// eval_decl scores the trained decl model over the held-out matrix
// produced by gen_decl_data: parse rate, compiles-standalone rate
// (go/types against a stub universe, real stdlib imports), and the
// structural-exact rate — per input form (phrasing vs structured
// spec) and per split cell (combination seen/unseen x frame
// seen/unseen), with the failure taxonomy.
//
// Run: go run ./scripts/eval_decl [-model path] [-heldout path] [-limit N]
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/training/decl"
)

func main() {
	modelPath := flag.String("model", "data/models/gob_models/real_tiny_seq2seq_decl.gob", "trained decl model")
	heldoutPath := flag.String("heldout", "data/training/decl_heldout.jsonl", "held-out eval items")
	outPath := flag.String("out", "data/training/decl_eval_report.json", "report output (empty = stdout only)")
	limit := flag.Int("limit", 0, "score only the first N items (0 = all)")
	flag.Parse()

	items, err := loadItems(*heldoutPath)
	if err != nil {
		fail(err)
	}
	if *limit > 0 && *limit < len(items) {
		items = items[:*limit]
	}

	placeholder, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		fail(err)
	}
	model, err := seq2seq.Load(*modelPath, placeholder)
	if err != nil {
		fail(fmt.Errorf("load model: %w (train it first: go run . -train-real-seq2seq -domain decl)", err))
	}
	if model.OutputVocab == nil || model.OutputVocab.Size() == 0 {
		fail(fmt.Errorf("model has no vocabulary"))
	}
	tok, err := tokenizer.NewTokenizer(model.OutputVocab)
	if err != nil {
		fail(err)
	}
	model.Tokenizer = tok
	if model.Decoder == nil || model.Decoder.MoE == nil {
		fail(fmt.Errorf("model was trained without the MoE decoder"))
	}

	predict := func(input string) (string, error) {
		return model.Predict(input, 64)
	}
	rep := decl.RunEval(items, predict)

	keys := make([]string, 0, len(rep.Cells))
	for k := range rep.Cells {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Printf("%-28s %5s %8s %9s %8s\n", "cell (form|combo|frame)", "N", "exact%", "parse%", "compile%")
	for _, k := range keys {
		c := rep.Cells[k]
		fmt.Printf("%-28s %5d %7.1f%% %8.1f%% %7.1f%%\n", k, c.N,
			100*float64(c.Exact)/float64(c.N),
			100*float64(c.Parsed)/float64(c.N),
			100*float64(c.Compiles)/float64(c.N))
	}
	fmt.Println("taxonomy:", formatCounts(rep.Taxonomy))

	if *outPath != "" {
		raw, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fail(err)
		}
		if err := os.WriteFile(*outPath, raw, 0o644); err != nil {
			fail(err)
		}
		fmt.Println("report written to", *outPath)
	}
}

func loadItems(path string) ([]decl.EvalItem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []decl.EvalItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var item decl.EvalItem
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, sc.Err()
}

func formatCounts(m map[string]int) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%d", k, m[k]))
	}
	return strings.Join(parts, " ")
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "eval_decl:", err)
	os.Exit(1)
}
