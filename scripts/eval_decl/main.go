// eval_decl scores the trained decl model over the held-out matrix
// produced by gen_decl_data: parse rate, compiles-standalone rate
// (go/types against a stub universe, real stdlib imports), and the
// structural-exact rate — per input form (phrasing vs structured
// spec) and per split cell (combination seen/unseen x frame
// seen/unseen), with the failure taxonomy.
//
// Run: go run ./scripts/eval_decl [-model path] [-heldout path] [-limit N]
// [-mode baseline|constrained|roundtrip]
//
// The default mode is the original unconstrained harness. The other
// modes wrap generation in an input-derived decoding constraint
// (automaton.go, constrain.go), and roundtrip additionally verifies
// spec-form candidates against the spec recovered from the input
// string itself (predict.go). Constrained modes write an extended
// report (mode, wall time, constraint statistics, and the standard
// EvalReport under "report").
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

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
	mode := flag.String("mode", "baseline", "decoding mode: baseline | constrained | roundtrip")
	attempts := flag.Int("attempts", 8, "roundtrip mode: max candidates per spec-form item")
	seed := flag.Int64("seed", 20261028, "roundtrip mode: sampling seed base")
	flag.Parse()
	if *mode != "baseline" && *mode != "constrained" && *mode != "roundtrip" {
		fail(fmt.Errorf("unknown -mode %q", *mode))
	}

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

	var rep *decl.EvalReport
	var pred *predictor
	start := time.Now()
	if *mode == "baseline" {
		predict := func(input string) (string, error) {
			return model.Predict(input, 64)
		}
		rep = decl.RunEval(items, predict)
	} else {
		initWordSyms(model)
		pred = newPredictor(model, *mode == "roundtrip", *attempts, *seed)
		rep = decl.RunEval(items, pred.predict)
	}
	wall := time.Since(start)

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
	fmt.Printf("mode=%s wall=%.1fs\n", *mode, wall.Seconds())
	if pred != nil {
		st := pred.stats
		fmt.Printf("constraint: kind-underivable=%d dead-end-steps=%d dead-end-items=%d anti-repeat-overrides=%d\n",
			st.kindUnderivable, st.deadEndSteps, st.deadEndItems, st.antiOverrides)
		if *mode == "roundtrip" {
			fmt.Printf("roundtrip: spec-items=%d skipped=%d attempts=%s\n",
				st.rtItems, st.rtSkipped, formatIntCounts(st.attemptsHist))
		}
	}

	if *outPath != "" {
		var raw []byte
		var err error
		if *mode == "baseline" {
			raw, err = json.MarshalIndent(rep, "", "  ")
		} else {
			raw, err = json.MarshalIndent(extendedReport{
				Mode:              *mode,
				Model:             *modelPath,
				MaxAttempts:       *attempts,
				WallSeconds:       wall.Seconds(),
				KindUnderivable:   pred.stats.kindUnderivable,
				RoundTripItems:    pred.stats.rtItems,
				RoundTripSkipped:  pred.stats.rtSkipped,
				AttemptsHistogram: pred.stats.attemptsHist,
				DeadEndSteps:      pred.stats.deadEndSteps,
				DeadEndItems:      pred.stats.deadEndItems,
				AntiOverrides:     pred.stats.antiOverrides,
				Report:            rep,
			}, "", "  ")
		}
		if err != nil {
			fail(err)
		}
		if err := os.WriteFile(*outPath, raw, 0o644); err != nil {
			fail(err)
		}
		fmt.Println("report written to", *outPath)
	}
}

// extendedReport wraps the standard EvalReport with the constrained
// modes' run statistics.
type extendedReport struct {
	Mode              string           `json:"mode"`
	Model             string           `json:"model"`
	MaxAttempts       int              `json:"max_attempts"`
	WallSeconds       float64          `json:"wall_seconds"`
	KindUnderivable   int              `json:"kind_underivable"`
	RoundTripItems    int              `json:"roundtrip_items"`
	RoundTripSkipped  int              `json:"roundtrip_skipped"`
	AttemptsHistogram map[int]int      `json:"attempts_histogram"`
	DeadEndSteps      int              `json:"dead_end_steps"`
	DeadEndItems      int              `json:"dead_end_items"`
	AntiOverrides     int              `json:"anti_repeat_overrides"`
	Report            *decl.EvalReport `json:"report"`
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

func formatIntCounts(m map[int]int) string {
	keys := make([]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%d:%d", k, m[k]))
	}
	return strings.Join(parts, " ")
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "eval_decl:", err)
	os.Exit(1)
}
