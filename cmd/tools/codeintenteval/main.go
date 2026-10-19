// Command codeintenteval scores the trained codeintent model on the held-out
// Bubble Tea distillation pairs (data/training/codeintent_heldout.jsonl).
//
// It decodes each held-out input with model.Predict (the same path the
// end-of-training probe uses) and reports unigram multiset F1 against the
// teacher answer, the UNK-token rate (a rough lost-ness proxy), and
// disambiguation-agreement for teacher answers that ask "which one did you
// mean". F1 vs the teacher is a rough proxy, not a verdict — the teacher
// answers are often disambiguation lists, so the worst/best examples are
// printed in full for human judgment.
//
// Usage: go run ./cmd/tools/codeintenteval -n 100   (0 or -1 = all 1551)
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	chat "github.com/golangast/gollemer/internal/ai/training/chat"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

type pair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

var wordRe = regexp.MustCompile(`[a-z0-9_]+`)

// tokens lowercases and splits on word characters; mirrors the probe's
// strings.ToLower treatment of the input.
func tokens(s string) []string { return wordRe.FindAllString(strings.ToLower(s), -1) }

// f1 is unigram multiset F1 between candidate and reference.
func f1(cand, ref []string) float64 {
	if len(cand) == 0 || len(ref) == 0 {
		return 0
	}
	counts := map[string]int{}
	for _, t := range ref {
		counts[t]++
	}
	overlap := 0
	for _, t := range cand {
		if counts[t] > 0 {
			counts[t]--
			overlap++
		}
	}
	p := float64(overlap) / float64(len(cand))
	r := float64(overlap) / float64(len(ref))
	if p+r == 0 {
		return 0
	}
	return 2 * p * r / (p + r)
}

func unkRate(toks []string) float64 {
	if len(toks) == 0 {
		return 1
	}
	n := 0
	for _, t := range toks {
		if t == "unk" {
			n++
		}
	}
	return float64(n) / float64(len(toks))
}

func loadModel(projectRoot, modelPath string) (*seq2seq.Seq2Seq, error) {
	// Same bootstrap as chat.loadRealModel: vocab lives inside the file.
	placeholder, err := tokenizer.NewTokenizer(mainvocab.NewVocabulary())
	if err != nil {
		return nil, err
	}
	if modelPath == "" {
		modelPath = chat.RealModelPath(projectRoot, chat.CodeIntentDomain)
	}
	model, err := seq2seq.Load(modelPath, placeholder)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.NewTokenizer(model.OutputVocab)
	if err != nil {
		return nil, err
	}
	model.Tokenizer = tok
	return model, nil
}

type result struct {
	idx      int
	f1       float64
	unk      float64
	genLen   int
	teacher  string
	gen      string
	disambT  bool // teacher asks "which one did you mean"
	disambG  bool // model asks it too
	question string
}

func main() {
	n := flag.Int("n", 100, "pairs to score (stride-sampled); <=0 scores all")
	maxTok := flag.Int("tokens", 100, "max decode tokens per answer")
	root := flag.String("root", "", "project root (default: repo root of this tool)")
	maskCode := flag.Bool("maskcode", false, "replace the [code] block with a placeholder before scoring (ablation: does the model use the code at all)")
	fullOut := flag.String("fullout", "", "write every decoded answer as JSONL to this path")
	modelPath := flag.String("model", "", "model .gob to score (default: production codeintent model)")
	pairsPath := flag.String("pairs", "", "JSONL pairs file to score (default: data/training/codeintent_heldout.jsonl)")
	flag.Parse()

	// maskInput replaces the [code] section of a codeintent input with a fixed
	// placeholder, keeping [file ...] and [question] intact. Identical decoder
	// outputs with and without the code prove the model ignores it.
	maskInput := func(in string) string {
		c := strings.Index(in, "[code]")
		q := strings.Index(in, "[question]")
		if c < 0 || q < 0 || q < c {
			return in
		}
		return in[:c+len("[code]")] + "\n[code masked]\n" + in[q:]
	}

	projectRoot := *root
	if projectRoot == "" {
		// cmd/tools/codeintenteval -> repo root is two levels up.
		exe, _ := os.Executable()
		_ = exe
		wd, err := os.Getwd()
		if err != nil {
			log.Fatal(err)
		}
		projectRoot = wd
		for i := 0; i < 3; i++ {
			if _, err := os.Stat(filepath.Join(projectRoot, "data", "training", "codeintent_heldout.jsonl")); err == nil {
				break
			}
			projectRoot = filepath.Dir(projectRoot)
		}
	}

	heldoutPath := filepath.Join(projectRoot, "data", "training", "codeintent_heldout.jsonl")
	if *pairsPath != "" {
		if filepath.IsAbs(*pairsPath) {
			heldoutPath = *pairsPath
		} else {
			heldoutPath = filepath.Join(projectRoot, *pairsPath)
		}
	}

	f, err := os.Open(heldoutPath)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	var pairs []pair
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		var p pair
		if err := json.Unmarshal(sc.Bytes(), &p); err != nil {
			log.Fatal(err)
		}
		pairs = append(pairs, p)
	}
	if err := sc.Err(); err != nil {
		log.Fatal(err)
	}

	// Stride-sample so the subset spreads across the whole file.
	sel := pairs
	if *n > 0 && *n < len(pairs) {
		stride := float64(len(pairs)) / float64(*n)
		sel = sel[:0]
		for i := 0; i < *n; i++ {
			sel = append(sel, pairs[int(float64(i)*stride)])
		}
	}

	model, err := loadModel(projectRoot, *modelPath)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("loaded codeintent model; scoring %d/%d held-out pairs, %d decode tokens each",
		len(sel), len(pairs), *maxTok)

	var res []result
	for i, p := range sel {
		in := p.Input
		if *maskCode {
			in = maskInput(in)
		}
		out, err := model.Predict(strings.ToLower(in), *maxTok)
		if err != nil {
			log.Printf("pair %d predict error: %v", i, err)
			continue
		}
		ct, rt := tokens(out), tokens(p.Output)
		q := p.Input
		if k := strings.LastIndex(p.Input, "[question]"); k >= 0 {
			q = strings.TrimSpace(p.Input[k+len("[question]"):])
		}
		res = append(res, result{
			idx: i, f1: f1(ct, rt), unk: unkRate(ct), genLen: len(ct),
			teacher: p.Output, gen: out,
			disambT: strings.Contains(strings.ToLower(p.Output), "which one did you mean"),
			disambG: strings.Contains(strings.ToLower(out), "which one did you mean"),
			question: q,
		})
		fmt.Printf("[%4d] F1=%.3f UNK=%.2f len=%4d Q=%s\n", i, res[len(res)-1].f1, res[len(res)-1].unk, len(ct), q)
	}

	if len(res) == 0 {
		log.Fatal("no results")
	}
	sort.Slice(res, func(i, j int) bool { return res[i].f1 < res[j].f1 })
	sum, sumU, med := 0.0, 0.0, res[len(res)/2].f1
	agree, disN := 0, 0
	for _, r := range res {
		sum += r.f1
		sumU += r.unk
		if r.disambT {
			disN++
			if r.disambG {
				agree++
			}
		}
	}
	fmt.Printf("\n=== codeintent held-out (%d pairs, %d decode tokens) ===\n", len(res), *maxTok)
	fmt.Printf("mean F1 %.3f | median F1 %.3f | mean UNK rate %.3f\n", sum/float64(len(res)), med, sumU/float64(len(res)))
	if disN > 0 {
		fmt.Printf("disambiguation agreement: %d/%d teacher-disambiguation cases also disambiguated by model\n", agree, disN)
	}
	for _, h := range []struct {
		name string
		rows []result
	}{
		{"WORST 3 by F1", res[:3]},
		{"BEST 3 by F1", res[len(res)-3:]},
	} {
		fmt.Printf("\n--- %s ---\n", h.name)
		for _, r := range h.rows {
			fmt.Printf("Q: %s\nTEACHER: %.300s\nMODEL  : %.300s\n(F1=%.3f UNK=%.2f)\n\n",
				r.question, strings.ReplaceAll(r.teacher, "\n", " "),
				strings.ReplaceAll(r.gen, "\n", " "), r.f1, r.unk)
		}
	}

	if *fullOut != "" {
		of, err := os.Create(*fullOut)
		if err != nil {
			log.Fatal(err)
		}
		w := bufio.NewWriter(of)
		for _, r := range res {
			rec := map[string]string{
				"question": r.question,
				"teacher":  r.teacher,
				"gen":      r.gen,
			}
			jb, _ := json.Marshal(rec)
			w.Write(jb)
			w.Write([]byte("\n"))
		}
		w.Flush()
		of.Close()
		log.Printf("wrote %d full outputs to %s", len(res), *fullOut)
	}
}
