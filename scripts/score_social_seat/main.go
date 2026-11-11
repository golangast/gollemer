// score_social_seat scores the social model against Section S of
// the frozen robustness eval with the typo normalizer off or on:
//
//	go run ./scripts/score_social_seat -model <gob> -eval <robustness_v1.jsonl> [-normalize]
//
// The protocol is the frozen scorer's (scripts/score_robustness),
// byte for byte: greedy Predict on the raw prompt (no situation
// line — the frozen prompts carry no session state), an item
// passes when the reply is non-empty and contains at least `need`
// of its want words. With -normalize, each prompt first passes
// through the chat seat's core (chat.NormalizeSocialLine) — the
// exact code the social input composition applies to a bare
// first-turn line — and the changed prompts are listed. The off
// run against the production gob must reproduce the pilot's
// frozen control numbers (42/160) before the on run means
// anything.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/training/chat"
)

type evalItem struct {
	ID      string   `json:"id"`
	Section string   `json:"section"`
	Family  string   `json:"family"`
	Variant string   `json:"variant"`
	Input   string   `json:"input"`
	Want    []string `json:"want"`
	Need    int      `json:"need"`
}

func loadModel(path string) (*seq2seq.Seq2Seq, error) {
	placeholder, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		return nil, err
	}
	model, err := seq2seq.Load(path, placeholder)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.NewTokenizer(model.OutputVocab)
	if err != nil {
		return nil, err
	}
	model.Tokenizer = tok
	model.BeamWidth = 0
	return model, nil
}

func main() {
	modelPath := flag.String("model", "", "social model .gob to score")
	evalPath := flag.String("eval", "", "robustness_v1.jsonl")
	normalize := flag.Bool("normalize", false, "apply the typo normalizer to each prompt first")
	flag.Parse()
	if *modelPath == "" || *evalPath == "" {
		fmt.Fprintln(os.Stderr, "score_social_seat: -model and -eval are required")
		os.Exit(2)
	}
	model, err := loadModel(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "score_social_seat: loading model: %v\n", err)
		os.Exit(1)
	}
	f, err := os.Open(*evalPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "score_social_seat: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()
	var items []evalItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		var it evalItem
		if err := json.Unmarshal(sc.Bytes(), &it); err != nil {
			fmt.Fprintf(os.Stderr, "score_social_seat: bad eval line: %v\n", err)
			os.Exit(1)
		}
		if it.Section == "S" {
			items = append(items, it)
		}
	}
	totals := map[string][2]int{}
	bump := func(key string, pass bool) {
		t := totals[key]
		t[1]++
		if pass {
			t[0]++
		}
		totals[key] = t
	}
	changed := 0
	for _, it := range items {
		input := it.Input
		if *normalize {
			if fixed := chat.NormalizeSocialLine(input, model.OutputVocab); fixed != input {
				changed++
				fmt.Printf("FIXED %s %q -> %q\n", it.ID, input, fixed)
				input = fixed
			}
		}
		reply, err := model.Predict(input, 60)
		if err != nil {
			fmt.Fprintf(os.Stderr, "score_social_seat: predict %s: %v\n", it.ID, err)
			os.Exit(1)
		}
		need := it.Need
		if need <= 0 {
			need = 1
		}
		hits := 0
		for _, w := range it.Want {
			if regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(w) + `\b`).MatchString(reply) {
				hits++
			}
		}
		pass := strings.TrimSpace(reply) != "" && hits >= need
		fmt.Printf("%s %s [%s/%s] %q -> %q (want-hit %d/%d)\n",
			map[bool]string{true: "PASS", false: "FAIL"}[pass], it.ID, it.Family, it.Variant, it.Input, reply, hits, len(it.Want))
		bump("S total", pass)
		bump("variant "+it.Variant, pass)
		if it.Variant == "typo" || it.Variant == "funcdrop" {
			bump("S mutated (typo+funcdrop)", pass)
		}
	}
	fmt.Println("---")
	if *normalize {
		fmt.Printf("prompts changed by the seat: %d/%d\n", changed, len(items))
	}
	for _, key := range []string{"S total", "S mutated (typo+funcdrop)", "variant clean", "variant paraphrase", "variant typo", "variant funcdrop"} {
		t := totals[key]
		fmt.Printf("%s: %d/%d\n", key, t[0], t[1])
	}
}
