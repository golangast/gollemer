package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func main() {
	projectRoot, _ := os.Getwd()
	bpePath := filepath.Join(projectRoot, "data/models/gob_models/bpe_tokenizer.gob")
	modelPath := filepath.Join(projectRoot, "data/models/gob_models/moe_social_model.gob")

	// Try to load BPE tokenizer first
	var tok *tokenizer.BPETokenizer
	if t, err := tokenizer.LoadBPETokenizer(bpePath); err == nil {
		tok = t
		log.Printf("Loaded BPE tokenizer (vocab=%d)", tok.Vocab.Size())
	} else {
		log.Printf("BPE tokenizer not found: %v", err)
	}

	model, err := moe.LoadIntentMoEModelWithFallback(modelPath)
	if err != nil {
		log.Fatalf("failed to load social model: %v", err)
	}
	model.RepairArchitecture()

	prompt := "add int after the return type of F in f/j.go"
	chatml := "<|im_start|>system\nYou are Gollemer, an expert Go AI assistant.\n<|im_end|>\n<|im_start|>user\n" + prompt + "\n<|im_end|>\n<|im_start|>assistant\n"

	var current []int
	if tok != nil {
		tokenIDs := tok.Encode(chatml)
		current = append([]int(nil), tokenIDs...)
	} else if model.SentenceVocab != nil {
		// Fallback: simple whitespace tokenization into sentence vocab IDs
		words := simpleSplit(prompt)
		for _, w := range words {
			id := model.SentenceVocab.GetTokenID(w)
			if id < 0 {
				id = model.SentenceVocab.GetTokenID("UNK")
				if id < 0 {
					id = 0
				}
			}
			current = append(current, id)
		}
	} else {
		log.Fatalf("no tokenizer or sentence vocab available")
	}

	var gen []int
	for i := 0; i < 256; i++ {
		next := model.PredictNextToken(current, nil)
		if next == 0 {
			break
		}
		gen = append(gen, next)
		current = append(current, next)
	}

	// Decode output
	if tok != nil {
		out := tok.Decode(gen)
		fmt.Println("--- GENERATED (BPE) ---")
		fmt.Println(out)
	} else if model.SentenceVocab != nil {
		var outWords []string
		for _, id := range gen {
			w := model.SentenceVocab.GetWord(id)
			if w == "" {
				continue
			}
			outWords = append(outWords, w)
		}
		fmt.Println("--- GENERATED (SentenceVocab) ---")
		fmt.Println(simpleJoin(outWords))
	} else {
		fmt.Println("--- GENERATED: (no decoder) ---")
	}
}

func simpleSplit(s string) []string {
	var res []string
	cur := ""
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ' ' || c == '\n' || c == '\t' || c == ',' || c == '.' || c == '/' {
			if cur != "" {
				res = append(res, cur)
				cur = ""
			}
			continue
		}
		cur += string(c)
	}
	if cur != "" {
		res = append(res, cur)
	}
	return res
}

func simpleJoin(words []string) string {
	out := ""
	for i, w := range words {
		if i > 0 {
			out += " "
		}
		out += w
	}
	return out
}
