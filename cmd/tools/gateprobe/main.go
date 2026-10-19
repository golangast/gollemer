// Command gateprobe loads a trained seq2seq model and reports the state of
// its copy gates: whether the parameters moved from initialization.
package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/training/chat"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func main() {
	modelPath := flag.String("model", "", "path to .gob model file")
	flag.Parse()
	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "need -model")
		os.Exit(1)
	}
	placeholder, err := tokenizer.NewTokenizer(mainvocab.NewVocabulary())
	if err != nil {
		fmt.Fprintf(os.Stderr, "tokenizer: %v\n", err)
		os.Exit(1)
	}
	m, err := seq2seq.Load(*modelPath, placeholder)
	if err != nil {
		fmt.Fprintf(os.Stderr, "model load: %v\n", err)
		os.Exit(1)
	}
	_ = chat.CodeIntentDomain
	reportGate("Decoder.EncoderCopy", m.Decoder.EncoderCopy)
}

func reportGate(name string, g *seq2seq.CopyGate) {
	if g == nil {
		fmt.Printf("=== %s: nil ===\n", name)
		return
	}
	bg := g.Bg.Data[0]
	g0 := 1.0 / (1.0 + math.Exp(float64(-bg)))
	var wnorm float64
	for _, w := range g.Wg.Data {
		wnorm += float64(w) * float64(w)
	}
	wnorm = math.Sqrt(wnorm)
	fmt.Printf("=== %s ===\n", name)
	fmt.Printf("Bg=%.4f (init -2.0) -> gate@zero-query=%.4f (init 0.1192)\n", bg, g0)
	fmt.Printf("||Wg||_2=%.6f (init 0.0), dim=%d\n", wnorm, g.HiddenDim)
	fmt.Printf("gate moved from init: %v\n", math.Abs(g0-0.1192) > 0.01 || wnorm > 1e-6)
}
