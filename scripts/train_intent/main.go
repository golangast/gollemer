// train_intent fits the intent classifier on
// data/training/intent_pairs.jsonl, evaluates it on the
// handwritten held-out set (the verdict), and writes
// data/intent_model.json.
//
// Run: go run ./scripts/train_intent
package main

import (
	"fmt"
	"sort"

	"github.com/golangast/gollemer/internal/ai/training/chat"
)

func main() {
	train, err := chat.LoadIntentPairs("data/training/intent_pairs.jsonl")
	if err != nil {
		panic(err)
	}
	held, err := chat.LoadIntentPairs("data/training/intent_heldout.jsonl")
	if err != nil {
		panic(err)
	}
	m := chat.TrainIntentModel(train)

	// Training accuracy (sanity only — memorization proves nothing).
	tOK := 0
	for _, p := range train {
		if got, _, _ := m.Predict(p.Input); got == p.Intent {
			tOK++
		}
	}
	fmt.Printf("train: %d/%d = %.1f%% (sanity only)\n", tOK, len(train), 100*float64(tOK)/float64(len(train)))

	// The verdict: handwritten phrasings the training never saw,
	// judged at the runtime gate (confidence + margin).
	type miss struct{ in, want, got string }
	var misses []miss
	hOK, gated, gatedOK := 0, 0, 0
	for _, p := range held {
		raw, _, _ := m.Predict(p.Input)
		got, _, ok := m.ClassifyIntent(p.Input)
		if ok {
			gated++
			if got == p.Intent {
				gatedOK++
			}
		}
		if raw == p.Intent {
			hOK++
		} else {
			misses = append(misses, miss{p.Input, p.Intent, raw + " (gated:" + got + ")"})
		}
	}
	fmt.Printf("HELD-OUT: %d/%d = %.1f%% raw; gated %d lines at %.1f%% precision (the rest fall through)\n",
		hOK, len(held), 100*float64(hOK)/float64(len(held)), gated, 100*float64(gatedOK)/float64(gated))
	sort.Slice(misses, func(i, j int) bool { return misses[i].want < misses[j].want })
	for _, ms := range misses {
		fmt.Printf("  MISS want=%-18s got=%-28s %q\n", ms.want, ms.got, ms.in)
	}
	if err := m.Save("data/intent_model.json"); err != nil {
		panic(err)
	}
	fmt.Println("wrote data/intent_model.json")
}
