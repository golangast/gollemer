// train_intent fits the intent classifier on
// data/training/intent_pairs.jsonl plus the real usage pairs
// scripts/mine_usage mined (data/training/intent_usage_pairs.jsonl),
// evaluates it on the handwritten held-out set (the verdict) and on
// the mined real held-out, and writes data/intent_model.json.
//
// Run: go run ./scripts/train_intent
package main

import (
	"fmt"
	"sort"

	"github.com/golangast/gollemer/internal/ai/training/chat"
)

func main() {
	gen, err := chat.LoadIntentPairs("data/training/intent_pairs.jsonl")
	if err != nil {
		panic(err)
	}
	// Real usage pairs, mined from the coordinator's outcome log by
	// scripts/mine_usage: the user's own confirmed lines, each
	// included exactly once and counted separately — never
	// multiplied the way generated templates are. A missing or
	// empty file simply means none have been mined yet.
	usage, err := chat.LoadIntentPairs("data/training/intent_usage_pairs.jsonl")
	if err != nil {
		usage = nil
	}
	train := append(append([]chat.IntentPair{}, gen...), usage...)
	fmt.Printf("pairs: %d generated + %d real usage = %d train\n", len(gen), len(usage), len(train))
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

	// The real held-out: the user's own typed lines, held out of
	// training by mine_usage's deterministic split. This is the
	// "does it guess HIS phrasings" number — reported separately,
	// never blended into the verdict above.
	if rh, err := chat.LoadIntentPairs("data/training/intent_usage_heldout.jsonl"); err == nil && len(rh) > 0 {
		rOK := 0
		for _, p := range rh {
			if got, _, _ := m.Predict(p.Input); got == p.Intent {
				rOK++
			}
		}
		fmt.Printf("REAL HELD-OUT (user's own lines): %d/%d = %.1f%% raw\n", rOK, len(rh), 100*float64(rOK)/float64(len(rh)))
	} else {
		fmt.Println("REAL HELD-OUT: none — mine_usage has not mined enough confirmed pairs to hold any out")
	}

	if err := m.Save("data/intent_model.json"); err != nil {
		panic(err)
	}
	fmt.Println("wrote data/intent_model.json")
}
