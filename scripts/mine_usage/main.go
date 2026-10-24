// mine_usage mines the Coordinator's outcome log
// (~/.gollemer_brains.jsonl) for CONFIRMED (line → intent) pairs:
// real lines a user actually typed, with the label their own
// behavior confirmed. It writes the pairs the intent classifier
// trains on next time scripts/train_intent runs:
//
//	data/training/intent_usage_pairs.jsonl    the train slice
//	data/training/intent_usage_heldout.jsonl  the real held-out
//
// Only two outcomes carry a trustworthy label:
//
//	dispatched     the classifier was confident and the line was
//	               routed to that intent
//	clarified-yes  the user answered a "did you mean…?" with yes —
//	               the strongest signal there is
//
// clarified-no lines are NEGATIVES (the guess was rejected; the true
// intent is unknown) and fell-through lines were never confirmed:
// both are excluded, never guessed at. Slash/exact commands and
// one-word control replies ("yes", "next", "1") are not phrasings.
// Any line equal to a handwritten held-out line is dropped, the
// same filter scripts/gen_intent_data applies, so the verdict set
// stays uncontaminated.
//
// The split is deterministic: pairs sorted by (line, intent), every
// 5th pair (index 4, 9, 14, …) goes to the real held-out, the rest
// to train. Re-run it whenever the log has grown:
//
//	go run ./scripts/mine_usage
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/training/chat"
)

// logEntry mirrors chat's brainLogEntry, including the outcome
// fields recordIntentOutcome writes (2026-10-24). Entries from
// before then have no Outcome — and never pair a line with an
// intent, which is exactly the gap the capture fix closed.
type logEntry struct {
	Brain   string  `json:"brain"`
	OK      bool    `json:"ok"`
	Note    string  `json:"note"`
	Line    string  `json:"line"`
	Intent  string  `json:"intent"`
	Conf    float64 `json:"conf"`
	Outcome string  `json:"outcome"`
}

// controlReplies are conversational plumbing, not phrasings: a
// "yes" answering a clarification is evidence about the PREVIOUS
// line (which is what gets recorded), never a pair itself.
func controlReply(line string) bool {
	switch strings.ToLower(strings.Trim(strings.TrimSpace(line), ".!?")) {
	case "", "y", "yes", "n", "no", "1", "2", "neither", "next", "quit", "exit", "cancel", "stop":
		return true
	}
	return false
}

func main() {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}
	brainsPath := flag.String("brains", filepath.Join(home, ".gollemer_brains.jsonl"), "coordinator outcome log to mine")
	actionsPath := flag.String("actions", filepath.Join(home, ".gollemer_actions.jsonl"), "predictor action log (counted, never mined: it records no labels)")
	heldoutPath := flag.String("heldout", "data/training/intent_heldout.jsonl", "handwritten held-out set to filter against")
	outPath := flag.String("out", "data/training/intent_usage_pairs.jsonl", "mined train pairs")
	realHeldoutPath := flag.String("realheldout", "data/training/intent_usage_heldout.jsonl", "mined real held-out pairs")
	flag.Parse()

	held, err := chat.LoadIntentPairs(*heldoutPath)
	if err != nil {
		panic(err)
	}
	heldInputs := map[string]bool{}
	for _, p := range held {
		heldInputs[p.Input] = true
	}

	data, err := os.ReadFile(*brainsPath)
	if err != nil {
		panic(err)
	}

	var (
		scanned, badLines                      int
		legacyIntent, legacySentence           int
		legacyEdit, legacyOther                int
		nDisp, nYes, nNo, nFell, nUnknown      int
		exSlash, exControl, exHeld, exConflict int
		exDup                                  int
		candidates                             []chat.IntentPair
	)
	for _, l := range strings.Split(string(data), "\n") {
		l = strings.TrimSpace(l)
		if l == "" {
			continue
		}
		scanned++
		var e logEntry
		if err := json.Unmarshal([]byte(l), &e); err != nil {
			badLines++
			continue
		}
		if e.Outcome == "" {
			// Legacy entry: the intent and the raw line were
			// never stored together, so no pair can be formed.
			switch {
			case strings.HasPrefix(e.Note, "intent "):
				legacyIntent++
			case strings.HasPrefix(e.Note, "sentence: "):
				legacySentence++
			case strings.HasPrefix(e.Note, "edit "):
				legacyEdit++
			default:
				legacyOther++
			}
			continue
		}
		switch e.Outcome {
		case chat.IntentOutcomeDispatched:
			nDisp++
		case chat.IntentOutcomeClarifiedYes:
			nYes++
		case chat.IntentOutcomeClarifiedNo:
			nNo++ // a rejected guess: negative evidence, not a pair
			continue
		case chat.IntentOutcomeFellThrough:
			nFell++ // never confirmed: not a pair
			continue
		default:
			nUnknown++
			continue
		}
		// A confirmed candidate. Apply the exclusions in order;
		// each candidate is counted under exactly one reason.
		line := strings.TrimSpace(e.Line)
		switch {
		case line == "" || e.Intent == "":
			exControl++
		case strings.HasPrefix(line, "/"):
			exSlash++
		case controlReply(line):
			exControl++
		case heldInputs[line]:
			exHeld++
		default:
			candidates = append(candidates, chat.IntentPair{Input: line, Intent: e.Intent})
		}
	}

	// One line, two confirmed labels: trust neither.
	intentsByLine := map[string]map[string]bool{}
	for _, p := range candidates {
		if intentsByLine[p.Input] == nil {
			intentsByLine[p.Input] = map[string]bool{}
		}
		intentsByLine[p.Input][p.Intent] = true
	}
	seen := map[chat.IntentPair]bool{}
	var pairs []chat.IntentPair
	for _, p := range candidates {
		if len(intentsByLine[p.Input]) > 1 {
			exConflict++
			continue
		}
		if seen[p] {
			exDup++
			continue
		}
		seen[p] = true
		pairs = append(pairs, p)
	}
	sort.Slice(pairs, func(i, j int) bool {
		if pairs[i].Input != pairs[j].Input {
			return pairs[i].Input < pairs[j].Input
		}
		return pairs[i].Intent < pairs[j].Intent
	})

	var train, realHeld []chat.IntentPair
	for i, p := range pairs {
		if i%5 == 4 {
			realHeld = append(realHeld, p)
		} else {
			train = append(train, p)
		}
	}
	writePairs := func(path string, ps []chat.IntentPair) {
		f, err := os.Create(path)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		for _, p := range ps {
			b, err := json.Marshal(p)
			if err != nil {
				panic(err)
			}
			if _, err := f.Write(append(b, '\n')); err != nil {
				panic(err)
			}
		}
	}
	writePairs(*outPath, train)
	writePairs(*realHeldoutPath, realHeld)

	fmt.Printf("mine_usage: scanned %d entries from %s (%d unparseable)\n", scanned, *brainsPath, badLines)
	fmt.Printf("  legacy entries, no outcome fields: %d (intent noted but raw line never recorded: %d; command sentence noted, no intent label: %d; edit outcomes, no raw line: %d; other notes: %d)\n",
		legacyIntent+legacySentence+legacyEdit+legacyOther, legacyIntent, legacySentence, legacyEdit, legacyOther)
	fmt.Printf("  outcome entries: dispatched %d, clarified-yes %d, clarified-no %d (negative, excluded), fell-through %d (unconfirmed, excluded), unknown %d\n",
		nDisp, nYes, nNo, nFell, nUnknown)
	fmt.Printf("  confirmed candidates: %d; excluded: slash commands %d, control replies/empty %d, held-out lines %d, conflicting labels %d, duplicates %d\n",
		len(candidates)+exSlash+exControl+exHeld, exSlash, exControl, exHeld, exConflict, exDup)
	if adata, err := os.ReadFile(*actionsPath); err == nil {
		n := 0
		for _, l := range strings.Split(string(adata), "\n") {
			if strings.TrimSpace(l) != "" {
				n++
			}
		}
		fmt.Printf("  actions log: %d raw lines in %s — no labels recorded there; counted, not mined\n", n, *actionsPath)
	}
	fmt.Printf("  MINED: %d pairs → %d train (%s), %d real held-out (%s)\n", len(pairs), len(train), *outPath, len(realHeld), *realHeldoutPath)
	if len(pairs) > 0 && len(pairs) <= 50 {
		for _, p := range pairs {
			fmt.Printf("    pair: %q -> %s\n", p.Input, p.Intent)
		}
	}
}
