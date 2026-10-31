package chat

// Phrasing layer, attempt 2 (2026-10-31): two candidate
// mechanisms measured head-to-head against run 1's judge
// (data/training/phrasing_heldout.jsonl, unmodified) under
// slot-exact scoring (phrasing2_slots.go):
//
//   - Arm A: the pointer-generator mapper, retrained on ~5x
//     denser, factor-crossed data with random-name pools,
//     indexed-UNK disambiguation (seq2seq's
//     PredictPtrGenVerbatimIndexed), and checkpoint selection
//     by validation instead of train loss (phrasing2_train.go).
//   - Arm B: classify + extract — an intent-lineage softmax
//     classifier over the closed family set, deterministic
//     glue segmentation for compounds, and per-family slot
//     extraction that renders the canonical sentence and must
//     survive the real parsers (this file).
//
// Outcome (same judge, slot-exact; bar: single-unseen >= 90%,
// compound-seenpair >= 85%): arm B single-unseen 98.2%,
// single-seen 100%, compound-seenpair 86.4%, compound-
// unseenpair 92.0%, zero none-class false claims — BOTH bars
// cleared, arm B seated (phrasing2_claim.go). Arm A, with all
// the levers applied, scored single-unseen 44.7% from its
// best-validation checkpoint (45.3% from its best-train-loss
// checkpoint) and compound-seenpair 9.7% — below even run 1's
// rescore (62.6% / 33.1%) — not earned, not seated; its
// trainer, data, and evals stay as the measured record. Run
// 1's baseline under slot-exact: 62.6% / 89.0% / 33.1% /
// 12.0% (single-unseen / single-seen / compound-seenpair /
// compound-unseenpair).
//
// PhrasingRoute is arm B's entry: a raw line in, the canonical
// piece(s) out — or ok=false, the layer's only other answer.
// It never guesses a slot it cannot point at in the line, and
// every rendered piece is validated by CanonicalSlots before
// it leaves, so what it returns is exactly what the claim
// chain would have parsed from a cleanly typed sentence.

import (
	"strings"
	"sync"
)

// PhrasingRoute classifies and extracts one line into its
// canonical piece(s). ok=false means no claim: segmentation
// found more than two pieces, the classifier's confidence or
// margin failed the intent layer's gate, a piece's extraction
// could not bind every slot, or a rendered canonical failed
// validation against the real parsers.
func PhrasingRoute(line string) (pieces []string, ok bool) {
	return phrasingRouteB(line)
}

// phrasingRouteB is arm B proper; it is implemented in
// phrasing2_extract.go. This indirection keeps PhrasingRoute
// stable as the eval/integration seam while the extractor is
// built and measured.

// --- arm B classifier artifact ---------------------------------
//
// The classifier is an IntentModel (intent.go's softmax over
// word n-gram + char-trigram features) trained on the v2
// fragments with family labels, saved as JSON at
// data/phrasing_model.json — the intent model's conventions.
// Absent file = arm B is off, exactly as an absent intent
// model turns that layer off.

var (
	phrasingModelOnce sync.Once
	phrasingModel     *IntentModel
)

// PhrasingModelPath is where arm B's trained classifier lives,
// relative to a project root.
const PhrasingModelRelPath = "data/phrasing_model.json"

func defaultPhrasingModel() *IntentModel {
	phrasingModelOnce.Do(func() {
		for _, base := range []string{unifiedProjectRoot, startFolder()} {
			if base == "" {
				continue
			}
			if m, err := LoadIntentModel(base + "/" + PhrasingModelRelPath); err == nil {
				phrasingModel = m
				return
			}
		}
	})
	return phrasingModel
}

// phrasingClassifier is the seam the route reads through — a
// variable so tests can count consultations (the
// deterministic-hit protection proof) and swap models.
var phrasingClassifier = func(line string) (family string, ok bool) {
	m := defaultPhrasingModel()
	if m == nil {
		return "", false
	}
	fam, _, ok := m.ClassifyIntent(line)
	return fam, ok
}

// PhrasingStripWrapper removes the politeness wrappers the
// generator dresses sentences in (train and held-out families
// alike — stripping is mechanism, not memorization: any
// leading/trailing courtesy phrase comes off). Segmentation,
// classification, and extraction all read the stripped form,
// and arm B's classifier trains on stripped fragments, so the
// whole arm sees one normal form.
func PhrasingStripWrapper(line string) string {
	s := strings.TrimSpace(line)
	for {
		low := strings.ToLower(s)
		trimmed := s
		for _, pre := range phrasingWrapperPrefixes {
			if strings.HasPrefix(low, pre) {
				trimmed = strings.TrimSpace(s[len(pre):])
				break
			}
		}
		for _, suf := range phrasingWrapperSuffixes {
			if strings.HasSuffix(strings.ToLower(trimmed), suf) {
				trimmed = strings.TrimSpace(trimmed[:len(trimmed)-len(suf)])
				break
			}
		}
		trimmed = strings.Trim(trimmed, " ,")
		if trimmed == s || trimmed == "" {
			break
		}
		s = trimmed
	}
	return s
}

var phrasingWrapperPrefixes = []string{
	"would you mind ", "go ahead and ", "i need you to ",
	"i'd like to ", "i want to ", "can you ", "could you ",
	"can we ", "let's ", "let us ", "hey gollemer, ", "hey gollemer ",
	"hey, ", "hey ", "please ", "kindly ", "just ", "quick, ", "quick ",
	"plese ",
}

var phrasingWrapperSuffixes = []string{
	" please", " for me", " when you can", ".", "?",
}

// PhrasingPieces normalizes a raw arm-A model output into the
// canonical pieces the layer validates: quote spacing
// collapsed (the decoder space-joins its tokens, so a quoted
// print text comes back as `" the results "`), then split on
// the " ## " separator. "none" and "" yield no pieces. The
// collapse is token-based, not regex: a `"` field toggles
// in-quote state, and fields are rejoined with no space after
// an opening quote and no space before a closing quote (the
// rule eval_phrasing measured; the scripts' copies must not
// drift from this one).
func PhrasingPieces(raw string) []string {
	s := collapsePhrasingQuoteSpacing(raw)
	if s == "" || s == "none" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(s, " ## ") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func collapsePhrasingQuoteSpacing(raw string) string {
	fields := strings.Fields(raw)
	var b strings.Builder
	inQuote := false
	prevOpen := false // previous field was an opening quote
	for i, f := range fields {
		closing := f == `"` && inQuote
		if i > 0 && !prevOpen && !closing {
			b.WriteByte(' ')
		}
		b.WriteString(f)
		prevOpen = false
		if f == `"` {
			if inQuote {
				inQuote = false
			} else {
				inQuote = true
				prevOpen = true
			}
		}
	}
	return b.String()
}
