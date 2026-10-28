package main

import (
	"hash/fnv"
	"math/rand"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/training/decl"
)

// The predictor behind -mode constrained and -mode roundtrip.
//
// constrained: one greedy decode per item under the input-derived
// automaton.
//
// roundtrip: for spec-form items (the input IS the spec string, so
// the reference spec is recovered from the input by
// decl.ParseSpecText), candidates are produced until one round-trips
// — the candidate is parsed back to a spec with the filter's own
// extraction (decl.ExtractSpec) and compared to the reference with
// the eval's comparator (decl.CompareSpec), tag presence only, since
// tag content is not present in the spec language — or the attempt
// budget (default 8) runs out, in which case the best-scoring
// candidate by the same comparator wins. Attempt 1 is the greedy
// constrained decode; later attempts sample with rising temperature
// under the same constraint, seeded per item for reproducibility.
// Phrasing-form items get the single greedy constrained decode (no
// in-band reference exists to round-trip against).

type runStats struct {
	kindUnderivable int
	rtSkipped       int // spec-form items whose input failed to parse
	rtItems         int // spec-form items round-tripped
	attemptsHist    map[int]int
	deadEndSteps    int
	deadEndItems    int
	antiOverrides   int
}

type predictor struct {
	model       *seq2seq.Seq2Seq
	roundtrip   bool
	maxAttempts int
	seed        int64
	stats       *runStats
	sampleTemps []float64
}

func newPredictor(m *seq2seq.Seq2Seq, roundtrip bool, maxAttempts int, seed int64) *predictor {
	return &predictor{
		model:       m,
		roundtrip:   roundtrip,
		maxAttempts: maxAttempts,
		seed:        seed,
		stats:       &runStats{attemptsHist: map[int]int{}},
		sampleTemps: []float64{0.7, 0.85, 1.0, 1.0, 1.15, 1.3, 1.3},
	}
}

// predict implements decl.PredictFunc.
func (p *predictor) predict(input string) (string, error) {
	if len(p.model.ExactMap) > 0 {
		if answer, ok := p.model.ExactMap[strings.ToLower(strings.TrimSpace(input))]; ok {
			return strings.TrimSpace(answer), nil
		}
	}
	con, err := buildConstraint(input)
	if err != nil {
		p.stats.kindUnderivable++
		return p.model.Predict(input, 64)
	}
	encH, encC, err := encodeInput(p.model, input)
	if err != nil {
		return "", err
	}
	greedy := p.decodeAttempt(encH, encC, con, 0, nil)
	if !p.roundtrip || !isSpecForm(input) {
		return greedy.text, nil
	}
	ref, err := decl.ParseSpecText(input)
	if err != nil {
		p.stats.rtSkipped++
		return greedy.text, nil
	}
	p.stats.rtItems++
	if rtExact(greedy.text, ref) {
		p.stats.attemptsHist[1]++
		return greedy.text, nil
	}
	candidates := []string{greedy.text}
	for attempt := 2; attempt <= p.maxAttempts; attempt++ {
		temp := p.sampleTemps[min(attempt-2, len(p.sampleTemps)-1)]
		rng := rand.New(rand.NewSource(p.attemptSeed(input, attempt)))
		out := p.decodeAttempt(encH, encC, con, temp, rng)
		candidates = append(candidates, out.text)
		if rtExact(out.text, ref) {
			p.stats.attemptsHist[attempt]++
			return out.text, nil
		}
	}
	p.stats.attemptsHist[p.maxAttempts]++
	return bestCandidate(candidates, ref), nil
}

func (p *predictor) attemptSeed(input string, attempt int) int64 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(input))
	return p.seed + int64(h.Sum64()%1_000_000) + int64(attempt)*7919
}

func (p *predictor) decodeAttempt(encH, encC *tensor.Tensor, con *constraint, temp float64, rng *rand.Rand) decodeOutcome {
	out := decode(p.model, encH, encC, con, temp, rng)
	p.stats.deadEndSteps += out.deadEnds
	p.stats.antiOverrides += out.antiOverrides
	if out.deadEnds > 0 {
		p.stats.deadEndItems++
	}
	return out
}

// rtExact reports whether a candidate round-trips to the reference
// spec under the eval's comparator (tag presence only).
func rtExact(candidate string, ref decl.Spec) bool {
	cat, _ := rtCompare(candidate, ref)
	return cat == decl.CatExact
}

// rtCompare extracts the candidate's spec and compares it with the
// reference via decl.CompareSpec with receiver names strict (the
// spec form carries them) and tags reduced to presence (tag content
// is not in the spec language — see decl.ParseSpecText).
func rtCompare(candidate string, ref decl.Spec) (string, string) {
	got, err := decl.ExtractSpec(candidate)
	if err != nil {
		return decl.CatUnparsable, err.Error()
	}
	return decl.CompareSpec(tagPresence(got), tagPresence(ref), true)
}

// tagPresence reduces every field tag to a presence marker.
func tagPresence(s decl.Spec) decl.Spec {
	out := s
	out.Fields = append([]decl.Field(nil), s.Fields...)
	for i := range out.Fields {
		if out.Fields[i].Tag != "" {
			out.Fields[i].Tag = "tagged"
		}
	}
	return out
}

// categoryRank orders comparator categories from best to worst for
// best-candidate selection.
func categoryRank(cat string) int {
	switch cat {
	case decl.CatExact:
		return 0
	case decl.CatWrongOrder:
		return 1
	case decl.CatTagMismatch:
		return 2
	case decl.CatWrongType:
		return 3
	case decl.CatMemberName:
		return 4
	case decl.CatMemberCount:
		return 5
	case decl.CatReceiverMismatch:
		return 6
	case decl.CatEmbedMismatch:
		return 7
	case decl.CatWrongName:
		return 8
	case decl.CatWrongKind:
		return 9
	default: // unparsable and anything unexpected
		return 10
	}
}

// bestCandidate returns the candidate the eval comparator scores
// best against the reference spec; ties keep the earliest attempt.
func bestCandidate(candidates []string, ref decl.Spec) string {
	best := candidates[0]
	bestRank := categoryRank(rtCat(candidates[0], ref))
	for _, cand := range candidates[1:] {
		if r := categoryRank(rtCat(cand, ref)); r < bestRank {
			best, bestRank = cand, r
		}
	}
	return best
}

func rtCat(candidate string, ref decl.Spec) string {
	cat, _ := rtCompare(candidate, ref)
	return cat
}
