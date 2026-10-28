package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Step-wise decoding with an automaton mask. The loop mirrors
// seq2seq.PredictWithTrace exactly — same input normalization, same
// encoder pass, same anti-degeneration masks (no immediate repeat,
// no repeated bigram), same copy-history handling, same 64-token
// cap — except that at each step the automaton's admissible set is
// intersected with the model's choices. With the constraint disabled
// (nil), the loop reproduces model.Predict token for token (verified
// against the snapshot checkpoint during development).

// wordSyms is the per-vocabulary-id symbolization, built once.
var wordSyms [][]sym

func initWordSyms(m *seq2seq.Seq2Seq) {
	v := m.OutputVocab
	wordSyms = make([][]sym, v.Size())
	for id := 0; id < v.Size(); id++ {
		wordSyms[id] = symbolizeWord(v.GetWord(id))
	}
}

// encodeInput runs the encoder half of Predict for one input.
func encodeInput(m *seq2seq.Seq2Seq, input string) (*tensor.Tensor, *tensor.Tensor, error) {
	query := seq2seq.NormalizeQuery(input)
	inputTokenIDs, err := m.Tokenizer.Encode(query)
	if err != nil {
		return nil, nil, fmt.Errorf("tokenize: %w", err)
	}
	data := make([]float32, len(inputTokenIDs))
	for i, id := range inputTokenIDs {
		data[i] = float32(id)
	}
	enc, cell, err := m.Encoder.Forward(tensor.NewTensor([]int{1, len(inputTokenIDs)}, data, true))
	if err != nil {
		return nil, nil, fmt.Errorf("encoder: %w", err)
	}
	return enc, cell, nil
}

func cloneTensor(t *tensor.Tensor) *tensor.Tensor {
	return tensor.NewTensor(append([]int(nil), t.Shape...), append([]float32(nil), t.Data...), true)
}

type decodeOutcome struct {
	text          string
	deadEnds      int  // steps where the constraint admitted no token at all
	antiOverrides int  // steps where anti-repeat masked every admissible token
	abandoned     bool // constraint dropped after a dead end
}

// maskStep computes, for every vocabulary id, whether emitting it
// keeps a live automaton configuration, and the resulting
// configurations when it does.
func maskStep(m *seq2seq.Seq2Seq, cfgs []config, con *constraint) ([]bool, [][]config) {
	v := m.OutputVocab
	allowed := make([]bool, v.Size())
	stepped := make([][]config, v.Size())
	done := acceptsEOS(cfgs)
	for id := 0; id < v.Size(); id++ {
		syms := wordSyms[id]
		if len(syms) == 1 && syms[0].kind == sEOS {
			allowed[id] = done
			continue
		}
		cur := cfgs
		ok := true
		for _, s := range syms {
			cur = stepAll(cur, s, con)
			if len(cur) == 0 {
				ok = false
				break
			}
		}
		if ok {
			allowed[id] = true
			stepped[id] = cur
		}
	}
	return allowed, stepped
}

// decode runs one generation. temp <= 0 selects greedy decoding;
// temp > 0 samples from the admissible set with that temperature.
func decode(m *seq2seq.Seq2Seq, encH, encC *tensor.Tensor, con *constraint, temp float64, rng *rand.Rand) decodeOutcome {
	var out decodeOutcome
	hidden, cell := cloneTensor(encH), cloneTensor(encC)
	v := m.OutputVocab
	var tokens []int
	cur := v.BosID
	seenBigrams := map[[2]int]bool{}
	useCopy := m.Decoder.Copy != nil
	histStates := [][][]float32{{}}
	histTokens := [][]int{{}}
	var cfgs []config
	if con != nil {
		cfgs = []config{{phase: phStart}}
	}
	for t := 0; t < 64; t++ {
		in := tensor.NewTensor([]int{1, 1}, []float32{float32(cur)}, true)
		pred, h, cl, err := m.Decoder.ForwardCopy(in, hidden, cell, histStates, histTokens)
		if err != nil {
			break
		}
		logits := pred.Data
		// Anti-degeneration masks, identical to PredictWithTrace.
		masked := make([]bool, len(logits))
		if len(tokens) > 0 {
			prev := tokens[len(tokens)-1]
			if prev >= 0 && prev < len(logits) {
				logits[prev] = -1e30
				masked[prev] = true
			}
			for cand := 0; cand < len(logits); cand++ {
				if seenBigrams[[2]int{prev, cand}] {
					logits[cand] = -1e30
					masked[cand] = true
				}
			}
		}
		var allowed []bool
		var stepped [][]config
		if cfgs != nil {
			allowed, stepped = maskStep(m, cfgs, con)
		}
		chosen := chooseToken(logits, masked, allowed, temp, rng, &out)
		if chosen < 0 {
			break
		}
		if out.abandoned {
			cfgs = nil // dead end: constraint dropped for this decode
		}
		if len(tokens) > 0 {
			seenBigrams[[2]int{tokens[len(tokens)-1], chosen}] = true
		}
		tokens = append(tokens, chosen)
		if chosen == v.EosID {
			break
		}
		if cfgs != nil && stepped != nil && stepped[chosen] != nil {
			cfgs = stepped[chosen]
		}
		if useCopy {
			hs := append([]float32(nil), h.Data...)
			histStates[0] = append(histStates[0], hs)
			histTokens[0] = append(histTokens[0], chosen)
		}
		cur = chosen
		hidden, cell = h, cl
	}
	out.text = v.Decode(tokens)
	return out
}

// chooseToken picks the next token id: the automaton's admissible
// set intersected with the anti-repeat masks, with honest fallbacks
// (recorded in the outcome) when an intersection is empty.
func chooseToken(logits []float32, masked, allowed []bool, temp float64, rng *rand.Rand, out *decodeOutcome) int {
	admissible := func(id int) bool { return allowed == nil || allowed[id] }
	// First choice: admissible and not anti-repeat masked.
	if id := pick(logits, temp, rng, func(id int) bool {
		return admissible(id) && !masked[id]
	}); id >= 0 {
		return id
	}
	if allowed != nil {
		// The constraint admits tokens but anti-repeat masked them
		// all: the automaton wins (gold streams never repeat tokens
		// or bigrams, so this signals a model-side loop).
		if id := pick(logits, temp, rng, func(id int) bool {
			return allowed[id]
		}); id >= 0 {
			out.antiOverrides++
			return id
		}
		// True dead end: the constraint admits nothing. Abandon it
		// for the rest of this decode rather than emit nothing.
		out.deadEnds++
		out.abandoned = true
	}
	// Unconstrained fallback (or unconstrained mode): Predict's rule.
	return pick(logits, 0, nil, func(id int) bool { return !masked[id] })
}

// pick selects a token: argmax when temp <= 0, temperature sampling
// otherwise, over the ids the filter admits. -1 if none.
func pick(logits []float32, temp float64, rng *rand.Rand, admit func(int) bool) int {
	if temp <= 0 || rng == nil {
		best := -1
		var bestVal float32
		for id := range logits {
			if !admit(id) {
				continue
			}
			if best < 0 || logits[id] > bestVal {
				best, bestVal = id, logits[id]
			}
		}
		return best
	}
	maxLogit := math.Inf(-1)
	for id := range logits {
		if admit(id) && float64(logits[id]) > maxLogit {
			maxLogit = float64(logits[id])
		}
	}
	if math.IsInf(maxLogit, -1) {
		return -1
	}
	var sum float64
	weights := make([]float64, len(logits))
	for id := range logits {
		if !admit(id) {
			continue
		}
		w := math.Exp((float64(logits[id]) - maxLogit) / temp)
		weights[id] = w
		sum += w
	}
	if sum <= 0 {
		return -1
	}
	r := rng.Float64() * sum
	for id, w := range weights {
		if w == 0 {
			continue
		}
		r -= w
		if r <= 0 {
			return id
		}
	}
	for id := len(weights) - 1; id >= 0; id-- {
		if weights[id] > 0 {
			return id
		}
	}
	return -1
}
