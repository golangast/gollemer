package seq2seq

// Verbatim decoding for the pointer-generator: PredictPtrGen
// with the output kept as SURFACE words from the user's own
// input wherever the copy branch did the work.
//
// Why this exists (the phrasing layer, 2026-10-30): the layer's
// held-out names are, by design, words the vocabulary has never
// seen — they encode to the UNK id, and the plain decoder can
// only ever hand back the string "UNK" for them. But the copy
// mechanism knows exactly WHERE the name sits: the attention
// distribution points at its input position. So when the
// winning token is UNK, the emitted word is the input word at
// the most-attended position. The same substitution applies
// when a known token wins mostly on copy mass — the vocabulary
// only knows names lowercased (training lowercases), so the
// surface word restores the case the user typed ("Jim", not
// "jim"). Generated words (grammar words the generation branch
// produced) come from the vocabulary as usual.
//
// Training meets this halfway: the phrasing trainer masks slot
// tokens to UNK in a fraction of its pairs, so "emit UNK by
// copying the attended position" is a behavior the model
// actually practiced, and the decoder state fed back after a
// copied word is the UNK id — exactly the masked-training
// condition.
//
// One discipline differs from PredictPtrGen: the repeated-
// bigram mask RESETS at the "##" piece separator. A compound
// answer legitimately repeats bigrams across its pieces ("add
// a … ## add a …"), and the mask's job — stopping degenerate
// loops — is per-piece work; the layer above validates every
// piece against the real parsers anyway.

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// verbatimSep is the piece separator of compound outputs. It
// must be a single token in the model's world: '#' is an
// ordinary word character to the tokenizer, so "##" is one
// token.
const verbatimSep = "##"

// VerbatimTokens splits a query into the words the verbatim
// decoder can copy back, with each word's original case kept.
// The splitting follows the tokenizer's rules (whitespace
// words, structural punctuation {}[]" as single tokens), with
// one deliberate difference from NormalizeQuery: a sentence
// mark (. ! ? , ; :) is dropped only at a word boundary —
// inside a word it stays, so "jim/main.go" and "time.Duration"
// survive as the single tokens the vocabulary knows them as.
// (NormalizeQuery drops dots everywhere, which would mangle
// file names into words the copy mechanism cannot restore.)
// The phrasing trainer encodes its inputs with this same
// function, so training and inference see identical streams.
func VerbatimTokens(query string) []string {
	runes := []rune(query)
	var toks []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			toks = append(toks, cur.String())
			cur.Reset()
		}
	}
	isMark := func(r rune) bool {
		switch r {
		case '.', '!', '?', ',', ';', ':':
			return true
		}
		return false
	}
	for i, r := range runes {
		switch {
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			flush()
		case r == '{' || r == '}' || r == '[' || r == ']' || r == '"':
			flush()
			toks = append(toks, string(r))
		case isMark(r):
			// A mark inside a word (followed by a letter or
			// digit) belongs to the word; at a boundary it is
			// sentence punctuation and drops.
			if i+1 < len(runes) && (runes[i+1] >= 'a' && runes[i+1] <= 'z' ||
				runes[i+1] >= 'A' && runes[i+1] <= 'Z' ||
				runes[i+1] >= '0' && runes[i+1] <= '9') {
				cur.WriteRune(r)
			}
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return toks
}

// verbatimWord chooses the emitted word for one decode step:
// the input's own surface word when the copy branch carried
// the win (or the winner is UNK, which only copying can mean),
// the vocabulary word otherwise.
func verbatimWord(best int, cache *ptrGenCache, inputIDs []int, surface []string, wordOf func(int) string, unkID int) string {
	if best == unkID {
		pos := argmaxFloat(cache.attn)
		if pos >= 0 && pos < len(surface) {
			return surface[pos]
		}
		return wordOf(best)
	}
	copyMass := (1 - cache.pGen) * cache.copyDist[best]
	genMass := cache.pGen * cache.pVocab[best]
	if copyMass > genMass {
		bestPos, bestAttn := -1, float32(-1)
		for i, id := range inputIDs {
			if id == best && i < len(cache.attn) && cache.attn[i] > bestAttn {
				bestPos, bestAttn = i, cache.attn[i]
			}
		}
		if bestPos >= 0 && bestPos < len(surface) {
			return surface[bestPos]
		}
	}
	return wordOf(best)
}

func argmaxFloat(xs []float32) int {
	if len(xs) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(xs); i++ {
		if xs[i] > xs[best] {
			best = i
		}
	}
	return best
}

// PredictPtrGenVerbatim greedily decodes from the
// pointer-generator mixture like PredictPtrGen, but returns
// the output with copied words in the input's own spelling
// (see the file header). The decoder's feedback is always the
// winning token ID, never the substituted word — the model's
// world stays ids end to end.
func (m *Seq2Seq) PredictPtrGenVerbatim(query string, maxLen int) (string, error) {
	if m.Decoder == nil || m.Decoder.PtrGen == nil {
		return "", fmt.Errorf("seq2seq PredictPtrGenVerbatim: model has no pointer-generator head")
	}
	if m.OutputVocab == nil {
		return "", fmt.Errorf("seq2seq PredictPtrGenVerbatim: model has no output vocabulary")
	}
	pg := m.Decoder.PtrGen
	H := m.HiddenDim
	v := m.OutputVocab

	surface := VerbatimTokens(query)
	if len(surface) == 0 {
		return "", fmt.Errorf("empty query")
	}
	inputIDs := make([]int, len(surface))
	for i, w := range surface {
		inputIDs[i] = v.GetTokenID(strings.ToLower(w))
	}
	sepID := v.GetTokenID(verbatimSep)

	encStates, hidden, cell, err := m.encodePositions(inputIDs)
	if err != nil {
		return "", err
	}

	words := []string{}
	seenBigrams := map[[2]int]bool{}
	current := v.BosID
	prev := -1

	for t := 0; t < maxLen; t++ {
		emb, err := m.Decoder.Embedding.Forward(tensor.NewTensor([]int{1, 1}, []float32{float32(current)}, true))
		if err != nil {
			return "", fmt.Errorf("verbatim decoder embedding failed at step %d: %w", t, err)
		}
		emb2D, err := emb.Reshape([]int{1, emb.Shape[2]})
		if err != nil {
			return "", fmt.Errorf("verbatim decoder embedding reshape at step %d: %w", t, err)
		}
		newHidden, newCell, err := m.Decoder.LSTM.Forward(emb2D, hidden, cell)
		if err != nil {
			return "", fmt.Errorf("verbatim decoder LSTM failed at step %d: %w", t, err)
		}
		logitsT, err := m.Decoder.Output.Forward(newHidden)
		if err != nil {
			return "", fmt.Errorf("verbatim output projection failed at step %d: %w", t, err)
		}
		final, cache := pg.ptrGenStep(newHidden.Data[:H], encStates, inputIDs, len(inputIDs), logitsT.Data)

		// Anti-repeat masking within the current piece,
		// mirroring PredictPtrGen; the separator resets it.
		if prev >= 0 {
			if prev < len(final) {
				final[prev] = 0
			}
			for c := 0; c < len(final); c++ {
				if seenBigrams[[2]int{prev, c}] {
					final[c] = 0
				}
			}
		}
		best := 0
		for i := 1; i < len(final); i++ {
			if final[i] > final[best] {
				best = i
			}
		}
		if best == v.EosID {
			break
		}
		if prev >= 0 {
			seenBigrams[[2]int{prev, best}] = true
		}
		words = append(words, verbatimWord(best, cache, inputIDs, surface, v.GetWord, v.UnkID))
		if best == sepID {
			seenBigrams = map[[2]int]bool{}
			prev = -1
		} else {
			prev = best
		}
		current = best
		hidden, cell = newHidden, newCell
	}
	return strings.TrimSpace(strings.Join(words, " ")), nil
}
