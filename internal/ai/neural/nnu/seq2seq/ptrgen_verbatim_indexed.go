package seq2seq

// Indexed-UNK verbatim decoding for phrasing attempt 2
// (2026-10-31). Run 1's verbatim decoder had a structural
// blindness: every out-of-vocabulary input word encodes to the
// same UNK id, so a sentence carrying TWO unseen slot values
// ("add instantiation of struct Tend to handler Sift", both
// names unseen) shows the encoder two identical tokens, and
// which one a copied UNK means is left to attention argmax —
// a guess that swaps type and handler whenever attention
// wavers.
//
// The indexed discipline removes the ambiguity. The model's
// vocabulary carries MaxIndexedUnk placeholder tokens,
// unk1..unk8. Encoding assigns each DISTINCT unknown word the
// next index in order of first appearance in the input; the
// trainer masks slot values with the same per-value index in
// input and target (see the phrasing2 trainer), so emitting
// unk<k> is a practiced, unambiguous act: "copy the k-th
// unknown". Decoding substitutes the k-th unknown's surface
// word deterministically — no attention guess. Plain UNK keeps
// run 1's attention substitution (wrapper noise trained it).
//
// Everything else — greedy mixture decoding, per-piece bigram
// reset at "##", surface-case restoration for known copied
// words — is PredictPtrGenVerbatim's discipline, unchanged.

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// MaxIndexedUnk is how many distinct unknown words one input
// can carry as indexed placeholders. Compounds bind at most
// six slot values; eight leaves room for an unseen wrapper
// word or two ranked ahead of them.
const MaxIndexedUnk = 8

// IndexedUnkToken names the k-th (1-based) indexed placeholder.
func IndexedUnkToken(k int) string { return "unk" + strconv.Itoa(k) }

// IndexedInputEncoding is one input prepared for the indexed
// decoder: the surface words, the id stream the encoder sees,
// and the restore map from placeholder index to the surface
// word that unknown was typed as.
type IndexedInputEncoding struct {
	Surface []string
	IDs     []int
	Restore map[int]string
}

// EncodeIndexedInput tokenizes a query verbatim and replaces
// each distinct out-of-vocabulary word with its indexed
// placeholder id, ranked by first appearance. Words beyond
// MaxIndexedUnk distinct unknowns fall back to plain UNK.
// Known words keep their vocabulary ids (the model's world,
// lowercased — surface case rides in Surface for restoration).
func (m *Seq2Seq) EncodeIndexedInput(query string) IndexedInputEncoding {
	v := m.OutputVocab
	surface := VerbatimTokens(query)
	enc := IndexedInputEncoding{
		Surface: surface,
		IDs:     make([]int, len(surface)),
		Restore: map[int]string{},
	}
	indexOf := map[string]int{}
	for i, w := range surface {
		low := strings.ToLower(w)
		id := v.GetTokenID(low)
		if id != v.UnkID {
			enc.IDs[i] = id
			continue
		}
		k, seen := indexOf[low]
		if !seen {
			k = len(indexOf) + 1
			indexOf[low] = k
			if k <= MaxIndexedUnk {
				enc.Restore[k] = w
			}
		}
		if k <= MaxIndexedUnk {
			if pid := v.GetTokenID(IndexedUnkToken(k)); pid != v.UnkID {
				enc.IDs[i] = pid
				continue
			}
		}
		enc.IDs[i] = v.UnkID
	}
	return enc
}

// indexedWord chooses the emitted word for one decode step of
// PredictPtrGenVerbatimIndexed: an indexed placeholder restores
// its own unknown deterministically; anything else follows
// verbatimWord's rule (plain UNK and copy-carried wins restore
// the attended input's surface word; generated words come from
// the vocabulary).
func (m *Seq2Seq) indexedWord(best int, cache *ptrGenCache, enc IndexedInputEncoding) string {
	v := m.OutputVocab
	if w := v.GetWord(best); strings.HasPrefix(w, "unk") {
		if k, err := strconv.Atoi(strings.TrimPrefix(w, "unk")); err == nil && k >= 1 {
			if surface, ok := enc.Restore[k]; ok {
				return surface
			}
			return w
		}
	}
	return verbatimWord(best, cache, enc.IDs, enc.Surface, v.GetWord, v.UnkID)
}

// PredictPtrGenVerbatimIndexed is PredictPtrGenVerbatim over
// the indexed encoding: distinct unknowns are distinct tokens
// end to end, and an emitted placeholder restores its own
// surface word. The decoder's feedback is always the winning
// token ID, never the substituted word.
func (m *Seq2Seq) PredictPtrGenVerbatimIndexed(query string, maxLen int) (string, error) {
	if m.Decoder == nil || m.Decoder.PtrGen == nil {
		return "", fmt.Errorf("seq2seq PredictPtrGenVerbatimIndexed: model has no pointer-generator head")
	}
	if m.OutputVocab == nil {
		return "", fmt.Errorf("seq2seq PredictPtrGenVerbatimIndexed: model has no output vocabulary")
	}
	pg := m.Decoder.PtrGen
	H := m.HiddenDim
	v := m.OutputVocab

	enc := m.EncodeIndexedInput(query)
	if len(enc.Surface) == 0 {
		return "", fmt.Errorf("empty query")
	}
	inputIDs := enc.IDs
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
			return "", fmt.Errorf("indexed decoder embedding failed at step %d: %w", t, err)
		}
		emb2D, err := emb.Reshape([]int{1, emb.Shape[2]})
		if err != nil {
			return "", fmt.Errorf("indexed decoder reshape at step %d: %w", t, err)
		}
		newHidden, newCell, err := m.Decoder.LSTM.Forward(emb2D, hidden, cell)
		if err != nil {
			return "", fmt.Errorf("indexed decoder LSTM failed at step %d: %w", t, err)
		}
		logitsT, err := m.Decoder.Output.Forward(newHidden)
		if err != nil {
			return "", fmt.Errorf("indexed output projection failed at step %d: %w", t, err)
		}
		final, cache := pg.ptrGenStep(newHidden.Data[:H], encStates, inputIDs, len(inputIDs), logitsT.Data)

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
		words = append(words, m.indexedWord(best, cache, enc))
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
