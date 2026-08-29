package chat

import (
	"fmt"
	"log"
	"math/rand"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

type Batch struct {
	Input        *tensor.Tensor // Shape: [BatchSize, MaxInputLen]
	Target       *tensor.Tensor // Shape: [BatchSize, MaxTargetLen]
	Grammar      *tensor.Tensor // Shape: [BatchSize, MaxTargetLen] (Ground-truth POS tags)
	QueryGrammar *tensor.Tensor // Shape: [BatchSize, MaxInputLen] (Ground-truth POS tags for query)
	Mask         []float32      // To tell the loss function to ignore <pad>
	LossMask     []float32      // 1.0 = compute gradient (assistant tokens), 0.0 = skip (user/control tokens)
	InputMask    *tensor.Tensor // Attention mask (0.0 for real, -1e9 for pad)
	Intents      []string       // Stored intent labels for RuleBook matching
	Weights      []float32      // Sample weights for evolutionary data control
}

type ChatDataIterator struct {
	pairs       []moe.TrainPair
	vocab       *mainvocab.Vocabulary
	unkID       int
	idx         int
	MaxLen      int
	Epoch       int
	PureSeq2Seq bool
}

func NewChatDataIterator(pairs []moe.TrainPair, vocab *mainvocab.Vocabulary, unkID int, pureSeq2Seq bool) *ChatDataIterator {
	// Shuffle pairs for better training
	rand.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	return &ChatDataIterator{
		pairs:       pairs,
		vocab:       vocab,
		unkID:       unkID,
		idx:         0,
		MaxLen:      48, // Reduced from 80: attention is O(seq²), saves ~64% attention memory
		PureSeq2Seq: pureSeq2Seq,
	}
}

func (it *ChatDataIterator) HasNext() bool {
	return it.idx < len(it.pairs)
}

// buildChatMLSequence builds a ChatML-formatted token sequence with loss mask
// from a TrainPair. Format:
//
//	<|im_start|>user\nQUERY<|im_end|>\n<|im_start|>assistant\nANSWER<|im_end|>
//
// lossMask[i] = 0.0 for user tokens, 1.0 for assistant tokens.
func (it *ChatDataIterator) buildChatMLSequence(pair moe.TrainPair) (inputIDs, targetIDs, lossMask []float32) {
	// Build ChatML: user turn + assistant turn
	userMsg := fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", pair.Q)
	assistantMsg := fmt.Sprintf("%s<|im_end|>", pair.A)

	// Tokenize user message (not used for loss)
	userTokens := cleanTokenize(userMsg)
	userIDs := make([]float32, len(userTokens))
	for i, t := range userTokens {
		userIDs[i] = float32(lookupVocab(t, it.vocab))
	}

	// Tokenize assistant message (used for loss) — convert to float32 IDs
	assistantTokens := cleanTokenize(assistantMsg)
	assistantIDs := make([]float32, len(assistantTokens))
	for i, t := range assistantTokens {
		assistantIDs[i] = float32(lookupVocab(t, it.vocab))
	}

	// Full sequence: user part (no loss) + assistant part (loss applied)
	fullTokens := make([]float32, len(userIDs)+len(assistantIDs))
	copy(fullTokens, userIDs)
	copy(fullTokens[len(userIDs):], assistantIDs)

	// Loss mask: 0 for user tokens, 1 for assistant tokens
	lm := make([]float32, len(fullTokens))
	for i := 0; i < len(userIDs); i++ {
		lm[i] = 0.0
	}
	for i := len(userIDs); i < len(fullTokens); i++ {
		lm[i] = 1.0
	}

	return fullTokens, fullTokens, lm
}

func (it *ChatDataIterator) Next() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	pair := it.pairs[it.idx]
	it.idx++

	q := pair.Q
	a := pair.A
	if !it.PureSeq2Seq && rand.Float32() < 0.3 {
		synonyms := map[string]string{
			"hello": "hi", "how are you": "how are you doing", "goodbye": "bye",
			"who are you": "what is your name", "what is your name": "who are you",
		}
		for old, neu := range synonyms {
			if strings.Contains(strings.ToLower(q), old) {
				q = strings.ReplaceAll(q, old, neu)
				break
			}
		}
	}

	if it.PureSeq2Seq {
		qTokens := cleanTokenize(q)
		qIDs := make([]float32, len(qTokens)+2)
		qIDs[0] = float32(it.vocab.BosID)
		for i, tok := range qTokens {
			qIDs[i+1] = float32(lookupVocab(tok, it.vocab))
		}
		qIDs[len(qIDs)-1] = float32(it.vocab.EosID)

		aTokens := cleanTokenize(a)
		aIDs := make([]float32, len(aTokens)+2)
		aIDs[0] = float32(it.vocab.BosID)
		for i, tok := range aTokens {
			aIDs[i+1] = float32(lookupVocab(tok, it.vocab))
		}
		aIDs[len(aIDs)-1] = float32(it.vocab.EosID)

		inputTensor := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
		targetTensor := tensor.NewTensor([]int{1, len(aIDs)}, aIDs, false)
		return inputTensor, targetTensor, tensor.NewTensor([]int{1, 1}, []float32{0}, false), tensor.NewTensor([]int{1, 1}, []float32{0}, false)
	}

	augmentedPair := moe.TrainPair{Q: q, A: a, Intent: pair.Intent}
	fullIDs, _, _ := it.buildChatMLSequence(augmentedPair)
	qIDs := fullIDs

	targetText := a
	aTokens := cleanTokenize(targetText)
	aIDs := make([]float32, len(aTokens)+2)
	aIDs[0] = float32(it.vocab.BosID)
	for i, t := range aTokens {
		aIDs[i+1] = float32(lookupVocab(t, it.vocab))
	}
	aIDs[len(aIDs)-1] = float32(it.vocab.EosID)

	aRoles := SimpleTagger(aTokens)
	gIDs := make([]float32, len(aIDs))
	gIDs[0] = 7
	for i := 0; i < len(aTokens); i++ {
		role := moe.GrammarRoleIndex(aRoles[i])
		if i > 0 && aRoles[i-1] == "PRON" && (aRoles[i] == "VERB" || aRoles[i] == "AUX") {
			gIDs[i+1] = float32(role) + 0.5
		} else {
			gIDs[i+1] = float32(role)
		}
	}
	gIDs[len(gIDs)-1] = 7

	qTokens := cleanTokenize(q)
	qRoles := SimpleTagger(qTokens)
	qgIDs := make([]float32, len(qIDs))
	for i := 0; i < len(qTokens) && i < len(qIDs); i++ {
		role := moe.GrammarRoleIndex(qRoles[i])
		qgIDs[i] = float32(role)
	}

	inputTensor := tensor.NewTensor([]int{1, len(qIDs)}, qIDs, false)
	targetTensor := tensor.NewTensor([]int{1, len(aIDs)}, aIDs, false)
	grammarTensor := tensor.NewTensor([]int{1, len(gIDs)}, gIDs, false)
	queryGrammarTensor := tensor.NewTensor([]int{1, len(qgIDs)}, qgIDs, false)
	return inputTensor, targetTensor, grammarTensor, queryGrammarTensor
}

func (it *ChatDataIterator) NextBatch(batchSize int) *Batch {
	var inputs [][]float32
	var targets [][]float32
	var grammars [][]float32
	var queryGrammars [][]float32
	var weights []float32
	var intents []string
	maxIn, maxOut := 0, 0

	for i := 0; i < batchSize && it.HasNext(); i++ {
		// Access pair directly to get intent and weight
		pair := it.pairs[it.idx]
		inp, tgt, gmr, qgmr := it.Next()
		// Sequence length constraint: respects curriculum limit
		if len(inp.Data) > it.MaxLen || len(tgt.Data) > it.MaxLen {
			continue
		}
		inputs = append(inputs, inp.Data)
		targets = append(targets, tgt.Data)
		grammars = append(grammars, gmr.Data)
		queryGrammars = append(queryGrammars, qgmr.Data)
		intents = append(intents, pair.Intent)

		w := pair.Weight
		if w == 0 {
			w = 1.0 // Default weight
		}
		weights = append(weights, w)

		if len(inp.Data) > maxIn {
			maxIn = len(inp.Data)
		}
		if len(tgt.Data) > maxOut {
			maxOut = len(tgt.Data)
		}
	}

	if len(inputs) == 0 {
		return &Batch{}
	}

	// SIMD alignment: round maxIn and maxOut up to the nearest multiple of 8
	// so that low-level vector operations always operate on aligned memory blocks.
	const simdAlign = 8
	if maxIn%simdAlign != 0 {
		maxIn = (maxIn/simdAlign + 1) * simdAlign
	}
	if maxOut%simdAlign != 0 {
		maxOut = (maxOut/simdAlign + 1) * simdAlign
	}

	paddedIn := make([]float32, len(inputs)*maxIn)
	paddedOut := make([]float32, len(targets)*maxOut)
	paddedGrammar := make([]float32, len(grammars)*maxOut)
	paddedQueryGrammar := make([]float32, len(queryGrammars)*maxIn)
	mask := make([]float32, len(targets)*maxOut)
	// LossMask: 1.0 for real answer tokens, 0.0 for pad positions.
	// The query (input) is never included in the target slice, so the target mask
	// already acts as the correct loss mask — real answer tokens get 1.0, padding 0.0.
	lossMask := make([]float32, len(targets)*maxOut)
	inputLogitMask := make([]float32, len(inputs)*maxIn) // For attention: 0 for real, -1e9 for pad
	padID := float32(it.vocab.PaddingTokenID)

	for i := range inputs {
		// --- Input row: copy real data, leave rest as padID ---
		inBase := i * maxIn
		copy(paddedIn[inBase:], inputs[i])
		// Fill tail with padID using a fast loop (Go will inline/vectorise this)
		for j := len(inputs[i]); j < maxIn; j++ {
			paddedIn[inBase+j] = padID
		}
		// Query grammar: copy real, fill tail with -1 (pad signal)
		copy(paddedQueryGrammar[inBase:], queryGrammars[i])
		for j := len(queryGrammars[i]); j < maxIn; j++ {
			paddedQueryGrammar[inBase+j] = -1
		}
		// Attention mask: 0 for real tokens, -1e9 for padding
		for j := 0; j < len(inputs[i]); j++ {
			inputLogitMask[inBase+j] = 0.0
		}
		for j := len(inputs[i]); j < maxIn; j++ {
			inputLogitMask[inBase+j] = -1e9
		}

		// --- Target row ---
		outBase := i * maxOut
		copy(paddedOut[outBase:], targets[i])
		for j := len(targets[i]); j < maxOut; j++ {
			paddedOut[outBase+j] = padID
		}
		// Grammar: copy real, fill tail with -1
		copy(paddedGrammar[outBase:], grammars[i])
		for j := len(grammars[i]); j < maxOut; j++ {
			paddedGrammar[outBase+j] = -1
		}
		// Mask/LossMask: 1 for real, 0 for padding
		realLen := len(targets[i])
		for j := 0; j < realLen; j++ {
			mask[outBase+j] = 1.0
			lossMask[outBase+j] = 1.0
		}
		// tail is already zero (slices zero-initialised by make)
	}

	// Reshape InputMask for attention: [Batch, 1, 1, SeqLen]
	inputMaskTensor := tensor.NewTensor([]int{len(inputs), 1, 1, maxIn}, inputLogitMask, false)

	return &Batch{
		Input:        tensor.NewTensor([]int{len(inputs), maxIn}, paddedIn, false),
		Target:       tensor.NewTensor([]int{len(targets), maxOut}, paddedOut, false),
		Grammar:      tensor.NewTensor([]int{len(grammars), maxOut}, paddedGrammar, false),
		QueryGrammar: tensor.NewTensor([]int{len(queryGrammars), maxIn}, paddedQueryGrammar, false),
		Mask:         mask,
		LossMask:     lossMask,
		InputMask:    inputMaskTensor,
		Intents:      intents,
		Weights:      weights,
	}
}

func (it *ChatDataIterator) Reset() {
	it.idx = 0
	rand.Shuffle(len(it.pairs), func(i, j int) { it.pairs[i], it.pairs[j] = it.pairs[j], it.pairs[i] })
	log.Println(" Shuffled training data for new epoch")
}
