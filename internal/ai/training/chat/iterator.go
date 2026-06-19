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
	pairs  []moe.TrainPair
	vocab  *mainvocab.Vocabulary
	unkID  int
	idx    int
	MaxLen int
	Epoch  int
}

func NewChatDataIterator(pairs []moe.TrainPair, vocab *mainvocab.Vocabulary, unkID int) *ChatDataIterator {
	// Shuffle pairs for better training
	rand.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	return &ChatDataIterator{
		pairs:  pairs,
		vocab:  vocab,
		unkID:  unkID,
		idx:    0,
		MaxLen: 80, // Default cap
	}
}

func (it *ChatDataIterator) HasNext() bool {
	return it.idx < len(it.pairs)
}

func (it *ChatDataIterator) Next() (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, *tensor.Tensor) {
	pair := it.pairs[it.idx]
	it.idx++

	// --- DYNAMIC AUGMENTATION ---
	// (Keeping the existing augmentation logic)
	q := pair.Q
	a := pair.A
	if rand.Float32() < 0.3 {
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

	// Query Format: Normalized structure with intent markers
	queryText := fmt.Sprintf("__intent__ %s : __ques__ %s __ans__", pair.Intent, q)
	qTokens := cleanTokenize(queryText)
	qIDs := make([]float32, len(qTokens))
	for i, t := range qTokens {
		qIDs[i] = float32(lookupVocab(t, it.vocab))
	}

	// Target Format: Raw answer from dataset
	targetText := a
	aTokens := cleanTokenize(targetText)
	aIDs := make([]float32, len(aTokens)+2)
	aIDs[0] = float32(it.vocab.BosID)
	for i, t := range aTokens {
		aIDs[i+1] = float32(lookupVocab(t, it.vocab))
	}
	aIDs[len(aIDs)-1] = float32(it.vocab.EosID)

	// Grammar Tags: Map role strings to indices
	aRoles := SimpleTagger(aTokens)
	gIDs := make([]float32, len(aIDs))
	gIDs[0] = 7 // BOS -> OTHER
	for i := 0; i < len(aTokens); i++ {
		//  Syntactic Boost: Bias toward linking PRON to VERB
		role := moe.GrammarRoleIndex(aRoles[i])
		if i > 0 && aRoles[i-1] == "PRON" && (aRoles[i] == "VERB" || aRoles[i] == "AUX") {
			gIDs[i+1] = float32(role) + 0.5 // Boost signal
		} else {
			gIDs[i+1] = float32(role)
		}
	}
	gIDs[len(gIDs)-1] = 7 // EOS -> OTHER

	// Query Grammar Tags: Map query tokens to roles
	qRoles := SimpleTagger(qTokens)
	qgIDs := make([]float32, len(qIDs))
	for i := 0; i < len(qTokens); i++ {
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
		for j := 0; j < maxIn; j++ {
			if j < len(inputs[i]) {
				paddedIn[i*maxIn+j] = inputs[i][j]
				paddedQueryGrammar[i*maxIn+j] = queryGrammars[i][j]
				inputLogitMask[i*maxIn+j] = 0.0
			} else {
				paddedIn[i*maxIn+j] = padID
				paddedQueryGrammar[i*maxIn+j] = -1 // Padding (ignore in routing loss)
				inputLogitMask[i*maxIn+j] = -1e9
			}
		}
		for j := 0; j < maxOut; j++ {
			if j < len(targets[i]) {
				paddedOut[i*maxOut+j] = targets[i][j]
				paddedGrammar[i*maxOut+j] = grammars[i][j]
				mask[i*maxOut+j] = 1.0
				// All real answer tokens contribute to the loss (1.0).
				// BOS/EOS at position 0 or last are kept — the model must learn sequence boundaries.
				lossMask[i*maxOut+j] = 1.0
			} else {
				paddedOut[i*maxOut+j] = padID
				paddedGrammar[i*maxOut+j] = -1 // Padding (ignore in routing loss)
				mask[i*maxOut+j] = 0.0
				lossMask[i*maxOut+j] = 0.0 // Never train on padding
			}
		}
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
