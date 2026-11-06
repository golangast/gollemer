package seq2seq

import (
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// SerializableSeq2Seq is a struct for saving and loading the model.
type SerializableSeq2Seq struct {
	Encoder     *Encoder
	Decoder     *Decoder
	OutputVocab *vocab.Vocabulary
	HiddenDim   int
	ExactMap    map[string]string
}

// Encoder represents the encoder part of the Seq2Seq model.
type Encoder struct {
	Embedding *nn.Embedding
	LSTM      *nn.LSTM
	// Add other layers as needed
}

// NewEncoder creates a new Encoder.
func NewEncoder(inputVocabSize, embeddingDim, hiddenDim int) (*Encoder, error) {
	lstm, err := nn.NewLSTM(embeddingDim, hiddenDim, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM for encoder: %w", err)
	}
	return &Encoder{
		Embedding: nn.NewEmbedding(inputVocabSize, embeddingDim),
		LSTM:      lstm,
	}, nil
}

// Forward performs a forward pass through the encoder.
func (e *Encoder) Forward(inputIDs *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	// inputIDs: [batch_size, sequence_length]
	embedded, err := e.Embedding.Forward(inputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("encoder embedding forward failed: %w", err)
	}

	batchSize := inputIDs.Shape[0]
	seqLength := inputIDs.Shape[1]
	hiddenSize := e.LSTM.HiddenSize

	hidden := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), true)
	cell := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), true)

	for t := range seqLength {
		// Get the input for the current time step
		timeStepInput, err := embedded.Slice(1, t, t+1)
		if err != nil {
			return nil, nil, err
		}
		timeStepInput, err = timeStepInput.Reshape([]int{batchSize, e.Embedding.DimModel})
		if err != nil {
			return nil, nil, err
		}

		hidden, cell, err = e.LSTM.Forward(timeStepInput, hidden, cell)
		if err != nil {
			return nil, nil, fmt.Errorf("encoder LSTM forward failed at step %d: %w", t, err)
		}
	}

	return hidden, cell, nil
}

// Parameters returns all learnable parameters of the Encoder.
func (e *Encoder) Parameters() []*tensor.Tensor { // Changed here
	params := []*tensor.Tensor{} // Changed here
	params = append(params, e.Embedding.Parameters()...)
	params = append(params, e.LSTM.Parameters()...)
	return params
}

// Decoder represents the decoder part of the Seq2Seq model.
type Decoder struct {
	Embedding *nn.Embedding
	LSTM      *nn.LSTM
	MoE       *nn.MoE // mixture-of-experts refines the LSTM state before projection
	Output    *nn.Linear
	// Copy is the pointer/copy mechanism over the decoder's own output
	// history. Nil disables it (all non-gocode domains): decoding is then
	// exactly the pre-copy behavior, so old checkpoints stay valid.
	Copy *CopyGate
	// PtrGen is the pointer-generator head over the ENCODER's input
	// positions (see ptrgen.go). Nil for every domain except the
	// speccopy probe: training and prediction then use the baseline
	// paths, so old checkpoints stay valid.
	PtrGen *PointerGen
}

// MoE configuration for the decoder.
const (
	decoderMoENumExperts = 4
	decoderMoETopK       = 2
)

// NewDecoder creates a new Decoder.
func NewDecoder(outputVocabSize, embeddingDim, hiddenDim int) (*Decoder, error) {
	outputLayer, err := nn.NewLinear(hiddenDim, outputVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer for decoder: %w", err)
	}
	lstm, err := nn.NewLSTM(embeddingDim, hiddenDim, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM for decoder: %w", err)
	}
	moe, err := nn.NewMoE(hiddenDim, decoderMoENumExperts, decoderMoETopK, hiddenDim/2)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE layer for decoder: %w", err)
	}
	return &Decoder{
		Embedding: nn.NewEmbedding(outputVocabSize, embeddingDim),
		LSTM:      lstm, // Input to LSTM will be embedded token + context
		MoE:       moe,
		Output:    outputLayer,
	}, nil
}

// Forward performs a forward pass through the decoder.
// It takes the previous output token, and the encoder's final hidden and cell states.
func (d *Decoder) Forward(inputTokenID *tensor.Tensor, hidden, cell *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	return d.ForwardCopy(inputTokenID, hidden, cell, nil, nil)
}

// ForwardCopy is Forward plus the copy mechanism. histStates[b] holds the
// decoder's own past LSTM states for batch row b and histTokens[b] the token
// ids that produced them; when d.Copy is nil (or history is empty) the copy
// bias is zero and this is exactly Forward.
func (d *Decoder) ForwardCopy(inputTokenID *tensor.Tensor, hidden, cell *tensor.Tensor, histStates [][][]float32, histTokens [][]int) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	// inputTokenID: [batch_size, 1] (single token ID)
	embedded, err := d.Embedding.Forward(inputTokenID)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder embedding forward failed: %w", err)
	}
	// embedded: [batch_size, 1, embedding_dim]

	// Reshape embedded to [batch_size, embedding_dim]
	embeddedReshaped, err := embedded.Reshape([]int{embedded.Shape[0], embedded.Shape[2]})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder embedding reshape failed: %w", err)
	}

	// LSTM expects input [batch_size, input_size]
	newHidden, newCell, err := d.LSTM.Forward(embeddedReshaped, hidden, cell)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder LSTM forward failed: %w", err)
	}
	// newHidden, newCell: [batch_size, hidden_dim]

	// Mixture-of-experts: route the LSTM state through the top-k experts
	// before projecting to vocabulary logits.
	moeIn, err := newHidden.Reshape([]int{newHidden.Shape[0], 1, newHidden.Shape[1]})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder MoE input reshape failed: %w", err)
	}
	moeOut, _, err := d.MoE.Forward(moeIn)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder MoE forward failed: %w", err)
	}
	moeOut2D, err := moeOut.Reshape([]int{moeOut.Shape[0], moeOut.Shape[2]})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder MoE output reshape failed: %w", err)
	}

	// Apply linear layer to the MoE-refined state
	prediction, err := d.Output.Forward(moeOut2D)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder output linear layer failed: %w", err)
	}
	// prediction: [batch_size, output_vocab_size] (logits for next token)

	// Copy mechanism: bias the logits toward tokens the decoder has
	// already emitted, so identifiers survive long-range dependencies.
	if d.Copy != nil && len(histStates) > 0 {
		batchSize := prediction.Shape[0]
		hiddenDim := moeOut2D.Shape[1]
		vocabSize := prediction.Shape[1]
		for b := 0; b < batchSize && b < len(histStates); b++ {
			var toks []int
			if b < len(histTokens) {
				toks = histTokens[b]
			}
			qOff := b * hiddenDim
			query := moeOut2D.Data[qOff : qOff+hiddenDim]
			lOff := b * vocabSize
			genLogits := prediction.Data[lOff : lOff+vocabSize]
			d.Copy.forwardCopyStep(query, genLogits, histStates[b], toks)
		}
	}

	return prediction, newHidden, newCell, nil
}

// Parameters returns all learnable parameters of the Decoder.
func (d *Decoder) Parameters() []*tensor.Tensor { // Changed here
	params := []*tensor.Tensor{} // Changed here
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.LSTM.Parameters()...)
	if d.MoE != nil {
		params = append(params, d.MoE.Parameters()...)
	}
	params = append(params, d.Output.Parameters()...)
	if d.Copy != nil {
		params = append(params, d.Copy.Parameters()...)
	}
	if d.PtrGen != nil {
		params = append(params, d.PtrGen.Parameters()...)
	}
	return params
}

// Seq2Seq represents the complete Encoder-Decoder model.
type Seq2Seq struct {
	Encoder     *Encoder
	Decoder     *Decoder
	Tokenizer   *tokenizer.Tokenizer // For encoding/decoding text
	OutputVocab *vocab.Vocabulary    // Vocabulary for the output descriptions
	HiddenDim   int
	ExactMap    map[string]string
	// BeamWidth > 1 makes PredictWithTrace decode by beam search
	// (cumulative log-probability, no length normalization) instead of
	// greedy argmax. Zero keeps the historical greedy loop. Only the
	// chat's social model opts in — beam width 4 was judged on the
	// social model alone (decode-upgrades probe, 2026-11-05: held-out
	// want-hits 9/41 vs greedy's 8/41, 3 judged wins to 1 loss, fully
	// deterministic); every other domain, and training/eval, keeps
	// greedy. Not serialized: a freshly loaded model stays greedy
	// until a caller opts in.
	BeamWidth int
}

// NewSeq2Seq creates a new Seq2Seq model.
func NewSeq2Seq(inputVocabSize, outputVocabSize, embeddingDim, hiddenDim int, tok *tokenizer.Tokenizer, outVocab *vocab.Vocabulary) (*Seq2Seq, error) {
	encoder, err := NewEncoder(inputVocabSize, embeddingDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	decoder, err := NewDecoder(outputVocabSize, embeddingDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	return &Seq2Seq{
		Encoder:     encoder,
		Decoder:     decoder,
		Tokenizer:   tok,
		OutputVocab: outVocab,
		HiddenDim:   hiddenDim,
		ExactMap:    make(map[string]string),
	}, nil
}

func (m *Seq2Seq) SetExactMap(mappings map[string]string) {
	if m == nil {
		return
	}
	if mappings == nil {
		m.ExactMap = make(map[string]string)
		return
	}
	m.ExactMap = mappings
}

// Parameters returns all learnable parameters of the Seq2Seq model.
func (m *Seq2Seq) Parameters() []*tensor.Tensor { // Changed here
	params := []*tensor.Tensor{} // Changed here
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.Decoder.Parameters()...)
	return params
}

// Forward performs a forward pass through the Seq2Seq model for training.
// It takes input sequence IDs and target output sequence IDs.
func (m *Seq2Seq) Forward(inputIDs, targetIDs *tensor.Tensor) (*tensor.Tensor, error) {
	// inputIDs: [batch_size, input_seq_len]
	// targetIDs: [batch_size, target_seq_len] (includes BOS and EOS)
	// Teacher forcing should predict the next token at each step, not the current token.
	// We feed targetIDs[:, t] and train against targetIDs[:, t+1], so the model learns
	// to map the prefix to the next token. This is the standard seq2seq objective.

	batchSize := inputIDs.Shape[0]
	targetSeqLen := targetIDs.Shape[1]
	if targetSeqLen <= 1 {
		return nil, fmt.Errorf("seq2seq target sequence must have at least 2 tokens (BOS + EOS)")
	}
	outputVocabSize := m.OutputVocab.Size()

	encoderHidden, encoderCell, err := m.Encoder.Forward(inputIDs)
	if err != nil {
		return nil, fmt.Errorf("seq2seq encoder forward failed: %w", err)
	}

	stepOutputs := make([]*tensor.Tensor, 0, targetSeqLen-1)
	decoderHidden := encoderHidden
	decoderCell := encoderCell

	// Copy history: per batch row, the decoder's own past LSTM states and
	// the token ids that produced them. Only used when the decoder carries
	// a copy gate.
	useCopy := m.Decoder.Copy != nil
	histStates := make([][][]float32, batchSize)
	histTokens := make([][]int, batchSize)

	for t := 0; t < targetSeqLen-1; t++ {
		decoderInputData := make([]float32, batchSize)
		for b := 0; b < batchSize; b++ {
			decoderInputData[b] = targetIDs.Data[b*targetSeqLen+t]
		}
		decoderInput := tensor.NewTensor([]int{batchSize, 1}, decoderInputData, true)

		prediction, hidden, cell, err := m.Decoder.ForwardCopy(decoderInput, decoderHidden, decoderCell, histStates, histTokens)
		if err != nil {
			return nil, fmt.Errorf("seq2seq decoder forward failed at step %d: %w", t, err)
		}

		reshapedPrediction, err := prediction.Reshape([]int{batchSize, 1, outputVocabSize})
		if err != nil {
			return nil, fmt.Errorf("seq2seq decoder reshape at step %d failed: %w", t, err)
		}
		stepOutputs = append(stepOutputs, reshapedPrediction)

		if useCopy {
			hiddenDim := hidden.Shape[1]
			for b := 0; b < batchSize; b++ {
				hOff := b * hiddenDim
				hs := append([]float32(nil), hidden.Data[hOff:hOff+hiddenDim]...)
				histStates[b] = append(histStates[b], hs)
				histTokens[b] = append(histTokens[b], int(decoderInputData[b]))
			}
		}

		decoderHidden = hidden
		decoderCell = cell
	}

	decoderOutputs, err := tensor.Concat(stepOutputs, 1)
	if err != nil {
		return nil, fmt.Errorf("seq2seq decoder output concat failed: %w", err)
	}
	return decoderOutputs, nil
}

// Predict generates a description given an input query.
// ThoughtStep records what the model did at one decoding step: the token it
// chose, the runners-up it considered, and which MoE experts it consulted
// (with their gate weights). This is the model's honest, observable "thought
// process" — routing decisions, not a verbal chain-of-thought (the model is
// far too small to reason in words).
type ThoughtStep struct {
	Token      string
	TokenID    int
	Experts    []int     // top-k expert indices consulted for this token
	Gates      []float32 // gate weight per consulted expert (sums to 1)
	Candidates []string  // top-3 candidate token texts after anti-repeat masking
}

// ThoughtTrace is the per-reply thought process: one step per generated token
// plus the overall expert usage histogram for the reply.
type ThoughtTrace struct {
	Steps       []ThoughtStep
	ExpertUsage []float32 // fraction of reply tokens routed to each expert
	NumExperts  int
}

func (m *Seq2Seq) Predict(query string, maxLen int) (string, error) {
	answer, _, err := m.PredictWithTrace(query, maxLen)
	return answer, err
}

// NormalizeQuery canonicalizes a user query before encoding: lowercased,
// sentence punctuation (.!?,;:) stripped, whitespace collapsed. "Tell me a
// joke." and "tell me a joke" must reach the encoder identically —
// punctuation carries no meaning for intent, and training uses the same
// normalization, so the model never has to relearn every phrasing twice.
func NormalizeQuery(q string) string {
	q = strings.ToLower(q)
	var b strings.Builder
	b.Grow(len(q))
	prevSpace := true
	for _, r := range q {
		switch {
		case r == '.' || r == '!' || r == '?' || r == ',' || r == ';' || r == ':':
			// drop sentence punctuation
		case r == ' ' || r == '\t' || r == '\n' || r == '\r':
			if !prevSpace {
				b.WriteRune(' ')
			}
			prevSpace = true
		default:
			b.WriteRune(r)
			prevSpace = false
		}
	}
	return strings.TrimSpace(b.String())
}

func (m *Seq2Seq) PredictWithTrace(query string, maxLen int) (string, *ThoughtTrace, error) {
	if m != nil && len(m.ExactMap) > 0 {
		if answer, ok := m.ExactMap[strings.ToLower(strings.TrimSpace(query))]; ok {
			return strings.TrimSpace(answer), nil, nil
		}
	}

	if m.Tokenizer == nil {
		return "", nil, fmt.Errorf("seq2seq tokenizer is nil")
	}

	query = NormalizeQuery(query)
	// Encode the input query
	inputTokenIDs, err := m.Tokenizer.Encode(query)
	if err != nil {
		return "", nil, fmt.Errorf("failed to tokenize query: %w", err)
	}

	// Convert token IDs to tensor
	inputTensorData := make([]float32, len(inputTokenIDs))
	for i, id := range inputTokenIDs {
		inputTensorData[i] = float32(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputTokenIDs)}, inputTensorData, true)

	// Encoder forward pass
	encoderHidden, encoderCell, err := m.Encoder.Forward(inputTensor)
	if err != nil {
		return "", nil, fmt.Errorf("prediction encoder forward failed: %w", err)
	}

	decoderHidden := encoderHidden
	decoderCell := encoderCell

	// Start with <SOS> token
	outputTokens := []int{}
	currentInputTokenID := float64(m.OutputVocab.BosID)

	trace := &ThoughtTrace{}
	if m.Decoder.MoE != nil {
		trace.NumExperts = m.Decoder.MoE.NumExperts
		trace.ExpertUsage = make([]float32, m.Decoder.MoE.NumExperts)
	}

	// Beam decoding, for callers that opted in via BeamWidth (the
	// chat's social model): keep the BeamWidth most probable partial
	// replies alive and return the best-scoring finished one, instead
	// of committing to the argmax token at every step. The thought
	// trace then honestly describes the winning beam's token path.
	if m.BeamWidth > 1 {
		return m.predictBeam(decoderHidden, decoderCell, maxLen, trace)
	}

	// Anti-degeneration: tiny models love to loop ("go go go", "the the the").
	// Forbid immediate token repeats and repeated bigrams during greedy
	// decoding — standard practice, costs nothing.
	seenBigrams := map[[2]int]bool{}

	// Copy history (batch=1 here): the decoder's own past LSTM states and
	// the token ids that produced them. Only used when the decoder carries
	// a copy gate.
	useCopy := m.Decoder.Copy != nil
	histStates := [][][]float32{{}}
	histTokens := [][]int{{}}

	for t := range maxLen {
		decoderInput := tensor.NewTensor([]int{1, 1}, []float32{float32(currentInputTokenID)}, true)

		prediction, hidden, cell, err := m.Decoder.ForwardCopy(decoderInput, decoderHidden, decoderCell, histStates, histTokens)
		if err != nil {
			return "", nil, fmt.Errorf("prediction decoder forward failed at step %d: %w", t, err)
		}

		if len(outputTokens) > 0 {
			prev := outputTokens[len(outputTokens)-1]
			if prev >= 0 && prev < prediction.Shape[1] {
				prediction.Data[prev] = -1e30 // no "go go"
			}
			for c := 0; c < prediction.Shape[1]; c++ {
				if seenBigrams[[2]int{prev, c}] {
					prediction.Data[c] = -1e30 // no repeated bigram
				}
			}
		}

		// Get the token with the highest probability (greedy decoding),
		// plus the runners-up for the thought trace.
		predictedTokenID := 0
		maxProb := prediction.Data[0]
		for i := 1; i < prediction.Shape[1]; i++ {
			if prediction.Data[i] > maxProb {
				maxProb = prediction.Data[i]
				predictedTokenID = i
			}
		}
		candidates := topKCandidates(prediction.Data, 3, m.OutputVocab)

		// Record the MoE routing for this step (single token => first K entries).
		step := ThoughtStep{
			Token:      m.OutputVocab.GetWord(predictedTokenID),
			TokenID:    predictedTokenID,
			Candidates: candidates,
		}
		if m.Decoder.MoE != nil {
			idx, gates := m.Decoder.MoE.LastRouting()
			k := m.Decoder.MoE.TopK
			if len(idx) >= k {
				step.Experts = append([]int(nil), idx[:k]...)
				step.Gates = append([]float32(nil), gates[:k]...)
				for _, e := range step.Experts {
					if e >= 0 && e < len(trace.ExpertUsage) {
						trace.ExpertUsage[e]++
					}
				}
			}
		}
		trace.Steps = append(trace.Steps, step)

		if len(outputTokens) > 0 {
			seenBigrams[[2]int{outputTokens[len(outputTokens)-1], predictedTokenID}] = true
		}
		outputTokens = append(outputTokens, predictedTokenID)
		if predictedTokenID == m.OutputVocab.EosID {
			break
		}

		// Record this step for the copy mechanism's history.
		if useCopy {
			hs := append([]float32(nil), hidden.Data...)
			histStates[0] = append(histStates[0], hs)
			histTokens[0] = append(histTokens[0], predictedTokenID)
		}

		// Use predicted token as next input
		currentInputTokenID = float64(predictedTokenID)
		decoderHidden = hidden
		decoderCell = cell
	}

	// Normalize the expert usage histogram (each step consults topK experts,
	// so usage sums to topK per step).
	if m.Decoder.MoE != nil {
		if denom := float32(len(trace.Steps) * m.Decoder.MoE.TopK); denom > 0 {
			for i := range trace.ExpertUsage {
				trace.ExpertUsage[i] /= denom
			}
		}
	}

	decodedDescription := m.OutputVocab.Decode(outputTokens)

	return decodedDescription, trace, nil
}

// ---------------------------------------------------------------------------
// Beam search decoding (PredictWithTrace when BeamWidth > 1).
//
// Score = cumulative log-probability (log-softmax of the banned logits),
// summed per step; NO length normalization. The greedy loop's hard bans
// apply per beam, relative to that beam's own history: its previous token
// and any bigram it has already produced are forbidden. Each live beam
// expands its top-BeamWidth tokens; the pool is trimmed to BeamWidth by
// score each step. A beam that emits EOS is finished and stops expanding
// but stays in the pool, competing by score. The winner is the
// highest-score finished beam, or the highest-score live beam if none
// finished by maxLen. Decoder state (LSTM hidden/cell, plus copy-gate
// history when the decoder carries one) is cloned per beam. The search
// is fully deterministic: same model, same input, same reply.
// ---------------------------------------------------------------------------

type beamState struct {
	tokens     []int
	hidden     *tensor.Tensor
	cell       *tensor.Tensor
	histStates [][][]float32
	histTokens [][]int
	score      float64
	steps      []ThoughtStep
	finished   bool
}

func cloneTensor(t *tensor.Tensor) *tensor.Tensor {
	if t == nil {
		return nil
	}
	c := *t
	c.Data = append([]float32(nil), t.Data...)
	c.Shape = append([]int(nil), t.Shape...)
	c.Grad = nil
	c.Mask = nil
	c.Creator = nil
	c.Operation = nil
	return &c
}

func cloneHistStates(h [][][]float32) [][][]float32 {
	out := make([][][]float32, len(h))
	for i := range h {
		out[i] = make([][]float32, len(h[i]))
		for j := range h[i] {
			out[i][j] = append([]float32(nil), h[i][j]...)
		}
	}
	return out
}

func cloneHistTokens(h [][]int) [][]int {
	out := make([][]int, len(h))
	for i := range h {
		out[i] = append([]int(nil), h[i]...)
	}
	return out
}

func (m *Seq2Seq) predictBeam(encoderHidden, encoderCell *tensor.Tensor, maxLen int, trace *ThoughtTrace) (string, *ThoughtTrace, error) {
	width := m.BeamWidth
	useCopy := m.Decoder.Copy != nil
	beams := []*beamState{{
		hidden:     cloneTensor(encoderHidden),
		cell:       cloneTensor(encoderCell),
		histStates: [][][]float32{{}},
		histTokens: [][]int{{}},
	}}

	for t := 0; t < maxLen; t++ {
		allFinished := true
		for _, b := range beams {
			if !b.finished {
				allFinished = false
				break
			}
		}
		if allFinished {
			break
		}
		var pool []*beamState
		for _, b := range beams {
			if b.finished {
				pool = append(pool, b)
				continue
			}
			lastTok := m.OutputVocab.BosID
			if len(b.tokens) > 0 {
				lastTok = b.tokens[len(b.tokens)-1]
			}
			decoderInput := tensor.NewTensor([]int{1, 1}, []float32{float32(lastTok)}, true)
			prediction, hidden, cell, err := m.Decoder.ForwardCopy(decoderInput, b.hidden, b.cell, b.histStates, b.histTokens)
			if err != nil {
				return "", nil, fmt.Errorf("beam prediction decoder forward failed at step %d: %w", t, err)
			}
			// MoE routing for this expansion (LastRouting reflects
			// THIS forward call, read before the next beam's).
			var expIdx []int
			var expGates []float32
			if m.Decoder.MoE != nil {
				idx, gates := m.Decoder.MoE.LastRouting()
				k := m.Decoder.MoE.TopK
				if len(idx) >= k {
					expIdx = append([]int(nil), idx[:k]...)
					expGates = append([]float32(nil), gates[:k]...)
				}
			}
			logits := prediction.Data
			vocabSize := prediction.Shape[1]
			// Hard bans, relative to this beam's own history.
			if len(b.tokens) > 0 {
				seen := map[[2]int]bool{}
				for i := 0; i+1 < len(b.tokens); i++ {
					seen[[2]int{b.tokens[i], b.tokens[i+1]}] = true
				}
				prev := b.tokens[len(b.tokens)-1]
				if prev >= 0 && prev < vocabSize {
					logits[prev] = -1e30
				}
				for c := 0; c < vocabSize; c++ {
					if seen[[2]int{prev, c}] {
						logits[c] = -1e30
					}
				}
			}
			candidates := topKCandidates(logits, 3, m.OutputVocab)
			// Log-softmax normalizer.
			maxL := float64(logits[0])
			for _, v := range logits {
				if float64(v) > maxL {
					maxL = float64(v)
				}
			}
			sum := 0.0
			for _, v := range logits {
				sum += math.Exp(float64(v) - maxL)
			}
			logNorm := maxL + math.Log(sum)
			// This beam's top-width next tokens.
			order := make([]int, vocabSize)
			for i := range order {
				order[i] = i
			}
			sort.Slice(order, func(a, bb int) bool { return logits[order[a]] > logits[order[bb]] })
			n := width
			if n > vocabSize {
				n = vocabSize
			}
			for _, tk := range order[:n] {
				child := &beamState{
					tokens:     append(append([]int(nil), b.tokens...), tk),
					hidden:     cloneTensor(hidden),
					cell:       cloneTensor(cell),
					histStates: cloneHistStates(b.histStates),
					histTokens: cloneHistTokens(b.histTokens),
					score:      b.score + (float64(logits[tk]) - logNorm),
					finished:   tk == m.OutputVocab.EosID,
				}
				step := ThoughtStep{
					Token:      m.OutputVocab.GetWord(tk),
					TokenID:    tk,
					Candidates: candidates,
					Experts:    expIdx,
					Gates:      expGates,
				}
				child.steps = append(append([]ThoughtStep(nil), b.steps...), step)
				// Copy-mechanism history, mirroring the greedy loop:
				// record (state, token) for every non-final token.
				if useCopy && !child.finished {
					hs := append([]float32(nil), hidden.Data...)
					child.histStates[0] = append(child.histStates[0], hs)
					child.histTokens[0] = append(child.histTokens[0], tk)
				}
				pool = append(pool, child)
			}
		}
		sort.SliceStable(pool, func(a, bb int) bool { return pool[a].score > pool[bb].score })
		if len(pool) > width {
			pool = pool[:width]
		}
		beams = pool
	}

	// Winner: highest-score finished beam; if none finished, best live.
	var winner *beamState
	for _, b := range beams {
		if b.finished && (winner == nil || b.score > winner.score) {
			winner = b
		}
	}
	if winner == nil {
		for _, b := range beams {
			if winner == nil || b.score > winner.score {
				winner = b
			}
		}
	}
	// The trace is the winning beam's path — the tokens the reply
	// actually consists of, with the routing each one used.
	trace.Steps = winner.steps
	if m.Decoder.MoE != nil {
		for _, st := range winner.steps {
			for _, e := range st.Experts {
				if e >= 0 && e < len(trace.ExpertUsage) {
					trace.ExpertUsage[e]++
				}
			}
		}
		if denom := float32(len(trace.Steps) * m.Decoder.MoE.TopK); denom > 0 {
			for i := range trace.ExpertUsage {
				trace.ExpertUsage[i] /= denom
			}
		}
	}
	return m.OutputVocab.Decode(winner.tokens), trace, nil
}

// topKCandidates returns the text of the k highest-logit tokens.
func topKCandidates(logits []float32, k int, v *vocab.Vocabulary) []string {
	type pair struct {
		id int
		v  float32
	}
	best := make([]pair, 0, k)
	for i, lv := range logits {
		inserted := false
		for j := range best {
			if lv > best[j].v {
				best = append(best, pair{})
				copy(best[j+1:], best[j:])
				best[j] = pair{i, lv}
				inserted = true
				break
			}
		}
		if !inserted && len(best) < k {
			best = append(best, pair{i, lv})
		}
		if len(best) > k {
			best = best[:k]
		}
	}
	out := make([]string, 0, len(best))
	for _, b := range best {
		out = append(out, v.GetWord(b.id))
	}
	return out
}

// Save saves the Seq2Seq model to a file.
func (m *Seq2Seq) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	// Create a serializable version of the model
	serializableModel := &SerializableSeq2Seq{
		Encoder:     m.Encoder,
		Decoder:     m.Decoder,
		OutputVocab: m.OutputVocab,
		HiddenDim:   m.HiddenDim,
		ExactMap:    m.ExactMap,
	}

	if err := encoder.Encode(serializableModel); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	log.Printf("Seq2Seq model saved to %s", filePath)
	return nil
}

// Load loads the Seq2Seq model from a file.
func Load(filePath string, tok *tokenizer.Tokenizer) (*Seq2Seq, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file for loading model: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	var serializableModel SerializableSeq2Seq
	if err := decoder.Decode(&serializableModel); err != nil {
		return nil, fmt.Errorf("failed to decode model: %w", err)
	}

	// Create a new Seq2Seq model from the loaded data
	model := &Seq2Seq{
		Encoder:     serializableModel.Encoder,
		Decoder:     serializableModel.Decoder,
		Tokenizer:   tok, // Tokenizer is not saved, it's passed in
		OutputVocab: serializableModel.OutputVocab,
		HiddenDim:   serializableModel.HiddenDim,
		ExactMap:    serializableModel.ExactMap,
	}

	log.Printf("Seq2Seq model loaded from %s", filePath)
	return model, nil
}
