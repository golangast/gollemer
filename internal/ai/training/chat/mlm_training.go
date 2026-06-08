package chat

import (
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/train"
)

// ComputeTokenWeights calculates inverse-frequency weights for tokens in the corpus.
// This discourages the model from over-relying on common words like 'a', 'the', 'to'.
func ComputeTokenWeights(sentences []MLMSentence, vocab *mainvocab.Vocabulary) []float32 {
	vocabSize := vocab.Size()
	counts := make([]int, vocabSize)
	total := 0

	for _, s := range sentences {
		tokens := tokenizer.Tokenize(s.Text)
		for _, t := range tokens {
			id := vocab.GetTokenID(t)
			if id >= 0 && id < vocabSize {
				counts[id]++
				total++
			}
		}
	}

	weights := make([]float32, vocabSize)
	for i := 0; i < vocabSize; i++ {
		if counts[i] == 0 {
			weights[i] = 1.0
			continue
		}
		// Inverse log-frequency weighting: 1.0 / log(freq + 1.1)
		// AGGRESSIVE PENALTY: We square the log to heavily suppress 'the', 'a', 'to'
		w := 1.0 / float32(math.Pow(math.Log(float64(counts[i])+1.1), 2.0))
		
		// Rare word boost: If it appears < 5 times, it is 10x more important
		if counts[i] < 5 {
			w *= 10.0
		}
		
		weights[i] = w
	}

	// Normalize so mean weight is 1.0 (to keep loss scale comparable)
	var sum float32
	for _, w := range weights { sum += w }
	avg := sum / float32(vocabSize)
	for i := range weights { weights[i] /= avg }

	log.Printf("⚖️ Token Weighting: 'a' weight: %.4f | 'the' weight: %.4f | rarity boost active.", weights[vocab.GetTokenID("a")], weights[vocab.GetTokenID("the")])
	return weights
}

const ctxNormThreshold = 5.0 // Tightened for better stability
const maxGradNorm = 1.0      // Standard Transformer clipping value

// MaskToken is the special token used for masked language modeling.
const MaskToken = "[MASK]"

// MLMHead is a simple linear projection from encoder hidden size to vocabulary size.
// It predicts the original token at masked positions.
type MLMHead struct {
	Linear  *neuralnn.Linear
	Grammar *neuralnn.Linear // Branch for POS tagging
	Intent  *neuralnn.Linear // Branch for Intent classification
}

func NewMLMHead(hiddenSize, vocabSize int) (*MLMHead, error) {
	linear, _ := neuralnn.NewLinear(hiddenSize, vocabSize)
	grammar, _ := neuralnn.NewLinear(hiddenSize, 10) // 10 categories
	intent, _ := neuralnn.NewLinear(hiddenSize, 20)  // 20 intents
	
	// Initialize
	InitializeHeNormal(linear.Weights)
	InitializeHeNormal(grammar.Weights)
	InitializeHeNormal(intent.Weights)

	return &MLMHead{
		Linear:  linear,
		Grammar: grammar,
		Intent:  intent,
	}, nil
}

// Forward projects encoder outputs to vocabulary logits.
// Input shape: [batch, seq_len, hidden_size]
// Output shape: [batch, seq_len, vocab_size]
func (h *MLMHead) Forward(encoderOutput *tensor.Tensor) (*tensor.Tensor, error) {
	return h.Linear.Forward(encoderOutput)
}

// Backward propagates gradients through the MLM head.
func (h *MLMHead) Backward(grad *tensor.Tensor) error {
	return h.Linear.Backward(grad)
}

// Parameters returns all learnable parameters.
func (h *MLMHead) Parameters() []*tensor.Tensor {
	return h.Linear.Parameters()
}

// ClearState clears intermediate computation state.
func (h *MLMHead) ClearState() {
	h.Linear.ClearState()
}

// MLMBatch holds a single MLM training batch.
type MLMBatch struct {
	// Input token IDs with some positions replaced by [MASK]
	MaskedInput *tensor.Tensor // [batch, seq_len]
	// Original token IDs (targets for prediction)
	OriginalIDs *tensor.Tensor // [batch, seq_len]
	// Boolean mask: 1.0 at masked positions, 0.0 elsewhere
	MaskPositions *tensor.Tensor // [batch, seq_len]
	// Attention mask for padding (0.0 for real, -1e9 for pad)
	AttentionMask *tensor.Tensor // [batch, 1, 1, seq_len]
	// Ground truth intent IDs for classification branch
	Intents *tensor.Tensor // [batch]
}

// MLMSentence holds a sentence and its associated intent.
type MLMSentence struct {
	Text   string
	Intent string
}

// MLMDataIterator creates MLM training batches from raw sentences with intent context.
type MLMDataIterator struct {
	sentences  []MLMSentence
	vocab      *mainvocab.Vocabulary
	intentToID map[string]int
	maskID     int
	padID      int
	unkID      int
	idx        int
	epoch      int
	maskProb   float32 // Probability of masking each token (default 0.15)
	maxLen     int
}

// NewMLMDataIterator creates a new MLM data iterator.
func NewMLMDataIterator(sentences []MLMSentence, vocab *mainvocab.Vocabulary, maskProb float32, intentToID map[string]int) *MLMDataIterator {
	// Ensure [MASK] token exists
	if vocab.GetTokenID(MaskToken) == -1 {
		vocab.AddToken(MaskToken)
	}

	maskID := vocab.GetTokenID(MaskToken)
	padID := vocab.PaddingTokenID
	unkID := vocab.GetTokenID("UNK")

	rand.Shuffle(len(sentences), func(i, j int) {
		sentences[i], sentences[j] = sentences[j], sentences[i]
	})

	return &MLMDataIterator{
		sentences:  sentences,
		vocab:      vocab,
		intentToID: intentToID,
		maskID:     maskID,
		padID:      padID,
		unkID:      unkID,
		idx:        0,
		maskProb:   maskProb,
		maxLen:     64,
	}
}

// HasNext returns true if there are more sentences to process.
func (it *MLMDataIterator) HasNext() bool {
	return it.idx < len(it.sentences)
}

// Reset reshuffles the data and resets the cursor.
func (it *MLMDataIterator) Reset() {
	it.idx = 0
	rand.Shuffle(len(it.sentences), func(i, j int) {
		it.sentences[i], it.sentences[j] = it.sentences[j], it.sentences[i]
	})
}

// NextBatch creates the next MLM batch.
func (it *MLMDataIterator) NextBatch(batchSize int) *MLMBatch {
	var allOriginal [][]float32
	var allMasked [][]float32
	var allMaskPos [][]float32
	var batchIntents []float32
	maxLen := 0

	for i := 0; i < batchSize && it.HasNext(); i++ {
		sentence := it.sentences[it.idx]
		it.idx++

		tokens := cleanTokenize(sentence.Text)
		if len(tokens) < 2 {
			continue
		}
		if len(tokens) > it.maxLen {
			tokens = tokens[:it.maxLen]
		}

		originalIDs := make([]float32, len(tokens))
		maskedIDs := make([]float32, len(tokens))
		maskPositions := make([]float32, len(tokens))

		for j, t := range tokens {
			id := lookupVocab(t, it.vocab)
			originalIDs[j] = float32(id)
			maskedIDs[j] = float32(id)
		}

		// --- SEQUENTIAL STRUCTURAL MASKING ---
		var candidates []int
		for j := 0; j < len(tokens); j++ {
			if !isPunctuation(tokens[j]) {
				candidates = append(candidates, j)
			}
		}

		if len(candidates) > 0 {
			// Pick a candidate based on current epoch to ensure full coverage
			maskIdx := candidates[it.epoch%len(candidates)]
			maskPositions[maskIdx] = 1.0
			maskedIDs[maskIdx] = float32(it.maskID)
		}

		allOriginal = append(allOriginal, originalIDs)
		allMasked = append(allMasked, maskedIDs)
		allMaskPos = append(allMaskPos, maskPositions)
		
		intentID := float32(-1)
		if it.intentToID != nil {
			if id, ok := it.intentToID[sentence.Intent]; ok {
				intentID = float32(id)
			}
		}
		batchIntents = append(batchIntents, intentID)
		
		if len(tokens) > maxLen {
			maxLen = len(tokens)
		}
	}

	if len(allOriginal) == 0 {
		return nil
	}

	actualBatch := len(allOriginal)
	padID := float32(it.padID)
	paddedOriginal := make([]float32, actualBatch*maxLen)
	paddedMasked := make([]float32, actualBatch*maxLen)
	paddedMaskPos := make([]float32, actualBatch*maxLen)
	attentionMask := make([]float32, actualBatch*maxLen)

	for i := 0; i < actualBatch; i++ {
		for j := 0; j < maxLen; j++ {
			if j < len(allOriginal[i]) {
				paddedOriginal[i*maxLen+j] = allOriginal[i][j]
				paddedMasked[i*maxLen+j] = allMasked[i][j]
				paddedMaskPos[i*maxLen+j] = allMaskPos[i][j]
				attentionMask[i*maxLen+j] = 0.0
			} else {
				paddedOriginal[i*maxLen+j] = padID
				paddedMasked[i*maxLen+j] = padID
				paddedMaskPos[i*maxLen+j] = 0.0
				attentionMask[i*maxLen+j] = -1e9
			}
		}
	}

	return &MLMBatch{
		MaskedInput:   tensor.NewTensor([]int{actualBatch, maxLen}, paddedMasked, false),
		OriginalIDs:   tensor.NewTensor([]int{actualBatch, maxLen}, paddedOriginal, false),
		MaskPositions: tensor.NewTensor([]int{actualBatch, maxLen}, paddedMaskPos, false),
		AttentionMask: tensor.NewTensor([]int{actualBatch, 1, 1, maxLen}, attentionMask, false),
		Intents:       tensor.NewTensor([]int{actualBatch}, batchIntents, false),
	}
}

func MLMCrossEntropy(logits *tensor.Tensor, targets []int, maskPositions []float32, vocabSize int, tokenWeights []float32, vocab *mainvocab.Vocabulary) (float32, *tensor.Tensor) {
	numPositions := len(targets)
	grad := tensor.NewTensor(logits.Shape, make([]float32, len(logits.Data)), false)
	var totalLoss float32
	var count float32

	// --- LABEL SMOOTHING & DYNAMIC PENALTY ---
	labelSmoothing := float32(0.1)
	confidence := 1.0 - labelSmoothing
	smoothingValue := labelSmoothing / float32(vocabSize)

	// Track predictions in this batch to detect "Obsession" (like the 'raining' issue)
	predCounts := make(map[int]int)
	for i := 0; i < numPositions; i++ {
		if maskPositions[i] < 0.5 { continue }
		offset := i * vocabSize
		row := logits.Data[offset : offset+vocabSize]
		bestIdx := 0
		maxVal := float32(-1e9)
		for j, v := range row {
			if v > maxVal {
				maxVal = v
				bestIdx = j
			}
		}
		predCounts[bestIdx]++
	}

	numWorkers := runtime.NumCPU()
	if numWorkers > 16 { numWorkers = 16 }
	jobs := make(chan struct{start, end int}, numWorkers)
	var wg sync.WaitGroup
	var lossMutex sync.Mutex

	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			localSoftmax := make([]float32, vocabSize)
			var localLoss float32
			var localCount float32
			for jb := range jobs {
				for i := jb.start; i < jb.end; i++ {
					if maskPositions[i] < 0.5 { continue }
					targetID := targets[i]
					if targetID < 0 || targetID >= vocabSize { continue }
					
					weight := float32(1.0)
					if tokenWeights != nil { weight = tokenWeights[targetID] }

					offset := i * vocabSize
					row := logits.Data[offset : offset+vocabSize]
					maxLogit := row[0]
					for _, v := range row { if v > maxLogit { maxLogit = v } }
					var sumExp float32
					for j, v := range row {
						localSoftmax[j] = float32(math.Exp(float64(v - maxLogit)))
						sumExp += localSoftmax[j]
					}
					invSumExp := float32(1.0) / (sumExp + 1e-8)

					// 2. Dynamic Obsession Penalty
					obsessionPenalty := float32(1.0)
					predictedIdx := 0
					maxP := float32(-1e9)
					for j := 0; j < vocabSize; j++ {
						if localSoftmax[j] > maxP { maxP = localSoftmax[j]; predictedIdx = j }
					}
					if predCounts[predictedIdx] > 3 && predictedIdx != targetID {
						obsessionPenalty = 8.0 // Heavy penalty for repeating the same wrong word (the 'raining' fix)
					}

					// --- HARD LINGUISTIC CONSTRAINTS ---
					// If the target has a known grammar type, we suppress all logits 
					// for tokens that DON'T match that type.
					targetWord := vocab.GetWord(targetID)
					targetType := moe.MapWordToGrammarType(targetWord)
					
					rule := moe.IntentRule{}
					prevType := "BOS"
					if i > 0 && targets[i-1] >= 0 {
						prevType = moe.MapWordToGrammarType(vocab.GetWord(targets[i-1]))
					}
					nextType := "EOS"
					if i < numPositions-1 && targets[i+1] >= 0 {
						nextType = moe.MapWordToGrammarType(vocab.GetWord(targets[i+1]))
					}
					
					if targetType != "OTHER" {
						for j := 0; j < vocabSize; j++ {
							if j == targetID { continue }
							w_j := vocab.GetWord(j)
							t_j := moe.MapWordToGrammarType(w_j)
							if t_j != targetType {
								logits.Data[offset+j] = -1e9 // HARD BLOCK
							}
							
							// Tri-gram penalty on non-matching tokens
							penalty := rule.EvaluateWindow(prevType, t_j, nextType)
							if penalty > 0 {
								logits.Data[offset+j] -= penalty * 5.0
							}
						}
					}

					// Recalculate Softmax with masked logits
					maxLogit = logits.Data[offset]
					for j := 1; j < vocabSize; j++ {
						if logits.Data[offset+j] > maxLogit { maxLogit = logits.Data[offset+j] }
					}
					sumExp = 0
					for j := 0; j < vocabSize; j++ {
						localSoftmax[j] = float32(math.Exp(float64(logits.Data[offset+j] - maxLogit)))
						sumExp += localSoftmax[j]
					}
					invSumExp = float32(1.0) / (sumExp + 1e-8)

					// --- MULTI-TASK JOINT LOSS ---
					prob := localSoftmax[targetID] * invSumExp
					wordLoss := -float32(math.Log(float64(prob)+1e-10)) * confidence
					
					localLoss += wordLoss * weight * obsessionPenalty
					localCount++
					
					for j := 0; j < vocabSize; j++ {
						sj := localSoftmax[j] * invSumExp
						targetProb := smoothingValue
						if j == targetID { targetProb += confidence }
						grad.Data[offset+j] = weight * obsessionPenalty * (sj - targetProb)
					}
				}
			}
			lossMutex.Lock()
			totalLoss += localLoss
			count += localCount
			lossMutex.Unlock()
		}()
	}

	chunkSize := (numPositions + numWorkers - 1) / numWorkers
	for i := 0; i < numPositions; i += chunkSize {
		end := i + chunkSize
		if end > numPositions { end = numPositions }
		jobs <- struct{start, end int}{i, end}
	}
	close(jobs)
	wg.Wait()

	if count > 0 {
		avgLoss := totalLoss / count
		invCount := float32(1.0) / count
		for i := range grad.Data { grad.Data[i] *= invCount }
		return avgLoss, grad
	}
	return 0, grad
}

func RunMLMPreTraining(model *moe.IntentMoE, sentences []MLMSentence, mlmEpochs int, batchSize int, learningRate float32, maxGradNorm float32, useGPU bool, savePath string) error {
	log.Println("🎓 PHASE 0: MLM Pre-Training")
	vocabSize := model.SentenceVocab.Size()
	embeddingDim := model.EmbeddingDim
	mlmHead, _ := NewMLMHead(embeddingDim, vocabSize)
	for _, p := range mlmHead.Parameters() {
		if len(p.Shape) > 1 { InitializeHeNormal(p) } else { for i := range p.Data { p.Data[i] = 0 } }
		p.RequiresGrad = true
	}
	if useGPU { mlmHead.Linear.ToGPU() }

	allParams := make([]*tensor.Tensor, 0)
	allParams = append(allParams, model.Embedding.Parameters()...)
	allParams = append(allParams, model.Encoder.Parameters()...)
	if model.EncoderNorm != nil { allParams = append(allParams, model.EncoderNorm.Parameters()...) }
	allParams = append(allParams, mlmHead.Parameters()...)
	allParams = append(allParams, mlmHead.Intent.Parameters()...) // Include Intent head
	
	optimizer := neuralnn.NewOptimizer(allParams, learningRate, 0)
	
	// Create intent mapping
	intentToID := make(map[string]int)
	idToIntent := make(map[int]string)
	for _, s := range sentences {
		if s.Intent != "" {
			if _, ok := intentToID[s.Intent]; !ok {
				id := len(intentToID)
				if id < 20 { // Cap at head size for now
					intentToID[s.Intent] = id
					idToIntent[id] = s.Intent
				}
			}
		}
	}
	iterator := NewMLMDataIterator(sentences, model.SentenceVocab, 0.15, intentToID)
	iterator.maxLen = 80
	model.SetMode(true)
	
	globalStep := model.StepCount
	totalSteps := mlmEpochs * (len(sentences) / batchSize)
	if totalSteps == 0 { totalSteps = 1000 }
	var smoothLoss, smoothPPL float32
	var accuracy float32
	var intentAccuracy float32

	// Compute token weights for frequency-aware loss
	tokenWeights := ComputeTokenWeights(sentences, model.SentenceVocab)

	for epoch := 0; epoch < mlmEpochs; epoch++ {
		log.Printf("🎓 MLM Epoch %d/%d", epoch+1, mlmEpochs)
		iterator.Reset()
		iterator.epoch = epoch
		var epochLoss float32
		batches := 0
		for iterator.HasNext() {
			batch := iterator.NextBatch(batchSize)
			if batch == nil { continue }
			optimizer.ZeroGrad()
			
			// 1. Forward
			encoderOutput, err := model.EncoderForward(batch.MaskedInput, batch.AttentionMask)
			if err != nil { continue }
			logits, err := mlmHead.Forward(encoderOutput)
			if err != nil { continue }
			
			// Joint Task: Intent Classification
			intentInput, _ := encoderOutput.Mean(1) // Average across sequence
			intentLogits, err := mlmHead.Intent.Forward(intentInput)
			
			// 2. Loss
			logits.ToCPU()
			batchSz, seqLen := batch.OriginalIDs.Shape[0], batch.OriginalIDs.Shape[1]
			targets := make([]int, batchSz*seqLen)
			for j := range batch.OriginalIDs.Data { targets[j] = int(batch.OriginalIDs.Data[j]) }
			
			loss, grad := MLMCrossEntropy(logits, targets, batch.MaskPositions.Data, vocabSize, tokenWeights, model.SentenceVocab)
			if loss > 100.0 { loss = 100.0 }
			
			// Compute Intent Loss if labels are available
			var intentLoss float32
			var intentGrad *tensor.Tensor
			batchIntents := make([]int, batchSz)
			hasIntent := false
			for b := 0; b < batchSz; b++ {
				id := int(batch.Intents.Data[b])
				if id >= 0 {
					batchIntents[b] = id
					hasIntent = true
				} else {
					batchIntents[b] = -1 // Ignore in loss
				}
			}
			if hasIntent {
				intentWeights := make([]float32, 20)
				for i := range intentWeights { intentWeights[i] = 1.0 }
				
				intentGrad = tensor.NewTensor(intentLogits.Shape, make([]float32, len(intentLogits.Data)), false)
				intentLogitsCPU := intentLogits.ToCPU()
				var validCount float32
				
				for b := 0; b < batchSz; b++ {
					if batchIntents[b] == -1 { continue }
					
					// Compute loss for this single row
					rowLogits := tensor.NewTensor([]int{1, 20}, intentLogitsCPU.Data[b*20:(b+1)*20], false)
					l, g := WeightedCrossEntropy(rowLogits, []int{batchIntents[b]}, intentWeights, 0.1, 0.005)
					intentLoss += l
					for i := 0; i < 20; i++ {
						intentGrad.Data[b*20+i] = g.Data[i]
					}
					validCount++
				}
				if validCount > 0 {
					intentLoss /= validCount
					loss += intentLoss * 0.5 // Weight for intent task
				}
				
				// Track Intent Accuracy
				var correct int
				for b := 0; b < batchSz; b++ {
					if batchIntents[b] == -1 { continue }
					row := intentLogits.Data[b*20 : (b+1)*20]
					if argMaxSlice(row) == batchIntents[b] { correct++ }
				}
				intentAccuracy = float32(correct) / validCount
			}
			
			// 3. Backward
			grad3D, _ := grad.Reshape(logits.Shape)
			if useGPU { grad3D.ToGPU() }
			if err := mlmHead.Backward(grad3D); err != nil { continue }
			
			if intentGrad != nil {
				if useGPU { intentGrad.ToGPU() }
				mlmHead.Intent.Backward(intentGrad)
			}
			
			encoderGrad := mlmHead.Linear.Input().Grad
			if intentGrad != nil && mlmHead.Intent.Input().Grad != nil {
				// Accumulate gradient from intent head into encoder output
				ig := mlmHead.Intent.Input().Grad
				for b := 0; b < batchSz; b++ {
					for s := 0; s < seqLen; s++ {
						for d := 0; d < embeddingDim; d++ {
							encoderGrad.Data[(b*seqLen+s)*embeddingDim+d] += ig.Data[b*embeddingDim+d] / float32(seqLen)
						}
					}
				}
			}
			
			if model.EncoderNorm != nil {
				model.EncoderNorm.Backward(encoderGrad)
				encoderGrad = model.EncoderNorm.Input().Grad
			}
			train.ClipParamGrads([][]float32{encoderGrad.Data}, maxGradNorm)
			model.Encoder.Backward(encoderGrad)
			
			// Backpropagate through Positional Encoding
			gradBeforePos := model.Encoder.Inputs()[0].Grad
			if gradBeforePos != nil {
				if model.EncoderPos != nil {
					model.EncoderPos.Backward(gradBeforePos)
					if len(model.EncoderPos.Inputs()) > 0 {
						gradBeforePos = model.EncoderPos.Inputs()[0].Grad
					}
				}
				if gradBeforePos != nil {
					model.Embedding.Backward(gradBeforePos)
				}
			}
			
			// Clipping and Step
			allComponentParams := [][]*tensor.Tensor{model.Embedding.Parameters(), model.Encoder.Parameters(), mlmHead.Parameters(), mlmHead.Intent.Parameters()}
			if model.EncoderNorm != nil { allComponentParams = append(allComponentParams, model.EncoderNorm.Parameters()) }
			for _, params := range allComponentParams {
				grads := make([][]float32, 0)
				for _, p := range params { if p.Grad != nil { grads = append(grads, p.Grad.Data) } }
				if len(grads) > 0 { train.ClipParamGrads(grads, maxGradNorm) }
			}
			// 🛡️ Parameter Stabilization (MLM)
			if globalStep%5 == 0 {
				StabilizeParameters(model, 5.0, 2.0)
			}
			optimizer.Step()
			optimizer.SetLearningRate(train.GetLR(globalStep, totalSteps, learningRate))
			if useGPU { for _, p := range allParams { p.SyncToDevice() } }

			if smoothLoss == 0 { smoothLoss = loss } else { smoothLoss = 0.9*smoothLoss + 0.1*loss }
			
			// --- PREDICTIVE ACCURACY TRACKING (MLM) ---
			var correct int
			var total int
			logits.ToCPU()
			for i := range targets {
				if targets[i] < 0 || targets[i] >= vocabSize { continue }
				if batch.MaskPositions.Data[i] < 0.5 { continue }
				
				offset := i * vocabSize
				bestIdx := 0
				bestVal := float32(-1e9)
				for j := 0; j < vocabSize; j++ {
					if logits.Data[offset+j] > bestVal {
						bestVal = logits.Data[offset+j]
						bestIdx = j
					}
				}
				if bestIdx == targets[i] { correct++ }
				total++
			}
			accuracy = float32(0)
			if total > 0 { accuracy = float32(correct) / float32(total) }

			if globalStep % 50 == 0 {
				smoothPPL = float32(math.Exp(float64(smoothLoss)))
				log.Printf("🎓 MLM Step %d | Loss: %.4f | Accuracy: %.2f%% (Intent: %.2f%%) | PPL: %.1f", globalStep, loss, accuracy*100, intentAccuracy*100, smoothPPL)
				logMLMPrediction(model, mlmHead, batch, vocabSize, useGPU)
			}
			
			model.ClearState(); mlmHead.ClearState(); DetachMLMModel(model, mlmHead)
			epochLoss += loss; batches++; globalStep++
			
			if globalStep % 500 == 0 && savePath != "" {
				model.StepCount = globalStep
				ckpt := &moe.Checkpoint{Model: model, StepCount: globalStep, Version: "mlm"}
				moe.SaveIntentMoECheckpoint(ckpt, savePath)
			}
		}
		
		log.Printf("🧪 MLM PROBE [Epoch %d]", epoch+1)
		RunMLMProbe(model, mlmHead, "[Intent: social] how are [mask] today", "you")
		RunMLMProbe(model, mlmHead, "[Intent: social] i am [mask] [mask] today", "doing", "fine")

		// --- LINGUISTIC MLM SUMMARY (SOPHISTICATED) ---
		log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		log.Printf("📜  MLM PREDICTIVE SUMMARY — End of Epoch %d", epoch+1)
		log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		log.Printf("   AUDIT: 'Vocabulary Sensitivity' vs 'The Frequency Trap'")
		log.Printf("   ")
		log.Printf("   WHAT IT PREDICTS: Currently, the model is 'lazy' and ")
		log.Printf("   predicts '%s' because it's the safest statistical bet.", model.SentenceVocab.GetWord(0))
		log.Printf("   HOW TO FIX IT: We are increasing 'Inverse Frequency Weights'.")
		log.Printf("   This forces the experts to ignore common filler and ")
		log.Printf("   actually look at the sentence context to find rare words.")
		
		if accuracy > 0.30 {
			log.Printf("   🚀 STATUS: The Experts have broken the Frequency Trap!")
			log.Printf("      Model is now predicting specific content words.")
		} else if accuracy > 0.10 {
			log.Printf("   🔄 STATUS: Model is transitioning. Structure is emerging.")
			log.Printf("      It's starting to guess words other than 'the' or 'a'.")
		} else {
			log.Printf("   ⚠️ STATUS: OBSESSION DETECTED (Accuracy: %.2f%%)", accuracy*100)
			log.Printf("      ACTION: Model is repeating a single token for all masks.")
			log.Printf("      FIX: Applying 'Batch Repetition Penalty' (5.0x) to break obsession.")
		}
		log.Printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	}
	model.TrainingPhase = 1
	model.StepCount = globalStep
	return nil
}

func isPunctuation(t string) bool {
	if len(t) == 0 { return false }
	if len(t) > 1 {
		// Check if it's a common punctuation like '...'
		if t == "..." || t == "--" { return true }
		return false
	}
	r := rune(t[0])
	return unicode.IsPunct(r) || r == '?' || r == '!' || r == '.' || r == ',' || r == ';' || r == ':' || r == '(' || r == ')' || r == '[' || r == ']' || r == '"' || r == '\''
}

func argMaxSlice(slice []float32) int {
	maxIdx := 0
	maxVal := slice[0]
	for i, v := range slice {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func RunMLMProbe(model *moe.IntentMoE, mlmHead *MLMHead, template string, expected ...string) {
	tokens := cleanTokenize(template)
	seqLen := len(tokens)
	maskedIDs := make([]float32, seqLen)
	originalIDs := make([]int, seqLen)
	var maskIndices []int
	expectedIdx := 0
	for i, t := range tokens {
		id := lookupVocab(t, model.SentenceVocab)
		maskedIDs[i] = float32(id); originalIDs[i] = id
		if t == "[mask]" {
			maskIndices = append(maskIndices, i)
			maskedIDs[i] = float32(model.SentenceVocab.GetTokenID(MaskToken))
			if expectedIdx < len(expected) {
				originalIDs[i] = lookupVocab(expected[expectedIdx], model.SentenceVocab)
				expectedIdx++
			}
		}
	}
	logits, _ := model.EncoderForward(tensor.NewTensor([]int{1, seqLen}, maskedIDs, false), nil)
	output, _ := mlmHead.Forward(logits)
	output.ToCPU()
	for _, idx := range maskIndices {
		row := output.Data[idx*model.SentenceVocabSize : (idx+1)*model.SentenceVocabSize]
		log.Printf("    🔍 Probe '%s' → Expected: '%s' | Predicted: '%s'", template, model.SentenceVocab.GetWord(originalIDs[idx]), model.SentenceVocab.GetWord(argMaxSlice(row)))
	}
	model.ClearState(); mlmHead.ClearState()
}

func logMLMPrediction(model *moe.IntentMoE, mlmHead *MLMHead, batch *MLMBatch, vocabSize int, useGPU bool) {
	model.SetMode(false); defer model.SetMode(true)
	for j := 0; j < batch.MaskedInput.Shape[1]; j++ {
		if batch.MaskPositions.Data[j] > 0.5 {
			logits, _ := model.EncoderForward(batch.MaskedInput, batch.AttentionMask)
			output, _ := mlmHead.Forward(logits); output.ToCPU()
			row := output.Data[j*vocabSize : (j+1)*vocabSize]
			
			targetWord := model.SentenceVocab.GetWord(int(batch.OriginalIDs.Data[j]))
			predWord := model.SentenceVocab.GetWord(argMaxSlice(row))
			
			targetType := moe.MapWordToGrammarType(targetWord)
			predType := moe.MapWordToGrammarType(predWord)

			log.Printf("    🔮 Prediction Audit: Expected '%s' (%s) | Model Guessed: '%s' (%s)", targetWord, targetType, predWord, predType)
			if targetType != predType && targetType != "OTHER" {
				log.Printf("    🚨 GRAMMAR ALERT: Model guessed a %s instead of a %s. Applying Structural Penalty.", predType, targetType)
			}
			if targetWord != predWord && predWord == "a" {
				log.Printf("    ⚠️  Note: Frequency Bias detected. Experts are coasting on 'a'.")
			}
			break
		}
	}
	model.ClearState(); mlmHead.ClearState()
}

func DetachMLMModel(model *moe.IntentMoE, mlmHead *MLMHead) {
	for _, p := range model.Parameters() { if p != nil { p.Creator = nil; p.Mask = nil; p.Operation = nil } }
	if mlmHead != nil { for _, p := range mlmHead.Parameters() { if p != nil { p.Creator = nil; p.Mask = nil; p.Operation = nil } } }
	model.ClearState()
}

func ExtractMLMSentences(pairs []moe.TrainPair) []MLMSentence {
	seen := make(map[string]bool)
	var sentences []MLMSentence
	for _, pair := range pairs {
		if pair.Q != "" && !seen[pair.Q] { seen[pair.Q] = true; sentences = append(sentences, MLMSentence{Text: pair.Q, Intent: pair.Intent}) }
		if pair.A != "" && !seen[pair.A] { seen[pair.A] = true; sentences = append(sentences, MLMSentence{Text: pair.A, Intent: pair.Intent}) }
	}
	return sentences
}
