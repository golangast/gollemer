package chat

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/debug"
	"strings"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	neuralnn "github.com/golangast/gollemer/internal/ai/neural/nn"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/train"
)

// MaskToken is the special token used for masked language modeling.
const MaskToken = "[MASK]"

// MLMHead is a simple linear projection from encoder hidden size to vocabulary size.
// It predicts the original token at masked positions.
type MLMHead struct {
	Linear *neuralnn.Linear
}

// NewMLMHead creates a new MLM prediction head.
func NewMLMHead(hiddenSize, vocabSize int) (*MLMHead, error) {
	linear, err := neuralnn.NewLinear(hiddenSize, vocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create MLM head: %w", err)
	}
	return &MLMHead{Linear: linear}, nil
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
}

// MLMSentence holds a sentence and its associated intent.
type MLMSentence struct {
	Text   string
	Intent string
}

// MLMDataIterator creates MLM training batches from raw sentences with intent context.
type MLMDataIterator struct {
	sentences []MLMSentence
	vocab     *mainvocab.Vocabulary
	maskID    int
	padID     int
	unkID     int
	idx       int
	maskProb  float32 // Probability of masking each token (default 0.15)
	maxLen    int
}

// NewMLMDataIterator creates a new MLM data iterator.
func NewMLMDataIterator(sentences []MLMSentence, vocab *mainvocab.Vocabulary, maskProb float32) *MLMDataIterator {
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
		sentences: sentences,
		vocab:     vocab,
		maskID:    maskID,
		padID:     padID,
		unkID:     unkID,
		idx:       0,
		maskProb:  maskProb,
		maxLen:    64,
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
// For each sentence:
//  1. Tokenize into word IDs
//  2. Randomly mask ~15% of tokens (not special tokens)
//  3. The model must predict the original token at masked positions
func (it *MLMDataIterator) NextBatch(batchSize int) *MLMBatch {
	var allOriginal [][]float32
	var allMasked [][]float32
	var allMaskPos [][]float32
	maxLen := 0

	for i := 0; i < batchSize && it.HasNext(); i++ {
		sentence := it.sentences[it.idx]
		it.idx++

		// Inject Intent as a prefix to teach context: "[Intent: social] hello world"
		intentPrefix := ""
		prefixTokenCount := 0
		if sentence.Intent != "" {
			intentPrefix = fmt.Sprintf("[Intent: %s]", sentence.Intent)
			prefixTokens := cleanTokenize(intentPrefix)
			prefixTokenCount = len(prefixTokens)
		}

		tokens := cleanTokenize(intentPrefix + " " + sentence.Text)
		if len(tokens) < 2 {
			continue // Skip very short sentences — not enough context
		}
		if len(tokens) > it.maxLen {
			tokens = tokens[:it.maxLen]
		}

		// Convert to IDs
		originalIDs := make([]float32, len(tokens))
		maskedIDs := make([]float32, len(tokens))
		maskPositions := make([]float32, len(tokens))

		for j, t := range tokens {
			id := lookupVocab(t, it.vocab)
			originalIDs[j] = float32(id)
			maskedIDs[j] = float32(id)
		}

		// Apply masking: for each token, with probability maskProb:
		//   80% → replace with [MASK]
		//   10% → replace with random word
		//   10% → keep original (but still predict it)
		// We NEVER mask the intent prefix.
		maskedCount := 0
		for j := range tokens {
			// Don't mask intent markers
			if j < prefixTokenCount {
				continue
			}

			if rand.Float32() < it.maskProb {
				maskPositions[j] = 1.0
				maskedCount++
				r := rand.Float32()
				if r < 0.8 {
					maskedIDs[j] = float32(it.maskID)
				} else if r < 0.9 {
					// Random token (avoid special tokens 0-3)
					maskedIDs[j] = float32(rand.Intn(it.vocab.Size()-4) + 4)
				}
				// else: keep original (10% of the time)
			}
		}

		// Ensure at least 1 token is masked per sentence
		if maskedCount == 0 && len(tokens) > 0 {
			pos := rand.Intn(len(tokens))
			maskPositions[pos] = 1.0
			maskedIDs[pos] = float32(it.maskID)
		}

		allOriginal = append(allOriginal, originalIDs)
		allMasked = append(allMasked, maskedIDs)
		allMaskPos = append(allMaskPos, maskPositions)

		if len(tokens) > maxLen {
			maxLen = len(tokens)
		}
	}

	if len(allOriginal) == 0 {
		return nil
	}

	actualBatch := len(allOriginal)
	padID := float32(it.padID)

	// Pad all sequences to maxLen
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
				attentionMask[i*maxLen+j] = 0.0 // Real token
			} else {
				paddedOriginal[i*maxLen+j] = padID
				paddedMasked[i*maxLen+j] = padID
				paddedMaskPos[i*maxLen+j] = 0.0 // Don't predict padding
				attentionMask[i*maxLen+j] = -1e9 // Mask padding in attention
			}
		}
	}

	return &MLMBatch{
		MaskedInput:   tensor.NewTensor([]int{actualBatch, maxLen}, paddedMasked, false),
		OriginalIDs:   tensor.NewTensor([]int{actualBatch, maxLen}, paddedOriginal, false),
		MaskPositions: tensor.NewTensor([]int{actualBatch, maxLen}, paddedMaskPos, false),
		AttentionMask: tensor.NewTensor([]int{actualBatch, 1, 1, maxLen}, attentionMask, false),
	}
}

// MLMCrossEntropy computes the cross-entropy loss ONLY at masked positions.
// logits: [batch, seq_len, vocab_size]
// targets: original token IDs [batch * seq_len]
// maskPositions: [batch * seq_len] — 1.0 at masked positions
// Returns: (loss, gradient tensor with same shape as logits)
func MLMCrossEntropy(logits *tensor.Tensor, targets []int, maskPositions []float32, vocabSize int) (float32, *tensor.Tensor) {
	numPositions := len(targets)
	grad := tensor.NewTensor(logits.Shape, make([]float32, len(logits.Data)), false)

	var totalLoss float32
	var count float32

	// Parallelize across rows to maximize CPU utilization on 'Mock' backends
	numWorkers := runtime.NumCPU()
	if numWorkers > 16 { numWorkers = 16 }
	
	type job struct {
		start, end int
	}
	jobs := make(chan job, numWorkers)
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

					offset := i * vocabSize
					row := logits.Data[offset : offset+vocabSize]

					// Numerical stability
					maxLogit := row[0]
					for _, v := range row {
						if v > maxLogit { maxLogit = v }
					}
					
					var sumExp float32
					for j, v := range row {
						localSoftmax[j] = float32(math.Exp(float64(v - maxLogit)))
						sumExp += localSoftmax[j]
					}
					invSumExp := float32(1.0) / (sumExp + 1e-8)

					// Loss
					prob := localSoftmax[targetID] * invSumExp
					localLoss -= float32(math.Log(float64(prob + 1e-12)))
					localCount++

					// Gradient
					for j := 0; j < vocabSize; j++ {
						sj := localSoftmax[j] * invSumExp
						targetProb := float32(0.0)
						if j == targetID { targetProb = 1.0 }
						grad.Data[offset+j] = sj - targetProb
					}
				}
			}
			lossMutex.Lock()
			totalLoss += localLoss
			count += localCount
			lossMutex.Unlock()
		}()
	}

	// Dispatch chunks of work
	chunkSize := (numPositions + numWorkers - 1) / numWorkers
	if chunkSize < 1 { chunkSize = 1 }
	for i := 0; i < numPositions; i += chunkSize {
		end := i + chunkSize
		if end > numPositions { end = numPositions }
		jobs <- job{i, end}
	}
	close(jobs)
	wg.Wait()

	if count > 0 {
		avgLoss := totalLoss / count
		invCount := float32(1.0) / count
		for i := range grad.Data {
			grad.Data[i] *= invCount
		}
		return avgLoss, grad
	}
	return 0, grad
}

// RunMLMPreTraining runs a pure MLM pre-training phase before the main seq2seq training.
// This teaches the encoder and embeddings word co-occurrence patterns (grammar).
func RunMLMPreTraining(
	model *moe.IntentMoE,
	sentences []MLMSentence,
	mlmEpochs int,
	batchSize int,
	learningRate float32,
	maxGradNorm float32,
	useGPU bool,
	savePath string, // Added savePath
) error {
	log.Println("🎓 ═══════════════════════════════════════════════")
	log.Println("🎓  PHASE 0: Masked Language Model Pre-Training")
	log.Println("🎓  Teaching grammar through word prediction...")
	log.Println("🎓 ═══════════════════════════════════════════════")

	vocabSize := model.SentenceVocab.Size()
	embeddingDim := model.EmbeddingDim

	// Ensure [MASK] token exists in vocabulary
	if model.SentenceVocab.GetTokenID(MaskToken) == -1 {
		model.SentenceVocab.AddToken(MaskToken)
		vocabSize = model.SentenceVocab.Size()
		log.Printf("✅ Added [MASK] token to vocabulary (new size: %d)", vocabSize)
	}

	// Create MLM prediction head
	mlmHead, err := NewMLMHead(embeddingDim, vocabSize)
	if err != nil {
		return fmt.Errorf("failed to create MLM head: %w", err)
	}

	// Initialize MLM head with He Normal
	for _, p := range mlmHead.Parameters() {
		InitializeHeNormal(p)
		p.RequiresGrad = true
	}

	if useGPU {
		mlmHead.Linear.ToGPU()
	}

	// Collect all trainable parameters (encoder + embedding + MLM head)
	allParams := make([]*tensor.Tensor, 0)
	allParams = append(allParams, model.Embedding.Parameters()...)
	allParams = append(allParams, model.Encoder.Parameters()...)
	allParams = append(allParams, mlmHead.Parameters()...)

	optimizer := neuralnn.NewOptimizer(allParams, learningRate, 0)

	// Create data iterator with 15% masking probability
	iterator := NewMLMDataIterator(sentences, model.SentenceVocab, 0.15)
	iterator.maxLen = 32

	model.SetMode(true)
	optimizer.SetLearningRate(learningRate)

	startTime := time.Now()
	globalStep := 0
	totalSteps := mlmEpochs * (len(sentences) / batchSize)
	if totalSteps == 0 {
		totalSteps = 1000
	}

	// Moving average for smoother logs
	var smoothLoss float32
	var smoothPPL float32

	for epoch := 0; epoch < mlmEpochs; epoch++ {
		log.Printf("🎓 MLM: Starting epoch %d with %d sentences (Batch Size: %d)", epoch+1, len(sentences), batchSize)
		iterator.Reset()
		var epochLoss float32
		batches := 0
		
		for iterator.HasNext() {
			batch := iterator.NextBatch(batchSize)
			if batch == nil || batch.MaskedInput == nil {
				continue
			}

			if globalStep == 0 {
				log.Printf("🚀 MLM: First batch loaded (SeqLen: %d). Starting forward pass...", batch.MaskedInput.Shape[1])
			}

			optimizer.ZeroGrad()

			maskedInput := batch.MaskedInput
			if useGPU {
				maskedInput.ToGPU()
			}

			// 1. Embed the masked input tokens
			embedded, err := model.Embedding.Forward(maskedInput)
			if err != nil {
				log.Printf("⚠️ MLM Embedding forward failed: %v", err)
				continue
			}

			// 2. Encode with MoE encoder to learn contextual representations
			encoderOutput, err := model.Encoder.Forward(embedded)
			if err != nil {
				log.Printf("⚠️ MLM Encoder forward failed: %v", err)
				continue
			}

			// Normalize encoder output (same as training)
			encoderOutput = model.NormalizeContextVector(encoderOutput)

			// 3. Project to vocabulary with MLM head
			logits, err := mlmHead.Forward(encoderOutput)
			if err != nil {
				log.Printf("⚠️ MLM Head forward failed: %v", err)
				continue
			}

			// 5. Compute loss only at masked positions
			logits.ToCPU()
			batchSz := batch.OriginalIDs.Shape[0]
			seqLen := batch.OriginalIDs.Shape[1]
			targets := make([]int, batchSz*seqLen)
			for j := 0; j < len(batch.OriginalIDs.Data) && j < len(targets); j++ {
				targets[j] = int(batch.OriginalIDs.Data[j])
			}

			// Flatten logits to [batch*seq, vocab]
			flatLogits, _ := logits.Reshape([]int{batchSz * seqLen, vocabSize})
			loss, grad := MLMCrossEntropy(flatLogits, targets, batch.MaskPositions.Data, vocabSize)

			if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
				log.Printf("⚠️ MLM loss is NaN/Inf at step %d, skipping", globalStep)
				model.ClearState()
				mlmHead.ClearState()
				continue
			}

			if smoothLoss == 0 {
				smoothLoss = loss
			} else {
				smoothLoss = 0.9*smoothLoss + 0.1*loss
			}

			// 6. Reshape gradient back to [batch, seq, vocab]
			grad3D, _ := grad.Reshape(logits.Shape)

			// 7. Backward through MLM head
			if useGPU {
				grad3D.ToGPU()
			}
			if err := mlmHead.Backward(grad3D); err != nil {
				log.Printf("⚠️ MLM Head backward failed: %v", err)
				continue
			}

			// Get gradient flowing to encoder output from MLM head
			encoderGrad := mlmHead.Linear.Input().Grad
			if encoderGrad == nil {
				encoderGrad = tensor.NewTensor(encoderOutput.Shape, make([]float32, len(encoderOutput.Data)), false)
			}

			// ACCOUNT FOR NORMALIZATION: NormalizeContextVector scales by (threshold / norm)
			// We should ideally scale gradients by the same factor (approx).
			const ctxNormThreshold = 5.0
			dim := encoderGrad.Shape[2]
			for b := 0; b < encoderGrad.Shape[0]; b++ {
				for s := 0; s < encoderGrad.Shape[1]; s++ {
					offset := (b*encoderGrad.Shape[1] + s) * dim
					var norm float32
					for d := 0; d < dim; d++ {
						v := encoderOutput.Data[offset+d]
						norm += v * v
					}
					norm = float32(math.Sqrt(float64(norm + 1e-8)))
					if norm > ctxNormThreshold {
						scale := ctxNormThreshold / norm
						for d := 0; d < dim; d++ {
							encoderGrad.Data[offset+d] *= scale
						}
					}
				}
			}

			// 8. Backward through encoder
			if err := model.Encoder.Backward(encoderGrad); err != nil {
				log.Printf("⚠️ MLM Encoder backward failed: %v", err)
				continue
			}

			// 9. Backward through embedding
			if len(model.Encoder.Inputs()) > 0 {
				embGrad := model.Encoder.Inputs()[0].Grad
				if embGrad != nil {
					model.Embedding.Backward(embGrad)
				}
			}

			// 10. Clip gradients
			paramGrads := make([][]float32, 0, len(allParams))
			for _, p := range allParams {
				if p.Grad != nil {
					paramGrads = append(paramGrads, p.Grad.Data)
				}
			}
			train.ClipParamGrads(paramGrads, maxGradNorm)

			// 11. Update weights
			optimizer.Step()

			// 12. Update Learning Rate (Cosine Decay with Warmup)
			currentLR := train.GetLR(globalStep, totalSteps, learningRate)
			optimizer.SetLearningRate(currentLR)

			if useGPU {
				for _, p := range allParams {
					p.SyncToDevice()
				}
			}
			// Logging moved here for better visibility of gradients before they are zeroed
			if globalStep == 0 {
				log.Println("✅ MLM: First batch update complete. Convergence started.")
			} else if globalStep % 10 == 0 {
				log.Printf("⏳ MLM: Step %d in progress...", globalStep)
			}

			if globalStep%50 == 0 {
				elapsed := time.Since(startTime).Seconds()
				ppl := float32(math.Exp(float64(smoothLoss)))
				if smoothPPL == 0 {
					smoothPPL = ppl
				} else {
					smoothPPL = 0.9*smoothPPL + 0.1*ppl
				}

				log.Printf("🎓 [MLM] Step %d | Loss: %.4f (smooth: %.4f) | PPL: %.1f | LR: %.6f | %.1f steps/s",
					globalStep, loss, smoothLoss, smoothPPL, currentLR, float64(globalStep)/elapsed)

				// Diagnostic: Compute gradient norms for visibility
				var embGradNorm, encGradNorm, headGradNorm float32
				for _, p := range model.Embedding.Parameters() {
					if p.Grad != nil {
						for _, g := range p.Grad.Data { embGradNorm += g * g }
					}
				}
				for _, p := range model.Encoder.Parameters() {
					if p.Grad != nil {
						for _, g := range p.Grad.Data { encGradNorm += g * g }
					}
				}
				for _, p := range mlmHead.Parameters() {
					if p.Grad != nil {
						for _, g := range p.Grad.Data { headGradNorm += g * g }
					}
				}
				log.Printf("📊 [Gradients] Embedding: %.6f | Encoder: %.6f | MLMHead: %.6f",
					math.Sqrt(float64(embGradNorm)), math.Sqrt(float64(encGradNorm)), math.Sqrt(float64(headGradNorm)))

				// Log a sample prediction at masked positions
				logMLMPrediction(model, mlmHead, batch, vocabSize, useGPU)
			}

			optimizer.ZeroGrad()

			// 13. Clear state
			model.ClearState()
			mlmHead.ClearState()
			DetachMLMModel(model, mlmHead)

			epochLoss += loss
			batches++
			globalStep++

				// Log a sample prediction at masked positions (moved up)

			// 💾 PERIODIC SAVING: Every 500 steps during MLM
			if globalStep > 0 && globalStep%500 == 0 && savePath != "" {
				log.Printf("💾 [MLM CHECKPOINT] Saving progress at Step %d...", globalStep)
				// Create a temporary checkpoint for storage
				ckpt := &moe.Checkpoint{
					Model:     model,
					StepCount: globalStep,
					Version:   "gollemer-mlm-pretrain",
				}
				if err := moe.SaveIntentMoECheckpoint(ckpt, savePath); err != nil {
					log.Printf("⚠️  MLM periodic save failed: %v", err)
				}
				// Also save vocabulary
				vocabPath := strings.Replace(savePath, ".gob", "_vocab.gob", 1)
				model.SentenceVocab.Save(vocabPath)
			}

			// 🛑 CIRCUIT BREAKER/SIGNAL CHECK
			if globalStep%100 == 0 {
				// Check for stop signal file in current directory
				if _, err := os.Stat(".stop"); err == nil {
					log.Println("🛑 [MLM] Stop signal detected. Saving and exiting Phase 0...")
					os.Remove(".stop")
					return nil // Graceful exit from MLM
				}
			}

			if globalStep%200 == 0 {
				runtime.GC()
				debug.FreeOSMemory()
			}
		}

		avgLoss := float32(0)
		if batches > 0 {
			avgLoss = epochLoss / float32(batches)
		}
		perplexity := float32(math.Exp(float64(avgLoss)))
		log.Printf("🎓 [MLM] Epoch %d/%d Complete | Avg Loss: %.4f | PPL: %.1f | Batches: %d",
			epoch+1, mlmEpochs, avgLoss, perplexity, batches)

		// Early stop if loss is very low (model has learned basic patterns)
		if avgLoss < 1.0 && epoch >= 2 {
			log.Printf("🎓 [MLM] Loss below 1.0 — grammar patterns learned. Moving to seq2seq training.")
			break
		}
	}

	log.Printf("🎓 ═══════════════════════════════════════════════")
	log.Printf("🎓  MLM Pre-Training Complete (%d steps, %.1fs)", globalStep, time.Since(startTime).Seconds())
	log.Printf("🎓  Encoder and embeddings now understand word context.")
	log.Printf("🎓 ═══════════════════════════════════════════════")

	// Free MLM head — it's no longer needed
	mlmHead = nil
	runtime.GC()

	return nil
}

// logMLMPrediction prints a sample fill-in-the-blank prediction for diagnostics.
func logMLMPrediction(model *moe.IntentMoE, mlmHead *MLMHead, batch *MLMBatch, vocabSize int, useGPU bool) {
	if model.SentenceVocab == nil || batch == nil {
		return
	}

	model.SetMode(false) // Switch to eval mode for diagnostic prediction
	defer model.SetMode(true)

	// Look at the first sequence in the batch
	seqLen := batch.MaskedInput.Shape[1]
	if seqLen == 0 {
		return
	}

	// Find a masked position
	for j := 0; j < seqLen; j++ {
		if batch.MaskPositions.Data[j] < 0.5 {
			continue
		}

		// Get the original word
		originalID := int(batch.OriginalIDs.Data[j])
		originalWord := model.SentenceVocab.GetWord(originalID)

		// Get surrounding context
		var contextBefore, contextAfter string
		if j > 0 {
			contextBefore = model.SentenceVocab.GetWord(int(batch.OriginalIDs.Data[j-1]))
		}
		if j < seqLen-1 && batch.MaskPositions.Data[j+1] < 0.5 {
			contextAfter = model.SentenceVocab.GetWord(int(batch.OriginalIDs.Data[j+1]))
		}

		// Run a quick forward to get prediction
		singleInput := tensor.NewTensor([]int{1, seqLen}, batch.MaskedInput.Data[:seqLen], false)
		if useGPU {
			singleInput.ToGPU()
		}
		emb, err := model.Embedding.Forward(singleInput)
		if err != nil {
			break
		}
		enc, err := model.Encoder.Forward(emb)
		if err != nil {
			break
		}
		enc = model.NormalizeContextVector(enc)
		logits, err := mlmHead.Forward(enc)
		if err != nil {
			break
		}
		logits.ToCPU()

		// Get top-3 predictions at the masked position
		offset := j * vocabSize
		if offset+vocabSize > len(logits.Data) {
			break
		}
		row := logits.Data[offset : offset+vocabSize]

		// Find max for softmax
		maxVal := row[0]
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
		var sumExp float32
		for _, v := range row {
			sumExp += float32(math.Exp(float64(v - maxVal)))
		}

		// Top-3
		type pred struct {
			id   int
			prob float32
		}
		top3 := make([]pred, 3)
		for k := 0; k < 3; k++ {
			top3[k] = pred{-1, -1}
		}
		for id, v := range row {
			p := float32(math.Exp(float64(v-maxVal))) / sumExp
			for k := 0; k < 3; k++ {
				if p > top3[k].prob {
					copy(top3[k+1:], top3[k:2])
					top3[k] = pred{id, p}
					break
				}
			}
		}

		predWords := make([]string, 3)
		for k := 0; k < 3; k++ {
			if top3[k].id >= 0 {
				predWords[k] = fmt.Sprintf("%s(%.1f%%)", model.SentenceVocab.GetWord(top3[k].id), top3[k].prob*100)
			}
		}

		correctMark := "❌"
		if top3[0].id == originalID {
			correctMark = "✅"
		}

		// Try to extract intent from context for display by looking for the token sequence: [ intent : <intent> ]
		intentDisplay := ""
		for k := 0; k < seqLen-3; k++ {
			t0 := model.SentenceVocab.GetWord(int(batch.OriginalIDs.Data[k]))
			t1 := model.SentenceVocab.GetWord(int(batch.OriginalIDs.Data[k+1]))
			t2 := model.SentenceVocab.GetWord(int(batch.OriginalIDs.Data[k+2]))
			if t0 == "[" && t1 == "intent" && t2 == ":" {
				intentValue := model.SentenceVocab.GetWord(int(batch.OriginalIDs.Data[k+3]))
				intentDisplay = fmt.Sprintf("[Intent: %s]", intentValue)
				break
			}
		}

		log.Printf("%s    🔍 '%s [___] %s' → Answer: '%s' | Predicted: %s %s",
			intentDisplay, contextBefore, contextAfter, originalWord, strings.Join(predWords, ", "), correctMark)

		// Clear the forward pass state
		model.ClearState()
		mlmHead.ClearState()
		break // Only show one example per check
	}
}

// DetachMLMModel detaches computation graphs from model and MLM head.
func DetachMLMModel(model *moe.IntentMoE, mlmHead *MLMHead) {
	for _, p := range model.Parameters() {
		if p != nil {
			p.Creator = nil
			p.Mask = nil
			p.Operation = nil
		}
	}
	if mlmHead != nil {
		for _, p := range mlmHead.Parameters() {
			if p != nil {
				p.Creator = nil
				p.Mask = nil
				p.Operation = nil
			}
		}
	}
	model.ClearState()
}

// ExtractMLMSentences extracts unique sentences from training pairs for MLM pre-training.
// It combines both questions and answers to maximize coverage of grammar patterns.
func ExtractMLMSentences(pairs []struct{ Q, A, Intent string }) []MLMSentence {
	seen := make(map[string]bool)
	var sentences []MLMSentence

	for _, pair := range pairs {
		// Add both questions and answers as independent sentences
		if pair.Q != "" && !seen[pair.Q] {
			seen[pair.Q] = true
			sentences = append(sentences, MLMSentence{Text: pair.Q, Intent: pair.Intent})
		}
		if pair.A != "" && !seen[pair.A] {
			seen[pair.A] = true
			sentences = append(sentences, MLMSentence{Text: pair.A, Intent: pair.Intent})
		}
	}

	log.Printf("📝 Extracted %d unique sentences for MLM pre-training", len(sentences))
	return sentences
}
