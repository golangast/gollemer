// Package training provides an adapter that wraps Gollemer's MoE model to satisfy
// the FIMTrainableModel interface, enabling Fill-In-The-Middle training on mined
// Go patch datasets.
package training

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// FIMAdapter wraps a MoE model stack to implement FIMTrainableModel.
// It handles token-to-embedding conversion, FIM-formatted forward passes,
// and gradient updates for patch-based code insertion training.
type FIMAdapter struct {
	model        *moe.MoEStack
	embedDim     int
	vocabSize    int
	learningRate float64
	step         int
	pool         sync.Pool // Reuse tensor data to reduce GC pressure
}

// NewFIMAdapter creates a new FIM adapter wrapping the given MoE stack.
func NewFIMAdapter(model *moe.MoEStack, embedDim, vocabSize int) *FIMAdapter {
	return &FIMAdapter{
		model:        model,
		embedDim:     embedDim,
		vocabSize:    vocabSize,
		learningRate: 1e-4,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]float32, 0, 1024*embedDim)
			},
		},
	}
}

// ForwardFIM runs a forward pass with FIM-formatted input.
// prefix, suffix, middle are token ID slices.
// Returns the cross-entropy loss and per-token logits.
func (a *FIMAdapter) ForwardFIM(prefix, suffix, middle []int) (float64, [][]float64, error) {
	// Build FIM input: <PRE> prefix <SUF> suffix <MID> middle
	inputIDs := make([]int, 0, len(prefix)+len(suffix)+len(middle)+3)
	inputIDs = append(inputIDs, tokenID("<PRE>"))
	inputIDs = append(inputIDs, prefix...)
	inputIDs = append(inputIDs, tokenID("<SUF>"))
	inputIDs = append(inputIDs, suffix...)
	inputIDs = append(inputIDs, tokenID("<MID>"))
	inputIDs = append(inputIDs, middle...)

	if len(inputIDs) == 0 {
		return 0, nil, fmt.Errorf("empty FIM input")
	}

	// Convert token IDs to input tensor
	inputTensor := a.tokensToTensor(inputIDs)
	defer a.releaseTensor(inputTensor)

	// Run forward pass through MoE stack
	output, err := a.model.Forward(inputTensor)
	if err != nil {
		return 0, nil, fmt.Errorf("model forward: %w", err)
	}

	// Compute cross-entropy loss on the middle portion only
	prefixSuffixLen := len(prefix) + len(suffix) + 2
	middleStart := prefixSuffixLen + 1

	var totalLoss float64
	var logits [][]float64

	if output != nil && output.Data != nil {
		outData := output.Data
		tokensInOutput := len(outData) / a.embedDim

		for i := 0; i < len(middle) && (middleStart+i) < tokensInOutput; i++ {
			startIdx := (middleStart + i) * a.embedDim
			endIdx := startIdx + a.embedDim
			if endIdx > len(outData) {
				break
			}

			tokenLogits := outData[startIdx:endIdx]
			// Convert float32 logits to float64 for loss computation
			logits64 := make([]float64, len(tokenLogits))
			for j, v := range tokenLogits {
				logits64[j] = float64(v)
			}
			logits = append(logits, logits64)

			// Cross-entropy: -log(softmax(target))
			targetID := middle[i]
			if targetID >= 0 && targetID < len(logits64) {
				var maxLogit float64
				for _, v := range logits64 {
					if v > maxLogit {
						maxLogit = v
					}
				}
				var sumExp float64
				for _, v := range logits64 {
					sumExp += math.Exp(v - maxLogit)
				}
				targetLogit := logits64[targetID] - maxLogit
				prob := math.Exp(targetLogit) / sumExp
				if prob > 0 {
					totalLoss -= math.Log(prob)
				}
			}
		}
	}

	if len(middle) > 0 {
		totalLoss /= float64(len(middle))
	}

	return totalLoss, logits, nil
}

// Backward updates model weights from the computed loss.
func (a *FIMAdapter) Backward(loss float64) error {
	if a.model == nil {
		return fmt.Errorf("model is nil")
	}

	// Run backward pass on the MoE stack
	// The MoEStack.Backward takes a gradient tensor
	gradShape := []int{1, 1, a.embedDim}
	gradData := make([]float32, a.embedDim)
	for i := range gradData {
		gradData[i] = float32(loss) / float32(a.embedDim)
	}
	grad := &tensor.Tensor{Shape: gradShape, Data: gradData}

	if err := a.model.Backward(grad); err != nil {
		return fmt.Errorf("model backward: %w", err)
	}

	a.step++
	return nil
}

// SaveCheckpoint saves training metadata (config, step, loss history) to JSON.
// The actual MoE model weights are managed by the main training loop.
func (a *FIMAdapter) SaveCheckpoint(path string) error {
	state := map[string]interface{}{
		"step":          a.step,
		"learning_rate": a.learningRate,
		"embed_dim":     a.embedDim,
		"vocab_size":    a.vocabSize,
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal checkpoint: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

// LoadCheckpoint loads training metadata from a JSON file.
func (a *FIMAdapter) LoadCheckpoint(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read checkpoint: %w", err)
	}
	var state map[string]interface{}
	if err := json.Unmarshal(data, &state); err != nil {
		return fmt.Errorf("unmarshal checkpoint: %w", err)
	}
	if step, ok := state["step"].(float64); ok {
		a.step = int(step)
	}
	if lr, ok := state["learning_rate"].(float64); ok {
		a.learningRate = lr
	}
	return nil
}

// GetLearningRate returns the current learning rate.
func (a *FIMAdapter) GetLearningRate() float64 {
	return a.learningRate
}

// SetLearningRate updates the learning rate.
func (a *FIMAdapter) SetLearningRate(lr float64) {
	a.learningRate = lr
}

// tokensToTensor converts token IDs to a flat float32 tensor with positional encoding.
// Uses a sync.Pool to reuse memory and reduce GC pressure.
func (a *FIMAdapter) tokensToTensor(ids []int) *tensor.Tensor {
	needed := len(ids) * a.embedDim
	raw := a.pool.Get().([]float32)
	if cap(raw) < needed {
		raw = make([]float32, needed)
	}
	data := raw[:needed]

	for i, id := range ids {
		base := i * a.embedDim
		for d := 0; d < a.embedDim; d++ {
			if d == 0 {
				data[base+d] = float32(id) / 10000.0
			} else {
				pos := float32(i)
				dim := float32(d)
				if d%2 == 0 {
					data[base+d] = float32(math.Sin(float64(pos) / math.Pow(10000, float64(dim)/float64(a.embedDim))))
				} else {
					data[base+d] = float32(math.Cos(float64(pos) / math.Pow(10000, float64(dim-1)/float64(a.embedDim))))
				}
			}
		}
	}

	t := &tensor.Tensor{
		Shape: []int{1, len(ids), a.embedDim},
		Data:  data,
	}
	return t
}

// releaseTensor returns tensor data to the pool for reuse.
func (a *FIMAdapter) releaseTensor(t *tensor.Tensor) {
	if t != nil && t.Data != nil {
		a.pool.Put(t.Data[:0]) // Reset length but keep capacity
	}
}

// tokenID returns a placeholder token ID for special FIM tokens.
func tokenID(token string) int {
	switch token {
	case "<PRE>":
		return 0
	case "<SUF>":
		return 1
	case "<MID>":
		return 2
	case "<PAD>":
		return 3
	case "<UNK>":
		return 4
	default:
		return 4
	}
}
