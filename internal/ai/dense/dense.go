// Package dense provides a minimal, self-contained dense feed-forward network
// intended to replace the Mixture-of-Experts routing stack.
//
// Rationale: if a single dense network cannot learn the target function, a
// Mixture-of-Experts router will only aggravate training by randomly assigning
// tokens to uninitialized experts. This package deliberately has no router, no
// experts, no gating, and no jitter — just a plain MLP.
package dense

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// DenseLayer is a single fully-connected layer with ReLU activation (except
// the output layer, which is followed by softmax during training).
type DenseLayer struct {
	InputDim  int
	OutputDim int
	Weights   []float32 // InputDim * OutputDim, row-major
	Bias      []float32 // OutputDim
}

// NewDenseLayer initializes weights with He initialization and zero bias.
func NewDenseLayer(inputDim, outputDim int) *DenseLayer {
	std := float32(math.Sqrt(2.0 / float64(inputDim)))
	w := make([]float32, inputDim*outputDim)
	for i := range w {
		w[i] = float32(rand.NormFloat64()) * std
	}
	return &DenseLayer{InputDim: inputDim, OutputDim: outputDim, Weights: w, Bias: make([]float32, outputDim)}
}

// DenseModel is the simplified single dense feed-forward network.
// Architecture: input -> [Linear+ReLU]x(n-1) -> Linear -> softmax(outputs).
type DenseModel struct {
	Layers  []*DenseLayer
	Outputs int
}

// NewDenseModel builds an MLP with the given hidden widths.
func NewDenseModel(inputDim int, hidden []int, outputs int) *DenseModel {
	m := &DenseModel{Outputs: outputs}
	prev := inputDim
	for _, h := range hidden {
		m.Layers = append(m.Layers, NewDenseLayer(prev, h))
		prev = h
	}
	m.Layers = append(m.Layers, NewDenseLayer(prev, outputs))
	return m
}

// Parameters returns all learnable weights and biases in flattened form.
func (m *DenseModel) Parameters() []float32 {
	var out []float32
	for _, l := range m.Layers {
		out = append(out, l.Weights...)
		out = append(out, l.Bias...)
	}
	return out
}

// Forward runs a forward pass and returns the pre-softmax logits (2D slim
// [batch][Outputs]) plus the cached layer pre-activations/activations needed
// for backward. It returns logits only; callers use Softmax/CrossEntropy.
func (m *DenseModel) Forward(inputs [][]float32) (logits [][]float32, acts [][][]float32, pre [][][]float32, err error) {
	cur := inputs
	for li, l := range m.Layers {
		isLast := li == len(m.Layers)-1
		out := make([][]float32, len(cur))
		preAct := make([][]float32, len(cur))
		for b := range cur {
			if len(cur[b]) != l.InputDim {
				return nil, nil, nil, fmt.Errorf("forward: expected input dim %d, got %d", l.InputDim, len(cur[b]))
			}
			z := matVec(l.Weights, l.Bias, cur[b], l.InputDim, l.OutputDim)
			preAct[b] = z
			a := make([]float32, len(z))
			copy(a, z)
			if !isLast {
				reluInPlace(a)
			}
			out[b] = a
		}
		cur = out
		pre = append(pre, preAct)
		acts = append(acts, out)
	}
	return cur, acts, pre, nil
}

// Backward computes gradients for all layers given cached activations and the
// softmax cross-entropy upstream gradient. Returns per-layer [weightsGrad, biasGrad].
func (m *DenseModel) Backward(inputs [][]float32, targets []int, logits [][]float32, acts [][][]float32, pre [][][]float32) ([][2][]float32, error) {
	// dLoss/dLogits = softmax(logits) - onehot(target), divided by batch.
	dOut := softmaxAndGrad(logits, targets)
	batch := float32(len(targets))
	for b := range dOut {
		for j := range dOut[b] {
			dOut[b][j] /= batch
		}
	}

	grads := make([][2][]float32, len(m.Layers))
	// Backprop through layers from last to first.
	for li := len(m.Layers) - 1; li >= 0; li-- {
		l := m.Layers[li]
		wGrad := make([]float32, len(l.Weights))
		bGrad := make([]float32, len(l.Bias))
		var inputBatch [][]float32
		if li == 0 {
			inputBatch = inputs
		} else {
			inputBatch = acts[li-1]
		}

		// dOut holds dL/dz for this layer (pre-activation).
		dZ := dOut
		// For non-last layers, dOut is dL/d(activation); multiply by ReLU' using pre.
		if li < len(m.Layers)-1 {
			for b := range dZ {
				for j := range dZ[b] {
					if pre[li][b][j] <= 0 {
						dZ[b][j] = 0
					}
				}
			}
		}

		// wGrad += dZ^T * input ; bGrad += sum_b dZ
		for b := range dZ {
			row := inputBatch[b]
			for j := 0; j < l.OutputDim; j++ {
				g := dZ[b][j]
				for i := 0; i < l.InputDim; i++ {
					wGrad[j*l.InputDim+i] += g * row[i]
				}
				bGrad[j] += g
			}
		}
		grads[li] = [2][]float32{wGrad, bGrad}

		// Compute dL/dInput for next (previous) layer: dInput = dZ * W^T
		if li > 0 {
			next := make([][]float32, len(dZ))
			for b := range dZ {
				din := make([]float32, l.InputDim)
				for i := 0; i < l.InputDim; i++ {
					var s float32
					for j := 0; j < l.OutputDim; j++ {
						s += dZ[b][j] * l.Weights[j*l.InputDim+i]
					}
					din[i] = s
				}
				next[b] = din
			}
			dOut = next
		}
	}
	return grads, nil
}

// Update applies gradient descent with the given learning rate to all layers.
func (m *DenseModel) Update(grads [][2][]float32, lr float32) {
	for li, l := range m.Layers {
		wg, bg := grads[li][0], grads[li][1]
		for i := range l.Weights {
			l.Weights[i] -= lr * wg[i]
		}
		for i := range l.Bias {
			l.Bias[i] -= lr * bg[i]
		}
	}
}

// Predict returns the argmax class index for each input.
func (m *DenseModel) Predict(inputs [][]float32) []int {
	logits, _, _, err := m.Forward(inputs)
	if err != nil {
		return nil
	}
	out := make([]int, len(logits))
	for b := range logits {
		best := 0
		for j := 1; j < len(logits[b]); j++ {
			if logits[b][j] > logits[b][best] {
				best = j
			}
		}
		out[b] = best
	}
	return out
}

// --- pure-math helpers (first principles, no external framework) ---

// matVec computes out[j] = sum_i W[j*in+i] * x[i] + bias[j].
func matVec(w []float32, b []float32, x []float32, in, out int) []float32 {
	res := make([]float32, out)
	for j := 0; j < out; j++ {
		var s float32
		base := j * in
		for i := 0; i < in; i++ {
			s += w[base+i] * x[i]
		}
		res[j] = s + b[j]
	}
	return res
}

func reluInPlace(a []float32) {
	for i := range a {
		if a[i] < 0 {
			a[i] = 0
		}
	}
}

func softmaxAndGrad(logits [][]float32, targets []int) [][]float32 {
	out := make([][]float32, len(logits))
	for b := range logits {
		row := logits[b]
		max := float32(math.Inf(-1))
		for _, v := range row {
			if v > max {
				max = v
			}
		}
		var sum float32
		probs := make([]float32, len(row))
		for j := range row {
			probs[j] = float32(math.Exp(float64(row[j] - max)))
			sum += probs[j]
		}
		for j := range probs {
			probs[j] /= sum
		}
		t := targets[b]
		for j := range probs {
			onehot := float32(0)
			if j == t {
				onehot = 1
			}
			probs[j] -= onehot
		}
		out[b] = probs
	}
	return out
}

// CrossEntropy computes mean softmax cross-entropy loss.
func CrossEntropy(logits [][]float32, targets []int) float32 {
	var total float32
	for b := range logits {
		row := logits[b]
		max := float32(math.Inf(-1))
		for _, v := range row {
			if v > max {
				max = v
			}
		}
		var expsum float32
		for _, v := range row {
			expsum += float32(math.Exp(float64(v - max)))
		}
		for j, v := range row {
			if j == targets[b] {
				total += max + float32(math.Log(float64(expsum))) - v
				break
			}
		}
	}
	return total / float32(len(logits))
}

// Accuracy returns the fraction of exact matches.
func Accuracy(pred, want []int) float32 {
	if len(pred) != len(want) {
		return 0
	}
	var ok int
	for i := range pred {
		if pred[i] == want[i] {
			ok++
		}
	}
	return float32(ok) / float32(len(pred))
}

// ErrNaN indicates the model diverged during training.
var ErrNaN = errors.New("nan or inf encountered during training")
