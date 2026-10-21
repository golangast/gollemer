package transformer

import (
	"fmt"
	"math"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// MultiHeadAttention implements scaled dot-product multi-head attention
// as in "Attention Is All You Need" (Vaswani et al. 2017).
type MultiHeadAttention struct {
	NumHeads int
	DModel   int
	DHead    int

	WQ *nn.Linear // Query projection: d_model -> d_model
	WK *nn.Linear // Key projection: d_model -> d_model
	WV *nn.Linear // Value projection: d_model -> d_model
	WO *nn.Linear // Output projection: d_model -> d_model
}

// NewMultiHeadAttention creates a multi-head attention layer.
// dModel must be divisible by numHeads.
func NewMultiHeadAttention(dModel, numHeads int) (*MultiHeadAttention, error) {
	if dModel%numHeads != 0 {
		return nil, fmt.Errorf("dModel (%d) must be divisible by numHeads (%d)", dModel, numHeads)
	}
	wq, err := nn.NewLinear(dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("WQ: %w", err)
	}
	wk, err := nn.NewLinear(dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("WK: %w", err)
	}
	wv, err := nn.NewLinear(dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("WV: %w", err)
	}
	wo, err := nn.NewLinear(dModel, dModel)
	if err != nil {
		return nil, fmt.Errorf("WO: %w", err)
	}
	return &MultiHeadAttention{
		NumHeads: numHeads,
		DModel:   dModel,
		DHead:    dModel / numHeads,
		WQ:       wq,
		WK:       wk,
		WV:       wv,
		WO:       wo,
	}, nil
}

// Forward performs multi-head attention.
// query, key, value: [batch, seq_len, d_model]
// mask (optional): [batch, 1, seq_q, seq_k] — additive mask (-inf for masked positions)
// Returns: [batch, seq_q, d_model]
func (m *MultiHeadAttention) Forward(query, key, value *tensor.Tensor, mask *tensor.Tensor) (*tensor.Tensor, error) {
	batch := query.Shape[0]
	seqQ := query.Shape[1]
	seqK := key.Shape[1]

	// Project to Q, K, V: [batch, seq, d_model]
	q, err := m.WQ.Forward(query)
	if err != nil {
		return nil, fmt.Errorf("WQ forward: %w", err)
	}
	k, err := m.WK.Forward(key)
	if err != nil {
		return nil, fmt.Errorf("WK forward: %w", err)
	}
	v, err := m.WV.Forward(value)
	if err != nil {
		return nil, fmt.Errorf("WV forward: %w", err)
	}

	// Split into heads: [batch, seq, num_heads, d_head] -> [batch*num_heads, seq, d_head]
	qHeads, err := splitHeads(q, m.NumHeads, m.DHead)
	if err != nil {
		return nil, fmt.Errorf("split Q heads: %w", err)
	}
	kHeads, err := splitHeads(k, m.NumHeads, m.DHead)
	if err != nil {
		return nil, fmt.Errorf("split K heads: %w", err)
	}
	vHeads, err := splitHeads(v, m.NumHeads, m.DHead)
	if err != nil {
		return nil, fmt.Errorf("split V heads: %w", err)
	}

	// Scaled dot-product attention for each head
	// scores = Q @ K^T / sqrt(d_head): [batch*heads, seq_q, seq_k]
	batchHeads := batch * m.NumHeads
	scale := 1.0 / math.Sqrt(float64(m.DHead))

	// K^T: [batch*heads, d_head, seq_k]
	kT, err := kHeads.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("K transpose: %w", err)
	}

	scores, err := qHeads.MatMul(kT)
	if err != nil {
		return nil, fmt.Errorf("QK^T: %w", err)
	}

	// Scale
	scoresData := scores.Data
	for i := range scoresData {
		scoresData[i] *= float32(scale)
	}

	// Apply mask (additive, -inf for masked)
	if mask != nil {
		// mask: [batch, 1, seq_q, seq_k] -> broadcast to [batch*heads, seq_q, seq_k]
		if err := applyMask(scores, mask, batch, m.NumHeads, seqQ, seqK); err != nil {
			return nil, fmt.Errorf("apply mask: %w", err)
		}
	}

	// Softmax over keys: [batch*heads, seq_q, seq_k]
	attnWeights, err := scores.Softmax(2)
	if err != nil {
		return nil, fmt.Errorf("softmax: %w", err)
	}

	// attn @ V: [batch*heads, seq_q, d_head]
	attnOut, err := attnWeights.MatMul(vHeads)
	if err != nil {
		return nil, fmt.Errorf("attn@V: %w", err)
	}

	// Combine heads: [batch*heads, seq_q, d_head] -> [batch, seq_q, d_model]
	combined, err := combineHeads(attnOut, batch, m.NumHeads, seqQ, m.DHead)
	if err != nil {
		return nil, fmt.Errorf("combine heads: %w", err)
	}

	// Output projection: [batch, seq_q, d_model]
	out, err := m.WO.Forward(combined)
	if err != nil {
		return nil, fmt.Errorf("WO forward: %w", err)
	}

	_ = batchHeads
	return out, nil
}

// splitHeads reshapes [batch, seq, d_model] -> [batch*num_heads, seq, d_head]
func splitHeads(t *tensor.Tensor, numHeads, dHead int) (*tensor.Tensor, error) {
	batch := t.Shape[0]
	seq := t.Shape[1]
	// Reshape to [batch, seq, num_heads, d_head]
	reshaped, err := t.Reshape([]int{batch, seq, numHeads, dHead})
	if err != nil {
		return nil, err
	}
	// Transpose to [batch, num_heads, seq, d_head]
	transposed, err := reshaped.Transpose(1, 2)
	if err != nil {
		return nil, err
	}
	// Reshape to [batch*num_heads, seq, d_head]
	return transposed.Reshape([]int{batch * numHeads, seq, dHead})
}

// combineHeads reshapes [batch*num_heads, seq, d_head] -> [batch, seq, d_model]
func combineHeads(t *tensor.Tensor, batch, numHeads, seq, dHead int) (*tensor.Tensor, error) {
	// Reshape to [batch, num_heads, seq, d_head]
	reshaped, err := t.Reshape([]int{batch, numHeads, seq, dHead})
	if err != nil {
		return nil, err
	}
	// Transpose to [batch, seq, num_heads, d_head]
	transposed, err := reshaped.Transpose(1, 2)
	if err != nil {
		return nil, err
	}
	// Reshape to [batch, seq, d_model]
	return transposed.Reshape([]int{batch, seq, numHeads * dHead})
}

// applyMask adds the mask to scores (mask contains 0 for keep, -inf for block)
// mask: [1, 1, seq_q, seq_k], scores: [batch*heads, seq_q, seq_k]
func applyMask(scores *tensor.Tensor, mask *tensor.Tensor, batch, numHeads, seqQ, seqK int) error {
	scoresData := scores.Data
	maskData := mask.Data
	for bh := 0; bh < batch*numHeads; bh++ {
		for i := 0; i < seqQ; i++ {
			for j := 0; j < seqK; j++ {
				maskVal := maskData[i*seqK+j]
				scoresData[bh*seqQ*seqK+i*seqK+j] += maskVal
			}
		}
	}
	return nil
}

// CausalMask creates a causal (lower-triangular) mask for decoder self-attention.
// Returns [1, 1, seq, seq] with 0 for allowed, -1e9 for masked.
func CausalMask(seq int) *tensor.Tensor {
	data := make([]float32, seq*seq)
	for i := 0; i < seq; i++ {
		for j := 0; j < seq; j++ {
			if j > i {
				data[i*seq+j] = -1e9
			}
		}
	}
	return tensor.NewTensor([]int{1, 1, seq, seq}, data, false)
}
