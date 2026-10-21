package transformer

import (
	"math"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// PositionalEncoding implements sinusoidal positional encoding
// as in "Attention Is All You Need".
type PositionalEncoding struct {
	DModel  int
	MaxLen  int
	peTable *tensor.Tensor // [max_len, d_model]
}

// NewPositionalEncoding creates sinusoidal positional encodings.
// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
func NewPositionalEncoding(dModel, maxLen int) *PositionalEncoding {
	data := make([]float32, maxLen*dModel)
	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < dModel; i += 2 {
			divTerm := math.Pow(10000, float64(2*(i/2))/float64(dModel))
			data[pos*dModel+i] = float32(math.Sin(float64(pos) / divTerm))
			if i+1 < dModel {
				data[pos*dModel+i+1] = float32(math.Cos(float64(pos) / divTerm))
			}
		}
	}
	return &PositionalEncoding{
		DModel:  dModel,
		MaxLen:  maxLen,
		peTable: tensor.NewTensor([]int{maxLen, dModel}, data, false),
	}
}

// Forward adds positional encoding to input.
// input: [batch, seq_len, d_model]
// Returns: [batch, seq_len, d_model]
//
// The addition goes through the tensor autograd graph (AddOperation) so
// gradients flow back to the input (e.g. the embedding lookup). A plain
// data copy here would silently detach the graph and freeze the embeddings.
func (p *PositionalEncoding) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	batch := input.Shape[0]
	seqLen := input.Shape[1]
	if seqLen > p.MaxLen {
		seqLen = p.MaxLen
	}

	// Build the positional contribution as its own tensor (no grad needed:
	// the table is fixed sinusoids).
	peData := p.peTable.Data
	peOnly := make([]float32, len(input.Data))
	for b := 0; b < batch; b++ {
		for s := 0; s < seqLen; s++ {
			for d := 0; d < p.DModel; d++ {
				idx := b*seqLen*p.DModel + s*p.DModel + d
				peOnly[idx] += peData[s*p.DModel+d]
			}
		}
	}
	pe := tensor.NewTensor(input.Shape, peOnly, false)

	return input.Add(pe)
}
