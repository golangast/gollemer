package transformer

import (
	"fmt"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// TransformerEncoderLayer is a single Transformer encoder layer:
// Multi-head self-attention + Feed-forward, each with residual + layer norm.
type TransformerEncoderLayer struct {
	DModel int

	SelfAttn *MultiHeadAttention
	Norm1    *nn.LayerNorm
	Norm2    *nn.LayerNorm

	FF1 *nn.Linear // d_model -> d_ff
	FF2 *nn.Linear // d_ff -> d_model
}

// NewTransformerEncoderLayer creates an encoder layer.
func NewTransformerEncoderLayer(dModel, numHeads, dFF int) (*TransformerEncoderLayer, error) {
	attn, err := NewMultiHeadAttention(dModel, numHeads)
	if err != nil {
		return nil, fmt.Errorf("self-attn: %w", err)
	}
	ff1, err := nn.NewLinear(dModel, dFF)
	if err != nil {
		return nil, fmt.Errorf("ff1: %w", err)
	}
	ff2, err := nn.NewLinear(dFF, dModel)
	if err != nil {
		return nil, fmt.Errorf("ff2: %w", err)
	}
	return &TransformerEncoderLayer{
		DModel:   dModel,
		SelfAttn: attn,
		Norm1:    nn.NewLayerNorm(dModel),
		Norm2:    nn.NewLayerNorm(dModel),
		FF1:      ff1,
		FF2:      ff2,
	}, nil
}

// Forward: input [batch, seq, d_model] -> [batch, seq, d_model]
func (l *TransformerEncoderLayer) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Self-attention with residual
	attnOut, err := l.SelfAttn.Forward(x, x, x, nil)
	if err != nil {
		return nil, fmt.Errorf("self-attn: %w", err)
	}
	res1, err := x.Add(attnOut)
	if err != nil {
		return nil, fmt.Errorf("residual1: %w", err)
	}
	norm1Out, err := l.Norm1.Forward(res1)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}

	// Feed-forward with residual
	ff1Out, err := l.FF1.Forward(norm1Out)
	if err != nil {
		return nil, fmt.Errorf("ff1: %w", err)
	}
	// ReLU activation (graph-tracked: gradients flow through correctly)
	reluOut, err := ff1Out.ReLU()
	if err != nil {
		return nil, fmt.Errorf("relu: %w", err)
	}
	ff2Out, err := l.FF2.Forward(reluOut)
	if err != nil {
		return nil, fmt.Errorf("ff2: %w", err)
	}
	res2, err := norm1Out.Add(ff2Out)
	if err != nil {
		return nil, fmt.Errorf("residual2: %w", err)
	}
	return l.Norm2.Forward(res2)
}

// TransformerDecoderLayer is a single Transformer decoder layer:
// Masked self-attention + Cross-attention + Feed-forward.
type TransformerDecoderLayer struct {
	DModel int

	SelfAttn  *MultiHeadAttention
	CrossAttn *MultiHeadAttention
	Norm1     *nn.LayerNorm
	Norm2     *nn.LayerNorm
	Norm3     *nn.LayerNorm

	FF1 *nn.Linear
	FF2 *nn.Linear
}

// NewTransformerDecoderLayer creates a decoder layer.
func NewTransformerDecoderLayer(dModel, numHeads, dFF int) (*TransformerDecoderLayer, error) {
	selfAttn, err := NewMultiHeadAttention(dModel, numHeads)
	if err != nil {
		return nil, fmt.Errorf("self-attn: %w", err)
	}
	crossAttn, err := NewMultiHeadAttention(dModel, numHeads)
	if err != nil {
		return nil, fmt.Errorf("cross-attn: %w", err)
	}
	ff1, err := nn.NewLinear(dModel, dFF)
	if err != nil {
		return nil, fmt.Errorf("ff1: %w", err)
	}
	ff2, err := nn.NewLinear(dFF, dModel)
	if err != nil {
		return nil, fmt.Errorf("ff2: %w", err)
	}
	return &TransformerDecoderLayer{
		DModel:    dModel,
		SelfAttn:  selfAttn,
		CrossAttn: crossAttn,
		Norm1:     nn.NewLayerNorm(dModel),
		Norm2:     nn.NewLayerNorm(dModel),
		Norm3:     nn.NewLayerNorm(dModel),
		FF1:       ff1,
		FF2:       ff2,
	}, nil
}

// Forward: x [batch, tgt_seq, d_model], memory [batch, src_seq, d_model]
// Returns: [batch, tgt_seq, d_model]
func (l *TransformerDecoderLayer) Forward(x, memory *tensor.Tensor) (*tensor.Tensor, error) {
	tgtSeq := x.Shape[1]

	// Masked self-attention with residual
	causalMask := CausalMask(tgtSeq)
	selfAttnOut, err := l.SelfAttn.Forward(x, x, x, causalMask)
	if err != nil {
		return nil, fmt.Errorf("self-attn: %w", err)
	}
	res1, err := x.Add(selfAttnOut)
	if err != nil {
		return nil, fmt.Errorf("residual1: %w", err)
	}
	norm1Out, err := l.Norm1.Forward(res1)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}

	// Cross-attention with residual
	crossAttnOut, err := l.CrossAttn.Forward(norm1Out, memory, memory, nil)
	if err != nil {
		return nil, fmt.Errorf("cross-attn: %w", err)
	}
	res2, err := norm1Out.Add(crossAttnOut)
	if err != nil {
		return nil, fmt.Errorf("residual2: %w", err)
	}
	norm2Out, err := l.Norm2.Forward(res2)
	if err != nil {
		return nil, fmt.Errorf("norm2: %w", err)
	}

	// Feed-forward with residual
	ff1Out, err := l.FF1.Forward(norm2Out)
	if err != nil {
		return nil, fmt.Errorf("ff1: %w", err)
	}
	// ReLU activation (graph-tracked: gradients flow through correctly)
	reluOut, err := ff1Out.ReLU()
	if err != nil {
		return nil, fmt.Errorf("relu: %w", err)
	}
	ff2Out, err := l.FF2.Forward(reluOut)
	if err != nil {
		return nil, fmt.Errorf("ff2: %w", err)
	}
	res3, err := norm2Out.Add(ff2Out)
	if err != nil {
		return nil, fmt.Errorf("residual3: %w", err)
	}
	return l.Norm3.Forward(res3)
}
