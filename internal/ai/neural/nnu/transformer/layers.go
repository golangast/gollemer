package transformer

import (
	"fmt"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// normOp adapts nn.LayerNorm to the tensor autograd Operation interface.
//
// The current tree's nn.LayerNorm computes its forward and backward
// correctly but never registers itself as its output's Creator (that
// hookup exists on the fix-nan lineage this package was ported from).
// Without it the norm output is a graph leaf: gamma/beta receive no
// gradient and — in the original post-norm layout — nearly all gradient
// flow through the stack died (44 of 46 params dead in the gradflow
// smoke test). Setting the Creator here restores the intended graph
// without touching the shared nn package.
type normOp struct {
	ln    *nn.LayerNorm
	input *tensor.Tensor
}

func (o *normOp) Inputs() []*tensor.Tensor { return []*tensor.Tensor{o.input} }

func (o *normOp) Backward(grad *tensor.Tensor) error { return o.ln.Backward(grad) }

// applyLayerNorm runs ln.Forward and re-attaches the output to the
// autograd graph via normOp.
func applyLayerNorm(ln *nn.LayerNorm, x *tensor.Tensor) (*tensor.Tensor, error) {
	out, err := ln.Forward(x)
	if err != nil {
		return nil, err
	}
	out.Creator = &normOp{ln: ln, input: x}
	return out, nil
}

// TransformerEncoderLayer is a single Transformer encoder layer:
// Multi-head self-attention + Feed-forward in PRE-norm form (LayerNorm
// before each sublayer, residual around it) — converted from the ported
// post-norm form for the mid-transformer experiment: post-norm + constant
// LR was the October NaN recipe; pre-norm keeps the residual stream
// unnormalized so gradient magnitudes stay stable from random init.
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
// Pre-norm: x + Attn(Norm1(x)), then res + FFN(Norm2(res)).
func (l *TransformerEncoderLayer) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	// Self-attention sublayer on the normalized stream
	norm1Out, err := applyLayerNorm(l.Norm1, x)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}
	attnOut, err := l.SelfAttn.Forward(norm1Out, norm1Out, norm1Out, nil)
	if err != nil {
		return nil, fmt.Errorf("self-attn: %w", err)
	}
	res1, err := x.Add(attnOut)
	if err != nil {
		return nil, fmt.Errorf("residual1: %w", err)
	}

	// Feed-forward sublayer on the normalized residual stream
	norm2Out, err := applyLayerNorm(l.Norm2, res1)
	if err != nil {
		return nil, fmt.Errorf("norm2: %w", err)
	}
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
	res2, err := res1.Add(ff2Out)
	if err != nil {
		return nil, fmt.Errorf("residual2: %w", err)
	}
	return res2, nil
}

// TransformerDecoderLayer is a single Transformer decoder layer:
// Masked self-attention + Cross-attention + Feed-forward, in PRE-norm
// form (see TransformerEncoderLayer for why).
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
// Pre-norm: x + SelfAttn(Norm1(x)), res + CrossAttn(Norm2(res), memory),
// res + FFN(Norm3(res)).
func (l *TransformerDecoderLayer) Forward(x, memory *tensor.Tensor) (*tensor.Tensor, error) {
	tgtSeq := x.Shape[1]

	// Masked self-attention sublayer on the normalized stream
	norm1Out, err := applyLayerNorm(l.Norm1, x)
	if err != nil {
		return nil, fmt.Errorf("norm1: %w", err)
	}
	causalMask := CausalMask(tgtSeq)
	selfAttnOut, err := l.SelfAttn.Forward(norm1Out, norm1Out, norm1Out, causalMask)
	if err != nil {
		return nil, fmt.Errorf("self-attn: %w", err)
	}
	res1, err := x.Add(selfAttnOut)
	if err != nil {
		return nil, fmt.Errorf("residual1: %w", err)
	}

	// Cross-attention sublayer on the normalized residual stream
	norm2Out, err := applyLayerNorm(l.Norm2, res1)
	if err != nil {
		return nil, fmt.Errorf("norm2: %w", err)
	}
	crossAttnOut, err := l.CrossAttn.Forward(norm2Out, memory, memory, nil)
	if err != nil {
		return nil, fmt.Errorf("cross-attn: %w", err)
	}
	res2, err := res1.Add(crossAttnOut)
	if err != nil {
		return nil, fmt.Errorf("residual2: %w", err)
	}

	// Feed-forward sublayer on the normalized residual stream
	norm3Out, err := applyLayerNorm(l.Norm3, res2)
	if err != nil {
		return nil, fmt.Errorf("norm3: %w", err)
	}
	ff1Out, err := l.FF1.Forward(norm3Out)
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
	res3, err := res2.Add(ff2Out)
	if err != nil {
		return nil, fmt.Errorf("residual3: %w", err)
	}
	return res3, nil
}
