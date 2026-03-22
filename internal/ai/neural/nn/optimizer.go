package nn

import (
	"math"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Optimizer interface defines the contract for optimizers.
type Optimizer interface {
	Step()
	ZeroGrad()
	ClipGradients()
	SetLearningRate(lr float64)
}

// Adam represents the Adam optimizer.
type Adam struct {
	parameters   []*Tensor
	learningRate float64
	beta1        float64
	beta2        float64
	epsilon      float64
	t            int
	m            map[*Tensor]*Tensor // 1st moment vector
	v            map[*Tensor]*Tensor // 2nd moment vector
	clipValue    float64
	WeightDecay  float64
}

// NewOptimizer creates a new Adam optimizer.
func NewOptimizer(parameters []*Tensor, learningRate float64, clipValue float64) Optimizer {
	return &Adam{
		parameters:   parameters,
		learningRate: learningRate,
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-8,
		t:            0,
		m:            make(map[*Tensor]*Tensor),
		v:            make(map[*Tensor]*Tensor),
		clipValue:    clipValue,
		WeightDecay:  0.0001, // Default weight decay
	}
}

// ClipGradients scales the gradients of all parameters if their total L2 norm exceeds clipValue.
func (o *Adam) ClipGradients() {
	if o.clipValue <= 0 {
		return
	}
	totalNorm := 0.0
	for _, p := range o.parameters {
		if p.Grad != nil {
			for _, g := range p.Grad.Data {
				totalNorm += g * g
			}
		}
	}
	totalNorm = math.Sqrt(totalNorm)
	if totalNorm > o.clipValue {
		scale := o.clipValue / totalNorm
		for _, p := range o.parameters {
			if p.Grad != nil {
				for i := range p.Grad.Data {
					p.Grad.Data[i] *= scale
				}
			}
		}
	}
}

// Step performs a single optimization step.
func (o *Adam) Step() {
	o.t++

	for _, p := range o.parameters {
		if p.Grad == nil {
			continue
		}

		if _, ok := o.m[p]; !ok {
			o.m[p] = NewTensor(p.Shape, make([]float64, len(p.Data)), false)
			o.v[p] = NewTensor(p.Shape, make([]float64, len(p.Data)), false)
		}

		m := o.m[p].Data
		v := o.v[p].Data
		grad := p.Grad.Data
		param := p.Data
		
		AdamWUpdate(param, grad, m, v, o.learningRate, o.beta1, o.beta2, o.epsilon, o.WeightDecay, o.t)
	}
}

// ZeroGrad resets the gradients of all parameters.
func (o *Adam) ZeroGrad() {
	for _, p := range o.parameters {
		p.ZeroGrad()
	}
}

// SetLearningRate updates the learning rate of the optimizer
func (o *Adam) SetLearningRate(lr float64) {
	o.learningRate = lr
}
