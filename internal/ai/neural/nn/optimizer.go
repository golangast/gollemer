package nn

import (
	"fmt"
	"math"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// Optimizer interface defines the contract for optimizers.
type Optimizer interface {
	Step()
	ZeroGrad()
	ClipGradients()
	SetLearningRate(lr float64)
	GetLearningRate() float64
}

// TrainingProfile represents a preset for training hyperparameters.
type TrainingProfile struct {
	Name           string
	LR             float64
	Lambda         float64 // Weight Decay
	ClipThreshold  float64 // Gradient Clipping
	HealThreshold  float64 // When to reset experts (e.g., 0.85)
	WarmupSteps    int     // Steps before applying full LR
}

func GetProfile(name string) TrainingProfile {
	switch name {
	case "aggressive":
		return TrainingProfile{
			Name: "Aggressive", LR: 2e-3, Lambda: 0.005, 
			ClipThreshold: 2.0, HealThreshold: 0.70, WarmupSteps: 100,
		}
	case "stable":
		return TrainingProfile{
			Name: "Stable", LR: 5e-4, Lambda: 0.01, 
			ClipThreshold: 1.0, HealThreshold: 0.90, WarmupSteps: 500,
		}
	default:
		// Default "Standard" profile
		return TrainingProfile{
			Name: "Standard", LR: 1e-3, Lambda: 0.01, 
			ClipThreshold: 1.0, HealThreshold: 0.85, WarmupSteps: 200,
		}
	}
}

// Adam represents the Adam optimizer.
type Adam struct {
	parameters    []*Tensor
	learningRate  float64
	beta1         float64
	beta2         float64
	epsilon       float64
	t             int
	m             map[*Tensor]*Tensor // 1st moment vector
	v             map[*Tensor]*Tensor // 2nd moment vector
	ClipThreshold float64
	Lambda        float64 // Weight decay (Lambda)
	RouterLR      float64 // Different learning rate for router parameters
}

// NewOptimizer creates a new Adam optimizer.
func NewOptimizer(parameters []*Tensor, learningRate float64, clipThreshold float64) Optimizer {
	return &Adam{
		parameters:    parameters,
		learningRate:  learningRate,
		beta1:         0.9,
		beta2:         0.999,
		epsilon:       1e-8,
		t:             0,
		m:             make(map[*Tensor]*Tensor),
		v:             make(map[*Tensor]*Tensor),
		ClipThreshold: clipThreshold,
		Lambda:        0.01, // Default weight decay
		RouterLR:      learningRate / 10.0, // Default to slower LR for routers
	}
}

// ClipGradients scales the gradients of all parameters if their total L2 norm exceeds ClipThreshold.
func (o *Adam) ClipGradients() {
	if o.ClipThreshold <= 0 {
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
	if totalNorm > o.ClipThreshold {
		scale := o.ClipThreshold / totalNorm
		for _, p := range o.parameters {
			if p.Grad != nil {
				// Use SIMD-accelerated scalar multiplication for consistent global scaling
				MulScalar(p.Grad.Data, scale, p.Grad.Data)
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
		
		lr := o.learningRate
		if p.IsRouter {
			lr = o.RouterLR
		}

		AdamWUpdate(param, grad, m, v, lr, o.beta1, o.beta2, o.epsilon, o.Lambda, o.t)
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

// GetLearningRate returns the current learning rate
func (o *Adam) GetLearningRate() float64 {
	return o.learningRate
}

// SetRouterLR updates the router-specific learning rate
func (o *Adam) SetRouterLR(lr float64) {
	o.RouterLR = lr
}

// GetRouterLR returns the current router learning rate
func (o *Adam) GetRouterLR() float64 {
	return o.RouterLR
}

// CoolingOptimizer wraps an existing Optimizer to provide a recovery period
type CoolingOptimizer struct {
	Base       Optimizer // Your SIMD AdamW / SGD
	Cooldown   int       // Remaining steps in cooldown
	IsActive   bool
	OriginalLR float64   // To store where we should be post-recovery
}

// Step executes the base optimizer but overrides LR if cooling
func (o *CoolingOptimizer) Step() {
	if o.IsActive {
		o.Cooldown--
		if o.Cooldown <= 0 {
			o.IsActive = false
			fmt.Printf("❄️ Recovery period over. Experts stabilized. Resuming LR: %.6f\n", o.OriginalLR)
			o.Base.SetLearningRate(o.OriginalLR)
		}
	}
	o.Base.Step()
}

// ZeroGrad resets the gradients of all parameters.
func (o *CoolingOptimizer) ZeroGrad() {
	o.Base.ZeroGrad()
}

// ClipGradients scales the gradients of all parameters.
func (o *CoolingOptimizer) ClipGradients() {
	o.Base.ClipGradients()
}

// SetLearningRate updates the learning rate of the optimizer
func (o *CoolingOptimizer) SetLearningRate(lr float64) {
	if o.IsActive {
		// During cooling, we don't allow permanent LR updates to the base,
		// but we store it as the original LR for when we finish cooling.
		o.OriginalLR = lr
	} else {
		o.Base.SetLearningRate(lr)
	}
}

// GetLearningRate returns the current learning rate
func (o *CoolingOptimizer) GetLearningRate() float64 {
	return o.Base.GetLearningRate()
}

// Trigger enters the cooling state
func (o *CoolingOptimizer) Trigger(steps int, reduction float64) {
	if !o.IsActive {
		o.OriginalLR = o.Base.GetLearningRate()
	}
	
	o.IsActive = true
	o.Cooldown = steps
	
	// Drop the LR immediately to allow "shaken" weights to settle
	current := o.Base.GetLearningRate()
	o.Base.SetLearningRate(current * reduction)
	fmt.Printf("📉 Circuit Breaker: Cooling for %d steps at LR %f (Was: %f)\n", steps, current*reduction, current)
}
