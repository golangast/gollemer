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
	SetLearningRate(lr float32)
	GetLearningRate() float32
	ResetStagnantMoments(t *Tensor)
}

// TrainingProfile represents a preset for training hyperparameters.
type TrainingProfile struct {
	Name           string
	LR             float32
	Lambda         float32 // Weight Decay
	ClipThreshold  float32 // Gradient Clipping
	HealThreshold  float32 // When to reset experts (e.g., 0.85)
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
	learningRate  float32
	beta1         float32
	beta2         float32
	epsilon       float32
	t             int
	m             map[*Tensor]*Tensor // 1st moment vector
	v             map[*Tensor]*Tensor // 2nd moment vector
	ClipThreshold float32
	Lambda        float32 // Weight decay (Lambda)
	RouterLR      float32 // Different learning rate for router parameters
}

// NewOptimizer creates a new Adam optimizer.
func NewOptimizer(parameters []*Tensor, learningRate float32, clipThreshold float32) Optimizer {
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
	totalNorm := float32(0.0)
	for _, p := range o.parameters {
		if p.Grad != nil {
			for _, g := range p.Grad.Data {
				totalNorm += g * g
			}
		}
	}
	totalNorm = float32(math.Sqrt(float64(totalNorm)))
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

// ResetStagnantMoments clears Adam moment vectors for weights identified as stagnant.
// This prevents the search history from damping the effect of a nudge (perturbation).
func (o *Adam) ResetStagnantMoments(t *Tensor) {
	if t.TimidMask() == nil {
		return
	}
	m, okM := o.m[t]
	v, okV := o.v[t]
	if !okM || !okV {
		return
	}
	for i, stagnant := range t.TimidMask() {
		if stagnant {
			m.Data[i] = 0
			v.Data[i] = 0
		}
	}
}

// Step performs a single optimization step, now with Timid-Aware LR boosting.
func (o *Adam) Step() {
	o.t++

	for _, p := range o.parameters {
		if p.Grad == nil {
			continue
		}

		if _, ok := o.m[p]; !ok {
			o.m[p] = NewTensor(p.Shape, make([]float32, len(p.Data)), false)
			o.v[p] = NewTensor(p.Shape, make([]float32, len(p.Data)), false)
		}

		m := o.m[p].Data
		v := o.v[p].Data
		grad := p.Grad.Data
		param := p.Data
		
		lr := o.learningRate
		if p.IsRouter {
			lr = o.RouterLR
		}

		// --- Timid Boost Integration ---
		// If TimidMask exists for this tensor, we apply localized LR boosting.
		// Note: This falls back to standard SIMD update if no mask is present.
		if p.TimidMask() != nil {
			// Bias correction terms
			b1t := float32(1.0 - math.Pow(float64(o.beta1), float64(o.t)))
			b2t := float32(1.0 - math.Pow(float64(o.beta2), float64(o.t)))
			
			mask := p.TimidMask()
			for i := range param {
				// 1. Update moments
				m[i] = o.beta1*m[i] + (1-o.beta1)*grad[i]
				v[i] = o.beta2*v[i] + (1-o.beta2)*grad[i]*grad[i]
				
				// 2. Localized LR boost
				effectiveLR := lr
				if mask[i] {
					effectiveLR *= 3.0 // Kick stagnant weights harder
				}
				
				// 3. AdamW Update: weight -= lr * (m_corrected / (sqrt(v_corrected) + eps) + lambda * weight)
				mHat := m[i] / b1t
				vHat := v[i] / b2t
				denom := float32(math.Sqrt(float64(vHat))) + o.epsilon
				
				update := (mHat / denom) + (o.Lambda * param[i])
				param[i] -= effectiveLR * update
			}
		} else {
			// Perform high-performance SIMD update when no mask is present
			AdamWUpdate(param, grad, m, v, lr, o.beta1, o.beta2, o.epsilon, o.Lambda, o.t)
		}
	}
}

// ZeroGrad resets the gradients of all parameters.
func (o *Adam) ZeroGrad() {
	for _, p := range o.parameters {
		p.ZeroGrad()
	}
}

// SetLearningRate updates the main learning rate and proportionally scales the RouterLR.
func (o *Adam) SetLearningRate(lr float32) {
	o.learningRate = lr
	// Keep RouterLR at 10% of the main learning rate to ensure stability in the MoE gating network.
	o.RouterLR = lr * 0.1
}

// GetLearningRate returns the current learning rate
func (o *Adam) GetLearningRate() float32 {
	return o.learningRate
}

// SetRouterLR updates the router-specific learning rate
func (o *Adam) SetRouterLR(lr float32) {
	o.RouterLR = lr
}

// GetRouterLR returns the current router learning rate
func (o *Adam) GetRouterLR() float32 {
	return o.RouterLR
}

// CoolingOptimizer wraps an existing Optimizer to provide a recovery period
type CoolingOptimizer struct {
	Base       Optimizer // Your SIMD AdamW / SGD
	Cooldown   int       // Remaining steps in cooldown
	IsActive   bool
	OriginalLR float32   // To store where we should be post-recovery
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
func (o *CoolingOptimizer) SetLearningRate(lr float32) {
	if o.IsActive {
		// During cooling, we don't allow permanent LR updates to the base,
		// but we store it as the original LR for when we finish cooling.
		o.OriginalLR = lr
	} else {
		o.Base.SetLearningRate(lr)
	}
}

// GetLearningRate returns the current learning rate
func (o *CoolingOptimizer) GetLearningRate() float32 {
	return o.Base.GetLearningRate()
}

// ResetStagnantMoments delegates to the base optimizer.
func (o *CoolingOptimizer) ResetStagnantMoments(t *Tensor) {
	o.Base.ResetStagnantMoments(t)
}

// Trigger enters the cooling state
func (o *CoolingOptimizer) Trigger(steps int, reduction float32) {
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
