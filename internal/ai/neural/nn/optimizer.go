package nn

import (
	"compress/gzip"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// OptimizerState holds serializable Adam optimizer state for checkpoint/resume.
// It lets training resume without the cold-start regression that occurs when
// Adam moments are zeroed on every process restart.
type OptimizerState struct {
	T     int         // Global step counter
	MData [][]float32 // 1st moment, indexed by parameter order
	VData [][]float32 // 2nd moment, indexed by parameter order
}

// Optimizer interface defines the contract for optimizers.
type Optimizer interface {
	Step()
	ZeroGrad()
	ClipGradients()
	SetLearningRate(lr float32)
	GetLearningRate() float32
	ResetStagnantMoments(t *Tensor)
	// ResetAllMoments wipes all Adam m/v accumulators so bad momentum can't
	// keep pushing weights in the wrong direction after a divergence event.
	ResetAllMoments()
}

// TrainingProfile represents a preset for training hyperparameters.
type TrainingProfile struct {
	Name          string
	LR            float32
	Lambda        float32 // Weight Decay
	ClipThreshold float32 // Gradient Clipping
	HealThreshold float32 // When to reset experts (e.g., 0.85)
	WarmupSteps   int     // Steps before applying full LR
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
	HealThreshold float32 // When to reset experts (e.g., 0.85)
	WarmupSteps   int     // Steps before applying full LR
	RouterLR      float32 // Different learning rate for router parameters
}

// NewOptimizer creates a new Adam optimizer.
// M/V moment tensors are allocated lazily on first Step() to avoid a
// burst memory spike (2× model params) that can trigger the OOM killer on
// memory-constrained hardware.
func NewOptimizer(parameters []*Tensor, learningRate float32, clipThreshold float32) Optimizer {
	o := &Adam{
		parameters:    parameters,
		learningRate:  learningRate,
		beta1:         0.9,
		beta2:         0.999,
		epsilon:       1e-8,
		t:             0,
		m:             make(map[*Tensor]*Tensor),
		v:             make(map[*Tensor]*Tensor),
		ClipThreshold: clipThreshold,
		Lambda:        0.001,
		RouterLR:      learningRate * 15.0,
	}
	return o
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

// ResetAllMoments wipes every Adam m/v accumulator to zero and resets the step
// counter. Call after a divergence rollback so stale momentum can't keep
// driving the weights in the wrong direction at the reduced learning rate.
func (o *Adam) ResetAllMoments() {
	for _, p := range o.parameters {
		if m, ok := o.m[p]; ok {
			for i := range m.Data {
				m.Data[i] = 0
			}
		}
		if v, ok := o.v[p]; ok {
			for i := range v.Data {
				v.Data[i] = 0
			}
		}
	}
	o.t = 0 // restart bias-correction schedule
}

// SnapshotParameters copies every parameter's current value into a flat map
// keyed by tensor pointer. Use this to take a "best checkpoint" snapshot.
func (o *Adam) SnapshotParameters() map[*Tensor][]float32 {
	snap := make(map[*Tensor][]float32, len(o.parameters))
	for _, p := range o.parameters {
		clone := make([]float32, len(p.Data))
		copy(clone, p.Data)
		snap[p] = clone
	}
	return snap
}

// RestoreParameters writes a snapshotted weight copy back into the live
// parameter tensors and resets all Adam moments so training restarts cleanly.
func (o *Adam) RestoreParameters(snap map[*Tensor][]float32) {
	for _, p := range o.parameters {
		if saved, ok := snap[p]; ok && len(saved) == len(p.Data) {
			copy(p.Data, saved)
		}
	}
	o.ResetAllMoments()
}

// SaveState serializes the Adam optimizer state (step counter + m/v moments)
// to a gzip-compressed gob file sequentially to avoid massive OOM spikes.
func (o *Adam) SaveState(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("optimizer SaveState: create file: %w", err)
	}
	defer f.Close()
	gz := gzip.NewWriter(f)
	defer gz.Close()
	enc := gob.NewEncoder(gz)

	if err := enc.Encode(o.t); err != nil {
		return err
	}
	if err := enc.Encode(len(o.parameters)); err != nil {
		return err
	}
	for _, p := range o.parameters {
		if m, ok := o.m[p]; ok {
			if err := enc.Encode(m.Data); err != nil {
				return err
			}
		} else {
			if err := enc.Encode([]float32(nil)); err != nil {
				return err
			}
		}
		if v, ok := o.v[p]; ok {
			if err := enc.Encode(v.Data); err != nil {
				return err
			}
		} else {
			if err := enc.Encode([]float32(nil)); err != nil {
				return err
			}
		}
	}
	return nil
}

// LoadState restores Adam optimizer state sequentially.
func (o *Adam) LoadState(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("optimizer LoadState: open: %w", err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("optimizer LoadState: gzip: %w", err)
	}
	defer gz.Close()
	dec := gob.NewDecoder(gz)

	var t, count int
	if err := dec.Decode(&t); err != nil {
		return err
	}
	if err := dec.Decode(&count); err != nil {
		return err
	}
	o.t = t

	for i, p := range o.parameters {
		if i >= count {
			break
		}
		var mData, vData []float32
		if err := dec.Decode(&mData); err != nil {
			return err
		}
		if err := dec.Decode(&vData); err != nil {
			return err
		}

		if len(mData) == len(p.Data) {
			o.m[p] = NewTensor(p.Shape, mData, false)
		}
		if len(vData) == len(p.Data) {
			o.v[p] = NewTensor(p.Shape, vData, false)
		}
	}
	return nil
}

// Step performs a single optimization step, now with Timid-Aware LR boosting.
func (o *Adam) Step() {
	o.t++

	// Pre-allocate M/V tensors for all parameters with gradients (single-threaded)
	// to avoid concurrent map read/write races in the worker goroutines.
	for _, p := range o.parameters {
		if p.Grad != nil {
			if _, ok := o.m[p]; !ok {
				o.m[p] = NewTensor(p.Shape, make([]float32, len(p.Data)), false)
				o.v[p] = NewTensor(p.Shape, make([]float32, len(p.Data)), false)
			}
		}
	}

	var wg sync.WaitGroup
	// Use limited concurrency to avoid context switching overhead
	numWorkers := runtime.NumCPU()
	if numWorkers > 16 {
		numWorkers = 16
	}

	paramChan := make(chan *Tensor, len(o.parameters))
	for _, p := range o.parameters {
		if p.Grad != nil {
			paramChan <- p
		}
	}
	close(paramChan)

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range paramChan {
				m := o.m[p].Data
				v := o.v[p].Data
				grad := p.Grad.Data
				param := p.Data

				lr := o.learningRate
				if p.IsRouter {
					lr = o.RouterLR
				}

				if p.TimidMask() != nil {
					b1t := float32(1.0 - math.Pow(float64(o.beta1), float64(o.t)))
					b2t := float32(1.0 - math.Pow(float64(o.beta2), float64(o.t)))

					mask := p.TimidMask()
					for i := range param {
						m[i] = o.beta1*m[i] + (1-o.beta1)*grad[i]
						v[i] = o.beta2*v[i] + (1-o.beta2)*grad[i]*grad[i]

						effectiveLR := lr
						if mask[i] {
							effectiveLR *= 3.0
						}

						mHat := m[i] / b1t
						vHat := v[i] / b2t
						denom := float32(math.Sqrt(float64(vHat))) + o.epsilon

						update := (mHat / denom) + (o.Lambda * param[i])
						param[i] -= effectiveLR * update
					}
				} else {
					AdamWUpdate(param, grad, m, v, lr, o.beta1, o.beta2, o.epsilon, o.Lambda, o.t)
				}

				// 🛡️ Continuous Clamping for Routers to prevent "Expert Obsession"
				if p.IsRouter {
					for i := range param {
						if param[i] > 2.5 {
							param[i] = 2.5
						}
						if param[i] < -2.5 {
							param[i] = -2.5
						}
					}
				}
			}
		}()
	}
	wg.Wait()
}

// ZeroGrad resets the gradients of all parameters in parallel.
func (o *Adam) ZeroGrad() {
	var wg sync.WaitGroup
	numWorkers := runtime.NumCPU()
	if numWorkers > 16 {
		numWorkers = 16
	}

	workCh := make(chan *Tensor, len(o.parameters))
	for _, p := range o.parameters {
		workCh <- p
	}
	close(workCh)

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for p := range workCh {
				p.ZeroGrad()
			}
		}()
	}
	wg.Wait()
}

func (o *Adam) SetLearningRate(lr float32) {
	o.learningRate = lr
	o.RouterLR = lr * 15.0 // Maintain higher router LR for exploration
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
	OriginalLR float32 // To store where we should be post-recovery
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

// ResetAllMoments delegates to the base optimizer.
func (o *CoolingOptimizer) ResetAllMoments() {
	o.Base.ResetAllMoments()
}

// SnapshotParameters delegates to the underlying Adam if available.
func (o *CoolingOptimizer) SnapshotParameters() map[*Tensor][]float32 {
	if adam, ok := o.Base.(*Adam); ok {
		return adam.SnapshotParameters()
	}
	return nil
}

// RestoreParameters delegates to the underlying Adam if available.
func (o *CoolingOptimizer) RestoreParameters(snap map[*Tensor][]float32) {
	if adam, ok := o.Base.(*Adam); ok {
		adam.RestoreParameters(snap)
	}
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

// SaveState saves the underlying Adam optimizer state to path.
func (o *CoolingOptimizer) SaveState(path string) error {
	if adam, ok := o.Base.(*Adam); ok {
		return adam.SaveState(path)
	}
	return nil // non-Adam base — no-op
}

// LoadState restores optimizer state from path into the underlying Adam.
func (o *CoolingOptimizer) LoadState(path string) error {
	if adam, ok := o.Base.(*Adam); ok {
		return adam.LoadState(path)
	}
	return nil // non-Adam base — no-op
}
