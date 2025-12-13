package nn

import (
	"math"

	. "github.com/golangast/gollemer/neural/tensor"
)

// Optimizer interface defines the contract for optimizers.
type Optimizer interface {
	Step()
	ZeroGrad()
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
	}
}

// Step performs a single optimization step.
func (o *Adam) Step() {
	o.t++
	biasCorrection1 := 1 - math.Pow(o.beta1, float64(o.t))
	biasCorrection2 := 1 - math.Pow(o.beta2, float64(o.t))

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
		unrollFactor := 8
		i := 0

		// Unrolled loop
		for ; i <= len(param)-unrollFactor; i += unrollFactor {
			// Unroll 1
			g1 := grad[i]
			if g1 > o.clipValue {
				g1 = o.clipValue
			} else if g1 < -o.clipValue {
				g1 = -o.clipValue
			}
			m[i] = o.beta1*m[i] + (1-o.beta1)*g1
			v[i] = o.beta2*v[i] + (1-o.beta2)*(g1*g1)
			mHat1 := m[i] / biasCorrection1
			vHat1 := v[i] / biasCorrection2
			param[i] -= o.learningRate * mHat1 / (math.Sqrt(vHat1) + o.epsilon)

			// Unroll 2
			g2 := grad[i+1]
			if g2 > o.clipValue {
				g2 = o.clipValue
			} else if g2 < -o.clipValue {
				g2 = -o.clipValue
			}
			m[i+1] = o.beta1*m[i+1] + (1-o.beta1)*g2
			v[i+1] = o.beta2*v[i+1] + (1-o.beta2)*(g2*g2)
			mHat2 := m[i+1] / biasCorrection1
			vHat2 := v[i+1] / biasCorrection2
			param[i+1] -= o.learningRate * mHat2 / (math.Sqrt(vHat2) + o.epsilon)

			// Unroll 3
			g3 := grad[i+2]
			if g3 > o.clipValue {
				g3 = o.clipValue
			} else if g3 < -o.clipValue {
				g3 = -o.clipValue
			}
			m[i+2] = o.beta1*m[i+2] + (1-o.beta1)*g3
			v[i+2] = o.beta2*v[i+2] + (1-o.beta2)*(g3*g3)
			mHat3 := m[i+2] / biasCorrection1
			vHat3 := v[i+2] / biasCorrection2
			param[i+2] -= o.learningRate * mHat3 / (math.Sqrt(vHat3) + o.epsilon)

			// Unroll 4
			g4 := grad[i+3]
			if g4 > o.clipValue {
				g4 = o.clipValue
			} else if g4 < -o.clipValue {
				g4 = -o.clipValue
			}
			m[i+3] = o.beta1*m[i+3] + (1-o.beta1)*g4
			v[i+3] = o.beta2*v[i+3] + (1-o.beta2)*(g4*g4)
			mHat4 := m[i+3] / biasCorrection1
			vHat4 := v[i+3] / biasCorrection2
			param[i+3] -= o.learningRate * mHat4 / (math.Sqrt(vHat4) + o.epsilon)

			// Unroll 5
			g5 := grad[i+4]
			if g5 > o.clipValue {
				g5 = o.clipValue
			} else if g5 < -o.clipValue {
				g5 = -o.clipValue
			}
			m[i+4] = o.beta1*m[i+4] + (1-o.beta1)*g5
			v[i+4] = o.beta2*v[i+4] + (1-o.beta2)*(g5*g5)
			mHat5 := m[i+4] / biasCorrection1
			vHat5 := v[i+4] / biasCorrection2
			param[i+4] -= o.learningRate * mHat5 / (math.Sqrt(vHat5) + o.epsilon)

			// Unroll 6
			g6 := grad[i+5]
			if g6 > o.clipValue {
				g6 = o.clipValue
			} else if g6 < -o.clipValue {
				g6 = -o.clipValue
			}
			m[i+5] = o.beta1*m[i+5] + (1-o.beta1)*g6
			v[i+5] = o.beta2*v[i+5] + (1-o.beta2)*(g6*g6)
			mHat6 := m[i+5] / biasCorrection1
			vHat6 := v[i+5] / biasCorrection2
			param[i+5] -= o.learningRate * mHat6 / (math.Sqrt(vHat6) + o.epsilon)

			// Unroll 7
			g7 := grad[i+6]
			if g7 > o.clipValue {
				g7 = o.clipValue
			} else if g7 < -o.clipValue {
				g7 = -o.clipValue
			}
			m[i+6] = o.beta1*m[i+6] + (1-o.beta1)*g7
			v[i+6] = o.beta2*v[i+6] + (1-o.beta2)*(g7*g7)
			mHat7 := m[i+6] / biasCorrection1
			vHat7 := v[i+6] / biasCorrection2
			param[i+6] -= o.learningRate * mHat7 / (math.Sqrt(vHat7) + o.epsilon)

			// Unroll 8
			g8 := grad[i+7]
			if g8 > o.clipValue {
				g8 = o.clipValue
			} else if g8 < -o.clipValue {
				g8 = -o.clipValue
			}
			m[i+7] = o.beta1*m[i+7] + (1-o.beta1)*g8
			v[i+7] = o.beta2*v[i+7] + (1-o.beta2)*(g8*g8)
			mHat8 := m[i+7] / biasCorrection1
			vHat8 := v[i+7] / biasCorrection2
			param[i+7] -= o.learningRate * mHat8 / (math.Sqrt(vHat8) + o.epsilon)
		}

		// Handle remaining elements
		for ; i < len(param); i++ {
			gradI := grad[i]
			// Clip gradients
			if gradI > o.clipValue {
				gradI = o.clipValue
			} else if gradI < -o.clipValue {
				gradI = -o.clipValue
			}

			// Update biased first and second moment estimates
			m[i] = o.beta1*m[i] + (1-o.beta1)*gradI
			v[i] = o.beta2*v[i] + (1-o.beta2)*math.Pow(gradI, 2)

			// Compute bias-corrected estimates and update parameters
			mHat := m[i] / biasCorrection1
			vHat := v[i] / biasCorrection2
			param[i] -= o.learningRate * mHat / (math.Sqrt(vHat) + o.epsilon)
		}
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
