package moe

import (
	"log"
	"math"
	"os"
	"runtime/pprof"
	"sync/atomic"
)

// FastExpert represents a single expert in the MoE layer with optimized memory layout.
type FastExpert struct {
	Weights []float32 // Flattened [InputSize * OutputSize]
	Bias    []float32
}

// FastMoELayer implements a high-performance Mixture of Experts layer.
type FastMoELayer struct {
	NumExperts int
	TopK       int
	Experts    []FastExpert

	// Router weights [InputSize * NumExperts]
	RouterWeights []float32

	// Monitoring / Training State
	AuxLossWeight float32
	ExpertCounts  []int64   // Use sync/atomic to update
	ExpertProbs   []float32 // Sum of probabilities for balancing
}

// NewMoELayerFast creates a new high-performance FastMoELayer.
func NewMoELayerFast(numExperts, inputSize, outputSize, topK int) *FastMoELayer {
	experts := make([]FastExpert, numExperts)
	for i := 0; i < numExperts; i++ {
		experts[i] = FastExpert{
			Weights: make([]float32, inputSize*outputSize),
			Bias:    make([]float32, outputSize),
		}
	}
	return &FastMoELayer{
		NumExperts:    numExperts,
		TopK:          topK,
		Experts:       experts,
		RouterWeights: make([]float32, inputSize*numExperts),
		ExpertCounts:  make([]int64, numExperts),
		ExpertProbs:   make([]float32, numExperts),
	}
}

// Forward performs the Scatter-Gather forward pass.
func (m *FastMoELayer) Forward(input []float32) []float32 {
	// 1. Get Router Logits (Linear layer)
	logits := m.computeRouterLogits(input)

	// 2. Fast Top-K selection (Top-2)
	idx1, idx2, val1, val2 := getTop2(logits)

	// 3. Normalize Top-2 scores (Softmax)
	w1, w2 := normalizeTwo(val1, val2)

	// 4. Update Balancing Metrics (Atomic for thread safety)
	atomic.AddInt64(&m.ExpertCounts[idx1], 1)
	atomic.AddInt64(&m.ExpertCounts[idx2], 1)

	// 5. Expert Computation (Unrolled Dot Products)
	out1 := m.Experts[idx1].Compute(input)
	out2 := m.Experts[idx2].Compute(input)

	// 6. Weighted Sum (Gather)
	finalOut := make([]float32, len(out1))
	for i := 0; i < len(out1); i++ {
		finalOut[i] = (out1[i] * w1) + (out2[i] * w2)
	}

	return finalOut
}

// computeRouterLogits computes the logits for the routing decision.
func (m *FastMoELayer) computeRouterLogits(input []float32) []float32 {
	inputSize := len(input)
	logits := make([]float32, m.NumExperts)
	for i := 0; i < m.NumExperts; i++ {
		dot := float32(0)
		offset := i * inputSize
		for j := 0; j < inputSize; j++ {
			dot += m.RouterWeights[offset+j] * input[j]
		}
		logits[i] = dot
	}
	return logits
}

// getTop2 finds the indices and values of the top 2 elements in a slice.
func getTop2(logits []float32) (int, int, float32, float32) {
	idx1, idx2 := 0, 1
	val1, val2 := logits[0], logits[1]

	if val2 > val1 {
		idx1, idx2 = 1, 0
		val1, val2 = val2, val1
	}

	for i := 2; i < len(logits); i++ {
		v := logits[i]
		if v > val1 {
			val2 = val1
			idx2 = idx1
			val1 = v
			idx1 = i
		} else if v > val2 {
			val2 = v
			idx2 = i
		}
	}
	return idx1, idx2, val1, val2
}

// normalizeTwo performs a softmax normalization for two values.
func normalizeTwo(v1, v2 float32) (float32, float32) {
	maxV := v1
	if v2 > maxV {
		maxV = v2
	}
	ev1 := float32(math.Exp(float64(v1 - maxV)))
	ev2 := float32(math.Exp(float64(v2 - maxV)))
	sum := ev1 + ev2
	return ev1 / sum, ev2 / sum
}

// Compute performs the expert's computation with unrolled dot products.
func (e *FastExpert) Compute(input []float32) []float32 {
	outputSize := len(e.Bias)
	inputSize := len(input)
	res := make([]float32, outputSize)

	// Hint to the compiler that we know the length to skip bounds checks
	_ = input[len(input)-1]
	_ = e.Weights[len(e.Weights)-1]

	for i := 0; i < outputSize; i++ {
		dot := float32(0)
		offset := i * inputSize
		row := e.Weights[offset : offset+inputSize]

		// Unrolled loop for 8-element processing
		j := 0
		for ; j <= inputSize-8; j += 8 {
			dot += row[j]*input[j] + row[j+1]*input[j+1] +
				row[j+2]*input[j+2] + row[j+3]*input[j+3] +
				row[j+4]*input[j+4] + row[j+5]*input[j+5] +
				row[j+6]*input[j+6] + row[j+7]*input[j+7]
		}
		for ; j < inputSize; j++ {
			dot += row[j] * input[j]
		}
		res[i] = dot + e.Bias[i]
	}
	return res
}

// CalculateAuxLoss computes the load balancing loss.
func (m *FastMoELayer) CalculateAuxLoss(batchSize int, counts []int64, probs []float32) float32 {
	if batchSize == 0 {
		return 0
	}

	var cvSum float32
	targetCount := float32(batchSize*m.TopK) / float32(m.NumExperts)

	for i := 0; i < m.NumExperts; i++ {
		diff := float32(counts[i]) - targetCount
		cvSum += diff * diff * probs[i]
	}

	return cvSum * m.AuxLossWeight
}

// GetExpertVariance calculates the variance of expert utilization.
func (m *FastMoELayer) GetExpertVariance() float64 {
	var sum, sumSq float64
	for _, count := range m.ExpertCounts {
		c := float64(count)
		sum += c
		sumSq += c * c
	}
	mean := sum / float64(m.NumExperts)
	return (sumSq / float64(m.NumExperts)) - (mean * mean)
}

// ProfileMoE runs the MoE forward pass multiple times with CPU profiling.
func ProfileMoE(m *FastMoELayer, input [][]float32) {
	f, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal(err)
	}
	defer pprof.StopCPUProfile()

	for i := 0; i < 1000; i++ {
		_ = m.Forward(input[i%len(input)])
	}
}

// AdamRouter optimizes router weights using the Adam optimizer.
type AdamRouter struct {
	M []float32 // First moment vector
	V []float32 // Second moment vector
	T int       // Time step (iteration)

	Beta1 float32
	Beta2 float32
	Eps   float32
	LR    float32
}

// NewAdamRouter initializes a new AdamRouter.
func NewAdamRouter(size int, lr float32) *AdamRouter {
	return &AdamRouter{
		M:     make([]float32, size),
		V:     make([]float32, size),
		Beta1: 0.9,
		Beta2: 0.999,
		Eps:   1e-8,
		LR:    lr,
	}
}

// Update updates the weights using the Adam optimizer with auxiliary gradients.
func (a *AdamRouter) Update(weights []float32, grads []float32) {
	a.T++

	// Bias correction terms
	b1t := 1.0 - float32(math.Pow(float64(a.Beta1), float64(a.T)))
	b2t := 1.0 - float32(math.Pow(float64(a.Beta2), float64(a.T)))

	// Unrolled for performance
	i := 0
	for ; i <= len(weights)-8; i += 8 {
		for k := 0; k < 8; k++ {
			idx := i + k
			g := grads[idx]

			// m = beta1 * m + (1 - beta1) * g
			a.M[idx] = a.Beta1*a.M[idx] + (1-a.Beta1)*g
			// v = beta2 * v + (1 - beta2) * g^2
			a.V[idx] = a.Beta2*a.V[idx] + (1-a.Beta2)*g*g

			mHat := a.M[idx] / b1t
			vHat := a.V[idx] / b2t

			weights[idx] -= a.LR * mHat / (float32(math.Sqrt(float64(vHat))) + a.Eps)
		}
	}

	// Handle remainder
	for ; i < len(weights); i++ {
		g := grads[i]
		a.M[i] = a.Beta1*a.M[i] + (1-a.Beta1)*g
		a.V[i] = a.Beta2*a.V[i] + (1-a.Beta2)*g*g

		mHat := a.M[i] / b1t
		vHat := a.V[i] / b2t

		weights[i] -= a.LR * mHat / (float32(math.Sqrt(float64(vHat))) + a.Eps)
	}
}
