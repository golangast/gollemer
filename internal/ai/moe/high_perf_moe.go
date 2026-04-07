package moe

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"unsafe"
)

// --- [Memory & SIMD Helpers] ---

// AlignedFloat32Slice mimics page-aligned memory for better cache performance.
func AlignedFloat32Slice(n int) []float32 {
	const align = 64 / 4 // 64 bytes = 16 float32s
	b := make([]float32, n+align)
	// Slicing to align (rough heuristic for pure-Go)
	misalignment := uintptr(unsafe.Pointer(&b[0])) % 64
	offset := 0
	if misalignment != 0 {
		offset = int((64 - misalignment) / 4)
	}
	return b[offset : offset+n]
}

// SparseDotProduct returns the sum of a[i] * b[i] with 4-way loop unrolling for auto-vectorization.
func SparseDotProduct(a, b []float32) float32 {
	if len(a) != len(b) {
		panic("slices must be equal length")
	}

	var sum0, sum1, sum2, sum3 float32
	n := len(a)

	i := 0
	for ; i <= n-4; i += 4 {
		sum0 += a[i] * b[i]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
	}

	finalSum := sum0 + sum1 + sum2 + sum3
	for ; i < n; i++ {
		finalSum += a[i] * b[i]
	}

	return finalSum
}

// AtomicUpdate subtracts the gradient (delta) from a weight safely using CAS.
func AtomicUpdate(addr *float32, delta float32) {
	for {
		oldVal := *addr
		newVal := oldVal - delta
		oldBits := math.Float32bits(oldVal)
		newBits := math.Float32bits(newVal)

		if atomic.CompareAndSwapUint32((*uint32)(unsafe.Pointer(addr)), oldBits, newBits) {
			break
		}
	}
}

// --- [SparseExpert Struct] ---

// SparseExpert represents a specialized neural layer in Gollemer.
type SparseExpert struct {
	ID      int
	Weights []float32
	Bias    []float32
}

// NewSparseExpert initializes a sparse expert with aligned memory.
func NewSparseExpert(id, inputDim, outputDim int) *SparseExpert {
	return &SparseExpert{
		ID:      id,
		Weights: AlignedFloat32Slice(inputDim * outputDim),
		Bias:    AlignedFloat32Slice(outputDim),
	}
}

// Compute performs the expert forward pass using SIMD-ready logic.
func (e *SparseExpert) Compute(input []float32) []float32 {
	outputSize := len(e.Bias)
	inputSize := len(input)
	res := make([]float32, outputSize)

	for i := 0; i < outputSize; i++ {
		offset := i * inputSize
		res[i] = SparseDotProduct(input, e.Weights[offset:offset+inputSize]) + e.Bias[i]
	}
	return res
}

// UpdateWeights performs a safe weight update using gradients.
func (e *SparseExpert) UpdateWeights(input []float32, errors []float32, lr float32) {
	inputSize := len(input)
	outputSize := len(errors)

	for i := 0; i < outputSize; i++ {
		err := errors[i]
		offset := i * inputSize
		for j := 0; j < inputSize; j++ {
			grad := input[j] * err
			AtomicUpdate(&e.Weights[offset+j], grad*lr)
		}
		AtomicUpdate(&e.Bias[i], err*lr)
	}
}

// SaveWeights persists the expert weights to disk in binary format.
func (e *SparseExpert) SaveWeights(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return binary.Write(f, binary.LittleEndian, e.Weights)
}

// LoadWeights loads binary weights into the expert's memory.
func (e *SparseExpert) LoadWeights(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	return binary.Read(f, binary.LittleEndian, e.Weights)
}

// --- [SparseGater decidion] ---

// SparseGater decides which experts to route tokens to.
type SparseGater struct {
	Weights      []float32 // [NumExperts * InputDim]
	NoiseWeights []float32 // [NumExperts * InputDim]
	NumExperts   int
	K            int
}

// NewSparseGater initializes a Gater with randomized weights.
func NewSparseGater(inputDim, numExperts, k int) *SparseGater {
	g := &SparseGater{
		Weights:      AlignedFloat32Slice(numExperts * inputDim),
		NoiseWeights: AlignedFloat32Slice(numExperts * inputDim),
		NumExperts:   numExperts,
		K:            k,
	}
	// Xavier-style initialization
	std := float32(math.Sqrt(2.0 / float64(inputDim)))
	for i := range g.Weights {
		g.Weights[i] = float32(rand.NormFloat64()) * std
		g.NoiseWeights[i] = float32(rand.NormFloat64()) * (std * 0.1)
	}
	return g
}

// Forward performs Noisy Top-K routing.
func (g *SparseGater) Forward(input []float32) ([]int, []float32) {
	inputDim := len(input)
	logits := make([]float32, g.NumExperts)

	for i := 0; i < g.NumExperts; i++ {
		// 1. Prediction + Noise Scaling
		cleanLogit := SparseDotProduct(input, g.Weights[i*inputDim:(i+1)*inputDim])
		noiseLogit := SparseDotProduct(input, g.NoiseWeights[i*inputDim:(i+1)*inputDim])

		// Softplus ensures positive noise variance
		noiseScale := float32(math.Log(1.0 + math.Exp(float64(noiseLogit))))
		epsilon := float32(rand.NormFloat64())

		logits[i] = cleanLogit + epsilon*noiseScale
	}

	// 2. Select Top-K
	indices := make([]int, g.NumExperts)
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool {
		return logits[indices[i]] > logits[indices[j]]
	})

	topKIndices := indices[:g.K]
	topKLogits := make([]float32, g.K)
	for i, idx := range topKIndices {
		topKLogits[i] = logits[idx]
	}

	// 3. Softmax across Top-K
	scores := g.softmax(topKLogits)
	return topKIndices, scores
}

func (g *SparseGater) softmax(logits []float32) []float32 {
	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := float32(0)
	scores := make([]float32, len(logits))
	for i, v := range logits {
		scores[i] = float32(math.Exp(float64(v - maxVal)))
		sum += scores[i]
	}
	for i := range scores {
		scores[i] /= (sum + 1e-12)
	}
	return scores
}

// UpdateGaterWeights updates routing weights based on importance signal.
func (g *SparseGater) UpdateGaterWeights(input []float32, gradOutput []float32, lr float32) {
	inputDim := len(input)
	for i := 0; i < g.NumExperts; i++ {
		for j := 0; j < inputDim; j++ {
			delta := input[j] * gradOutput[i] * lr
			AtomicUpdate(&g.Weights[i*inputDim+j], delta)
		}
	}
}

// --- [SparseModel Orchestrator] ---

// SparseModel wraps the Gater and Experts into a single MoE interface.
type SparseModel struct {
	Gater      *SparseGater
	Experts    []*SparseExpert
	AllWeights []float32 // For flat binary save/mmap
}

// Predict performs one-shot inference using parallel experts.
func (m *SparseModel) Predict(input []float32) ([]float32, []int) {
	indices, scores := m.Gater.Forward(input)

	var wg sync.WaitGroup
	results := make([][]float32, len(indices))

	for i, idx := range indices {
		wg.Add(1)
		go func(slot, expertIdx int) {
			defer wg.Done()
			results[slot] = m.Experts[expertIdx].Compute(input)
		}(i, idx)
	}
	wg.Wait()

	// Accumulate weighted expert outputs (Gather)
	outputSize := len(results[0])
	finalOutput := make([]float32, outputSize)
	for i := range indices {
		s := scores[i]
		for j := 0; j < outputSize; j++ {
			finalOutput[j] += results[i][j] * s
		}
	}
	return finalOutput, indices
}

// SaveCheckpoint saves the model state as binary data.
func (m *SparseModel) SaveCheckpoint(epoch int, loss float32) error {
	path := fmt.Sprintf("data/models/checkpoints/gollemer_E%d_L%.4f.bin", epoch, loss)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	binary.Write(f, binary.LittleEndian, int32(epoch))
	binary.Write(f, binary.LittleEndian, loss)

	// Collect all weights for saving
	// (Simplified for now: write Gater weights then all expert weights)
	binary.Write(f, binary.LittleEndian, m.Gater.Weights)
	binary.Write(f, binary.LittleEndian, m.Gater.NoiseWeights)
	for _, e := range m.Experts {
		binary.Write(f, binary.LittleEndian, e.Weights)
		binary.Write(f, binary.LittleEndian, e.Bias)
	}
	return nil
}

// LoadSparseModel loads model state from binary data.
func LoadSparseModel(filename string, inputDim, numExperts, k int) (*SparseModel, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var epoch int32
	var loss float32
	binary.Read(f, binary.LittleEndian, &epoch)
	binary.Read(f, binary.LittleEndian, &loss)

	gater := NewSparseGater(inputDim, numExperts, k)
	binary.Read(f, binary.LittleEndian, gater.Weights)
	binary.Read(f, binary.LittleEndian, gater.NoiseWeights)

	experts := make([]*SparseExpert, numExperts)
	for i := 0; i < numExperts; i++ {
		expert := NewSparseExpert(i, inputDim, inputDim) // assuming square for demo
		binary.Read(f, binary.LittleEndian, expert.Weights)
		binary.Read(f, binary.LittleEndian, expert.Bias)
		experts[i] = expert
	}

	return &SparseModel{Gater: gater, Experts: experts}, nil
}

// --- [Diagnostics & Trace] ---

// Trace shows which expert handled which token with terminal colors.
func (m *SparseModel) Trace(sentence string, tokenizer interface {
	Tokenize(string) []string
	Embed(string) []float32
}) {
	tokens := tokenizer.Tokenize(sentence)
	colors := []string{"\033[31m", "\033[32m", "\033[33m", "\033[34m", "\033[35m", "\033[36m"}
	reset := "\033[0m"

	fmt.Printf("Trace for: \"%s\"\n", sentence)
	for _, token := range tokens {
		vec := tokenizer.Embed(token)
		indices, _ := m.Gater.Forward(vec)
		expertID := indices[0]
		color := colors[expertID%len(colors)]
		fmt.Printf("%s%s[E%d]%s ", color, token, expertID, reset)
	}
	fmt.Println()
}
