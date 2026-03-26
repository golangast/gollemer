package model

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// KVCache stores keys and values for attention to optimize inference.
type KVCache struct {
	// [Layer][SequenceLength][KVHeads][HeadDim]
	Keys   [][][][]float32
	Values [][][][]float32

	NumQHeads  int
	NumKVHeads int
	HeadDim    int
	MaxLen     int
	Cursor     int
}

// NewKVCache initializes a new KV cache with pre-allocated memory.
func NewKVCache(layers, qHeads, kvHeads, headDim, maxLen int) *KVCache {
	keys := make([][][][]float32, layers)
	values := make([][][][]float32, layers)

	for i := 0; i < layers; i++ {
		keys[i] = make([][][]float32, maxLen)
		values[i] = make([][][]float32, maxLen)
		for j := 0; j < maxLen; j++ {
			keys[i][j] = make([][]float32, kvHeads)
			values[i][j] = make([][]float32, kvHeads)
			for k := 0; k < kvHeads; k++ {
				keys[i][j][k] = make([]float32, headDim)
				values[i][j][k] = make([]float32, headDim)
			}
		}
	}

	return &KVCache{
		Keys:       keys,
		Values:     values,
		NumQHeads:  qHeads,
		NumKVHeads: kvHeads,
		HeadDim:    headDim,
		MaxLen:     maxLen,
	}
}

// Reset clears the cache state.
func (c *KVCache) Reset() {
	c.Cursor = 0
	for l := range c.Keys {
		for t := range c.Keys[l] {
			for h := range c.Keys[l][t] {
				for i := range c.Keys[l][t][h] {
					c.Keys[l][t][h][i] = 0
					c.Values[l][t][h][i] = 0
				}
			}
		}
	}
}

// MoEModel represents the next-gen architecture for Gollemer.
type MoEModel struct {
	Tokenizer *tokenizer.Tokenizer
	Cache     *KVCache
	Config    MoEConfig
}

type MoEConfig struct {
	MaxLen     int
	HiddenSize int
	Layers     int
}

// NewMoEModel creates a new instance of the model.
func NewMoEModel(t *tokenizer.Tokenizer, config MoEConfig) *MoEModel {
	// Example GQA settings
	qHeads := 8
	kvHeads := 2
	headDim := config.HiddenSize / qHeads

	return &MoEModel{
		Tokenizer: t,
		Cache:     NewKVCache(config.Layers, qHeads, kvHeads, headDim, config.MaxLen),
		Config:    config,
	}
}

// Forward is a placeholder for the actual forward pass through MoE layers.
func (m *MoEModel) Forward(tokens []int) []float32 {
	// In a real implementation, this would pass through embeddings,
	// multiple MoE layers with GQA and RoPE, and finally a projection layer.
	// For now, we return dummy logits of the vocabulary size.
	vocabSize := m.Tokenizer.Vocabulary.Size()
	logits := make([]float32, vocabSize)
	for i := range logits {
		logits[i] = rand.Float32() // Dummy predictions
	}
	return logits
}

// SoftmaxSIMD normalizes the logits using a SIMD-ready approach.
func SoftmaxSIMD(logits []float32) {
	var max float32 = -math.MaxFloat32
	for _, v := range logits {
		if v > max {
			max = v
		}
	}

	var sum float64
	for i := 0; i < len(logits); i++ {
		logits[i] = float32(math.Exp(float64(logits[i] - max)))
		sum += float64(logits[i])
	}

	invSum := float32(1.0 / sum)
	for i := 0; i < len(logits); i++ {
		logits[i] *= invSum
	}
}

// TokenProb helps us sort tokens by their probability while keeping their original ID.
type TokenProb struct {
	ID   int
	Prob float32
}

// SampleTopP performs Nucleus (Top-P) sampling.
func (m *MoEModel) SampleTopP(probs []float32, p float32) int {
	tokenProbs := make([]TokenProb, len(probs))
	for i, prob := range probs {
		tokenProbs[i] = TokenProb{ID: i, Prob: prob}
	}

	sort.Slice(tokenProbs, func(i, j int) bool {
		return tokenProbs[i].Prob > tokenProbs[j].Prob
	})

	var cumulativeProb float32
	cutoffIndex := len(tokenProbs) - 1
	for i, tp := range tokenProbs {
		cumulativeProb += tp.Prob
		if cumulativeProb >= p {
			cutoffIndex = i
			break
		}
	}

	nucleus := tokenProbs[:cutoffIndex+1]
	var nucleusSum float32
	for _, tp := range nucleus {
		nucleusSum += tp.Prob
	}

	r := rand.Float32() * nucleusSum
	var currentSum float32
	for _, tp := range nucleus {
		currentSum += tp.Prob
		if r <= currentSum {
			return tp.ID
		}
	}

	return nucleus[0].ID
}

// GenerateWithStats handles the autoregressive generation with real-time TPS tracking.
func (m *MoEModel) GenerateWithStats(prompt string) {
	tokens, _ := m.Tokenizer.Encode(prompt)
	start := time.Now()
	generatedCount := 0

	fmt.Printf("/ʕ◡ϖ◡ʔ/ > ")

	for generatedCount < m.Config.MaxLen {
		logits := m.Forward(tokens)
		
		// 1. Softmax
		SoftmaxSIMD(logits)
		
		// 2. Sample
		nextToken := m.SampleTopP(logits, 0.9)

		// 3. Decode and print
		word, _ := m.Tokenizer.Decode([]int{nextToken})
		fmt.Print(word)
		if !strings.HasSuffix(word, " ") {
			fmt.Print(" ")
		}

		tokens = append(tokens, nextToken)
		generatedCount++

		// Calculate Stats
		elapsed := time.Since(start).Seconds()
		tps := float64(generatedCount) / elapsed

		// Update terminal status line
		fmt.Printf("\033[s\033[K\n[ TPS: %.2f | Tokens: %d ]\033[u", tps, generatedCount)

		// Check for EOS (Placeholder ID for demonstration)
		if nextToken == 0 { // Assume 0 is EOS for now
			break
		}
	}
	fmt.Println("\n— Done.")
}

// GQA-related methods as requested

func (m *MoEModel) AttendGQA(qLayer [][]float32, layerIdx int, cache *KVCache) []float32 {
	groupSize := cache.NumQHeads / cache.NumKVHeads
	output := make([]float32, cache.NumQHeads*cache.HeadDim)

	for qIdx := 0; qIdx < cache.NumQHeads; qIdx++ {
		kvIdx := qIdx / groupSize
		qHead := qLayer[qIdx]
		headOutput := m.ComputeScaledDotProduct(qHead, cache.Keys[layerIdx], cache.Values[layerIdx], kvIdx, cache.Cursor)
		copy(output[qIdx*cache.HeadDim:], headOutput)
	}
	return output
}

func (m *MoEModel) ComputeScaledDotProduct(qHead []float32, kCache [][][]float32, vCache [][][]float32, kvIdx int, cursor int) []float32 {
	headDim := len(qHead)
	scale := 1.0 / math.Sqrt(float64(headDim))
	scores := make([]float32, cursor+1)

	for t := 0; t <= cursor; t++ {
		kHead := kCache[t][kvIdx]
		var sum float32
		// Parallelizable over headDim
		for i := 0; i < headDim; i++ {
			sum += qHead[i] * kHead[i]
		}
		scores[t] = sum * float32(scale)
	}

	SoftmaxSIMD(scores)

	output := make([]float32, headDim)
	for t := 0; t <= cursor; t++ {
		vHead := vCache[t][kvIdx]
		s := scores[t]
		for i := 0; i < headDim; i++ {
			output[i] += s * vHead[i]
		}
	}
	return output
}

// ApplyRoPE rotates the query or key vector based on its absolute position.
func ApplyRoPE(vec []float32, position int, headDim int, theta float32) {
	for i := 0; i < headDim/2; i++ {
		freq := float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(headDim)))
		angle := float32(position) * freq
		
		cos := float32(math.Cos(float64(angle)))
		sin := float32(math.Sin(float64(angle)))

		v0 := vec[i]
		v1 := vec[i+headDim/2]
		vec[i] = v0*cos - v1*sin
		vec[i+headDim/2] = v0*sin + v1*cos
	}
}
