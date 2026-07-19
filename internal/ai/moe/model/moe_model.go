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
	Tokenizer  *tokenizer.Tokenizer
	Cache      *KVCache
	Config     MoEConfig
	MockBigram map[int][]int
}

type MoEConfig struct {
	MaxLen     int
	HiddenSize int
	Layers     int
}

// GenerationOptions defines parameters for text generation.
type GenerationOptions struct {
	MaxLen            int
	Temperature       float32
	TopP              float32
	TopK              int
	Echo              bool
	StopTokens        []int
	RouterTemperature float32
}

// DefaultGenerationOptions returns the standard configuration for generation.
func DefaultGenerationOptions() GenerationOptions {
	return GenerationOptions{
		MaxLen:            50,
		Temperature:       1.0,
		TopP:              0.9,
		TopK:              40,
		Echo:              true,
		RouterTemperature: 1.0,
	}
}

// NewMoEModel creates a new instance of the model.
func NewMoEModel(t *tokenizer.Tokenizer, config MoEConfig) *MoEModel {
	// Example GQA settings
	qHeads := 8
	kvHeads := 2
	headDim := config.HiddenSize / qHeads

	m := &MoEModel{
		Tokenizer: t,
		Cache:     NewKVCache(config.Layers, qHeads, kvHeads, headDim, config.MaxLen),
		Config:    config,
	}
	m.InitializeMockBigram()
	return m
}

// InitializeMockBigram populates the mock bigram model to make dummy generation more realistic.
func (m *MoEModel) InitializeMockBigram() {
	m.MockBigram = make(map[int][]int)
	vocabWords := m.Tokenizer.Vocabulary.TokenToWord

	for id1, w1 := range vocabWords {
		for id2, w2 := range vocabWords {

			// Simple rules to favor certain transitions
			isGood := false
			lw1, lw2 := strings.ToLower(w1), strings.ToLower(w2)

			// Determiner -> Noun
			if (lw1 == "the" || lw1 == "a") && !(lw2 == "the" || lw2 == "a") {
				isGood = true
			}
			// Noun -> Verb
			if (lw1 == "gollemer" || lw1 == "network" || lw1 == "system") && (lw2 == "is" || lw2 == "processes" || lw2 == "thinks") {
				isGood = true
			}
			// Verb -> Adjective/Adverb/Preposition
			if (lw1 == "is" || lw1 == "becomes") && (lw2 == "fast" || lw2 == "powerful" || lw2 == "smart") {
				isGood = true
			}

			if isGood {
				m.MockBigram[id1] = append(m.MockBigram[id1], id2)
			}
		}
	}
}

func (m *MoEModel) Forward(tokens []int) []float32 {
	vocabSize := m.Tokenizer.Vocabulary.Size()
	logits := make([]float32, vocabSize)

	lastToken := -1
	if len(tokens) > 0 {
		lastToken = tokens[len(tokens)-1]
	}

	// 1. Base randomness
	for i := range logits {
		logits[i] = rand.Float32() * 0.1 // Lowered base randomness
	}

	// 2. Apply bigram bias
	if lastToken != -1 {
		if followers, ok := m.MockBigram[lastToken]; ok {
			for _, id := range followers {
				logits[id] += 2.0 // Strong boost to likely followers
			}
		}
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

// Sample performs sampling using Temperature, Top-K, and Top-P (Nucleus) sampling.
func (m *MoEModel) Sample(logits []float32, opts GenerationOptions) int {
	// 1. Apply Temperature
	if opts.Temperature != 1.0 && opts.Temperature > 0 {
		for i := range logits {
			logits[i] /= opts.Temperature
		}
	}

	// 2. Softmax
	SoftmaxSIMD(logits)

	tokenProbs := make([]TokenProb, len(logits))
	for i, prob := range logits {
		tokenProbs[i] = TokenProb{ID: i, Prob: prob}
	}

	// Sort by probability descending
	sort.Slice(tokenProbs, func(i, j int) bool {
		return tokenProbs[i].Prob > tokenProbs[j].Prob
	})

	// 3. Apply Top-K
	if opts.TopK > 0 && opts.TopK < len(tokenProbs) {
		tokenProbs = tokenProbs[:opts.TopK]
	}

	// 4. Apply Top-P (Nucleus)
	if opts.TopP > 0 && opts.TopP < 1.0 {
		var cumulativeProb float32
		cutoffIndex := len(tokenProbs) - 1
		for i, tp := range tokenProbs {
			cumulativeProb += tp.Prob
			if cumulativeProb >= opts.TopP {
				cutoffIndex = i
				break
			}
		}
		tokenProbs = tokenProbs[:cutoffIndex+1]
	}

	// 5. Sample from the filtered distribution
	var totalProb float32
	for _, tp := range tokenProbs {
		totalProb += tp.Prob
	}

	r := rand.Float32() * totalProb
	var currentSum float32
	for _, tp := range tokenProbs {
		currentSum += tp.Prob
		if r <= currentSum {
			return tp.ID
		}
	}

	return tokenProbs[0].ID
}

// GenerateWithStats handles the autoregressive generation with real-time TPS tracking.
func (m *MoEModel) GenerateWithStats(prompt string) string {
	return m.GenerateCustom(prompt, DefaultGenerationOptions())
}

// GenerateCustom performs generation with specific options.
func (m *MoEModel) GenerateCustom(prompt string, opts GenerationOptions) string {
	tokens, _ := m.Tokenizer.Encode(prompt)
	start := time.Now()
	generatedCount := 0
	var result strings.Builder

	if opts.Echo {
		fmt.Printf("/ʕ◡ϖ◡ʔ/ > ")
	}

	for generatedCount < opts.MaxLen {
		logits := m.Forward(tokens)

		// Sample with options
		nextToken := m.Sample(logits, opts)

		// Decode and append
		word, _ := m.Tokenizer.Decode([]int{nextToken})
		result.WriteString(word)
		if !strings.HasSuffix(word, " ") {
			result.WriteString(" ")
		}

		if opts.Echo {
			fmt.Print(word)
			if !strings.HasSuffix(word, " ") {
				fmt.Print(" ")
			}

			// Calculate Stats
			elapsed := time.Since(start).Seconds()
			tps := 0.0
			if elapsed > 0 {
				tps = float64(generatedCount+1) / elapsed
			}

			// Update terminal status line
			fmt.Printf("\033[s\033[K\n[ TPS: %.2f | Tokens: %d | Temp: %.1f | TopP: %.2f ]\033[u",
				tps, generatedCount+1, opts.Temperature, opts.TopP)
		}

		tokens = append(tokens, nextToken)
		generatedCount++

		// Check for EOS
		isStop := false
		for _, stopID := range opts.StopTokens {
			if nextToken == stopID {
				isStop = true
				break
			}
		}
		if isStop || nextToken == 0 { // 0 is assumed EOS
			break
		}
	}

	if opts.Echo {
		fmt.Println("\n— Done.")
	}
	return result.String()
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
