package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe/model"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func main() {
	fmt.Println("🚀 Gollemer Sentence Generation Control Tester")
	fmt.Println("============================================")

	// 1. Initialize Vocabulary and Tokenizer
	v := vocab.NewVocabulary()
	
	// Add a diverse set of tokens to make the random generation look like sentences
	vocabularyWords := []string{
		"The", "A", "Gollemer", "intelligence", "network", "system", "MoE", "module",
		"is", "was", "will", "be", "becomes", "acts", "processes", "thinks",
		"fast", "slow", "efficient", "powerful", "modular", "adaptive", "smart",
		"quickly", "smoothly", "internally", "optimally", "automatically",
		"data", "input", "output", "tensor", "gradient", "expert", "router",
		"and", "but", "or", "while", "though", "if", "unless",
		"in", "on", "at", "by", "with", "from", "into", "through",
		"performance", "latency", "throughput", "capacity", "accuracy",
		".", ",", "!", "?",
	}
	
	for _, tok := range vocabularyWords {
		v.AddToken(tok)
	}

	t, _ := tokenizer.NewTokenizer(v)
	
	// 2. Setup Model
	// Note: Forward pass currently uses random logits for demonstration.
	m := model.NewMoEModel(t, model.MoEConfig{
		MaxLen:     50,
		HiddenSize: 128,
		Layers:     4,
	})

	prompts := []string{
		"Gollemer is",
		"The network processes",
		"MoE architecture enables",
	}

	// 3. Define control sets to test
	configs := []struct {
		Name string
		Opts model.GenerationOptions
	}{
		{
			Name: "Balanced (Vibrant)",
			Opts: model.GenerationOptions{MaxLen: 15, Temperature: 1.0, TopP: 0.9, TopK: 40, RouterTemperature: 1.0, Echo: false},
		},
		{
			Name: "Precise (Focused)",
			Opts: model.GenerationOptions{MaxLen: 15, Temperature: 0.7, TopP: 0.8, TopK: 10, RouterTemperature: 0.8, Echo: false},
		},
		{
			Name: "Creative (Diverse)",
			Opts: model.GenerationOptions{MaxLen: 15, Temperature: 1.4, TopP: 0.95, TopK: 100, RouterTemperature: 1.5, Echo: false},
		},
		{
			Name: "Strict (Deterministic)",
			Opts: model.GenerationOptions{MaxLen: 15, Temperature: 0.2, TopP: 0.3, TopK: 5, RouterTemperature: 0.5, Echo: false},
		},
	}

	// 4. Run tests
	for _, prompt := range prompts {
		fmt.Printf("\n\nPROMPT: \"%s\"\n", prompt)
		fmt.Println(strings.Repeat("-", len(prompt)+10))

		for _, cfg := range configs {
			fmt.Printf("\n%-25s | Temp: %.1f | TopP: %.2f | TopK: %d | RouterTemp: %.1f\n", 
				cfg.Name, cfg.Opts.Temperature, cfg.Opts.TopP, cfg.Opts.TopK, cfg.Opts.RouterTemperature)
			
			var totalScore float32
			numTrials := 3
			for i := 1; i <= numTrials; i++ {
				output := m.GenerateCustom(prompt, cfg.Opts)
				score := scoreSentence(output)
				totalScore += score
				fmt.Printf("  Trial %d: %s %s [Score: %.1f]\n", i, prompt, output, score)
			}
			fmt.Printf("  >> Avg Score: %.1f\n", totalScore/float32(numTrials))
		}
	}

	fmt.Println("\n\n✅ Generation tests complete.")
}

// scoreSentence provides a heuristic quality score for a generated sentence.
// Higher is generally better for "best sentences" tests.
func scoreSentence(text string) float32 {
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0
	}

	score := float32(10.0)

	// 1. Penalize very short sentences
	if len(words) < 5 {
		score -= 3.0
	}

	// 2. Penalize excessive repetition
	seen := make(map[string]int)
	for _, w := range words {
		seen[w]++
	}
	repeatCost := 0.0
	for _, count := range seen {
		if count > 1 {
			repeatCost += math.Pow(float64(count), 1.5)
		}
	}
	score -= float32(repeatCost) * 0.5

	// 3. Reward varied vocabulary
	score += float32(len(seen)) * 0.2

	// 4. Reward punctuation presence
	if strings.ContainsAny(text, ".!?") {
		score += 2.0
	}

	// Clamp score
	if score < 0 {
		score = 0
	}
	return score
}
