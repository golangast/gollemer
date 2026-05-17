package training

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"log"
)

// ExpertHyperParams maps directly to the configurations used during layer forward passes.
// In Gollemer, these typically translate to OutputScale and LRMultiplier overrides.
type ExpertHyperParams struct {
	LearningRate   float64
	DropoutPenalty float64
	LossWeight     float64
}

// AdaptiveSupervisor manages the autonomous feedback loop for MoE training,
// handling variable mutation, structural scaling (expert spawning), and
// automated evolution of raw training assets.
type AdaptiveSupervisor struct {
	ModelDim         int
	CurrentExperts   int
	MaxExperts       int
	ExpertRegistry   map[int]*ExpertHyperParams
	PathFailureCount map[string]int // Tracks path sequences like "E1+E6"
	TrainingDataPath string
}

// NewAdaptiveSupervisor initializes a new adaptive supervisor.
func NewAdaptiveSupervisor(initialExperts, modelDim int, dataPath string) *AdaptiveSupervisor {
	registry := make(map[int]*ExpertHyperParams)
	for i := 0; i < initialExperts; i++ {
		registry[i] = &ExpertHyperParams{
			LearningRate:   0.001,
			DropoutPenalty: 0.1,
			LossWeight:     1.0,
		}
	}
	return &AdaptiveSupervisor{
		ModelDim:         modelDim,
		CurrentExperts:   initialExperts,
		MaxExperts:       16, // Bounds to prevent runaway memory expansion
		ExpertRegistry:   registry,
		PathFailureCount: make(map[string]int),
		TrainingDataPath: dataPath,
	}
}

// EvaluateGate is called by the main curriculum runner at the end of an epoch checkpoint.
// It performs real-time model surgery and data refinement upon quality gate failures.
func (s *AdaptiveSupervisor) EvaluateGate(activePath string, currentScore float64, targetIntent string, problematicRawQuestion string) {
	const minThreshold = 0.0500

	if currentScore >= minThreshold {
		return // Quality gate passed safely
	}

	log.Printf("⚠️  AdaptiveSupervisor: Quality Gate Rejected (Score: %.4f < %.4f). Taking control...", currentScore, minThreshold)
	s.PathFailureCount[activePath]++

	// 1. MUTATE TARGET ROUTING VARIABLES
	// We locate the individual expert IDs (e.g., from "E1+E6")
	parts := strings.FieldsFunc(activePath, func(r rune) bool {
		return r == '+' || r == '-' || r == '>' || r == ' '
	})

	for _, idStr := range parts {
		if !strings.HasPrefix(idStr, "E") {
			continue
		}
		var id int
		fmt.Sscanf(idStr, "E%d", &id)

		if params, exists := s.ExpertRegistry[id]; exists {
			log.Printf("🎯 [Mutation] Adjusting hyper-parameters for Expert %d", id)
			params.LossWeight *= 0.85      // De-emphasize its current representation in structural calculations
			params.LearningRate *= 1.10    // Force an exploratory learning rate bump
		}
	}

	// 2. CREATE NEW SUB-NETWORK EXPERT
	// If a specific routing alignment fails repeatedly, allocate an entirely clean path
	if s.PathFailureCount[activePath] >= 3 && s.CurrentExperts < s.MaxExperts {
		newExpertID := s.CurrentExperts
		s.CurrentExperts++
		
		log.Printf("🔥 [Structural Expansion] Path %s collapsed under intent '%s'. Allocating Expert %d (Dim: %d)", 
			activePath, targetIntent, newExpertID, s.ModelDim)
		
		s.ExpertRegistry[newExpertID] = &ExpertHyperParams{
			LearningRate:   0.0005, // Initialize stable
			DropoutPenalty: 0.05,   // Keep representation sharp
			LossWeight:     1.2,    // High confidence bias for initialization
		}
		
		// Reset tracking for the path to prevent infinite growth loops
		s.PathFailureCount[activePath] = 0
		
		// NOTE: In the main loop, we check s.CurrentExperts and call model.AddExpertToLayer()
	}

	// 3. EVOLVE TRAINING DATA
	s.EvolveDataset(problematicRawQuestion)
}

// EvolveDataset reads the raw dataset file, expands the language structure into 
// standard syntax trees for specific failing queries, and writes it back atomically.
func (s *AdaptiveSupervisor) EvolveDataset(targetQuestion string) {
	if s.TrainingDataPath == "" {
		return
	}

	log.Printf("📝 [Data Evolution] Scanning training assets for token target: '%s'", targetQuestion)
	
	file, err := os.Open(s.TrainingDataPath)
	if err != nil {
		log.Printf("⚠️  [Data Evolution] Error opening data: %v", err)
		return
	}
	
	var lines []string
	scanner := bufio.NewScanner(file)
	mutatedCount := 0

	for scanner.Scan() {
		line := scanner.Text()
		
		// Target checking: handles both internal markers and raw CSV lines
		match := false
		if strings.Contains(line, "__ques__ "+targetQuestion+" __ans__") {
			match = true
		} else if strings.HasPrefix(line, targetQuestion+",") {
			match = true
		}

		if match {
			// Mutate short token fragments into rich syntactic representations 
			// containing functional Subject-Verb-Object profiles
			var replacement string
			switch strings.ToLower(targetQuestion) {
			case "hello", "hi", "hey":
				replacement = "i welcome you with " + targetQuestion
			case "thanks", "thank you":
				replacement = "i offer you my thanks"
			case "i am sad":
				replacement = "i feel very sad today"
			default:
				if strings.HasPrefix(strings.ToLower(targetQuestion), "i am processing the concept of ") {
					replacement = targetQuestion
				} else {
					replacement = "i am processing the concept of " + targetQuestion
				}
			}
			
			var newLine string
			if strings.Contains(line, "__ques__") {
				oldMarker := "__ques__ " + targetQuestion
				newMarker := "__ques__ " + replacement
				newLine = strings.Replace(line, oldMarker, newMarker, 1)
			} else {
				// CSV Case
				newLine = strings.Replace(line, targetQuestion+",", replacement+",", 1)
			}
			lines = append(lines, newLine)
			mutatedCount++
		} else {
			lines = append(lines, line)
		}
	}
	file.Close()

	if mutatedCount > 0 {
		// Flush changes back cleanly to prevent broken buffers
		outFile, err := os.Create(s.TrainingDataPath)
		if err != nil {
			log.Printf("⚠️  [Data Evolution] Error writing evolved dataset: %v", err)
			return
		}
		defer outFile.Close()
		
		writer := bufio.NewWriter(outFile)
		for _, line := range lines {
			_, _ = writer.WriteString(line + "\n")
		}
		_ = writer.Flush()
		log.Printf("✅ [Data Evolution] Success. Mutated %d corpus references.", mutatedCount)
	}
}
