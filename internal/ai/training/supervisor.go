package training

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
)

// ExpertHyperParams holds routing and learning overrides for an expert.
type ExpertHyperParams struct {
	LearningRate   float64
	DropoutPenalty float64
	LossWeight     float64
}

// AdaptiveSupervisor manages the MoE training feedback loop.
type AdaptiveSupervisor struct {
	ModelDim         int
	CurrentExperts   int
	MaxExperts       int
	ExpertRegistry   map[int]*ExpertHyperParams
	PathFailureCount map[string]int // Tracks path failure sequences (e.g., "E1+E6")
	TrainingDataPath string
	ActiveMode       string
}

// NewAdaptiveSupervisor initializes a new supervisor.
func NewAdaptiveSupervisor(experts, dim int, dataPath string) *AdaptiveSupervisor {
	reg := make(map[int]*ExpertHyperParams)
	for i := 0; i < experts; i++ {
		reg[i] = &ExpertHyperParams{
			LearningRate:   0.001,
			DropoutPenalty: 0.1,
			LossWeight:     1.0,
		}
	}
	return &AdaptiveSupervisor{
		ModelDim:         dim,
		CurrentExperts:   experts,
		MaxExperts:       16, // Bounds to prevent runaway memory expansion
		ExpertRegistry:   reg,
		PathFailureCount: make(map[string]int),
		TrainingDataPath: dataPath,
	}
}

// AssessSpawningPacing dynamically sets the spawning gate based on the intent domain.
func (s *AdaptiveSupervisor) AssessSpawningPacing(intent string) int {
	if s.ActiveMode == "OverfitMode" && intent == "social" {
		return 20
	}
	if intent == "social" || intent == "grammar_baseline" {
		return 15
	}
	return 5
}

// EvaluateGate performs real-time model surgery upon quality gate failures.
func (s *AdaptiveSupervisor) EvaluateGate(path string, score float64, intent, question string) {
	const threshold = 0.05

	if score >= threshold {
		return
	}

	log.Printf("⚠️  AdaptiveSupervisor: Quality Gate Rejected (Score: %.4f < %.4f). Taking control...", score, threshold)

	if s.CurrentExperts >= s.MaxExperts {
		log.Printf("⚠️ [Supervisor] Hard cap reached. Freezing mutations.")
		return
	}

	s.PathFailureCount[path]++
	s.mutateRouting(path)

	// Expand structural capacity if the path repeatedly fails.
	if s.PathFailureCount[path] >= 3 && s.CurrentExperts < s.MaxExperts {
		id := s.CurrentExperts
		s.CurrentExperts++

		log.Printf("🔥 [Structural Expansion] Path %s collapsed. Allocating Expert %d", path, id)

		s.ExpertRegistry[id] = &ExpertHyperParams{
			LearningRate:   0.0005,
			DropoutPenalty: 0.05,
			LossWeight:     1.2,
		}
		s.PathFailureCount[path] = 0
	}

	s.EvolveDataset(question)
}

func (s *AdaptiveSupervisor) mutateRouting(path string) {
	parts := strings.FieldsFunc(path, func(r rune) bool {
		return r == '+' || r == '-' || r == '>' || r == ' '
	})

	for _, p := range parts {
		if !strings.HasPrefix(p, "E") {
			continue
		}
		var id int
		fmt.Sscanf(p, "E%d", &id)

		if params, ok := s.ExpertRegistry[id]; ok {
			log.Printf("🎯 [Mutation] Adjusting hyper-parameters for Expert %d", id)
			params.LossWeight *= 0.85
			params.LearningRate *= 1.10
		}
	}
}

// ResetMetrics clears historical tracking and resets hyper-parameters.
func (s *AdaptiveSupervisor) ResetMetrics() {
	log.Printf("🔄 [Supervisor] Cold-Resetting metrics.")
	s.PathFailureCount = make(map[string]int)
	
	for id, param := range s.ExpertRegistry {
		param.LearningRate = 0.001
		param.DropoutPenalty = 0.1
		param.LossWeight = 1.0
		log.Printf("🔄 [Supervisor] Reset Expert %d to base weights.", id)
	}
}

// EvolveDataset enriches the standard syntax trees for specific failing queries.
func (s *AdaptiveSupervisor) EvolveDataset(question string) {
	if s.TrainingDataPath == "" {
		return
	}

	log.Printf("📝 [Data Evolution] Scanning training assets for target: '%s'", question)

	f, err := os.Open(s.TrainingDataPath)
	if err != nil {
		log.Printf("⚠️  [Data Evolution] Error opening data: %v", err)
		return
	}

	var lines []string
	scan := bufio.NewScanner(f)
	mutations := 0

	for scan.Scan() {
		line := scan.Text()
		
		if !strings.Contains(line, "__ques__ "+question+" __ans__") && !strings.HasPrefix(line, question+",") {
			lines = append(lines, line)
			continue
		}

		repl := replaceTarget(question)
		if strings.Contains(line, "__ques__") {
			line = strings.Replace(line, "__ques__ "+question, "__ques__ "+repl, 1)
		} else {
			line = strings.Replace(line, question+",", repl+",", 1)
		}
		
		lines = append(lines, line)
		mutations++
	}
	f.Close()

	if mutations == 0 {
		return
	}

	// Flush changes.
	out, err := os.Create(s.TrainingDataPath)
	if err != nil {
		log.Printf("⚠️  [Data Evolution] Error writing dataset: %v", err)
		return
	}
	defer out.Close()

	w := bufio.NewWriter(out)
	for _, l := range lines {
		w.WriteString(l + "\n")
	}
	w.Flush()
	
	log.Printf("✅ [Data Evolution] Success. Mutated %d references.", mutations)
}

func replaceTarget(q string) string {
	switch strings.ToLower(q) {
	case "hello", "hi", "hey":
		return "i welcome you with " + q
	case "thanks", "thank you":
		return "i offer you my thanks"
	case "i am sad":
		return "i feel very sad today"
	default:
		return q
	}
}
