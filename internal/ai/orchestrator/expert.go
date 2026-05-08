package orchestrator

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

type TrainingMetrics struct {
	Epoch             int
	AverageLoss       float32
	SemanticSinksHits int     // How many times experts reset this epoch
	GatingEntropy     float32 // Gating network diversity (lower means collapsing to 1 expert)
	GrammarScore      float32 // From sentence testing
	SimilarityScore   float32 // From sentence testing
	TestResults       []TestProbeResult
	LayerResets       []map[int]int // Per-layer expert reset counts
	LayerUsage        []map[int]int // Per-layer expert utilization
}

type TestProbeResult struct {
	Prompt   string
	Response string
	Path     string // Expert path (e.g. "word(E1+E8) -> word(E4+E12)")
}

type SurgeryPerformer interface {
	PerformSurgery(layerIdx, alphaID, sinkID int)
}

type HyperparameterExpert struct {
	SafeCfg      *SafeConfig
	ExpertHealth map[int]float32 // expertID -> grammar contribution score
}

func NewHyperparameterExpert(cfg *SafeConfig) *HyperparameterExpert {
	return &HyperparameterExpert{
		SafeCfg:      cfg,
		ExpertHealth: make(map[int]float32),
	}
}

// Step is the high-level Supervisor logic called at the end of every epoch
func (e *HyperparameterExpert) Step(metrics TrainingMetrics, surgery SurgeryPerformer) {
	// 1. Update Expert Health based on test paths and scores
	e.UpdateHealth(metrics)

	// 2. ANALYZE & MUTATE
	e.AnalyzeAndAdjust(metrics)
	
	// 3. VERIFY: Expert Path Check
	e.VerifyExpertPaths(metrics)

	// 4. SURGERY: If we have consistent sinks, perform cloning
	if metrics.Epoch > 50 && metrics.SemanticSinksHits > 0 {
		e.PerformSurgery(metrics, surgery)
	}

	// 5. AUTO-STOP: If loss is low and grammar is high, stop training
	if metrics.AverageLoss < 0.2 && metrics.GrammarScore > 25.0 {
		fmt.Printf("\n🎓 [Supervisor] BRAIN MATURATION COMPLETE! Loss: %.4f | Grammar: %.1f\n", 
			metrics.AverageLoss, metrics.GrammarScore)
		fmt.Println("🚀 Triggering Graceful Shutdown.")
		os.Exit(0)
	}
}

// UpdateHealth correlates expert paths with overall epoch success
func (e *HyperparameterExpert) UpdateHealth(metrics TrainingMetrics) {
	for _, res := range metrics.TestResults {
		// Parse expert IDs from path: "word(E1+E8) -> word(E4+E12)"
		experts := e.parseExpertsFromPath(res.Path)
		
		// If the response is semi-coherent, reward the experts
		reward := float32(0.01)
		if metrics.GrammarScore < 10.0 {
			reward = -0.01
		}
		
		for _, eid := range experts {
			e.ExpertHealth[eid] += reward
		}
	}
}

func (e *HyperparameterExpert) parseExpertsFromPath(path string) []int {
	var experts []int
	// Simple parsing for E[number]
	parts := strings.Split(path, "E")
	for i := 1; i < len(parts); i++ {
		var id int
		_, err := fmt.Sscanf(parts[i], "%d", &id)
		if err == nil {
			experts = append(experts, id)
		}
	}
	return experts
}

// AnalyzeAndAdjust reads the latest training metrics and tweaks variables on-the-fly
func (e *HyperparameterExpert) AnalyzeAndAdjust(metrics TrainingMetrics) {
	e.SafeCfg.Update(func(cfg *TrainingConfig) {
		fmt.Printf("\n🧠 [Supervisor Analyze] Epoch %d | Loss: %.4f | Grammar: %.1f | Similarity: %.1f%%\n", 
			metrics.Epoch, metrics.AverageLoss, metrics.GrammarScore, metrics.SimilarityScore)

		// Log Health for top experts
		type healthRank struct { id int; score float32 }
		ranks := []healthRank{}
		for id, s := range e.ExpertHealth { ranks = append(ranks, healthRank{id, s}) }
		sort.Slice(ranks, func(i, j int) bool { return ranks[i].score > ranks[j].score })
		
		fmt.Print("🏥 Expert Health: ")
		for i := 0; i < 3 && i < len(ranks); i++ {
			fmt.Printf("E%d(%.2f) ", ranks[i].id, ranks[i].score)
		}
		fmt.Println()

		// A. Detect "Mumbling" (Word Salad)
		if metrics.GrammarScore < 15.0 && metrics.Epoch > 50 {
			cfg.ContextMultiplier = min(30.0, cfg.ContextMultiplier+0.5)
			cfg.RouterTemperature = max(0.5, cfg.RouterTemperature*0.98)
			fmt.Printf("📝 Detect Mumbling: Increased ContextMultiplier to %.1f, Sharpened Temp to %.2f\n", 
				cfg.ContextMultiplier, cfg.RouterTemperature)
		}

		// B. Tackle Semantic Sinks / Expert Collapse
		if metrics.SemanticSinksHits > 2 {
			cfg.RouterNoise = min(0.50, cfg.RouterNoise+0.05)
			cfg.LoadBalancingWeight = min(1.0, cfg.LoadBalancingWeight+0.02)
			fmt.Printf("⚠️  Sink Detected! Increased Router Noise to %.3f, LBW to %.3f\n", 
				cfg.RouterNoise, cfg.LoadBalancingWeight)
		}

		// C. Adjust Learning Rate based on plateauing
		if metrics.AverageLoss > 5.5 && metrics.Epoch > 20 {
			cfg.LearningRate = max(1e-6, cfg.LearningRate*0.95)
			fmt.Printf("📉 Plateau detected. Scaled down Learning Rate to %e\n", cfg.LearningRate)
		}
	})
	
	// PERSIST
	e.SafeCfg.RLock()
	data, _ := json.MarshalIndent(e.SafeCfg.Config, "", "  ")
	_ = os.WriteFile("data/config/social_train.json", data, 0644)
	e.SafeCfg.RUnlock()
}

// PerformSurgery identifies alpha and sink experts and triggers cloning
func (e *HyperparameterExpert) PerformSurgery(metrics TrainingMetrics, surgery SurgeryPerformer) {
	if surgery == nil {
		return
	}

	for lIdx, resets := range metrics.LayerResets {
		// Find worst expert in this layer (most resets)
		worstID := -1
		maxResets := 0
		for eid, count := range resets {
			if count > maxResets {
				maxResets = count
				worstID = eid
			}
		}

		if worstID != -1 && maxResets > 1 {
			// Find best expert (Alpha) - the one with highest health
			bestID := -1
			maxHealth := float32(-1e9)
			for eid, health := range e.ExpertHealth {
				if health > maxHealth {
					maxHealth = health
					bestID = eid
				}
			}
			
			if bestID != -1 && bestID != worstID {
				surgery.PerformSurgery(lIdx, bestID, worstID)
			}
		}
	}
}

// VerifyExpertPaths checks if the routing decisions make linguistic sense
func (e *HyperparameterExpert) VerifyExpertPaths(metrics TrainingMetrics) {
	for _, res := range metrics.TestResults {
		isQuestion := strings.Contains(strings.ToLower(res.Prompt), "__ques__")
		if isQuestion {
			hasQuestionExpert := strings.Contains(res.Path, "E8") || strings.Contains(res.Path, "E10")
			if !hasQuestionExpert && metrics.Epoch > 100 {
				fmt.Printf("🕵️ [Supervisor Verify] Question Path Nonsense: '%s' (Path: %s). Path lacks PRON/AUX specialization.\n", 
					res.Prompt, res.Path)
				e.SafeCfg.Update(func(cfg *TrainingConfig) {
					cfg.LoadBalancingWeight = min(1.0, cfg.LoadBalancingWeight + 0.01)
				})
			}
		}
	}
}

func min(a, b float32) float32 { if a < b { return a }; return b }
func max(a, b float32) float32 { if a > b { return a }; return b }
