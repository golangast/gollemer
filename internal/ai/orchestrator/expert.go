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
	OverfitMode       bool
}

type TestProbeResult struct {
	Prompt   string
	Response string
	Path     string // Expert path (e.g. "word(E1+E8) -> word(E4+E12)")
}

type SurgeryPerformer interface {
	PerformSurgery(layerIdx, alphaID, sinkID int)
	HealExpert(layerIdx, expertIdx int, alphaIDs []int)
	SetHealthyExperts(layerIdx int, expertIDs []int)
	ResetRouters(layerIdx int)
}

// expertRecord tracks cumulative health and recovery cooldowns for one expert.
type expertRecord struct {
	Health      float32 // EMA health score
	UsageEMA    float32 // EMA of how often it gets used
	HealCount   int     // How many times it has been healed
	CooldownEpochs int  // Epochs remaining before healing can be triggered again
	TrendDown   int     // Consecutive epochs with negative health delta
}

type HyperparameterExpert struct {
	SafeCfg      *SafeConfig
	ExpertHealth map[int]float32    // deprecated alias — kept for backward compat
	records      map[int]*expertRecord
	BoostEpochs  int
	// Trend tracking
	lossHistory    []float32
	grammarHistory []float32
	simHistory     []float32
	// Running averages across epochs
	epochsElapsed int
	// Snapshots for 20-epoch window
	windowLoss    float32
	windowGrammar float32
	windowSim     float32
}

func NewHyperparameterExpert(cfg *SafeConfig) *HyperparameterExpert {
	return &HyperparameterExpert{
		SafeCfg:      cfg,
		ExpertHealth: make(map[int]float32),
		records:      make(map[int]*expertRecord),
	}
}

func (e *HyperparameterExpert) getRecord(id int) *expertRecord {
	if r, ok := e.records[id]; ok {
		return r
	}
	r := &expertRecord{Health: 0.0, UsageEMA: 0.1}
	e.records[id] = r
	return r
}

// Step is the high-level Supervisor logic called at the end of every epoch.
func (e *HyperparameterExpert) Step(metrics TrainingMetrics, surgery SurgeryPerformer) {
	e.epochsElapsed++

	// Tick down cooldowns
	for _, r := range e.records {
		if r.CooldownEpochs > 0 {
			r.CooldownEpochs--
		}
	}

	// 1. Update expert health with momentum-aware scoring
	e.updateHealthMomentum(metrics)

	// 2. Track trends (loss/grammar/sim) for smarter decisions
	e.trackTrends(metrics)

	// 3. Analyze & mutate hyperparameters
	e.AnalyzeAndAdjust(metrics)

	// 4. Verify routing paths
	e.VerifyExpertPaths(metrics)

	// 5. SURGERY: Clone strong experts into collapsed ones (after warmup)
	if metrics.Epoch > 50 && metrics.SemanticSinksHits > 0 && !metrics.OverfitMode {
		e.PerformSurgery(metrics, surgery)
	}

	// 6. HEALING: Blend-reset experts that are critically unhealthy
	e.runHealingPass(metrics, surgery)

	// 7. Propagate healthy expert set to all layers for regularization
	e.propagateHealthyExperts(metrics, surgery)

	// 8. 20-Epoch Linguistic Audit
	if e.epochsElapsed > 0 && e.epochsElapsed%20 == 0 {
		e.AnalyzeLinguisticWindow(metrics, surgery)
	}

	// 9. AUTO-STOP: Maturation check
	if metrics.AverageLoss < 0.2 && metrics.GrammarScore > 25.0 {
		fmt.Printf("\n🎓 [Supervisor] BRAIN MATURATION COMPLETE! Loss: %.4f | Grammar: %.1f\n",
			metrics.AverageLoss, metrics.GrammarScore)
		fmt.Println("🚀 Triggering Graceful Shutdown.")
		os.Exit(0)
	}
}

// updateHealthMomentum updates per-expert health using a momentum-weighted EMA.
// Experts that appear in coherent response paths get rewarded; those absent get penalized.
func (e *HyperparameterExpert) updateHealthMomentum(metrics TrainingMetrics) {
	// Determine overall training quality this epoch
	const (
		healthDecay   = 0.92 // EMA decay for health
		usageDecay    = 0.90
	)

	// Quality signal: combined score normalized to [-1, +1]
	quality := float32(0)
	if metrics.GrammarScore > 0 {
		quality += (metrics.GrammarScore/30.0)*0.5 - 0.25 // grammar: 0..30 → -0.25..+0.25
	}
	if metrics.SimilarityScore > 0 {
		quality += (metrics.SimilarityScore/100.0)*0.5 - 0.25 // sim: 0..100% → -0.25..+0.25
	}
	if metrics.AverageLoss < 5.0 {
		quality += 0.3 // Loss is getting reasonable
	}

	expertsSeen := make(map[int]bool)
	for _, res := range metrics.TestResults {
		experts := e.parseExpertsFromPath(res.Path)
		for _, eid := range experts {
			expertsSeen[eid] = true
		}
	}

	// Update records
	for _, res := range metrics.TestResults {
		experts := e.parseExpertsFromPath(res.Path)
		for _, eid := range experts {
			r := e.getRecord(eid)
			delta := quality * 0.6 // scale reward
			r.Health = r.Health*healthDecay + delta*(1-healthDecay)
			r.UsageEMA = r.UsageEMA*usageDecay + 1.0*(1-usageDecay)
			e.ExpertHealth[eid] = r.Health
		}
	}

	// Penalize experts never seen in any test path this epoch
	for id, r := range e.records {
		if !expertsSeen[id] {
			r.Health = r.Health*healthDecay + (-0.05)*(1-healthDecay)
			r.UsageEMA = r.UsageEMA * usageDecay
			r.TrendDown++
			e.ExpertHealth[id] = r.Health
		} else {
			r.TrendDown = 0
		}
	}
}

func (e *HyperparameterExpert) trackTrends(metrics TrainingMetrics) {
	const maxHistory = 20 // Window of 20 epochs as requested
	e.lossHistory = append(e.lossHistory, metrics.AverageLoss)
	e.grammarHistory = append(e.grammarHistory, metrics.GrammarScore)
	e.simHistory = append(e.simHistory, metrics.SimilarityScore)
	if len(e.lossHistory) > maxHistory {
		e.lossHistory = e.lossHistory[len(e.lossHistory)-maxHistory:]
		e.grammarHistory = e.grammarHistory[len(e.grammarHistory)-maxHistory:]
		e.simHistory = e.simHistory[len(e.simHistory)-maxHistory:]
	}
}

// lossImproving returns true if the last N epochs show a downward loss trend.
func (e *HyperparameterExpert) lossImproving(n int) bool {
	h := e.lossHistory
	if len(h) < n+1 {
		return false
	}
	h = h[len(h)-n-1:]
	return h[n] < h[0]
}

// grammarStagnant returns true if grammar hasn't improved in n epochs.
func (e *HyperparameterExpert) grammarStagnant(n int) bool {
	h := e.grammarHistory
	if len(h) < n {
		return false
	}
	h = h[len(h)-n:]
	max := h[0]
	for _, v := range h {
		if v > max {
			max = v
		}
	}
	// Stagnant if best in window equals first value (no improvement)
	return max <= h[0]+0.5
}

// runHealingPass triggers genetic blending for critically unhealthy experts.
func (e *HyperparameterExpert) runHealingPass(metrics TrainingMetrics, surgery SurgeryPerformer) {
	// Build sorted list of healthiest experts to use as anchors
	type rank struct {
		id     int
		health float32
	}
	var allRanks []rank
	for id, r := range e.records {
		allRanks = append(allRanks, rank{id, r.Health})
	}
	sort.Slice(allRanks, func(i, j int) bool { return allRanks[i].health > allRanks[j].health })

	// Global anchor pool: top 3 healthy experts
	var globalAnchors []int
	for _, r := range allRanks {
		if r.health > -0.2 && len(globalAnchors) < 3 {
			globalAnchors = append(globalAnchors, r.id)
		}
	}
	if len(globalAnchors) == 0 {
		globalAnchors = []int{14, 9} // Structural defaults: GREET, VERB
	}

	criticalThreshold := float32(-1.0)

	for id, r := range e.records {
		if r.Health >= criticalThreshold {
			continue
		}
		if r.CooldownEpochs > 0 {
			fmt.Printf("🏥 [Supervisor] Expert E%d health critical (%.2f) but on cooldown (%d epochs).\n",
				id, r.Health, r.CooldownEpochs)
			continue
		}

		fmt.Printf("🏥 [Supervisor] Expert E%d health is critical (%.2f). Triggering Healing pass.\n", id, r.Health)

		// Reset health to break the trigger cycle
		r.Health = 0.0
		e.ExpertHealth[id] = 0.0
		r.HealCount++
		// Cooldown grows with each successive healing (max 5 epochs)
		r.CooldownEpochs = 1 + r.HealCount
		if r.CooldownEpochs > 5 {
			r.CooldownEpochs = 5
		}

		// Use per-expert anchors: prefer experts that are healthy AND different from the sick one
		anchors := make([]int, 0, 3)
		for _, rk := range allRanks {
			if rk.id != id && rk.health > -0.1 && len(anchors) < 3 {
				anchors = append(anchors, rk.id)
			}
		}
		if len(anchors) == 0 {
			anchors = globalAnchors
		}

		for lIdx := range metrics.LayerUsage {
			surgery.HealExpert(lIdx, id, anchors)
		}
	}
}

// propagateHealthyExperts updates all layers with the current healthy expert set.
func (e *HyperparameterExpert) propagateHealthyExperts(metrics TrainingMetrics, surgery SurgeryPerformer) {
	type healthRank struct {
		id    int
		score float32
	}
	ranks := []healthRank{}
	for id, s := range e.ExpertHealth {
		ranks = append(ranks, healthRank{id, s})
	}
	sort.Slice(ranks, func(i, j int) bool { return ranks[i].score > ranks[j].score })

	healthyIDs := []int{}
	// Always try the grammar anchors first
	for _, id := range []int{14, 9} {
		if e.ExpertHealth[id] > -0.5 {
			healthyIDs = append(healthyIDs, id)
		}
	}
	// Add top performers
	for i := 0; i < 3 && i < len(ranks); i++ {
		if ranks[i].score > 0.1 {
			found := false
			for _, hid := range healthyIDs {
				if hid == ranks[i].id {
					found = true
					break
				}
			}
			if !found {
				healthyIDs = append(healthyIDs, ranks[i].id)
			}
		}
	}

	if len(healthyIDs) > 0 {
		for lIdx := range metrics.LayerUsage {
			surgery.SetHealthyExperts(lIdx, healthyIDs)
		}
	}
}

// UpdateHealth is the legacy method kept for compatibility.
func (e *HyperparameterExpert) UpdateHealth(metrics TrainingMetrics) {
	e.updateHealthMomentum(metrics)
}

func (e *HyperparameterExpert) parseExpertsFromPath(path string) []int {
	var experts []int
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

// AnalyzeAndAdjust reads the latest training metrics and tweaks hyperparameters on-the-fly.
func (e *HyperparameterExpert) AnalyzeAndAdjust(metrics TrainingMetrics) {
	e.SafeCfg.Update(func(cfg *TrainingConfig) {
		if metrics.Epoch%20 == 0 {
			fmt.Printf("\n🧠 [Supervisor Analyze] Epoch %d | Loss: %.4f | Grammar: %.1f | Similarity: %.1f%% | UnkPen: %.3f | BiasInt: %.1f\n",
				metrics.Epoch, metrics.AverageLoss, metrics.GrammarScore, metrics.SimilarityScore, cfg.UnkPenalty, cfg.StructuralBiasIntensity)

			// Log top-3 and bottom-3 expert health
			type healthRank struct{ id int; score float32 }
			var ranks []healthRank
			for id, s := range e.ExpertHealth {
				ranks = append(ranks, healthRank{id, s})
			}
			sort.Slice(ranks, func(i, j int) bool { return ranks[i].score > ranks[j].score })

			fmt.Print("🏥 Expert Health: ")
			for i := 0; i < 3 && i < len(ranks); i++ {
				fmt.Printf("E%d(%.2f) ", ranks[i].id, ranks[i].score)
			}
			if len(ranks) > 6 {
				fmt.Print("... ")
				for i := len(ranks) - 3; i < len(ranks); i++ {
					fmt.Printf("E%d(%.2f) ", ranks[i].id, ranks[i].score)
				}
			}
			fmt.Println()
		}

		// ── OverfitMode: Lockdown ───────────────────────────────────────────
		if cfg.OverfitMode {
			fmt.Println("🎯 Supervisor: OverfitMode active. Lockdown engaged.")
			cfg.RouterNoise = 0.0
			cfg.RouterTemperature = 0.05 // Even lower for perfect reproduction
			// Note: We DO NOT return here anymore; we want to adjust ContextMultiplier and LR even in OverfitMode
		}

		// ── Exploration Boost ───────────────────────────────────────────────
		if e.BoostEpochs > 0 {
			fmt.Printf("🔥 [Supervisor] EXPLORATION BOOST: %d epochs remaining\n", e.BoostEpochs)
			cfg.RouterTemperature = 0.7
			cfg.RouterNoise = 0.06
			e.BoostEpochs--
			return
		}

		// Trigger exploration boost for chronic word salad
		if metrics.Epoch > 30 && metrics.GrammarScore < 10.0 && e.BoostEpochs == 0 && e.grammarStagnant(5) {
			fmt.Println("🚨 [Supervisor] Grammar stagnant. Triggering 8-epoch Exploration Boost.")
			e.BoostEpochs = 8
		}

		// ── A: Context Multiplier — boost if model is mumbling ────────────
		if metrics.GrammarScore < 15.0 && metrics.Epoch > 20 {
			cfg.ContextMultiplier = min32(30.0, cfg.ContextMultiplier+0.3)
			cfg.RouterTemperature = max32(0.5, cfg.RouterTemperature*0.97)
			fmt.Printf("📝 Mumbling: ContextMultiplier→%.1f, Temp→%.2f\n",
				cfg.ContextMultiplier, cfg.RouterTemperature)
		} else if metrics.GrammarScore >= 20.0 {
			// Cool down temperature as model matures
			cfg.RouterTemperature = max32(0.4, cfg.RouterTemperature*0.99)
		}

		// ── B: Expert Collapse ────────────────────────────────────────────
		if metrics.SemanticSinksHits > 2 {
			cfg.RouterNoise = min32(0.50, cfg.RouterNoise+0.04)
			cfg.LoadBalancingWeight = min32(1.0, cfg.LoadBalancingWeight+0.02)
			fmt.Printf("⚠️  Sink: Noise→%.3f, LBW→%.3f\n", cfg.RouterNoise, cfg.LoadBalancingWeight)
		} else if metrics.GatingEntropy > 3.0 {
			// Good entropy: dial down noise slightly to let routing solidify
			cfg.RouterNoise = max32(0.01, cfg.RouterNoise*0.97)
		}

		// ── C: Learning Rate — plateau detection ──────────────────────────
		if metrics.AverageLoss > 5.5 && metrics.Epoch > 20 && !e.lossImproving(3) {
			cfg.LearningRate = max32(1e-6, cfg.LearningRate*0.93)
			fmt.Printf("📉 Plateau: LR→%e\n", cfg.LearningRate)
		} else if metrics.AverageLoss < 3.0 && e.lossImproving(5) && metrics.Epoch > 50 {
			// Loss is improving well — allow slight LR recovery
			cfg.LearningRate = min32(cfg.LearningRate*1.05, 5e-4)
		}

		// ── D: Structural Routing Bias Decay ─────────────────────────────
		if metrics.GrammarScore >= 20.0 {
			decay := 5.0 - (metrics.GrammarScore-20.0)*0.4
			if decay < 1.0 {
				decay = 1.0
			}
			cfg.StructuralRoutingWeight = decay
			fmt.Printf("📉 Bias Decay: StructuralRoutingWeight→%.2f\n", cfg.StructuralRoutingWeight)
			
			// Also decay the bias intensity to let the model "learn for real"
			cfg.StructuralBiasIntensity = max32(2.0, 8.0 - (metrics.GrammarScore-20.0)*0.5)
		} else {
			cfg.StructuralRoutingWeight = 5.0
			cfg.StructuralBiasIntensity = 4.0
		}

		// ── UNK Penalty ──────────────────────────────────────────────────
		if metrics.GrammarScore < 10.0 {
			cfg.UnkPenalty = 0.15 // Standard suppression
		} else if metrics.GrammarScore > 20.0 {
			cfg.UnkPenalty = 0.005 // Extreme suppression to force variety
		}

		// ── E: Similarity-driven feedback ────────────────────────────────
		if metrics.SimilarityScore < 25.0 && metrics.Epoch > 10 {
			cfg.ContextMultiplier = min32(60.0, cfg.ContextMultiplier+0.8)
			cfg.StructuralBiasIntensity = min32(20.0, cfg.StructuralBiasIntensity+1.5)
			fmt.Printf("🎯 Low Similarity (%.1f%%): ContextMultiplier→%.1f, BiasInt→%.1f\n",
				metrics.SimilarityScore, cfg.ContextMultiplier, cfg.StructuralBiasIntensity)
		} else if metrics.SimilarityScore > 45.0 {
			cfg.StructuralRoutingWeight = max32(1.0, cfg.StructuralRoutingWeight*0.95)
			cfg.ContextMultiplier = max32(1.0, cfg.ContextMultiplier*0.98)
		}

		// ── F: LR Defibrillation — escape deep local minima ──────────────
		// When LR has decayed below 5e-06 AND grammar is stuck below 15, shock the weights.
		if cfg.LearningRate < 5e-6 && metrics.GrammarScore < 15.0 && metrics.Epoch > 50 && e.grammarStagnant(5) {
			cfg.LearningRate = 8e-5
			fmt.Printf("⚡ [Supervisor] LR Defibrillation! Bumping LR 1e-06→8e-05 to escape local minima.\n")
		}

		// ── G: Expert Regularization (enable after warmup) ───────────────
		if metrics.Epoch > 80 && cfg.ExpertRegularizationWeight == 0 {
			cfg.ExpertRegularizationWeight = 0.001
			fmt.Println("✅ Expert regularization enabled.")
		}

		// ── H: Noise Decay — once model is stable, taper noise ───────────
		if metrics.GrammarScore > 18.0 && metrics.SimilarityScore > 15.0 {
			cfg.RouterNoise = max32(0.005, cfg.RouterNoise*0.97)
			fmt.Printf("🔕 Stable: Router noise decayed to %.4f\n", cfg.RouterNoise)
		}
	})

	// Persist config
	e.SafeCfg.RLock()
	data, _ := json.MarshalIndent(e.SafeCfg.Config, "", "  ")
	_ = os.WriteFile("data/config/social_train.json", data, 0644)
	e.SafeCfg.RUnlock()
}

// AnalyzeLinguisticWindow evaluates performance over a 20-epoch block and intervenes if necessary.
func (e *HyperparameterExpert) AnalyzeLinguisticWindow(metrics TrainingMetrics, surgery SurgeryPerformer) {
	if len(e.grammarHistory) < 20 {
		return
	}

	var sumScore, sumGrammar, sumSim float32
	for i := 0; i < 20; i++ {
		sumGrammar += e.grammarHistory[i]
		sumSim += e.simHistory[i]
		// Combined Score estimate (matching UI progress: Grammar + Sim Contribution)
		sumScore += e.grammarHistory[i] + (e.simHistory[i] * 0.3)
	}
	avgGrammar := sumGrammar / 20.0
	avgSim := sumSim / 20.0
	avgScore := sumScore / 20.0

	fmt.Printf("\n🧠 [Supervisor Audit] 20-Epoch Window | Avg Score: %.1f | Avg Grammar: %.1f | Avg Sim: %.1f%%\n",
		avgScore, avgGrammar, avgSim)

	// Intervention Targets: Grammar < 18 or Sim < 15% after window
	if avgGrammar < 18.0 || avgSim < 15.0 {
		fmt.Println("🚨 [Supervisor Audit] Progress below target. Triggering GLOBAL EXPERT UPDATE.")
		e.GlobalExpertRefresh(surgery)
		
		// Also nudge hyperparameters to break the plateau
		e.SafeCfg.Update(func(cfg *TrainingConfig) {
			cfg.RouterTemperature = min32(1.2, cfg.RouterTemperature*1.2) // Increase exploration
			cfg.ContextMultiplier = min32(50.0, cfg.ContextMultiplier*1.1)
			cfg.StructuralBiasIntensity = min32(15.0, cfg.StructuralBiasIntensity+2.0)
		})
	} else {
		fmt.Println("✅ [Supervisor Audit] Linguistic progress is on track. Smoothing expert weights...")
		e.SafeCfg.Update(func(cfg *TrainingConfig) {
			cfg.RouterTemperature = max32(0.2, cfg.RouterTemperature*0.9) // Sharpen
		})
	}
}

// GlobalExpertRefresh updates all experts by blending them with known healthy anchors.
func (e *HyperparameterExpert) GlobalExpertRefresh(surgery SurgeryPerformer) {
	// Identify anchors (Top 3 healthiest experts)
	type rank struct{ id int; health float32 }
	var allRanks []rank
	for id, r := range e.records {
		allRanks = append(allRanks, rank{id, r.Health})
	}
	sort.Slice(allRanks, func(i, j int) bool { return allRanks[i].health > allRanks[j].health })

	var anchors []int
	for _, r := range allRanks {
		if r.health > -0.1 && len(anchors) < 3 {
			anchors = append(anchors, r.id)
		}
	}
	if len(anchors) == 0 {
		anchors = []int{14, 9, 8} // Defaults: GREET, VERB, PRON
	}

	fmt.Printf("🔄 [Supervisor] Global Refresh: Updating ALL experts using anchors %v\n", anchors)
	
	// Apply Healing to ALL experts (0-15 typically)
	for lIdx := 0; lIdx < 16; lIdx++ { // Safeguard: loop many layers, interface checks bounds
		surgery.ResetRouters(lIdx)
		for eIdx := 0; eIdx < 16; eIdx++ {
			surgery.HealExpert(lIdx, eIdx, anchors)
		}
	}
}

// PerformSurgery identifies alpha and sink experts and triggers cloning.
func (e *HyperparameterExpert) PerformSurgery(metrics TrainingMetrics, surgery SurgeryPerformer) {
	if surgery == nil {
		return
	}

	for lIdx, resets := range metrics.LayerResets {
		worstID := -1
		maxResets := 0
		for eid, count := range resets {
			if count > maxResets {
				maxResets = count
				worstID = eid
			}
		}
		if worstID == -1 || maxResets <= 1 {
			continue
		}

		// Alpha: highest-health expert that is NOT the worst AND has positive health.
		// If no expert is healthy enough, skip surgery to avoid sick→sick cloning.
		bestID := -1
		maxHealth := float32(-0.1) // minimum threshold: must be at least slightly positive
		for eid, r := range e.records {
			if eid == worstID {
				continue
			}
			if r.Health > maxHealth {
				maxHealth = r.Health
				bestID = eid
			}
		}

		if bestID != -1 {
			fmt.Printf("🔬 [Supervisor Surgery] Layer %d: Cloning E%d (health=%.2f) → E%d (resets=%d)\n",
				lIdx, bestID, maxHealth, worstID, maxResets)
			surgery.PerformSurgery(lIdx, bestID, worstID)
		} else {
			fmt.Printf("⏭️ [Supervisor Surgery] Layer %d: Skipping — no healthy alpha available (all experts sick)\n", lIdx)
		}
	}
}

// VerifyExpertPaths checks if routing decisions make linguistic sense.
func (e *HyperparameterExpert) VerifyExpertPaths(metrics TrainingMetrics) {
	for _, res := range metrics.TestResults {
		isQuestion := strings.Contains(strings.ToLower(res.Prompt), "__ques__")
		if isQuestion {
			// Check for PRON (E8) or AUX (E10) experts in question context
			hasQuestionExpert := strings.Contains(res.Path, "E8") || strings.Contains(res.Path, "E10")
			
			// Relational check: Does it have a structure like PRON -> AUX or PRON -> VERB?
			hasRelation := (strings.Contains(res.Path, "E8") && strings.Contains(res.Path, "E10")) ||
				(strings.Contains(res.Path, "E8") && strings.Contains(res.Path, "E9"))

			if !hasQuestionExpert && metrics.Epoch > 100 {
				fmt.Printf("🕵️ [Supervisor Verify] Question lacks PRON/AUX expert: '%s' (Path: %s)\n",
					res.Prompt, res.Path)
				e.SafeCfg.Update(func(cfg *TrainingConfig) {
					cfg.LoadBalancingWeight = min32(1.0, cfg.LoadBalancingWeight+0.01)
					cfg.StructuralRoutingWeight = min32(10.0, cfg.StructuralRoutingWeight+0.5)
				})
			}

			if !hasRelation && metrics.Epoch > 150 {
				fmt.Printf("🕵️ [Supervisor Verify] Weak token relations (No PRON-AUX/VERB link): '%s'\n", res.Prompt)
				// Nudge the model to care more about structure
				e.SafeCfg.Update(func(cfg *TrainingConfig) {
					cfg.EntropyWeight = min32(0.5, cfg.EntropyWeight+0.005)
				})
			}
		}
	}
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// Keep old names to avoid breaking any callers from the old code.
func min(a, b float32) float32 { return min32(a, b) }
func max(a, b float32) float32 { return max32(a, b) }
