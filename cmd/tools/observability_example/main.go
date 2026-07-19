package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// ExampleObservabilityIntegration demonstrates how to use all 4 observability features.
func main() {
	// ============================================================================
	// SETUP: Initialize your model, vocabulary, and Word2Vec
	// ============================================================================

	// Load vocabulary
	vocab := vocab.NewVocabulary()
	for _, word := range []string{"hello", "world", "the", "quick", "brown", "fox", "jumped", "over", "lazy", "dog", "today", "tomorrow", "learn", "model", "token", "routing", "expert"} {
		vocab.AddToken(word)
	}

	// Load or initialize Word2Vec model
	w2vModel := &word2vec.SimpleWord2Vec{
		VocabSize:      100, // Placeholder
		VectorSize:     64,
		WordVectorsF32: make(map[int][]float32),
	}
	// (Populate with embeddings)

	// Initialize your MoE model (pseudo-code - replace with actual init)
	_ = vocab    // placeholder
	_ = w2vModel // placeholder

	// Create trainer with observability
	trainer := &moe.Trainer{
		BestModelPath: "data/models/gob_models/golden_checkpoint.gob",
	}

	// Create a small demo model to use if no checkpoint is available
	var demoModel *moe.IntentMoE
	if dm, err := moe.NewIntentMoE(vocab.Size(), 64, 8, vocab.Size(), vocab.Size(), vocab.Size(), 2, w2vModel); err == nil {
		demoModel = dm
	}

	// Register a simple on-demand inference hook for demo purposes.
	moe.OnDemandInference = func(prompt string, maxLen int) ([]map[string]interface{}, error) {
		// Simulate tokenization
		tokens := strings.Fields(strings.ToLower(prompt))
		tokenIDs := make([]int, len(tokens))
		for i, t := range tokens {
			id := vocab.GetTokenID(t)
			if id < 0 || id == vocab.UnkID {
				id = vocab.AddToken(t)
			}
			tokenIDs[i] = id
		}

		// Try to run a real inference using a loaded checkpoint if available
		if trainer.BestModelPath != "" {
			if _, err := os.Stat(trainer.BestModelPath); err == nil {
				ckpt, err := moe.LoadIntentMoECheckpoint(trainer.BestModelPath)
				if err == nil && ckpt != nil && ckpt.Model != nil {
					// Build query tensor (batch=1, seq=len(tokens))
					qdata := make([]float32, len(tokenIDs))
					for i, id := range tokenIDs {
						qdata[i] = float32(id)
					}
					queryTensor := tensor.NewTensor([]int{1, len(tokenIDs)}, qdata, false)

					// Minimal target tensor placeholder (required by Forward signature)
					tgtLen := maxLen
					if tgtLen < len(tokenIDs) {
						tgtLen = len(tokenIDs)
					}
					if tgtLen < 1 {
						tgtLen = 1
					}
					tgt := make([]float32, tgtLen)
					targetTensor := tensor.NewTensor([]int{1, tgtLen}, tgt, false)

					// Set temporary token buffer on ObservabilityInstance and call Forward
					if moe.ObservabilityInstance != nil {
						moe.ObservabilityInstance.SetTempTokenIDs(tokenIDs)
					}
					_, _, err = ckpt.Model.Forward(0.0, queryTensor, targetTensor)
					// Clear the temp buffer
					if moe.ObservabilityInstance != nil {
						moe.ObservabilityInstance.ClearTempTokenIDs()
					}
					if err == nil {
						traceSummary := []map[string]interface{}{{"prompt": prompt, "tokens": tokenIDs}}
						return traceSummary, nil
					}
				}
			}
		}

		// If no checkpoint forward succeeded, try the in-memory demo model
		if demoModel != nil {
			qdata := make([]float32, len(tokenIDs))
			for i, id := range tokenIDs {
				qdata[i] = float32(id)
			}
			queryTensor := tensor.NewTensor([]int{1, len(tokenIDs)}, qdata, false)
			tgtLen := maxLen
			if tgtLen < len(tokenIDs) {
				tgtLen = len(tokenIDs)
			}
			if tgtLen < 1 {
				tgtLen = 1
			}
			tgt := make([]float32, tgtLen)
			targetTensor := tensor.NewTensor([]int{1, tgtLen}, tgt, false)
			if moe.ObservabilityInstance != nil {
				moe.ObservabilityInstance.SetTempTokenIDs(tokenIDs)
			}
			_, _, _ = demoModel.Forward(0.0, queryTensor, targetTensor)
			if moe.ObservabilityInstance != nil {
				moe.ObservabilityInstance.ClearTempTokenIDs()
			}
			traceSummary := []map[string]interface{}{{"prompt": prompt, "tokens": tokens}}
			return traceSummary, nil
		}

		// If there are active layers instrumented, use them, otherwise fabricate 3 layers for demo
		if len(moe.ActiveLayers) > 0 {
			for li, layer := range moe.ActiveLayers {
				selected := make([][]int, len(tokenIDs))
				confs := make([][]float32, len(tokenIDs))
				for ti := range tokenIDs {
					expert := ti % len(layer.Experts)
					selected[ti] = []int{expert}
					confs[ti] = []float32{0.8}
				}
				moe.ObservabilityInstance.SetLayerSelection(li, tokenIDs, selected, confs)
			}
		} else {
			// Fabricate 3 layers with 8 experts each
			for li := 0; li < 3; li++ {
				selected := make([][]int, len(tokenIDs))
				confs := make([][]float32, len(tokenIDs))
				for ti := range tokenIDs {
					expert := (ti + li) % 8
					selected[ti] = []int{expert}
					confs[ti] = []float32{0.7 + float32(li)*0.1}
				}
				moe.ObservabilityInstance.SetLayerSelection(li, tokenIDs, selected, confs)
			}
		}

		// Return a lightweight trace summary
		traceSummary := []map[string]interface{}{{"prompt": prompt, "tokens": tokens}}
		return traceSummary, nil
	}

	// ============================================================================
	// FEATURE 1: Initialize Observability
	// ============================================================================

	log.Println("🚀 Initializing Observability...")
	trainer.InitializeObservability(
		8,   // numExperts
		500, // windowSize (metrics collected every 500 steps)
		vocab,
		w2vModel,
	)

	// ============================================================================
	// FEATURE 2: Start Metrics HTTP Server
	// ============================================================================
	// Register static handlers for dashboard
	setupDashboardServer(vocab)

	port := os.Getenv("MOE_OBSERVABILITY_PORT")
	if port == "" {
		port = "8080"
	}
	portNum, err := strconv.Atoi(port)
	if err != nil || portNum <= 0 {
		portNum = 8080
		port = "8080"
	}

	aggregator := moe.StartMetricsServer(trainer, vocab, fmt.Sprintf(":%d", portNum))
	log.Printf("📊 Metrics server started at http://localhost:%s", port)
	log.Printf("📈 Open dashboard at: http://localhost:%s/docs/dashboard-enhanced.html", port)

	// ============================================================================
	// FEATURE 3: Simulate Training Loop
	// ============================================================================

	epochs := 10
	stepsPerEpoch := 500
	batchSize := 32

	for epoch := 0; epoch < epochs; epoch++ {
		log.Printf("\n📌 Epoch %d/%d\n", epoch+1, epochs)

		totalLoss := float32(0)
		totalSteps := 0

		// Training loop
		for step := 0; step < stepsPerEpoch; step++ {
			// ================================================================
			// Simulate batch data
			// ================================================================
			batch := generateBatch(batchSize, vocab)

			// ================================================================
			// Forward pass (pseudo-code)
			// ================================================================
			expertIDs := make([]int, batchSize)
			tokenIDs := make([]int, 0)

			for i := 0; i < batchSize; i++ {
				// In real code: call model.Router() to get expert assignment
				expertIDs[i] = i % 8 // Simulate expert routing

				// Collect token IDs from batch
				tokenIDs = append(tokenIDs, batch[i]...)
			}

			// Simulate loss (in reality: crossentropy loss)
			loss := simulateLoss(epoch, step)

			// ================================================================
			// CRITICAL: Record metrics before backward pass
			// ================================================================
			trainer.RecordTrainingStep(expertIDs, tokenIDs, loss)
			totalLoss += loss
			totalSteps++
			aggregator.CollectMetrics()

			// Record weight snapshots for velocity tracking (every 10 steps)
			if step%10 == 0 {
				layer0Weights := make([]float32, 128)
				for i := range layer0Weights {
					layer0Weights[i] = float32((step+i)%20) * 0.01
				}
				trainer.RecordWeightSnapshot("layer_0", layer0Weights)
			}

			// ================================================================
			// Backward pass (pseudo-code)
			// ================================================================
			// loss_tensor := model.Backward(loss)
			// optimizer.Step()

			// ================================================================
			// Update weight velocity after parameters change
			// ================================================================
			if (step+1)%10 == 0 {
				newLayer0Weights := make([]float32, 128)
				for i := range newLayer0Weights {
					newLayer0Weights[i] = float32((step+i+5)%20) * 0.01
				}
				embeddingSize := max(vocab.Size(), 8)
				embeddingData := make([]float32, embeddingSize*64)
				for i := range embeddingData {
					embeddingData[i] = float32((step+i)%15) * 0.02
				}
				embeddings := tensor.NewTensor([]int{embeddingSize, 64}, embeddingData, false)
				trainer.UpdateWeightVelocity("layer_0", newLayer0Weights, embeddings)
				trainer.UpdateEmbeddingGalaxy(vocab, embeddings, 20)

				for idx := 0; idx < 3 && idx < len(tokenIDs); idx++ {
					tokenID := tokenIDs[idx]
					tokenStr := vocab.GetWord(tokenID)
					if tokenStr == "" {
						tokenStr = fmt.Sprintf("ID:%d", tokenID)
					}
					expertPath := []int{idx % 8, (idx + 1) % 8, (idx + 2) % 8}
					confidences := []float32{0.8, 0.7, 0.9}
					trainer.RecordTokenTrajectory(tokenID, tokenStr, expertPath, confidences)
				}
			}

			// Give the dashboard a moment to reflect each training step.
			time.Sleep(2 * time.Millisecond)

			// Print progress
		}

		// ====================================================================
		// End of epoch: Log observability report and reset windowed metrics
		// ====================================================================
		trainer.FinishEpoch(vocab)

		// Optional: Export metrics to JSON for archival
		// ====================================================================
		if epoch%2 == 0 {
			// In real code:
			// metricsJSON, err := trainer.ExportObservabilityMetrics(vocab)
			// if err == nil {
			//     log.Printf("📁 Saved metrics to epoch_%d.json\n", epoch)
			//     // write metricsJSON to file
			// }
		}

		// ====================================================================
		// Check health indicators for auto-healing triggers
		// ====================================================================
		checkHealthAndReact(aggregator)
	}

	// ====================================================================
	// Final Report
	// ====================================================================
	log.Println("\n✅ Training Complete!")
	aggregator.LogHealthReport()

	// Keep server running for dashboard access
	log.Println("\n📊 Dashboard continues running. Press Ctrl+C to exit.")
	log.Println("   Access: http://localhost:8080/docs/dashboard-enhanced.html")
	select {} // Block forever
}

// ============================================================================
// Helper Functions
// ============================================================================

// generateBatch creates a fake batch of token sequences.
func generateBatch(batchSize int, vocab *vocab.Vocabulary) [][]int {
	batch := make([][]int, batchSize)
	for i := 0; i < batchSize; i++ {
		// Simulate a sequence of 10 tokens per sample
		batch[i] = make([]int, 10)
		for j := 0; j < 10; j++ {
			batch[i][j] = (i + j) % vocab.Size() // Fake token ID
		}
	}
	return batch
}

// simulateLoss creates a fake loss value that decreases over time.
func simulateLoss(epoch, step int) float32 {
	// Simulate training loss: decreases with epoch, varies per step
	baseDecay := 1.0 - float32(epoch)*0.05
	stepNoise := 0.1 * float32(step%10) / 10.0
	return baseDecay * (1.0 + stepNoise)
}

// checkHealthAndReact demonstrates using observability for auto-healing.
func checkHealthAndReact(aggregator *moe.MetricsAggregator) {
	payload := aggregator.BuildDashboardPayload()

	// Check 1: Expert Balance
	indicators := payload.HealthIndicators
	if util, ok := indicators["expert_utilization"].(float32); ok {
		if util > 0.9 {
			log.Println("⚠️  WARNING: Expert utilization very high! Consider:")
			log.Println("    - Increasing router noise to encourage diversity")
			log.Println("    - Adding more experts to the model")
		} else if util < 0.3 {
			log.Println("⚠️  WARNING: Expert utilization very low! Consider:")
			log.Println("    - Reducing router temperature")
			log.Println("    - Checking if most experts are collapsed")
		}
	}

	// Check 2: Learning Intensity
	if maxVel, ok := indicators["max_weight_velocity"].(float32); ok {
		if maxVel > 0.15 {
			log.Println("🔥 High weight velocity detected - model is learning aggressively")
		} else if maxVel < 0.001 {
			log.Println("❄️ Very low weight velocity - check if training has plateaued")
		}
	}

	// Check 3: Token Category Balance
	if payload.CategoryLoss != nil {
		for catName, stats := range payload.CategoryLoss {
			if improvement, ok := stats["improvement"].(float32); ok {
				if improvement < -0.1 { // Loss getting worse
					log.Printf("📍 Alert: Category '%s' losing ground (improvement: %.4f)\n", catName, improvement)
					log.Println("   Consider adjusting curriculum or learning rate")
				}
			}
		}
	}

	// Check 4: Semantic Drift Stagnation
	if drifts := payload.SemanticDrift; drifts != nil {
		if len(drifts) == 0 {
			log.Println("⚠️  No semantic drift detected - embeddings may be frozen")
		}
	}
}

// ============================================================================
// Minimal HTTP server setup for dashboard
// ============================================================================

func setupDashboardServer(vocab *vocab.Vocabulary) {
	// Serve dashboard files from the docs directory under /docs/
	fs := http.FileServer(http.Dir("docs"))
	http.Handle("/docs/", http.StripPrefix("/docs/", fs))

	// Metrics endpoints already registered by StartMetricsServer()

	// Temporary testing endpoint: POST JSON mapping of expertID->[]tokens
	// Example payload: {"0":["alpha","bravo"],"1":["charlie"]}
	http.HandleFunc("/api/metrics/inject", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		if moe.ObservabilityInstance == nil {
			http.Error(w, "observability not enabled", http.StatusBadRequest)
			return
		}
		var payload map[string][]string
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "bad JSON", http.StatusBadRequest)
			return
		}
		// For each expert->tokens list, map tokens to IDs and record routes
		for k, toks := range payload {
			eid, err := strconv.Atoi(k)
			if err != nil {
				continue
			}
			for _, t := range toks {
				tid := vocab.GetTokenID(t)
				if tid < 0 || tid == vocab.UnkID {
					tid = vocab.AddToken(t)
				}
				moe.ObservabilityInstance.ExpertLexicon.RecordTokenRoute(eid, tid)
			}
		}
		json.NewEncoder(w).Encode(map[string]bool{"ok": true})
	})

	// Also expose a non-prefixed inject endpoint to avoid mux prefix conflicts
	http.HandleFunc("/observability/inject", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		if moe.ObservabilityInstance == nil {
			http.Error(w, "observability not enabled", http.StatusBadRequest)
			return
		}
		var payload map[string][]string
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			http.Error(w, "bad JSON", http.StatusBadRequest)
			return
		}
		for k, toks := range payload {
			eid, err := strconv.Atoi(k)
			if err != nil {
				continue
			}
			for _, t := range toks {
				tid := vocab.GetTokenID(t)
				if tid < 0 || tid == vocab.UnkID {
					tid = vocab.AddToken(t)
				}
				moe.ObservabilityInstance.ExpertLexicon.RecordTokenRoute(eid, tid)
			}
		}
		json.NewEncoder(w).Encode(map[string]bool{"ok": true})
	})
}

// ============================================================================
// Example of advanced usage: Custom metrics export
// ============================================================================

func exportMetricsForAnalysis(trainer *moe.Trainer, vocab *vocab.Vocabulary) error {
	// Get current observability metrics
	metricsJSON, err := trainer.ExportObservabilityMetrics(vocab)
	if err != nil {
		return err
	}

	// Parse and analyze
	log.Printf("📊 Exported metrics (length: %d bytes)\n", len(metricsJSON))
	log.Println("Sample JSON structure:")
	log.Println(metricsJSON[:min(len(metricsJSON), 500)] + "...")

	return nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// Example of advanced usage: Custom alerts
// ============================================================================

func setupAlertSystem(aggregator *moe.MetricsAggregator) {
	go func() {
		for {
			<-time.After(10 * time.Second)

			current := aggregator.GetCurrentMetrics()
			if catLoss, ok := current["category_loss"].(map[string]interface{}); ok {
				for catName, stats := range catLoss {
					if statsMap, ok := stats.(map[string]interface{}); ok {
						if avgLoss, ok := statsMap["average_loss"].(float32); ok {
							// Alert if any category loss exceeds threshold
							if avgLoss > 2.0 {
								log.Printf("🚨 ALERT: Category '%s' loss too high: %.4f\n", catName, avgLoss)
								// Send notification, adjust hyperparameters, etc.
							}
						}
					}
				}
			}
		}
	}()
}

// Note: Import time package for the alert system
// import "time"

// ============================================================================
// Tips for Production Use
// ============================================================================

/*
1. INITIALIZATION
   - Always call trainer.InitializeObservability() at the start
   - Ensure vocabulary is fully loaded before initializing

2. RECORDING DURING TRAINING
   - Call trainer.RecordTrainingStep() for every training step
   - Record weight snapshots every N steps (not every step) to reduce overhead
   - Call trainer.UpdateWeightVelocity() after parameter updates

3. ENDPOINT EPOCH
   - Always call trainer.FinishEpoch(vocab) at the end of each epoch
   - This logs the report and resets windowed metrics

4. DASHBOARD ACCESS
   - Open docs/dashboard-enhanced.html in browser while training
   - Dashboard polls metrics endpoint every 1 second
   - No impact on training loop (non-blocking)

5. CONFIGURATION
   - Edit data/config/observability_config.json to customize categories
   - Adjust window_size based on your training speed
   - Reduce update frequency if CPU usage is high

6. TROUBLESHOOTING
   - Check /api/metrics/current to verify data collection
   - Look at console logs from trainer.FinishEpoch()
   - Ensure RecordTrainingStep() is called with correct parameters

7. PERFORMANCE
   - Observability adds ~1-2% overhead to training
   - Memory footprint is O(N_experts × N_vocab) for lexicon tracking
   - Use subset of experts/vocabulary if memory is constrained
*/
