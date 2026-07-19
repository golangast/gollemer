package moe

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"

	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// MetricsAggregator collects and serves observability metrics for the dashboard.
type MetricsAggregator struct {
	trainer          *Trainer
	vocab            *vocab.Vocabulary
	mu               sync.RWMutex
	metricsHistory   []map[string]interface{}
	maxHistoryPoints int
	lastUpdate       time.Time
	// testOverrides holds developer/testing overrides that should be merged
	// into the latest metrics returned to clients. This allows tests to inject
	// synthetic data without being immediately clobbered by the periodic
	// collector.
	testOverrides map[string]interface{}
}

// on-demand inference lock to avoid concurrent trace runs that clobber global state
var onDemandInferenceLock sync.Mutex

// NewMetricsAggregator creates a new metrics aggregator.
func NewMetricsAggregator(trainer *Trainer, vocab *vocab.Vocabulary) *MetricsAggregator {
	return &MetricsAggregator{
		trainer:          trainer,
		vocab:            vocab,
		metricsHistory:   make([]map[string]interface{}, 0),
		maxHistoryPoints: 200, // Keep last 200 metric samples
		lastUpdate:       time.Now(),
		testOverrides:    make(map[string]interface{}),
	}
}

// readGitCommit attempts to resolve current git commit SHA (short) from .git
func readGitCommit() string {
	headPath := ".git/HEAD"
	b, err := os.ReadFile(headPath)
	if err != nil {
		return ""
	}
	ref := strings.TrimSpace(string(b))
	if strings.HasPrefix(ref, "ref: ") {
		refPath := strings.TrimPrefix(ref, "ref: ")
		full := filepath.Join(".git", refPath)
		rb, err := os.ReadFile(full)
		if err != nil {
			return strings.TrimSpace(ref)
		}
		return strings.TrimSpace(string(rb))
	}
	return ref
}

// CollectMetrics snapshots current observability metrics.
func (m *MetricsAggregator) CollectMetrics() map[string]interface{} {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.trainer == nil || !m.trainer.ObservabilityEnabled || m.trainer.Observability == nil {
		return map[string]interface{}{
			"error": "Observability not initialized",
		}
	}

	metrics := m.trainer.Observability.GetDashboardMetrics(m.vocab)
	// Provide predicted_sentences to help the dashboard show what examples
	// the model can currently handle. We expose a structured array of
	// objects with sentence, avg_conf, timestamp and source fields. Prefer
	// persisted recent traces, then fall back to a simple heuristic combining
	// expert lexicon tokens.
	structured := make([]map[string]interface{}, 0)
	if ObservabilityInstance != nil {
		traces := ObservabilityInstance.GetTraceHistory()
		// Most recent first
		for i := len(traces) - 1; i >= 0 && len(structured) < 12; i-- {
			rec := traces[i]
			prompt, _ := rec["prompt"].(string)
			if strings.TrimSpace(prompt) == "" {
				continue
			}
			// compute avg_conf from trajectories if available
			avgConf := float32(0)
			if trjs, ok := rec["trajectories"].([]map[string]interface{}); ok && len(trjs) > 0 {
				var sum float32
				var count int
				for _, tj := range trjs {
					if v, ok := tj["avg_conf"].(float32); ok {
						sum += v
						count++
					} else if vf, ok := tj["avg_conf"].(float64); ok {
						sum += float32(vf)
						count++
					}
				}
				if count > 0 {
					avgConf = sum / float32(count)
				}
			} else if arr, ok := rec["trajectories"].([]interface{}); ok && len(arr) > 0 {
				var sum float32
				var count int
				for _, raw := range arr {
					if mobj, ok := raw.(map[string]interface{}); ok {
						if vf, ok := mobj["avg_conf"].(float64); ok {
							sum += float32(vf)
							count++
						} else if vf2, ok := mobj["avg_conf"].(float32); ok {
							sum += vf2
							count++
						}
					}
				}
				if count > 0 {
					avgConf = sum / float32(count)
				}
			}
			ts := int64(0)
			if tsv, ok := rec["timestamp"].(int64); ok {
				ts = tsv
			} else if tvf, ok := rec["timestamp"].(float64); ok {
				ts = int64(tvf)
			}
			structured = append(structured, map[string]interface{}{
				"sentence":  prompt,
				"avg_conf":  avgConf,
				"timestamp": ts,
				"source":    "trace",
			})
		}
	}
	if len(structured) == 0 {
		// Try expert lexicon heuristics
		if lex, ok := metrics["expert_lexicon"].(map[int][]string); ok {
			// build short example phrases by taking first tokens from different experts
			tops := make([]string, 0)
			for i := 0; i < 8; i++ {
				if toks, ok := lex[i]; ok && len(toks) > 0 {
					tops = append(tops, toks[0])
				}
			}
			if len(tops) > 0 {
				for i := 0; i < 8 && len(structured) < 12; i++ {
					parts := make([]string, 0)
					for j := 0; j < 4 && j < len(tops); j++ {
						parts = append(parts, tops[(i+j)%len(tops)])
					}
					structured = append(structured, map[string]interface{}{
						"sentence":  strings.Join(parts, " "),
						"avg_conf":  float32(0),
						"timestamp": time.Now().Unix(),
						"source":    "heuristic",
					})
				}
			}
		}
	}
	if len(structured) > 0 {
		// Provide both legacy array-of-strings and new structured array for compatibility
		preds := make([]string, 0, len(structured))
		for _, s := range structured {
			if txt, ok := s["sentence"].(string); ok {
				preds = append(preds, txt)
			}
		}
		metrics["predicted_sentences"] = preds
		metrics["predicted_sentences_structured"] = structured
	}
	metrics["timestamp"] = time.Now().Unix()
	metrics["time_str"] = time.Now().Format("15:04:05")
	metrics["health_indicators"] = m.buildHealthIndicatorsFromMetrics(metrics)

	// Keep history trimmed
	m.metricsHistory = append(m.metricsHistory, metrics)
	if len(m.metricsHistory) > m.maxHistoryPoints {
		m.metricsHistory = m.metricsHistory[1:]
	}

	m.lastUpdate = time.Now()
	return metrics
}

// GetCurrentMetrics returns the latest collected metrics.
func (m *MetricsAggregator) GetCurrentMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.metricsHistory) == 0 {
		return map[string]interface{}{
			"error": "No metrics collected yet",
		}
	}
	// Return a shallow copy of the latest metrics so callers cannot mutate internal state.
	latest := m.metricsHistory[len(m.metricsHistory)-1]
	out := make(map[string]interface{}, len(latest)+len(m.testOverrides))
	for k, v := range latest {
		out[k] = v
	}
	// Merge any test overrides (they take precedence)
	for k, v := range m.testOverrides {
		out[k] = v
	}
	return out
}

// GetMetricsHistory returns all collected metrics over time.
func (m *MetricsAggregator) GetMetricsHistory() []map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Return a deep copy to prevent external mutation
	history := make([]map[string]interface{}, len(m.metricsHistory))
	copy(history, m.metricsHistory)
	return history
}

// GetTimeSeriesMetrics returns metrics suitable for time-series charts.
func (m *MetricsAggregator) GetTimeSeriesMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Extract time series data from history
	timestamps := make([]string, len(m.metricsHistory))
	categoryLossTimeSeries := make(map[string][]float32)
	weightVelocityTimeSeries := make([]float32, len(m.metricsHistory))

	for i, metrics := range m.metricsHistory {
		if ts, ok := metrics["time_str"].(string); ok {
			timestamps[i] = ts
		}

		// Extract category loss
		if catLoss, ok := metrics["category_loss"].(map[string]interface{}); ok {
			for catName, stats := range catLoss {
				if statsMap, ok := stats.(map[string]interface{}); ok {
					if avgLoss, ok := statsMap["average_loss"].(float32); ok {
						if categoryLossTimeSeries[catName] == nil {
							categoryLossTimeSeries[catName] = make([]float32, len(m.metricsHistory))
						}
						categoryLossTimeSeries[catName][i] = avgLoss
					}
				}
			}
		}

		// Extract weight velocity
		if wv, ok := metrics["weight_velocity"].(map[string]interface{}); ok {
			if maxVel, ok := wv["max_velocity"].(float32); ok {
				weightVelocityTimeSeries[i] = maxVel
			}
		}
	}

	return map[string]interface{}{
		"timestamps":             timestamps,
		"category_loss_series":   categoryLossTimeSeries,
		"weight_velocity_series": weightVelocityTimeSeries,
		"data_points":            len(m.metricsHistory),
	}
}

// ServeHTTP serves metrics over HTTP in JSON format.
func (m *MetricsAggregator) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	path := strings.TrimPrefix(r.URL.Path, "/")
	switch path {
	case "metrics/predicted", "predicted":
		// Return structured predicted sentences (sentence, avg_conf, timestamp, source)
		cur := m.GetCurrentMetrics()
		if ps, ok := cur["predicted_sentences_structured"]; ok {
			json.NewEncoder(w).Encode(map[string]interface{}{"predicted": ps})
			return
		}
		// Fallback: collect fresh metrics to build the field
		snap := m.CollectMetrics()
		if ps, ok := snap["predicted_sentences_structured"]; ok {
			json.NewEncoder(w).Encode(map[string]interface{}{"predicted": ps})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"predicted": []map[string]interface{}{}})
		return

	case "metrics/current", "current":
		metrics := m.GetCurrentMetrics()
		json.NewEncoder(w).Encode(metrics)

	case "metrics/history", "history":
		history := m.GetMetricsHistory()
		json.NewEncoder(w).Encode(map[string]interface{}{
			"history": history,
			"count":   len(history),
		})

	case "metrics/timeseries", "timeseries":
		ts := m.GetTimeSeriesMetrics()
		json.NewEncoder(w).Encode(ts)

	case "metrics/snapshot", "snapshot":
		snapshot := m.CollectMetrics()
		json.NewEncoder(w).Encode(snapshot)

	case "metrics/layersnapshot", "layersnapshot":
		if m.trainer == nil || m.trainer.Observability == nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "observability not enabled"})
			return
		}
		json.NewEncoder(w).Encode(m.trainer.Observability.GetLayerRoutingSnapshot())

	case "metrics/inject", "inject":
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			json.NewEncoder(w).Encode(map[string]string{"error": "POST only"})
			return
		}
		var payload map[string][]string
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": "bad JSON"})
			return
		}
		if m.trainer == nil || m.trainer.Observability == nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": "observability not enabled"})
			return
		}
		for k, toks := range payload {
			eid, err := strconv.Atoi(k)
			if err != nil {
				continue
			}
			for _, t := range toks {
				tid := -1
				if m.vocab != nil {
					tid = m.vocab.GetTokenID(t)
					if tid == m.vocab.UnkID {
						tid = m.vocab.AddToken(t)
					}
				}
				m.trainer.Observability.ExpertLexicon.RecordTokenRoute(eid, tid)
			}
		}
		json.NewEncoder(w).Encode(map[string]bool{"ok": true})

	case "metrics/inject_metrics", "inject_metrics":
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			json.NewEncoder(w).Encode(map[string]string{"error": "POST only"})
			return
		}
		var payload map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": "bad JSON"})
			return
		}
		// Ensure we have a latest metrics snapshot to modify and store overrides
		_latest := m.CollectMetrics()
		m.mu.Lock()
		if len(m.metricsHistory) == 0 {
			m.metricsHistory = append(m.metricsHistory, _latest)
		}
		lastIdx := len(m.metricsHistory) - 1
		for k, v := range payload {
			// Only allow a small set of test keys
			switch k {
			case "weight_velocity", "category_loss", "expert_lexicon":
				// Apply to the latest history entry for immediate effect
				m.metricsHistory[lastIdx][k] = v
				// Also persist into testOverrides so future GetCurrentMetrics merges it
				m.testOverrides[k] = v
			default:
				// ignore other keys
			}
		}
		m.mu.Unlock()
		json.NewEncoder(w).Encode(map[string]bool{"ok": true})

	case "metrics/pca", "pca":
		if m.trainer == nil || m.trainer.Observability == nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "observability not enabled"})
			return
		}
		json.NewEncoder(w).Encode(m.trainer.Observability.EmbeddingProjection.GetProjectionCoordinates())

	case "metrics/similarity", "similarity":
		if m.trainer == nil || m.trainer.Observability == nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "observability not enabled"})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"matrix":   m.trainer.Observability.ExpertSimilarity.GetSimilarityMatrix(),
			"warnings": m.trainer.Observability.ExpertSimilarity.GetRedundancyWarnings(0.85),
		})

	case "metrics/trajectories", "trajectories":
		// Prefer assembling trajectories from ObservabilityInstance.LayerSelections (includes token IDs)
		if ObservabilityInstance == nil || len(ObservabilityInstance.LayerSelections) == 0 {
			json.NewEncoder(w).Encode(map[string]interface{}{"trajectories": []map[string]interface{}{}})
			return
		}

		// Determine number of layers and tokens from stored selections
		numLayers := 0
		for li := range ObservabilityInstance.LayerSelections {
			if li+1 > numLayers {
				numLayers = li + 1
			}
		}

		// Use first available layer selection to determine token count
		var firstSel *LayerSelection
		for _, sel := range ObservabilityInstance.LayerSelections {
			firstSel = sel
			break
		}
		if firstSel == nil {
			json.NewEncoder(w).Encode(map[string]interface{}{"trajectories": []map[string]interface{}{}})
			return
		}

		numTokens := len(firstSel.Selected)

		// Assemble trajectories per token
		trajectories := make([]map[string]interface{}, 0, numTokens)
		for t := 0; t < numTokens; t++ {
			path := make([]int, numLayers)
			confs := make([]float32, numLayers)
			tokenID := -1
			for li := 0; li < numLayers; li++ {
				if sel, ok := ObservabilityInstance.LayerSelections[li]; ok {
					if t < len(sel.Selected) && len(sel.Selected[t]) > 0 {
						path[li] = sel.Selected[t][0]
						if t < len(sel.Confidences) && len(sel.Confidences[t]) > 0 {
							confs[li] = sel.Confidences[t][0]
						}
					} else {
						path[li] = -1
						confs[li] = 0
					}
					if tokenID == -1 && t < len(sel.TokenIDs) {
						tokenID = sel.TokenIDs[t]
					}
				} else {
					path[li] = -1
					confs[li] = 0
				}
			}
			word := ""
			if tokenID >= 0 {
				if m.vocab != nil {
					word = m.vocab.GetWord(tokenID)
				}
				if word == "" && ObservabilityInstance != nil && ObservabilityInstance.SemanticDrift != nil && ObservabilityInstance.SemanticDrift.vocab != nil {
					word = ObservabilityInstance.SemanticDrift.vocab.GetWord(tokenID)
				}
			}
			trajectories = append(trajectories, map[string]interface{}{
				"token_index": t,
				"token_id":    tokenID,
				"token":       word,
				"layer_path":  path,
				"confidences": confs,
				"avg_conf":    avgFloat32(confs),
			})
		}

		json.NewEncoder(w).Encode(map[string]interface{}{"trajectories": trajectories})

	case "metrics/trace", "trace":
		// On-demand trace: run a prompt through the model (via hook) and return fresh LayerSelections
		if OnDemandInference == nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "OnDemandInference not configured"})
			return
		}

		// Parse body for prompt and max_len (allow GET fallback via query)
		var payload struct {
			Prompt string `json:"prompt"`
			MaxLen int    `json:"max_len"`
		}
		if r.Method == "POST" {
			_ = json.NewDecoder(r.Body).Decode(&payload)
		} else {
			payload.Prompt = r.URL.Query().Get("prompt")
			if ml := r.URL.Query().Get("max_len"); ml != "" {
				fmt.Sscanf(ml, "%d", &payload.MaxLen)
			}
		}

		if payload.Prompt == "" {
			json.NewEncoder(w).Encode(map[string]string{"error": "prompt required"})
			return
		}

		promptTokens := strings.Fields(payload.Prompt)

		onDemandInferenceLock.Lock()
		defer onDemandInferenceLock.Unlock()

		if ObservabilityInstance != nil {
			ObservabilityInstance.ClearLayerSelections()
		}

		traceID := fmt.Sprintf("trace-%d", time.Now().UnixNano())
		start := time.Now()
		if ObservabilityInstance != nil {
			ObservabilityInstance.SetCurrentTraceID(traceID)
		}
		res, err := OnDemandInference(payload.Prompt, payload.MaxLen)
		for _, entry := range res {
			if entry == nil {
				continue
			}
			if rawTokens, ok := entry["tokens"]; ok {
				switch tokens := rawTokens.(type) {
				case []int:
					textTokens := make([]interface{}, len(tokens))
					for i, id := range tokens {
						text := ""
						if id >= 0 {
							if m.vocab != nil {
								text = m.vocab.GetWord(id)
							}
							if text == "" && ObservabilityInstance != nil && ObservabilityInstance.SemanticDrift != nil && ObservabilityInstance.SemanticDrift.vocab != nil {
								text = ObservabilityInstance.SemanticDrift.vocab.GetWord(id)
							}
							if text == "" || text == "UNK" || text == "<unk>" || text == "<pad>" || text == "<s>" || text == "</s>" {
								if i < len(promptTokens) {
									text = promptTokens[i]
								}
							}
							if text == "" {
								text = fmt.Sprintf("ID:%d", id)
							}
						}
						textTokens[i] = text
					}
					entry["tokens"] = textTokens
				case []interface{}:
					textTokens := make([]interface{}, len(tokens))
					for i, token := range tokens {
						switch v := token.(type) {
						case string:
							textTokens[i] = v
						case int:
							text := ""
							if v >= 0 {
								if m.vocab != nil {
									text = m.vocab.GetWord(v)
								}
								if text == "" && ObservabilityInstance != nil && ObservabilityInstance.SemanticDrift != nil && ObservabilityInstance.SemanticDrift.vocab != nil {
									text = ObservabilityInstance.SemanticDrift.vocab.GetWord(v)
								}
								if text == "" || text == "UNK" || text == "<unk>" || text == "<pad>" || text == "<s>" || text == "</s>" {
									if i < len(promptTokens) {
										text = promptTokens[i]
									}
								}
								if text == "" {
									text = fmt.Sprintf("ID:%d", v)
								}
							}
							textTokens[i] = text
						case float64:
							text := ""
							if int(v) >= 0 {
								if m.vocab != nil {
									text = m.vocab.GetWord(int(v))
								}
								if text == "" && ObservabilityInstance != nil && ObservabilityInstance.SemanticDrift != nil && ObservabilityInstance.SemanticDrift.vocab != nil {
									text = ObservabilityInstance.SemanticDrift.vocab.GetWord(int(v))
								}
								if text == "" || text == "UNK" || text == "<unk>" || text == "<pad>" || text == "<s>" || text == "</s>" {
									if i < len(promptTokens) {
										text = promptTokens[i]
									}
								}
								if text == "" {
									text = fmt.Sprintf("ID:%d", int(v))
								}
							}
							textTokens[i] = text
						default:
							textTokens[i] = token
						}
					}
					entry["tokens"] = textTokens
				}
			}
		}
		dur := time.Since(start)
		if ObservabilityInstance != nil {
			ObservabilityInstance.ClearCurrentTraceID()
		}
		if err != nil {
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}

		if ObservabilityInstance == nil || len(ObservabilityInstance.LayerSelections) == 0 {
			json.NewEncoder(w).Encode(map[string]interface{}{"trace": res, "trajectories": []map[string]interface{}{}})
			return
		}

		numLayers := 0
		for li := range ObservabilityInstance.LayerSelections {
			if li+1 > numLayers {
				numLayers = li + 1
			}
		}

		var firstSel *LayerSelection
		for _, sel := range ObservabilityInstance.LayerSelections {
			firstSel = sel
			break
		}
		if firstSel == nil {
			json.NewEncoder(w).Encode(map[string]interface{}{"trace": res, "trajectories": []map[string]interface{}{}})
			return
		}

		numTokens := len(firstSel.Selected)
		trajectories := make([]map[string]interface{}, 0, numTokens)
		for t := 0; t < numTokens; t++ {
			path := make([]int, numLayers)
			confs := make([]float32, numLayers)
			tokenID := -1
			for li := 0; li < numLayers; li++ {
				if sel, ok := ObservabilityInstance.LayerSelections[li]; ok {
					if t < len(sel.Selected) && len(sel.Selected[t]) > 0 {
						path[li] = sel.Selected[t][0]
						if t < len(sel.Confidences) && len(sel.Confidences[t]) > 0 {
							confs[li] = sel.Confidences[t][0]
						}
					} else {
						path[li] = -1
						confs[li] = 0
					}
					if tokenID == -1 && t < len(sel.TokenIDs) {
						tokenID = sel.TokenIDs[t]
					}
				} else {
					path[li] = -1
					confs[li] = 0
				}
			}

			word := ""
			if tokenID >= 0 {
				if m.vocab != nil {
					word = m.vocab.GetWord(tokenID)
				}
				if word == "" && ObservabilityInstance != nil && ObservabilityInstance.SemanticDrift != nil && ObservabilityInstance.SemanticDrift.vocab != nil {
					word = ObservabilityInstance.SemanticDrift.vocab.GetWord(tokenID)
				}
				if word == "" || word == "UNK" || word == "<unk>" || word == "<pad>" || word == "<s>" || word == "</s>" {
					if t < len(promptTokens) {
						word = promptTokens[t]
					}
				}
			}

			trajectories = append(trajectories, map[string]interface{}{
				"token_index": t,
				"token_id":    tokenID,
				"token":       word,
				"layer_path":  path,
				"confidences": confs,
				"avg_conf":    avgFloat32(confs),
			})
		}

		if ObservabilityInstance != nil {
			record := map[string]interface{}{
				"id":           traceID,
				"prompt":       payload.Prompt,
				"timestamp":    time.Now().Unix(),
				"duration_ms":  dur.Milliseconds(),
				"trace":        res,
				"trajectories": trajectories,
			}
			if m.trainer != nil {
				record["model"] = m.trainer.BestModelPath
			}
			lat := ObservabilityInstance.GetTempLayerLatencies()
			if len(lat) > 0 {
				record["layer_latencies_ms"] = lat
			}
			comp := ObservabilityInstance.GetTempLayerComponentLatencies()
			if len(comp) > 0 {
				record["layer_component_latencies_ms"] = comp
			}
			ObservabilityInstance.AppendTrace(record)
			ObservabilityInstance.ClearTempLayerLatencies()
		}

		json.NewEncoder(w).Encode(map[string]interface{}{"trace": res, "trajectories": trajectories})

	case "metrics/trace_history", "trace_history":
		if ObservabilityInstance == nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "observability not enabled"})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"traces": ObservabilityInstance.GetTraceHistory()})

	case "metrics/trace_history/delete", "trace_history/delete":
		if ObservabilityInstance == nil {
			json.NewEncoder(w).Encode(map[string]string{"error": "observability not enabled"})
			return
		}
		id := r.URL.Query().Get("id")
		if id == "" {
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]string{"error": "missing id parameter"})
			return
		}
		removed := ObservabilityInstance.DeleteTraceByID(id)
		if !removed {
			w.WriteHeader(http.StatusNotFound)
			json.NewEncoder(w).Encode(map[string]string{"error": "trace not found"})
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "id": id})

	default:
		w.WriteHeader(http.StatusNotFound)
		json.NewEncoder(w).Encode(map[string]string{"error": "Unknown endpoint"})
	}
}

// Expose trace history endpoint: /metrics/trace_history
// Add handler in ServeHTTP before default

// StartMetricsServer starts an HTTP server serving observability metrics.
func StartMetricsServer(trainer *Trainer, vocab *vocab.Vocabulary, addr string) *MetricsAggregator {
	aggregator := NewMetricsAggregator(trainer, vocab)

	// Start periodic collection
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond) // Collect every 500ms
		defer ticker.Stop()

		for range ticker.C {
			aggregator.CollectMetrics()

			// Additionally, if observability is enabled, sample expert weights from all active MoE layers
			if trainer != nil && trainer.ObservabilityEnabled && len(ActiveLayers) > 0 {
				// Use composite keys to avoid collisions: key = layerIndex*10000 + expertIndex
				for lidx, layer := range ActiveLayers {
					numExperts := len(layer.Experts)
					for eid := 0; eid < numExperts; eid++ {
						params := layer.Experts[eid].Parameters()
						// Flatten representative parameters
						flat := make([]float32, 0)
						for _, p := range params {
							if p != nil {
								flat = append(flat, p.Data...)
							}
						}
						compositeID := lidx*10000 + eid
						trainer.RecordExpertForSimilarity(compositeID, flat)
					}
				}
				// Trigger recompute of similarity matrix from collected expert weights
				trainer.ComputeExpertRedundancy(0)
			}
		}
	}()

	// Setup HTTP routes
	http.Handle("/api/metrics/", http.StripPrefix("/api/metrics", aggregator))

	// Start server
	go func() {
		log.Printf("📊 Starting Metrics Server on %s\n", addr)
		if err := http.ListenAndServe(addr, nil); err != nil {
			log.Printf("⚠️  Metrics server error: %v\n", err)
		}
	}()

	return aggregator
}

// ==============================================================================
// Dashboard Payload Builder
// ==============================================================================

// DashboardPayload structures all observability data for frontend visualization.
type DashboardPayload struct {
	Epoch            int                               `json:"epoch"`
	Timestamp        int64                             `json:"timestamp"`
	ExpertLexicon    map[int][]string                  `json:"expert_lexicon"`
	CategoryLoss     map[string]map[string]interface{} `json:"category_loss"`
	WeightVelocity   map[string]interface{}            `json:"weight_velocity"`
	SemanticDrift    []map[string]interface{}          `json:"semantic_drift"`
	TimeSeries       map[string]interface{}            `json:"time_series"`
	HealthIndicators map[string]interface{}            `json:"health_indicators"`
}

// BuildDashboardPayload constructs the complete dashboard payload.
func (m *MetricsAggregator) BuildDashboardPayload() DashboardPayload {
	m.mu.RLock()
	defer m.mu.RUnlock()

	current := m.GetCurrentMetrics()
	timeSeries := m.GetTimeSeriesMetrics()

	payload := DashboardPayload{
		Timestamp: time.Now().Unix(),
	}

	// Extract fields with type assertions
	if epoch, ok := current["epoch"].(int); ok {
		payload.Epoch = epoch
	}

	if lexicon, ok := current["expert_lexicon"].(map[int][]string); ok {
		payload.ExpertLexicon = lexicon
	}

	if catLoss, ok := current["category_loss"].(map[string]map[string]interface{}); ok {
		payload.CategoryLoss = catLoss
	}

	if wv, ok := current["weight_velocity"].(map[string]interface{}); ok {
		payload.WeightVelocity = wv
	}

	if drift, ok := current["semantic_drift"].([]map[string]interface{}); ok {
		payload.SemanticDrift = drift
	}

	payload.TimeSeries = timeSeries

	// Build health indicators
	payload.HealthIndicators = m.buildHealthIndicatorsFromMetrics(current)

	return payload
}

// buildHealthIndicators builds aggregated health metrics.
func (m *MetricsAggregator) buildHealthIndicators() map[string]interface{} {
	return m.buildHealthIndicatorsFromMetrics(m.GetCurrentMetrics())
}

func (m *MetricsAggregator) buildHealthIndicatorsFromMetrics(current map[string]interface{}) map[string]interface{} {
	indicators := make(map[string]interface{})

	// Expert balance health
	if lexicon, ok := current["expert_lexicon"].(map[int][]string); ok {
		indicators["expert_count"] = len(lexicon)
		usedExperts := 0
		for _, tokens := range lexicon {
			if len(tokens) > 0 {
				usedExperts++
			}
		}
		indicators["active_experts"] = usedExperts
		indicators["expert_utilization"] = float32(usedExperts) / float32(len(lexicon))
	}

	// Token category health
	if catLoss, ok := current["category_loss"].(map[string]map[string]interface{}); ok {
		var bestCategory string
		var bestImprovement float32
		for catName, stats := range catLoss {
			if improvement, ok := stats["improvement"].(float32); ok {
				if improvement > bestImprovement {
					bestImprovement = improvement
					bestCategory = catName
				}
			}
		}
		if bestCategory != "" {
			indicators["best_improving_category"] = bestCategory
			indicators["best_improvement"] = bestImprovement
		}
	}

	// Weight velocity health
	if wv, ok := current["weight_velocity"].(map[string]interface{}); ok {
		if maxVel, ok := wv["max_velocity"].(float32); ok {
			indicators["max_weight_velocity"] = maxVel
			if maxVel > 0.1 {
				indicators["learning_intensity"] = "HIGH"
			} else if maxVel > 0.01 {
				indicators["learning_intensity"] = "MEDIUM"
			} else {
				indicators["learning_intensity"] = "LOW"
			}
		}
	}

	// Semantic drift health
	if drift, ok := current["semantic_drift"].([]map[string]interface{}); ok {
		indicators["semantic_drift_count"] = len(drift)
		if len(drift) > 0 {
			indicators["max_semantic_drift"] = drift[0]["drift"]
		}
	}

	return indicators
}

// ExportDashboardPayloadJSON serializes the dashboard payload to JSON.
func (m *MetricsAggregator) ExportDashboardPayloadJSON() (string, error) {
	payload := m.BuildDashboardPayload()
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// LogHealthReport logs a comprehensive health report.
func (m *MetricsAggregator) LogHealthReport() {
	current := m.GetCurrentMetrics()

	fmt.Printf("╔════════════════════════════════════════════════╗\n")
	fmt.Printf("║          OBSERVABILITY HEALTH REPORT          ║\n")
	fmt.Printf("╚════════════════════════════════════════════════╝\n\n")

	if epoch, ok := current["epoch"].(int); ok {
		fmt.Printf("📊 Epoch: %d\n", epoch)
	}

	// Expert Lexicon report
	fmt.Println("\n📚 Expert Lexicon:")
	if lexicon, ok := current["expert_lexicon"].(map[int][]string); ok {
		for expertID := 0; expertID < len(lexicon); expertID++ {
			if tokens, ok := lexicon[expertID]; ok && len(tokens) > 0 {
				fmt.Printf("   Expert %d: %v\n", expertID, tokens[:min(3, len(tokens))])
			}
		}
	}

	// Category Loss report
	fmt.Println("\n📈 Category Loss Analysis:")
	if catLoss, ok := current["category_loss"].(map[string]map[string]interface{}); ok {
		for catName, stats := range catLoss {
			if avgLoss, ok := stats["average_loss"].(float32); ok {
				if improvement, ok := stats["improvement"].(float32); ok {
					status := "📉"
					if improvement < 0 {
						status = "📈"
					}
					fmt.Printf("   %s %s: Loss=%.4f, Δ=%.4f\n", status, catName, avgLoss, improvement)
				}
			}
		}
	}

	// Weight Velocity report
	fmt.Println("\n🔥 Weight Velocity Hotspots:")
	if wv, ok := current["weight_velocity"].(map[string]interface{}); ok {
		if maxVel, ok := wv["max_velocity"].(float32); ok {
			intensity := "🔵"
			if maxVel > 0.1 {
				intensity = "🔴"
			} else if maxVel > 0.01 {
				intensity = "🟡"
			}
			fmt.Printf("   %s Max Velocity: %.6f\n", intensity, maxVel)
		}
	}

	// Semantic Drift report
	fmt.Println("\n🔄 Semantic Drift Tracking:")
	if drift, ok := current["semantic_drift"].([]map[string]interface{}); ok {
		for i, d := range drift {
			if i >= 3 {
				break
			}
			if token, ok := d["token"].(string); ok {
				if driftVal, ok := d["drift"].(float32); ok {
					fmt.Printf("   Token \"%s\" drifted by %.4f\n", token, driftVal)
				}
			}
		}
	}

	fmt.Println()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
