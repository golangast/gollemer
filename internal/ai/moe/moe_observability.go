package moe

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"

	"gonum.org/v1/gonum/mat"
)

// ObservabilityInstance is a global reference set when a Trainer enables observability.
var ObservabilityInstance *MoEObservability

// OnDemandInference is an optional hook that runs a single inference for tracing.
// Signature: prompt, maxLen -> optional metadata or error. The function should set
// `ObservabilityInstance.LayerSelections` during the run so the trace endpoint can
// assemble trajectories. If nil, trace endpoint returns an error.
var OnDemandInference func(prompt string, maxLen int) ([]map[string]interface{}, error)

// LayerSelection stores per-layer selected experts and confidences along with token IDs.
type LayerSelection struct {
	TokenIDs    []int
	Selected    [][]int
	Confidences [][]float32
}

// ==============================================================================
// FEATURE 1: Expert Lexicon (Top In-Domain Tokens)
// ==============================================================================

// TokenRoutingHistogram tracks which token IDs route to which experts over N steps.
type TokenRoutingHistogram struct {
	// expertToTokens[expertID] = map[tokenID]count
	expertToTokens map[int]map[int]int
	mu             sync.Mutex
	windowSize     int
	currentStep    int
}

// NewTokenRoutingHistogram creates a histogram tracking tokens per expert over a window.
func NewTokenRoutingHistogram(numExperts, windowSize int) *TokenRoutingHistogram {
	h := &TokenRoutingHistogram{
		expertToTokens: make(map[int]map[int]int),
		windowSize:     windowSize,
		currentStep:    0,
	}
	for i := 0; i < numExperts; i++ {
		h.expertToTokens[i] = make(map[int]int)
	}
	return h
}

// RecordTokenRoute records that a token routed to an expert.
func (h *TokenRoutingHistogram) RecordTokenRoute(expertID, tokenID int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if expertID >= 0 && expertID < len(h.expertToTokens) {
		h.expertToTokens[expertID][tokenID]++
	}
	h.currentStep++
}

// RecordBatchTokenRoutes records multiple token-to-expert routings.
// If the batch contains more tokens than expert assignments, the expert list is cycled
// across the sequence so the dashboard still receives meaningful routing activity.
func (h *TokenRoutingHistogram) RecordBatchTokenRoutes(expertIDs, tokenIDs []int) {
	if len(expertIDs) == 0 || len(tokenIDs) == 0 {
		return
	}
	if len(expertIDs) == len(tokenIDs) {
		for i := range expertIDs {
			h.RecordTokenRoute(expertIDs[i], tokenIDs[i])
		}
		return
	}
	for i, tokenID := range tokenIDs {
		h.RecordTokenRoute(expertIDs[i%len(expertIDs)], tokenID)
	}
}

// Reset clears the histogram for the next window.
func (h *TokenRoutingHistogram) Reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	for i := range h.expertToTokens {
		h.expertToTokens[i] = make(map[int]int)
	}
	h.currentStep = 0
}

// GetTopKTokensPerExpert returns the top K tokens for each expert, decoded to strings.
func (h *TokenRoutingHistogram) GetTopKTokensPerExpert(k int, vocab *vocab.Vocabulary) map[int][]string {
	h.mu.Lock()
	defer h.mu.Unlock()

	result := make(map[int][]string)
	for expertID, tokenCounts := range h.expertToTokens {
		if len(tokenCounts) == 0 {
			result[expertID] = []string{}
			continue
		}

		// Sort by count descending
		type tokenCount struct {
			tokenID int
			count   int
		}
		var sorted []tokenCount
		for tokenID, count := range tokenCounts {
			sorted = append(sorted, tokenCount{tokenID, count})
		}
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i].count > sorted[j].count
		})

		// Take top K and decode
		for i := 0; i < k && i < len(sorted); i++ {
			tokenID := sorted[i].tokenID
			word := ""
			if vocab != nil {
				word = vocab.GetWord(tokenID)
			}
			// Fallback: try semantic drift vocab if available
			if word == "" && ObservabilityInstance != nil && ObservabilityInstance.SemanticDrift != nil && ObservabilityInstance.SemanticDrift.vocab != nil {
				word = ObservabilityInstance.SemanticDrift.vocab.GetWord(tokenID)
			}
			if word == "" || word == "UNK" || word == "<unk>" {
				word = fmt.Sprintf("ID:%d", tokenID)
			}
			result[expertID] = append(result[expertID], word)
		}
	}
	return result
}

// ==============================================================================
// FEATURE 2: Live Loss Delta per Token Category
// ==============================================================================

// TokenCategory represents a semantic grouping of tokens.
type TokenCategory struct {
	Name     string
	TokenIDs []int
	Color    string
}

// LoadTokenCategories loads token categories from config (or hardcoded defaults).
func LoadTokenCategories(vocab *vocab.Vocabulary) []TokenCategory {
	// Hardcoded defaults. In production, load from JSON config.
	categories := []TokenCategory{
		{
			Name:  "Structural Words",
			Color: "#6366f1",
		},
		{
			Name:  "Core Verbs",
			Color: "#ec4899",
		},
		{
			Name:  "Technical Terms",
			Color: "#f59e0b",
		},
		{
			Name:  "Punctuation",
			Color: "#8b5cf6",
		},
	}

	// Map common words to categories (simplified)
	structuralWords := []string{"the", "a", "an", "is", "are", "to", "and", "or", "of", "in", "at", "on"}
	coreVerbs := []string{"make", "go", "run", "do", "get", "set", "see", "come", "go", "use", "help", "turn"}
	technicalTerms := []string{"file", "directory", "command", "script", "code", "function", "class", "module"}
	punctuation := []string{".", ",", "!", "?", ":", ";", "-", "(", ")", "[", "]"}

	for _, word := range structuralWords {
		id := vocab.GetTokenID(word)
		if id == -1 {
			// Ensure common category tokens exist in the vocabulary for observability
			id = vocab.AddToken(word)
		}
		if id != -1 {
			categories[0].TokenIDs = append(categories[0].TokenIDs, id)
		}
	}
	for _, word := range coreVerbs {
		id := vocab.GetTokenID(word)
		if id == -1 {
			id = vocab.AddToken(word)
		}
		if id != -1 {
			categories[1].TokenIDs = append(categories[1].TokenIDs, id)
		}
	}
	for _, word := range technicalTerms {
		id := vocab.GetTokenID(word)
		if id == -1 {
			id = vocab.AddToken(word)
		}
		if id != -1 {
			categories[2].TokenIDs = append(categories[2].TokenIDs, id)
		}
	}
	for _, word := range punctuation {
		id := vocab.GetTokenID(word)
		if id == -1 {
			id = vocab.AddToken(word)
		}
		if id != -1 {
			categories[3].TokenIDs = append(categories[3].TokenIDs, id)
		}
	}

	return categories
}

// TokenCategoryLossTracker tracks loss improvements per token category.
type TokenCategoryLossTracker struct {
	categories     []TokenCategory
	categoryLosses map[string]*CategoryLossStats
	mu             sync.Mutex
}

// CategoryLossStats tracks loss metrics for a category.
type CategoryLossStats struct {
	TotalLoss     float32
	Count         int
	PreviousDelta float32
	CurrentDelta  float32
	Improvement   float32
}

// NewTokenCategoryLossTracker creates a loss tracker for token categories.
func NewTokenCategoryLossTracker(categories []TokenCategory) *TokenCategoryLossTracker {
	tracker := &TokenCategoryLossTracker{
		categories:     categories,
		categoryLosses: make(map[string]*CategoryLossStats),
	}
	for _, cat := range categories {
		tracker.categoryLosses[cat.Name] = &CategoryLossStats{}
	}
	return tracker
}

// RecordLossForTokens records the loss contribution for specific token IDs.
func (t *TokenCategoryLossTracker) RecordLossForTokens(loss float32, tokenIDs []int) {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Map token to categories
	matchedAny := false
	for _, tokenID := range tokenIDs {
		for _, cat := range t.categories {
			for _, catTokenID := range cat.TokenIDs {
				if catTokenID == tokenID {
					stats := t.categoryLosses[cat.Name]
					stats.TotalLoss += loss
					stats.Count++
					matchedAny = true
					break
				}
			}
		}
	}
	if !matchedAny && len(tokenIDs) > 0 {
		// log.Printf(" [DEBUG] No tokens matched categories. Sample token: %d", tokenIDs[0])
	}
}

// GetCategoryLossMetrics returns loss stats per category.
func (t *TokenCategoryLossTracker) GetCategoryLossMetrics() map[string]map[string]interface{} {
	t.mu.Lock()
	defer t.mu.Unlock()

	result := make(map[string]map[string]interface{})
	for catName, stats := range t.categoryLosses {
		avgLoss := float32(0)
		if stats.Count > 0 {
			avgLoss = stats.TotalLoss / float32(stats.Count)
		}
		result[catName] = map[string]interface{}{
			"average_loss":     avgLoss,
			"sample_count":     stats.Count,
			"total_loss":       stats.TotalLoss,
			"improvement":      stats.Improvement,
			"improvement_rate": float32(math.Abs(float64(stats.Improvement)) / float64(math.Max(float64(stats.PreviousDelta), 1e-6))),
		}
	}
	return result
}

// UpdateImprovement calculates loss improvement between epochs.
func (t *TokenCategoryLossTracker) UpdateImprovement() {
	t.mu.Lock()
	defer t.mu.Unlock()

	for _, stats := range t.categoryLosses {
		avgLoss := float32(0)
		if stats.Count > 0 {
			avgLoss = stats.TotalLoss / float32(stats.Count)
		}
		stats.PreviousDelta = stats.CurrentDelta
		stats.CurrentDelta = avgLoss
		stats.Improvement = stats.PreviousDelta - stats.CurrentDelta
		stats.TotalLoss = 0
		stats.Count = 0
	}
}

// ==============================================================================
// FEATURE 3: Weight Velocity (Frobenius Norm Heatmap)
// ==============================================================================

// WeightVelocityTracker tracks weight changes per layer/expert to detect learning hotspots.
type WeightVelocityTracker struct {
	previousWeights map[string][]float32
	velocities      map[string]float32
	mu              sync.Mutex
}

// NewWeightVelocityTracker creates a velocity tracker.
func NewWeightVelocityTracker() *WeightVelocityTracker {
	return &WeightVelocityTracker{
		previousWeights: make(map[string][]float32),
		velocities:      make(map[string]float32),
	}
}

// RecordWeightSnapshot captures current weight state for delta calculation.
func (w *WeightVelocityTracker) RecordWeightSnapshot(layerName string, weights []float32) {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Copy weights for next iteration
	snapshot := make([]float32, len(weights))
	copy(snapshot, weights)
	w.previousWeights[layerName] = snapshot
}

// UpdateWeightVelocity calculates velocity (Frobenius norm of weight delta).
func (w *WeightVelocityTracker) UpdateWeightVelocity(layerName string, currentWeights []float32) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if prev, ok := w.previousWeights[layerName]; ok {
		if len(prev) != len(currentWeights) {
			return
		}

		// Calculate Frobenius norm: sqrt(sum(dW_ij^2))
		var sumSq float32
		for i := range currentWeights {
			delta := currentWeights[i] - prev[i]
			sumSq += delta * delta
		}
		velocity := float32(math.Sqrt(float64(sumSq)))
		w.velocities[layerName] = velocity
	}
}

// GetVelocityHeatmap returns the velocity grid for visualization.
func (w *WeightVelocityTracker) GetVelocityHeatmap() map[string]interface{} {
	w.mu.Lock()
	defer w.mu.Unlock()

	// Normalize velocities to [0, 1] for heatmap coloring
	var maxVel float32
	for _, vel := range w.velocities {
		if vel > maxVel {
			maxVel = vel
		}
	}
	if maxVel == 0 {
		maxVel = 1
	}

	normalized := make(map[string]float32)
	for name, vel := range w.velocities {
		normalized[name] = vel / maxVel
	}

	return map[string]interface{}{
		"velocities":   w.velocities,
		"normalized":   normalized,
		"max_velocity": maxVel,
	}
}

// ==============================================================================
// FEATURE 4: Semantic Drift Tracker
// ==============================================================================

// SemanticDriftTracker tracks embedding space shifts during training.
type SemanticDriftTracker struct {
	baselineEmbeddings map[int][]float32
	currentEmbeddings  map[int][]float32
	w2vModel           *word2vec.SimpleWord2Vec
	vocab              *vocab.Vocabulary
	mu                 sync.Mutex
}

// NewSemanticDriftTracker creates a drift tracker with Word2Vec baseline.
func NewSemanticDriftTracker(w2vModel *word2vec.SimpleWord2Vec, vocab *vocab.Vocabulary) *SemanticDriftTracker {
	tracker := &SemanticDriftTracker{
		baselineEmbeddings: make(map[int][]float32),
		currentEmbeddings:  make(map[int][]float32),
		w2vModel:           w2vModel,
		vocab:              vocab,
	}

	// Initialize baseline from Word2Vec
	if w2vModel != nil && w2vModel.WordVectorsF32 != nil {
		for tokenID, vec := range w2vModel.WordVectorsF32 {
			baseline := make([]float32, len(vec))
			copy(baseline, vec)
			tracker.baselineEmbeddings[tokenID] = baseline
		}
	}

	return tracker
}

// RecordEmbeddingState captures current embedding vectors for specific tokens.
func (s *SemanticDriftTracker) RecordEmbeddingState(embeddingTensor *tensor.Tensor) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if embeddingTensor == nil {
		return
	}

	// Flatten embeddings for analysis (simplified approach)
	// Assumes embeddingTensor contains [vocabSize, embeddingDim] shape
	data := embeddingTensor.Data
	s.currentEmbeddings = make(map[int][]float32)

	// This is a simplified version; adapt to your actual tensor layout
	numTokens := len(data) / 64 // Assuming 64-dim embeddings (adjust as needed)
	for i := 0; i < numTokens && i*64 < len(data); i++ {
		embedding := make([]float32, 64)
		copy(embedding, data[i*64:(i+1)*64])
		s.currentEmbeddings[i] = embedding
	}

	if len(s.baselineEmbeddings) == 0 && len(s.currentEmbeddings) > 0 {
		s.baselineEmbeddings = make(map[int][]float32, len(s.currentEmbeddings))
		for tokenID, vec := range s.currentEmbeddings {
			baseline := make([]float32, len(vec))
			copy(baseline, vec)
			s.baselineEmbeddings[tokenID] = baseline
		}
	}
}

// cosineSimilarityInternal calculates cosine distance between two vectors.
// Note: Use the moe_layer.CosineSimilarity for consistency
func cosineSimilarityInternal(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProd, normA, normB float32
	for i := range a {
		dotProd += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = float32(math.Sqrt(float64(normA)))
	normB = float32(math.Sqrt(float64(normB)))

	if normA == 0 || normB == 0 {
		return 0
	}
	return dotProd / (normA * normB)
}

// GetTopSemanticShifts returns the top N tokens with largest embedding shifts.
func (s *SemanticDriftTracker) GetTopSemanticShifts(topN int) []map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()

	type drift struct {
		tokenID  int
		distance float32
		word     string
	}

	var drifts []drift
	for tokenID, currentVec := range s.currentEmbeddings {
		if baseVec, ok := s.baselineEmbeddings[tokenID]; ok {
			// Use cosine distance (1 - similarity) as drift metric
			similarity := cosineSimilarityInternal(currentVec, baseVec)
			distance := 1 - similarity
			word := s.vocab.GetWord(tokenID)
			if word == "" {
				word = fmt.Sprintf("[%d]", tokenID)
			}
			drifts = append(drifts, drift{tokenID, distance, word})
		}
	}

	// Sort by distance descending
	sort.Slice(drifts, func(i, j int) bool {
		return drifts[i].distance > drifts[j].distance
	})

	// Return top N
	result := make([]map[string]interface{}, 0)
	for i := 0; i < topN && i < len(drifts); i++ {
		result = append(result, map[string]interface{}{
			"token":    drifts[i].word,
			"drift":    drifts[i].distance,
			"token_id": drifts[i].tokenID,
		})
	}

	return result
}

// ==============================================================================
// Master Observability Aggregator
// ==============================================================================

// MoEObservability is the master controller combining all observability features.
type MoEObservability struct {
	ExpertLexicon       *TokenRoutingHistogram
	TokenLossTracker    *TokenCategoryLossTracker
	WeightVelocity      *WeightVelocityTracker
	SemanticDrift       *SemanticDriftTracker
	TokenTrajectory     *TokenTrajectoryTracker
	EmbeddingProjection *PCAProjection
	ExpertSimilarity    *ExpertSimilarityMatrix
	currentEpoch        int
	metricsBuffer       []map[string]interface{}
	mu                  sync.Mutex
	LayerSelections     map[int]*LayerSelection
	// TraceHistory keeps recent on-demand traces for UI browsing
	TraceHistory    []map[string]interface{}
	TraceHistoryMax int
	// TempTokenIDs holds temporary token IDs for the currently running forward
	// (used for on-demand inference traces). Access via setters/getters.
	TempTokenIDs []int
	// CurrentTraceID stores the active on-demand trace id (if any).
	CurrentTraceID string
	// TempLayerLatencies stores per-layer latency (ms) collected during the current on-demand trace.
	TempLayerLatencies map[int]int64
	// TempLayerComponentLatencies stores per-layer per-component latencies (ms) collected during the current on-demand trace.
	TempLayerComponentLatencies map[int]map[string]int64
}

// NewMoEObservability creates the master observability controller.
func NewMoEObservability(numExperts int, windowSize int, vocab *vocab.Vocabulary, w2vModel *word2vec.SimpleWord2Vec) *MoEObservability {
	categories := LoadTokenCategories(vocab)
	m := &MoEObservability{
		ExpertLexicon:       NewTokenRoutingHistogram(numExperts, windowSize),
		TokenLossTracker:    NewTokenCategoryLossTracker(categories),
		WeightVelocity:      NewWeightVelocityTracker(),
		SemanticDrift:       NewSemanticDriftTracker(w2vModel, vocab),
		TokenTrajectory:     NewTokenTrajectoryTracker(200),
		EmbeddingProjection: NewPCAProjection(),
		ExpertSimilarity:    NewExpertSimilarityMatrix(numExperts),
		metricsBuffer:       make([]map[string]interface{}, 0),
		LayerSelections:     make(map[int]*LayerSelection),
		TraceHistory:        make([]map[string]interface{}, 0),
		TraceHistoryMax:     50,
	}
	ObservabilityInstance = m
	return m
}

// RecordStep records a single training step's metrics.
func (m *MoEObservability) RecordStep(expertIDs, tokenIDs []int, loss float32) {
	m.ExpertLexicon.RecordBatchTokenRoutes(expertIDs, tokenIDs)
	m.TokenLossTracker.RecordLossForTokens(loss, tokenIDs)
}

// RecordWeights captures weight state for velocity tracking.
func (m *MoEObservability) RecordWeights(layerName string, weights []float32) {
	m.WeightVelocity.RecordWeightSnapshot(layerName, weights)
}

// FinishStep updates velocity and semantic drift tracking.
func (m *MoEObservability) FinishStep(layerName string, currentWeights []float32, embeddingTensor *tensor.Tensor) {
	m.WeightVelocity.UpdateWeightVelocity(layerName, currentWeights)
	m.SemanticDrift.RecordEmbeddingState(embeddingTensor)
}

// RecordTokenTrajectory records the full routing path of a token through layers.
func (m *MoEObservability) RecordTokenTrajectory(tokenID int, tokenStr string, expertPath []int, confidences []float32) {
	m.TokenTrajectory.RecordTokenTrajectory(tokenID, tokenStr, expertPath, confidences)
}

// UpdateEmbeddingProjection updates the 2D PCA projection of embeddings.
func (m *MoEObservability) UpdateEmbeddingProjection(vocab *vocab.Vocabulary, embeddingTensor *tensor.Tensor, topN int) {
	m.EmbeddingProjection.ComputeEmbeddingProjection(vocab, embeddingTensor, topN)
}

// RecordExpertWeights records expert weight matrices for similarity computation.
func (m *MoEObservability) RecordExpertWeights(expertID int, weights []float32) {
	m.ExpertSimilarity.RecordExpertWeights(expertID, weights)
}

// ComputeExpertSimilarity computes the expert redundancy matrix.
func (m *MoEObservability) ComputeExpertSimilarity() {
	m.ExpertSimilarity.ComputeSimilarityMatrix()
}

// ResetForEpoch resets windowed metrics for the next epoch.
func (m *MoEObservability) ResetForEpoch() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.ExpertLexicon.Reset()
	m.TokenLossTracker.UpdateImprovement()
	m.currentEpoch++
}

// GetDashboardMetrics returns all observability metrics as JSON.
func (m *MoEObservability) GetDashboardMetrics(vocab *vocab.Vocabulary) map[string]interface{} {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Get top 10 tokens per expert
	expertLexicon := m.ExpertLexicon.GetTopKTokensPerExpert(10, vocab)

	// Get loss per category
	categoryLoss := m.TokenLossTracker.GetCategoryLossMetrics()

	// Get weight velocity heatmap
	velocityHeatmap := m.WeightVelocity.GetVelocityHeatmap()

	// Get top 3 semantic drifts
	semanticDrift := m.SemanticDrift.GetTopSemanticShifts(3)

	// Get token trajectories
	trajectories := m.TokenTrajectory.GetTrajectories()

	// Get embedding projection
	embeddingProjection := m.EmbeddingProjection.GetProjectionCoordinates()

	// Get expert similarity matrix
	similarityMatrix := m.ExpertSimilarity.GetSimilarityMatrix()
	redundancyWarnings := m.ExpertSimilarity.GetRedundancyWarnings(0.85)

	return map[string]interface{}{
		"epoch":                m.currentEpoch,
		"expert_lexicon":       expertLexicon,
		"category_loss":        categoryLoss,
		"weight_velocity":      velocityHeatmap,
		"semantic_drift":       semanticDrift,
		"token_trajectories":   trajectories,
		"embedding_projection": embeddingProjection,
		"expert_similarity":    similarityMatrix,
		"redundancy_warnings":  redundancyWarnings,
		"timestamp":            fmt.Sprintf("%d", int64(math.Ceil(float64(m.currentEpoch)*1000))),
	}
}

// GetLayerRoutingSnapshot returns per-layer routing selections and confidences for the last forward pass.
func (m *MoEObservability) GetLayerRoutingSnapshot() []map[string]interface{} {
	m.mu.Lock()
	defer m.mu.Unlock()

	snapshot := make([]map[string]interface{}, 0)
	for li, layer := range ActiveLayers {
		layerInfo := make(map[string]interface{})
		selected := layer.GetSelectedExperts()
		// Collect confidences if available
		confidences := make([][]float32, 0)
		if layer.GateOutputs != nil {
			numExperts := len(layer.Experts)
			totalTokens := len(layer.GateOutputs.Data) / numExperts
			for t := 0; t < totalTokens; t++ {
				row := make([]float32, 0, numExperts)
				base := t * numExperts
				for e := 0; e < numExperts; e++ {
					row = append(row, layer.GateOutputs.Data[base+e])
				}
				confidences = append(confidences, row)
			}
		}

		layerInfo["layer_index"] = li
		layerInfo["selected_experts"] = selected
		layerInfo["confidences"] = confidences

		// If we have an observability-stored selection for this layer, include token IDs
		if ObservabilityInstance != nil {
			if sel, ok := ObservabilityInstance.LayerSelections[li]; ok {
				layerInfo["token_ids"] = sel.TokenIDs
			}
		}

		snapshot = append(snapshot, layerInfo)
	}
	return snapshot
}

// SetLayerSelection stores per-layer selection and confidences for later assembly.
func (m *MoEObservability) SetLayerSelection(layerIdx int, tokenIDs []int, selected [][]int, confidences [][]float32) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.LayerSelections[layerIdx] = &LayerSelection{
		TokenIDs:    append([]int{}, tokenIDs...),
		Selected:    append([][]int{}, selected...),
		Confidences: append([][]float32{}, confidences...),
	}
}

// ClearLayerSelections clears stored per-layer selections (call at epoch boundaries if desired).
func (m *MoEObservability) ClearLayerSelections() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.LayerSelections = make(map[int]*LayerSelection)
}

// ExportMetricsJSON serializes metrics to JSON string.
func (m *MoEObservability) ExportMetricsJSON(vocab *vocab.Vocabulary) (string, error) {
	metrics := m.GetDashboardMetrics(vocab)
	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// AppendTrace appends an on-demand trace record to the history (bounded).
func (m *MoEObservability) AppendTrace(record map[string]interface{}) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if record == nil {
		return
	}
	m.TraceHistory = append(m.TraceHistory, record)
	if len(m.TraceHistory) > m.TraceHistoryMax {
		m.TraceHistory = m.TraceHistory[1:]
	}
	// Persist trace to disk as JSONL
	go func(rec map[string]interface{}) {
		dir := filepath.Join("data", "observability")
		_ = os.MkdirAll(dir, 0755)
		fpath := filepath.Join(dir, "traces.jsonl")
		f, err := os.OpenFile(fpath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Printf("failed to open trace file: %v", err)
			return
		}
		defer f.Close()
		b, err := json.Marshal(rec)
		if err != nil {
			log.Printf("failed to marshal trace record: %v", err)
			return
		}
		if _, err := f.Write(append(b, '\n')); err != nil {
			log.Printf("failed to write trace record: %v", err)
			return
		}
	}(record)
}

// GetTraceHistory returns a shallow copy of recent traces.
func (m *MoEObservability) GetTraceHistory() []map[string]interface{} {
	m.mu.Lock()
	defer m.mu.Unlock()
	hist := make([]map[string]interface{}, len(m.TraceHistory))
	copy(hist, m.TraceHistory)
	return hist
}

// SetTempTokenIDs sets temporary token IDs for the currently running forward.
func (m *MoEObservability) SetTempTokenIDs(ids []int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if ids == nil {
		m.TempTokenIDs = nil
		return
	}
	m.TempTokenIDs = make([]int, len(ids))
	copy(m.TempTokenIDs, ids)
}

// GetTempTokenIDs returns a copy of the current temporary token IDs (may be nil).
func (m *MoEObservability) GetTempTokenIDs() []int {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.TempTokenIDs == nil {
		return nil
	}
	out := make([]int, len(m.TempTokenIDs))
	copy(out, m.TempTokenIDs)
	return out
}

// ClearTempTokenIDs clears the temporary token ID buffer.
func (m *MoEObservability) ClearTempTokenIDs() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.TempTokenIDs = nil
}

// AddLayerLatency records latency (ms) for a given layer index during the current trace.
func (m *MoEObservability) AddLayerLatency(layerIdx int, durMs int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.TempLayerLatencies == nil {
		m.TempLayerLatencies = make(map[int]int64)
	}
	// accumulate if called multiple times
	m.TempLayerLatencies[layerIdx] += durMs
}

// GetTempLayerLatencies returns a copy of recorded per-layer latencies (ms).
func (m *MoEObservability) GetTempLayerLatencies() map[int]int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[int]int64, len(m.TempLayerLatencies))
	for k, v := range m.TempLayerLatencies {
		out[k] = v
	}
	return out
}

// ClearTempLayerLatencies clears per-layer latency buffer.
func (m *MoEObservability) ClearTempLayerLatencies() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.TempLayerLatencies = nil
}

// AddLayerComponentLatency records latency (ms) for a given layer and component.
func (m *MoEObservability) AddLayerComponentLatency(layerIdx int, component string, durMs int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.TempLayerComponentLatencies == nil {
		m.TempLayerComponentLatencies = make(map[int]map[string]int64)
	}
	if m.TempLayerComponentLatencies[layerIdx] == nil {
		m.TempLayerComponentLatencies[layerIdx] = make(map[string]int64)
	}
	m.TempLayerComponentLatencies[layerIdx][component] += durMs
}

// GetTempLayerComponentLatencies returns a deep copy of component latencies.
func (m *MoEObservability) GetTempLayerComponentLatencies() map[int]map[string]int64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[int]map[string]int64, len(m.TempLayerComponentLatencies))
	for li, comp := range m.TempLayerComponentLatencies {
		out[li] = make(map[string]int64, len(comp))
		for k, v := range comp {
			out[li][k] = v
		}
	}
	return out
}

// ClearTempLayerComponentLatencies clears component latency buffer.
func (m *MoEObservability) ClearTempLayerComponentLatencies() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.TempLayerComponentLatencies = nil
}

// SetCurrentTraceID sets the active trace id for the running on-demand inference.
func (m *MoEObservability) SetCurrentTraceID(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CurrentTraceID = id
}

// GetCurrentTraceID returns the current active trace id (may be empty).
func (m *MoEObservability) GetCurrentTraceID() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.CurrentTraceID
}

// ClearCurrentTraceID clears the active trace id.
func (m *MoEObservability) ClearCurrentTraceID() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CurrentTraceID = ""
}

// DeleteTraceByID deletes traces matching the given id from history.
func (m *MoEObservability) DeleteTraceByID(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if id == "" {
		return false
	}
	newHist := make([]map[string]interface{}, 0, len(m.TraceHistory))
	removed := false
	for _, rec := range m.TraceHistory {
		if rid, ok := rec["id"].(string); ok && rid == id {
			removed = true
			continue
		}
		newHist = append(newHist, rec)
	}
	if removed {
		m.TraceHistory = newHist
	}
	return removed
}

// Log outputs a human-readable summary of current metrics.
func (m *MoEObservability) Log(vocab *vocab.Vocabulary) {
	m.mu.Lock()
	defer m.mu.Unlock()

	log.Printf("╔════ MoE Observability Report (Epoch %d) ════╗\n", m.currentEpoch)

	// Feature 1: Expert Lexicon
	log.Println("📚 Expert Lexicon (Top 5 tokens per expert):")
	lexicon := m.ExpertLexicon.GetTopKTokensPerExpert(5, vocab)
	for expertID := 0; expertID < len(lexicon); expertID++ {
		if tokens, ok := lexicon[expertID]; ok && len(tokens) > 0 {
			log.Printf("   Expert %d: %v\n", expertID, tokens)
		}
	}

	// Feature 2: Category Loss
	log.Println("📊 Category Loss Breakdown:")
	categoryLoss := m.TokenLossTracker.GetCategoryLossMetrics()
	for catName, metrics := range categoryLoss {
		avgLoss := metrics["average_loss"]
		improvement := metrics["improvement"]
		log.Printf("   %s: Loss=%.4f, Improvement=%.4f\n", catName, avgLoss, improvement)
	}

	// Feature 3: Weight Velocity
	log.Println("🔥 Weight Velocity Hotspots:")
	velocityMap := m.WeightVelocity.GetVelocityHeatmap()
	maxVel := velocityMap["max_velocity"]
	log.Printf("   Max Velocity: %.6f\n", maxVel)

	// Feature 4: Semantic Drift
	log.Println("🔄 Top Semantic Drifts:")
	drifts := m.SemanticDrift.GetTopSemanticShifts(3)
	for _, drift := range drifts {
		token := drift["token"]
		distance := drift["drift"]
		log.Printf("   Token \"%v\" drifted by %.4f\n", token, distance)
	}

	// Feature 5: Token Trajectories
	log.Println("🔀 Sample Token Trajectories:")
	trajectories := m.TokenTrajectory.GetTrajectories()
	for i, traj := range trajectories {
		if i >= 3 {
			break
		}
		token := traj["token"]
		path := traj["layer_path"]
		avgConf := traj["avg_conf"]
		log.Printf("   \"%v\" → path: %v (avg conf: %.3f)\n", token, path, avgConf)
	}

	// Feature 6: Embedding Projection
	log.Println("🌌 Embedding Galaxy (Top 5 words):")
	projection := m.EmbeddingProjection.GetProjectionCoordinates()
	for i, point := range projection {
		if i >= 5 {
			break
		}
		word := point["word"]
		x := point["x"]
		y := point["y"]
		cluster := point["cluster"]
		log.Printf("   \"%v\" → (%.2f, %.2f) cluster %v\n", word, x, y, cluster)
	}

	// Feature 7: Expert Similarity
	log.Println("⚠️  Expert Redundancy Warnings:")
	warnings := m.ExpertSimilarity.GetRedundancyWarnings(0.85)
	if len(warnings) == 0 {
		log.Println("   ✅ All experts are sufficiently diverse")
	} else {
		for _, warning := range warnings {
			expA := warning["expert_a"]
			expB := warning["expert_b"]
			sim := warning["similarity"]
			log.Printf("   Expert %v ↔ Expert %v: similarity %.3f\n", expA, expB, sim)
		}
	}

	log.Println("╚═════════════════════════════════════════════╝")
}

// ==============================================================================
// FEATURE 5: Token Trajectory / Sankey Flow Pipeline
// ==============================================================================

// TokenTrajectory tracks how a token is routed through experts across layers.
type TokenTrajectory struct {
	tokenID    int
	tokenStr   string
	layers     []int     // expertID per layer
	confidence []float32 // routing confidence per layer
}

// TokenTrajectoryTracker records full routing paths for tokens through the model.
type TokenTrajectoryTracker struct {
	trajectories []TokenTrajectory
	mu           sync.Mutex
	maxSize      int
}

// NewTokenTrajectoryTracker creates a trajectory tracker.
func NewTokenTrajectoryTracker(maxSize int) *TokenTrajectoryTracker {
	return &TokenTrajectoryTracker{
		trajectories: make([]TokenTrajectory, 0),
		maxSize:      maxSize,
	}
}

// RecordTokenTrajectory records the full layer-by-layer routing path for a token.
func (t *TokenTrajectoryTracker) RecordTokenTrajectory(tokenID int, tokenStr string, expertPath []int, confidences []float32) {
	t.mu.Lock()
	defer t.mu.Unlock()

	trajectory := TokenTrajectory{
		tokenID:    tokenID,
		tokenStr:   tokenStr,
		layers:     append([]int{}, expertPath...),
		confidence: append([]float32{}, confidences...),
	}

	t.trajectories = append(t.trajectories, trajectory)

	// Keep only recent trajectories
	if len(t.trajectories) > t.maxSize {
		t.trajectories = t.trajectories[len(t.trajectories)-t.maxSize:]
	}
}

// GetTrajectories returns all recorded token trajectories.
func (t *TokenTrajectoryTracker) GetTrajectories() []map[string]interface{} {
	t.mu.Lock()
	defer t.mu.Unlock()

	result := make([]map[string]interface{}, len(t.trajectories))
	for i, traj := range t.trajectories {
		result[i] = map[string]interface{}{
			"token":      traj.tokenStr,
			"token_id":   traj.tokenID,
			"layer_path": traj.layers,
			"confidence": traj.confidence,
			"avg_conf":   avgFloat32(traj.confidence),
		}
	}
	return result
}

// ==============================================================================
// FEATURE 6: PCA Projection of Embedding Space
// ==============================================================================

// PCAProjection represents a 2D projection of embeddings using PCA-like method.
type PCAProjection struct {
	words       []string
	coordinates []map[string]float32 // {x, y} per word
	cluster     map[string]int       // cluster assignment per word
	mu          sync.Mutex
}

// NewPCAProjection creates a PCA projection tracker.
func NewPCAProjection() *PCAProjection {
	return &PCAProjection{
		words:       make([]string, 0),
		coordinates: make([]map[string]float32, 0),
		cluster:     make(map[string]int),
	}
}

// ComputeEmbeddingProjection computes a 2D PCA-like projection of embeddings.
func (p *PCAProjection) ComputeEmbeddingProjection(vocab *vocab.Vocabulary, embeddingTensor *tensor.Tensor, topN int) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if embeddingTensor == nil || vocab == nil {
		return
	}
	// Real PCA using SVD (gonum)
	// Expect embeddingTensor.Shape == [vocabSize, embDim]
	if len(embeddingTensor.Shape) < 2 {
		return
	}
	vocabSize := embeddingTensor.Shape[0]
	embDim := embeddingTensor.Shape[1]

	if topN <= 0 || topN > vocabSize {
		topN = vocabSize
	}

	// Build data matrix (topN x embDim) as float64
	data := make([]float64, topN*embDim)
	words := make([]string, 0, topN)
	for i := 0; i < topN; i++ {
		word := vocab.GetWord(i)
		if word == "" || word == "[PAD]" || word == "[UNK]" {
			words = append(words, word)
		} else {
			words = append(words, word)
		}
		for j := 0; j < embDim; j++ {
			idx := i*embDim + j
			if idx < len(embeddingTensor.Data) {
				data[i*embDim+j] = float64(embeddingTensor.Data[idx])
			} else {
				data[i*embDim+j] = 0
			}
		}
	}

	A := mat.NewDense(topN, embDim, data)

	// Center columns (subtract mean)
	colMean := make([]float64, embDim)
	for j := 0; j < embDim; j++ {
		var sum float64
		for i := 0; i < topN; i++ {
			sum += A.At(i, j)
		}
		colMean[j] = sum / float64(topN)
		for i := 0; i < topN; i++ {
			A.Set(i, j, A.At(i, j)-colMean[j])
		}
	}

	// SVD
	var svd mat.SVD
	ok := svd.Factorize(A, mat.SVDThin)
	if !ok {
		return
	}

	var V mat.Dense
	svd.VTo(&V)

	// Take first two principal directions (columns of V)
	r, c := V.Dims()
	k := 2
	if c < 2 {
		k = c
	}
	Vsub := mat.NewDense(r, k, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < k; j++ {
			Vsub.Set(i, j, V.At(i, j))
		}
	}

	// Project: coords = A * Vsub  (topN x k)
	var coords mat.Dense
	coords.Mul(A, Vsub)

	// Save results
	p.words = make([]string, topN)
	p.coordinates = make([]map[string]float32, topN)
	p.cluster = make(map[string]int)
	for i := 0; i < topN; i++ {
		x := float32(coords.At(i, 0))
		y := float32(0)
		if k >= 2 {
			y = float32(coords.At(i, 1))
		}
		// Normalize for visualization scale
		nx := normalizeCoord(x)
		ny := normalizeCoord(y)
		p.words[i] = words[i]
		p.coordinates[i] = map[string]float32{"x": nx, "y": ny}
		p.cluster[words[i]] = i % 5
	}
}

// GetProjectionCoordinates returns the 2D coordinates for visualization.
func (p *PCAProjection) GetProjectionCoordinates() []map[string]interface{} {
	p.mu.Lock()
	defer p.mu.Unlock()

	result := make([]map[string]interface{}, len(p.words))
	for i, word := range p.words {
		cluster := 0
		if c, ok := p.cluster[word]; ok {
			cluster = c
		}

		result[i] = map[string]interface{}{
			"word":    word,
			"x":       p.coordinates[i]["x"],
			"y":       p.coordinates[i]["y"],
			"cluster": cluster,
		}
	}
	return result
}

// ==============================================================================
// FEATURE 7: Expert Similarity / Redundancy Matrix
// ==============================================================================

// ExpertSimilarityMatrix computes cosine similarity between expert weight matrices.
type ExpertSimilarityMatrix struct {
	expertWeights map[int][][]float32 // expertID → weights
	similarities  [][]float32         // similarity matrix
	mu            sync.Mutex
}

// NewExpertSimilarityMatrix creates a similarity tracker.
func NewExpertSimilarityMatrix(numExperts int) *ExpertSimilarityMatrix {
	return &ExpertSimilarityMatrix{
		expertWeights: make(map[int][][]float32),
		similarities:  nil,
	}
}

// RecordExpertWeights records the weight matrices of experts.
func (e *ExpertSimilarityMatrix) RecordExpertWeights(expertID int, weights []float32) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Store weights as 2D array (simplified: reshape from 1D)
	e.expertWeights[expertID] = reshapeWeights(weights, 16) // 16 columns for simplicity
}

// ComputeSimilarityMatrix computes cosine similarity between all expert pairs.
func (e *ExpertSimilarityMatrix) ComputeSimilarityMatrix() {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Build a list of expert keys
	keys := make([]int, 0, len(e.expertWeights))
	for k := range e.expertWeights {
		keys = append(keys, k)
	}
	sort.Ints(keys)

	n := len(keys)
	if n == 0 {
		e.similarities = nil
		return
	}

	// Initialize similarity matrix
	e.similarities = make([][]float32, n)
	for i := range e.similarities {
		e.similarities[i] = make([]float32, n)
	}

	// Pre-flatten all weights
	flats := make([][]float32, n)
	for i, k := range keys {
		flats[i] = e.flattenWeights(e.expertWeights[k])
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				e.similarities[i][j] = 1.0
				continue
			}
			e.similarities[i][j] = cosineSimilarityInternal(flats[i], flats[j])
		}
	}
}

// GetSimilarityMatrix returns the similarity matrix as a 2D array.
func (e *ExpertSimilarityMatrix) GetSimilarityMatrix() [][]float32 {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Return deep copy
	if e.similarities == nil {
		return nil
	}
	result := make([][]float32, len(e.similarities))
	for i := range e.similarities {
		result[i] = make([]float32, len(e.similarities[i]))
		copy(result[i], e.similarities[i])
	}
	return result
}

// GetRedundancyWarnings returns list of experts that are too similar.
func (e *ExpertSimilarityMatrix) GetRedundancyWarnings(threshold float32) []map[string]interface{} {
	e.mu.Lock()
	defer e.mu.Unlock()

	warnings := make([]map[string]interface{}, 0)

	for i := 0; i < len(e.similarities); i++ {
		for j := i + 1; j < len(e.similarities); j++ {
			if e.similarities[i][j] > threshold {
				warnings = append(warnings, map[string]interface{}{
					"expert_a":   i,
					"expert_b":   j,
					"similarity": e.similarities[i][j],
					"alert":      "⚠️ Experts are too similar - consider forcing surgery mutation",
				})
			}
		}
	}

	return warnings
}

// flattenWeights converts 2D weight matrix to 1D array.
func (e *ExpertSimilarityMatrix) flattenWeights(weights [][]float32) []float32 {
	result := make([]float32, 0)
	for _, row := range weights {
		result = append(result, row...)
	}
	return result
}

// ==============================================================================
// Helper Functions
// ==============================================================================

func normalizeCoord(val float32) float32 {
	// Normalize to [-1, 1] using tanh-like scaling
	return float32(math.Tanh(float64(val)))
}

func reshapeWeights(flat []float32, cols int) [][]float32 {
	if len(flat) == 0 || cols == 0 {
		return [][]float32{}
	}

	rows := (len(flat) + cols - 1) / cols
	result := make([][]float32, rows)

	for i := 0; i < rows; i++ {
		start := i * cols
		end := start + cols
		if end > len(flat) {
			end = len(flat)
		}
		result[i] = flat[start:end]
	}

	return result
}

func avgFloat32(arr []float32) float32 {
	if len(arr) == 0 {
		return 0
	}
	var sum float32
	for _, v := range arr {
		sum += v
	}
	return sum / float32(len(arr))
}
