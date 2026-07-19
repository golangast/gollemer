package moe

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// ExpertMonitor tracks token dispatch counts during a single training session
// (epoch or a configurable window). It decouples monitoring from the MoELayer
// struct, making it safe to reset independently without touching model weights.
type ExpertMonitor struct {
	Counts     []int // Token count per expert
	Total      int   // Total tokens processed
	NumExperts int

	mu        sync.Mutex
	History   []map[string]interface{} // Rolling utilization samples for ASCII/CLI dashboards
	MaxEvents int
}

// NewExpertMonitor creates a monitor for the given number of experts.
func NewExpertMonitor(numExperts int) *ExpertMonitor {
	return &ExpertMonitor{
		Counts:     make([]int, numExperts),
		NumExperts: numExperts,
		MaxEvents:  80,
	}
}

// LogSelection records that a token was dispatched to the given expert.
func (m *ExpertMonitor) LogSelection(expertID int) {
	if expertID >= 0 && expertID < m.NumExperts {
		m.mu.Lock()
		m.Counts[expertID]++
		m.Total++
		m.recordSample(expertID)
		m.mu.Unlock()
		if GlobalTelemetry != nil {
			GlobalTelemetry.RecordTrace("router", "dispatch", map[string]interface{}{"expert_id": expertID, "total": m.Total})
		}
	}
}

// LogSelections records a batch of expert selections at once.
func (m *ExpertMonitor) LogSelections(expertIDs []int) {
	for _, id := range expertIDs {
		m.LogSelection(id)
	}
}

// Reset clears all accumulated counts for the next epoch/window.
func (m *ExpertMonitor) Reset() {
	m.mu.Lock()
	for i := range m.Counts {
		m.Counts[i] = 0
	}
	m.Total = 0
	m.mu.Unlock()
}

// Fractions returns per-expert utilisation as [0, 1] float fractions.
func (m *ExpertMonitor) Fractions() []float64 {
	m.mu.Lock()
	defer m.mu.Unlock()

	fracs := make([]float64, m.NumExperts)
	if m.Total == 0 {
		return fracs
	}
	for i, c := range m.Counts {
		fracs[i] = float64(c) / float64(m.Total)
	}
	return fracs
}

// LoadLoss computes a mean-squared-error penalty for unbalanced expert usage.
// A perfectly balanced model returns 0. A model routing everything to one
// expert returns a large positive value. Use this as an auxiliary loss term.
func (m *ExpertMonitor) LoadLoss() float64 {
	return CalculateLoadLoss(m.Counts)
}

// CalculateLoadLoss computes the MSE-based load-balance penalty from integer
// hard-counts. This is complementary to CalculateImportanceLossTensor (which
// uses soft router probabilities). Use this for post-forward diagnostics.
//
//   - usage: slice of token counts per expert in the current forward pass.
//   - Returns 0 when perfectly balanced; larger when skewed.
func CalculateLoadLoss(usage []int) float64 {
	numExperts := len(usage)
	if numExperts == 0 {
		return 0
	}

	var totalTokens int
	for _, count := range usage {
		totalTokens += count
	}
	if totalTokens == 0 {
		return 0
	}

	// Target: perfectly uniform distribution
	target := float64(totalTokens) / float64(numExperts)

	var variance float64
	for _, count := range usage {
		diff := float64(count) - target
		variance += diff * diff
	}

	// Mean squared error of the distribution, normalised by expert count
	return variance / float64(numExperts)
}

// Report logs a visual utilisation bar chart to stdout.
func (m *ExpertMonitor) Report() {
	fmt.Println("\n─── Expert Utilisation Report ───────────────────────")
	fracs := m.Fractions()
	m.mu.Lock()
	counts := append([]int(nil), m.Counts...)
	total := m.Total
	m.mu.Unlock()
	for i, f := range fracs {
		pct := f * 100
		bar := strings.Repeat("█", int(pct/2)) // 1 block per 2%
		fmt.Printf("  Expert %d │ [%-50s] %5.1f%%  (%d tokens)\n",
			i, bar, pct, counts[i])
	}
	fmt.Printf("  Total tokens dispatched: %d\n", total)
	fmt.Printf("  Load Imbalance (MSE):    %.4f\n", m.LoadLoss())
	fmt.Println("─────────────────────────────────────────────────────")
}

// HeatmapASCII emits a compact CLI-native heatmap for recent routing utilization.
func (m *ExpertMonitor) HeatmapASCII(label string) string {
	m.mu.Lock()
	defer m.mu.Unlock()

	fracs := make([]float64, m.NumExperts)
	if m.Total > 0 {
		for i, c := range m.Counts {
			fracs[i] = float64(c) / float64(m.Total)
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "[%s]\n", label)
	for i := 0; i < m.NumExperts; i++ {
		pct := int(math.Round(fracs[i] * 100))
		bar := strings.Repeat("#", int(math.Max(1, float64(pct)/5)))
		if pct == 0 {
			bar = "-"
		}
		fmt.Fprintf(&b, "E%d | %-24s %3d%% (%d)\n", i, bar, pct, m.Counts[i])
	}
	return b.String()
}

func (m *ExpertMonitor) recordSample(expertID int) {
	if m.MaxEvents <= 0 {
		m.MaxEvents = 80
	}
	m.History = append(m.History, map[string]interface{}{"expert_id": expertID, "count": 1, "total": m.Total})
	if len(m.History) > m.MaxEvents {
		m.History = m.History[len(m.History)-m.MaxEvents:]
	}
}

// Snapshot writes a JSON file with a compact routing snapshot for debugging.
func (m *ExpertMonitor) Snapshot(path string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	fracs := make([]float64, m.NumExperts)
	if m.Total > 0 {
		for i, c := range m.Counts {
			fracs[i] = float64(c) / float64(m.Total)
		}
	}
	payload := map[string]interface{}{
		"total":       m.Total,
		"counts":      append([]int(nil), m.Counts...),
		"fractions":   fracs,
		"history":     append([]map[string]interface{}(nil), m.History...),
		"num_experts": m.NumExperts,
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// MaxImbalance returns the largest deviation any single expert has from
// the perfect target, expressed as a fraction (0 = perfect, 1 = all tokens).
func (m *ExpertMonitor) MaxImbalance() float64 {
	if m.Total == 0 || m.NumExperts == 0 {
		return 0
	}
	target := float64(m.Total) / float64(m.NumExperts)
	maxDev := 0.0
	for _, c := range m.Counts {
		dev := math.Abs(float64(c)-target) / float64(m.Total)
		if dev > maxDev {
			maxDev = dev
		}
	}
	return maxDev
}
