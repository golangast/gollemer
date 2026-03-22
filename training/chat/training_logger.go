package chat

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	moe "github.com/golangast/gollemer/internal/moe"
)

// ─────────────────────────────────────────────────────────────────────────────
// TrainingLogger: writes per-epoch metrics to a CSV file for external analysis.
// ─────────────────────────────────────────────────────────────────────────────

// TrainingLogger writes per-epoch metrics to a CSV file.
type TrainingLogger struct {
	File   *os.File
	Writer *csv.Writer
	Path   string
}

// NewTrainingLogger creates (or appends to) a CSV log file.
// If the file already exists its header is preserved; new rows are appended.
func NewTrainingLogger(filename string) (*TrainingLogger, error) {
	// Ensure the directory exists
	if err := os.MkdirAll(filepath.Dir(filename), 0o755); err != nil {
		return nil, fmt.Errorf("failed to create log directory: %w", err)
	}

	// Decide: create a fresh file or append
	needsHeader := false
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		needsHeader = true
	}

	f, err := os.OpenFile(filename, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("failed to open training log: %w", err)
	}

	w := csv.NewWriter(f)
	if needsHeader {
		header := []string{
			"epoch", "avg_loss", "lb_loss", "perplexity", "learning_rate",
			"e0_util", "e1_util", "e2_util", "e3_util",
			"e4_util", "e5_util", "e6_util", "e7_util",
			"frozen_experts",
		}
		if err := w.Write(header); err != nil {
			return nil, fmt.Errorf("failed to write CSV header: %w", err)
		}
		w.Flush()
	}

	return &TrainingLogger{File: f, Writer: w, Path: filename}, nil
}

// LogEpoch appends one row of metrics to the CSV.
//   - utils: per-expert utilisation fractions (0-1), one per expert
//   - frozenExperts: IDs frozen by ThawScheduler this epoch
func (l *TrainingLogger) LogEpoch(epoch int, avgLoss, lbLoss, perplexity, lr float64, utils []float64, frozenExperts []int) {
	row := []string{
		strconv.Itoa(epoch),
		fmt.Sprintf("%.4f", avgLoss),
		fmt.Sprintf("%.4f", lbLoss),
		fmt.Sprintf("%.2f", perplexity),
		fmt.Sprintf("%.8f", lr),
	}

	// Pad or trim utils to always have 8 columns
	for i := 0; i < 8; i++ {
		if i < len(utils) {
			row = append(row, fmt.Sprintf("%.4f", utils[i]))
		} else {
			row = append(row, "0.0000")
		}
	}

	// Encode frozen list as a compact string e.g. "0,1,2" or ""
	frozenStr := ""
	for i, id := range frozenExperts {
		if i > 0 {
			frozenStr += ","
		}
		frozenStr += strconv.Itoa(id)
	}
	row = append(row, frozenStr)

	_ = l.Writer.Write(row)
	l.Writer.Flush()
}

// Close flushes and closes the underlying file.
func (l *TrainingLogger) Close() {
	l.Writer.Flush()
	_ = l.File.Close()
}

// ─────────────────────────────────────────────────────────────────────────────
// GenerateProgressReport: reads the CSV and prints an expert MVP/slacker report
// ─────────────────────────────────────────────────────────────────────────────

// ExpertStat holds aggregated performance data for a single expert.
type ExpertCsvStat struct {
	ID      int
	AvgUtil float64
	Growth  float64 // Δ utilisation over the last 10 epochs
}

// GenerateProgressReport reads training_log.csv and prints a concise analysis.
func GenerateProgressReport(logPath string) {
	f, err := os.Open(logPath)
	if err != nil {
		fmt.Printf("⚠️  Could not open log file %s: %v\n", logPath, err)
		return
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil || len(records) < 2 { // at least header + 1 data row
		fmt.Println("⚠️  Not enough data in log file yet.")
		return
	}

	// records[0] = header; records[1:] = data
	data := records[1:]
	numRows := len(data)

	// Columns: epoch, avg_loss, lb_loss, perplexity, lr, e0..e7, frozen
	const firstUtilCol = 5 // index of e0_util in each row
	const numExperts = 8

	utilSum := make([]float64, numExperts)
	utilRecent := make([]float64, numExperts) // last 10 epochs average
	utilEarly := make([]float64, numExperts)  // first 10 epochs average
	recentWindow := 10
	earlyWindow := 10

	for rowIdx, row := range data {
		for e := 0; e < numExperts; e++ {
			col := firstUtilCol + e
			if col >= len(row) {
				continue
			}
			val, _ := strconv.ParseFloat(row[col], 64)
			utilSum[e] += val

			if numRows <= recentWindow || rowIdx >= numRows-recentWindow {
				utilRecent[e] += val
			}
			if rowIdx < earlyWindow {
				utilEarly[e] += val
			}
		}
	}

	// Normalise windows
	recentCount := float64(min(recentWindow, numRows))
	earlyCount := float64(min(earlyWindow, numRows))

	stats := make([]ExpertCsvStat, numExperts)
	for e := 0; e < numExperts; e++ {
		stats[e].ID = e
		stats[e].AvgUtil = utilSum[e] / float64(numRows)
		if earlyCount > 0 && recentCount > 0 {
			stats[e].Growth = (utilRecent[e] / recentCount) - (utilEarly[e] / earlyCount)
		}
	}

	// Sort by growth descending for MVP; ascending for slacker
	byCrowth := make([]ExpertCsvStat, numExperts)
	copy(byCrowth, stats)
	sort.Slice(byCrowth, func(i, j int) bool {
		return byCrowth[i].Growth > byCrowth[j].Growth
	})

	mvp := byCrowth[0]
	slacker := byCrowth[numExperts-1]

	// Also find the most-used overall
	byUtil := make([]ExpertCsvStat, numExperts)
	copy(byUtil, stats)
	sort.Slice(byUtil, func(i, j int) bool {
		return byUtil[i].AvgUtil > byUtil[j].AvgUtil
	})

	// Parse final perplexity
	finalPPLStr := ""
	if len(data[numRows-1]) > 3 {
		finalPPLStr = data[numRows-1][3]
	}
	finalPPL, _ := strconv.ParseFloat(finalPPLStr, 64)

	fmt.Println("\n🏆 ─── Gollemer Expert MVP Report ───────────────────────")
	fmt.Printf("   📈 Epochs logged    : %d\n", numRows)
	fmt.Printf("   📉 Final Perplexity : %.2f\n", finalPPL)
	fmt.Println()
	fmt.Println("   Expert  │  Avg Util  │  Growth (last 10 vs first 10)")
	fmt.Println("   ────────┼────────────┼──────────────────────────────")
	for _, s := range byUtil {
		arrow := "→"
		if s.Growth > 0.01 {
			arrow = "↑"
		} else if s.Growth < -0.01 {
			arrow = "↓"
		}
		fmt.Printf("     E%-2d   │   %5.1f%%   │ %s  %.2f%%\n",
			s.ID, s.AvgUtil*100, arrow, s.Growth*100)
	}
	fmt.Println()
	fmt.Printf("   ✅ Current MVP    : Expert %d  (Growth +%.2f%%)\n", mvp.ID, mvp.Growth*100)
	fmt.Printf("   ⚠️  Action Needed : Expert %d  (Growth %.2f%% — consider Reset)\n", slacker.ID, slacker.Growth*100)
	fmt.Println("   ──────────────────────────────────────────────────────")
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint helpers: periodic epoch snapshots independent of best-model saves
// ─────────────────────────────────────────────────────────────────────────────

// SavePeriodicCheckpoint saves the model every N epochs to a dated file.
// It does NOT overwrite the "best" checkpoint — it is an insurance snapshot.
func SavePeriodicCheckpoint(model *moe.IntentMoE, checkpointDir string, epoch int, every int) {
	if epoch%every != 0 {
		return
	}
	if err := os.MkdirAll(checkpointDir, 0o755); err != nil {
		fmt.Printf("⚠️  Could not create checkpoint dir: %v\n", err)
		return
	}
	path := filepath.Join(checkpointDir, fmt.Sprintf("gollemer_epoch_%04d.gob", epoch))
	if err := moe.SaveIntentMoEModelToGOB(model, path); err != nil {
		fmt.Printf("⚠️  Periodic checkpoint save failed (epoch %d): %v\n", epoch, err)
	} else {
		fmt.Printf("💾 Periodic checkpoint saved: %s\n", path)
	}

	// Prune old checkpoints: keep only the 3 most recent to save disk space
	pruneOldCheckpoints(checkpointDir, 3)
}

// pruneOldCheckpoints keeps only the `keep` most recent *.gob files in dir.
func pruneOldCheckpoints(dir string, keep int) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}
	var gobs []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".gob" {
			gobs = append(gobs, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(gobs) // ascending → oldest first
	for len(gobs) > keep {
		_ = os.Remove(gobs[0])
		gobs = gobs[1:]
	}
}

// CollectUtilisationFractions builds a []float64 of per-expert utilisation
// fractions (0-1) from ALL active encoder MoE layers (averaged across layers).
func CollectUtilisationFractions() []float64 {
	allLayers := moe.ActiveLayers
	if len(allLayers) == 0 {
		return nil
	}

	numExperts := len(allLayers[0].Experts)
	totals := make([]float64, numExperts)
	var grandTotal float64

	for _, layer := range allLayers {
		for i, count := range layer.AccumulatedUtilization {
			if i < numExperts {
				totals[i] += float64(count)
				grandTotal += float64(count)
			}
		}
	}

	fracs := make([]float64, numExperts)
	if grandTotal > 0 {
		for i := range fracs {
			fracs[i] = totals[i] / grandTotal
		}
	}
	return fracs
}

// min returns the smaller of two ints (helper for logger, Go 1.21 generics not assumed).
func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ─────────────────────────────────────────────────────────────────────────────
// Ensure math import doesn't get flagged unused if compiler optimises
// ─────────────────────────────────────────────────────────────────────────────
var _ = math.MaxFloat64 // keep math import live
