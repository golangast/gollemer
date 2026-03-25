package moe

import (
	"fmt"
	"math"
	"strings" // Added for strings.Repeat
	"encoding/csv"
	"os"
	"strconv"
	"time"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// GetMaxUtilization calculates the dominance of the most used expert.
func GetMaxUtilization(counts []int) float32 {
	if len(counts) == 0 {
		return 0
	}

	total := 0
	max := 0
	for _, c := range counts {
		total += c
		if c > max {
			max = c
		}
	}

	if total == 0 {
		return 0
	}

	// Return the percentage of the most dominant expert
	return float32(max) / float32(total)
}

// LogExpertHealth saves the current utilization state to a CSV file.
func LogExpertHealth(filename string, epoch int, layerID int, counts []int) {
	file, err := os.OpenFile(filename, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening log file:", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Calculate percentages
	total := 0
	for _, c := range counts {
		total += c
	}

	// Prepare row: Timestamp, Epoch, Layer, E0%, E1%, ..., EN%, MaxDominance
	row := []string{
		time.Now().Format("2006-01-02 15:04:05"),
		strconv.Itoa(epoch),
		strconv.Itoa(layerID),
	}

	maxDom := 0.0
	for _, c := range counts {
		perc := 0.0
		if total > 0 {
			perc = float64(c) / float64(total)
		}
		row = append(row, fmt.Sprintf("%.4f", perc))
		if perc > maxDom {
			maxDom = perc
		}
	}
	row = append(row, fmt.Sprintf("%.4f", maxDom))

	writer.Write(row)
}

// LogWeightStretch visualizes the "Commitment Level" of the model weights.
func LogWeightStretch(m *IntentMoE) {
	var highCommit, active, timid int
	params := m.Parameters()
	total := 0
	for _, p := range params {
		total += len(p.Data)
		for _, w := range p.Data {
			absW := math.Abs(w)
			if absW > 0.50 {
				highCommit++ // The "Confidence" Zone
			} else if absW > 0.20 {
				active++     // The "Learning" Zone
			} else {
				timid++      // The "Noise" Zone
			}
		}
	}

	percentHigh := (float32(highCommit) / float32(total)) * 100
	percentActive := (float32(active) / float32(total)) * 100

	fmt.Printf("⚖️  Weight Stretch: [High: %.2f%%] [Active: %.2f%%] [Timid: %d units]\n", 
		percentHigh, percentActive, timid)
	
	// Auto-Heal Trigger suggestion logic
	if percentHigh < 0.1 {
		fmt.Println("📢 Suggestion: Increase Weight Decay. Weights are too clustered near zero.")
	}
}

// CheckSaturation calculates Max weight and L2 norm to detect training divergence.
func CheckSaturation(m *IntentMoE, epoch int) {
	var maxWeight float64
	var sumSq float64
	params := m.Parameters()
	total := 0
	for _, p := range params {
		total += len(p.Data)
		for _, w := range p.Data {
			absW := math.Abs(w)
			if absW > maxWeight {
				maxWeight = absW
			}
			sumSq += w * w
		}
	}
	
	l2Norm := math.Sqrt(sumSq) / float64(total)

	// 🚩 ALERT: Weight Saturation detected
	if maxWeight > 2.0 {
		fmt.Printf("⚠️  SATURATION WARNING: Max Weight reached %.2f!\n", maxWeight)
		fmt.Println("👉 Recommendation: Lower LR or increase Weight Decay immediately to prevent divergence.")
	}

	// ❄️ ALERT: Vanishing Gradient detected
	if l2Norm < 1e-5 && epoch > 2 {
		fmt.Printf("❄️  FREEZE WARNING: L2 Norm is critically low (%.2e).\n", l2Norm)
		fmt.Println("👉 Recommendation: Model is 'stuck'. Increase LR or check activation functions.")
	}
}

// MoEStats holds utilization statistics for a batch to calculate auxiliary loss.
type MoEStats struct {
	RouterProbSum []float64 // Sum of probabilities for each expert
	ExpertCounts  []float64 // Hard count of how many times each expert was chosen
}

// PrintExpertWeightHistogram prints a detailed visual distribution of expert weights.
func PrintExpertWeightHistogram(label string, expert Expert) {
	params := expert.Parameters()
	if len(params) == 0 {
		return
	}

	const numBins = 10
	bins := make([]int, numBins)
	minVal, maxVal := -1.0, 1.0

	var totalWeights int
	for _, p := range params {
		totalWeights += len(p.Data)
		for _, w := range p.Data {
			binIdx := int((w - minVal) / (maxVal - minVal) * float64(numBins))
			if binIdx >= 0 && binIdx < numBins {
				bins[binIdx]++
			}
		}
	}

	if totalWeights == 0 {
		return
	}

	fmt.Printf("\n📉 [%s] Weight Distribution (Total: %d):\n", label, totalWeights)
	for i, count := range bins {
		binStart := minVal + float64(i)*(maxVal-minVal)/float64(numBins)
		binEnd := minVal + float64(i+1)*(maxVal-minVal)/float64(numBins)
		
		barLen := (count * 40) / totalWeights // Scale to 40 chars
		bar := strings.Repeat("█", barLen)
		fmt.Printf("[%5.2f to %5.2f]: %s (%d)\n", binStart, binEnd, bar, count)
	}
}

// PrintLayerWeightHistogram visualizes the health of an entire MoE layer.
func PrintLayerWeightHistogram(label string, layer *MoELayer) {
	params := layer.Parameters()
	if len(params) == 0 {
		return
	}

	const numBins = 10
	bins := make([]int, numBins)
	minVal, maxVal := -1.0, 1.0

	var totalWeights int
	for _, p := range params {
		totalWeights += len(p.Data)
		for _, w := range p.Data {
			binIdx := int((w - minVal) / (maxVal - minVal) * float64(numBins))
			if binIdx >= 0 && binIdx < numBins {
				bins[binIdx]++
			}
		}
	}

	if totalWeights == 0 {
		return
	}

	fmt.Printf("\n� Layer [%s] Weight Distribution (Total: %d):\n", label, totalWeights)
	for i, count := range bins {
		binStart := minVal + float64(i)*(maxVal-minVal)/float64(numBins)
		binEnd := minVal + float64(i+1)*(maxVal-minVal)/float64(numBins)
		
		barLen := (count * 40) / totalWeights
		bar := strings.Repeat("█", barLen)
		fmt.Printf("[%5.2f to %5.2f]: %s (%d)\n", binStart, binEnd, bar, count)
	}
}

// CalculateSparsityPenalty calculates the L1 norm of weights to encourage sparse connections.
func CalculateSparsityPenalty(params []*tensor.Tensor, lambda float64) float64 {
	var l1Loss float64
	for _, p := range params {
		for _, w := range p.Data {
			l1Loss += math.Abs(w)
		}
	}
	return l1Loss * lambda
}

// CalculateDiversityLoss calculates the Shannon Entropy of probs to force sharp peaks.
func CalculateDiversityLoss(probs *tensor.Tensor) float64 {
	var entropy float64
	epsilon := 1e-10 // Prevent log(0)

	for _, p := range probs.Data {
		if p > epsilon {
			entropy -= p * math.Log2(p)
		}
	}
	
	// We want to MINIMIZE entropy to force sharp peaks.
	// Scale by small factor (0.01) so it doesn't overwhelm Cross-Entropy loss.
	return (entropy / float64(len(probs.Data))) * 0.01
}

// CalculateUsageVariance calculates the variance of expert usage to discourage monopolies.
func CalculateUsageVariance(usageMap map[int]int, numExperts int, totalTokens int) float64 {
    if numExperts == 0 || totalTokens == 0 {
        return 0
    }
    targetUsage := float64(totalTokens) / float64(numExperts)
    var variance float64

    for i := 0; i < numExperts; i++ {
        count := usageMap[i]
        diff := float64(count) - targetUsage
        variance += diff * diff
    }
    
    // Normalize and scale
    return (variance / float64(totalTokens)) * 0.01
}

// PerformWeightSurgery kills weights that fall below the "signal" threshold.
func PerformWeightSurgery(expert Expert, threshold float64) int {
    killCount := 0
    params := expert.Parameters()
    for _, p := range params {
        for i, w := range p.Data {
            if w > -threshold && w < threshold {
                p.Data[i] = 0.0
                killCount++
            }
        }
    }
    return killCount
}

// PrintExpertHeatmap visualizes weight density as a text-based heat map.
func PrintExpertHeatmap(label string, expert Expert, threshold float64) {
    params := expert.Parameters()
    if len(params) == 0 {
        return
    }

    // Use the largest parameter (usually the weight matrix)
    var weights []float64
    maxLen := 0
    for _, p := range params {
        if len(p.Data) > maxLen {
            maxLen = len(p.Data)
            weights = p.Data
        }
    }

    // Assuming a near-square matrix for visualization
    size := int(math.Sqrt(float64(len(weights))))
    if size > 64 {
        size = 64 // Cap size for logs
    }
    
    fmt.Printf("\n🗺️ Sparsity Heatmap [%s] (Threshold: %0.3f)\n", label, threshold)
    
    for i := 0; i < size; i++ {
        rowStr := ""
        for j := 0; j < size; j++ {
            if i*size+j >= len(weights) {
                break
            }
            w := weights[i*size+j]
            if w > -threshold && w < threshold {
                rowStr += "░" // Dead
            } else if w > 0 {
                rowStr += "█" // Positive Active
            } else {
                rowStr += "▒" // Negative Active
            }
        }
        fmt.Println(rowStr)
    }
}
// ValidateExpertHealth checks if the Router is distributing tokens across all experts.
// 'counts' is the slice of how many tokens each expert received in the last epoch.
func ValidateExpertHealth(layerName string, counts []int) {
	totalTokens := 0
	for _, c := range counts {
		totalTokens += c
	}

	if totalTokens == 0 {
		fmt.Printf("--- Health Check: %s [No Data] ---\n", layerName)
		return
	}

	fmt.Printf("--- Health Check: %s ---\n", layerName)
	
	var entropy float64
	numExperts := len(counts)

	for i, c := range counts {
		utilization := float64(c) / float64(totalTokens)
		fmt.Printf("   Expert %d: [%.2f%% utilization]\n", i, utilization*100)
		
		if utilization > 0 {
			entropy -= utilization * math.Log2(utilization)
		}
	}

	// Max possible entropy is log2(numExperts)
	maxEntropy := math.Log2(float64(numExperts))
	healthScore := (entropy / (maxEntropy + 1e-10)) * 100

	fmt.Printf("Overall Layer Health Score: %.2f%%\n", healthScore)
	if healthScore < 50.0 {
		fmt.Println("⚠️ WARNING: Expert Collapse detected. Increase Auxiliary Loss weight.")
	}
	fmt.Println("-------------------------------")
}
