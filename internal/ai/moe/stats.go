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
			absW := float32(math.Abs(float64(w)))
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
	var maxWeight float32
	var sumSq float32
	params := m.Parameters()
	total := 0
	for _, p := range params {
		total += len(p.Data)
		for _, w := range p.Data {
			absW := float32(math.Abs(float64(w)))
			if absW > maxWeight {
				maxWeight = absW
			}
			sumSq += w * w
		}
	}
	
	l2Norm := float32(math.Sqrt(float64(sumSq))) / float32(total)

	// 🚩 ALERT: Weight Saturation detected
	if maxWeight > 2.0 {
		fmt.Printf("⚠️  SATURATION WARNING: Max Weight reached %.2f!\n", float64(maxWeight))
		fmt.Println("👉 Recommendation: Lower LR or increase Weight Decay immediately to prevent divergence.")
	}

	// ❄️ ALERT: Vanishing Gradient detected
	if l2Norm < 1e-5 && epoch > 2 {
		fmt.Printf("❄️  FREEZE WARNING: L2 Norm is critically low (%.2e).\n", float64(l2Norm))
		fmt.Println("👉 Recommendation: Model is 'stuck'. Increase LR or check activation functions.")
	}
}

// MoEStats holds utilization statistics for a batch to calculate auxiliary loss.
type MoEStats struct {
	RouterProbSum []float32 // Sum of probabilities for each expert
	ExpertCounts  []int     // Hard count of how many times each expert was chosen
}

// PrintExpertWeightHistogram prints a detailed visual distribution of expert weights.
func PrintExpertWeightHistogram(label string, expert Expert) {
	params := expert.Parameters()
	if len(params) == 0 {
		return
	}

	const numBins = 10
	bins := make([]int, numBins)
	var minVal, maxVal float32 = -1.0, 1.0

	var totalWeights int
	for _, p := range params {
		totalWeights += len(p.Data)
		for _, w := range p.Data {
			binIdx := int((w - minVal) / (maxVal - minVal) * float32(numBins))
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
		binStart := minVal + float32(i)*(maxVal-minVal)/float32(numBins)
		binEnd := minVal + float32(i+1)*(maxVal-minVal)/float32(numBins)
		
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
	var minVal, maxVal float32 = -1.0, 1.0

	var totalWeights int
	for _, p := range params {
		totalWeights += len(p.Data)
		for _, w := range p.Data {
			binIdx := int((w - minVal) / (maxVal - minVal) * float32(numBins))
			if binIdx >= 0 && binIdx < numBins {
				bins[binIdx]++
			}
		}
	}

	if totalWeights == 0 {
		return
	}

	fmt.Printf("\n Layer [%s] Weight Distribution (Total: %d):\n", label, totalWeights)
	for i, count := range bins {
		binStart := minVal + float32(i)*(maxVal-minVal)/float32(numBins)
		binEnd := minVal + float32(i+1)*(maxVal-minVal)/float32(numBins)
		
		barLen := (count * 40) / totalWeights
		bar := strings.Repeat("█", barLen)
		fmt.Printf("[%5.2f to %5.2f]: %s (%d)\n", binStart, binEnd, bar, count)
	}
}

// CalculateSparsityPenalty calculates the L1 norm of weights to encourage sparse connections.
func CalculateSparsityPenalty(params []*tensor.Tensor, lambda float32) float32 {
	var l1Loss float32
	for _, p := range params {
		for _, w := range p.Data {
			l1Loss += float32(math.Abs(float64(w)))
		}
	}
	return l1Loss * lambda
}

// CalculateDiversityLoss calculates the Shannon Entropy of probs to force sharp peaks.
func CalculateDiversityLoss(probs *tensor.Tensor) float32 {
	var entropy float32
	var epsilon float32 = 1e-10 // Prevent log(0)

	for _, p := range probs.Data {
		if p > epsilon {
			entropy -= p * float32(math.Log2(float64(p)))
		}
	}
	
	// We want to MINIMIZE entropy to force sharp peaks.
	// Scale by small factor (0.01) so it doesn't overwhelm Cross-Entropy loss.
	return (entropy / float32(len(probs.Data))) * 0.01
}

// CalculateUsageVariance calculates the variance of expert usage to discourage monopolies.
func CalculateUsageVariance(usageMap map[int]int, numExperts int, totalTokens int) float32 {
    if numExperts == 0 || totalTokens == 0 {
        return 0
    }
    targetUsage := float32(totalTokens) / float32(numExperts)
    var variance float32

    for i := 0; i < numExperts; i++ {
        count := usageMap[i]
        diff := float32(count) - targetUsage
        variance += diff * diff
    }
    
    // Normalize and scale
    return (variance / float32(totalTokens)) * 0.01
}

// PerformWeightSurgery kills weights that fall below the "signal" threshold.
func PerformWeightSurgery(expert Expert, threshold float32) int {
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
func PrintExpertHeatmap(label string, expert Expert, threshold float32) {
    params := expert.Parameters()
    if len(params) == 0 {
        return
    }

    // Use the largest parameter (usually the weight matrix)
    var weights []float32
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
    
    fmt.Printf("\n🗺️ Sparsity Heatmap [%s] (Threshold: %0.3f)\n", label, float32(threshold))
    
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

// CalculateImportanceLoss computes the penalty for unbalanced expert usage.
// probs: A 2D slice [batch_size][num_experts] from the router's softmax.
func CalculateImportanceLoss(probs [][]float32) float32 {
	if len(probs) == 0 {
		return 0
	}
	batchSize := float32(len(probs))
	numExperts := len(probs[0])
	
	// Sum probabilities for each expert across the batch
	expertSums := make([]float32, numExperts)
	for _, sample := range probs {
		for i, p := range sample {
			expertSums[i] += p
		}
	}

	// Calculate the mean probability per expert
	var loss float32
	for _, sum := range expertSums {
		meanProb := sum / batchSize
		loss += meanProb * meanProb
	}

	return loss * float32(numExperts)
}

// LogUtilization prints a visual bar chart of how much each expert is used.
func LogUtilization(gateProbs [][]float32) {
	if len(gateProbs) == 0 {
		return
	}
	numExperts := len(gateProbs[0])
	counts := make([]float32, numExperts)

	// Count "hard" selections (which expert had the highest prob)
	for _, sample := range gateProbs {
		maxIdx := 0
		for i, p := range sample {
			if p > sample[maxIdx] {
				maxIdx = i
			}
		}
		counts[maxIdx]++
	}

	fmt.Printf("\n--- Expert Utilization (Batch) ---\n")
	for i, count := range counts {
		percentage := (count / float32(len(gateProbs))) * 100
		bar := strings.Repeat("█", int(percentage/2)) // 1 block per 2%
		fmt.Printf("Expert %d: [%-50s] %.1f%%\n", i, bar, float64(percentage))
	}
}

// CalculateImportanceLossTensor is a version of CalculateImportanceLoss that takes a Tensor.
func CalculateImportanceLossTensor(probs *tensor.Tensor) float32 {
	if probs == nil || len(probs.Data) == 0 {
		return 0
	}
	numExperts := probs.Shape[len(probs.Shape)-1]
	numTokens := len(probs.Data) / numExperts
	
	expertSums := make([]float32, numExperts)
	for t := 0; t < numTokens; t++ {
		base := t * numExperts
		for e := 0; e < numExperts; e++ {
			expertSums[e] += probs.Data[base+e]
		}
	}

	var loss float32
	for _, sum := range expertSums {
		meanProb := sum / float32(numTokens)
		loss += meanProb * meanProb
	}

	return loss * float32(numExperts)
}

// LogUtilizationTensor is a version of LogUtilization that takes a Tensor.
func LogUtilizationTensor(gateProbs *tensor.Tensor) {
	if gateProbs == nil || len(gateProbs.Data) == 0 {
		return
	}
	numExperts := gateProbs.Shape[len(gateProbs.Shape)-1]
	numTokens := len(gateProbs.Data) / numExperts
	counts := make([]float32, numExperts)

	for t := 0; t < numTokens; t++ {
		base := t * numExperts
		maxIdx := 0
		for e := 1; e < numExperts; e++ {
			if gateProbs.Data[base+e] > gateProbs.Data[base+maxIdx] {
				maxIdx = e
			}
		}
		counts[maxIdx]++
	}

	fmt.Printf("\n--- Expert Utilization (Batch Tensor) ---\n")
	for i, count := range counts {
		percentage := (count / float32(numTokens)) * 100
		bar := strings.Repeat("█", int(percentage/2)) // 1 block per 2%
		fmt.Printf("Expert %d: [%-50s] %.1f%%\n", i, bar, float64(percentage))
	}
}
