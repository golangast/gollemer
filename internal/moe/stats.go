package moe

import (
	"fmt"
	"math"
	"strings" // Added for strings.Repeat
	"github.com/golangast/gollemer/neural/tensor"
)

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
