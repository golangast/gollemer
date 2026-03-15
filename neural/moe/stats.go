package moe

import (
	"fmt"
	"math"
	"github.com/golangast/gollemer/neural/tensor"
)

// PrintExpertWeightHistogram provides a quick visual of the "health" of an Expert's brain.
// If you see a giant spike at 0, your expert is "brain dead."
// If you see spikes at the far left and right, it's over-saturated.
func PrintExpertWeightHistogram(label string, expert Expert) {
	const bins = 10
	var counts [bins]int
	minVal, maxVal := -1.0, 1.0 // Expected range

	params := expert.Parameters()
	if len(params) == 0 {
		return
	}

	totalWeights := 0
	for _, p := range params {
		totalWeights += len(p.Data)
		for _, w := range p.Data {
			idx := int((w - minVal) / (maxVal - minVal) * float64(bins))
			if idx >= 0 && idx < bins {
				counts[idx]++
			}
		}
	}

	if totalWeights == 0 {
		return
	}

	fmt.Printf("\n📈 Weight Dist [%s]: ", label)
	for _, c := range counts {
		barLen := (c * 20) / totalWeights
		bar := ""
		for i := 0; i < barLen; i++ {
			bar += "█"
		}
		fmt.Print(bar + "░")
	}
	fmt.Println()
}

// PrintLayerWeightHistogram visualizes the health of an entire MoE layer.
func PrintLayerWeightHistogram(label string, layer *MoELayer) {
	const bins = 10
	var counts [bins]int
	minVal, maxVal := -1.0, 1.0

	params := layer.Parameters()
	totalWeights := 0
	for _, p := range params {
		totalWeights += len(p.Data)
		for _, w := range p.Data {
			idx := int((w - minVal) / (maxVal - minVal) * float64(bins))
			if idx >= 0 && idx < bins {
				counts[idx]++
			}
		}
	}

	if totalWeights == 0 {
		return
	}

	fmt.Printf("\n📈 Layer Weight Dist [%s]: ", label)
	for _, c := range counts {
		barLen := (c * 20) / totalWeights
		bar := ""
		for i := 0; i < barLen; i++ {
			bar += "█"
		}
		fmt.Print(bar + "░")
	}
	fmt.Println()
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
