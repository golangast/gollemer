package chat

import (
	"math"
	"sync"
)

// CosineDecay calculates the next temperature based on the current progress.
func CosineDecay(step, maxSteps int, tMax, tMin float32) float32 {
	if step >= maxSteps {
		return tMin
	}
	// T_cur = T_min + 0.5 * (T_max - T_min) * (1 + cos(pi * step / maxSteps))
	ratio := float32(step) / float32(maxSteps)
	return tMin + 0.5*(tMax-tMin)*(1+float32(math.Cos(math.Pi*float64(ratio))))
}

type ThawScheduler struct {
	mu          sync.Mutex
	CurrentStep int
	MaxSteps    int
	StartTemp   float32
	MinTemp     float32
	// LayerThresholds maps a Temp to a Layer index to "thaw"
	LayerThresholds []float32
}

// Next calculates the next temperature using exponential decay and returns the number of active layer clusters.
func (s *ThawScheduler) Next() (float32, int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.CurrentStep < s.MaxSteps {
		s.CurrentStep++
	}

	// Exponential Decay calculation: Temp = Initial * e^(-decayRate * step)
	// decayRate 0.001 ensures we cool down over the first ~1k steps.
	const decayRate = 0.001
	temp := s.StartTemp * float32(math.Exp(-decayRate*float64(s.CurrentStep)))
	if temp < s.MinTemp {
		temp = s.MinTemp
	}

	// Layer clusters are thawed as the network "cools" and becomes more certain.
	// We reverse the logic here: smaller temp means more experts are thawed (matured).
	activeLayers := 0
	for _, threshold := range s.LayerThresholds {
		if temp <= threshold {
			activeLayers++
		}
	}
	// Sanity floor: always at least 2 Experts (1 Cluster)
	if activeLayers == 0 { activeLayers = 1 }

	return temp, activeLayers
}
