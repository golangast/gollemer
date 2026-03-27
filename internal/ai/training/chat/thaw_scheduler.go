package chat

import (
	"math"
	"sync"
)

// CosineDecay calculates the next temperature based on the current progress.
func CosineDecay(step, maxSteps int, tMax, tMin float64) float64 {
	if step >= maxSteps {
		return tMin
	}
	// T_cur = T_min + 0.5 * (T_max - T_min) * (1 + cos(pi * step / maxSteps))
	ratio := float64(step) / float64(maxSteps)
	return tMin + 0.5*(tMax-tMin)*(1+math.Cos(math.Pi*ratio))
}

type ThawScheduler struct {
	mu          sync.Mutex
	CurrentStep int
	MaxSteps    int
	StartTemp   float64
	MinTemp     float64
	// LayerThresholds maps a Temp to a Layer index to "thaw"
	LayerThresholds []float64
}

// Next calculates the next temperature and returns the number of active layers.
func (s *ThawScheduler) Next() (float64, int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.CurrentStep < s.MaxSteps {
		s.CurrentStep++
	}

	// Cosine Decay calculation
	ratio := float64(s.CurrentStep) / float64(s.MaxSteps)
	temp := s.MinTemp + 0.5*(s.StartTemp-s.MinTemp)*(1+math.Cos(math.Pi*ratio))

	// Determine how many layers are "thawed" based on the current temp
	activeLayers := 0
	for _, threshold := range s.LayerThresholds {
		if temp <= threshold {
			activeLayers++
		}
	}

	return temp, activeLayers
}
