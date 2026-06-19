package chat

import (
	"math"
	"sync"
)

// ThawScheduler gradually unfreezes model layers as training progresses.
type ThawScheduler struct {
	mu         sync.Mutex
	Step       int
	MaxSteps   int
	StartTemp  float32
	MinTemp    float32
	Thresholds []float32 // Thresholds map a temp to a layer index to thaw.
}

// Next advances the scheduler and returns the new temperature and number of active layers.
func (s *ThawScheduler) Next() (float32, int) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.Step < s.MaxSteps {
		s.Step++
	}

	// Exponential decay: temp = start * e^(-rate * step)
	const rate = 0.001
	temp := s.StartTemp * float32(math.Exp(-rate*float64(s.Step)))
	if temp < s.MinTemp {
		temp = s.MinTemp
	}

	active := 0
	for _, t := range s.Thresholds {
		if temp <= t {
			active++
		}
	}

	if active == 0 {
		active = 1
	}

	return temp, active
}
