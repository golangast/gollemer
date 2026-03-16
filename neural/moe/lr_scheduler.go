package moe

import "fmt"

type LRScheduler struct {
	CurrentLR      float64
	DecayFactor    float64 // e.g., 0.5 (halves the LR)
	Patience       int     // How many epochs to wait before dropping
	FailCount      int
	BestPerplexity float64
	MinLR          float64
}

func (s *LRScheduler) Update(currentPPL float64) float64 {
	if currentPPL < s.BestPerplexity || s.BestPerplexity == 0 {
		s.BestPerplexity = currentPPL
		s.FailCount = 0
		return s.CurrentLR
	}

	s.FailCount++
	if s.FailCount >= s.Patience {
		newLR := s.CurrentLR * s.DecayFactor
		if newLR >= s.MinLR {
			fmt.Printf("📉 No improvement for %d epochs. Decaying LR: %f -> %f\n", s.FailCount, s.CurrentLR, newLR)
			s.CurrentLR = newLR
		}
		s.FailCount = 0 // Reset patience after decay
	}
	return s.CurrentLR
}
