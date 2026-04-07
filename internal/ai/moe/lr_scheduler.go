package moe

import "fmt"

type LRScheduler struct {
	CurrentLR      float32
	DecayFactor    float32 // e.g., 0.5 (halves the LR)
	Patience       int     // How many epochs to wait before dropping
	FailCount      int
	BestPerplexity float32
	MinLR          float32
}

func (s *LRScheduler) Update(currentPPL float32) float32 {
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

// ApplyStepDecay reduces the learning rate by a factor every N epochs.
func ApplyStepDecay(currentLR float32, epoch int, stepSize int, gamma float32) float32 {
	if epoch > 0 && epoch%stepSize == 0 {
		return currentLR * gamma
	}
	return currentLR
}
