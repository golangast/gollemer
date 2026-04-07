package train

import "math"

type Annealer struct {
	StartTemp float32
	MinTemp   float32
	Decay     float32
	WarmUp    int // Number of epochs to stay at StartTemp
}

// GetTemp calculates the temperature for the current epoch
func (a *Annealer) GetTemp(epoch int) float32 {
	// Phase 1: Warm-up (Hold temperature constant)
	if epoch < a.WarmUp {
		return a.StartTemp
	}

	// Phase 2: Exponential Decay
	// Subtract warm-up epochs so decay starts from the peak
	decayEpoch := float64(epoch - a.WarmUp)
	temp := float32(float64(a.StartTemp) * math.Pow(float64(a.Decay), decayEpoch))

	if temp < a.MinTemp {
		return a.MinTemp
	}
	return temp
}
