package train

import (
	"fmt"
	"math"
)

// PlateauConfig defines the policy for reacting to stagnating performance.
type PlateauConfig struct {
	Patience      int     // Epochs to wait before taking action
	Cooldown      int     // Epochs to ignore after an action is taken
	Factor        float64 // Multiplier to reduce the Learning Rate
	TempDecay     float64 // Multiplier to reduce the Router Temperature
	MinLR         float64 // Minimum allowable learning rate
}

// PlateauState tracks the training progress metrics for plateau detection.
type PlateauState struct {
	BestPPL       float64
	BadEpochs     int
	CooldownTimer int
}

// Update evaluates the current Perplexity and adjusts LR/Temperature if progress has stalled.
// It returns a status message describing the decision.
func (s *PlateauState) Update(currentPPL float64, config PlateauConfig, currentLR *float64, currentTemp *float64) string {
	if s.CooldownTimer > 0 {
		s.CooldownTimer--
		return "⏳ Cooldown active (waiting for stabilization)"
	}

	if currentPPL < s.BestPPL {
		s.BestPPL = currentPPL
		s.BadEpochs = 0
		return "✅ New best PPL achieved"
	}

	s.BadEpochs++
	if s.BadEpochs >= config.Patience {
		// Response Action: Drop Learning Rate and reduce Gate Exploration (Temperature)
		oldLR := *currentLR
		*currentLR = math.Max(config.MinLR, (*currentLR * config.Factor))
		
		// We decay the router temperature to encourage expert commitment on stagnation
		oldTemp := *currentTemp
		*currentTemp = math.Max(0.1, (*currentTemp * config.TempDecay))
		
		s.BadEpochs = 0
		s.CooldownTimer = config.Cooldown
		return fmt.Sprintf("📉 Plateau detected! Dropping LR (%.6f -> %.6f) and Router Temp (%.2f -> %.2f)", 
			oldLR, *currentLR, oldTemp, *currentTemp)
	}

	return fmt.Sprintf("⚠️ No improvement in PPL (%d/%d epochs)", s.BadEpochs, config.Patience)
}
