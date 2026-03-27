package chat

import (
	"testing"
)

func TestThawScheduler(t *testing.T) {
	scheduler := &ThawScheduler{
		CurrentStep:     0,
		MaxSteps:        100,
		StartTemp:       1.0,
		MinTemp:         0.1,
		LayerThresholds: []float64{0.8, 0.5, 0.2},
	}

	// Check initial state
	temp, active := scheduler.Next()
	if temp >= 1.0 || active != 0 {
		t.Errorf("Expected decay to start. Got Temp: %f, Active: %d", temp, active)
	}

	// Simulate halfway point
	scheduler.CurrentStep = 50
	temp, active = scheduler.Next()
	// Cosine at 50/100 (pi/2) should be roughly the midpoint
	if active < 1 {
		t.Errorf("At midpoint (Step 50), at least one layer should be thawed. Got Temp: %f, Active: %d", temp, active)
	}
	
	// Simulate near end
	scheduler.CurrentStep = 95
	temp, active = scheduler.Next()
	if active < 3 {
		t.Errorf("Near end (Step 95), more layers should be thawed. Got Temp: %f, Active: %d", temp, active)
	}
}
