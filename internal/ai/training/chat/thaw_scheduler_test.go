package chat

import (
	"testing"
)

func TestThawScheduler(t *testing.T) {
	scheduler := &ThawScheduler{
		CurrentStep:     0,
		MaxSteps:        1000,
		StartTemp:       1.0,
		MinTemp:         0.1,
		LayerThresholds: []float32{0.8, 0.5, 0.2},
	}

	// Check initial state after 1 step
	temp, active := scheduler.Next()
	if temp >= 1.0 || active != 1 {
		t.Errorf("Expected decay to start. Got Temp: %f, Active: %d", temp, active)
	}

	// Simulate halfway point (Step 500)
	scheduler.CurrentStep = 500
	temp, active = scheduler.Next()
	if active < 1 {
		t.Errorf("At Step 500, at least one layer should be thawed. Got Temp: %f, Active: %d", temp, active)
	}
	
	// Simulate near end (Step 2000)
	scheduler.CurrentStep = 2000
	temp, active = scheduler.Next()
	if active < 3 {
		t.Errorf("Near end (Step 2000), more layers should be thawed. Got Temp: %f, Active: %d", temp, active)
	}
}
