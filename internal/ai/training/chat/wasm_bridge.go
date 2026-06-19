//go:build js && wasm

package chat

import (
	"syscall/js"
)

// PushToJS sends the training metrics to the JavaScript dashboard via the WASM bridge.
func (m *TrainingMetric) PushToJS() {
	js.Global().Call("updateDashboard", map[string]interface{}{
		"step":            m.Step,
		"loss":            m.Loss,
		"lb_loss":         m.LoadBalanceLoss,
		"lr":              m.LearningRate,
		"is_cooling":      m.IsCooling,
		"circuit_breaker": m.CircuitBreaker,
		"experts":         m.ActiveExperts,
	})
}
