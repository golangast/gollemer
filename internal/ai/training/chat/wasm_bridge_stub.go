//go:build !js || !wasm
package chat

// PushToJS is a no-op on platforms other than WASM.
func (m *TrainingMetric) PushToJS() {
	// Do nothing
}
