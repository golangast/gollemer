//go:build gpu && !wasm

package moe

// ToGPU for BornExpert is disabled because it relies on Rust-based backends.
// Please use GoffiExpert for GPU-accelerated training.
func (e *BornExpert) ToGPU() {
	// No-op to avoid CGO/Rust dependencies.
}
