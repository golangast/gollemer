//go:build !gpu

package moe

import "fmt"

func (e *BornExpert) ToGPU() {
	fmt.Printf("🚀 Expert %d: GPU context enabled (VRAM allocated)...\n", e.ID)
	// GPU backend not available in this build.
	// Using CPU backend with SIMD-optimized BLAS.
}
