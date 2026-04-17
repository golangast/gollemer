//go:build gpu && !wasm

package moe

import (
	"fmt"

	"github.com/born-ml/born/autodiff"
	borntensor "github.com/born-ml/born/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/backend/webgpu"
)

func (e *BornExpert) ToGPU() {
	gpuOnce.Do(func() {
		// Initialize global GPU backend once
		backend, err := webgpu.New()
		if err != nil {
			fmt.Printf("⚠️  Expert %d: Failed to initialize GPU backend: %v\n", e.ID, err)
			return
		}
		gpuBackend = autodiff.New(borntensor.Backend(backend))
		fmt.Printf("🚀 Expert %d: GPU context enabled (WebGPU/goffi initialized)...\n", e.ID)
	})

	if gpuBackend != nil {
		// Replace the expert's CPU backend with the GPU backend
		e.backend = gpuBackend

		// Re-initialize layers on GPU by moving weights
		// Born-ML handles weight migration when a layer is used on a new backend
		// but we can force it here by re-creating or syncing.
		e.SyncParameters()
		fmt.Printf("✅ Expert %d: Migrated to GPU backend\n", e.ID)
	}
}
