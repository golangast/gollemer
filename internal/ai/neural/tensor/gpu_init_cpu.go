//go:build !gpu

package tensor

import (
	"fmt"
	"sync"
)

var (
	gpuOnce sync.Once
)

func initGPU() error {
	// GPU initialization is handled by the compute platform
	// In goffi mode, we use direct library loading.
	return nil
}

func (t *Tensor) ToGPU() *Tensor {
	// CPU-only mode: no-op.
	return t
}

func (t *Tensor) ToCPU() *Tensor {
	t.Device = CPU
	return t
}

func (t *Tensor) Release() {
	// CPU-only mode: no-op.
}

func (t *Tensor) SyncToDevice() {
	// CPU-only mode: no-op.
}

func DispatchGPUMatMul(a, b *Tensor) (*Tensor, error) {
	return nil, fmt.Errorf("GPU support not compiled")
}
