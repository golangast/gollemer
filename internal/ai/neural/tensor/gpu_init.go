package tensor

import (
	"fmt"
	"sync"

	"github.com/gogpu/wgpu"
)

var (
	gpuInstance *wgpu.Instance
	gpuDevice   *wgpu.Device
	gpuQueue    *wgpu.Queue
	gpuOnce     sync.Once
)

func initGPU() error {
	var err error
	gpuOnce.Do(func() {
		gpuInstance, err = wgpu.CreateInstance(&wgpu.InstanceDescriptor{
			Backends: wgpu.BackendsVulkan,
		})
		if err != nil {
			return
		}

		adapter, aErr := gpuInstance.RequestAdapter(&wgpu.RequestAdapterOptions{
			PowerPreference: wgpu.PowerPreferenceHighPerformance,
		})
		if aErr != nil {
			err = aErr
			return
		}

		gpuDevice, err = adapter.RequestDevice(nil)
		if err != nil {
			return
		}

		gpuQueue = gpuDevice.Queue()
		fmt.Printf("✅ GPU Accelerator Initialized: %s\n", adapter.Info().Name)
	})
	return err
}

func (t *Tensor) ToGPU() *Tensor {
	if t.Device == GPU {
		return t
	}

	if err := initGPU(); err != nil {
		fmt.Printf("⚠️ GPU Initialization failed, falling back to CPU: %v\n", err)
		return t
	}

	t.Device = GPU
	// Marks the tensor for GPU dispatch.
	// The actual buffer upload happens lazily or during dispatch.
	return t
}
