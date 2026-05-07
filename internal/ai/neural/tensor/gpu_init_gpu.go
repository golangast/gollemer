//go:build gpu

package tensor

// ToGPU moves the tensor to GPU memory using the Native OpenCL/Goffi backend.
func (t *Tensor) ToGPU() *Tensor {
	if t.Device == GPU && t.gpuData != nil {
		return t // Already on GPU
	}
	t.openclUpload()
	return t
}

// ToCPU moves the tensor back to host memory.
func (t *Tensor) ToCPU() *Tensor {
	if t.Device == CPU {
		return t
	}
	t.openclDownload()
	return t
}

// Release explicitly frees the GPU buffer.
func (t *Tensor) Release() {
	if t.Device == GPU {
		t.openclRelease()
	}
}

// SyncToDevice ensures the GPU data matches the host data.
func (t *Tensor) SyncToDevice() {
	if t.Device != GPU || t.gpuData == nil {
		return
	}
	// In simplicity: re-upload
	t.openclUpload()
}

// DispatchGPUMatMul dispatches matrix multiplication to the Native OpenCL/Goffi backend.
func DispatchGPUMatMul(a, b *Tensor) (*Tensor, error) {
	return DispatchOpenCLMatMul(a, b)
}
