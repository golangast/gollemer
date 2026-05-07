//go:build !wasm

package tensor

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/go-webgpu/goffi/ffi"
	"github.com/go-webgpu/goffi/types"
)

var (
	libOpenCL unsafe.Pointer
	clOnce    sync.Once
	clReady   bool

	// OpenCL State
	clContext      uintptr
	clQueue        uintptr
	clProgram      uintptr
	clKernelMatMul uintptr

	// Symbols
	procGetPlatformIDs          unsafe.Pointer
	procGetDeviceIDs            unsafe.Pointer
	procCreateContext           unsafe.Pointer
	procCreateCommandQueue      unsafe.Pointer
	procCreateProgramWithSource unsafe.Pointer
	procBuildProgram            unsafe.Pointer
	procCreateKernel            unsafe.Pointer
	procCreateBuffer            unsafe.Pointer
	procEnqueueWriteBuffer      unsafe.Pointer
	procEnqueueReadBuffer       unsafe.Pointer
	procEnqueueNDRangeKernel    unsafe.Pointer
	procSetKernelArg            unsafe.Pointer
	procFinish                  unsafe.Pointer
	procReleaseMemObject        unsafe.Pointer

	// CIFs
	cifGetPlatformIDs     types.CallInterface
	cifGetDeviceIDs       types.CallInterface
	cifCreateContext      types.CallInterface
	cifCreateCommandQueue types.CallInterface
	cifCreateProgSource   types.CallInterface
	cifBuildProgram       types.CallInterface
	cifCreateKernel       types.CallInterface
	cifCreateBuffer       types.CallInterface
	cifRWBuffer           types.CallInterface
	cifNDRange            types.CallInterface
	cifSetArg             types.CallInterface
	cifFinish             types.CallInterface
	cifRelease            types.CallInterface
)

const matmulSource = `
__kernel void matmul(
    const int M, const int N, const int K,
    __global const float* A,
    __global const float* B,
    __global float* C)
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}
`

func initOpenCL() {
	fmt.Fprintln(os.Stderr, "🔍 Goffi: Starting Native OpenCL Handshake (No CGO, No Rust)...")
	var err error
	libOpenCL, err = ffi.LoadLibrary("libOpenCL.so.1")
	if err != nil {
		fmt.Fprintf(os.Stderr, "⚠️  OpenCL: Failed to load libOpenCL.so.1: %v\n", err)
		return
	}

	// Load Symbols
	procGetPlatformIDs, _ = ffi.GetSymbol(libOpenCL, "clGetPlatformIDs")
	procGetDeviceIDs, _ = ffi.GetSymbol(libOpenCL, "clGetDeviceIDs")
	procCreateContext, _ = ffi.GetSymbol(libOpenCL, "clCreateContext")
	procCreateCommandQueue, _ = ffi.GetSymbol(libOpenCL, "clCreateCommandQueue")
	procCreateProgramWithSource, _ = ffi.GetSymbol(libOpenCL, "clCreateProgramWithSource")
	procBuildProgram, _ = ffi.GetSymbol(libOpenCL, "clBuildProgram")
	procCreateKernel, _ = ffi.GetSymbol(libOpenCL, "clCreateKernel")
	procCreateBuffer, _ = ffi.GetSymbol(libOpenCL, "clCreateBuffer")
	procEnqueueWriteBuffer, _ = ffi.GetSymbol(libOpenCL, "clEnqueueWriteBuffer")
	procEnqueueReadBuffer, _ = ffi.GetSymbol(libOpenCL, "clEnqueueReadBuffer")
	procEnqueueNDRangeKernel, _ = ffi.GetSymbol(libOpenCL, "clEnqueueNDRangeKernel")
	procSetKernelArg, _ = ffi.GetSymbol(libOpenCL, "clSetKernelArg")
	procFinish, _ = ffi.GetSymbol(libOpenCL, "clFinish")
	procReleaseMemObject, _ = ffi.GetSymbol(libOpenCL, "clReleaseMemObject")

	if procGetPlatformIDs == nil || procGetDeviceIDs == nil || procCreateContext == nil || procCreateBuffer == nil {
		return
	}

	// Internal zero value helper for NULL pointers
	var zero uintptr = 0

	// Prepare CIFs
	pDescriptors := types.PointerTypeDescriptor
	u32 := types.UInt32TypeDescriptor
	u64 := types.UInt64TypeDescriptor
	i32 := types.SInt32TypeDescriptor
	
	ffi.PrepareCallInterface(&cifGetPlatformIDs, types.DefaultCall, i32, []*types.TypeDescriptor{u32, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifGetDeviceIDs, types.DefaultCall, i32, []*types.TypeDescriptor{pDescriptors, u64, u32, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifCreateContext, types.DefaultCall, pDescriptors, []*types.TypeDescriptor{pDescriptors, u32, pDescriptors, pDescriptors, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifCreateCommandQueue, types.DefaultCall, pDescriptors, []*types.TypeDescriptor{pDescriptors, pDescriptors, u64, pDescriptors})
	ffi.PrepareCallInterface(&cifCreateProgSource, types.DefaultCall, pDescriptors, []*types.TypeDescriptor{pDescriptors, u32, pDescriptors, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifBuildProgram, types.DefaultCall, i32, []*types.TypeDescriptor{pDescriptors, u32, pDescriptors, pDescriptors, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifCreateKernel, types.DefaultCall, pDescriptors, []*types.TypeDescriptor{pDescriptors, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifCreateBuffer, types.DefaultCall, pDescriptors, []*types.TypeDescriptor{pDescriptors, u64, u64, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifRWBuffer, types.DefaultCall, i32, []*types.TypeDescriptor{pDescriptors, pDescriptors, u32, u64, u64, pDescriptors, u32, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifNDRange, types.DefaultCall, i32, []*types.TypeDescriptor{pDescriptors, pDescriptors, u32, pDescriptors, pDescriptors, pDescriptors, u32, pDescriptors, pDescriptors})
	ffi.PrepareCallInterface(&cifSetArg, types.DefaultCall, i32, []*types.TypeDescriptor{pDescriptors, u32, u64, pDescriptors})
	ffi.PrepareCallInterface(&cifFinish, types.DefaultCall, i32, []*types.TypeDescriptor{pDescriptors})
	ffi.PrepareCallInterface(&cifRelease, types.DefaultCall, i32, []*types.TypeDescriptor{pDescriptors})

	var clRet int32
	var numPlatforms uint32
	pNumPlatforms := uintptr(unsafe.Pointer(&numPlatforms))

	// 1. Get Platform
	platformCount := uint32(1)
	ffi.CallFunction(&cifGetPlatformIDs, procGetPlatformIDs, unsafe.Pointer(&clRet), []unsafe.Pointer{
		unsafe.Pointer(&platformCount), unsafe.Pointer(&zero), unsafe.Pointer(&pNumPlatforms),
	})
	if numPlatforms == 0 { 
		fmt.Fprintln(os.Stderr, "⚠️  OpenCL: No platforms found.")
		return 
	}
	platforms := make([]uintptr, numPlatforms)
	pPlatforms := uintptr(unsafe.Pointer(&platforms[0]))
	ffi.CallFunction(&cifGetPlatformIDs, procGetPlatformIDs, unsafe.Pointer(&clRet), []unsafe.Pointer{
		unsafe.Pointer(&numPlatforms), unsafe.Pointer(&pPlatforms), unsafe.Pointer(&zero),
	})
	activePlatform := platforms[0]
	fmt.Fprintf(os.Stderr, "✅ OpenCL: Found platform %v\n", activePlatform)

	// 2. Get Device (GPU)
	var device uintptr
	pDevice := uintptr(unsafe.Pointer(&device))
	var numDevices uint32
	pNumDevices := uintptr(unsafe.Pointer(&numDevices))
	clDeviceTypeGPU := uint64(1 << 2) // CL_DEVICE_TYPE_GPU
	deviceCount := uint32(1)
	ffi.CallFunction(&cifGetDeviceIDs, procGetDeviceIDs, unsafe.Pointer(&clRet), []unsafe.Pointer{
		unsafe.Pointer(&activePlatform), unsafe.Pointer(&clDeviceTypeGPU), unsafe.Pointer(&deviceCount), unsafe.Pointer(&pDevice), unsafe.Pointer(&pNumDevices),
	})
	if device == 0 { 
		fmt.Fprintln(os.Stderr, "⚠️  OpenCL: No GPU device found on this platform.")
		return 
	}
	fmt.Fprintf(os.Stderr, "✅ OpenCL: Found GPU device %v\n", device)

	// 3. Create Context
	ffi.CallFunction(&cifCreateContext, procCreateContext, unsafe.Pointer(&clContext), []unsafe.Pointer{
		unsafe.Pointer(&zero), unsafe.Pointer(&deviceCount), unsafe.Pointer(&device), unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&clRet),
	})

	// 4. Create Queue
	ffi.CallFunction(&cifCreateCommandQueue, procCreateCommandQueue, unsafe.Pointer(&clQueue), []unsafe.Pointer{
		unsafe.Pointer(&clContext), unsafe.Pointer(&device), unsafe.Pointer(&zero), unsafe.Pointer(&clRet),
	})

	// 5. Build Program
	src := matmulSource
	srcBytes := []byte(src)
	srcPtr := uintptr(unsafe.Pointer(&srcBytes[0]))
	srcLen := uint64(len(srcBytes))
	count1 := uint32(1)
	ffi.CallFunction(&cifCreateProgSource, procCreateProgramWithSource, unsafe.Pointer(&clProgram), []unsafe.Pointer{
		unsafe.Pointer(&clContext), unsafe.Pointer(&count1), unsafe.Pointer(&srcPtr), unsafe.Pointer(&srcLen), unsafe.Pointer(&clRet),
	})
	
	ffi.CallFunction(&cifBuildProgram, procBuildProgram, unsafe.Pointer(&clRet), []unsafe.Pointer{
		unsafe.Pointer(&clProgram), unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&zero),
	})

	// 6. Create Kernel
	kName := "matmul\x00"
	kNamePtr := uintptr(unsafe.Pointer(&[]byte(kName)[0]))
	ffi.CallFunction(&cifCreateKernel, procCreateKernel, unsafe.Pointer(&clKernelMatMul), []unsafe.Pointer{
		unsafe.Pointer(&clProgram), unsafe.Pointer(&kNamePtr), unsafe.Pointer(&clRet),
	})

	if clKernelMatMul != 0 {
		clReady = true
		fmt.Fprintf(os.Stderr, "🚀 Goffi: OpenCL GPU Pipeline Ready (Context: 0x%x)\n", clContext)
	}
}

func DispatchOpenCLMatMul(a, b *Tensor) (*Tensor, error) {
	clOnce.Do(initOpenCL)
	if !clReady {
		return nil, fmt.Errorf("OpenCL not available")
	}

	m, k, n := int32(a.Shape[0]), int32(a.Shape[1]), int32(b.Shape[1])
	res := NewTensor([]int{int(m), int(n)}, nil, a.RequiresGrad || b.RequiresGrad)
	res.Device = GPU

	// Ensure tensors are on GPU
	a.ToGPU()
	b.ToGPU()
	res.ToGPU()

	bufA := a.gpuData.(uintptr)
	bufB := b.gpuData.(uintptr)
	bufC := res.gpuData.(uintptr)

	// Set Args
	s32 := uint64(4)
	sPtr := uint64(8)
	arg0 := uint32(0)
	arg1 := uint32(1)
	arg2 := uint32(2)
	arg3 := uint32(3)
	arg4 := uint32(4)
	arg5 := uint32(5)
	
	ffi.CallFunction(&cifSetArg, procSetKernelArg, nil, []unsafe.Pointer{unsafe.Pointer(&clKernelMatMul), unsafe.Pointer(&arg0), unsafe.Pointer(&s32), unsafe.Pointer(&m)})
	ffi.CallFunction(&cifSetArg, procSetKernelArg, nil, []unsafe.Pointer{unsafe.Pointer(&clKernelMatMul), unsafe.Pointer(&arg1), unsafe.Pointer(&s32), unsafe.Pointer(&n)})
	ffi.CallFunction(&cifSetArg, procSetKernelArg, nil, []unsafe.Pointer{unsafe.Pointer(&clKernelMatMul), unsafe.Pointer(&arg2), unsafe.Pointer(&s32), unsafe.Pointer(&k)})
	ffi.CallFunction(&cifSetArg, procSetKernelArg, nil, []unsafe.Pointer{unsafe.Pointer(&clKernelMatMul), unsafe.Pointer(&arg3), unsafe.Pointer(&sPtr), unsafe.Pointer(&bufA)})
	ffi.CallFunction(&cifSetArg, procSetKernelArg, nil, []unsafe.Pointer{unsafe.Pointer(&clKernelMatMul), unsafe.Pointer(&arg4), unsafe.Pointer(&sPtr), unsafe.Pointer(&bufB)})
	ffi.CallFunction(&cifSetArg, procSetKernelArg, nil, []unsafe.Pointer{unsafe.Pointer(&clKernelMatMul), unsafe.Pointer(&arg5), unsafe.Pointer(&sPtr), unsafe.Pointer(&bufC)})

	// Dispatch
	globalWorkSize := []uint64{uint64(m), uint64(n)}
	pGlobalWorkSize := uintptr(unsafe.Pointer(&globalWorkSize[0]))
	workDim := uint32(2)
	var zero uintptr = 0
	
	ffi.CallFunction(&cifNDRange, procEnqueueNDRangeKernel, nil, []unsafe.Pointer{
		unsafe.Pointer(&clQueue), unsafe.Pointer(&clKernelMatMul), unsafe.Pointer(&workDim),
		unsafe.Pointer(&zero), unsafe.Pointer(&pGlobalWorkSize), unsafe.Pointer(&zero),
		unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&zero),
	})

	// Finish
	ffi.CallFunction(&cifFinish, procFinish, nil, []unsafe.Pointer{unsafe.Pointer(&clQueue)})

	return res, nil
}

// Global Help for ToGPU in OpenCL mode
func (t *Tensor) openclUpload() {
	clOnce.Do(initOpenCL)
	if !clReady { return }

	size := uint64(len(t.Data) * 4)
	flags := uint64(1 << 2) // CL_MEM_READ_WRITE
	
	var clRet int32
	var buf uintptr
	var zero uintptr = 0

	// Release previous buffer if it exists to prevent memory leaks
	if t.gpuData != nil {
		oldBuf := t.gpuData.(uintptr)
		ffi.CallFunction(&cifRelease, procReleaseMemObject, nil, []unsafe.Pointer{unsafe.Pointer(&oldBuf)})
		t.gpuData = nil
	}

	ffi.CallFunction(&cifCreateBuffer, procCreateBuffer, unsafe.Pointer(&buf), []unsafe.Pointer{
		unsafe.Pointer(&clContext), unsafe.Pointer(&flags), unsafe.Pointer(&size), unsafe.Pointer(&zero), unsafe.Pointer(&clRet),
	})
	
	if buf != 0 {
		ptr := uintptr(unsafe.Pointer(&t.Data[0]))
		blocking := uint32(1)
		offset := uint64(0)
		ffi.CallFunction(&cifRWBuffer, procEnqueueWriteBuffer, nil, []unsafe.Pointer{
			unsafe.Pointer(&clQueue), unsafe.Pointer(&buf), unsafe.Pointer(&blocking),
			unsafe.Pointer(&offset), unsafe.Pointer(&size), unsafe.Pointer(&ptr),
			unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&zero),
		})
		t.gpuData = buf
		t.Device = GPU

		// 🛡️ Proactive Memory Management: Register a finalizer to ensure that
		// when Go's GC collects this Tensor, the OpenCL buffer is also freed.
		runtime.SetFinalizer(t, func(obj *Tensor) {
			obj.Release()
		})
	}
}

func (t *Tensor) openclDownload() {
	if t.gpuData == nil {
		return
	}
	buf := t.gpuData.(uintptr)
	size := uint64(len(t.Data) * 4)
	ptr := uintptr(unsafe.Pointer(&t.Data[0]))
	blocking := uint32(1)
	offset := uint64(0)
	var zero uintptr = 0

	ffi.CallFunction(&cifRWBuffer, procEnqueueReadBuffer, nil, []unsafe.Pointer{
		unsafe.Pointer(&clQueue), unsafe.Pointer(&buf), unsafe.Pointer(&blocking),
		unsafe.Pointer(&offset), unsafe.Pointer(&size), unsafe.Pointer(&ptr),
		unsafe.Pointer(&zero), unsafe.Pointer(&zero), unsafe.Pointer(&zero),
	})

	// Release the GPU buffer after successful download
	t.openclRelease()
}

// openclRelease explicitly frees the OpenCL memory object.
func (t *Tensor) openclRelease() {
	if t.gpuData == nil {
		return
	}
	buf := t.gpuData.(uintptr)
	ffi.CallFunction(&cifRelease, procReleaseMemObject, nil, []unsafe.Pointer{unsafe.Pointer(&buf)})
	t.gpuData = nil
	t.Device = CPU
	// Clear the finalizer since we've manually released it
	runtime.SetFinalizer(t, nil)
}
