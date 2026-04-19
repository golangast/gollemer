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
	libOpenBLAS   unsafe.Pointer
	procSgemm     unsafe.Pointer
	cifSgemm      types.CallInterface
	blasOnce      sync.Once
	blasAvailable bool
)

func initOpenBLAS() {
	fmt.Fprintf(os.Stderr, "🔍 Goffi: Starting Native CPU Handshake (OpenBLAS)...")
	var err error
	paths := []string{
		"/usr/lib/x86_64-linux-gnu/libopenblas.so.3",
		"libopenblas.so.3",
		"libopenblas.so",
	}

	for _, path := range paths {
		libOpenBLAS, err = ffi.LoadLibrary(path)
		if err == nil {
			fmt.Fprintf(os.Stderr, "✅ Goffi: Loaded OpenBLAS from %s\n", path)
			break
		}
	}

	if libOpenBLAS == nil {
		fmt.Fprintln(os.Stderr, "❌ Goffi: Failed to load OpenBLAS library (all paths failed)")
		return
	}

	procSgemm, err = ffi.GetSymbol(libOpenBLAS, "cblas_sgemm")
	if err != nil || procSgemm == nil {
		fmt.Fprintf(os.Stderr, "❌ Goffi: Failed to find cblas_sgemm symbol: %v\n", err)
		return
	}

	// Prepare CIF for cblas_sgemm. 
	i32 := types.SInt32TypeDescriptor
	f32 := types.FloatTypeDescriptor
	ptr := types.PointerTypeDescriptor
	
	err = ffi.PrepareCallInterface(&cifSgemm, types.DefaultCall, types.VoidTypeDescriptor, []*types.TypeDescriptor{
		i32, i32, i32, i32, i32, i32, f32, ptr, i32, ptr, i32, f32, ptr, i32,
	})
	
	if err == nil {
		// Manual test call to avoid sync.Once deadlock
		tA := []float32{1, 2, 3, 4}; tB := []float32{5, 6, 7, 8}; tC := make([]float32, 4)
		order := uintptr(101); tA_ := uintptr(111); tB_ := uintptr(111); M := uintptr(2); N := uintptr(2); K := uintptr(2)
		alpha := float32(1.0); beta := float32(0.0)
		ld := uintptr(2)
		uA := uintptr(unsafe.Pointer(&tA[0])); uB := uintptr(unsafe.Pointer(&tB[0])); uC := uintptr(unsafe.Pointer(&tC[0]))
		
		errTest := ffi.CallFunction(&cifSgemm, procSgemm, nil, []unsafe.Pointer{
			unsafe.Pointer(&order), unsafe.Pointer(&tA_), unsafe.Pointer(&tB_),
			unsafe.Pointer(&M), unsafe.Pointer(&N), unsafe.Pointer(&K),
			unsafe.Pointer(&alpha), unsafe.Pointer(&uA), unsafe.Pointer(&ld),
			unsafe.Pointer(&uB), unsafe.Pointer(&ld), unsafe.Pointer(&beta),
			unsafe.Pointer(&uC), unsafe.Pointer(&ld),
		})
		
		if errTest == nil && tC[0] == 19.0 {
			blasAvailable = true
			fmt.Fprintf(os.Stderr, "🚀 Goffi: OpenBLAS CPU Backend Ready (Test Passed)\n")
		} else {
			fmt.Fprintf(os.Stderr, "⚠️  Goffi: OpenBLAS Test Failed (got %.1f), falling back to Pure-Go\n", tC[0])
			blasAvailable = false
		}
	} else {
		fmt.Fprintf(os.Stderr, "❌ Goffi: Failed to prepare CIF: %v\n", err)
	}
}

// GoffiMatMul performs a high-performance matrix multiplication using OpenBLAS via FFI.
func GoffiMatMul(a, b, c []float32, m, n, k int) error {
	if !blasAvailable {
		blasOnce.Do(initOpenBLAS)
		if !blasAvailable {
			return fmt.Errorf("OpenBLAS unavailable")
		}
	}

	if len(a) == 0 || len(b) == 0 || len(c) == 0 {
		return nil
	}

	// 1. BLAS parameters 
	order := uintptr(101)  // CblasRowMajor
	tA := uintptr(111)     // CblasNoTrans
	tB := uintptr(111)     // CblasNoTrans
	M := uintptr(m); N := uintptr(n); K := uintptr(k)
	alpha := float32(1.0); beta := float32(0.0)
	lda := uintptr(k); ldb := uintptr(n); ldc := uintptr(n)

	// 2. Capture data addresses into uintptr (explicit 8-byte storage)
	uA := uintptr(unsafe.Pointer(&a[0]))
	uB := uintptr(unsafe.Pointer(&b[0]))
	uC := uintptr(unsafe.Pointer(&c[0]))

	// 3. Execute synchronous FFI call
	err := ffi.CallFunction(&cifSgemm, procSgemm, nil, []unsafe.Pointer{
		unsafe.Pointer(&order),
		unsafe.Pointer(&tA),
		unsafe.Pointer(&tB),
		unsafe.Pointer(&M),
		unsafe.Pointer(&N),
		unsafe.Pointer(&K),
		unsafe.Pointer(&alpha),
		unsafe.Pointer(&uA), 
		unsafe.Pointer(&lda),
		unsafe.Pointer(&uB),
		unsafe.Pointer(&ldb),
		unsafe.Pointer(&beta),
		unsafe.Pointer(&uC),
		unsafe.Pointer(&ldc),
	})

	runtime.KeepAlive(a)
	runtime.KeepAlive(b)
	runtime.KeepAlive(c)

	if err != nil {
		return err
	}

	// 4. Sanity Check - Only trigger if major parts of the matrix are zeroed despite non-zero energy
	if len(c) > 1 && c[0] == 0 && c[len(c)-1] == 0 {
		var normA float32
		for i := 0; i < 8 && i < len(a); i++ { normA += a[i] * a[i] }
		if normA > 1e-12 {
			if c[len(c)/2] == 0 {
				return fmt.Errorf("GoffiMatMul returned zeros for non-zero input")
			}
		}
	}

	return nil
}
