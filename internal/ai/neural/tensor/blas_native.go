//go:build cgo && openblas

package tensor

/*
#cgo LDFLAGS: -lopenblas
#include <cblas.h>
*/
import "C"
import "unsafe"

// NativeMatMul calls OpenBLAS sgemm directly via CGO.
func NativeMatMul(aData, bData, cData []float32, m, n, k int) {
	C.cblas_sgemm(
		C.CblasRowMajor, C.CblasNoTrans, C.CblasNoTrans,
		C.int(m), C.int(n), C.int(k),
		1.0,
		(*C.float)(unsafe.Pointer(&aData[0])), C.int(k),
		(*C.float)(unsafe.Pointer(&bData[0])), C.int(n),
		0.0,
		(*C.float)(unsafe.Pointer(&cData[0])), C.int(n),
	)
}
