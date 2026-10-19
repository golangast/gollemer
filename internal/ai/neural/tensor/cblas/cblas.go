//go:build openblas

package cblas

/*
#cgo LDFLAGS: -lopenblas
#include <cblas.h>
*/
import "C"
import "unsafe"

// Sgemm calls OpenBLAS cblas_sgemm directly via CGO.
// Row-major, NoTrans/NoTrans: C = A * B
func Sgemm(aData, bData, cData []float32, m, n, k int) {
	if len(aData) == 0 || len(bData) == 0 || len(cData) == 0 {
		return
	}
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
