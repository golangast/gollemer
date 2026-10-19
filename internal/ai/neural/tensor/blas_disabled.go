//go:build !openblas

package tensor

// nativeBLASAvailable is false when building without the openblas tag.
// The pure-Go MatMul implementation is used.
const nativeBLASAvailable = false

// matMulDispatch routes to the pure-Go implementation when OpenBLAS is unavailable.
func matMulDispatch(a, b, res []float32, m, n, k int) {
	MatMulRaw(a, b, res, m, n, k)
}
