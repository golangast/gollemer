//go:build openblas

package tensor

import "github.com/golangast/gollemer/internal/ai/neural/tensor/cblas"

// nativeBLASAvailable is true when building with the openblas tag.
const nativeBLASAvailable = true

// matMulDispatch routes to OpenBLAS for large matrices where it wins,
// otherwise uses the hand-tuned AVX2 Go kernel.
//
// Threshold based on benchmarks (2026-10-18):
// - N < 2048: AVX2 is faster (CGO overhead dominates for skinny matrices)
// - N >= 2048: OpenBLAS is 1.7x-2.6x faster (compute dominates)
// The M*N*K threshold was removed: TrainGocode (N=512, M*N*K=2.5e8)
// was slower on OpenBLAS (2.57ms vs 2.2ms AVX2). N is the reliable signal.
func matMulDispatch(a, b, res []float32, m, n, k int) {
	// Use OpenBLAS for wide output projections (vocab layers)
	if n >= 2048 {
		cblas.Sgemm(a, b, res, m, n, k)
		return
	}
	// Otherwise use hand-tuned AVX2 (faster for small/medium)
	MatMulRaw(a, b, res, m, n, k)
}
