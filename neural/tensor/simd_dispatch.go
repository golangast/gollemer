package tensor

// This file defines the entry points for SIMD-accelerated operations.
// The actual implementations are in simd_generic.go and simd_arch.go.

import "fmt"

func AddVectors(a, b, res []float64) {
	vecAdd(a, b, res)
}

func SubVectors(a, b, res []float64) {
	vecSub(a, b, res)
}

func MulVectors(a, b, res []float64) {
	vecMul(a, b, res)
}

func DivVectors(a, b, res []float64) {
	vecDiv(a, b, res)
}

func MulScalar(a []float64, scalar float64, res []float64) {
	vecMulScalar(a, scalar, res)
}

func DivScalar(a []float64, scalar float64, res []float64) {
	vecDivScalar(a, scalar, res)
}

func AddScalar(a []float64, scalar float64, res []float64) {
	vecAddScalar(a, scalar, res)
}

func AddAccumulate(res, a []float64) {
	if len(res) < len(a) {
		panic(fmt.Sprintf("AddAccumulate: destination slice is smaller than source slice (%d < %d)", len(res), len(a)))
	}
	for i, v := range a {
		res[i] += v
	}
}

func MulAccumulate(res, a, b []float64) {
	for i, v := range a {
		res[i] += v * b[i]
	}
}

func DivAccumulate(res, a []float64, scalar float64) {
	for i, v := range a {
		res[i] += v / scalar
	}
}

func MulScalarAccumulate(res, a []float64, scalar float64) {
	for i, v := range a {
		res[i] += v * scalar
	}
}

func SumVector(a []float64) float64 {
	return vecSum(a)
}

func DotProduct(a, b []float64) float64 {
	return vecDot(a, b)
}

// Internal aliases for same-package use
func addVectors(a, b, res []float64) { vecAdd(a, b, res) }
func subVectors(a, b, res []float64) { vecSub(a, b, res) }
func mulVectors(a, b, res []float64) { vecMul(a, b, res) }
func divVectors(a, b, res []float64) { vecDiv(a, b, res) }
func mulScalar(a []float64, scalar float64, res []float64) { vecMulScalar(a, scalar, res) }
func divScalar(a []float64, scalar float64, res []float64) { vecDivScalar(a, scalar, res) }
func addScalar(a []float64, scalar float64, res []float64) { vecAddScalar(a, scalar, res) }
func addAccumulate(res, a []float64) {
	if len(res) < len(a) {
		panic(fmt.Sprintf("addAccumulate: destination slice is smaller than source slice (%d < %d)", len(res), len(a)))
	}
	for i, v := range a {
		res[i] += v
	}
}
func mulAccumulate(res, a, b []float64) {
	for i, v := range a {
		res[i] += v * b[i]
	}
}
func divAccumulate(res, a []float64, scalar float64) {
	for i, v := range a {
		res[i] += v / scalar
	}
}
func mulScalarAccumulate(res, a []float64, scalar float64) {
	for i, v := range a {
		res[i] += v * scalar
	}
}
func sumVector(a []float64) float64 { return vecSum(a) }

// SoftmaxBackwardRow computes the softmax Jacobian-vector product for one attention row.
// out[i] = p[i] * (dp[i] - dot(dp, p)) where p is the softmax output and dp is the upstream gradient.
func SoftmaxBackwardRow(p, dp, out []float64) {
	if len(p) != len(dp) || len(p) != len(out) {
		panic(fmt.Sprintf("SoftmaxBackwardRow: slice length mismatch. p:%d, dp:%d, out:%d", len(p), len(dp), len(out)))
	}
	var dot float64
	for i := 0; i < len(p); i++ {
		dot += dp[i] * p[i]
	}
	for i := range p {
		out[i] = p[i] * (dp[i] - dot)
	}
}
