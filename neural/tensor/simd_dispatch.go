package tensor

// This file defines the entry points for SIMD-accelerated operations.
// The actual implementations are in simd_generic.go and simd_arch.go.

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
	vecAddAccumulate(res, a)
}

func MulAccumulate(res, a, b []float64) {
	vecMulAccumulate(res, a, b)
}

func DivAccumulate(res, a []float64, scalar float64) {
	vecDivAccumulate(res, a, scalar)
}

func MulScalarAccumulate(res, a []float64, scalar float64) {
	vecMulScalarAccumulate(res, a, scalar)
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
func addAccumulate(res, a []float64) { vecAddAccumulate(res, a) }
func mulAccumulate(res, a, b []float64) { vecMulAccumulate(res, a, b) }
func divAccumulate(res, a []float64, scalar float64) { vecDivAccumulate(res, a, scalar) }
func mulScalarAccumulate(res, a []float64, scalar float64) { vecMulScalarAccumulate(res, a, scalar) }
func sumVector(a []float64) float64 { return vecSum(a) }

// SoftmaxBackwardRow computes the softmax Jacobian-vector product for one attention row.
// out[i] = p[i] * (dp[i] - dot(dp, p)) where p is the softmax output and dp is the upstream gradient.
func SoftmaxBackwardRow(p, dp, out []float64) {
	vecSoftmaxBackwardRow(p, dp, out)
}
