package tensor

// This file defines the entry points for SIMD-accelerated operations.
// The actual implementations are in simd_generic.go and simd_arch.go.

func AddVectors(a, b, res []float32) {
	vecAdd(a, b, res)
}

func SubVectors(a, b, res []float32) {
	vecSub(a, b, res)
}

func MulVectors(a, b, res []float32) {
	vecMul(a, b, res)
}

func DivVectors(a, b, res []float32) {
	vecDiv(a, b, res)
}

func MulScalar(a []float32, scalar float32, res []float32) {
	vecMulScalar(a, scalar, res)
}

func DivScalar(a []float32, scalar float32, res []float32) {
	vecDivScalar(a, scalar, res)
}

func AddScalar(a []float32, scalar float32, res []float32) {
	vecAddScalar(a, scalar, res)
}

func AddAccumulate(res, a []float32) {
	vecAddAccumulate(res, a)
}

func MulAccumulate(res, a, b []float32) {
	vecMulAccumulate(res, a, b)
}

func DivAccumulate(res, a []float32, scalar float32) {
	vecDivAccumulate(res, a, scalar)
}

func MulScalarAccumulate(res, a []float32, scalar float32) {
	vecMulScalarAccumulate(res, a, scalar)
}

func SumVector(a []float32) float32 {
	return vecSum(a)
}

func DotProduct(a, b []float32) float32 {
	return vecDot(a, b)
}

// Internal aliases for same-package use
func addVectors(a, b, res []float32)                       { vecAdd(a, b, res) }
func subVectors(a, b, res []float32)                       { vecSub(a, b, res) }
func mulVectors(a, b, res []float32)                       { vecMul(a, b, res) }
func divVectors(a, b, res []float32)                       { vecDiv(a, b, res) }
func mulScalar(a []float32, scalar float32, res []float32) { vecMulScalar(a, scalar, res) }
func divScalar(a []float32, scalar float32, res []float32) { vecDivScalar(a, scalar, res) }
func addScalar(a []float32, scalar float32, res []float32) { vecAddScalar(a, scalar, res) }
func addAccumulate(res, a []float32)                       { vecAddAccumulate(res, a) }
func mulAccumulate(res, a, b []float32)                    { vecMulAccumulate(res, a, b) }
func divAccumulate(res, a []float32, scalar float32)       { vecDivAccumulate(res, a, scalar) }
func mulScalarAccumulate(res, a []float32, scalar float32) { vecMulScalarAccumulate(res, a, scalar) }
func sumVector(a []float32) float32                        { return vecSum(a) }

// SoftmaxBackwardRow computes the softmax Jacobian-vector product for one attention row.
// out[i] = p[i] * (dp[i] - dot(dp, p)) where p is the softmax output and dp is the upstream gradient.
func SoftmaxBackwardRow(p, dp, out []float32) {
	vecSoftmaxBackwardRow(p, dp, out)
}

func ReLUVector(a []float32) {
	vecReLU(a)
}

func ScaleGradients(grads []float32, maxNorm float32) {
	vecScaleGradients(grads, maxNorm)
}

func MaxSlice(a []float32) float32 {
	return vecMaxSlice(a)
}

func AdamWUpdate(weights, grads, m, v []float32, lr, beta1, beta2, eps, weightDecay float32, t int) {
	vecAdamWUpdate(weights, grads, m, v, lr, beta1, beta2, eps, weightDecay, t)
}

func ClipWeights(data []float32, maxVal float32) {
	vecClipWeights(data, maxVal)
}

func TopKZero(data []float32, k int) {
	vecTopKZero(data, k)
}

func LeakyReLUVectors(data []float32, alpha float32) {
	vecLeakyReLU(data, alpha)
}
