//go:build simd

package tensor

func IsSIMDEnabled() bool {
	return true
}

const step4 = 4

func vecAdd(a, b, res []float64) {
	for i := 0; i < len(a); i++ {
		res[i] = a[i] + b[i]
	}
}

func vecSub(a, b, res []float64) {
	for i := 0; i < len(a); i++ {
		res[i] = a[i] - b[i]
	}
}

func vecMul(a, b, res []float64) {
	for i := 0; i < len(a); i++ {
		res[i] = a[i] * b[i]
	}
}

func vecDiv(a, b, res []float64) {
	for i := 0; i < len(a); i++ {
		res[i] = a[i] / b[i]
	}
}

func vecMulScalar(a []float64, scalar float64, res []float64) {
	for i := 0; i < len(a); i++ {
		res[i] = a[i] * scalar
	}
}

func vecDivScalar(a []float64, scalar float64, res []float64) {
	for i := 0; i < len(a); i++ {
		res[i] = a[i] / scalar
	}
}

func vecAddScalar(a []float64, scalar float64, res []float64) {
	for i := 0; i < len(a); i++ {
		res[i] = a[i] + scalar
	}
}

func vecAddAccumulate(res, a []float64) {
	for i := 0; i < len(a); i++ {
		res[i] += a[i]
	}
}

func vecMulAccumulate(res, a, b []float64) {
	for i := 0; i < len(a); i++ {
		res[i] += a[i] * b[i]
	}
}

func vecDivAccumulate(res, a []float64, scalar float64) {
	for i := 0; i < len(a); i++ {
		res[i] += a[i] / scalar
	}
}

func vecMulScalarAccumulate(res, a []float64, scalar float64) {
	for i := 0; i < len(a); i++ {
		res[i] += a[i] * scalar
	}
}

func vecSum(a []float64) float64 {
	var sum float64
	for i := 0; i < len(a); i++ {
		sum += a[i]
	}
	return sum
}

func vecDot(a, b []float64) float64 {
	var total float64
	for i := 0; i < len(a); i++ {
		total += a[i] * b[i]
	}
	return total
}

// vecSoftmaxBackwardRow computes the softmax Jacobian-vector product for one row:
// out[i] = p[i] * (dp[i] - dot(dp, p))
// p is the softmax output row, dp is the upstream gradient row, out is the result.
func vecSoftmaxBackwardRow(p, dp, out []float64) {
	dot := vecDot(dp, p)
	for i := range p {
		out[i] = p[i] * (dp[i] - dot)
	}
}
