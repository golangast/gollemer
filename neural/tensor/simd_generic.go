//go:build !simd

package tensor

import (
	"os"
	"strings"
)

func IsSIMDEnabled() bool {
	return strings.Contains(os.Getenv("GOEXPERIMENT"), "simd")
}

func vecAdd(a, b, res []float64) {
	for i := range a {
		res[i] = a[i] + b[i]
	}
}

func vecSub(a, b, res []float64) {
	for i := range a {
		res[i] = a[i] - b[i]
	}
}

func vecMul(a, b, res []float64) {
	for i := range a {
		res[i] = a[i] * b[i]
	}
}

func vecDiv(a, b, res []float64) {
	for i := range a {
		res[i] = a[i] / b[i]
	}
}

func vecMulScalar(a []float64, scalar float64, res []float64) {
	for i := range a {
		res[i] = a[i] * scalar
	}
}

func vecDivScalar(a []float64, scalar float64, res []float64) {
	for i := range a {
		res[i] = a[i] / scalar
	}
}

func vecAddScalar(a []float64, scalar float64, res []float64) {
	for i := range a {
		res[i] = a[i] + scalar
	}
}

func vecAddAccumulate(res, a []float64) {
	for i := range a {
		res[i] += a[i]
	}
}

func vecMulAccumulate(res, a, b []float64) {
	for i := range a {
		res[i] += a[i] * b[i]
	}
}

func vecDivAccumulate(res, a []float64, scalar float64) {
	for i := range a {
		res[i] += a[i] / scalar
	}
}

func vecMulScalarAccumulate(res, a []float64, scalar float64) {
	for i := range a {
		res[i] += a[i] * scalar
	}
}

func vecSum(a []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i]
	}
	return sum
}

func vecDot(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// vecSoftmaxBackwardRow computes out[i] = p[i] * (dp[i] - dot(dp, p)) for one row.
func vecSoftmaxBackwardRow(p, dp, out []float64) {
	dot := vecDot(dp, p)
	for i := range p {
		out[i] = p[i] * (dp[i] - dot)
	}
}
