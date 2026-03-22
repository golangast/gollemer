//go:build !goexperiment.simd

package tensor

import (
	"math"
	"os"
	"sort"
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

func vecSoftmaxBackwardRow(p, dp, out []float64) {
	dot := vecDot(dp, p)
	for i := range p {
		out[i] = p[i] * (dp[i] - dot)
	}
}

func vecReLU(data []float64) {
	for i := range data {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func vecScaleGradients(grads []float64, maxNorm float64) {
	sumSq := vecDot(grads, grads)
	norm := math.Sqrt(sumSq)
	if norm > maxNorm {
		scaleFactor := maxNorm / (norm + 1e-8)
		vecMulScalar(grads, scaleFactor, grads)
	}
}

func vecMaxSlice(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	maxVal := data[0]
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	return maxVal
}

func vecAdamWUpdate(weights, grads, m, v []float64, lr, beta1, beta2, eps, weightDecay float64, t int) {
	biasCorrection1 := 1.0 - math.Pow(beta1, float64(t))
	biasCorrection2 := 1.0 - math.Pow(beta2, float64(t))
	for i := range weights {
		m[i] = beta1*m[i] + (1.0-beta1)*grads[i]
		v[i] = beta2*v[i] + (1.0-beta2)*grads[i]*grads[i]
		mHat := m[i] / biasCorrection1
		vHat := v[i] / biasCorrection2
		weights[i] -= lr * (mHat/(math.Sqrt(vHat)+eps) + weightDecay*weights[i])
	}
}

func vecClipWeights(data []float64, maxVal float64) {
	for i := range data {
		if data[i] > maxVal {
			data[i] = maxVal
		} else if data[i] < -maxVal {
			data[i] = -maxVal
		}
	}
}

func vecTopKZero(data []float64, k int) {
	n := len(data)
	if k >= n || k <= 0 {
		return
	}
	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)
	threshold := sorted[n-k]
	for i := range data {
		if data[i] < threshold {
			data[i] = 0
		}
	}
}

func vecLeakyReLU(data []float64, alpha float64) {
	for i := range data {
		if data[i] < 0 {
			data[i] *= alpha
		}
	}
}
