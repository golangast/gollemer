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

func vecAdd(a, b, res []float32) {
	for i := range a {
		res[i] = a[i] + b[i]
	}
}

func vecSub(a, b, res []float32) {
	for i := range a {
		res[i] = a[i] - b[i]
	}
}

func vecMul(a, b, res []float32) {
	for i := range a {
		res[i] = a[i] * b[i]
	}
}

func vecDiv(a, b, res []float32) {
	for i := range a {
		res[i] = a[i] / b[i]
	}
}

func vecMulScalar(a []float32, scalar float32, res []float32) {
	for i := range a {
		res[i] = a[i] * scalar
	}
}

func vecDivScalar(a []float32, scalar float32, res []float32) {
	for i := range a {
		res[i] = a[i] / scalar
	}
}

func vecAddScalar(a []float32, scalar float32, res []float32) {
	for i := range a {
		res[i] = a[i] + scalar
	}
}

func vecAddAccumulate(res, a []float32) {
	for i := range a {
		res[i] += a[i]
	}
}

func vecMulAccumulate(res, a, b []float32) {
	for i := range a {
		res[i] += a[i] * b[i]
	}
}

func vecDivAccumulate(res, a []float32, scalar float32) {
	for i := range a {
		res[i] += a[i] / scalar
	}
}

func vecMulScalarAccumulate(res, a []float32, scalar float32) {
	for i := range a {
		res[i] += a[i] * scalar
	}
}

func vecSum(a []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i]
	}
	return sum
}

func vecDot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func vecSoftmaxBackwardRow(p, dp, out []float32) {
	dot := vecDot(dp, p)
	for i := range p {
		out[i] = p[i] * (dp[i] - dot)
	}
}

func vecReLU(data []float32) {
	for i := range data {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func vecScaleGradients(grads []float32, maxNorm float32) {
	sumSq := vecDot(grads, grads)
	norm := float32(math.Sqrt(float64(sumSq)))
	if norm > maxNorm {
		scaleFactor := maxNorm / (norm + 1e-8)
		vecMulScalar(grads, scaleFactor, grads)
	}
}

func vecMaxSlice(data []float32) float32 {
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

func vecAdamWUpdate(weights, grads, m, v []float32, lr, beta1, beta2, eps, weightDecay float32, t int) {
	biasCorrection1 := float32(1.0 - math.Pow(float64(beta1), float64(t)))
	biasCorrection2 := float32(1.0 - math.Pow(float64(beta2), float64(t)))
	for i := range weights {
		m[i] = beta1*m[i] + (1.0-beta1)*grads[i]
		v[i] = beta2*v[i] + (1.0-beta2)*grads[i]*grads[i]
		mHat := m[i] / biasCorrection1
		vHat := v[i] / biasCorrection2
		weights[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + weightDecay*weights[i])
	}
}

func vecClipWeights(data []float32, maxVal float32) {
	for i := range data {
		if data[i] > maxVal {
			data[i] = maxVal
		} else if data[i] < -maxVal {
			data[i] = -maxVal
		}
	}
}

func vecTopKZero(data []float32, k int) {
	n := len(data)
	if k >= n || k <= 0 {
		return
	}
	sorted := make([]float32, n)
	copy(sorted, data)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	threshold := sorted[n-k]
	for i := range data {
		if data[i] < threshold {
			data[i] = 0
		}
	}
}

func vecLeakyReLU(data []float32, alpha float32) {
	for i := range data {
		if data[i] < 0 {
			data[i] *= alpha
		}
	}
}
