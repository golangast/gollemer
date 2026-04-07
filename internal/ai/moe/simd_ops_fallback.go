//go:build !goexperiment.simd

package moe

import (
	"math/rand"
)

// computeRouterLogitsSIMD is the pure-Go fallback.
func computeRouterLogitsSIMD(
	inputFlat []float32,
	routerWeightsData []float32,
	numTokens, numExperts, inputDim int,
	logitsOut []float32,
) {
	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		tokenBase := tokenIdx * inputDim
		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			var dot float32
			for k := 0; k < inputDim; k++ {
				dot += inputFlat[tokenBase+k] * routerWeightsData[k*numExperts+expertIdx]
			}
			logitsOut[tokenIdx*numExperts+expertIdx] = dot
		}
	}
}

// computeRouterGradSIMD is the pure-Go fallback for the router gradient.
func computeRouterGradSIMD(
	inputFlat []float32,
	routerWeightsData []float32,
	logitsGradFlat []float32,
	dWeightsOut []float32,
	dInputOut []float32,
	numTokens, numExperts, inputDim int,
	scaleFactor float32,
) {
	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		tokenBase := tokenIdx * inputDim
		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			gradVal := logitsGradFlat[tokenIdx*numExperts+expertIdx] * scaleFactor
			if gradVal == 0 {
				continue
			}
			for k := 0; k < inputDim; k++ {
				dWeightsOut[k*numExperts+expertIdx] += inputFlat[tokenBase+k] * gradVal
				if dInputOut != nil {
					dInputOut[tokenBase+k] += routerWeightsData[k*numExperts+expertIdx] * gradVal
				}
			}
		}
	}
}

// updateWeightsSIMD fallback
func updateWeightsSIMD(weights, gradients, inputs []float32, delta float32) {
	n := len(inputs)
	if len(gradients) < n {
		n = len(gradients)
	}
	for i := 0; i < n; i++ {
		gradients[i] += inputs[i] * delta
	}
}

// --- float32 operations for MoELayer ---

func simdScaleF32(data []float32, factor float32) {
	for i := range data {
		data[i] *= factor
	}
}

func simdAddScalarMulF32(dst, src []float32, scalar float32) {
	for i := range dst {
		dst[i] += src[i] * scalar
	}
}

func simdMulScalarF32(dst, src []float32, scalar float32) {
	for i := range dst {
		dst[i] = src[i] * scalar
	}
}

func simdDotProductGenericF32(a, b []float32) float32 {
	return simdDotProductF32(a, b)
}

func simdDotProductF32(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

func simdSoftmaxBackwardRowF32(p, dp, out []float32) {
	var dot float32
	for i := range p {
		dot += p[i] * dp[i]
	}
	for i := range out {
		out[i] = p[i] * (dp[i] - dot)
	}
}

func simdReLUF32(data []float32) {
	for i := range data {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func simdSquareOfSumsLossF32(counts []int, total int, weight float32) float32 {
	if total == 0 {
		return 0
	}
	tInv := 1.0 / float32(total)
	var sumSq float32
	for _, c := range counts {
		f := float32(c) * tInv
		sumSq += f * f
	}
	return sumSq * weight
}

func simdMaxSliceF32(data []float32) float32 {
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

func simdAddJitterF32(dst, src []float32, jitterStdDev float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	for i := 0; i < n; i++ {
		jitter := 1.0 + (rand.NormFloat64() * float64(jitterStdDev))
		dst[i] = src[i] * float32(jitter)
	}
}
