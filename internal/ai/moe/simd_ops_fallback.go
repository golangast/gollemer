//go:build !goexperiment.simd

package moe

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// computeRouterLogitsSIMD is the pure-Go fallback.
func computeRouterLogitsSIMD(
	inputFlat []float32,
	routerWeightsData []float32,
	numTokens, numExperts, inputDim int,
	logitsOut []float32,
) {
	numWorkers := runtime.NumCPU()
	if numTokens < 64 {
		numWorkers = 1
	}
	tokensPerWorker := (numTokens + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * tokensPerWorker
		end := (w + 1) * tokensPerWorker
		if start >= numTokens {
			break
		}
		if end > numTokens {
			end = numTokens
		}

		wg.Add(1)
		go func(tokenStart, tokenEnd int) {
			defer wg.Done()
			for tokenIdx := tokenStart; tokenIdx < tokenEnd; tokenIdx++ {
				tokenBase := tokenIdx * inputDim
				outBase := tokenIdx * numExperts
				for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
					var dot float32
					k := 0
					// 8-way unroll for dot product
					for ; k+8 <= inputDim; k += 8 {
						dot += inputFlat[tokenBase+k] * routerWeightsData[k*numExperts+expertIdx]
						dot += inputFlat[tokenBase+k+1] * routerWeightsData[(k+1)*numExperts+expertIdx]
						dot += inputFlat[tokenBase+k+2] * routerWeightsData[(k+2)*numExperts+expertIdx]
						dot += inputFlat[tokenBase+k+3] * routerWeightsData[(k+3)*numExperts+expertIdx]
						dot += inputFlat[tokenBase+k+4] * routerWeightsData[(k+4)*numExperts+expertIdx]
						dot += inputFlat[tokenBase+k+5] * routerWeightsData[(k+5)*numExperts+expertIdx]
						dot += inputFlat[tokenBase+k+6] * routerWeightsData[(k+6)*numExperts+expertIdx]
						dot += inputFlat[tokenBase+k+7] * routerWeightsData[(k+7)*numExperts+expertIdx]
					}
					for ; k < inputDim; k++ {
						dot += inputFlat[tokenBase+k] * routerWeightsData[k*numExperts+expertIdx]
					}
					logitsOut[outBase+expertIdx] = dot
				}
			}
		}(start, end)
	}
	wg.Wait()
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
	numWorkers := runtime.NumCPU()
	if numTokens < 64 {
		numWorkers = 1
	}
	tokensPerWorker := (numTokens + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		start := w * tokensPerWorker
		end := (w + 1) * tokensPerWorker
		if start >= numTokens {
			break
		}
		if end > numTokens {
			end = numTokens
		}

		wg.Add(1)
		go func(tokenStart, tokenEnd int) {
			defer wg.Done()
			for tokenIdx := tokenStart; tokenIdx < tokenEnd; tokenIdx++ {
				tokenBase := tokenIdx * inputDim
				outBase := tokenIdx * numExperts
				for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
					gradVal := logitsGradFlat[outBase+expertIdx] * scaleFactor
					if gradVal == 0 {
						continue
					}
					k := 0
					// 8-way unroll for gradient updates
					for ; k+8 <= inputDim; k += 8 {
						dWeightsOut[k*numExperts+expertIdx] += inputFlat[tokenBase+k] * gradVal
						dWeightsOut[(k+1)*numExperts+expertIdx] += inputFlat[tokenBase+k+1] * gradVal
						dWeightsOut[(k+2)*numExperts+expertIdx] += inputFlat[tokenBase+k+2] * gradVal
						dWeightsOut[(k+3)*numExperts+expertIdx] += inputFlat[tokenBase+k+3] * gradVal
						dWeightsOut[(k+4)*numExperts+expertIdx] += inputFlat[tokenBase+k+4] * gradVal
						dWeightsOut[(k+5)*numExperts+expertIdx] += inputFlat[tokenBase+k+5] * gradVal
						dWeightsOut[(k+6)*numExperts+expertIdx] += inputFlat[tokenBase+k+6] * gradVal
						dWeightsOut[(k+7)*numExperts+expertIdx] += inputFlat[tokenBase+k+7] * gradVal

						if dInputOut != nil {
							dInputOut[tokenBase+k] += routerWeightsData[k*numExperts+expertIdx] * gradVal
							dInputOut[tokenBase+k+1] += routerWeightsData[(k+1)*numExperts+expertIdx] * gradVal
							dInputOut[tokenBase+k+2] += routerWeightsData[(k+2)*numExperts+expertIdx] * gradVal
							dInputOut[tokenBase+k+3] += routerWeightsData[(k+3)*numExperts+expertIdx] * gradVal
							dInputOut[tokenBase+k+4] += routerWeightsData[(k+4)*numExperts+expertIdx] * gradVal
							dInputOut[tokenBase+k+5] += routerWeightsData[(k+5)*numExperts+expertIdx] * gradVal
							dInputOut[tokenBase+k+6] += routerWeightsData[(k+6)*numExperts+expertIdx] * gradVal
							dInputOut[tokenBase+k+7] += routerWeightsData[(k+7)*numExperts+expertIdx] * gradVal
						}
					}
					for ; k < inputDim; k++ {
						dWeightsOut[k*numExperts+expertIdx] += inputFlat[tokenBase+k] * gradVal
						if dInputOut != nil {
							dInputOut[tokenBase+k] += routerWeightsData[k*numExperts+expertIdx] * gradVal
						}
					}
				}
			}
		}(start, end)
	}
	wg.Wait()
}

// updateWeightsSIMD fallback
func updateWeightsSIMD(weights, gradients, inputs []float32, delta float32) {
	n := len(inputs)
	if len(gradients) < n {
		n = len(gradients)
	}
	i := 0
	// 8-way unroll
	for ; i+8 <= n; i += 8 {
		gradients[i] += inputs[i] * delta
		gradients[i+1] += inputs[i+1] * delta
		gradients[i+2] += inputs[i+2] * delta
		gradients[i+3] += inputs[i+3] * delta
		gradients[i+4] += inputs[i+4] * delta
		gradients[i+5] += inputs[i+5] * delta
		gradients[i+6] += inputs[i+6] * delta
		gradients[i+7] += inputs[i+7] * delta
	}
	for ; i < n; i++ {
		gradients[i] += inputs[i] * delta
	}
}

// --- float32 operations for MoELayer ---

func SimdScaleF32(data []float32, factor float32) {
	i := 0
	n := len(data)
	// 8-way unroll
	for ; i+8 <= n; i += 8 {
		data[i] *= factor
		data[i+1] *= factor
		data[i+2] *= factor
		data[i+3] *= factor
		data[i+4] *= factor
		data[i+5] *= factor
		data[i+6] *= factor
		data[i+7] *= factor
	}
	for ; i < n; i++ {
		data[i] *= factor
	}
}

func SimdAddScalarMulF32(dst, src []float32, scalar float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	i := 0
	// 8-way unroll
	for ; i+8 <= n; i += 8 {
		dst[i] += src[i] * scalar
		dst[i+1] += src[i+1] * scalar
		dst[i+2] += src[i+2] * scalar
		dst[i+3] += src[i+3] * scalar
		dst[i+4] += src[i+4] * scalar
		dst[i+5] += src[i+5] * scalar
		dst[i+6] += src[i+6] * scalar
		dst[i+7] += src[i+7] * scalar
	}
	for ; i < n; i++ {
		dst[i] += src[i] * scalar
	}
}

func SimdMulScalarF32(dst, src []float32, scalar float32) {
	for i := range dst {
		dst[i] = src[i] * scalar
	}
}

func SimdDotProductGenericF32(a, b []float32) float32 {
	return SimdDotProductF32(a, b)
}

func SimdDotProductF32(a, b []float32) float32 {
	var dot float32
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	// 8-way unroll
	for ; i+8 <= n; i += 8 {
		dot += a[i] * b[i]
		dot += a[i+1] * b[i+1]
		dot += a[i+2] * b[i+2]
		dot += a[i+3] * b[i+3]
		dot += a[i+4] * b[i+4]
		dot += a[i+5] * b[i+5]
		dot += a[i+6] * b[i+6]
		dot += a[i+7] * b[i+7]
	}
	for ; i < n; i++ {
		dot += a[i] * b[i]
	}
	return dot
}

func SimdSoftmaxBackwardRowF32(p, dp, out []float32) {
	var dot float32
	for i := range p {
		dot += p[i] * dp[i]
	}
	for i := range out {
		out[i] = p[i] * (dp[i] - dot)
	}
}

func SimdReLUF32(data []float32) {
	for i := range data {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func SimdSquareOfSumsLossF32(counts []int, total int, weight float32) float32 {
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

func SimdMaxSliceF32(data []float32) float32 {
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

func SimdArgMaxF32(data []float32) int {
	if len(data) == 0 {
		return -1
	}
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func SimdAddF32(a, b []float32) {
	for i := range a {
		a[i] += b[i]
	}
}

func SimdSubF32(a, b []float32) {
	for i := range a {
		a[i] -= b[i]
	}
}

func SimdIsFiniteF32(data []float32) bool {
	for _, v := range data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return false
		}
	}
	return true
}

func SimdExpF32(data []float32) {
	for i := range data {
		data[i] = float32(math.Exp(float64(data[i])))
	}
}

func SimdSoftmaxF32(data []float32) float32 {
	if len(data) == 0 {
		return 0
	}
	max := SimdMaxSliceF32(data)
	var sum float32
	for i := range data {
		data[i] = float32(math.Exp(float64(data[i] - max)))
		sum += data[i]
	}
	if sum > 0 {
		invSum := 1.0 / sum
		for i := range data {
			data[i] *= invSum
		}
	}
	return sum
}
