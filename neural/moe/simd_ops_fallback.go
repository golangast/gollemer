//go:build !goexperiment.simd

package moe

// computeRouterLogitsSIMD is the pure-Go fallback used when the
// GOEXPERIMENT=simd build flag is NOT set. It produces identical results
// to the SIMD version but without hardware vectorisation.
//
// inputFlat is shaped [numTokens * inputDim] (float64).
// routerWeightsData is shaped [inputDim * numExperts] (float64), column-major
// (i.e. nn.Linear weight layout: W[k][expert] = routerWeightsData[k*numExperts+expert]).
// logitsOut is shaped [numTokens * numExperts] (float64) and is overwritten.
func computeRouterLogitsSIMD(
	inputFlat []float64,
	routerWeightsData []float64,
	numTokens, numExperts, inputDim int,
	logitsOut []float64,
) {
	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		tokenBase := tokenIdx * inputDim
		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			var dot float64
			for k := 0; k < inputDim; k++ {
				dot += inputFlat[tokenBase+k] * routerWeightsData[k*numExperts+expertIdx]
			}
			logitsOut[tokenIdx*numExperts+expertIdx] = dot
		}
	}
}

// computeRouterGradSIMD is the pure-Go fallback for the router gradient.
// It accumulates weight gradients and input gradients for the router's
// linear layer:
//   dW[k][expertIdx] += inputFlat[token][k] * logitsGradFlat[token][expertIdx]
//   dInputOut[token][k] += routerWeightsData[k][expertIdx] * logitsGradFlat[token][expertIdx]
func computeRouterGradSIMD(
	inputFlat []float64,
	routerWeightsData []float64,
	logitsGradFlat []float64,
	dWeightsOut []float64,
	dInputOut []float64,
	numTokens, numExperts, inputDim int,
) {
	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		tokenBase := tokenIdx * inputDim
		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			gradVal := logitsGradFlat[tokenIdx*numExperts+expertIdx]
			if gradVal == 0 {
				continue
			}
			for k := 0; k < inputDim; k++ {
				dWeightsOut[k*numExperts+expertIdx] += inputFlat[tokenBase+k] * gradVal
				dInputOut[tokenBase+k] += routerWeightsData[k*numExperts+expertIdx] * gradVal
			}
		}
	}
}

// --- float64 operations for MoELayer ---

func simdScaleF64(data []float64, factor float64) {
	for i := range data {
		data[i] *= factor
	}
}

func simdAddScalarMulF64(dst, src []float64, scalar float64) {
	for i := range dst {
		dst[i] += src[i] * scalar
	}
}

func simdMulScalarF64(dst, src []float64, scalar float64) {
	for i := range dst {
		dst[i] = src[i] * scalar
	}
}

func simdDotProductF64(a, b []float64) float64 {
	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

func simdSoftmaxBackwardRowF64(out, p, dp []float64, sumDP float64) {
	for k := range out {
		out[k] = p[k] * (dp[k] - sumDP)
	}
}

func simdReLUF64(data []float64) {
	for i := range data {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func simdMaxSliceF64(data []float64) float64 {
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

func simdSubScalarF64(data []float64, scalar float64) {
	for i := range data {
		data[i] -= scalar
	}
}
