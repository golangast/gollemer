//go:build goexperiment.simd

package moe

import (
	"simd/archsimd"
)

// simdDotProductF32 computes the dot product of two float32 slices using SIMD.
// It processes 8 elements at a time (AVX2 256-bit), then cleans up the remainder
// with 4-wide (SSE) vectors, and finally scalar. The result is returned as float32.
func simdDotProductF32(a, b []float32) float32 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}

	var acc8 archsimd.Float32x8
	i := 0

	// Process 8 elements at a time (AVX2 256-bit)
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		acc8 = acc8.Add(va.Mul(vb))
	}

	// Process 4 elements at a time (SSE 128-bit)
	// Horizontal sum of acc8: split into lo and hi 4-wide vectors, add, then sum pairs
	lo8 := acc8.GetLo() // Float32x4 — lower 4 lanes of acc8
	hi8 := acc8.GetHi() // Float32x4 — upper 4 lanes of acc8
	acc4 := lo8.Add(hi8)

	var acc4b archsimd.Float32x4
	for ; i+4 <= n; i += 4 {
		va4 := archsimd.LoadFloat32x4Slice(a[i:])
		vb4 := archsimd.LoadFloat32x4Slice(b[i:])
		acc4b = acc4b.Add(va4.Mul(vb4))
	}
	acc4 = acc4.Add(acc4b)

	// Horizontal sum of acc4: AddPairs folds [a0, a1, a2, a3] -> [a0+a1, a2+a3, a0+a1, a2+a3]
	// then AddPairs again -> full horizontal sum in each lane
	sumVec := acc4.AddPairs(archsimd.BroadcastFloat32x4(0)).AddPairs(archsimd.BroadcastFloat32x4(0))
	var buf [4]float32
	sumVec.Store(&buf)
	sum := buf[0] + buf[1] + buf[2] + buf[3] // AddPairs doesn't fully reduce, so sum manually

	// Scalar tail
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// simdMulAccumulateF32 performs element-wise multiplication of a and b, adding
// the result into dst. All slices must have the same length.
// Operates 8 elements at a time using AVX2 SIMD, then 4 (SSE), then scalar tail.
func simdMulAccumulateF32(dst, a, b []float32) {
	n := len(dst)
	if len(a) < n {
		n = len(a)
	}
	if len(b) < n {
		n = len(b)
	}

	i := 0
	// Process 8 at a time
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		vd := archsimd.LoadFloat32x8Slice(dst[i:])
		result := vd.Add(va.Mul(vb))
		result.StoreSlice(dst[i:])
	}

	// Process 4 at a time
	for ; i+4 <= n; i += 4 {
		va4 := archsimd.LoadFloat32x4Slice(a[i:])
		vb4 := archsimd.LoadFloat32x4Slice(b[i:])
		vd4 := archsimd.LoadFloat32x4Slice(dst[i:])
		result4 := vd4.Add(va4.Mul(vb4))
		result4.StoreSlice(dst[i:])
	}

	// Scalar tail
	for ; i < n; i++ {
		dst[i] += a[i] * b[i]
	}
}

// computeRouterLogitsSIMD computes router logits for all (token, expert) pairs
// using SIMD-accelerated float32 dot products. Weights are stored in row-major
// order as [numExperts][inputDim] in float64; they are converted to float32 per
// expert weight row on first call and cached in routerWeightsF32.
//
// inputFlat is shaped [numTokens * inputDim] (float64).
// routerWeightsData is shaped [numExperts * inputDim] (float64).
// logitsOut is shaped [numTokens * numExperts] (float64) and will be overwritten.
func computeRouterLogitsSIMD(
	inputFlat []float64,
	routerWeightsData []float64,
	numTokens, numExperts, inputDim int,
	logitsOut []float64,
) {
	// Convert the full router weight matrix to float32.
	// In the actual GatingNetwork, routerWeightsData is W of shape [inputDim, numExperts]
	// (column-major from nn.Linear), so expert i's weights are routerWeightsData[k*numExperts+i]
	// for k in [0, inputDim). We'll compute each expert's dot product accordingly.
	// Note: nn.Linear stores weights as [inputDim, outputDim] = [inputDim, numExperts].
	// So expert i's weight is the i-th column: W[k][i] = routerWeightsData[k*numExperts + i].

	// Convert the full router weight matrix to float32 once.
	weightsF32 := make([]float32, len(routerWeightsData))
	for i, val := range routerWeightsData {
		weightsF32[i] = float32(val)
	}

	// Build float32 input row buffer (reused per token)
	inputF32 := make([]float32, inputDim)
	// Expert weight column buffer (reused per expert)
	expertWeightF32 := make([]float32, inputDim)

	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		// Convert this token's input row to float32
		tokenBase := tokenIdx * inputDim
		for k := 0; k < inputDim; k++ {
			inputF32[k] = float32(inputFlat[tokenBase+k])
		}

		// For each expert, extract its weight column and compute dot product
		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			// Expert expertIdx's weights: column expertIdx of [inputDim x numExperts] matrix
			for k := 0; k < inputDim; k++ {
				expertWeightF32[k] = weightsF32[k*numExperts+expertIdx]
			}
			logitsOut[tokenIdx*numExperts+expertIdx] = float64(simdDotProductF32(inputF32, expertWeightF32))
		}
	}
}

// computeRouterGradSIMD accumulates weight gradients and input gradients for
// the router's linear layer using SIMD multiply-accumulate. This mirrors the
// backward of a linear layer:
//   dW[k][i]     += inputFlat[token][k] * logitsGrad[token][i]
//   dInputFlat[token][k] += routerWeights[k][i] * logitsGrad[token][i]
//
// All float64 data is converted to float32 for SIMD computation, then cast back.
func computeRouterGradSIMD(
	inputFlat []float64,
	routerWeightsData []float64,
	logitsGradFlat []float64,
	dWeightsOut []float64,
	dInputOut []float64,
	numTokens, numExperts, inputDim int,
) {
	// Pre-convert weights to float32
	weightsF32 := make([]float32, len(routerWeightsData))
	for i, val := range routerWeightsData {
		weightsF32[i] = float32(val)
	}

	inputF32 := make([]float32, inputDim)
	expertWeightF32 := make([]float32, inputDim)
	dInputF32 := make([]float32, inputDim)
	dWeightsF32 := make([]float32, len(routerWeightsData))

	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		tokenBase := tokenIdx * inputDim
		// Convert this token's input to float32
		for k := 0; k < inputDim; k++ {
			inputF32[k] = float32(inputFlat[tokenBase+k])
		}
		// Reset dInput for this token
		for i := 0; i < inputDim; i++ {
			dInputF32[i] = 0
		}

		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			gradVal := float32(logitsGradFlat[tokenIdx*numExperts+expertIdx])
			if gradVal == 0 {
				continue
			}

			// Extract expert weight column from pre-converted weightsF32
			for k := 0; k < inputDim; k++ {
				expertWeightF32[k] = weightsF32[k*numExperts+expertIdx]
			}

			// SIMD accumulation for dInput and scalar for dWeights
			gradVec := archsimd.BroadcastFloat32x8(gradVal)
			k := 0
			for ; k+8 <= inputDim; k += 8 {
				// dInput[:] += gradVal * expertWeight[:]
				vw := archsimd.LoadFloat32x8Slice(expertWeightF32[k:])
				vdi := archsimd.LoadFloat32x8Slice(dInputF32[k:])
				vdi = vdi.Add(vw.Mul(gradVec))
				vdi.StoreSlice(dInputF32[k:])

				// dWeights is column-major so scalar fallback
				for kk := k; kk < k+8; kk++ {
					dWeightsF32[kk*numExperts+expertIdx] += inputF32[kk] * gradVal
				}
			}
			// Tail
			for ; k < inputDim; k++ {
				dWeightsF32[k*numExperts+expertIdx] += inputF32[k] * gradVal
				dInputF32[k] += expertWeightF32[k] * gradVal
			}
		}

		// Accumulate dInput back to float64 for this token
		if dInputOut != nil {
			for k := 0; k < inputDim; k++ {
				dInputOut[tokenBase+k] += float64(dInputF32[k])
			}
		}
	}

	// Accumulate dWeights back to float64
	for idx := range dWeightsOut {
		dWeightsOut[idx] += float64(dWeightsF32[idx])
	}
}

// --- float64 operations for MoELayer ---

// simdScaleF64 multiplies every element in data by factor.
func simdScaleF64(data []float64, factor float64) {
	n := len(data)
	if n == 0 {
		return
	}
	vecFactor := archsimd.BroadcastFloat64x4(factor)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat64x4Slice(data[i:])
		v = v.Mul(vecFactor)
		v.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		data[i] *= factor
	}
}

// simdAddScalarMulF64 performs dst[i] += src[i] * scalar.
func simdAddScalarMulF64(dst, src []float64, scalar float64) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n == 0 {
		return
	}
	vecScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		vd := archsimd.LoadFloat64x4Slice(dst[i:])
		vs := archsimd.LoadFloat64x4Slice(src[i:])
		vd = vd.Add(vs.Mul(vecScalar))
		vd.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += src[i] * scalar
	}
}

// simdMulScalarF64 performs dst[i] = src[i] * scalar.
func simdMulScalarF64(dst, src []float64, scalar float64) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n == 0 {
		return
	}
	vecScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		vs := archsimd.LoadFloat64x4Slice(src[i:])
		vd := vs.Mul(vecScalar)
		vd.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = src[i] * scalar
	}
}

// simdDotProductF64 computes the dot product of two float64 slices.
func simdDotProductF64(a, b []float64) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	var acc archsimd.Float64x4
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vb := archsimd.LoadFloat64x4Slice(b[i:])
		acc = acc.Add(va.Mul(vb))
	}

	var buf [4]float64
	acc.Store(&buf)
	sum := buf[0] + buf[1] + buf[2] + buf[3]

	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// simdSoftmaxBackwardRowF64 computes out[k] = p[k] * (dp[k] - sumDP).
func simdSoftmaxBackwardRowF64(out, p, dp []float64, sumDP float64) {
	n := len(out)
	if len(p) < n || len(dp) < n {
		return
	}
	vecSumDP := archsimd.BroadcastFloat64x4(sumDP)
	i := 0
	for ; i+4 <= n; i += 4 {
		vp := archsimd.LoadFloat64x4Slice(p[i:])
		vdp := archsimd.LoadFloat64x4Slice(dp[i:])
		resVec := vp.Mul(vdp.Sub(vecSumDP))
		resVec.StoreSlice(out[i:])
	}
	for ; i < n; i++ {
		out[i] = p[i] * (dp[i] - sumDP)
	}
}

// simdReLUF64 performs in-place ReLU: x = max(0, x)
func simdReLUF64(data []float64) {
	n := len(data)
	if n == 0 {
		return
	}
	zeros := archsimd.BroadcastFloat64x4(0)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat64x4Slice(data[i:])
		v = v.Max(zeros)
		v.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

// simdMaxSliceF64 finds the maximum value in a slice.
func simdMaxSliceF64(data []float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}
	maxVal := data[0]
	i := 0
	if n >= 4 {
		vMax := archsimd.LoadFloat64x4Slice(data)
		i = 4
		for ; i+4 <= n; i += 4 {
			v := archsimd.LoadFloat64x4Slice(data[i:])
			vMax = vMax.Max(v)
		}
		var buf [4]float64
		vMax.Store(&buf)
		for _, v := range buf {
			if v > maxVal {
				maxVal = v
			}
		}
	}
	for ; i < n; i++ {
		if data[i] > maxVal {
			maxVal = data[i]
		}
	}
	return maxVal
}

// simdSubScalarF64 performs in-place subtraction: x = x - scalar
func simdSubScalarF64(data []float64, scalar float64) {
	n := len(data)
	if n == 0 {
		return
	}
	vecScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat64x4Slice(data[i:])
		v = v.Sub(vecScalar)
		v.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		data[i] -= scalar
	}
}
