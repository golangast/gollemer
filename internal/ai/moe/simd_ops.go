//go:build goexperiment.simd

package moe

import (
	"math/rand"
	"simd/archsimd"
)

// simdDotProductF32 computes the dot product of two float32 slices using SIMD.
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
	lo8 := acc8.GetLo()
	hi8 := acc8.GetHi()
	acc4 := lo8.Add(hi8)

	var acc4b archsimd.Float32x4
	for ; i+4 <= n; i += 4 {
		va4 := archsimd.LoadFloat32x4Slice(a[i:])
		vb4 := archsimd.LoadFloat32x4Slice(b[i:])
		acc4b = acc4b.Add(va4.Mul(vb4))
	}
	acc4 = acc4.Add(acc4b)

	// Horizontal sum
	sumVec := acc4.AddPairs(archsimd.BroadcastFloat32x4(0)).AddPairs(archsimd.BroadcastFloat32x4(0))
	var buf [4]float32
	sumVec.Store(&buf)
	sum := buf[0] + buf[1] + buf[2] + buf[3]

	// Scalar tail
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// computeRouterLogitsSIMD computes router logits for all (token, expert) pairs.
// inputFlat: [numTokens, inputDim]
// routerWeightsData: [inputDim, numExperts] (column-major)
// logitsOut: [numTokens, numExperts]
func computeRouterLogitsSIMD(
	inputFlat []float32,
	routerWeightsData []float32,
	numTokens, numExperts, inputDim int,
	logitsOut []float32,
) {
	// Expert weight column buffer (reused per expert)
	expertWeightF32 := make([]float32, inputDim)

	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		tokenBase := tokenIdx * inputDim
		inputF32 := inputFlat[tokenBase : tokenBase+inputDim]

		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			// Extract expert weight column: W[k][expertIdx] = routerWeightsData[k*numExperts + expertIdx]
			for k := 0; k < inputDim; k++ {
				expertWeightF32[k] = routerWeightsData[k*numExperts+expertIdx]
			}
			logitsOut[tokenIdx*numExperts+expertIdx] = simdDotProductF32(inputF32, expertWeightF32)
		}
	}
}

// computeRouterGradSIMD accumulates weight gradients and input gradients for the router.
func computeRouterGradSIMD(
	inputFlat []float32,
	routerWeightsData []float32,
	logitsGradFlat []float32,
	dWeightsOut []float32,
	dInputOut []float32,
	numTokens, numExperts, inputDim int,
	scaleFactor float32,
) {
	dInputRow := make([]float32, inputDim)
	expertWeightF32 := make([]float32, inputDim)

	for tokenIdx := 0; tokenIdx < numTokens; tokenIdx++ {
		tokenBase := tokenIdx * inputDim
		inputRow := inputFlat[tokenBase : tokenBase+inputDim]

		// Reset dInputRow for this token
		for i := range dInputRow {
			dInputRow[i] = 0
		}

		for expertIdx := 0; expertIdx < numExperts; expertIdx++ {
			gradVal := logitsGradFlat[tokenIdx*numExperts+expertIdx] * scaleFactor
			if gradVal == 0 {
				continue
			}

			// Extract weight column
			for k := 0; k < inputDim; k++ {
				expertWeightF32[k] = routerWeightsData[k*numExperts+expertIdx]
			}

			gradVec := archsimd.BroadcastFloat32x8(gradVal)
			k := 0
			for ; k+8 <= inputDim; k += 8 {
				// dInputRow[:] += gradVal * expertWeight[:]
				vw := archsimd.LoadFloat32x8Slice(expertWeightF32[k:])
				vdi := archsimd.LoadFloat32x8Slice(dInputRow[k:])
				vdi = vdi.Add(vw.Mul(gradVec))
				vdi.StoreSlice(dInputRow[k:])

				// dWeights is column-major
				for kk := k; kk < k+8; kk++ {
					dWeightsOut[kk*numExperts+expertIdx] += inputRow[kk] * gradVal
				}
			}
			// Tail
			for ; k < inputDim; k++ {
				dWeightsOut[k*numExperts+expertIdx] += inputRow[k] * gradVal
				dInputRow[k] += expertWeightF32[k] * gradVal
			}
		}

		// Accumulate dInputRow to dInputOut
		if dInputOut != nil {
			for k := 0; k < inputDim; k++ {
				dInputOut[tokenBase+k] += dInputRow[k]
			}
		}
	}
}

// updateWeightsSIMD performs a vectorized weight gradient update.
func updateWeightsSIMD(weights, gradients, inputs []float32, delta float32) {
	deltaVec := archsimd.BroadcastFloat32x8(delta)
	n := len(inputs)
	if len(gradients) < n {
		n = len(gradients)
	}

	i := 0
	for ; i+8 <= n; i += 8 {
		inputVec := archsimd.LoadFloat32x8Slice(inputs[i:])
		gradVec := inputVec.Mul(deltaVec)
		existingGrad := archsimd.LoadFloat32x8Slice(gradients[i:])
		newGrad := existingGrad.Add(gradVec)
		newGrad.StoreSlice(gradients[i:])
	}

	deltaVec4 := archsimd.BroadcastFloat32x4(delta)
	for ; i+4 <= n; i += 4 {
		inputVec4 := archsimd.LoadFloat32x4Slice(inputs[i:])
		gradVec4 := inputVec4.Mul(deltaVec4)
		existingGrad4 := archsimd.LoadFloat32x4Slice(gradients[i:])
		newGrad4 := existingGrad4.Add(gradVec4)
		newGrad4.StoreSlice(gradients[i:])
	}

	for ; i < n; i++ {
		gradients[i] += inputs[i] * delta
	}
}

// --- float32 operations for MoELayer ---

func simdScaleF32(data []float32, factor float32) {
	n := len(data)
	if n == 0 {
		return
	}
	vecFactor := archsimd.BroadcastFloat32x8(factor)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		v = v.Mul(vecFactor)
		v.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		data[i] *= factor
	}
}

func simdAddScalarMulF32(dst, src []float32, scalar float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n == 0 {
		return
	}
	vecScalar := archsimd.BroadcastFloat32x8(scalar)
	i := 0
	for ; i+8 <= n; i += 8 {
		vd := archsimd.LoadFloat32x8Slice(dst[i:])
		vs := archsimd.LoadFloat32x8Slice(src[i:])
		vd = vd.Add(vs.Mul(vecScalar))
		vd.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += src[i] * scalar
	}
}

func simdMulScalarF32(dst, src []float32, scalar float32) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n == 0 {
		return
	}
	vecScalar := archsimd.BroadcastFloat32x8(scalar)
	i := 0
	for ; i+8 <= n; i += 8 {
		vs := archsimd.LoadFloat32x8Slice(src[i:])
		vd := vs.Mul(vecScalar)
		vd.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = src[i] * scalar
	}
}

func simdDotProductGenericF32(a, b []float32) float32 {
	return simdDotProductF32(a, b)
}

func simdReLUF32(data []float32) {
	n := len(data)
	if n == 0 {
		return
	}
	zeros := archsimd.BroadcastFloat32x8(0)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		v = v.Max(zeros)
		v.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

// simdSoftmaxBackwardRowF32 computes out[k] = p[k] * (dp[k] - sumDP).
func simdSoftmaxBackwardRowF32(p, dp, out []float32) {
	n := len(out)
	if len(p) < n || len(dp) < n {
		return
	}
	
	// Calculate sumDP = dot(dp, p)
	sumDP := simdDotProductF32(dp, p)
	
	i := 0
	for ; i+8 <= n; i += 8 {
		vp := archsimd.LoadFloat32x8Slice(p[i:])
		vdp := archsimd.LoadFloat32x8Slice(dp[i:])
		resVec := vp.Mul(vdp.Sub(archsimd.BroadcastFloat32x8(sumDP)))
		resVec.StoreSlice(out[i:])
	}
	for ; i < n; i++ {
		out[i] = p[i] * (dp[i] - sumDP)
	}
}

func simdMaxSliceF32(data []float32) float32 {
	n := len(data)
	if n == 0 {
		return 0
	}
	maxVal := data[0]
	i := 0
	if n >= 8 {
		vMax := archsimd.LoadFloat32x8Slice(data)
		i = 8
		for ; i+8 <= n; i += 8 {
			v := archsimd.LoadFloat32x8Slice(data[i:])
			vMax = vMax.Max(v)
		}
		var buf [8]float32
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

// simdSquareOfSumsLossF32 computes sum((counts[i]/total)^2) * weight.
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
