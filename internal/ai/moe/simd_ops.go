//go:build goexperiment.simd

package moe

import (
	"math"
	"math/rand"
	"simd/archsimd"
	"sync"

	. "github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// SimdDotProductF32 computes the dot product of two float32 slices using SIMD.
func SimdDotProductF32(a, b []float32) float32 {
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
	numWorkers := 8
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

				// Optimized: Process experts in blocks of 8 (contiguous weights)
				e := 0
				for ; e+8 <= numExperts; e += 8 {
					var acc8 archsimd.Float32x8
					k := 0
					// Manual unroll factor 4 for k-loop (input dimensions)
					for ; k+4 <= inputDim; k += 4 {
						v0 := archsimd.BroadcastFloat32x8(inputFlat[tokenBase+k])
						w0 := archsimd.LoadFloat32x8Slice(routerWeightsData[k*numExperts+e:])
						acc8 = acc8.Add(v0.Mul(w0))

						v1 := archsimd.BroadcastFloat32x8(inputFlat[tokenBase+k+1])
						w1 := archsimd.LoadFloat32x8Slice(routerWeightsData[(k+1)*numExperts+e:])
						acc8 = acc8.Add(v1.Mul(w1))

						v2 := archsimd.BroadcastFloat32x8(inputFlat[tokenBase+k+2])
						w2 := archsimd.LoadFloat32x8Slice(routerWeightsData[(k+2)*numExperts+e:])
						acc8 = acc8.Add(v2.Mul(w2))

						v3 := archsimd.BroadcastFloat32x8(inputFlat[tokenBase+k+3])
						w3 := archsimd.LoadFloat32x8Slice(routerWeightsData[(k+3)*numExperts+e:])
						acc8 = acc8.Add(v3.Mul(w3))
					}
					// k-tail
					for ; k < inputDim; k++ {
						vk := archsimd.BroadcastFloat32x8(inputFlat[tokenBase+k])
						wk := archsimd.LoadFloat32x8Slice(routerWeightsData[k*numExperts+e:])
						acc8 = acc8.Add(vk.Mul(wk))
					}
					acc8.StoreSlice(logitsOut[outBase+e:])
				}

				// Expert-tail (if numExperts not multiple of 8)
				for ; e < numExperts; e++ {
					var dot float32
					k := 0
					for ; k+4 <= inputDim; k += 4 {
						dot += inputFlat[tokenBase+k] * routerWeightsData[k*numExperts+e]
						dot += inputFlat[tokenBase+k+1] * routerWeightsData[(k+1)*numExperts+e]
						dot += inputFlat[tokenBase+k+2] * routerWeightsData[(k+2)*numExperts+e]
						dot += inputFlat[tokenBase+k+3] * routerWeightsData[(k+3)*numExperts+e]
					}
					for ; k < inputDim; k++ {
						dot += inputFlat[tokenBase+k] * routerWeightsData[k*numExperts+e]
					}
					logitsOut[outBase+e] = dot
				}
			}
		}(start, end)
	}
	wg.Wait()
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
	numWorkers := 8
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
				lgRow := logitsGradFlat[outBase : outBase+numExperts]

				// 1. Accumulate Weight Gradients (Atomic)
				// dWeights[k][e] += lgRow[e] * input[k]
				for e := 0; e < numExperts; e++ {
					lg := lgRow[e] * scaleFactor
					if lg == 0 {
						continue
					}
					k := 0
					for ; k+4 <= inputDim; k += 4 {
						AtomicAddFloat32(&dWeightsOut[k*numExperts+e], inputFlat[tokenBase+k]*lg)
						AtomicAddFloat32(&dWeightsOut[(k+1)*numExperts+e], inputFlat[tokenBase+k+1]*lg)
						AtomicAddFloat32(&dWeightsOut[(k+2)*numExperts+e], inputFlat[tokenBase+k+2]*lg)
						AtomicAddFloat32(&dWeightsOut[(k+3)*numExperts+e], inputFlat[tokenBase+k+3]*lg)
					}
					for ; k < inputDim; k++ {
						AtomicAddFloat32(&dWeightsOut[k*numExperts+e], inputFlat[tokenBase+k]*lg)
					}
				}

				// 2. Accumulate Input Gradients (if needed)
				// dInput[k] += sum_e( lgRow[e] * W[k][e] )
				if dInputOut != nil {
					dInputRow := dInputOut[tokenBase : tokenBase+inputDim]
					k := 0
					for ; k < inputDim; k++ {
						wRow := routerWeightsData[k*numExperts : (k+1)*numExperts]
						// We can use SimdDotProductF32 here but with 4-way unroll for k
						dInputRow[k] += SimdDotProductF32(lgRow, wRow) * scaleFactor
					}
				}
			}
		}(start, end)
	}
	wg.Wait()
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

func SimdScaleF32(data []float32, factor float32) {
	n := len(data)
	if n == 0 {
		return
	}
	vecFactor := archsimd.BroadcastFloat32x8(factor)
	i := 0
	// 4-way unroll (32 elements per iteration)
	for ; i+32 <= n; i += 32 {
		v0 := archsimd.LoadFloat32x8Slice(data[i:])
		v1 := archsimd.LoadFloat32x8Slice(data[i+8:])
		v2 := archsimd.LoadFloat32x8Slice(data[i+16:])
		v3 := archsimd.LoadFloat32x8Slice(data[i+24:])

		v0.Mul(vecFactor).StoreSlice(data[i:])
		v1.Mul(vecFactor).StoreSlice(data[i+8:])
		v2.Mul(vecFactor).StoreSlice(data[i+16:])
		v3.Mul(vecFactor).StoreSlice(data[i+24:])
	}
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		v.Mul(vecFactor).StoreSlice(data[i:])
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
	if n == 0 {
		return
	}
	vecScalar := archsimd.BroadcastFloat32x8(scalar)
	i := 0
	// 4-way unroll
	for ; i+32 <= n; i += 32 {
		vs0 := archsimd.LoadFloat32x8Slice(src[i:])
		vd0 := archsimd.LoadFloat32x8Slice(dst[i:])
		vs1 := archsimd.LoadFloat32x8Slice(src[i+8:])
		vd1 := archsimd.LoadFloat32x8Slice(dst[i+8:])
		vs2 := archsimd.LoadFloat32x8Slice(src[i+16:])
		vd2 := archsimd.LoadFloat32x8Slice(dst[i+16:])
		vs3 := archsimd.LoadFloat32x8Slice(src[i+24:])
		vd3 := archsimd.LoadFloat32x8Slice(dst[i+24:])

		vd0.Add(vs0.Mul(vecScalar)).StoreSlice(dst[i:])
		vd1.Add(vs1.Mul(vecScalar)).StoreSlice(dst[i+8:])
		vd2.Add(vs2.Mul(vecScalar)).StoreSlice(dst[i+16:])
		vd3.Add(vs3.Mul(vecScalar)).StoreSlice(dst[i+24:])
	}
	for ; i+8 <= n; i += 8 {
		vd := archsimd.LoadFloat32x8Slice(dst[i:])
		vs := archsimd.LoadFloat32x8Slice(src[i:])
		vd.Add(vs.Mul(vecScalar)).StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += src[i] * scalar
	}
}

func SimdMulScalarF32(dst, src []float32, scalar float32) {
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

// SimdDotProductGenericF32 is a generic wrapper.
func SimdDotProductGenericF32(a, b []float32) float32 {
	return SimdDotProductF32(a, b)
}

// SimdReLUF32 performs ReLU in-place.
func SimdReLUF32(data []float32) {
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

// SimdSoftmaxBackwardRowF32 computes out[k] = p[k] * (dp[k] - sumDP).
func SimdSoftmaxBackwardRowF32(p, dp, out []float32) {
	n := len(out)
	if len(p) < n || len(dp) < n {
		return
	}

	// Calculate sumDP = dot(dp, p)
	sumDP := SimdDotProductF32(dp, p)

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

// SimdMaxSliceF32 returns the maximum value in a slice.
func SimdMaxSliceF32(data []float32) float32 {
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

// SimdSquareOfSumsLossF32 computes balanced load loss.
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

// SimdArgMaxF32 finds the index of the maximum value in a slice.
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

// SimdAddF32 computes a[i] += b[i] using SIMD.
func SimdAddF32(a, b []float32) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	// 4-way unroll
	for ; i+32 <= n; i += 32 {
		va0 := archsimd.LoadFloat32x8Slice(a[i:])
		vb0 := archsimd.LoadFloat32x8Slice(b[i:])
		va1 := archsimd.LoadFloat32x8Slice(a[i+8:])
		vb1 := archsimd.LoadFloat32x8Slice(b[i+8:])
		va2 := archsimd.LoadFloat32x8Slice(a[i+16:])
		vb2 := archsimd.LoadFloat32x8Slice(b[i+16:])
		va3 := archsimd.LoadFloat32x8Slice(a[i+24:])
		vb3 := archsimd.LoadFloat32x8Slice(b[i+24:])

		va0.Add(vb0).StoreSlice(a[i:])
		va1.Add(vb1).StoreSlice(a[i+8:])
		va2.Add(vb2).StoreSlice(a[i+16:])
		va3.Add(vb3).StoreSlice(a[i+24:])
	}
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		va.Add(vb).StoreSlice(a[i:])
	}
	for ; i < n; i++ {
		a[i] += b[i]
	}
}

// SimdSubF32 computes a[i] -= b[i] using SIMD.
func SimdSubF32(a, b []float32) {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	i := 0
	// 4-way unroll
	for ; i+32 <= n; i += 32 {
		va0 := archsimd.LoadFloat32x8Slice(a[i:])
		vb0 := archsimd.LoadFloat32x8Slice(b[i:])
		va1 := archsimd.LoadFloat32x8Slice(a[i+8:])
		vb1 := archsimd.LoadFloat32x8Slice(b[i+8:])
		va2 := archsimd.LoadFloat32x8Slice(a[i+16:])
		vb2 := archsimd.LoadFloat32x8Slice(b[i+16:])
		va3 := archsimd.LoadFloat32x8Slice(a[i+24:])
		vb3 := archsimd.LoadFloat32x8Slice(b[i+24:])

		va0.Sub(vb0).StoreSlice(a[i:])
		va1.Sub(vb1).StoreSlice(a[i+8:])
		va2.Sub(vb2).StoreSlice(a[i+16:])
		va3.Sub(vb3).StoreSlice(a[i+24:])
	}
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		va.Sub(vb).StoreSlice(a[i:])
	}
	for ; i < n; i++ {
		a[i] -= b[i]
	}
}

// SimdIsFiniteF32 returns true if all elements in the slice are finite (not NaN or Inf).
func SimdIsFiniteF32(data []float32) bool {
	n := len(data)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		// Check for NaN/Inf: (v == v) && (abs(v) < MaxFloat32)
		// Simpler for SIMD: check if v - v == 0
		diff := v.Sub(v)
		var buf [8]float32
		diff.Store(&buf)
		for _, d := range buf {
			if d != 0 {
				return false
			}
		}
	}
	for ; i < n; i++ {
		if math.IsNaN(float64(data[i])) || math.IsInf(float64(data[i]), 0) {
			return false
		}
	}
	return true
}

// SimdExpF32 computes exp(x) for every element in data.
// The bit-twiddling approximation used earlier was numerically unstable and can
// generate invalid probability distributions during training, so we use the
// mathematically correct scalar path here to keep the loss objective reliable.
func SimdExpF32(data []float32) {
	for i := range data {
		data[i] = float32(math.Exp(float64(data[i])))
	}
}

// SimdSoftmaxF32 performs an in-place softmax on a row.
func SimdSoftmaxF32(data []float32) float32 {
	n := len(data)
	if n == 0 {
		return 0
	}

	// 1. Find max
	maxVal := SimdMaxSliceF32(data)

	// 2. Subtract max and compute Exp via SIMD
	vMax := archsimd.BroadcastFloat32x8(maxVal)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		v.Sub(vMax).StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		data[i] -= maxVal
	}

	// Vectorized Exp approximation
	SimdExpF32(data)

	// 3. Sum and Normalize
	var sumExp float32
	i = 0
	var vSum8 archsimd.Float32x8
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		vSum8 = vSum8.Add(v)
	}

	// Horizontal sum
	var buf [8]float32
	vSum8.Store(&buf)
	for _, v := range buf {
		sumExp += v
	}
	for ; i < n; i++ {
		sumExp += data[i]
	}

	if sumExp > 0 {
		SimdScaleF32(data, 1.0/sumExp)
	}
	return sumExp
}
