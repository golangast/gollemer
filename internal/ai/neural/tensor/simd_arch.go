//go:build goexperiment.simd

package tensor

import (
	"math"
	"simd/archsimd"
	"sort"
	"sync"
)

func IsSIMDEnabled() bool {
	return true
}

func vecAdd(a, b, res []float32) {
	n := len(a)
	i := 0
	// Try 8-wide (AVX2)
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		vr := va.Add(vb)
		vr.StoreSlice(res[i:])
	}
	// Try 4-wide (SSE)
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vb := archsimd.LoadFloat32x4Slice(b[i:])
		vr := va.Add(vb)
		vr.StoreSlice(res[i:])
	}
	// Tail
	for ; i < n; i++ {
		res[i] = a[i] + b[i]
	}
}

func vecSub(a, b, res []float32) {
	n := len(a)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		vr := va.Sub(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vb := archsimd.LoadFloat32x4Slice(b[i:])
		vr := va.Sub(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] - b[i]
	}
}

func vecMul(a, b, res []float32) {
	n := len(a)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		vr := va.Mul(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vb := archsimd.LoadFloat32x4Slice(b[i:])
		vr := va.Mul(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] * b[i]
	}
}

func vecDiv(a, b, res []float32) {
	n := len(a)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		vr := va.Div(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vb := archsimd.LoadFloat32x4Slice(b[i:])
		vr := va.Div(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] / b[i]
	}
}

func vecMulScalar(a []float32, scalar float32, res []float32) {
	n := len(a)
	vScalar8 := archsimd.BroadcastFloat32x8(scalar)
	vScalar4 := archsimd.BroadcastFloat32x4(scalar)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vr := va.Mul(vScalar8)
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vr := va.Mul(vScalar4)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] * scalar
	}
}

func vecDivScalar(a []float32, scalar float32, res []float32) {
	n := len(a)
	vScalar8 := archsimd.BroadcastFloat32x8(scalar)
	vScalar4 := archsimd.BroadcastFloat32x4(scalar)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vr := va.Div(vScalar8)
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vr := va.Div(vScalar4)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] / scalar
	}
}

func vecAddScalar(a []float32, scalar float32, res []float32) {
	n := len(a)
	vScalar8 := archsimd.BroadcastFloat32x8(scalar)
	vScalar4 := archsimd.BroadcastFloat32x4(scalar)
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vr := va.Add(vScalar8)
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vr := va.Add(vScalar4)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] + scalar
	}
}

func vecAddAccumulate(res, a []float32) {
	n := len(a)
	i := 0
	for ; i+8 <= n; i += 8 {
		vr := archsimd.LoadFloat32x8Slice(res[i:])
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vr = vr.Add(va)
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat32x4Slice(res[i:])
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vr = vr.Add(va)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i]
	}
}

func vecMulAccumulate(res, a, b []float32) {
	n := len(a)
	i := 0
	for ; i+8 <= n; i += 8 {
		vr := archsimd.LoadFloat32x8Slice(res[i:])
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		vr = vr.Add(va.Mul(vb))
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat32x4Slice(res[i:])
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vb := archsimd.LoadFloat32x4Slice(b[i:])
		vr = vr.Add(va.Mul(vb))
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i] * b[i]
	}
}

func vecDivAccumulate(res, a []float32, scalar float32) {
	n := len(a)
	vScalar8 := archsimd.BroadcastFloat32x8(scalar)
	vScalar4 := archsimd.BroadcastFloat32x4(scalar)
	i := 0
	for ; i+8 <= n; i += 8 {
		vr := archsimd.LoadFloat32x8Slice(res[i:])
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vr = vr.Add(va.Div(vScalar8))
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat32x4Slice(res[i:])
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vr = vr.Add(va.Div(vScalar4))
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i] / scalar
	}
}

func vecMulScalarAccumulate(res, a []float32, scalar float32) {
	n := len(a)
	vScalar8 := archsimd.BroadcastFloat32x8(scalar)
	vScalar4 := archsimd.BroadcastFloat32x4(scalar)
	i := 0
	for ; i+8 <= n; i += 8 {
		vr := archsimd.LoadFloat32x8Slice(res[i:])
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vr = vr.Add(va.Mul(vScalar8))
		vr.StoreSlice(res[i:])
	}
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat32x4Slice(res[i:])
		va := archsimd.LoadFloat32x4Slice(a[i:])
		vr = vr.Add(va.Mul(vScalar4))
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i] * scalar
	}
}

func vecSum(a []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	var acc8 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		acc8 = acc8.Add(va)
	}
	
	// Horizontal reduction
	lo8 := acc8.GetLo()
	hi8 := acc8.GetHi()
	acc4 := lo8.Add(hi8)
	
	for ; i+4 <= n; i += 4 {
		va4 := archsimd.LoadFloat32x4Slice(a[i:])
		acc4 = acc4.Add(va4)
	}
	
	var buf [4]float32
	acc4.Store(&buf)
	sum := buf[0] + buf[1] + buf[2] + buf[3]
	
	for ; i < n; i++ {
		sum += a[i]
	}
	return sum
}

func vecDot(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	var acc8 archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		acc8 = acc8.Add(va.Mul(vb))
	}

	lo8 := acc8.GetLo()
	hi8 := acc8.GetHi()
	acc4 := lo8.Add(hi8)
	
	for ; i+4 <= n; i += 4 {
		va4 := archsimd.LoadFloat32x4Slice(a[i:])
		vb4 := archsimd.LoadFloat32x4Slice(b[i:])
		acc4 = acc4.Add(va4.Mul(vb4))
	}
	
	var buf [4]float32
	acc4.Store(&buf)
	sum := buf[0] + buf[1] + buf[2] + buf[3]
	
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func vecSoftmaxBackwardRow(p, dp, out []float32) {
	dot := vecDot(dp, p)
	n := len(p)
	vDot8 := archsimd.BroadcastFloat32x8(dot)
	vDot4 := archsimd.BroadcastFloat32x4(dot)
	i := 0
	for ; i+8 <= n; i += 8 {
		vp := archsimd.LoadFloat32x8Slice(p[i:])
		vdp := archsimd.LoadFloat32x8Slice(dp[i:])
		vr := vp.Mul(vdp.Sub(vDot8))
		vr.StoreSlice(out[i:])
	}
	for ; i+4 <= n; i += 4 {
		vp := archsimd.LoadFloat32x4Slice(p[i:])
		vdp := archsimd.LoadFloat32x4Slice(dp[i:])
		vr := vp.Mul(vdp.Sub(vDot4))
		vr.StoreSlice(out[i:])
	}
	for ; i < n; i++ {
		out[i] = p[i] * (dp[i] - dot)
	}
}

func vecReLU(data []float32) {
	n := len(data)
	zeros8 := archsimd.BroadcastFloat32x8(0)
	zeros4 := archsimd.BroadcastFloat32x4(0)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		v = v.Max(zeros8)
		v.StoreSlice(data[i:])
	}
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat32x4Slice(data[i:])
		v = v.Max(zeros4)
		v.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		if data[i] < 0 {
			data[i] = 0
		}
	}
}

func vecScaleGradients(grads []float32, maxNorm float32) {
	n := len(grads)
	if n == 0 {
		return
	}
	sumSq := vecDot(grads, grads)
	norm := float32(math.Sqrt(float64(sumSq)))
	if norm > maxNorm {
		scaleFactor := maxNorm / (norm + 1e-8)
		vecMulScalar(grads, scaleFactor, grads)
	}
}

func vecMaxSlice(data []float32) float32 {
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
		// Reduce vMax to 4-wide
		vMax4 := vMax.GetLo().Max(vMax.GetHi())
		var buf [4]float32
		vMax4.Store(&buf)
		for _, v := range buf {
			if v > maxVal {
				maxVal = v
			}
		}
	} else if n >= 4 {
		vMax4 := archsimd.LoadFloat32x4Slice(data)
		i = 4
		var buf [4]float32
		vMax4.Store(&buf)
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

func vecAdamWUpdate(weights, grads, m, v []float32, lr, beta1, beta2, eps, weightDecay float32, t int) {
	n := len(weights)
	biasCorrection1 := float32(1.0 - math.Pow(float64(beta1), float64(t)))
	biasCorrection2 := float32(1.0 - math.Pow(float64(beta2), float64(t)))
	
	vB1 := archsimd.BroadcastFloat32x8(beta1)
	vB1Inv := archsimd.BroadcastFloat32x8(1.0 - beta1)
	vB2 := archsimd.BroadcastFloat32x8(beta2)
	vB2Inv := archsimd.BroadcastFloat32x8(1.0 - beta2)
	vBC1 := archsimd.BroadcastFloat32x8(biasCorrection1)
	vBC2 := archsimd.BroadcastFloat32x8(biasCorrection2)

	i := 0
	for ; i+8 <= n; i += 8 {
		w := archsimd.LoadFloat32x8Slice(weights[i:])
		g := archsimd.LoadFloat32x8Slice(grads[i:])
		mv := archsimd.LoadFloat32x8Slice(m[i:])
		vv := archsimd.LoadFloat32x8Slice(v[i:])

		// m = beta1 * m + (1 - beta1) * grad
		mv = vB1.Mul(mv).Add(vB1Inv.Mul(g))
		mv.StoreSlice(m[i:])

		// v = beta2 * v + (1 - beta2) * grad^2
		vv = vB2.Mul(vv).Add(vB2Inv.Mul(g.Mul(g)))
		vv.StoreSlice(v[i:])

		// mHat / (sqrt(vHat) + eps)
		mHat := mv.Div(vBC1)
		vHat := vv.Div(vBC2)
		
		var mH, vH [8]float32
		mHat.StoreSlice(mH[:])
		vHat.StoreSlice(vH[:])
		
		var wData [8]float32
		w.StoreSlice(wData[:])
		
		for j := 0; j < 8; j++ {
			update := mH[j] / (float32(math.Sqrt(float64(vH[j]))) + eps)
			// AdamW: w = w - lr * update - lr * weightDecay * w
			wData[j] -= lr * (update + weightDecay*wData[j])
		}
		newW := archsimd.LoadFloat32x8Slice(wData[:])
		newW.StoreSlice(weights[i:])
	}
	
	// Fallback
	for ; i < n; i++ {
		m[i] = beta1*m[i] + (1.0-beta1)*grads[i]
		v[i] = beta2*v[i] + (1.0-beta2)*grads[i]*grads[i]
		mHat := m[i] / biasCorrection1
		vHat := v[i] / biasCorrection2
		weights[i] -= lr * (mHat/(float32(math.Sqrt(float64(vHat)))+eps) + weightDecay*weights[i])
	}
}

func vecClipWeights(data []float32, maxVal float32) {
	n := len(data)
	vMax8 := archsimd.BroadcastFloat32x8(maxVal)
	vMin8 := archsimd.BroadcastFloat32x8(-maxVal)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		v = v.Min(vMax8).Max(vMin8)
		v.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
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
	
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		d := [8]float32{}
		v.StoreSlice(d[:])
		for j := 0; j < 8; j++ {
			if d[j] < threshold {
				d[j] = 0
			}
		}
		newV := archsimd.LoadFloat32x8Slice(d[:])
		newV.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		if data[i] < threshold {
			data[i] = 0
		}
	}
}


func vecLeakyReLU(data []float32, alpha float32) {
	n := len(data)
	vAlpha8 := archsimd.BroadcastFloat32x8(alpha)
	vZero8 := archsimd.BroadcastFloat32x8(0.0)
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(data[i:])
		vPos := v.Max(vZero8)
		vNeg := v.Min(vZero8).Mul(vAlpha8)
		res := vPos.Add(vNeg)
		res.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		if data[i] < 0 {
			data[i] *= alpha
		}
	}
}

// vecMatMul performs a SIMD-accelerated matrix multiplication (C = A @ B).
// This implementation uses a row-major blocked approach (IKJ order) for cache-friendliness
// and processes B in 256-bit (AVX2) or 128-bit (SSE) chunks.
func MatMulRaw(a, b, res []float32, m, n, k int) {
	if m < 4 {
		matMulRawSequential(a, b, res, m, n, k, 0, m)
		return
	}

	numWorkers := 8
	if m < 8 {
		numWorkers = m
	}

	var wg sync.WaitGroup
	rowsPerWorker := (m + numWorkers - 1) / numWorkers
	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if startRow >= m {
			break
		}
		if endRow > m {
			endRow = m
		}
		wg.Add(1)
		go func(sRow, eRow int) {
			defer wg.Done()
			matMulRawSequential(a, b, res, m, n, k, sRow, eRow)
		}(startRow, endRow)
	}
	wg.Wait()
}

func matMulRawSequential(a, b, res []float32, m, n, k, startRow, endRow int) {
	for i := startRow; i < endRow; i++ {
		rowA := a[i*k : (i+1)*k]
		rowRes := res[i*n : (i+1)*n]
		for ik := 0; ik < k; ik++ {
			aik := rowA[ik]
			if aik == 0 {
				continue // Sparse optimization
			}
			
			rowB := b[ik*n : (ik+1)*n]
			vA8 := archsimd.BroadcastFloat32x8(aik)
			vA4 := archsimd.BroadcastFloat32x4(aik)
			
			j := 0
			// 8-wide (AVX2)
			for ; j+8 <= n; j += 8 {
				vB8 := archsimd.LoadFloat32x8Slice(rowB[j:])
				vR8 := archsimd.LoadFloat32x8Slice(rowRes[j:])
				vR8 = vR8.Add(vA8.Mul(vB8))
				vR8.StoreSlice(rowRes[j:])
			}
			// 4-wide (SSE)
			for ; j+4 <= n; j += 4 {
				vB4 := archsimd.LoadFloat32x4Slice(rowB[j:])
				vR4 := archsimd.LoadFloat32x4Slice(rowRes[j:])
				vR4 = vR4.Add(vA4.Mul(vB4))
				vR4.StoreSlice(rowRes[j:])
			}
			// Tail
			for ; j < n; j++ {
				rowRes[j] += aik * rowB[j]
			}
		}
	}
}
