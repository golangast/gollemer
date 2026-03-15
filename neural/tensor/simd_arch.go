//go:build goexperiment.simd

package tensor

import (
	"math"
	"simd/archsimd"
	"sort"
)

func IsSIMDEnabled() bool {
	return true
}

func vecAdd(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vb := archsimd.LoadFloat64x4Slice(b[i:])
		vr := va.Add(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] + b[i]
	}
}

func vecSub(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vb := archsimd.LoadFloat64x4Slice(b[i:])
		vr := va.Sub(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] - b[i]
	}
}

func vecMul(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vb := archsimd.LoadFloat64x4Slice(b[i:])
		vr := va.Mul(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] * b[i]
	}
}

func vecDiv(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vb := archsimd.LoadFloat64x4Slice(b[i:])
		vr := va.Div(vb)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] / b[i]
	}
}

func vecMulScalar(a []float64, scalar float64, res []float64) {
	n := len(a)
	vScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vr := va.Mul(vScalar)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] * scalar
	}
}

func vecDivScalar(a []float64, scalar float64, res []float64) {
	n := len(a)
	vScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vr := va.Div(vScalar)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] / scalar
	}
}

func vecAddScalar(a []float64, scalar float64, res []float64) {
	n := len(a)
	vScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vr := va.Add(vScalar)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] = a[i] + scalar
	}
}

func vecAddAccumulate(res, a []float64) {
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat64x4Slice(res[i:])
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vr = vr.Add(va)
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i]
	}
}

func vecMulAccumulate(res, a, b []float64) {
	n := len(a)
	i := 0
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat64x4Slice(res[i:])
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vb := archsimd.LoadFloat64x4Slice(b[i:])
		vr = vr.Add(va.Mul(vb))
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i] * b[i]
	}
}

func vecDivAccumulate(res, a []float64, scalar float64) {
	n := len(a)
	vScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat64x4Slice(res[i:])
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vr = vr.Add(va.Div(vScalar))
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i] / scalar
	}
}

func vecMulScalarAccumulate(res, a []float64, scalar float64) {
	n := len(a)
	vScalar := archsimd.BroadcastFloat64x4(scalar)
	i := 0
	for ; i+4 <= n; i += 4 {
		vr := archsimd.LoadFloat64x4Slice(res[i:])
		va := archsimd.LoadFloat64x4Slice(a[i:])
		vr = vr.Add(va.Mul(vScalar))
		vr.StoreSlice(res[i:])
	}
	for ; i < n; i++ {
		res[i] += a[i] * scalar
	}
}

func vecSum(a []float64) float64 {
	n := len(a)
	if n == 0 {
		return 0
	}
	var acc archsimd.Float64x4
	i := 0
	for ; i+4 <= n; i += 4 {
		va := archsimd.LoadFloat64x4Slice(a[i:])
		acc = acc.Add(va)
	}
	var buf [4]float64
	acc.Store(&buf)
	sum := buf[0] + buf[1] + buf[2] + buf[3]
	for ; i < n; i++ {
		sum += a[i]
	}
	return sum
}

func vecDot(a, b []float64) float64 {
	n := len(a)
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

func vecSoftmaxBackwardRow(p, dp, out []float64) {
	dot := vecDot(dp, p)
	n := len(p)
	vDot := archsimd.BroadcastFloat64x4(dot)
	i := 0
	for ; i+4 <= n; i += 4 {
		vp := archsimd.LoadFloat64x4Slice(p[i:])
		vdp := archsimd.LoadFloat64x4Slice(dp[i:])
		vr := vp.Mul(vdp.Sub(vDot))
		vr.StoreSlice(out[i:])
	}
	for ; i < n; i++ {
		out[i] = p[i] * (dp[i] - dot)
	}
}

func vecReLU(data []float64) {
	n := len(data)
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

func vecScaleGradients(grads []float64, maxNorm float64) {
	n := len(grads)
	if n == 0 {
		return
	}
	sumSq := vecDot(grads, grads)
	norm := math.Sqrt(sumSq)
	if norm > maxNorm {
		scaleFactor := maxNorm / (norm + 1e-8)
		vecMulScalar(grads, scaleFactor, grads)
	}
}

func vecMaxSlice(data []float64) float64 {
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

func vecAdamWUpdate(weights, grads, m, v []float64, lr, beta1, beta2, eps, weightDecay float64, t int) {
	n := len(weights)
	biasCorrection1 := 1.0 - math.Pow(beta1, float64(t))
	biasCorrection2 := 1.0 - math.Pow(beta2, float64(t))
	
	vB1 := archsimd.BroadcastFloat64x4(beta1)
	vB1Inv := archsimd.BroadcastFloat64x4(1.0 - beta1)
	vB2 := archsimd.BroadcastFloat64x4(beta2)
	vB2Inv := archsimd.BroadcastFloat64x4(1.0 - beta2)
	vBC1 := archsimd.BroadcastFloat64x4(biasCorrection1)
	vBC2 := archsimd.BroadcastFloat64x4(biasCorrection2)

	i := 0
	for ; i+4 <= n; i += 4 {
		w := archsimd.LoadFloat64x4Slice(weights[i:])
		g := archsimd.LoadFloat64x4Slice(grads[i:])
		mv := archsimd.LoadFloat64x4Slice(m[i:])
		vv := archsimd.LoadFloat64x4Slice(v[i:])

		// m = beta1 * m + (1 - beta1) * grad
		mv = vB1.Mul(mv).Add(vB1Inv.Mul(g))
		mv.StoreSlice(m[i:])

		// v = beta2 * v + (1 - beta2) * grad^2
		vv = vB2.Mul(vv).Add(vB2Inv.Mul(g.Mul(g)))
		vv.StoreSlice(v[i:])

		// mHat / (sqrt(vHat) + eps)
		mHat := mv.Div(vBC1)
		vHat := vv.Div(vBC2)
		
		// Approximate sqrt using a loop or wait for archsimd.Sqrt? 
		// For now, let's use a small helper or process element-wise for the final step 
		// if archsimd doesn't have Sqrt. 
		// Actually, archsimd Float64x4 has Sqrt! (verified in typical SIMD libraries, let's assume it does or check)
		// Wait, I should verify if Sqrt exists.
		
		var mH, vH [4]float64
		mHat.Store(&mH)
		vHat.Store(&vH)
		
		var wData [4]float64
		w.Store(&wData)
		
		for j := 0; j < 4; j++ {
			update := mH[j] / (math.Sqrt(vH[j]) + eps)
			// AdamW: w = w - lr * update - lr * weightDecay * w
			wData[j] -= lr * (update + weightDecay*wData[j])
		}
		newW := archsimd.LoadFloat64x4Slice(wData[:])
		newW.StoreSlice(weights[i:])
	}
	
	// Fallback
	for ; i < n; i++ {
		m[i] = beta1*m[i] + (1.0-beta1)*grads[i]
		v[i] = beta2*v[i] + (1.0-beta2)*grads[i]*grads[i]
		mHat := m[i] / biasCorrection1
		vHat := v[i] / biasCorrection2
		weights[i] -= lr * (mHat/(math.Sqrt(vHat)+eps) + weightDecay*weights[i])
	}
}

func vecClipWeights(data []float64, maxVal float64) {
	n := len(data)
	vMax := archsimd.BroadcastFloat64x4(maxVal)
	vMin := archsimd.BroadcastFloat64x4(-maxVal)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat64x4Slice(data[i:])
		// v = min(maxVal, max(-maxVal, v))
		v = v.Min(vMax).Max(vMin)
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

func vecTopKZero(data []float64, k int) {
	n := len(data)
	if k >= n || k <= 0 {
		return
	}
	
	// Find threshold (generic logic)
	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)
	threshold := sorted[n-k]
	
	// Zero out anything below threshold (SIMD)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat64x4Slice(data[i:])
		// mask = v >= threshold
		// We can use a trick: if v < threshold, v = 0
		// archsimd doesn't have a direct "Select" or "Masked Store" for float64x4 easily 
		// depending on the version. Let's use element-wise but we can still use the threshold.
		// Actually, I'll use a SIMD-like approach or just a fast loop if Select is missing.
		
		d := [4]float64{}
		v.Store(&d)
		for j := 0; j < 4; j++ {
			if d[j] < threshold {
				d[j] = 0
			}
		}
		newV := archsimd.LoadFloat64x4Slice(d[:])
		newV.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		if data[i] < threshold {
			data[i] = 0
		}
	}
}

func vecLeakyReLU(data []float64, alpha float64) {
	n := len(data)
	vAlpha := archsimd.BroadcastFloat64x4(alpha)
	vZero := archsimd.BroadcastFloat64x4(0.0)
	i := 0
	for ; i+4 <= n; i += 4 {
		v := archsimd.LoadFloat64x4Slice(data[i:])
		// v_pos = max(0, v)
		// v_neg = min(0, v) * alpha
		vPos := v.Max(vZero)
		vNeg := v.Min(vZero).Mul(vAlpha)
		res := vPos.Add(vNeg)
		res.StoreSlice(data[i:])
	}
	for ; i < n; i++ {
		if data[i] < 0 {
			data[i] *= alpha
		}
	}
}
