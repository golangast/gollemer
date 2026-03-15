//go:build goexperiment.simd

package tensor

import (
	"math"
	"simd/archsimd"
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
