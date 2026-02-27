//go:build simd

package tensor

import (
	"simd/archsimd"
)

func IsSIMDEnabled() bool {
	return true
}

const step4 = 4

func vecAdd(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vb := archsimd.LoadFloat64x4Slice(b[i : i+step4])
		vr := va.Add(vb)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] = a[i] + b[i]
	}
}

func vecSub(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vb := archsimd.LoadFloat64x4Slice(b[i : i+step4])
		vr := va.Sub(vb)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] = a[i] - b[i]
	}
}

func vecMul(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vb := archsimd.LoadFloat64x4Slice(b[i : i+step4])
		vr := va.Mul(vb)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] = a[i] * b[i]
	}
}

func vecDiv(a, b, res []float64) {
	n := len(a)
	i := 0
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vb := archsimd.LoadFloat64x4Slice(b[i : i+step4])
		vr := va.Div(vb)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] = a[i] / b[i]
	}
}

func vecMulScalar(a []float64, scalar float64, res []float64) {
	n := len(a)
	i := 0
	vscalar := archsimd.BroadcastFloat64x4(scalar)
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vr := va.Mul(vscalar)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] = a[i] * scalar
	}
}

func vecDivScalar(a []float64, scalar float64, res []float64) {
	n := len(a)
	i := 0
	vscalar := archsimd.BroadcastFloat64x4(scalar)
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vr := va.Div(vscalar)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] = a[i] / scalar
	}
}

func vecAddScalar(a []float64, scalar float64, res []float64) {
	n := len(a)
	i := 0
	vscalar := archsimd.BroadcastFloat64x4(scalar)
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vr := va.Add(vscalar)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] = a[i] + scalar
	}
}

func vecAddAccumulate(res, a []float64) {
	n := len(a)
	i := 0
	for ; i <= n-step4; i += step4 {
		vr := archsimd.LoadFloat64x4Slice(res[i : i+step4])
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vr = va.Add(vr) // res += a
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] += a[i]
	}
}

func vecMulAccumulate(res, a, b []float64) {
	n := len(a)
	i := 0
	for ; i <= n-step4; i += step4 {
		vr := archsimd.LoadFloat64x4Slice(res[i : i+step4])
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vb := archsimd.LoadFloat64x4Slice(b[i : i+step4])
		vr = va.MulAdd(vb, vr) // res += a * b
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] += a[i] * b[i]
	}
}

func vecDivAccumulate(res, a []float64, scalar float64) {
	n := len(a)
	i := 0
	v_inv_scalar := archsimd.BroadcastFloat64x4(1.0 / scalar)
	for ; i <= n-step4; i += step4 {
		vr := archsimd.LoadFloat64x4Slice(res[i : i+step4])
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vr = va.MulAdd(v_inv_scalar, vr) // res += a * (1/scalar)
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] += a[i] / scalar
	}
}

func vecMulScalarAccumulate(res, a []float64, scalar float64) {
	n := len(a)
	i := 0
	vscalar := archsimd.BroadcastFloat64x4(scalar)
	for ; i <= n-step4; i += step4 {
		vr := archsimd.LoadFloat64x4Slice(res[i : i+step4])
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vr = va.MulAdd(vscalar, vr) // res += a * scalar
		vr.StoreSlice(res[i : i+step4])
	}
	for ; i < n; i++ {
		res[i] += a[i] * scalar
	}
}

func vecSum(a []float64) float64 {
	n := len(a)
	i := 0
	vsum := archsimd.BroadcastFloat64x4(0)
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vsum = vsum.Add(va)
	}

	// Horizontal sum of vsum
	var res [4]float64
	vsum.Store(&res)
	sum := res[0] + res[1] + res[2] + res[3]

	for ; i < n; i++ {
		sum += a[i]
	}
	return sum
}

func vecDot(a, b []float64) float64 {
	n := len(a)
	i := 0
	vsum := archsimd.BroadcastFloat64x4(0)
	for ; i <= n-step4; i += step4 {
		va := archsimd.LoadFloat64x4Slice(a[i : i+step4])
		vb := archsimd.LoadFloat64x4Slice(b[i : i+step4])
		vsum = vsum.MulAdd(va, vb) // vsum += va * vb
	}
	
	// Horizontal sum
	var sums [4]float64
	vsum.Store(&sums)
	total := sums[0] + sums[1] + sums[2] + sums[3]
	
	for ; i < n; i++ {
		total += a[i] * b[i]
	}
	return total
}

// vecSoftmaxBackwardRow computes the softmax Jacobian-vector product for one row:
// out[i] = p[i] * (dp[i] - dot(dp, p))
// p is the softmax output row, dp is the upstream gradient row, out is the result.
func vecSoftmaxBackwardRow(p, dp, out []float64) {
	n := len(p)
	dot := vecDot(dp, p)
	vdot := archsimd.BroadcastFloat64x4(dot)
	i := 0
	for ; i <= n-step4; i += step4 {
		vp := archsimd.LoadFloat64x4Slice(p[i : i+step4])
		vdp := archsimd.LoadFloat64x4Slice(dp[i : i+step4])
		// vdp - vdot
		vdiff := vdp.Sub(vdot)
		// vp * vdiff
		vout := vp.Mul(vdiff)
		vout.StoreSlice(out[i : i+step4])
	}
	for ; i < n; i++ {
		out[i] = p[i] * (dp[i] - dot)
	}
}
