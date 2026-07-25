//go:build !amd64 && !arm64

package tensor

func vecDot(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}
	_ = a[n-1]
	_ = b[n-1]

	var sum0, sum1, sum2, sum3 float32
	i := 0
	for ; i <= n-4; i += 4 {
		sum0 += a[i] * b[i]
		sum1 += a[i+1] * b[i+1]
		sum2 += a[i+2] * b[i+2]
		sum3 += a[i+3] * b[i+3]
	}
	sum := (sum0 + sum1) + (sum2 + sum3)
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

func vecAddAccumulate(dst, src []float32) {
	n := len(dst)
	if n == 0 {
		return
	}
	_ = dst[n-1]
	_ = src[n-1]

	i := 0
	for ; i <= n-4; i += 4 {
		dst[i] += src[i]
		dst[i+1] += src[i+1]
		dst[i+2] += src[i+2]
		dst[i+3] += src[i+3]
	}
	for ; i < n; i++ {
		dst[i] += src[i]
	}
}

func vecSub(a, b, res []float32) {
	for i := range a {
		res[i] = a[i] - b[i]
	}
}

func vecMul(a, b, res []float32) {
	for i := range a {
		res[i] = a[i] * b[i]
	}
}

func vecSoftmaxBackwardRow(p, dp, out []float32) {
	dot := vecDot(dp, p)
	for i := range p {
		out[i] = p[i] * (dp[i] - dot)
	}
}
