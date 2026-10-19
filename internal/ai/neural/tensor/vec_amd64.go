//go:build amd64

package tensor

func vecDot(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return vecDotAVX2(a, b)
}

func vecAddAccumulate(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	vecAddAVX2(dst, src)
}

// Declarations matching the Go Assembly signatures
func vecDotAVX2(a, b []float32) float32
func vecAddAVX2(dst, src []float32)
func vecSubAVX2(a, b, res []float32)
func vecMulAVX2(a, b, res []float32)
func vecSoftmaxBackwardRowAVX2(p, dp, out []float32)
func vecSaxpyAVX2(dst, src []float32, alpha float32)

func vecSub(a, b, res []float32) {
	if len(a) == 0 {
		return
	}
	vecSubAVX2(a, b, res)
}

func vecMul(a, b, res []float32) {
	if len(a) == 0 {
		return
	}
	vecMulAVX2(a, b, res)
}

func vecSoftmaxBackwardRow(p, dp, out []float32) {
	if len(p) == 0 {
		return
	}
	vecSoftmaxBackwardRowAVX2(p, dp, out)
}

func vecSaxpy(dst, src []float32, alpha float32) {
	if len(dst) == 0 {
		return
	}
	vecSaxpyAVX2(dst, src, alpha)
}

func vecAdamWUpdateAVX2(weights, grads, m, v []float32, lr, oneMinusB1, oneMinusB2, beta1, beta2, eps, wd, invBc1, invBc2 float32)

func vecLayerNormFusedAVX2(out, norm, in, gamma, beta []float32, mean, rstd float32)

// VecLayerNormFused computes norm=(in-mean)*rstd and out=norm*gamma+beta in a single AVX2 pass.
// Exported for use by the nn package (LayerNorm).
func VecLayerNormFused(out, norm, in, gamma, beta []float32, mean, rstd float32) {
	if len(out) == 0 {
		return
	}
	// Process AVX2-aligned part
	n := len(out)
	vecCount := n &^ 7
	if vecCount > 0 {
		vecLayerNormFusedAVX2(out[:vecCount], norm[:vecCount], in[:vecCount], gamma[:vecCount], beta[:vecCount], mean, rstd)
	}
	// Scalar tail
	for i := vecCount; i < n; i++ {
		nv := (in[i] - mean) * rstd
		norm[i] = nv
		out[i] = nv*gamma[i] + beta[i]
	}
}
