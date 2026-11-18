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
