//go:build arm64

package tensor

func vecDot(a, b []float32) float32 {
	if len(a) == 0 {
		return 0
	}
	return vecDotNEON(a, b)
}

func vecAddAccumulate(dst, src []float32) {
	if len(dst) == 0 {
		return
	}
	vecAddNEON(dst, src)
}

// Declarations matching the Go Assembly signatures
func vecDotNEON(a, b []float32) float32
func vecAddNEON(dst, src []float32)
func vecSubNEON(a, b, res []float32)
func vecMulNEON(a, b, res []float32)
func vecSoftmaxBackwardRowNEON(p, dp, out []float32)

func vecSub(a, b, res []float32) {
	if len(a) == 0 {
		return
	}
	vecSubNEON(a, b, res)
}

func vecMul(a, b, res []float32) {
	if len(a) == 0 {
		return
	}
	vecMulNEON(a, b, res)
}

func vecSoftmaxBackwardRow(p, dp, out []float32) {
	if len(p) == 0 {
		return
	}
	vecSoftmaxBackwardRowNEON(p, dp, out)
}
