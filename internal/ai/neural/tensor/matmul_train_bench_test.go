package tensor

// Training-realistic matmul shapes: Linear layers in the seq2seq trainer run
// as [batch*steps, in] @ [in, out]. Social (128/256): 32*40=1280 rows.
// Decoder output projection: [1280, 256] @ [256, vocab~3000].
// Gocode (256/512): wider. These mirror what -train-real-seq2seq actually does.

import (
	"math/rand"
	"testing"
)

func benchMatMulShape(b *testing.B, m, n, k int) {
	b.Helper()
	a := make([]float32, m*k)
	bb := make([]float32, k*n)
	for i := range a {
		a[i] = float32(rand.NormFloat64())
	}
	for i := range bb {
		bb[i] = float32(rand.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := make([]float32, m*n)
		MatMulRaw(a, bb, res, m, n, k)
	}
}

func BenchmarkMatMulTrainSocial(b *testing.B) { benchMatMulShape(b, 1280, 256, 128) }
func BenchmarkMatMulTrainOutput(b *testing.B) { benchMatMulShape(b, 1280, 3000, 256) }
func BenchmarkMatMulTrainGocode(b *testing.B) { benchMatMulShape(b, 1920, 512, 256) }
