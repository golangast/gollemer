package ops

import (
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func BenchmarkAddVectors(b *testing.B) {
	a := make([]float32, 4096)
	w := make([]float32, 4096)
	res := make([]float32, 4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor.AddVectors(a, w, res)
	}
}

func BenchmarkDotProduct(b *testing.B) {
	a := make([]float32, 4096)
	w := make([]float32, 4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tensor.DotProduct(a, w)
	}
}

func BenchmarkSoftmaxBackwardRow(b *testing.B) {
	p := make([]float32, 4096)
	dp := make([]float32, 4096)
	out := make([]float32, 4096)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor.SoftmaxBackwardRow(p, dp, out)
	}
}
