package tensor

import (
	"math/rand"
	"testing"
)

func BenchmarkMatMul(b *testing.B) {
	// Define dimensions for the benchmark
	rows := 128
	cols := 128
	common := 128

	// Initialize Tensor A
	shapeA := []int{rows, common}
	dataA := make([]float64, rows*common)
	for i := range dataA {
		dataA[i] = rand.Float64()
	}
	tA := NewTensor(shapeA, dataA, false)

	// Initialize Tensor B
	shapeB := []int{common, cols}
	dataB := make([]float64, common*cols)
	for i := range dataB {
		dataB[i] = rand.Float64()
	}
	tB := NewTensor(shapeB, dataB, false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := tA.MatMul(tB)
		if err != nil {
			b.Fatal(err)
		}
	}
}
