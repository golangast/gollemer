package tensor

import (
	"math/rand"
	"testing"
)

func TestMatMul2D(t *testing.T) {
	a := NewTensor([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}, false)
	b := NewTensor([]int{3, 2}, []float32{7, 8, 9, 10, 11, 12}, false)
	// [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
	// [[7+18+33, 8+20+36], [28+45+66, 32+50+72]]
	// [[58, 64], [139, 154]]
	res, err := a.MatMul(b)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float32{58, 64, 139, 154}
	for i, v := range res.Data {
		if v != expected[i] {
			t.Errorf("expected %f, got %f", expected[i], v)
		}
	}
}

func TestMatMul3D(t *testing.T) {
	// Simple 3D case: batch size 2, 2x2 matrices
	a := NewTensor([]int{2, 2, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, false)
	b := NewTensor([]int{2, 2, 2}, []float32{1, 0, 0, 1, 0, 1, 1, 0}, false)
	// Slice 0: [[1, 2], [3, 4]] * [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
	// Slice 1: [[5, 6], [7, 8]] * [[0, 1], [1, 0]] = [[6, 5], [8, 7]]
	res, err := a.MatMul(b)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float32{1, 2, 3, 4, 6, 5, 8, 7}
	for i, v := range res.Data {
		if v != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestMatMul4D(t *testing.T) {
	// Case: batch=1, heads=2, 2x2 matrices
	a := NewTensor([]int{1, 2, 2, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, false)
	b := NewTensor([]int{1, 2, 2, 2}, []float32{1, 0, 0, 1, 0, 1, 1, 0}, false)
	// Head 0: [[1, 2], [3, 4]] * [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
	// Head 1: [[5, 6], [7, 8]] * [[0, 1], [1, 0]] = [[6, 5], [8, 7]]
	res, err := a.MatMul(b)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float32{1, 2, 3, 4, 6, 5, 8, 7}
	for i, v := range res.Data {
		if v != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func TestMatMul3Dx2D(t *testing.T) {
	// Case 4: 3D [2, 2, 2] x 2D [2, 2]
	a := NewTensor([]int{2, 2, 2}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, false)
	b := NewTensor([]int{2, 2}, []float32{1, 1, 1, 1}, false)
	// Slice 0: [[1, 2], [3, 4]] * [[1, 1], [1, 1]] = [[3, 3], [7, 7]]
	// Slice 1: [[5, 6], [7, 8]] * [[1, 1], [1, 1]] = [[11, 11], [15, 15]]
	res, err := a.MatMul(b)
	if err != nil {
		t.Fatal(err)
	}
	expected := []float32{3, 3, 7, 7, 11, 11, 15, 15}
	for i, v := range res.Data {
		if v != expected[i] {
			t.Errorf("At index %d: expected %f, got %f", i, expected[i], v)
		}
	}
}

func BenchmarkMatMul(b *testing.B) {
	// Define dimensions for the benchmark
	rows := 512
	cols := 512
	common := 512

	// Initialize Tensor A
	shapeA := []int{rows, common}
	dataA := make([]float32, rows*common)
	for i := range dataA {
		dataA[i] = float32(rand.Float64())
	}
	tA := NewTensor(shapeA, dataA, false)

	// Initialize Tensor B
	shapeB := []int{common, cols}
	dataB := make([]float32, common*cols)
	for i := range dataB {
		dataB[i] = float32(rand.Float64())
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
