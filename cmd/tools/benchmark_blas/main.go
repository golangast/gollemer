package main

import (
	"fmt"
	"time"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func main() {
	m, n, k := 1024, 1024, 1024
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	c := make([]float32, m*n)

	for i := range a { a[i] = 1.0 }
	for i := range b { b[i] = 1.0 }

	fmt.Println("🚀 Benchmarking GoffiMatMul (1024x1024)...")
	start := time.Now()
	err := tensor.GoffiMatMul(a, b, c, m, n, k)
	if err != nil {
		fmt.Printf("❌ GoffiMatMul failed: %v\n", err)
		return
	}
	duration := time.Since(start)
	
	gflops := float64(2*m*n*k) / duration.Seconds() / 1e9
	fmt.Printf("✅ GoffiMatMul finished in %v (%.2f GFLOPS)\n", duration, gflops)
}
