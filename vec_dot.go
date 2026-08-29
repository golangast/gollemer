package main

// vecDot computes the dot product of vectors a and b.
func vecDot(a, b []float32) float32 {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	var sum float32
	for i := 0; i < minLen; i++ {
		sum += a[i] * b[i]
	}
	return sum
}
