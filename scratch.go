package main

import (
	"fmt"
	"math"
)

func main() {
	var f float32 = 9659998850252800.0
	fmt.Printf("%f\n", f)
	fmt.Printf("%b\n", math.Float32bits(f))
}
