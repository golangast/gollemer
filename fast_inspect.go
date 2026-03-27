package main

import (
	"encoding/gob"
	"fmt"
	"os"
)

type Checkpoint struct {
	StepCount  int
	Commitment float64
}

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run fast_inspect.go <path>")
		return
	}
	path := os.Args[1]
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer file.Close()

	var ckpt Checkpoint
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&ckpt) // Should ignore other fields
	if err != nil {
		fmt.Printf("Error decoding %s: %v\n", path, err)
		return
	}

	fmt.Printf("%s | Steps: %d | IQ: %.4f\n", path, ckpt.StepCount, ckpt.Commitment)
}
