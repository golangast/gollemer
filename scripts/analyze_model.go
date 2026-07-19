//go:build ignore
// +build ignore

package main

import (
	"encoding/gob"
	"fmt"
	"os"
	"reflect"

	"github.com/golangast/gollemer/internal/ai/moe"
	_ "github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run scripts/analyze_model.go <checkpoint_path>")
		os.Exit(1)
	}

	path := os.Args[1]
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Error opening file: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	var ckpt moe.Checkpoint
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&ckpt)
	if err != nil {
		fmt.Printf("Error decoding checkpoint: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Analysis of Checkpoint: %s\n", path)
	fmt.Printf("Step: %d, Version: %s\n", ckpt.StepCount, ckpt.Version)

	analyzeModel(ckpt.Model)
}

func analyzeModel(m *moe.IntentMoE) {
	totalParams := 0
	tensorCount := 0
	
	v := reflect.ValueOf(m).Elem()
	findTensors(v, &totalParams, &tensorCount)

	fmt.Printf("Total Tensors Found: %d\n", tensorCount)
	fmt.Printf("Total Parameters Found: %d\n", totalParams)
	fmt.Printf("Estimated Data Size (FP64): %.2f GB\n", float64(totalParams*8)/1e9)
}

func findTensors(v reflect.Value, totalParams *int, tensorCount *int) {
	if !v.IsValid() {
		return
	}

	if v.Type() == reflect.TypeOf(&tensor.Tensor{}) && !v.IsNil() {
		t := v.Interface().(*tensor.Tensor)
		*tensorCount++
		*totalParams += len(t.Data)
		return
	}

	switch v.Kind() {
	case reflect.Ptr, reflect.Interface:
		if !v.IsNil() {
			findTensors(v.Elem(), totalParams, tensorCount)
		}
	case reflect.Struct:
		for i := 0; i < v.NumField(); i++ {
			findTensors(v.Field(i), totalParams, tensorCount)
		}
	case reflect.Slice, reflect.Array:
		for i := 0; i < v.Len(); i++ {
			findTensors(v.Index(i), totalParams, tensorCount)
		}
	case reflect.Map:
		for _, key := range v.MapKeys() {
			findTensors(v.MapIndex(key), totalParams, tensorCount)
		}
	}
}
