package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/golangast/gollemer/internal/ai/moe"
)

type CartridgeSpec struct {
	Filename  string
	Namespace string
}

func main() {
	outDir := "data/models/intents"
	if err := os.MkdirAll(outDir, 0755); err != nil {
		log.Fatalf("Failed to create dir: %v", err)
	}

	specs := []CartridgeSpec{
		{Filename: "goroutine_fix.cartridge", Namespace: "goroutine_fix"},
		{Filename: "sql_builder.cartridge", Namespace: "sql_builder"},
		{Filename: "unit_test.cartridge", Namespace: "unit_test"},
		{Filename: "gin_router.cartridge", Namespace: "gin_router"},
		{Filename: "gorm_model.cartridge", Namespace: "gorm_model"},
		{Filename: "protobuf.cartridge", Namespace: "protobuf"},
	}

	inputDim := 768
	hiddenDim := 256
	outputDim := 768

	for _, spec := range specs {
		outPath := filepath.Join(outDir, spec.Filename)
		expert, err := moe.NewFeedForwardExpert(inputDim, hiddenDim, outputDim)
		if err != nil {
			log.Fatalf("Failed to create expert for %s: %v", spec.Filename, err)
		}

		file, err := os.Create(outPath)
		if err != nil {
			log.Fatalf("Failed to create file %s: %v", outPath, err)
		}

		header := moe.CartridgeHeader{
			Version:   1,
			InputDim:  uint32(inputDim),
			HiddenDim: uint32(hiddenDim),
			OutputDim: uint32(outputDim),
		}
		copy(header.Magic[:], "GLMR_CRT")
		ns := []byte(spec.Namespace)
		if len(ns) > 32 {
			ns = ns[:32]
		}
		copy(header.Namespace[:], ns)

		if err := binary.Write(file, binary.LittleEndian, &header); err != nil {
			file.Close()
			log.Fatalf("Failed to write header for %s: %v", outPath, err)
		}

		if err := binary.Write(file, binary.LittleEndian, expert.Layer1.Weights.Data); err != nil {
			file.Close()
			log.Fatalf("Failed to write layer 1 weights for %s: %v", outPath, err)
		}
		if expert.Layer1.Biases != nil {
			if err := binary.Write(file, binary.LittleEndian, expert.Layer1.Biases.Data); err != nil {
				file.Close()
				log.Fatalf("Failed to write layer 1 biases for %s: %v", outPath, err)
			}
		}

		if err := binary.Write(file, binary.LittleEndian, expert.Layer2.Weights.Data); err != nil {
			file.Close()
			log.Fatalf("Failed to write layer 2 weights for %s: %v", outPath, err)
		}
		if expert.Layer2.Biases != nil {
			if err := binary.Write(file, binary.LittleEndian, expert.Layer2.Biases.Data); err != nil {
				file.Close()
				log.Fatalf("Failed to write layer 2 biases for %s: %v", outPath, err)
			}
		}

		file.Close()
		fmt.Printf("✅ Created cartridge: %s (Namespace: %s)\n", outPath, spec.Namespace)
	}
}
