package main

import (
	"encoding/binary"
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	inPath := flag.String("weights", "", "Path to the input .gob expert weights")
	namespace := flag.String("namespace", "", "The intent namespace this cartridge registers under")
	outPath := flag.String("out", "", "Path to the output .cartridge file")
	flag.Parse()

	if *inPath == "" || *namespace == "" || *outPath == "" {
		flag.Usage()
		log.Fatal("Missing required arguments")
	}

	inFile, err := os.Open(*inPath)
	if err != nil {
		log.Fatalf("Failed to open input weights: %v", err)
	}
	defer inFile.Close()

	var expert moe.Expert
	decoder := gob.NewDecoder(inFile)
	if err := decoder.Decode(&expert); err != nil {
		log.Fatalf("Failed to decode input weights: %v", err)
	}

	ffExpert, ok := expert.(*moe.FeedForwardExpert)
	if !ok {
		log.Fatalf("Compiled cartridges only support FeedForwardExpert right now")
	}

	outFile, err := os.Create(*outPath)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer outFile.Close()

	header := moe.CartridgeHeader{
		Version:   1,
		InputDim:  uint32(ffExpert.Layer1.Weights.Shape[0]),
		HiddenDim: uint32(ffExpert.Layer1.Weights.Shape[1]),
		OutputDim: uint32(ffExpert.Layer2.Weights.Shape[1]),
	}
	copy(header.Magic[:], "GLMR_CRT")

	// Copy namespace, padding with zeros if shorter, truncating if longer
	ns := []byte(*namespace)
	if len(ns) > 32 {
		ns = ns[:32]
	}
	copy(header.Namespace[:], ns)

	if err := binary.Write(outFile, binary.LittleEndian, &header); err != nil {
		log.Fatalf("Failed to write header: %v", err)
	}

	// Write weights sequentially
	if err := binary.Write(outFile, binary.LittleEndian, ffExpert.Layer1.Weights.Data); err != nil {
		log.Fatalf("Failed to write Layer1 Weights: %v", err)
	}
	if ffExpert.Layer1.Biases != nil {
		if err := binary.Write(outFile, binary.LittleEndian, ffExpert.Layer1.Biases.Data); err != nil {
			log.Fatalf("Failed to write Layer1 Biases: %v", err)
		}
	}

	if err := binary.Write(outFile, binary.LittleEndian, ffExpert.Layer2.Weights.Data); err != nil {
		log.Fatalf("Failed to write Layer2 Weights: %v", err)
	}
	if ffExpert.Layer2.Biases != nil {
		if err := binary.Write(outFile, binary.LittleEndian, ffExpert.Layer2.Biases.Data); err != nil {
			log.Fatalf("Failed to write Layer2 Biases: %v", err)
		}
	}

	fmt.Printf("✅ Successfully compiled cartridge: %s (Namespace: %s)\n", *outPath, *namespace)
}
