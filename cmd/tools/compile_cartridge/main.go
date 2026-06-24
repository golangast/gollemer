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

	// --- Dynamic Unpacking of Expert Types ---
	var ffExpert *moe.FeedForwardExpert

	switch e := expert.(type) {
	case *moe.FeedForwardExpert:
		ffExpert = e
		log.Println("Successfully resolved native FeedForwardExpert structural layout.")

	case *moe.InternalExpert:
		log.Println("Mapping InternalExpert parameters to cartridge specifications...")
		// Bridge InternalExpert layers over to the target compiler structural representation
		ffExpert = &moe.FeedForwardExpert{
			Layer1:        e.GetFC1(), // See implementation extension below if fields are unexported
			Layer2:        e.GetFC2(),
			ActivationEMA: e.GetHealth(),
		}

	default:
		log.Fatalf("Compiled cartridges do not support expert type: %T", expert)
	}
	// ------------------------------------------

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

	ns := []byte(*namespace)
	if len(ns) > 32 {
		ns = ns[:32]
	}
	copy(header.Namespace[:], ns)

	if err := binary.Write(outFile, binary.LittleEndian, &header); err != nil {
		log.Fatalf("Failed to write header: %v", err)
	}

	// Write weights sequentially for Mmap compatibility
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
