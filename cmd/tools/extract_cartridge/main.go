package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/golangast/gollemer/internal/ai/moe"
)

func main() {
	inPath := flag.String("model", "data/models/gob_models/moe_social_model.gob", "Path to the full MoE model checkpoint")
	outPath := flag.String("out", "data/models/gob_models/extracted_expert.gob", "Path to save the extracted expert")
	expertIdx := flag.Int("expert", 0, "Index of the expert to extract from the OutputMoE layer")
	flag.Parse()

	file, err := os.Open(*inPath)
	if err != nil {
		log.Fatalf("Failed to open model: %v", err)
	}
	defer file.Close()

	var model moe.IntentMoE
	if err := gob.NewDecoder(file).Decode(&model); err != nil {
		log.Fatalf("Failed to decode model: %v", err)
	}

	if model.Decoder == nil || model.Decoder.OutputMoE == nil {
		log.Fatalf("Model does not have an OutputMoE layer")
	}

	if *expertIdx < 0 || *expertIdx >= len(model.Decoder.OutputMoE.Experts) {
		log.Fatalf("Invalid expert index: %d", *expertIdx)
	}

	expert := model.Decoder.OutputMoE.Experts[*expertIdx]

	out, err := os.Create(*outPath)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer out.Close()

	if err := gob.NewEncoder(out).Encode(&expert); err != nil {
		log.Fatalf("Failed to encode expert: %v", err)
	}

	fmt.Printf("✅ Extracted expert %d to %s\n", *expertIdx, *outPath)
}
