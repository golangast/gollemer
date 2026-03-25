package main

import (
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nn"
)

// Minimal version of Checkpoint for headers
type CheckpointHeader struct {
	StepCount       int
	Version         string
	Commitment      float64
	TokensProcessed int64
	TotalDuration   string
	LastProfile     nn.TrainingProfile
}

func main() {
	exportFlag := flag.Bool("export", false, "Export metadata to JSON")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Println("Usage: go run inspect_model.go [--export] <path_to_gob>")
		return
	}

	path := flag.Arg(0)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("❌ Error opening file: %v\n", err)
		return
	}
	defer file.Close()

	var ckpt moe.Checkpoint
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&ckpt)
	if err != nil {
		fmt.Printf("❌ Error decoding checkpoint: %v\n", err)
		return
	}

	fmt.Println("------------------------------------------")
	fmt.Printf("📂 Model Checkpoint: %s\n", path)
	fmt.Printf("🆔 Version:         %s\n", ckpt.Version)
	fmt.Printf("🔢 Total Steps:     %d\n", ckpt.StepCount)
	fmt.Printf("🛠️  Last Profile:    %s\n", ckpt.LastProfile.Name)
	fmt.Printf("📉 Learning Rate:   %.6f\n", ckpt.LastProfile.LR)
	fmt.Printf("⚖️  Weight Decay:    %.6f (Lambda)\n", ckpt.LastProfile.Lambda)
	fmt.Printf("🧠 Commitment:      %.2f%%\n", ckpt.Commitment)
	fmt.Printf("⌛ Duration:        %v\n", ckpt.TotalDuration)
	if ckpt.TotalDuration > 0 {
		fmt.Printf("⚡ Throughput:      %.2f tokens/sec\n", float64(ckpt.TokensProcessed)/ckpt.TotalDuration.Seconds())
	}
	fmt.Println("------------------------------------------")

	if *exportFlag {
		jsonPath := path + ".json"
		report := CheckpointHeader{
			StepCount:       ckpt.StepCount,
			Version:         ckpt.Version,
			Commitment:      ckpt.Commitment,
			TokensProcessed: ckpt.TokensProcessed,
			TotalDuration:   ckpt.TotalDuration.String(),
			LastProfile:     ckpt.LastProfile,
		}

		jsonData, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Printf("❌ Error marshaling JSON: %v\n", err)
			return
		}

		err = os.WriteFile(jsonPath, jsonData, 0644)
		if err != nil {
			fmt.Printf("❌ Error saving JSON: %v\n", err)
			return
		}
		// fmt.Printf("📊 Report exported to: %s\n", jsonPath)
	}
}
