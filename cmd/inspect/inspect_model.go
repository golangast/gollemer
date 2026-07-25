package main

import (
	"encoding/binary"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nn"
)

// CartridgeHeader mirrors the binary layout from cartridge.go for inspection.
type CartridgeHeader struct {
	Magic     [8]byte  // "GLMR_CRT"
	Version   uint32   // Engine version
	Namespace [32]byte // Intent namespace
	InputDim  uint32
	HiddenDim uint32
	OutputDim uint32
}

// Minimal version of Checkpoint for headers
type CheckpointHeader struct {
	StepCount       int
	Version         string
	Commitment      float32
	TokensProcessed int64
	TotalDuration   string
	LastProfile     nn.TrainingProfile
}

func main() {
	exportFlag := flag.Bool("export", false, "Export metadata to JSON")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Println("Usage: go run inspect_model.go [--export] <path_to_gob_or_cartridge>")
		return
	}

	path := flag.Arg(0)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("❌ Error opening file: %v\n", err)
		return
	}
	defer file.Close()

	fi, err := file.Stat()
	if err != nil || fi.Size() == 0 {
		fmt.Println("❌ Error: model file is empty or unreadable")
		return
	}

	sep := strings.Repeat("─", 52)

	// ── 0. Binary .cartridge format ──────────────────────────────────────────
	{
		var hdr CartridgeHeader
		if err := binary.Read(file, binary.LittleEndian, &hdr); err == nil && string(hdr.Magic[:]) == "GLMR_CRT" {
			namespace := strings.TrimRight(string(hdr.Namespace[:]), "\x00")
			fmt.Println(sep)
			fmt.Printf("📂 File:           %s\n", path)
			fmt.Printf("📦 Size:           %.1f MB\n", float64(fi.Size())/1_000_000)
			fmt.Printf("🏷️  Format:         Binary Cartridge (GLMR_CRT)\n")
			fmt.Printf("🆔 Cartridge v%d\n", hdr.Version)
			fmt.Printf("🔤 Namespace:      %s\n", namespace)
			fmt.Printf("📐 Input Dim:      %d\n", hdr.InputDim)
			fmt.Printf("🔧 Hidden Dim:     %d\n", hdr.HiddenDim)
			fmt.Printf("📊 Output Dim:     %d\n", hdr.OutputDim)
			fmt.Println(sep)

			if *exportFlag {
				jsonData, _ := json.MarshalIndent(map[string]interface{}{
					"type":       "cartridge",
					"version":    hdr.Version,
					"namespace":  namespace,
					"input_dim":  hdr.InputDim,
					"hidden_dim": hdr.HiddenDim,
					"output_dim": hdr.OutputDim,
				}, "", "  ")
				jsonPath := path + ".inspect.json"
				os.WriteFile(jsonPath, jsonData, 0644)
				fmt.Printf("📊 Exported to: %s\n", jsonPath)
			}
			return
		}
		_, _ = file.Seek(0, io.SeekStart)
	}

	// ── 1. Gob Checkpoint (legacy) ───────────────────────────────────────────
	var ckpt moe.Checkpoint
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&ckpt)
	if err != nil {
		// ── 2. Fallback: try decoding as a raw Expert ────────────────────────────
		_, _ = file.Seek(0, io.SeekStart)
		var expert moe.Expert
		if decErr2 := gob.NewDecoder(file).Decode(&expert); decErr2 == nil {
			fmt.Println(sep)
			fmt.Printf("📂 Expert:          %s\n", path)
			fmt.Printf("📦 Size:            %.1f MB\n", float64(fi.Size())/1_000_000)
			fmt.Printf("🏷️  Format:          Gob Expert\n")
			fmt.Printf("🔤 Type:            %s\n", expert.Description())

			if *exportFlag {
				jsonData, _ := json.MarshalIndent(map[string]interface{}{
					"type":   "expert",
					"format": "gob",
					"desc":   expert.Description(),
				}, "", "  ")
				jsonPath := path + ".inspect.json"
				os.WriteFile(jsonPath, jsonData, 0644)
				fmt.Printf("📊 Exported to: %s\n", jsonPath)
			}
			fmt.Println(sep)
			return
		}

		fmt.Printf("❌ Error decoding checkpoint: %v\n", err)
		return
	}

	fmt.Println(sep)
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
	fmt.Println(sep)

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
		fmt.Printf("📊 Exported to: %s\n", jsonPath)
	}
}
