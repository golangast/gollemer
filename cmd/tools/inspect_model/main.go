package main

import (
	"bufio"
	"compress/gzip"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nn"
)

// InspectReport is the JSON-exportable summary of a loaded model.
type InspectReport struct {
	Type            string             `json:"type"`
	StepCount       int                `json:"step_count"`
	TrainingPhase   int                `json:"training_phase"`
	Version         string             `json:"version,omitempty"`
	Commitment      float32            `json:"commitment,omitempty"`
	TokensProcessed int64              `json:"tokens_processed,omitempty"`
	TotalDuration   string             `json:"total_duration,omitempty"`
	LastProfile     nn.TrainingProfile `json:"last_profile,omitempty"`
	VocabSize       int                `json:"vocab_size"`
	EmbeddingDim    int                `json:"embedding_dim"`
	Layers          []LayerReport      `json:"layers"`
}

type LayerReport struct {
	Name              string         `json:"name"`
	NumExperts        int            `json:"num_experts"`
	K                 int            `json:"k"`
	RouterWeightMag   float64        `json:"router_weight_magnitude"`
	RouterTemperature float32        `json:"router_temperature"`
	Experts           []ExpertReport `json:"experts"`
}

type ExpertReport struct {
	ID           int    `json:"id"`
	Frozen       bool   `json:"frozen"`
	StepStagnant int    `json:"step_stagnant_counter"`
	Status       string `json:"status"`
}

func main() {
	exportFlag := flag.Bool("export", false, "Export metadata to JSON")
	verboseFlag := flag.Bool("v", false, "Verbose: show per-expert details for all layers")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Println("Usage: go run main.go [--export] [-v] <path_to_gob>")
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

	var ckpt *moe.Checkpoint
	var model *moe.IntentMoE
	var isCheckpoint bool

	// ── 1. gzip Checkpoint wrapper ─────────────────────────────────────────
	{
		_, _ = file.Seek(0, io.SeekStart)
		if gz, gzErr := gzip.NewReader(file); gzErr == nil {
			var dc moe.Checkpoint
			if decErr := gob.NewDecoder(gz).Decode(&dc); decErr == nil && dc.Model != nil {
				ckpt, model, isCheckpoint = &dc, dc.Model, true
			}
			gz.Close()
		}
	}

	// ── 2. gzip raw IntentMoE ───────────────────────────────────────────────
	if model == nil {
		_, _ = file.Seek(0, io.SeekStart)
		if gz, gzErr := gzip.NewReader(file); gzErr == nil {
			var dm moe.IntentMoE
			if decErr := gob.NewDecoder(gz).Decode(&dm); decErr == nil {
				model = &dm
			}
			gz.Close()
		}
	}

	// ── 3. raw gob Checkpoint ───────────────────────────────────────────────
	if model == nil {
		_, _ = file.Seek(0, io.SeekStart)
		var dc moe.Checkpoint
		if decErr := gob.NewDecoder(bufio.NewReader(file)).Decode(&dc); decErr == nil && dc.Model != nil {
			ckpt, model, isCheckpoint = &dc, dc.Model, true
		}
	}

	// ── 4. raw gob IntentMoE (legacy) ─────────────────────────────────────
	if model == nil {
		_, _ = file.Seek(0, io.SeekStart)
		var dm moe.IntentMoE
		if decErr := gob.NewDecoder(bufio.NewReader(file)).Decode(&dm); decErr == nil {
			model = &dm
		}
	}

	if model == nil {
		fmt.Println("❌ Failed to decode in all formats (gzip-checkpoint, gzip-model, raw-checkpoint, raw-gob)")
		return
	}

	model.RepairArchitecture()

	// ── Collect MoE layers ─────────────────────────────────────────────────
	layers := model.Encoder.GetMoELayers()
	if model.Decoder != nil && model.Decoder.OutputMoE != nil {
		layers = append(layers, model.Decoder.OutputMoE)
	}

	// ── Print header ────────────────────────────────────────────────────────
	sep := strings.Repeat("─", 52)
	fmt.Println(sep)
	fmt.Printf("📂 File:           %s\n", path)
	fmt.Printf("📦 Size:           %.1f MB\n", float64(fi.Size())/1_000_000)

	if isCheckpoint && ckpt != nil {
		fmt.Printf("🏷️  Format:         Gzipped Checkpoint\n")
		fmt.Printf("🆔 Version:        %s\n", ckpt.Version)
		fmt.Printf("🔢 Step Count:     %d\n", ckpt.StepCount)
		fmt.Printf("🧠 Commitment:     %.2f%%\n", ckpt.Commitment)
		fmt.Printf("⌛ Duration:       %v\n", ckpt.TotalDuration)
		if ckpt.TotalDuration.Seconds() > 0 {
			fmt.Printf("⚡ Throughput:     %.1f tok/s\n",
				float64(ckpt.TokensProcessed)/ckpt.TotalDuration.Seconds())
		}
		fmt.Printf("🛠️  Profile:        %s  LR=%.2e  λ=%.2e\n",
			ckpt.LastProfile.Name, ckpt.LastProfile.LR, ckpt.LastProfile.Lambda)
	} else {
		fmt.Printf("🏷️  Format:         Direct IntentMoE (gzip)\n")
		fmt.Printf("🔢 Step Count:     %d\n", model.StepCount)
		fmt.Printf("📈 Phase:          %d\n", model.TrainingPhase)
	}

	// ── Model summary ────────────────────────────────────────────────────────
	fmt.Println(sep)
	fmt.Printf("📖 Vocab Size:     %d\n", model.SentenceVocabSize)
	fmt.Printf("📐 Embedding Dim:  %d\n", model.EmbeddingDim)

	decoderDim := 0
	if model.Decoder != nil && model.Decoder.LSTM != nil {
		decoderDim = model.Decoder.LSTM.HiddenSize
		fmt.Printf("💡 Decoder Dim:    %d\n", decoderDim)
	}
	fmt.Printf("🗂️  MoE Layers:     %d\n", len(layers))

	// ── Per-layer expert table ───────────────────────────────────────────────
	report := InspectReport{
		Type:          "model",
		StepCount:     model.StepCount,
		TrainingPhase: model.TrainingPhase,
		VocabSize:     model.SentenceVocabSize,
		EmbeddingDim:  model.EmbeddingDim,
	}
	if isCheckpoint && ckpt != nil {
		report.Type = "checkpoint"
		report.Version = ckpt.Version
		report.Commitment = ckpt.Commitment
		report.TokensProcessed = ckpt.TokensProcessed
		report.TotalDuration = ckpt.TotalDuration.String()
		report.LastProfile = ckpt.LastProfile
	}

	for li, layer := range layers {
		layerName := fmt.Sprintf("Encoder Layer %d", li)
		if li == len(layers)-1 && model.Decoder != nil && model.Decoder.OutputMoE == layer {
			layerName = "Decoder Output MoE"
		}

		// Router weight magnitude (L1)
		routerMag := 0.0
		if layer.GatingNetwork != nil && layer.GatingNetwork.Linear != nil &&
			layer.GatingNetwork.Linear.Weights != nil {
			for _, v := range layer.GatingNetwork.Linear.Weights.Data {
				routerMag += math.Abs(float64(v))
			}
		}
		routerHealth := "✅ OK"
		if routerMag == 0 {
			routerHealth = "🚨 ZERO — all tokens will pin to E0"
		} else if routerMag < 0.1 {
			routerHealth = "⚠️  weak"
		}

		frozen, active := 0, 0
		for _, f := range layer.ExpertFrozen {
			if f {
				frozen++
			} else {
				active++
			}
		}

		fmt.Println(sep)
		fmt.Printf("🔧 %s\n", layerName)
		fmt.Printf("   Experts: %d total  |  Active: %d  |  Frozen: %d\n",
			layer.NumExperts, active, frozen)
		fmt.Printf("   K (top-k): %d  |  Temperature: %.2f  |  LB-weight: %.3f\n",
			layer.K, layer.RouterTemperature, layer.LoadBalancingWeight)
		fmt.Printf("   Router mag: %.4f  %s\n", routerMag, routerHealth)

		// Expert table
		if *verboseFlag || len(layers) == 1 {
			fmt.Printf("   %-4s  %-6s  %-10s  %s\n", "ID", "Frozen", "Stagnant", "Status")
			fmt.Printf("   %s\n", strings.Repeat("-", 35))
		}

		lr := LayerReport{
			Name:              layerName,
			NumExperts:        layer.NumExperts,
			K:                 layer.K,
			RouterWeightMag:   routerMag,
			RouterTemperature: layer.RouterTemperature,
		}

		for ei := range layer.NumExperts {
			frozen := ei < len(layer.ExpertFrozen) && layer.ExpertFrozen[ei]
			stagnant := 0
			if ei < len(layer.StepStagnationCounters) {
				stagnant = layer.StepStagnationCounters[ei]
			}

			status := "active"
			icon := "  "
			if frozen {
				status = "frozen"
				icon = "❄️ "
			} else if stagnant > 1000 {
				status = "stagnant"
				icon = "⚠️ "
			}

			if *verboseFlag || len(layers) == 1 {
				frozenStr := "no"
				if frozen {
					frozenStr = "YES"
				}
				fmt.Printf("   %s%-3d  %-6s  %-10d  %s\n", icon, ei, frozenStr, stagnant, status)
			}

			lr.Experts = append(lr.Experts, ExpertReport{
				ID:           ei,
				Frozen:       frozen,
				StepStagnant: stagnant,
				Status:       status,
			})
		}
		report.Layers = append(report.Layers, lr)
	}

	fmt.Println(sep)

	if !*verboseFlag && len(layers) > 1 {
		fmt.Println("   (use -v for per-expert detail on all layers)")
	}

	// ── JSON export ─────────────────────────────────────────────────────────
	if *exportFlag {
		jsonPath := path + ".inspect.json"
		jsonData, _ := json.MarshalIndent(report, "", "  ")
		if err := os.WriteFile(jsonPath, jsonData, 0644); err != nil {
			fmt.Printf("❌ Error saving JSON: %v\n", err)
		} else {
			fmt.Printf("📊 Exported to: %s\n", jsonPath)
		}
	}
}
