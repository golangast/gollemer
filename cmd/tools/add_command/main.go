package main

// add_command registers a new voice command without retraining the GRU.
//
// It works by:
//   1. Recording a few samples of your new command.
//   2. Running each sample through the frozen (already-trained) GRU encoder.
//   3. Averaging the resulting embedding vectors into a single "prototype".
//   4. Appending that prototype to the saved model JSON.
//
// At inference time, voice_capture uses cosine similarity between the live
// audio embedding and all stored prototypes — so new commands "just work".
//
// Usage:
//   go run cmd/tools/add_command/main.go "TURN OFF FAN"
//   go run cmd/tools/add_command/main.go "WHAT IS THE WEATHER"

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
)

const (
	SampleRate  = 16000
	FrameSize   = 400
	NumSamples  = SampleRate * 1 // 1 second window
	NumRecordings = 3
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run cmd/tools/add_command/main.go \"YOUR COMMAND HERE\"")
		fmt.Println("Example: go run cmd/tools/add_command/main.go \"TURN OFF FAN\"")
		return
	}

	// Build intent name from args: "turn off fan" → "TURN_OFF_FAN"
	rawCmd := strings.Join(os.Args[1:], " ")
	intent := strings.ToUpper(strings.ReplaceAll(strings.TrimSpace(rawCmd), " ", "_"))
	fmt.Printf("📝 Registering new command: '%s'\n", intent)
	fmt.Println()

	// Load existing trained model
	ae, te, headW, headB, classNames, prototypes, err := moe.LoadAudioModel("models/audio_gru.json")
	if err != nil {
		fmt.Println("❌ No trained model found. Run 'go run cmd/tools/train_audio/main.go' first.")
		return
	}

	if prototypes == nil {
		prototypes = make(map[string][]float32)
	}

	fmt.Printf("✅ Loaded trained model with %d existing commands.\n", len(prototypes))
	fmt.Printf("   Existing: %v\n\n", classNames)

	os.MkdirAll("dataset/audio", 0755)
	reader := bufio.NewReader(os.Stdin)

	var embeddings [][]float32

	// Record NumRecordings samples and compute their embeddings
	for i := 1; i <= NumRecordings; i++ {
		fmt.Printf("🎙️  Sample %d of %d — say '%s'\n", i, NumRecordings, rawCmd)
		fmt.Println("   Press [ENTER] when ready...")
		reader.ReadBytes('\n')

		filename := fmt.Sprintf("dataset/audio/%s_%d.raw", intent, i)
		fmt.Println("🔴 RECORDING (1.5 seconds)...")

		cmd := exec.Command("ffmpeg",
			"-y",
			"-f", "alsa", "-i", "default",
			"-t", "1.5",
			"-ac", "1", "-ar", "16000",
			"-f", "s16le", filename,
		)
		if err := cmd.Run(); err != nil {
			fmt.Printf("❌ Recording failed: %v\n", err)
			return
		}
		fmt.Printf("✅ Recorded to %s\n\n", filename)

		// Load and process the recording
		emb, err := computeEmbedding(filename, ae, te)
		if err != nil {
			fmt.Printf("❌ Failed to compute embedding: %v\n", err)
			return
		}
		embeddings = append(embeddings, emb)
	}

	// Average all embeddings into a single prototype
	hiddenDim := te.HiddenDim
	prototype := make([]float32, hiddenDim)
	for _, emb := range embeddings {
		for d := 0; d < hiddenDim; d++ {
			prototype[d] += emb[d]
		}
	}
	for d := range prototype {
		prototype[d] /= float32(len(embeddings))
	}

	// Normalize prototype to unit length for stable cosine similarity
	var norm float64
	for _, v := range prototype {
		norm += float64(v) * float64(v)
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for d := range prototype {
			prototype[d] = float32(float64(prototype[d]) / norm)
		}
	}

	// Check if command already exists
	if _, exists := prototypes[intent]; exists {
		fmt.Printf("⚠️  Updating existing prototype for '%s'\n", intent)
	} else {
		// Add to classNames if new
		classNames = append(classNames, intent)
		fmt.Printf("🆕 Added new command '%s' to model!\n", intent)
	}

	prototypes[intent] = prototype

	// Save model back with updated prototypes
	// Reload raw JSON to preserve all weights, only patch Prototypes + ClassNames
	if err := patchModelFile("models/audio_gru.json", classNames, prototypes); err != nil {
		// Fallback: full save
		err2 := moe.SaveAudioModel("models/audio_gru.json", ae, te, headW, headB, classNames, prototypes)
		if err2 != nil {
			fmt.Printf("❌ Failed to save: %v\n", err2)
			return
		}
	}

	fmt.Printf("\n🎉 Done! '%s' is now registered.\n", intent)
	fmt.Println("Run 'go run cmd/tools/voice_capture/main.go' to try it out!")
	fmt.Println()
	fmt.Println("All registered commands:")
	for name := range prototypes {
		fmt.Printf("  - %s\n", name)
	}
}

// computeEmbedding loads a raw audio file, finds the loudest 1-second window,
// and returns the GRU embedding vector for that audio.
func computeEmbedding(filename string, ae *moe.AudioEncoder, te *moe.TemporalEncoder) ([]float32, error) {
	rawBytes, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	samples := make([]float32, len(rawBytes)/2)
	for i := range samples {
		s16 := int16(binary.LittleEndian.Uint16(rawBytes[i*2 : i*2+2]))
		samples[i] = float32(s16) / 32768.0
	}

	// Find loudest 1-second window
	target := NumSamples
	if len(samples) > target {
		bestStart, bestEnergy := 0, float32(0)
		for start := 0; start <= len(samples)-target; start += FrameSize {
			var energy float32
			for j := 0; j < target; j++ {
				energy += samples[start+j] * samples[start+j]
			}
			if energy > bestEnergy {
				bestEnergy = energy
				bestStart = start
			}
		}
		samples = samples[bestStart : bestStart+target]
	} else {
		padded := make([]float32, target)
		copy(padded, samples)
		samples = padded
	}

	// Chunk into frames and run through encoder
	numFrames := len(samples) / FrameSize
	frames := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		frames[i] = samples[i*FrameSize : (i+1)*FrameSize]
	}

	audioTokens := ae.Forward(frames)
	embedding := te.Forward(audioTokens)
	return embedding, nil
}

// patchModelFile reads the existing JSON and only updates ClassNames + Prototypes,
// preserving all trained weight arrays exactly as-is.
func patchModelFile(path string, classNames []string, prototypes map[string][]float32) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	classJSON, err := json.Marshal(classNames)
	if err != nil {
		return err
	}
	raw["ClassNames"] = classJSON

	protoJSON, err := json.Marshal(prototypes)
	if err != nil {
		return err
	}
	raw["Prototypes"] = protoJSON

	out, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0644)
}
