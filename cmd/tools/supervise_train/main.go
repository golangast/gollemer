// supervise_train: A Knowledge Distillation tool.
//
// Whisper (Teacher) listens to your voice, transcribes the exact words you said,
// and automatically labels the acoustic embedding so Gollemer (Student) can
// learn to recognize the phrase without you having to record it manually as a
// named command.
//
// Usage:
//
//	go run cmd/tools/supervise_train/main.go
package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

const (
	sampleRate  = 16000
	frameSize   = 400
	recordSecs  = "3.5"
	numSamples  = sampleRate * 3 // must match voice.go AudioWindow capacity logic
	modelPath   = "models/audio_gru.json"
	datasetDir  = "dataset/audio"
	whisperBin  = "/tmp/whisper.cpp/build/bin/whisper-cli"
	whisperModel = "/tmp/whisper.cpp/models/ggml-tiny.en.bin"
)

// computeEmbedding reads a raw s16le file and returns the GRU motion vector.
func computeEmbedding(rawFile string, ae *moe.AudioEncoder, te *moe.TemporalEncoder) ([]float32, error) {
	rawBytes, err := os.ReadFile(rawFile)
	if err != nil {
		return nil, fmt.Errorf("read raw: %w", err)
	}

	samples := make([]float32, len(rawBytes)/2)
	for i := range samples {
		s16 := int16(rawBytes[i*2]) | int16(rawBytes[i*2+1])<<8
		samples[i] = float32(s16) / 32768.0
	}

	// Trim or pad to numSamples, choosing the highest-energy window
	if len(samples) > numSamples {
		bestStart, bestEnergy := 0, float32(0)
		for start := 0; start <= len(samples)-numSamples; start += frameSize {
			var energy float32
			for _, s := range samples[start : start+numSamples] {
				energy += s * s
			}
			if energy > bestEnergy {
				bestEnergy = energy
				bestStart = start
			}
		}
		samples = samples[bestStart : bestStart+numSamples]
	} else if len(samples) < numSamples {
		padded := make([]float32, numSamples)
		copy(padded, samples)
		samples = padded
	}

	numFrames := len(samples) / frameSize
	frames := make([][]float32, numFrames)
	for i := range frames {
		frames[i] = samples[i*frameSize : (i+1)*frameSize]
	}

	audioTokens := ae.Forward(frames)
	motionVec := te.Forward(audioTokens)
	return motionVec, nil
}

// cleanTranscript turns Whisper output like " Who are you?" into "WHO_ARE_YOU"
func cleanTranscript(text string) string {
	text = strings.ToUpper(strings.TrimSpace(text))
	for _, ch := range []string{".", ",", "?", "!", "'", "\"", "(", ")", "[", "]", "-"} {
		text = strings.ReplaceAll(text, ch, "")
	}
	// Collapse multiple spaces then replace with underscore
	fields := strings.Fields(text)
	return strings.Join(fields, "_")
}

// cosineSim computes cosine similarity between two vectors.
func cosineSim(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}

func main() {
	fmt.Println("🎓 Gollemer Whisper Teacher — Knowledge Distillation Mode")
	fmt.Println("==========================================================")
	fmt.Println("Speak naturally. Whisper transcribes your voice, then Gollemer")
	fmt.Println("learns the acoustic pattern for that exact phrase automatically.")
	fmt.Println()

	// Sanity-check whisper
	if _, err := os.Stat(whisperBin); os.IsNotExist(err) {
		fmt.Println("❌ Whisper binary not found at", whisperBin)
		fmt.Println("   Ensure /tmp/whisper.cpp has been built first.")
		os.Exit(1)
	}
	if _, err := os.Stat(whisperModel); os.IsNotExist(err) {
		fmt.Println("❌ Whisper model not found at", whisperModel)
		os.Exit(1)
	}

	// Load Gollemer audio model
	ae, te, headW, headB, classNames, prototypes, err := moe.LoadAudioModel(modelPath)
	if err != nil {
		fmt.Printf("❌ Failed to load %s: %v\n", modelPath, err)
		os.Exit(1)
	}
	if prototypes == nil {
		prototypes = make(map[string][]float32)
	}
	fmt.Printf("✅ Loaded audio model (%d commands)\n\n", len(classNames))

	os.MkdirAll(datasetDir, 0755)
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Println("─────────────────────────────────────────────")
		fmt.Print("Press [ENTER] to record (or type 'quit' to exit): ")
		line, _ := reader.ReadString('\n')
		if strings.TrimSpace(strings.ToLower(line)) == "quit" {
			fmt.Println("👋 Exiting teacher.")
			break
		}

		wavFile := datasetDir + "/teacher_temp.wav"
		rawFile := datasetDir + "/teacher_temp.raw"

		fmt.Printf("🔴 Recording %.1ss — speak now!\n", recordSecs)
		recCmd := exec.Command("ffmpeg",
			"-y", "-f", "alsa", "-i", "default",
			"-t", recordSecs,
			"-ac", "1", "-ar", "16000",
			wavFile,
		)
		recCmd.Stderr = nil
		if err := recCmd.Run(); err != nil {
			fmt.Printf("❌ ffmpeg error: %v\n", err)
			continue
		}

		fmt.Println("🔬 Transcribing with Whisper...")
		// CRITICAL: Route stderr to /dev/null so whisper's debug/log lines
		// ("read_audio_data:", "whisper_", etc.) don't pollute the transcript.
		// Only stdout contains the actual transcribed text.
		devNull, _ := os.Open(os.DevNull)
		defer devNull.Close()
		wCmd := exec.Command(whisperBin, "-m", whisperModel, "-f", wavFile, "-nt")
		wCmd.Stderr = devNull // discard all debug/log output
		stdout, err := wCmd.Output()
		if err != nil {
			fmt.Printf("⚠️  Whisper returned error (may still have transcript): %v\n", err)
		}

		// Stdout is now ONLY the transcript lines — no debug noise
		var transcriptParts []string
		for _, raw := range strings.Split(string(stdout), "\n") {
			trimmed := strings.TrimSpace(raw)
			if trimmed == "" {
				continue
			}
			transcriptParts = append(transcriptParts, trimmed)
		}
		rawTranscript := strings.Join(transcriptParts, " ")
		intent := cleanTranscript(rawTranscript)

		if intent == "" {
			fmt.Println("⚠️  Whisper heard nothing useful. Try again, speaking louder.")
			continue
		}

		fmt.Printf("🧠 Whisper heard: \"%s\"\n", rawTranscript)
		fmt.Printf("🏷️  Intent label:  %s\n", intent)

		// Convert wav → raw s16le for embedding computation
		convCmd := exec.Command("ffmpeg",
			"-y", "-i", wavFile,
			"-f", "s16le", "-ac", "1", "-ar", "16000",
			rawFile,
		)
		convCmd.Stderr = nil
		convCmd.Run()

		// Compute the GRU embedding for this audio
		emb, err := computeEmbedding(rawFile, ae, te)
		if err != nil {
			fmt.Printf("❌ Embedding error: %v\n", err)
			continue
		}

		// If we already have a prototype, show similarity before update
		if existing, ok := prototypes[intent]; ok {
			sim := cosineSim(existing, emb)
			fmt.Printf("📐 Similarity to existing prototype: %.1f%%\n", sim*100)
		}

		// Update prototype (running average if already exists)
		if existing, ok := prototypes[intent]; ok {
			averaged := make([]float32, len(emb))
			for i := range emb {
				averaged[i] = (existing[i] + emb[i]) * 0.5
			}
			prototypes[intent] = averaged
		} else {
			prototypes[intent] = emb
			// Add to classNames if new
			classNames = append(classNames, intent)
			fmt.Printf("🆕 New command added: %s\n", intent)
		}

		// Save updated model
		if err := moe.SaveAudioModel(modelPath, ae, te, headW, headB, classNames, prototypes); err != nil {
			fmt.Printf("❌ Save failed: %v\n", err)
			continue
		}

		// Persist the raw file for future retraining
		finalRaw := fmt.Sprintf("%s/%s_%d.raw", datasetDir, intent, time.Now().Unix())
		os.Rename(rawFile, finalRaw)

		fmt.Printf("✅ Gollemer now knows: \"%s\" → %s\n", rawTranscript, intent)
		fmt.Printf("   (Saved to %s)\n", finalRaw)
	}

	fmt.Printf("\n📊 Final command vocabulary (%d commands):\n", len(classNames))
	for i, name := range classNames {
		fmt.Printf("   %2d. %s\n", i+1, name)
	}
}
