package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
)

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic Audio Generators (16kHz sample rate)
// ─────────────────────────────────────────────────────────────────────────────

const SampleRate = 16000
const DurationSeconds = 1
const NumSamples = SampleRate * DurationSeconds
const FrameSize = 400 // 25ms per frame

// GenerateSineWave generates a pure tone at a specific frequency
func GenerateSineWave(frequency float64) []float32 {
	samples := make([]float32, NumSamples)
	for i := 0; i < NumSamples; i++ {
		t := float64(i) / float64(SampleRate)
		samples[i] = float32(math.Sin(2 * math.Pi * frequency * t))
	}
	return samples
}

// GenerateWhiteNoise generates random noise (silence/background noise simulation)
func GenerateWhiteNoise() []float32 {
	samples := make([]float32, NumSamples)
	for i := 0; i < NumSamples; i++ {
		samples[i] = rand.Float32()*2 - 1 // -1 to 1
	}
	return samples
}

// ChunkAudio breaks a continuous audio buffer into smaller frames
func ChunkAudio(samples []float32, frameSize int) [][]float32 {
	numFrames := len(samples) / frameSize
	frames := make([][]float32, numFrames)
	for i := 0; i < numFrames; i++ {
		frames[i] = samples[i*frameSize : (i+1)*frameSize]
	}
	return frames
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func softmax(logits []float32) []float32 {
	out := make([]float32, len(logits))
	maxV := logits[0]
	for _, v := range logits {
		if v > maxV {
			maxV = v
		}
	}
	var sum float32
	for i, v := range logits {
		out[i] = float32(math.Exp(float64(v - maxV)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

func crossEntropyLoss(probs []float32, targetClass int) (float32, []float32) {
	loss := -float32(math.Log(float64(probs[targetClass]) + 1e-9))
	grad := make([]float32, len(probs))
	copy(grad, probs)
	grad[targetClass] -= 1.0
	return loss, grad
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Training Loop
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	fmt.Println("Initialising Temporal Audio Training…")
	fmt.Println("Classes: 0=LOW_PITCH(440Hz)  1=HIGH_PITCH(880Hz)  2=NOISE")
	fmt.Println()

	// ── Architecture ─────────────────────────────────────────────────────────
	//
	//  [Raw Audio PCM: 16000 samples]
	//       │  Chunking into 400-sample frames (40 frames)
	//  [AudioEncoder: 400 → 16 dim]        → sequence of 40 x 16-dim vectors
	//       │
	//  [TemporalEncoder GRU: 16 → 32 dim]  → single 32-dim context vector
	//       │
	//  [Linear head]                       → 3 class logits
	//
	inputDim := FrameSize
	audioFeatureDim := 64
	gruHiddenDim := 32

	ae := moe.NewAudioEncoder(inputDim, audioFeatureDim)
	te := moe.NewTemporalEncoder(audioFeatureDim, gruHiddenDim)

	// ── Dataset ──────────────────────────────────────────────────────────────
	type sample struct {
		frames      [][]float32
		targetClass int
		label       string
	}

	var dataset []sample
	var classNames []string
	classMap := make(map[string]int)

	// Try loading real voice samples first
	files, err := os.ReadDir("dataset/audio")
	if err == nil && len(files) > 0 {
		fmt.Println("Loading real voice samples from dataset/audio/...")
		for _, f := range files {
			if strings.HasSuffix(f.Name(), ".raw") {
				// Filename format: INTENT_NAME_sample1.raw
				parts := strings.Split(f.Name(), "_")
				if len(parts) < 2 { continue }
				
				// Rejoin intent name if it had underscores (e.g., TURN_ON_LIGHTS_1.raw)
				intentName := strings.Join(parts[:len(parts)-1], "_")
				
				if _, exists := classMap[intentName]; !exists {
					classMap[intentName] = len(classNames)
					classNames = append(classNames, intentName)
				}
				classIdx := classMap[intentName]

				// Load PCM float32 samples (using absolute amplitude envelope for better GRU tracking)
				rawBytes, err := os.ReadFile("dataset/audio/" + f.Name())
				if err == nil {
					samples := make([]float32, len(rawBytes)/2)
					maxAmp := float32(0)
					for i := 0; i < len(samples); i++ {
						sample16 := int16(binary.LittleEndian.Uint16(rawBytes[i*2 : i*2+2]))
						v := float32(sample16) / 32768.0
						samples[i] = v
						if math.Abs(float64(v)) > float64(maxAmp) {
							maxAmp = float32(math.Abs(float64(v)))
						}
					}
					fmt.Printf("Loaded %s (Max Amplitude: %.4f)\n", f.Name(), maxAmp)
					
					// Find the loudest 1.0 second (16000 samples) window
					targetSamples := 16000 
					if len(samples) > targetSamples {
						bestStart := 0
						bestEnergy := float32(0)
						for start := 0; start <= len(samples)-targetSamples; start += 400 {
							var energy float32
							for j := 0; j < targetSamples; j++ {
								energy += samples[start+j] * samples[start+j]
							}
							if energy > bestEnergy {
								bestEnergy = energy
								bestStart = start
							}
						}
						samples = samples[bestStart : bestStart+targetSamples]
					} else {
						padded := make([]float32, targetSamples)
						copy(padded, samples)
						samples = padded
					}
					
					dataset = append(dataset, sample{
						frames:      ChunkAudio(samples, FrameSize),
						targetClass: classIdx,
						label:       intentName,
					})
				}
			}
		}
	}

	// Fallback to synthetic if no real data
	if len(dataset) == 0 {
		fmt.Println("No real audio found. Using synthetic frequency samples.")
		classNames = []string{"LOW_PITCH", "HIGH_PITCH", "NOISE"}
		dataset = []sample{
			{ChunkAudio(GenerateSineWave(440), FrameSize), 0, "LOW_PITCH"},
			{ChunkAudio(GenerateSineWave(880), FrameSize), 1, "HIGH_PITCH"},
			{ChunkAudio(GenerateWhiteNoise(), FrameSize), 2, "NOISE"},
		}
	}

	// Balance dataset by duplicating minority classes
	classCounts := make(map[int]int)
	maxCount := 0
	for _, s := range dataset {
		classCounts[s.targetClass]++
		if classCounts[s.targetClass] > maxCount {
			maxCount = classCounts[s.targetClass]
		}
	}
	
	var balancedDataset []sample
	for classIdx, count := range classCounts {
		// Find all samples of this class
		var classSamples []sample
		for _, s := range dataset {
			if s.targetClass == classIdx {
				classSamples = append(classSamples, s)
			}
		}
		
		// Duplicate until we hit maxCount
		for i := 0; i < maxCount; i++ {
			balancedDataset = append(balancedDataset, classSamples[i%count])
		}
	}
	dataset = balancedDataset

	numClasses := len(classNames)
	fmt.Printf("Dataset prepared: %d balanced samples across %d intents.\n\n", len(dataset), numClasses)

	// Xavier init for classification head
	headW := make([]float32, numClasses*gruHiddenDim)
	headB := make([]float32, numClasses)
	limit := float32(math.Sqrt(1.0 / float64(gruHiddenDim)))
	for i := range headW {
		headW[i] = (rand.Float32()*2.0 - 1.0) * limit
	}

	headGradW := make([]float32, len(headW))
	headGradB := make([]float32, len(headB))

	// ── Training ─────────────────────────────────────────────────────────────
	epochs := 2500
	lr := float32(0.001)

	for epoch := 1; epoch <= epochs; epoch++ {
		totalLoss := float32(0)

		for _, s := range dataset {
			// 1. Audio Projection
			audioTokens := ae.Forward(s.frames)

			// 2. GRU
			motionVec := te.Forward(audioTokens)

			// 3. Linear head
			logits := make([]float32, numClasses)
			for c := 0; c < numClasses; c++ {
				for d := 0; d < gruHiddenDim; d++ {
					logits[c] += headW[c*gruHiddenDim+d] * motionVec[d]
				}
				logits[c] += headB[c]
			}

			// 4. Loss
			probs := softmax(logits)
			loss, dLogits := crossEntropyLoss(probs, s.targetClass)
			totalLoss += loss

			// ── Backward ──
			for i := range headGradW { headGradW[i] = 0 }
			for i := range headGradB { headGradB[i] = 0 }
			
			dMotion := make([]float32, gruHiddenDim)
			for c := 0; c < numClasses; c++ {
				for d := 0; d < gruHiddenDim; d++ {
					headGradW[c*gruHiddenDim+d] = dLogits[c] * motionVec[d]
					dMotion[d] += headW[c*gruHiddenDim+d] * dLogits[c]
				}
				headGradB[c] = dLogits[c]
			}

			// Backprop through GRU
			dAudioTokens := te.Backward(dMotion, lr)

			// Backprop through AudioEncoder
			ae.Backward(s.frames, dAudioTokens, lr)

			// Update head weights
			for i := range headW { headW[i] -= lr * headGradW[i] }
			for i := range headB { headB[i] -= lr * headGradB[i] }
		}

		if epoch%20 == 0 {
			fmt.Printf("Epoch %3d/%d  avg-loss=%.4f\n", epoch, epochs, totalLoss/float32(len(dataset)))
		}
	}

	// ── Evaluation ───────────────────────────────────────────────────────────
	fmt.Println()
	fmt.Println("─── Final Evaluation ───")
	correct := 0
	for _, s := range dataset {
		audioTokens := ae.Forward(s.frames)
		motionVec := te.Forward(audioTokens)
		
		logits := make([]float32, numClasses)
		for c := 0; c < numClasses; c++ {
			for d := 0; d < gruHiddenDim; d++ {
				logits[c] += headW[c*gruHiddenDim+d] * motionVec[d]
			}
			logits[c] += headB[c]
		}
		
		probs := softmax(logits)
		pred := 0
		for c := 1; c < numClasses; c++ {
			if probs[c] > probs[pred] {
				pred = c
			}
		}
		
		status := "✗"
		if pred == s.targetClass {
			correct++
			status = "✓"
		}
		fmt.Printf("  %s  Ground-truth=%-20s  Predicted=%-12s  Confidence=%.1f%%\n",
			status, s.label, classNames[pred], probs[pred]*100)
	}
	fmt.Printf("\nAccuracy: %d/%d (%.0f%%)\n", correct, len(dataset), float64(correct)/float64(len(dataset))*100)

	// ── Compute Prototypes (mean embedding per class) ─────────────────────────
	// This allows zero-shot command addition without retraining.
	prototypes := make(map[string][]float32)
	protoCounts := make(map[string]int)
	for _, s := range dataset {
		audioTokens := ae.Forward(s.frames)
		emb := te.Forward(audioTokens) // 32-dim embedding

		if prototypes[s.label] == nil {
			prototypes[s.label] = make([]float32, gruHiddenDim)
		}
		for d := 0; d < gruHiddenDim; d++ {
			prototypes[s.label][d] += emb[d]
		}
		protoCounts[s.label]++
	}
	// Average to get the mean prototype
	for label, vec := range prototypes {
		count := float32(protoCounts[label])
		for d := range vec {
			vec[d] /= count
		}
	}
	fmt.Printf("✅ Computed %d prototype embeddings.\n", len(prototypes))

	// Save the model
	os.MkdirAll("models", 0755)
	err = moe.SaveAudioModel("models/audio_gru.json", ae, te, headW, headB, classNames, prototypes)
	if err != nil {
		fmt.Printf("Failed to save audio model: %v\n", err)
	} else {
		fmt.Println("✅ Saved trained audio model to models/audio_gru.json")
		fmt.Println()
		fmt.Println("💡 To add a NEW command without retraining, run:")
		fmt.Println("   go run cmd/tools/add_command/main.go \"YOUR NEW COMMAND\"")
	}
}
