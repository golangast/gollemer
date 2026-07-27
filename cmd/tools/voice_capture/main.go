package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

// ─────────────────────────────────────────────────────────────────────────────
// Pure-Go Audio Capture Pipeline (No CGO)
// ─────────────────────────────────────────────────────────────────────────────

const (
	SampleRate      = 16000
	Channels        = 1
	BytesPerSample  = 2
	FrameSize       = 400  // 25ms of audio at 16kHz
	EnergyThreshold = 0.01 // Tuned for normalized float32 (-1.0 to 1.0)
)

type AudioBuffer struct {
	mu      sync.Mutex
	samples []float32
}

// StartAudioCapture uses ffmpeg to capture audio from the default ALSA device
// and streams it as raw 16-bit PCM to stdout, reading it into a shared buffer.
func StartAudioCapture(ctx context.Context, ab *AudioBuffer) error {
	// Use ffmpeg to capture from ALSA (default mic) and output raw s16le PCM
	cmd := exec.CommandContext(ctx, "ffmpeg",
		"-f", "alsa", "-i", "default",
		"-ac", "1", "-ar", "16000",
		"-f", "s16le", "-",
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return err
	}

	go func() {
		// Read raw 16-bit PCM (2 bytes per sample)
		buf := make([]byte, FrameSize*BytesPerSample)
		for {
			select {
			case <-ctx.Done():
				return
			default:
				n, err := stdout.Read(buf)
				if err != nil || n == 0 {
					continue
				}

				// Convert bytes to float32 samples (amplitude envelope)
				samples := make([]float32, n/BytesPerSample)
				for i := 0; i < len(samples); i++ {
					// little-endian
					sample16 := int16(binary.LittleEndian.Uint16(buf[i*2 : i*2+2]))
					samples[i] = float32(sample16) / 32768.0
				}

				ab.mu.Lock()
				ab.samples = append(ab.samples, samples...)
				ab.mu.Unlock()
			}
		}
	}()

	return nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio Window (Ring Buffer for GRU)
// ─────────────────────────────────────────────────────────────────────────────

type AudioWindow struct {
	mu         sync.RWMutex
	Capacity   int
	Frames     [][]float32
	Head       int
	filled     int
	audioEnc   *moe.AudioEncoder
	temporal   *moe.TemporalEncoder
	headW      []float32
	headB      []float32
	ClassNames []string
	Prototypes map[string][]float32 // class name → mean embedding (for cosine similarity)
}

func NewAudioWindow(capacity int, ae *moe.AudioEncoder, te *moe.TemporalEncoder, headW, headB []float32, classNames []string, prototypes map[string][]float32) *AudioWindow {
	return &AudioWindow{
		Capacity:   capacity,
		Frames:     make([][]float32, capacity),
		audioEnc:   ae,
		temporal:   te,
		headW:      headW,
		headB:      headB,
		ClassNames: classNames,
		Prototypes: prototypes,
	}
}

func (aw *AudioWindow) Push(frame []float32) {
	aw.mu.Lock()
	defer aw.mu.Unlock()
	aw.Frames[aw.Head] = frame
	aw.Head = (aw.Head + 1) % aw.Capacity
	if aw.filled < aw.Capacity {
		aw.filled++
	}
}

func (aw *AudioWindow) Ready() bool {
	aw.mu.RLock()
	defer aw.mu.RUnlock()
	return aw.filled >= aw.Capacity
}

func (aw *AudioWindow) GetOrderedSequence() [][]float32 {
	aw.mu.RLock()
	defer aw.mu.RUnlock()
	seq := make([][]float32, aw.Capacity)
	for i := 0; i < aw.Capacity; i++ {
		idx := (aw.Head + i) % aw.Capacity
		seq[i] = aw.Frames[idx]
	}
	return seq
}

func cosineSim(a, b []float32) float32 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(normA) * math.Sqrt(normB)))
}

func (aw *AudioWindow) Classify() (string, float32) {
	seq := aw.GetOrderedSequence()

	// 1. AudioEncoder → 2. GRU → single embedding vector
	audioTokens := aw.audioEnc.Forward(seq)
	motionVec := aw.temporal.Forward(audioTokens)

	// ── Prototype cosine similarity (preferred — supports zero-shot commands) ──
	if len(aw.Prototypes) > 0 {
		bestLabel := ""
		bestSim := float32(-2.0)
		for label, proto := range aw.Prototypes {
			sim := cosineSim(motionVec, proto)
			if sim > bestSim {
				bestSim = sim
				bestLabel = label
			}
		}
		// Convert cosine similarity (-1..1) to a 0..1 confidence score
		conf := (bestSim + 1.0) / 2.0
		return bestLabel, conf
	}

	// ── Fallback: linear head + softmax (original approach) ──
	numClasses := len(aw.ClassNames)
	hiddenDim := aw.temporal.HiddenDim
	logits := make([]float32, numClasses)
	for c := 0; c < numClasses; c++ {
		for d := 0; d < hiddenDim; d++ {
			logits[c] += aw.headW[c*hiddenDim+d] * motionVec[d]
		}
		logits[c] += aw.headB[c]
	}

	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}
	sum := float32(0)
	for i, v := range logits {
		logits[i] = float32(math.Exp(float64(v - maxVal)))
		sum += logits[i]
	}
	for c := 0; c < numClasses; c++ {
		logits[c] /= sum
	}

	best := 0
	for c := 1; c < numClasses; c++ {
		if logits[c] > logits[best] {
			best = c
		}
	}
	return aw.ClassNames[best], logits[best]
}

// getRMS calculates Root Mean Square energy to detect Voice Activity
func getRMS(samples []float32) float32 {
	var sum float32
	for _, s := range samples {
		sum += s * s
	}
	return float32(math.Sqrt(float64(sum / float32(len(samples)))))
}

func loadTrainedHead() (*moe.AudioEncoder, *moe.TemporalEncoder, []float32, []float32, []string, map[string][]float32) {
	ae, te, headW, headB, classNames, prototypes, err := moe.LoadAudioModel("models/audio_gru.json")
	if err == nil {
		log.Printf("✅ Loaded trained audio model (%d commands, %d prototypes)", len(classNames), len(prototypes))
		return ae, te, headW, headB, classNames, prototypes
	}

	log.Println("⚠️  No trained audio model found. Run 'go run cmd/tools/train_audio/main.go' first.")
	log.Println("Using dummy untrained weights (will output random predictions).")

	classNames = []string{"SILENCE", "WAKE_WORD_DETECTED", "UNKNOWN_SPEECH"}
	numClasses := len(classNames)
	hiddenDim := 32

	ae = moe.NewAudioEncoder(FrameSize, 64)
	te = moe.NewTemporalEncoder(64, hiddenDim)

	headW = make([]float32, numClasses*hiddenDim)
	headB = make([]float32, numClasses)
	limit := float32(math.Sqrt(1.0 / float64(hiddenDim)))
	for i := range headW {
		headW[i] = (float32(i%7) - 3.0) * limit * 0.1
	}

	return ae, te, headW, headB, classNames, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Loop
// ─────────────────────────────────────────────────────────────────────────────

func main() {
	log.Println("Gollemer Voice Capture — Pure Go Implementation (No CGO)")
	log.Println("Using AudioEncoder + TemporalEncoder GRU for intent processing.")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigs
		fmt.Println("\n[Voice] Shutting down...")
		cancel()
	}()

	ab := &AudioBuffer{}

	err := StartAudioCapture(ctx, ab)
	if err != nil {
		log.Printf("⚠️  Could not start ffmpeg audio capture: %v", err)
		log.Println("Falling back to simulated audio stream.")

		// Simulate audio capture with white noise
		go func() {
			ticker := time.NewTicker(25 * time.Millisecond)
			for {
				select {
				case <-ctx.Done():
					return
				case <-ticker.C:
					// Create 400 frames of silence
					ab.mu.Lock()
					ab.samples = append(ab.samples, make([]float32, 400)...)
					ab.mu.Unlock()
				}
			}
		}()
	} else {
		log.Println("✅ Ffmpeg audio capture started successfully.")
	}

	// 40 frames * 25ms = 1 second rolling window
	ae, te, headW, headB, classNames, prototypes := loadTrainedHead()
	aw := NewAudioWindow(40, ae, te, headW, headB, classNames, prototypes)

	fmt.Println("\n🎧 Listening...")

	ticker := time.NewTicker(25 * time.Millisecond) // process exactly 1 frame per tick
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ab.mu.Lock()
			if len(ab.samples) >= FrameSize {
				// Grab one frame
				frame := ab.samples[:FrameSize]
				// Shift buffer
				ab.samples = ab.samples[FrameSize:]
				ab.mu.Unlock()

				rms := getRMS(frame)
				aw.Push(frame)

				if aw.Ready() && rms > EnergyThreshold {
					// Voice detected! Classify the 1-second window
					intent, conf := aw.Classify()

					// Avoid spamming the log with silence/noise or low confidence guesses
					if conf > 0.80 && intent != "SILENCE" && intent != "NOISE" && intent != "UNKNOWN_SPEECH" {
						fmt.Printf("\n🤖 Audio GRU: %s (Confidence: %.2f, RMS: %.3f)\n", intent, conf, rms)
						// In a full implementation, you would trigger the MoE intent handler here:
						// processWithGollemer(intent)

						// Debounce: clear the buffer to avoid multi-triggering on the same sound
						aw.mu.Lock()
						aw.filled = 0
						aw.mu.Unlock()
					} else if conf <= 0.80 && intent != "SILENCE" && intent != "NOISE" {
						// Print low confidence matches for debugging
						fmt.Printf("   [Low Confidence Guess: %s (%.0f%%)]\n", intent, conf*100)
					}
				}
			} else {
				ab.mu.Unlock()
			}
		}
	}
}
