package llm

import (
	"context"
	"encoding/binary"
	"log"
	"math"
	"os/exec"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

const (
	SampleRate      = 16000
	Channels        = 1
	BytesPerSample  = 2
	FrameSize       = 400 // 25ms of audio at 16kHz
	EnergyThreshold = 0.01 // Tuned for normalized float32 (-1.0 to 1.0)
)

type AudioBuffer struct {
	mu      sync.Mutex
	samples []float32
}

func StartAudioCapture(ctx context.Context, ab *AudioBuffer) error {
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

				samples := make([]float32, n/BytesPerSample)
				for i := 0; i < len(samples); i++ {
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
	Prototypes map[string][]float32
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
	audioTokens := aw.audioEnc.Forward(seq)
	motionVec := aw.temporal.Forward(audioTokens)

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
		conf := (bestSim + 1.0) / 2.0
		return bestLabel, conf
	}

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

	log.Println("⚠️  No trained audio model found. Voice commands disabled.")
	return nil, nil, nil, nil, nil, nil
}

func StartVoiceListener(ctx context.Context, inputChan chan<- string) {
	ae, te, headW, headB, classNames, prototypes := loadTrainedHead()
	if ae == nil {
		return
	}

	ab := &AudioBuffer{}
	if err := StartAudioCapture(ctx, ab); err != nil {
		log.Printf("⚠️  Could not start ffmpeg audio capture: %v", err)
		return
	}

	aw := NewAudioWindow(40, ae, te, headW, headB, classNames, prototypes)
	ticker := time.NewTicker(25 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			ab.mu.Lock()
			if len(ab.samples) >= FrameSize {
				frame := ab.samples[:FrameSize]
				ab.samples = ab.samples[FrameSize:]
				ab.mu.Unlock()

				rms := getRMS(frame)
				aw.Push(frame)

				if aw.Ready() && rms > EnergyThreshold {
					intent, conf := aw.Classify()
					
					if conf > 0.80 && intent != "SILENCE" && intent != "NOISE" && intent != "UNKNOWN_SPEECH" {
						log.Printf("\n🤖 [Voice] Heard: %s (Confidence: %.2f)", intent, conf)
						inputChan <- intent
						
						// Debounce
						aw.mu.Lock()
						aw.filled = 0
						aw.mu.Unlock()
					}
				}
			} else {
				ab.mu.Unlock()
			}
		}
	}
}
