package llm

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
)

// isSpeaking is set to 1 while the LLM is generating/printing a response.
// The voice listener checks this flag and mutes itself to prevent self-hearing.
var isSpeaking int32

// MuteVoice silences the microphone listener while the LLM is speaking.
func MuteVoice() { atomic.StoreInt32(&isSpeaking, 1) }

// UnmuteVoice re-enables the microphone listener after the LLM finishes speaking.
func UnmuteVoice() { atomic.StoreInt32(&isSpeaking, 0) }

const (
	SampleRate      = 16000
	Channels        = 1
	BytesPerSample  = 2
	FrameSize       = 400  // 25ms of audio at 16kHz
	EnergyThreshold = 0.15 // Increased to 0.15 to ignore loud room fans and hum
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

// whisperAutoTeach snapshots the current audio window frames, runs Whisper STT on them,
// and automatically updates the in-memory prototype map with the new label.
// This is called in a goroutine when the acoustic model is uncertain (conf 0.75–0.98).
func whisperAutoTeach(frames [][]float32, aw *AudioWindow, ae *moe.AudioEncoder, te *moe.TemporalEncoder, headW, headB []float32) {
	const (
		whisperBin   = "/tmp/whisper.cpp/build/bin/whisper-cli"
		whisperModel = "/tmp/whisper.cpp/models/ggml-tiny.en.bin"
		rawTmp       = "dataset/audio/live_teach_temp.raw"
		wavTmp       = "dataset/audio/live_teach_temp.wav"
	)

	if _, err := os.Stat(whisperBin); os.IsNotExist(err) {
		return // whisper not available, skip silently
	}

	os.MkdirAll("dataset/audio", 0755)

	// Write the current window frames to a raw s16le file
	f, err := os.Create(rawTmp)
	if err != nil {
		return
	}
	for _, frame := range frames {
		for _, s := range frame {
			sample16 := int16(s * 32768.0)
			_ = binary.Write(f, binary.LittleEndian, sample16)
		}
	}
	f.Close()

	// Convert raw → wav for Whisper
	exec.Command("ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
		"-i", rawTmp, wavTmp).Run()

	// Run Whisper, discard its stderr debug logs
	devNull, _ := os.Open(os.DevNull)
	defer devNull.Close()
	wCmd := exec.Command(whisperBin, "-m", whisperModel, "-f", wavTmp, "-nt")
	wCmd.Stderr = devNull
	out, _ := wCmd.Output()

	// Parse transcript from stdout only
	var parts []string
	for _, line := range strings.Split(string(out), "\n") {
		t := strings.TrimSpace(line)
		if t != "" {
			parts = append(parts, t)
		}
	}
	rawTranscript := strings.Join(parts, " ")

	// Clean to intent label: uppercase, strip punctuation, spaces → underscores
	label := strings.ToUpper(strings.TrimSpace(rawTranscript))
	for _, ch := range []string{".", ",", "?", "!", "'", "\"", "(", ")", "[", "]", "-"} {
		label = strings.ReplaceAll(label, ch, "")
	}
	label = strings.Join(strings.Fields(label), "_")

	if label == "" || len(label) > 60 {
		log.Printf("🎤 [AutoTeach] Whisper transcript empty or too long, skipping.")
		return
	}

	log.Printf("🎤 [AutoTeach] Whisper heard: \"%s\" → teaching label: %s", rawTranscript, label)

	// Compute embedding for this audio
	audioTokens := ae.Forward(frames)
	newEmb := te.Forward(audioTokens)

	// Average with existing prototype or insert new one
	aw.mu.Lock()
	if existing, ok := aw.Prototypes[label]; ok {
		avg := make([]float32, len(newEmb))
		for i := range newEmb {
			avg[i] = (existing[i] + newEmb[i]) * 0.5
		}
		aw.Prototypes[label] = avg
		log.Printf("📐 [AutoTeach] Updated existing prototype for %s", label)
	} else {
		aw.Prototypes[label] = newEmb
		aw.ClassNames = append(aw.ClassNames, label)
		log.Printf("🆕 [AutoTeach] New command added: %s", label)
	}
	currentNames := aw.ClassNames
	currentProtos := aw.Prototypes
	aw.mu.Unlock()

	// Persist updated model to disk
	if err := moe.SaveAudioModel("models/audio_gru.json", ae, te, headW, headB, currentNames, currentProtos); err != nil {
		log.Printf("⚠️  [AutoTeach] Failed to save model: %v", err)
	} else {
		finalRaw := fmt.Sprintf("dataset/audio/%s_live_%d.raw", label, time.Now().Unix())
		os.Rename(rawTmp, finalRaw)
		log.Printf("✅ [AutoTeach] Model saved. %s prototype persisted.", label)
	}
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

	aw := NewAudioWindow(120, ae, te, headW, headB, classNames, prototypes)
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

				if aw.Ready() && rms > EnergyThreshold && atomic.LoadInt32(&isSpeaking) == 0 {
					intent, conf := aw.Classify()

					if conf > 0.98 && intent != "SILENCE" && intent != "NOISE" && intent != "UNKNOWN_SPEECH" && intent != "BLANK_AUDIO" {
						// High confidence — fire the command
						log.Printf("\n🤖 [Voice] Heard: %s (Confidence: %.2f)", intent, conf)
						inputChan <- intent

						// Hard cooldown: flush window + buffer, sleep 4s
						aw.mu.Lock()
						aw.filled = 0
						aw.mu.Unlock()
						ab.mu.Lock()
						ab.samples = ab.samples[:0]
						ab.mu.Unlock()
						time.Sleep(4 * time.Second)

					} else if conf > 0.93 {
						// Medium-high confidence — heard real speech but uncertain.
						// Run Whisper in the background to auto-teach the new phrase.
						log.Printf("\n🤔 [Voice] Uncertain (best: %s, conf: %.2f) — auto-teaching via Whisper...", intent, conf)
						snapshot := aw.GetOrderedSequence()
						go whisperAutoTeach(snapshot, aw, ae, te, headW, headB)

						// Flush and pause to avoid re-triggering on the same audio
						aw.mu.Lock()
						aw.filled = 0
						aw.mu.Unlock()
						ab.mu.Lock()
						ab.samples = ab.samples[:0]
						ab.mu.Unlock()
						time.Sleep(4 * time.Second)
					}
				}
			} else {
				ab.mu.Unlock()
			}
		}
	}
}
