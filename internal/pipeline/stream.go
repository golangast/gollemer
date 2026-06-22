package pipeline

import (
	"context"
	"fmt"
	"sync"
)

// Frame represents raw image data from the incoming camera.
type Frame []byte

// AudioChunk represents raw PCM audio data from the microphone.
type AudioChunk []int16

// CentralPipeline holds the buffered channels for all concurrent data streams.
type CentralPipeline struct {
	AudioChan  chan AudioChunk
	VideoChan  chan Frame
	IntentChan chan string
}

// NewPipeline constructs a CentralPipeline with sensibly-sized channel buffers.
func NewPipeline() *CentralPipeline {
	return &CentralPipeline{
		AudioChan:  make(chan AudioChunk, 100), // Buffer up to 100 audio segments
		VideoChan:  make(chan Frame, 5),        // Keep video buffer small to avoid lag
		IntentChan: make(chan string, 10),
	}
}

// StartLoops ignites the non-blocking concurrent workers.
// Each goroutine owns exactly one data stream so a slow camera
// never blocks the microphone listener and vice-versa.
func (p *CentralPipeline) StartLoops(ctx context.Context) {
	// Worker 1: Microphone Stream Ingestion Listener
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case chunk := <-p.AudioChan:
				// Guard: do not process audio while the speaker is active to
				// prevent acoustic feedback into Gollemer's own voice pipeline.
				if State.IsSpeaking {
					continue
				}
				// Forward raw PCM waves directly into Gollemer's TemporalEncoder.
				_ = chunk
			}
		}
	}()

	// Worker 2: Video Stream Processing Layer
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case frame := <-p.VideoChan:
				// Process raw array matrices for motion or visual verification.
				_ = frame
			}
		}
	}()

	// Worker 3: Intent Dispatch
	go func() {
		for {
			select {
			case <-ctx.Done():
				return
			case intent := <-p.IntentChan:
				fmt.Printf("[Pipeline] Dispatched intent: %s\n", intent)
			}
		}
	}()
}

// SystemState is a thread-safe flag set that reflects the physical audio/speaker state.
type SystemState struct {
	mu           sync.Mutex
	IsSpeaking   bool
	IsProcessing bool
}

// State is the singleton used across Gollemer's voice and hardware subsystems.
var State = &SystemState{}

// SetSpeaking toggles the speaker-active flag and prints a status line so
// the microphone worker can suppress its own pipeline while TTS is playing.
func (s *SystemState) SetSpeaking(speaking bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.IsSpeaking = speaking

	if speaking {
		fmt.Println("[SYSTEM STATE] Speaker active. Suppressing microphone input processing to prevent feedback loops.")
	} else {
		fmt.Println("[SYSTEM STATE] Speaker idle. Microphone processing resumed.")
	}
}

// SetProcessing marks whether the NLP model is currently generating a response.
func (s *SystemState) SetProcessing(processing bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.IsProcessing = processing
}
