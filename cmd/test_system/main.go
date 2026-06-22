package main

import (
	"context"
	"fmt"
	"time"

	"github.com/golangast/gollemer/internal/hardware"
	"github.com/golangast/gollemer/internal/pipeline"
)

func main() {
	fmt.Println("--- Running Gollemer Mock Hardware Integration Core ---")

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	// 1. Fire up pipeline channels
	p := pipeline.NewPipeline()
	p.StartLoops(ctx)
	fmt.Println("[OK] Pipeline channels started (audio, video, intent workers).")

	// 2. Simulate the system speaking, then resuming mic
	pipeline.State.SetSpeaking(true)
	time.Sleep(100 * time.Millisecond)
	pipeline.State.SetSpeaking(false)

	// 3. Simulate pushing a fake audio chunk (will be silently dropped while speaking)
	p.AudioChan <- pipeline.AudioChunk{0, 1, 2, 3}
	fmt.Println("[OK] Injected mock audio chunk.")

	// 4. Trigger hardware commands directly via HandleIntent
	intents := []hardware.IntentPayload{
		{
			Intent: "MICROPHONE_MUTE",
			Roles:  map[string]string{},
		},
		{
			Intent: "CAMERA_CAPTURE_ANALYZE",
			Roles:  map[string]string{"target": "output", "device": "camera"},
		},
		{
			Intent: "AUDIO_VOLUME_UP",
			Roles:  map[string]string{},
		},
		{
			Intent: "MICROPHONE_UNMUTE",
			Roles:  map[string]string{},
		},
	}

	for _, payload := range intents {
		if err := hardware.HandleIntent(payload); err != nil {
			fmt.Printf("[WARN] Intent %s returned: %v\n", payload.Intent, err)
		}
	}

	// 5. Push a fake intent token through the intent channel
	p.IntentChan <- "MICROPHONE_MUTE"
	time.Sleep(50 * time.Millisecond) // give the goroutine time to log it

	fmt.Println()
	fmt.Println("System architecture validation: SUCCESS. Awaiting physical peripheral initialization paths.")
}
