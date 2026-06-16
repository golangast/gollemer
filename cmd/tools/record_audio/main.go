package main

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// This tool records 1.5 seconds of audio from the microphone and saves it as raw 16kHz PCM
func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: go run record_audio/main.go <intent_name> <sample_number>")
		fmt.Println("Example: go run record_audio/main.go TURN_ON_LIGHTS 1")
		return
	}

	sampleNum := os.Args[len(os.Args)-1]

	// Join all arguments except the program name and the last argument (sample number)
	var intentParts []string
	for i := 1; i < len(os.Args)-1; i++ {
		intentParts = append(intentParts, strings.ToUpper(os.Args[i]))
	}
	intent := strings.Join(intentParts, "_")

	filename := fmt.Sprintf("dataset/audio/%s_%s.raw", intent, sampleNum)

	os.MkdirAll("dataset/audio", 0755)

	fmt.Printf("\n🎤 Get ready to say '%s'...\n", intent)
	time.Sleep(1 * time.Second)
	fmt.Println("🔴 RECORDING NOW (1.5 seconds)...")

	// Record 1.5 seconds of audio at 16kHz to a raw file using ffmpeg
	cmd := exec.Command("ffmpeg",
		"-y",
		"-f", "alsa", "-i", "default",
		"-t", "1.5",
		"-ac", "1", "-ar", "16000",
		"-f", "s16le", filename,
	)

	err := cmd.Run()
	if err != nil {
		fmt.Printf("❌ Failed to record: %v\n", err)
		return
	}

	fmt.Printf("✅ Saved to %s\n\n", filename)
	fmt.Println("Record 3-5 samples per intent for good accuracy.")
}
