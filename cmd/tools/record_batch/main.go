package main

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"
)

// This tool reads a list of commands from a text file and automatically
// prompts the user to record each one 3 times.
func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run cmd/tools/record_batch/main.go <commands_file.txt>")
		fmt.Println("Create a simple text file with one command per line.")
		return
	}

	filepath := os.Args[1]
	file, err := os.Open(filepath)
	if err != nil {
		fmt.Printf("❌ Could not open file: %v\n", err)
		return
	}
	defer file.Close()

	os.MkdirAll("dataset/audio", 0755)

	var commands []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			commands = append(commands, line)
		}
	}

	if len(commands) == 0 {
		fmt.Println("⚠️ No commands found in file.")
		return
	}

	fmt.Printf("🚀 Loaded %d commands to record.\n", len(commands))
	fmt.Println("We will record 3 samples for each command.")
	fmt.Println("Press [ENTER] when you are ready to begin...")
	bufio.NewReader(os.Stdin).ReadBytes('\n')

	for _, cmdStr := range commands {
		// Format to intent name: "Turn on lights" -> "TURN_ON_LIGHTS"
		intent := strings.ToUpper(strings.ReplaceAll(cmdStr, " ", "_"))

		fmt.Printf("\n==========================================\n")
		fmt.Printf("🎙️  NEW COMMAND: '%s'\n", cmdStr)
		fmt.Printf("==========================================\n")
		
		for sampleNum := 1; sampleNum <= 3; sampleNum++ {
			fmt.Printf("\nGet ready to say '%s' (Sample %d of 3)...\n", cmdStr, sampleNum)
			fmt.Println("Press [ENTER] to start recording...")
			bufio.NewReader(os.Stdin).ReadBytes('\n')

			filename := fmt.Sprintf("dataset/audio/%s_%d.raw", intent, sampleNum)

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

			fmt.Printf("✅ Saved to %s\n", filename)
			time.Sleep(500 * time.Millisecond)
		}
	}

	fmt.Printf("\n🎉 All %d commands recorded successfully!\n", len(commands))
	fmt.Println("You can now run 'go run cmd/tools/train_audio/main.go' to train the network.")
}
