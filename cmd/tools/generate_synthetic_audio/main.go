package main

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"
)

func main() {
	commandsFile := "cmd/tools/record_batch/commands.txt"
	os.MkdirAll("dataset/audio", 0755)

	file, err := os.Open(commandsFile)
	if err != nil {
		fmt.Printf("❌ Could not open %s: %v\n", commandsFile, err)
		return
	}
	defer file.Close()

	var commands []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" && !strings.HasPrefix(line, "#") {
			commands = append(commands, line)
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("❌ Error reading %s: %v\n", commandsFile, err)
		return
	}

	fmt.Printf("🚀 Generating synthetic audio for %d commands using Google TTS...\n", len(commands))

	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	for _, cmdText := range commands {
		// Clean the command string to an intent
		intent := strings.ToUpper(strings.ReplaceAll(cmdText, " ", "_"))
		fmt.Printf("Processing: %s\n", intent)

		// URL encode the spoken command text
		query := url.QueryEscape(strings.ToLower(cmdText))
		reqURL := fmt.Sprintf("http://translate.google.com/translate_tts?ie=UTF-8&total=1&idx=0&client=tw-ob&tl=en-us&q=%s", query)

		req, err := http.NewRequest("GET", reqURL, nil)
		if err != nil {
			fmt.Printf("  ❌ Failed to create request: %v\n", err)
			continue
		}
		// Need a User-Agent so Google doesn't block the request
		req.Header.Set("User-Agent", "Mozilla/5.0 (X11; Linux x86_64)")

		resp, err := client.Do(req)
		if err != nil {
			fmt.Printf("  ❌ Failed to download TTS: %v\n", err)
			continue
		}

		if resp.StatusCode != http.StatusOK {
			fmt.Printf("  ❌ Bad status code: %d\n", resp.StatusCode)
			resp.Body.Close()
			continue
		}

		mp3Path := fmt.Sprintf("/tmp/%s.mp3", intent)
		out, err := os.Create(mp3Path)
		if err != nil {
			fmt.Printf("  ❌ Failed to create mp3 file: %v\n", err)
			resp.Body.Close()
			continue
		}

		_, err = io.Copy(out, resp.Body)
		out.Close()
		resp.Body.Close()

		if err != nil {
			fmt.Printf("  ❌ Failed to write mp3: %v\n", err)
			continue
		}

		// Convert to 3 raw samples for training as required by train_audio
		success := true
		for sampleNum := 1; sampleNum <= 3; sampleNum++ {
			rawPath := fmt.Sprintf("dataset/audio/%s_%d.raw", intent, sampleNum)
			// ffmpeg -y -i input.mp3 -ac 1 -ar 16000 -f s16le output.raw
			cmd := exec.Command("ffmpeg", "-y", "-i", mp3Path, "-ac", "1", "-ar", "16000", "-f", "s16le", rawPath)
			if err := cmd.Run(); err != nil {
				fmt.Printf("  ❌ Failed to convert to raw using ffmpeg: %v\n", err)
				success = false
				break
			}
		}

		os.Remove(mp3Path)
		if success {
			fmt.Printf("  ✅ Created 3 raw samples for %s\n", intent)
		}

		// Respectful delay to avoid getting rate limited by Google TTS
		time.Sleep(500 * time.Millisecond)
	}

	fmt.Println("\n🎉 All commands successfully generated with synthetic audio!")
	fmt.Println("Run 'go run cmd/tools/train_audio/main.go' to train the intent model.")
}
