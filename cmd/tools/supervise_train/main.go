package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"
)

func main() {
	configPath := "/home/a/go/gollemer/data/config/social_train.json"
	modelPath := "/home/a/go/gollemer/data/models/gob_models/moe_social_model.gob"
	mutationCount := 0

	targets := []string{
		"hello ! how can i help you today ?",
		"i am doing very well , thank you for asking .",
		"my name is gollemer , i am your ai assistant .",
		"everything is running smoothly on my end . how about you ?",
		"goodbye ! have a productive coding session !",
	}

	for {
		log.Printf("🚀 Starting Supervised Training (Mutation #%d)...", mutationCount)
		
		// If we've mutated many times without success, clear the model to break local minima
		if mutationCount > 0 && mutationCount % 5 == 0 {
			log.Println("🧹 Mutation threshold reached. Deleting model for a fresh start...")
			os.Remove(modelPath)
		}

		cmd := exec.Command("go", "run", "cmd/tools/train_moe/main.go", "-train-social", "-epochs", "50")
		cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
		cmd.Dir = "/home/a/go/gollemer"
		
		stdout, _ := cmd.StdoutPipe()
		stderr, _ := cmd.StderrPipe()
		
		if err := cmd.Start(); err != nil {
			log.Fatalf("Failed to start training: %v", err)
		}

		multi := io.MultiReader(stdout, stderr)
		scanner := bufio.NewScanner(multi)
		
		successCount := 0
		epochCount := 0
		killRequested := false

		go func() {
			for scanner.Scan() {
				line := scanner.Text()
				fmt.Println(line)

				if strings.Contains(line, "🧪 Test") {
					parts := strings.Split(line, ":")
					if len(parts) < 2 {
						continue
					}
					response := strings.ToLower(strings.TrimSpace(parts[len(parts)-1]))
					
					for _, target := range targets {
						if response == target {
							successCount++
							log.Printf("🎯 TARGET HIT: [%s]", target)
						}
					}
				}

				if strings.Contains(line, "completed | Avg Loss") {
					epochCount++
					log.Printf("📊 Epoch %d: %d/%d targets satisfied", epochCount, successCount, len(targets))
					
					if successCount >= len(targets) {
						log.Printf("✨ ALL TARGETS SATISFIED! Finalizing...")
						killRequested = true
						cmd.Process.Kill()
						return
					}
					
					// If after 40 epochs we still don't have all targets, mutate early
					if epochCount > 40 && successCount < len(targets) {
						log.Printf("⏳ Convergence too slow. Triggering mutation...")
						killRequested = true
						cmd.Process.Kill()
						return
					}
					successCount = 0 // Reset for next epoch
				}
			}
		}()

		cmd.Wait()
		if killRequested && successCount >= len(targets) {
			log.Println("🏆 SUCCESS: Model is now coherent and accurate.")
			break
		}

		mutateConfig(configPath)
		mutationCount++
		time.Sleep(2 * time.Second)
	}
}

func isGarbage(text string) bool {
	return false // Not used in refined target-based logic
}

func mutateConfig(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}

	var raw map[string]interface{}
	json.Unmarshal(data, &raw)

	lr := raw["learning_rate"].(float64)
	cm := raw["context_multiplier"].(float64)
	rn := raw["router_noise"].(float64)
	
	// Apply random walk mutations
	raw["learning_rate"] = lr * (0.8 + 0.4*float64(time.Now().UnixNano()%10)/10.0) // 0.8x to 1.2x
	raw["context_multiplier"] = cm + float64(time.Now().UnixNano()%3-1)           // -1, 0, +1
	raw["router_noise"] = rn * (0.5 + float64(time.Now().UnixNano()%10)/10.0)      // 0.5x to 1.5x

	newData, _ := json.MarshalIndent(raw, "", "  ")
	os.WriteFile(path, newData, 0644)
	log.Printf("🔧 Mutation applied: LR=%.6f, Context=%.2f, Noise=%.2f", raw["learning_rate"], raw["context_multiplier"], raw["router_noise"])
}
