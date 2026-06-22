package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type OllamaReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
	Format string `json:"format"`
}

type OllamaResp struct {
	Response string `json:"response"`
}

func main() {
	logFile := "logs/training_log.csv"
	configFile := "data/config/social_train.json"

	fmt.Println("🤖 Teacher AI Supervisor starting up...")
	fmt.Println("Watching", logFile, "and dynamically hot-patching", configFile)

	for {
		time.Sleep(15 * time.Second)

		// 1. Read Logs (last 10 lines)
		content, err := os.ReadFile(logFile)
		if err != nil {
			fmt.Println("Waiting for training logs to generate...")
			continue
		}
		lines := strings.Split(strings.TrimSpace(string(content)), "\n")
		if len(lines) < 3 {
			continue
		}

		// Get header and last 10 lines
		header := lines[0]
		startIdx := len(lines) - 10
		if startIdx < 1 {
			startIdx = 1
		}
		recentLogs := header + "\n" + strings.Join(lines[startIdx:], "\n")

		// 2. Read current config
		configData, err := os.ReadFile(configFile)
		if err != nil {
			fmt.Println("Error reading config:", err)
			continue
		}

		// 3. Prompt Ollama
		prompt := fmt.Sprintf(`You are an expert AI Training Supervisor dynamically tuning hyperparameters for an MoE model.
Current Configuration:
%s

Recent Training Logs (epoch,avg_loss,lb_loss,perplexity,learning_rate,e0_util...):
%s

Analyze the loss and perplexity trends. If loss is plateauing or jumping, you may want to adjust "learning_rate" or "router_temperature".
Output a JSON object containing ONLY the parameters you want to change. If training is going perfectly, output an empty JSON {}. 
Do not output anything outside the JSON. Example: {"learning_rate": 0.00002}
`, string(configData), recentLogs)

		reqBody, _ := json.Marshal(OllamaReq{
			Model:  "qwen2.5:3b",
			Prompt: prompt,
			Stream: false,
			Format: "json",
		})

		resp, err := http.Post("http://127.0.0.1:11434/api/generate", "application/json", bytes.NewBuffer(reqBody))
		if err != nil {
			fmt.Println("Teacher offline:", err)
			continue
		}

		var oResp OllamaResp
		json.NewDecoder(resp.Body).Decode(&oResp)
		resp.Body.Close()

		text := strings.TrimSpace(oResp.Response)
		if text == "" || text == "{}" {
			fmt.Println("🤖 Teacher evaluated logs: No parameter changes needed right now.")
			continue
		}

		// Parse the Teacher's JSON update
		var teacherUpdates map[string]interface{}
		if err := json.Unmarshal([]byte(text), &teacherUpdates); err != nil {
			fmt.Printf("Teacher generated invalid JSON: %s\n", text)
			continue
		}

		if len(teacherUpdates) == 0 {
			continue
		}

		// 4. Merge changes into existing config
		var currentConfig map[string]interface{}
		json.Unmarshal(configData, &currentConfig)

		changesMade := false
		for k, v := range teacherUpdates {
			if oldV, exists := currentConfig[k]; exists {
				if fmt.Sprintf("%v", oldV) != fmt.Sprintf("%v", v) {
					fmt.Printf("  -> 🔧 Teacher patching %s: %v -> %v\n", k, oldV, v)
					currentConfig[k] = v
					changesMade = true
				}
			}
		}

		// 5. Save updated config (Hot-reloader in training loop will pick this up automatically!)
		if changesMade {
			newConfigData, _ := json.MarshalIndent(currentConfig, "", "  ")
			if err := os.WriteFile(configFile, newConfigData, 0644); err == nil {
				fmt.Println("✅ Teacher dynamically hot-patched social_train.json!")
			}
		} else {
			fmt.Println("🤖 Teacher evaluated logs: Training stable.")
		}
	}
}
