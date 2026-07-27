package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
)

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type OllamaResponse struct {
	Response string `json:"response"`
}

func main() {
	outputFile := "data/training/trainingdata/conversing.csv"

	outFile, err := os.OpenFile(outputFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Println("Error opening output file:", err)
		return
	}
	defer outFile.Close()

	writer := csv.NewWriter(outFile)
	defer writer.Flush()

	fmt.Println("🤖 Teacher Model is generating foundational grammar lessons...")

	prompt := `You are an expert linguistics teacher. Generate 100 extremely basic, perfectly grammatical conversational Q&A pairs to teach a brand new AI model how to speak English from scratch. 
Start with simple subject-verb-object structures, then basic greetings, then simple questions and responses.
Keep every sentence under 8 words. No complex punctuation. 

Output format MUST be EXACTLY:
Q: <query>
A: <answer>
Q: <query>
A: <answer>`

	reqBody, _ := json.Marshal(OllamaRequest{
		Model:  "qwen2.5:3b",
		Prompt: prompt,
		Stream: false,
	})

	resp, err := http.Post("http://127.0.0.1:11434/api/generate", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		fmt.Println("  -> Error connecting to Ollama:", err)
		return
	}
	defer resp.Body.Close()

	var ollamaResp OllamaResponse
	json.NewDecoder(resp.Body).Decode(&ollamaResp)

	lines := strings.Split(ollamaResp.Response, "\n")
	var currentQ string
	var currentA string
	addedCount := 0

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "Q:") {
			currentQ = strings.TrimSpace(strings.TrimPrefix(line, "Q:"))
			currentQ = strings.Trim(currentQ, "\"'")
		} else if strings.HasPrefix(line, "A:") {
			currentA = strings.TrimSpace(strings.TrimPrefix(line, "A:"))
			currentA = strings.Trim(currentA, "\"'")

			if currentQ != "" && currentA != "" {
				// Append with a "grammar_lesson" intent
				writer.Write([]string{currentQ, currentA, "grammar_lesson", "OTHER"})
				addedCount++
				currentQ = ""
				currentA = ""
			}
		}
	}

	fmt.Printf("✅ Teacher successfully generated and injected %d foundational grammar lessons into conversing.csv!\n", addedCount)
	fmt.Println("Restart 'make train' so Gollemer can learn these basic sentence structures first.")
}
