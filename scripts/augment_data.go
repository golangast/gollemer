package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const (
	OllamaModel = "llama3"
	OllamaURL   = "http://localhost:11434/api/generate"
)

type Variation struct {
	Query  string `json:"query"`
	Answer string `json:"answer"`
}

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
	Format string `json:"format"`
}

type OllamaResponse struct {
	Response string `json:"response"`
}

func generateVariations(q, a, intent string) []Variation {
	prompt := fmt.Sprintf(`You are an expert conversational data generator for an AI system.
Given the following user query and assistant response pair, generate 5 distinct, high-quality variations of the user query and corresponding variations of the assistant response.
Keep the original intent exactly the same.
For the assistant response, add a brief, thoughtful "reasoning prefix" in brackets where appropriate (e.g., "[Evaluating user status...] I am doing well, thank you!"). This helps the model's MoE router learn structured thinking before generating text.

Original Query: %s
Original Answer: %s
Intent: %s

Respond ONLY with a JSON array of objects, where each object has "query" and "answer" keys. Do not include markdown formatting, markdown code blocks, or any other text outside the JSON array.
`, q, a, intent)

	reqData := OllamaRequest{
		Model:  OllamaModel,
		Prompt: prompt,
		Stream: false,
		Format: "json",
	}

	reqBytes, err := json.Marshal(reqData)
	if err != nil {
		fmt.Printf("Error marshaling request: %v\n", err)
		return nil
	}

	resp, err := http.Post(OllamaURL, "application/json", bytes.NewReader(reqBytes))
	if err != nil {
		fmt.Printf("Connection Error: Is Ollama running on localhost:11434? (%v)\n", err)
		return nil
	}
	defer resp.Body.Close()

	respBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("Error reading response: %v\n", err)
		return nil
	}

	var ollamaResp OllamaResponse
	if err := json.Unmarshal(respBytes, &ollamaResp); err != nil {
		fmt.Printf("Error parsing ollama response: %v\n", err)
		return nil
	}

	text := strings.TrimSpace(ollamaResp.Response)
	if strings.HasPrefix(text, "```json") {
		text = text[7:]
	}
	if strings.HasPrefix(text, "```") {
		text = text[3:]
	}
	if strings.HasSuffix(text, "```") {
		text = text[:len(text)-3]
	}
	text = strings.TrimSpace(text)

	var variations []Variation
	if err := json.Unmarshal([]byte(text), &variations); err != nil {
		fmt.Printf("Error parsing JSON array from model: %v\n", err)
		return nil
	}

	return variations
}
