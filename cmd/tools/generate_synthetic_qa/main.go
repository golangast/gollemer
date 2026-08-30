package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type ollamaReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type ollamaResp struct {
	Response string `json:"response"`
}

func main() {
	prompt := `Generate 10 Go training Q&A pairs with explicit chain-of-thought reasoning in the assistant's answer.

Format each pair as a JSON object:
{"q": "...", "a": "To give you the best answer, let me break this down. First, ... Second, ... Therefore, ..."}

Rules:
1. Questions should be natural user queries about Go programming
2. Answers MUST follow this exact structure:
   - Start with: "To give you the best answer, let me break this down."
   - Then "First, [point]" and "Second, [point]"
   - End with "Therefore, [conclusion]"
3. Mix of social greetings and technical Go questions
4. Keep answers concise (2-4 sentences total)
5. Return ONLY a JSON array of 10 objects, no extra text

Example:
[{"q": "What is a channel?", "a": "To give you the best answer, let me break this down. First, a channel is a typed conduit for passing values between goroutines. Second, it synchronizes execution by blocking sends and receives. Therefore, channels are the primary mechanism for safe concurrent communication in Go."}]`

	reqBody, _ := json.Marshal(ollamaReq{
		Model:  "qwen2.5:3b",
		Prompt: prompt,
		Stream: false,
	})

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post("http://127.0.0.1:11434/api/generate", "application/json", strings.NewReader(string(reqBody)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Ollama connection failed: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	var oResp ollamaResp
	if err := json.NewDecoder(resp.Body).Decode(&oResp); err != nil {
		fmt.Fprintf(os.Stderr, "JSON decode error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println(oResp.Response)
}
