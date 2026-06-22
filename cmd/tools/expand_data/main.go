package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
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
	inputFile := "data/training/trainingdata/conversing.csv"
	outputFile := "data/training/trainingdata/conversations.csv"

	file, err := os.Open(inputFile)
	if err != nil {
		fmt.Println("Error opening input file:", err)
		return
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error reading CSV:", err)
		return
	}

	outFile, err := os.Create(outputFile)
	if err != nil {
		fmt.Println("Error creating output file:", err)
		return
	}
	defer outFile.Close()

	writer := csv.NewWriter(outFile)
	defer writer.Flush()

	fmt.Printf("Loaded %d rows from %s. Starting expansion...\n", len(records), inputFile)

	count := 0
	for _, record := range records {
		if len(record) < 3 {
			continue
		}
		q := record[0]
		a := record[1]
		intent := record[2]
		grammar := "OTHER"
		if len(record) > 3 {
			grammar = record[3]
		}

		// Write original
		writer.Write([]string{q, a, intent, grammar})

		teacherPrompt := fmt.Sprintf(`You are a master AI data generator. Reword the following query and answer pair into 3 diverse, natural, grammatically flawless variations for a social chat dataset. Keep them concise (single short sentences). 
For the answer, add a brief, thoughtful "reasoning prefix" in brackets where appropriate (e.g., "[Evaluating user status...] I am doing well, thank you!").

Original Query: "%s"
Original Answer: "%s"

Output format:
Q: <query variation 1>
A: <answer variation 1>
Q: <query variation 2>
A: <answer variation 2>
Q: <query variation 3>
A: <answer variation 3>`, q, a)

		reqBody, _ := json.Marshal(OllamaRequest{
			Model:  "qwen2.5:3b",
			Prompt: teacherPrompt,
			Stream: false,
		})

		fmt.Printf("[%d/%d] Expanding: %s\n", count+1, len(records), q)
		resp, err := http.Post("http://127.0.0.1:11434/api/generate", "application/json", bytes.NewBuffer(reqBody))
		if err != nil {
			fmt.Println("  -> Error connecting to Ollama:", err)
			time.Sleep(2 * time.Second)
			continue
		}

		var ollamaResp OllamaResponse
		json.NewDecoder(resp.Body).Decode(&ollamaResp)
		resp.Body.Close()

		lines := strings.Split(ollamaResp.Response, "\n")
		var currentQ string
		var currentA string

		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "Q:") {
				currentQ = strings.TrimSpace(strings.TrimPrefix(line, "Q:"))
			} else if strings.HasPrefix(line, "A:") {
				currentA = strings.TrimSpace(strings.TrimPrefix(line, "A:"))
				if currentQ != "" && currentA != "" {
					writer.Write([]string{currentQ, currentA, intent, grammar})
					currentQ = ""
					currentA = ""
				}
			}
		}

		count++
		if count%5 == 0 {
			writer.Flush()
			fmt.Printf("  -> Saved checkpoint to %s\n", outputFile)
		}
	}

	fmt.Println("🚀 Finished expanding dataset!")
}
