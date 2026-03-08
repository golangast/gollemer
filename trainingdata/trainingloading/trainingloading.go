package trainingloading

import (
	"encoding/json"
	"os"
	"strings"
	"log"
)


func LoadWikiQA(filePath string, pairs *[]struct{ Q, A string }) {
	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("⚠️  Could not open wikiqa file %s: %v. Skipping.", filePath, err)
		return
	}
	defer file.Close()

	var data []struct {
		Query       string `json:"Query"`
		Description string `json:"description"`
	}

	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		log.Printf("⚠️  Could not decode wikiqa json %s: %v. Skipping.", filePath, err)
		return
	}

	loadedCount := 0
	for _, item := range data {
		if item.Query != "" && item.Description != "" {
			*pairs = append(*pairs, struct{ Q, A string }{item.Query, item.Description})
			loadedCount++
		}
	}
	log.Printf("✅ Loaded %d pairs from %s", loadedCount, filePath)
}

func LoadQATxt(filePath string, pairs *[]struct{ Q, A string }) {
	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("⚠️  Could not open qa.txt file %s: %v. Skipping.", filePath, err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	scanner.Scan() // Skip header line

	count := 0
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) >= 3 {
			question := parts[1]
			answer := parts[2]
			if question != "" && answer != "" && answer != "NULL" {
				*pairs = append(*pairs, struct{ Q, A string }{question, answer})
				count++
			}
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("⚠️  Error reading qa.txt file %s: %v", filePath, err)
	} else {
		log.Printf("✅ Loaded %d pairs from %s", count, filePath)
	}
}