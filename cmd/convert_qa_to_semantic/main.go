package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// QAEntry represents a single Q&A entry from the TSV file
type QAEntry struct {
	ArticleTitle             string
	Question                 string
	Answer                   string
	DifficultyFromQuestioner string
	DifficultyFromAnswerer   string
	ArticleFile              string
}

// IntentTrainingExample represents the output format for training
type IntentTrainingExample struct {
	Query          string                  `json:"query"`
	SemanticOutput semantic.SemanticOutput `json:"semantic_output"`
}

func main() {
	const inputPath = "trainingdata/qa/qa.txt"
	const outputPath = "trainingdata/qa_semantic_output.json"

	// Open the TSV file
	file, err := os.Open(inputPath)
	if err != nil {
		log.Fatalf("Failed to open input file %s: %v", inputPath, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var trainingExamples []IntentTrainingExample
	lineNum := 0

	// Skip header line
	if scanner.Scan() {
		lineNum++
	}

	// Process each line
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}

		// Parse TSV line
		fields := strings.Split(line, "\t")
		if len(fields) < 6 {
			log.Printf("Skipping line %d: insufficient fields (%d)", lineNum, len(fields))
			continue
		}

		entry := QAEntry{
			ArticleTitle:             fields[0],
			Question:                 fields[1],
			Answer:                   fields[2],
			DifficultyFromQuestioner: fields[3],
			DifficultyFromAnswerer:   fields[4],
			ArticleFile:              fields[5],
		}

		// Skip entries with NULL answers or empty questions
		if entry.Answer == "NULL" || strings.TrimSpace(entry.Answer) == "" ||
			strings.TrimSpace(entry.Question) == "" {
			continue
		}

		// Convert to semantic output format
		semanticOutput := semantic.SemanticOutput{
			Operation: "answer_question",
			Command:   "provide_answer",
			TargetResource: &semantic.Resource{
				Type: "qa_pair",
				Name: entry.ArticleTitle,
				Properties: map[string]interface{}{
					"question":   entry.Question,
					"answer":     entry.Answer,
					"topic":      entry.ArticleTitle,
					"difficulty": entry.DifficultyFromAnswerer,
				},
			},
			Context: semantic.Context{
				UserRole: "user",
			},
		}

		trainingExample := IntentTrainingExample{
			Query:          entry.Question,
			SemanticOutput: semanticOutput,
		}

		trainingExamples = append(trainingExamples, trainingExample)
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading input file: %v", err)
	}

	log.Printf("Converted %d Q&A pairs to semantic format", len(trainingExamples))

	// Write to output JSON file
	outputFile, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Failed to create output file %s: %v", outputPath, err)
	}
	defer outputFile.Close()

	encoder := json.NewEncoder(outputFile)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(trainingExamples); err != nil {
		log.Fatalf("Failed to encode JSON: %v", err)
	}

	fmt.Printf("Successfully converted Q&A data to %s\n", outputPath)
	fmt.Printf("Total training examples: %d\n", len(trainingExamples))
}
