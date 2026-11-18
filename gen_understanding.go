// Command: generate understanding training pairs from Go source.
// Usage: go run main.go -gen-understanding -src <dir> -out <file>
//
// This analyzes Go code with the AST analyzer, extracts structural facts,
// and creates (facts → explanation) training pairs. The model learns HOW
// THE PROGRAM KNOWS CODE — not character prediction.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

func runGenUnderstanding(srcDir, outPath string) error {
	fmt.Printf("Analyzing %s...\n", srcDir)

	// Use the analyzer to understand the code.
	project, err := analyze.Analyze(srcDir)
	if err != nil {
		return fmt.Errorf("analyze failed: %w", err)
	}

	fmt.Printf("Found %d packages\n", len(project.Packages))

	// Generate structure→language pairs.
	pairs := analyze.GenerateStructurePairs(project)
	fmt.Printf("Generated %d pairs\n", len(pairs))

	// Write as chat_pairs.jsonl format with domain="understanding".
	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer f.Close()

	count := 0
	for _, p := range pairs {
		entry := map[string]string{
			"input":  p.Input,
			"output": p.Output,
			"domain": "understanding",
		}
		data, err := json.Marshal(entry)
		if err != nil {
			continue
		}
		f.Write(append(data, '\n'))
		count++
	}

	fmt.Printf("Wrote %d pairs to %s\n", count, outPath)
	return nil
}
