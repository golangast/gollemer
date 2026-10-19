package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// This tool generates harder gocode training data from real Go repos.
// For each function, it creates: ("write a function that [intent]", "[code]").
// The intent comes from the analyzer's deep understanding; the code is real.

type ChatPair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
	Domain string `json:"domain"`
}

func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: gendistill-gocode <repo-path> <output.jsonl>")
		os.Exit(1)
	}
	repoPath := os.Args[1]
	outputPath := os.Args[2]

	proj, err := analyze.Analyze(repoPath)
	if err != nil {
		fmt.Printf("Analyze failed: %v\n", err)
		os.Exit(1)
	}

	out, err := os.Create(outputPath)
	if err != nil {
		fmt.Printf("Create failed: %v\n", err)
		os.Exit(1)
	}
	defer out.Close()
	w := bufio.NewWriter(out)
	defer w.Flush()

	count := 0
	for _, pkg := range proj.Packages {
		for _, fn := range pkg.Funcs {
			// Skip trivial: main, init, test, too long, too short.
			if fn.IsMain || fn.IsInit || fn.IsTest {
				continue
			}
			if len(fn.Src) < 50 || len(fn.Src) > 2000 {
				continue
			}
			// Get the intent.
			intent := fn.Intent()
			if intent == "" || len(intent) < 10 {
				continue
			}
			// Create the pair: "write a function to [intent]" -> [code].
			// Strip the function name from the start if present (e.g.,
			// "DefaultPipelineOptions returns..." -> "returns...").
			cleanIntent := strings.TrimSuffix(intent, ".")
			// Remove leading function name if it matches.
			lowerIntent := strings.ToLower(cleanIntent)
			lowerName := strings.ToLower(fn.Name)
			if strings.HasPrefix(lowerIntent, lowerName) {
				cleanIntent = strings.TrimSpace(cleanIntent[len(fn.Name):])
			}
			cleanIntent = strings.ToLower(cleanIntent)
			input := fmt.Sprintf("write a function that %s", cleanIntent)
			// Output is the function source, cleaned.
			output := strings.TrimSpace(fn.Src)
			if output == "" {
				continue
			}
			pair := ChatPair{Input: input, Output: output, Domain: "gocode"}
			data, _ := json.Marshal(pair)
			w.Write(data)
			w.WriteString("\n")
			count++
		}
	}
	fmt.Printf("Generated %d pairs -> %s\n", count, outputPath)
	_ = filepath.Separator // avoid unused import
}
