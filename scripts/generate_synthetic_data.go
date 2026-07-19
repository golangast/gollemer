//go:build ignore

package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

var goSyntaxPatterns = []string{
	"func %s(%s) %s { return %s }",
	"var %s = %s",
	"const %s = %s",
	"type %s struct { %s %s }",
	"if %s { %s }",
	"for i := 0; i < %s; i++ { %s }",
	"for _, %s := range %s { %s }",
	"switch %s { case %s: %s }",
	"select { case <-%s: %s }",
	"go func() { %s }()",
	"defer %s()",
	"return %s, nil",
	"if err != nil { return %s, err }",
	"%s := make([]%s, %s)",
	"%s := map[%s]%s{}",
	"ch := make(chan %s, %s)",
	"close(%s)",
	"%s, ok := %s.(%s)",
	"import ( %q )",
	"package %s",
	"%s.%s(%s)",
	"len(%s)",
	"append(%s, %s)",
	"copy(%s, %s)",
	"delete(%s, %s)",
	"%s := %s + %s",
	"%s := %s - %s",
	"%s := %s * %s",
	"%s := %s / %s",
	"func init() { %s }",
	"fmt.Sprintf(%q, %s)",
	"%s = append(%s, %s...)",
	"json.Unmarshal(%s, &%s)",
	"json.Marshal(%s)",
	"io.ReadAll(%s)",
	"os.Open(%s)",
	"http.Get(%s)",
	"context.WithTimeout(%s, %s)",
	"sync.WaitGroup{}",
	"mu.Lock(); defer mu.Unlock()",
}

var goTypes = []string{"int", "string", "bool", "float64", "error", "[]byte", "chan struct{}", "map[string]interface{}", "time.Duration", "int64", "uint32", "interface{}", "func() error"}

var goVars = []string{"data", "result", "err", "ctx", "buf", "val", "key", "item", "tmp", "cfg", "resp", "req", "wg", "mu", "ch", "size", "count", "idx", "dst", "src"}
var goValues = []string{"10", "true", "false", "nil", "0", "\"hello\"", "\"example\"", "\"test\"", "\"\"", "make([]int, 0)", "time.Second * 5"}
var goLabels = []string{"api", "handler", "server", "client", "parser", "storage", "router", "logger", "config", "worker", "callback", "middleware", "encoder", "decoder", "validator", "mapper", "loader", "scanner", "filter", "merger"}

var englishPatterns = []string{
	"How do I %s in Go?",
	"What is the best way to %s?",
	"Can you explain how %s works?",
	"Why does my %s return an error?",
	"Show me an example of %s.",
	"What is the difference between %s?",
	"How can I optimize %s?",
	"Please help me debug %s.",
	"Write a function that %s.",
	"Explain the concept of %s.",
	"How does %s work under the hood?",
	"Should I use %s or %s?",
	"What causes %s to fail?",
	"I need to implement %s, any advice?",
	"Can you review my %s implementation?",
}

var englishAnswers = []string{
	"You should use the standard library package for %s. It provides built-in support.",
	"The %s pattern is idiomatic Go. Here's a simple example: func example() error { return nil }",
	"Start by defining a struct for %s, then implement the required interface methods.",
	"For %s, you need to handle errors properly. Always check the returned error value.",
	"The %s approach works well for concurrent programs. Use goroutines and channels.",
	"Make sure to benchmark %s before optimizing. Use the testing package with -bench flag.",
	"When debugging %s, check the error message first. It usually tells you what's wrong.",
	"A common pattern for %s is to use a sync.Mutex to protect shared state.",
	"Use context.Context with %s to support cancellation and timeouts.",
	"The %s convention in Go is to return (result, error) pairs from functions.",
	"Prefer composition over inheritance for %s. Embed types rather than subclassing.",
	"Use interfaces for %s to keep your code testable and loosely coupled.",
	"For %s, start simple and add complexity only when benchmarks prove you need it.",
	"Read the Go blog post about %s. It explains the design rationale clearly.",
	"The %s approach is clean because it separates concerns and is easy to test.",
}

func randomChoice(list []string) string {
	return list[rand.Intn(len(list))]
}

func randomChoices(list []string, n int) []string {
	result := make([]string, n)
	for i := range n {
		result[i] = randomChoice(list)
	}
	return result
}

func main() {
	rand.Seed(42)

	// Track seen Q/A to avoid duplicates
	seen := make(map[string]bool)
	var pairs []struct{ query, answer string }

	// --- Generate Go syntax technical pairs ---
	for i := 0; i < 4000; i++ {
		pat := randomChoice(goSyntaxPatterns)
		args := randomChoices(goVars, strings.Count(pat, "%s"))
		qtArgs := randomChoices(goLabels, strings.Count(pat, "%q"))

		// Build query
		qIdx := 0
		qQtIdx := 0
		query := pat
		for strings.Contains(query, "%s") {
			query = strings.Replace(query, "%s", args[qIdx], 1)
			qIdx++
		}
		for strings.Contains(query, "%q") {
			query = strings.Replace(query, "%q", fmt.Sprintf("%q", qtArgs[qQtIdx]), 1)
			qQtIdx++
		}

		if seen[query] {
			continue
		}
		seen[query] = true

		answer := fmt.Sprintf("This Go pattern uses %s to handle %s efficiently. Example: %s",
			randomChoice(goTypes), randomChoice(goLabels), query)

		pairs = append(pairs, struct{ query, answer string }{query, answer})
	}

	// --- Generate English Q&A technical pairs ---
	for i := 0; i < 3000; i++ {
		pat := randomChoice(englishPatterns)
		args := randomChoices(goLabels, strings.Count(pat, "%s"))

		query := pat
		for strings.Contains(query, "%s") {
			query = strings.Replace(query, "%s", args[0], 1)
		}

		if seen[query] {
			continue
		}
		seen[query] = true

		answer := randomChoice(englishAnswers)
		for strings.Contains(answer, "%s") {
			answer = strings.Replace(answer, "%s", randomChoice(goLabels), 1)
		}

		pairs = append(pairs, struct{ query, answer string }{query, answer})
	}

	// Truncate to target
	if len(pairs) > 10000 {
		pairs = pairs[:10000]
	}
	if len(pairs) < 5000 {
		// Pad with generic pairs
		for i := len(pairs); i < 5000; i++ {
			q := fmt.Sprintf("How to use %s in Go?", randomChoice(goLabels))
			a := fmt.Sprintf("Use the %s package from the standard library. It provides %s functions.",
				randomChoice(goLabels), randomChoice(goLabels))
			if seen[q] {
				continue
			}
			pairs = append(pairs, struct{ query, answer string }{q, a})
		}
	}

	log.Printf("Generated %d synthetic training pairs", len(pairs))

	// --- Write to conversations.csv format (append mode) ---
	outputPath := filepath.Join("data", "training", "trainingdata", "synthetic_pairs.csv")
	f, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Failed to create %s: %v", outputPath, err)
	}
	defer f.Close()

	writer := csv.NewWriter(f)
	defer writer.Flush()

	// Write header
	writer.Write([]string{"query", "answer", "intent", "grammar"})
	if err := writer.Error(); err != nil {
		log.Fatalf("CSV write error: %v", err)
	}

	for _, p := range pairs {
		grammar := ""
		if strings.Contains(p.query, "func") || strings.Contains(p.query, "var ") || strings.Contains(p.query, "type ") {
			grammar = "code_syntax"
		} else {
			grammar = "technical_qa"
		}
		writer.Write([]string{p.query, p.answer, "technical", grammar})
		if err := writer.Error(); err != nil {
			log.Fatalf("CSV write error: %v", err)
		}
	}

	log.Printf("✅ Wrote %d synthetic pairs to %s", len(pairs), outputPath)

	// --- Also write the Go syntax examples as a structured text file for ingestion ---
	txtPath := filepath.Join("data", "training", "trainingdata", "syntax_blocks.txt")
	txtF, err := os.Create(txtPath)
	if err != nil {
		log.Fatalf("Failed to create %s: %v", txtPath, err)
	}
	defer txtF.Close()

	for i := 0; i < 1000; i++ {
		pat := randomChoice(goSyntaxPatterns)
		args := randomChoices(goVars, strings.Count(pat, "%s"))
		line := pat
		for strings.Contains(line, "%s") {
			line = strings.Replace(line, "%s", args[0], 1)
		}
		for strings.Contains(line, "%q") {
			line = strings.Replace(line, "%q", fmt.Sprintf("%q", randomChoice(goLabels)), 1)
		}
		fmt.Fprintln(txtF, line)
	}
	log.Printf("✅ Wrote 1000 syntax blocks to %s", txtPath)
}
