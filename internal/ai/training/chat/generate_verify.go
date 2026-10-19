package chat

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// This file does "both": the neural model writes code, then the analyzer
// deeply understands what was written. The understanding acts as a check
// on the generation — if the analyzer's reading doesn't match the intent,
// the code is flagged.

// GeneratedCode is code written by the model plus the analyzer's
// understanding of it.
type GeneratedCode struct {
	Prompt   string // what was asked for
	Code     string // what the model wrote
	Analysis string // what the analyzer understands it does
	Matches  bool   // does the analysis match the intent?
}

// GenerateAndUnderstand writes code with the model, then analyzes it.
func GenerateAndUnderstand(projectRoot, prompt string) (*GeneratedCode, error) {
	// 1. Generate with the neural model.
	model, err := loadRealModel(projectRoot, GocodeDomain)
	if err != nil {
		return nil, fmt.Errorf("no gocode model: %w", err)
	}
	code, err := model.Predict(prompt, 100)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	// 2. Write to temp file and analyze.
	tmpDir, err := os.MkdirTemp("", "gencode")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	// Wrap in package main if needed.
	if !strings.Contains(code, "package ") {
		code = "package main\n\n" + code
	}
	codeFile := filepath.Join(tmpDir, "gen.go")
	if err := os.WriteFile(codeFile, []byte(code), 0644); err != nil {
		return nil, err
	}
	goMod := filepath.Join(tmpDir, "go.mod")
	os.WriteFile(goMod, []byte("module gencode\n\ngo 1.24\n"), 0644)

	proj, err := analyze.Analyze(tmpDir)
	if err != nil {
		return &GeneratedCode{
			Prompt:   prompt,
			Code:     code,
			Analysis: fmt.Sprintf("Failed to analyze: %v", err),
			Matches:  false,
		}, nil
	}

	// 3. Get the analyzer's understanding of the generated function.
	var analysis strings.Builder
	var funcName string
	for _, pkg := range proj.Packages {
		for _, fn := range pkg.Funcs {
			// Find the generated function (not main, not init, not test).
			if !fn.IsMain && !fn.IsTest && !fn.IsInit {
				funcName = strings.ToLower(fn.Name)
				analysis.WriteString(fmt.Sprintf("Function: %s\n", fn.Sig))
				analysis.WriteString(fmt.Sprintf("Does: %s\n", fn.Summary()))
				analysis.WriteString(fmt.Sprintf("Control: %s\n", fn.ControlPhrase()))
				break
			}
		}
		if analysis.Len() > 0 {
			break
		}
	}
	if analysis.Len() == 0 {
		analysis.WriteString("No function found in generated code.")
	}

	// 4. Check if it matches intent: the function name should contain
	// a key term from the prompt.
	matches := intentMatches(prompt, funcName)

	return &GeneratedCode{
		Prompt:   prompt,
		Code:     code,
		Analysis: analysis.String(),
		Matches:  matches,
	}, nil
}

func intentMatches(prompt, funcName string) bool {
	prompt = strings.ToLower(prompt)
	funcName = strings.ToLower(funcName)
	// Extract key terms from the prompt.
	// For "checks if a number is even": key terms include "even".
	// For "reverses a string": key terms include "reverse".
	// The generated function's NAME should contain a key term.
	words := strings.Fields(prompt)
	skip := map[string]bool{"write": true, "a": true, "function": true, "that": true, "if": true, "is": true, "the": true, "an": true}
	for _, w := range words {
		w = strings.Trim(w, ",.")
		if !skip[w] && len(w) > 2 {
			w = strings.TrimSuffix(w, "s") // "checks"->"check", "reverses"->"reverse"
			if strings.Contains(funcName, w) {
				return true
			}
		}
	}
	return false
}
