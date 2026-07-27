// Package errors provides a Prompt-to-AST code synthesis engine.
// Instead of hardcoding specific fixers, the engine acts as an open-ended
// generator: given a natural language prompt like "Create a feedforward
// neural network with backpropagation from scratch", it generates the
// structural code, then uses the compiler as an iterative architect to
// auto-heal compilation errors through the hybrid multi-pass repair loop.
package errors

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// =============================================================================
// 1. SynthesizerEngine – Prompt-to-AST Code Generator
// =============================================================================

// SynthesizerEngine generates arbitrary Go code from natural language prompts.
// It uses the LLM layer to draft implementations, then leverages the multi-pass
// repair loop to auto-heal compilation errors until the code compiles cleanly.
//
// The compiler acts as the iterative architect: each error becomes a roadmap
// for the next fix. The engine iterates until go build passes or max passes
// are exhausted.
type SynthesizerEngine struct {
	LLM       *LLMFixer
	Hybrid    *HybridEngine
	MaxPasses int
	Verbose   bool
}

// NewSynthesizerEngine creates a new code synthesis engine.
func NewSynthesizerEngine(llm *LLMFixer, hybrid *HybridEngine, maxPasses int, verbose bool) *SynthesizerEngine {
	if maxPasses <= 0 {
		maxPasses = 10
	}
	return &SynthesizerEngine{
		LLM:       llm,
		Hybrid:    hybrid,
		MaxPasses: maxPasses,
		Verbose:   verbose,
	}
}

// =============================================================================
// 2. Architecture Prompts – Generative Intent Archetypes
// =============================================================================

// GenerativeIntent defines the high-level request type and template logic.
type GenerativeIntent struct {
	Name         string
	Description  string
	SystemPrompt string
	Examples     []string
}

// GenerativeIntentCatalog defines known generative intent archetypes.
// These are high-level patterns rather than specific fixers — they guide
// the LLM on what structure to generate, not how to fix individual errors.
var GenerativeIntentCatalog = map[string]GenerativeIntent{
	"neural_network": {
		Name:        "neural_network",
		Description: "Create a feedforward neural network with backpropagation",
		SystemPrompt: `You are an expert Go programmer generating neural network code.
Generate a complete, compilable Go implementation of a feedforward neural network with:
- A Network struct holding layers
- A Layer struct with weights and biases
- Activation functions (sigmoid, relu, tanh)
- Forward propagation
- Backpropagation with gradient descent
- Training loop example
- Use only the standard library (no external dependencies).
- Use float64 for all computations.
- Include proper error handling for dimension mismatches.`,
		Examples: []string{
			"feedforward neural network with backpropagation",
			"multi-layer perceptron from scratch",
			"neural network with gradient descent training",
		},
	},
	"matrix_math": {
		Name:        "matrix_math",
		Description: "Create matrix/linear algebra utilities from scratch",
		SystemPrompt: `You are an expert Go programmer generating matrix math code.
Generate a complete, compilable Go implementation of matrix operations including:
- Matrix struct with Rows and Cols
- NewMatrix, NewOnes, NewZeros, NewIdentity
- Add, Subtract, Multiply (element-wise and dot product)
- Transpose, Inverse
- Apply (element-wise function application)
- Use float64 for all values.
- Use only the standard library.`,
		Examples: []string{
			"matrix operations library from scratch",
			"linear algebra utilities",
			"matrix multiply and transpose",
		},
	},
	"data_loader": {
		Name:        "data_loader",
		Description: "Create a CSV/dataset loader with batching and shuffling",
		SystemPrompt: `You are an expert Go programmer generating data loading code.
Generate a complete, compilable Go implementation of a data loader with:
- Dataset struct holding features and labels
- CSV loading from file path
- Train/test split
- Mini-batch iteration
- Shuffling
- Normalization/standardization
- Use only the standard library.`,
		Examples: []string{
			"data loader with batching and shuffling",
			"CSV dataset reader with train test split",
			"mini-batch data iterator",
		},
	},
	"optimizer": {
		Name:        "optimizer",
		Description: "Create optimization algorithms (SGD, Adam) from scratch",
		SystemPrompt: `You are an expert Go programmer generating optimization code.
Generate a complete, compilable Go implementation of optimization algorithms with:
- Optimizer interface
- SGD (Stochastic Gradient Descent)
- Adam optimizer
- Learning rate scheduling
- Gradient clipping
- Use float64 for all values.
- Use only the standard library.`,
		Examples: []string{
			"optimization algorithms from scratch",
			"SGD and Adam optimizers in Go",
			"gradient descent with momentum",
		},
	},
}

// DetectIntent matches a natural language prompt to a generative intent.
// Uses simple keyword matching as a fast path — the LLM can override if needed.
func DetectIntent(prompt string) *GenerativeIntent {
	lower := strings.ToLower(prompt)

	// Score each intent by keyword match count
	type scored struct {
		intent *GenerativeIntent
		score  int
	}
	var scoredIntents []scored

	for _, intent := range GenerativeIntentCatalog {
		score := 0
		for _, example := range intent.Examples {
			if strings.Contains(lower, strings.ToLower(example)) {
				score += 3
			}
		}
		// Check description keywords
		descWords := strings.Fields(strings.ToLower(intent.Description))
		for _, word := range descWords {
			if len(word) > 3 && strings.Contains(lower, word) {
				score++
			}
		}
		if score > 0 {
			scoredIntents = append(scoredIntents, scored{intent: &intent, score: score})
		}
	}

	if len(scoredIntents) == 0 {
		return nil
	}

	// Return the best match
	best := scoredIntents[0]
	for _, s := range scoredIntents[1:] {
		if s.score > best.score {
			best = s
		}
	}
	return best.intent
}

// =============================================================================
// 3. Code Generation Prompt Templates
// =============================================================================

// CodeGenPromptTemplate is the prompt template for generating Go code from a
// natural language description. It uses the GenerativeIntent's system prompt
// to guide the LLM toward the correct architecture.
const CodeGenPromptTemplate = `You are Gollemer, an expert Go code generator.

%s

Requirements:
- Output ONLY valid Go source code, nothing else.
- The code MUST compile with the standard library only.
- Include package declaration, imports, type definitions, and functions.
- Include a main() function demonstrating usage.
- Use proper error handling.
- Comment complex logic.

Generate a complete Go file for:

%s`

// RepairPromptTemplate is used when the initial generation fails to compile.
// It asks the LLM to fix only the failing parts based on the compiler error output.
const RepairPromptTemplate = `You are Gollemer, an expert Go code repair assistant.

The following Go file has compilation errors:

=== FILE: %s ===
%s

=== COMPILER ERRORS ===
%s

Fix ALL compilation errors. Output the ENTIRE corrected file as valid Go source code.
Only output the corrected Go code, no explanations.`

// =============================================================================
// 4. Synthesize – Generate Code from Prompt
// =============================================================================

// SynthesizeResult holds the result of a code synthesis operation.
type SynthesizeResult struct {
	FilePath     string
	Passes       int
	FinalBuildOK bool
	History      []SynthesizeAttempt
}

// SynthesizeAttempt records a single generation or repair attempt.
type SynthesizeAttempt struct {
	Pass    int
	Phase   string // "generate" or "repair"
	Content string
	Errors  string
}

// Synthesize generates Go code from a natural language prompt and iteratively
// repairs it until compilation succeeds. The compiler serves as the iterative
// architect — each error provides a roadmap for the next repair pass.
func (se *SynthesizerEngine) Synthesize(prompt string, targetDir string, filename string) (*SynthesizeResult, error) {
	if filename == "" {
		filename = "generated.go"
	}
	filePath := filepath.Join(targetDir, filename)

	result := &SynthesizeResult{
		FilePath: filePath,
		History:  make([]SynthesizeAttempt, 0),
	}

	// ── Step 1: Detect intent for guided generation ──────────────────────
	intent := DetectIntent(prompt)
	systemPrompt := ""
	if intent != nil {
		systemPrompt = intent.SystemPrompt
		if se.Verbose {
			fmt.Printf("🧠 Detected generative intent: %s\n", intent.Name)
		}
	}
	if systemPrompt == "" {
		systemPrompt = "You are an expert Go programmer. Generate complete, compilable Go code."
	}

	// ── Step 2: Generate initial code via LLM ────────────────────────────
	genPrompt := fmt.Sprintf(CodeGenPromptTemplate, systemPrompt, prompt)
	if se.Verbose {
		fmt.Printf("🤖 Generating code from prompt...\n")
	}

	code, err := se.LLM.Client.Complete(genPrompt, 2048, 0.1)
	if err != nil {
		return nil, fmt.Errorf("initial generation failed: %w", err)
	}

	// Extract just the Go code (strip markdown fences if present)
	code = extractGoCode(code)

	if se.Verbose {
		fmt.Printf("📄 Generated %d bytes of Go code\n", len(code))
	}

	result.History = append(result.History, SynthesizeAttempt{
		Pass:    0,
		Phase:   "generate",
		Content: code,
	})

	// Write initial code to file
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return nil, fmt.Errorf("create target directory: %w", err)
	}
	if err := os.WriteFile(filePath, []byte(code), 0644); err != nil {
		return nil, fmt.Errorf("write generated file: %w", err)
	}

	// ── Step 3: Iterative Multi-Pass Repair ──────────────────────────────
	// The compiler acts as the iterative architect. Each compilation error
	// provides a roadmap for what to fix next. We loop until build passes
	// or max passes are exhausted.
	for pass := 1; pass <= se.MaxPasses; pass++ {
		if se.Verbose {
			fmt.Printf("\n📦 Compilation pass %d/%d\n", pass, se.MaxPasses)
		}

		// Check if code compiles
		buildOutput, buildErr := se.runBuild(targetDir)
		if buildErr == nil {
			result.Passes = pass
			result.FinalBuildOK = true
			if se.Verbose {
				fmt.Printf("✅ Build passed on pass %d!\n", pass)
			}
			return result, nil
		}

		if se.Verbose {
			fmt.Printf("  ❌ Build failed:\n  %s\n", truncateString(buildOutput, 300))
		}

		// Try hybrid engine first (fast AST-level fixes)
		hybridResults, hybridErr := se.Hybrid.ProcessError(buildOutput)
		if hybridErr == nil && len(hybridResults) > 0 {
			hasFixes := false
			for _, r := range hybridResults {
				if strings.HasPrefix(r, "✅") {
					hasFixes = true
				}
				if se.Verbose {
					fmt.Printf("  %s\n", r)
				}
			}
			if hasFixes {
				continue // Try next pass with AST fixes applied
			}
		}

		// If hybrid engine couldn't fix, use LLM for semantic repair
		// Read current file content (may have been modified by hybrid engine)
		currentContent, err := os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("read current file: %w", err)
		}

		repairPrompt := fmt.Sprintf(RepairPromptTemplate,
			filename, string(currentContent), buildOutput)
		repaired, err := se.LLM.Client.Complete(repairPrompt, 2048, 0.1)
		if err != nil {
			return nil, fmt.Errorf("LLM repair pass %d failed: %w", pass, err)
		}

		repaired = extractGoCode(repaired)

		if repaired == code {
			if se.Verbose {
				fmt.Printf("  ⚠️  LLM returned identical code — no progress made\n")
			}
		}

		code = repaired
		result.History = append(result.History, SynthesizeAttempt{
			Pass:    pass,
			Phase:   "repair",
			Content: code,
			Errors:  buildOutput,
		})

		// Write repaired code
		if err := os.WriteFile(filePath, []byte(code), 0644); err != nil {
			return nil, fmt.Errorf("write repaired file: %w", err)
		}
	}

	// Final build check
	_, buildErr := se.runBuild(targetDir)
	result.FinalBuildOK = buildErr == nil

	if !result.FinalBuildOK {
		if se.Verbose {
			fmt.Printf("❌ Build still failing after %d passes\n", se.MaxPasses)
		}
	}

	return result, nil
}

// runBuild executes go build in the specified directory.
func (se *SynthesizerEngine) runBuild(dir string) (string, error) {
	cmd := exec.Command("go", "build", ".")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	outputStr := string(output)
	if err != nil {
		return outputStr, fmt.Errorf("build failed: %s", outputStr)
	}
	return outputStr, nil
}

// =============================================================================
// 5. Utility Functions
// =============================================================================

// extractGoCode strips markdown code fences and extracts the raw Go source.
func extractGoCode(text string) string {
	// Remove markdown code fences
	text = strings.TrimSpace(text)

	// Handle ```go ... ``` fences
	if strings.HasPrefix(text, "```go") {
		text = text[5:]
	} else if strings.HasPrefix(text, "```") {
		text = text[3:]
	}

	if strings.HasSuffix(text, "```") {
		text = text[:len(text)-3]
	} else if last := strings.LastIndex(text, "```"); last >= 0 {
		text = text[:last]
	}

	text = strings.TrimSpace(text)
	return text
}

// =============================================================================
// 6. High-Level Generative Handler
// =============================================================================

// HandleGenerativePrompt is the top-level entry point for code synthesis.
// Given a natural language prompt and a target package, it:
//  1. Asks the LLM/generative engine to draft the file structure
//  2. Writes it safely to the target file
//  3. Leverages the multi-pass engine to auto-fix compilation errors
//
// Example:
//
//	err := HandleGenerativePrompt(
//	    "Create a feedforward neural network with backpropagation from scratch",
//	    "./generated_projects/neural_net",
//	    "neural.go",
//	)
func HandleGenerativePrompt(prompt string, targetPackage string, filename string) error {
	// Initialize the LLM client
	llmConfig := DefaultLLMFixerConfig()
	llm := NewLLMFixer(llmConfig)

	// Initialize the hybrid engine with regex classifier
	classifier := NewRegexClassifier(targetPackage)
	hybrid := NewHybridEngine(targetPackage, classifier, true)

	// Create the synthesizer with generous max passes
	synthesizer := NewSynthesizerEngine(llm, hybrid, 10, true)

	fmt.Printf("🧬 Code Synthesis Engine\n")
	fmt.Printf("   Prompt: %s\n", prompt)
	fmt.Printf("   Target: %s/%s\n", targetPackage, filename)
	fmt.Println()

	// Step 1-3: Generate, write, and iteratively repair
	result, err := synthesizer.Synthesize(prompt, targetPackage, filename)
	if err != nil {
		return fmt.Errorf("synthesis failed: %w", err)
	}

	// Print results
	fmt.Printf("\n📊 Synthesis Result:\n")
	fmt.Printf("   File:     %s\n", result.FilePath)
	fmt.Printf("   Passes:   %d\n", result.Passes)
	fmt.Printf("   Build:    ")
	if result.FinalBuildOK {
		fmt.Println("✅ PASSED")
	} else {
		fmt.Println("❌ FAILED (after max passes)")
	}
	fmt.Printf("   History:  %d attempts\n", len(result.History))

	return nil
}

// =============================================================================
// 7. AST-Level Synthesis Support
// =============================================================================

// ASTGenerator generates Go AST nodes from natural language descriptions.
// This enables the engine to operate at the AST level rather than raw text,
// providing stronger structural guarantees.
type ASTGenerator struct {
	Fset *token.FileSet
}

// NewASTGenerator creates a new AST generator.
func NewASTGenerator() *ASTGenerator {
	return &ASTGenerator{
		Fset: token.NewFileSet(),
	}
}

// GenerateFile generates a complete *ast.File from a Go source string.
// This is used to convert LLM-generated code into AST form for hybrid repairs.
func (ag *ASTGenerator) GenerateFile(source string) (*ast.File, error) {
	return parser.ParseFile(ag.Fset, "", source, parser.ParseComments)
}

// FormatFile formats an *ast.File back to source code.
func (ag *ASTGenerator) FormatFile(file *ast.File) (string, error) {
	var buf strings.Builder
	if err := format.Node(&buf, ag.Fset, file); err != nil {
		return "", fmt.Errorf("format AST: %w", err)
	}
	return buf.String(), nil
}

// InjectFunction adds a new function declaration to an existing AST file.
func (ag *ASTGenerator) InjectFunction(file *ast.File, funcDecl *ast.FuncDecl) {
	file.Decls = append(file.Decls, funcDecl)
}

// CreateFunctionDecl creates an *ast.FuncDecl from a name, params, results, and body.
func (ag *ASTGenerator) CreateFunctionDecl(name string, params, results []string, body string) (*ast.FuncDecl, error) {
	// Parse the body as a function literal to extract the AST
	src := fmt.Sprintf("package p\nfunc %s(%s) %s {\n%s\n}", name,
		strings.Join(params, ", "),
		strings.Join(results, ", "),
		body)

	f, err := parser.ParseFile(ag.Fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse function: %w", err)
	}

	if len(f.Decls) == 0 {
		return nil, fmt.Errorf("no declaration generated")
	}

	funcDecl, ok := f.Decls[0].(*ast.FuncDecl)
	if !ok {
		return nil, fmt.Errorf("generated declaration is not a function")
	}

	return funcDecl, nil
}

// =============================================================================
// 8. Synthesis Quality Metrics
// =============================================================================

// SynthesisMetrics tracks quality metrics for generated code.
type SynthesisMetrics struct {
	LinesOfCode   int
	FunctionCount int
	TypeCount     int
	ImportCount   int
	HasTests      bool
	HasMain       bool
}

// AnalyzeCode analyzes generated Go code for quality metrics.
func AnalyzeCode(source string) (*SynthesisMetrics, error) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "", source, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse source: %w", err)
	}

	metrics := &SynthesisMetrics{
		LinesOfCode: strings.Count(source, "\n"),
		ImportCount: len(file.Imports),
	}

	// Count functions and types
	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			metrics.FunctionCount++
			if d.Name.Name == "main" {
				metrics.HasMain = true
			}
		case *ast.GenDecl:
			if d.Tok == token.TYPE {
				metrics.TypeCount += len(d.Specs)
			}
		}
	}

	return metrics, nil
}

// PrintMetrics prints the synthesis metrics in a human-readable format.
func (sm *SynthesisMetrics) PrintMetrics() {
	fmt.Printf("📊 Code Metrics:\n")
	fmt.Printf("   Lines:      %d\n", sm.LinesOfCode)
	fmt.Printf("   Functions:  %d\n", sm.FunctionCount)
	fmt.Printf("   Types:      %d\n", sm.TypeCount)
	fmt.Printf("   Imports:    %d\n", sm.ImportCount)
	fmt.Printf("   Has main(): %v\n", sm.HasMain)
}

// Ensure the JSON package is used (avoids unused import error)
var _ = json.Marshal
