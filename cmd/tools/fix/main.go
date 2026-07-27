// Package main implements the 'gollemer fix' CLI subcommand.
// It runs the auto-fix loop using Gollemer's own local MoE model for error
// intent classification and AST-based patch generation.
// No external APIs are needed — everything runs locally.
//
// The loop:
//  1. Runs `go build <package>` for compile errors
//  2. Parses and classifies errors through the MoE classification model
//  3. Applies AST-based fixes automatically (adds missing symbols, imports, handlers)
//  4. Repeats until build passes or max iterations are reached
//
// Usage:
//
//	gollemer fix ./cmd/tools/multi_orchestrator
//	gollemer fix -retries=5 -auto-apply ./...
//	gollemer fix -verbose -test-mode ./pkg/mypackage
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/errors"
	"github.com/golangast/gollemer/internal/ai/llm"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/agent"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func main() {
	maxRetries := flag.Int("retries", 5, "Maximum auto-fix iterations")
	autoApply := flag.Bool("auto-apply", false, "Automatically apply AST-based fixes without confirmation")
	testMode := flag.Bool("test-mode", false, "Run tests instead of build for validation")
	verbose := flag.Bool("verbose", false, "Print detailed output")
	flag.Parse()

	pkgTarget := flag.Arg(0)
	if pkgTarget == "" {
		fmt.Fprintf(os.Stderr, "Usage: gollemer fix [flags] <package>\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  gollemer fix ./cmd/tools/multi_orchestrator\n")
		fmt.Fprintf(os.Stderr, "  gollemer fix -retries=5 -auto-apply ./...\n")
		fmt.Fprintf(os.Stderr, "  gollemer fix -verbose -test-mode .\n")
		os.Exit(1)
	}

	workingDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	projectRoot, err := llm.FindProjectRoot()
	if err != nil {
		projectRoot = workingDir
	}

	fmt.Printf("🔧 Gollemer Auto-Fix\n")
	fmt.Printf("   Package:       %s\n", pkgTarget)
	fmt.Printf("   Max retries:   %d\n", *maxRetries)
	fmt.Printf("   Auto-apply:    %v\n", *autoApply)
	fmt.Printf("   Test mode:     %v\n", *testMode)
	fmt.Printf("   Working dir:   %s\n", workingDir)
	fmt.Printf("   Project root:  %s\n", projectRoot)
	fmt.Println()

	// If auto-apply is enabled, use the MoE-based ErrorRouter directly
	if *autoApply {
		err := runAutoApplyFix(projectRoot, pkgTarget, *maxRetries, *testMode, *verbose)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\n❌ Auto-apply fix failed: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Fallback to LLM-driven auto-fix loop (original behavior)
	runLLMFix(pkgTarget, workingDir, projectRoot, *maxRetries, *verbose)
}

// runAutoApplyFix uses the MoE-based ErrorRouter for automatic AST-level fixes
// without needing the full LLM pipeline.
func runAutoApplyFix(projectRoot string, pkgTarget string, maxRetries int, testMode bool, verbose bool) error {
	// Initialize the error router with the MoE classification model
	router, err := errors.NewErrorRouter(projectRoot, verbose)
	if err != nil {
		if verbose {
			log.Printf("⚠️  Could not initialize MoE classifier: %v", err)
			log.Printf("   Falling back to regex-based classification")
		}
	}

	// Handle recursive package scanning (./...)
	packages := expandPackages(pkgTarget, projectRoot)
	if len(packages) == 0 {
		packages = []string{pkgTarget}
	}

	if verbose {
		log.Printf("Packages to process: %v", packages)
	}

	for _, pkg := range packages {
		fmt.Printf("📦 Processing package: %s\n", pkg)

		for i := 0; i < maxRetries; i++ {
			if verbose {
				log.Printf("  Iteration %d/%d", i+1, maxRetries)
			}

			// Run validation (build or test)
			var output string
			var buildErr error

			if testMode {
				output, buildErr = runTest(pkg)
			} else {
				output, buildErr = runBuild(pkg)
			}

			if buildErr == nil {
				fmt.Printf("  ✅ Build passed for %s\n", pkg)
				break // Package is clean, move to next
			}

			if verbose {
				fmt.Printf("  ❌ Build failed:\n  %s\n", truncateString(output, 300))
			}

			// Process errors through the router
			var result *errors.RouterResult
			if router != nil {
				result = router.ProcessCompilerOutput(output)
			} else {
				// Use raw parsing only
				result = &errors.RouterResult{}
				parsed := errors.ParseCompilerOutput(output)
				result.TotalCount = len(parsed)
				result.Errors = parsed
				for _, pe := range parsed {
					fixer := errors.GetFixer(pe.Intent)
					if fixer != nil {
						msg, fixErr := fixer(pe, projectRoot)
						if fixErr == nil {
							result.FixedCount++
							result.FixResults = append(result.FixResults, errors.FixResult{
								ParsedError: pe,
								Fixed:       true,
								Message:     msg,
							})
						}
					}
				}
			}

			if result.FixedCount == 0 {
				if i == maxRetries-1 {
					fmt.Printf("  ⚠️  No fixes could be applied for %s\n", pkg)
				} else {
					if verbose {
						log.Printf("  ⚠️  No fixes applied, retrying...")
					}
				}
				continue
			}

			// Print results
			for _, fr := range result.FixResults {
				if fr.Fixed {
					fmt.Printf("    ✅ %s\n", fr.Message)
				}
			}
		}
	}

	return nil
}

// runLLMFix runs the original LLM-driven auto-fix loop as a fallback.
func runLLMFix(pkgTarget string, workingDir string, projectRoot string, maxRetries int, verbose bool) {
	// Initialize Gollemer's local inference engine
	inference, err := newLocalInference(workingDir)
	if err != nil {
		fmt.Printf("⚠️  Could not initialize local model: %v\n", err)
		fmt.Printf("   Running in test-only mode (no auto-fixes)\n")
	}

	// First try the MoE-based router for fast fixes
	router, routerErr := errors.NewErrorRouter(projectRoot, verbose)
	if routerErr == nil {
		// Run build first to see if there are errors
		buildOutput, buildErr := runBuild(pkgTarget)
		if buildErr != nil {
			if verbose {
				log.Printf("📋 MoE Router processing build errors...")
			}
			result := router.ProcessCompilerOutput(buildOutput)
			if result.FixedCount > 0 {
				fmt.Printf("📋 MoE Router applied %d fixes:\n", result.FixedCount)
				for _, fr := range result.FixResults {
					if fr.Fixed {
						fmt.Printf("  ✅ %s\n", fr.Message)
					}
				}
				// Rebuild to check if fixes resolved the issue
				_, buildErr2 := runBuild(pkgTarget)
				if buildErr2 == nil {
					fmt.Printf("✅ Build passed after MoE router fixes!\n")
					return
				}
				if verbose {
					log.Printf("MoE router fixes insufficient, falling back to LLM loop...")
				}
			}
		} else {
			fmt.Printf("✅ Build passed! No fixes needed.\n")
			return
		}
	}

	// Fallback: Create the LLM callback using Gollemer's local MoE model
	callLLM := createLocalLLMCallback(inference, verbose)

	// Run the auto-fix loop
	if err := agent.RunAutoFix(pkgTarget, workingDir, callLLM); err != nil {
		fmt.Fprintf(os.Stderr, "\n❌ Auto-fix failed: %v\n", err)
		os.Exit(1)
	}
}

// runBuild executes 'go build' and returns output and any error.
func runBuild(pkg string) (string, error) {
	cmd := exec.Command("go", "build", pkg)
	output, err := cmd.CombinedOutput()
	outputStr := string(output)
	if err != nil {
		return outputStr, fmt.Errorf("build failed: %v", err)
	}
	return outputStr, nil
}

// runTest executes 'go test' and returns output and any error.
func runTest(pkg string) (string, error) {
	cmd := exec.Command("go", "test", "-v", "-timeout", "30s", pkg)
	output, err := cmd.CombinedOutput()
	outputStr := string(output)
	if err != nil {
		return outputStr, fmt.Errorf("test failed: %v", err)
	}
	return outputStr, nil
}

// expandPackages expands ./... patterns to individual packages.
func expandPackages(pattern string, projectRoot string) []string {
	if !strings.HasSuffix(pattern, "/...") {
		return nil // Not a recursive pattern
	}

	baseDir := strings.TrimSuffix(pattern, "/...")
	if baseDir == "" {
		baseDir = "."
	}

	fullBase := filepath.Join(projectRoot, baseDir)
	var packages []string

	err := filepath.Walk(fullBase, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() {
			// Check if it's a Go package (has .go files)
			hasGoFiles, _ := filepath.Glob(filepath.Join(path, "*.go"))
			if len(hasGoFiles) > 0 {
				relPath, _ := filepath.Rel(projectRoot, path)
				packages = append(packages, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return nil
	}

	return packages
}

// truncateString truncates a string to the specified max length.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// ============================================================================
// Below is the existing LLM-driven fix code kept for backward compatibility.
// ============================================================================

// LocalInference wraps Gollemer's local MoE model and BPE tokenizer
// for generating patch suggestions from test failure logs.
type LocalInference struct {
	SocialModel *moe.IntentMoE
	BPETok      *tokenizer.BPETokenizer
	ProjectRoot string
	Verbose     bool
}

// newLocalInference initializes the local inference engine by loading
// Gollemer's MoE model and BPE tokenizer from the project's data directory.
func newLocalInference(workingDir string) (*LocalInference, error) {
	projectRoot, err := llm.FindProjectRoot()
	if err != nil {
		return nil, fmt.Errorf("find project root: %w", err)
	}

	inf := &LocalInference{
		ProjectRoot: projectRoot,
	}

	// Try to load BPE tokenizer
	bpePath := filepath.Join(projectRoot, "data/models/gob_models/bpe_tokenizer.gob")
	if _, statErr := os.Stat(bpePath); statErr == nil {
		if tok, loadErr := tokenizer.LoadBPETokenizer(bpePath); loadErr == nil {
			inf.BPETok = tok
			log.Printf("✅ BPE Tokenizer loaded from %s (vocab=%d)", bpePath, tok.Vocab.Size())
		}
	} else {
		log.Printf("ℹ️  BPE Tokenizer not found at %s", bpePath)
	}

	// Try to load the social MoE model for inference
	modelPaths := []string{
		filepath.Join(projectRoot, "data/models/gob_models/moe_social_model.gob"),
		filepath.Join(projectRoot, "data/models/gob_models/moe_classification_model.gob"),
		filepath.Join(projectRoot, "data/models/checkpoints/latest_periodic.gob"),
		filepath.Join(projectRoot, "data/models/gob_models/golden_checkpoint.gob"),
	}

	for _, p := range modelPaths {
		if _, statErr := os.Stat(p); statErr != nil {
			continue
		}
		loaded, loadErr := moe.LoadIntentMoEModelWithFallback(p)
		if loadErr == nil && loaded != nil {
			loaded.RepairArchitecture()
			inf.SocialModel = loaded
			log.Printf("✅ MoE model loaded from %s", filepath.Base(p))
			break
		}
	}

	if inf.SocialModel == nil {
		log.Printf("ℹ️  No MoE model found. Run training first to enable auto-fixes.")
	}

	return inf, nil
}

// Generate generates a response from the local MoE model given a prompt.
func (li *LocalInference) Generate(prompt string) (string, error) {
	if li.BPETok == nil || li.SocialModel == nil {
		return generateFallbackPatch(prompt), nil
	}

	// Format as ChatML for the local model
	chatmlPrompt := fmt.Sprintf("<|im_start|>system\nYou are Gollemer, an expert Go AI assistant. You fix failing tests by applying SEARCH/REPLACE patches.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt)

	// Tokenize with BPE
	tokenIDs := li.BPETok.Encode(chatmlPrompt)
	if len(tokenIDs) == 0 {
		return generateFallbackPatch(prompt), nil
	}

	// Generate tokens using the MoE model
	var generatedIDs []int
	currentSequence := append([]int(nil), tokenIDs...)

	imEndID := -1
	eosID := -1
	if li.BPETok.Vocab != nil {
		imEndID = li.BPETok.Vocab.GetTokenID("<|im_end|>")
		eosID = li.BPETok.Vocab.GetTokenID("</s>")
	}

	maxNewTokens := 256
	for i := 0; i < maxNewTokens; i++ {
		nextTokenID := li.SocialModel.PredictNextToken(currentSequence)

		if nextTokenID == imEndID || nextTokenID == eosID || nextTokenID == 0 {
			break
		}

		generatedIDs = append(generatedIDs, nextTokenID)
		currentSequence = append(currentSequence, nextTokenID)
	}

	responseText := li.BPETok.Decode(generatedIDs)
	if responseText == "" {
		return generateFallbackPatch(prompt), nil
	}

	return responseText, nil
}

// createLocalLLMCallback returns a function that uses Gollemer's local MoE model
// to generate patch suggestions from test failure logs.
func createLocalLLMCallback(inference *LocalInference, verbose bool) func(prompt string, tools []agent.LLMTool) ([]agent.ToolCall, string, error) {
	return func(prompt string, tools []agent.LLMTool) ([]agent.ToolCall, string, error) {
		response, err := inference.Generate(prompt)
		if err != nil {
			return nil, "", fmt.Errorf("local model error: %w", err)
		}

		if verbose {
			fmt.Printf("\n🧠 Local Model Response:\n%s\n", response)
		}

		toolCalls := parsePatchFromResponse(response)
		if len(toolCalls) > 0 {
			return toolCalls, response, nil
		}

		fix := generateFallbackPatch(prompt)
		toolCalls = parsePatchFromResponse(fix)
		return toolCalls, fix, nil
	}
}

// parsePatchFromResponse extracts SEARCH/REPLACE patches from the model's response.
func parsePatchFromResponse(response string) []agent.ToolCall {
	if !strings.Contains(response, "<<<<<<< SEARCH") {
		return nil
	}

	filePath := extractFilePath(response)
	if filePath == "" {
		filePath = "fix.go"
	}

	searchIdx := strings.Index(response, "<<<<<<< SEARCH")
	if searchIdx == -1 {
		return nil
	}

	patchBlock := response[searchIdx:]
	replaceEnd := strings.Index(patchBlock, ">>>>>>> REPLACE")
	if replaceEnd != -1 {
		patchBlock = patchBlock[:replaceEnd+len(">>>>>>> REPLACE")]
	}

	return []agent.ToolCall{
		{
			Name: "apply_patch",
			Arguments: func() []byte {
				json := fmt.Sprintf(`{"file":"%s","patch":"%s"}`, filePath, escapeJSON(patchBlock))
				return []byte(json)
			}(),
		},
	}
}

// extractFilePath tries to extract a file path from a response text.
func extractFilePath(text string) string {
	patterns := []string{"file: ", "in file ", "File: ", "In file ", "path: ", "Path: "}
	for _, p := range patterns {
		if idx := strings.Index(text, p); idx != -1 {
			candidate := text[idx+len(p):]
			if nl := strings.IndexAny(candidate, "\n "); nl != -1 {
				candidate = candidate[:nl]
			}
			candidate = strings.TrimSpace(candidate)
			if candidate != "" && (strings.HasSuffix(candidate, ".go") || !strings.Contains(candidate, ".")) {
				return candidate
			}
		}
	}
	return ""
}

// generateFallbackPatch creates a simple fix suggestion based on common error patterns.
func generateFallbackPatch(testOutput string) string {
	lower := strings.ToLower(testOutput)

	if strings.Contains(lower, "undefined:") || strings.Contains(lower, "undeclared name:") {
		var symbol string
		for _, line := range strings.Split(testOutput, "\n") {
			lowLine := strings.ToLower(line)
			if strings.Contains(lowLine, "undefined:") {
				parts := strings.Split(line, "undefined:")
				if len(parts) > 1 {
					symbol = strings.TrimSpace(strings.Split(parts[1], "\n")[0])
				}
				break
			}
			if strings.Contains(lowLine, "undeclared name:") {
				parts := strings.Split(line, "undeclared name:")
				if len(parts) > 1 {
					symbol = strings.TrimSpace(strings.Split(parts[1], "\n")[0])
				}
				break
			}
		}
		if symbol != "" && !strings.Contains(symbol, " ") {
			return fmt.Sprintf("The test references undefined symbol '%s'. Add the missing declaration or import.", symbol)
		}
	}

	if strings.Contains(lower, "t.error") || strings.Contains(lower, "t.fatal") {
		return "The test contains a hardcoded error. Remove or fix the t.Error/t.Fatal call to make the test pass."
	}

	return "Fix the failing test by reading the relevant file with read_file and applying a SEARCH/REPLACE patch to correct the issue."
}

// escapeJSON escapes a string for safe embedding in JSON.
func escapeJSON(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "\"", "\\\"")
	s = strings.ReplaceAll(s, "\n", "\\n")
	s = strings.ReplaceAll(s, "\t", "\\t")
	return s
}
