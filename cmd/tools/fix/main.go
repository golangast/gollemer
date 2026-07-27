// Package main implements the 'gollemer fix' CLI subcommand.
// It runs the LLM-driven auto-fix loop using Gollemer's own local MoE model.
// No external APIs are needed — everything runs locally.
//
// The loop:
//  1. Runs `go test -v <package>`
//  2. If tests fail, sends the error log to Gollemer's local MoE model
//  3. Executes the model's tool calls (read_file, apply_patch)
//  4. Repeats until tests pass or max iterations are reached
//
// Usage:
//
//	gollemer fix ./cmd/tools/multi_orchestrator
//	gollemer fix -retries=3 ./pkg/mypackage
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/llm"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/agent"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func main() {
	maxRetries := flag.Int("retries", 5, "Maximum auto-fix iterations")
	verbose := flag.Bool("verbose", false, "Print detailed output")
	flag.Parse()

	pkgTarget := flag.Arg(0)
	if pkgTarget == "" {
		fmt.Fprintf(os.Stderr, "Usage: gollemer fix [flags] <package>\n\n")
		fmt.Fprintf(os.Stderr, "Flags:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  gollemer fix ./cmd/tools/multi_orchestrator\n")
		fmt.Fprintf(os.Stderr, "  gollemer fix -retries=3 ./pkg/mypackage\n")
		fmt.Fprintf(os.Stderr, "  gollemer fix -verbose .\n")
		os.Exit(1)
	}

	workingDir, err := os.Getwd()
	if err != nil {
		log.Fatalf("Failed to get working directory: %v", err)
	}

	fmt.Printf("🔧 Gollemer Auto-Fix\n")
	fmt.Printf("   Package:     %s\n", pkgTarget)
	fmt.Printf("   Max retries: %d\n", *maxRetries)
	fmt.Printf("   Working dir: %s\n", workingDir)
	fmt.Println()

	// Initialize Gollemer's local inference engine
	inference, err := newLocalInference(workingDir)
	if err != nil {
		fmt.Printf("⚠️  Could not initialize local model: %v\n", err)
		fmt.Printf("   Running in test-only mode (no auto-fixes)\n")
	}

	// Create the LLM callback using Gollemer's local MoE model
	callLLM := createLocalLLMCallback(inference, *verbose)

	// Run the auto-fix loop
	if err := agent.RunAutoFix(pkgTarget, workingDir, callLLM); err != nil {
		fmt.Fprintf(os.Stderr, "\n❌ Auto-fix failed: %v\n", err)
		os.Exit(1)
	}
}

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
// Uses the same approach as GenerateSocialResponse in client.go:
// formats as ChatML, tokenizes with BPE, then runs PredictNextToken in a loop.
func (li *LocalInference) Generate(prompt string) (string, error) {
	if li.BPETok == nil || li.SocialModel == nil {
		return generateFallbackPatch(prompt), nil
	}

	// Format as ChatML for the local model (same format as GenerateSocialResponse)
	chatmlPrompt := fmt.Sprintf("<|im_start|>system\nYou are Gollemer, an expert Go AI assistant. You fix failing tests by applying SEARCH/REPLACE patches.<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt)

	// Tokenize with BPE
	tokenIDs := li.BPETok.Encode(chatmlPrompt)
	if len(tokenIDs) == 0 {
		return generateFallbackPatch(prompt), nil
	}

	// Generate tokens using the MoE model (same loop as GenerateSocialResponse)
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

		// Stop on ChatML end token, EOS, or zero
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
		// Generate response from local model
		response, err := inference.Generate(prompt)
		if err != nil {
			return nil, "", fmt.Errorf("local model error: %w", err)
		}

		if verbose {
			fmt.Printf("\n🧠 Local Model Response:\n%s\n", response)
		}

		// Parse the response for SEARCH/REPLACE patches
		toolCalls := parsePatchFromResponse(response)
		if len(toolCalls) > 0 {
			return toolCalls, response, nil
		}

		// Fallback: generate a basic fix suggestion
		fix := generateFallbackPatch(prompt)
		toolCalls = parsePatchFromResponse(fix)
		return toolCalls, fix, nil
	}
}

// parsePatchFromResponse extracts SEARCH/REPLACE patches from the model's response text.
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

// generateFallbackPatch creates a simple fix suggestion based on common test failure patterns.
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
