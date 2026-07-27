package errors

import (
	"fmt"
	"log"
	"os/exec"
	"strings"
)

// ErrorRouter orchestrates the error classification → fixer mapping pipeline.
// It takes raw compiler output, parses it, classifies intents through the
// MoE model, and triggers the appropriate AST-based fixers.
type ErrorRouter struct {
	Classifier *ErrorClassifier
	Verbose    bool
}

// NewErrorRouter creates a new error router with the trained MoE classifier.
func NewErrorRouter(projectRoot string, verbose bool) (*ErrorRouter, error) {
	classifier, err := NewErrorClassifier(projectRoot, verbose)
	if err != nil {
		return nil, fmt.Errorf("create error classifier: %w", err)
	}

	return &ErrorRouter{
		Classifier: classifier,
		Verbose:    verbose,
	}, nil
}

// NewErrorRouterDefault creates a router using automatic project root detection.
func NewErrorRouterDefault(verbose bool) (*ErrorRouter, error) {
	classifier, err := NewErrorClassifierDefault(verbose)
	if err != nil {
		return nil, fmt.Errorf("create error classifier: %w", err)
	}

	return &ErrorRouter{
		Classifier: classifier,
		Verbose:    verbose,
	}, nil
}

// FixResult represents the result of applying a fix for a single error.
type FixResult struct {
	ParsedError *ParsedError
	Fixed       bool
	Message     string
	Err         error
}

// RouterResult represents the overall result of processing compiler output.
type RouterResult struct {
	Errors      []*ParsedError
	FixResults  []FixResult
	FixedCount  int
	TotalCount  int
	BuildPassed bool
}

// ProcessCompilerOutput takes raw compiler output, classifies errors, and applies fixes.
func (r *ErrorRouter) ProcessCompilerOutput(output string) *RouterResult {
	result := &RouterResult{}

	// Parse and classify all errors
	parsedErrors := ParseCompilerOutput(output)
	result.TotalCount = len(parsedErrors)
	result.Errors = parsedErrors

	if r.Verbose {
		log.Printf("Parsed %d errors from compiler output", len(parsedErrors))
	}

	// Classify each error through the MoE model
	for _, pe := range parsedErrors {
		intent, confidence := r.Classifier.ClassifyIntent(pe.Raw)
		pe.Intent = intent
		pe.Confidence = confidence

		if r.Verbose {
			log.Printf("  Error: %s → Intent: %s (confidence: %.2f)", pe.Message, intent, confidence)
		}
	}

	// Apply fixes for classified errors
	for _, pe := range parsedErrors {
		fixer := GetFixer(pe.Intent)
		if fixer == nil {
			result.FixResults = append(result.FixResults, FixResult{
				ParsedError: pe,
				Fixed:       false,
				Message:     fmt.Sprintf("No fixer available for intent %s", pe.Intent),
			})
			continue
		}

		projectRoot := r.Classifier.ProjectRoot
		msg, err := fixer(pe, projectRoot)
		if err != nil {
			result.FixResults = append(result.FixResults, FixResult{
				ParsedError: pe,
				Fixed:       false,
				Err:         err,
				Message:     err.Error(),
			})
			continue
		}

		result.FixedCount++
		result.FixResults = append(result.FixResults, FixResult{
			ParsedError: pe,
			Fixed:       true,
			Message:     msg,
		})
	}

	if r.Verbose {
		log.Printf("Fixed %d/%d errors", result.FixedCount, result.TotalCount)
	}

	return result
}

// RunWithAutoFix runs a build command, processes errors through the router,
// and re-runs until the build passes or max retries are reached.
func (r *ErrorRouter) RunWithAutoFix(pkgTarget string, retries int) error {
	currentPkg := pkgTarget

	for i := 0; i < retries; i++ {
		if r.Verbose {
			log.Printf("🔧 Auto-fix iteration %d/%d for %s", i+1, retries, currentPkg)
		}

		// Run go build and capture output
		buildOutput, err := r.runBuild(currentPkg)
		if err == nil {
			fmt.Printf("✅ Build passed for %s\n", currentPkg)
			return nil
		}

		if r.Verbose {
			log.Printf("❌ Build failed (iteration %d/%d)", i+1, retries)
			log.Printf("   Output: %s", truncateString(string(buildOutput), 500))
		}

		// Process the error output through the MoE router
		result := r.ProcessCompilerOutput(string(buildOutput))

		if result.FixedCount == 0 {
			if i == retries-1 {
				return fmt.Errorf("no fixes could be applied for %s after %d iterations", currentPkg, retries)
			}
			if r.Verbose {
				log.Printf("⚠️  No fixes applied on iteration %d, retrying...", i+1)
			}
			continue
		}

		// Print fix results
		for _, fr := range result.FixResults {
			if fr.Fixed {
				fmt.Printf("  ✅ %s\n", fr.Message)
			} else {
				fmt.Printf("  ❌ %s\n", fr.Message)
			}
		}
	}

	return fmt.Errorf("failed to fix %s within %d retries", currentPkg, retries)
}

// runBuild executes 'go build' on the specified package and returns the output.
func (r *ErrorRouter) runBuild(pkg string) (string, error) {
	cmd := exec.Command("go", "build", pkg)
	output, err := cmd.CombinedOutput()
	outputStr := string(output)

	if err != nil {
		return outputStr, fmt.Errorf("build failed: %s", outputStr)
	}

	return outputStr, nil
}

// truncateString truncates a string to the specified max length.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// formatRouterResult formats the RouterResult for display.
func formatRouterResult(result *RouterResult) string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("📊 Results: Fixed %d/%d errors\n", result.FixedCount, result.TotalCount))
	for i, fr := range result.FixResults {
		status := "❌"
		if fr.Fixed {
			status = "✅"
		}
		sb.WriteString(fmt.Sprintf("  %d. %s %s\n", i+1, status, fr.Message))
	}

	return sb.String()
}

// PrintResult prints the router result in a human-readable format.
func (r *ErrorRouter) PrintResult(result *RouterResult) {
	fmt.Println(formatRouterResult(result))
}
