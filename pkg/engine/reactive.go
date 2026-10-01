// Package engine is the reactive execution loop that turns a natural
// language prompt into verified, tuned, explainable Go code.
//
// ExecutePipeline runs four stages in order:
//
//  1. Synthesis   — the prompt becomes formatted Go source (pkg/beginner).
//  2. Proving     — the AST is checked for unchecked nil dereferences
//     and missing resource defers (pkg/analysis).
//  3. Auto-tuning — slice capacities are pre-sized and wide value
//     receivers narrowed (pkg/synthesis).
//  4. Visuals     — the call graph becomes a beginner walkthrough plus
//     a real-world analogy (pkg/xray).
//
// Every stage is pure stdlib plus the repo's own packages; the only
// outside world is the context.Context, which is honored between
// stages so a cancelled run stops promptly.
package engine

import (
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"unicode"

	"github.com/golangast/gollemer/pkg/analysis"
	gast "github.com/golangast/gollemer/pkg/ast"
	"github.com/golangast/gollemer/pkg/beginner"
	"github.com/golangast/gollemer/pkg/synthesis"
	"github.com/golangast/gollemer/pkg/xray"
)

// PipelineResult is the outcome of one reactive pipeline run.
type PipelineResult struct {
	// FinalCode is the synthesized, tuned Go source, gofmt-formatted.
	FinalCode string `json:"finalCode"`
	// IsSafetyVerified is true when the symbolic prover found zero
	// unchecked nil dereferences and zero missing resource defers.
	IsSafetyVerified bool `json:"isSafetyVerified"`
	// Analogy is a plain-English real-world comparison for the
	// dominant Go concept in the generated code.
	Analogy string `json:"analogy"`
	// VisualSteps is the step-by-step beginner walkthrough of the
	// program's execution, traced from main.
	VisualSteps []xray.ExecutionStep `json:"visualSteps"`
	// PerformanceDelta summarizes the allocations the auto-tuner
	// removed, e.g. "Reduced allocations from 4 to 0 allocs/op".
	PerformanceDelta string `json:"performanceDelta"`
}

// ExecutePipeline runs the four-stage reactive loop over input.
//
// input is a natural language prompt ("Create a worker pool using
// channels and WaitGroup") or a code refactor instruction the
// synthesizer understands. codeCtx is optional: when non-nil its
// package name labels the generated file for the visual walkthrough.
//
// It returns an error when the prompt is empty, the context is
// cancelled, the synthesizer cannot handle the request, the prover or
// tuner rejects the code, or the final code fails to type-check.
func ExecutePipeline(ctx context.Context, input string, codeCtx *gast.CodebaseContext) (*PipelineResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, errors.New("engine: empty prompt")
	}
	fileName := pipelineFileName(codeCtx)

	// Stage 1 — synthesis: natural language -> formatted Go AST code.
	code, _, err := beginner.GenerateGoFromCommand(input)
	if err != nil {
		return nil, fmt.Errorf("engine stage 1 (synthesis): %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}

	// Stage 2 — symbolic proving: no unchecked nil dereferences,
	// no missing resource defers.
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, fileName, code, 0)
	if err != nil {
		return nil, fmt.Errorf("engine stage 2 (proving): parse: %w", err)
	}
	report, err := analysis.ProveSafetyWithFileSet(file, nil, fset)
	if err != nil {
		return nil, fmt.Errorf("engine stage 2 (proving): %w", err)
	}
	verified := zeroNilAndLeakViolations(report)
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}

	// Stage 3 — auto-tuning: pre-size slices, narrow wide receivers.
	opt := synthesis.OptimizeAllocations(fset, file)
	if opt.OptimizedCode == "" {
		return nil, errors.New("engine stage 3 (auto-tuning): optimizer produced no code")
	}
	final := opt.OptimizedCode
	optSet := token.NewFileSet()
	optFile, err := parser.ParseFile(optSet, fileName, final, 0)
	if err != nil {
		return nil, fmt.Errorf("engine stage 3 (auto-tuning): re-parse: %w", err)
	}
	if err := checkFinalCompiles(optSet, optFile); err != nil {
		return nil, fmt.Errorf("engine stage 3 (auto-tuning): tuned code does not type-check: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("engine: %w", err)
	}

	// Stage 4 — visual synthesis: execution walkthrough + analogy.
	steps, err := xray.TraceExecutionPath(optFile, "main")
	if err != nil {
		return nil, fmt.Errorf("engine stage 4 (visuals): %w", err)
	}
	// TraceExecutionPath normalizes positions by re-rendering, so it
	// labels steps "snippet.go"; restore the pipeline's file label.
	for i := range steps {
		steps[i].FilePath = fileName
	}

	return &PipelineResult{
		FinalCode:        final,
		IsSafetyVerified: verified,
		Analogy:          xray.GetConceptAnalogy(detectPrimitive(final, input)),
		VisualSteps:      steps,
		PerformanceDelta: formatDelta(opt.AllocsBefore, opt.AllocsAfter),
	}, nil
}

// zeroNilAndLeakViolations reports whether the safety report contains
// no unchecked nil dereferences and no missing resource defers.
// Other warning categories (e.g. slice-bounds risk) do not fail the
// proof; they are advisory.
func zeroNilAndLeakViolations(report *analysis.SafetyReport) bool {
	if report == nil {
		return false
	}
	for _, v := range report.Violations {
		switch v.Category {
		case analysis.CategoryCriticalNilDereference, analysis.CategoryResourceLeakWarning:
			return false
		}
	}
	return true
}

// formatDelta renders the allocation summary in allocs/op terms.
func formatDelta(before, after int) string {
	if before == after {
		return fmt.Sprintf("No unbounded allocations found — already at %d allocs/op", after)
	}
	return fmt.Sprintf("Reduced allocations from %d to %d allocs/op", before, after)
}

// pipelineFileName labels the generated file for trace output. It
// prefers the codebase package name when it is a valid Go identifier.
func pipelineFileName(codeCtx *gast.CodebaseContext) string {
	if codeCtx != nil && isGoIdentifier(codeCtx.PackageName) {
		return codeCtx.PackageName + ".go"
	}
	return "pipeline.go"
}

func isGoIdentifier(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		if r != '_' && !unicode.IsLetter(r) && (i == 0 || !unicode.IsDigit(r)) {
			return false
		}
	}
	return true
}

// detectPrimitive picks the dominant Go concept in the generated code
// (falling back to the prompt text) so the analogy matches what the
// user will actually read. The return is always a key of xray's
// analogy table.
func detectPrimitive(code, input string) string {
	haystack := strings.ToLower(code + "\n" + input)
	switch {
	case strings.Contains(haystack, "chan "):
		return "channel"
	case hasGoStatement(code):
		return "goroutine"
	case strings.Contains(haystack, "select {"):
		return "select"
	case strings.Contains(haystack, "sync.waitgroup"):
		return "waitgroup"
	case strings.Contains(haystack, "interface{") || strings.Contains(haystack, "interface {"):
		return "interface"
	case strings.Contains(haystack, "struct{") || strings.Contains(haystack, "struct {"):
		return "struct"
	case strings.Contains(haystack, "sync.mutex"):
		return "mutex"
	case strings.Contains(haystack, "defer "):
		return "defer"
	case strings.Contains(haystack, "append("):
		return "slice"
	case strings.Contains(haystack, "map["):
		return "map"
	case strings.Contains(haystack, "context."):
		return "context"
	default:
		return "function"
	}
}

// hasGoStatement reports whether code launches a goroutine: any
// line whose first token is the go keyword.
func hasGoStatement(code string) bool {
	for _, line := range strings.Split(code, "\n") {
		if strings.HasPrefix(strings.TrimSpace(line), "go ") {
			return true
		}
	}
	return false
}

// checkFinalCompiles type-checks the tuned program as a standalone
// package. It mirrors pkg/beginner's compile gate so the pipeline
// never returns code that does not build.
func checkFinalCompiles(fset *token.FileSet, file *ast.File) error {
	var first error
	cfg := types.Config{
		Importer: importer.Default(),
		Error: func(e error) {
			if first == nil {
				first = e
			}
		},
	}
	_, _ = cfg.Check("pipeline", fset, []*ast.File{file}, nil)
	return first
}

/*
Runnable example: run a prompt through the pipeline and print the
PipelineResult as JSON.

package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/golangast/gollemer/pkg/engine"
)

func main() {
	res, err := engine.ExecutePipeline(context.Background(),
		"Create a worker pool using channels and WaitGroup", nil)
	if err != nil {
		panic(err)
	}
	out, _ := json.MarshalIndent(res, "", "  ")
	fmt.Println(string(out))
	fmt.Println("verified:", res.IsSafetyVerified)
	fmt.Println("delta:", res.PerformanceDelta)
	for _, s := range res.VisualSteps {
		fmt.Printf("step %d: %s\n", s.StepNumber, s.Title)
	}
}
*/
