package chat

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// This file does compositional generation: break a hard problem into simple
// steps the model CAN handle, generate each step, then compose them.
// Harder code via decomposition, not a bigger model.

// Step is one piece of a decomposed problem.
type Step struct {
	Description string // "validate the email"
	Prompt      string // "write a function that validates an email"
	Code        string // generated code for this step
}

// Decompose breaks a hard problem into simple steps.
// If p is not nil, it uses the analyzer's deep knowledge: finds similar
// functions in the project and extracts their steps by analogy.
// Otherwise, falls back to keyword-based decomposition.
func Decompose(problem string, p *analyze.Project) []Step {
	problem = strings.ToLower(problem)
	var steps []Step

	// Find similar functions in the analyzed project to learn the steps.
	// E.g., if problem mentions "handler", find handlers and see what they do.
	if p != nil && (strings.Contains(problem, "handler") || strings.Contains(problem, "http")) {
		// Learn from real handlers: what steps do they take?
		for _, fn := range p.AllFuncs() {
			if isHandlerFunc(fn) {
				// Extract the steps from its call graph.
				for _, calleeID := range fn.Calls {
					if callee := p.FuncByID(calleeID); callee != nil {
						steps = append(steps, Step{
							Description: fmt.Sprintf("%s step", callee.Name),
							Prompt:      fmt.Sprintf("write a function that %s", callee.Summary()),
						})
					}
				}
				if len(steps) > 0 {
					break // Use the first handler as template.
				}
			}
		}
	}

	// Fallback to keyword-based if no analogy found.
	if len(steps) == 0 {
		if strings.Contains(problem, "valid") {
			steps = append(steps, Step{
				Description: "validation step",
				Prompt:      "write a function that validates input",
			})
		}
		if strings.Contains(problem, "create") || strings.Contains(problem, "make") || strings.Contains(problem, "new") {
			steps = append(steps, Step{
				Description: "construction step",
				Prompt:      "write a function that creates a struct",
			})
		}
		if strings.Contains(problem, "http") || strings.Contains(problem, "handler") {
			steps = append(steps, Step{
				Description: "http handler step",
				Prompt:      "write an http handler function",
			})
		}
	}
	if len(steps) == 0 {
		steps = append(steps, Step{Description: "single step", Prompt: problem})
	}
	return steps
}

func isHandlerFunc(fn *analyze.Func) bool {
	// Handler pattern: has http.ResponseWriter param.
	for _, t := range fn.ParamTypes {
		if strings.Contains(t, "ResponseWriter") {
			return true
		}
	}
	return false
}

// ComposeCompositional generates hard code by decomposition.
func ComposeCompositional(projectRoot, problem string) (string, error) {
	steps := Decompose(problem, nil)

	model, err := loadRealModel(projectRoot, GocodeDomain)
	if err != nil {
		return "", fmt.Errorf("no gocode model: %w", err)
	}

	var parts []string
	parts = append(parts, fmt.Sprintf("// Generated from: %s", problem))
	parts = append(parts, "")

	for i, step := range steps {
		code, err := model.Predict(step.Prompt, 100)
		if err != nil {
			return "", fmt.Errorf("step %d failed: %w", i, err)
		}
		// Clean up: remove package clause if present (we'll add one).
		code = strings.TrimPrefix(code, "package main\n\n")
		code = strings.TrimPrefix(code, "package main\n")
		parts = append(parts, fmt.Sprintf("// Step %d: %s", i+1, step.Description))
		parts = append(parts, code)
		parts = append(parts, "")
	}

	// Wrap in package main.
	result := "package main\n\n" + strings.Join(parts, "\n")
	return result, nil
}
