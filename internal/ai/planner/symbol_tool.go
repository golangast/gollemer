package planner

import (
	"fmt"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/agent"
	"github.com/golangast/gollemer/internal/ai/symbol"
)

// ─── Symbol Search Tool ─────────────────────────────────────────────────────────

// SymbolSearchTool wraps the symbol graph for use as an agent tool.
type SymbolSearchTool struct {
	symbolGraph *symbol.SymbolGraph
}

// NewSymbolSearchTool creates a new symbol search tool.
func NewSymbolSearchTool(sg *symbol.SymbolGraph) *SymbolSearchTool {
	return &SymbolSearchTool{symbolGraph: sg}
}

func (t *SymbolSearchTool) Name() string {
	return "search_symbols"
}

func (t *SymbolSearchTool) Description() string {
	return "Searches the workspace symbol graph for definitions, references, and callers of a symbol"
}

func (t *SymbolSearchTool) Schema() agent.ToolSchema {
	return agent.ToolSchema{
		Parameters: map[string]agent.ParameterDef{
			"query": {
				Type:        "string",
				Description: "Symbol name or search query to find in the codebase",
			},
			"mode": {
				Type:        "string",
				Description: "Search mode: 'definitions', 'references', 'callers', 'implementations', or 'all'",
				Enum:        []string{"definitions", "references", "callers", "implementations", "all"},
				Default:     "all",
			},
		},
		Required: []string{"query"},
	}
}

func (t *SymbolSearchTool) Execute(args map[string]any) (agent.ToolResult, error) {
	query, ok := args["query"].(string)
	if !ok || query == "" {
		return agent.ToolResult{
			Success: false,
			Error:   fmt.Errorf("query is required"),
		}, nil
	}

	mode, _ := args["mode"].(string)
	if mode == "" {
		mode = "all"
	}

	var output strings.Builder
	output.WriteString(fmt.Sprintf("## Symbol Search Results for %q\n\n", query))

	// Always collect definitions
	defs := t.symbolGraph.FindDefinitions(query)
	output.WriteString(fmt.Sprintf("### Definitions (%d found)\n", len(defs)))
	for _, sym := range defs {
		output.WriteString(fmt.Sprintf("- %s (%s) in %s:%d\n", sym.Name, sym.Kind, sym.File, sym.Line))
		if sym.Signature != "" {
			output.WriteString(fmt.Sprintf("  Signature: %s\n", sym.Signature))
		}
		if sym.DocComment != "" {
			output.WriteString(fmt.Sprintf("  Doc: %s\n", strings.TrimSpace(sym.DocComment)))
		}
	}
	output.WriteString("\n")

	if mode != "definitions" {
		// References
		refs := t.symbolGraph.FindReferences(query)
		output.WriteString(fmt.Sprintf("### References (%d found)\n", len(refs)))
		for _, ref := range refs {
			output.WriteString(fmt.Sprintf("- %s:%d - %s\n", ref.File, ref.Line, ref.Context))
		}
		output.WriteString("\n")

		// Callers
		callers := t.symbolGraph.FindCallers(query)
		output.WriteString(fmt.Sprintf("### Callers (%d found)\n", len(callers)))
		for _, caller := range callers {
			output.WriteString(fmt.Sprintf("- %s in %s:%d\n", caller.Name, caller.File, caller.Line))
		}
		output.WriteString("\n")

		// Implementations
		impls := t.symbolGraph.FindImplementations(query)
		output.WriteString(fmt.Sprintf("### Implementations (%d found)\n", len(impls)))
		for _, impl := range impls {
			output.WriteString(fmt.Sprintf("- %s in %s:%d\n", impl.Name, impl.File, impl.Line))
		}
		output.WriteString("\n")
	}

	// Also show call graph if available
	callGraph := t.symbolGraph.TraceCallGraph(query, 3)
	if len(callGraph) > 0 {
		output.WriteString("### Call Graph\n")
		for caller, callees := range callGraph {
			output.WriteString(fmt.Sprintf("- %s calls: %s\n", caller, strings.Join(callees, ", ")))
		}
	}

	return agent.ToolResult{
		Success: true,
		Output:  output.String(),
		Metadata: map[string]any{
			"query": query,
			"mode":  mode,
		},
	}, nil
}

// ─── Plan Execution Tool ────────────────────────────────────────────────────────

// PlanExecutionTool wraps the planner for use as an agent tool.
type PlanExecutionTool struct {
	planner *Planner
}

// NewPlanExecutionTool creates a new plan execution tool.
func NewPlanExecutionTool(p *Planner) *PlanExecutionTool {
	return &PlanExecutionTool{planner: p}
}

func (t *PlanExecutionTool) Name() string {
	return "execute_plan"
}

func (t *PlanExecutionTool) Description() string {
	return "Executes the three-phase pipeline: explore symbols, create plan, and verify changes"
}

func (t *PlanExecutionTool) Schema() agent.ToolSchema {
	return agent.ToolSchema{
		Parameters: map[string]agent.ParameterDef{
			"goal": {
				Type:        "string",
				Description: "The goal or change request to analyze and plan",
			},
			"mode": {
				Type:        "string",
				Description: "Pipeline mode: 'explore' (phase 1 only), 'plan' (phases 1+2), 'full' (all 3 phases)",
				Enum:        []string{"explore", "plan", "full"},
				Default:     "plan",
			},
		},
		Required: []string{"goal"},
	}
}

func (t *PlanExecutionTool) Execute(args map[string]any) (agent.ToolResult, error) {
	goal, ok := args["goal"].(string)
	if !ok || goal == "" {
		return agent.ToolResult{
			Success: false,
			Error:   fmt.Errorf("goal is required"),
		}, nil
	}

	mode, _ := args["mode"].(string)
	if mode == "" {
		mode = "plan"
	}

	var output strings.Builder

	switch mode {
	case "explore":
		exploration, err := t.planner.Explore(goal)
		if err != nil {
			return agent.ToolResult{
				Success: false,
				Error:   fmt.Errorf("exploration failed: %w", err),
			}, nil
		}
		output.WriteString(fmt.Sprintf("## Exploration Results\n\n"))
		output.WriteString(fmt.Sprintf("Query: %q\n", goal))
		output.WriteString(fmt.Sprintf("Symbols found: %d\n", len(exploration.FoundSymbols)))
		output.WriteString(fmt.Sprintf("Affected files: %d\n", len(exploration.AffectedFiles)))
		output.WriteString(fmt.Sprintf("References found: %d\n", len(exploration.References)))
		output.WriteString(fmt.Sprintf("Duration: %v\n\n", exploration.Duration))

		output.WriteString("### Affected Files\n")
		for _, f := range exploration.AffectedFiles {
			output.WriteString(fmt.Sprintf("- [%s] %s (%s)\n", priorityLabel(f.Priority), f.Path, f.Reason))
		}

		if len(exploration.CallGraph) > 0 {
			output.WriteString("\n### Call Graph\n")
			for caller, callees := range exploration.CallGraph {
				output.WriteString(fmt.Sprintf("- %s -> %s\n", caller, strings.Join(callees, ", ")))
			}
		}

	case "plan":
		exploration, err := t.planner.Explore(goal)
		if err != nil {
			return agent.ToolResult{
				Success: false,
				Error:   fmt.Errorf("exploration failed: %w", err),
			}, nil
		}

		plan, err := t.planner.Plan(goal, exploration)
		if err != nil {
			return agent.ToolResult{
				Success: false,
				Error:   fmt.Errorf("planning failed: %w", err),
			}, nil
		}

		output.WriteString(fmt.Sprintf("## Execution Plan\n\n"))
		output.WriteString(fmt.Sprintf("Goal: %s\n", plan.Goal))
		output.WriteString(fmt.Sprintf("Risk Level: %s\n", plan.RiskLevel))
		output.WriteString(fmt.Sprintf("Estimated Changes: %d\n", plan.EstimatedChanges))
		output.WriteString(fmt.Sprintf("Created: %s\n\n", plan.CreatedAt.Format(time.RFC3339)))

		output.WriteString("### Reasoning\n")
		output.WriteString(plan.Reasoning)
		output.WriteString("\n")

		output.WriteString("### Phases\n")
		for _, phase := range plan.Phases {
			output.WriteString(fmt.Sprintf("\n#### %s: %s\n", phase.Name, phase.Description))
			for _, step := range phase.Steps {
				output.WriteString(fmt.Sprintf("- Step %d: [%s] %s\n", step.ID, step.Action, step.Description))
				if step.Details != "" {
					output.WriteString(fmt.Sprintf("  Details: %s\n", step.Details))
				}
			}
		}

	case "full":
		exploration, plan, results, verification, err := t.planner.ThreePhasePipeline(goal)
		if err != nil {
			return agent.ToolResult{
				Success: false,
				Error:   fmt.Errorf("pipeline failed: %w", err),
			}, nil
		}

		output.WriteString(fmt.Sprintf("## Pipeline Results\n\n"))
		output.WriteString(fmt.Sprintf("### Phase 1: Exploration\n"))
		output.WriteString(fmt.Sprintf("Symbols found: %d, Files affected: %d\n", len(exploration.FoundSymbols), len(exploration.AffectedFiles)))

		output.WriteString(fmt.Sprintf("\n### Phase 2: Plan\n"))
		output.WriteString(fmt.Sprintf("Risk: %s, Changes: %d\n", plan.RiskLevel, plan.EstimatedChanges))

		output.WriteString(fmt.Sprintf("\n### Phase 3: Execution\n"))
		successCount := 0
		for _, r := range results {
			status := "✅"
			if !r.Success {
				status = "❌"
			} else {
				successCount++
			}
			output.WriteString(fmt.Sprintf("- %s Step %d: %s (%v)\n", status, r.StepID, r.File, r.Duration))
			if r.ErrorMessage != "" {
				output.WriteString(fmt.Sprintf("  Error: %s\n", r.ErrorMessage))
			}
		}

		output.WriteString(fmt.Sprintf("\n### Verification\n"))
		if verification.Success {
			output.WriteString("✅ All checks passed!\n")
		} else {
			output.WriteString("❌ Verification failed:\n")
			if verification.GoVet != "" {
				output.WriteString(fmt.Sprintf("- go vet: %s\n", verification.GoVet))
			}
			if verification.GoBuild != "" {
				output.WriteString(fmt.Sprintf("- go build: %s\n", verification.GoBuild))
			}
			if verification.GoTest != "" {
				output.WriteString(fmt.Sprintf("- go test: %s\n", verification.GoTest))
			}
		}

		output.WriteString(fmt.Sprintf("\nSummary: %d/%d steps succeeded\n", successCount, len(results)))
	}

	return agent.ToolResult{
		Success: true,
		Output:  output.String(),
		Metadata: map[string]any{
			"goal": goal,
			"mode": mode,
		},
	}, nil
}

func priorityLabel(p int) string {
	switch p {
	case 2:
		return "HIGH"
	case 1:
		return "MED"
	default:
		return "LOW"
	}
}
