package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/planner"
	"github.com/golangast/gollemer/internal/ai/symbol"
)

func main() {
	mode := flag.String("mode", "plan", "Pipeline mode: explore, plan, full")
	outputJSON := flag.Bool("json", false, "Output results as JSON")
	symbolFlag := flag.String("symbol", "", "Target symbol for exploration")
	goalFlag := flag.String("goal", "", "Target goal for plan/full modes")
	flag.Parse()

	query := ""
	args := flag.Args()
	if len(args) >= 1 {
		query = args[0]
	} else if *symbolFlag != "" {
		query = *symbolFlag
	} else if *goalFlag != "" {
		query = *goalFlag
	}

	if query == "" {
		fmt.Fprintf(os.Stderr, "Usage: smart_agent [-mode explore|plan|full] [-json] [-symbol <name>] [-goal <text>] <query>\n")
		os.Exit(1)
	}

	rootDir, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting working directory: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "🤖 Gollemer Smart Agent\n")
	fmt.Fprintf(os.Stderr, "   Root: %s\n", rootDir)
	fmt.Fprintf(os.Stderr, "   Query: %q\n", query)
	fmt.Fprintf(os.Stderr, "   Mode: %s\n\n", *mode)

	fmt.Fprintf(os.Stderr, "📚 Indexing workspace symbol graph...\n")
	start := time.Now()

	sg := symbol.NewSymbolGraph(rootDir)
	if err := sg.IndexWorkspace(); err != nil {
		fmt.Fprintf(os.Stderr, "Error indexing workspace: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "   %s\n", sg.Summary())
	fmt.Fprintf(os.Stderr, "   Indexing took: %v\n\n", time.Since(start))

	p := planner.NewPlanner(sg, rootDir, nil)

	switch *mode {
	case "explore":
		exploration, err := p.Explore(query)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Exploration failed: %v\n", err)
			os.Exit(1)
		}

		if *outputJSON {
			data, _ := planner.ExportExploration(exploration)
			fmt.Println(string(data))
		} else {
			printExploration(exploration)
		}

	case "plan":
		exploration, err := p.Explore(query)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Exploration failed: %v\n", err)
			os.Exit(1)
		}

		plan, err := p.Plan(query, exploration)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Planning failed: %v\n", err)
			os.Exit(1)
		}

		if *outputJSON {
			data, _ := planner.ExportPlan(plan)
			fmt.Println(string(data))
		} else {
			printPlan(plan)
		}

	case "full":
		exploration, plan, results, verification, err := p.ThreePhasePipeline(query)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Pipeline failed: %v\n", err)
			os.Exit(1)
		}

		if *outputJSON {
			output := map[string]any{
				"exploration":  exploration,
				"plan":         plan,
				"results":      results,
				"verification": verification,
			}
			data, _ := json.MarshalIndent(output, "", "  ")
			fmt.Println(string(data))
		} else {
			printExploration(exploration)
			printPlan(plan)
			printResults(results, verification)
		}

	default:
		fmt.Fprintf(os.Stderr, "Unknown mode: %s\n", *mode)
		os.Exit(1)
	}
}

func printExploration(e *planner.ExplorationResult) {
	fmt.Println("=" + strings.Repeat("=", 60))
	fmt.Println("🔍 PHASE 1: EXPLORATION & MAPPING")
	fmt.Println("=" + strings.Repeat("=", 60))
	fmt.Printf("Query: %q\n", e.Query)
	fmt.Printf("Symbols found: %d\n", len(e.FoundSymbols))
	fmt.Printf("Affected files: %d\n", len(e.AffectedFiles))
	fmt.Printf("References found: %d\n", len(e.References))
	fmt.Printf("Duration: %v\n\n", e.Duration)

	if len(e.FoundSymbols) > 0 {
		fmt.Println("📌 Found Symbols:")
		for _, sym := range e.FoundSymbols {
			fmt.Printf("  - %s (%s) in %s:%d\n", sym.Name, sym.Kind, sym.File, sym.Line)
			if sym.Signature != "" {
				fmt.Printf("    Signature: %s\n", sym.Signature)
			}
		}
		fmt.Println()
	}

	if len(e.AffectedFiles) > 0 {
		fmt.Println("📁 Affected Files:")
		for _, f := range e.AffectedFiles {
			priority := "🟢 LOW"
			if f.Priority == 2 {
				priority = "🔴 HIGH"
			} else if f.Priority == 1 {
				priority = "🟡 MED"
			}
			fmt.Printf("  %s: %s\n", priority, f.Path)
			fmt.Printf("    Reason: %s\n", f.Reason)
			fmt.Printf("    Symbols: %s\n", strings.Join(f.Symbols, ", "))
		}
		fmt.Println()
	}

	if len(e.CallGraph) > 0 {
		fmt.Println("🔗 Call Graph:")
		for caller, callees := range e.CallGraph {
			fmt.Printf("  %s -> %s\n", caller, strings.Join(callees, ", "))
		}
		fmt.Println()
	}
}

func printPlan(p *planner.ExecutionPlan) {
	fmt.Println("=" + strings.Repeat("=", 60))
	fmt.Println("📋 PHASE 2: EXECUTION PLAN")
	fmt.Println("=" + strings.Repeat("=", 60))
	fmt.Printf("Goal: %s\n", p.Goal)
	fmt.Printf("Risk Level: %s\n", p.RiskLevel)
	fmt.Printf("Estimated Changes: %d\n", p.EstimatedChanges)
	fmt.Printf("Created: %s\n\n", p.CreatedAt.Format(time.RFC3339))

	fmt.Println("📝 Reasoning:")
	fmt.Println(p.Reasoning)
	fmt.Println()

	fmt.Println("📋 Execution Phases:")
	for _, phase := range p.Phases {
		fmt.Printf("\n  Phase: %s\n", phase.Name)
		fmt.Printf("  Description: %s\n", phase.Description)
		for _, step := range phase.Steps {
			fmt.Printf("    Step %d: [%s] %s\n", step.ID, step.Action, step.Description)
			if step.Details != "" {
				fmt.Printf("      Details: %s\n", step.Details)
			}
		}
	}
	fmt.Println()
}

func printResults(results []planner.PatchResult, verification *planner.VerificationResult) {
	fmt.Println("=" + strings.Repeat("=", 60))
	fmt.Println("🔧 PHASE 3: PATCH EXECUTION & VERIFICATION")
	fmt.Println("=" + strings.Repeat("=", 60))

	successCount := 0
	for _, r := range results {
		status := "✅"
		if !r.Success {
			status = "❌"
		} else {
			successCount++
		}
		fmt.Printf("  %s Step %d: %s (%v)\n", status, r.StepID, r.File, r.Duration)
		if r.ErrorMessage != "" {
			fmt.Printf("    Error: %s\n", r.ErrorMessage)
		}
		if r.Output != "" {
			fmt.Printf("    Output: %s\n", r.Output)
		}
	}
	fmt.Printf("\n  Summary: %d/%d steps succeeded\n\n", successCount, len(results))

	fmt.Println("✅ Verification Results:")
	if verification.Success {
		fmt.Println("  All checks passed!")
	} else {
		if verification.GoFmt != "" {
			fmt.Printf("  ❌ gofmt: %s\n", verification.GoFmt)
		}
		if verification.GoVet != "" {
			fmt.Printf("  ❌ govet: %s\n", verification.GoVet)
		}
		if verification.GoBuild != "" {
			fmt.Printf("  ❌ gobuild: %s\n", verification.GoBuild)
		}
		if verification.GoTest != "" {
			fmt.Printf("  ❌ gotest: %s\n", verification.GoTest)
		}
	}
	fmt.Println()
}
