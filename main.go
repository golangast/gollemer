package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
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
	editFlag := flag.String("edit", "", "File edit command to execute")
	intentFlag := flag.String("intent", "", "Natural language intent for AST orchestration (e.g. 'add web server to ft/server.go')")
	corpusFlag := flag.String("corpus", "./ft", "Source directory to build/use as corpus")
	corpusJSONFlag := flag.String("corpus-json", "./corpus.json", "Path to corpus JSON knowledge base")
	memoryJSONFlag := flag.String("memory-json", "./memory.json", "Path to memory JSON experience store")
	targetFlag := flag.String("target", "", "Target file for intent injection")
	repoFlag := flag.String("repo", "", "GitHub repository URL to clone/update (e.g. https://github.com/user/repo)")
	repoDestFlag := flag.String("repo-dest", "", "Local destination directory for the cloned repository")
	mapNLPFlag := flag.String("map-nlp", "", "Natural language instruction to map against the indexed corpus")
	planIntentFlag := flag.String("plan-intent", "", "High-level natural language intent for multi-step execution planner")
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

	if *editFlag != "" {
		query := *editFlag
		mode := "full"
		rootDir, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error getting working directory: %v\n", err)
			os.Exit(1)
		}

		fmt.Fprintf(os.Stderr, "🤖 Gollemer File Editor\n")
		fmt.Fprintf(os.Stderr, "   Root: %s\n", rootDir)
		fmt.Fprintf(os.Stderr, "   Query: %q\n", query)
		fmt.Fprintf(os.Stderr, "   Mode: %s\n\n", mode)

		fmt.Fprintf(os.Stderr, "📚 Indexing workspace symbol graph...\n")
		start := time.Now()

		sg := symbol.NewSymbolGraph(rootDir)
		if err := sg.IndexWorkspace(); err != nil {
			fmt.Fprintf(os.Stderr, "Error indexing workspace: %v\n", err)
			os.Exit(1)
		}

		fmt.Fprintf(os.Stderr, "   %s\n", sg.Summary())
		fmt.Fprintf(os.Stderr, "   Indexing took: %v\n\n", time.Since(start))

		p := planner.NewPlanner(sg, rootDir, &planner.GollemerNativeEngine{})
		plan, err := p.AddFileEditCommand(query)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error creating file edit plan: %v\n", err)
			os.Exit(1)
		}

		fmt.Fprintf(os.Stderr, "## File Edit Plan\n\n")
		fmt.Fprintf(os.Stderr, "Goal: %s\n", plan.Goal)
		fmt.Fprintf(os.Stderr, "Risk Level: %s\n", plan.RiskLevel)
		fmt.Fprintf(os.Stderr, "Estimated Changes: %d\n", plan.EstimatedChanges)
		fmt.Fprintf(os.Stderr, "Created: %s\n\n", plan.CreatedAt.Format(time.RFC3339))

		fmt.Fprintf(os.Stderr, "### Reasoning\n")
		fmt.Fprintf(os.Stderr, "%s\n\n", plan.Reasoning)

		fmt.Fprintf(os.Stderr, "### Phases\n")
		for _, phase := range plan.Phases {
			fmt.Fprintf(os.Stderr, "\n#### %s: %s\n", phase.Name, phase.Description)
			for _, step := range phase.Steps {
				fmt.Fprintf(os.Stderr, "- Step %d: [%s] %s\n", step.ID, step.Action, step.Description)
				if step.Details != "" {
					fmt.Fprintf(os.Stderr, "  Details: %s\n", step.Details)
				}
			}
		}

		exploration := &planner.ExplorationResult{
			Query: query,
		}
		results, verification, err := p.ExecutePlan(plan, exploration)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error executing file edit plan: %v\n", err)
			os.Exit(1)
		}

		fmt.Fprintf(os.Stderr, "\n## File Edit Results\n\n")
		fmt.Fprintf(os.Stderr, "Query: %q\n", query)
		fmt.Fprintf(os.Stderr, "File: %s\n\n", plan.Phases[0].Steps[0].File)

		fmt.Fprintf(os.Stderr, "### Execution Results\n")
		successCount := 0
		for _, r := range results {
			status := "✅"
			if !r.Success {
				status = "❌"
			} else {
				successCount++
			}
			fmt.Fprintf(os.Stderr, "- %s Step %d: %s (%v)\n", status, r.StepID, r.File, r.Duration)
			if r.ErrorMessage != "" {
				fmt.Fprintf(os.Stderr, "  Error: %s\n", r.ErrorMessage)
			}
		}

		fmt.Fprintf(os.Stderr, "\n### Verification\n")
		if verification.Success {
			fmt.Fprintf(os.Stderr, "✅ All checks passed!\n")
		} else {
			fmt.Fprintf(os.Stderr, "❌ Verification failed:\n")
			if verification.GoVet != "" {
				fmt.Fprintf(os.Stderr, "- go vet: %s\n", verification.GoVet)
			}
			if verification.GoBuild != "" {
				fmt.Fprintf(os.Stderr, "- go build: %s\n", verification.GoBuild)
			}
			if verification.GoTest != "" {
				fmt.Fprintf(os.Stderr, "- go test: %s\n", verification.GoTest)
			}
		}

		fmt.Fprintf(os.Stderr, "\nSummary: %d/%d steps succeeded\n", successCount, len(results))
		os.Exit(0)
	}

	// --- Remote Repository Cloner ---
	if *repoFlag != "" {
		dest := *repoDestFlag
		if dest == "" {
			// Default destination: ./repos/<last segment of URL>
			segments := strings.Split(strings.TrimSuffix(*repoFlag, "/"), "/")
			dest = filepath.Join("repos", segments[len(segments)-1])
		}
		fmt.Fprintf(os.Stderr, "📥 Cloning / updating repository %s → %s\n", *repoFlag, dest)
		if err := CloneOrUpdateRepo(*repoFlag, dest); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Clone/update failed: %v\n", err)
			os.Exit(1)
		}

		// Automatically build corpus from the freshly cloned repo.
		corpusOut := *corpusJSONFlag
		fmt.Fprintf(os.Stderr, "📚 Indexing cloned repo into %s...\n", corpusOut)
		if err := BuildCodeCorpus(dest, corpusOut); err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  Corpus build warning: %v\n", err)
		}
		fmt.Fprintf(os.Stderr, "✅ Repository indexed at %s\n", corpusOut)
		return
	}

	// --- NLP→Codebase Concept Mapper ---
	if *mapNLPFlag != "" {
		fmt.Fprintf(os.Stderr, "🗺️  Mapping NLP concept: %q\n", *mapNLPFlag)
		patternID, err := MapNLPToCodebase(*mapNLPFlag, *corpusJSONFlag, *memoryJSONFlag)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ NLP mapping failed: %v\n", err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "✅ Best match pattern ID: %s (saved to memory)\n", patternID)
		return
	}

	// --- Multi-Step Execution Planner ---
	if *planIntentFlag != "" {
		fmt.Fprintf(os.Stderr, "🗺️  Gollemer Multi-Step Execution Planner\n")
		fmt.Fprintf(os.Stderr, "   Goal: %q\n", *planIntentFlag)

		// Ensure corpus exists
		fmt.Fprintf(os.Stderr, "📚 Indexing corpus from %s...\n", *corpusFlag)
		if err := BuildCodeCorpus(*corpusFlag, *corpusJSONFlag); err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  Corpus build warning: %v\n", err)
		}

		targetDir := ""
		if *targetFlag != "" {
			targetDir = filepath.Dir(*targetFlag)
		}

		plan, err := PlanHighLevelIntent(*planIntentFlag, targetDir, *corpusJSONFlag, *memoryJSONFlag)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ Planning failed: %v\n", err)
			os.Exit(1)
		}

		fmt.Fprintf(os.Stderr, "📋 Execution Plan Generated (%d steps):\n", len(plan.Steps))
		for _, step := range plan.Steps {
			fmt.Fprintf(os.Stderr, "  %d. %s\n", step.Index+1, step.Description)
		}

		fmt.Fprintf(os.Stderr, "\n🚀 Executing Plan...\n")
		results := RunExecutionPlan(plan, *corpusJSONFlag, *memoryJSONFlag)

		successCount := 0
		for _, res := range results {
			if res.Success {
				successCount++
			}
		}

		fmt.Fprintf(os.Stderr, "\n✅ Plan Execution Summary: %d/%d steps succeeded.\n", successCount, len(plan.Steps))
		if successCount != len(plan.Steps) {
			os.Exit(1)
		}
		return
	}

	// --- Intent Router (AST Orchestrator) ---
	if *intentFlag != "" {
		instruction := *intentFlag
		corpusJSON := *corpusJSONFlag
		memoryJSON := *memoryJSONFlag
		targetFile := *targetFlag

		fmt.Fprintf(os.Stderr, "🧠 Gollemer AST Orchestrator\n")
		fmt.Fprintf(os.Stderr, "   Intent: %q\n", instruction)

		// 1. Build/refresh corpus from the corpus source directory
		fmt.Fprintf(os.Stderr, "📚 Indexing corpus from %s...\n", *corpusFlag)
		if err := BuildCodeCorpus(*corpusFlag, corpusJSON); err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  Corpus build warning: %v\n", err)
		}

		// 2. Route: web/http server intent → use blueprint target if no explicit target given
		instructionLower := strings.ToLower(instruction)
		if targetFile == "" {
			if strings.Contains(instructionLower, "web server") ||
				strings.Contains(instructionLower, "http server") ||
				strings.Contains(instructionLower, "webserver") {
				targetFile = "./ft/generated_server.go"
				fmt.Fprintf(os.Stderr, "🌐 Web server intent detected — targeting %s\n", targetFile)
			} else {
				fmt.Fprintf(os.Stderr, "❌ No target file specified. Use -target=<file>\n")
				os.Exit(1)
			}
		}

		// 3. Seed target file if it does not exist
		if _, err := os.Stat(targetFile); os.IsNotExist(err) {
			pkg := "ft"
			if err := os.MkdirAll(strings.Split(targetFile, "/")[0], 0755); err == nil {
				os.WriteFile(targetFile, []byte("package "+pkg+"\n"), 0644)
			}
		}

		// 3.5 Check for database generation intent
		if structName, parsedTarget, err := ParseDatabaseIntent(instruction); err == nil {
			fmt.Fprintf(os.Stderr, "💾 Database generation intent detected for struct %q in %q\n", structName, parsedTarget)
			targetFile = parsedTarget

			model, err := InspectStructAST(targetFile, structName)
			if err != nil {
				fmt.Fprintf(os.Stderr, "❌ Failed to inspect struct: %v\n", err)
				os.Exit(1)
			}

			dbCode := GenerateDatabaseCode(model)

			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, "", "package dummy\n"+dbCode, 0)
			if err != nil || len(f.Decls) == 0 {
				fmt.Fprintf(os.Stderr, "❌ Failed to parse generated code: %v\n", err)
				os.Exit(1)
			}

			funcDecl, ok := f.Decls[0].(*ast.FuncDecl)
			if !ok {
				fmt.Fprintf(os.Stderr, "❌ Expected FuncDecl in generated code\n")
				os.Exit(1)
			}

			fmt.Fprintf(os.Stderr, "🔧 Injecting generated database code...\n")
			if err := InjectAndValidate(targetFile, structName, funcDecl); err != nil {
				fmt.Fprintf(os.Stderr, "❌ Injection failed: %v\n", err)
				os.Exit(1)
			}

			fmt.Fprintf(os.Stderr, "✅ Done — Database code generated successfully\n")
			return
		}

		// 4. Run the 4-stage semantic orchestrator
		fmt.Fprintf(os.Stderr, "🔧 Running 4-stage semantic orchestrator...\n")
		if err := OrchestrateAndLearn(targetFile, instruction, corpusJSON, memoryJSON); err != nil {
			fmt.Fprintf(os.Stderr, "❌ Orchestration failed: %v\n", err)
			os.Exit(1)
		}

		// 5. Run AutoFixPipeline for final import/syntax cleanup
		fmt.Fprintf(os.Stderr, "🩹 Running AutoFixPipeline...\n")
		if err := AutoFixPipeline(targetFile); err != nil {
			fmt.Fprintf(os.Stderr, "⚠️  AutoFix warning: %v\n", err)
		}

		fmt.Fprintf(os.Stderr, "✅ Done — %s updated successfully\n", targetFile)
		os.Exit(0)
	}

	if query == "" {
		fmt.Fprintf(os.Stderr, "Usage: gollemer [-mode explore|plan|full] [-json] [-symbol <name>] [-goal <text>] <query>\n")
		fmt.Fprintf(os.Stderr, "       gollemer -intent=\"<instruction>\" [-target=<file>] [-corpus=./ft] [-corpus-json=./corpus.json] [-memory-json=./memory.json]\n")
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

	p := planner.NewPlanner(sg, rootDir, &planner.GollemerNativeEngine{})
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

func test(a, b int) int {
	return a + b
}
