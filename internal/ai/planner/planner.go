// Package planner implements a three-phase step-by-step execution pipeline:
//
// Phase 1: Exploration & Mapping - uses the symbol graph to find affected files
// Phase 2: Architectural Execution Plan - drafts a plain-text plan
// Phase 3: Targeted Patch Execution - emits surgical patches per plan
//
// This enables chain-of-thought reasoning before any code changes are made.
package planner

import (
	"context"
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/knowledge"
	"github.com/golangast/gollemer/internal/ai/symbol"
	"github.com/golangast/gollemer/internal/ai/training"
)

// ─── Phase 1: Exploration & Mapping ─────────────────────────────────────────────

// ExplorationResult captures the output of Phase 1.
type ExplorationResult struct {
	Query            string              `json:"query"`
	FoundSymbols     []*symbol.Symbol    `json:"found_symbols"`
	AffectedFiles    []AffectedFile      `json:"affected_files"`
	CallGraph        map[string][]string `json:"call_graph"`
	References       []*symbol.Reference `json:"references"`
	FileDependencies map[string][]string `json:"file_dependencies"`
	Duration         time.Duration       `json:"duration"`
}

// AffectedFile describes a file that needs modification.
type AffectedFile struct {
	Path       string   `json:"path"`
	Package    string   `json:"package"`
	Reason     string   `json:"reason"`
	Symbols    []string `json:"symbols"`
	Priority   int      `json:"priority"` // 0=low, 1=medium, 2=high
	IsTestFile bool     `json:"is_test_file"`
}

// ─── Phase 2: Execution Plan ────────────────────────────────────────────────────

// ExecutionPlan is the output of Phase 2.
type ExecutionPlan struct {
	Goal             string      `json:"goal"`
	Phases           []PlanPhase `json:"phases"`
	Reasoning        string      `json:"reasoning"`
	RiskLevel        string      `json:"risk_level"` // "low", "medium", "high"
	EstimatedChanges int         `json:"estimated_changes"`
	CreatedAt        time.Time   `json:"created_at"`
}

// PlanPhase is a logical grouping of steps.
type PlanPhase struct {
	Name        string     `json:"name"`
	Description string     `json:"description"`
	Steps       []PlanStep `json:"steps"`
}

// PlanStep is a single surgical operation.
type PlanStep struct {
	ID           int    `json:"id"`
	Action       string `json:"action"` // "modify", "create", "delete", "refactor"
	File         string `json:"file"`
	Description  string `json:"description"`
	Details      string `json:"details"`
	Dependencies []int  `json:"dependencies"`  // Step IDs this depends on
	DryRunCheck  string `json:"dry_run_check"` // What to verify before applying
}

// ─── Phase 3: Patch Execution ───────────────────────────────────────────────────

// PatchResult captures the outcome of applying a patch.
type PatchResult struct {
	StepID       int           `json:"step_id"`
	File         string        `json:"file"`
	Success      bool          `json:"success"`
	ErrorMessage string        `json:"error_message,omitempty"`
	Output       string        `json:"output,omitempty"`
	Duration     time.Duration `json:"duration"`
}

// VerificationResult captures the outcome of the verification loop.
type VerificationResult struct {
	Success        bool     `json:"success"`
	GoFmt          string   `json:"gofmt,omitempty"`
	GoVet          string   `json:"govet,omitempty"`
	GoBuild        string   `json:"gobuild,omitempty"`
	GoTest         string   `json:"gotest,omitempty"`
	FailedSteps    []int    `json:"failed_steps,omitempty"`
	CompilerErrors []string `json:"compiler_errors,omitempty"`
}

// StructuredPlan is a JSON-serializable plan step used for LLM-generated plans.
type StructuredPlan struct {
	TargetFile  string `json:"target_file"`
	Action      string `json:"action"`
	Description string `json:"description"`
	CodeSnippet string `json:"code_snippet"`
}

// ─── LLM Engine Interface ──────────────────────────────────

// LLMEngine abstracts Gollemer text generation capability.
type LLMEngine interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// ─── Planner ────────────────────────────────────────────────────────────────────

// Planner orchestrates the three-phase pipeline.
type Planner struct {
	symbolGraph    *symbol.SymbolGraph
	rootDir        string
	workDir        string // Temp directory for dry-run patches
	maxRetries     int
	llm            LLMEngine       // Native Gollemer model instance
	conceptMatcher *ConceptMatcher // Maps query terms to Go idiom patterns
}

// NewPlanner creates a new planner with the given symbol graph.
func NewPlanner(sg *symbol.SymbolGraph, rootDir string, engine LLMEngine) *Planner {
	return &Planner{
		symbolGraph:    sg,
		rootDir:        rootDir,
		workDir:        filepath.Join(rootDir, ".gollemer_work"),
		maxRetries:     3,
		llm:            engine,
		conceptMatcher: NewConceptMatcher(),
	}
}

// ─── Phase 1: Exploration ───────────────────────────────────────────────────────

// Explore performs Phase 1: search for symbols, find references, trace call graph.
// Accepts both plain symbol names and natural language queries (e.g. "Add caching layer to SymbolGraph").
// The query is tokenized into candidate words, each of which is searched in the symbol graph.
func (p *Planner) Explore(query string) (*ExplorationResult, error) {

	start := time.Now()
	result := &ExplorationResult{
		Query:            query,
		FoundSymbols:     make([]*symbol.Symbol, 0),
		AffectedFiles:    make([]AffectedFile, 0),
		CallGraph:        make(map[string][]string),
		References:       make([]*symbol.Reference, 0),
		FileDependencies: make(map[string][]string),
	}

	// 1. Extract clean identifier candidates from the prompt
	candidates := extractSymbolCandidates(query)
	// Build candidate list: full query first, then individual words >= 3 chars.
	// Also add CamelCase sub-tokens (e.g. "SymbolGraph" -> "Symbol", "Graph").

	// 2. Search symbol graph for all candidates, deduplicating results
	seenSymbols := make(map[string]bool)
	seenFiles := make(map[string]bool)
	for _, cand := range candidates {
		syms := p.symbolGraph.SearchSymbols(cand)
		for _, sym := range syms {
			key := fmt.Sprintf("%s:%s:%d", sym.Name, sym.File, sym.Line)
			if !seenSymbols[key] {
				seenSymbols[key] = true
				result.FoundSymbols = append(result.FoundSymbols, sym)
			}
		}
	}
	// 2. Search SymbolGraph for matched candidates
	seenSymbols = make(map[string]bool)
	for _, cand := range candidates {
		syms := p.symbolGraph.SearchSymbols(cand)
		for _, sym := range syms {
			key := fmt.Sprintf("%s:%s:%d", sym.Name, sym.File, sym.Line)
			if !seenSymbols[key] {
				seenSymbols[key] = true
				result.FoundSymbols = append(result.FoundSymbols, sym)
			}
		}
	}
	// 3. For each found symbol, collect definitions, references, and implementations
	for _, sym := range result.FoundSymbols {
		// Track affected files from the symbol itself
		if !seenFiles[sym.File] {
			seenFiles[sym.File] = true
			result.AffectedFiles = append(result.AffectedFiles, AffectedFile{
				Path:       sym.File,
				Package:    sym.Package,
				Reason:     fmt.Sprintf("Contains %s %q", sym.Kind, sym.Name),
				Symbols:    []string{sym.Name},
				Priority:   2,
				IsTestFile: strings.HasSuffix(sym.File, "_test.go"),
			})
		}

		// Collect references
		refs := p.symbolGraph.FindReferences(sym.Name)
		result.References = append(result.References, refs...)

		// Trace call graph for functions
		if sym.Kind == symbol.KindFunction || sym.Kind == symbol.KindMethod {
			callGraph := p.symbolGraph.TraceCallGraph(sym.Name, 3)
			for k, v := range callGraph {
				result.CallGraph[k] = v
			}

			// Find callers
			callers := p.symbolGraph.FindCallers(sym.Name)
			for _, caller := range callers {
				if !seenFiles[caller.File] {
					seenFiles[caller.File] = true
					result.AffectedFiles = append(result.AffectedFiles, AffectedFile{
						Path:       caller.File,
						Package:    caller.Package,
						Reason:     fmt.Sprintf("Calls %q (caller)", sym.Name),
						Symbols:    []string{caller.Name, sym.Name},
						Priority:   1,
						IsTestFile: strings.HasSuffix(caller.File, "_test.go"),
					})
				}
			}

			// Trace callees (functions this symbol calls)
			if callees, ok := callGraph[sym.Name]; ok {
				for _, callee := range callees {
					defs := p.symbolGraph.FindDefinitions(callee)
					for _, def := range defs {
						if !seenFiles[def.File] {
							seenFiles[def.File] = true
							result.AffectedFiles = append(result.AffectedFiles, AffectedFile{
								Path:       def.File,
								Package:    def.Package,
								Reason:     fmt.Sprintf("Called by %q (callee)", sym.Name),
								Symbols:    []string{callee, sym.Name},
								Priority:   1,
								IsTestFile: strings.HasSuffix(def.File, "_test.go"),
							})
						}
					}
				}
			}
		}

		// Find implementations for interfaces
		if sym.Kind == symbol.KindInterface {
			impls := p.symbolGraph.FindImplementations(sym.Name)
			for _, impl := range impls {
				if !seenFiles[impl.File] {
					seenFiles[impl.File] = true
					result.AffectedFiles = append(result.AffectedFiles, AffectedFile{
						Path:       impl.File,
						Package:    impl.Package,
						Reason:     fmt.Sprintf("Implements interface %q", sym.Name),
						Symbols:    []string{impl.Name, sym.Name},
						Priority:   1,
						IsTestFile: strings.HasSuffix(impl.File, "_test.go"),
					})
				}
			}
		}
	}

	// 3. Sort affected files by priority (descending) and then by path
	sort.Slice(result.AffectedFiles, func(i, j int) bool {
		if result.AffectedFiles[i].Priority != result.AffectedFiles[j].Priority {
			return result.AffectedFiles[i].Priority > result.AffectedFiles[j].Priority
		}
		return result.AffectedFiles[i].Path < result.AffectedFiles[j].Path
	})

	// 4. Build file dependency map for topological ordering
	for _, file := range result.AffectedFiles {
		deps := p.symbolGraph.GetSymbolsByFile(file.Path)
		var depFiles []string
		for _, dep := range deps {
			// Find files that define symbols referenced in this file
			for _, ref := range dep.References {
				refFile := ref.File
				if refFile != "" && refFile != file.Path && seenFiles[refFile] {
					depFiles = append(depFiles, refFile)
				}
			}
		}
		// Deduplicate
		depSet := make(map[string]bool)
		var uniqueDeps []string
		for _, d := range depFiles {
			if !depSet[d] {
				depSet[d] = true
				uniqueDeps = append(uniqueDeps, d)
			}
		}
		result.FileDependencies[file.Path] = uniqueDeps
	}

	result.Duration = time.Since(start)
	return result, nil
}

// ─── Phase 2: Planning ──────────────────────────────────────────────────────────

// Plan performs Phase 2: create an execution plan from exploration results.
func (p *Planner) Plan(goal string, exploration *ExplorationResult) (*ExecutionPlan, error) {
	plan := &ExecutionPlan{
		Goal:      goal,
		CreatedAt: time.Now(),
		Phases:    make([]PlanPhase, 0),
	}

	// Count affected files
	plan.EstimatedChanges = len(exploration.AffectedFiles)

	// Assess risk level
	directChanges := 0
	testChanges := 0
	highPriority := 0
	for _, f := range exploration.AffectedFiles {
		if f.Priority >= 2 {
			highPriority++
		}
		if f.IsTestFile {
			testChanges++
		} else {
			directChanges++
		}
	}

	if highPriority > 5 || directChanges > 10 {
		plan.RiskLevel = "high"
	} else if highPriority > 2 || directChanges > 3 {
		plan.RiskLevel = "medium"
	} else {
		plan.RiskLevel = "low"
	}

	// Build reasoning
	var reasoning strings.Builder
	reasoning.WriteString(fmt.Sprintf("## Exploration Summary\n\n"))
	reasoning.WriteString(fmt.Sprintf("Query: %q\n", goal))
	reasoning.WriteString(fmt.Sprintf("Symbols found: %d\n", len(exploration.FoundSymbols)))
	reasoning.WriteString(fmt.Sprintf("Affected files: %d (direct: %d, tests: %d)\n", len(exploration.AffectedFiles), directChanges, testChanges))
	reasoning.WriteString(fmt.Sprintf("References found: %d\n", len(exploration.References)))
	reasoning.WriteString(fmt.Sprintf("Risk level: %s\n\n", plan.RiskLevel))

	reasoning.WriteString("## Affected Files\n\n")
	for i, f := range exploration.AffectedFiles {
		priority := ""
		switch f.Priority {
		case 2:
			priority = "[HIGH]"
		case 1:
			priority = "[MED]"
		default:
			priority = "[LOW]"
		}
		reasoning.WriteString(fmt.Sprintf("%d. %s %s - %s\n", i+1, priority, f.Path, f.Reason))
		reasoning.WriteString(fmt.Sprintf("   Symbols: %s\n", strings.Join(f.Symbols, ", ")))
	}

	if len(exploration.CallGraph) > 0 {
		reasoning.WriteString("\n## Call Graph\n\n")
		for caller, callees := range exploration.CallGraph {
			reasoning.WriteString(fmt.Sprintf("  %s -> %s\n", caller, strings.Join(callees, ", ")))
		}
	}

	plan.Reasoning = reasoning.String()

	// Build execution phases
	// Phase A: Preparation - create test files first if needed
	if testChanges > 0 {
		testPhase := PlanPhase{
			Name:        "Test Preparation",
			Description: "Update test fixtures and test files to match expected new behavior",
			Steps:       make([]PlanStep, 0),
		}
		for _, f := range exploration.AffectedFiles {
			if f.IsTestFile {
				step := PlanStep{
					ID:           len(testPhase.Steps) + 1,
					Action:       "modify",
					File:         f.Path,
					Description:  fmt.Sprintf("Update test expectations in %s", filepath.Base(f.Path)),
					Details:      fmt.Sprintf("Modify tests related to: %s", strings.Join(f.Symbols, ", ")),
					Dependencies: []int{},
				}
				testPhase.Steps = append(testPhase.Steps, step)
			}
		}
		if len(testPhase.Steps) > 0 {
			plan.Phases = append(plan.Phases, testPhase)
		}
	}

	// Phase B: Core changes - modify the primary files
	primaryPhase := PlanPhase{
		Name:        "Core Implementation",
		Description: "Apply the primary code changes",
		Steps:       make([]PlanStep, 0),
	}
	for _, f := range exploration.AffectedFiles {
		if !f.IsTestFile {
			step := PlanStep{
				ID:           len(primaryPhase.Steps) + 1,
				Action:       "modify",
				File:         f.Path,
				Description:  fmt.Sprintf("Update %s - %s", filepath.Base(f.Path), f.Reason),
				Details:      fmt.Sprintf("Symbols affected: %s", strings.Join(f.Symbols, ", ")),
				Dependencies: []int{},
				DryRunCheck:  fmt.Sprintf("Verify %s parses correctly after change", filepath.Base(f.Path)),
			}
			primaryPhase.Steps = append(primaryPhase.Steps, step)
		}
	}
	if len(primaryPhase.Steps) > 0 {
		plan.Phases = append(plan.Phases, primaryPhase)
	}

	// Phase C: Verification
	verifyPhase := PlanPhase{
		Name:        "Verification",
		Description: "Run tests and linters to validate all changes",
		Steps: []PlanStep{
			{
				ID:          1,
				Action:      "verify",
				File:        p.rootDir,
				Description: "Run go vet on all modified files",
				Details:     "Check for common Go mistakes",
			},
			{
				ID:          2,
				Action:      "verify",
				File:        p.rootDir,
				Description: "Run go build to check compilation",
				Details:     "Ensure the entire project compiles",
			},
			{
				ID:          3,
				Action:      "verify",
				File:        p.rootDir,
				Description: "Run go test on affected packages",
				Details:     "Ensure all tests pass",
			},
		},
	}
	plan.Phases = append(plan.Phases, verifyPhase)

	return plan, nil
}

// ─── Phase 3: Patch Execution & Verification ────────────────────────────────────

// ExecutePlan runs the plan with verification after each step.
func (p *Planner) ExecutePlan(plan *ExecutionPlan, exploration *ExplorationResult) ([]PatchResult, *VerificationResult, error) {
	results := make([]PatchResult, 0)

	// Create work directory for dry runs
	if err := os.MkdirAll(p.workDir, 0755); err != nil {
		return nil, nil, fmt.Errorf("create work dir: %w", err)
	}
	defer os.RemoveAll(p.workDir)

	// Execute each phase
	for _, phase := range plan.Phases {
		fmt.Fprintf(os.Stderr, "\n=== Phase: %s ===\n", phase.Name)
		fmt.Fprintf(os.Stderr, "%s\n", phase.Description)

		for _, step := range phase.Steps {
			fmt.Fprintf(os.Stderr, "  Step %d: %s\n", step.ID, step.Description)

			start := time.Now()
			result := PatchResult{
				StepID: step.ID,
				File:   step.File,
			}

			// For verification steps, run checks
			if step.Action == "verify" {
				verification := p.runVerification(step)
				result.Success = verification.Success
				result.Output = verification.GoBuild
				if !verification.Success {
					result.ErrorMessage = p.buildErrorSummary(verification)
				}
			} else {
				// For modify/create steps, validate the file parses as Go
				if strings.HasSuffix(step.File, ".go") {
					err := p.dryRunParse(step.File)
					if err != nil {
						result.Success = false
						result.ErrorMessage = fmt.Sprintf("dry-run parse failed: %v", err)
					} else {
						result.Success = true
						result.Output = fmt.Sprintf("File %s parses correctly", filepath.Base(step.File))
					}
				} else {
					result.Success = true
					result.Output = "Non-Go file, skipping parse check"
				}
			}

			result.Duration = time.Since(start)
			results = append(results, result)
		}
	}

	// Final verification
	finalVerification := p.runFullVerification()

	return results, finalVerification, nil
}

// dryRunParse parses a Go file in memory to check for syntax errors.
func (p *Planner) dryRunParse(filePath string) error {
	absPath := filepath.Join(p.rootDir, filePath)
	if !filepath.IsAbs(filePath) {
		absPath = filepath.Join(p.rootDir, filePath)
	} else {
		absPath = filePath
	}

	cmd := exec.Command("go", "fmt", absPath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("go fmt failed: %s", string(output))
	}

	// Also check with go vet for the package containing the file
	pkgDir := filepath.Dir(absPath)
	cmd = exec.Command("go", "vet", pkgDir)
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("go vet failed: %s", string(output))
	}

	return nil
}

// runVerification runs a single verification step.
func (p *Planner) runVerification(step PlanStep) VerificationResult {
	result := VerificationResult{Success: true}

	switch step.ID {
	case 1: // go vet
		cmd := exec.Command("go", "vet", "./...")
		cmd.Dir = p.rootDir
		if output, err := cmd.CombinedOutput(); err != nil {
			result.Success = false
			result.GoVet = string(output)
			result.CompilerErrors = append(result.CompilerErrors, string(output))
		}
	case 2: // go build
		cmd := exec.Command("go", "build", "./...")
		cmd.Dir = p.rootDir
		if output, err := cmd.CombinedOutput(); err != nil {
			result.Success = false
			result.GoBuild = string(output)
			result.CompilerErrors = append(result.CompilerErrors, string(output))
		}
	case 3: // go test
		cmd := exec.Command("go", "test", "./...")
		cmd.Dir = p.rootDir
		if output, err := cmd.CombinedOutput(); err != nil {
			result.Success = false
			result.GoTest = string(output)
		}
	}

	return result
}

// runFullVerification runs the complete verification suite.
func (p *Planner) runFullVerification() *VerificationResult {
	result := &VerificationResult{Success: true}

	// go fmt
	cmd := exec.Command("go", "fmt", "./...")
	cmd.Dir = p.rootDir
	if output, err := cmd.CombinedOutput(); err != nil {
		return &VerificationResult{
			Success: false,
			GoFmt:   string(output),
		}
	}

	// go vet
	cmd = exec.Command("go", "vet", "./...")
	cmd.Dir = p.rootDir
	if output, err := cmd.CombinedOutput(); err != nil {
		result.Success = false
		result.GoVet = string(output)
		result.CompilerErrors = append(result.CompilerErrors, string(output))
	}

	// go build
	cmd = exec.Command("go", "build", "./...")
	cmd.Dir = p.rootDir
	if output, err := cmd.CombinedOutput(); err != nil {
		result.Success = false
		result.GoBuild = string(output)
		result.CompilerErrors = append(result.CompilerErrors, string(output))
	}

	// go test
	cmd = exec.Command("go", "test", "./...")
	cmd.Dir = p.rootDir
	if output, err := cmd.CombinedOutput(); err != nil {
		result.Success = false
		result.GoTest = string(output)
	}

	return result
}

// buildErrorSummary creates a concise error summary from verification results.
func (p *Planner) buildErrorSummary(v VerificationResult) string {
	var parts []string
	if v.GoVet != "" {
		parts = append(parts, "vet: "+v.GoVet)
	}
	if v.GoBuild != "" {
		parts = append(parts, "build: "+v.GoBuild)
	}
	if v.GoTest != "" {
		parts = append(parts, "test: "+v.GoTest)
	}
	return strings.Join(parts, "\n")
}

// ─── Multi-Candidate Sampling ───────────────────────────────────────────────────

// PatchCandidate represents one implementation approach.
type PatchCandidate struct {
	ID          int            `json:"id"`
	Description string         `json:"description"`
	Strategy    string         `json:"strategy"`
	Plan        *ExecutionPlan `json:"plan"`
	Results     []PatchResult  `json:"results"`
}

// GenerateCandidates creates multiple implementation approaches for a goal.
func (p *Planner) GenerateCandidates(goal string, exploration *ExplorationResult) ([]*PatchCandidate, error) {
	candidates := make([]*PatchCandidate, 0)

	// Candidate 1: Minimal changes - only modify the core files
	candidate1 := &PatchCandidate{
		ID:          1,
		Description: "Minimal: Only modify directly affected files",
		Strategy:    "Make the smallest possible change set. Only modify files that directly contain the target symbols.",
	}
	plan1 := &ExecutionPlan{
		Goal:      goal,
		CreatedAt: time.Now(),
		RiskLevel: "low",
		Reasoning: "Minimal approach - only touching directly affected files and their immediate tests.",
	}
	corePhase := PlanPhase{
		Name:        "Core Changes",
		Description: "Minimal changes to directly affected files",
		Steps:       make([]PlanStep, 0),
	}
	for _, f := range exploration.AffectedFiles {
		if f.Priority >= 2 {
			step := PlanStep{
				ID:          len(corePhase.Steps) + 1,
				Action:      "modify",
				File:        f.Path,
				Description: fmt.Sprintf("Update %s", filepath.Base(f.Path)),
				Details:     f.Reason,
			}
			corePhase.Steps = append(corePhase.Steps, step)
		}
	}
	plan1.Phases = append(plan1.Phases, corePhase)
	candidate1.Plan = plan1
	candidates = append(candidates, candidate1)

	// Candidate 2: Comprehensive - modify all affected files including callers
	candidate2 := &PatchCandidate{
		ID:          2,
		Description: "Comprehensive: Modify all affected files including callers",
		Strategy:    "Update the core symbols and all their callers for consistency. Add new tests.",
	}
	plan2 := &ExecutionPlan{
		Goal:      goal,
		CreatedAt: time.Now(),
		RiskLevel: highRiskLevel(exploration),
		Reasoning: "Comprehensive approach - updating the full call chain for consistency.",
	}
	plan2.Phases = append(plan2.Phases, PlanPhase{
		Name:        "All Changes",
		Description: "Update all affected files",
		Steps:       buildStepsFromAllFiles(exploration),
	})
	candidate2.Plan = plan2
	candidates = append(candidates, candidate2)

	// Candidate 3: Safe with tests - modify files and add/update tests
	candidate3 := &PatchCandidate{
		ID:          3,
		Description: "Safe: Modify files and update/add comprehensive tests",
		Strategy:    "Make changes with parallel test updates. Run tests after each change.",
	}
	plan3 := &ExecutionPlan{
		Goal:      goal,
		CreatedAt: time.Now(),
		RiskLevel: "medium",
		Reasoning: "Safe approach - tests are updated alongside implementation changes.",
		Phases: []PlanPhase{
			{
				Name:        "Test Updates",
				Description: "Update test files first",
				Steps:       buildTestSteps(exploration),
			},
			{
				Name:        "Implementation",
				Description: "Apply core changes",
				Steps:       buildCoreSteps(exploration),
			},
		},
	}
	candidate3.Plan = plan3
	candidates = append(candidates, candidate3)

	return candidates, nil
}

func highRiskLevel(e *ExplorationResult) string {
	if len(e.AffectedFiles) > 10 {
		return "high"
	}
	return "medium"
}

func buildStepsFromAllFiles(e *ExplorationResult) []PlanStep {
	steps := make([]PlanStep, 0)
	for i, f := range e.AffectedFiles {
		steps = append(steps, PlanStep{
			ID:          i + 1,
			Action:      "modify",
			File:        f.Path,
			Description: fmt.Sprintf("Update %s", filepath.Base(f.Path)),
			Details:     f.Reason,
		})
	}
	return steps
}

func buildTestSteps(e *ExplorationResult) []PlanStep {
	steps := make([]PlanStep, 0)
	for i, f := range e.AffectedFiles {
		if f.IsTestFile {
			steps = append(steps, PlanStep{
				ID:          i + 1,
				Action:      "modify",
				File:        f.Path,
				Description: fmt.Sprintf("Update test %s", filepath.Base(f.Path)),
				Details:     f.Reason,
			})
		}
	}
	return steps
}

func buildCoreSteps(e *ExplorationResult) []PlanStep {
	steps := make([]PlanStep, 0)
	for i, f := range e.AffectedFiles {
		if !f.IsTestFile {
			steps = append(steps, PlanStep{
				ID:          i + 1,
				Action:      "modify",
				File:        f.Path,
				Description: fmt.Sprintf("Update %s", filepath.Base(f.Path)),
				Details:     f.Reason,
			})
		}
	}
	return steps
}

// ─── RLAIF Integration ──────────────────────────────────────────────────────────

// ExecutePatchAndVerify applies a SEARCH/REPLACE patch and runs the full Go
// toolchain verification (go/parser, go vet, go build). This is the core
// compiler-driven RL loop that generates reward signals for model training.
//
// Returns a PatchOutcome with reward: +1.0 for success, -1.0 for failure,
// -0.5 for vet warnings. Compiler errors are fed back for error ingestion.
func (p *Planner) ExecutePatchAndVerify(patch, beforeCode, filePath string) training.PatchOutcome {
	// Use the RLAIF verification logic directly
	config := training.DefaultRLAIFConfig()
	config.TempDir = p.workDir

	// Create a minimal trainer just for verification
	// (model updates happen externally via the RLAIFTrainer)
	outcome := training.PatchOutcome{
		Reward:  config.RewardFailure,
		ValidGo: true,
	}

	// Step 1: Parse before code
	fset := token.NewFileSet()
	if _, err := parser.ParseFile(fset, "", beforeCode, parser.AllErrors); err != nil {
		outcome.ValidGo = false
		outcome.CompilerErrors = fmt.Sprintf("before code parse error: %v", err)
		return outcome
	}

	// Step 2: Extract after code from patch
	afterCode := extractAfterFromPatchRLAIF(patch)
	if afterCode == "" {
		outcome.ValidGo = false
		outcome.CompilerErrors = "could not extract REPLACE section from patch"
		return outcome
	}

	// Step 3: Parse after code
	fset = token.NewFileSet()
	if _, err := parser.ParseFile(fset, "", afterCode, parser.AllErrors); err != nil {
		outcome.ValidGo = false
		outcome.CompilerErrors = fmt.Sprintf("after code parse error: %v", err)
		return outcome
	}
	outcome.ValidGo = true

	// Step 4: Write to temp file and run go vet
	if filePath == "" {
		filePath = "patch_output.go"
	}
	tempFile := filepath.Join(p.workDir, filepath.Base(filePath))
	if err := os.MkdirAll(p.workDir, 0755); err != nil {
		outcome.CompilerErrors = fmt.Sprintf("create work dir: %v", err)
		return outcome
	}
	if err := os.WriteFile(tempFile, []byte(afterCode), 0644); err != nil {
		outcome.CompilerErrors = fmt.Sprintf("write temp file: %v", err)
		return outcome
	}

	// Run go vet
	cmd := exec.Command("go", "vet", tempFile)
	if output, err := cmd.CombinedOutput(); err != nil {
		outcome.VetPassed = false
		outcome.CompilerErrors = string(output)
		outcome.Reward = config.RewardVetWarn
		return outcome
	}
	outcome.VetPassed = true

	// Step 5: Try go build
	modDir := filepath.Dir(tempFile)
	if _, err := os.Stat(filepath.Join(modDir, "go.mod")); os.IsNotExist(err) {
		initCmd := exec.Command("go", "mod", "init", "gollemer_rlaif_sandbox")
		initCmd.Dir = modDir
		_ = initCmd.Run()
	}

	buildCmd := exec.Command("go", "build", "-o", "/dev/null", tempFile)
	buildCmd.Dir = modDir
	if output, err := buildCmd.CombinedOutput(); err != nil {
		outcome.BuildPassed = false
		outcome.CompilerErrors = string(output)
		outcome.Reward = config.RewardFailure
		return outcome
	}
	outcome.BuildPassed = true
	outcome.Reward = config.RewardSuccess

	return outcome
}

// extractAfterFromPatchRLAIF extracts the REPLACE section from a SEARCH/REPLACE patch.
func extractAfterFromPatchRLAIF(patch string) string {
	parts := strings.Split(patch, "=======\n")
	if len(parts) != 2 {
		return ""
	}
	after := strings.TrimSuffix(parts[1], "\n>>>>>>> REPLACE")
	after = strings.TrimSuffix(after, ">>>>>>> REPLACE")
	return after
}

// ─── Public API ─────────────────────────────────────────────────────────────────

// ThreePhasePipeline runs all three phases in sequence.
// Returns the exploration, plan, patch results, and verification.
func (p *Planner) ThreePhasePipeline(goal string) (*ExplorationResult, *ExecutionPlan, []PatchResult, *VerificationResult, error) {
	fmt.Fprintf(os.Stderr, "\n🔍 Phase 1: Exploration & Mapping\n")
	fmt.Fprintf(os.Stderr, "   Query: %q\n", goal)

	exploration, err := p.Explore(goal)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("exploration failed: %w", err)
	}

	fmt.Fprintf(os.Stderr, "   Found %d symbols, %d affected files\n", len(exploration.FoundSymbols), len(exploration.AffectedFiles))

	fmt.Fprintf(os.Stderr, "\n📋 Phase 2: Architectural Execution Plan\n")

	plan, err := p.Plan(goal, exploration)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("planning failed: %w", err)
	}

	fmt.Fprintf(os.Stderr, "   Risk level: %s, Estimated changes: %d\n", plan.RiskLevel, plan.EstimatedChanges)

	fmt.Fprintf(os.Stderr, "\n🔧 Phase 3: Targeted Patch Execution\n")

	results, verification, err := p.ExecutePlan(plan, exploration)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("execution failed: %w", err)
	}

	return exploration, plan, results, verification, nil
}

// ─── JSON Export/Import ────────────────────────────────────────────────────────

// ExportExploration serializes exploration results to JSON.
func ExportExploration(e *ExplorationResult) ([]byte, error) {
	return json.MarshalIndent(e, "", "  ")
}

// ExportPlan serializes the execution plan to JSON.
func ExportPlan(p *ExecutionPlan) ([]byte, error) {
	return json.MarshalIndent(p, "", "  ")
}

// ExportResults serializes patch results to JSON.
func ExportResults(r []PatchResult) ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// splitCamelCase splits a CamelCase or PascalCase identifier into its component words.
// e.g. "SymbolGraph" -> ["Symbol", "Graph"], "JWTToken" -> ["JWT", "Token"].
func splitCamelCase(s string) []string {
	if s == "" {
		return nil
	}
	var tokens []string
	var current strings.Builder
	runes := []rune(s)
	for i, r := range runes {
		if r >= 'A' && r <= 'Z' {
			// Start of a new uppercase segment
			if current.Len() > 0 {
				// Check if previous was uppercase and next is lowercase (e.g. "JWTToken" -> "JWT", "Token")
				if i > 0 && runes[i-1] >= 'A' && runes[i-1] <= 'Z' && i+1 < len(runes) && runes[i+1] >= 'a' && runes[i+1] <= 'z' {
					tokens = append(tokens, current.String())
					current.Reset()
				} else if runes[i-1] >= 'a' && runes[i-1] <= 'z' {
					// Previous was lowercase, this is a new word boundary
					tokens = append(tokens, current.String())
					current.Reset()
				}
			}
			current.WriteRune(r)
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

func extractSymbolCandidates(query string) []string {
	// 1. Common programming/English stop words to ignore during symbol search
	stopWords := map[string]bool{
		"add": true, "create": true, "remove": true, "delete": true, "update": true,
		"modify": true, "fix": true, "implement": true, "layer": true, "to": true,
		"in": true, "for": true, "the": true, "a": true, "an": true, "and": true,
		"or": true, "resolution": true, "caching": true, "cache": true, "struct": true,
		"method": true, "methods": true, "function": true, "file": true,
	}

	words := strings.FieldsFunc(query, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_')
	})

	var candidates []string

	for _, w := range words {
		lower := strings.ToLower(w)

		// Skip short words and stop words
		if len(w) <= 2 || stopWords[lower] {
			continue
		}

		// Keep PascalCase/camelCase identifiers (e.g. SymbolGraph)
		candidates = append(candidates, w)
	}

	// Always prefer exact composite identifier matches first if present
	return candidates
}

// ─── LLM Architect (Phase 2 Enhancement) ────────────────────────────────

func (p *Planner) ExecutePlanWithRetry(plan *ExecutionPlan, exploration *ExplorationResult) ([]PatchResult, *VerificationResult, error) {
	var lastVerification *VerificationResult

	for attempt := 1; attempt <= p.maxRetries; attempt++ {
		fmt.Fprintf(os.Stderr, "\n=== Attempt %d/%d ===\n", attempt, p.maxRetries)

		results, verification, err := p.ExecutePlan(plan, exploration)
		if err != nil {
			return nil, nil, fmt.Errorf("execution failed on attempt %d: %w", attempt, err)
		}

		// Check if verification passed
		if verification.Success {
			if attempt > 1 {
				fmt.Fprintf(os.Stderr, "✅ Self-healing succeeded on attempt %d\n", attempt)
			}
			return results, verification, nil
		}

		lastVerification = verification

		// Collect all error messages
		var errorMessages []string
		if verification.GoVet != "" {
			errorMessages = append(errorMessages, "vet: "+verification.GoVet)
		}
		if verification.GoBuild != "" {
			errorMessages = append(errorMessages, "build: "+verification.GoBuild)
		}
		if verification.GoTest != "" {
			errorMessages = append(errorMessages, "test: "+verification.GoTest)
		}
		combinedError := strings.Join(errorMessages, "\n")

		fmt.Fprintf(os.Stderr, "⚠️  Verification failed on attempt %d: %s\n", attempt, combinedError)

		// Don't replan on the last attempt
		if attempt >= p.maxRetries {
			break
		}

		// Feed error back into Phase 2 to generate corrected plan
		fmt.Fprintf(os.Stderr, "🔄 Feeding error back into Phase 2 for replanning...\n")
		plan, err = p.ReplanWithError(plan, combinedError, exploration)
		if err != nil {
			return results, verification, fmt.Errorf("replanning failed after attempt %d: %w", attempt, err)
		}
	}

	return nil, lastVerification, fmt.Errorf("self-healing exhausted %d attempts: %s", p.maxRetries, p.buildErrorSummary(*lastVerification))
}

// GenerateExecutionPlan uses the native LLM engine to produce a structured
// execution plan from Phase 1 exploration results, falling back to a
// deterministic plan if the model is unavailable.
// The prompt is augmented with concept templates extracted from the query,
// providing the LLM with Go pattern blueprints to follow.
func (p *Planner) GenerateExecutionPlan(ctx context.Context, exp *ExplorationResult) (*ExecutionPlan, error) {
	// Extract concepts from the query to augment the prompt
	concepts := p.conceptMatcher.ExtractConcepts(exp.Query)
	prompt := p.buildConceptAugmentedPrompt(exp, concepts)

	if p.llm != nil {
		response, err := p.llm.Generate(ctx, prompt)
		if err == nil {
			return p.parseLLMResponseToPlan(response, exp)
		}
	}

	return p.buildDeterministicPlan(exp), nil
}

// buildConceptAugmentedPrompt constructs a prompt that includes concept
// blueprints alongside the standard exploration context. This forces the
// LLM to fill in the parameters of proven Go patterns rather than
// generating code from scratch.
func (p *Planner) buildConceptAugmentedPrompt(exp *ExplorationResult, concepts []knowledge.ConceptTemplate) string {
	var conceptInstructions string
	for _, c := range concepts {
		conceptInstructions += fmt.Sprintf(
			"Concept Detected: %s\n"+
				"Required Go Primitives: %v\n"+
				"Structural Code Patterns To Apply:\n",
			c.Term, c.RequiredConstructs,
		)
		for _, m := range c.ASTMutations {
			conceptInstructions += fmt.Sprintf("- Pattern [%s]: %s\n", m.Type, m.CodeTemplate)
		}
	}

	return fmt.Sprintf(
		"### COMMAND: %s\n\n"+
			"### GO PATTERN BLUEPRINTS:\n%s\n"+
			"### LOCAL AST SYMBOLS:\n%v\n"+
			"### TARGET FILES:\n%v\n\n"+
			"Instructions: Apply the Go pattern blueprints above to the target symbols in the affected files.",
		exp.Query, conceptInstructions, exp.FoundSymbols, exp.AffectedFiles,
	)
}

func (p *Planner) parseLLMResponseToPlan(response string, exp *ExplorationResult) (*ExecutionPlan, error) {
	var steps []StructuredPlan
	if err := json.Unmarshal([]byte(response), &steps); err != nil {
		return p.buildDeterministicPlan(exp), nil
	}

	plan := &ExecutionPlan{
		Goal:      exp.Query,
		CreatedAt: time.Now(),
		Phases:    make([]PlanPhase, 0),
		RiskLevel: "low",
		Reasoning: "LLM-generated plan with structured patch instructions",
	}

	for _, s := range steps {
		plan.Phases = append(plan.Phases, PlanPhase{
			Name:        s.Description,
			Description: s.Action,
			Steps: []PlanStep{
				{
					ID:          len(plan.Phases) + 1,
					Action:      s.Action,
					File:        s.TargetFile,
					Description: s.Description,
					Details:     s.CodeSnippet,
				},
			},
		})
	}

	return plan, nil
}

func (p *Planner) buildDeterministicPlan(exp *ExplorationResult) *ExecutionPlan {
	plan := &ExecutionPlan{
		Goal:      exp.Query,
		CreatedAt: time.Now(),
		Phases:    make([]PlanPhase, 0),
		Reasoning: "Deterministic fallback plan",
	}

	corePhase := PlanPhase{
		Name:        "Core Changes",
		Description: "Modify affected files",
		Steps:       make([]PlanStep, 0),
	}

	for _, f := range exp.AffectedFiles {
		if !f.IsTestFile {
			corePhase.Steps = append(corePhase.Steps, PlanStep{
				ID:          len(corePhase.Steps) + 1,
				Action:      "modify",
				File:        f.Path,
				Description: fmt.Sprintf("Update %s", filepath.Base(f.Path)),
				Details:     f.Reason,
			})
		}
	}

	if len(corePhase.Steps) > 0 {
		plan.Phases = append(plan.Phases, corePhase)
	}

	return plan
}

// ReplanWithError generates a revised execution plan by feeding verification
// errors back into the planning phase. This enables automated self-healing.
func (p *Planner) ReplanWithError(currentPlan *ExecutionPlan, errorLog string, exploration *ExplorationResult) (*ExecutionPlan, error) {
	// Build error context for replanning
	errorContext := p.analyzeErrors(errorLog, exploration)

	// Create a revised plan that addresses the detected issues
	revisedPlan := &ExecutionPlan{
		Goal:      currentPlan.Goal,
		CreatedAt: time.Now(),
		Phases:    make([]PlanPhase, 0),
		Reasoning: fmt.Sprintf("Replanned due to errors:\n%s\n\nOriginal reasoning:\n%s", errorLog, currentPlan.Reasoning),
	}

	// Filter out steps that are failing and add corrective steps
	for _, phase := range currentPlan.Phases {
		if phase.Name == "Verification" {
			continue // Skip old verification phase
		}
		revisedPlan.Phases = append(revisedPlan.Phases, phase)
	}

	// Add corrective phase if there are error patterns to address
	if len(errorContext) > 0 {
		correctivePhase := PlanPhase{
			Name:        "Corrective Actions",
			Description: "Fix verification failures detected in previous attempt",
			Steps:       make([]PlanStep, 0),
		}

		for i, ctx := range errorContext {
			correctivePhase.Steps = append(correctivePhase.Steps, PlanStep{
				ID:          len(correctivePhase.Steps) + 1,
				Action:      "modify",
				File:        ctx.file,
				Description: fmt.Sprintf("Fix: %s", ctx.description),
				Details:     ctx.detail,
			})
			_ = i // index used for step ID generation above
		}

		// Add verification step
		correctivePhase.Steps = append(correctivePhase.Steps, PlanStep{
			ID:          len(correctivePhase.Steps) + 1,
			Action:      "verify",
			File:        p.rootDir,
			Description: "Re-verify fixes with go vet and go test",
			Details:     "Ensure all verification errors are resolved",
		})

		revisedPlan.Phases = append(revisedPlan.Phases, correctivePhase)
	}

	// Add final verification phase
	revisedPlan.Phases = append(revisedPlan.Phases, PlanPhase{
		Name:        "Final Verification",
		Description: "Run complete verification suite",
		Steps: []PlanStep{
			{
				ID:          1,
				Action:      "verify",
				File:        p.rootDir,
				Description: "Run go vet on all modified files",
				Details:     "Check for common Go mistakes",
			},
			{
				ID:          2,
				Action:      "verify",
				File:        p.rootDir,
				Description: "Run go build to check compilation",
				Details:     "Ensure the entire project compiles",
			},
			{
				ID:          3,
				Action:      "verify",
				File:        p.rootDir,
				Description: "Run go test on affected packages",
				Details:     "Ensure all tests pass",
			},
		},
	})

	return revisedPlan, nil
}

// errorContextEntry represents a single verification error with location info.
type errorContextEntry struct {
	file        string
	description string
	detail      string
}

// analyzeErrors parses verification error output and maps errors to affected files.
func (p *Planner) analyzeErrors(errorLog string, exploration *ExplorationResult) []errorContextEntry {
	var entries []errorContextEntry

	for _, f := range exploration.AffectedFiles {
		// Check if this file appears in the error log
		if strings.Contains(errorLog, filepath.Base(f.Path)) || strings.Contains(errorLog, f.Path) {
			// Try to extract a meaningful description from the error
			description := "Fix verification errors in this file"
			detail := fmt.Sprintf("File %s produced errors during verification", f.Path)

			if strings.Contains(errorLog, "vet:") {
				description = "Fix go vet errors"
				detail = fmt.Sprintf("go vet reported issues in %s", filepath.Base(f.Path))
			} else if strings.Contains(errorLog, "build:") {
				description = "Fix compilation errors"
				detail = fmt.Sprintf("Build failed for %s", filepath.Base(f.Path))
			} else if strings.Contains(errorLog, "test:") {
				description = "Fix test failures"
				detail = fmt.Sprintf("Tests failed for %s", filepath.Base(f.Path))
			}

			entries = append(entries, errorContextEntry{
				file:        f.Path,
				description: description,
				detail:      detail,
			})
		}
	}

	return entries
}
