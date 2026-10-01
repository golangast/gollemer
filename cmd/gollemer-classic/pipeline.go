// Core orchestration engine: the "crazy smart" Go code intelligence
// pipeline.
//
// RunPipeline executes the full closed loop for one task:
//
//  1. 🔍 Resolving Graph — type-resolved package context
//     (pkg/ast.LoadPackageContext) plus a hybrid Graph + Vector memory
//     index (pkg/memory): vector search for seed nodes matching the
//     task, BFS expansion to depth 2 over CALLS and IMPLEMENTS edges,
//     all rendered into the LLM's context window.
//  2. 🧠 Drafting candidates — the LLM proposes N diverse solutions.
//  3. 🛡️ Symbolic Proving — every candidate is checked by the
//     path-sensitive symbolic prover (pkg/analysis) before anything
//     executes: candidates with CRITICAL violations (unchecked nil
//     dereference, unclosed resource, unguarded slice index) are
//     discarded (or heavily penalized if nothing survives); WARNINGs
//     subtract from fitness.
//  4. 🧪 MCTS Parallel Sandboxes — survivors are validated and
//     benchmarked concurrently in isolated sandboxes
//     (pkg/synthesis, pkg/runner) and ranked by fitness:
//     (TestsPassed * 50) + (ZeroLintWarnings * 20)
//     - (AllocsPerOp * 2) - (NsPerOp / 1000).
//  5. 🔧 Self-Healing — if the winner fails validation, its exact
//     JSON line-number errors ([]runner.TestError) are fed back to
//     the generator, up to 3 repair attempts.
//  6. ⚡ Auto-Tuning Memory — if the winner allocates, AST
//     micro-mutations (pre-sized slices, strings.Builder, pointer
//     receivers) are tried; each is kept only if allocations drop
//     and tests still pass.
//  7. ✅ Committed — the final files are written atomically via a
//     validated patch transaction (pkg/runner.ApplyPatchSet).
package main

import (
	"context"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"sort"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/pkg/analysis"
	gast "github.com/golangast/gollemer/pkg/ast"
	"github.com/golangast/gollemer/pkg/memory"
	"github.com/golangast/gollemer/pkg/runner"
	"github.com/golangast/gollemer/pkg/synthesis"
)

// PipelineOptions tunes the pipeline. Use DefaultPipelineOptions and
// override fields as needed.
type PipelineOptions struct {
	NumCandidates   int // LLM candidates drafted per task
	MaxHealAttempts int // self-healing repair attempts for the winner
	TopK            int // vector-search seed nodes
	GraphDepth      int // BFS expansion depth over CALLS/IMPLEMENTS
	TuneIters       int // max auto-tuning mutation attempts
	EmbedDim        int // embedding dimension for lexical vectors
}

// DefaultPipelineOptions returns the standard pipeline configuration.
func DefaultPipelineOptions() PipelineOptions {
	return PipelineOptions{
		NumCandidates:   3,
		MaxHealAttempts: 3,
		TopK:            5,
		GraphDepth:      2,
		TuneIters:       3,
		EmbedDim:        256,
	}
}

// PipelineResult summarizes a completed pipeline run.
type PipelineResult struct {
	Files          map[string]string // final committed files
	WinnerID       string            // winning candidate id
	Fitness        float64           // safety-adjusted final fitness
	SafetyCritical int               // CRITICAL violations on the winner
	SafetyWarnings int               // WARNING violations on the winner
	AllocsBefore   int64             // winner allocs/op before tuning
	AllocsAfter    int64             // winner allocs/op after tuning
	Mutations      []string          // auto-tuning mutations kept
	Healed         bool              // winner needed self-healing repair
	HealAttempts   int               // repair attempts used
}

// RunPipeline executes the full code-intelligence pipeline for task
// against the Go module rooted at targetDir and atomically commits
// the winning files. It returns the result summary.
func RunPipeline(targetDir, task string, opts PipelineOptions) (*PipelineResult, error) {
	if targetDir == "" {
		return nil, fmt.Errorf("pipeline: empty targetDir")
	}
	if task == "" {
		return nil, fmt.Errorf("pipeline: empty task")
	}
	if opts.NumCandidates < 1 {
		opts.NumCandidates = 1
	}
	ctx := context.Background()

	// 1. 🔍 Resolving Graph
	fmt.Println("🔍 Resolving Graph")
	codeCtx, err := gast.LoadPackageContext(targetDir)
	if err != nil {
		return nil, fmt.Errorf("pipeline: load package context: %w", err)
	}
	chunks := loadChunks(targetDir, 1<<20)
	graphChunks := graphContextChunks(codeCtx, chunks, task, opts)
	fmt.Printf("   module %s: %d structs, %d interfaces, %d context chunks\n",
		codeCtx.ModulePath, len(codeCtx.Structs), len(codeCtx.Interfaces), len(graphChunks))
	system := buildSystemPrompt(codeCtx, graphChunks)

	// 2. 🧠 Drafting candidates
	fmt.Printf("🧠 Drafting %d candidates\n", opts.NumCandidates)
	candidates, err := generateCandidates(task, system, opts.NumCandidates)
	if err != nil {
		return nil, err
	}

	// 3. 🛡️ Symbolic Proving (before anything executes)
	fmt.Println("🛡️ Symbolic Proving")
	penalties, survivors := proveAndFilter(candidates)
	fmt.Printf("   %d/%d candidates survived proving\n", len(survivors), len(candidates))

	// 4. 🧪 MCTS Parallel Sandboxes
	fmt.Printf("🧪 MCTS Parallel Sandboxes [%d/%d]\n", len(survivors), len(survivors))
	ranked, err := synthesis.EvaluateCandidatesRanked(ctx, targetDir, survivors)
	if err != nil {
		return nil, fmt.Errorf("pipeline: candidate evaluation: %w", err)
	}
	best := pickBest(ranked, penalties)
	fmt.Printf("   winner: %s  fitness=%.1f  allocs/op=%d  ns/op=%.0f\n",
		best.node.ID, best.final, best.node.Metrics.AllocsPerOp, best.node.Metrics.NsPerOp)
	result := &PipelineResult{
		WinnerID: best.node.ID,
		Fitness:  best.final,
	}

	// 5. 🔧 Self-Healing (only when the winner fails validation —
	// benchmark-only hiccups are not code bugs and are not repaired)
	winnerFiles := cloneFiles(best.node.CodeMap)
	winnerMetrics := &best.node.Metrics
	vres, verr := runner.ValidateGeneratedCode(targetDir, winnerFiles)
	if verr != nil {
		return nil, fmt.Errorf("pipeline: winner re-validation: %w", verr)
	}
	if !vres.Passed {
		fmt.Println("🔧 Self-Healing")
		healed, metrics, attempts, herr := healWinner(targetDir, task, winnerFiles, vres.Errors, winnerMetrics, opts.MaxHealAttempts)
		if herr != nil {
			return nil, herr
		}
		winnerFiles = healed
		winnerMetrics = metrics
		result.Healed = true
		result.HealAttempts = attempts
	} else {
		fmt.Println("🔧 Self-Healing: winner already green — skipping")
	}

	// 6. ⚡ Auto-Tuning Memory
	fmt.Println("⚡ Auto-Tuning Memory")
	result.AllocsBefore = winnerMetrics.AllocsPerOp
	if winnerMetrics.AllocsPerOp > 0 {
		tuned, tunedMetrics, kept := autoTuneAllocs(ctx, targetDir, winnerFiles, winnerMetrics, opts.TuneIters)
		winnerFiles, winnerMetrics = tuned, tunedMetrics
		result.Mutations = kept
	} else {
		fmt.Println("   zero allocs/op — nothing to tune")
	}
	result.AllocsAfter = winnerMetrics.AllocsPerOp
	winnerPenalty := 0
	if best.index >= 0 && best.index < len(penalties) {
		winnerPenalty = penalties[best.index]
	}
	result.Fitness = synthesis.CalculateFitness(
		&runner.ExecutionResult{Passed: true}, winnerMetrics) - float64(winnerPenalty)

	// 7. ✅ Committed
	fmt.Println("✅ Committed")
	var patches runner.PatchSet
	for _, name := range sortedKeys(winnerFiles) {
		patches.Add(name, winnerFiles[name])
	}
	res, err := runner.ApplyPatchSet(targetDir, patches)
	if err != nil {
		return nil, fmt.Errorf("pipeline: commit: %w", err)
	}
	if !res.Passed {
		return nil, fmt.Errorf("pipeline: commit validation failed:\n%s", formatErrors(res.Errors))
	}
	result.Files = winnerFiles
	crit, warn, _ := proveCandidate(winnerFiles)
	result.SafetyCritical, result.SafetyWarnings = crit, warn
	fmt.Printf("   committed %d file(s): %s\n", len(winnerFiles), joinNames(sortedKeys(winnerFiles)))
	return result, nil
}

// ---------------------------------------------------------------------------
// 1. Graph + vector context
// ---------------------------------------------------------------------------

// graphContextChunks builds the hybrid Graph + Vector memory index over
// the codebase chunks, embeds the task, and returns the seed nodes plus
// their depth-bounded graph neighborhood as prompt chunks. Any failure
// degrades gracefully to flat chunk context — retrieval must never
// break generation.
func graphContextChunks(codeCtx *gast.CodebaseContext, chunks []gast.CodeChunk, task string, opts PipelineOptions) []gast.CodeChunk {
	flat := func() []gast.CodeChunk {
		if len(chunks) > maxPromptChunks {
			return chunks[:maxPromptChunks]
		}
		return chunks
	}
	if len(chunks) == 0 {
		return nil
	}
	graph, err := memory.BuildGraph(codeCtx, chunks)
	if err != nil {
		fmt.Printf("   [graph] build failed (%v); using flat context\n", err)
		return flat()
	}
	dim := opts.EmbedDim
	if dim <= 0 {
		dim = 256
	}
	for id := range graph.Nodes {
		node, ok := graph.GetNode(id)
		if !ok {
			continue
		}
		graph.SetEmbedding(id, memory.EmbedText(node.CodeContent+"\n"+node.DocComment, dim))
	}
	topK := opts.TopK
	if topK <= 0 {
		topK = 5
	}
	depth := opts.GraphDepth
	if depth < 0 {
		depth = 0
	}
	seeds, err := memory.QueryContext(graph, memory.EmbedText(task, dim), topK, depth)
	if err != nil || len(seeds) == 0 {
		fmt.Printf("   [graph] vector search failed (%v); using flat context\n", err)
		return flat()
	}
	byID := make(map[string]gast.CodeChunk, len(chunks))
	for _, c := range chunks {
		byID[c.ID] = c
	}
	var out []gast.CodeChunk
	for _, s := range seeds {
		if c, ok := byID[s.ID]; ok {
			out = append(out, c)
		}
	}
	if len(out) == 0 {
		return flat()
	}
	fmt.Printf("   [graph] %d nodes, vector search + depth-%d expansion -> %d context chunks\n",
		len(graph.Nodes), depth, len(out))
	return out
}

// ---------------------------------------------------------------------------
// 2. Candidate generation
// ---------------------------------------------------------------------------

// generateCandidates asks the LLM for n diverse solutions to task.
// Candidates whose responses contain no file blocks are skipped; at
// least one usable candidate is required.
func generateCandidates(task, system string, n int) ([]map[string]string, error) {
	var out []map[string]string
	for i := 0; i < n; i++ {
		fmt.Printf("   candidate %d/%d: prompting LLM...\n", i+1, n)
		variant := fmt.Sprintf("%s\n\n(This is candidate %d of %d. Use a meaningfully different implementation approach from the other candidates — different data structures, algorithms, or code structure.)", task, i+1, n)
		raw, err := callLLM(system + "\nTask:\n" + variant + "\n")
		if err != nil {
			return nil, fmt.Errorf("pipeline: candidate %d: LLM call failed: %w", i, err)
		}
		files, err := parseFilesFromResponse(raw)
		if err != nil {
			fmt.Printf("   candidate %d: no file blocks (%v) — skipping\n", i, err)
			continue
		}
		out = append(out, formatCandidates(files))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("pipeline: no usable candidates generated")
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// 3. Symbolic safety proving
// ---------------------------------------------------------------------------

// proveCandidate runs the path-sensitive symbolic prover over every
// non-test Go file in files and returns the CRITICAL and WARNING
// counts plus human-readable violation lines. Files that do not parse
// are skipped here — the sandbox reports their syntax errors.
func proveCandidate(files map[string]string) (critical, warnings int, lines []string) {
	for _, path := range sortedKeys(files) {
		if !isTunableFile(path) {
			continue
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, path, files[path], parser.ParseComments)
		if err != nil {
			continue
		}
		info := &types.Info{
			Types: make(map[ast.Expr]types.TypeAndValue),
			Defs:  make(map[*ast.Ident]types.Object),
			Uses:  make(map[*ast.Ident]types.Object),
		}
		// Best-effort type information: even a failed Check leaves
		// partial facts, and the prover degrades gracefully.
		conf := types.Config{Importer: importer.Default(), Error: func(error) {}}
		_, _ = conf.Check("pipeline_candidate", fset, []*ast.File{f}, info)
		report, err := analysis.ProveSafetyWithFileSet(f, info, fset)
		if err != nil {
			continue
		}
		for _, v := range report.Violations {
			lines = append(lines, fmt.Sprintf("[%s] %s:%d %s", v.Severity, path, v.LineNumber, v.Message))
			switch v.Severity {
			case analysis.SeverityCritical:
				critical++
			case analysis.SeverityWarning:
				warnings++
			}
		}
	}
	return critical, warnings, lines
}

// proveAndFilter proves every candidate before execution. Candidates
// with CRITICAL violations are discarded; each surviving candidate's
// penalty is 25 per WARNING. If every candidate fails proving, none
// are discarded — instead each carries 100 per CRITICAL plus 25 per
// WARNING, so the least-bad still wins. Penalties align with the
// returned survivor slice.
func proveAndFilter(candidates []map[string]string) (penalties []int, survivors []map[string]string) {
	crit := make([]int, len(candidates))
	warn := make([]int, len(candidates))
	for i, c := range candidates {
		c, w, lines := proveCandidate(c)
		crit[i], warn[i] = c, w
		for _, l := range lines {
			fmt.Printf("   candidate-%d %s\n", i, l)
		}
		if c == 0 && w == 0 {
			fmt.Printf("   candidate-%d: clean ✅\n", i)
		}
	}
	for i, c := range candidates {
		if crit[i] == 0 {
			penalties = append(penalties, 25*warn[i])
			survivors = append(survivors, c)
		} else {
			fmt.Printf("   candidate-%d: %d CRITICAL violation(s) — discarded\n", i, crit[i])
		}
	}
	if len(survivors) == 0 {
		fmt.Println("   all candidates failed proving — keeping all with heavy penalties")
		for i, c := range candidates {
			penalties = append(penalties, 100*crit[i]+25*warn[i])
			survivors = append(survivors, c)
		}
	}
	return penalties, survivors
}

// scoredCandidate pairs an evaluated node with its safety-adjusted
// fitness and its index into the survivor/penalty slices.
type scoredCandidate struct {
	node  synthesis.CandidateNode
	final float64
	index int // survivor index, or -1 when the node ID is unrecognized
}

// survivorIndexOf resolves a ranked node's ID back to its index in the
// survivor/penalty slices. synthesis.EvaluateCandidatesRanked names
// nodes "candidate-N" in survivor order, so the penalty travels with
// the candidate no matter how evaluation reorders them.
func survivorIndexOf(id string) (int, bool) {
	const prefix = "candidate-"
	if !strings.HasPrefix(id, prefix) {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimPrefix(id, prefix))
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// pickBest applies safety penalties to the MCTS-ranked nodes and
// returns the highest safety-adjusted fitness (ID tiebreak).
func pickBest(ranked []synthesis.CandidateNode, penalties []int) scoredCandidate {
	scored := make([]scoredCandidate, 0, len(ranked))
	for _, node := range ranked {
		pen, idx := 0, -1
		if i, ok := survivorIndexOf(node.ID); ok && i < len(penalties) {
			pen, idx = penalties[i], i
		}
		final := node.Score - float64(pen)
		allocs := "n/a"
		if len(node.Errors) == 0 {
			allocs = fmt.Sprintf("%d", node.Metrics.AllocsPerOp)
		}
		scored = append(scored, scoredCandidate{node: node, final: final, index: idx})
		fmt.Printf("   %s: raw=%.1f safety=-%d final=%.1f allocs/op=%s errors=%d\n",
			node.ID, node.Score, pen, final, allocs, len(node.Errors))
	}
	sort.Slice(scored, func(a, b int) bool {
		if scored[a].final != scored[b].final {
			return scored[a].final > scored[b].final
		}
		return scored[a].node.ID < scored[b].node.ID
	})
	return scored[0]
}

// ---------------------------------------------------------------------------
// 5. Self-healing
// ---------------------------------------------------------------------------

// healWinner repairs a failing winner: its exact structured errors
// ([]runner.TestError with file/line/message) are fed back to the
// generator, up to maxAttempts times. Each repair is re-proved and
// re-validated; the first green repair wins. It returns the healed
// files, benchmark metrics (refreshed when the re-benchmark succeeds,
// otherwise the previous metrics), and the attempts used.
func healWinner(targetDir, task string, files map[string]string, firstErrs []runner.TestError, current *runner.BenchmarkMetrics, maxAttempts int) (map[string]string, *runner.BenchmarkMetrics, int, error) {
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	errs := firstErrs
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		fmt.Printf("   attempt %d/%d: %d error(s), re-prompting generator...\n", attempt, maxAttempts, len(errs))
		for _, e := range errs {
			fmt.Printf("     [%s] %s:%d: %s\n", e.Category, e.FilePath, e.LineNumber, e.Message)
		}
		raw, err := callLLM(buildRepairPrompt(task, files, errs))
		if err != nil {
			return nil, nil, attempt, fmt.Errorf("pipeline: heal attempt %d: LLM call failed: %w", attempt, err)
		}
		parsed, err := parseFilesFromResponse(raw)
		if err != nil {
			fmt.Printf("   attempt %d: repair had no file blocks (%v) — retrying\n", attempt, err)
			continue
		}
		files = formatCandidates(parsed)
		if crit, _, lines := proveCandidate(files); crit > 0 {
			fmt.Printf("   attempt %d: repair introduced %d CRITICAL violation(s) — retrying\n", attempt, crit)
			for _, l := range lines {
				fmt.Printf("     %s\n", l)
			}
			continue
		}
		res, err := runner.ValidateGeneratedCode(targetDir, files)
		if err != nil {
			return nil, nil, attempt, fmt.Errorf("pipeline: heal validation: %w", err)
		}
		if res.Passed {
			fmt.Printf("   healed on attempt %d ✅\n", attempt)
			if bm := benchmarkFiles(targetDir, files); bm != nil {
				current = bm
			}
			return files, current, attempt, nil
		}
		errs = res.Errors
	}
	return nil, nil, maxAttempts, fmt.Errorf("pipeline: self-healing exhausted %d attempts", maxAttempts)
}

func joinNames(names []string) string {
	out := ""
	for i, n := range names {
		if i > 0 {
			out += ", "
		}
		out += n
	}
	return out
}
