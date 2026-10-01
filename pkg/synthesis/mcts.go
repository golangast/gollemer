// Package synthesis evaluates candidate code patches with a Monte
// Carlo Tree Search discipline: each candidate is a node, the rollout
// is sandbox validation plus benchmark profiling, backpropagation
// writes the fitness score onto the node, and selection picks the
// highest-scoring candidate. Candidate generation (tree expansion) is
// the caller's job — typically an LLM producing CodeMap patches;
// this package scores them in parallel across isolated sandboxes.
package synthesis

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/golangast/gollemer/pkg/runner"
)

// DefaultBenchPattern selects which benchmarks a candidate is scored
// on. "." runs every benchmark in the sandbox and the first result is
// used for the fitness penalties.
const DefaultBenchPattern = "."

// benchmarkTimeout bounds a single candidate's benchmark phase. The
// runner enforces its own 30s cap; this tighter bound keeps the whole
// evaluation responsive. A timed-out benchmark is abandoned (its
// sandbox is reaped when the in-flight run finishes) and the
// candidate is scored on correctness alone.
const benchmarkTimeout = 15 * time.Second

// CandidateNode is one evaluated candidate: the proposed file map,
// its fitness score, any errors observed, and its benchmark metrics
// (zero when no benchmark ran).
type CandidateNode struct {
	ID      string
	CodeMap map[string]string
	Score   float64
	Errors  []string
	Metrics runner.BenchmarkMetrics
}

// EvaluationResult is the outcome of evaluating a candidate set:
// nodes sorted by descending fitness and the count evaluated.
type EvaluationResult struct {
	BestCandidate       CandidateNode
	EvaluatedNodesCount int
}

// CalculateFitness scores a candidate: +50 for passing tests, +20 for
// compiling with no syntax/compiler errors, minus 2 per allocation
// per op and minus ns/op divided by 1000. Nil arguments contribute
// nothing.
func CalculateFitness(res *runner.ExecutionResult, bench *runner.BenchmarkMetrics) float64 {
	score := 0.0
	if res != nil {
		if res.Passed {
			score += 50
		}
		compilesClean := true
		for _, e := range res.Errors {
			if e.Category == runner.CategoryCompiler || e.Category == runner.CategorySyntax {
				compilesClean = false
				break
			}
		}
		if compilesClean {
			score += 20
		}
	}
	if bench != nil {
		score -= float64(bench.AllocsPerOp) * 2
		score -= bench.NsPerOp / 1000
	}
	return score
}

// EvaluateCandidatesRanked runs every candidate through sandbox validation
// and benchmark profiling in parallel (at most runtime.NumCPU()
// concurrent evaluations), scores each with CalculateFitness, and
// returns all nodes sorted by descending score (ID tiebreak). Callers
// that need to apply their own penalties (e.g. safety) use this and
// re-rank; EvaluateCandidates is the convenience wrapper that returns
// only the best.
func EvaluateCandidatesRanked(ctx context.Context, targetDir string, candidates []map[string]string) ([]CandidateNode, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if targetDir == "" {
		return nil, fmt.Errorf("synthesis: empty targetDir")
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("synthesis: no candidates to evaluate")
	}
	if _, err := os.Stat(filepath.Join(targetDir, "go.mod")); err != nil {
		return nil, fmt.Errorf("synthesis: %q is not a Go module root: %w", targetDir, err)
	}

	g, gctx := errgroup.WithContext(ctx)
	g.SetLimit(runtime.NumCPU())

	resultsCh := make(chan *CandidateNode, len(candidates))
	for i, codeMap := range candidates {
		i, codeMap := i, codeMap
		g.Go(func() error {
			resultsCh <- evaluateCandidate(gctx, targetDir, fmt.Sprintf("candidate-%d", i), codeMap)
			return nil
		})
	}
	go func() {
		_ = g.Wait()
		close(resultsCh)
	}()

	var nodes []*CandidateNode
	for n := range resultsCh {
		nodes = append(nodes, n)
	}
	if err := g.Wait(); err != nil {
		return nil, fmt.Errorf("synthesis: evaluation failed: %w", err)
	}

	sort.Slice(nodes, func(i, j int) bool {
		if nodes[i].Score != nodes[j].Score {
			return nodes[i].Score > nodes[j].Score
		}
		return nodes[i].ID < nodes[j].ID
	})
	out := make([]CandidateNode, 0, len(nodes))
	for _, n := range nodes {
		out = append(out, *n)
	}
	return out, nil
}

// EvaluateCandidates runs every candidate through sandbox validation
// and benchmark profiling in parallel (at most runtime.NumCPU()
// concurrent evaluations), scores each with CalculateFitness, and
// returns the nodes sorted by descending score. A nil ctx is treated
// as context.Background. Individual candidate failures are recorded
// on their node; only evaluation-wide failures (bad targetDir, no
// candidates, worker errors) return an error.
func EvaluateCandidates(ctx context.Context, targetDir string, candidates []map[string]string) (*EvaluationResult, error) {
	nodes, err := EvaluateCandidatesRanked(ctx, targetDir, candidates)
	if err != nil {
		return nil, err
	}
	return &EvaluationResult{BestCandidate: nodes[0], EvaluatedNodesCount: len(nodes)}, nil
}

// evaluateCandidate runs one rollout: validate the candidate in a
// sandbox, then profile its benchmarks in a second sandbox.
func evaluateCandidate(ctx context.Context, targetDir, id string, codeMap map[string]string) *CandidateNode {
	node := &CandidateNode{ID: id, CodeMap: codeMap}
	if ctx.Err() != nil {
		node.Errors = []string{ctx.Err().Error()}
		return node
	}

	res, err := runner.ValidateGeneratedCode(targetDir, codeMap)
	if err != nil {
		node.Errors = []string{err.Error()}
		return node
	}
	if !res.Passed {
		for _, e := range res.Errors {
			node.Errors = append(node.Errors, e.Message)
		}
		node.Score = CalculateFitness(res, nil)
		return node
	}

	sandbox, cleanup, err := runner.CreateSandbox(targetDir, codeMap)
	if err != nil {
		node.Errors = []string{err.Error()}
		node.Score = CalculateFitness(res, nil)
		return node
	}

	type benchOutcome struct {
		metrics *runner.BenchmarkMetrics
		err     error
	}
	benchCh := make(chan benchOutcome, 1)
	go func() {
		m, berr := runner.RunBenchmarkValidation(sandbox, DefaultBenchPattern)
		benchCh <- benchOutcome{m, berr}
	}()

	benchCtx, cancel := context.WithTimeout(ctx, benchmarkTimeout)
	defer cancel()
	select {
	case out := <-benchCh:
		cleanup()
		if out.err != nil {
			node.Errors = append(node.Errors, "benchmark: "+out.err.Error())
			node.Score = CalculateFitness(res, nil)
			return node
		}
		node.Metrics = *out.metrics
		node.Score = CalculateFitness(res, out.metrics)
	case <-benchCtx.Done():
		// Report the timeout now; reap the sandbox once the
		// in-flight benchmark finishes (it carries the runner's
		// own 30s cap, so this is bounded).
		go func() { <-benchCh; cleanup() }()
		node.Errors = append(node.Errors, fmt.Sprintf("benchmark timed out after %s", benchmarkTimeout))
		node.Score = CalculateFitness(res, nil)
	}
	return node
}

/*
Runnable example: score candidate patches for a target module.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/golangast/gollemer/pkg/synthesis"
)

func main() {
	candidates := []map[string]string{
		{"sum.go": "package m\n\nfunc Sum(xs []int) int {\n\ts := 0\n\tfor _, x := range xs {\n\t\ts += x\n\t}\n\treturn s\n}\n"},
		{"sum.go": "package m\n\nfunc Sum(xs []int) int {\n\treturn 42 // wrong on purpose\n}\n"},
	}
	res, err := synthesis.EvaluateCandidates(context.Background(), "./target", candidates)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("evaluated %d candidates\n", res.EvaluatedNodesCount)
	fmt.Printf("best: %s score=%.2f errors=%v\n",
		res.BestCandidate.ID, res.BestCandidate.Score, res.BestCandidate.Errors)
}
*/
