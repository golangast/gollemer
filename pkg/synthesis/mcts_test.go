package synthesis

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/golangast/gollemer/pkg/runner"
)

// mctsFixture builds a target module: a unit test and a benchmark for
// Sum. The candidate under test supplies sum.go.
func mctsFixture(t *testing.T, benchBody string) string {
	t.Helper()
	root := t.TempDir()
	write := func(name, content string) {
		p := root + "/" + name
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/mctstarget\n\ngo 1.26.0\n")
	write("sum_test.go", `package target

import "testing"

func TestSum(t *testing.T) {
	if got := Sum([]int{1, 2, 3}); got != 6 {
		t.Fatalf("Sum = %d, want 6", got)
	}
}
`)
	write("sum_bench_test.go", "package target\n\nimport \"testing\"\n\nfunc BenchmarkSum(b *testing.B) {\n"+benchBody+"\n}\n")
	return root
}

func TestCalculateFitness(t *testing.T) {
	execRes := func(passed bool, cats ...string) *runner.ExecutionResult {
		r := &runner.ExecutionResult{Passed: passed}
		for _, c := range cats {
			r.Errors = append(r.Errors, runner.TestError{Category: c, Message: "x"})
		}
		return r
	}
	for _, tc := range []struct {
		name  string
		res   *runner.ExecutionResult
		bench *runner.BenchmarkMetrics
		want  float64
	}{
		{"pass clean fast", execRes(true), &runner.BenchmarkMetrics{AllocsPerOp: 2, NsPerOp: 1500}, 64.5},
		{"pass clean no bench", execRes(true), nil, 70},
		{"fail assertion only", execRes(false, runner.CategoryAssertion), nil, 20},
		{"fail compiler", execRes(false, runner.CategoryCompiler), nil, 0},
		{"fail syntax", execRes(false, runner.CategorySyntax), nil, 0},
		{"nil nil", nil, nil, 0},
		{"nil res with bench", nil, &runner.BenchmarkMetrics{AllocsPerOp: 1, NsPerOp: 500}, -2.5},
		{"pass with slow bench", execRes(true), &runner.BenchmarkMetrics{NsPerOp: 3000}, 67},
	} {
		if got := CalculateFitness(tc.res, tc.bench); got != tc.want {
			t.Errorf("%s: CalculateFitness = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestEvaluateCandidatesPicksBest(t *testing.T) {
	root := mctsFixture(t, `
		xs := []int{1, 2, 3, 4, 5}
		for i := 0; i < b.N; i++ {
			Sum(xs)
		}`)
	good := map[string]string{"sum.go": "package target\n\nfunc Sum(xs []int) int {\n\ts := 0\n\tfor _, x := range xs {\n\t\ts += x\n\t}\n\treturn s\n}\n"}
	alloc := map[string]string{"sum.go": "package target\n\nfunc Sum(xs []int) int {\n\ttmp := append([]int(nil), xs...)\n\ts := 0\n\tfor _, x := range tmp {\n\t\ts += x\n\t}\n\treturn s\n}\n"}
	wrong := map[string]string{"sum.go": "package target\n\nfunc Sum(xs []int) int {\n\treturn 42\n}\n"}
	broken := map[string]string{"sum.go": "package target\n\nfunc Sum(xs []int) int {\n"}

	res, err := EvaluateCandidates(context.Background(), root, []map[string]string{good, alloc, wrong, broken})
	if err != nil {
		t.Fatalf("EvaluateCandidates: %v", err)
	}
	if res.EvaluatedNodesCount != 4 {
		t.Errorf("EvaluatedNodesCount = %d, want 4", res.EvaluatedNodesCount)
	}
	best := res.BestCandidate
	if best.ID != "candidate-0" {
		t.Errorf("BestCandidate.ID = %q, want candidate-0", best.ID)
	}
	if best.Score < 60 {
		t.Errorf("BestCandidate.Score = %v, want >= 60 (50 pass + 20 clean)", best.Score)
	}
	if len(best.Errors) != 0 {
		t.Errorf("BestCandidate.Errors = %v, want none", best.Errors)
	}
	if best.Metrics.AllocsPerOp != 0 {
		t.Errorf("BestCandidate allocs = %d, want 0", best.Metrics.AllocsPerOp)
	}
}

func TestEvaluateCandidatesErrors(t *testing.T) {
	root := mctsFixture(t, `for i := 0; i < b.N; i++ {}`)
	good := map[string]string{"sum.go": "package target\n\nfunc Sum(xs []int) int { return 0 }\n"}
	if _, err := EvaluateCandidates(context.Background(), "", []map[string]string{good}); err == nil {
		t.Error("expected error for empty targetDir")
	}
	if _, err := EvaluateCandidates(context.Background(), root, nil); err == nil {
		t.Error("expected error for no candidates")
	}
	if _, err := EvaluateCandidates(context.Background(), t.TempDir(), []map[string]string{good}); err == nil {
		t.Error("expected error for targetDir without go.mod")
	}
	// A nil context is tolerated.
	if _, err := EvaluateCandidates(nil, root, []map[string]string{good}); err != nil {
		t.Errorf("nil ctx: %v", err)
	}
}

func TestEvaluateCandidatesBenchmarkTimeout(t *testing.T) {
	root := mctsFixture(t, `time.Sleep(20 * time.Second)`)
	// The bench file needs the time import.
	p := root + "/sum_bench_test.go"
	content, _ := os.ReadFile(p)
	s := strings.Replace(string(content), `"testing"`, "\"testing\"\n\nimport \"time\"", 1)
	if err := os.WriteFile(p, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
	good := map[string]string{"sum.go": "package target\n\nfunc Sum(xs []int) int {\n\ts := 0\n\tfor _, x := range xs {\n\t\ts += x\n\t}\n\treturn s\n}\n"}

	start := time.Now()
	res, err := EvaluateCandidates(context.Background(), root, []map[string]string{good})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("EvaluateCandidates: %v", err)
	}
	if res.EvaluatedNodesCount != 1 {
		t.Errorf("EvaluatedNodesCount = %d, want 1", res.EvaluatedNodesCount)
	}
	found := false
	for _, e := range res.BestCandidate.Errors {
		if strings.Contains(e, "timed out") {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a timeout error, got %v", res.BestCandidate.Errors)
	}
	// The 15s synthesis bound must fire well before the runner's 30s cap.
	if elapsed > 25*time.Second {
		t.Errorf("evaluation took %v, want < 25s (15s benchmark timeout)", elapsed)
	}
}
