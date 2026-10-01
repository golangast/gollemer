package engine

import (
	"context"
	"go/format"
	"strings"
	"testing"

	gast "github.com/golangast/gollemer/pkg/ast"
)

func TestExecutePipelineWorkerPool(t *testing.T) {
	res, err := ExecutePipeline(context.Background(),
		"Create a worker pool using channels and WaitGroup", nil)
	if err != nil {
		t.Fatalf("ExecutePipeline: %v", err)
	}
	if !strings.Contains(res.FinalCode, "func main()") {
		t.Errorf("FinalCode missing main:\n%s", res.FinalCode)
	}
	// Must be gofmt-stable.
	if got := string(mustFormat(t, res.FinalCode)); got != res.FinalCode {
		t.Error("FinalCode is not gofmt-stable")
	}
	if !res.IsSafetyVerified {
		t.Error("IsSafetyVerified = false, want true for the worker pool template")
	}
	wantAnalogy := "A conveyor belt: moves data safely between workers without needing explicit locks."
	if res.Analogy != wantAnalogy {
		t.Errorf("Analogy = %q, want %q", res.Analogy, wantAnalogy)
	}
	if len(res.VisualSteps) == 0 {
		t.Fatal("VisualSteps empty")
	}
	if res.VisualSteps[0].Title == "" || res.VisualSteps[0].StepNumber != 1 {
		t.Errorf("first step malformed: %+v", res.VisualSteps[0])
	}
	if !strings.Contains(res.PerformanceDelta, "allocs/op") {
		t.Errorf("PerformanceDelta = %q", res.PerformanceDelta)
	}
	t.Logf("delta: %s", res.PerformanceDelta)
	for _, s := range res.VisualSteps {
		t.Logf("step %d: %s", s.StepNumber, s.Title)
	}
}

func TestExecutePipelineCodebaseContext(t *testing.T) {
	codeCtx := &gast.CodebaseContext{PackageName: "worker"}
	res, err := ExecutePipeline(context.Background(),
		"Create a worker pool using channels and WaitGroup", codeCtx)
	if err != nil {
		t.Fatalf("ExecutePipeline: %v", err)
	}
	for _, s := range res.VisualSteps {
		if s.FilePath != "worker.go" {
			t.Errorf("FilePath = %q, want worker.go", s.FilePath)
		}
	}
	// Invalid package names fall back to pipeline.go, never crash.
	bad := &gast.CodebaseContext{PackageName: "not a package!"}
	res, err = ExecutePipeline(context.Background(), "Create a worker pool", bad)
	if err != nil {
		t.Fatalf("ExecutePipeline: %v", err)
	}
	for _, s := range res.VisualSteps {
		if s.FilePath != "pipeline.go" {
			t.Errorf("FilePath = %q, want pipeline.go", s.FilePath)
		}
	}
}

func TestExecutePipelineErrors(t *testing.T) {
	if _, err := ExecutePipeline(context.Background(), "   ", nil); err == nil {
		t.Error("empty prompt: want error")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := ExecutePipeline(ctx, "Create a worker pool", nil); err == nil {
		t.Error("cancelled context: want error")
	}
	if _, err := ExecutePipeline(context.Background(), "knit me a sweater", nil); err == nil {
		t.Error("unknown prompt: want error")
	}
}

func TestFormatDelta(t *testing.T) {
	if got := formatDelta(4, 0); got != "Reduced allocations from 4 to 0 allocs/op" {
		t.Errorf("formatDelta(4, 0) = %q", got)
	}
	if got := formatDelta(0, 0); !strings.Contains(got, "0 allocs/op") {
		t.Errorf("formatDelta(0, 0) = %q", got)
	}
}

func TestDetectPrimitive(t *testing.T) {
	cases := map[string]string{
		"jobs := make(chan int)":            "channel",
		"go worker(jobs)":                   "goroutine",
		"type S struct{ X int }":            "struct",
		"type I interface{ M() }":           "interface",
		"var m sync.Mutex":                  "mutex",
		"defer f.Close()":                   "defer",
		"xs = append(xs, 1)":                "slice",
		"m := map[string]int{}":             "map",
		"select {\ncase x := <-c:\n}":       "select",
		"var wg sync.WaitGroup":             "waitgroup",
		"ctx, cancel := context.WithCancel": "context",
		"fmt.Println(1)":                    "function",
	}
	for code, want := range cases {
		if got := detectPrimitive(code, ""); got != want {
			t.Errorf("detectPrimitive(%q) = %q, want %q", code, got, want)
		}
	}
}

func mustFormat(t *testing.T, src string) []byte {
	t.Helper()
	out, err := format.Source([]byte(src))
	if err != nil {
		t.Fatalf("format: %v", err)
	}
	return out
}
