package engine

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/pkg/xray"
)

func TestRenderBeginner(t *testing.T) {
	r := &PipelineResult{
		FinalCode:        "package main\n\nfunc main() {}\n",
		IsSafetyVerified: true,
		Analogy:          "A light switch: flipping it runs the program.",
		VisualSteps: []xray.ExecutionStep{
			{StepNumber: 1, Title: "Step 1: Main", Description: "Main — entry point."},
			{StepNumber: 2, Title: "Step 2: Helper", Description: "Helper."},
		},
		PerformanceDelta: "No unbounded allocations found — already at 0 allocs/op",
	}
	out := r.RenderBeginner()
	for _, want := range []string{
		"Here's your Go program:",
		"```go",
		"func main() {}",
		"Think of it like this: A light switch",
		"What it does, step by step:",
		"1. Main — entry point.",
		"2. Helper",
		"✅ Safety: checked",
		"⚡ Speed: already lean",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("RenderBeginner missing %q\n---\n%s", want, out)
		}
	}
	if strings.Contains(out, "Step 1:") {
		t.Errorf("RenderBeginner leaked raw step prefix:\n%s", out)
	}
}

func TestRenderBeginnerUnverified(t *testing.T) {
	r := &PipelineResult{FinalCode: "package main\n", IsSafetyVerified: false}
	out := r.RenderBeginner()
	if !strings.Contains(out, "⚠️ Safety: couldn't fully verify") {
		t.Errorf("want unverified safety line, got:\n%s", out)
	}
}
