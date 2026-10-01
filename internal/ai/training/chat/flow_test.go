package chat

import (
	"strings"
	"testing"
)

func TestHandleFlowCommand(t *testing.T) {
	out, ok := handleFlowCommand("/flow create a worker pool")
	if !ok {
		t.Fatal("want ok=true for /flow with a prompt")
	}
	for _, want := range []string{
		"Here's your Go program:",
		"```go",
		"Think of it like this:",
		"What it does, step by step:",
		"✅ Safety: checked",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("flow output missing %q\n---\n%s", want, out)
		}
	}
}

func TestHandleFlowCommandUsage(t *testing.T) {
	out, ok := handleFlowCommand("/flow")
	if !ok {
		t.Fatal("want ok=true for bare /flow")
	}
	if !strings.Contains(out, "usage:") {
		t.Errorf("want usage line, got %q", out)
	}
}

func TestHandleFlowCommandNotMatched(t *testing.T) {
	for _, line := range []string{"/shell x", "flow create a worker pool", "hello"} {
		if _, ok := handleFlowCommand(line); ok {
			t.Errorf("handleFlowCommand(%q) = ok, want false", line)
		}
	}
}

func TestHandleFlowCommandUnknownPrompt(t *testing.T) {
	out, ok := handleFlowCommand("/flow teleport a sandwich")
	if !ok {
		t.Fatal("want ok=true even for unknown prompts")
	}
	if !strings.Contains(out, "flow error:") {
		t.Errorf("want a flow error for an unknown prompt, got %q", out)
	}
}
