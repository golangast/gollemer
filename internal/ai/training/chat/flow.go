package chat

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/golangast/gollemer/pkg/engine"
)

// FlowDomain tags replies produced by the /flow command.
const FlowDomain = "flow"

// flowTimeout bounds one /flow invocation.
const flowTimeout = 60 * time.Second

// handleFlowCommand handles "/flow <prompt>": it runs the prompt through
// the reactive engine pipeline (synthesis, safety proving, auto-tuning,
// visual trace) in-process and returns the beginner-friendly rendering.
// The engine is imported directly: no subprocess, no shell.
func handleFlowCommand(line string) (string, bool) {
	if line != "/flow" && !strings.HasPrefix(line, "/flow ") {
		return "", false
	}
	prompt := strings.TrimSpace(strings.TrimPrefix(line, "/flow"))
	if prompt == "" {
		return "usage: /flow <prompt> — e.g. /flow create a worker pool", true
	}
	ctx, cancel := context.WithTimeout(context.Background(), flowTimeout)
	defer cancel()
	res, err := engine.ExecutePipeline(ctx, prompt, nil)
	if err != nil {
		return fmt.Sprintf("flow error: %v", err), true
	}
	return res.RenderBeginner(), true
}
