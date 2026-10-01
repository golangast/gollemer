package chat

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"
)

// ShellDomain tags replies produced by the /shell command.
const ShellDomain = "shell"

// shellTimeout bounds one /shell invocation.
const shellTimeout = 90 * time.Second

// runShellOnce executes one shell prompt through the standalone REPL
// pipeline (cmd/gollemer) and returns its terminal output. It is a var
// so tests can stub it. The prompt travels as a single argv element —
// no shell is involved, so it cannot inject commands.
var runShellOnce = func(prompt string) (string, error) {
	dir := unifiedProjectRoot
	if dir == "" {
		dir = "."
	}
	ctx, cancel := context.WithTimeout(context.Background(), shellTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "run", "./cmd/gollemer", "-once", prompt)
	cmd.Dir = dir
	var out, errb bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errb
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("shell: %v: %s", err, strings.TrimSpace(errb.String()))
	}
	return out.String(), nil
}

// handleShellCommand handles "/shell <prompt>": it runs the prompt
// through the reactive pipeline (synthesis, safety, tuning, trace)
// and returns the rendered terminal output.
func handleShellCommand(line string) (string, bool) {
	if line != "/shell" && !strings.HasPrefix(line, "/shell ") {
		return "", false
	}
	prompt := strings.TrimSpace(strings.TrimPrefix(line, "/shell"))
	if prompt == "" {
		return "usage: /shell <prompt> — e.g. /shell build a list of squares", true
	}
	out, err := runShellOnce(prompt)
	if err != nil {
		return fmt.Sprintf("shell error: %v", err), true
	}
	return strings.TrimRight(out, "\n"), true
}
