// Benchmark profiling: run benchmarks against candidate Go code and
// extract memory allocation metrics for LLM optimization feedback.
//
// RunBenchmarkValidation executes `go test -bench=<pattern> -benchmem
// -json` inside a sandbox directory, parses the JSON event stream for
// benchmark result lines, and returns the first match's metrics.
// Benchmarks that fail still report their metrics with Passed=false;
// a pattern matching nothing is an error. Like the sandbox runner,
// the benchmark tree runs in its own process group and is reaped on
// the 30-second timeout.
package runner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// benchmarkTimeout caps a benchmark validation run.
const benchmarkTimeout = 30 * time.Second

// BenchmarkMetrics holds the measured performance of one benchmark.
type BenchmarkMetrics struct {
	Name            string  `json:"name"`
	N               int     `json:"n"`
	NsPerOp         float64 `json:"nsPerOp"`
	AllocBytesPerOp int64   `json:"allocBytesPerOp"`
	AllocsPerOp     int64   `json:"allocsPerOp"`
	Passed          bool    `json:"passed"`
}

// FormatLLMFeedback converts the metrics into structured prompt
// feedback for an LLM optimization pass.
func (m *BenchmarkMetrics) FormatLLMFeedback() string {
	if m == nil {
		return ""
	}
	var b strings.Builder
	b.WriteString("Benchmark Performance Report:\n")
	if !m.Passed {
		b.WriteString("- Result: FAILED (metrics may be incomplete)\n")
	}
	fmt.Fprintf(&b, "- Speed: %g ns/op\n", m.NsPerOp)
	fmt.Fprintf(&b, "- Memory: %d B/op across %d allocations\n", m.AllocBytesPerOp, m.AllocsPerOp)
	if m.AllocsPerOp > 0 {
		b.WriteString("Recommendation: optimize memory allocation by avoiding slice re-allocations or heap escapes.\n")
	} else {
		b.WriteString("Recommendation: no allocations detected; allocation behavior is already optimal.\n")
	}
	return b.String()
}

// RunBenchmarkValidation runs `go test -bench=<benchPattern>
// -benchmem -json ./...` in sandboxDir and returns the metrics of the
// first benchmark matching the pattern. A benchmark that ran but
// reported no metrics (failed before finishing) yields zero-valued
// metrics with Passed=false; a pattern matching nothing is an error.
func RunBenchmarkValidation(sandboxDir string, benchPattern string) (*BenchmarkMetrics, error) {
	if sandboxDir == "" {
		return nil, fmt.Errorf("runner: empty sandboxDir")
	}
	if benchPattern == "" {
		return nil, fmt.Errorf("runner: empty benchPattern")
	}
	info, err := os.Stat(sandboxDir)
	if err != nil {
		return nil, fmt.Errorf("runner: sandboxDir: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("runner: sandboxDir %q is not a directory", sandboxDir)
	}
	if _, err := os.Stat(filepath.Join(sandboxDir, "go.mod")); err != nil {
		return nil, fmt.Errorf("runner: no go.mod in %q: sandboxDir must be a Go module root", sandboxDir)
	}

	ctx, cancel := context.WithTimeout(context.Background(), benchmarkTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "go", "test", "-bench="+benchPattern, "-benchmem", "-json", "./...")
	cmd.Dir = sandboxDir
	// New process group so a timeout reaps the whole benchmark tree,
	// not just the `go` process. Unix-specific; this package targets
	// unix environments.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	runErr := cmd.Run()
	var execErr *exec.Error
	if errors.As(runErr, &execErr) {
		return nil, fmt.Errorf("runner: go toolchain not available: %w", execErr)
	}
	timedOut := ctx.Err() == context.DeadlineExceeded
	if timedOut && cmd.Process != nil {
		// The leader is already dead via CommandContext; kill the
		// rest of the group (runaway benchmark binary).
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
	if timedOut {
		return nil, fmt.Errorf("runner: benchmark timed out after %s", benchmarkTimeout)
	}

	results, failed, matched := parseBenchmarkOutput(stdout.String(), benchPattern)
	if len(results) == 0 {
		if len(matched) > 0 {
			// The benchmark ran but reported no metrics (e.g. it
			// failed before finishing): report the failure.
			return &BenchmarkMetrics{Name: matched[0], Passed: false}, nil
		}
		msg := lastMeaningfulLine(stdout.String() + "\n" + stderr.String())
		if msg == "" {
			msg = "no output"
		}
		return nil, fmt.Errorf("runner: no benchmarks matched pattern %q: %s", benchPattern, msg)
	}
	m := results[0]
	m.Passed = runErr == nil && !failed[m.Name]
	return m, nil
}

var (
	// BenchmarkAdd-8 \t 1000000 \t 123.4 ns/op ...
	benchResultRe = regexp.MustCompile(`^(Benchmark\S+?)(?:-\d+)?\s+(\d+)\s+([\d.]+)\s+ns/op`)
	benchBytesRe  = regexp.MustCompile(`(\d+)\s+B/op`)
	benchAllocsRe = regexp.MustCompile(`(\d+)\s+allocs/op`)
)

// parseBenchmarkOutput extracts benchmark metrics from a `go test
// -json` stream, records per-benchmark failures by base name, and
// lists the test names matching benchPattern in first-seen order
// (mirroring how -bench matches benchmark names).
//
// Output events are accumulated per test before parsing: the testing
// framework prints the benchmark name before running and the numbers
// after, so under load the result line can arrive as two (or more)
// separate Output events.
func parseBenchmarkOutput(text, benchPattern string) (results []*BenchmarkMetrics, failed map[string]bool, matched []string) {
	failed = map[string]bool{}
	seen := map[string]bool{}
	outputs := map[string]*strings.Builder{}
	benchRe, _ := regexp.Compile(benchPattern)
	for _, line := range strings.Split(text, "\n") {
		var ev testEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		if ev.Test != "" && benchRe != nil && benchRe.MatchString(ev.Test) && !seen[ev.Test] {
			seen[ev.Test] = true
			matched = append(matched, ev.Test)
		}
		switch ev.Action {
		case "fail":
			if ev.Test != "" {
				failed[ev.Test] = true
			}
		case "output":
			sb, ok := outputs[ev.Test]
			if !ok {
				sb = &strings.Builder{}
				outputs[ev.Test] = sb
			}
			sb.WriteString(ev.Output)
		}
	}
	for _, name := range matched {
		sb, ok := outputs[name]
		if !ok {
			continue
		}
		for _, l := range strings.Split(sb.String(), "\n") {
			if m := parseBenchmarkLine(l); m != nil {
				results = append(results, m)
			}
		}
	}
	return results, failed, matched
}

// parseBenchmarkLine parses one benchmark result line, e.g.
// "BenchmarkAdd-8  1000000  123.4 ns/op  48 B/op  2 allocs/op".
// The -N GOMAXPROCS suffix is stripped; MB/s columns are tolerated.
// Returns nil for non-result lines.
func parseBenchmarkLine(output string) *BenchmarkMetrics {
	line := strings.TrimSpace(output)
	parts := benchResultRe.FindStringSubmatch(line)
	if parts == nil {
		return nil
	}
	n, _ := strconv.Atoi(parts[2])
	nsPerOp, _ := strconv.ParseFloat(parts[3], 64)
	m := &BenchmarkMetrics{Name: parts[1], N: n, NsPerOp: nsPerOp}
	if b := benchBytesRe.FindStringSubmatch(line); b != nil {
		m.AllocBytesPerOp, _ = strconv.ParseInt(b[1], 10, 64)
	}
	if a := benchAllocsRe.FindStringSubmatch(line); a != nil {
		m.AllocsPerOp, _ = strconv.ParseInt(a[1], 10, 64)
	}
	return m
}

/*
Runnable example: parse benchmark metrics from a sandbox directory.

package main

import (
	"fmt"
	"log"

	"github.com/golangast/gollemer/pkg/runner"
)

func main() {
	m, err := runner.RunBenchmarkValidation("./mycandidate", "BenchmarkAdd")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("benchmark: %s (n=%d, passed=%v)\n", m.Name, m.N, m.Passed)
	fmt.Print(m.FormatLLMFeedback())
}
*/
