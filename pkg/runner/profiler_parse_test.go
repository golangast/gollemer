package runner

import (
	"strings"
	"testing"
)

// The testing framework prints the benchmark name before running and
// the numbers after; under load those arrive as separate `go test
// -json` Output events. Metrics must be recovered from the joined
// chunks, not from a single event.
func TestParseBenchmarkOutputSplitChunks(t *testing.T) {
	stream := strings.Join([]string{
		`{"Action":"run","Package":"p","Test":"BenchmarkAdd"}`,
		`{"Action":"output","Package":"p","Test":"BenchmarkAdd","Output":"=== RUN   BenchmarkAdd\n"}`,
		`{"Action":"output","Package":"p","Test":"BenchmarkAdd","Output":"BenchmarkAdd-8   \t"}`,
		`{"Action":"output","Package":"p","Test":"BenchmarkAdd","Output":"1000000\t   123.4 ns/op\t    48 B/op\t       2 allocs/op\n"}`,
		`{"Action":"pass","Package":"p","Test":"BenchmarkAdd"}`,
		`{"Action":"output","Package":"p","Output":"PASS\n"}`,
	}, "\n")
	results, failed, matched := parseBenchmarkOutput(stream, "BenchmarkAdd")
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	m := results[0]
	if m.Name != "BenchmarkAdd" || m.N != 1000000 || m.NsPerOp != 123.4 ||
		m.AllocBytesPerOp != 48 || m.AllocsPerOp != 2 {
		t.Errorf("wrong metrics: %+v", m)
	}
	if len(matched) != 1 || matched[0] != "BenchmarkAdd" {
		t.Errorf("matched = %v, want [BenchmarkAdd]", matched)
	}
	if len(failed) != 0 {
		t.Errorf("failed = %v, want empty", failed)
	}
}

// A failing benchmark's name must still be reported as matched so the
// caller can distinguish "ran and failed" from "no match".
func TestParseBenchmarkOutputFailedBenchmark(t *testing.T) {
	stream := strings.Join([]string{
		`{"Action":"run","Package":"p","Test":"BenchmarkFlaky"}`,
		`{"Action":"output","Package":"p","Test":"BenchmarkFlaky","Output":"    bench_test.go:25: boom\n"}`,
		`{"Action":"fail","Package":"p","Test":"BenchmarkFlaky"}`,
	}, "\n")
	results, failed, matched := parseBenchmarkOutput(stream, "BenchmarkFlaky")
	if len(results) != 0 {
		t.Errorf("got %d results, want 0", len(results))
	}
	if !failed["BenchmarkFlaky"] {
		t.Errorf("failed = %v, want BenchmarkFlaky marked", failed)
	}
	if len(matched) != 1 || matched[0] != "BenchmarkFlaky" {
		t.Errorf("matched = %v, want [BenchmarkFlaky]", matched)
	}
}
