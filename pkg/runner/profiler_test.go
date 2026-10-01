package runner

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// benchFixture builds a module with three benchmarks:
// BenchmarkAdd (no allocs), BenchmarkAlloc (1 alloc of 128 B/op),
// BenchmarkFlaky (reports metrics, then fails via b.Error).
func benchFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(name, content string) {
		p := filepath.Join(root, name)
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/benchmine\n\ngo 1.26.0\n")
	write("bench_test.go", `package bench

import "testing"

func Add(a, b int) int { return a + b }

func BenchmarkAdd(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Add(1, 2)
	}
}

var sink []byte

func BenchmarkAlloc(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sink = make([]byte, 128)
	}
}

func BenchmarkFlaky(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Add(1, 2)
	}
	b.Error("intentional failure")
}
`)
	return root
}

func TestRunBenchmarkValidationNoAllocs(t *testing.T) {
	root := benchFixture(t)
	m, err := RunBenchmarkValidation(root, "BenchmarkAdd")
	if err != nil {
		t.Fatalf("RunBenchmarkValidation: %v", err)
	}
	if m.Name != "BenchmarkAdd" {
		t.Errorf("Name = %q, want BenchmarkAdd", m.Name)
	}
	if m.N <= 0 {
		t.Errorf("N = %d, want > 0", m.N)
	}
	if m.NsPerOp < 0 {
		t.Errorf("NsPerOp = %v, want >= 0", m.NsPerOp)
	}
	if m.AllocBytesPerOp != 0 || m.AllocsPerOp != 0 {
		t.Errorf("allocs = %d B/op x %d, want 0 x 0", m.AllocBytesPerOp, m.AllocsPerOp)
	}
	if !m.Passed {
		t.Error("Passed = false, want true")
	}
	fb := m.FormatLLMFeedback()
	for _, want := range []string{
		"Benchmark Performance Report:",
		"ns/op",
		"0 B/op across 0 allocations",
		"already optimal",
	} {
		if !strings.Contains(fb, want) {
			t.Errorf("feedback missing %q:\n%s", want, fb)
		}
	}
}

func TestRunBenchmarkValidationWithAllocs(t *testing.T) {
	root := benchFixture(t)
	m, err := RunBenchmarkValidation(root, "BenchmarkAlloc")
	if err != nil {
		t.Fatalf("RunBenchmarkValidation: %v", err)
	}
	if m.Name != "BenchmarkAlloc" {
		t.Errorf("Name = %q, want BenchmarkAlloc", m.Name)
	}
	if m.AllocsPerOp != 1 {
		t.Errorf("AllocsPerOp = %d, want 1", m.AllocsPerOp)
	}
	if m.AllocBytesPerOp != 128 {
		t.Errorf("AllocBytesPerOp = %d, want 128", m.AllocBytesPerOp)
	}
	if !m.Passed {
		t.Error("Passed = false, want true")
	}
	fb := m.FormatLLMFeedback()
	if !strings.Contains(fb, "128 B/op across 1 allocations") {
		t.Errorf("feedback missing allocation line:\n%s", fb)
	}
	if !strings.Contains(fb, "avoiding slice re-allocations or heap escapes") {
		t.Errorf("feedback missing optimization recommendation:\n%s", fb)
	}
}

func TestRunBenchmarkValidationFailed(t *testing.T) {
	root := benchFixture(t)
	m, err := RunBenchmarkValidation(root, "BenchmarkFlaky")
	if err != nil {
		t.Fatalf("RunBenchmarkValidation: %v", err)
	}
	if m.Name != "BenchmarkFlaky" {
		t.Errorf("Name = %q, want BenchmarkFlaky", m.Name)
	}
	if m.Passed {
		t.Error("Passed = true, want false for a failing benchmark")
	}
	if fb := m.FormatLLMFeedback(); !strings.Contains(fb, "FAILED") {
		t.Errorf("feedback should flag the failure:\n%s", fb)
	}
}

func TestRunBenchmarkValidationErrors(t *testing.T) {
	root := benchFixture(t)
	if _, err := RunBenchmarkValidation("", "BenchmarkAdd"); err == nil {
		t.Error("expected error for empty sandboxDir")
	}
	if _, err := RunBenchmarkValidation(root, ""); err == nil {
		t.Error("expected error for empty benchPattern")
	}
	if _, err := RunBenchmarkValidation(filepath.Join(root, "nope"), "BenchmarkAdd"); err == nil {
		t.Error("expected error for missing directory")
	}
	nogomod := t.TempDir()
	if _, err := RunBenchmarkValidation(nogomod, "BenchmarkAdd"); err == nil {
		t.Error("expected error for directory without go.mod")
	}
	if _, err := RunBenchmarkValidation(root, "BenchmarkDoesNotExist"); err == nil {
		t.Error("expected error for unmatched pattern")
	} else if !strings.Contains(err.Error(), "BenchmarkDoesNotExist") {
		t.Errorf("error should name the pattern, got: %v", err)
	}
}

func TestParseBenchmarkLine(t *testing.T) {
	for _, tc := range []struct {
		in    string
		name  string
		n     int
		ns    float64
		bop   int64
		aop   int64
		isNil bool
	}{
		{"BenchmarkAdd-8  \t1000000\t   123.4 ns/op\t    48 B/op\t       2 allocs/op\n", "BenchmarkAdd", 1000000, 123.4, 48, 2, false},
		{"BenchmarkAdd-8  \t1000000\t   123.4 ns/op\t       0 B/op\t       0 allocs/op\n", "BenchmarkAdd", 1000000, 123.4, 0, 0, false},
		{"BenchmarkAdd  \t10\t   5.5 ns/op\n", "BenchmarkAdd", 10, 5.5, 0, 0, false},
		{"BenchmarkAdd/sub-8  \t100\t   1.2 ns/op\t    12.3 MB/s\t    16 B/op\t       1 allocs/op\n", "BenchmarkAdd/sub", 100, 1.2, 16, 1, false},
		{"    BenchmarkAdd-8  \t100\t   1.2 ns/op\n", "BenchmarkAdd", 100, 1.2, 0, 0, false},
		{"PASS\n", "", 0, 0, 0, 0, true},
		{"ok  \texample.com/benchmine\t1.234s\n", "", 0, 0, 0, 0, true},
		{"--- FAIL: BenchmarkFlaky-8 (0.00s)\n", "", 0, 0, 0, 0, true},
		{"", "", 0, 0, 0, 0, true},
	} {
		m := parseBenchmarkLine(tc.in)
		if tc.isNil {
			if m != nil {
				t.Errorf("parseBenchmarkLine(%q) = %+v, want nil", tc.in, m)
			}
			continue
		}
		if m == nil {
			t.Errorf("parseBenchmarkLine(%q) = nil, want %+v", tc.in, tc)
			continue
		}
		if m.Name != tc.name || m.N != tc.n || m.NsPerOp != tc.ns || m.AllocBytesPerOp != tc.bop || m.AllocsPerOp != tc.aop {
			t.Errorf("parseBenchmarkLine(%q) = %+v, want name=%s n=%d ns=%v bop=%d aop=%d",
				tc.in, m, tc.name, tc.n, tc.ns, tc.bop, tc.aop)
		}
	}
}

func TestFormatLLMFeedbackNil(t *testing.T) {
	var m *BenchmarkMetrics
	if fb := m.FormatLLMFeedback(); fb != "" {
		t.Errorf("nil feedback = %q, want empty", fb)
	}
}
