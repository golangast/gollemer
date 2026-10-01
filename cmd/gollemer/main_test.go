package main

import (
	"bytes"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func TestSynthesizeAllIntents(t *testing.T) {
	prompts := map[string]string{
		"create a worker pool":     "worker pool",
		"write an http server":     "http server",
		"read a file line by line": "read file",
		"encode a struct to json":  "json",
		"make a mutex counter":     "mutex counter",
		"run a ticker":             "ticker",
		"build a list of squares":  "build a list",
	}
	for prompt, want := range prompts {
		code, pattern, err := synthesize(prompt)
		if err != nil {
			t.Errorf("synthesize(%q): %v", prompt, err)
			continue
		}
		if pattern != want {
			t.Errorf("synthesize(%q) pattern = %q, want %q", prompt, pattern, want)
		}
		if _, ok := analogies[pattern]; !ok {
			t.Errorf("no analogy for pattern %q", pattern)
		}
		fmtd, err := format.Source([]byte(code))
		if err != nil {
			t.Errorf("synthesize(%q): format: %v", prompt, err)
			continue
		}
		if string(fmtd) != code {
			t.Errorf("synthesize(%q): template not gofmt-stable", prompt)
		}
		if _, err := parser.ParseFile(token.NewFileSet(), "t.go", code, 0); err != nil {
			t.Errorf("synthesize(%q): parse: %v", prompt, err)
		}
	}
}

func TestSynthesizeUnknown(t *testing.T) {
	if _, _, err := synthesize("knit me a sweater"); err == nil {
		t.Error("want error for unknown prompt")
	} else if !strings.Contains(err.Error(), "worker pool") {
		t.Errorf("error should list capabilities, got: %v", err)
	}
}

func TestRunPipelineStages(t *testing.T) {
	out, err := runPipeline("write an http server")
	if err != nil {
		t.Fatal(err)
	}
	if out.Pattern != "http server" {
		t.Errorf("pattern = %q", out.Pattern)
	}
	if !strings.Contains(out.Code, "package main") {
		t.Error("code missing package clause")
	}
	if out.Analogy == "" {
		t.Error("empty analogy")
	}
	if !out.Safety.Verified || out.Safety.Panics != 0 {
		t.Errorf("http server should verify: %+v", out.Safety)
	}
	if len(out.Steps) != 2 || out.Steps[0].Title != "Main" || out.Steps[1].Title != "Health" {
		t.Errorf("steps = %+v", out.Steps)
	}
	if !strings.Contains(out.Steps[0].Subtitle, "package main in repl.go") {
		t.Errorf("subtitle = %q", out.Steps[0].Subtitle)
	}
}

func TestRunPipelinePrealloc(t *testing.T) {
	out, err := runPipeline("build a list of squares")
	if err != nil {
		t.Fatal(err)
	}
	if out.Perf.AllocsBefore != 1 || out.Perf.AllocsAfter != 0 {
		t.Errorf("allocs = %d -> %d, want 1 -> 0", out.Perf.AllocsBefore, out.Perf.AllocsAfter)
	}
	if !strings.Contains(out.Code, "make([]int, 0, n)") {
		t.Errorf("code should preallocate, got:\n%s", out.Code)
	}
	// Tuned code must still parse (verified by the pipeline re-parse).
	if _, err := parser.ParseFile(token.NewFileSet(), "t.go", out.Code, 0); err != nil {
		t.Errorf("tuned code does not parse: %v", err)
	}
}

func TestGeneticTuneConverges(t *testing.T) {
	src := `package main

func main() {
	ch := make(chan int, 64)
	for i := 0; i < 8; i++ {
		ch <- i
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "t.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	sites := collectCapSites(f)
	if len(sites) != 1 {
		t.Fatalf("sites = %d, want 1", len(sites))
	}
	if sites[0].demand != 8 {
		t.Fatalf("demand = %d, want 8", sites[0].demand)
	}
	// Clear any memory from other tests for this pattern key.
	delete(tuningMemory, "ga-test")
	tc := geneticTune(sites, "ga-test")
	if got := sites[0].get(); got != 8 {
		t.Errorf("tuned capacity = %d, want 8 (demand)", got)
	}
	if tc.wasteBefore != 56 || tc.wasteAfter != 0 {
		t.Errorf("waste = %d -> %d, want 56 -> 0", tc.wasteBefore, tc.wasteAfter)
	}
	if len(tc.changes) != 1 {
		t.Errorf("changes = %v", tc.changes)
	}
	if mem := tuningMemory["ga-test"]; len(mem) != 1 || mem[0] != 8 {
		t.Errorf("memory = %v, want [8]", mem)
	}
	// Second run: memory seeds the population, converges again.
	tc2 := geneticTune(sites, "ga-test")
	if sites[0].get() != 8 || len(tc2.changes) != 0 {
		t.Errorf("second run should hold at 8, got cap=%d changes=%v", sites[0].get(), tc2.changes)
	}
}

func TestGeneticTuneShortfall(t *testing.T) {
	src := `package main

func main() {
	ch := make(chan int, 2)
	for i := 0; i < 8; i++ {
		ch <- i
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "t.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	sites := collectCapSites(f)
	if len(sites) != 1 {
		t.Fatalf("sites = %d, want 1", len(sites))
	}
	delete(tuningMemory, "ga-test2")
	geneticTune(sites, "ga-test2")
	if got := sites[0].get(); got != 8 {
		t.Errorf("tuned capacity = %d, want 8 (was undersized)", got)
	}
}

func TestCollectCapSitesSkipsUnknownDemand(t *testing.T) {
	src := `package main

func main() {
	results := make(chan int, 10)
	for r := range results {
		_ = r
	}
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "t.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sites := collectCapSites(f); len(sites) != 0 {
		t.Errorf("range-loop demand is unknown, sites should be empty, got %d", len(sites))
	}
}

func TestHighlightGo(t *testing.T) {
	got := highlightGo("package main\n// hi\nvar s = \"x\"\nconst n = 42\n")
	for _, want := range []string{
		"\033[32mpackage\033[0m", // keyword green
		"\033[90m// hi\033[0m",   // comment gray
		"\033[33m\"x\"\033[0m",   // string yellow
		"\033[36m42\033[0m",      // number cyan
	} {
		if !strings.Contains(got, want) {
			t.Errorf("highlight missing %q in %q", want, got)
		}
	}
}

func TestRenderTerminalOutput(t *testing.T) {
	out, err := runPipeline("write an http server")
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	renderTo(&buf, out)
	got := buf.String()
	for _, want := range []string{
		"=== GENERATED GO SOURCE ===",
		"💡 BEGINNER CONCEPT ANALOGY",
		"\033[33m", // yellow analogy
		"─── VISUAL EXECUTION TRACE ───",
		"\033[35m[1]\033[0m", // magenta step number
		"\033[90m└──\033[0m", // gray tree char
		"STATUS BADGES",
		"\033[32m✓ SAFE\033[0m", // green badge
		"\033[36m⚡ PERF\033[0m", // cyan perf badge
		"allocs/op",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("rendered output missing %q", want)
		}
	}
}

func TestHumanize(t *testing.T) {
	if got := humanize("main"); got != "Main" {
		t.Errorf("humanize(main) = %q", got)
	}
	if got := humanize("handleLogin"); got != "Handle Login" {
		t.Errorf("humanize(handleLogin) = %q", got)
	}
}
