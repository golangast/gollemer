package beginner

import (
	"go/format"
	"strings"
	"testing"
)

func mustGenerate(t *testing.T, prompt string) (string, string) {
	t.Helper()
	code, expl, err := GenerateGoFromCommand(prompt)
	if err != nil {
		t.Fatalf("GenerateGoFromCommand(%q): %v", prompt, err)
	}
	if strings.TrimSpace(code) == "" {
		t.Fatalf("GenerateGoFromCommand(%q): empty code", prompt)
	}
	if strings.Count(expl, ". ") < 1 {
		t.Errorf("GenerateGoFromCommand(%q): explanation too short: %q", prompt, expl)
	}
	// Formatted output must be gofmt-stable.
	again, err := format.Source([]byte(code))
	if err != nil {
		t.Fatalf("GenerateGoFromCommand(%q): output does not parse: %v", prompt, err)
	}
	if string(again) != code {
		t.Errorf("GenerateGoFromCommand(%q): output is not gofmt-stable", prompt)
	}
	if err := checkCompiles(code); err != nil {
		t.Errorf("GenerateGoFromCommand(%q): does not type-check: %v", prompt, err)
	}
	return code, expl
}

func TestGenerateGoFromCommandSpecExamples(t *testing.T) {
	code, expl := mustGenerate(t, "Create an HTTP server listening on port 8080 with a health check endpoint")
	for _, want := range []string{`":8080"`, "/healthz", "http.NewServeMux", "http.ListenAndServe"} {
		if !strings.Contains(code, want) {
			t.Errorf("http server code missing %q", want)
		}
	}
	if !strings.Contains(expl, "ServeMux") {
		t.Errorf("http server explanation missing rationale: %q", expl)
	}

	code, _ = mustGenerate(t, "Write a function to read a file line by line safely")
	for _, want := range []string{"bufio.NewScanner", "defer f.Close()", "scanner.Err()", "func readLines"} {
		if !strings.Contains(code, want) {
			t.Errorf("read-lines code missing %q", want)
		}
	}

	code, _ = mustGenerate(t, "Create a worker pool using channels and WaitGroup")
	for _, want := range []string{"sync.WaitGroup", "wg.Wait()", "close(jobs)", "range jobs"} {
		if !strings.Contains(code, want) {
			t.Errorf("worker pool code missing %q", want)
		}
	}
}

func TestGenerateGoFromCommandAllIntentsCompile(t *testing.T) {
	prompts := []string{
		"Create a worker pool using channels and WaitGroup",
		"build a safe counter with a mutex",
		"Write a function to read a file line by line safely",
		"write data to a file",
		"create a web server",
		"encode this struct to json",
		"use a context with timeout",
		"build a string efficiently",
		"sort a slice of names",
		"use a ticker",
		"run something every second",
		"read command line arguments",
		"read an environment variable",
	}
	for _, p := range prompts {
		mustGenerate(t, p)
	}
}

func TestGenerateGoFromCommandUnknown(t *testing.T) {
	_, _, err := GenerateGoFromCommand("bake a cake with quantum frosting")
	if err == nil {
		t.Fatal("want error for unknown request")
	}
	if !strings.Contains(err.Error(), "worker pools") {
		t.Errorf("error should list capabilities, got: %v", err)
	}
}

func TestGenerateGoFromCommandParams(t *testing.T) {
	code, _, err := GenerateGoFromCommand("Create an HTTP server listening on port 9090")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(code, `":9090"`) {
		t.Errorf("port not extracted into code:\n%s", code)
	}

	code, _, err = GenerateGoFromCommand("Create a worker pool with 7 workers")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(code, "w <= 7") {
		t.Errorf("worker count not extracted into code:\n%s", code)
	}

	code, _, err = GenerateGoFromCommand(`write "data.txt" to a file`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(code, `"data.txt"`) {
		t.Errorf("filename not extracted into code:\n%s", code)
	}
}

func TestExplainGoConcept(t *testing.T) {
	c, err := ExplainGoConcept("channels")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(c.Analogy, "conveyor belt") {
		t.Errorf("analogy = %q", c.Analogy)
	}
	if len(c.KeyRules) < 2 {
		t.Errorf("want at least 2 key rules, got %d", len(c.KeyRules))
	}
	if err := checkCompiles(c.Snippet); err != nil {
		t.Errorf("channels snippet does not compile: %v", err)
	}

	// Case-insensitive + aliases.
	for _, name := range []string{"Channels", "goroutine", "ERROR HANDLING", "struct"} {
		if _, err := ExplainGoConcept(name); err != nil {
			t.Errorf("ExplainGoConcept(%q): %v", name, err)
		}
	}

	if _, err := ExplainGoConcept("quantum baking"); err == nil {
		t.Error("want error for unknown concept")
	}
}

func TestExplainGoConceptAllSnippetsCompile(t *testing.T) {
	for _, name := range conceptNames() {
		c, err := ExplainGoConcept(name)
		if err != nil {
			t.Fatalf("ExplainGoConcept(%q): %v", name, err)
		}
		if err := checkCompiles(c.Snippet); err != nil {
			t.Errorf("concept %q snippet does not compile: %v", name, err)
		}
	}
}

const visualSnippet = `package main

import (
	"fmt"
	"os"
)

type Config struct {
	Port int
}

func load(path string) Config {
	data, _ := os.ReadFile(path)
	fmt.Println(string(data))
	return Config{Port: 8080}
}

func main() {
	cfg := load("app.conf")
	fmt.Println(cfg.Port)
}
`

func TestBuildVisualMap(t *testing.T) {
	vm, err := BuildVisualMap(visualSnippet)
	if err != nil {
		t.Fatalf("BuildVisualMap: %v", err)
	}
	if len(vm.Imports) != 2 || vm.Imports[0] != "fmt" || vm.Imports[1] != "os" {
		t.Errorf("imports = %v", vm.Imports)
	}
	if len(vm.Structs) != 1 || vm.Structs[0] != "Config" {
		t.Errorf("structs = %v", vm.Structs)
	}
	if len(vm.Functions) != 2 {
		t.Fatalf("functions = %d, want 2", len(vm.Functions))
	}
	var mainCalls []string
	for _, fn := range vm.Functions {
		if fn.Name == "main" {
			mainCalls = fn.Calls
		}
		if fn.Line <= 0 {
			t.Errorf("function %q missing line number", fn.Name)
		}
	}
	found := false
	for _, c := range mainCalls {
		if c == "load" {
			found = true
		}
	}
	if !found {
		t.Errorf("main calls = %v, want it to include load", mainCalls)
	}
	wantSteps := []string{"main", "load"}
	if len(vm.CallSteps) != len(wantSteps) {
		t.Fatalf("callSteps = %v, want %v", vm.CallSteps, wantSteps)
	}
	for i, want := range wantSteps {
		if vm.CallSteps[i] != want {
			t.Errorf("callSteps[%d] = %q, want %q", i, vm.CallSteps[i], want)
		}
	}
	if vm.Summary == "" {
		t.Error("empty summary")
	}
}

func TestBuildVisualMapFragment(t *testing.T) {
	vm, err := BuildVisualMap(`func add(a, b int) int { return a + b }`)
	if err != nil {
		t.Fatalf("BuildVisualMap(fragment): %v", err)
	}
	if len(vm.Functions) != 1 || vm.Functions[0].Name != "add" {
		t.Errorf("functions = %+v", vm.Functions)
	}
	if len(vm.CallSteps) != 1 || vm.CallSteps[0] != "add" {
		t.Errorf("callSteps = %v", vm.CallSteps)
	}
}

func TestBuildVisualMapErrors(t *testing.T) {
	if _, err := BuildVisualMap("   "); err == nil {
		t.Error("empty snippet: want error")
	}
	if _, err := BuildVisualMap("package main\nfunc broken( {"); err == nil {
		t.Error("unparseable snippet: want error")
	}
}
