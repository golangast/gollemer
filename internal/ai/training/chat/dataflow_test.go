package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// dataflowChatFixture writes a tiny two-file project whose Items
// slice is declared in one file and appended to in another.
func dataflowChatFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/chatdflow\n\ngo 1.21\n")
	write("store/store.go", `package store

var Items []string = []string{"a"}

func Reset() {
	Items[0] = "z"
}
`)
	write("store/more.go", `package store

func Add(s string) {
	Items = append(Items, s)
}
`)
	return dir
}

// The four plain-sentence shapes reach the dataflow answer through
// the chat's codebase-question route, with the declaration and the
// cross-file change sites in the reply.
func TestDataflowQuestionRoutes(t *testing.T) {
	oldProj, oldRoot, oldURoot := lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot
	defer func() { lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot = oldProj, oldRoot, oldURoot }()

	dir := dataflowChatFixture(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	lastAnalyzeProject = p
	lastAnalyzeRoot = dir

	for _, q := range []string{
		"where does Items get its data",
		"where does Items come from",
		"where does Items change",
		"what changes Items",
	} {
		out, ok := tryCodebaseQuestion(q)
		if !ok {
			t.Fatalf("tryCodebaseQuestion(%q) not handled", q)
		}
		for _, want := range []string{"`Items`", "store/store.go", "store/more.go", "new items are added with", "element [0] is set to"} {
			if !strings.Contains(out, want) {
				t.Errorf("%q: answer missing %q\n---\n%s", q, want, out)
			}
		}
	}

	// With no project loaded, a dataflow-shaped question makes the
	// chat analyze the folder it runs in, like the other codebase
	// question shapes.
	lastAnalyzeProject = nil
	lastAnalyzeRoot = ""
	unifiedProjectRoot = dir
	out, ok := tryCodebaseQuestion("where does Items get its data")
	if !ok || !strings.Contains(out, "`Items`") {
		t.Errorf("auto-analyze route failed: ok=%v out=%q", ok, out)
	}

	// A question naming nothing the project tracks declines, so the
	// other brains keep their sentences.
	lastAnalyzeProject = p
	if _, ok := tryCodebaseQuestion("what does a goroutine do"); ok {
		t.Error("tryCodebaseQuestion stole a Go-concept question")
	}
}

// The shipped intent model claims the four shapes for dataflow_q,
// and slot extraction recovers the variable name from each.
func TestDataflowIntentAndSlot(t *testing.T) {
	oldProj := lastAnalyzeProject
	defer func() { lastAnalyzeProject = oldProj }()

	root := filepath.Join("..", "..", "..", "..")
	m, err := LoadIntentModel(filepath.Join(root, "data", "intent_model.json"))
	if err != nil {
		t.Fatalf("LoadIntentModel: %v", err)
	}
	dir := dataflowChatFixture(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	lastAnalyzeProject = p

	for _, q := range []string{
		"where does Items get its data",
		"where does Items come from",
		"where does Items change",
		"what changes Items",
	} {
		if intent, _, ok := m.ClassifyIntent(q); !ok || intent != "dataflow_q" {
			t.Errorf("ClassifyIntent(%q) = %q, %v; want dataflow_q", q, intent, ok)
		}
		name := slotVarName(q)
		if name == "" {
			t.Errorf("slotVarName(%q) = empty", q)
			continue
		}
		if _, ok := p.DataflowText(name); !ok {
			t.Errorf("slotVarName(%q) = %q, which does not resolve", q, name)
		}
	}
}
