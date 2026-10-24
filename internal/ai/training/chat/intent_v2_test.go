package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// intentV2Fixture is a tiny project with doc comments (so the
// pieces learner has something to learn) and a function whose doc
// mentions retries (so /find has something to find).
func intentV2Fixture(t *testing.T) string {
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
	write("go.mod", "module example.com/v2\n\ngo 1.21\n")
	write("store/store.go", `package store

// LoadRetries loads the retry settings for the store.
func LoadRetries() int { return 3 }

// Store holds the items for sale.
type Store struct{ Items []string }
`)
	return dir
}

// withRealIntentModel injects the shipped model file into the
// package var the dispatch reads. (Under a bare `go test`,
// defaultIntentModel finds no data/ next to the package and the
// layer is off — the older dispatch test passes vacuously without
// this.)
func withRealIntentModel(t *testing.T) *IntentModel {
	t.Helper()
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadIntentModel(filepath.Join(wd, "..", "..", "..", "..", "data", "intent_model.json"))
	if err != nil {
		t.Fatalf("load shipped intent model: %v", err)
	}
	old, oldPending := intentModel, pendingClarify
	intentModel = m
	pendingClarify = nil
	t.Cleanup(func() { intentModel, pendingClarify = old, oldPending })
	return m
}

// The neighbor sentences the v2 build had to keep straight, checked
// against the shipped model at the classifier's own gate.
func TestIntentV2ProbeSentences(t *testing.T) {
	m := withRealIntentModel(t)
	probes := []struct{ line, want string }{
		{"where does Items get its data", "dataflow_q"},
		{"where does the config come from", "dataflow_q"},
		{"where do I change the router", "where"},
		{"why does Names lock", "why_func"},
		{"how does Add compare to Get", "compare"},
	}
	for _, p := range probes {
		got, _, ok := m.ClassifyIntent(p.line)
		if !ok || got != p.want {
			t.Errorf("ClassifyIntent(%q) = %q, ok=%v; want %q", p.line, got, ok, p.want)
		}
	}
}

// Brain-owned intents are never dispatched and never clarified:
// the line falls through with no pending question left behind.
func TestIntentV2BrainOwnedFallThrough(t *testing.T) {
	withRealIntentModel(t)
	for _, line := range []string{
		"what is a goroutine",              // concept_q
		"write a function that sorts ints", // write_code
		"tell me a joke",                   // social
		"run make eval",                    // run_make
		"the router thing",                 // concept_q again, vague form
	} {
		if tryIntentDispatch(line, nil) {
			t.Errorf("tryIntentDispatch(%q) claimed a brain-owned line", line)
		}
		if pendingClarify != nil {
			t.Errorf("tryIntentDispatch(%q) left a pending clarification", line)
		}
	}
}

// Deliverable 1: the four newly wired dispatches reach their
// features from free phrasing.
func TestIntentV2DispatchFind(t *testing.T) {
	withRealIntentModel(t)
	withEditsProject(t, intentV2Fixture(t))
	var handled bool
	out := captureStdout(t, func() { handled = tryIntentDispatch("got anything on retries", nil) })
	if !handled {
		t.Fatalf("find phrasing not dispatched:\n%s", out)
	}
	if !strings.Contains(out, `match(es) for "retries"`) || !strings.Contains(out, "LoadRetries") {
		t.Errorf("find did not search for retries:\n%s", out)
	}
}

func TestIntentV2DispatchLearnPieces(t *testing.T) {
	withRealIntentModel(t)
	withEditsProject(t, intentV2Fixture(t))
	var handled bool
	out := captureStdout(t, func() { handled = tryIntentDispatch("learn pieces pls", nil) })
	if !handled || !strings.Contains(out, "pieces from doc comments") {
		t.Errorf("learn_pieces dispatch = %v:\n%s", handled, out)
	}
}

func TestIntentV2DispatchExportUnderstanding(t *testing.T) {
	withRealIntentModel(t)
	dir := intentV2Fixture(t)
	withEditsProject(t, dir)
	var handled bool
	out := captureStdout(t, func() { handled = tryIntentDispatch("export the pairs", nil) })
	if !handled || !strings.Contains(out, "structure→language pairs") {
		t.Errorf("export_understanding dispatch = %v:\n%s", handled, out)
	}
	path := filepath.Join(dir, "understanding_pairs.jsonl")
	if !strings.Contains(out, path) {
		t.Errorf("export_understanding did not report the durable project path %q:\n%s", path, out)
	}
	if _, err := os.Stat(path); err != nil {
		t.Errorf("export_understanding did not write %q: %v", path, err)
	}
}

func TestIntentV2DispatchBuildAPI(t *testing.T) {
	withRealIntentModel(t)
	dir := t.TempDir()
	t.Chdir(dir)
	var handled bool
	out := captureStdout(t, func() { handled = tryIntentDispatch("build an api with /users and /health", nil) })
	if !handled || !strings.Contains(out, "Composed API") {
		t.Fatalf("build_api dispatch = %v:\n%s", handled, out)
	}
	if _, err := os.Stat(filepath.Join(dir, "compose", "main.go")); err != nil {
		t.Errorf("compose/main.go not written: %v", err)
	}
	// No routes in the sentence: guidance, not silence.
	out = captureStdout(t, func() { handled = tryIntentDispatch("build api", nil) })
	if !handled || !strings.Contains(out, "Tell me the routes") {
		t.Errorf("build_api without routes = %v:\n%s", handled, out)
	}
}

// The edits parser keeps its priority: its sentence forms parse as
// edits (the loops call tryAssembledEdit before tryIntentDispatch,
// so a parsed form never reaches the intent layer).
func TestIntentV2EditsParserStillClaims(t *testing.T) {
	withEditsProject(t, editsChatFixture(t))
	if _, ok := parseAssembledEdit("add a Timeout field to Config"); !ok {
		t.Error("edits parser no longer claims its field form")
	}
	var handled bool
	out := captureStdout(t, func() {
		handled = tryAssembledEdit("add a Timeout field to Config", NewLineReader(strings.NewReader("n\n")))
	})
	if !handled {
		t.Errorf("tryAssembledEdit did not claim the form (output:\n%s)", out)
	}
}

// The clarification band, at the unit level: a top-2 tie names
// both; too-vague and brain-owned tops name nothing.
func TestClarifyCandidatesBand(t *testing.T) {
	m := withRealIntentModel(t)
	cands := clarifyCandidates(m, "the words and pieces")
	if len(cands) != 2 || cands[0] != "learn_pieces" || cands[1] != "glossary" {
		t.Errorf("clarifyCandidates(tie) = %v, want [learn_pieces glossary]", cands)
	}
	if c := clarifyCandidates(m, "show the thing"); len(c) != 0 {
		t.Errorf("clarifyCandidates(below floor) = %v, want none", c)
	}
	if c := clarifyCandidates(m, "the router thing"); len(c) != 0 {
		t.Errorf("clarifyCandidates(brain-owned top) = %v, want none", c)
	}
}

// The full exchange: one prompt, "1" routes to the first
// candidate, "neither" falls through untouched, "2" routes to the
// second, and an unrelated reply clears the question and is
// processed fresh.
func TestClarifyExchange(t *testing.T) {
	withRealIntentModel(t)
	withEditsProject(t, intentV2Fixture(t))

	ask := func() string {
		var handled bool
		out := captureStdout(t, func() { handled = tryIntentDispatch("the words and pieces", nil) })
		if !handled {
			t.Fatalf("near miss not claimed by the clarification offer:\n%s", out)
		}
		if pendingClarify == nil {
			t.Fatalf("no pending clarification after the offer:\n%s", out)
		}
		return out
	}

	// One prompt, naming both candidates.
	out := ask()
	if strings.Count(out, "Did you mean") != 1 || !strings.Contains(out, "(1)") || !strings.Contains(out, "(2)") {
		t.Errorf("offer output wrong:\n%s", out)
	}

	// "1" routes to the first candidate (learn the pieces).
	out = captureStdout(t, func() {
		if !tryIntentDispatch("1", nil) {
			t.Error("reply 1 not handled")
		}
	})
	if !strings.Contains(out, "pieces from doc comments") {
		t.Errorf("reply 1 routed wrong:\n%s", out)
	}
	if pendingClarify != nil {
		t.Error("pending clarification not cleared after answering")
	}

	// "neither" falls through to exactly the old behavior.
	ask()
	var handled bool
	captureStdout(t, func() { handled = tryIntentDispatch("neither", nil) })
	if handled {
		t.Error("'neither' should fall through unhandled")
	}
	if pendingClarify != nil {
		t.Error("pending clarification not cleared after 'neither'")
	}

	// "2" routes to the second candidate (the glossary).
	ask()
	captureStdout(t, func() { handled = tryIntentDispatch("2", nil) })
	if !handled {
		t.Error("reply 2 not handled")
	}
	if pendingClarify != nil {
		t.Error("pending clarification not cleared after reply 2")
	}

	// An unrelated confident line clears the question and is
	// processed fresh — no hijack, no second prompt.
	ask()
	out = captureStdout(t, func() { handled = tryIntentDispatch("words", nil) })
	if !handled || strings.Contains(out, "Did you mean") {
		t.Errorf("unrelated line mishandled (handled=%v):\n%s", handled, out)
	}
	if pendingClarify != nil {
		t.Error("pending clarification survived an unrelated line")
	}
}

// A confident sentence never triggers the question.
func TestClarifyConfidentNeverAsks(t *testing.T) {
	withRealIntentModel(t)
	withEditsProject(t, intentV2Fixture(t))
	for _, line := range []string{"words", "got anything on retries", "learn pieces pls"} {
		var handled bool
		out := captureStdout(t, func() { handled = tryIntentDispatch(line, nil) })
		if !handled {
			t.Errorf("confident line %q not dispatched", line)
		}
		if strings.Contains(out, "Did you mean") || pendingClarify != nil {
			t.Errorf("confident line %q triggered a clarification:\n%s", line, out)
		}
	}
}

// The single-candidate wording path: a pending question with one
// candidate routes on "yes" and uses the original line.
func TestClarifySingleCandidateYes(t *testing.T) {
	withRealIntentModel(t)
	withEditsProject(t, intentV2Fixture(t))
	pendingClarify = &pendingClarification{line: "export the pairs", cands: []string{"export_understanding"}}
	var handled bool
	out := captureStdout(t, func() { handled = tryIntentDispatch("yes", nil) })
	if !handled || !strings.Contains(out, "structure→language pairs") {
		t.Errorf("single-candidate yes = %v:\n%s", handled, out)
	}
	if pendingClarify != nil {
		t.Error("pending clarification not cleared after yes")
	}
}
