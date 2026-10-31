package chat

import (
	"strings"
	"sync"
	"testing"
)

// The speccopy proposer (proposeddecl.go): a gated proposer
// that fires only on concept-book grammar misses, and only
// when a model is available and its answer parses. These
// tests drive the seams: a fake predictor stands in for the
// model (the real checkpoint is a local-only file, never a
// test fixture), and the claim chain is the loop's own order.

// withProposer swaps the prediction seam for a test's fake —
// the withConceptEntries pattern (value + a spent Once,
// restored after).
func withProposer(t *testing.T, fn func(string) (string, error)) {
	t.Helper()
	oldFn, oldOnce := proposerFunc, proposerOnce
	t.Cleanup(func() { proposerFunc, proposerOnce = oldFn, oldOnce })
	proposerFunc = fn
	proposerOnce = &sync.Once{}
	proposerOnce.Do(func() {})
}

// withProposerAbsent restores the un-loaded state, so the
// real loader runs (and finds no gob under a test's roots).
func withProposerAbsent(t *testing.T) {
	t.Helper()
	oldFn, oldOnce := proposerFunc, proposerOnce
	t.Cleanup(func() { proposerFunc, proposerOnce = oldFn, oldOnce })
	proposerFunc = nil
	proposerOnce = &sync.Once{}
}

// The trigger's shape: write verb + declaration kind word +
// named target, with the target clause stripped from the
// phrasing the model sees (its training sentences never carry
// a destination).
func TestParseProposedAskClaims(t *testing.T) {
	cases := []struct {
		line     string
		target   string
		phrasing string
	}{
		{"add a struct called Point with X as an int and Y as a string to jim/main.go",
			"jim/main.go", "add a struct called point with x as an int and y as a string"},
		{"add a function in main.go that takes a string and returns an error",
			"main.go", "add a function that takes a string and returns an error"},
		{"create an interface called Store with a save method to store.go",
			"store.go", "create an interface called store with a save method"},
		{"add a method called Receive on a pointer to Buffer to store.go",
			"store.go", "add a method called receive on a pointer to buffer"},
		{"put a struct called Config with a timeout into the file main.go",
			"main.go", "put a struct called config with a timeout"},
	}
	for _, c := range cases {
		pa, ok := parseProposedAsk(c.line)
		if !ok {
			t.Errorf("parseProposedAsk(%q) not claimed", c.line)
			continue
		}
		if pa.target != c.target || pa.phrasing != c.phrasing {
			t.Errorf("parseProposedAsk(%q) = {target:%q phrasing:%q}, want {%q %q}",
				c.line, pa.target, pa.phrasing, c.target, c.phrasing)
		}
	}
}

// Non-claims: no target, no kind word, a question, the
// delegate's behaviour clause, and a generation ask all stay
// with their owners.
func TestParseProposedAskNonClaims(t *testing.T) {
	for _, line := range []string{
		"add a struct called Point with X as an int", // no target named
		"add a field of type int to Config",          // the field form's shape, no kind word
		"how do i add a struct to main.go",           // a question
		"add a function in main.go that calls Load",  // the delegate's clause
		"write a function that reverses a string",    // the gocode brain's generation ask
		"add a print line to jim/main.go",            // a statement, not a declaration
		"add a webserver to jim/main.go",             // add-server's, no kind word
		"tell me about structs in main.go",           // not an imperative
	} {
		if pa, ok := parseProposedAsk(line); ok {
			t.Errorf("parseProposedAsk(%q) claimed %+v — that line is not the proposer's", line, pa)
		}
	}
}

// A sentence the concept book serves never reaches the model:
// the run-level check refuses before the predictor exists in
// the story, even with a model installed.
func TestProposedGrammarHitSkipsModel(t *testing.T) {
	calls := 0
	withProposer(t, func(string) (string, error) {
		calls++
		return "spec struct config", nil
	})
	if runProposedDecl("add a struct named Config to jim/main.go", yes) {
		t.Error("grammar-hit line claimed by the proposer")
	}
	if calls != 0 {
		t.Errorf("model consulted %d times on a grammar hit, want 0", calls)
	}
}

// No model file under the session's roots: the loader finds
// nothing, the claim never fires, and the line gets the
// boundary's answer — the chat behaving exactly as before.
func TestProposedGobAbsentFallsToBoundary(t *testing.T) {
	withProposerAbsent(t)
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	line := "add a struct called Point with X as an int and Y as a string to jim/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("line fell through the claim chain:\n%s", out)
	}
	if !strings.Contains(out, "I can't add a struct yet") {
		t.Errorf("gob-absent line did not get the boundary's answer:\n%s", out)
	}
	if proposerPredictor() != nil {
		t.Error("loader found a model under a fixture root — the feature should be off")
	}
}

// A model answer that can't be parsed as a spec — and one
// that parses but names things outside the registry's pools,
// so case restoration refuses it — both fall through
// silently: no plan, no model error, the boundary's exact
// answer.
func TestProposedUnusableAnswerFallsToBoundary(t *testing.T) {
	for _, raw := range []string{
		"the model has opinions but no spec here",
		"spec struct zzzqqq field qqq www",
	} {
		withProposer(t, func(string) (string, error) { return raw, nil })
		root := liveSessionRoot(t)
		withChatSession(t, root, true)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)
		line := "add a struct called Point with X as an int and Y as a string to jim/main.go"
		if runProposedDecl(line, yes) {
			t.Errorf("raw %q: unusable answer claimed the line", raw)
			continue
		}
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled || !strings.Contains(out, "I can't add a struct yet") {
			t.Errorf("raw %q: line did not land on the boundary:\n%s", raw, out)
		}
		if got := readBack(t, root, "jim/main.go"); got != "package main\n" {
			t.Errorf("raw %q: a fall-through wrote to jim/main.go:\n%s", raw, got)
		}
	}
}

// The full path: a parseable proposal reaches the plan stage
// (marked as the model's, before the gate), a "no" writes
// nothing, and a "yes" lands the rendered declaration
// through the spine's build + verify. The print line goes
// first — the beginner's own order — because it grows the
// main a bare package-main file needs before any declaration
// can join it (the spine's build is absolute; see
// TestConceptFunctionScaffoldLiveSession).
func TestProposedParseableEndToEnd(t *testing.T) {
	calls := 0
	var gotPhrasing string
	withProposer(t, func(phrasing string) (string, error) {
		calls++
		gotPhrasing = phrasing
		return "spec struct handler field items []uint field score bool", nil
	})
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	if calls != 0 {
		t.Fatalf("model consulted %d times before any grammar miss", calls)
	}

	line := "add a struct called Handler with items and a score to jim/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, no, profile) })
	if !handled {
		t.Fatalf("proposed line fell through the claim chain:\n%s", out)
	}
	if gotPhrasing != "add a struct called handler with items and a score" {
		t.Errorf("model was fed %q, want the sentence without its target clause", gotPhrasing)
	}
	if !strings.Contains(out, "proposed by my small trained model") {
		t.Errorf("plan did not own the model's authorship before the gate:\n%s", out)
	}
	if !strings.Contains(out, "type Handler struct") {
		t.Errorf("plan did not show the rendered declaration:\n%s", out)
	}
	if !strings.Contains(out, "Not applied") {
		t.Errorf("decline not reported:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); strings.Contains(got, "Handler") {
		t.Errorf("declined proposal wrote to jim/main.go:\n%s", got)
	}

	out = captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("proposed line fell through on the yes run:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	for _, want := range []string{"type Handler struct {", "Items []uint", "Score bool"} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q after the approved proposal:\n%s\noutput:\n%s", want, got, out)
		}
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}
