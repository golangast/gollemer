package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The live-session regression suite for the add-a-webserver
// claim — the second failure (2026-10-25 13:42) happened in a
// real session, not in any state the first suite set up:
//
//   - the b0aae9c tests called runAddServer directly, so no
//     earlier claim in the unified loop — and no fall-through
//     past it — was ever exercised;
//   - they hand-wrote jim/main.go instead of creating it through
//     the create feature, the way the user's session did;
//   - withServerProject left lastAnalyzeProject nil, so the
//     "project already analyzed, index older than the file"
//     state the user was actually in never occurred;
//   - the grammar table never included the designator form the
//     user typed ("add webserver to folder jim").
//
// These tests therefore drive the loop's deterministic claim
// chain in the loop's own order (real_seq2seq_chat.go: learn,
// beginner, assembled edits, rename, build resources, creations,
// add-server, delegate — everything before tryIntentDispatch,
// the only road to the analyzer's report and its "is not a
// directory" resolver error), build jim/ through the create
// feature itself, and run both session states: analyzed-first
// (the user's) and cold.

// claimChain runs one line through the unified loop's
// deterministic claims, in order, with a scripted gate answer.
func claimChain(line string, ask editAsk, profile *UserProfile) bool {
	if tryLearnCommand(line) {
		return true
	}
	if handleBeginnerCommand(line, profile) {
		return true
	}
	if runAssembledEdit(line, ask) {
		return true
	}
	if runRenameEdit(line, ask) {
		return true
	}
	if tryBuildResources(line, profile) {
		return true
	}
	if runCreate(line, ask) {
		return true
	}
	if runAddServer(line, ask) {
		return true
	}
	if runDelegate(line, ask) {
		return true
	}
	return false
}

// withChatSession points the chat globals at dir. analyzed=true
// reproduces the user's session state: the project was analyzed
// (his learn-path render) BEFORE jim existed, so the analyzed
// index predates the file the edit targets. analyzed=false is
// the cold session: no project analyzed, creations resolve
// against the folder the chat started in.
func withChatSession(t *testing.T, dir string, analyzed bool) {
	t.Helper()
	oldProj, oldRoot, oldURoot, oldSnap := lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot, lastAnalyzeSnapshot
	t.Cleanup(func() {
		lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot, lastAnalyzeSnapshot = oldProj, oldRoot, oldURoot, oldSnap
		learnResetForTest()
	})
	learnResetForTest()
	unifiedProjectRoot = dir
	if analyzed {
		if !analyzeQuiet(dir) {
			t.Fatal("fixture project did not analyze")
		}
	} else {
		lastAnalyzeProject = nil
		lastAnalyzeRoot = ""
		lastAnalyzeSnapshot = nil
	}
}

// liveSessionRoot is the fixture module BEFORE jim: a go.mod and
// a root package file (real projects — and the user's scratch
// checkout — have more than the one package being edited).
func liveSessionRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/livesession\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package livesession\n")
	return dir
}

// createJimThroughTheChain runs the user's own first two lines —
// folder, then the bare-clause file — through the claim chain,
// exactly as his session created them.
func createJimThroughTheChain(t *testing.T, profile *UserProfile) {
	t.Helper()
	for _, line := range []string{"create a folder named jim", "create file jim/main.go"} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Fatalf("creation line %q not claimed by the chain:\n%s", line, out)
		}
		if !strings.Contains(out, "Done — created") {
			t.Fatalf("creation line %q did not create:\n%s", line, out)
		}
	}
}

// analyzeReportMarkers are the strings a webserver sentence must
// never produce: the analyzer resolver's error (the first live
// failure) and the full analyze report's furniture (the second).
var analyzeReportMarkers = []string{"couldn't read that codebase", "is not a directory", "Entry points"}

func requireNoAnalyzeOutput(t *testing.T, line, out string) {
	t.Helper()
	for _, m := range analyzeReportMarkers {
		if strings.Contains(out, m) {
			t.Errorf("line %q produced analyzer output (marker %q):\n%s", line, m, out)
		}
	}
}

// The user's third line, verbatim, in his session state: the
// project analyzed first, jim and its bare-clause main.go
// created through the chat itself moments before.
func TestAddServerLiveSessionFileLine(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	if got := readBack(t, root, "jim/main.go"); got != "package main\n" {
		t.Fatalf("the create feature's file is not the bare clause his session had: %q", got)
	}
	line := "add a webserver to jim/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("his line fell through the whole claim chain:\n%s", out)
	}
	requireNoAnalyzeOutput(t, line, out)
	if !strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("the composer did not own the answer:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); got != analyze.WebServerSource {
		t.Errorf("jim/main.go was not filled with the shared component:\n%s", got)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}

// Every webserver line the user has typed live, plus the bare
// folder form — each in BOTH session states (analyzed-first like
// his session, and cold), each landing as the filled component.
func TestAddServerLiveSessionAllHisLines(t *testing.T) {
	lines := []string{
		"add a webserver to jim/main.go",     // 2026-10-25, the file form
		"add a webserver to the jim/main.go", // 2026-10-24, his first phrasing
		"add webserver to folder jim",        // 2026-10-25, the folder-word form
		"add a webserver to jim/",            // 2026-10-24, the trailing slash
		"add a webserver to jim",             // the bare folder name
		// 2026-10-25 20:10, on the shipped stack with the
		// designator fix live: "the" BEFORE webserver — the
		// determiner slot read only a/an, so both fell
		// through to the analyzer beside their working
		// designator twin.
		"add the webserver to jim/main.go",
		"add the webserver to the file jim/main.go",
	}
	for _, analyzed := range []bool{true, false} {
		for _, line := range lines {
			root := liveSessionRoot(t)
			withChatSession(t, root, analyzed)
			profile := &UserProfile{}
			createJimThroughTheChain(t, profile)
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
			if !handled {
				t.Errorf("analyzed=%v: %q fell through the claim chain:\n%s", analyzed, line, out)
				continue
			}
			requireNoAnalyzeOutput(t, line, out)
			if got := readBack(t, root, "jim/main.go"); got != analyze.WebServerSource {
				t.Errorf("analyzed=%v: %q did not fill jim/main.go with the component:\n%s\noutput:\n%s", analyzed, line, got, out)
			}
		}
	}
}

// The wording guard: webserver-shaped lines that cannot be
// served still belong to this claim — the answer names the
// target and the problem in webserver terms, never an analyze
// report or a resolver error. (If any of these fell through the
// chain, the intent layer would be free to answer with exactly
// those — handled=false is the failure this test exists for.)
func TestAddServerChainGuardWording(t *testing.T) {
	refusals := []struct {
		line string
		want string
	}{
		// A missing folder named with the designator word: the
		// refusal names the folder the user meant and the create
		// line that would write its server — never "folder nope".
		{"add webserver to folder nope", `create a file nope/main.go with a webserver`},
		// A missing file: the same create alternative.
		{"add a webserver to nope/main.go", `create a file nope/main.go with a webserver`},
		// The designator with no name at all: claimed, and asked
		// in webserver terms.
		{"add a webserver to folder", "doesn't name the file or folder"},
		{"add a webserver to the", "doesn't name the file or folder"},
	}
	for _, tc := range refusals {
		root := liveSessionRoot(t)
		withChatSession(t, root, true)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(tc.line, yes, profile) })
		if !handled {
			t.Errorf("%q fell through the claim chain:\n%s", tc.line, out)
			continue
		}
		requireNoAnalyzeOutput(t, tc.line, out)
		if !strings.Contains(out, tc.want) {
			t.Errorf("%q: refusal does not say %q:\n%s", tc.line, tc.want, out)
		}
		if strings.Contains(out, "folder nope") {
			t.Errorf("%q: the designator word leaked into the target:\n%s", tc.line, out)
		}
	}

	// The designator forms that DO resolve serve the component.
	served := []string{
		"add a webserver to the folder jim",
		"add a webserver to file jim/main.go",
		"put a web server into the folder jim",
	}
	for _, line := range served {
		root := liveSessionRoot(t)
		withChatSession(t, root, true)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Errorf("%q fell through the claim chain:\n%s", line, out)
			continue
		}
		requireNoAnalyzeOutput(t, line, out)
		if got := readBack(t, root, "jim/main.go"); got != analyze.WebServerSource {
			t.Errorf("%q did not fill jim/main.go:\n%s\noutput:\n%s", line, got, out)
		}
	}
}

// The designator grammar, at the parse level: the kind word is a
// slot (as in the creations grammar), never part of the target.
func TestParseAddServerDesignatorForms(t *testing.T) {
	claims := map[string]string{
		"add webserver to folder jim":             "jim",
		"add a webserver to the folder jim":       "jim",
		"add a webserver to folder jim/":          "jim/",
		"add a webserver in folder jim":           "jim",
		"add a webserver to file jim/main.go":     "jim/main.go",
		"add a webserver to the file jim/main.go": "jim/main.go",
		"add a webserver to directory jim":        "jim",
	}
	for line, want := range claims {
		got, ok := parseAddServer(line)
		if !ok {
			t.Errorf("parseAddServer(%q) not claimed", line)
			continue
		}
		if got != want {
			t.Errorf("parseAddServer(%q) = %q, want %q", line, got, want)
		}
	}
	// Claimed but nameless: the flow answers, the line never
	// falls through.
	for _, line := range []string{"add a webserver to folder", "add a webserver to the", "add a webserver to file"} {
		if got, ok := parseAddServer(line); !ok || got != "" {
			t.Errorf("parseAddServer(%q) = %q, %v — want (\"\", true), claimed with no target", line, got, ok)
		}
	}
	// Still not this claim's: nothing after the preposition, and
	// the neighbours' sentences.
	for _, line := range []string{"add a webserver to", "add a webserver", "create a folder named jim"} {
		if got, ok := parseAddServer(line); ok {
			t.Errorf("parseAddServer(%q) claimed target %q — not its sentence", line, got)
		}
	}
}
