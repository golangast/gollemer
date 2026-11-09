package chat

// The goroutine-result piece's chat pins: the grammar's
// claims and declines, and the full session behaviour —
// John's exact stolen line landing the channel
// transformation through the claim chain, the gate, the
// which-one question, and the refusals writing nothing.

import (
	"strings"
	"testing"
)

func TestParseGoroutineResultClaims(t *testing.T) {
	cases := []struct {
		line string
		want goroutineResultRequest
	}{
		{"have the goroutine return a string",
			goroutineResultRequest{typ: "string"}},
		{"make the goroutine return an int",
			goroutineResultRequest{typ: "int"}},
		{"update the goroutine to return a string",
			goroutineResultRequest{typ: "string"}},
		{"can you update the goroutine to return a string in jim/main.go",
			goroutineResultRequest{typ: "string", file: "jim/main.go"}},
		{"could you have the goroutine return a bool?",
			goroutineResultRequest{typ: "bool"}},
		{"please make the goroutine return a float64",
			goroutineResultRequest{typ: "float64"}},
		{"return a string from the goroutine",
			goroutineResultRequest{typ: "string"}},
		{"have the goroutine in main return a string",
			goroutineResultRequest{typ: "string", fn: "main"}},
		{"have the goroutine in jim/main.go return a string",
			goroutineResultRequest{typ: "string", file: "jim/main.go"}},
		{"have the goroutine in main in jim/main.go return a string",
			goroutineResultRequest{typ: "string", fn: "main", file: "jim/main.go"}},
		{"have the goroutine return a string in main",
			goroutineResultRequest{typ: "string", fn: "main"}},
		{"have the goroutine return the string",
			goroutineResultRequest{typ: "string"}},
		// A bare identifier after "in" is the grammar's
		// function slot — deterministic, and the planner
		// answers honestly when no such function exists.
		{"have the goroutine return a string in jim",
			goroutineResultRequest{typ: "string", fn: "jim"}},
		{"have the second goroutine return a string",
			goroutineResultRequest{typ: "string", ordinal: 2}},
		{"update the first goroutine in helper to return an int in jim/main.go",
			goroutineResultRequest{typ: "int", fn: "helper", file: "jim/main.go", ordinal: 1}},
		{"have the go routine return a string",
			goroutineResultRequest{typ: "string"}},
		{"have the goroutine return a string.",
			goroutineResultRequest{typ: "string"}},
		// A type outside the closed set still claims: the
		// shape is this piece's, and the refusal is the
		// piece's own words.
		{"have the goroutine return a byte",
			goroutineResultRequest{typ: "byte"}},
	}
	for _, tc := range cases {
		got, ok := parseGoroutineResult(tc.line)
		if !ok {
			t.Errorf("%q did not claim", tc.line)
			continue
		}
		if got != tc.want {
			t.Errorf("%q = %+v, want %+v", tc.line, got, tc.want)
		}
	}
}

func TestParseGoroutineResultDeclines(t *testing.T) {
	for _, line := range []string{
		"have the goroutine print a string",         // printing is a different ask
		"make the goroutine stop",                   // no return at all
		"have the goroutine return",                 // no type named
		"the goroutine returns a string",            // a statement, not an ask
		"what is a goroutine",                       // the concept book's question
		"add a goroutine to main",                   // the concept book's own sentence
		"add a string to jim/main.go",               // the theft's rewrite — not this piece
		"have the server return a string",           // a different subject
		"update the server to return a string",      // a different subject
		"return a string from the function",         // not the goroutine
		"have the goroutine in jim/main.go",         // no return clause
		"have the sixth goroutine return a string",  // positions stop at fifth
		"make the goroutine return a string please", // trailing words break the shape
		"run the goroutine",                         // someone else's sentence
		"explain what a goroutine is",               // the talk seats' question
	} {
		if got, ok := parseGoroutineResult(line); ok {
			t.Errorf("%q claimed as %+v — it must not", line, got)
		}
	}
}

// goroutineJimRoot is John's scratch shape: a module with a
// root package file and jim/main.go holding one anonymous
// goroutine.
func goroutineJimRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/goroutinejim\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package goroutinejim\n")
	writeFile(t, dir, "jim/main.go", "package main\n\nfunc main() {\n\tgo func() {\n\t\t// TODO: work goes here\n\t}()\n}\n")
	return dir
}

// John's literal line — the one the phrasing layer stole
// into "add a string to jim/main.go" on 2026-11-09 — claims
// THIS piece through the whole chain, shows the plan, and
// lands the channel transformation with the partner
// confirmation. No string statement is added anywhere.
func TestGoroutineResultJohnsLineLands(t *testing.T) {
	dir := goroutineJimRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	line := "can you update the goroutine to return a string in jim/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("John's line fell through the whole claim chain:\n%s", out)
	}
	if !strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("the composer did not own the answer:\n%s", out)
	}
	if strings.Contains(out, "add a string") {
		t.Errorf("the phrasing layer's theft wording appeared:\n%s", out)
	}
	got := readBack(t, dir, "jim/main.go")
	for _, want := range []string{
		"resultCh := make(chan string)",
		"resultCh <- \"\"",
		"result := <-resultCh",
		"_ = result",
		"// TODO: work goes here",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q after the landing:\n%s", want, got)
		}
	}
	if !strings.Contains(out, "Done — the goroutine in main now sends its string back on the resultCh channel") {
		t.Errorf("the partner confirmation did not name the landing:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked that it's really there") {
		t.Errorf("the spine's build/verify was not confirmed:\n%s", out)
	}
}

// The gate declines: nothing is written, byte for byte.
func TestGoroutineResultGateDeclineWritesNothing(t *testing.T) {
	dir := goroutineJimRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	before := readBack(t, dir, "jim/main.go")
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("have the goroutine return a string in jim/main.go", no, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if got := readBack(t, dir, "jim/main.go"); got != before {
		t.Errorf("a declined gate still wrote:\n%s", got)
	}
}

// Several goroutines, none named: the piece asks which
// one, and the number completes the original ask on the
// second goroutine.
func TestGoroutineResultSeveralAsksAndCompletes(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/goroutinetwo\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tgo func() {\n\t\tprintln(\"one\")\n\t}()\n\tgo func() {\n\t\tprintln(\"two\")\n\t}()\n}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("have the goroutine return a string in main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "which one should get the result channel") {
		t.Fatalf("no which-one question was asked:\n%s", out)
	}
	if got := readBack(t, dir, "main.go"); strings.Contains(got, "resultCh") {
		t.Fatalf("the question itself wrote the edit:\n%s", got)
	}
	out = captureStdout(t, func() {
		handled = claimChain("2", yes, profile)
	})
	if !handled {
		t.Fatalf("the answer fell through:\n%s", out)
	}
	got := readBack(t, dir, "main.go")
	if n := strings.Count(got, "resultCh <- \"\""); n != 1 {
		t.Fatalf("sends = %d, want exactly 1:\n%s", n, got)
	}
	if strings.Index(got, "resultCh <- \"\"") < strings.Index(got, `println("two")`) {
		t.Errorf("the send landed in the first goroutine, not the second:\n%s", got)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("the completion did not land through the spine:\n%s", out)
	}
}

// A go statement calling a named function declines in the
// piece's own words, through the chain, writing nothing.
func TestGoroutineResultNamedCallRefusedThroughChain(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/goroutinenamed\n\ngo 1.21\n")
	src := "package main\n\nfunc work() {}\n\nfunc main() {\n\tgo work()\n}\n"
	writeFile(t, dir, "main.go", src)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("have the goroutine return a string in main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "runs work") || !strings.Contains(out, "different edit") {
		t.Errorf("the named-call refusal did not arrive in the piece's words:\n%s", out)
	}
	if got := readBack(t, dir, "main.go"); got != src {
		t.Errorf("a refusal still wrote:\n%s", got)
	}
}

// No file named: the session's last worked-on file is the
// reading, printed before the plan.
func TestGoroutineResultSessionDefaultFile(t *testing.T) {
	dir := goroutineJimRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	workCtx.root = dir
	workCtx.lastFile = "jim/main.go"
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("have the goroutine return a string", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "reading that as jim/main.go") {
		t.Errorf("the session reading was not printed:\n%s", out)
	}
	if got := readBack(t, dir, "jim/main.go"); !strings.Contains(got, "resultCh := make(chan string)") {
		t.Errorf("the edit did not land in the session's file:\n%s", got)
	}
}

// A type outside the set refuses at the claim, in the
// piece's words, before any plan is shown.
func TestGoroutineResultUnknownTypeRefused(t *testing.T) {
	dir := goroutineJimRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	before := readBack(t, dir, "jim/main.go")
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("have the goroutine return a byte in jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "\"byte\" isn't a result type I can build this with yet") {
		t.Errorf("the unknown-type refusal did not arrive:\n%s", out)
	}
	if got := readBack(t, dir, "jim/main.go"); got != before {
		t.Errorf("a refusal still wrote:\n%s", got)
	}
}

// A missing named file: the piece edits what exists and
// says so — it never creates the file.
func TestGoroutineResultMissingFileRefused(t *testing.T) {
	dir := goroutineJimRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("have the goroutine return a string in nope/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "there's no file nope/main.go in this project") {
		t.Errorf("the missing-file refusal did not arrive:\n%s", out)
	}
}
