package chat

// The function-edit catch's pins: the grammar's claims and
// declines, and the session behaviour — John's two stolen
// lines (2026-11-21) landing project-side in the composer's
// voice, questions about a function keeping their brains, an
// unresolvable name falling through untouched, the ambiguous
// name asking which one, and the file-delete refusal the
// catch was modelled on staying byte-identical.

import (
	"strings"
	"testing"
)

func TestParseFuncEditClaims(t *testing.T) {
	cases := []struct {
		line string
		want funcEditRequest
	}{
		{"remove the function named jake",
			funcEditRequest{kind: funcEditRemove, name: "jake"}},
		{"delete the function jake",
			funcEditRequest{kind: funcEditRemove, name: "jake"}},
		{"remove function jake",
			funcEditRequest{kind: funcEditRemove, name: "jake"}},
		{"remove the function jake from jake/main.go",
			funcEditRequest{kind: funcEditRemove, name: "jake", file: "jake/main.go"}},
		{"delete func jake in jim/main.go",
			funcEditRequest{kind: funcEditRemove, name: "jake", file: "jim/main.go"}},
		{"can you remove the function named jake",
			funcEditRequest{kind: funcEditRemove, name: "jake"}},
		{"please delete the function named jake.",
			funcEditRequest{kind: funcEditRemove, name: "jake"}},
		{"have the function jake return a string",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "string"}},
		{"make jake return a string",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "string"}},
		{"change the function jake to return a string",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "string"}},
		{"jake should return a string",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "string"}},
		{"have jake return an int",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "int"}},
		{"make the function jake return a Point",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "Point"}},
		{"have the function jake return a time.Duration",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "time.Duration"}},
		{"have the function Store.Save return a string",
			funcEditRequest{kind: funcEditReturn, name: "Store.Save", typ: "string"}},
		{"could you have the function jake return a bool?",
			funcEditRequest{kind: funcEditReturn, name: "jake", typ: "bool"}},
	}
	for _, tc := range cases {
		got, ok := parseFuncEdit(tc.line)
		if !ok {
			t.Errorf("%q did not claim", tc.line)
			continue
		}
		if got != tc.want {
			t.Errorf("%q = %+v, want %+v", tc.line, got, tc.want)
		}
	}
}

func TestParseFuncEditDeclines(t *testing.T) {
	for _, line := range []string{
		// Questions about a function keep their brains.
		"what does jake return",
		"how does the function jake work",
		"does jake return a string?",
		"why does jake return a string",
		// The file/folder boundary's own sentences.
		"remove the file jake/main.go",
		"delete the folder jim",
		// No function word, no claim.
		"remove jake",
		"delete jake from the project",
		// Neighbours' sentences.
		"rename the function jake to john", // the rename claim's
		"add a function named jake",        // a creation
		"add a function like jake named jake2",
		// Not type-shaped tails.
		"make jake return the favor",
		"make jake return",
		"jake should return",
		"jake should go home",
		// Statements, not asks.
		"the function should return a string",
		"jake returns a string",
		"the goroutine returns a string",
		// Incomplete shapes.
		"remove the function",
		"remove the function named",
		"have the function return a string",
		"change jake to a string",
	} {
		if got, ok := parseFuncEdit(line); ok {
			t.Errorf("%q claimed as %+v — it must not", line, got)
		}
	}
}

// funcEditJakeRoot is the screenshot's shape in miniature: a
// module whose jake/main.go holds one empty func jake.
func funcEditJakeRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/funceditjake\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package funceditjake\n")
	writeFile(t, dir, "jake/main.go", "package main\n\nfunc jake() {}\n\nfunc main() {}\n")
	return dir
}

const funcEditRemoveAnswer = "I can't remove a function from your code — taking code out is yours alone, like deleting a file. jake is in jake/main.go (line 3): delete its declaration there in your editor, along with any lines that call it. What I can do: rename jake everywhere it appears, or explain what it does and what calls it."

const funcEditReturnAnswer = "I can't change what a function returns yet — that edit isn't one of my pieces. You asked for jake to return a string; jake is in jake/main.go (line 3), and its signature today is func jake(). Making it return one means editing that declaration in your editor and returning a value wherever it finishes. What I can do: rename jake or one of its parameters, add a parameter to it, or explain how it's called today."

// John's first stolen line: "remove the function named
// jake" — answered by the talk brain with a greeting on
// 2026-11-21 — now lands project-side with the composer's
// decline, and writes nothing.
func TestFuncEditRemoveLineLandsProjectSide(t *testing.T) {
	dir := funcEditJakeRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	before := readBack(t, dir, "jake/main.go")
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("remove the function named jake", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through the whole claim chain:\n%s", out)
	}
	if !strings.Contains(out, funcEditRemoveAnswer) {
		t.Errorf("the composer's decline was not the answer:\n%s", out)
	}
	if strings.Contains(out, "good to see you") {
		t.Errorf("the talk brain's greeting appeared:\n%s", out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != before {
		t.Errorf("the decline still wrote:\n%s", got)
	}
}

// John's second stolen line: "have the function jake
// return a string" — answered by the [go] brain with a
// string-comparison lecture on 2026-11-21 — now lands
// project-side with the composer's decline.
func TestFuncEditReturnLineLandsProjectSide(t *testing.T) {
	dir := funcEditJakeRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("have the function jake return a string", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through the whole claim chain:\n%s", out)
	}
	if !strings.Contains(out, funcEditReturnAnswer) {
		t.Errorf("the composer's decline was not the answer:\n%s", out)
	}
	if strings.Contains(out, "EqualFold") {
		t.Errorf("the [go] brain's string lecture appeared:\n%s", out)
	}
}

// Questions about jake are not edits: the catch does not
// fire, and the lines keep the fate they have today (the
// claim chain declines them; routing and the go/analyze
// brains own them from there).
func TestFuncEditQuestionsFallThrough(t *testing.T) {
	dir := funcEditJakeRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	for _, line := range []string{
		"what does jake return",
		"how does the function jake work",
		"does jake return a string?",
	} {
		if runFuncEdit(line, yes) {
			t.Errorf("%q was claimed by the function-edit catch", line)
		}
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if handled {
			t.Errorf("%q was claimed by the chain — it belongs to the go/analyze brains:\n%s", line, out)
		}
	}
}

// The resolution gate: the same shapes naming a function
// the project doesn't have decline silently — the line
// keeps today's fate, so social talk carrying the word
// "function" is never stolen.
func TestFuncEditUnresolvedNameFallsThrough(t *testing.T) {
	dir := funcEditJakeRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	// The goroutine piece's own sentence parses in this
	// grammar with the name "goroutine" — and still is not
	// this catch's: no function by that name resolves here,
	// and the goroutine claim above owns the line regardless.
	if runFuncEdit("have the goroutine return a string", yes) {
		t.Errorf("the goroutine sentence was claimed by the function-edit catch")
	}
	for _, line := range []string{
		"remove the function named jim",
		"delete the function bella",
		"have the function bella return a string",
		"remove the function named jake from other/main.go", // right name, wrong file
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if handled {
			t.Errorf("%q was claimed — the name does not resolve here:\n%s", line, out)
		}
	}
}

// Two functions with one name: the catch asks which one,
// and the number completes the ask with that function's
// decline.
func TestFuncEditAmbiguousNameAsks(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/funceditamb\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package funceditamb\n")
	writeFile(t, dir, "a/a.go", "package a\n\nfunc jake() {}\n")
	writeFile(t, dir, "b/b.go", "package b\n\nfunc jake() {}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("remove the function named jake", yes, profile)
	})
	if !handled {
		t.Fatalf("the line fell through:\n%s", out)
	}
	if !strings.Contains(out, "There's more than one function named jake — which one do you mean? Nothing's been written.") {
		t.Fatalf("no which-one question was asked:\n%s", out)
	}
	if !strings.Contains(out, "1. a/a.go (line 3)") || !strings.Contains(out, "2. b/b.go (line 3)") {
		t.Fatalf("the candidates were not both named:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("1", yes, profile) })
	if !handled {
		t.Fatalf("the answer fell through:\n%s", out)
	}
	want := strings.Replace(funcEditRemoveAnswer, "jake/main.go", "a/a.go", 1)
	if !strings.Contains(out, want) {
		t.Errorf("the completion did not decline for a/a.go's jake:\n%s", out)
	}
}

// The refusal the catch was modelled on is untouched:
// removing a FILE still gets the create boundary's answer,
// byte for byte.
func TestFuncEditFileDeleteRefusalUnchanged(t *testing.T) {
	dir := funcEditJakeRoot(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("remove the file jake/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("the file line fell through:\n%s", out)
	}
	want := "I can't delete files or folders — deleting is yours alone. In your terminal, the command is: rm jake/main.go. What I can do: create a file or folder here behind a [y/n] gate, and explain any command in plain words before you run it."
	if !strings.Contains(out, want) {
		t.Errorf("the file-delete refusal changed:\n%s", out)
	}
}
