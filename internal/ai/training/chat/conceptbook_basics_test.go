package chat

import (
	"strings"
	"testing"
)

// The Go-basics expansion of the concept book (2026-10-30): the
// gobyexample checklist, as rows. These tests pin two ask forms
// per shipped row at parse level, and walk the WHOLE shipped
// book end to end — every row, through the claim chain, in a
// scratch project, asserting the spine built and checked it.
// The walk iterates the book itself (never a hand-list), so a
// row added to data/concepts.json is exercised the day it
// lands: if its template does not survive the spine's own AST
// check + build verification in a bare program file, its
// subtest fails by name.

func TestParseConceptClaimsBasics(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, id string }{
		{"add a variable to jim/main.go", "variable"},
		{"add a new variable to jim/main.go", "variable"},
		{"add a constant to jim/main.go", "constant"},
		{"put a const to jim/main.go", "constant"},
		{"add a for loop to jim/main.go", "for-loop"},
		{"add a loop to jim/main.go", "for-loop"},
		{"add a range loop to jim/main.go", "range-loop"},
		{"add a range to jim/main.go", "range-loop"},
		{"add an if statement to jim/main.go", "if-else"},
		{"add an if else to jim/main.go", "if-else"},
		{"add a switch statement to jim/main.go", "switch"},
		{"add a switch to jim/main.go", "switch"},
		{"add an array to jim/main.go", "array"},
		{"put an array to jim/main.go", "array"},
		{"add a slice to jim/main.go", "slice"},
		{"add a slice in main to jim/main.go", "slice"},
		{"add a map to jim/main.go", "map"},
		{"add a dictionary to jim/main.go", "map"},
		{"add a closure to jim/main.go", "closure"},
		{"add an anonymous function to jim/main.go", "closure"},
		{"add a pointer to jim/main.go", "pointers"},
		{"add pointers to jim/main.go", "pointers"},
		{"add a string to jim/main.go", "strings-runes"},
		{"add strings and runes to jim/main.go", "strings-runes"},
		{"add an error to jim/main.go", "errors"},
		{"add a new error to jim/main.go", "errors"},
		{"add a goroutine to jim/main.go", "goroutine"},
		{"add a go routine to jim/main.go", "goroutine"},
		{"add a channel to jim/main.go", "channel"},
		{"add a channel in main to jim/main.go", "channel"},
		{"add a buffered channel to jim/main.go", "buffered-channel"},
		{"add channel synchronization to jim/main.go", "channel-sync"},
		{"add a channel sync to jim/main.go", "channel-sync"},
		{"add a select to jim/main.go", "select"},
		{"add a non-blocking channel to jim/main.go", "select"},
		{"add a timeout to jim/main.go", "timeout"},
		{"add a close channel to jim/main.go", "close-channel"},
		{"add closing a channel to jim/main.go", "close-channel"},
		{"add a channel range to jim/main.go", "channel-range"},
		{"add a range over a channel to jim/main.go", "channel-range"},
		{"add a timer to jim/main.go", "timer"},
		{"add a ticker to jim/main.go", "ticker"},
		{"add a wait group to jim/main.go", "waitgroup"},
		{"add a waitgroup to jim/main.go", "waitgroup"},
		{"add an atomic counter to jim/main.go", "atomic-counter"},
		{"add an atomic to jim/main.go", "atomic-counter"},
		{"add a mutex to jim/main.go", "mutex"},
		{"add sorting to jim/main.go", "sorting"},
		{"add a sort to jim/main.go", "sorting"},
		{"add a custom sort to jim/main.go", "sorting-by"},
		{"add a sort by function to jim/main.go", "sorting-by"},
		{"add a defer to jim/main.go", "defer"},
		{"add a deferred call to jim/main.go", "defer"},
		{"add a recover to jim/main.go", "recover"},
		{"add panic recovery to jim/main.go", "recover"},
		{"add string functions to jim/main.go", "string-functions"},
		{"add string formatting to jim/main.go", "string-format"},
		{"add a sprintf to jim/main.go", "string-format"},
		{"add a text template to jim/main.go", "text-template"},
		{"add a regular expression to jim/main.go", "regexp"},
		{"add a regex to jim/main.go", "regexp"},
		{"add json to jim/main.go", "json"},
		{"add a marshal to json to jim/main.go", "json"},
		{"add xml to jim/main.go", "xml"},
		{"add a marshal to xml to jim/main.go", "xml"},
		{"add the current time to jim/main.go", "time-now"},
		{"add a time to jim/main.go", "time-now"},
		{"add an epoch to jim/main.go", "epoch"},
		{"add unix time to jim/main.go", "epoch"},
		{"add time formatting to jim/main.go", "time-format"},
		{"add a format a time to jim/main.go", "time-format"},
		{"add a random number to jim/main.go", "random"},
		{"add a random to jim/main.go", "random"},
		{"add number parsing to jim/main.go", "number-parsing"},
		{"add parse numbers to jim/main.go", "number-parsing"},
		{"add url parsing to jim/main.go", "url-parsing"},
		{"add a parse a url to jim/main.go", "url-parsing"},
		{"add a sha256 hash to jim/main.go", "sha256"},
		{"add a hash to jim/main.go", "sha256"},
		{"add base64 encoding to jim/main.go", "base64"},
		{"add a base64 to jim/main.go", "base64"},
		{"add file reading to jim/main.go", "read-file"},
		{"add a read file to jim/main.go", "read-file"},
		{"add file writing to jim/main.go", "write-file"},
		{"add a write file to jim/main.go", "write-file"},
		{"add a file path to jim/main.go", "file-paths"},
		{"add file paths to jim/main.go", "file-paths"},
		{"add a list a directory to jim/main.go", "directories"},
		{"add list files to jim/main.go", "directories"},
		{"add a temporary file to jim/main.go", "temp-file"},
		{"add a temp file to jim/main.go", "temp-file"},
		{"add a log line to jim/main.go", "log-line"},
		{"add logging to jim/main.go", "log-line"},
		{"add an http client to jim/main.go", "http-client"},
		{"add an http get to jim/main.go", "http-client"},
		{"add a context to jim/main.go", "context"},
		{"add a spawn a process to jim/main.go", "spawn-process"},
		{"add an external command to jim/main.go", "spawn-process"},
		{"add an enum named Level to jim/main.go", "enum"},
		{"add an enumeration named Level to jim/main.go", "enum"},
		{"add an embedded struct named Car to jim/main.go", "embedding"},
		{"add struct embedding named Car to jim/main.go", "embedding"},
		{"add a recursive function named Walk to jim/main.go", "recursion"},
		{"add recursion named Walk to jim/main.go", "recursion"},
	}
	for _, c := range cases {
		req, ok := parseConceptFrom(c.line, book)
		if !ok {
			t.Errorf("parseConcept(%q) not claimed, want row %q", c.line, c.id)
			continue
		}
		if req.entry.ID != c.id {
			t.Errorf("parseConcept(%q) = row %q, want %q", c.line, req.entry.ID, c.id)
		}
	}
}

// New-book names must not widen the claim past the family's
// rules: a mangled phrasing whose slot zone carries an unknown
// word still refuses, and the topics deliberately left out of
// the book (methods above all) still fall through.
func TestParseConceptBasicsNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	for _, line := range []string{
		"add a time out to jim/main.go",         // "out" is no slot word; the timeout row is one word
		"add a loop over to jim/main.go",        // "over" is no slot word
		"add a method to jim/main.go",           // methods are not a book row (a receiver can't be bound)
		"add a generic function to jim/main.go", // generics need a type-parameter slot the book doesn't have
		"add a flag to jim/main.go",             // command-line flags: a whole-program shape, not a row
	} {
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed row %q — that line is not the composer's", line, req.entry.ID)
		}
	}
}

// The every-row walk: each shipped row, one fresh scratch
// project, one sentence built from the row's own first phrasing
// — statements land in a grown main, declarations join a file
// that already has one (the spine's build is absolute: a bare
// package-main file cannot take a declaration until it has a
// main, the shipped session tests' own lesson).
func TestConceptBookEveryRowEndToEnd(t *testing.T) {
	book := defaultBookEntries(t)
	for _, entry := range book {
		entry := entry
		t.Run(entry.ID, func(t *testing.T) {
			root := liveSessionRoot(t)
			withChatSession(t, root, true)
			profile := &UserProfile{}
			createJimThroughTheChain(t, profile)

			line := "add a " + entry.Names[0] + " to jim/main.go"
			if entry.Kind == "declaration" {
				var grown bool
				captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
				if !grown {
					t.Fatal("print line did not grow jim's main")
				}
				line = "add a " + entry.Names[0] + " named Rowtest to jim/main.go"
			}
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
			if !handled {
				t.Fatalf("%q fell through the claim chain:\n%s", line, out)
			}
			requireNoAnalyzeOutput(t, line, out)
			if !strings.Contains(out, "built the project and checked") {
				t.Fatalf("row %s: spine build/verify not reported for %q:\n%s", entry.ID, line, out)
			}
			got := readBack(t, root, "jim/main.go")
			marker := conceptRowMarker(entry.Template)
			if marker == "" || !strings.Contains(got, marker) {
				t.Errorf("row %s: jim/main.go does not contain %q after %q:\n%s", entry.ID, marker, line, got)
			}
			if entry.Kind == "declaration" && !strings.Contains(got, "Rowtest") {
				t.Errorf("row %s: the declaration's name did not land:\n%s", entry.ID, got)
			}
		})
	}
}

// conceptRowMarker renders a row's template with its slots
// defaulted and returns the first line's first token — the
// smallest piece of the row that must be visible in the file
// the spine wrote.
func conceptRowMarker(template string) string {
	r := strings.NewReplacer("{name}", "Rowtest", "{text}", "", "{methods}", "")
	first := strings.TrimSpace(strings.SplitN(r.Replace(template), "\n", 2)[0])
	if i := strings.IndexAny(first, " \t"); i >= 0 {
		first = first[:i]
	}
	return first
}
