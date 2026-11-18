package chat

import "testing"

// The basics sweep (2026-11-01): YourBasic's go-to guide,
// Callicoder's tutorial series, and the official anchors
// (A Tour of Go, the go.dev tutorial list) mapped the
// five-sources way. The topics the book's grammar can
// honestly assemble became the eighteen statement rows
// pinned here; the reference-shaped topics seat on the
// resources shelf (pinned below); the rest were duplicates
// of existing rows or exclusions with named walls — the
// full mapping is in the commit message.

// The new rows claim their own lines, in the family's
// natural forms and in the every-row walk's built form.
func TestParseConceptClaimsSweep(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, id string }{
		{"add an append to a slice to jim/main.go", "append-slice"},
		{"add an append to slice in main to jim/main.go", "append-slice"},
		{"add a slice append to jim/main.go", "append-slice"},
		{"add a map lookup to jim/main.go", "map-lookup"},
		{"add a look up a key to jim/main.go", "map-lookup"},
		{"add a find a key in a map to jim/main.go", "map-lookup"},
		{"add an environment variable to jim/main.go", "env-var"},
		{"add an env var in main to jim/main.go", "env-var"},
		{"add a current working directory to jim/main.go", "working-directory"},
		{"add a working directory to jim/main.go", "working-directory"},
		{"add a read a file line by line to jim/main.go", "read-lines"},
		{"add read lines from a file to jim/main.go", "read-lines"},
		{"add an append to a file to jim/main.go", "append-file"},
		{"add append text to a file in main to jim/main.go", "append-file"},
		{"add a walk a directory to jim/main.go", "walk-directory"},
		{"add list files recursively to jim/main.go", "walk-directory"},
		{"add a log to a file to jim/main.go", "log-to-file"},
		{"add logging to a file to jim/main.go", "log-to-file"},
		{"add a string builder to jim/main.go", "string-builder"},
		{"add a build a string to jim/main.go", "string-builder"},
		{"add a split a string to jim/main.go", "split-string"},
		{"add a split string in main to jim/main.go", "split-string"},
		{"add a trim whitespace to jim/main.go", "trim-string"},
		{"add a trim a string to jim/main.go", "trim-string"},
		{"add a reverse a string to jim/main.go", "reverse-string"},
		{"add a string reverse to jim/main.go", "reverse-string"},
		{"add a delete from a slice to jim/main.go", "delete-from-slice"},
		{"add a remove an element from a slice to jim/main.go", "delete-from-slice"},
		{"add a clear a slice to jim/main.go", "clear-slice"},
		{"add an empty a slice to jim/main.go", "clear-slice"},
		{"add a slice search to jim/main.go", "slice-search"},
		{"add a find in a slice to jim/main.go", "slice-search"},
		{"add a slice copy to jim/main.go", "copy-slice"},
		{"add a copy a slice to jim/main.go", "copy-slice"},
		{"add a shuffle a slice to jim/main.go", "shuffle"},
		{"add a shuffle to jim/main.go", "shuffle"},
		{"add an execution time to jim/main.go", "execution-time"},
		{"add a measure execution time in main to jim/main.go", "execution-time"},
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

// The new names must not widen the claim into a
// neighbour's lines: the older rows keep their shortest
// forms, and the standing exclusions still fall through.
func TestParseConceptSweepNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	neighbours := []struct{ line, id string }{
		{"add a map to jim/main.go", "map"},
		{"add a slice to jim/main.go", "slice"},
		{"add a string to jim/main.go", "strings-runes"},
		{"add string functions to jim/main.go", "string-functions"},
		{"add sorting to jim/main.go", "sorting"},
		{"add a log line to jim/main.go", "log-line"},
		{"add logging to jim/main.go", "log-line"},
		{"add a read file to jim/main.go", "read-file"},
		{"add a write file to jim/main.go", "write-file"},
		{"add a list files to jim/main.go", "directories"},
		{"add a temporary file to jim/main.go", "temp-file"},
		{"add a random number to jim/main.go", "random"},
		{"add a current time to jim/main.go", "time-now"},
	}
	for _, c := range neighbours {
		req, ok := parseConceptFrom(c.line, book)
		if !ok {
			t.Errorf("parseConcept(%q) not claimed, want neighbour row %q", c.line, c.id)
			continue
		}
		if req.entry.ID != c.id {
			t.Errorf("parseConcept(%q) = row %q, want neighbour %q", c.line, req.entry.ID, c.id)
		}
	}
	for _, line := range []string{
		"add a method to jim/main.go",       // receiver wall (yourbasic/callicoder/tour methods)
		"add a generic type to jim/main.go", // type-parameter slot (tour generics)
		"add a flag to jim/main.go",         // whole-program shape (command-line flags)
		"add a type alias to jim/main.go",   // the aliased type is a slot the book lacks
		"add a stringer to jim/main.go",     // a String() method: receiver wall
		"add a queue to jim/main.go",        // data-structure design, not a piece
		"add a prime check to jim/main.go",  // an algorithm, not a language piece
		"add a uuid to jim/main.go",         // no stdlib UUID: external dependency
	} {
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed row %q — that line is not the composer's", line, req.entry.ID)
		}
	}
}

// The sweep's shelf: each new entry is earned by its own
// topic's goal, under its own source's header.
func TestSweepShelf(t *testing.T) {
	want := []struct {
		goal  string
		label string
		url   string
	}{
		{"go vs java", "Go vs. Java: 15 main differences / Java to Go in-depth tutorial", yourbasicURL},
		{"nil map", "Gotcha: Assignment to entry in nil map", yourbasicURL},
		{"nil pointer dereference", "Gotcha: Invalid memory address or nil pointer dereference", yourbasicURL},
		{"single-value context", "Gotcha: Multiple-value in single-value context", yourbasicURL},
		{"arrays are values", "Gotcha: Array won't change (arrays are values)", yourbasicURL},
		{"regexp cheat sheet", "Regexp tutorial and cheat sheet", yourbasicURL},
		{"string handling", "Go string handling overview [cheat sheet]", yourbasicURL},
		{"printf formatting", "fmt.Printf formatting tutorial and cheat sheet", yourbasicURL},
		{"bitwise operators", "Bitwise operators [cheat sheet]", yourbasicURL},
		{"operators", "Operators: complete list", yourbasicURL},
		{"project layout", "Start a new Go project [standard layout]", yourbasicURL},
		{"data races", "Data races explained", yourbasicURL},
		{"race detector", "How to detect data races", yourbasicURL},
		{"deadlocks", "How to debug deadlocks", yourbasicURL},
		{"parallel computation", "3 rules for efficient parallel computation", yourbasicURL},
		{"object-oriented programming", "Object-oriented programming without inheritance", yourbasicURL},
		{"zero values", "Default zero values for all Go types", yourbasicURL},
		{"int vs int64", "Pick the right one: int vs. int64", yourbasicURL},
		{"redeclaring variables", "Redeclaring variables", yourbasicURL},
		{"initialization order", "Package initialization and program execution order", yourbasicURL},
		{"basic types", "Basic Types, Operators, and Type Conversion", callicoderTypesURL},
		{"data types", "Basic Types, Operators, and Type Conversion", callicoderTypesURL},
	}
	for _, w := range want {
		found := false
		for _, l := range referenceLabels(w.goal) {
			if l == w.label+"|"+w.url {
				found = true
			}
		}
		if !found {
			t.Errorf("goal %q: shelf entry %q not earned; got %v", w.goal, w.label, referenceLabels(w.goal))
		}
	}
}
