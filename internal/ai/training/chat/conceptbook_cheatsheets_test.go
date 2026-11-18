package chat

import "testing"

// The cheatsheet sweep (2026-11-01): Devhints' Go sheet,
// GoInterview.dev's cheatsheet, and programming-idioms.org's
// Go cheatsheet (336 idioms with a published Go answer),
// mapped the five-sources way. What the book's grammar can
// honestly assemble became the twenty-one statement rows
// pinned here (book 114 -> 135); the three sheets seat on
// the resources shelf as reading (pinned below); everything
// else was a duplicate of an existing row or an exclusion
// with a named wall — algorithms, whole programs, data
// structures, expression fragments, compositions, and the
// signature machinery (multiple returns, variadic, function
// parameters — the slots carry no signatures). The full
// per-topic mapping is in the commit message.

// The new rows claim their own lines, in the family's
// natural forms and in the every-row walk's built form.
func TestParseConceptClaimsCheatsheets(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, id string }{
		{"add a while loop to jim/main.go", "while-loop"},
		{"add a while in main to jim/main.go", "while-loop"},
		{"add an infinite loop to jim/main.go", "infinite-loop"},
		{"add a forever loop to jim/main.go", "infinite-loop"},
		{"add a type conversion to jim/main.go", "type-conversion"},
		{"add a type cast in main to jim/main.go", "type-conversion"},
		{"add a sleep to jim/main.go", "sleep"},
		{"add a pause execution to jim/main.go", "sleep"},
		{"add a delete from a map to jim/main.go", "delete-from-map"},
		{"add a delete a map entry to jim/main.go", "delete-from-map"},
		{"add a clear a map to jim/main.go", "clear-map"},
		{"add an empty a map to jim/main.go", "clear-map"},
		{"add map keys to jim/main.go", "map-keys"},
		{"add a list of map keys to jim/main.go", "map-keys"},
		{"add a read input to jim/main.go", "read-input"},
		{"add a read a line from stdin to jim/main.go", "read-input"},
		{"add a swap values to jim/main.go", "swap-values"},
		{"add a swap to jim/main.go", "swap-values"},
		{"add a sum of a slice to jim/main.go", "sum"},
		{"add a sum to jim/main.go", "sum"},
		{"add a join strings to jim/main.go", "join-strings"},
		{"add a join to jim/main.go", "join-strings"},
		{"add a repeat a string to jim/main.go", "repeat-string"},
		{"add a string repeat to jim/main.go", "repeat-string"},
		{"add a string contains to jim/main.go", "string-contains"},
		{"add a check if a string contains to jim/main.go", "string-contains"},
		{"add a file exists to jim/main.go", "file-exists"},
		{"add a check if a file exists to jim/main.go", "file-exists"},
		{"add a delete a file to jim/main.go", "delete-file"},
		{"add a delete file to jim/main.go", "delete-file"},
		{"add a rename a file to jim/main.go", "rename-file"},
		{"add a move a file to jim/main.go", "rename-file"},
		{"add a hex encoding to jim/main.go", "hex-encoding"},
		{"add a hex to jim/main.go", "hex-encoding"},
		{"add an insert into a slice to jim/main.go", "insert-into-slice"},
		{"add an insert an element into a slice to jim/main.go", "insert-into-slice"},
		{"add a remove a value from a slice to jim/main.go", "remove-value-from-slice"},
		{"add a remove all occurrences from a slice to jim/main.go", "remove-value-from-slice"},
		{"add a reverse a slice to jim/main.go", "reverse-slice"},
		{"add a reverse a list to jim/main.go", "reverse-slice"},
		{"add a print to stderr to jim/main.go", "print-stderr"},
		{"add a write to stderr to jim/main.go", "print-stderr"},
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

// The new names must not widen the claim into a neighbour's
// lines: the older rows keep their shortest forms, the
// insert-led database families keep theirs, and the
// standing exclusions still fall through.
func TestParseConceptCheatsheetsNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	neighbours := []struct{ line, id string }{
		{"add a map to jim/main.go", "map"},
		{"add a slice to jim/main.go", "slice"},
		{"add a reverse a string to jim/main.go", "reverse-string"},
		{"add a clear a slice to jim/main.go", "clear-slice"},
		{"add a delete from a slice to jim/main.go", "delete-from-slice"},
		{"add a remove an element from a slice to jim/main.go", "delete-from-slice"},
		{"add a print line to jim/main.go", "print-line"},
		{"add a join paths to jim/main.go", "file-paths"},
		{"add an insert into database to jim/main.go", "database-exec"},
		{"add a rqlite insert to jim/main.go", "rqlite-insert"},
		{"add a read file to jim/main.go", "read-file"},
		{"add a loop to jim/main.go", "for-loop"},
		{"add a string to jim/main.go", "strings-runes"},
		{"add string functions to jim/main.go", "string-functions"},
		{"add a trim a string to jim/main.go", "trim-string"},
		{"add a copy a slice to jim/main.go", "copy-slice"},
		{"add an append to a slice to jim/main.go", "append-slice"},
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
		"add an assert to jim/main.go",               // Go has no assert; panic is no building block
		"add a binary search to jim/main.go",         // an algorithm, not a language piece
		"add a stack to jim/main.go",                 // data-structure design (and generic at that)
		"add a trie to jim/main.go",                  // data-structure design
		"add a command line flag to jim/main.go",     // whole-program shape (standing exclusion)
		"add a getter to jim/main.go",                // receiver wall
		"add an uppercase conversion to jim/main.go", // the strings toolbox is reading + the string-functions row, not a claim on this phrasing
		"add a memoization to jim/main.go",           // a generic helper function: signature + type-parameter walls
		"add a quine to jim/main.go",                 // a whole program, and a parlour trick
		"add a substring to jim/main.go",             // an expression fragment — the indexes are the user's own
		"add a factorial to jim/main.go",             // an algorithm (the recursion row is where recursive functions come from)
		"add a fizzbuzz to jim/main.go",              // a whole exercise: a composition of loop + if + print
	} {
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed row %q — that line is not the composer's", line, req.entry.ID)
		}
	}
}

// The sweep's shelf: each sheet is earned by its own topic's
// goal, under its own source's header.
func TestCheatsheetsShelf(t *testing.T) {
	want := []struct {
		goal  string
		label string
		url   string
	}{
		{"cheatsheet", "Go cheatsheet — the one-page syntax reference", devhintsURL},
		{"go cheatsheet", "Go cheatsheet — the one-page syntax reference", devhintsURL},
		{"quick reference", "Go cheatsheet — the one-page syntax reference", devhintsURL},
		{"gointerview", "Go Cheatsheet 2026 — interactive reference with code examples", gointerviewURL},
		{"interview cheatsheet", "Go Cheatsheet 2026 — interactive reference with code examples", gointerviewURL},
		{"interview questions", "Go Cheatsheet 2026 — interactive reference with code examples", gointerviewURL},
		{"idioms", "Programming Idioms — Go: small tasks, each answered in Go", programmingIdiomsURL},
		{"programming idioms", "Programming Idioms — Go: small tasks, each answered in Go", programmingIdiomsURL},
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
	// The single-word goals earn exactly their own sheet —
	// no pile-on from a neighbouring source.
	if got := referenceLabels("cheatsheet"); len(got) != 1 {
		t.Errorf("goal cheatsheet earned %d entries, want exactly 1: %v", len(got), got)
	}
	if got := referenceLabels("idioms"); len(got) != 1 {
		t.Errorf("goal idioms earned %d entries, want exactly 1: %v", len(got), got)
	}
	if got := referenceLabels("gointerview"); len(got) != 1 {
		t.Errorf("goal gointerview earned %d entries, want exactly 1: %v", len(got), got)
	}
}
