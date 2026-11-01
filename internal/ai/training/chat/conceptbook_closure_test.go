package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// closureSeedProject builds the hostile fixture: a scratch
// project whose jim/main.go main has already taken the routing
// row, so its body holds a mux.HandleFunc closure and ends
// with the routing template's trailing TODO comment — the
// environment the type-switch bug needed.
func closureSeedProject(t *testing.T) (string, *UserProfile) {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/closure\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	jim := filepath.Join(dir, "jim")
	if err := os.MkdirAll(jim, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, jim, "main.go", "package main\n\nfunc main() {\n\tprintln(\"ok\")\n}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var h bool
	out := captureStdout(t, func() { h = claimChain("add routing in main to jim/main.go", yes, profile) })
	if !h {
		t.Fatalf("seed: routing line fell through the chain:\n%s", out)
	}
	if got := readBack(t, dir, "jim/main.go"); !strings.Contains(got, "mux.HandleFunc") {
		t.Fatalf("seed: routing did not land its closure in jim/main.go:\n%s", got)
	}
	return dir, profile
}

// Regression pin for the type-switch inserter bug (found
// 2026-10-31 by the main chat's live verification): a concept
// STATEMENT added to a function whose body already contains a
// closure was refused by the assembler's own containment check
// ("main doesn't contain the statement after assembly") even
// though the same statement lands fine in a fresh function.
// The routing row is the natural seed: its template puts a
// mux.HandleFunc closure inside main.
func TestConceptStatementIntoClosureBody(t *testing.T) {
	withPhrasingModel(t)
	dir, profile := closureSeedProject(t)

	var h2 bool
	out2 := captureStdout(t, func() { h2 = claimChain("add a type switch in main to jim/main.go", yes, profile) })
	if !h2 {
		t.Fatalf("type switch line fell through the chain:\n%s", out2)
	}
	if strings.Contains(out2, "internal error in my assembler") {
		t.Fatalf("type switch was refused by the assembler's self-check:\n%s", out2)
	}
	got := readBack(t, dir, "jim/main.go")
	if !strings.Contains(got, "switch value.(type)") {
		t.Fatalf("type switch did not land in jim/main.go:\n%s", got)
	}
	tsAt := strings.Index(got, "switch value.(type)")
	muxAt := strings.Index(got, "mux.HandleFunc")
	if tsAt < muxAt {
		t.Fatalf("type switch landed before the routing code, want it at the end of main:\n%s", got)
	}
}

// The class proof for the same bug: EVERY statement row in the
// shipped book, one per fresh project seeded with the routing
// landing (a closure in main, the body ending in the routing
// template's trailing comment), must land and pass the spine's
// build verification — the same bar the every-row walk sets in
// a bare main, now in the hostile environment. The walk
// iterates the book itself, so a statement row added tomorrow
// is held to this too.
//
// One named exception: the routing row itself cannot be added
// to a routing-seeded main — its template declares `mux :=`
// and the seed already declares it, so the build honestly
// rejects the duplicate. Its subtest seeds the closure with
// the closure row instead (still a function body holding a
// closure) and holds routing to the same landing+build bar
// there.
func TestConceptBookEveryStatementRowIntoClosureBody(t *testing.T) {
	withPhrasingModel(t)
	book := defaultBookEntries(t)
	for _, entry := range book {
		entry := entry
		if entry.Kind != "statement" {
			continue
		}
		t.Run(entry.ID, func(t *testing.T) {
			var dir string
			var profile *UserProfile
			if entry.ID == "routing" {
				// Alternate seed (see the comment above): the
				// closure row lands a closure in main.
				var p *UserProfile
				dir, p = closureSeedProjectAlt(t)
				profile = p
			} else {
				dir, profile = closureSeedProject(t)
			}
			line := "add a " + entry.Names[0] + " to jim/main.go"
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
			if !handled {
				t.Fatalf("%q fell through the claim chain:\n%s", line, out)
			}
			if strings.Contains(out, "internal error in my assembler") {
				t.Fatalf("row %s refused by the assembler's self-check:\n%s", entry.ID, out)
			}
			if !strings.Contains(out, "built the project and checked") {
				t.Fatalf("row %s: spine build/verify not reported for %q:\n%s", entry.ID, line, out)
			}
			got := readBack(t, dir, "jim/main.go")
			marker := conceptRowMarker(entry.Template)
			if marker == "" || !strings.Contains(got, marker) {
				t.Errorf("row %s: jim/main.go does not contain %q after %q:\n%s", entry.ID, marker, line, got)
			}
		})
	}
}

// closureSeedProjectAlt seeds jim/main.go's main with the
// closure row instead of routing — used only by the routing
// row's matrix subtest (see above).
func closureSeedProjectAlt(t *testing.T) (string, *UserProfile) {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/closure\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	jim := filepath.Join(dir, "jim")
	if err := os.MkdirAll(jim, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, jim, "main.go", "package main\n\nfunc main() {\n\tprintln(\"ok\")\n}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var h bool
	out := captureStdout(t, func() { h = claimChain("add a closure in main to jim/main.go", yes, profile) })
	if !h {
		t.Fatalf("alt seed: closure line fell through the chain:\n%s", out)
	}
	if got := readBack(t, dir, "jim/main.go"); !strings.Contains(got, "func(") {
		t.Fatalf("alt seed: closure did not land in jim/main.go:\n%s", got)
	}
	return dir, profile
}
