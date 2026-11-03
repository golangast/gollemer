package chat

import (
	"strings"
	"testing"
)

// The no-host guidance (index-aware routing fix, ITEM 1):
// a concept statement aimed at a file that declares NO
// functions used to get a refusal spelling a sentence that
// could never work in that file — "add a database in main
// to jake/database/database.go", when that file is package
// database and declares no functions at all, main included.
// Now the planner, told the statement's book sibling by
// runConcept, spells the sibling's declaration sentence —
// or, for a statement with no sibling, two honest steps.

// seedNoHostProject builds the defect's ground truth: a
// program main.go, plus jake/database/database.go holding
// only its package clause — no functions, no main.
func seedNoHostProject(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/nohost\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {}\n")
	writeFile(t, dir, "jake/database/database.go", "package database\n")
	return dir
}

const wantD2Guidance = `jake/database/database.go declares no functions, so there's nowhere to put the database — that piece lives inside a function. The piece that fits a file like this is the database connection function, a function of its own: say "add a database connection function named Connect to jake/database/database.go".`

// The D2 line verbatim: the refusal spells the sibling's
// sentence exactly, writes nothing — and the spelled
// sentence, fed back, lands the Connect declaration in that
// very file. Guidance that works, proven.
func TestNoHostDatabaseGuidanceVerbatimAndWorks(t *testing.T) {
	root := seedNoHostProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	before := readBack(t, root, "jake/database/database.go")

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add database connection to jake/database/database.go", yes, profile)
	})
	if !handled {
		t.Fatal("the D2 line fell through the chain")
	}
	if !strings.Contains(out, wantD2Guidance) {
		t.Errorf("refusal does not contain the exact sibling guidance:\nwant substring:\n%s\noutput:\n%s", wantD2Guidance, out)
	}
	if strings.Contains(out, "has no main to put") || strings.Contains(out, "Apply this edit?") {
		t.Errorf("the defective refusal or a gate survived:\n%s", out)
	}
	if after := readBack(t, root, "jake/database/database.go"); after != before {
		t.Errorf("the refusal changed the file:\nbefore:\n%s\nafter:\n%s", before, after)
	}

	// The spelled sentence, fed back verbatim, lands.
	out = captureStdout(t, func() {
		handled = claimChain("add a database connection function named Connect to jake/database/database.go", yes, profile)
	})
	if !handled {
		t.Fatalf("the spelled sibling sentence fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Fatalf("the spelled sibling sentence did not land through the spine:\n%s", out)
	}
	got := readBack(t, root, "jake/database/database.go")
	if !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("the spelled sentence did not land Connect in the file:\n%s", got)
	}
}

// The explicit-name variant: a named function that doesn't
// exist, in a file with no functions at all, gets the same
// sibling guidance — not the old "has no function named"
// text naming a host that was never there.
func TestNoHostDatabaseExplicitMissingGuidance(t *testing.T) {
	root := seedNoHostProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	before := readBack(t, root, "jake/database/database.go")

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a database connection in Missing to jake/database/database.go", yes, profile)
	})
	if !handled {
		t.Fatal("the explicit-name line fell through the chain")
	}
	if !strings.Contains(out, wantD2Guidance) {
		t.Errorf("explicit refusal does not contain the exact sibling guidance:\nwant substring:\n%s\noutput:\n%s", wantD2Guidance, out)
	}
	if strings.Contains(out, "has no function named") {
		t.Errorf("the old explicit refusal survived:\n%s", out)
	}
	if after := readBack(t, root, "jake/database/database.go"); after != before {
		t.Error("the explicit refusal changed the file")
	}
}

// The two-step case: print line is a statement with NO
// declaration sibling — the shipped-book sweep
// (TestConceptDeclarationSiblingShippedBook) pins database
// as the book's only statement with a sibling, and no
// declaration row carries "print line function" (or any
// other print-line name plus " function"). It is not
// gorqlite-gated (its only import is fmt). Into a
// function-less file it gets the two-step guidance exactly,
// and nothing is written.
func TestNoHostPrintLineTwoStepGuidance(t *testing.T) {
	root := seedNoHostProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	before := readBack(t, root, "jake/database/database.go")

	want := `jake/database/database.go declares no functions, so there's nowhere to put the print line — that piece lives inside a function. Give the file a function first: say "add a function named <Name> to jake/database/database.go", then say "add a print line in <Name> to jake/database/database.go" with your function's name in both places.`
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a print line to jake/database/database.go", yes, profile)
	})
	if !handled {
		t.Fatal("the two-step line fell through the chain")
	}
	if !strings.Contains(out, want) {
		t.Errorf("refusal does not contain the exact two-step guidance:\nwant substring:\n%s\noutput:\n%s", want, out)
	}
	if strings.Contains(out, "Apply this edit?") {
		t.Errorf("the two-step refusal reached the gate:\n%s", out)
	}
	if after := readBack(t, root, "jake/database/database.go"); after != before {
		t.Error("the two-step refusal changed the file")
	}
}
