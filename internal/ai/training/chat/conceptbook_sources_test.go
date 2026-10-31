package chat

import (
	"strings"
	"testing"
)

// The five-sources expansion of the concept book (2026-10-31):
// the gowebexamples web family, the go-database-sql.org database
// family, and the Switch wiki's type switch, as rows. The
// every-row walk (conceptbook_basics_test.go) already exercises
// each new row end to end off the shipped book; these tests pin
// two ask forms per new row at parse level, the deliberate
// exclusions' fall-through, and one honesty point in the spine:
// a handler-shaped declaration carries real parameters and a
// real body, so its plan must NOT wear the function scaffold's
// "empty scaffold" note.

func TestParseConceptClaimsSources(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, id string }{
		// Part 4 — the Switch wiki's type switch.
		{"add a type switch to jim/main.go", "type-switch"},
		{"add a switch on type to jim/main.go", "type-switch"},
		{"add a type assertion switch to jim/main.go", "type-switch"},
		// Part 1 — gowebexamples: routing (stdlib ServeMux form).
		{"add routing to jim/main.go", "routing"},
		{"add a router to jim/main.go", "routing"},
		{"add a serve mux to jim/main.go", "routing"},
		// Templates (html/template; the text row keeps its own).
		{"add an html template to jim/main.go", "html-template"},
		{"add a web template to jim/main.go", "html-template"},
		{"add a template to jim/main.go", "html-template"},
		// Assets and Files.
		{"add static files to jim/main.go", "static-files"},
		{"add serve static files to jim/main.go", "static-files"},
		// Forms.
		{"add a form handler named Contact to jim/main.go", "form-handler"},
		{"add an html form named Signup to jim/main.go", "form-handler"},
		{"add a form named Contact to jim/main.go", "form-handler"},
		// Middleware (Basic).
		{"add a middleware named Logging to jim/main.go", "middleware"},
		{"add a logging middleware named Log to jim/main.go", "middleware"},
		// Middleware (Advanced): the timing factory + the chain.
		{"add a timing middleware named Timing to jim/main.go", "middleware-timing"},
		{"add advanced middleware named Timer to jim/main.go", "middleware-timing"},
		{"add a middleware chain named Chain to jim/main.go", "middleware-chain"},
		{"add chain middleware named Chain to jim/main.go", "middleware-chain"},
		// Sessions (plain-cookie form).
		{"add a session handler named Login to jim/main.go", "session-handler"},
		{"add a session named Login to jim/main.go", "session-handler"},
		{"add a session cookie named Auth to jim/main.go", "session-handler"},
		// Part 2 — go-database-sql.org: open + ping (+ the pool).
		{"add a database to jim/main.go", "database"},
		{"add a database connection to jim/main.go", "database"},
		{"add an open database to jim/main.go", "database"},
		// Retrieving: many rows, one row, prepared.
		{"add a database query to jim/main.go", "database-query"},
		{"add query rows to jim/main.go", "database-query"},
		{"add a database query row to jim/main.go", "database-query-row"},
		{"add query one row to jim/main.go", "database-query-row"},
		{"add a single row query to jim/main.go", "database-query-row"},
		{"add a prepared statement to jim/main.go", "database-prepared"},
		{"add prepare a statement to jim/main.go", "database-prepared"},
		// Modifying + transactions.
		{"add a database exec to jim/main.go", "database-exec"},
		{"add a database insert to jim/main.go", "database-exec"},
		{"add a database transaction to jim/main.go", "database-transaction"},
		{"add a transaction to jim/main.go", "database-transaction"},
		// NULLs.
		{"add a database null to jim/main.go", "database-null"},
		{"add a nullable column to jim/main.go", "database-null"},
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

// The deliberate exclusions stay unclaimed — websockets (no
// stdlib server package) and password hashing (bcrypt is
// x/crypto, outside the book's stdlib discipline) have no row,
// so their sentences fall to the boundary's honest answer; a
// near-miss spelling of a shipped name refuses too.
func TestParseConceptSourcesNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	for _, line := range []string{
		"add a websocket to jim/main.go",
		"add websockets to jim/main.go",
		"add password hashing to jim/main.go",
		"add a type switcheroo to jim/main.go",
	} {
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed row %q — that line is not the composer's", line, req.entry.ID)
		}
	}
}

// A handler-shaped row end to end: the middleware lands with
// its real signature and body, and the plan does NOT call it an
// empty scaffold (the note belongs to parameterless scaffolds —
// the book's function and recursion rows — alone).
func TestConceptMiddlewareEndToEndNoScaffoldNote(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	line := "add a middleware named Logging to jim/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	if strings.Contains(out, "This is an empty scaffold") {
		t.Errorf("a middleware with parameters and a body was called an empty scaffold:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	for _, want := range []string{
		"func Logging(next http.HandlerFunc) http.HandlerFunc {",
		"log.Println(r.URL.Path)",
		`"net/http"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q:\n%s\noutput:\n%s", want, got, out)
		}
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}
