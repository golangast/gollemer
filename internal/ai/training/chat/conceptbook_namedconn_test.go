package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The named-connection build (2026-11-01): a name belongs to
// a declaration. John's live line — "add a database
// connection named Connet to jake/db.go", into a file
// already holding a Connect function and no main — matched
// the inline database STATEMENT (the row owning the name
// "database connection"), dropped the name in silence, and
// fabricated a main to hold the snippet when what he wanted
// was the connection function in that file. Now a named
// statement ask resolves to the book's declaration sibling
// when one exists, and refuses with guidance when none
// does — the name is never dropped.

// seedJohnsDatabaseProject builds John's screenshot setup:
// jake/main.go holds a webserver, jake/db.go holds a working
// Connect declaration and no main.
func seedJohnsDatabaseProject(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/namedconn\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package namedconn\n")
	writeFile(t, dir, "jake/main.go", `package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "hi")
	})
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`)
	writeFile(t, dir, "jake/db.go", `package main

import (
	"context"
	"database/sql"
)

func Connect() (*sql.DB, error) {
	db, err := sql.Open("TODO: driver name", "TODO: connection string")
	if err != nil {
		return nil, err
	}
	if err := db.PingContext(context.Background()); err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(10)
	return db, nil
}
`)
	return dir
}

// John's exact line: the declaration plan for Connet in
// jake/db.go — the same plan the declaration's own sentence
// produces — with no fabricated main and no inline snippet.
func TestNamedConnectionResolvesToDeclaration(t *testing.T) {
	root := seedJohnsDatabaseProject(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a database connection named Connet to jake/db.go", yes, profile)
	})
	if !handled {
		t.Fatal("John's line fell through the chain")
	}
	for _, want := range []string{
		"Add the database connect Connet to",
		"from the concept book",
		"built the project and checked",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("resolved plan missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"had no main", "new main to live in", "at the end of main"} {
		if strings.Contains(out, banned) {
			t.Errorf("the statement's fabrication survived in the output (%q):\n%s", banned, out)
		}
	}
	got := readBack(t, root, "jake/db.go")
	for _, want := range []string{"func Connet() (*sql.DB, error)", "func Connect() (*sql.DB, error)"} {
		if !strings.Contains(got, want) {
			t.Errorf("jake/db.go missing %q after the resolved ask:\n%s", want, got)
		}
	}
	if strings.Contains(got, "func main(") {
		t.Errorf("a main was fabricated in jake/db.go:\n%s", got)
	}
}

// The resolution is the statement ROW's, not one name's: the
// statement's other names resolve too, and the "name is"
// introducer binds the same way — all landing as the
// declaration.
func TestNamedConnectionResolutionVariants(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)

	for _, c := range []struct{ line, decl string }{
		{"add a sql database named Store to jim/main.go", "func Store() (*sql.DB, error)"},
		{"add an open database named Opener to jim/main.go", "func Opener() (*sql.DB, error)"},
		{"add a database connection name is Linker to jim/main.go", "func Linker() (*sql.DB, error)"},
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through the chain:\n%s", c.line, out)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Fatalf("%q did not land through the spine:\n%s", c.line, out)
		}
		if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, c.decl) {
			t.Errorf("%q did not land %s in jim/main.go:\n%s", c.line, c.decl, got)
		}
	}
}

// An untargeted named connection ask takes the declaration's
// own folder default: it is created as database/<name>.go.
func TestNamedConnectionUntargetedFolderDefault(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a database connection named Pool", yes, profile)
	})
	if !handled {
		t.Fatalf("the untargeted named connection fell through:\n%s", out)
	}
	got := readBack(t, root, "database/pool.go")
	if !strings.Contains(got, "package database") || !strings.Contains(got, "func Pool() (*sql.DB, error)") {
		t.Errorf("database/pool.go does not hold the resolved declaration:\n%s", got)
	}
}

// The generic floor: a named statement with no declaration
// sibling refuses — the name is named as lost, the piece is
// named as a statement inside a function, and the working
// sentence is the same ask unnamed. Nothing is written, no
// gate is shown. The database sibling statements are pinned
// here by name, per the rule.
func TestNamedStatementWithoutSiblingRefuses(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	before := readBack(t, root, "jim/main.go")

	for _, c := range []struct{ line, name, suggestion string }{
		{"add a database query named Finder to jim/main.go", "Finder",
			`"add a database query in main to jim/main.go"`},
		{"add a database exec named Runner to jim/main.go", "Runner",
			`"add a database exec in main to jim/main.go"`},
		{"add a database query row named One to jim/main.go", "One",
			`"add a database query row in main to jim/main.go"`},
		{"add a database prepared statement named Prep to jim/main.go", "Prep",
			`"add a prepared statement in main to jim/main.go"`},
		{"add a database transaction named Tx to jim/main.go", "Tx",
			`"add a database transaction in main to jim/main.go"`},
		{"add a for loop named Spinner in Run to jim/main.go", "Spinner",
			`"add a for loop in Run to jim/main.go"`},
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through the chain", c.line)
		}
		for _, want := range []string{
			"Not adding that:",
			"is a statement — a piece of code that lives inside a function",
			"doesn't take a name of its own",
			`"` + c.name + `" would be lost`,
			c.suggestion,
		} {
			if !strings.Contains(out, want) {
				t.Errorf("%q: refusal missing %q:\n%s", c.line, want, out)
			}
		}
		if strings.Contains(out, "Apply this edit?") {
			t.Errorf("%q reached the gate instead of refusing:\n%s", c.line, out)
		}
	}
	if after := readBack(t, root, "jim/main.go"); after != before {
		t.Errorf("a named-statement refusal changed jim/main.go:\nbefore:\n%s\nafter:\n%s", before, after)
	}
}

// The other direction holds: the UNNAMED statement lines
// keep landing exactly as before — including "database
// connection", whose name ownership by the statement row is
// deliberate (0051e26) when no name rides the ask. Each case
// gets a fresh project: the statement rows open their own
// db, so two in one main would redeclare it.
func TestUnnamedDatabaseStatementsStillLand(t *testing.T) {
	for _, c := range []struct{ line, marker string }{
		{"add a database connection to jim/main.go", "err = db.Ping()"},
		{"add a database query to jim/main.go", "db.Query("},
	} {
		root := liveSessionRoot(t)
		withChatSession(t, root, true)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)

		var handled bool
		out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through the chain:\n%s", c.line, out)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Fatalf("%q did not land through the spine:\n%s", c.line, out)
		}
		got := readBack(t, root, "jim/main.go")
		if !strings.Contains(got, c.marker) {
			t.Errorf("%q did not land its statement (%q missing):\n%s", c.line, c.marker, got)
		}
		if strings.Contains(got, "(*sql.DB, error)") {
			t.Errorf("%q produced the declaration instead of the statement:\n%s", c.line, got)
		}
	}
}

// The bare word still belongs to the ambiguous-piece
// guidance, name and all: "add a database named Store" is
// bare (no connection word), so it guides — it does not
// resolve, and it writes nothing.
func TestBareNamedDatabaseStillGuides(t *testing.T) {
	root := seedJohnsDatabaseProject(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	before := readBack(t, root, "jake/db.go")

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a database named Store to jake", yes, profile)
	})
	if !handled {
		t.Fatal("the bare named database ask fell through the chain")
	}
	for _, want := range []string{
		`"add a database connection function named Store to jake"`,
		`"add a database in main to jake/main.go"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the bare ask did not get the guidance (%q missing):\n%s", want, out)
		}
	}
	if strings.Contains(out, "Apply this edit?") || strings.Contains(out, "Not adding that:") {
		t.Errorf("the bare ask was planned or refused instead of guided:\n%s", out)
	}
	if after := readBack(t, root, "jake/db.go"); after != before {
		t.Error("the guidance changed jake/db.go")
	}
	if _, err := os.Stat(filepath.Join(root, "database")); !os.IsNotExist(err) {
		t.Error("the guidance created a database/ folder")
	}
}

// Parse is untouched by the resolution: the grammar still
// reports the statement match and the bound name — the
// resolution is runConcept's, downstream of the parse.
func TestParseNamedConnectionStillStatement(t *testing.T) {
	book := defaultBookEntries(t)
	req, ok := parseConceptFrom("add a database connection named Connet to jake/db.go", book)
	if !ok {
		t.Fatal("the named connection line no longer parses")
	}
	if req.entry.ID != "database" || req.name != "Connet" || req.target != "jake/db.go" {
		t.Errorf("parse = row %q name %q target %q, want database/Connet/jake/db.go",
			req.entry.ID, req.name, req.target)
	}
}

// The sibling rule against the shipped book: exactly one
// statement row in the book has a declaration sibling today
// — database → database-connect. The sweep pins the pairing
// data-driven and singular, so a future name collision
// surfaces here instead of silently re-pairing an ask.
func TestConceptDeclarationSiblingShippedBook(t *testing.T) {
	book := defaultBookEntries(t)
	for _, e := range book {
		sib, ok := conceptDeclarationSibling(e, book)
		switch {
		case e.ID == "database":
			if !ok || sib.ID != "database-connect" {
				t.Errorf("database sibling = %q, %v — want database-connect, true", sib.ID, ok)
			}
		case e.Kind == analyze.ConceptStatement && ok:
			t.Errorf("statement %q unexpectedly has declaration sibling %q", e.ID, sib.ID)
		}
	}
}

// The sibling rule against fixture books: the pairing is
// carried by names alone (a statement name plus " function"
// on a declaration), and two declarations claiming the same
// derived name pair with nothing — ambiguity never resolves.
func TestConceptDeclarationSiblingFixtureBooks(t *testing.T) {
	stmt := analyze.ConceptEntry{ID: "gadget", Kind: analyze.ConceptStatement,
		Names: []string{"gadget", "gadget hook"}, Template: "x := 1"}
	decl := analyze.ConceptEntry{ID: "gadget-fn", Kind: analyze.ConceptDeclaration,
		Names: []string{"gadget function"}, Template: "func {name}() {}"}
	other := analyze.ConceptEntry{ID: "sprocket", Kind: analyze.ConceptDeclaration,
		Names: []string{"sprocket"}, Template: "func {name}() {}"}

	if sib, ok := conceptDeclarationSibling(stmt, []analyze.ConceptEntry{stmt, decl, other}); !ok || sib.ID != "gadget-fn" {
		t.Errorf("fixture sibling = %q, %v — want gadget-fn, true", sib.ID, ok)
	}
	rival := analyze.ConceptEntry{ID: "gadget-alt", Kind: analyze.ConceptDeclaration,
		Names: []string{"gadget function"}, Template: "func {name}() {}"}
	if _, ok := conceptDeclarationSibling(stmt, []analyze.ConceptEntry{stmt, decl, rival}); ok {
		t.Error("two declarations claiming the derived name resolved anyway")
	}
	if _, ok := conceptDeclarationSibling(stmt, []analyze.ConceptEntry{stmt, other}); ok {
		t.Error("a book with no derived declaration name resolved anyway")
	}
}
