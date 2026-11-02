package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The ambiguous-piece guidance (ambiguouspiece.go): the
// 2026-11-01 screenshot line "add handler Jim" was answered by
// the write boundary with the generic "not one of the pieces I
// can assemble today" — true and useless, since the book
// carries TWO handlers and the line just doesn't say which.
// These tests pin the disambiguation that replaced it for the
// bare-handler shapes, and the narrowness that keeps every
// other line's owner untouched.

// The parse table: the claimed shapes, with the name and
// target each one carries into its answer.
func TestParseAmbiguousPieceClaims(t *testing.T) {
	cases := []struct {
		line, name, target string
	}{
		{"add handler Jim", "Jim", ""},         // the screenshot line
		{"add a handler Jim", "Jim", ""},       // the articled bare-name form
		{"add handler named Jim", "Jim", ""},   // the named slot, no article
		{"add a handler named Jim", "Jim", ""}, // the named slot
		{"add a handler named Jim to main.go", "Jim", "main.go"},
		{"add handler Jim to jim/main.go", "Jim", "jim/main.go"},
		{"add handler named Jim to jim/main.go", "Jim", "jim/main.go"},
		{"add a handler to jim/main.go", "", "jim/main.go"}, // nameless, targeted
		{"add handler to main.go", "", "main.go"},
		{"add a handler", "", ""}, // nameless, untargeted
		{"add handler", "", ""},
		{"add handler Jim.", "Jim", ""}, // trailing full stop, trimmed like the boundary trims
	}
	for _, c := range cases {
		req, ok := parseAmbiguousPiece(c.line)
		if !ok {
			t.Errorf("parseAmbiguousPiece(%q) not claimed", c.line)
			continue
		}
		if req.name != c.name || req.target != c.target {
			t.Errorf("parseAmbiguousPiece(%q) = (name %q, target %q), want (%q, %q)",
				c.line, req.name, req.target, c.name, c.target)
		}
		if len(req.rows) != 2 {
			t.Errorf("parseAmbiguousPiece(%q) resolved %d rows, want the two handler rows", c.line, len(req.rows))
		}
	}
}

// The calibration table: lines this claim must NEVER take —
// the concrete pieces (the concept book's), other owners'
// shapes, questions, and near-miss sentences whose leftover
// words would make the guidance a lie.
func TestParseAmbiguousPieceNonClaims(t *testing.T) {
	for _, line := range []string{
		"add a form handler named Jim",            // the concept book serves it
		"add a form handler named Jim to main.go", // … targeted, too
		"add a session handler named Jim",         // the other concrete piece
		"add a form handler to main.go",           // a concrete kind word, no name
		"what is a handler",                       // a question — the go brain's
		"how do i add a handler",                  // never verb-led
		"can you add a handler named Jim",         // never verb-led
		"put handler Jim",                         // only the add shapes are this fix's
		"insert a handler named Jim",
		"add handlers Jim",                                   // the plural is not the kind word
		"add handler for the login page",                     // "for" is not a name
		"add a handler for Jim",                              // … with an article, either
		"add handler Jim that reads the form",                // a behaviour clause rides along
		"add handler named Jim to main.go that serves forms", // … in the tail, too
		"add handler named",                                  // a dangling named slot
		"add handler the",                                    // "the" is not a name
		"add a middleware named Jim",                         // a different concrete piece
		"add handler Jim on line 3",                          // a line phrase is not a target tail
		"hello how are you",
	} {
		if req, ok := parseAmbiguousPiece(line); ok {
			t.Errorf("parseAmbiguousPiece(%q) claimed as %+v — that line is not this claim's", line, req)
		}
		if runAmbiguousPiece(line) {
			t.Errorf("runAmbiguousPiece(%q) claimed a line that isn't its", line)
		}
	}
}

// The answers, verbatim where the report needs them: the
// screenshot line and the with-target variant.
func TestAmbiguousPieceAnswers(t *testing.T) {
	out := captureStdout(t, func() {
		if !runAmbiguousPiece("add handler Jim") {
			t.Error("runAmbiguousPiece did not claim the screenshot line")
		}
	})
	want := `"handler" names two different pieces here, and I can't tell which one you mean: a form handler (reads a submitted form) or a session handler (works with the user's session). Say "add a form handler named Jim" or "add a session handler named Jim" — with no file named, the piece is created in a handlers/ folder.`
	if !strings.Contains(out, want) {
		t.Errorf("screenshot-line answer:\n got %q\nwant it to contain %q", out, want)
	}

	out = captureStdout(t, func() {
		if !runAmbiguousPiece("add a handler named Jim to main.go") {
			t.Error("runAmbiguousPiece did not claim the with-target line")
		}
	})
	want = `"handler" names two different pieces here, and I can't tell which one you mean: a form handler (reads a submitted form) or a session handler (works with the user's session). Say "add a form handler named Jim to main.go" or "add a session handler named Jim to main.go".`
	if !strings.Contains(out, want) {
		t.Errorf("with-target answer:\n got %q\nwant it to contain %q", out, want)
	}
	if strings.Contains(out, "handlers/ folder") {
		t.Errorf("with-target answer mentions the folder default it isn't using:\n%s", out)
	}

	// The nameless form shows the slot honestly, in both
	// suggested sentences.
	out = captureStdout(t, func() {
		if !runAmbiguousPiece("add a handler") {
			t.Error("runAmbiguousPiece did not claim the nameless line")
		}
	})
	if !strings.Contains(out, `"add a form handler named <Name>" or "add a session handler named <Name>"`) {
		t.Errorf("nameless answer does not show the <Name> slot in both lines:\n%s", out)
	}
}

// The mechanism is rendered from the book: rename a row's
// canonical name in a fixture book and the suggested sentence
// follows it; drop a row and the ambiguity — and the claim —
// is gone.
func TestAmbiguousPieceRenderedFromTheBook(t *testing.T) {
	book := []analyze.ConceptEntry{
		{ID: "form-handler", Names: []string{"form reader", "form handler"}, Kind: "declaration", Folder: "handlers"},
		{ID: "session-handler", Names: []string{"session handler"}, Kind: "declaration", Folder: "handlers"},
	}
	req, ok := parseAmbiguousPieceFrom("add handler Jim", book)
	if !ok {
		t.Fatal("the fixture book's line was not claimed")
	}
	if ans := req.answer(); !strings.Contains(ans, `"add a form reader named Jim"`) {
		t.Errorf("the suggested sentence did not use the book's own canonical name:\n%s", ans)
	}

	one := []analyze.ConceptEntry{
		{ID: "form-handler", Names: []string{"form handler"}, Kind: "declaration", Folder: "handlers"},
	}
	if _, ok := parseAmbiguousPieceFrom("add handler Jim", one); ok {
		t.Error("claimed against a one-handler book — with one survivor there is no ambiguity to explain")
	}
	if _, ok := parseAmbiguousPieceFrom("add handler Jim", nil); ok {
		t.Error("claimed against an empty book")
	}
}

// Chain level: the concrete piece still lands through the
// concept book (the claim never sees it), the bare shapes get
// the guidance through the whole chain and write nothing, and
// the boundary keeps every other line it owns.
func TestAmbiguousPieceChainNonTheft(t *testing.T) {
	// The served line: the folder-default form handler,
	// exactly as before this claim existed.
	root := liveSessionRoot(t)
	withChatSession(t, root, false)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a form handler named SignUp", yes, profile) })
	if !handled {
		t.Fatalf("the concept-served line fell through the chain:\n%s", out)
	}
	if got := readBack(t, root, "handlers/signup.go"); !strings.Contains(got, "func SignUp(") {
		t.Errorf("the form handler did not land in handlers/signup.go:\n%s", got)
	}

	// The screenshot line, through the whole chain: claimed,
	// answered with both exact lines, and nothing written —
	// no handlers/ folder, no file of any kind.
	root = liveSessionRoot(t)
	withChatSession(t, root, false)
	out = captureStdout(t, func() { handled = claimChain("add handler Jim", yes, profile) })
	if !handled {
		t.Fatalf("the screenshot line fell through the chain:\n%s", out)
	}
	for _, want := range []string{`"add a form handler named Jim"`, `"add a session handler named Jim"`} {
		if !strings.Contains(out, want) {
			t.Errorf("chain answer for the screenshot line lacks %s:\n%s", want, out)
		}
	}
	if _, err := os.Stat(filepath.Join(root, "handlers")); !os.IsNotExist(err) {
		t.Errorf("the guidance wrote something: handlers/ exists in the fixture root")
	}
	if strings.Contains(out, "Apply this edit?") {
		t.Errorf("the guidance opened a gate — it must only talk:\n%s", out)
	}

	// The with-target line through the chain: the target is
	// carried into both suggested sentences, and the named
	// file is NOT created.
	out = captureStdout(t, func() { handled = claimChain("add a handler named Jim to main.go", yes, profile) })
	if !handled {
		t.Fatalf("the with-target line fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, `"add a form handler named Jim to main.go"`) ||
		!strings.Contains(out, `"add a session handler named Jim to main.go"`) {
		t.Errorf("chain answer did not carry the target into both lines:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "main.go")); !os.IsNotExist(err) {
		t.Errorf("the guidance created main.go — it must only talk")
	}
}

// The database claim, parsed: only the bare shape — no
// function context, no sibling-row words, an optional
// "named" name and an optional target. Its names never ride
// the bare-name slot.
func TestParseAmbiguousDatabaseClaims(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, name, target string }{
		{"add a database to jake", "", "jake"},
		{"add a database to jake/main.go", "", "jake/main.go"},
		{"add a database", "", ""},
		{"add database to jake", "", "jake"},
		{"add a database named Store to jake", "Store", "jake"},
		{"add a database in jake/main.go", "", "jake/main.go"},
		{"add a database to the file jake/main.go", "", "jake/main.go"},
	}
	for _, c := range cases {
		req, ok := parseAmbiguousDatabaseFrom(c.line, book)
		if !ok {
			t.Errorf("parseAmbiguousDatabase(%q) not claimed", c.line)
			continue
		}
		if req.word != "database" || req.name != c.name || req.target != c.target {
			t.Errorf("parseAmbiguousDatabase(%q) = word %q name %q target %q, want database/%q/%q",
				c.line, req.word, req.name, req.target, c.name, c.target)
		}
		if len(req.rows) != 2 {
			t.Errorf("parseAmbiguousDatabase(%q) resolved %d candidates, want 2", c.line, len(req.rows))
		}
	}
}

// The database claim's non-claims: every explicit database
// form keeps its book owner — the function-context statement,
// the connect declaration by any of its names, the sibling
// statement rows, and the connection-named statement row.
func TestParseAmbiguousDatabaseNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	for _, line := range []string{
		// A function context makes it the book's statement.
		"add a database in main to jake/main.go",
		"add a database in main",
		"add a database in Login to app/main.go",
		// The declaration's own sentences.
		"add a database connection function named Connect",
		"add a database connect named Connect to jim/main.go",
		"add a db connect function named OpenDB to store.go",
		// The statement row's own connection name.
		"add a database connection to jim/main.go",
		// The sibling rows: their verbs are not the bare piece.
		"add a database query in main to jim/main.go",
		"add a database query to jim/main.go",
		"add a database exec to jim/main.go",
		"add a database insert to jim/main.go",
		"add a database transaction to jim/main.go",
		"add a database null to jim/main.go",
		// Other families and shapes.
		"add a gorqlite connection named Connect",
		"add a handler named Jim",
		"what is a database",
		"add database Store to jake",
		"add a database to jake that users log into",
	} {
		if req, ok := parseAmbiguousDatabaseFrom(line, book); ok {
			t.Errorf("parseAmbiguousDatabase(%q) claimed (%s) — not its shape", line, req.answer())
		}
	}
	// The seats are disjoint: the chain-end claim never
	// takes the database word, and vice versa.
	if _, ok := parseAmbiguousPieceFrom("add a database to jake", book); ok {
		t.Error("the chain-end claim took a database line")
	}
	if _, ok := parseAmbiguousDatabaseFrom("add handler Jim", book); ok {
		t.Error("the pre-book claim took a handler line")
	}
}

// The database guidance's wording, pinned: both exact
// working lines, the target carried in as given, the inline
// line spelled with the function context the bare ask
// lacked, and the created-as note on the untargeted form.
func TestAmbiguousDatabaseAnswers(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, want string }{
		{"add a database to jake",
			`"database" names two different pieces here, and I can't tell which one you mean: a database connection function (a connection pool you open once and keep) or the inline database piece (sql.Open and a Ping inside one function). Say "add a database connection function named Connect to jake" or "add a database in main to jake/main.go".`},
		{"add a database to jake/main.go",
			`"database" names two different pieces here, and I can't tell which one you mean: a database connection function (a connection pool you open once and keep) or the inline database piece (sql.Open and a Ping inside one function). Say "add a database connection function named Connect to jake/main.go" or "add a database in main to jake/main.go".`},
		{"add a database",
			`"database" names two different pieces here, and I can't tell which one you mean: a database connection function (a connection pool you open once and keep) or the inline database piece (sql.Open and a Ping inside one function). Say "add a database connection function named Connect" or "add a database in main to main.go" — with no file named, the connection function is created as database/connect.go.`},
		{"add a database named Store to jake",
			`"database" names two different pieces here, and I can't tell which one you mean: a database connection function (a connection pool you open once and keep) or the inline database piece (sql.Open and a Ping inside one function). Say "add a database connection function named Store to jake" or "add a database in main to jake/main.go".`},
	}
	for _, c := range cases {
		req, ok := parseAmbiguousDatabaseFrom(c.line, book)
		if !ok {
			t.Errorf("parseAmbiguousDatabase(%q) not claimed", c.line)
			continue
		}
		if got := req.answer(); got != c.want {
			t.Errorf("answer(%q) =\n%s\nwant:\n%s", c.line, got, c.want)
		}
	}
}

// ambiguousFixtureBook builds a book from the default one,
// keeping only the given rows with their canonical names
// replaced — the seam for proving the guidance renders from
// whatever book is loaded.
func ambiguousFixtureBook(t *testing.T, names map[string]string) []analyze.ConceptEntry {
	t.Helper()
	var book []analyze.ConceptEntry
	for _, e := range defaultBookEntries(t) {
		canon, keep := names[e.ID]
		if !keep {
			continue
		}
		e.Names = append([]string{canon}, e.Names[1:]...)
		book = append(book, e)
	}
	return book
}

// The database guidance is rendered from the book like the
// handler one: when the row no longer carries the name the
// table prefers, the suggested sentence follows the book's
// own canonical name; drop either row and the ambiguity is
// gone, so the claim lets go.
func TestAmbiguousDatabaseRenderedFromTheBook(t *testing.T) {
	var renamed []analyze.ConceptEntry
	for _, e := range defaultBookEntries(t) {
		switch e.ID {
		case "database-connect":
			e.Names = []string{"connection maker"}
			renamed = append(renamed, e)
		case "database":
			renamed = append(renamed, e)
		}
	}
	req, ok := parseAmbiguousDatabaseFrom("add a database to jake", renamed)
	if !ok {
		t.Fatal("renamed book: bare database ask not claimed")
	}
	if got := req.answer(); !strings.Contains(got, `"add a connection maker named Connect to jake"`) {
		t.Errorf("renamed book: the connect sentence did not follow the book:\n%s", got)
	}
	oneOnly := ambiguousFixtureBook(t, map[string]string{
		"database-connect": "database connection function",
	})
	if _, ok := parseAmbiguousDatabaseFrom("add a database to jake", oneOnly); ok {
		t.Error("a book missing the statement row still claimed the bare database ask")
	}
	otherOnly := ambiguousFixtureBook(t, map[string]string{
		"database": "database",
	})
	if _, ok := parseAmbiguousDatabaseFrom("add a database to jake", otherOnly); ok {
		t.Error("a book missing the connect row still claimed the bare database ask")
	}
}

// The preempt, through the chain, on John's exact situation:
// jake/main.go holds a webserver, he asks for "a database"
// in jake, and instead of sql.Open landing in main he gets
// both exact sentences — the file byte-unchanged, no
// database/ folder, no gate. The two explicit forms the
// guidance names still land exactly as before.
func TestAmbiguousDatabaseChainPreempt(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	for _, line := range []string{
		"create a folder named jake",
		"create file jake/main.go",
		"add a webserver to jake/main.go",
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Fatalf("setup line %q fell through the chain:\n%s", line, out)
		}
	}
	before := readBack(t, root, "jake/main.go")
	if !strings.Contains(before, "ListenAndServe") {
		t.Fatalf("setup: jake/main.go does not hold the webserver:\n%s", before)
	}

	// The bare ask: guidance, and nothing else happens.
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a database to jake", yes, profile) })
	if !handled {
		t.Fatal("the bare database ask fell through the chain")
	}
	for _, want := range []string{
		`"add a database connection function named Connect to jake"`,
		`"add a database in main to jake/main.go"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance missing %s:\n%s", want, out)
		}
	}
	if strings.Contains(out, "Apply this edit?") {
		t.Errorf("the guidance asked the gate:\n%s", out)
	}
	if after := readBack(t, root, "jake/main.go"); after != before {
		t.Errorf("the guidance changed jake/main.go:\nbefore:\n%s\nafter:\n%s", before, after)
	}
	if _, err := os.Stat(filepath.Join(root, "database")); !os.IsNotExist(err) {
		t.Error("the guidance created a database/ folder")
	}

	// The explicit statement form the guidance names still
	// lands in main.
	out = captureStdout(t, func() { handled = claimChain("add a database in main to jake/main.go", yes, profile) })
	if !handled {
		t.Fatal("the explicit database statement fell through the chain")
	}
	if got := readBack(t, root, "jake/main.go"); !strings.Contains(got, "sql.Open") {
		t.Errorf("the explicit statement did not land sql.Open in jake/main.go:\n%s", got)
	}

	// And the declaration form lands its own file.
	root2 := liveSessionRoot(t)
	withChatSession(t, root2, true)
	profile2 := &UserProfile{}
	out = captureStdout(t, func() { handled = claimChain("add a database connection function named Connect", yes, profile2) })
	if !handled {
		t.Fatalf("the database connection function fell through the chain:\n%s", out)
	}
	if got := readBack(t, root2, "database/connect.go"); !strings.Contains(got, "func Connect()") {
		t.Errorf("database/connect.go lacks the declaration:\n%s", got)
	}

	// The untargeted bare ask guides too, and writes nothing.
	root3 := liveSessionRoot(t)
	withChatSession(t, root3, true)
	profile3 := &UserProfile{}
	out = captureStdout(t, func() { handled = claimChain("add a database", yes, profile3) })
	if !handled {
		t.Fatal("the untargeted bare database ask fell through the chain")
	}
	if !strings.Contains(out, "database/connect.go") {
		t.Errorf("the untargeted guidance lacks the created-as note:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root3, "database")); !os.IsNotExist(err) {
		t.Error("the untargeted guidance created a database/ folder")
	}
}
