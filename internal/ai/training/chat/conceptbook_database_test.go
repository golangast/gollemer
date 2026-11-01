package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The database build (2026-11-01): the database-connect
// declaration (folder "database", the driver rule unchanged —
// no driver import is ever written), the SQLInterface shelf,
// and the gorqlite family behind its dependency gate.

// gorqliteStubSource is the offline stand-in for
// github.com/rqlite/gorqlite in fixture projects: exactly
// the API surface the gorqlite book rows' templates use
// (Open, QueryOne, WriteOneParameterized, and the two
// result/statement types), with the real client's
// signatures. Fixture projects require the module and
// replace it with this local package, so the gate sees the
// requirement in go.mod and the spine's build resolves the
// import without any network.
const gorqliteStubSource = `package gorqlite

// Connection is the stub form of the client's connection.
type Connection struct{}

// Open mirrors the real client's signature.
func Open(connURL string) (*Connection, error) { return &Connection{}, nil }

// QueryResult mirrors the real client's result: its Err
// field is the part the book's shapes name.
type QueryResult struct {
	Err error
}

// QueryOne mirrors the real client's single-statement query.
func (c *Connection) QueryOne(sql string) (QueryResult, error) {
	return QueryResult{}, nil
}

// ParameterizedStatement mirrors the real client's
// statement-plus-arguments form.
type ParameterizedStatement struct {
	Query     string
	Arguments []interface{}
}

// WriteResult mirrors the real client's write result.
type WriteResult struct {
	Err          error
	RowsAffected int64
	LastInsertID int64
}

// WriteOneParameterized mirrors the real client's
// single-statement parameterized write.
func (c *Connection) WriteOneParameterized(ps ParameterizedStatement) (WriteResult, error) {
	return WriteResult{}, nil
}
`

// seedGorqliteStub gives a fixture project the gorqlite
// dependency in its offline form: go.mod gains the require
// line and a replace pointing at a local stub package.
func seedGorqliteStub(t *testing.T, root string) {
	t.Helper()
	stub := filepath.Join(root, "gorqlitestub")
	if err := os.MkdirAll(stub, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, stub, "go.mod", "module github.com/rqlite/gorqlite\n\ngo 1.21\n")
	writeFile(t, stub, "gorqlite.go", gorqliteStubSource)
	gomod := readBack(t, root, "go.mod")
	writeFile(t, root, "go.mod", strings.TrimSpace(gomod)+
		"\n\nrequire github.com/rqlite/gorqlite v0.0.0\n\nreplace github.com/rqlite/gorqlite => ./gorqlitestub\n")
}

// closureSeedProjectGorqlite is the closure walk's habitat
// for the gorqlite statement rows: closureSeedProject's
// routing-seeded main, in a project that requires the client
// through the offline stub.
func closureSeedProjectGorqlite(t *testing.T) (string, *UserProfile) {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/closure\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	jim := filepath.Join(dir, "jim")
	if err := os.MkdirAll(jim, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, jim, "main.go", "package main\n\nfunc main() {\n\tprintln(\"ok\")\n}\n")
	seedGorqliteStub(t, dir)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var h bool
	out := captureStdout(t, func() { h = claimChain("add routing in main to jim/main.go", yes, profile) })
	if !h {
		t.Fatalf("gorqlite habitat seed: routing line fell through the chain:\n%s", out)
	}
	if got := readBack(t, dir, "jim/main.go"); !strings.Contains(got, "mux.HandleFunc") {
		t.Fatalf("gorqlite habitat seed: routing did not land its closure in jim/main.go:\n%s", got)
	}
	return dir, profile
}

// resetPendingGorqlite isolates a test from the gate's
// session state in both directions.
func resetPendingGorqlite(t *testing.T) {
	t.Helper()
	pendingGorqlite = nil
	t.Cleanup(func() { pendingGorqlite = nil })
}

// The claim matrix for the new rows, with the naming
// discipline pinned in both directions: gorqlite asks name
// the client and land on the gorqlite rows; rqlite asks keep
// landing on the HTTP rows; the existing database statement
// keeps its lines, including bare "database connection".
func TestParseConceptClaimsDatabase(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, id string }{
		{"add a database connect named Connect to jim/main.go", "database-connect"},
		{"add a database connection function named Connect to jim/main.go", "database-connect"},
		{"add a db connect function named OpenDB to store.go", "database-connect"},
		{"add a gorqlite connection named Connect to jim/main.go", "gorqlite-connect"},
		{"add a gorqlite connect named Connect to jim/main.go", "gorqlite-connect"},
		{"add a connect with gorqlite named Client to jim/main.go", "gorqlite-connect"},
		{"add a gorqlite client named Connect to jim/main.go", "gorqlite-connect"},
		{"add a gorqlite query in main to jim/main.go", "gorqlite-query"},
		{"add a query with gorqlite in main to jim/main.go", "gorqlite-query"},
		{"add a gorqlite write in main to jim/main.go", "gorqlite-write"},
		{"add a gorqlite insert in main to jim/main.go", "gorqlite-write"},
		{"add an insert with gorqlite in main to jim/main.go", "gorqlite-write"},
		// The rqlite HTTP family keeps every line it owns.
		{"add a rqlite connection in main to jim/main.go", "rqlite-connect"},
		{"add a rqlite query in main to jim/main.go", "rqlite-query"},
		{"add a rqlite insert in main to jim/main.go", "rqlite-insert"},
		// The database statement rows keep theirs.
		{"add a database connection to jim/main.go", "database"},
		{"add a database to jim/main.go", "database"},
		{"add a database query to jim/main.go", "database-query"},
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

// go.mod reading, the gate's eyes: block and single-line
// requires count; comments and other modules do not.
func TestGoModRequires(t *testing.T) {
	dir := t.TempDir()
	write := func(content string) {
		writeFile(t, dir, "go.mod", content)
	}
	if goModRequires(dir, gorqliteModule) {
		t.Error("no go.mod at all read as requiring the module")
	}
	write("module example.com/x\n\ngo 1.21\n")
	if goModRequires(dir, gorqliteModule) {
		t.Error("bare go.mod read as requiring the module")
	}
	write("module example.com/x\n\ngo 1.21\n\nrequire github.com/rqlite/gorqlite v1.0.0\n")
	if !goModRequires(dir, gorqliteModule) {
		t.Error("single-line require not seen")
	}
	write("module example.com/x\n\ngo 1.21\n\nrequire (\n\tgithub.com/rqlite/gorqlite v1.0.0 // indirect\n\tgolang.org/x/mod v0.41.0\n)\n")
	if !goModRequires(dir, gorqliteModule) {
		t.Error("block require not seen")
	}
	write("module example.com/x\n\ngo 1.21\n\n// require github.com/rqlite/gorqlite v1.0.0\nrequire golang.org/x/mod v0.41.0\n")
	if goModRequires(dir, gorqliteModule) {
		t.Error("commented-out require counted")
	}
	write("module example.com/x\n\ngo 1.21\n\nrequire github.com/rqlite/gorqlite/v2 v2.0.0\n")
	if goModRequires(dir, gorqliteModule) {
		t.Error("the v2 module path counted as the v1 requirement")
	}
}

// Part 1 end to end on a named target: the declaration
// lands, builds, and carries the driver rule in its own
// text — the TODO blank-import comment, PingContext, the
// pool setting — plus the plan note's plain warning.
func TestDatabaseConnectEndToEnd(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a database connect named Connect to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("database connect fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	if !strings.Contains(out, "unknown driver") {
		t.Errorf("plan does not carry the note's unknown-driver warning:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	for _, want := range []string{
		"func Connect() (*sql.DB, error)",
		`sql.Open("TODO: driver name", "TODO: connection string")`,
		"db.PingContext(context.Background())",
		"db.SetMaxOpenConns(10)",
		"blank import",
		`"database/sql"`,
		`"context"`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q:\n%s", want, got)
		}
	}
	// No driver import is ever written: the import block
	// holds only the standard library (the TODO comment
	// names an example driver in prose, like the database
	// statement row's own comment — that is not an import).
	imp := got[strings.Index(got, "import ("):strings.Index(got, ")")]
	for _, banned := range []string{"go-sql-driver", "lib/pq", "go-sqlite3"} {
		if strings.Contains(imp, banned) {
			t.Errorf("a driver import was written (%s):\n%s", banned, got)
		}
	}
}

// Part 1's folder default: a no-file ask creates
// database/<name>.go as package database.
func TestDatabaseConnectFolderDefault(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a database connect named Connect", yes, profile)
	})
	if !handled {
		t.Fatalf("no-file database connect fell through the chain:\n%s", out)
	}
	got := readBack(t, root, "database/connect.go")
	if !strings.Contains(got, "package database") {
		t.Errorf("database/connect.go is not package database:\n%s", got)
	}
	if !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("database/connect.go lacks the declaration:\n%s", got)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}

// A named file still wins over the folder default: nothing
// is created under database/.
func TestDatabaseConnectNamedTargetWins(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	var handled bool
	captureStdout(t, func() {
		handled = claimChain("add a database connect named Connect to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatal("named-target database connect fell through")
	}
	if _, err := os.Stat(filepath.Join(root, "database")); !os.IsNotExist(err) {
		t.Errorf("a database/ folder was created despite the named target")
	}
}

// The gate, end to end: dep absent → guidance + the ask
// held, nothing written; "installed" while still absent → a
// plain not-yet answer, ask still held; dep seeded +
// "installed" → the ORIGINAL plan re-offered, and a yes
// lands it.
func TestGorqliteGateAbsentThenInstalled(t *testing.T) {
	resetPendingGorqlite(t)
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	before := readBack(t, root, "jim/main.go")

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a gorqlite query in main to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("gorqlite ask fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "go get github.com/rqlite/gorqlite") {
		t.Errorf("gate guidance lacks the exact go get line:\n%s", out)
	}
	if strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("a plan was offered despite the missing dependency:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); got != before {
		t.Errorf("the gated ask wrote to jim/main.go:\n%s", got)
	}
	if pendingGorqlite == nil {
		t.Fatal("the gated ask was not held in the session")
	}
	if !strings.Contains(pendingGorqlite.piece, "gorqlite query") {
		t.Errorf("held piece = %q, want the gorqlite query", pendingGorqlite.piece)
	}

	// "installed" while go.mod still lacks the module.
	out = captureStdout(t, func() { handled = claimChain("installed", yes, profile) })
	if !handled {
		t.Fatalf("the follow-up fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "isn't in its require list yet") {
		t.Errorf("still-absent follow-up not answered plainly:\n%s", out)
	}
	if pendingGorqlite == nil {
		t.Error("the ask stopped being held after a still-absent follow-up")
	}
	if got := readBack(t, root, "jim/main.go"); got != before {
		t.Errorf("the still-absent follow-up wrote to jim/main.go")
	}

	// Install (the offline stub form), then "installed".
	seedGorqliteStub(t, root)
	out = captureStdout(t, func() { handled = claimChain("installed", yes, profile) })
	if !handled {
		t.Fatalf("the installed follow-up fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("the original plan was not re-offered:\n%s", out)
	}
	if pendingGorqlite != nil {
		t.Error("the ask is still held after the re-offer")
	}
	got := readBack(t, root, "jim/main.go")
	if !strings.Contains(got, "conn.QueryOne(") {
		t.Errorf("the re-offered plan did not land:\n%s", got)
	}
	if !strings.Contains(got, `"github.com/rqlite/gorqlite"`) {
		t.Errorf("the landed code does not import the client:\n%s", got)
	}
}

// A no at the re-offered plan's gate writes nothing — the
// word "installed" itself never writes, and neither does a
// declined plan.
func TestGorqliteGateReofferDeclineWritesNothing(t *testing.T) {
	resetPendingGorqlite(t)
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	before := readBack(t, root, "jim/main.go")
	captureStdout(t, func() { claimChain("add a gorqlite write in main to jim/main.go", yes, profile) })
	if pendingGorqlite == nil {
		t.Fatal("the gated write ask was not held")
	}
	seedGorqliteStub(t, root)
	out := captureStdout(t, func() { claimChain("it's installed", no, profile) })
	if !strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("the plan was not re-offered before the decline:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); got != before {
		t.Errorf("a declined re-offer wrote to jim/main.go:\n%s", got)
	}
	if pendingGorqlite != nil {
		t.Error("the ask is still held after its plan was declined")
	}
}

// With the dependency present from the start, a gorqlite
// ask plans directly — no guidance, nothing held — and the
// follow-up shapes claim nothing.
func TestGorqliteDepPresentDirectPlan(t *testing.T) {
	resetPendingGorqlite(t)
	root := liveSessionRoot(t)
	seedGorqliteStub(t, root)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a gorqlite write in main to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("gorqlite write fell through the chain:\n%s", out)
	}
	if strings.Contains(out, "doesn't have it yet") {
		t.Errorf("gate guidance printed despite the dependency:\n%s", out)
	}
	if pendingGorqlite != nil {
		t.Error("an ask was held despite the dependency being present")
	}
	got := readBack(t, root, "jim/main.go")
	if !strings.Contains(got, "WriteOneParameterized(gorqlite.ParameterizedStatement{") {
		t.Errorf("the gorqlite write did not land:\n%s", got)
	}
	// The follow-up shapes claim nothing with no held ask.
	for _, line := range []string{"installed", "done", "gorqlite is installed"} {
		var h bool
		captureStdout(t, func() { h = claimChain(line, yes, profile) })
		if h {
			t.Errorf("claimChain(%q) claimed with no held ask", line)
		}
	}
}

// The gated declaration in its folder form: dep absent, the
// guidance names the database/<name>.go address the piece
// WOULD take; after the install and the follow-up, the file
// is created there as package database.
func TestGorqliteConnectFolderGateAndLanding(t *testing.T) {
	resetPendingGorqlite(t)
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a gorqlite connection named Connect", yes, profile)
	})
	if !handled {
		t.Fatalf("gorqlite connection fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "database/connect.go") {
		t.Errorf("gate guidance does not name the folder address:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "database")); !os.IsNotExist(err) {
		t.Error("a database/ folder was created by the gated ask")
	}
	seedGorqliteStub(t, root)
	out = captureStdout(t, func() { handled = claimChain("gorqlite is installed", yes, profile) })
	if !handled {
		t.Fatalf("the gorqlite-is-installed follow-up fell through:\n%s", out)
	}
	got := readBack(t, root, "database/connect.go")
	if !strings.Contains(got, "package database") {
		t.Errorf("database/connect.go is not package database:\n%s", got)
	}
	if !strings.Contains(got, "func Connect() (*gorqlite.Connection, error)") {
		t.Errorf("database/connect.go lacks the declaration:\n%s", got)
	}
	if !strings.Contains(got, `gorqlite.Open("http://localhost:4001")`) {
		t.Errorf("database/connect.go lacks the Open call:\n%s", got)
	}
}

// The rqlite HTTP family is never gated: in a project with
// no gorqlite dependency its rows plan and land as always.
func TestRqliteRowsUnaffectedByGate(t *testing.T) {
	resetPendingGorqlite(t)
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a rqlite query in main to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("rqlite query fell through the chain:\n%s", out)
	}
	if strings.Contains(out, "go get github.com/rqlite/gorqlite") {
		t.Errorf("the HTTP rqlite row was gated:\n%s", out)
	}
	if pendingGorqlite != nil {
		t.Error("an HTTP rqlite ask was held by the gorqlite gate")
	}
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "/db/query") {
		t.Errorf("the rqlite query did not land:\n%s", got)
	}
}

// The phrasing route knows the gorqlite families under the
// shipped artifact (their judge cells are 100%): messy forms
// map onto the gorqlite canonicals, and an insert-led line
// that names no client never routes to gorqlite.
func TestPhrasingGorqliteFamiliesRoute(t *testing.T) {
	withPhrasingModel(t)
	routeCases := []struct {
		line string
		want []string
	}{
		{"could you stick a gorqlite query in main to main.go",
			[]string{"add a gorqlite query in main to main.go"}},
		{"kindly place a gorqlite write in main to web.go",
			[]string{"add a gorqlite write in main to web.go"}},
		{"please add a gorqlite insert in main to main.go",
			[]string{"add a gorqlite write in main to main.go"}},
		{"hey gollemer, put a gorqlite connection named Connect to store.go",
			[]string{"add a gorqlite connection named Connect to store.go"}},
	}
	for _, c := range routeCases {
		got, ok := PhrasingRoute(c.line)
		if !ok {
			t.Errorf("PhrasingRoute(%q) not claimed, want %q", c.line, c.want)
			continue
		}
		if strings.Join(got, " ## ") != strings.Join(c.want, " ## ") {
			t.Errorf("PhrasingRoute(%q) = %q, want %q", c.line, got, c.want)
		}
	}
}

// The gate holds on the phrasing path too: a messy gorqlite
// ask the grammar cannot read is mapped by the route onto
// its canonical — which then meets the same gate in
// runConcept, so a dep-absent project gets the guidance and
// the held ask, never a plan.
func TestGorqliteGateHoldsOnThePhrasingPath(t *testing.T) {
	resetPendingGorqlite(t)
	withPhrasingModel(t)
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	var grown bool
	captureStdout(t, func() { grown = claimChain("add print line to jim/main.go", yes, profile) })
	if !grown {
		t.Fatal("print line did not grow jim's main")
	}
	before := readBack(t, root, "jim/main.go")
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("could you stick a gorqlite query in main to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("messy gorqlite ask fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "go get github.com/rqlite/gorqlite") {
		t.Errorf("the phrasing path bypassed the gate:\n%s", out)
	}
	if strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("a plan was offered on the phrasing path despite the missing dependency:\n%s", out)
	}
	if pendingGorqlite == nil {
		t.Error("the phrasing-path gated ask was not held")
	}
	if got := readBack(t, root, "jim/main.go"); got != before {
		t.Error("the phrasing-path gated ask wrote to jim/main.go")
	}
}

// Part 2's shelf: the SQLInterface sections answer database
// resource asks under the Go Wiki header, and the connecting
// section leads a connecting goal.
func TestDatabaseShelf(t *testing.T) {
	labels := referenceLabels("database")
	joined := strings.Join(labels, "\n")
	if !strings.Contains(joined, "SQLInterface: how database/sql works|"+sqlInterfaceURL) {
		t.Errorf("referenceLabels(database) lacks the SQLInterface entry:\n%s", joined)
	}
	labels = referenceLabels("connecting to a database")
	if len(labels) == 0 || !strings.HasPrefix(labels[0], "SQLInterface: Connecting to a database") {
		t.Errorf("referenceLabels(connecting to a database) = %v", labels)
	}
	labels = referenceLabels("null values")
	if len(labels) == 0 || !strings.Contains(labels[0], "Dealing with NULL") {
		t.Errorf("referenceLabels(null values) = %v", labels)
	}
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))
	out := captureStdout(t, func() {
		if !tryBuildResources("resources for databases", nil) {
			t.Error("explicit databases resources ask not claimed")
		}
	})
	if !strings.Contains(out, "From the Go Wiki:") {
		t.Errorf("the pack does not head the Wiki entries:\n%s", out)
	}
	if !strings.Contains(out, "SQLInterface") {
		t.Errorf("the pack lacks the SQLInterface sections:\n%s", out)
	}
}
