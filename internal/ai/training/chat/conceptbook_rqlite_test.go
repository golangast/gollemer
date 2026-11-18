package chat

import (
	"strings"
	"testing"
)

// The rqlite family (2026-10-31): four concept-book rows that
// teach a beginner's program to talk to a rqlite node over its
// HTTP API — connect (a /status check), create table, insert,
// query — plus the "how do I run rqlite" guidance claim. The
// every-row walk (conceptbook_basics_test.go) already exercises
// each row end to end off the shipped book; these tests pin
// the ask forms at parse level, one landing per row with its
// content and its runtime note, the guidance claim's tight
// shape, and the phrasing layer's hold on the new families —
// including the regression the retrain first shipped with:
// the insert-led thefts the naming-evidence guard closed.

func TestParseConceptClaimsRqlite(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, id string }{
		// Connect: the base address, a client, a /status check.
		{"add a rqlite connection to jim/main.go", "rqlite-connect"},
		{"add a connect to rqlite to jim/main.go", "rqlite-connect"},
		{"add a rqlite client to jim/main.go", "rqlite-connect"},
		// Create table: an execute carrying a CREATE TABLE.
		{"add a rqlite create table to jim/main.go", "rqlite-create-table"},
		{"add a rqlite table to jim/main.go", "rqlite-create-table"},
		{"add a create table in rqlite to jim/main.go", "rqlite-create-table"},
		// Insert: a parameterized execute.
		{"add a rqlite insert to jim/main.go", "rqlite-insert"},
		{"add an insert into rqlite to jim/main.go", "rqlite-insert"},
		{"add a rqlite insert row to jim/main.go", "rqlite-insert"},
		// Query: a /db/query whose answer is decoded.
		{"add a rqlite query to jim/main.go", "rqlite-query"},
		{"add a query rqlite to jim/main.go", "rqlite-query"},
		{"add a select from rqlite to jim/main.go", "rqlite-query"},
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

// What the build deliberately did NOT give rows stays
// unclaimed: clustering and backups are node operations, not
// program code; a websocket row would leave the stdlib; and
// bare "rqlite" names no row at all.
func TestParseConceptRqliteNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	for _, line := range []string{
		"add a rqlite websocket to jim/main.go",
		"add a rqlite to jim/main.go",
		"add a rqlite backup to jim/main.go",
	} {
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed row %q — that line is not the composer's", line, req.entry.ID)
		}
	}
}

// One landing per row, in a fresh jim each time: the content
// markers land, the project builds, and the plan carries the
// row's honesty note — the code compiles without a node, but
// a running rqlited is what answers at runtime, and the node
// owns the database file.
func TestConceptRqliteEndToEnd(t *testing.T) {
	cases := []struct {
		id   string
		line string
		want []string
		note string
	}{
		{"rqlite-connect", "add a rqlite connection to jim/main.go",
			[]string{`rqliteAddr := "http://localhost:4001"`, "rqliteClient := &http.Client{Timeout: 10 * time.Second}", `rqliteAddr + "/status"`, `"net/http"`, `"time"`},
			"The node owns the database file"},
		{"rqlite-create-table", "add a rqlite create table to jim/main.go",
			[]string{"/db/execute", "CREATE TABLE TODO: your table", "json.Marshal"},
			"created inside the node's own database file"},
		{"rqlite-insert", "add a rqlite insert to jim/main.go",
			[]string{"/db/execute", "INSERT INTO TODO: your table(name) VALUES(?)", "[]any{[]any{"},
			"must already be running"},
		{"rqlite-query", "add a rqlite query to jim/main.go",
			[]string{"/db/query", "SELECT TODO: your query", "level=strong", "Columns []string"},
			"must already be running"},
	}
	for _, c := range cases {
		t.Run(c.id, func(t *testing.T) {
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
			out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
			if !handled {
				t.Fatalf("%q fell through the claim chain:\n%s", c.line, out)
			}
			if !strings.Contains(out, c.note) {
				t.Errorf("plan output missing the row's note %q:\n%s", c.note, out)
			}
			if !strings.Contains(out, "built the project and checked") {
				t.Errorf("spine build/verify not reported:\n%s", out)
			}
			got := readBack(t, root, "jim/main.go")
			for _, want := range c.want {
				if !strings.Contains(got, want) {
					t.Errorf("jim/main.go missing %q:\n%s", want, got)
				}
			}
		})
	}
}

// The guidance claim: tight shapes only, and the answer is
// printed commands the user runs himself — the install
// routes and the node start, with the chat's no-install
// honesty up front.
func TestRqliteGuidePins(t *testing.T) {
	for _, line := range []string{
		"how do i run rqlite",
		"how do i install rqlite?",
		"how can i get rqlite running",
		"how to start rqlite",
		"what is rqlite",
		"install rqlite",
	} {
		if !parseRqliteGuide(line) {
			t.Errorf("parseRqliteGuide(%q) = false, want the guidance claim", line)
		}
	}
	for _, line := range []string{
		"add a rqlite query to jim/main.go",
		"how do i run a webserver",
		"which package is rqlite in",
		"how do i install go",
		"how do i use rqlite with my existing cluster of five nodes",
	} {
		if parseRqliteGuide(line) {
			t.Errorf("parseRqliteGuide(%q) = true — that line belongs to another claim", line)
		}
	}
	out := captureStdout(t, func() {
		if !tryRqliteGuide("how do i run rqlite") {
			t.Error("tryRqliteGuide declined its own shape")
		}
	})
	for _, want := range []string{
		"curl -fsSL https://rqlite.io/install.sh | sh",
		"docker run -p 4001:4001 rqlite/rqlite",
		"brew install rqlite",
		"rqlited -node-id=1 data/",
		"the database file lives inside it, owned by the node",
		"I can't install or run programs for you",
		"http://localhost:4001",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance answer missing %q:\n%s", want, out)
		}
	}
}

// The phrasing layer hears the new families in held-out
// dress, and the naming-evidence guard holds the line the
// first retrain crossed: an insert-led ask for some OTHER
// piece is never read as a rqlite insert on the shared verb.
func TestPhrasingRqliteFamiliesEndToEnd(t *testing.T) {
	withPhrasingModel(t)
	routeCases := []struct {
		line string
		want []string
	}{
		{"could you stick a rqlite query in main to main.go",
			[]string{"add a rqlite query in main to main.go"}},
		{"kindly place a rqlite connection in main to web.go",
			[]string{"add a rqlite connection in main to web.go"}},
		{"please add a rqlite insert in main to main.go",
			[]string{"add a rqlite insert in main to main.go"}},
		{"hey gollemer, put a rqlite create table in main to store.go",
			[]string{"add a rqlite create table in main to store.go"}},
		// The theft shapes, measured on the held-out judges:
		// the generic insert-led dialect belongs to the piece
		// the line names, never to rqlite.
		{"insert an http server into api/main.go",
			[]string{"add a webserver to api/main.go"}},
		{"kindly insert a router in Nomad to api/main.go",
			[]string{"add a routing in Nomad to api/main.go"}},
		{"create a session with the name Vespa plus insert a xml inside it to nest/main.go",
			[]string{"add a session handler named Vespa to nest/main.go", "add a xml in Vespa to nest/main.go"}},
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
