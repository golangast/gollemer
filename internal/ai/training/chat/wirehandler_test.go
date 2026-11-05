package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The wiring claim: the grammar's shapes and its non-claims
// at the parser, then John's 2026-11-03 journey end to end
// through the claim chain — a form handler landed in
// jim/main.go, then the sentence that used to fall through:
// "connect the handler to the webserver".

func TestParseWireHandlerClaims(t *testing.T) {
	cases := []struct {
		line    string
		name    string
		route   string
		partial bool
	}{
		{"connect the handler to the webserver", "", "", false},
		{"connect the Jake handler to the webserver", "Jake", "", false},
		{"connect the handler Jake to the webserver", "Jake", "", false},
		{"wire the Jake handler to the server", "Jake", "", false},
		{"wire Jake to the webserver", "Jake", "", false},
		{"wire up the handler to the webserver", "", "", false},
		{"hook up the Jake handler to the web server", "Jake", "", false},
		{"register the Jake handler with the webserver", "Jake", "", false},
		{"attach a handler to the server", "", "", false},
		{"connect the jake handler to the webserver", "jake", "", false},
		{"connect the Jake handler to the webserver at /contact", "Jake", "/contact", false},
		{"connect the handler to the webserver at /", "", "/", false},
		// The 2026-11-05 widening: link/plug verbs, the
		// "into" preposition, and the split phrasal — the
		// particle behind the handler's name.
		{"wire Jake up to the webserver", "Jake", "", false},
		{"hook Jake up to the webserver", "Jake", "", false},
		{"connect the handler up to the webserver", "", "", false},
		{"link the Jake handler to the webserver", "Jake", "", false},
		{"link Jake to the server", "Jake", "", false},
		{"plug Jake into the webserver", "Jake", "", false},
		{"connect Jake into the webserver", "Jake", "", false},
		{"wire Jake into the webserver at /jake", "Jake", "/jake", false},
		{"wire Jake up", "Jake", "", true},
		// Started, then stopped: claimed, answered in wiring terms.
		{"connect the handler", "", "", true},
		{"connect the Jake handler", "Jake", "", true},
		{"connect the handler to", "", "", true},
		{"connect the handler to the", "", "", true},
		{"wire Jake to", "Jake", "", true},
	}
	for _, c := range cases {
		req, ok := parseWireHandler(c.line)
		if !ok {
			t.Errorf("%q: not claimed", c.line)
			continue
		}
		if req.name != c.name || req.route != c.route || req.partial != c.partial {
			t.Errorf("%q: got {name:%q route:%q partial:%v}, want {name:%q route:%q partial:%v}",
				c.line, req.name, req.route, req.partial, c.name, c.route, c.partial)
		}
	}
}

func TestParseWireHandlerNonClaims(t *testing.T) {
	for _, line := range []string{
		// The neighbours' sentences — the claim requires a
		// handler reference AND a server word, in this order.
		"add a webserver to jim/main.go",
		"add a database connection function named Connect to jim/main.go",
		"add a database connection named Connect to jake/database/database.go",
		"add a gorqlite connection named Rq to store/db.go",
		"add a database query in main to jim/main.go",
		"serve home.html to webtest/main.go",
		"serve an html file",
		"connect to the database",
		"connect the webserver to the handler", // direction is fixed: handler into server
		// "point" was probed and deliberately not widened
		// (2026-11-05): "at" is this grammar's route slot,
		// so the metaphor can't be read without breaking
		// "…at /path".
		"point Jake at the webserver",
		"link to the database",
		// Bare "hook" without its particle, in the named
		// form too: the phrasal verb claims with "up" or
		// not at all (the unnamed form is pinned below in
		// the family's original non-claims).
		"hook Jake to the webserver",
		"connect the handler to the database",  // no server word
		"connect the handler to the webserver please",
		"connect the handler to the webserver at",
		"how do I connect a handler to a server?",
		"make a function named Jake in jim/main.go and call it in jim/main.go",
		"run the program in jim/main.go",
		"add a form handler named Jake to jim/main.go",
		"connect",
		"wire",
		"hook the handler to the webserver", // "hook" needs its "up"
	} {
		if req, ok := parseWireHandler(line); ok {
			t.Errorf("%q: claimed as %+v, want no claim", line, req)
		}
	}
}

// wireJimRoot is John's screenshot state as a fixture: a
// webserver program in jim/main.go with the form handler Jake
// already landed in it.
func wireJimRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/wirejim\n\ngo 1.21\n")
	// A root library package beside jim/: with jim/ as the
	// project's ONLY package (a lone main in a subfolder),
	// the spine's `go build ./...` tries to write the binary
	// onto the jim directory itself and fails — the artifact
	// the paces workout's builds() helper documents. John's
	// real projects always carry more packages; so does this
	// fixture.
	writeFile(t, dir, "keep.go", "package wirejim\n\n// Keep is a placeholder so the fixture has a library package.\nfunc Keep() {}\n")
	writeFile(t, dir, "jim/main.go", `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	fmt.Fprintln(w, "thanks")
}

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Printf("visit: %s %s\n", r.Method, r.URL.Path)
		fmt.Fprintln(w, "hello — your webserver is running")
	})
	fmt.Println("listening at http://localhost:8080 — press Ctrl+C to stop")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`)
	return dir
}

// The headline journey: the bare sentence wires the one
// unregistered handler in the project — Jake — into jim's
// webserver, behind the gate, build-verified.
func TestWireHandlerJohnsJourneyThroughChain(t *testing.T) {
	root := wireJimRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	line := "connect the handler to the webserver"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the claim chain:\n%s", line, out)
	}
	requireNoAnalyzeOutput(t, line, out)
	got := readBack(t, root, "jim/main.go")
	reg := `http.HandleFunc("/jake", Jake)`
	ri, li := strings.Index(got, reg), strings.Index(got, "http.ListenAndServe")
	if ri < 0 {
		t.Fatalf("the registration never landed:\n%s", got)
	}
	if ri > li {
		t.Errorf("the registration landed after the serve call:\n%s", got)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	if !strings.Contains(out, "registers Jake at /jake") {
		t.Errorf("done line does not name the wiring:\n%s", out)
	}
	// Ask again (still the bare sentence): the plain
	// already-registered answer for the bare form — every
	// reachable handler is registered, Jake's answer named —
	// and no second line appears.
	out2 := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("repeat %q fell through:\n%s", line, out2)
	}
	for _, want := range []string{"already registered", "Jake answers at /jake", "Nothing to add"} {
		if !strings.Contains(out2, want) {
			t.Errorf("repeat ask's answer missing %q:\n%s", want, out2)
		}
	}
	if n := strings.Count(readBack(t, root, "jim/main.go"), reg); n != 1 {
		t.Errorf("jim/main.go holds %d registrations for Jake, want 1", n)
	}
	// And at another path: refused with both paths named.
	var handled3 bool
	out3 := captureStdout(t, func() {
		handled3 = claimChain("connect the Jake handler to the webserver at /signup", yes, profile)
	})
	if !handled3 {
		t.Fatalf("at-path repeat fell through:\n%s", out3)
	}
	if !strings.Contains(out3, "already registered at /jake") || !strings.Contains(out3, "/signup") {
		t.Errorf("different-path ask did not get the refusal:\n%s", out3)
	}
}

// The gate decline writes nothing.
func TestWireHandlerGateDeclineWritesNothing(t *testing.T) {
	root := wireJimRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	before, _ := os.ReadFile(filepath.Join(root, "jim", "main.go"))
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("connect the Jake handler to the webserver", no, profile) })
	if !handled {
		t.Fatalf("named wiring fell through the chain:\n%s", out)
	}
	after, _ := os.ReadFile(filepath.Join(root, "jim", "main.go"))
	if string(after) != string(before) {
		t.Errorf("a declined gate changed the file:\n%s", out)
	}
	if !strings.Contains(out, "Not applied") {
		t.Errorf("decline not reported:\n%s", out)
	}
}

// The unfinished sentence is claimed and answered in wiring
// terms — it never reaches the intent layer.
func TestWireHandlerPartialClaimed(t *testing.T) {
	root := wireJimRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("connect the handler", yes, profile) })
	if !handled {
		t.Fatalf("unfinished wiring sentence fell through:\n%s", out)
	}
	if !strings.Contains(out, "connect the Jake handler to the webserver") {
		t.Errorf("partial answer does not spell the full sentence:\n%s", out)
	}
}

// Non-theft through the chain: the database family's connect
// lines keep their own claims with the wiring claim seated
// ahead of the book. (The named-connection suite pins the
// landing itself; this pins that the wiring claim does not
// answer them.)
func TestWireHandlerDoesNotStealDatabaseConnect(t *testing.T) {
	for _, line := range []string{
		"connect the handler to the webserver",
		"connect the Jake handler to the webserver",
	} {
		if _, ok := parseWireHandler(line); !ok {
			t.Errorf("%q should be the wiring claim's", line)
		}
	}
	for _, line := range []string{
		"add a database connection function named Connect to jim/main.go",
		"add a database connection named Connect",
		"connect to the database",
	} {
		if _, ok := parseWireHandler(line); ok {
			t.Errorf("%q stolen by the wiring grammar", line)
		}
	}
}
