package chat

import (
	"errors"
	"fmt"
	"net"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The run-the-program suite. The regression at its heart is
// John's screenshot (2026-11-01): "run the program
// jake/main.go" ran the HOST project's `make start` and then
// narrated "It ran fine" — so these tests drive the real
// claim chain against fixture projects, one of which HAS a
// Makefile with a start target, and check what actually ran.

// withRunSession points the chat globals at dir (cold: no
// analyzed project), resets the session's program, and stubs
// the browser opener so no test ever opens a real browser.
// The stub records the URLs it was asked to open.
func withRunSession(t *testing.T, dir string) *[]string {
	t.Helper()
	withChatSession(t, dir, false)
	oldProg := sessionProgram
	oldOpen := openBrowserURL
	opened := &[]string{}
	openBrowserURL = func(u string) error {
		*opened = append(*opened, u)
		return nil
	}
	t.Cleanup(func() {
		stopProgramAtChatEnd()
		sessionProgram = oldProg
		openBrowserURL = oldOpen
	})
	sessionProgram = nil
	return opened
}

func runFixtureRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/runfixture\n\ngo 1.21\n")
	return dir
}

const quickProgram = `package main

import "fmt"

func main() {
	fmt.Println("hello from the quick program")
}
`

// waitForLog polls the session program's log until it
// contains want (the visit-log case), or fails the test.
func waitForLog(t *testing.T, want string) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if sessionProgram != nil {
			if lines, _ := sessionProgram.log.snapshot(); strings.Contains(strings.Join(lines, "\n"), want) {
				return
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("program log never contained %q", want)
}

// portFree reports whether nothing is listening on the port
// (the stopped-program proof), retrying briefly while the
// socket winds down.
func portFree(port string) bool {
	for i := 0; i < 20; i++ {
		ln, err := net.Listen("tcp", ":"+port)
		if err == nil {
			ln.Close()
			return true
		}
		time.Sleep(100 * time.Millisecond)
	}
	return false
}

func TestParseProgramLine(t *testing.T) {
	cases := []struct {
		line   string
		shape  programShape
		target string
	}{
		{"run the program jake/main.go", shapeRunProgram, "jake/main.go"},
		{"run the program in jake/main.go", shapeRunProgram, "jake/main.go"},
		{"top", shapeTop, ""},
		{"/top", shapeTop, ""},
		{"what's running", shapeTop, ""},
		{"show running programs", shapeTop, ""},
		{"stop the program in jake", shapeNone, ""},
		{"run jake/main.go", shapeRunProgram, "jake/main.go"},
		{"run program jake/main.go", shapeRunProgram, "jake/main.go"},
		{"run ./jake/main.go", shapeRunProgram, "./jake/main.go"},
		{"run the program jake/main.go.", shapeRunProgram, "jake/main.go"},
		// start|launch read as the run verb (2026-11-05) —
		// same rest shapes, so a build ask still declines.
		{"start the program in jake/main.go", shapeRunProgram, "jake/main.go"},
		{"start jake/main.go", shapeRunProgram, "jake/main.go"},
		{"launch jake/main.go", shapeRunProgram, "jake/main.go"},
		{"start a webserver in jake/main.go", shapeNone, ""},
		// Politeness wrappers (2026-11-09): the closed set
		// comes off and the bare shape applies — John's exact
		// line first. The set is closed: "will you" is not in
		// it, and a wrapped build ask still declines.
		{"can you run jim/main.go", shapeRunProgram, "jim/main.go"},
		{"can you run the program jim/main.go", shapeRunProgram, "jim/main.go"},
		{"can you run the program in jim/main.go", shapeRunProgram, "jim/main.go"},
		{"could you run jim/main.go", shapeRunProgram, "jim/main.go"},
		{"please run jim/main.go", shapeRunProgram, "jim/main.go"},
		{"run jim/main.go please", shapeRunProgram, "jim/main.go"},
		{"can you run jim/main.go please", shapeRunProgram, "jim/main.go"},
		{"can you please run jim/main.go", shapeRunProgram, "jim/main.go"},
		{"can you start jim/main.go", shapeRunProgram, "jim/main.go"},
		{"can you launch the program in jim/main.go", shapeRunProgram, "jim/main.go"},
		{"can you stop the program", shapeStopProgram, ""},
		{"could you stop it", shapeStopProgram, ""},
		{"can you stop jim/main.go", shapeStopProgram, "jim/main.go"},
		{"please stop", shapeStopBare, ""},
		{"can you show the program log", shapeShowLog, ""},
		{"can you show visits", shapeShowLog, ""},
		{"show the program log please", shapeShowLog, ""},
		{"can you show running programs", shapeTop, ""},
		{"can you run make start", shapeNone, ""},
		{"can you run make eval", shapeNone, ""},
		{"can you start a webserver in jim", shapeNone, ""},
		{"can you start a webserver in jim/main.go", shapeNone, ""},
		{"can you run it", shapeNone, ""},
		{"can you run the program", shapeNone, ""},
		// The server shapes (2026-11-09): the target is a
		// folder to resolve at claim time, or a .go file —
		// no pathiness test at parse, resolution is the gate.
		{"run the server jim", shapeRunServer, "jim"},
		{"run server jim", shapeRunServer, "jim"},
		{"start the server jim", shapeRunServer, "jim"},
		{"launch the server jim", shapeRunServer, "jim"},
		{"run the server in jim", shapeRunServer, "jim"},
		{"run server in jim", shapeRunServer, "jim"},
		{"run the server jim/main.go", shapeRunServer, "jim/main.go"},
		{"run the server jim/", shapeRunServer, "jim/"},
		{"can you run the server jim", shapeRunServer, "jim"},
		{"can you start the server jim", shapeRunServer, "jim"},
		{"run the server", shapeNone, ""},
		{"run server", shapeNone, ""},
		{"can you kill the program", shapeNone, ""},
		{"will you run jim/main.go", shapeNone, ""},
		{"can you", shapeNone, ""},
		{"please", shapeNone, ""},
		{"kill the program", shapeNone, ""},
		{"run the program notes.txt", shapeRunProgram, "notes.txt"},
		{"run make eval", shapeNone, ""},
		{"run make start", shapeNone, ""},
		{"run it", shapeNone, ""},
		{"run the project", shapeNone, ""},
		{"run the program", shapeNone, ""}, // beginner's watch keeps it
		{"run go mod tidy", shapeNone, ""},
		{"stop", shapeStopBare, ""},
		{"stop it", shapeStopProgram, ""},
		{"stop the program", shapeStopProgram, ""},
		{"stop the server", shapeStopProgram, ""},
		{"stop jake/main.go", shapeStopProgram, "jake/main.go"},
		{"stop teaching", shapeNone, ""},
		{"show visits", shapeShowLog, ""},
		{"show the program log", shapeShowLog, ""},
		{"show the log", shapeShowLog, ""},
		{"show me routeDomain", shapeNone, ""},
		{"add a print line in main to main.go", shapeNone, ""},
	}
	for _, tc := range cases {
		shape, target := parseProgramLine(tc.line)
		if shape != tc.shape || target != tc.target {
			t.Errorf("parseProgramLine(%q) = (%v, %q), want (%v, %q)", tc.line, shape, target, tc.shape, tc.target)
		}
	}
}

// The screenshot pin: a fixture project whose Makefile HAS a
// start target (the host project's shape), a program beside
// it, and the ask — the program must run; make start must
// not; and no success may be narrated for anything but the
// program's real finish.
func TestRunProgramNotMakeStart(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "Makefile", "start:\n\t@echo HOST MAKE START RAN\n")
	writeFile(t, dir, "jake/main.go", quickProgram)
	withRunSession(t, dir)
	profile := &UserProfile{Directives: map[string]bool{"do_all_means_all": true}}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program jake/main.go", yes, profile) })
	if !handled {
		t.Fatalf("the run ask was not claimed:\n%s", out)
	}
	for _, want := range []string{"hello from the quick program", "It ran and finished (exit code 0)", "[auto-yes: your 'do all' directive]"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"HOST MAKE START RAN", "make start", "It ran fine"} {
		if strings.Contains(out, banned) {
			t.Errorf("output contains %q — the make-start failure shape:\n%s", banned, out)
		}
	}
	if sessionProgram == nil || !sessionProgram.hasExited() {
		t.Error("the quick program should have run and finished inside the window")
	}
}

// The 2026-11-09 screenshot pin, end to end: John's layout —
// a jim/ folder holding a web server (two files, so the
// package is the program), the HOST project's own Makefile
// with a start target — and his exact line, "can you run
// jim/main.go". The prefixed ask must claim; the gate must
// name jim's package; jim's server must be the thing that
// runs, with its URL read from its own output; "top" must
// list it; "stop the program" must stop it — and the invented
// narration from the make-start path must appear nowhere in
// the whole exchange.
func TestRunProgramCanYouPrefixRunsJim(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "Makefile", "start:\n\t@echo MAKE START OPENED THE CHAT INSTEAD\n")
	writeFile(t, dir, "jim/main.go", `package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Printf("visit: %s %s\n", r.Method, r.URL.Path)
		fmt.Fprintln(w, greeting())
	})
	fmt.Println("listening at http://localhost:8080 — jim's own server")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`)
	writeFile(t, dir, "jim/help.go", "package main\n\nfunc greeting() string {\n\treturn \"hello from jim\"\n}\n")
	opened := withRunSession(t, dir)
	profile := &UserProfile{}

	var prompt string
	recYes := func(p string) (string, error) { prompt = p; return "y", nil }

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("can you run jim/main.go", recYes, profile) })
	if !handled {
		t.Fatalf("John's prefixed run ask was not claimed:\n%s", out)
	}
	if !strings.Contains(prompt, "go run ./jim") {
		t.Errorf("gate prompt = %q, want it to name jim's package (go run ./jim)", prompt)
	}
	for _, want := range []string{"is running in the background", "http://localhost:8080"} {
		if !strings.Contains(out, want) {
			t.Errorf("server start report missing %q:\n%s", want, out)
		}
	}
	if len(*opened) != 1 || (*opened)[0] != "http://localhost:8080" {
		t.Errorf("browser opened %v, want [http://localhost:8080]", *opened)
	}
	if !programRunning() {
		t.Fatal("jim's server should be running in the background")
	}
	transcript := out

	// A real visit proves jim's program — not the host's
	// Makefile — is what answered.
	resp, err := http.Get("http://localhost:8080/")
	if err != nil {
		t.Fatalf("visit to jim's server failed: %v", err)
	}
	body := make([]byte, 128)
	n, _ := resp.Body.Read(body)
	resp.Body.Close()
	if !strings.Contains(string(body[:n]), "hello from jim") {
		t.Errorf("visit body = %q, want jim's own answer", string(body[:n]))
	}
	waitForLog(t, "visit: GET /")

	out = captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "jim/main.go — running") {
		t.Errorf("top does not list jim's program as running:\n%s", out)
	}
	transcript += out

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || !strings.Contains(out, "Stopped jim/main.go") || programRunning() {
		t.Errorf("\"stop the program\" did not stop jim's server:\n%s", out)
	}
	transcript += out
	if !portFree("8080") {
		t.Error("port 8080 still bound after stop")
	}

	// The invented narration and the host's make-start output
	// must appear nowhere in the whole exchange.
	for _, banned := range []string{"It ran fine", "the program talking", "MAKE START OPENED", "make start"} {
		if strings.Contains(transcript, banned) {
			t.Errorf("transcript contains %q — the make-start failure shape:\n%s", banned, transcript)
		}
	}
}

// A quick program behind the plain gate: asked, answered yes,
// runs, finishes, and its own line is what gets reported.
func TestRunProgramQuickExit(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "hello/main.go", quickProgram)
	withRunSession(t, dir)

	// The scripted ask echoes its prompt the way the real
	// LineReader does (ReadLine prints the prompt it is given),
	// so the gate's wording is visible to assert on.
	echoYes := func(prompt string) (string, error) {
		fmt.Print(prompt)
		return "y", nil
	}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run hello/main.go", echoYes, &UserProfile{}) })
	if !handled {
		t.Fatalf("not claimed:\n%s", out)
	}
	if !strings.Contains(out, "Run it? go run hello/main.go") {
		t.Errorf("gate prompt missing or wrong:\n%s", out)
	}
	if !strings.Contains(out, "hello from the quick program") || !strings.Contains(out, "It ran and finished (exit code 0)") {
		t.Errorf("run report wrong:\n%s", out)
	}
}

// Declining the gate runs nothing.
func TestRunProgramGateDecline(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "hello/main.go", quickProgram)
	withRunSession(t, dir)
	no := func(string) (string, error) { return "n", nil }

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program hello/main.go", no, &UserProfile{}) })
	if !handled || !strings.Contains(out, "OK — not run.") {
		t.Errorf("decline not honored:\n%s", out)
	}
	if sessionProgram != nil {
		t.Error("a declined run must not start anything")
	}
}

// The form rule: a file with siblings runs as its package —
// the sibling's code is in the program that runs.
func TestRunProgramPackageForm(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "pkg/main.go", "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(greeting())\n}\n")
	writeFile(t, dir, "pkg/help.go", "package main\n\nfunc greeting() string {\n\treturn \"from the sibling file\"\n}\n")
	withRunSession(t, dir)

	if arg, cmdText, note := programForm(mustAbs(t, dir), "pkg/main.go"); arg != "./pkg" || cmdText != "go run ./pkg" || note == "" {
		t.Errorf("programForm multi-file = (%q, %q, %q), want package form", arg, cmdText, note)
	}
	writeFile(t, dir, "solo/main.go", quickProgram)
	if arg, cmdText, note := programForm(mustAbs(t, dir), "solo/main.go"); arg != "solo/main.go" || cmdText != "go run solo/main.go" || note != "" {
		t.Errorf("programForm lone file = (%q, %q, %q), want file form", arg, cmdText, note)
	}

	var prompt string
	recAsk := func(p string) (string, error) { prompt = p; return "y", nil }
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program pkg/main.go", recAsk, &UserProfile{}) })
	if !handled {
		t.Fatalf("not claimed:\n%s", out)
	}
	if !strings.Contains(prompt, "go run ./pkg") {
		t.Errorf("gate prompt = %q, want the package form", prompt)
	}
	if !strings.Contains(out, "from the sibling file") {
		t.Errorf("the sibling file's code did not run:\n%s", out)
	}
}

func mustAbs(t *testing.T, dir string) string {
	t.Helper()
	abs, err := filepath.EvalSymlinks(dir)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

// A program that doesn't compile: the compiler's words,
// verbatim, reported as a compile failure — never as a run.
func TestRunProgramCompileError(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "bad/main.go", "package main\n\nfunc main() {\n\tprintln(nopeUndefined())\n}\n")
	withRunSession(t, dir)

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program bad/main.go", yes, &UserProfile{}) })
	if !handled {
		t.Fatalf("not claimed:\n%s", out)
	}
	for _, want := range []string{"It didn't run", "doesn't compile yet", "undefined: nopeUndefined"} {
		if !strings.Contains(out, want) {
			t.Errorf("compile report missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"It ran and finished", "It ran fine", "is running in the background"} {
		if strings.Contains(out, banned) {
			t.Errorf("compile failure narrated as %q:\n%s", banned, out)
		}
	}
	if sessionProgram != nil {
		t.Error("nothing may be left running after a compile failure")
	}
}

// Refusals by name: missing file, non-.go path, a library
// package, a main package with no func main.
func TestRunProgramRefusals(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "notes.txt", "hello\n")
	writeFile(t, dir, "greet.go", "package greet\n\nfunc Hi() {}\n")
	writeFile(t, dir, "nomain.go", "package main\n\nfunc NotMain() {}\n")
	withRunSession(t, dir)
	profile := &UserProfile{}

	cases := []struct{ line, want string }{
		{"run the program ghost/main.go", "no such file"},
		{"run the program notes.txt", "isn't a Go file"},
		{"run the program greet.go", "is package greet"},
		{"run the program nomain.go", "has no func main"},
		{"run the program ../outside.go", "outside this project"},
	}
	for _, tc := range cases {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(tc.line, yes, profile) })
		if !handled {
			t.Errorf("%q not claimed", tc.line)
			continue
		}
		if !strings.Contains(out, tc.want) {
			t.Errorf("%q: output missing %q:\n%s", tc.line, tc.want, out)
		}
		if sessionProgram != nil {
			t.Errorf("%q left a program behind", tc.line)
			sessionProgram = nil
		}
	}
	// "run make …" is never this claim's.
	if h := claimChain("run make eval", yes, profile); h {
		t.Error("\"run make eval\" must not be claimed by run-the-program")
	}
}

// The full server lifecycle: start, browser, double-run,
// a real visit in the log, show visits, the bare-"stop"
// fork, and a port left free.
func TestRunProgramServerLifecycle(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", analyze.WebServerSource)
	opened := withRunSession(t, dir)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
	if !handled {
		t.Fatalf("server run not claimed:\n%s", out)
	}
	for _, want := range []string{"is running in the background", "http://localhost:8080", "Opening http://localhost:8080 in your browser."} {
		if !strings.Contains(out, want) {
			t.Errorf("server start report missing %q:\n%s", want, out)
		}
	}
	if len(*opened) != 1 || (*opened)[0] != "http://localhost:8080" {
		t.Errorf("browser opened %v, want [http://localhost:8080]", *opened)
	}
	if !programRunning() {
		t.Fatal("the server should be running in the background")
	}

	// Running it again answers with its live status.
	out = captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "already running") {
		t.Errorf("double-run answer wrong:\n%s", out)
	}

	// A real visit lands in the log the chat keeps.
	resp, err := http.Get("http://localhost:8080/")
	if err != nil {
		t.Fatalf("visit failed: %v", err)
	}
	body := make([]byte, 128)
	n, _ := resp.Body.Read(body)
	resp.Body.Close()
	if !strings.Contains(string(body[:n]), "hello — your webserver is running") {
		t.Errorf("visit body = %q", string(body[:n]))
	}
	waitForLog(t, "visit: GET /")

	out = captureStdout(t, func() { handled = claimChain("show visits", yes, profile) })
	if !handled || !strings.Contains(out, "visit: GET /") || !strings.Contains(out, "It's still running.") {
		t.Errorf("show visits wrong:\n%s", out)
	}

	// While it runs, bare "stop" is the program's, not the
	// chat's — the fork both ways is pinned.
	if stopMeansQuit("stop") {
		t.Error("stopMeansQuit(stop) = true while a program runs; the chat would quit instead of stopping the program")
	}
	out = captureStdout(t, func() { handled = claimChain("stop", yes, profile) })
	if !handled || !strings.Contains(out, "Stopped srv/main.go") {
		t.Errorf("bare stop did not stop the program:\n%s", out)
	}
	if programRunning() {
		t.Error("program still running after stop")
	}
	if !stopMeansQuit("stop") {
		t.Error("stopMeansQuit(stop) = false with nothing running; bare stop must quit the chat again")
	}
	if h := runRunProgram("stop", yes, profile); h {
		t.Error("bare \"stop\" with nothing running must fall through to the loop's quit check")
	}
	if !portFree("8080") {
		t.Error("port 8080 still bound after stop")
	}
}

// The stop shapes, against a live server and against air.
func TestRunProgramStopShapes(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", analyze.WebServerSource)
	withRunSession(t, dir)
	profile := &UserProfile{}

	// Nothing running: the explicit shapes answer honestly.
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || !strings.Contains(out, "No program is running") {
		t.Errorf("stop with nothing running:\n%s", out)
	}

	start := func() {
		t.Helper()
		out := captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
		if !handled || !programRunning() {
			t.Fatalf("server did not start:\n%s", out)
		}
	}
	for _, shape := range []string{"stop the program", "stop it", "stop the server", "stop srv/main.go"} {
		start()
		out := captureStdout(t, func() { handled = claimChain(shape, yes, profile) })
		if !handled || !strings.Contains(out, "Stopped srv/main.go") || programRunning() {
			t.Errorf("%q did not stop the program:\n%s", shape, out)
		}
		if !portFree("8080") {
			t.Fatalf("port 8080 still bound after %q", shape)
		}
	}

	// A stop aimed at a different file names the right one.
	start()
	out = captureStdout(t, func() { handled = claimChain("stop other/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "isn't the program that's running") || !programRunning() {
		t.Errorf("stop of the wrong file:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// A port already in use: the program's own bind error is the
// report — never a running claim, never a browser.
func TestRunProgramPortConflict(t *testing.T) {
	ln, err := net.Listen("tcp", ":18099")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv2/main.go", `package main

import (
	"fmt"
	"net/http"
)

func main() {
	fmt.Println("listening at http://localhost:18099")
	if err := http.ListenAndServe(":18099", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`)
	opened := withRunSession(t, dir)

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program srv2/main.go", yes, &UserProfile{}) })
	if !handled {
		t.Fatalf("not claimed:\n%s", out)
	}
	if !strings.Contains(out, "address already in use") {
		t.Errorf("the bind error is not in the report:\n%s", out)
	}
	// The fixture prints the error and returns from main —
	// exit code 0, exactly like the webserver template. The
	// narration must still call it what it is: a server that
	// never served, not a clean finish.
	if !strings.Contains(out, "It stopped right away") {
		t.Errorf("the bind failure was not reported as one:\n%s", out)
	}
	for _, banned := range []string{"is running in the background", "Opening http", "It ran and finished"} {
		if strings.Contains(out, banned) {
			t.Errorf("port conflict narrated as %q:\n%s", banned, out)
		}
	}
	if len(*opened) != 0 {
		t.Errorf("browser opened for a server that never came up: %v", *opened)
	}
	if programRunning() {
		t.Error("nothing may be running after the bind failure")
	}
}

// The chat-end hook (what /quit and EOF call): the program
// dies with the chat.
func TestRunProgramChatEndCleanup(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", analyze.WebServerSource)
	withRunSession(t, dir)

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, &UserProfile{}) })
	if !handled || !programRunning() {
		t.Fatalf("server did not start:\n%s", out)
	}
	out = captureStdout(t, func() { stopProgramAtChatEnd() })
	if programRunning() {
		t.Error("the chat-end hook left the program running")
	}
	if !strings.Contains(out, "stopped srv/main.go") {
		t.Errorf("chat-end stop not announced:\n%s", out)
	}
	if !portFree("8080") {
		t.Error("port 8080 still bound after chat-end cleanup")
	}
}

// The prompt-cycle notice: new lines are announced once,
// never twice; an exit is announced once.
func TestRunProgramPromptNotice(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", analyze.WebServerSource)
	withRunSession(t, dir)

	var handled bool
	captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, &UserProfile{}) })
	if !handled || !programRunning() {
		t.Fatal("server did not start")
	}
	if _, err := http.Get("http://localhost:8080/"); err != nil {
		t.Fatal(err)
	}
	waitForLog(t, "visit: GET /")
	// The startup lines were the run report's business; the
	// visit came after. Reset the notice counter the way a
	// fresh prompt cycle would see it: only the visit is new.
	_, total := sessionProgram.log.snapshot()
	sessionProgram.log.mu.Lock()
	sessionProgram.log.notified = total - 1
	sessionProgram.log.mu.Unlock()

	out := captureStdout(t, func() { noteProgramActivity() })
	if !strings.Contains(out, "printed 1 new lines") {
		t.Errorf("notice missing the new line:\n%s", out)
	}
	out = captureStdout(t, func() { noteProgramActivity() })
	if strings.Contains(out, "new lines") {
		t.Errorf("notice repeated itself:\n%s", out)
	}
	captureStdout(t, func() { handled = claimChain("stop the program", yes, &UserProfile{}) })
	if !handled {
		t.Error("cleanup stop not handled")
	}
}

// Narration is a function of observed state, pinned at the
// source: the exit report for each state, and a build
// failure text that cannot call itself a run.
func TestRunProgramNarrationFromState(t *testing.T) {
	mk := func(lines []string, err error) *programRun {
		p := &programRun{rel: "x/main.go", log: &programLog{}, done: make(chan struct{})}
		for _, l := range lines {
			p.log.add(l)
		}
		p.exited = true
		p.exitErr = err
		return p
	}
	out := captureStdout(t, func() { reportProgramExit(mk([]string{"some output"}, nil)) })
	if !strings.Contains(out, "It ran and finished (exit code 0). The lines above are x/main.go's own output.") {
		t.Errorf("finished-with-output narration wrong:\n%s", out)
	}
	out = captureStdout(t, func() { reportProgramExit(mk(nil, nil)) })
	if !strings.Contains(out, "It ran and finished (exit code 0), and printed nothing.") {
		t.Errorf("finished-quiet narration wrong:\n%s", out)
	}
	out = captureStdout(t, func() { reportProgramExit(mk([]string{"boom"}, errors.New("exit status 3"))) })
	if !strings.Contains(out, "It stopped with an error (exit status 3)") {
		t.Errorf("failed narration wrong:\n%s", out)
	}
	fail := programBuildFailText("x/main.go", "# example.com/x\nx/main.go:2:1: undefined: nope\n")
	if strings.Contains(fail, "It ran and finished") || strings.Contains(fail, "It ran fine") {
		t.Errorf("build-failure text claims a run:\n%s", fail)
	}
	if !strings.Contains(fail, "undefined: nope") {
		t.Errorf("build-failure text lost the compiler's words:\n%s", fail)
	}
}

// The created webserver logs its visits: the template line
// is pinned, and a served file carries it.
func TestWebserverTemplateLogsVisits(t *testing.T) {
	want := `fmt.Printf("visit: %s %s\n", r.Method, r.URL.Path)`
	if !strings.Contains(analyze.WebServerMainBody, want) {
		t.Error("WebServerMainBody lost the visit log line")
	}
	if !strings.Contains(analyze.WebServerSource, want) {
		t.Error("WebServerSource lost the visit log line")
	}
}

// John's second screenshot, as a sequence: "run the program
// in jake/main.go" (with "in") runs the program; when the run
// never got going (a compile error here), "stop the program"
// is this claim's honest no-program answer — never a make
// offer, never a success narration.
func TestRunProgramInShapeThenStopAfterFailedRun(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "Makefile", "start:\n\t@echo HOST MAKE START RAN\n")
	writeFile(t, dir, "jake/main.go", "package main\n\nfunc main() {\n\tnopeUndefined()\n}\n")
	withRunSession(t, dir)
	profile := &UserProfile{Directives: map[string]bool{"do_all_means_all": true}}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program in jake/main.go", yes, profile) })
	if !handled {
		t.Fatal("\"run the program in jake/main.go\" was not claimed")
	}
	if strings.Contains(out, "HOST MAKE START RAN") || strings.Contains(out, "make start") {
		t.Errorf("the host Makefile's start target fired:\n%s", out)
	}
	if !strings.Contains(out, "doesn't compile yet") || !strings.Contains(out, "undefined: nopeUndefined") {
		t.Errorf("the compile error was not the answer:\n%s", out)
	}
	if programRunning() {
		t.Fatal("a program that failed to compile is marked running")
	}
	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || !strings.Contains(out, "No program is running") {
		t.Errorf("stop after the failed run did not answer no-program-running:\n%s", out)
	}
	if strings.Contains(out, "Run it?") || strings.Contains(out, "It ran fine") || strings.Contains(out, "HOST MAKE START RAN") {
		t.Errorf("the stop line produced a run offer or a success claim:\n%s", out)
	}
}

// Top: this chat's own programs only — empty state, a live
// server with its visit counted, and the truthful exited
// entry after a stop.
func TestRunProgramTop(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", analyze.WebServerSource)
	opened := withRunSession(t, dir)
	_ = opened
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "No programs are running right now") {
		t.Errorf("top with nothing running:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("server did not start:\n%s", out)
	}
	if _, err := http.Get("http://localhost:8080/"); err != nil {
		t.Fatal(err)
	}
	waitForLog(t, "visit: GET /")
	out = captureStdout(t, func() { handled = claimChain("what's running", yes, profile) })
	if !handled || !strings.Contains(out, "srv/main.go") || !strings.Contains(out, "running for") ||
		!strings.Contains(out, "http://localhost:8080") || !strings.Contains(out, "1 visit") ||
		!strings.Contains(out, "only the ones it started") {
		t.Errorf("top with the server running:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Fatalf("stop failed:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("show running programs", yes, profile) })
	if !handled || !strings.Contains(out, "no longer running") || strings.Contains(out, "running for") {
		t.Errorf("top after stop must list the program as exited, not live:\n%s", out)
	}
}

// Show-log with no program this session is not this claim's.
func TestRunProgramShowLogNoProgram(t *testing.T) {
	dir := runFixtureRoot(t)
	withRunSession(t, dir)
	if h := claimChain("show visits", yes, &UserProfile{}); h {
		t.Error("\"show visits\" with no program this session must fall through")
	}
}
