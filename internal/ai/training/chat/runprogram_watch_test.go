package chat

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"
)

// The watch-mode suite (2026-11-09, John's ask: "a mode to
// restart it on updating the files or html"). These tests
// drive the real claim chain against a fixture server, then
// edit its files the way a person — or the chat itself —
// would, and check the restart from observed state: the
// watcher's notes (recorded through the watchAnnounce stub,
// the openBrowserURL pattern), the session program's
// identity and log, and real HTTP answers from old and new
// processes.

const watchSrvV1 = `package main

import (
	"fmt"
	"net/http"
)

func main() {
	fmt.Println("watch server ready: version one")
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "version one")
	})
	_ = http.ListenAndServe("127.0.0.1:18237", nil)
}
`

const watchSrvV2 = `package main

import (
	"fmt"
	"net/http"
)

func main() {
	fmt.Println("watch server ready: version two")
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "version two")
	})
	_ = http.ListenAndServe("127.0.0.1:18237", nil)
}
`

const watchSrvBroken = `package main

import (
	"fmt"
	"net/http"
)

func main() {
	fmt.Println("watch server ready: broken")
	this is not go code
}
`

// watchNotes records the watcher's out-of-band notes.
type watchNotes struct {
	mu    sync.Mutex
	lines []string
}

func (n *watchNotes) add(format string, args ...any) {
	n.mu.Lock()
	defer n.mu.Unlock()
	n.lines = append(n.lines, fmt.Sprintf(format, args...))
}

func (n *watchNotes) all() []string {
	n.mu.Lock()
	defer n.mu.Unlock()
	return append([]string(nil), n.lines...)
}

func (n *watchNotes) count(substr string) int {
	c := 0
	for _, l := range n.all() {
		if strings.Contains(l, substr) {
			c++
		}
	}
	return c
}

// waitNote polls the recorded notes until one contains
// substr, or fails after the deadline — rebuilds under load
// are slow, so the deadlines here are generous.
func (n *watchNotes) waitFor(t *testing.T, substr string) string {
	t.Helper()
	deadline := time.Now().Add(60 * time.Second)
	for time.Now().Before(deadline) {
		for _, l := range n.all() {
			if strings.Contains(l, substr) {
				return l
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("watch note %q never arrived; notes so far: %v", substr, n.all())
	return ""
}

// withWatchSession shrinks the watcher's timings, stubs its
// notes into a recorder, and registers the cleanup that
// ends the watcher and waits for its goroutine — a leaked
// watcher ticking into the next test would print into that
// test's captured stdout.
func withWatchSession(t *testing.T) *watchNotes {
	t.Helper()
	oldPoll, oldQuiet := watchPollInterval, watchQuietPeriod
	watchPollInterval, watchQuietPeriod = 25*time.Millisecond, 120*time.Millisecond
	notes := &watchNotes{}
	oldAnnounce := watchAnnounce
	watchAnnounce = notes.add
	t.Cleanup(func() {
		w := sessionWatcher
		endWatching()
		if w != nil {
			select {
			case <-w.done:
			case <-time.After(3 * time.Second):
				t.Error("watcher goroutine did not end")
			}
		}
		sessionWatcher = nil
		watchPollInterval, watchQuietPeriod = oldPoll, oldQuiet
		watchAnnounce = oldAnnounce
	})
	return notes
}

// getBody fetches the fixture server's root until it answers
// with a body containing want, or fails after the deadline.
func getBody(t *testing.T, want string) string {
	t.Helper()
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.Now().Add(30 * time.Second)
	var last string
	for time.Now().Before(deadline) {
		resp, err := client.Get("http://127.0.0.1:18237/")
		if err == nil {
			b, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			last = string(b)
			if strings.Contains(last, want) {
				return last
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatalf("server never answered %q; last body %q", want, last)
	return ""
}

// settleWatch waits past several poll cycles so the
// watcher's baseline snapshot certainly predates the next
// edit a test makes.
func settleWatch() { time.Sleep(300 * time.Millisecond) }

// The watch grammar, at parse: the run shapes with "watch"
// as the verb, the "restart … when files change" synonym,
// and the stop-watching shapes — with the declines that
// keep the neighbours' sentences theirs.
func TestParseProgramWatchShapes(t *testing.T) {
	cases := []struct {
		line   string
		shape  programShape
		target string
	}{
		{"watch jim/main.go", shapeWatchProgram, "jim/main.go"},
		{"watch the program jim/main.go", shapeWatchProgram, "jim/main.go"},
		{"watch program jim/main.go", shapeWatchProgram, "jim/main.go"},
		{"watch the program in jim/main.go", shapeWatchProgram, "jim/main.go"},
		{"can you watch jim/main.go", shapeWatchProgram, "jim/main.go"},
		{"watch jim/main.go please", shapeWatchProgram, "jim/main.go"},
		{"restart jim/main.go when files change", shapeWatchProgram, "jim/main.go"},
		{"restart the program jim/main.go when files change", shapeWatchProgram, "jim/main.go"},
		{"restart program jim/main.go when files change", shapeWatchProgram, "jim/main.go"},
		{"stop watching jim/main.go", shapeStopWatching, "jim/main.go"},
		{"stop watching the program jim/main.go", shapeStopWatching, "jim/main.go"},
		{"stop watching", shapeStopWatching, ""},
		{"can you stop watching jim/main.go", shapeStopWatching, "jim/main.go"},
		// Declines: the shapes stay tight.
		{"watch it", shapeNone, ""},
		{"watch the server", shapeNone, ""},
		{"watch", shapeNone, ""},
		{"restart jim/main.go", shapeNone, ""}, // no when-files-change tail, no claim
		{"restart the program", shapeNone, ""},
		{"stop watching tv", shapeNone, ""},
		{"stop teaching", shapeNone, ""},
		{"watch a webserver in jim/main.go", shapeNone, ""},
	}
	for _, tc := range cases {
		shape, target := parseProgramLine(tc.line)
		if shape != tc.shape || target != tc.target {
			t.Errorf("parseProgramLine(%q) = (%v, %q), want (%v, %q)", tc.line, shape, target, tc.shape, tc.target)
		}
	}
}

// Watch from cold: the claim runs the program through the
// run path, watching starts, a .go edit restarts it — the
// note names the changed file, the new process's output is
// in its log, and it answers as the new version. Top shows
// the watching.
func TestWatchStartsProgramAndRestartsOnGoEdit(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled {
		t.Fatalf("watch not claimed:\n%s", out)
	}
	for _, want := range []string{"is running in the background", "Watching srv/main.go"} {
		if !strings.Contains(out, want) {
			t.Errorf("watch start report missing %q:\n%s", want, out)
		}
	}
	if !programWatching("srv/main.go") {
		t.Fatal("the session should be watching srv/main.go")
	}
	getBody(t, "version one")
	first := sessionProgram

	out = captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "watching") {
		t.Errorf("top does not show the watching:\n%s", out)
	}

	settleWatch()
	writeFile(t, dir, "srv/main.go", watchSrvV2)
	note := notes.waitFor(t, "restarted")
	if strings.TrimSpace(note) != "[srv/main.go restarted — main.go changed]" {
		t.Errorf("restart note = %q, want the changed file named", note)
	}
	if sessionProgram == first {
		t.Error("the restart did not swap the process")
	}
	waitForLog(t, "version two")
	getBody(t, "version two")

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
	if programWatching("srv/main.go") {
		t.Error("stopping the program must end its watching")
	}
}

// Any file type restarts it — John's "or html": a new page
// in the watched subtree restarts the program, and the
// note names the page.
func TestWatchRestartsOnHTMLEdit(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("watch did not start the program:\n%s", out)
	}
	getBody(t, "version one")
	first := sessionProgram

	settleWatch()
	writeFile(t, dir, "srv/page.html", "<html><body>hi</body></html>\n")
	note := notes.waitFor(t, "restarted")
	if strings.TrimSpace(note) != "[srv/main.go restarted — page.html changed]" {
		t.Errorf("restart note = %q, want the html file named", note)
	}
	if sessionProgram == first {
		t.Error("the html edit did not restart the program")
	}
	getBody(t, "version one") // same code, new process

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// A burst of edits inside the quiet period is one restart,
// not one per file.
func TestWatchBurstCoalesces(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatal("watch did not start the program")
	}
	getBody(t, "version one")

	settleWatch()
	writeFile(t, dir, "srv/main.go", watchSrvV2)
	writeFile(t, dir, "srv/extra.css", "body {}\n")
	writeFile(t, dir, "srv/extra.js", "console.log(1)\n")
	notes.waitFor(t, "restarted")
	// Well past another quiet period: no second restart may
	// be hiding behind the first.
	time.Sleep(800 * time.Millisecond)
	if c := notes.count("restarted"); c != 1 {
		t.Errorf("the burst produced %d restarts, want 1; notes: %v", c, notes.all())
	}
	getBody(t, "version two")

	out := captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// A broken edit restarts nothing: the failure is announced,
// the OLD process keeps answering, and watching survives —
// fixing the file restarts it again.
func TestWatchBuildFailureKeepsOldProcess(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatal("watch did not start the program")
	}
	getBody(t, "version one")
	first := sessionProgram

	settleWatch()
	writeFile(t, dir, "srv/main.go", watchSrvBroken)
	note := notes.waitFor(t, "didn't restart")
	if !strings.Contains(note, "build failed") {
		t.Errorf("build-failure note = %q", note)
	}
	if sessionProgram != first {
		t.Error("a failed build swapped the process")
	}
	if !programRunning() {
		t.Error("the old process died on a failed build")
	}
	getBody(t, "version one")

	// The fix lands and the watching is still alive.
	writeFile(t, dir, "srv/main.go", watchSrvV2)
	notes.waitFor(t, "restarted")
	getBody(t, "version two")

	out := captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// "stop watching" ends the watching and only the watching:
// the program keeps running and later edits restart
// nothing. Watching a program that is already running
// turns watching on; the synonym sentence starts it
// watched from cold.
func TestWatchStopWatchingAndTurnOn(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	// Start plain, then turn watching on.
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("plain run did not start:\n%s", out)
	}
	getBody(t, "version one")
	out = captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "Watching srv/main.go") {
		t.Errorf("turning watching on failed:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "already watching") {
		t.Errorf("double watch answer wrong:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("stop watching srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "Stopped watching srv/main.go") || !strings.Contains(out, "still running") {
		t.Errorf("stop watching answer wrong:\n%s", out)
	}
	if !programRunning() {
		t.Error("stop watching must leave the program running")
	}

	settleWatch()
	writeFile(t, dir, "srv/main.go", watchSrvV2)
	time.Sleep(800 * time.Millisecond)
	if c := notes.count("restarted"); c != 0 {
		t.Errorf("an edit after stop watching restarted the program; notes: %v", notes.all())
	}
	getBody(t, "version one")

	out = captureStdout(t, func() { handled = claimChain("stop watching srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "I'm not watching any program") {
		t.Errorf("second stop watching answer wrong:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}

	// The synonym, from cold.
	out = captureStdout(t, func() { handled = claimChain("restart srv/main.go when files change", yes, profile) })
	if !handled || !strings.Contains(out, "Watching srv/main.go") || !programRunning() {
		t.Errorf("the restart-when-changed synonym did not start it watched:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// The refusals, honestly: a watch aimed at a program the
// chat isn't running while another one is, a stop-watching
// aimed at the wrong program, a non-.go watch target, and a
// watch of a file that isn't there.
func TestWatchRefusals(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "srv2/main.go", quickProgram)
	withRunSession(t, dir)
	withWatchSession(t)
	profile := &UserProfile{}

	// Nothing watched yet.
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("stop watching srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "I'm not watching any program") {
		t.Errorf("stop watching with nothing watched:\n%s", out)
	}

	captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatal("watch did not start the program")
	}

	out = captureStdout(t, func() { handled = claimChain("watch srv2/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "One at a time") {
		t.Errorf("watch of a second program was not refused:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop watching srv2/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "isn't the program I'm watching") {
		t.Errorf("stop watching the wrong program was not refused:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("watch srv/page.html", yes, profile) })
	if !handled || !strings.Contains(out, "isn't a Go file") {
		t.Errorf("watch of an html file was not refused by name:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("watch nope/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "no such file") {
		t.Errorf("watch of a missing file was not refused by name:\n%s", out)
	}
}

// A watched program that ends on its own: the watcher
// notes it once and the watching pauses — the session's
// watcher is gone afterwards. (The fixture outlives the
// startup window, then returns from main by itself.)
func TestWatchProgramExitsOnItsOwn(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", `package main

import (
	"fmt"
	"time"
)

func main() {
	fmt.Println("short lived, but past the startup window")
	time.Sleep(4 * time.Second)
	fmt.Println("done now")
}
`)
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "Watching srv/main.go") {
		t.Fatalf("watch did not start:\n%s", out)
	}
	note := notes.waitFor(t, "watching is paused")
	if !strings.Contains(note, "finished on its own") {
		t.Errorf("self-exit note = %q", note)
	}
	// The watcher winds down on its own.
	deadline := time.Now().Add(5 * time.Second)
	for sessionWatcher != nil && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if sessionWatcher != nil {
		t.Error("the watcher outlived the program it watched")
	}
}
