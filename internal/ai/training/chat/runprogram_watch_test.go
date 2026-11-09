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
	oldPending := pendingWatchFolders
	pendingWatchFolders = nil
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
		pendingWatchFolders = oldPending
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

// The folder grammar, at parse: "watch folder <path>" and
// its "the folder" spelling, the stop-watching-folder
// shapes, the softeners — and the declines.
func TestParseProgramWatchFolderShapes(t *testing.T) {
	cases := []struct {
		line   string
		shape  programShape
		target string
	}{
		{"watch folder jim/temp", shapeWatchFolder, "jim/temp"},
		{"watch the folder jim/temp", shapeWatchFolder, "jim/temp"},
		{"watch folder assets", shapeWatchFolder, "assets"},
		{"can you watch folder jim/temp", shapeWatchFolder, "jim/temp"},
		{"watch folder jim/temp please", shapeWatchFolder, "jim/temp"},
		{"stop watching folder jim/temp", shapeStopWatchingFolder, "jim/temp"},
		{"stop watching the folder jim/temp", shapeStopWatchingFolder, "jim/temp"},
		{"can you stop watching folder assets", shapeStopWatchingFolder, "assets"},
		// Declines: the shapes stay tight.
		{"watch folder", shapeNone, ""},
		{"watch the folder", shapeNone, ""},
		{"watch folders jim/temp", shapeNone, ""},
		{"stop watching folder", shapeNone, ""},
		{"stop watching folders jim/temp", shapeNone, ""},
	}
	for _, tc := range cases {
		shape, target := parseProgramLine(tc.line)
		if shape != tc.shape || target != tc.target {
			t.Errorf("parseProgramLine(%q) = (%v, %q), want (%v, %q)", tc.line, shape, target, tc.shape, tc.target)
		}
	}
}

// A folder outside the program's subtree: "watch folder"
// on a running-but-unwatched program turns watching on
// with the folder included, top lists it, and an html edit
// — then a js edit — under it restarts the program, each
// note naming the file project-relatively.
func TestWatchFolderOutsideSubtreeRestarts(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "assets/page.html", "<html><body>one</body></html>\n")
	writeFile(t, dir, "assets/app.js", "console.log(1)\n")
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("plain run did not start:\n%s", out)
	}
	getBody(t, "version one")

	out = captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "Watching srv/main.go") || !strings.Contains(out, "assets") {
		t.Errorf("watch folder did not turn watching on with the folder:\n%s", out)
	}
	if !programWatching("srv/main.go") {
		t.Fatal("watch folder should leave the session watching srv/main.go")
	}
	out = captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "watching for changes (also assets)") {
		t.Errorf("top does not list the folder:\n%s", out)
	}

	settleWatch()
	writeFile(t, dir, "assets/page.html", "<html><body>two</body></html>\n")
	note := notes.waitFor(t, "restarted")
	if strings.TrimSpace(note) != "[srv/main.go restarted — assets/page.html changed]" {
		t.Errorf("restart note = %q, want the file named project-relatively", note)
	}
	getBody(t, "version one") // same code, new process

	settleWatch()
	writeFile(t, dir, "assets/app.js", "console.log(2)\n")
	note = notes.waitFor(t, "assets/app.js changed")
	if strings.TrimSpace(note) != "[srv/main.go restarted — assets/app.js changed]" {
		t.Errorf("restart note = %q, want the js file named project-relatively", note)
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
	if programWatching("srv/main.go") {
		t.Error("stopping the program must end its watching, folders included")
	}
}

// Naming the program's own folder, or a folder inside a
// watched root, is a plain already-watching answer and no
// state change: the extra set stays empty.
func TestWatchFolderAlreadyCovered(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "srv/templates/index.html", "<html></html>\n")
	writeFile(t, dir, "assets/app.js", "console.log(1)\n")
	withRunSession(t, dir)
	withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("watch did not start the program:\n%s", out)
	}
	for _, line := range []string{"watch folder srv", "watch folder srv/templates"} {
		out = captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled || !strings.Contains(out, "already watching") {
			t.Errorf("%q was not the already-watching answer:\n%s", line, out)
		}
		if got := watchedExtraFolderNames("srv/main.go"); len(got) != 0 {
			t.Errorf("%q added extra roots %v, want none", line, got)
		}
	}

	// Added once, answered as already there the second time.
	out = captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "Now watching the folder assets too") {
		t.Errorf("adding the folder failed:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "already watching") {
		t.Errorf("naming the same folder twice was not already-watching:\n%s", out)
	}
	if got := watchedExtraFolderNames("srv/main.go"); len(got) != 1 || got[0] != "assets" {
		t.Errorf("extra roots = %v, want just [assets]", got)
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// "stop watching folder" removes just that root: later
// edits there restart nothing, while edits in the
// program's own folder still do.
func TestWatchFolderStopRemovesOnlyIt(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "assets/page.html", "<html><body>one</body></html>\n")
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("watch did not start the program:\n%s", out)
	}
	getBody(t, "version one")
	out = captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "Now watching the folder assets too") {
		t.Fatalf("adding the folder failed:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop watching folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "Stopped watching the folder assets") {
		t.Errorf("stop watching folder answer wrong:\n%s", out)
	}
	if !programWatching("srv/main.go") {
		t.Error("removing a folder must leave the program watched")
	}

	settleWatch()
	writeFile(t, dir, "assets/page.html", "<html><body>two</body></html>\n")
	time.Sleep(800 * time.Millisecond)
	if c := notes.count("restarted"); c != 0 {
		t.Errorf("an edit in the removed folder restarted the program; notes: %v", notes.all())
	}
	getBody(t, "version one")

	// The base root is still watched.
	writeFile(t, dir, "srv/main.go", watchSrvV2)
	notes.waitFor(t, "restarted")
	getBody(t, "version two")

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// The folder refusals, honestly: a declaration with no
// program is noted as pending rather than refused (the
// declaration itself is the refusal-free part — the
// invalid folders below still refuse by name), a folder
// that isn't there, a file named as a folder, a path out
// of the project, and a stop-watching-folder aimed at the
// program's own folder.
func TestWatchFolderRefusals(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "assets/app.js", "console.log(1)\n")
	withRunSession(t, dir)
	withWatchSession(t)
	profile := &UserProfile{}

	// No program yet: the declaration pends, and stopping
	// it again takes the note back off the list.
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "No program is running") {
		t.Errorf("watch folder with no program was not answered:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop watching folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "only noted") {
		t.Errorf("stop watching a pending folder:\n%s", out)
	}
	if len(pendingWatchFolders) != 0 {
		t.Errorf("pending folders = %v, want none after the removal", pendingWatchFolderNames())
	}

	captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatal("watch did not start the program")
	}

	out = captureStdout(t, func() { handled = claimChain("watch folder nope", yes, profile) })
	if !handled || !strings.Contains(out, "no folder by that name") {
		t.Errorf("watch of a missing folder was not refused by name:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("watch folder srv/main.go", yes, profile) })
	if !handled || !strings.Contains(out, "is a file, not a folder") {
		t.Errorf("watch of a file as a folder was not refused by name:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("watch folder ../outside", yes, profile) })
	if !handled || !strings.Contains(out, "outside this project") {
		t.Errorf("watch of a folder outside the project was not refused:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop watching folder srv", yes, profile) })
	if !handled || !strings.Contains(out, "comes with watching the program") {
		t.Errorf("stop watching the program's own folder was not explained:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("stop watching folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "I'm not watching the folder assets") {
		t.Errorf("stop watching an unwatched folder was not answered honestly:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
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

// John's ordering (2026-11-09, live on the pushed build):
// in a fresh session he named two folders BEFORE running
// anything and got the no-program refusal twice. Declaring
// is intent, in order — valid folders are noted as
// pending, duplicates and folders inside a noted one are
// already-noted answers, invalid ones still refuse by
// name at declare time, and top shows the waiting list.
func TestWatchFolderPendingDeclare(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "srv/templates/index.html", "<html></html>\n")
	writeFile(t, dir, "assets/app.js", "console.log(1)\n")
	writeFile(t, dir, "assets/deep/x.txt", "x\n")
	withRunSession(t, dir)
	withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "noted") || !strings.Contains(out, "watch <file>") {
		t.Errorf("declare with no program was not noted as pending:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("watch folder srv/templates", yes, profile) })
	if !handled || !strings.Contains(out, "noted") {
		t.Errorf("second declare was not noted:\n%s", out)
	}
	if got := pendingWatchFolderNames(); len(got) != 2 || got[0] != "assets" || got[1] != "srv/templates" {
		t.Fatalf("pending folders = %v, want [assets srv/templates]", got)
	}

	out = captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "already noted") {
		t.Errorf("duplicate declare was not already-noted:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("watch folder assets/deep", yes, profile) })
	if !handled || !strings.Contains(out, "already noted") {
		t.Errorf("declare inside a noted folder was not already-noted:\n%s", out)
	}
	if len(pendingWatchFolders) != 2 {
		t.Errorf("pending folders changed on already-noted answers: %v", pendingWatchFolderNames())
	}

	for _, tc := range []struct{ line, want string }{
		{"watch folder jim/temp", "no folder by that name"}, // John's words; absent in this fixture
		{"watch folder nope", "no folder by that name"},
		{"watch folder assets/app.js", "is a file, not a folder"},
		{"watch folder ../outside", "outside this project"},
	} {
		out = captureStdout(t, func() { handled = claimChain(tc.line, yes, profile) })
		if !handled || !strings.Contains(out, tc.want) {
			t.Errorf("%q was not refused by name (%q):\n%s", tc.line, tc.want, out)
		}
	}
	if len(pendingWatchFolders) != 2 {
		t.Errorf("a refusal pended a folder: %v", pendingWatchFolderNames())
	}

	out = captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "Watch folders waiting for a program: assets, srv/templates") {
		t.Errorf("top does not show the waiting folders:\n%s", out)
	}
}

// Watch start answers the pending list: a folder outside
// the program's subtree attaches (and an edit there
// restarts, named in full), while one inside the
// program's own folder drops with a spoken note — John's
// jim/temp case. The pending list ends empty.
func TestWatchFolderPendingAttachAtWatchStart(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "srv/templates/index.html", "<html></html>\n")
	writeFile(t, dir, "assets/page.html", "<html><body>one</body></html>\n")
	withRunSession(t, dir)
	notes := withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	for _, line := range []string{"watch folder assets", "watch folder srv/templates"} {
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled || !strings.Contains(out, "noted") {
			t.Fatalf("%q was not noted as pending:\n%s", line, out)
		}
	}

	out := captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("watch did not start the program:\n%s", out)
	}
	if !strings.Contains(out, "srv/templates") || !strings.Contains(out, "already covers it") {
		t.Errorf("the inside-the-base pending folder did not drop with a note:\n%s", out)
	}
	if !strings.Contains(out, "including the folder assets") {
		t.Errorf("the watch start did not name the attached folder:\n%s", out)
	}
	if len(pendingWatchFolders) != 0 {
		t.Errorf("pending folders survived the watch start: %v", pendingWatchFolderNames())
	}
	if got := watchedExtraFolderNames("srv/main.go"); len(got) != 1 || got[0] != "assets" {
		t.Fatalf("extra roots = %v, want just [assets]", got)
	}
	out = captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "watching for changes (also assets)") || strings.Contains(out, "waiting for a program") {
		t.Errorf("top after attach wrong:\n%s", out)
	}
	getBody(t, "version one")

	settleWatch()
	writeFile(t, dir, "assets/page.html", "<html><body>two</body></html>\n")
	note := notes.waitFor(t, "restarted")
	if strings.TrimSpace(note) != "[srv/main.go restarted — assets/page.html changed]" {
		t.Errorf("restart note = %q, want the attached folder's file named project-relatively", note)
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// "stop watching folder" removes a PENDING folder: the
// answer says it was only noted, and a later watch start
// does not attach it — while its noted neighbour still
// attaches.
func TestWatchFolderPendingStopRemoves(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "assets/page.html", "<html></html>\n")
	writeFile(t, dir, "assets2/page.html", "<html></html>\n")
	withRunSession(t, dir)
	withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	for _, line := range []string{"watch folder assets", "watch folder assets2"} {
		captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Fatalf("%q was not claimed", line)
		}
	}
	out := captureStdout(t, func() { handled = claimChain("stop watching folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "Stopped watching the folder assets") || !strings.Contains(out, "only noted") {
		t.Errorf("stopping a pending folder:\n%s", out)
	}
	if got := pendingWatchFolderNames(); len(got) != 1 || got[0] != "assets2" {
		t.Fatalf("pending folders = %v, want just [assets2]", got)
	}
	out = captureStdout(t, func() { handled = claimChain("stop watching folder assets", yes, profile) })
	if !handled || !strings.Contains(out, "not one of the folders waiting") {
		t.Errorf("stopping an unnoted folder was not answered with the waiting list:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("watch srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("watch did not start the program:\n%s", out)
	}
	if strings.Contains(out, "including the folder assets,") || strings.Contains(out, "including the folder assets.") {
		t.Errorf("the removed folder attached anyway:\n%s", out)
	}
	if got := watchedExtraFolderNames("srv/main.go"); len(got) != 1 || got[0] != "assets2" {
		t.Errorf("extra roots = %v, want just [assets2]", got)
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("cleanup stop failed:\n%s", out)
	}
}

// A plain run does not consume pending folders — they
// wait for a WATCH start, show as waiting on top while
// the program runs unwatched, survive the program's
// stop, and bare "stop watching" clears them with a
// spoken line.
func TestWatchFolderPendingSurvivesPlainRun(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", watchSrvV1)
	writeFile(t, dir, "assets/page.html", "<html></html>\n")
	withRunSession(t, dir)
	withWatchSession(t)
	profile := &UserProfile{}

	var handled bool
	captureStdout(t, func() { handled = claimChain("watch folder assets", yes, profile) })
	if !handled || len(pendingWatchFolders) != 1 {
		t.Fatal("the declaration did not pend")
	}

	out := captureStdout(t, func() { handled = claimChain("run the program srv/main.go", yes, profile) })
	if !handled || !programRunning() {
		t.Fatalf("plain run did not start:\n%s", out)
	}
	if programWatching("srv/main.go") {
		t.Fatal("a plain run must not turn watching on")
	}
	if len(pendingWatchFolders) != 1 {
		t.Errorf("a plain run consumed the pending folders: %v", pendingWatchFolderNames())
	}
	out = captureStdout(t, func() { handled = claimChain("top", yes, profile) })
	if !handled || !strings.Contains(out, "Watch folders waiting for a program: assets") {
		t.Errorf("top does not show the folder still waiting under a running program:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || programRunning() {
		t.Errorf("stop failed:\n%s", out)
	}
	if len(pendingWatchFolders) != 1 {
		t.Errorf("stopping the program dropped folders that never attached: %v", pendingWatchFolderNames())
	}

	out = captureStdout(t, func() { handled = claimChain("stop watching", yes, profile) })
	if !handled || !strings.Contains(out, "cleared the folders that were waiting for a program: assets") {
		t.Errorf("bare stop watching did not clear the pending folders:\n%s", out)
	}
	if len(pendingWatchFolders) != 0 {
		t.Errorf("pending folders survived the clear: %v", pendingWatchFolderNames())
	}
}
