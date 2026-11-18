package chat

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The talk seat's unit pins: the responder seam (absent =
// never fires, the speccopy rule), the talk tag, the
// claim-checker's marking wording on a FORCED hallucination,
// and the deterministic arms (project facts, traps, "it")
// answering from the fixture index alone — no social model
// needed, so these run offline and fast.

// talkFixtureProject builds the probe fixture: a module whose
// main calls greet.Hello and store.Save, a greet package with
// a function and a struct, and a store package with two
// functions. Every fact the seat may state about it is known.
func talkFixtureProject(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/talkprobe\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", `package main

import (
	"example.com/talkprobe/greet"
	"example.com/talkprobe/store"
)

func main() {
	msg := greet.Hello("world")
	_ = store.Save("data.txt", []byte(msg))
}
`)
	writeFile(t, dir, "greet/greet.go", `package greet

// Hello returns a greeting for the named person.
func Hello(name string) string {
	return "Hello, " + name
}

// Greeter greets people with a prefix.
type Greeter struct {
	Prefix string
}
`)
	writeFile(t, dir, "store/store.go", `package store

// Save writes data to the file at path.
func Save(path string, data []byte) error {
	return nil
}

// Load reads the file at path.
func Load(path string) ([]byte, error) {
	return nil, nil
}
`)
	return dir
}

// installTalkResponder swaps the talk responder seam the way
// the proposer tests swap theirs: a fresh Once, already fired,
// so the lazy loader never runs and the given responder (nil
// for the unavailable state) is the one the seat consults.
func installTalkResponder(t *testing.T, resp TalkResponder) {
	t.Helper()
	oldResp, oldOnce := talkResponderVar, talkOnce
	t.Cleanup(func() { talkResponderVar, talkOnce = oldResp, oldOnce })
	talkResponderVar = resp
	talkOnce = &sync.Once{}
	talkOnce.Do(func() {})
}

// fakeTalkResponder is a scripted TalkResponder for seat
// plumbing tests.
type fakeTalkResponder struct {
	resp TalkResponse
	ok   bool
}

func (f fakeTalkResponder) Respond(ctx TalkContext, line string) (TalkResponse, bool) {
	return f.resp, f.ok
}

// talkAskSeat drives one line through the seat exactly as the
// unified loop's tail does: the user turn is recorded first,
// then the seat is consulted.
func talkAskSeat(t *testing.T, conv *Conversation, line string, socialRoute bool) (bool, string) {
	t.Helper()
	conv.AddUser(line)
	var handled bool
	out := captureStdout(t, func() {
		handled = tryTalkSeat(line, SocialDomain, conv, socialRoute, false)
	})
	return handled, out
}

func TestTalkSeatNilResponderNeverFires(t *testing.T) {
	dir := talkFixtureProject(t)
	withChatSession(t, dir, true)
	installTalkResponder(t, nil)
	handled, out := talkAskSeat(t, NewConversation(), "what does Save take?", false)
	if handled || out != "" {
		t.Errorf("nil responder: handled=%v out=%q — the seat must never fire and print nothing", handled, out)
	}
}

func TestTalkSeatFakeResponderTagAndRecord(t *testing.T) {
	dir := talkFixtureProject(t)
	withChatSession(t, dir, true)
	installTalkResponder(t, fakeTalkResponder{resp: TalkResponse{Text: "hello there, friend", Factual: true}, ok: true})
	conv := NewConversation()
	handled, out := talkAskSeat(t, conv, "hello", true)
	if !handled {
		t.Fatalf("fake responder declined through the seat:\n%s", out)
	}
	if !strings.Contains(out, "gollemer [talk]>") || !strings.Contains(out, "hello there, friend") {
		t.Errorf("seat reply lacks the talk tag or the text:\n%s", out)
	}
	turns := conv.Turns()
	if len(turns) == 0 || turns[len(turns)-1].Domain != TalkDomain {
		t.Errorf("seat reply not recorded under the talk domain: %+v", turns)
	}
}

// The forced hallucination: a fabricated reply naming a
// project function that does not exist must never print as
// bare fact — its sentence carries the marker, and a real
// name's sentence stays clean.
func TestTalkCheckerMarksInventedFunction(t *testing.T) {
	dir := talkFixtureProject(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	before := "Try calling Frobnicate() — it sorts the cache for you. Save(path string, data []byte) error is the real save function here."
	after := talkCheckReply(before, p)
	t.Logf("before: %s", before)
	t.Logf("after:  %s", after)
	wantMarked := "Try calling Frobnicate() — it sorts the cache for you (I don't see that in your project)."
	if !strings.Contains(after, wantMarked) {
		t.Errorf("invented-name sentence not marked as pinned:\n got: %s\nwant substring: %s", after, wantMarked)
	}
	if n := strings.Count(after, talkInventedMarker); n != 1 {
		t.Errorf("marker count = %d, want exactly 1 (the real Save sentence stays clean):\n%s", n, after)
	}
	if bad := talkUnmarkedInventions(after, p); len(bad) != 0 {
		t.Errorf("unmarked inventions remain after checking: %v", bad)
	}
}

// A bare fragment sentence that is little more than the
// invented reference itself is suppressed, not marked.
func TestTalkCheckerSuppressesBareFragment(t *testing.T) {
	dir := talkFixtureProject(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	after := talkCheckReply("Use Frobnicate(). Hello(name string) string is real and stays.", p)
	if strings.Contains(after, "Frobnicate") {
		t.Errorf("bare invented fragment was not suppressed:\n%s", after)
	}
	if !strings.Contains(after, "Hello(name string) string is real and stays.") {
		t.Errorf("clean sentence was damaged:\n%s", after)
	}
}

// Paths and dotted project references are checked too;
// stdlib dotted names are general knowledge and pass.
func TestTalkCheckerPathsAndDotted(t *testing.T) {
	dir := talkFixtureProject(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	before := "Look in cache/magic.go and call store.Zoom() for the details today. greet.Hello() is the greeter entry point, and fmt.Println is standard."
	after := talkCheckReply(before, p)
	if n := strings.Count(after, talkInventedMarker); n != 1 {
		t.Errorf("marker count = %d, want 1 (one bad sentence):\n%s", n, after)
	}
	if !strings.Contains(after, "greet.Hello() is the greeter entry point, and fmt.Println is standard.") {
		t.Errorf("clean sentence was marked or damaged:\n%s", after)
	}
	if bad := talkUnmarkedInventions(after, p); len(bad) != 0 {
		t.Errorf("unmarked inventions remain: %v in\n%s", bad, after)
	}
}

// The deterministic arms through the real seat, with a
// social-less local responder: project facts, the honest
// trap, and the file/package answers.
func TestTalkSeatProjectFacts(t *testing.T) {
	dir := talkFixtureProject(t)
	withChatSession(t, dir, true)
	installTalkResponder(t, &localTalkResponder{})
	conv := NewConversation()
	cases := []struct {
		line string
		want []string
	}{
		{"what does Save take?", []string{"gollemer [talk]>", "path string", "data []byte"}},
		{"what does Load return?", []string{"[]byte", "error"}},
		{"what does Frobnicate do?", []string{"Frobnicate", "I don't see anything called Frobnicate in this project."}},
		{"what's in greet/greet.go?", []string{"greet/greet.go", "Hello", "Greeter"}},
		{"what functions are in the store package?", []string{"Save", "Load"}},
		{"what's in the store package?", []string{"Save", "Load"}},
		{"does this project have a function called Teleport?", []string{"I don't see a function called Teleport in this project."}},
		{"what type is Greeter?", []string{"Greeter", "struct"}},
		{"how does the Blorp handler work?", []string{"Blorp", "I don't see anything called Blorp in this project."}},
		{"what's in cache/magic.go?", []string{"I don't see a file called cache/magic.go in this project."}},
	}
	for _, tc := range cases {
		handled, out := talkAskSeat(t, conv, tc.line, false)
		if !handled {
			t.Errorf("%q: seat declined", tc.line)
			continue
		}
		for _, w := range tc.want {
			if !strings.Contains(out, w) {
				t.Errorf("%q: reply lacks %q:\n%s", tc.line, w, out)
			}
		}
	}
	// A general-Go concept is NOT a missing project name: the
	// knowledge base answers it, unmarked and honest.
	handled, out := talkAskSeat(t, conv, "explain it to me simply: what is a channel?", false)
	if !handled || !strings.Contains(out, "Channels move values between goroutines") {
		t.Errorf("general-Go line not answered by the knowledge base arm:\n%s", out)
	}
}

// "It" resolves from the session working context, and the
// answer names the reading.
func TestTalkSeatItResolution(t *testing.T) {
	dir := talkFixtureProject(t)
	withChatSession(t, dir, true)
	installTalkResponder(t, &localTalkResponder{})
	rootAbs, ok := claimRootAbs(dir)
	if !ok {
		t.Fatal("claim root did not settle")
	}
	noteLandedFile(rootAbs, "greet/greet.go", "created", "function", "Hello")
	conv := NewConversation()
	handled, out := talkAskSeat(t, conv, "what does it do?", false)
	if !handled {
		t.Fatalf("it-question declined:\n%s", out)
	}
	if !strings.Contains(out, "Hello") || !strings.Contains(out, "greet/greet.go") {
		t.Errorf("it-answer does not name the reading (Hello in greet/greet.go):\n%s", out)
	}
	handled, out = talkAskSeat(t, conv, "is it in the store package?", false)
	if !handled || !strings.Contains(out, "No") || !strings.Contains(out, "greet") {
		t.Errorf("it-package answer wrong:\n%s", out)
	}
	// No context at all — no work in progress and a fresh
	// conversation that never mentioned a subject: the seat
	// declines and today's fallback owns the line.
	workCtx = workContext{}
	handled, _ = talkAskSeat(t, NewConversation(), "what does it do?", false)
	if handled {
		t.Errorf("context-less it-question was claimed — silent guessing is the cardinal sin")
	}
}

// The new talk files must add no network imports — the
// zero-network constraint, pinned in the suite itself.
func TestTalkSeatFilesHaveNoNetImports(t *testing.T) {
	for _, rel := range []string{"talkseat.go", "talkseat_check.go"} {
		data, err := os.ReadFile(filepath.Join(".", rel))
		if err != nil {
			t.Fatal(err)
		}
		for _, banned := range []string{`"net"`, `"net/http"`, `"net/url"`, `"net/`} {
			if strings.Contains(string(data), banned) {
				t.Errorf("%s contains banned network import text %s", rel, banned)
			}
		}
	}
}
