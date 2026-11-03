package chat

import (
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
	"time"
)

// Memory/state tests use an isolated store under a temp
// home. The store must never appear inside a fixture project.

func isolateMemoryStore(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	oldPath, oldBackend := memoryStorePathOverride, activeMemoryBackend
	oldRecall, oldShort := memoryRecall, shortTermMemory
	memoryStorePathOverride = filepath.Join(home, ".gollemer_memory.json")
	activeMemoryBackend = memoryBackendJSON
	resetMemoryRecall()
	resetShortTermMemory()
	t.Cleanup(func() {
		memoryStorePathOverride, activeMemoryBackend = oldPath, oldBackend
		memoryRecall, shortTermMemory = oldRecall, oldShort
	})
	return home
}

func memoryAsk(t *testing.T, conv *Conversation, line string) (bool, string) {
	t.Helper()
	var handled bool
	out := captureStdout(t, func() { handled = tryMemoryCommand(line, conv) })
	return handled, out
}

func assertNoMemoryStoreInTree(t *testing.T, root string) {
	t.Helper()
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && info.Name() == ".gollemer_memory.json" {
			t.Errorf("memory store leaked into the project tree: %s", path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestMemoryCaptureEchoOrdinaryLineAndRecall(t *testing.T) {
	home := isolateMemoryStore(t)
	root := talkFixtureProject(t)
	withChatSession(t, root, true)
	conv := NewConversation()

	handled, out := memoryAsk(t, conv, "remember: I prefer tabs over spaces")
	if !handled {
		t.Fatal("explicit remember line was not claimed")
	}
	if !strings.Contains(out, "gollemer [talk]> I'll remember: I prefer tabs over spaces.") {
		t.Errorf("capture was not echoed in the pinned words:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(home, ".gollemer_memory.json")); err != nil {
		t.Fatalf("store was not written under the isolated home: %v", err)
	}
	assertNoMemoryStoreInTree(t, root)

	handled, _ = memoryAsk(t, conv, "I prefer bananas while coding")
	if handled {
		t.Fatal("an ordinary line was captured — preferences may only come from explicit shapes")
	}
	d, err := loadLongTermMemory()
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Preferences) != 1 {
		t.Fatalf("stored preferences = %v, want exactly the explicit one", d.Preferences)
	}

	handled, out = memoryAsk(t, conv, "what do you remember")
	if !handled || !strings.Contains(out, "I prefer tabs over spaces") || !strings.Contains(out, canonicalMemoryRoot(root)) {
		t.Errorf("recall did not show the preference scoped to this project:\n%s", out)
	}
}

func TestMemoryCrossSessionRoundTripAndProjectIsolation(t *testing.T) {
	isolateMemoryStore(t)
	root := talkFixtureProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	conv := NewConversation()
	if handled, out := memoryAsk(t, conv, "remember that I prefer small functions"); !handled || !strings.Contains(out, "I'll remember: I prefer small functions.") {
		t.Fatalf("preference capture failed:\n%s", out)
	}
	handled, out := ctxRun(t, "add a struct named Profile to profile.go", yes, profile)
	if !handled || !strings.Contains(out, "built the project and checked") {
		t.Fatalf("the piece did not land:\n%s", out)
	}
	saved := finishMemorySession()
	if !strings.Contains(saved, "I'll remember this session:") || !strings.Contains(saved, "Profile") {
		t.Fatalf("session finish did not echo a summary naming the piece: %q", saved)
	}

	// Session B: fresh in-process context, same project and
	// same store.
	resetWorkContext()
	withChatSession(t, root, true)
	line := beginMemorySession(root)
	if !strings.HasPrefix(line, "Last time here: ") || !strings.Contains(line, "Profile") {
		t.Fatalf("continuity line = %q, want the stored summary naming Profile", line)
	}
	handled, out = memoryAsk(t, NewConversation(), "what do you remember")
	if !handled || !strings.Contains(out, "I prefer small functions") || !strings.Contains(out, "Profile") {
		t.Errorf("session B recall lacks the preference or the piece:\n%s", out)
	}
	ctx := buildTalkContext("what does Profile do?", SocialDomain, NewConversation(), false)
	if !strings.Contains(ctx.MemoryText, "I prefer small functions") || !strings.Contains(ctx.MemoryText, "Profile") {
		t.Errorf("session B talk context slice lacks the preference or summary:\n%s", ctx.MemoryText)
	}
	if !strings.Contains(ctx.StateText, "I prefer small functions") {
		t.Errorf("state block lacks recalled preferences:\n%s", ctx.StateText)
	}

	// A different project gets its own project slice: the
	// first project's summary and notes do not travel. The
	// user's global preferences do, by design.
	other := talkFixtureProject(t)
	withChatSession(t, other, true)
	if line := beginMemorySession(other); line != "" {
		t.Errorf("different project got a continuity line from the first project: %q", line)
	}
	otherCtx := buildTalkContext("hello", SocialDomain, NewConversation(), false)
	if strings.Contains(otherCtx.MemoryText, "Profile") || strings.Contains(otherCtx.MemoryText, "Last session here") {
		t.Errorf("different project context leaked the first project's slice:\n%s", otherCtx.MemoryText)
	}
	if !strings.Contains(otherCtx.MemoryText, "I prefer small functions") {
		t.Errorf("different project context lost the global preference:\n%s", otherCtx.MemoryText)
	}
}

func TestMemoryStateArms(t *testing.T) {
	isolateMemoryStore(t)
	root := talkFixtureProject(t)
	if err := os.MkdirAll(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".git", "HEAD"), []byte("ref: refs/heads/feature/memory-state\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	withChatSession(t, root, true)
	installTalkResponder(t, &localTalkResponder{})
	conv := NewConversation()

	handled, out := talkAskSeat(t, conv, "what branch am I on?", false)
	if !handled || !strings.Contains(out, "branch feature/memory-state") {
		t.Errorf("branch arm did not read .git/HEAD:\n%s", out)
	}
	handled, out = talkAskSeat(t, conv, "what operating system are we on?", false)
	if !handled || !strings.Contains(out, runtime.GOOS) {
		t.Errorf("OS arm did not answer from runtime state:\n%s", out)
	}

	rootAbs, ok := claimRootAbs(root)
	if !ok {
		t.Fatal("claim root did not settle")
	}
	noteLandedFile(rootAbs, "profile.go", "created", "struct", "Profile")
	handled, out = talkAskSeat(t, conv, "what have we made?", false)
	if !handled || !strings.Contains(out, "Profile") || !strings.Contains(out, "profile.go") {
		t.Errorf("done arm did not answer from the session summary:\n%s", out)
	}

	// Change two Go files after the analyzed read: one
	// modified, one new. The seat captures the baseline
	// before its silent re-read, so both are reported, with
	// the honest not-a-full-git-status caveat.
	if err := os.WriteFile(filepath.Join(root, "greet", "greet.go"), []byte("package greet\n\n// changed after the read\nfunc Hello(name string) string { return \"Hello, \" + name }\n\ntype Greeter struct{ Prefix string }\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeFile(t, root, "new.go", "package main\n")
	handled, out = talkAskSeat(t, conv, "what files changed?", false)
	if !handled || !strings.Contains(out, "modified: greet/greet.go") || !strings.Contains(out, "new: new.go") || !strings.Contains(out, "not a full git status") {
		t.Errorf("changed-files arm did not report the baseline subset honestly:\n%s", out)
	}

	// Time is asserted by shape, never by the wall clock.
	fixed := time.Date(2026, time.November, 3, 7, 4, 5, 0, time.FixedZone("CST", -6*60*60))
	ans, ok := talkStateAnswer(TalkContext{State: systemState{Now: fixed}}, "what time is it?")
	if !ok || !regexp.MustCompile(`^It's 7:04:05 AM CST on Tuesday, November 3, 2026\.$`).MatchString(ans) {
		t.Errorf("time arm answer = %q", ans)
	}
}

func TestMemoryForgetShapes(t *testing.T) {
	isolateMemoryStore(t)
	root := talkFixtureProject(t)
	other := talkFixtureProject(t)
	withChatSession(t, root, true)
	conv := NewConversation()
	for _, line := range []string{
		"remember: I prefer tabs over spaces",
		"remember: I prefer dark mode",
		"note: the API lives in the store package",
	} {
		if handled, out := memoryAsk(t, conv, line); !handled {
			t.Fatalf("capture %q was not claimed:\n%s", line, out)
		}
	}
	if err := addLongTermProjectNote(other, "other project note"); err != nil {
		t.Fatal(err)
	}

	handled, out := memoryAsk(t, conv, "forget my preference")
	if !handled || !strings.Contains(out, "Nothing was forgotten") {
		t.Errorf("bare preference forget should change nothing and explain:\n%s", out)
	}
	d, _ := loadLongTermMemory()
	if len(d.Preferences) != 2 {
		t.Fatalf("bare forget changed preferences: %v", d.Preferences)
	}

	handled, out = memoryAsk(t, conv, "forget my tabs over spaces")
	if !handled || !strings.Contains(out, "Forgotten: your preference \"I prefer tabs over spaces\"") {
		t.Errorf("preference forget did not confirm the exact removal:\n%s", out)
	}
	d, _ = loadLongTermMemory()
	if len(d.Preferences) != 1 || d.Preferences[0] != "I prefer dark mode" {
		t.Fatalf("wrong preferences remain: %v", d.Preferences)
	}
	if d.Projects[canonicalMemoryRoot(root)] == nil || len(d.Projects[canonicalMemoryRoot(root)].Notes) != 1 {
		t.Fatalf("preference forget touched project notes: %+v", d.Projects)
	}

	handled, out = memoryAsk(t, conv, "forget this project")
	if !handled || !strings.Contains(out, "Forgotten: this project's notes and session summaries") {
		t.Errorf("project forget did not confirm its exact scope:\n%s", out)
	}
	d, _ = loadLongTermMemory()
	if d.Projects[canonicalMemoryRoot(root)] != nil {
		t.Errorf("project slice survived forget-this-project: %+v", d.Projects[canonicalMemoryRoot(root)])
	}
	if len(d.Preferences) != 1 || d.Projects[canonicalMemoryRoot(other)] == nil {
		t.Errorf("project forget touched preferences or another project: %+v", d)
	}

	handled, out = memoryAsk(t, conv, "forget everything")
	if !handled || !strings.Contains(out, "Forgotten: everything I had stored") {
		t.Errorf("forget-everything did not confirm its scope:\n%s", out)
	}
	d, _ = loadLongTermMemory()
	if len(d.Preferences) != 0 || len(d.Projects) != 0 {
		t.Errorf("store survived forget-everything: %+v", d)
	}
}

func TestMemoryStoreLocationAndNoNetImports(t *testing.T) {
	home := isolateMemoryStore(t)
	root := talkFixtureProject(t)
	withChatSession(t, root, true)
	if handled, _ := memoryAsk(t, NewConversation(), "remember: I prefer local files"); !handled {
		t.Fatal("capture was not claimed")
	}
	storePath := memoryStorePath()
	if filepath.Dir(storePath) != home {
		t.Errorf("store path = %s, want directly under the isolated home %s", storePath, home)
	}
	if strings.HasPrefix(storePath, canonicalMemoryRoot(root)+string(filepath.Separator)) {
		t.Errorf("store path is inside the project tree: %s", storePath)
	}
	assertNoMemoryStoreInTree(t, root)

	data, err := os.ReadFile(filepath.Join(".", "memory_state.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, banned := range []string{`"net"`, `"net/http"`, `"net/url"`, `"net/`} {
		if strings.Contains(string(data), banned) {
			t.Errorf("memory_state.go contains banned network import text %s", banned)
		}
	}
	if _, err := openMemoryStoreForBackend(memoryBackendRqlite); err == nil {
		t.Error("rqlite backend unexpectedly opened — it must stay an unwired seam in this build")
	}
}
