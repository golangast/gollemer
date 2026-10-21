package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFriendRepeatTopicTrigger(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	friendResetForTest()

	friendNoteFunc("Freeze")
	friendNoteFunc("Freeze")
	if friendPending != "" {
		t.Fatalf("Friend spoke too early (2 asks): %q", friendPending)
	}
	friendNoteFunc("Freeze")
	if !strings.Contains(friendPending, "Freeze") {
		t.Fatalf("after 3 asks the repeat-topic suggestion should be queued, got %q", friendPending)
	}
	// Same suggestion must not re-queue after it fires.
	key := friendPendingKey
	friendFiredSession[key] = true
	friendPending, friendPendingKey = "", ""
	friendNoteFunc("Freeze")
	if friendPending != "" {
		t.Fatalf("repeat-topic suggestion fired twice in one session: %q", friendPending)
	}
}

func TestFriendCooldownAndStuckTrigger(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	friendResetForTest()

	friendObserveBrain("comparer", false)
	friendObserveBrain("comparer", false)
	if !strings.Contains(friendPending, "Comparer") {
		t.Fatalf("two straight failures should queue a stuck note, got %q", friendPending)
	}
	// Cooldown: turn 0, last spoke at -1000 — the FIRST-ever suggestion
	// may speak immediately; after that, 10 turns must pass.
	friendMaybeSpeak()
	if friendPending != "" || friendLastSpokeTurn != 0 {
		t.Fatalf("first suggestion should have spoken: pending=%q spoke=%d", friendPending, friendLastSpokeTurn)
	}
	friendSetPending("topic:X", "queued")
	friendTurn = 5
	friendMaybeSpeak()
	if friendPending == "" {
		t.Fatal("cooldown violated: spoke again after only 5 turns")
	}
	friendTurn = 10
	friendMaybeSpeak()
	if friendPending != "" {
		t.Fatal("suggestion should speak once the cooldown passes")
	}
}

func TestFriendMemoryPersists(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	friendResetForTest()

	friendNoteFunc("Names")
	friendNoteFunc("Names")
	data, err := os.ReadFile(filepath.Join(home, ".gollemer_friend.json"))
	if err != nil {
		t.Fatalf("friend memory not written: %v", err)
	}
	if !strings.Contains(string(data), `"Names": 2`) {
		t.Fatalf("friend memory missing the ask counts:\n%s", data)
	}
	friendResetForTest() // simulate a new process
	if got := friendLoad().FuncAsks["Names"]; got != 2 {
		t.Fatalf("friend forgot across sessions: Names=%d", got)
	}
	if friendLoad().LastTopic != "Names" {
		t.Fatalf("friend forgot the last topic: %q", friendLoad().LastTopic)
	}
}

func TestFriendWatchDigest(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	friendResetForTest()

	entry := `{"t":"2020-01-01T10:00:00Z","kind":"sig-changed","func":"Counter.Freeze","detail":"signature changed — 3 caller(s) use it","root":"/x"}` + "\n" +
		`{"t":"2020-01-01T10:01:00Z","kind":"untested-new","func":"Store","detail":"new exported function — no test exercises it yet","root":"/x"}` + "\n"
	if err := os.WriteFile(filepath.Join(home, ".gollemer_watch.jsonl"), []byte(entry), 0600); err != nil {
		t.Fatal(err)
	}
	notes := friendWatchNotes()
	if len(notes) != 2 {
		t.Fatalf("expected 2 watcher notes, got %v", notes)
	}
	friendBeginSession()
	if !strings.Contains(friendPending, "Watcher noticed") || !strings.Contains(friendPending, "Counter.Freeze") {
		t.Fatalf("session should open with the watcher digest queued, got %q", friendPending)
	}
	if friendLoad().WatchRead == "" {
		t.Fatal("watch cursor must advance after the digest")
	}
	if again := friendWatchNotes(); len(again) != 0 {
		t.Fatalf("digested events must not repeat: %v", again)
	}
}
