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
