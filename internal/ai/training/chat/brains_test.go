package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBrainRegistryAndRouting(t *testing.T) {
	for _, b := range brainRegistry {
		if b.Key == "" || b.Name == "" || b.Job == "" || b.Works == "" || b.Fails == "" {
			t.Errorf("brain %+v is missing a field — the coordinator must know each brain's job and limits", b)
		}
		if brainByKey(b.Key) == nil {
			t.Errorf("brain %s not findable by key", b.Key)
		}
	}
	// Every intent the coordinator routes must land on a registered brain.
	for _, intent := range []string{
		"summarize", "analyze", "tour", "example", "whatis", "show_func",
		"walkthrough", "arch", "why_import", "why_func", "compare",
		"glossary", "learn_path", "run_project", "where", "comment",
		"test_scaffold", "review", "uexplain", "tell_more", "what_now",
		"help", "brains", "why_brain",
	} {
		key := intentBrain(intent)
		if key == "" {
			t.Errorf("intent %s has no brain in the routing table", intent)
			continue
		}
		if brainByKey(key) == nil {
			t.Errorf("intent %s routes to unregistered brain %q", intent, key)
		}
	}
}

func TestCoordinatorRecordsAndExplains(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	stat := func(k string) (int, int) {
		brainMu.Lock()
		defer brainMu.Unlock()
		if st := brainStats[k]; st != nil {
			return st.OK, st.Fail
		}
		return 0, 0
	}
	whyOK0, _ := stat("why")
	_, stFail0 := stat("storyteller")

	recordBrain("why", true, "intent why_func")
	recordBrain("storyteller", false, "verifier rejected: invents behavior")

	brainMu.Lock()
	key, ok, note := lastBrain, lastBrainOK, lastNote
	brainMu.Unlock()
	if key != "storyteller" || ok || !strings.Contains(note, "rejected") {
		t.Fatalf("last dispatch not recorded: %s %v %q", key, ok, note)
	}
	whyOK1, _ := stat("why")
	_, stFail1 := stat("storyteller")
	if whyOK1 != whyOK0+1 || stFail1 != stFail0+1 {
		t.Fatalf("session stats did not increment: why %d->%d, storyteller-fail %d->%d",
			whyOK0, whyOK1, stFail0, stFail1)
	}

	data, err := os.ReadFile(filepath.Join(home, ".gollemer_brains.jsonl"))
	if err != nil {
		t.Fatalf("brain log not written: %v", err)
	}
	if !strings.Contains(string(data), `"brain":"storyteller"`) ||
		!strings.Contains(string(data), `"ok":false`) {
		t.Fatalf("brain log missing the failure entry:\n%s", data)
	}
	counts := brainLogCounts()
	if counts["why"] == nil || counts["why"].OK < 1 || counts["storyteller"] == nil || counts["storyteller"].Fail < 1 {
		t.Fatalf("all-time counts missing the new entries: %+v", counts)
	}
}

func TestBeginnerRawSentenceRecordsBrain(t *testing.T) {
	// Regression: loops hand handleBeginnerCommand RAW sentences; the
	// coordinator must key off the translated form or a tour records
	// nothing and "why did you do that" claims no brain has answered.
	brainMu.Lock()
	before := 0
	if st := brainStats["analyst"]; st != nil {
		before = st.OK
	}
	brainMu.Unlock()

	if !handleBeginnerCommand("give me a tour", nil) {
		t.Fatal("raw tour sentence should be handled")
	}
	brainMu.Lock()
	after := 0
	if st := brainStats["analyst"]; st != nil {
		after = st.OK
	}
	last := lastBrain
	brainMu.Unlock()
	if after != before+1 || last != "analyst" {
		t.Fatalf("tour via raw sentence not recorded: analyst %d->%d, lastBrain=%q", before, after, last)
	}
}
