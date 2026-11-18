package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// usage_test.go — the intent layer's outcome capture (2026-10-24):
// every classified line lands in the brain log with its raw text,
// the chosen intent, the confidence, and how it ended up. These
// entries are what scripts/mine_usage turns into training pairs —
// but only the confirmed outcomes (dispatched, clarified-yes) may
// ever be mined; these tests pin the recording itself.

// outcomeEntries reads a brain log (in a temp HOME) and returns
// just the intent-outcome entries.
func outcomeEntries(t *testing.T, home string) []brainLogEntry {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(home, ".gollemer_brains.jsonl"))
	if err != nil {
		t.Fatalf("read brain log: %v", err)
	}
	var out []brainLogEntry
	for _, l := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(l) == "" {
			continue
		}
		var e brainLogEntry
		if err := json.Unmarshal([]byte(l), &e); err != nil {
			t.Fatalf("parse brain log line %q: %v", l, err)
		}
		if e.Outcome != "" {
			out = append(out, e)
		}
	}
	return out
}

func TestRecordIntentOutcomeEntry(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	recordBrain("why", true, "intent why_func")
	recordIntentOutcome("wheres the logi handled", "find", 0.87, IntentOutcomeDispatched)

	// The extended entry still parses for the log's older
	// consumers, under the coordinator's name.
	counts := brainLogCounts()
	if counts["why"] == nil || counts["why"].OK != 1 {
		t.Errorf("brainLogCounts lost the plain entry: %+v", counts)
	}
	if counts["coordinator"] == nil || counts["coordinator"].OK != 1 {
		t.Errorf("brainLogCounts lost the outcome entry: %+v", counts)
	}
	// And it must not steal "why did you do that" from the brain
	// that actually answered.
	brainMu.Lock()
	last := lastBrain
	brainMu.Unlock()
	if last != "why" {
		t.Errorf("recordIntentOutcome clobbered lastBrain: %q", last)
	}

	entries := outcomeEntries(t, home)
	if len(entries) != 1 {
		t.Fatalf("outcome entries = %d, want 1", len(entries))
	}
	e := entries[0]
	if e.Line != "wheres the logi handled" || e.Intent != "find" ||
		e.Outcome != IntentOutcomeDispatched || e.Conf < 0.86 || e.Conf > 0.88 {
		t.Errorf("outcome entry wrong: %+v", e)
	}
}

func TestIntentDispatchRecordsOutcomes(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	withRealIntentModel(t)
	withEditsProject(t, intentV2Fixture(t))

	var handled bool
	// A confident dispatch is recorded with the raw line.
	captureStdout(t, func() { handled = tryIntentDispatch("got anything on retries", nil) })
	if !handled {
		t.Fatal("find phrasing not dispatched")
	}
	// A confident brain-owned line falls through — and the record
	// says fell-through, not dispatched: nothing confirmed it as
	// an intent-layer success.
	captureStdout(t, func() { handled = tryIntentDispatch("what is a goroutine", nil) })
	if handled {
		t.Fatal("brain-owned line should fall through")
	}
	// A line under the clarification floor gets no offer and is
	// recorded as an unconfirmed fall-through.
	captureStdout(t, func() { handled = tryIntentDispatch("show the thing", nil) })
	if handled {
		t.Fatal("below-floor line should fall through unhandled")
	}
	// A clarification answered "1": the ORIGINAL line is recorded
	// with the user-confirmed intent. The pending question is
	// constructed directly (as in TestClarifySingleCandidateYes) so
	// the recording is tested deterministically, not via one model
	// instance's borderline tie.
	pendingClarify = &pendingClarification{line: "the words and pieces", cands: []string{"learn_pieces", "glossary"}, conf: 0.49}
	captureStdout(t, func() { handled = tryIntentDispatch("1", nil) })
	if !handled {
		t.Fatal("clarification reply 1 not handled")
	}
	// A clarification declined: recorded as a negative, with the
	// rejected candidate named for the record (never mined).
	pendingClarify = &pendingClarification{line: "the words and pieces", cands: []string{"learn_pieces", "glossary"}, conf: 0.49}
	captureStdout(t, func() { handled = tryIntentDispatch("neither", nil) })
	if handled {
		t.Fatal("'neither' should fall through unhandled")
	}

	entries := outcomeEntries(t, home)
	if len(entries) != 5 {
		t.Fatalf("outcome entries = %d, want 5: %+v", len(entries), entries)
	}
	byLineOutcome := map[string]brainLogEntry{}
	for _, e := range entries {
		byLineOutcome[e.Line+"|"+e.Outcome] = e
	}
	check := func(line, outcome, intent string) {
		e, ok := byLineOutcome[line+"|"+outcome]
		if !ok {
			t.Errorf("no %s entry for %q in %+v", outcome, line, entries)
			return
		}
		if intent != "" && e.Intent != intent {
			t.Errorf("entry {%q %s} intent = %s, want %s", line, outcome, e.Intent, intent)
		}
	}
	check("got anything on retries", IntentOutcomeDispatched, "find")
	check("what is a goroutine", IntentOutcomeFellThrough, "concept_q")
	check("show the thing", IntentOutcomeFellThrough, "")
	check("the words and pieces", IntentOutcomeClarifiedYes, "learn_pieces")
	check("the words and pieces", IntentOutcomeClarifiedNo, "learn_pieces")
	if e := byLineOutcome["got anything on retries|"+IntentOutcomeDispatched]; e.Conf < intentMinConf {
		t.Errorf("dispatched entry conf = %v, below the dispatch gate it supposedly passed", e.Conf)
	}
	if e := byLineOutcome["the words and pieces|"+IntentOutcomeClarifiedYes]; e.Conf != 0.49 {
		t.Errorf("clarified-yes entry conf = %v, want the offered 0.49", e.Conf)
	}
}
