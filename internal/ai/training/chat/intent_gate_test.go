package chat

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The session-thread gates (intent.go): a what_now / tell_more
// verdict from the intent classifier is honored only with
// deterministic session evidence — a family marker in the line
// and, for tell_more, a continuable context. Measured 2026-11-15
// against the shipped classifier: "i am nervous about beekeeping"
// classified as tell_more at 0.994 and was dispatched to the
// beginner guidance; "what is today" and "what is go" classified
// as what_now at 0.995 / 1.000 and got the same. These tests pin
// both directions: the stolen lines fall through, the genuine
// session asks still dispatch.

// freshSessionState clears the session-thread state the gates
// and handlers consult, and returns a restore function.
func freshSessionState(t *testing.T) func() {
	t.Helper()
	oldKind, oldSubject := lastActionKind, lastSubject
	oldStops, oldIdx := tourStops, tourIdx
	oldProj := lastAnalyzeProject
	lastActionKind, lastSubject = "", ""
	tourStops, tourIdx = nil, 0
	lastAnalyzeProject = nil
	return func() {
		lastActionKind, lastSubject = oldKind, oldSubject
		tourStops, tourIdx = oldStops, oldIdx
		lastAnalyzeProject = oldProj
	}
}

func TestSessionThreadGateShapes(t *testing.T) {
	// Stolen shapes: no session marker for either gate.
	for _, line := range []string{
		"i am nervous about beekeeping",
		"what is today",
		"what is go",
		"i am worried about my exam",
		"i feel nervous about the dentist",
		"what is rust",
		"what's new?",
		"how was your day",
		"tell me something interesting",
	} {
		if whatNowShape(line) {
			t.Errorf("whatNowShape(%q) = true, want false (no session marker)", line)
		}
		if tellMoreShape(line) {
			t.Errorf("tellMoreShape(%q) = true, want false (no more-family marker)", line)
		}
	}
	// Genuine what_now family shapes.
	for _, line := range []string{
		"im stuck", "i am stuck", "im lost", "im confused",
		"what should i do now", "what should i do", "what now",
		"what do i do next", "what do i do", "what next", "whats next",
		"what to do now", "where do i go from here", "help me",
		"how do i get started", "whats the next step", "next steps",
		"directions please", "please guide me",
		"hwat should i do", "pls wht now", // the family's typo variants
	} {
		if !whatNowShape(line) {
			t.Errorf("whatNowShape(%q) = false, want true (genuine family shape)", line)
		}
	}
	// Genuine tell_more family shapes.
	for _, line := range []string{
		"tell me more", "tell me more about that", "more", "go on",
		"continue", "keep going", "what else", "im listening",
		"and then", "what happened next", "details pls", "dig deeper",
		"tell me omre", // the family's typo variants
	} {
		if !tellMoreShape(line) {
			t.Errorf("tellMoreShape(%q) = false, want true (genuine family shape)", line)
		}
	}
}

// The gates' shapes are derived from the intent families, so the
// families themselves are the recall pin: every what_now line in
// the training pairs and the held-out file must pass whatNowShape,
// and every tell_more line tellMoreShape. A future family addition
// with a new stem must extend the gate — this test is what says so.
func TestSessionThreadGateFamilyRecall(t *testing.T) {
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	counts := map[string]int{}
	var misses []string
	for _, name := range []string{"intent_pairs.jsonl", "intent_heldout.jsonl"} {
		f, err := os.Open(filepath.Join(repoRoot, "data", "training", name))
		if err != nil {
			t.Fatal(err)
		}
		sc := bufio.NewScanner(f)
		sc.Buffer(make([]byte, 1024*1024), 1024*1024)
		for sc.Scan() {
			var pair struct {
				Input  string `json:"input"`
				Intent string `json:"intent"`
			}
			if err := json.Unmarshal(sc.Bytes(), &pair); err != nil {
				continue
			}
			switch pair.Intent {
			case "what_now":
				counts["what_now"]++
				if !whatNowShape(pair.Input) {
					misses = append(misses, "what_now: "+pair.Input)
				}
			case "tell_more":
				counts["tell_more"]++
				if !tellMoreShape(pair.Input) {
					misses = append(misses, "tell_more: "+pair.Input)
				}
			}
		}
		f.Close()
	}
	if counts["what_now"] == 0 || counts["tell_more"] == 0 {
		t.Fatalf("family files yielded no pairs to check: %v", counts)
	}
	if len(misses) > 0 {
		show := misses
		if len(show) > 10 {
			show = show[:10]
		}
		t.Errorf("gate shapes decline %d of %d family lines (first misses: %s)",
			len(misses), counts["what_now"]+counts["tell_more"], strings.Join(show, "; "))
	}
}

func TestIntentGateDeclinesStolenVerdicts(t *testing.T) {
	restore := freshSessionState(t)
	defer restore()

	// The measured thefts, with the classifier's verdicts forced:
	// on a fresh session none of them may be handled.
	if dispatchIntent("tell_more", "i am nervous about beekeeping", nil) {
		t.Error("tell_more verdict claimed a social line, want the gate to fall through")
	}
	if dispatchIntent("what_now", "what is today", nil) {
		t.Error("what_now verdict claimed a social line, want the gate to fall through")
	}
	if dispatchIntent("what_now", "what is go", nil) {
		t.Error("what_now verdict claimed a knowledge question, want the gate to fall through")
	}
	// Siblings of the same shapes.
	if dispatchIntent("what_now", "i am worried about my exam", nil) {
		t.Error("what_now verdict claimed a first-person feeling statement")
	}
	if dispatchIntent("tell_more", "i feel nervous about the dentist", nil) {
		t.Error("tell_more verdict claimed a first-person feeling statement")
	}
	if dispatchIntent("what_now", "what is rust", nil) {
		t.Error("what_now verdict claimed a bare what-is question")
	}
	// Each gate condition alone is not enough: a marker with no
	// continuable context declines (fresh session — nothing to
	// continue), and a context with no marker declines.
	if dispatchIntent("tell_more", "tell me more", nil) {
		t.Error("tell_more with a marker but no session context dispatched, want decline")
	}
	lastActionKind = "example"
	if dispatchIntent("tell_more", "i am nervous about beekeeping", nil) {
		t.Error("tell_more with context but no marker claimed a social line, want decline")
	}
	lastActionKind = ""
}

func TestIntentGateKeepsGenuineDispatches(t *testing.T) {
	restore := freshSessionState(t)
	defer restore()

	for _, line := range []string{"im stuck", "what should i do now", "what now"} {
		var handled bool
		out := captureStdout(t, func() { handled = dispatchIntent("what_now", line, nil) })
		if !handled || !strings.Contains(out, "tour") {
			t.Errorf("genuine what_now dispatch changed for %q: handled=%v out=%q", line, handled, out)
		}
	}
	// tell_more with both conditions: a tour in progress is a
	// continuable context, and "tell me more" carries the marker.
	tourStops = []string{"first stop", "second stop"}
	var handled bool
	captureStdout(t, func() { handled = dispatchIntent("tell_more", "tell me more", nil) })
	if !handled {
		t.Error("tell_more with context and marker declined, want the continuation dispatch")
	}
	tourStops = nil
	// A recorded last action is also a continuable context.
	lastActionKind = "example"
	captureStdout(t, func() { handled = dispatchIntent("tell_more", "tell me more about that", nil) })
	if !handled {
		t.Error("tell_more after a recorded action declined, want the continuation dispatch")
	}
	lastActionKind = ""
}

// The theft through the real layer: with the SHIPPED classifier
// live, the measured stolen lines must fall through
// tryIntentDispatch untouched, while a genuine what-now line
// still dispatches — the same arrangement as
// TestIntentLayerLeavesStrongConceptQuestions.
func TestIntentLayerLeavesStolenSocialLines(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	oldRoot := unifiedProjectRoot
	unifiedProjectRoot = repoRoot
	defer func() { unifiedProjectRoot = oldRoot }()
	restore := freshSessionState(t)
	defer restore()

	m := defaultIntentModel()
	if m == nil {
		t.Skip("intent model unavailable in this test process (its lazy loader was already consumed without a data root); the gates are pinned by the dispatch-level tests and the live smoke")
	}
	// The thefts are real, not hypothetical: assert the shipped
	// classifier's verdicts so a future model that stops
	// misfiring makes this test say so instead of silently
	// going vacuous.
	for line, want := range map[string]string{
		"i am nervous about beekeeping": "tell_more",
		"what is today":                 "what_now",
		"what is go":                    "what_now",
	} {
		if intent, _, ok := m.ClassifyIntent(line); !ok || intent != want {
			t.Fatalf("shipped classifier verdict for %q = %q, ok=%v; want the recorded %s theft", line, intent, ok, want)
		}
		if tryIntentDispatch(line, nil) {
			t.Errorf("intent layer claimed %q, want fall-through past the session gates", line)
		}
	}
	// The intent layer's own territory still dispatches.
	var handled bool
	out := captureStdout(t, func() { handled = tryIntentDispatch("im stuck", nil) })
	if !handled {
		t.Error("intent layer declined \"im stuck\", want the what_now dispatch to survive the gate")
	}
	if !strings.Contains(out, "Nothing's open yet") && !strings.Contains(out, "tour") {
		t.Errorf("what_now dispatch for \"im stuck\" printed %q, want the beginner guidance", out)
	}
}
