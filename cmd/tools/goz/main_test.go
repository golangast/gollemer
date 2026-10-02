package main

import "testing"

func newTestPicker() *picker {
	p := &picker{}
	p.choices = []Choice{
		{Command: "chat", Comment: "chat with gollemer", Raw: "chat     chat with gollemer"},
		{Command: "sel", Comment: "pick a target", Raw: "sel      pick a target"},
		{Command: "smarter", Comment: "retrain", Raw: "smarter  retrain"},
	}
	p.refilter()
	return p
}

// single column: rows == total
func TestBurstTypingKeepsEveryByte(t *testing.T) {
	p := newTestPicker()
	// "chat" arriving in one read must register all four keystrokes.
	consumed, action := p.processKeys([]byte("chat"), 3, 3)
	if consumed != 4 || action != keyNone {
		t.Fatalf("consumed=%d action=%q, want 4/%q", consumed, action, keyNone)
	}
	if p.query != "chat" {
		t.Fatalf("query=%q, want %q", p.query, "chat")
	}
	if len(p.filtered) != 1 || p.filtered[0].Command != "chat" {
		t.Fatalf("filtered=%v, want only the chat target", p.filtered)
	}
}

func TestBackspaceInBurst(t *testing.T) {
	p := newTestPicker()
	consumed, _ := p.processKeys([]byte("ab\x7fc"), 3, 3)
	if consumed != 4 {
		t.Fatalf("consumed=%d, want 4", consumed)
	}
	if p.query != "ac" {
		t.Fatalf("query=%q, want %q", p.query, "ac")
	}
}

func TestEnter(t *testing.T) {
	p := newTestPicker()
	consumed, action := p.processKeys([]byte("\r"), 3, 3)
	if consumed != 1 || action != keyDone {
		t.Fatalf("consumed=%d action=%q, want 1/%q", consumed, action, keyDone)
	}
}

func TestCtrlC(t *testing.T) {
	p := newTestPicker()
	consumed, action := p.processKeys([]byte("\x03"), 3, 3)
	if consumed != 1 || action != keyQuit {
		t.Fatalf("consumed=%d action=%q, want 1/%q", consumed, action, keyQuit)
	}
}

func TestArrowKeys(t *testing.T) {
	p := newTestPicker()
	if _, _ = p.processKeys([]byte("\x1b[B"), 3, 3); p.sel != 1 {
		t.Fatalf("after down sel=%d, want 1", p.sel)
	}
	if _, _ = p.processKeys([]byte("\x1b[A"), 3, 3); p.sel != 0 {
		t.Fatalf("after up sel=%d, want 0", p.sel)
	}
	if _, _ = p.processKeys([]byte("\x0e"), 3, 3); p.sel != 1 { // Ctrl+N
		t.Fatalf("after Ctrl+N sel=%d, want 1", p.sel)
	}
	if _, _ = p.processKeys([]byte("\x10"), 3, 3); p.sel != 0 { // Ctrl+P
		t.Fatalf("after Ctrl+P sel=%d, want 0", p.sel)
	}
}

func TestSplitEscapeSequenceWaitsForRest(t *testing.T) {
	p := newTestPicker()
	// ESC alone: nothing consumed, nothing happens yet.
	consumed, action := p.processKeys([]byte("\x1b"), 3, 3)
	if consumed != 0 || action != keyNone {
		t.Fatalf("consumed=%d action=%q, want 0/%q", consumed, action, keyNone)
	}
	// The rest arrives on the next read: full sequence consumed, key applied.
	consumed, _ = p.processKeys([]byte("\x1b[B"), 3, 3)
	if consumed != 3 || p.sel != 1 {
		t.Fatalf("consumed=%d sel=%d, want 3 and 1", consumed, p.sel)
	}
}

func TestLoneEscapeDroppedRestKept(t *testing.T) {
	p := newTestPicker()
	// ESC followed by non-'[' bytes: ESC ignored, the rest typed normally.
	consumed, _ := p.processKeys([]byte("\x1bab"), 3, 3)
	if consumed != 3 {
		t.Fatalf("consumed=%d, want 3", consumed)
	}
	if p.query != "ab" {
		t.Fatalf("query=%q, want %q", p.query, "ab")
	}
}

func TestMouseReportSwallowed(t *testing.T) {
	p := newTestPicker()
	// ESC [ M + 3 coordinate bytes: all 6 consumed, query untouched.
	consumed, action := p.processKeys([]byte("\x1b[M !\""), 3, 3)
	if consumed != 6 || action != keyNone {
		t.Fatalf("consumed=%d action=%q, want 6/%q", consumed, action, keyNone)
	}
	if p.query != "" {
		t.Fatalf("query=%q, want empty (click coords must not leak)", p.query)
	}
}

func TestMixedBurst(t *testing.T) {
	p := newTestPicker()
	// Typing, an arrow key, and more typing all in one read.
	consumed, _ := p.processKeys([]byte("s\x1b[Al"), 3, 3)
	if consumed != 5 {
		t.Fatalf("consumed=%d, want 5", consumed)
	}
	if p.query != "sl" {
		t.Fatalf("query=%q, want %q", p.query, "sl")
	}
	if p.sel != 0 {
		t.Fatalf("sel=%d, want 0 (up arrow at top is a no-op)", p.sel)
	}
}
