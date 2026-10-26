package chat

import (
	"strings"
	"testing"
	"time"
	"unicode/utf8"
)

// pace_test.go — the typewriter emission plan (2026-10-25):
// printPaced used to print a whole line, then take the line's
// whole pause, so answers arrived as blocks — "sentence by
// sentence", in the user's words. The plan now spends each
// line's pause across small chunks of the line itself. These
// tests pin the plan (never wall-clock timing): many small
// pieces, per-line pause totals identical to the schedule
// paceDelayFor prices, escapes and runes never split, budget
// behaviour unchanged, and byte-identical output either way.

// pieceJoin reassembles a plan's texts.
func pieceJoin(pieces []pacePiece) string {
	var b strings.Builder
	for _, p := range pieces {
		b.WriteString(p.text)
	}
	return b.String()
}

func pieceSleep(pieces []pacePiece) time.Duration {
	var total time.Duration
	for _, p := range pieces {
		total += p.sleep
	}
	return total
}

// endsMidEscape reports whether s cuts off inside a CSI escape
// sequence (an ESC[ with no final byte before the string ends).
func endsMidEscape(s string) bool {
	for i := 0; i+1 < len(s); i++ {
		if s[i] != 0x1b || s[i+1] != '[' {
			continue
		}
		j := i + 2
		for j < len(s) && !(s[j] >= '@' && s[j] <= '~') {
			j++
		}
		if j >= len(s) {
			return true
		}
		i = j
	}
	return false
}

func TestPacePlanTypewriterChunks(t *testing.T) {
	sample := "The story of this project, told in plain words for a newcomer.\n" +
		"It reads Go code and explains it, one small piece at a time.\n" +
		"Start where the program starts: main, in main.go, line three.\n" +
		"The engine room is where the most-called functions live.\n" +
		"Understand those and the rest of the project falls into place.\n"
	lines := splitLines(sample)
	pieces := pacePlan(lines)

	if got := pieceJoin(pieces); got != sample {
		t.Fatalf("plan pieces do not reassemble the input:\n%q\nwant:\n%q", got, sample)
	}
	// Many small pieces, not one per line: each paused line of
	// ~60 runes becomes ~20 chunks.
	if len(pieces) < 60 {
		t.Errorf("plan has %d pieces for %d lines — not typewriter granularity", len(pieces), len(lines))
	}
	// Every piece but the last (the final line, which never
	// pauses and so stays whole) is at most a chunk wide.
	for i, p := range pieces[:len(pieces)-1] {
		if w := visibleLen(p.text); w > paceChunkRunes {
			t.Errorf("piece %d is %d runes wide, want <= %d: %q", i, w, paceChunkRunes, p.text)
		}
	}
	// The plan's total pause is the old schedule's, to the
	// nanosecond: paceDelayFor over every line but the last.
	var want time.Duration
	for _, ln := range lines[:len(lines)-1] {
		want += paceDelayFor(visibleLen(ln))
	}
	if got := pieceSleep(pieces); got != want {
		t.Errorf("plan pause total = %v, want the line schedule's %v", got, want)
	}
	// And no single pause is a whole line's lump: each is one
	// chunk's small share.
	for i, p := range pieces {
		if p.sleep >= paceDelayFor(40) {
			t.Errorf("piece %d pauses %v — a line-sized lump, not a chunk share", i, p.sleep)
		}
	}
}

func TestPacePlanEscapesAndRunesStayWhole(t *testing.T) {
	lines := []string{
		"\x1b[31mRed words ◆ drawn here\x1b[0m and plain words after them.\n",
		"A second line, long enough to be cut into many small pieces.\n",
		"A third line so the plan has more than one paused line in it.\n",
	}
	joined := strings.Join(lines, "")
	pieces := pacePlan(lines)
	if got := pieceJoin(pieces); got != joined {
		t.Fatalf("plan pieces do not reassemble the input:\n%q\nwant:\n%q", got, joined)
	}
	for i, p := range pieces {
		if !utf8.ValidString(p.text) {
			t.Errorf("piece %d is not valid UTF-8 (a rune was split): %q", i, p.text)
		}
		if endsMidEscape(p.text) {
			t.Errorf("piece %d ends inside an escape sequence: %q", i, p.text)
		}
	}
	for i, p := range pieces[:len(pieces)-1] {
		if w := visibleLen(p.text); w > paceChunkRunes {
			t.Errorf("piece %d is %d visible runes, want <= %d: %q", i, w, paceChunkRunes, p.text)
		}
	}
}

func TestPacePlanBudgetFlowsWholeLines(t *testing.T) {
	line := strings.Repeat("x", 100) + "\n" // 101 visible, pause 2.22s
	var lines []string
	for i := 0; i < 20; i++ {
		lines = append(lines, line)
	}
	pieces := pacePlan(lines)
	if got := pieceJoin(pieces); got != strings.Join(lines, "") {
		t.Fatal("plan pieces do not reassemble the input")
	}
	total := pieceSleep(pieces)
	if total == 0 || total > paceBurstBudget {
		t.Errorf("plan pause total = %v, want within (0, %v]", total, paceBurstBudget)
	}
	// Once the budget is spent, later lines flow as single
	// whole pieces — the plan grows the chunk to the line
	// rather than dropping or hurrying content. (The final
	// line is always whole, so look before it.)
	flowed := false
	for _, p := range pieces[:len(pieces)-1] {
		if visibleLen(p.text) > paceChunkRunes {
			flowed = true
		}
	}
	if !flowed {
		t.Error("no whole-line piece before the end — budget exhaustion did not switch to flowing")
	}
}

// Paced and unpaced printPaced must emit the same bytes; only
// the timing differs. Runs both through the real printPaced
// with the sleep hook stubbed, on plain and decorated text.
func TestPrintPacedContentIdentity(t *testing.T) {
	oldOn, oldInter, oldSleep := paceOn, paceInteractive, paceSleep
	defer func() { paceOn, paceInteractive, paceSleep = oldOn, oldInter, oldSleep }()
	paceInteractive = true
	pauses := 0
	paceSleep = func(time.Duration) { pauses++ }

	samples := []string{
		"First a plain sentence, in ordinary words for a reader.\n" +
			"Then another, because one line never paces on its own.\n" +
			"A third line carries the answer the rest of the way home.\n",
		"\x1b[36m◆ A heading with colour and a ◆ rune in it\x1b[0m\n" +
			"The body under it types out just the same, rune by rune.\n" +
			"And a last line to close the sample out for the plan.\n",
	}
	for _, sample := range samples {
		pauses = 0
		paceOn = false
		instant := captureStdout(t, func() { printPaced(sample) })
		paceOn = true
		paced := captureStdout(t, func() { printPaced(sample) })
		if paced != instant {
			t.Errorf("paced output differs from instant:\npaced:   %q\ninstant: %q", paced, instant)
		}
		if pauses == 0 {
			t.Error("paced run never paused — the typewriter path did not run")
		}
	}
}
