package chat

import (
	"strings"
	"testing"
	"time"
)

// color_test.go — the pretty-printing layer: color only on a real
// terminal, clean text everywhere else, and the width math the raw
// line editor depends on.

func TestColorizeRespectsSwitch(t *testing.T) {
	old := colorEnabled
	defer func() { colorEnabled = old }()

	colorEnabled = false
	if got := colorize("red", "", "boom"); got != "boom" {
		t.Fatalf("color off must return the plain string, got %q", got)
	}
	if got := cPrompt("you> "); got != "you> " {
		t.Fatalf("prompt must stay plain when color is off, got %q", got)
	}

	colorEnabled = true
	got := colorize("red", "", "boom")
	if !strings.HasPrefix(got, "\x1b[31m") || !strings.HasSuffix(got, "\x1b[0m") {
		t.Fatalf("red should wrap in SGR 31 + reset, got %q", got)
	}
	if got := colorize("black", "cyan", "L"); !strings.Contains(got, "30;46") {
		t.Fatalf("fg+bg should combine codes, got %q", got)
	}
	if got := colorize("dpurple", "", "x"); !strings.Contains(got, "\x1b[23m") {
		t.Fatalf("John's dpurple name must map to 23, got %q", got)
	}
}

func TestStripANSIAndVisibleLen(t *testing.T) {
	old := colorEnabled
	defer func() { colorEnabled = old }()
	colorEnabled = true

	p := cPrompt("you> ")
	if visibleLen(p) != 5 {
		t.Fatalf("colored prompt visible width = %d, want 5", visibleLen(p))
	}
	if stripANSI(p) != "you> " {
		t.Fatalf("stripANSI = %q, want the plain prompt", stripANSI(p))
	}
	if visibleLen("plain") != 5 {
		t.Fatal("plain text width must be its rune count")
	}
}

// TestPrintPacedBudget: a long answer streams (pauses happen) but the
// total added delay never exceeds the burst budget, and short output
// never pauses at all.
func TestPrintPacedBudget(t *testing.T) {
	oldOn, oldInter, oldSleep := paceOn, paceInteractive, paceSleep
	defer func() { paceOn, paceInteractive, paceSleep = oldOn, oldInter, oldSleep }()
	paceOn, paceInteractive = true, true

	var pauses int
	var total time.Duration
	paceSleep = func(d time.Duration) { pauses++; total += d }

	printPaced("one line\n")
	if pauses != 0 {
		t.Fatalf("short output must not pace, got %d pauses", pauses)
	}

	var sb strings.Builder
	for i := 0; i < 500; i++ {
		sb.WriteString("a fairly long line of output that the eye has to travel across the screen\n")
	}
	printPaced(sb.String())
	if pauses == 0 {
		t.Fatal("a 500-line answer should stream, not dump")
	}
	if total > paceBurstBudget {
		t.Fatalf("pacing added %v, budget is %v", total, paceBurstBudget)
	}

	paceInteractive = false
	pauses = 0
	printPaced(sb.String())
	if pauses != 0 {
		t.Fatal("piped (non-interactive) output must never pace")
	}
}

// TestTeachTranslations pins the sentence forms that open, advance,
// and close the training tour — and that they do not steal phrases
// owned by other flows.
func TestTeachTranslations(t *testing.T) {
	cases := map[string]string{
		"teach me":                "/teach",
		"Teach me!":               "/teach",
		"training":                "/teach",
		"beginner tour":           "/teach",
		"show me what you can do": "/teach",
		"next lesson":             "/nextlesson",
		"skip the lesson":         "/nextlesson",
		"stop teaching":           "/teachstop",
		"speed":                   "/speed",
		"speed off":               "/speed off",
		"type slower":             "/speed on",
		// Owned elsewhere — must NOT be stolen:
		"how do i use this": "/what-now",
		"take the tour":     "/tour",
		"run it":            "/watch",
	}
	for in, want := range cases {
		if got := translateBeginnerSentence(in); got != want {
			t.Errorf("translateBeginnerSentence(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestLessonRepsMatchTheirHints: every lesson's TRY phrase must be
// accepted by its own checker — a tour that rejects the exact phrase
// it told the user to say is worse than no tour.
func TestLessonRepsMatchTheirHints(t *testing.T) {
	// lessonList reads the open project; with none open the names
	// degrade, so open the bundled demo the way ensureProject does.
	if !ensureProject() {
		t.Skip("no project available for lessons")
	}
	for i, l := range lessonList() {
		low := strings.Trim(strings.ToLower(l.tryHint), ".!? ")
		if !l.try(low) {
			t.Errorf("lesson %d (%s): its own hint %q is not accepted by its checker", i+1, l.title, l.tryHint)
		}
	}
}
