package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// beginner_pass_test.go — the beginner pass (2026-10-25): one
// walk at a time with one numbering, per-project curricula
// (no Gollemer-self stops inside someone else's project), the
// concepts first screen's density caps, and the slowed pacing
// default. The analyzer-side ranking/description tests live in
// analyze/beginner_screen_test.go.

// The wails screenshot sequence: a guided walk is in flight,
// the user starts a project tour, says "next" — and must get
// tour stop 2, never a guided stop. And the mirror image.
func TestNextNeverWandersBetweenWalks(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	friendResetForTest()
	guideActive, guideIdx, guideRoot, guideSteps = false, 0, "", nil
	tourStops, tourIdx = nil, 0
	defer func() {
		guideActive, tourStops, guideSteps = false, nil, nil
	}()
	oldProj := lastAnalyzeProject
	defer func() { lastAnalyzeProject = oldProj }()
	if !ensureProject() {
		t.Skip("no project available")
	}

	// Guided walk first, then the tour claims "next".
	captureStdout(t, func() { handleBeginnerCommand("start", nil) })
	if !guideActive {
		t.Fatal("guided walk did not start")
	}
	out := captureStdout(t, func() { handleBeginnerCommand("tour", nil) })
	if !strings.Contains(out, "Stop 1 of") {
		t.Fatalf("tour did not open with its own stop 1:\n%s", firstLines(out, 4))
	}
	if guideActive {
		t.Fatal("starting the tour left the guided walk active")
	}
	out = captureStdout(t, func() { handleBeginnerCommand("next", nil) })
	if strings.Contains(out, "Guided stop") || !strings.Contains(out, "Stop 2 of") {
		t.Fatalf("'next' wandered out of the tour:\n%s", firstLines(out, 4))
	}

	// Mirror: tour in flight, "start" claims "next". (Fresh
	// notebook, so the walk starts at stop 1 rather than
	// resuming the first leg's bookmark.)
	friendResetForTest()
	if m := friendLoad(); m.Guided != nil {
		delete(m.Guided, lastAnalyzeProject.Root)
		friendSave()
	}
	guideActive, guideSteps, tourStops, tourIdx = false, nil, nil, 0
	captureStdout(t, func() { handleBeginnerCommand("tour", nil) })
	captureStdout(t, func() { handleBeginnerCommand("start", nil) })
	if !guideActive {
		t.Fatal("guided walk did not restart")
	}
	if tourStops != nil {
		t.Fatal("starting the guided walk left the tour active")
	}
	out = captureStdout(t, func() { handleBeginnerCommand("next", nil) })
	if !strings.Contains(out, "Stop 2 of") {
		t.Fatalf("'next' did not advance the guided walk:\n%s", firstLines(out, 4))
	}
}

// The curriculum is per-project: the Friend stop belongs to
// the self-tour only.
func TestGuideStepsPerProject(t *testing.T) {
	self := guideStepsFor(gollemerModule)
	if len(self) != len(guideCurriculum) {
		t.Fatalf("self walk = %d steps, want the full %d", len(self), len(guideCurriculum))
	}
	ext := guideStepsFor("example.com/someone-elses-project")
	if len(ext) != len(guideCurriculum)-1 {
		t.Fatalf("external walk = %d steps, want %d", len(ext), len(guideCurriculum)-1)
	}
	for _, s := range ext {
		if s.kind == "friend" {
			t.Fatal("external walk still contains the Friend stop")
		}
	}
}

// extWalkFixture builds a small external project (not
// gollemer) with six exported types — enough concepts to
// exercise the first-screen caps.
func extWalkFixture(t *testing.T) *analyze.Project {
	t.Helper()
	root := t.TempDir()
	write := func(rel, body string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/extwalk\n\ngo 1.21\n")
	write("main.go", `// Command extwalk is a tiny demo program.
package main

func main() {
	a := &Alpha{}
	a.Greet()
	b := &Beta{}
	b.Open()
}
`)
	write("demo.go", `// Package main holds the demo's types.
package main

// Alpha is the first thing the program greets with.
type Alpha struct{ Name string }

// Greet says hello.
func (a *Alpha) Greet() {}

// Beta opens the demo's main window.
type Beta struct{ Title string }

// Open shows the window.
func (b *Beta) Open() {}

// Gamma holds the demo's settings for one run.
type Gamma struct{ Width int }

// Delta is a counter of demo events.
type Delta struct{ Count int }

// Epsilon formats demo output lines.
type Epsilon struct{ Prefix string }

// Zeta is the demo's closing screen.
type Zeta struct{ Done bool }
`)
	p, err := analyze.Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// A full guided walk over an external project: five stops,
// one header system, sequential numbering, no Friend lecture,
// and a concepts stop that respects the first-screen caps.
func TestExternalGuidedWalk(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	friendResetForTest()
	oldProj := lastAnalyzeProject
	defer func() { lastAnalyzeProject = oldProj }()
	lastAnalyzeProject = extWalkFixture(t)
	guideActive, guideSteps = false, nil
	defer func() { guideActive, guideSteps = false, nil }()

	var all strings.Builder
	out := captureStdout(t, func() { handleStart(false) })
	all.WriteString(out)
	if !strings.Contains(out, "Five short stops") {
		t.Fatalf("external walk intro should promise five stops:\n%s", firstLines(out, 3))
	}
	for guideActive {
		out = captureStdout(t, func() {
			if !guideNext() {
				t.Fatal("guideNext refused mid-walk")
			}
		})
		all.WriteString(out)
	}
	text := all.String()
	if strings.Contains(text, "Guided stop") {
		t.Error("old two-system header survived somewhere in the walk")
	}
	if strings.Contains(text, "quiet member of the team") {
		t.Error("the Friend stop lectured during an external project's walk")
	}
	for i := 1; i <= 5; i++ {
		hdr := "Stop " + itoaTest(i) + " of 5"
		if !strings.Contains(text, hdr) {
			t.Errorf("walk missing header %q", hdr)
		}
	}
	// The concepts stop (last one) shows the first screen:
	// at most 4 numbered concepts and the more-pointer.
	last := strings.LastIndex(text, "Stop 5 of 5")
	conceptStop := text[last:]
	numbered := 0
	for _, ln := range strings.Split(conceptStop, "\n") {
		trim := strings.TrimSpace(ln)
		if len(trim) > 2 && trim[0] >= '1' && trim[0] <= '9' && trim[1] == '.' {
			numbered++
		}
	}
	if numbered == 0 || numbered > 4 {
		t.Errorf("concepts stop shows %d numbered concepts, want 1..4:\n%s", numbered, firstLines(conceptStop, 14))
	}
	if !strings.Contains(conceptStop, "… and") {
		t.Errorf("concepts stop lacks the reachable-remainder pointer:\n%s", firstLines(conceptStop, 14))
	}
}

func itoaTest(n int) string {
	return string(rune('0' + n))
}

// The pacing default: a representative five-line answer now
// streams at a reading pace (tens of chars/sec, not ~1,600),
// pauses grow with line length, and /speed still switches
// pacing fully off and back on.
func TestPaceDefaultRate(t *testing.T) {
	sample := "The story of this project, told in plain words for a newcomer.\n" +
		"It reads Go code and explains it, one small piece at a time.\n" +
		"Start where the program starts: main, in main.go, line three.\n" +
		"The engine room is where the most-called functions live.\n" +
		"Understand those and the rest of the project falls into place.\n"
	lines := splitLines(sample)
	var chars int
	var total time.Duration
	for i, ln := range lines {
		chars += visibleLen(ln)
		if i < len(lines)-1 {
			total += paceDelayFor(visibleLen(ln))
		}
	}
	rate := float64(chars) / total.Seconds()
	if rate < 30 || rate > 75 {
		t.Errorf("default pacing = %.0f chars/sec, want a reading pace (30..75)", rate)
	}
	if paceDelayFor(80) <= paceDelayFor(10) {
		t.Error("longer lines should pause longer")
	}
	if paceBurstBudget < 10*time.Second {
		t.Errorf("burst budget %v too small for the slower pace", paceBurstBudget)
	}
}

// /speed off must make printPaced instant again; /speed on
// restores the pauses. Uses the paceSleep hook, so no real
// waiting happens.
func TestSpeedToggleStillControlsPacing(t *testing.T) {
	oldOn, oldInteractive, oldSleep := paceOn, paceInteractive, paceSleep
	defer func() { paceOn, paceInteractive, paceSleep = oldOn, oldInteractive, oldSleep }()
	pauses := 0
	paceSleep = func(time.Duration) { pauses++ }
	paceInteractive = true
	long := strings.Repeat("A line of the answer that a beginner is reading.\n", 6)

	paceOn = true
	captureStdout(t, func() { printPaced(long) })
	if pauses == 0 {
		t.Fatal("pacing on: printPaced never paused")
	}
	pauses = 0
	captureStdout(t, func() { setSpeed(false) })
	if paceOn {
		t.Fatal("setSpeed(false) left paceOn set")
	}
	captureStdout(t, func() { printPaced(long) })
	if pauses != 0 {
		t.Errorf("pacing off: printPaced paused %d times", pauses)
	}
	captureStdout(t, func() { setSpeed(true) })
	if !paceOn {
		t.Fatal("setSpeed(true) did not restore paceOn")
	}
}
