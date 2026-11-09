package chat

import (
	"errors"
	"strings"
	"testing"
)

// The "watch it work" narration pins. runWatchIt runs a
// command (a make target, `go run`, `go test`) to completion
// and frames its output. The old frame said "It ran fine.
// What you see above is the program talking" for ANY exit-0
// command — including the host repo's `make start`, whose
// own output said it needs a real terminal and started
// nothing (John's screenshot, 2026-11-09). The frame may now
// only say what the runner observed: the exit state and the
// command's own lines. These tests pin the narration
// function directly, one case per rule.

func TestWatchItNarrationFailure(t *testing.T) {
	got := watchItNarration(errors.New("exit status 1"), false, false, []string{"boom: it broke"})
	for _, want := range []string{"It failed", "exit status 1"} {
		if !strings.Contains(got, want) {
			t.Errorf("failure frame missing %q: %q", want, got)
		}
	}
	for _, banned := range []string{"ran fine", "It works", "the program talking", "finished with no error"} {
		if strings.Contains(got, banned) {
			t.Errorf("failure frame contains %q: %q", banned, got)
		}
	}
}

// The screenshot case itself: exit 0, and the command's own
// output says it needs a real terminal. The frame must
// report exactly that — the command started nothing — and
// never a word of success against that evidence.
func TestWatchItNarrationNeedsTerminal(t *testing.T) {
	lines := []string{
		"make[1]: Entering directory '/home/zendrulat/g/gollemer'",
		"bash scripts/start.sh",
		"make start opens the chat and needs a real terminal — for the guide, run: make guide",
		"make[1]: Leaving directory '/home/zendrulat/g/gollemer'",
	}
	got := watchItNarration(nil, false, false, lines)
	for _, want := range []string{"needs a real terminal", "started nothing"} {
		if !strings.Contains(got, want) {
			t.Errorf("terminal frame missing %q: %q", want, got)
		}
	}
	for _, banned := range []string{"ran fine", "the program talking", "It ran", "finished with no error"} {
		if strings.Contains(got, banned) {
			t.Errorf("terminal frame contains %q — success claimed against the output's own words: %q", banned, got)
		}
	}
}

// Exit 0 with ordinary output: a neutral frame. It may say
// the command finished and that the lines are its output;
// it may not claim a program ran, is serving, or talked.
func TestWatchItNarrationPlainSuccess(t *testing.T) {
	got := watchItNarration(nil, false, false, []string{"some ordinary output"})
	if !strings.Contains(got, "finished with no error") {
		t.Errorf("plain frame should report the clean exit: %q", got)
	}
	for _, banned := range []string{"ran fine", "the program talking", "It ran", "is running", "serving"} {
		if strings.Contains(got, banned) {
			t.Errorf("plain frame contains %q — an unobserved program claim: %q", banned, got)
		}
	}
}

// Exit 0 with no output at all: the same neutrality, and no
// invented story about quiet programs doing hidden work.
func TestWatchItNarrationSilentSuccess(t *testing.T) {
	got := watchItNarration(nil, false, false, nil)
	if !strings.Contains(got, "printed nothing") {
		t.Errorf("silent frame should say it printed nothing: %q", got)
	}
	for _, banned := range []string{"ran fine", "quietly", "the program talking", "It ran"} {
		if strings.Contains(got, banned) {
			t.Errorf("silent frame contains %q: %q", banned, got)
		}
	}
}

// The two frames that survive with their claims intact,
// because the runner genuinely observed them: a timeout
// (the process was alive at the cap) and the test leg
// (go test's exit 0 is the tests' own verdict).
func TestWatchItNarrationObservedClaims(t *testing.T) {
	got := watchItNarration(nil, true, false, []string{"serving at :8080"})
	if !strings.Contains(got, "ran for a full minute") || !strings.Contains(got, "I stopped it") {
		t.Errorf("timeout frame changed: %q", got)
	}
	got = watchItNarration(nil, false, true, []string{"ok \texample.com/a", "ok \texample.com/b", "?   \texample.com/c\t[no test files]"})
	if !strings.Contains(got, "2 package(s) of tests passed, none failed") {
		t.Errorf("test frame should count go test's own ok lines: %q", got)
	}
}

func TestOutputNeedsTerminal(t *testing.T) {
	yes := []string{
		"make start opens the chat and needs a real terminal — for the guide, run: make guide",
		"This tool requires an interactive terminal.",
		"NEEDS A TERMINAL",
	}
	for _, l := range yes {
		if !outputNeedsTerminal([]string{l}) {
			t.Errorf("outputNeedsTerminal(%q) = false, want true", l)
		}
	}
	no := []string{
		"listening at http://localhost:8080",
		"the terminal UI renders in color", // names a terminal, claims nothing about needing one
		"tests passed",
		"",
	}
	for _, l := range no {
		if outputNeedsTerminal([]string{l}) {
			t.Errorf("outputNeedsTerminal(%q) = true, want false", l)
		}
	}
}
