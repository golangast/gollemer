package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/pkg/termui"
)

// readability_test.go — the 2026-10-24 readability pass: per-brain
// reply tags, the decorate pass over non-termui long answers, and
// the start screen's shape.

// Reply tags take the answering brain's hue: green for gocode,
// cyan for gocli, blue for go, white for goanalyze, purple default.
func TestCTagPerBrain(t *testing.T) {
	old := colorEnabled
	defer func() { colorEnabled = old }()
	colorEnabled = false
	if got := cTag("gollemer [gocode]>"); got != "gollemer [gocode]>" {
		t.Fatalf("color off: tag must be plain, got %q", got)
	}
	colorEnabled = true
	cases := map[string]string{
		"gollemer [gocode]>":    "32m", // green
		"gollemer [gocli]>":     "36m", // cyan
		"gollemer [go]>":        "34m", // blue
		"gollemer [goanalyze]>": "37m", // white
		"gollemer [social]>":    "35m", // purple
		"gollemer [memory]>":    "35m", // unknown brains keep the default
	}
	for tag, code := range cases {
		if got := cTag(tag); !strings.Contains(got, code) {
			t.Errorf("cTag(%q) = %q, want SGR %s", tag, got, code)
		}
	}
	// The six brains' tags are all different colors.
	seen := map[string]string{}
	for _, d := range []string{"social", "go", "gocode", "gocli", "makefile", "goanalyze"} {
		got := cTag("gollemer [" + d + "]>")
		for other, prev := range seen {
			if prev == got {
				t.Errorf("brains %s and %s share a tag color", other, d)
			}
		}
		seen[d] = got
	}
}

// The decorate pass is the identity with color off, and lights
// commands, backticks, and file:line refs with color on. Lines
// already carrying ANSI pass through untouched.
func TestDecorateOutput(t *testing.T) {
	oldC, oldT := colorEnabled, termui.Enabled
	defer func() { colorEnabled, termui.Enabled = oldC, oldT }()

	sample := "Run make chat to start\nsee `tour` and main.go:12 for details\nENGINE ROOM stuff happens"
	colorEnabled, termui.Enabled = false, false
	if got := decorateOutput(sample); got != sample {
		t.Fatalf("color off: decorate must be identity, got %q", got)
	}
	colorEnabled, termui.Enabled = true, true
	got := decorateOutput(sample)
	if !strings.Contains(got, "\x1b[36mmake chat\x1b[0m") {
		t.Errorf("make command not lit: %q", got)
	}
	if !strings.Contains(got, "\x1b[36mtour\x1b[0m") {
		t.Errorf("backticked span not lit: %q", got)
	}
	if !strings.Contains(got, "main.go") || !strings.Contains(got, "\x1b[1;33m12\x1b[0m") {
		t.Errorf("file:line not styled: %q", got)
	}
	styled := "\x1b[36malready\x1b[0m make chat"
	if out := decorateOutput(styled); out != styled {
		t.Errorf("pre-styled line must pass through untouched, got %q", out)
	}
}

// The start screen is one panel: the name in the border, what it
// is, the three starter lines, and the folder chip — plus the
// capability block beneath the panel — and nothing from the old
// startup dump (no phrase catalog, no most-run line). The screen
// is record-driven now, so the shape test pins an EMPTY home: the
// generic screen a fresh install gets.
func TestStartScreenShape(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	friendResetForTest()
	defer friendResetForTest()
	oldT := termui.Enabled
	defer func() { termui.Enabled = oldT }()
	termui.Enabled = false
	out := startScreenText("/home/user/projects/widget")
	for _, want := range []string{"GOLLEMER", "start", "help", "/quit", "widget", "brains ready"} {
		if !strings.Contains(out, want) {
			t.Errorf("start screen missing %q:\n%s", want, out)
		}
	}
	for _, banned := range []string{"Say things like:", "most-run", "what are the main types"} {
		if strings.Contains(out, banned) {
			t.Errorf("start screen contains old startup dump %q:\n%s", banned, out)
		}
	}
	if strings.Contains(out, "\x1b[") {
		t.Errorf("start screen must carry no escapes with styling off:\n%q", out)
	}
	// Height budget: the panel (11 lines at most, with the teach
	// line) plus the 4 capability lines John's intro-screen ask
	// added — and a line of headroom. Anything taller is bloat.
	if n := len(strings.Split(out, "\n")); n > 16 {
		t.Errorf("start screen too tall (%d lines) — print less:\n%s", n, out)
	}
}
