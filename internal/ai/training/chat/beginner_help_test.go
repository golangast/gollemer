package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/pkg/termui"
)

// The help catalogs moved from plain heredocs to a termui-rendered
// data table. These tests pin the deal: EVERY phrase, command and
// explanation from the old text must still be present (content
// parity), the section structure must render with ◆ headers in
// plain mode too, and styling must add escapes without adding or
// losing a single visible character.

// collapseWS flattens whitespace runs so alignment changes don't
// read as content changes.
func collapseWS(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func helpNorm(s string) string { return collapseWS(stripANSI(s)) }

// checkInventory asserts each item survives in the normalized
// output. Items are matched exactly (case-sensitive), except the
// folded titles, which are matched case-insensitively because the
// section headers render in the house ALL-CAPS header style.
func checkInventory(t *testing.T, name, out string, items, folded []string) {
	t.Helper()
	norm := helpNorm(out)
	for _, it := range items {
		if !strings.Contains(norm, it) {
			t.Errorf("%s lost content %q:\n%s", name, it, out)
		}
	}
	upper := strings.ToUpper(norm)
	for _, it := range folded {
		if !strings.Contains(upper, strings.ToUpper(it)) {
			t.Errorf("%s lost section title %q:\n%s", name, it, out)
		}
	}
}

func TestHelpContentParity(t *testing.T) {
	items := []string{
		// Opener (quotes are part of the text).
		`New here? Say "start" and I'll drive — six short stops, you just say "next".`,
		`Or say "teach me" — six tiny lessons that train YOU on what I can do, by trying each one.`,
		"Three situations cover most of it:",
		// Every sayable phrase and its explanation.
		"start", `the guided walk (or "tour <github link>" to open a project)`,
		"what does this program do", "the plain-English story",
		"what should I do now", "I'll name the next step from where you stand",
		"what does Names do", "any function name",
		"why does Names lock", "the reasons the code itself shows",
		"compare Add and Total", "two functions, side by side",
		"paste any Go error", "I'll translate it into plain words",
		"where for <github issue link>", "where to edit for an issue",
		// Closing lines.
		`Two more worth knowing: "brains" (who works here, and their limits) and "any suggestions" (the Friend's notebook on how you work).`,
		`The full catalog: say "help all".`,
	}
	folded := []string{"I'm lost", "I want to understand one thing", "Something's broken"}
	checkInventory(t, "help", helpText(), items, folded)
}

func TestHelpAllContentParity(t *testing.T) {
	items := []string{
		"tour <github link>", "walk it one stop at a time (then say: next)",
		"give me a tour", "tour the project already open",
		"what does this program do", "the plain-English summary",
		"what are the main types", "its key words, one plain line each",
		"what should I learn first", "its ideas, in the order to learn them",
		"example Names", "see a function used for real (any name works)",
		"run it", "watch the project run, narrated",
		"teach me", "learn what I can do — six tiny lessons, you try each one",
		"/speed", "long answers stream in readably instead of dumping (say it again for full speed)",
		"what should I do now", "I'll look at where you are and name the next step",
		"any suggestions", "the Friend speaks — patterns it noticed in how you work",
		"tell me more", "go one step deeper into whatever we just did",
		"help", "this list",
		"analyze <github link or folder>",
		"what does Names do",
		"why does Names lock", "the reasons the code itself shows",
		"compare Add and Total", "two functions, side by side",
		"where for <github issue link>", "where to edit for an issue",
		"(another terminal) gollemer -watch <folder>", "the Watcher diffs your code as you save; the Friend reports what mattered",
		"brains", "who does the work here, and their honest limits",
		"how are your brains doing", "the scoreboard — who answered, who failed",
		"why did you do that", "I'll name the brain and the recorded reason",
		"paste any Go error", "I'll translate it into plain words",
		"/fix undefined: foo", "(shortcut for the same idea)",
		"write a function that reverses a string",
		"Or just talk — ask me anything. Say 'exit' to leave.",
	}
	folded := []string{"Things you can say", "Learn a project", "When you're stuck",
		"Ask about code", "The team behind the answers", "Fix and write"}
	checkInventory(t, "help all", helpAllText(), items, folded)
}

// setStyling flips both styling switches (the chat colorizer and
// termui's) together, the way a real terminal / NO_COLOR would.
func setStyling(t *testing.T, on bool) {
	t.Helper()
	oldC, oldT := colorEnabled, termui.Enabled
	colorEnabled, termui.Enabled = on, on
	t.Cleanup(func() { colorEnabled, termui.Enabled = oldC, oldT })
}

func TestHelpPlainModeNoEscapesButGlyphs(t *testing.T) {
	setStyling(t, false)
	for name, out := range map[string]string{"help": helpText(), "help all": helpAllText()} {
		if strings.Contains(out, "\x1b") {
			t.Errorf("%s plain mode carries escapes:\n%q", name, out)
		}
		if !strings.Contains(out, "◆") {
			t.Errorf("%s plain mode lost its ◆ section headers:\n%s", name, out)
		}
	}
}

func TestHelpStyledMode(t *testing.T) {
	setStyling(t, false)
	plainHelp, plainAll := helpText(), helpAllText()
	setStyling(t, true)
	for name, pair := range map[string][2]string{
		"help":     {helpText(), plainHelp},
		"help all": {helpAllText(), plainAll},
	} {
		styled, plain := pair[0], pair[1]
		if !strings.Contains(styled, "\x1b[") {
			t.Errorf("%s styled mode carries no SGR sequences:\n%q", name, styled)
		}
		// The sayable column must be colored: cyan is SGR 36.
		if !strings.Contains(styled, "\x1b[36m") {
			t.Errorf("%s styled mode has no cyan sayables:\n%q", name, styled)
		}
		// Styling must never add or lose visible content.
		if stripANSI(styled) != plain {
			t.Errorf("%s visible text differs between styled and plain modes:\nstyled:\n%s\nplain:\n%s", name, stripANSI(styled), plain)
		}
	}
}
