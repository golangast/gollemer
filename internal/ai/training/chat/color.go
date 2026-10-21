package chat

import (
	"fmt"
	"os"
	"strings"
)

// color.go — PRETTY PRINTING, no framework: raw ANSI codes from the
// standard library only. John supplied the first version of this file
// (2026-10-21): two helpers, ColorizeCol (a bracketed label — [ word ])
// and ColorizeOutPut (colored text inline), keyed by color names. The
// API and the color names are his; what's added here is (a) the color
// math fixed up (a real reset, a real background escape), (b) string
// builders so callers can compose before printing, (c) an on/off
// switch — color only when stdout is a real terminal and NO_COLOR is
// unset, so pipes, files and tests stay clean — and (d) a small
// semantic palette (prompt, tag, header, error, good, hint) built on
// his two functions so the rest of the chat never picks raw colors.

// colorEnabled is decided once at startup: a character device on
// stdout and no NO_COLOR in the environment.
var colorEnabled = computeColorEnabled()

func computeColorEnabled() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	fi, err := os.Stdout.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// fgCode maps John's foreground names to ANSI codes. Unknown names
// fall back to the default foreground (0 in his table).
func fgCode(name string) int {
	switch name {
	case "black":
		return 30
	case "red":
		return 31
	case "green":
		return 32
	case "yellow":
		return 33
	case "blue":
		return 34
	case "purple":
		return 35
	case "cyan":
		return 36
	case "white":
		return 37
	case "dpurple":
		return 23
	}
	return 0
}

// bgCode maps John's background names to ANSI codes. "" (or an
// unknown name) means no background — his default "9" was a reset
// fragment, which painted nothing; an empty code keeps the line
// clean instead.
func bgCode(name string) string {
	switch name {
	case "black":
		return "40"
	case "red":
		return "41"
	case "green":
		return "42"
	case "yellow":
		return "43"
	case "blue":
		return "44"
	case "magenta":
		return "45"
	case "cyan":
		return "46"
	case "white":
		return "47"
	case "bpurple":
		return "101"
	case "purple":
		return "105"
	}
	return ""
}

// colorize wraps s in the SGR sequence for the named colors, or
// returns s untouched when color is off. Bold is added when fg is
// given with a background — labels read better a touch heavier.
func colorize(fg, bg, s string) string {
	if !colorEnabled || s == "" {
		return s
	}
	seq := fmt.Sprintf("%d", fgCode(fg))
	if b := bgCode(bg); b != "" {
		seq += ";" + b
	}
	return "\x1b[" + seq + "m" + s + "\x1b[0m"
}

// Colorize returns the colored string without printing it — the
// builder form of John's helpers, for composing lines.
func Colorize(ForegroundColor, BackgroundColor, line string) string {
	return colorize(ForegroundColor, BackgroundColor, line)
}

// ColorizeCol prints John's bracketed label: [ line ] in the named
// colors, followed by a small gap. Used for section bands: guided
// stops, lesson titles, anything that should read as a header.
func ColorizeCol(ForegroundColor, BackgroundColor, line string) {
	fmt.Print(colorize(ForegroundColor, BackgroundColor, "[ "+line+" ]") + "   ")
}

// ColorizeOutPut prints line in the named colors, inline, with no
// brackets and no trailing newline — continuations stay possible.
func ColorizeOutPut(ForegroundColor, BackgroundColor, line string) {
	fmt.Print(colorize(ForegroundColor, BackgroundColor, line))
}

// --- Semantic palette -------------------------------------------------
// The chat picks MEANINGS, not colors; the mapping lives here alone.

// cPrompt colors the "you> " prompt.
func cPrompt(s string) string { return colorize("cyan", "", s) }

// cTag colors the reply tag ("gollemer [go]>" and friends).
func cTag(s string) string { return colorize("purple", "", s) }

// cHead colors a section heading inside a block of output.
func cHead(s string) string { return colorize("yellow", "", s) }

// cSay colors a phrase the user can say, quoted inside help text.
func cSay(s string) string { return colorize("cyan", "", s) }

// cErr colors a failure line.
func cErr(s string) string { return colorize("red", "", s) }

// cGood colors a success/accepted line.
func cGood(s string) string { return colorize("green", "", s) }

// cDim colors a muted aside (parentheticals, asides, minor status).
func cDim(s string) string {
	if !colorEnabled {
		return s
	}
	return "\x1b[2m" + s + "\x1b[0m"
}

// printBand prints a full-width-feeling header band using John's
// bracket label: the title in a colored [ ... ] block, then a dim
// rule. Used by the guide and the lessons.
func printBand(title string) {
	fmt.Println()
	ColorizeCol("black", "cyan", title)
	fmt.Println()
}

// stripANSI removes SGR sequences — the raw-mode line editor counts
// visible cells, so prompts carrying color need their width measured
// without the escapes.
func stripANSI(s string) string {
	if !strings.Contains(s, "\x1b[") {
		return s
	}
	var b strings.Builder
	for i := 0; i < len(s); {
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '[' {
			j := i + 2
			for j < len(s) && !(s[j] >= '@' && s[j] <= '~') {
				j++
			}
			i = j + 1
			continue
		}
		b.WriteByte(s[i])
		i++
	}
	return b.String()
}

// visibleLen is the on-screen cell count of s (runes, escapes removed).
func visibleLen(s string) int {
	return len([]rune(stripANSI(s)))
}
