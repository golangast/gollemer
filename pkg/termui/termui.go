// Package termui is Gollemer's home-grown terminal UI library:
// a named style palette, a small symbol vocabulary, and components
// (section headers, bar charts, tables, stat lines, path lists,
// prose blocks) that render plain data into readable terminal text.
//
// Pure Go standard library, no dependencies. Two rules govern
// everything here:
//
//   - ANSI escapes are decoration, never content: styling applies
//     only when Enabled — stdout is a real terminal and NO_COLOR is
//     unset, the same gate chat/color.go uses. With styling off,
//     every Style.Apply returns its input untouched, so piped and
//     NO_COLOR output contains zero escape bytes.
//   - GLYPHS are content: the symbols, bar tracks, and layout the
//     components produce appear in plain mode too, because they
//     carry the readability (a dim track shows a bar's scale even
//     in a log file). Width math is always visible-width aware, so
//     styled text aligns exactly like its plain form.
package termui

import (
	"os"
	"strings"
)

// Enabled is the styling switch, decided once at startup: a
// character device on stdout and no NO_COLOR in the environment.
// Tests and embedders may set it directly.
var Enabled = computeEnabled()

func computeEnabled() bool {
	if os.Getenv("NO_COLOR") != "" {
		return false
	}
	fi, err := os.Stdout.Stat()
	return err == nil && fi.Mode()&os.ModeCharDevice != 0
}

// fgCode maps color names to ANSI foreground codes. The names match
// the palette in chat/color.go (John's names) so the whole program
// speaks one color vocabulary.
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
	}
	return 0
}

// Style is a text treatment: a named foreground color, an
// optional background color ("" = none), optionally strong (bold)
// or faint (dim). The zero Style is plain text.
type Style struct {
	fg     string
	bg     string
	strong bool
	faint  bool
}

// Fg returns a style in the named foreground color.
func Fg(name string) Style { return Style{fg: name} }

// Strong returns a bold style in the named foreground color.
func Strong(name string) Style { return Style{fg: name, strong: true} }

// Faint returns a dim style in the named foreground color
// ("" keeps the default color).
func Faint(name string) Style { return Style{fg: name, faint: true} }

// Bg returns a style with the named background color (and no
// foreground change) — the base for chips and highlight rows.
func Bg(name string) Style { return Style{bg: name} }

// Apply wraps s in the style's SGR sequence, or returns s untouched
// when styling is disabled or the style is plain.
func (s Style) Apply(v string) string {
	if !Enabled || v == "" || (s.fg == "" && s.bg == "" && !s.strong && !s.faint) {
		return v
	}
	var codes []string
	if s.strong {
		codes = append(codes, "1")
	}
	if s.faint {
		codes = append(codes, "2")
	}
	if s.fg != "" {
		codes = append(codes, itoa(fgCode(s.fg)))
	}
	if s.bg != "" {
		codes = append(codes, itoa(fgCode(s.bg)+10))
	}
	return "\x1b[" + strings.Join(codes, ";") + "m" + v + "\x1b[0m"
}

// Named semantic styles: the palette the components draw from.
var (
	Accent = Fg("cyan")       // links, commands, secondary headers
	Gold   = Strong("yellow") // section marks, ranks, key numbers
	Good   = Fg("green")      // positive direction (fan-in, success)
	Bad    = Fg("red")        // failures, parse errors
	Info   = Fg("blue")       // neutral direction (fan-out)
	Violet = Fg("purple")     // social / companion surfaces
	Bright = Strong("white")  // names the eye should land on
	Muted  = Faint("")        // labels, legends, tracks, asides
)

// Paint colors v with a named foreground — the one-call form.
func Paint(fg, v string) string { return Fg(fg).Apply(v) }

// BoldText emphasizes v in the default color.
func BoldText(v string) string { return Style{strong: true}.Apply(v) }

// DimText mutes v.
func DimText(v string) string { return Muted.Apply(v) }

// BrainStyle gives each of the six chat brains one consistent hue,
// everywhere it appears: reply tags, cast lists, the start guide.
// Other cast members map to their nearest brain.
func BrainStyle(name string) Style {
	return Fg(BrainColorName(name))
}

// BrainColorName is BrainStyle as a palette name, for callers with
// their own coloring helpers (chat's colorize).
func BrainColorName(name string) string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "social", "companion", "friend":
		return "purple"
	case "go":
		return "blue"
	case "gocode":
		return "green"
	case "gocli":
		return "cyan"
	case "makefile", "maker":
		return "yellow"
	case "goanalyze", "analyst", "analyzer", "coordinator":
		return "white"
	}
	return "purple"
}

// The symbol vocabulary: one marker per job, everywhere.
const (
	SectionMark = "◆" // a major section opens here
	SubMark     = "▸" // a sub-section opens here
	Bullet      = "•" // a list item
	Star        = "★" // the top entry / an entry point
	Arrow       = "→" // depends on / calls / leads to
	ArrowBack   = "←" // depended on by / comes from
	RuleChar    = "─" // a horizontal rule segment
	BarChar     = "│" // prose left rule / table separator
	FillChar    = "█" // a bar's filled part
	TrackChar   = "░" // a bar's unfilled track
	BranchMid   = "├──"
	BranchEnd   = "└──"
)

// StripANSI removes SGR sequences from s.
func StripANSI(s string) string {
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

// VisibleWidth is the on-screen cell count of s (runes, escapes
// removed).
func VisibleWidth(s string) int {
	return len([]rune(StripANSI(s)))
}

// PadRight pads s with spaces until its visible width is w.
// Already-wide strings are returned as-is.
func PadRight(s string, w int) string {
	if d := w - VisibleWidth(s); d > 0 {
		return s + strings.Repeat(" ", d)
	}
	return s
}

// PadLeft pads s on the left until its visible width is w.
func PadLeft(s string, w int) string {
	if d := w - VisibleWidth(s); d > 0 {
		return strings.Repeat(" ", d) + s
	}
	return s
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var digs []byte
	for n > 0 {
		digs = append([]byte{byte('0' + n%10)}, digs...)
		n /= 10
	}
	return string(digs)
}
