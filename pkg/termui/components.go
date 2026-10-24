package termui

import (
	"fmt"
	"strings"
)

// components.go — the building blocks every long-form surface is
// rendered with. Each takes plain data and returns a string; none
// prints. Multi-line results carry no trailing newline — callers
// own line termination, like the renderers they replace.

// SectionHeader opens a major section: a gold ◆ and the title in
// the section color. In plain mode the glyph remains — it is the
// landmark the eye scans for.
func SectionHeader(title string) string {
	return Gold.Apply(SectionMark + " " + title)
}

// SubHeader opens a sub-section: a cyan ▸, quieter than a section.
func SubHeader(title string) string {
	return Accent.Apply(SubMark + " " + title)
}

// Rule is a dim horizontal line of the given visible width.
func Rule(width int) string {
	if width <= 0 {
		return ""
	}
	return Muted.Apply(strings.Repeat(RuleChar, width))
}

// Badge renders a small label chip: with styling on, a filled
// chip — the label's hue as background, black strong text, the
// hledger-tab look. With styling off it degrades to the plain
// bracketed form "[ text ]" so logs keep a readable marker.
func Badge(text string, st Style) string {
	if Enabled && st.fg != "" {
		return Style{fg: "black", bg: st.fg, strong: true}.Apply(" " + text + " ")
	}
	return st.Apply("[ " + text + " ]")
}

// Highlight wraps a whole row in a background wash of the given
// hue with black strong text — for the ONE row in a panel that
// matters most (the #1 function, a rejected verdict). With
// styling off the row is returned unchanged: the highlight is
// emphasis, never information.
func Highlight(text string, st Style) string {
	if st.fg == "" {
		return text
	}
	return Style{fg: "black", bg: st.fg, strong: true}.Apply(text)
}

// HBar renders one horizontal bar: a fill in the chart's hue over
// a dim track, scaled so value == max fills the width. The fill
// carries an intensity gradient — strong head, plain middle, faint
// tail — so magnitude reads even before the number does. The track
// is drawn in plain mode too — it is what tells the eye how big
// the fill is relative to full.
func HBar(value, max, width int, st Style) string {
	if width <= 0 {
		return ""
	}
	fill := 0
	if value > 0 && max > 0 {
		fill = 1 + (width-1)*value/max
		if fill > width {
			fill = width
		}
	}
	return gradientFill(fill, st) +
		Muted.Apply(strings.Repeat(TrackChar, width-fill))
}

// gradientFill renders n fill cells in st's hue with intensity
// steps: the first third strong, the middle plain, the last third
// faint. With styling off it is simply n fill cells.
func gradientFill(n int, st Style) string {
	if n <= 0 {
		return ""
	}
	head := (n + 2) / 3
	mid := (n - head + 1) / 2
	tail := n - head - mid
	strong := Style{fg: st.fg, strong: true}
	faint := Style{fg: st.fg, faint: true}
	return strong.Apply(strings.Repeat(FillChar, head)) +
		st.Apply(strings.Repeat(FillChar, mid)) +
		faint.Apply(strings.Repeat(FillChar, tail))
}

// ThresholdBar renders a ratio meter (value/max as a fraction)
// whose hue steps with the fraction: green under half, yellow
// under three quarters, red above — the system-monitor convention.
func ThresholdBar(value, max, width int) string {
	frac := 0.0
	if max > 0 {
		frac = float64(value) / float64(max)
	}
	st := Good
	switch {
	case frac >= 0.75:
		st = Bad
	case frac >= 0.5:
		st = Fg("yellow")
	}
	return HBar(value, max, width, st)
}

// BarItem is one row of a BarChart: a label and its value.
type BarItem struct {
	Label string
	Value int
}

// BarChart renders a ranked bar chart: the top row carries a gold
// ★, the rest dim rank numbers; labels are bright, values strong,
// and an optional dim suffix (" callers") trails each value.
// Items must arrive sorted, biggest first.
func BarChart(items []BarItem, width int, st Style, valueSuffix string) string {
	return barChart(items, width, st, valueSuffix, false)
}

// BarChartTop is BarChart with the first row washed in the chart
// hue's highlight — the dashboard "this row matters" treatment.
// With styling off it is exactly BarChart (the ★ still marks it).
func BarChartTop(items []BarItem, width int, st Style, valueSuffix string) string {
	return barChart(items, width, st, valueSuffix, true)
}

func barChart(items []BarItem, width int, st Style, valueSuffix string, highlightTop bool) string {
	if len(items) == 0 {
		return ""
	}
	max, labelW := 0, 0
	for _, it := range items {
		if it.Value > max {
			max = it.Value
		}
		if w := VisibleWidth(it.Label); w > labelW {
			labelW = w
		}
	}
	row := func(i int, it BarItem, plain bool) string {
		mark := fmt.Sprintf("%d.", i+1)
		if i == 0 {
			mark = Star
		}
		label, val, suf := it.Label, itoa(it.Value), valueSuffix
		if !plain {
			if i == 0 {
				mark = Gold.Apply(mark)
			} else {
				mark = Muted.Apply(mark)
			}
			label = Bright.Apply(label)
			val = Strong("").Apply(val)
			suf = Muted.Apply(suf)
		}
		return fmt.Sprintf("  %s %s %s %s%s", mark, PadRight(label, labelW),
			HBar(it.Value, max, width, st), val, suf)
	}
	rows := make([]string, len(items))
	chartW := 0
	for i, it := range items {
		rows[i] = row(i, it, false)
		if w := VisibleWidth(rows[i]); w > chartW {
			chartW = w
		}
	}
	if highlightTop && Enabled && len(rows) > 0 {
		rows[0] = Highlight(PadRight(StripANSI(rows[0]), chartW), st)
	}
	return strings.Join(rows, "\n")
}

// Pair is one key/value statistic.
type Pair struct {
	Key   string
	Value string
}

// KeyValues renders statistics on one line: warm-yellow labels
// (the dashboard field-label color), strong values, separated by
// dim dots — "Packages 25 · Go files 236".
func KeyValues(pairs ...Pair) string {
	parts := make([]string, 0, len(pairs))
	for _, p := range pairs {
		parts = append(parts, Fg("yellow").Apply(p.Key+" ")+Strong("").Apply(p.Value))
	}
	return strings.Join(parts, Muted.Apply(" · "))
}

// RankedItem is one row of a RankedList: a label and an optional
// trailing detail (a description, a bar, a count).
type RankedItem struct {
	Label  string
	Detail string
}

// RankedList renders a numbered list: dim numbers, bright labels,
// details in the default color — for reading orders and rankings.
func RankedList(items []RankedItem) string {
	var b strings.Builder
	for i, it := range items {
		fmt.Fprintf(&b, "  %s %s", Muted.Apply(fmt.Sprintf("%d.", i+1)), Bright.Apply(it.Label))
		if it.Detail != "" {
			b.WriteString("  " + it.Detail)
		}
		b.WriteString("\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// StylePath renders a file path the way the eye wants it: the
// directory dimmed, the file name bright. With styling off it
// returns the path unchanged.
func StylePath(path string) string {
	i := strings.LastIndex(path, "/")
	if i < 0 {
		return Bright.Apply(path)
	}
	return Muted.Apply(path[:i+1]) + Bright.Apply(path[i+1:])
}

// StyleFileLine renders "path:line" with the path styled and the
// line number in gold.
func StyleFileLine(path string, line int) string {
	return StylePath(path) + Muted.Apply(":") + Gold.Apply(itoa(line))
}

// PathList renders file paths as a bulleted list, one per line.
func PathList(paths []string) string {
	var b strings.Builder
	for _, p := range paths {
		fmt.Fprintf(&b, "  %s %s\n", Accent.Apply(Bullet), StylePath(p))
	}
	return strings.TrimRight(b.String(), "\n")
}

// Table renders a simple aligned table: a styled header row, a dim
// rule under it, and rows whose columns line up by visible width
// even when cells carry ANSI styling.
func Table(headers []string, rows [][]string) string {
	cols := len(headers)
	if cols == 0 {
		return ""
	}
	widths := make([]int, cols)
	for i, h := range headers {
		widths[i] = VisibleWidth(h)
	}
	for _, r := range rows {
		for i := 0; i < cols && i < len(r); i++ {
			if w := VisibleWidth(r[i]); w > widths[i] {
				widths[i] = w
			}
		}
	}
	sep := " " + Muted.Apply(BarChar) + " "
	renderRow := func(cells []string, st *Style) string {
		parts := make([]string, cols)
		for i := 0; i < cols; i++ {
			c := ""
			if i < len(cells) {
				c = cells[i]
			}
			if st != nil {
				c = st.Apply(StripANSI(c))
			}
			parts[i] = PadRight(c, widths[i])
		}
		return "  " + strings.Join(parts, sep)
	}
	var b strings.Builder
	head := Accent
	b.WriteString(renderRow(headers, &head))
	b.WriteString("\n")
	total := 2
	for _, w := range widths {
		total += w
	}
	total += 3 * (cols - 1)
	b.WriteString("  " + Rule(total-2))
	for _, r := range rows {
		b.WriteString("\n")
		b.WriteString(renderRow(r, nil))
	}
	return b.String()
}

// Panel sizing: a box wider than MaxPanelWidth sprawls; below
// MinPanelWidth a border does more harm than good and Panel
// degrades to a borderless section header.
const (
	MaxPanelWidth = 100
	MinPanelWidth = 40
)

// WidthOverride pins TerminalWidth when set (> 0) — tests and
// embedders rendering for a known target use it.
var WidthOverride int

// TerminalWidth is the output width in columns: the override, the
// real terminal size, or 80 when unknown.
func TerminalWidth() int {
	if WidthOverride > 0 {
		return WidthOverride
	}
	if w := terminalWidth(); w > 0 {
		return w
	}
	return 80
}

// Panel renders content in a bordered box with the title set into
// the top border, bpytop-style:
//
//	╭─ ENGINE ROOM ─────────────╮
//	│ content, padded one space │
//	╰───────────────────────────╯
//
// The border takes the section hue dimmed, the title the same hue
// strong. Content is padded to its widest line; the box never
// exceeds the terminal width (capped at MaxPanelWidth). When the
// terminal is narrower than MinPanelWidth, or the content or title
// cannot fit the cap, Panel degrades to a borderless section
// header over the raw content rather than wrapping into garbage.
// With an empty content the result is a title-only box — a header
// block. Panels are printed blocks for a scrolling transcript,
// never a screen takeover.
func Panel(title, content string, st Style) string {
	return panel(title, content, st, 0)
}

// PanelW is Panel at a fixed box width — the building block for
// side-by-side Rows. A width the content cannot fit falls back to
// Panel's auto sizing (never truncation).
func PanelW(title, content string, st Style, width int) string {
	return panel(title, content, st, width)
}

func panel(title, content string, st Style, fixedW int) string {
	border := Style{fg: st.fg, faint: true}
	if st.fg == "" {
		border = Muted
	}
	titleStyle := Style{fg: st.fg, strong: true}
	if st.fg == "" {
		titleStyle = Bright
	}
	lines := []string{}
	if content != "" {
		lines = strings.Split(strings.TrimRight(content, "\n"), "\n")
	}
	contentW := 0
	for _, l := range lines {
		if w := VisibleWidth(l); w > contentW {
			contentW = w
		}
	}
	titleW := VisibleWidth(title)
	boxW := contentW + 4 // border + one space of padding per side
	if need := titleW + 6; boxW < need {
		boxW = need // the top border must hold the title
	}
	if fixedW > boxW {
		boxW = fixedW
	}
	cap := TerminalWidth()
	if cap > MaxPanelWidth && fixedW == 0 {
		cap = MaxPanelWidth
	}
	if TerminalWidth() < MinPanelWidth || boxW > cap {
		head := SectionHeader(title)
		if content == "" {
			return head
		}
		return head + "\n" + strings.Join(lines, "\n")
	}
	var b strings.Builder
	fills := boxW - 5 - titleW
	if fills < 1 {
		fills = 1
	}
	b.WriteString(border.Apply("╭─ ") + titleStyle.Apply(title) +
		border.Apply(" "+strings.Repeat("─", fills)+"╮"))
	for _, l := range lines {
		b.WriteString("\n" + border.Apply("│ ") + PadRight(l, boxW-4) + border.Apply(" │"))
	}
	b.WriteString("\n" + border.Apply("╰"+strings.Repeat("─", boxW-2)+"╯"))
	return b.String()
}

// Row lays rendered blocks (usually Panels) side by side with a
// two-space gutter, padding shorter blocks with blank lines so
// the columns share a baseline — the dashboard multibox layout.
// When the combined width exceeds the terminal, the blocks stack
// vertically instead; nothing ever wraps mid-box.
func Row(blocks ...string) string {
	if len(blocks) == 0 {
		return ""
	}
	if len(blocks) == 1 {
		return blocks[0]
	}
	split := make([][]string, len(blocks))
	widths := make([]int, len(blocks))
	total := 2 * (len(blocks) - 1)
	height := 0
	for i, blk := range blocks {
		split[i] = strings.Split(blk, "\n")
		for _, l := range split[i] {
			if w := VisibleWidth(l); w > widths[i] {
				widths[i] = w
			}
		}
		total += widths[i]
		if len(split[i]) > height {
			height = len(split[i])
		}
	}
	if total > TerminalWidth() {
		return strings.Join(blocks, "\n")
	}
	var b strings.Builder
	for r := 0; r < height; r++ {
		for c := range blocks {
			if c > 0 {
				b.WriteString("  ")
			}
			cell := ""
			if r < len(split[c]) {
				cell = split[c][r]
			}
			b.WriteString(PadRight(cell, widths[c]))
		}
		if r < height-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// Prose renders a long text block with a dim left rule on every
// line, so paragraphs read as one quoted unit instead of a wall.
// The prose itself keeps the default color — the rule is the cue.
func Prose(text string) string {
	lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
	for i, l := range lines {
		if l == "" {
			lines[i] = Muted.Apply(BarChar)
		} else {
			lines[i] = Muted.Apply(BarChar+" ") + l
		}
	}
	return strings.Join(lines, "\n")
}
