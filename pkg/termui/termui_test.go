package termui

import (
	"io"
	"os"
	"strings"
	"testing"
)

func withEnabled(t *testing.T, on bool) {
	t.Helper()
	old := Enabled
	Enabled = on
	t.Cleanup(func() { Enabled = old })
}

// With styling off, components keep their glyphs and layout but
// must not emit a single escape byte — piped and NO_COLOR output
// stays clean.
func TestDisabledHasGlyphsButNoEscapes(t *testing.T) {
	withEnabled(t, false)
	out := SectionHeader("ENGINE ROOM") + "\n" +
		BarChart([]BarItem{{"pkg.Foo", 10}, {"pkg.Bar", 4}}, 10, Accent, " callers") + "\n" +
		KeyValues(Pair{"Packages", "3"}, Pair{"Files", "9"}) + "\n" +
		PathList([]string{"store/store.go"}) + "\n" +
		Table([]string{"package", "in"}, [][]string{{"store", "1"}}) + "\n" +
		Prose("one\n\ntwo")
	if strings.Contains(out, "\x1b[") {
		t.Fatalf("disabled styling leaked escapes:\n%q", out)
	}
	for _, want := range []string{"◆ ENGINE ROOM", "★", "█", "░", "Packages 3", "•", "store.go", "│ one", "package"} {
		if !strings.Contains(out, want) {
			t.Errorf("plain rendering missing %q:\n%s", want, out)
		}
	}
}

// With styling on, Apply wraps and StripANSI/VisibleWidth see
// through it: styled text must align like its plain form.
func TestStyledAlignsLikePlain(t *testing.T) {
	withEnabled(t, true)
	styled := Bright.Apply("store")
	if !strings.Contains(styled, "\x1b[") {
		t.Fatalf("enabled style produced no escapes: %q", styled)
	}
	if StripANSI(styled) != "store" {
		t.Fatalf("StripANSI = %q, want %q", StripANSI(styled), "store")
	}
	if VisibleWidth(styled) != 5 {
		t.Fatalf("VisibleWidth = %d, want 5", VisibleWidth(styled))
	}
	if got := PadRight(styled, 8); VisibleWidth(got) != 8 {
		t.Fatalf("PadRight visible width = %d, want 8 (%q)", VisibleWidth(got), got)
	}
	// StylePath must round-trip to the original path once stripped.
	if got := StripANSI(StylePath("store/store.go")); got != "store/store.go" {
		t.Fatalf("StylePath stripped = %q", got)
	}
	// Table: a styled cell and a plain cell in the same column
	// must start the next column at the same visible offset.
	tbl := Table([]string{"name", "n"}, [][]string{
		{Bright.Apply("longname"), "1"},
		{"x", "2"},
	})
	lines := strings.Split(StripANSI(tbl), "\n")
	if len(lines) != 4 { // header, rule, two rows
		t.Fatalf("table lines = %d:\n%s", len(lines), tbl)
	}
	r1, r2 := lines[2], lines[3]
	if strings.Index(r1, "│") != strings.Index(r2, "│") {
		t.Errorf("table columns misaligned:\n%s\n%s", r1, r2)
	}
}

// Every brain owns a distinct hue: the styled forms of the six
// brain names must all differ.
func TestBrainStylesDistinct(t *testing.T) {
	withEnabled(t, true)
	seen := map[string]string{}
	for _, b := range []string{"social", "go", "gocode", "gocli", "makefile", "goanalyze"} {
		got := BrainStyle(b).Apply("x")
		if prev, dup := seen[got]; dup {
			t.Errorf("brains %s and %s share a style %q", prev, b, got)
		}
		seen[got] = b
	}
}

// HBar: full value fills the track, zero value is all track, and
// the visible width is always exactly the requested width.
func TestHBarShape(t *testing.T) {
	withEnabled(t, false)
	for _, tc := range []struct {
		v, max, w int
		fills     int
	}{
		{10, 10, 12, 12},
		{0, 10, 12, 0},
		{5, 10, 12, 6},
		{3, 0, 12, 0},
	} {
		bar := HBar(tc.v, tc.max, tc.w, Accent)
		if got := VisibleWidth(bar); got != tc.w {
			t.Errorf("HBar(%d,%d,%d) width = %d", tc.v, tc.max, tc.w, got)
		}
		if got := strings.Count(bar, FillChar); got != tc.fills {
			t.Errorf("HBar(%d,%d,%d) fills = %d, want %d (%q)", tc.v, tc.max, tc.w, got, tc.fills, bar)
		}
	}
}

// Panel: title in the top border, every line the same visible
// width, content padded inside.
func TestPanelShape(t *testing.T) {
	withEnabled(t, false)
	old := WidthOverride
	WidthOverride = 80
	defer func() { WidthOverride = old }()

	out := Panel("ENGINE ROOM", "one\nlonger line", Accent)
	lines := strings.Split(out, "\n")
	if len(lines) != 4 {
		t.Fatalf("panel lines = %d:\n%s", len(lines), out)
	}
	if !strings.HasPrefix(lines[0], "╭─ ENGINE ROOM ─") || !strings.HasSuffix(lines[0], "╮") {
		t.Errorf("top border wrong: %q", lines[0])
	}
	if !strings.HasPrefix(lines[3], "╰") || !strings.HasSuffix(lines[3], "╯") {
		t.Errorf("bottom border wrong: %q", lines[3])
	}
	w := VisibleWidth(lines[0])
	for _, l := range lines {
		if VisibleWidth(l) != w {
			t.Errorf("panel line width %d, want %d: %q", VisibleWidth(l), w, l)
		}
	}
	if !strings.Contains(lines[1], "│ one") || !strings.Contains(lines[2], "│ longer line") {
		t.Errorf("panel content wrong:\n%s", out)
	}
	// A title-only panel is a clean two-border header block.
	head := Panel("Guided stop 1 of 6", "", Accent)
	hl := strings.Split(head, "\n")
	if len(hl) != 2 || VisibleWidth(hl[0]) != VisibleWidth(hl[1]) {
		t.Errorf("title-only panel misaligned:\n%s", head)
	}
}

// Below MinPanelWidth the panel degrades to a borderless header —
// no border glyphs, content untouched.
func TestPanelNarrowDegrades(t *testing.T) {
	withEnabled(t, false)
	old := WidthOverride
	WidthOverride = 30
	defer func() { WidthOverride = old }()

	out := Panel("ENGINE ROOM", "body", Accent)
	if strings.Contains(out, "╭") || strings.Contains(out, "│") {
		t.Errorf("narrow panel kept its border:\n%s", out)
	}
	if !strings.Contains(out, "◆ ENGINE ROOM") || !strings.Contains(out, "body") {
		t.Errorf("narrow panel lost content:\n%s", out)
	}
}

// Row places two panels side by side when they fit and stacks
// them when they don't.
func TestRowSideBySideAndStacked(t *testing.T) {
	withEnabled(t, false)
	old := WidthOverride
	defer func() { WidthOverride = old }()

	a := Panel("A", "x", Accent)
	b := Panel("B", "yy", Info)
	WidthOverride = 80
	side := Row(a, b)
	lines := strings.Split(side, "\n")
	if len(lines) != 3 {
		t.Fatalf("side-by-side row lines = %d:\n%s", len(lines), side)
	}
	if !strings.Contains(lines[0], "╭─ A ─") || !strings.Contains(lines[0], "╭─ B ─") {
		t.Errorf("row top line lacks both panel titles: %q", lines[0])
	}
	WidthOverride = 10 // narrower than the two panels side by side
	if got := Row(a, b); got != a+"\n"+b {
		t.Errorf("narrow row should stack:\n%s", got)
	}
}

// Badge is a filled chip when styled, a bracketed label when not.
func TestBadgeChip(t *testing.T) {
	withEnabled(t, false)
	if got := Badge("out →", Info); got != "[ out → ]" {
		t.Errorf("plain badge = %q", got)
	}
	withEnabled(t, true)
	got := Badge("out →", Info)
	if !strings.Contains(got, "44m") || !strings.Contains(got, "out →") {
		t.Errorf("styled badge lacks blue bg chip: %q", got)
	}
}

// BarChartTop highlights exactly the first row, and only when
// styling is on.
func TestBarChartTopHighlight(t *testing.T) {
	items := []BarItem{{Label: "Big", Value: 10}, {Label: "Small", Value: 2}}
	withEnabled(t, false)
	if got := BarChartTop(items, 10, Accent, ""); strings.Contains(got, "\x1b[") {
		t.Errorf("plain BarChartTop has escapes: %q", got)
	}
	withEnabled(t, true)
	got := BarChartTop(items, 10, Accent, "")
	lines := strings.Split(got, "\n")
	if !strings.Contains(lines[0], "46m") { // cyan background
		t.Errorf("top row not highlighted: %q", lines[0])
	}
	if strings.Contains(lines[1], "46m") {
		t.Errorf("second row must not be highlighted: %q", lines[1])
	}
}

// Off a terminal the spinner is exactly one line per phase —
// no carriage returns ever reach a pipe.
func TestSpinnerPlainMode(t *testing.T) {
	withEnabled(t, false)
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	sp := Spin("Reading…")
	sp.Stop("Ready.")
	w.Close()
	os.Stdout = old
	out, _ := io.ReadAll(r)
	if strings.Contains(string(out), "\r") {
		t.Errorf("plain spinner emitted carriage returns: %q", out)
	}
	if string(out) != "Reading…\nReady.\n" {
		t.Errorf("plain spinner lines = %q", out)
	}
}

// ThresholdBar picks its hue by fraction: green low, red high.
func TestThresholdBarHues(t *testing.T) {
	withEnabled(t, true)
	if got := ThresholdBar(2, 10, 10); !strings.Contains(got, "\x1b[32m") && !strings.Contains(got, "\x1b[1;32m") {
		t.Errorf("low threshold bar not green: %q", got)
	}
	if got := ThresholdBar(9, 10, 10); !strings.Contains(got, "31m") {
		t.Errorf("high threshold bar not red: %q", got)
	}
}

// Prose keeps the words untouched and rules every line.
func TestProseRules(t *testing.T) {
	withEnabled(t, false)
	out := Prose("alpha\nbeta")
	if StripANSI(out) != "│ alpha\n│ beta" {
		t.Fatalf("Prose = %q", StripANSI(out))
	}
}
