package analyze

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/pkg/termui"
)

// This file renders the terminal visuals: several charts that show
// a codebase's shape at a glance, rendered through pkg/termui (the
// home-grown UI library: section headers, tracked bars, tables).
// VisualReport bundles them into the one report the chat prints
// for "analyze <repo>" and "show me a visual".

// pkgLabel renders "" (the root dir) as ".".
func pkgLabel(dir string) string {
	if dir == "" {
		return "."
	}
	return dir
}

// PackageSizes draws lines-of-code per package as a horizontal bar chart,
// biggest first. Big bars are where the code — and the complexity — lives.
func (p *Project) PackageSizes() string {
	return p.packageSizes(0) + "\n"
}

// packageSizes renders the sizes panel; width > 0 pins the panel
// width (and shrinks the bars) for side-by-side Row layouts.
func (p *Project) packageSizes(width int) string {
	pkgs := append([]*Package(nil), p.Packages...)
	sort.Slice(pkgs, func(i, j int) bool {
		if pkgs[i].Lines != pkgs[j].Lines {
			return pkgs[i].Lines > pkgs[j].Lines
		}
		return pkgs[i].Dir < pkgs[j].Dir
	})
	items := make([]termui.BarItem, 0, len(pkgs))
	for _, pkg := range pkgs {
		items = append(items, termui.BarItem{Label: pkgLabel(pkg.Dir), Value: pkg.Lines})
	}
	barW := 30
	if width > 0 {
		barW = 14
	}
	chart := termui.BarChart(items, barW, termui.Fg("yellow"), "")
	if width > 0 {
		return termui.PanelW("PACKAGE SIZES (lines of code)", chart, termui.Fg("yellow"), width)
	}
	return termui.Panel("PACKAGE SIZES (lines of code)", chart, termui.Fg("yellow"))
}

// Coupling draws fan-out (project packages this one imports) and fan-in
// (project packages importing this one) per package, sorted by fan-in.
// High fan-in = a hub everybody depends on; high fan-out = a package
// that knows a lot about the rest of the project.
func (p *Project) Coupling() string {
	return p.coupling(0) + "\n"
}

// coupling renders the coupling panel; width > 0 pins the panel
// width (and shrinks the bars) for side-by-side Row layouts.
func (p *Project) coupling(width int) string {
	fanIn := map[string]int{}
	for _, pkg := range p.Packages {
		for _, dep := range pkg.Internal {
			fanIn[dep]++
		}
	}
	pkgs := append([]*Package(nil), p.Packages...)
	sort.Slice(pkgs, func(i, j int) bool {
		fi, fj := fanIn[pkgs[i].Dir], fanIn[pkgs[j].Dir]
		if fi != fj {
			return fi > fj
		}
		return len(pkgs[i].Internal) > len(pkgs[j].Internal)
	})
	maxOut, maxIn := 0, 0
	for _, pkg := range pkgs {
		if len(pkg.Internal) > maxOut {
			maxOut = len(pkg.Internal)
		}
		if fanIn[pkg.Dir] > maxIn {
			maxIn = fanIn[pkg.Dir]
		}
	}
	barW := 12
	var inner strings.Builder
	if width > 0 {
		barW = 5
		fmt.Fprintf(&inner, "%s imports   %s importing it\n",
			termui.Badge("out "+termui.Arrow, termui.Info),
			termui.Badge(termui.ArrowBack+" in", termui.Good))
	} else {
		fmt.Fprintf(&inner, "%s packages it imports   %s packages importing it\n",
			termui.Badge("out "+termui.Arrow, termui.Info),
			termui.Badge(termui.ArrowBack+" in", termui.Good))
	}
	rows := make([][]string, 0, len(pkgs))
	for _, pkg := range pkgs {
		rows = append(rows, []string{
			pkgLabel(pkg.Dir),
			"out " + termui.HBar(len(pkg.Internal), maxOut, barW, termui.Info),
			"in " + termui.HBar(fanIn[pkg.Dir], maxIn, barW, termui.Good),
			fmt.Sprintf("%d", fanIn[pkg.Dir]),
		})
	}
	headers := []string{"package", "out", "in", "imported by"}
	if width > 0 {
		headers = []string{"package", "out", "in", "by"}
	}
	inner.WriteString(termui.Table(headers, rows))
	if width > 0 {
		return termui.PanelW("COUPLING (who depends on whom)", inner.String(), termui.Info, width)
	}
	return termui.Panel("COUPLING (who depends on whom)", inner.String(), termui.Info)
}

// BiggestFiles draws the n largest Go files as a bar chart. Big files
// are the usual suspects when something feels hard to change.
func (p *Project) BiggestFiles(n int) string {
	type fl struct {
		rel   string
		lines int
	}
	var files []fl
	for _, pkg := range p.Packages {
		for _, rel := range pkg.Files {
			data, err := os.ReadFile(filepath.Join(p.Root, rel))
			if err != nil {
				continue
			}
			files = append(files, fl{rel, countLines(data)})
		}
	}
	sort.Slice(files, func(i, j int) bool { return files[i].lines > files[j].lines })
	if len(files) > n {
		files = files[:n]
	}
	max := 0
	for _, f := range files {
		if f.lines > max {
			max = f.lines
		}
	}
	var inner strings.Builder
	for i, f := range files {
		fmt.Fprintf(&inner, "%s %s %s %s\n",
			termui.Muted.Apply(fmt.Sprintf("%d.", i+1)),
			termui.PadRight(termui.StylePath(p.LinkPath(f.rel)), 40),
			termui.HBar(f.lines, max, 24, termui.Violet),
			termui.Strong("").Apply(fmt.Sprintf("%d", f.lines)))
	}
	return termui.Panel(fmt.Sprintf("BIGGEST FILES (top %d)", len(files)),
		inner.String(), termui.Violet) + "\n"
}

func countLines(data []byte) int {
	n := 0
	for _, c := range data {
		if c == '\n' {
			n++
		}
	}
	if len(data) > 0 && data[len(data)-1] != '\n' {
		n++
	}
	return n
}

// VisualReport bundles every terminal visual into one report: package
// sizes, coupling, the engine room (most-called functions), and the
// package dependency graph. This is what the chat prints for
// "analyze <repo>" and for a bare "show me a visual".
func (p *Project) VisualReport() string {
	name := p.Module
	if name == "" {
		name = filepathBase(p.Root)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n%s\n\n", termui.SectionHeader("VISUALS: "+name), termui.Rule(46))
	b.WriteString(p.Pipeline())
	b.WriteString("\n")
	if termui.TerminalWidth() >= 116 {
		// Wide terminal: the dashboard layout — sizes and coupling
		// as side-by-side multibox panels.
		b.WriteString(termui.Row(p.packageSizes(57), p.coupling(57)))
		b.WriteString("\n\n")
	} else {
		b.WriteString(p.PackageSizes())
		b.WriteString("\n")
		b.WriteString(p.Coupling())
		b.WriteString("\n")
	}
	b.WriteString(p.EngineRoom(8))
	b.WriteString("\n")
	b.WriteString(p.BiggestFiles(8))
	b.WriteString("\n")
	b.WriteString(RenderASCIIGraph(p))
	return b.String()
}
