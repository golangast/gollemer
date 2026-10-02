package analyze

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// This file renders the terminal visuals: several ASCII charts that show
// a codebase's shape at a glance. VisualReport bundles them into the one
// report the chat prints for "analyze <repo>" and "show me a visual".

// bars scales v against max into a width-wide block bar.
func bars(v, max, width int) string {
	if max <= 0 || v <= 0 {
		return ""
	}
	n := 1 + (width-1)*v/max
	return strings.Repeat("█", n)
}

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
	pkgs := append([]*Package(nil), p.Packages...)
	sort.Slice(pkgs, func(i, j int) bool {
		if pkgs[i].Lines != pkgs[j].Lines {
			return pkgs[i].Lines > pkgs[j].Lines
		}
		return pkgs[i].Dir < pkgs[j].Dir
	})
	max := 0
	for _, pkg := range pkgs {
		if pkg.Lines > max {
			max = pkg.Lines
		}
	}
	var b strings.Builder
	b.WriteString("PACKAGE SIZES (lines of code)\n")
	for _, pkg := range pkgs {
		fmt.Fprintf(&b, "  %-28s %s %d\n", pkgLabel(pkg.Dir), bars(pkg.Lines, max, 30), pkg.Lines)
	}
	return b.String()
}

// Coupling draws fan-out (project packages this one imports) and fan-in
// (project packages importing this one) per package, sorted by fan-in.
// High fan-in = a hub everybody depends on; high fan-out = a package
// that knows a lot about the rest of the project.
func (p *Project) Coupling() string {
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
	var b strings.Builder
	b.WriteString("COUPLING (who depends on whom)\n")
	b.WriteString("  out = packages it imports · in = packages importing it\n")
	for _, pkg := range pkgs {
		fmt.Fprintf(&b, "  %-28s out %-16s in %s %d\n",
			pkgLabel(pkg.Dir),
			bars(len(pkg.Internal), maxOut, 12),
			bars(fanIn[pkg.Dir], maxIn, 12),
			fanIn[pkg.Dir])
	}
	return b.String()
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
	var b strings.Builder
	fmt.Fprintf(&b, "BIGGEST FILES (top %d)\n", len(files))
	max := 0
	for _, f := range files {
		if f.lines > max {
			max = f.lines
		}
	}
	for _, f := range files {
		fmt.Fprintf(&b, "  %-40s %s %d\n", f.rel, bars(f.lines, max, 24), f.lines)
	}
	return b.String()
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
	fmt.Fprintf(&b, "VISUALS: %s\n%s\n\n", name, strings.Repeat("=", 46))
	b.WriteString(p.Pipeline())
	b.WriteString("\n")
	b.WriteString(p.PackageSizes())
	b.WriteString("\n")
	b.WriteString(p.Coupling())
	b.WriteString("\n")
	b.WriteString(p.EngineRoom(8))
	b.WriteString("\n")
	b.WriteString(p.BiggestFiles(8))
	b.WriteString("\n")
	b.WriteString(RenderASCIIGraph(p))
	return b.String()
}
