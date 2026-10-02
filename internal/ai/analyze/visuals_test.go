package analyze

import (
	"strings"
	"testing"
)

// The visual report bundles every chart: sizes, coupling, engine room,
// biggest files, and the dependency graph.
func TestVisualReportSections(t *testing.T) {
	root := fixture(t)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	out := p.VisualReport()
	for _, want := range []string{
		"VISUALS:",
		"PACKAGE SIZES",
		"COUPLING",
		"ENGINE ROOM",
		"BIGGEST FILES",
		"Dependency graph:",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("VisualReport missing section %q", want)
		}
	}
}

// Package sizes: both fixture packages appear with their line counts.
func TestPackageSizes(t *testing.T) {
	root := fixture(t)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	out := p.PackageSizes()
	if !strings.Contains(out, "store") {
		t.Errorf("PackageSizes missing store package:\n%s", out)
	}
	if !strings.Contains(out, "█") {
		t.Errorf("PackageSizes has no bars:\n%s", out)
	}
	// The store package has more lines than the tiny main package.
	storeLines, rootLines := -1, -1
	for _, pkg := range p.Packages {
		if pkg.Dir == "store" {
			storeLines = pkg.Lines
		}
		if pkg.Dir == "" {
			rootLines = pkg.Lines
		}
	}
	if storeLines <= 0 || rootLines <= 0 {
		t.Fatalf("per-package lines not recorded: store=%d root=%d", storeLines, rootLines)
	}
	if storeLines <= rootLines {
		t.Errorf("store lines (%d) should exceed root lines (%d)", storeLines, rootLines)
	}
}

// Coupling: the fixture's main package imports store, so store has
// fan-in 1 and the root package has fan-out 1.
func TestCoupling(t *testing.T) {
	root := fixture(t)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	out := p.Coupling()
	if !strings.Contains(out, "in") || !strings.Contains(out, "out") {
		t.Errorf("Coupling missing fan-in/fan-out legend:\n%s", out)
	}
	lines := strings.Split(out, "\n")
	found := false
	for _, l := range lines {
		if strings.Contains(l, "store") && strings.Contains(l, " in ") {
			found = true
			// fan-in count for store should be 1 (imported by main).
			if !strings.HasSuffix(strings.TrimSpace(l), "1") {
				t.Errorf("store fan-in should be 1: %q", l)
			}
		}
	}
	if !found {
		t.Errorf("Coupling has no row for store:\n%s", out)
	}
}

// Biggest files lists the fixture's Go files.
func TestBiggestFiles(t *testing.T) {
	root := fixture(t)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	out := p.BiggestFiles(8)
	if !strings.Contains(out, "store/store.go") || !strings.Contains(out, "main.go") {
		t.Errorf("BiggestFiles missing fixture files:\n%s", out)
	}
}
