package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// guideFixture builds a tiny project with dispatch shapes: an if-chain
// router and a switch dispatcher, so GuideChange has patterns to find.
func guideFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/guide\n\ngo 1.21\n")
	write("main.go", `package main

// main is the entry point.
func main() {
	run("/flow")
}

func run(cmd string) {
	route(cmd)
}
`)
	write("router.go", `package main

// route sends a command to its handler.
func route(cmd string) {
	if cmd == "/flow" {
		handleFlow(cmd)
	} else if cmd == "/analyze" {
		handleAnalyze(cmd)
	} else if cmd == "/help" {
		handleHelp(cmd)
	}
}

func handleFlow(cmd string)    {}
func handleAnalyze(cmd string) {}
func handleHelp(cmd string)    {}
`)
	write("dispatch.go", `package main

// dispatch picks a source by kind.
func dispatch(kind string) string {
	switch kind {
	case "yaml":
		return "yaml-src"
	case "file":
		return "file-src"
	case "http":
		return "http-src"
	}
	return ""
}
`)
	return root
}

// The guided answer names one anchor, the pattern to imitate, and the
// call chain the change sits in.
func TestGuideChange(t *testing.T) {
	p, err := Analyze(guideFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out := p.GuideChange("add a new chat command")
	for _, want := range []string{
		"TO ADD",
		"START HERE:",
		"router.go",
		"YOU ARE HERE:",
		"route",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("GuideChange missing %q\n%s", want, out)
		}
	}
	if !strings.Contains(out, "PUT IT HERE:") {
		t.Errorf("GuideChange should find the if-chain dispatch site\n%s", out)
	}
}

// The dispatch scanner finds switch and if-chain extension points.
func TestFindDispatchSites(t *testing.T) {
	p, err := Analyze(guideFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	fn, _, _, _ := p.resolve("route")
	sites := findDispatchSites(p.Root, fn)
	if len(sites) == 0 {
		t.Fatal("no dispatch sites found in route")
	}
	if sites[0].kind != "chain" {
		t.Errorf("route dispatch kind = %q, want chain", sites[0].kind)
	}
	if len(sites[0].examples) < 2 {
		t.Errorf("route examples = %v, want the command branches", sites[0].examples)
	}

	sfn, _, _, _ := p.resolve("dispatch")
	ssites := findDispatchSites(p.Root, sfn)
	if len(ssites) == 0 || ssites[0].kind != "switch" {
		t.Errorf("dispatch sites = %+v, want a switch", ssites)
	}
}

// The caller chain walks up to main.
func TestCallerChain(t *testing.T) {
	p, err := Analyze(guideFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	fn, _, _, _ := p.resolve("route")
	chain := callerChain(p, fn, 5)
	if len(chain) < 3 {
		t.Fatalf("chain = %v, want main → run → route", chain)
	}
	if chain[0].Name != "main" {
		t.Errorf("chain starts at %q, want main", chain[0].Name)
	}
	if chain[len(chain)-1].Name != "route" {
		t.Errorf("chain ends at %q, want route", chain[len(chain)-1].Name)
	}
}

// No function anchor: falls back to the ranked list, never empty words.
func TestGuideChangeFallback(t *testing.T) {
	p, err := Analyze(guideFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out := p.GuideChange("zzzznothing")
	if !strings.Contains(out, "No strong matches") {
		t.Errorf("fallback = %q, want the no-matches message", out)
	}
}
