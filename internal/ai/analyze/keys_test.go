package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// keysFixture is a tiny TUI app with key-dispatch switches: one at the
// app level, one in a view, plus a non-key switch that must not match.
func keysFixture(t *testing.T) string {
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
	write("go.mod", "module example.com/keys\n\ngo 1.21\n")
	write("main.go", "package main\n\nfunc main() { run() }\n")
	write("tui/app.go", `package tui

// App is the root model.
type App struct{}

// Update handles messages.
func (a *App) Update(msg string) string {
	switch msg {
	case "ctrl+c", "q":
		return "quit"
	case "esc":
		return "back"
	case "enter":
		return "select"
	}
	return ""
}
`)
	write("tui/views/clean.go", `package views

// CleanModel shows the file list.
type CleanModel struct{}

// Update handles view messages.
func (m *CleanModel) Update(msg string) string {
	switch msg {
	case "up", "down":
		return "move"
	case " ":
		return "toggle"
	}
	return ""
}
`)
	write("tui/menu.go", `package tui

// Menu picks an item.
func Menu(choice int) string {
	switch choice {
	case 1:
		return "clean"
	case 2:
		return "rules"
	}
	return ""
}
`)
	return root
}

// Key-dispatch switches are found; the int switch is not.
func TestFindKeyDispatchSites(t *testing.T) {
	p, err := Analyze(keysFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	sites := p.FindKeyDispatchSites()
	if len(sites) != 2 {
		t.Fatalf("sites = %d, want 2: %+v", len(sites), sites)
	}
	for _, s := range sites {
		if s.Func == nil {
			t.Errorf("site %s:%d has no func", s.File, s.Line)
		}
	}
	if !strings.Contains(sites[0].File, "app.go") {
		t.Errorf("first site = %s, want the app switch", sites[0].File)
	}
	has := func(ss []string, w string) bool {
		for _, s := range ss {
			if s == w {
				return true
			}
		}
		return false
	}
	if !has(sites[0].Keys, "ctrl+c") || !has(sites[0].Keys, "esc") {
		t.Errorf("app keys = %v", sites[0].Keys)
	}
}

// wantsKeys fires on keyboard/shortcut/hotkey language.
func TestWantsKeys(t *testing.T) {
	yes := ExtractIssueConcepts("GUI: Add keyboard shortcuts\n\nImplement basic hotkeys.")
	if !wantsKeys(yes) {
		t.Error("wantsKeys = false for a shortcuts issue")
	}
	no := ExtractIssueConcepts("Feature Request: retry+backoff\n\nRetry failed http fetches.")
	if wantsKeys(no) {
		t.Error("wantsKeys = true for a retry issue")
	}
}

// `Ctrl+A` becomes the "ctrl+a" case literal.
func TestKeyLiterals(t *testing.T) {
	c := ExtractIssueConcepts("Add hotkeys:\n\n- `Ctrl+A` select all\n- `Del` delete\n- `Esc` clear")
	lits := keyLiterals(c)
	for _, want := range []string{"ctrl+a", "del", "esc"} {
		found := false
		for _, l := range lits {
			if l == want {
				found = true
			}
		}
		if !found {
			t.Errorf("lits = %v, want %q", lits, want)
		}
	}
}

// The full guided answer for a shortcuts issue names the dispatch
// switches and the hotkeys to add.
func TestGuideIssueKeys(t *testing.T) {
	p, err := Analyze(keysFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	c := ExtractIssueConcepts("GUI: Add keyboard shortcuts\n\nImplement basic hotkeys.\n\n- `Ctrl+A` select all\n- `Del` delete")
	out := p.GuideIssue(c)
	for _, want := range []string{
		"TO ADD",
		"BEHAVIOR — add the hotkeys here:",
		"app.go",
		"ctrl+c",
		"Add `ctrl+a`, `del` as new cases",
		"YOU ARE HERE:",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("GuideIssue missing %q\n%s", want, out)
		}
	}
	t.Logf("\n%s", out)
}
