package analyze

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// framekitFixture builds a small framework-shaped project on disk:
//
//   - ui.App / ui.Window / ui.Config / ui.Page — the public face:
//     constructed and used by main, a few steps from the front door.
//   - internal/build.Builder — internal assembly machinery with
//     MORE methods than any public type, so the old method-count
//     ordering promotes it over the public face.
//   - internal/logx.Logger — a seam interface whose doc echoes the
//     screenshot's circular sentence, implemented only by adapter
//     subpackages (stdlog, syslogx).
//   - internal/menu.MenuAdapter — a platform-adapter interface with
//     a dotted-identifier doc line, implemented in a subpackage,
//     touched only by the internal builder.
//
// The split is deliberate: every fact the beginner ranking uses
// (entry-point distance, who references a type, package location,
// implementer layout) separates the two classes on structure alone.
func framekitFixture(t *testing.T) *Project {
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
	write("go.mod", "module example.com/framekit\n\ngo 1.21\n")
	write("main.go", `package main

import "example.com/framekit/ui"

func main() {
	cfg := ui.Config{Title: "Demo", Width: 800}
	app := ui.NewApp(cfg)
	home := ui.NewWindow("Home")
	home.Add(ui.NewPage("Start"))
	app.Show(home)
	app.Run()
}
`)
	write("ui/ui.go", `// Package ui is the public face of the framekit framework.
package ui

import (
	"example.com/framekit/internal/build"
	"example.com/framekit/internal/logx"
)

// Config holds the choices a user makes when starting an
// application: its window title and how wide it opens.
type Config struct {
	Title string
	Width int
}

// App is the whole application: it owns the windows and runs
// the event loop until the user quits.
type App struct {
	cfg     Config
	windows []*Window
	log     logx.Logger
}

// NewApp builds an application from its config.
func NewApp(cfg Config) *App {
	b := build.NewBuilder()
	b.WithLogger(logx.NewStd())
	_ = b
	return &App{cfg: cfg}
}

// Show puts a window on the screen.
func (a *App) Show(w *Window) { a.windows = append(a.windows, w) }

// Run starts the event loop.
func (a *App) Run() int { return 0 }

// Title returns the configured title.
func (a *App) Title() string { return a.cfg.Title }

// Window is a single window the user sees and clicks in.
type Window struct {
	Title string
	pages []*Page
}

// NewWindow makes a window with the given title.
func NewWindow(title string) *Window { return &Window{Title: title} }

// Add puts a page inside the window.
func (w *Window) Add(p *Page) { w.pages = append(w.pages, p) }

// Page is one screen of content inside a window.
type Page struct {
	Name string
	Body string
}

// NewPage makes a page with the given name.
func NewPage(name string) *Page { return &Page{Name: name} }

// Render draws the page's content.
func (p *Page) Render() string { return p.Body }
`)
	write("internal/build/build.go", `// Package build assembles an application before it runs.
package build

import (
	"example.com/framekit/internal/logx"
	"example.com/framekit/internal/menu"
)

// Builder defines a builder that can build framekit applications.
type Builder struct {
	steps int
	log   logx.Logger
	menu  menu.MenuAdapter
}

// NewBuilder starts an empty build.
func NewBuilder() *Builder { return &Builder{} }

// WithLogger sets the build's logger.
func (b *Builder) WithLogger(l logx.Logger) *Builder { b.log = l; return b }

// UseMenu sets the build's menu adapter.
func (b *Builder) UseMenu(m menu.MenuAdapter) { b.menu = m }

// AddFonts registers the fonts step.
func (b *Builder) AddFonts() { b.steps++ }

// AddIcons registers the icons step.
func (b *Builder) AddIcons() { b.steps++ }

// AddStyles registers the styles step.
func (b *Builder) AddStyles() { b.steps++ }

// AddRoutes registers the routes step.
func (b *Builder) AddRoutes() { b.steps++ }

// AddAssets registers the assets step.
func (b *Builder) AddAssets() { b.steps++ }

// Freeze ends the build.
func (b *Builder) Freeze() int { return b.steps }
`)
	write("internal/logx/logx.go", `// Package logx is the framework's logging seam.
package logx

import "example.com/framekit/internal/logx/stdlog"

// Logger defines what a user can do with a logger.
type Logger interface {
	Log(msg string)
	SetLevel(level int)
}

// NewStd returns the standard logger.
func NewStd() Logger { return stdlog.StdLogger{} }
`)
	write("internal/logx/stdlog/stdlog.go", `// Package stdlog adapts the standard library logger.
package stdlog

// StdLogger writes log lines to standard error.
type StdLogger struct{}

// Log writes one line.
func (StdLogger) Log(msg string) {}

// SetLevel sets the minimum level.
func (StdLogger) SetLevel(level int) {}
`)
	write("internal/logx/syslogx/syslogx.go", `// Package syslogx adapts the system log.
package syslogx

// SysLogger writes log lines to the system log.
type SysLogger struct{}

// Log writes one line.
func (SysLogger) Log(msg string) {}

// SetLevel sets the minimum level.
func (SysLogger) SetLevel(level int) {}
`)
	write("internal/menu/menu.go", `// Package menu is the framework's menu seam.
package menu

// MenuAdapter is the com.canonical.dbusmenu interface.
type MenuAdapter interface {
	BuildMenu() error
	SetItem(id string, label string)
}
`)
	write("internal/menu/gtk/gtk.go", `// Package gtk adapts the platform menu.
package gtk

// GtkMenu is the platform menu implementation.
type GtkMenu struct{}

// BuildMenu builds the menu.
func (GtkMenu) BuildMenu() error { return nil }

// SetItem sets one item.
func (GtkMenu) SetItem(id string, label string) {}
`)
	p, err := Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestFramekitCapture prints the concept screens for the report's
// before/after evidence. It asserts the fixture analyzed sanely;
// the ranking assertions live in the beginner-screen tests.
func TestFramekitCapture(t *testing.T) {
	p := framekitFixture(t)
	var names []string
	for _, c := range p.LearnPath() {
		names = append(names, c.Name)
	}
	if len(names) < 7 {
		t.Fatalf("fixture should yield 7 concepts, got %v", names)
	}
	fmt.Println("=== LearnPath order (old machinery):", strings.Join(names, ", "))
	fmt.Println("=== LearnPathText first screen:")
	fmt.Println(firstNLines(p.LearnPathText(), 14))
	fmt.Println("=== BeginnerConcepts first screen (new machinery):")
	for i, c := range p.BeginnerConcepts(4) {
		fmt.Printf("%d. %s — %s\n", i+1, c.Name, c.OneLiner)
	}
	fmt.Println("=== BeginnerConcepts full order:")
	var all []string
	for _, c := range p.BeginnerConcepts(0) {
		all = append(all, c.Name)
	}
	fmt.Println(strings.Join(all, ", "))
}

// The fixture's first screen must be the user-facing types —
// and ONLY them: every piece of plumbing (the builder, the
// logger seam and its two implementations, the platform-adapter
// seam and its implementation) ranks below the screen.
func TestBeginnerScreenFixture(t *testing.T) {
	p := framekitFixture(t)
	screen := p.BeginnerConcepts(4)
	if len(screen) != 4 {
		t.Fatalf("screen has %d concepts, want 4", len(screen))
	}
	want := map[string]bool{"App": true, "Window": true, "Config": true, "Page": true}
	for _, c := range screen {
		if !want[c.Name] {
			t.Errorf("plumbing on the first screen: %s — %s", c.Name, c.OneLiner)
		}
		delete(want, c.Name)
	}
	if len(want) > 0 {
		t.Errorf("user-facing types missing from the first screen: %v", want)
	}
	if screen[0].Name != "App" && screen[0].Name != "Window" && screen[0].Name != "Config" {
		t.Errorf("first concept is %s, want a user-facing type", screen[0].Name)
	}
	// No circular doc echo anywhere in the full beginner order.
	for _, c := range p.BeginnerConcepts(0) {
		for _, bad := range []string{"defines a builder", "defines what a user can do", "com.canonical.dbusmenu"} {
			if strings.Contains(c.OneLiner, bad) {
				t.Errorf("%s still carries the circular line %q: %s", c.Name, bad, c.OneLiner)
			}
		}
	}
}

// The replaced descriptions are built from facts.
func TestBeginnerDescriptionsFromFacts(t *testing.T) {
	p := framekitFixture(t)
	byName := map[string]LearnConcept{}
	for _, c := range p.BeginnerConcepts(0) {
		byName[c.Name] = c
	}
	if got := byName["Builder"].OneLiner; !strings.Contains(got, "struct holding") ||
		!strings.Contains(got, "steps") || !strings.Contains(got, "ui.NewApp") {
		t.Errorf("Builder description not fact-built: %q", got)
	}
	if got := byName["Logger"].OneLiner; !strings.Contains(got, "can Log") ||
		!strings.Contains(got, "SetLevel") {
		t.Errorf("Logger description not fact-built: %q", got)
	}
	if got := byName["MenuAdapter"].OneLiner; !strings.Contains(got, "BuildMenu") {
		t.Errorf("MenuAdapter description not fact-built: %q", got)
	}
	if got := byName["Config"].OneLiner; !strings.Contains(got, "choices a user makes") {
		t.Errorf("Config's good doc was replaced: %q", got)
	}
}

// Circular detection: the screenshot sentences must flag, real
// explanations must not.
func TestLowInformation(t *testing.T) {
	ambient := []string{"framekit", "wails"}
	circular := []struct{ sentence, name string }{
		{"Builder defines a builder that can build framekit applications.", "Builder"},
		{"Logger defines what a user can do with a logger.", "Logger"},
		{"MenuAdapter is the com.canonical.dbusmenu interface.", "MenuAdapter"},
		{"Builder defines a builder that can build Wails applications.", "Builder"},
		{"CustomLogger is a custom logger.", "CustomLogger"},
	}
	for _, c := range circular {
		if !lowInformation(c.sentence, c.name, ambient) {
			t.Errorf("not flagged circular: %q (%s)", c.sentence, c.name)
		}
	}
	informative := []struct{ sentence, name string }{
		{"Container holds everything.", "Container"},
		{"App is the whole application: it owns the windows and runs the event loop until the user quits.", "App"},
		{"Window is a single window the user sees and clicks in.", "Window"},
		{"Page is one screen of content inside a window.", "Page"},
		{"Config holds the choices a user makes when starting an application.", "Config"},
	}
	for _, c := range informative {
		if lowInformation(c.sentence, c.name, ambient) {
			t.Errorf("wrongly flagged circular: %q (%s)", c.sentence, c.name)
		}
	}
}

// When no facts support a description, the fallback is the
// type's kind and home — never an invented sentence.
func TestStructuralDescNoFacts(t *testing.T) {
	p := &Project{}
	bare := &Type{Name: "Marker", Kind: "struct", Pkg: "ui", PkgDir: "ui"}
	if got := p.structuralDesc(bare, nil); got != "a struct in package ui" {
		t.Errorf("no-facts struct = %q", got)
	}
	iface := &Type{Name: "Empty", Kind: "interface", Pkg: "ui", PkgDir: "ui"}
	if got := p.structuralDesc(iface, nil); got != "an interface in package ui" {
		t.Errorf("no-facts interface = %q", got)
	}
	named := &Type{Name: "Level", Kind: "int", Pkg: "ui", PkgDir: "ui"}
	if got := p.structuralDesc(named, nil); got != "a int type in package ui" {
		t.Errorf("no-facts named type = %q", got)
	}
}

func firstNLines(s string, n int) string {
	lines := strings.Split(s, "\n")
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.Join(lines, "\n")
}
