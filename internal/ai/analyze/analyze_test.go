package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fixture writes a tiny two-package project and returns its root.
func fixture(t *testing.T) string {
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
	write("go.mod", "module example.com/demo\n\ngo 1.21\n")
	write("main.go", `package main

import "example.com/demo/store"

// main is the entry point.
func main() {
	s := store.New()
	s.Save("hello")
}
`)
	write("store/store.go", `package store

// Item is the stored thing.
type Item struct{ Name string }

// Store keeps items.
type Store struct{ items []Item }

// New makes a store.
func New() *Store { return &Store{} }

// Save stores an item.
func (s *Store) Save(name string) { s.items = append(s.items, Item{name}) }

// Count returns the item count.
func (s *Store) Count() int { return len(s.items) }
`)
	write("store/broken.go", `package store

this is not valid go (
`)
	return root
}

func TestAnalyzeFixture(t *testing.T) {
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Packages) != 2 {
		t.Fatalf("want 2 packages, got %d", len(p.Packages))
	}
	if p.Module != "example.com/demo" {
		t.Fatalf("want module example.com/demo, got %q", p.Module)
	}
	if len(p.ParseErr) != 1 {
		t.Fatalf("want 1 parse error, got %v", p.ParseErr)
	}
	mains := p.EntryPoints()
	if len(mains) != 1 || mains[0].Name != "main" {
		t.Fatalf("want 1 entry point, got %v", mains)
	}
	// store.New should have a caller (main.main); Save too.
	byID := map[string]*Func{}
	for _, fn := range p.byID {
		byID[fn.ID] = fn
	}
	newFn := byID["store.New"]
	if newFn == nil {
		t.Fatal("store.New not found")
	}
	if len(newFn.Callers) != 1 {
		t.Fatalf("store.New want 1 caller, got %v", newFn.Callers)
	}
	saveFn := byID["store.Store.Save"]
	if saveFn == nil {
		t.Fatal("store.Store.Save not found")
	}
	if len(saveFn.Callers) != 1 {
		t.Fatalf("Store.Save want 1 caller, got %v", saveFn.Callers)
	}
	// New is called, so it should outrank the never-called Count.
	countFn := byID["store.Store.Count"]
	if countFn == nil {
		t.Fatal("store.Store.Count not found")
	}
	if newFn.Score <= countFn.Score {
		t.Fatalf("called func should outrank uncalled: New=%d Count=%d", newFn.Score, countFn.Score)
	}
	// Import graph: root imports store.
	var rootPkg *Package
	for _, pkg := range p.Packages {
		if pkg.Dir == "" {
			rootPkg = pkg
		}
	}
	if rootPkg == nil || len(rootPkg.Internal) != 1 || rootPkg.Internal[0] != "store" {
		t.Fatalf("root internal imports: %+v", rootPkg)
	}
}

func TestWhereToChange(t *testing.T) {
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	hits := p.WhereToChange("add a method to save items")
	if len(hits) == 0 {
		t.Fatal("no hits")
	}
	found := false
	for _, h := range hits {
		if strings.Contains(h.What, "Save") {
			found = true
		}
	}
	if !found {
		t.Fatalf("want a Save hit, got %+v", hits)
	}
}

func TestAnalyzeNotADir(t *testing.T) {
	if _, err := Analyze(filepath.Join(t.TempDir(), "nope")); err == nil {
		t.Fatal("want error for missing dir")
	}
	if _, err := Analyze(""); err == nil {
		t.Fatal("want error for empty dir")
	}
}

func TestRenderASCIIGraphFixture(t *testing.T) {
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out := RenderASCIIGraph(p)
	for _, want := range []string{"Dependency graph", "A → B means A imports B", "📦"} {
		if !strings.Contains(out, want) {
			t.Fatalf("ascii graph missing %q\n---\n%s", want, out)
		}
	}
}
