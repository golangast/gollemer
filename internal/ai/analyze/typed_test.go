package analyze

import (
	"os"
	"path/filepath"
	"testing"
)

// typedFixture: Child embeds base.Base across packages; c.Helper()
// resolves only through the type checker's method set.
func typedFixture(t *testing.T) *Project {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/tm\n\ngo 1.21\n")
	write("base/base.go", `package base

type Base struct{}

func (b *Base) Helper() string { return "help" }
`)
	write("main.go", `package main

import "example.com/tm/base"

type Child struct {
	base.Base
}

func Do(c *Child) string {
	return c.Helper()
}

func main() { print(Do(&Child{})) }
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestTypedEmbeddedMethod(t *testing.T) {
	p := typedFixture(t)
	if !p.Typed {
		t.Fatal("expected type info to load for stdlib-only fixture")
	}
	var do, helper *Func
	for _, fn := range p.byID {
		switch fn.Name {
		case "Do":
			do = fn
		case "Helper":
			helper = fn
		}
	}
	if do == nil || helper == nil {
		t.Fatal("fixture funcs missing")
	}
	found := false
	for _, id := range do.Calls {
		if id == helper.ID {
			found = true
		}
	}
	if !found {
		t.Errorf("Do.Calls = %v, want embedded base.Base.Helper (%s)", do.Calls, helper.ID)
	}
}
