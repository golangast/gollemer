package chat

import (
	"strings"
	"testing"
)

// The use shape's chat face: the sentence is claimed beside
// the delegate's, the target and symbol resolve against the
// project on disk, and the gated flow lands the assembled
// function (or the plan's one retry) with the same guarantees
// as every assembled edit.

// usesChatFixture builds the classic two-package project:
// store/ holds the symbols, app/ (package main) is the usual
// target.
func usesChatFixture(t *testing.T, extra map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/uchat\n\ngo 1.21\n",
		"store/store.go": `package store

import "errors"

type Item struct {
	ID   int
	Name string
}

func Load(id int) (Item, error) {
	if id < 0 {
		return Item{}, errors.New("bad id")
	}
	return Item{ID: id}, nil
}

func load(id int) Item { return Item{ID: id} }
`,
		"app/main.go": "package main\n\nfunc main() {}\n",
	}
	for rel, src := range extra {
		files[rel] = src
	}
	for rel, src := range files {
		writeFile(t, dir, rel, src)
	}
	return dir
}

func TestParseUsesForms(t *testing.T) {
	claims := []struct {
		line, file, symbol, from, name string
		forceType                      bool
	}{
		{"add a function in web.go that uses LoadUser from store.go", "web.go", "LoadUser", "store.go", "", false},
		{"add a function Report in web.go that uses Fetch", "web.go", "Fetch", "", "Report", false},
		{"in web.go add a function that uses LoadUser from store.go", "web.go", "LoadUser", "store.go", "", false},
		{"in web.go, add a function named Report that uses the type Item from store.go", "web.go", "Item", "store.go", "Report", true},
		{"add a function in web.go that uses the type Item", "web.go", "Item", "", "", true},
		{"add a function in web.go that uses store.Load", "web.go", "store.Load", "", "", false},
	}
	for _, c := range claims {
		file, symbol, from, name, forceType, ok := parseUses(c.line)
		if !ok {
			t.Errorf("parseUses(%q) didn't claim", c.line)
			continue
		}
		if file != c.file || symbol != c.symbol || from != c.from || name != c.name || forceType != c.forceType {
			t.Errorf("parseUses(%q) = (%q, %q, %q, %q, %v), want (%q, %q, %q, %q, %v)",
				c.line, file, symbol, from, name, forceType, c.file, c.symbol, c.from, c.name, c.forceType)
		}
	}
	notMine := []string{
		"add a function in web.go that calls LoadUser from store.go", // the delegate's
		"add a handler named Login to web.go",                        // the concept book's
		"add an Item in main in web.go",                              // instantiate's shape family
		"add a function in web.go",                                   // no use tail
		"what package do i use for json",                             // a question, not an edit
		"delegate LoadUser into web.go",                              // not a sentence shape at all
	}
	for _, line := range notMine {
		if _, _, _, _, _, ok := parseUses(line); ok {
			t.Errorf("parseUses(%q) claimed a line that isn't a use", line)
		}
	}
}

func TestUseCrossPackageEndToEnd(t *testing.T) {
	dir := usesChatFixture(t, nil)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !runUses("in app/main.go add a function that uses Load from store.go", yes) {
			t.Error("runUses didn't claim the canonical line")
		}
	})
	if !strings.Contains(out, "Here's the use I assembled") {
		t.Fatalf("no plan was shown:\n%s", out)
	}
	main := readBack(t, dir, "app/main.go")
	for _, want := range []string{
		`"example.com/uchat/store"`,
		"func UseLoad() {",
		"item, err := store.Load(0)",
		"if err != nil {",
		"_ = item",
	} {
		if !strings.Contains(main, want) {
			t.Errorf("landed use missing %q:\n%s", want, main)
		}
	}
	if n := countDelegateBackups(t, dir); n != 0 {
		t.Errorf("%d backup files left after the use, want 0", n)
	}
}

func TestUseTypeEndToEnd(t *testing.T) {
	dir := usesChatFixture(t, nil)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !runUses("add a function in app/main.go that uses the type Item from store.go", yes) {
			t.Error("runUses didn't claim the type line")
		}
	})
	if !strings.Contains(out, "Done —") {
		t.Fatalf("type use didn't land:\n%s", out)
	}
	main := readBack(t, dir, "app/main.go")
	for _, want := range []string{"func UseItem() {", "item := store.Item{", "_ = item"} {
		if !strings.Contains(main, want) {
			t.Errorf("landed type use missing %q:\n%s", want, main)
		}
	}
}

func TestUseDeclinedAtGate(t *testing.T) {
	dir := usesChatFixture(t, nil)
	withServerProject(t, dir)
	before := readBack(t, dir, "app/main.go")
	no := func(string) (string, error) { return "n", nil }
	out := captureStdout(t, func() {
		if !runUses("add a function in app/main.go that uses Load from store.go", no) {
			t.Error("runUses didn't claim the line")
		}
	})
	if !strings.Contains(out, "Not applied") {
		t.Errorf("decline not reported:\n%s", out)
	}
	if after := readBack(t, dir, "app/main.go"); after != before {
		t.Errorf("declined use changed the file:\n%s", after)
	}
}

func TestUseRefusalsAtTheChat(t *testing.T) {
	dir := usesChatFixture(t, nil)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !runUses("add a function in app/main.go that uses load from store.go", yes) {
			t.Error("runUses didn't claim the unexported line")
		}
	})
	if !strings.Contains(out, "isn't exported") {
		t.Errorf("unexported refusal missing:\n%s", out)
	}
	if strings.Contains(readBack(t, dir, "app/main.go"), "UseLoad") {
		t.Error("a refused use wrote to the file")
	}
	out = captureStdout(t, func() {
		if !runUses("add a function in app/main.go that uses Missing from store.go", yes) {
			t.Error("runUses didn't claim the missing-symbol line")
		}
	})
	if !strings.Contains(out, "can't find") {
		t.Errorf("missing-symbol refusal missing:\n%s", out)
	}
}

// The retry, exercised end to end: main.go already imports a
// package whose declared name is ALSO store (a legacy/ folder
// declaring package store). The use's plain form adds the real
// store as a colliding plain import — "store redeclared" — and
// the plan's one alternate form (the pkg-suffixed alias) is
// what lands, reported as the retry.
func TestUseRetryOnQualifierCollision(t *testing.T) {
	dir := usesChatFixture(t, map[string]string{
		"legacy/legacy.go": `package store

func Legacy() int { return 7 }
`,
		"app/main.go": `package main

import (
	"example.com/uchat/legacy"
)

func main() { _ = store.Legacy() }
`,
	})
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !runUses("add a function in app/main.go that uses Load from store.go", yes) {
			t.Error("runUses didn't claim the line")
		}
	})
	if !strings.Contains(out, "the retry the plan named") {
		t.Fatalf("the collision use didn't land via the retry:\n%s", out)
	}
	main := readBack(t, dir, "app/main.go")
	for _, want := range []string{
		`storepkg "example.com/uchat/store"`,
		"item, err := storepkg.Load(0)",
		"store.Legacy()", // the pre-existing call, untouched
	} {
		if !strings.Contains(main, want) {
			t.Errorf("retry-landed file missing %q:\n%s", want, main)
		}
	}
	if n := strings.Count(main, `"example.com/uchat/store"`); n != 1 {
		t.Errorf("store imported %d times, want exactly 1 (the aliased retry form):\n%s", n, main)
	}
	if n := countDelegateBackups(t, dir); n != 0 {
		t.Errorf("%d backup files left after the retried use, want 0", n)
	}
}

func TestUseClaimedInTheChain(t *testing.T) {
	dir := usesChatFixture(t, nil)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !claimChain("add a function in app/main.go that uses Load from store.go", yes, nil) {
			t.Error("the chain didn't claim the use line")
		}
	})
	if !strings.Contains(out, "Here's the use I assembled") {
		t.Errorf("chain-claimed use didn't run the use flow:\n%s", out)
	}
}
