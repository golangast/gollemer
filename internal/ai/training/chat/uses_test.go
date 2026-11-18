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
		{"add a function named Report in web.go that uses Fetch", "web.go", "Fetch", "", "Report", false},
		{"add a function named Report in api/report.go that uses Validate from api/validate.go", "api/report.go", "Validate", "api/validate.go", "Report", false},
		{"add a function named Report in web.go that uses the type Item from store.go", "web.go", "Item", "store.go", "Report", true},
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

// The paces hybrid, end to end (2026-11-01 workout, defect
// 3): "add a function named Report in api/report.go that
// uses Validate from api/validate.go" carries BOTH the book's
// name introducer and the uses tail. It used to fall between
// the two grammars — the uses grammar wanted the bare name,
// the book's function claim met the "that uses" clause
// unprepared — and land on the write boundary's extra-clause
// refusal, whose listed forms never mention the uses
// sentence. Now the introducer routes to the same assembled
// edit the bare-name form produces.
func TestUseNamedHybridEndToEnd(t *testing.T) {
	extra := map[string]string{
		"api/main.go":     "package main\n\nfunc main() {}\n",
		"api/validate.go": "package main\n\nfunc Validate() bool { return true }\n",
		"api/report.go":   "package main\n",
	}
	hybrid := "add a function named Report in api/report.go that uses Validate from api/validate.go"

	dir := usesChatFixture(t, extra)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !runUses(hybrid, yes) {
			t.Error("runUses didn't claim the named hybrid")
		}
	})
	if !strings.Contains(out, "Here's the use I assembled from this project's own code") {
		t.Fatalf("the hybrid didn't produce the assembled use plan:\n%s", out)
	}
	if strings.Contains(out, "extra clause") {
		t.Errorf("the hybrid still met the boundary's extra-clause refusal:\n%s", out)
	}
	report := readBack(t, dir, "api/report.go")
	for _, want := range []string{"func Report(", "Validate("} {
		if !strings.Contains(report, want) {
			t.Errorf("landed hybrid missing %q:\n%s", want, report)
		}
	}

	// The same line through the whole claim chain: the uses
	// claim takes it before the book or the boundary can.
	dir2 := usesChatFixture(t, extra)
	withServerProject(t, dir2)
	out = captureStdout(t, func() {
		if !claimChain(hybrid, yes, nil) {
			t.Error("the chain didn't claim the named hybrid")
		}
	})
	if !strings.Contains(out, "Here's the use I assembled") {
		t.Errorf("chain-claimed hybrid didn't run the use flow:\n%s", out)
	}
	if !strings.Contains(readBack(t, dir2, "api/report.go"), "func Report(") {
		t.Error("chain-claimed hybrid didn't land Report")
	}
}

// The paces step-6 sequence in ONE session: the target file
// is created by the chat itself, then the hybrid lands in
// it. A chat-created file is baselined in the freshness
// snapshot but is not in the index's file lists until the
// next re-read — the planner resolves it by directory, and
// the post-gate package lookup must agree, or the flow
// answered "I lost track of api/report.go's package" after
// the gate and wrote nothing (the workout's step-6 failure
// once the hybrid could route at all).
func TestUseNamedHybridIntoJustCreatedFile(t *testing.T) {
	dir := usesChatFixture(t, map[string]string{
		"api/main.go":     "package main\n\nfunc main() {}\n",
		"api/validate.go": "package main\n\nfunc Validate() bool { return true }\n",
	})
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		if !claimChain("create api/report.go", yes, profile) {
			t.Error("the chain didn't claim the create")
		}
		if !claimChain("add a function named Report in api/report.go that uses Validate from api/validate.go", yes, profile) {
			t.Error("the chain didn't claim the hybrid after the create")
		}
	})
	if strings.Contains(out, "I lost track") {
		t.Errorf("the use into a just-created file lost its package:\n%s", out)
	}
	report := readBack(t, dir, "api/report.go")
	for _, want := range []string{"func Report(", "Validate("} {
		if !strings.Contains(report, want) {
			t.Errorf("just-created file missing %q after the hybrid:\n%s", want, report)
		}
	}
}

// The hybrid's honest refusals are the uses route's own: a
// symbol that doesn't exist, and a target file that doesn't
// exist (a use goes into a file that already exists — the
// create is a separate, named step).
func TestUseNamedHybridRefusals(t *testing.T) {
	extra := map[string]string{
		"api/main.go":     "package main\n\nfunc main() {}\n",
		"api/validate.go": "package main\n\nfunc Validate() bool { return true }\n",
		"api/report.go":   "package main\n",
	}
	dir := usesChatFixture(t, extra)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !runUses("add a function named Report in api/report.go that uses Missing from api/validate.go", yes) {
			t.Error("runUses didn't claim the missing-symbol hybrid")
		}
	})
	if !strings.Contains(out, "Not doing that use:") || !strings.Contains(out, "can't find") {
		t.Errorf("missing-symbol hybrid refusal is not the uses route's:\n%s", out)
	}
	out = captureStdout(t, func() {
		if !runUses("add a function named Report in api/absent.go that uses Validate from api/validate.go", yes) {
			t.Error("runUses didn't claim the missing-file hybrid")
		}
	})
	if !strings.Contains(out, "there's no api/absent.go in this project") {
		t.Errorf("missing-file hybrid refusal is not the uses route's:\n%s", out)
	}
	if strings.Contains(readBack(t, dir, "api/report.go"), "func Report(") {
		t.Error("a refused hybrid wrote to report.go")
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
