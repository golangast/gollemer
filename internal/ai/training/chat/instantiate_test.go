package chat

import (
	"strings"
	"testing"
)

// Instantiation (instantiate.go + analyze/editinstantiate.go):
// "add instantiation of struct Jim to handler Jake" makes a
// value of an existing project type inside an existing project
// function. These tests pin the parse family, the claim
// discipline (instance-worded claims even on a miss; the plain
// form only on full resolution), the end-to-end landings in a
// scratch project that declares the jim/Jake pattern, and the
// honest refusals writing nothing.

func TestParseInstantiateClaims(t *testing.T) {
	cases := []struct {
		line             string
		typ, fn, varName string
		explicit         bool
	}{
		{"add instantiation of struct Jim to handler Jake", "Jim", "Jake", "", true},
		{"add instantiation of Jim to handler Jake", "Jim", "Jake", "", true},
		{"add an instance of Jim in Jake", "Jim", "Jake", "", true},
		{"add instance of Jim in Jake", "Jim", "Jake", "", true},
		{"add a Jim in Jake", "Jim", "Jake", "", false},
		{"add a Jim to Jake", "Jim", "Jake", "", false},
		{"add a Jim into the function Jake", "Jim", "Jake", "", false},
		{"put a Jim in Jake", "Jim", "Jake", "", false},
		{"add a Jim in Jake named j", "Jim", "Jake", "j", false},
		{"add instantiation of struct Jim to handler Jake named jimmy", "Jim", "Jake", "jimmy", true},
		{"put a store.Jim into the handler Jake called sj", "store.Jim", "Jake", "sj", false},
		{"add a jim in jake", "jim", "jake", "", false},
	}
	for _, c := range cases {
		req, ok := parseInstantiate(c.line)
		if !ok {
			t.Errorf("parseInstantiate(%q) not claimed", c.line)
			continue
		}
		if req.typeName != c.typ || req.funcName != c.fn || req.varName != c.varName || req.explicit != c.explicit {
			t.Errorf("parseInstantiate(%q) = %+v, want type %q func %q var %q explicit %v",
				c.line, req, c.typ, c.fn, c.varName, c.explicit)
		}
	}
}

func TestParseInstantiateNonClaims(t *testing.T) {
	for _, line := range []string{
		"add a struct named Jim to jim/main.go",         // the book's declaration
		"add a struct name is Jim to jim/main.go",       // the book's, spelled out
		"add a function named Jake to jim/main.go",      // the book's scaffold
		"add a jim to jim/main.go",                      // a FILE target is never this shape
		"add a Jim in jim/main.go",                      // file target, no type/func reading
		"add a Jim to handler",                          // the target names nothing
		"add a Jim",                                     // no target clause at all
		"add instantiation of struct Jim",               // instance wording, still no target
		"add a Jim in Jake please",                      // trailing words refuse the family
		"add a Jim in Jake that prints",                 // a behaviour clause is not this shape
		"create a Jim in Jake",                          // creations' verb
		"add a Jim in Jake named",                       // the name slot opens, nothing follows
		"add a method named M to jim/main.go",           // kind word in the object slot
		"add a function in jim/main.go that calls Load", // the delegate's
	} {
		if req, ok := parseInstantiate(line); ok {
			t.Errorf("parseInstantiate(%q) claimed %+v — that line is not this shape's", line, req)
		}
	}
}

// instantiateFixture writes the jim/Jake scratch project: a
// struct with fields of several shapes and a handler to hold
// the statement, in one main package with its main present
// (the spine's build is absolute).
func instantiateFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/instantiate\n\ngo 1.21\n")
	writeFile(t, dir, "jim.go", `package main

type Jim struct {
	Name  string
	Count int
	Tags  []string
	Home  *Jim
}

type Tags []string

type Greeter interface {
	Greet() string
}

func Jake() {
	println("ok")
}

func main() { Jake() }
`)
	return dir
}

func TestInstantiateJohnsFormEndToEnd(t *testing.T) {
	for _, analyzed := range []bool{true, false} {
		dir := instantiateFixture(t)
		withChatSession(t, dir, analyzed)
		profile := &UserProfile{}
		line := "add instantiation of struct Jim to handler Jake"
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Fatalf("analyzed=%v: %q fell through the claim chain:\n%s", analyzed, line, out)
		}
		requireNoAnalyzeOutput(t, line, out)
		got := readBack(t, dir, "jim.go")
		for _, want := range []string{
			"jim := Jim{",
			`Name:  "",`,
			"Count: 0,",
			"Tags:  nil,",
			"Home:  nil,",
			"_ = jim",
		} {
			if !strings.Contains(got, want) {
				t.Errorf("analyzed=%v: jim.go missing %q:\n%s\noutput:\n%s", analyzed, want, got, out)
			}
		}
		// The statement landed INSIDE Jake, not anywhere else.
		jakeAt := strings.Index(got, "func Jake()")
		jimAt := strings.Index(got, "jim := Jim{")
		mainAt := strings.Index(got, "func main()")
		if !(jakeAt >= 0 && jimAt > jakeAt && jimAt < mainAt) {
			t.Errorf("analyzed=%v: the literal did not land inside Jake:\n%s", analyzed, got)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Errorf("analyzed=%v: spine build/verify not reported:\n%s", analyzed, out)
		}
	}
}

func TestInstantiateVariantsEndToEnd(t *testing.T) {
	cases := []struct {
		line string
		want string
	}{
		{"add a Jim in Jake", "jim := Jim{"},
		{"add an instance of Jim to Jake named j", "j := Jim{"},
		{"add instantiation of Tags in Jake", "tags := Tags{}"},
	}
	for _, c := range cases {
		dir := instantiateFixture(t)
		withChatSession(t, dir, true)
		profile := &UserProfile{}
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
		if !handled {
			t.Errorf("%q fell through the claim chain:\n%s", c.line, out)
			continue
		}
		got := readBack(t, dir, "jim.go")
		if !strings.Contains(got, c.want) {
			t.Errorf("%q: jim.go missing %q:\n%s\noutput:\n%s", c.line, c.want, got, out)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Errorf("%q: spine build/verify not reported:\n%s", c.line, out)
		}
	}
}

// The near-miss discipline: instance wording claims even when
// a side can't be found, and the answer names what wasn't
// found — with nothing written. The plain form, by contrast,
// does not claim at all when resolution fails.
func TestInstantiateRefusalsWriteNothing(t *testing.T) {
	dir := instantiateFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	before := readBack(t, dir, "jim.go")
	for _, c := range []struct{ line, want string }{
		{"add instantiation of Zed to handler Jake", "can't find a type named"},
		{"add instantiation of Jim to handler Zed", "can't find a function named"},
		{"add instantiation of Greeter to handler Jake", "is an interface"},
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
		if !handled {
			t.Errorf("%q fell through the claim chain — the instance-worded form claims its misses:\n%s", c.line, out)
			continue
		}
		if !strings.Contains(out, c.want) {
			t.Errorf("%q: answer does not say %q:\n%s", c.line, c.want, out)
		}
	}
	if got := readBack(t, dir, "jim.go"); got != before {
		t.Errorf("a refusal wrote to jim.go:\n%s", got)
	}

	// The plain form with an unresolvable side is not claimed
	// by this shape at all.
	var claimed bool
	captureStdout(t, func() { claimed = runInstantiate("add a Zed in Jake", yes) })
	if claimed {
		t.Error("plain form claimed an unknown type — it must resolve both sides before claiming")
	}
	captureStdout(t, func() { claimed = runInstantiate("add a Jim in Zed", yes) })
	if claimed {
		t.Error("plain form claimed an unknown function — it must resolve both sides before claiming")
	}
}

// A name the function already uses refuses with the escape
// named; the escape then works.
func TestInstantiateNameCollisionAndEscape(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/instcollide\n\ngo 1.21\n")
	writeFile(t, dir, "jim.go", `package main

type Jim struct {
	Name string
}

func Jake() {
	jim := 0
	println(jim)
}

func main() { Jake() }
`)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add instantiation of Jim to handler Jake", yes, profile) })
	if !handled || !strings.Contains(out, "already uses the name") {
		t.Fatalf("collision was not refused in the family's voice:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("add instantiation of Jim to handler Jake named james", yes, profile) })
	if !handled {
		t.Fatalf("named escape fell through:\n%s", out)
	}
	got := readBack(t, dir, "jim.go")
	if !strings.Contains(got, "james := Jim{") {
		t.Errorf("the named variable did not land:\n%s\noutput:\n%s", got, out)
	}
}

// Cross-package: the type lives in store, the handler in main.
// The literal is qualified, the import is added, and the
// unexported field is left out with the plan's note naming it.
func TestInstantiateCrossPackage(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/xpkg\n\ngo 1.21\n")
	writeFile(t, dir, "store/store.go", `package store

type Jim struct {
	Name   string
	secret int
}
`)
	writeFile(t, dir, "main.go", `package main

func Jake() {
	println("ok")
}

func main() { Jake() }
`)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add instantiation of Jim to handler Jake", yes, profile) })
	if !handled {
		t.Fatalf("cross-package line fell through:\n%s", out)
	}
	got := readBack(t, dir, "main.go")
	for _, want := range []string{
		`"example.com/xpkg/store"`,
		"jim := store.Jim{",
		`Name: "",`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("main.go missing %q:\n%s\noutput:\n%s", want, got, out)
		}
	}
	if strings.Contains(got, "secret") {
		t.Errorf("the unexported field was set from another package:\n%s", got)
	}
	if !strings.Contains(out, "Left out of the literal: secret") {
		t.Errorf("the plan did not own the skipped field:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}

// The neighbours keep their sentences inside the same project:
// the book's struct declaration is still the book's, and a
// file target is never read as a function.
func TestInstantiateNeighboursKeepTheirLines(t *testing.T) {
	dir := instantiateFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a struct named Extra to jim.go", yes, profile) })
	if !handled || !strings.Contains(out, "concept book") {
		t.Errorf("the book's struct line was stolen or unhandled:\n%s", out)
	}
	var claimed bool
	captureStdout(t, func() { claimed = runInstantiate("add a jim to jim.go", yes) })
	if claimed {
		t.Error("a file target was claimed as an instantiation")
	}
}
