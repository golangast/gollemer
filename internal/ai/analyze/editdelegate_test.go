package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// delegateProject writes a small multi-package module and
// analyzes it: a store package with the delegate sources, an app
// (package main) and a lib package as targets, and a second
// Fetch in other/ so ambiguity is one keystroke away.
func delegateProject(t *testing.T, extra map[string]string) (string, *Project) {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/delegate\n\ngo 1.21\n",
		"store/store.go": `package store

import (
	"errors"
	"strings"
	"time"
)

type Item struct{ ID int }

func (i Item) Label() string { return "item" }

func Load(id int) (Item, error) {
	if id < 0 {
		return Item{}, errors.New("bad id")
	}
	return Item{ID: id}, nil
}

func Fetch(key string) int { return len(key) }

func Reset() {}

func Join(parts ...string) string { return strings.Join(parts, ",") }

func At(t time.Time) time.Time { return t }

func First[T any](items []T) (T, bool) {
	if len(items) == 0 {
		var zero T
		return zero, false
	}
	return items[0], true
}

func load(id int) Item { return Item{ID: id} }
`,
		"other/other.go": `package other

func Fetch() int { return 1 }
`,
		"app/main.go": "package main\n\nfunc main() {}\n",
		"lib/lib.go": `package lib

func Local(n int) int { return n * 2 }
`,
		"lib/other.go": "package lib\n",
	}
	for rel, src := range extra {
		files[rel] = src
	}
	for rel, src := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze delegate fixture: %v", err)
	}
	return dir, p
}

func TestDelegatePlanCrossPackage(t *testing.T) {
	dir, p := delegateProject(t, nil)
	plan := mustPlan(t, p, EditRequest{
		Kind:         EditDelegate,
		DelegateFile: "app/main.go",
		SourceName:   "store.Load", // the qualified spelling resolves too
	})
	if len(plan.Files) != 2 {
		t.Fatalf("plan files = %d, want 2 (wrapper + test skeleton): %+v", len(plan.Files), plan.Files)
	}
	main := plan.Files[0]
	if main.Rel != "app/main.go" {
		t.Fatalf("first plan file = %s, want app/main.go", main.Rel)
	}
	for _, want := range []string{
		`"example.com/delegate/store"`,
		"func Load(id int) (store.Item, error)",
		"return store.Load(id)",
	} {
		if !strings.Contains(string(main.After), want) {
			t.Errorf("wrapper file missing %q:\n%s", want, main.After)
		}
	}
	// The provenance line names the source fact the plan was
	// built from.
	for _, want := range []string{"store/store.go", "func Load(id int) (Item, error)"} {
		if !strings.Contains(plan.Provenance, want) {
			t.Errorf("provenance %q missing %q", plan.Provenance, want)
		}
	}
	// The failure-mode note: Load can fail, and the plan says the
	// wrapper inherits that.
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "straight through") {
		t.Errorf("plan notes missing the failure-mode note:\n%s", notes)
	}
	// The scaffold rides along, labeled as one.
	stub := plan.Files[1]
	if stub.Rel != "app/main_test.go" || len(stub.Before) != 0 {
		t.Errorf("scaffold = %s (before %d bytes), want a created app/main_test.go", stub.Rel, len(stub.Before))
	}
	for _, want := range []string{"func TestLoad(t *testing.T)", "TODO", "tests := []struct"} {
		if !strings.Contains(string(stub.After), want) {
			t.Errorf("scaffold missing %q:\n%s", want, stub.After)
		}
	}
	if !strings.Contains(notes, "test scaffold") || !strings.Contains(notes, "not part of how the edit is verified") {
		t.Errorf("plan notes don't label the scaffold as a non-verifying scaffold:\n%s", notes)
	}
	// The retry form is pre-assembled and named: the source
	// imported under the alias, the call qualified by it.
	if len(plan.AltFiles) != 2 {
		t.Fatalf("alt files = %d, want 2 (wrapper form + same skeleton)", len(plan.AltFiles))
	}
	alt := string(plan.AltFiles[0].After)
	for _, want := range []string{`storepkg "example.com/delegate/store"`, "return storepkg.Load(id)"} {
		if !strings.Contains(alt, want) {
			t.Errorf("retry form missing %q:\n%s", want, alt)
		}
	}
	if !strings.Contains(plan.AltDesc, "storepkg") {
		t.Errorf("AltDesc doesn't name the retry form: %q", plan.AltDesc)
	}
	// Applied, the whole fixture still builds, and the plan's
	// structural verify finds the wrapper calling the source.
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("plan.Verify false on the applied fixture")
	}
}

func TestDelegatePlanSamePackage(t *testing.T) {
	dir, p := delegateProject(t, nil)
	// Unnamed, a same-package delegate is refused: the source's
	// name is taken by the source itself, and the wrapper needs
	// a name of its own. The refusal carries the working
	// sentence.
	mustRefuse(t, p, EditRequest{
		Kind:         EditDelegate,
		DelegateFile: "lib/other.go",
		SourceName:   "Local",
	}, "needs its own name")
	plan := mustPlan(t, p, EditRequest{
		Kind:         EditDelegate,
		DelegateFile: "lib/other.go",
		SourceName:   "Local",
		WrapperName:  "Double",
	})
	after := string(plan.Files[0].After)
	if !strings.Contains(after, "func Double(n int) int") || !strings.Contains(after, "return Local(n)") {
		t.Errorf("same-package wrapper wrong:\n%s", after)
	}
	if strings.Contains(after, "example.com/delegate/lib") {
		t.Errorf("same-package delegate imported its own package:\n%s", after)
	}
	if len(plan.AltFiles) != 0 {
		t.Errorf("same-package plan has a retry form (%d files) — there is no import to vary", len(plan.AltFiles))
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("plan.Verify false on the applied same-package fixture")
	}
}

func TestDelegatePlanUserNamedAndFallback(t *testing.T) {
	_, p := delegateProject(t, nil)
	named := mustPlan(t, p, EditRequest{
		Kind:         EditDelegate,
		DelegateFile: "app/main.go",
		SourceName:   "Load",
		WrapperName:  "LoadItem",
	})
	if !strings.Contains(string(named.Files[0].After), "func LoadItem(id int) (store.Item, error)") {
		t.Errorf("user-named wrapper missing:\n%s", named.Files[0].After)
	}
	if !strings.Contains(string(named.Files[1].After), "func TestLoadItem(t *testing.T)") {
		t.Errorf("skeleton not named for the user's wrapper:\n%s", named.Files[1].After)
	}

	// The fallback rule: when package main already declares Load,
	// the wrapper is <Source>From<Pkg> — LoadFromStore. (The
	// source is named qualified here: with a Load of its own in
	// main, the bare name is honestly ambiguous.)
	_, p2 := delegateProject(t, map[string]string{
		"app/main.go": "package main\n\nfunc main() {}\n\nfunc Load() {}\n",
	})
	fb := mustPlan(t, p2, EditRequest{
		Kind:         EditDelegate,
		DelegateFile: "app/main.go",
		SourceName:   "store.Load",
	})
	if !strings.Contains(string(fb.Files[0].After), "func LoadFromStore(id int) (store.Item, error)") {
		t.Errorf("fallback-named wrapper missing:\n%s", fb.Files[0].After)
	}
	// A user-given name that collides is a refusal, never a
	// silent rename.
	mustRefuse(t, p2, EditRequest{
		Kind:         EditDelegate,
		DelegateFile: "app/main.go",
		SourceName:   "store.Load",
		WrapperName:  "Load",
	}, "already exists in package main")
}

func TestDelegatePlanShapes(t *testing.T) {
	_, p := delegateProject(t, nil)
	// A void source: a bare call statement, no return.
	void := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "Reset"})
	after := string(void.Files[0].After)
	if !strings.Contains(after, "func Reset() {") || !strings.Contains(after, "\tstore.Reset()\n") {
		t.Errorf("void delegate wrong:\n%s", after)
	}
	if strings.Contains(after, "return store.Reset()") {
		t.Errorf("void delegate returns a call:\n%s", after)
	}
	// A variadic source: the wrapper is variadic and forwards
	// with ....
	vari := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "Join"})
	after = string(vari.Files[0].After)
	if !strings.Contains(after, "func Join(parts ...string) string") || !strings.Contains(after, "return store.Join(parts...)") {
		t.Errorf("variadic delegate wrong:\n%s", after)
	}
	// A signature mentioning an imported type: the type keeps
	// its qualifier and the import rides along.
	imp := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "At"})
	after = string(imp.Files[0].After)
	if !strings.Contains(after, "func At(t time.Time) time.Time") || !strings.Contains(after, `"time"`) {
		t.Errorf("imported-type delegate wrong:\n%s", after)
	}
	// A generic source: the type parameters ride along and the
	// call instantiates them explicitly.
	gen := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "First"})
	after = string(gen.Files[0].After)
	if !strings.Contains(after, "func First[T any](items []T) (T, bool)") || !strings.Contains(after, "return store.First[T](items)") {
		t.Errorf("generic delegate wrong:\n%s", after)
	}
}

func TestDelegateRefusals(t *testing.T) {
	_, p := delegateProject(t, nil)
	base := EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go"}

	r := base
	r.SourceName = "Item"
	mustRefuse(t, p, r, "not a function")
	mustRefuse(t, p, r, "delegates functions only")

	r = base
	r.SourceName = "Label"
	mustRefuse(t, p, r, "is a method on Item")

	r = base
	r.SourceName = "Loaad"
	ref := mustRefuse(t, p, r, `can't find a function called "Loaad"`)
	if !strings.Contains(ref, `"Load"`) {
		t.Errorf("missing-symbol refusal should name the near match:\n%s", ref)
	}

	r = base
	r.SourceName = "Zanzibar"
	ref = mustRefuse(t, p, r, `can't find a function called "Zanzibar"`)
	if strings.Contains(ref, "Did you mean") {
		t.Errorf("no near match exists for Zanzibar, but the refusal offers one:\n%s", ref)
	}

	r = base
	r.SourceName = "Fetch"
	ref = mustRefuse(t, p, r, "could mean more than one function")
	for _, want := range []string{"store/store.go", "other/other.go"} {
		if !strings.Contains(ref, want) {
			t.Errorf("ambiguity refusal should list %s:\n%s", want, ref)
		}
	}

	r = base
	r.SourceName = "load"
	mustRefuse(t, p, r, "isn't exported")

	r = base
	r.SourceName = "main"
	mustRefuse(t, p, r, "entry point")

	r = base
	r.SourceName = "Load"
	r.DelegateFile = "app/main_test.go"
	mustRefuse(t, p, r, "test file")

	r = base
	r.DelegateFile = "testdata/x.go"
	mustRefuse(t, p, r, "isn't part of the project")
}

func TestDelegateSelfCheck(t *testing.T) {
	exp := delegateExpect{
		wrapper:   "Load",
		srcName:   "Load",
		srcQual:   "store",
		results:   2,
		args:      1,
		addedImps: []string{"example.com/delegate/store"},
	}
	good := []byte(`package main

import "example.com/delegate/store"

func Load(id int) (store.Item, error) {
	return store.Load(id)
}
`)
	if bad := selfCheckDelegate(good, exp); bad != "" {
		t.Errorf("self-check rejected a correct delegate: %s", bad)
	}
	bads := map[string][]byte{
		"drops the results": []byte(`package main

import "example.com/delegate/store"

func Load(id int) (store.Item, error) {
	store.Load(id)
	return store.Item{}, nil
}
`),
		"wrong argument count": []byte(`package main

import "example.com/delegate/store"

func Load(id int) (store.Item, error) {
	return store.Load(id, id)
}
`),
		"calls a different function": []byte(`package main

import "example.com/delegate/store"

func Load(id int) (store.Item, error) {
	return store.Fetch(id)
}
`),
		"undeclared qualifier": []byte(`package main

func Load(id int) (store.Item, error) {
	return store.Load(id)
}
`),
		"added import unused": []byte(`package main

import (
	"example.com/delegate/other"
	"example.com/delegate/store"
)

func Load(id int) (store.Item, error) {
	return store.Load(id)
}
`),
		"map write in the body": []byte(`package main

import "example.com/delegate/store"

func Load(id int) (store.Item, error) {
	m := map[string]int{}
	m["x"] = id
	return store.Load(id)
}
`),
		"map literal as the argument": []byte(`package main

import "example.com/delegate/store"

func Load(id int) (store.Item, error) {
	_ = map[string]int{"x": id}
	return store.Load(id)
}
`),
	}
	expUnused := exp
	expUnused.addedImps = []string{"example.com/delegate/store", "example.com/delegate/other"}
	for name, src := range bads {
		e := exp
		if name == "added import unused" {
			e = expUnused
		}
		if bad := selfCheckDelegate(src, e); bad == "" {
			t.Errorf("self-check accepted %s", name)
		}
	}
	// Variadic forwarding is checked both ways.
	vexp := delegateExpect{wrapper: "Join", srcName: "Join", srcQual: "store", results: 1, args: 1, variadic: true, addedImps: []string{"example.com/delegate/store"}}
	noDots := []byte(`package main

import "example.com/delegate/store"

func Join(parts ...string) string {
	return store.Join(parts)
}
`)
	if bad := selfCheckDelegate(noDots, vexp); bad == "" {
		t.Error("self-check accepted a variadic delegate that doesn't forward with ...")
	}
}
