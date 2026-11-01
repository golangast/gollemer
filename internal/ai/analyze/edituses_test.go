package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// usesProject writes a small multi-package module and
// analyzes it: a store package holding the symbols to use
// (functions of several shapes, types of several kinds), an
// app (package main) and a lib package as targets, and an
// other/ package whose Fetch and Item make ambiguity one
// keystroke away.
func usesProject(t *testing.T, extra map[string]string) (string, *Project) {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/uses\n\ngo 1.21\n",
		"store/store.go": `package store

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"time"
)

type Item struct {
	ID   int
	Name string
}

func (i Item) Label() string { return "item" }

type Config struct {
	Retries int
	Verbose bool
}

type Handler interface{ Handle() }

type IDs []int

type UserID string

func Load(id int) (Item, error) {
	if id < 0 {
		return Item{}, errors.New("bad id")
	}
	return Item{ID: id}, nil
}

func Fetch(key string) int { return len(key) }

func Reset() {}

func Join(parts ...string) string { return strings.Join(parts, ",") }

func Save(item Item) error {
	if item.ID < 0 {
		return errors.New("bad item")
	}
	return nil
}

func WithCtx(ctx context.Context, id int) (Item, error) { return Item{ID: id}, nil }

func At(t time.Time) time.Time { return t }

func LogRequest(r http.Request) { _ = r }

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

type Item struct{ X int }
`,
		"app/main.go": "package main\n\nfunc main() {}\n",
		"lib/lib.go": `package lib

func Local(n int) int { return n * 2 }

func Config() int { return 3 }
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
		t.Fatalf("Analyze uses fixture: %v", err)
	}
	return dir, p
}

func TestUsePlanCrossPackageFunc(t *testing.T) {
	_, p := usesProject(t, nil)
	plan := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.Load",
	})
	main := plan.Files[0]
	if main.Rel != "app/main.go" {
		t.Fatalf("first plan file = %s, want app/main.go", main.Rel)
	}
	for _, want := range []string{
		`"example.com/uses/store"`,
		"func UseLoad() {",
		"item, err := store.Load(0)",
		"if err != nil {",
		"_ = item",
	} {
		if !strings.Contains(string(main.After), want) {
			t.Errorf("use file missing %q:\n%s", want, main.After)
		}
	}
	// The provenance names the source fact the plan was built from.
	for _, want := range []string{"store/store.go", "func Load(id int) (Item, error)"} {
		if !strings.Contains(plan.Provenance, want) {
			t.Errorf("provenance %q missing %q", plan.Provenance, want)
		}
	}
	// The retry form exists (cross-package) and calls through
	// the alias.
	if len(plan.AltFiles) == 0 {
		t.Fatal("cross-package use plan has no retry form")
	}
	alt := string(plan.AltFiles[0].After)
	for _, want := range []string{`storepkg "example.com/uses/store"`, "storepkg.Load(0)"} {
		if !strings.Contains(alt, want) {
			t.Errorf("retry form missing %q:\n%s", want, alt)
		}
	}
	// The placeholder note names the parameters it filled.
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "placeholder values for its parameters (id int)") {
		t.Errorf("plan notes missing the placeholder inventory:\n%s", notes)
	}
}

func TestUsePlanFromFileDisambiguates(t *testing.T) {
	_, p := usesProject(t, nil)
	plan := mustPlan(t, p, EditRequest{
		Kind:        EditUse,
		UseFile:     "app/main.go",
		UseSymbol:   "Load",
		UseFromFile: "store.go",
	})
	if !strings.Contains(string(plan.Files[0].After), "store.Load(0)") {
		t.Errorf("from-file use didn't resolve to store.Load:\n%s", plan.Files[0].After)
	}
	// The wrong from-file refuses, naming where the symbol lives.
	mustRefuse(t, p, EditRequest{
		Kind:        EditUse,
		UseFile:     "app/main.go",
		UseSymbol:   "Load",
		UseFromFile: "lib.go",
	}, "doesn't declare Load")
}

func TestUsePlanSamePackageFunc(t *testing.T) {
	_, p := usesProject(t, nil)
	plan := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "lib/other.go",
		UseSymbol: "Local",
	})
	body := string(plan.Files[0].After)
	for _, want := range []string{"func UseLocal() {", "value := Local(0)", "_ = value"} {
		if !strings.Contains(body, want) {
			t.Errorf("same-package use missing %q:\n%s", want, body)
		}
	}
	if len(plan.AltFiles) != 0 {
		t.Errorf("same-package plan has a retry form (%d files) — there is no import to vary", len(plan.AltFiles))
	}
}

func TestUsePlanStructArgAndCtxArg(t *testing.T) {
	_, p := usesProject(t, nil)
	// A struct parameter gets the instantiate literal, from
	// the type's real definition, inside the call.
	save := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.Save",
	})
	body := string(save.Files[0].After)
	for _, want := range []string{"if err := store.Save(store.Item{", "ID:   0,", `Name: "",`, "}); err != nil {"} {
		if !strings.Contains(body, want) {
			t.Errorf("struct-arg use missing %q:\n%s", want, body)
		}
	}
	// A context parameter gets context.Background() and the
	// import rides along.
	ctxPlan := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.WithCtx",
	})
	body = string(ctxPlan.Files[0].After)
	for _, want := range []string{`"context"`, "item, err := store.WithCtx(context.Background(), 0)"} {
		if !strings.Contains(body, want) {
			t.Errorf("ctx-arg use missing %q:\n%s", want, body)
		}
	}
	// A time.Time gets its public zero.
	at := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.At",
	})
	if !strings.Contains(string(at.Files[0].After), "t := store.At(time.Time{})") {
		t.Errorf("time-arg use wrong:\n%s", at.Files[0].After)
	}
	// A no-result call is a bare statement; a variadic call
	// passes the tail nothing.
	reset := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.Reset",
	})
	if !strings.Contains(string(reset.Files[0].After), "\tstore.Reset()\n") {
		t.Errorf("no-arg use wrong:\n%s", reset.Files[0].After)
	}
	join := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.Join",
	})
	if !strings.Contains(string(join.Files[0].After), "value := store.Join()") {
		t.Errorf("variadic use wrong:\n%s", join.Files[0].After)
	}
	if notes := strings.Join(join.Notes, "\n"); !strings.Contains(notes, "variadic") {
		t.Errorf("variadic use notes don't say the tail was passed nothing:\n%s", notes)
	}
}

func TestUsePlanTypeCrossPackage(t *testing.T) {
	_, p := usesProject(t, nil)
	plan := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.Item",
	})
	body := string(plan.Files[0].After)
	for _, want := range []string{
		`"example.com/uses/store"`,
		"func UseItem() {",
		"item := store.Item{",
		"ID:   0,",
		`Name: "",`,
		"_ = item",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("type use missing %q:\n%s", want, body)
		}
	}
	if len(plan.AltFiles) == 0 {
		t.Error("cross-package type use has no retry form")
	}
	// The "type" designator forces the type reading of a bare
	// name the from-file pins down.
	forced := mustPlan(t, p, EditRequest{
		Kind:        EditUse,
		UseFile:     "app/main.go",
		UseSymbol:   "Item",
		UseFromFile: "store.go",
		UseType:     true,
	})
	if !strings.Contains(string(forced.Files[0].After), "item := store.Item{") {
		t.Errorf("forced type use wrong:\n%s", forced.Files[0].After)
	}
}

func TestUsePlanTypeDefinedKinds(t *testing.T) {
	_, p := usesProject(t, nil)
	// A defined basic gets its conversion zero.
	uid := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.UserID",
	})
	if !strings.Contains(string(uid.Files[0].After), `userID := store.UserID("")`) {
		t.Errorf("defined-basic type use wrong:\n%s", uid.Files[0].After)
	}
	// A defined slice gets the empty literal.
	ids := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "app/main.go",
		UseSymbol: "store.IDs",
	})
	if !strings.Contains(string(ids.Files[0].After), "iDs := store.IDs{}") {
		t.Errorf("defined-slice type use wrong:\n%s", ids.Files[0].After)
	}
	// A package-clause-only target file: the import block and
	// the new function both anchor at EOF — the imports must
	// still land BEFORE the function (the same-offset merge).
	clauseOnly := mustPlan(t, p, EditRequest{
		Kind:      EditUse,
		UseFile:   "lib/other.go",
		UseSymbol: "store.Item",
		UseType:   true,
	})
	body := string(clauseOnly.Files[0].After)
	impAt := strings.Index(body, `"example.com/uses/store"`)
	fnAt := strings.Index(body, "func UseItem()")
	if impAt < 0 || fnAt < 0 || impAt > fnAt {
		t.Errorf("clause-only target assembled out of order (import at %d, func at %d):\n%s", impAt, fnAt, body)
	}
}

func TestUseRefusals(t *testing.T) {
	_, p := usesProject(t, nil)
	cases := []struct {
		name string
		req  EditRequest
		want string
	}{
		{"ambiguous function", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "Fetch"}, "could mean more than one function"},
		{"missing symbol", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "Nope"}, "can't find a function or type called"},
		{"unexported cross-package", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "store.load"}, "isn't exported"},
		{"underivable argument", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "store.LogRequest"}, "no starting value I can derive"},
		{"generic function", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "store.First"}, "is generic"},
		{"interface type", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "store.Handler"}, "is an interface"},
		{"method", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "Item.Label"}, "is a method on"},
		{"ambiguous type", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "Item", UseType: true}, "declared in more than one package"},
		{"function and type", EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "Config", UseFromFile: ""}, "both a function and a type"},
		{"name collision", EditRequest{Kind: EditUse, UseFile: "lib/other.go", UseSymbol: "Local", UseName: "Config"}, "already exists in package lib"},
		{"test file target", EditRequest{Kind: EditUse, UseFile: "lib/lib_test.go", UseSymbol: "Local"}, "is a test file"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			mustRefuse(t, p, c.req, c.want)
		})
	}
	// The underivable refusal names the parameter and its type.
	_, refusal := p.PlanEdit(EditRequest{Kind: EditUse, UseFile: "app/main.go", UseSymbol: "store.LogRequest"})
	if !strings.Contains(refusal, "http.Request") {
		t.Errorf("underivable refusal doesn't name the type:\n%s", refusal)
	}
}

func TestUseSelfCheck(t *testing.T) {
	good := []byte(`package main

import "example.com/uses/store"

func UseLoad() {
	item, err := store.Load(0)
	if err != nil {
		return
	}
	_ = item
}
`)
	if bad := selfCheckUses(good, usesExpect{name: "UseLoad", srcName: "Load", srcQual: "store", args: 1, hasError: true, errVar: "err", valVars: []string{"item"}, addedImps: []string{"example.com/uses/store"}}); bad != "" {
		t.Errorf("self-check rejected a good use body: %s", bad)
	}
	// The error dropped on the floor: no guard.
	dropped := []byte(`package main

import "example.com/uses/store"

func UseLoad() {
	item, err := store.Load(0)
	_ = err
	_ = item
}
`)
	if bad := selfCheckUses(dropped, usesExpect{name: "UseLoad", srcName: "Load", srcQual: "store", args: 1, hasError: true, errVar: "err", valVars: []string{"item"}}); !strings.Contains(bad, "doesn't check") {
		t.Errorf("self-check let a dropped error through: %q", bad)
	}
	// A passthrough in use's clothing: parameters of its own.
	passthrough := []byte(`package main

import "example.com/uses/store"

func UseLoad(id int) (store.Item, error) {
	return store.Load(id)
}
`)
	if bad := selfCheckUses(passthrough, usesExpect{name: "UseLoad", srcName: "Load", srcQual: "store", args: 1, hasError: true, errVar: "err"}); bad == "" {
		t.Error("self-check let a passthrough through — that's the delegate's shape, not a use")
	}
	// Type form: the value built is the wrong type.
	wrongType := []byte(`package main

func UseItem() {
	item := 0
	_ = item
}
`)
	if bad := selfCheckUses(wrongType, usesExpect{name: "UseItem", typeRef: "store.Item", valVar: "item"}); !strings.Contains(bad, "doesn't build") {
		t.Errorf("self-check let a wrong type form through: %q", bad)
	}
}
