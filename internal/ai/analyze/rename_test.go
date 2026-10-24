package analyze

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// renameFixture builds a small module exercising every rename
// shape: a plain function used across packages and from tests, a
// method pair on one type, a type used in signatures and literals,
// a package var and const, struct fields used via selectors and
// literal keys, an embedded field, a String method, a local that
// shadows a package function's name, and a main package.
func renameFixture(t *testing.T) (string, *Project) {
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
	write("go.mod", "module example.com/rename\n\ngo 1.21\n")
	write("lib/lib.go", `package lib

type Widget struct {
	Name string
	size int
}

func (w *Widget) Save() string { return w.Name }
func (w *Widget) Load() string { return w.Name }

func Greet(name string) string { return "hi " + name }

func Shadow() string {
	Greet := "local"
	return Greet
}

var DefaultName = "anon"

const Version = "1.0"

type Config struct {
	Host string
	Port int
}

func NewConfig() Config { return Config{Host: "localhost", Port: 80} }

type Base struct {
	X int
}

type Embedded struct {
	Base
	Extra string
}

type Label struct {
	Text string
}

func (l Label) String() string { return l.Text }
`)
	write("lib/lib_test.go", `package lib

import "testing"

func TestGreet(t *testing.T) {
	if Greet("x") != "hi x" {
		t.Fatal("bad greet")
	}
	w := &Widget{Name: "w"}
	if w.Save() != "w" {
		t.Fatal("bad save")
	}
	_ = DefaultName
	_ = Version
}
`)
	write("app/app.go", `package app

import "example.com/rename/lib"

func Run() string {
	c := lib.NewConfig()
	_ = c.Host
	return lib.Greet(lib.DefaultName) + lib.Version
}
`)
	write("app/app_test.go", `package app

import (
	"testing"

	"example.com/rename/lib"
)

func TestRun(t *testing.T) {
	_ = lib.Greet("y")
	_ = Run()
}
`)
	write("cmd/tool/main.go", `package main

import "example.com/rename/lib"

func main() {
	println(lib.Greet("z"))
}

func helper() string { return lib.Version }
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze fixture: %v", err)
	}
	return dir, p
}

func applyPlanToDisk(t *testing.T, dir string, plan *EditPlan) {
	t.Helper()
	for _, f := range plan.Files {
		if err := os.WriteFile(filepath.Join(dir, f.Rel), f.After, 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func buildFixtureProject(t *testing.T, dir string) {
	t.Helper()
	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture no longer builds: %v\n%s", err, out)
	}
}

func mustRenamePlan(t *testing.T, p *Project, req EditRequest) *EditPlan {
	t.Helper()
	plan, refusal := p.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("PlanEdit(%+v) refused: %s", req, refusal)
	}
	return plan
}

func renameReq(from, to string) EditRequest {
	return EditRequest{Kind: EditRename, RenameFrom: from, RenameTo: to}
}

func readFixtureFile(t *testing.T, dir, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, rel))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// A function rename rewrites the definition and every use —
// another package, both test files, a main package — and never the
// same-named local that shadows it in Shadow.
func TestRenameFuncAcrossPackages(t *testing.T) {
	dir, p := renameFixture(t)
	plan := mustRenamePlan(t, p, renameReq("Greet", "GreetUser"))
	applyPlanToDisk(t, dir, plan)
	for rel, want := range map[string]string{
		"lib/lib.go":       "func GreetUser(name string) string",
		"lib/lib_test.go":  "GreetUser(\"x\")",
		"app/app.go":       "lib.GreetUser(lib.DefaultName)",
		"app/app_test.go":  "lib.GreetUser(\"y\")",
		"cmd/tool/main.go": "lib.GreetUser(\"z\")",
	} {
		if got := readFixtureFile(t, dir, rel); !strings.Contains(got, want) {
			t.Errorf("%s does not contain %q after rename:\n%s", rel, want, got)
		}
	}
	shadow := readFixtureFile(t, dir, "lib/lib.go")
	if !strings.Contains(shadow, "Greet := \"local\"\n\treturn Greet") {
		t.Errorf("the shadowing local was renamed too — collateral damage:\n%s", shadow)
	}
	buildFixtureProject(t, dir)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("plan.Verify false after a successful rename")
	}
}

// A method rename follows selectors everywhere, tests included.
func TestRenameMethod(t *testing.T) {
	dir, p := renameFixture(t)
	plan := mustRenamePlan(t, p, renameReq("Widget.Save", "Store"))
	applyPlanToDisk(t, dir, plan)
	if got := readFixtureFile(t, dir, "lib/lib.go"); !strings.Contains(got, "func (w *Widget) Store() string") {
		t.Errorf("method definition not renamed:\n%s", got)
	}
	if got := readFixtureFile(t, dir, "lib/lib_test.go"); !strings.Contains(got, "w.Store()") {
		t.Errorf("test use not renamed:\n%s", got)
	}
	buildFixtureProject(t, dir)
}

// A type rename rewrites signatures, composite literals, and the
// embedded-field reference (which is a use of the type).
func TestRenameType(t *testing.T) {
	dir, p := renameFixture(t)
	plan := mustRenamePlan(t, p, renameReq("Config", "Settings"))
	applyPlanToDisk(t, dir, plan)
	got := readFixtureFile(t, dir, "lib/lib.go")
	for _, want := range []string{"type Settings struct", "func NewConfig() Settings", "Settings{Host: \"localhost\", Port: 80}"} {
		if !strings.Contains(got, want) {
			t.Errorf("lib.go missing %q after type rename:\n%s", want, got)
		}
	}
	buildFixtureProject(t, dir)

	// The embedded field is a use of Base's type: renaming the
	// type rewrites it, keeping Embedded valid. (The bare name
	// "Base" is ambiguous — type and field — so the kind hint
	// picks the type.)
	p2, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan = mustRenamePlan(t, p2, EditRequest{Kind: EditRename, RenameFrom: "Base", RenameTo: "Root", RenameKind: "type"})
	applyPlanToDisk(t, dir, plan)
	got = readFixtureFile(t, dir, "lib/lib.go")
	if !strings.Contains(got, "type Embedded struct {\n\tRoot\n") {
		t.Errorf("embedded reference not renamed with its type:\n%s", got)
	}
	buildFixtureProject(t, dir)

	// Renaming a type with methods rewrites the receiver
	// declarations and composite literals too.
	dir2, pw := renameFixture(t)
	plan = mustRenamePlan(t, pw, renameReq("Widget", "Gadget"))
	applyPlanToDisk(t, dir2, plan)
	got = readFixtureFile(t, dir2, "lib/lib.go")
	for _, want := range []string{"type Gadget struct", "func (w *Gadget) Save() string", "func (w *Gadget) Load() string"} {
		if !strings.Contains(got, want) {
			t.Errorf("lib.go missing %q after Widget rename:\n%s", want, got)
		}
	}
	if got := readFixtureFile(t, dir2, "lib/lib_test.go"); !strings.Contains(got, "&Gadget{Name: \"w\"}") {
		t.Errorf("composite literal not renamed:\n%s", got)
	}
	buildFixtureProject(t, dir2)
}

// Package-level var and const renames follow cross-package uses.
func TestRenameVarAndConst(t *testing.T) {
	dir, p := renameFixture(t)
	plan := mustRenamePlan(t, p, renameReq("DefaultName", "FallbackName"))
	applyPlanToDisk(t, dir, plan)
	p2, _ := Analyze(dir)
	plan = mustRenamePlan(t, p2, renameReq("Version", "ReleaseVersion"))
	applyPlanToDisk(t, dir, plan)
	if got := readFixtureFile(t, dir, "app/app.go"); !strings.Contains(got, "lib.FallbackName") || !strings.Contains(got, "lib.ReleaseVersion") {
		t.Errorf("cross-package uses not renamed:\n%s", got)
	}
	buildFixtureProject(t, dir)
}

// A field rename rewrites selector uses in other packages and
// keyed composite literal entries at home.
func TestRenameField(t *testing.T) {
	dir, p := renameFixture(t)
	plan := mustRenamePlan(t, p, renameReq("Config.Host", "Hostname"))
	applyPlanToDisk(t, dir, plan)
	if got := readFixtureFile(t, dir, "app/app.go"); !strings.Contains(got, "c.Hostname") {
		t.Errorf("selector use not renamed:\n%s", got)
	}
	if got := readFixtureFile(t, dir, "lib/lib.go"); !strings.Contains(got, "Hostname: \"localhost\"") || !strings.Contains(got, "Hostname string") {
		t.Errorf("declaration or literal key not renamed:\n%s", got)
	}
	buildFixtureProject(t, dir)
}

// Ambiguity is reported with candidates, never guessed at:
// "Base" is both a type and (embedded) a field of Embedded.
func TestRenameAmbiguousCandidates(t *testing.T) {
	_, p := renameFixture(t)
	cands := p.RenameCandidates(renameReq("Base", "Root"))
	if len(cands) != 2 {
		t.Fatalf("RenameCandidates(Base) = %d, want 2 (type + embedded field): %+v", len(cands), cands)
	}
	_, refusal := p.PlanEdit(renameReq("Base", "Root"))
	if !strings.Contains(refusal, "could mean more than one") {
		t.Fatalf("ambiguous PlanEdit refusal = %q", refusal)
	}
	// The kind hint narrows it back to one.
	cands = p.RenameCandidates(EditRequest{Kind: EditRename, RenameFrom: "Base", RenameTo: "Root", RenameKind: "type"})
	if len(cands) != 1 || cands[0].Kind != "type" {
		t.Fatalf("kind-hinted candidates = %+v", cands)
	}
}

// Collisions refuse before any plan exists.
func TestRenameCollisionRefusals(t *testing.T) {
	_, p := renameFixture(t)
	if _, refusal := p.PlanEdit(renameReq("Greet", "NewConfig")); !strings.Contains(refusal, "already taken") {
		t.Errorf("func->existing func refusal = %q", refusal)
	}
	if _, refusal := p.PlanEdit(renameReq("Widget.Save", "Load")); !strings.Contains(refusal, "already has a method") {
		t.Errorf("method->existing method refusal = %q", refusal)
	}
	if _, refusal := p.PlanEdit(renameReq("Greet", "Greet")); !strings.Contains(refusal, "already named") {
		t.Errorf("same-name refusal = %q", refusal)
	}
	if _, refusal := p.PlanEdit(renameReq("Greet", "123bad")); !strings.Contains(refusal, "isn't a name Go allows") {
		t.Errorf("invalid-name refusal = %q", refusal)
	}
	if _, refusal := p.PlanEdit(renameReq("Greet", "func")); !strings.Contains(refusal, "isn't a name Go allows") {
		t.Errorf("keyword-name refusal = %q", refusal)
	}
	if _, refusal := p.PlanEdit(renameReq("Nope", "Yep")); !strings.Contains(refusal, "couldn't find") {
		t.Errorf("not-found refusal = %q", refusal)
	}
}

// main and init keep their names; Error and String too.
func TestRenameProtectedNames(t *testing.T) {
	_, p := renameFixture(t)
	if _, refusal := p.PlanEdit(renameReq("main", "entry")); !strings.Contains(refusal, "entry point") {
		t.Errorf("main refusal = %q", refusal)
	}
	if _, refusal := p.PlanEdit(renameReq("Label.String", "Render")); !strings.Contains(refusal, "fmt.Stringer") {
		t.Errorf("String refusal = %q", refusal)
	}
}

// An embedded field refuses and points at the type rename instead.
func TestRenameEmbeddedFieldRefused(t *testing.T) {
	_, p := renameFixture(t)
	var field *RenameTarget
	for _, c := range p.RenameCandidates(renameReq("Base", "Root")) {
		if c.Kind == "field" {
			cc := c
			field = &cc
		}
	}
	if field == nil {
		t.Fatal("no field candidate for the embedded Base")
	}
	_, refusal := p.PlanRenameTarget(*field, "Root")
	if !strings.Contains(refusal, "embedded field") {
		t.Fatalf("embedded refusal = %q", refusal)
	}
}

// A symbol declared in a generated file refuses, and so does a
// rename whose uses reach into one.
func renameGeneratedFixture(t *testing.T) (string, *Project) {
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
	write("go.mod", "module example.com/genrename\n\ngo 1.21\n")
	write("lib/lib.go", `package lib

func UsedByGen() string { return "u" }

func Plain() string { return "p" }
`)
	write("lib/gen.go", `// Code generated by a tool. DO NOT EDIT.

package lib

func GenFunc() string { return UsedByGen() }
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return dir, p
}

func TestRenameGeneratedRefusals(t *testing.T) {
	_, p := renameGeneratedFixture(t)
	if _, refusal := p.PlanEdit(renameReq("GenFunc", "MadeFunc")); !strings.Contains(refusal, "generated") {
		t.Errorf("decl-in-generated refusal = %q", refusal)
	}
	if _, refusal := p.PlanEdit(renameReq("UsedByGen", "UsedByHand")); !strings.Contains(refusal, "generated") {
		t.Errorf("use-in-generated refusal = %q", refusal)
	}
	// A symbol the generated file never touches renames fine.
	if _, refusal := p.PlanEdit(renameReq("Plain", "Simple")); refusal != "" {
		t.Errorf("untouched symbol refused: %s", refusal)
	}
}

// A use in a file this platform's build can't see (a windows-only
// variant) refuses the rename rather than leaving the use behind.
func TestRenameBuildTagGuard(t *testing.T) {
	dir, _ := renameFixture(t)
	win := "lib/win.go"
	src := "//go:build windows\n\npackage lib\n\nfunc WinGreet() string { return Greet(\"w\") }\n"
	if err := os.WriteFile(filepath.Join(dir, win), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	p2, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, refusal := p2.PlanEdit(renameReq("Greet", "GreetUser"))
	if !strings.Contains(refusal, "isn't part of this platform's build") {
		t.Fatalf("build-tag guard refusal = %q", refusal)
	}
	// An unrelated symbol the windows file never mentions is fine.
	if _, refusal := p2.PlanEdit(renameReq("Version", "ReleaseVersion")); refusal != "" {
		t.Errorf("unrelated rename refused by the guard: %s", refusal)
	}
}

// A method that satisfies a project interface refuses: the
// interface owns the name too.
func TestRenameInterfaceMethodRefused(t *testing.T) {
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
	write("go.mod", "module example.com/ifrename\n\ngo 1.21\n")
	write("lib/lib.go", `package lib

type Door struct{}

func (d Door) Open() string { return "open" }
func (d Door) Close() string { return "closed" }
`)
	write("app/app.go", `package app

import "example.com/ifrename/lib"

type Opener interface {
	Open() string
}

var _ Opener = lib.Door{}

func Use(o Opener) string { return o.Open() }
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, refusal := p.PlanEdit(renameReq("Door.Open", "Unlock")); !strings.Contains(refusal, "Opener") {
		t.Fatalf("interface-method refusal = %q", refusal)
	}
	// Close is on no interface: it renames.
	if _, refusal := p.PlanEdit(renameReq("Door.Close", "Shut")); refusal != "" {
		t.Fatalf("non-interface method refused: %s", refusal)
	}
}
