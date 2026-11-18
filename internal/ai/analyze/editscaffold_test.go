package analyze

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// scaffoldProject is the fixture for the delegate's test
// scaffold: a store package holding the signature zoo the type
// table must handle — scalars, a populated-able struct, a
// struct with unexported fields, a context, containers, and an
// interface it must refuse to invent — and an app package to
// delegate into.
func scaffoldProject(t *testing.T, extra map[string]string) (string, *Project) {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/scaffold\n\ngo 1.21\n",
		"store/store.go": `package store

import (
	"context"
	"io"
)

type Receipt struct {
	Total int
	Note  string
}

type Options struct {
	Name  string
	Count int
}

type Secret struct {
	Public string
	hidden int
}

func Process(id string, qty int) (*Receipt, error) {
	if qty < 0 {
		return nil, io.EOF
	}
	return &Receipt{Total: qty}, nil
}

func Configure(opts *Options) error { return nil }

func ConfigureSecret(opts *Secret) error { return nil }

func WithContext(ctx context.Context, name string) string { return name }

func Batch(tags []string, counts map[string]int) int { return len(tags) + len(counts) }

func Lookup(r io.Reader) (string, error) { return "", nil }
`,
		"app/main.go": "package main\n\nfunc main() {}\n",
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
		t.Fatalf("Analyze scaffold fixture: %v", err)
	}
	return dir, p
}

// runGoIn runs the go tool in dir and returns its combined
// output and success.
func runGoIn(t *testing.T, dir string, args ...string) (string, bool) {
	t.Helper()
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH")
	}
	cmd := exec.Command(goBin, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// scContains reports whether the scaffold contains want,
// ignoring whitespace runs — gofmt aligns literal columns, and
// the scaffold's content, not its padding, is what these tests
// pin.
func scContains(sc, want string) bool {
	norm := func(s string) string { return strings.Join(strings.Fields(s), " ") }
	return strings.Contains(norm(sc), norm(want))
}

// scaffoldOf returns the plan's scaffold file text.
func scaffoldOf(t *testing.T, plan *EditPlan) string {
	t.Helper()
	if len(plan.Files) != 2 {
		t.Fatalf("plan files = %d, want 2 (wrapper + scaffold)", len(plan.Files))
	}
	stub := plan.Files[1]
	if stub.Rel != "app/main_test.go" {
		t.Fatalf("scaffold file = %s, want app/main_test.go", stub.Rel)
	}
	return string(stub.After)
}

func TestScaffoldScalarsAndPointerResult(t *testing.T) {
	dir, p := scaffoldProject(t, nil)
	plan := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "store.Process"})
	sc := scaffoldOf(t, plan)
	for _, want := range []string{
		// The args struct mirrors the signature.
		"type args struct",
		"id string",
		"qty int",
		// The zero case, from the type table.
		`name: "zero values"`,
		`id: "",`,
		"qty: 0,",
		// The named edge cases.
		`name: "edge: id — non-empty string"`,
		`id: "test-string",`,
		`name: "edge: qty — negative"`,
		"qty: -1,",
		// Expectations are placeholders, labeled as such.
		"want: nil, // TODO: the result you expect",
		"wantErr: false, // TODO: set true if this case should return an error.",
		// The standard loop, calling the wrapper by name.
		"got, err := Process(tc.args.id, tc.args.qty)",
		"if (err != nil) != tc.wantErr {",
		"if !reflect.DeepEqual(got, tc.want) {",
		// The honesty machinery: header + self-skip.
		"inputs are generated from the parameter types",
		"t.Skip(\"scaffold: fill in the want/wantErr TODOs below",
		`"reflect"`,
		`"testing"`,
	} {
		if !scContains(sc, want) {
			t.Errorf("scaffold missing %q:\n%s", want, sc)
		}
	}
	// The plan labels the scaffold a scaffold, not evidence.
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "test scaffold") || !strings.Contains(notes, "not part of how the edit is verified") {
		t.Errorf("plan notes don't label the scaffold honestly:\n%s", notes)
	}
	// Applied: the fixture builds, vets, and its tests compile —
	// and the scaffold test itself reports SKIP, never a pass
	// and never a failure.
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if out, ok := runGoIn(t, dir, "vet", "./..."); !ok {
		t.Errorf("fixture vet after scaffold:\n%s", out)
	}
	out, ok := runGoIn(t, dir, "test", "-v", "-run", "TestProcess", "./app/")
	if !ok {
		t.Fatalf("scaffold test run failed:\n%s", out)
	}
	if !strings.Contains(out, "--- SKIP: TestProcess") {
		t.Errorf("scaffold test should report SKIP:\n%s", out)
	}
	if out, ok := runGoIn(t, dir, "test", "./..."); !ok {
		t.Errorf("fixture suite after scaffold:\n%s", out)
	}
}

func TestScaffoldStructPointerEdges(t *testing.T) {
	_, p := scaffoldProject(t, nil)
	plan := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "Configure"})
	sc := scaffoldOf(t, plan)
	for _, want := range []string{
		"opts: nil,", // zero: a nil pointer
		`name: "edge: opts — populated"`,
		`opts: &store.Options{Name: "test-string", Count: -1},`, // populated from the fields
		"err := Configure(tc.args.opts)",
		"wantErr: false, // TODO",
	} {
		if !scContains(sc, want) {
			t.Errorf("populated-struct scaffold missing %q:\n%s", want, sc)
		}
	}

	// Unexported fields: no populated literal may be offered —
	// the scaffold says why, in the file and in the plan note.
	plan2 := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "ConfigureSecret"})
	sc2 := scaffoldOf(t, plan2)
	if strings.Contains(sc2, "&store.Secret{") {
		t.Errorf("scaffold populated a struct with unexported fields:\n%s", sc2)
	}
	if !strings.Contains(sc2, "unexported fields") {
		t.Errorf("scaffold doesn't explain the missing edge case:\n%s", sc2)
	}
	if !strings.Contains(strings.Join(plan2.Notes, "\n"), "unexported fields") {
		t.Errorf("plan note doesn't carry the scaffold's TODO:\n%s", strings.Join(plan2.Notes, "\n"))
	}
	if !strings.Contains(sc2, "opts: nil,") {
		t.Errorf("secret scaffold lost its zero case:\n%s", sc2)
	}
}

func TestScaffoldContextParam(t *testing.T) {
	dir, p := scaffoldProject(t, nil)
	plan := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "WithContext"})
	sc := scaffoldOf(t, plan)
	for _, want := range []string{
		`"context"`,
		"ctx: context.Background(),",
		`name: "edge: ctx — cancelled context"`,
		"ctx: gollemerCancelledContext(),",
		"func gollemerCancelledContext() context.Context {",
		"got := WithContext(tc.args.ctx, tc.args.name)",
	} {
		if !scContains(sc, want) {
			t.Errorf("context scaffold missing %q:\n%s", want, sc)
		}
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	if out, ok := runGoIn(t, dir, "test", "./app/"); !ok {
		t.Errorf("context scaffold doesn't run clean:\n%s", out)
	}
}

func TestScaffoldSliceAndMapParams(t *testing.T) {
	_, p := scaffoldProject(t, nil)
	plan := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "Batch"})
	sc := scaffoldOf(t, plan)
	for _, want := range []string{
		"tags: nil,",
		"counts: nil,",
		`name: "edge: tags — one element"`,
		`tags: []string{"test-string"},`,
		`name: "edge: counts — one entry"`,
		`counts: map[string]int{"test-string": -1},`,
		"got := Batch(tc.args.tags, tc.args.counts)",
		"want: 0, // TODO: the result you expect",
	} {
		if !scContains(sc, want) {
			t.Errorf("container scaffold missing %q:\n%s", want, sc)
		}
	}
	if strings.Contains(sc, "wantErr bool") || strings.Contains(sc, "wantErr:") {
		t.Errorf("Batch returns no error, but the scaffold has a wantErr field:\n%s", sc)
	}
}

func TestScaffoldInterfaceParamTODO(t *testing.T) {
	_, p := scaffoldProject(t, nil)
	plan := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "Lookup"})
	sc := scaffoldOf(t, plan)
	// The reader's zero is nil (an interface's honest zero), but
	// no edge value may be invented for it, and the file must
	// say so.
	if !strings.Contains(sc, "r: nil,") {
		t.Errorf("interface param lost its nil zero:\n%s", sc)
	}
	if strings.Contains(sc, "edge: r") {
		t.Errorf("scaffold invented an edge case for an interface param:\n%s", sc)
	}
	if !strings.Contains(sc, "real implementation") {
		t.Errorf("scaffold doesn't explain the interface TODO:\n%s", sc)
	}
	for _, invented := range []string{"bytes.NewReader", "strings.NewReader", "io.LimitReader"} {
		if strings.Contains(sc, invented) {
			t.Errorf("scaffold invented a reader value (%s):\n%s", invented, sc)
		}
	}
}

func TestScaffoldAppendsToExistingTestFile(t *testing.T) {
	dir, p := scaffoldProject(t, map[string]string{
		"app/main_test.go": `package main

import "testing"

func TestExisting(t *testing.T) {}
`,
	})
	plan := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "store.Process"})
	if len(plan.Files) != 2 {
		t.Fatalf("plan files = %d, want 2", len(plan.Files))
	}
	sc := string(plan.Files[1].After)
	if !strings.Contains(sc, "func TestExisting") || !strings.Contains(sc, "func TestProcess") {
		t.Errorf("appended scaffold lost a test:\n%s", sc)
	}
	if n := strings.Count(sc, `"testing"`); n != 1 {
		t.Errorf("testing imported %d times after append:\n%s", n, sc)
	}
	if !strings.Contains(sc, `"reflect"`) {
		t.Errorf("appended scaffold missing its reflect import:\n%s", sc)
	}
	applyPlan(t, dir, plan)
	buildFixture(t, dir)
	out, ok := runGoIn(t, dir, "test", "-v", "./app/")
	if !ok {
		t.Fatalf("appended scaffold's package doesn't test clean:\n%s", out)
	}
	if !strings.Contains(out, "--- PASS: TestExisting") || !strings.Contains(out, "--- SKIP: TestProcess") {
		t.Errorf("appended run should pass the old test and skip the scaffold:\n%s", out)
	}
}

func TestScaffoldGenericOmittedAndVariadicWorks(t *testing.T) {
	dir, p := delegateProject(t, nil)
	// A generic source gets no scaffold — and the plan says
	// why, in plain words.
	gen := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "First"})
	if len(gen.Files) != 1 {
		t.Fatalf("generic plan files = %d, want 1 (wrapper only)", len(gen.Files))
	}
	if !strings.Contains(strings.Join(gen.Notes, "\n"), "generic") {
		t.Errorf("generic plan doesn't explain the missing scaffold:\n%s", strings.Join(gen.Notes, "\n"))
	}
	// A variadic source's scaffold types the args field as a
	// slice and forwards with ....
	vari := mustPlan(t, p, EditRequest{Kind: EditDelegate, DelegateFile: "app/main.go", SourceName: "Join"})
	sc := scaffoldOf(t, vari)
	if !scContains(sc, "parts []string") || !scContains(sc, "Join(tc.args.parts...)") {
		t.Errorf("variadic scaffold wrong:\n%s", sc)
	}
	if !scContains(sc, `parts: []string{"test-string"},`) {
		t.Errorf("variadic scaffold missing its edge case:\n%s", sc)
	}
	applyPlan(t, dir, vari)
	buildFixture(t, dir)
	if out, ok := runGoIn(t, dir, "test", "./app/"); !ok {
		t.Errorf("variadic scaffold's package doesn't test clean:\n%s", out)
	}
}

func TestScaffoldSelfCheck(t *testing.T) {
	b := &scaffoldBuilder{
		g:      &delegateGen{wrapper: "Load"},
		params: []scaffoldParam{{name: "id", typ: "int"}},
	}
	good := `// TestLoad is a test scaffold.
func TestLoad(t *testing.T) {
	t.Skip("scaffold")

	type args struct {
		id int
	}
	tests := []struct {
		name string
		args args
	}{
		{name: "zero values", args: args{id: 0}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			Load(tc.args.id)
		})
	}
}
`
	if bad := selfCheckScaffold(good, "TestLoad", b); bad != "" {
		t.Errorf("self-check rejected a correct scaffold: %s", bad)
	}
	bads := map[string]string{
		"no skip": strings.Replace(good, "\tt.Skip(\"scaffold\")\n", "", 1),
		"wrong field type": strings.Replace(good,
			"\t\tid int\n", "\t\tid string\n", 1),
		"wrong field name": strings.Replace(good,
			"\t\tid int\n", "\t\tn int\n", 1),
		"never calls the wrapper": strings.Replace(good,
			"Load(tc.args.id)", "Other(tc.args.id)", 1),
		"missing test": strings.Replace(good,
			"func TestLoad(", "func TestOther(", 1),
	}
	for name, body := range bads {
		if bad := selfCheckScaffold(body, "TestLoad", b); bad == "" {
			t.Errorf("self-check accepted a scaffold with %s", name)
		}
	}
}
