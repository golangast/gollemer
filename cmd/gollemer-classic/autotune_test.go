package main

import (
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// applyMut renders src through m and returns the result.
func applyMut(t *testing.T, m mutation, src string) (string, bool) {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "x.go", src, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	changed := m.apply(f)
	var sb strings.Builder
	if err := format.Node(&sb, fset, f); err != nil {
		t.Fatalf("format: %v", err)
	}
	return sb.String(), changed
}

func mustContain(t *testing.T, s, sub string) {
	t.Helper()
	if !strings.Contains(s, sub) {
		t.Errorf("expected %q in:\n%s", sub, s)
	}
}

func TestPreallocAppendRange(t *testing.T) {
	src := `package p

func Squares(xs []int) []int {
	var squares []int
	for _, x := range xs {
		squares = append(squares, x*x)
	}
	return squares
}
`
	out, changed := applyMut(t, allocMutations[0], src)
	if !changed {
		t.Fatal("expected change")
	}
	mustContain(t, out, "squares := make([]int, 0, len(xs))")
	if strings.Contains(out, "var squares []int") {
		t.Error("old declaration still present")
	}
}

func TestPreallocAppendClassicFor(t *testing.T) {
	src := `package p

func Fill(n int) []int {
	var out []int
	for i := 0; i < n; i++ {
		out = append(out, i)
	}
	return out
}
`
	out, changed := applyMut(t, allocMutations[0], src)
	if !changed {
		t.Fatal("expected change")
	}
	mustContain(t, out, "out := make([]int, 0, n)")
}

func TestPreallocAppendNoAppend(t *testing.T) {
	src := `package p

func F(xs []int) []int {
	var out []int
	for _, x := range xs {
		println(x)
	}
	return out
}
`
	_, changed := applyMut(t, allocMutations[0], src)
	if changed {
		t.Error("must not change when nothing is appended")
	}
}

func TestPreallocAppendShadowed(t *testing.T) {
	src := `package p

func F(xs []int) []int {
	var out []int
	for _, x := range xs {
		out := append(out, x)
		_ = out
	}
	return out
}
`
	_, changed := applyMut(t, allocMutations[0], src)
	if changed {
		t.Error("must not change when the loop shadows the variable")
	}
}

func TestStringBuilder(t *testing.T) {
	src := `package p

func Concat(words []string) string {
	var s string
	for _, w := range words {
		s += w
	}
	return s
}
`
	out, changed := applyMut(t, allocMutations[1], src)
	if !changed {
		t.Fatal("expected change")
	}
	mustContain(t, out, "var s strings.Builder")
	mustContain(t, out, "s.WriteString(w)")
	mustContain(t, out, "return s.String()")
	mustContain(t, out, `"strings"`)
	if strings.Contains(out, "s += w") {
		t.Error("old concat still present")
	}
}

func TestStringBuilderReassignmentBails(t *testing.T) {
	src := `package p

func F(words []string) string {
	var s string
	for _, w := range words {
		s += w
	}
	s = "reset"
	return s
}
`
	_, changed := applyMut(t, allocMutations[1], src)
	if changed {
		t.Error("must not change when s is reassigned")
	}
}

func TestStringBuilderAddressTakenBails(t *testing.T) {
	src := `package p

func F(words []string) string {
	var s string
	for _, w := range words {
		s += w
	}
	_ = &s
	return s
}
`
	_, changed := applyMut(t, allocMutations[1], src)
	if changed {
		t.Error("must not change when &s is taken")
	}
}

func TestStringBuilderFieldNameUntouched(t *testing.T) {
	src := `package p

type T struct{ s string }

func F(words []string, t T) string {
	var s string
	for _, w := range words {
		s += w
	}
	return s + t.s
}
`
	out, changed := applyMut(t, allocMutations[1], src)
	if !changed {
		t.Fatal("expected change")
	}
	mustContain(t, out, "t.s") // struct field must not become t.s.String()
	if strings.Contains(out, "t.s.String()") {
		t.Error("struct field was wrongly rewritten")
	}
}

func TestPointerReceiver(t *testing.T) {
	src := `package p

type Store struct{ n int }

func (s Store) Get() int { return s.n }
func (s *Store) Set(n int) { s.n = n }
`
	out, changed := applyMut(t, allocMutations[2], src)
	if !changed {
		t.Fatal("expected change")
	}
	mustContain(t, out, "func (s *Store) Get() int")
	mustContain(t, out, "func (s *Store) Set(n int)")
}

func TestPointerReceiverAlreadyPointer(t *testing.T) {
	src := `package p

type Store struct{ n int }

func (s *Store) Set(n int) { s.n = n }
`
	_, changed := applyMut(t, allocMutations[2], src)
	if changed {
		t.Error("must not change when all receivers are already pointers")
	}
}

func TestTryMutationEndToEnd(t *testing.T) {
	files := map[string]string{
		"a.go":      "package p\n\nfunc F(xs []int) []int {\n\tvar out []int\n\tfor _, x := range xs {\n\t\tout = append(out, x)\n\t}\n\treturn out\n}\n",
		"a_test.go": "package p\n", // test files are never mutated
	}
	out, changed := tryMutation(files, allocMutations[0])
	if !changed {
		t.Fatal("expected change")
	}
	mustContain(t, out["a.go"], "make([]int, 0, len(xs))")
	if out["a_test.go"] != files["a_test.go"] {
		t.Error("test file was modified")
	}
}

// Ensure the mutated AST is still valid Go by re-parsing it.
func TestMutatedOutputParses(t *testing.T) {
	srcs := []string{
		"package p\n\nfunc F(xs []int) []int {\n\tvar out []int\n\tfor _, x := range xs {\n\t\tout = append(out, x)\n\t}\n\treturn out\n}\n",
		"package p\n\nfunc G(ws []string) string {\n\tvar s string\n\tfor _, w := range ws {\n\t\ts += w\n\t}\n\treturn s\n}\n",
		"package p\n\ntype T struct{}\nfunc (t T) M() {}\n",
	}
	for i, m := range allocMutations {
		out, changed := applyMut(t, m, srcs[i])
		if !changed {
			t.Fatalf("mutation %d: expected change", i)
		}
		if _, err := parser.ParseFile(token.NewFileSet(), "x.go", out, 0); err != nil {
			t.Fatalf("mutation %d: output does not parse: %v\n%s", i, err, out)
		}
		var _ = ast.NewIdent // keep ast import if unused in future edits
	}
}

func TestEnsureImportKeepsDocComment(t *testing.T) {
	src := "package p\n\n// Concat concatenates words.\nfunc Concat() string {\n\tvar s string\n\tfor _, w := range []string{\"a\"} {\n\t\ts += w\n\t}\n\treturn s\n}\n"
	out, changed := applyMut(t, allocMutations[1], src)
	if !changed {
		t.Fatal("expected change")
	}
	// The doc comment must stay with the function, not leak onto the
	// import line (go/printer interleaves comments by position).
	bad := "import \"strings\" // Concat"
	if strings.Contains(out, bad) {
		t.Errorf("doc comment leaked onto import line:\n%s", out)
	}
	lines := strings.Split(out, "\n")
	foundImport, foundDoc := -1, -1
	for i, l := range lines {
		if strings.HasPrefix(l, "import \"strings\"") {
			foundImport = i
		}
		if strings.HasPrefix(l, "// Concat") {
			foundDoc = i
		}
	}
	if foundImport < 0 || foundDoc < 0 {
		t.Fatalf("import or doc comment missing:\n%s", out)
	}
	if foundImport > foundDoc {
		t.Errorf("import must precede the doc comment:\n%s", out)
	}
}
