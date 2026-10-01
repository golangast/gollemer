package synthesis

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

func mustParseOpt(t *testing.T, src string) (*token.FileSet, *ast.File) {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "opt.go", src, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	return fset, f
}

func TestOptimizeAllocationsPreallocates(t *testing.T) {
	src := `package main

import "fmt"

func build(n int) []int {
	var out []int
	for i := 0; i < n; i++ {
		out = append(out, i*2)
	}
	fmt.Println(out)
	return out
}
`
	fset, f := mustParseOpt(t, src)
	rep := OptimizeAllocations(fset, f)
	if rep.AllocsBefore != 1 {
		t.Errorf("AllocsBefore = %d, want 1", rep.AllocsBefore)
	}
	if rep.AllocsAfter != 0 {
		t.Errorf("AllocsAfter = %d, want 0", rep.AllocsAfter)
	}
	if !strings.Contains(rep.OptimizedCode, "out := make([]int, 0, n)") {
		t.Errorf("missing preallocation:\n%s", rep.OptimizedCode)
	}
	if len(rep.Changes) == 0 || !strings.Contains(rep.Changes[0], "preallocated out") {
		t.Errorf("changes = %v", rep.Changes)
	}
	// The rewritten code must still parse.
	if _, err := parser.ParseFile(token.NewFileSet(), "out.go", rep.OptimizedCode, 0); err != nil {
		t.Errorf("optimized code does not parse: %v", err)
	}
}

func TestOptimizeAllocationsLiteralBound(t *testing.T) {
	src := `package main

func tens() []int {
	var out []int
	for i := 0; i < 10; i++ {
		out = append(out, i)
	}
	return out
}
`
	fset, f := mustParseOpt(t, src)
	rep := OptimizeAllocations(fset, f)
	if !strings.Contains(rep.OptimizedCode, "make([]int, 0, 10)") {
		t.Errorf("missing literal-bound preallocation:\n%s", rep.OptimizedCode)
	}
	if rep.AllocsAfter != 0 {
		t.Errorf("AllocsAfter = %d, want 0", rep.AllocsAfter)
	}
}

func TestOptimizeAllocationsSkipsSideEffectBound(t *testing.T) {
	src := `package main

func next() int { return 3 }

func build() []int {
	var out []int
	for i := 0; i < next(); i++ {
		out = append(out, i)
	}
	return out
}
`
	fset, f := mustParseOpt(t, src)
	rep := OptimizeAllocations(fset, f)
	if strings.Contains(rep.OptimizedCode, "make([]int") {
		t.Errorf("must not hoist a side-effecting bound:\n%s", rep.OptimizedCode)
	}
	if rep.AllocsAfter != rep.AllocsBefore {
		t.Errorf("AllocsAfter = %d, want unchanged %d", rep.AllocsAfter, rep.AllocsBefore)
	}
}

func TestOptimizeAllocationsNarrowsReceiver(t *testing.T) {
	src := `package main

type Big struct {
	A, B, C, D, E int
}

func (b Big) Sum() int { return b.A + b.B + b.C + b.D + b.E }

type Small struct{ X int }

func (s Small) Get() int { return s.X }
`
	fset, f := mustParseOpt(t, src)
	rep := OptimizeAllocations(fset, f)
	if !strings.Contains(rep.OptimizedCode, "func (b *Big) Sum() int") {
		t.Errorf("Big receiver not narrowed:\n%s", rep.OptimizedCode)
	}
	if strings.Contains(rep.OptimizedCode, "func (s *Small) Get() int") {
		t.Errorf("Small receiver must stay a value receiver:\n%s", rep.OptimizedCode)
	}
}

func TestOptimizeAllocationsNoop(t *testing.T) {
	src := `package main

import "fmt"

func main() { fmt.Println("hi") }
`
	fset, f := mustParseOpt(t, src)
	rep := OptimizeAllocations(fset, f)
	if rep.AllocsBefore != 0 || rep.AllocsAfter != 0 {
		t.Errorf("allocs = %d -> %d, want 0 -> 0", rep.AllocsBefore, rep.AllocsAfter)
	}
	if strings.TrimSpace(rep.OptimizedCode) == "" {
		t.Error("empty optimized code")
	}
}
