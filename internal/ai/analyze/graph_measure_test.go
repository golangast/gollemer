package analyze

// Graph measurement harness: compares the heuristic-only call graph
// against ground truth computed with the real type checker, on a
// fixture project shaped to stress the two heuristic gaps this file
// exists to measure — constructor-held values whose method names
// collide with unrelated same-package methods, interface dispatch
// (parameter- and field-held), and embedded promotion across
// packages. Run with GOLLEMER_MEASURE=1 to print precision/recall;
// the assertions live in graph_sharpen_test.go.

import (
	"fmt"
	"go/ast"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

// measureFiles is the refinable fixture: everything compiles, so
// refineTypes runs on it under plain Analyze; analyzeDir(dir, true)
// exposes the heuristic-only graph for comparison.
func measureFiles() map[string]string {
	return map[string]string{
		"go.mod": "module example.com/measure\n\ngo 1.21\n",
		"base/base.go": `package base

type Base struct{}

func (b *Base) Helper() string { return "help" }
func (b *Base) Shared() string { return "base" }
`,
		"child/child.go": `package child

import "example.com/measure/base"

type Child struct {
	base.Base
}

func (c *Child) Shared() string { return "child" }
`,
		"store/cache.go": `package store

type Cache struct{}

func (c *Cache) Save() string { return "cache" }
`,
		"store/store.go": `package store

type Store struct{ n int }

func New() *Store { return &Store{} }

func NewStore() *Store { return &Store{} }

func NewWithErr() (*Store, error) { return &Store{}, nil }

func (s *Store) Save() string { return "store" }

func (s *Store) Load() string { return "load" }

func SaveNew() string {
	s := New()
	return s.Save()
}

func SaveCache() string {
	c := &Cache{}
	return c.Save()
}
`,
		"saver/saver.go": `package saver

// Two methods, so the fixture's collision baits (Local and Cache,
// each with a lone Save) do NOT implement it: store.Store is the
// only implementation, and the only constructor-built one.
type Saver interface {
	Save() string
	Load() string
}
`,
		"app/app.go": `package app

import (
	"example.com/measure/child"
	"example.com/measure/saver"
	"example.com/measure/store"
)

type Local struct{}

func (l *Local) Save() string { return "local" }

type App struct {
	s saver.Saver
}

func ViaCtor() string {
	st := store.New()
	return st.Save()
}

func ViaCtorTuple() string {
	st, _ := store.NewWithErr()
	return st.Save()
}

func ViaNewStore() string {
	st := store.NewStore()
	return st.Load()
}

func ViaIfaceParam(s saver.Saver) string {
	return s.Save()
}

func ViaField(a *App) string {
	return a.s.Save()
}

func ViaFieldLocal(a *App) string {
	sv := a.s
	return sv.Save()
}

func ViaChild(c *child.Child) string {
	return c.Helper()
}

func ViaChildShadow(c *child.Child) string {
	return c.Shared()
}
`,
	}
}

// brokenFiles is the same stress shapes plus one package whose
// import cannot resolve, so packages.Load fails and refineTypes
// bails: the heuristic graph must stand alone.
func brokenFiles() map[string]string {
	files := map[string]string{
		"go.mod":       "module example.com/broken\n\ngo 1.21\n",
		"junk/junk.go": "package junk\n\nimport _ \"nowhere.invalid/nope\"\n",
	}
	for rel, src := range measureFiles() {
		if rel == "go.mod" {
			continue
		}
		files[rel] = strings.ReplaceAll(src, "example.com/measure", "example.com/broken")
	}
	return files
}

func writeMeasureFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for rel, src := range files {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// edge is one caller -> callee pair of model IDs.
type edge struct{ caller, callee string }

func heuristicEdges(p *Project) map[edge]bool {
	out := map[edge]bool{}
	for _, fn := range p.AllFuncs() {
		for _, c := range fn.Calls {
			out[edge{fn.ID, c}] = true
		}
	}
	return out
}

// measureCallerClass buckets the fixture's caller functions into the
// edge classes the measurement itemizes.
func measureCallerClass(callerID string) string {
	name := callerID[strings.LastIndex(callerID, ".")+1:]
	switch name {
	case "SaveNew", "ViaCtor", "ViaCtorTuple":
		return "ctor-collision"
	case "ViaIfaceParam", "ViaField", "ViaFieldLocal":
		return "interface"
	case "ViaChild":
		return "promotion"
	default:
		return "control"
	}
}

// truthEdges computes the fixture's real call edges with the type
// checker. Interface-method selections have no concrete target in
// the model; for this fixture the sole constructor-built implementer
// of saver.Saver is store.Store, so those edges are scored against
// store.Store's method.
func truthEdges(t *testing.T, dir string, proj *Project) map[edge]bool {
	t.Helper()
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedTypes | packages.NeedSyntax |
			packages.NeedTypesInfo | packages.NeedImports | packages.NeedDeps,
		Dir: dir,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		t.Fatalf("truth load: %v", err)
	}
	out := map[edge]bool{}
	for _, pkg := range pkgs {
		info := pkg.TypesInfo
		if info == nil {
			continue
		}
		rel := strings.TrimPrefix(strings.TrimPrefix(pkg.PkgPath, proj.Module), "/")
		for _, file := range pkg.Syntax {
			for _, decl := range file.Decls {
				fd, ok := decl.(*ast.FuncDecl)
				if !ok || fd.Body == nil {
					continue
				}
				caller := proj.byID[funcID(rel, declRecv(fd), fd.Name.Name)]
				if caller == nil {
					continue
				}
				ast.Inspect(fd.Body, func(n ast.Node) bool {
					call, ok := n.(*ast.CallExpr)
					if !ok {
						return true
					}
					if callee := truthTarget(proj, info, call); callee != nil && callee != caller {
						out[edge{caller.ID, callee.ID}] = true
					}
					return true
				})
			}
		}
	}
	return out
}

// truthTarget maps one call expression to the model function the
// type checker proves it invokes (or the sole implementer, for an
// interface-method selection in this fixture).
func truthTarget(proj *Project, info *types.Info, call *ast.CallExpr) *Func {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		if obj, ok := info.Uses[fun].(*types.Func); ok {
			return proj.funcForObject(obj)
		}
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[fun]; ok {
			obj, ok := sel.Obj().(*types.Func)
			if !ok || obj == nil {
				return nil
			}
			if target := proj.funcForObject(obj); target != nil {
				return target
			}
			// Interface method: no Func in the model. In this
			// fixture saver.Saver's only constructor-built
			// implementation is store.Store.
			if sig, ok := obj.Type().(*types.Signature); ok && sig.Recv() != nil {
				if _, isIface := sig.Recv().Type().Underlying().(*types.Interface); isIface {
					for _, fn := range proj.byMethod[obj.Name()] {
						if fn.Receiver == "Store" {
							return fn
						}
					}
				}
			}
			return nil
		}
		// pkg.Func(): a qualified identifier, not a selection.
		if _, ok := info.Uses[fun.Sel].(*types.Func); ok {
			return proj.funcForObject(info.Uses[fun.Sel].(*types.Func))
		}
	}
	return nil
}

func scoreEdges(name string, got, truth map[edge]bool) {
	classes := map[string][2]int{} // class -> [hits, truthTotal]
	gotByClass := map[string]int{}
	falseByClass := map[string]int{}
	hits, total := 0, 0
	for e := range truth {
		total++
		c := measureCallerClass(e.caller)
		ct := classes[c]
		ct[1]++
		if got[e] {
			hits++
			ct[0]++
		}
		classes[c] = ct
	}
	ngot := 0
	for e := range got {
		ngot++
		c := measureCallerClass(e.caller)
		gotByClass[c]++
		if !truth[e] {
			falseByClass[c]++
		}
	}
	prec := 0.0
	if ngot > 0 {
		prec = float64(hits) / float64(ngot)
	}
	rec := 0.0
	if total > 0 {
		rec = float64(hits) / float64(total)
	}
	fmt.Printf("[%s] edges: truth=%d heuristic=%d hits=%d precision=%.3f recall=%.3f\n",
		name, total, ngot, hits, prec, rec)
	var names []string
	for c := range classes {
		names = append(names, c)
	}
	sort.Strings(names)
	for _, c := range names {
		ct := classes[c]
		fmt.Printf("  class %-15s truth=%d hit=%d got=%d false=%d\n",
			c, ct[1], ct[0], gotByClass[c], falseByClass[c])
	}
}

func TestGraphMeasure(t *testing.T) {
	if os.Getenv("GOLLEMER_MEASURE") == "" {
		t.Skip("measurement report only; set GOLLEMER_MEASURE=1")
	}
	dir := writeMeasureFixture(t, measureFiles())
	proj, err := analyzeDir(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	if proj.Typed {
		t.Fatal("noRefine run must not be typed")
	}
	truth := truthEdges(t, dir, proj)
	scoreEdges("heuristic-only vs type-checker truth (refinable fixture)", heuristicEdges(proj), truth)

	// The same fixture fully analyzed: refinement may only add the
	// edges the heuristics could not prove.
	full, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !full.Typed {
		t.Fatal("fixture should load with type info")
	}
	scoreEdges("heuristic+refined vs truth (refinable fixture)", heuristicEdges(full), truth)
}
