package analyze

// Unit tests for the sharpened heuristic resolution (methodset.go):
// the constructor table, promoted method sets with Go's depth and
// ambiguity rules, constructor/interface/field dispatch, the
// refusal cases where the new machinery must NOT guess, and the
// load-broken module where the heuristic graph stands alone.

import (
	"strings"
	"testing"
)

// unitFiles is a compiling module exercising the constructor and
// dispatch shapes in isolation.
func unitFiles() map[string]string {
	return map[string]string{
		"go.mod": "module example.com/unit\n\ngo 1.21\n",
		"saver/saver.go": `package saver

type Saver interface {
	Save() string
	Load() string
}

type Mer interface {
	M() string
}
`,
		"store/store.go": `package store

import "example.com/unit/saver"

type Store struct{}

func New() *Store { return &Store{} }

func NewStore() Store { return Store{} }

func NewWithErr() (*Store, error) { return &Store{}, nil }

func NewThing() *Cache { return &Cache{} }

func NewSaver() saver.Saver { return &Store{} }

func Make() *Store { return &Store{} }

func (s *Store) Save() string { return "store" }
func (s *Store) Load() string { return "load" }

type Cache struct{}

func (c *Cache) Save() string { return "cache" }
`,
		"shapes/shapes.go": `package shapes

import "example.com/unit/saver"

type Inner struct{}

func (i Inner) VM() string  { return "vm" }
func (i *Inner) PM() string { return "pm" }

type OuterV struct{ Inner }
type OuterP struct{ *Inner }

type Mid struct{ Inner }
type Top struct{ Mid }

type Shad struct{ Inner }

func (s *Shad) VM() string { return "shad" }

type Base struct{}

func (b *Base) DM() string { return "dm" }

type A1 struct{ Base }
type A2 struct{ Base }
type D struct {
	A1
	A2
}

type Deep struct{ A2 }
type E struct {
	A1
	Deep
}

type HasM struct{}

func (h *HasM) M() string { return "m" }

type V struct{ HasM }

type S struct {
	saver.Mer
	V
}

type GBase[T any] struct{}

func (g *GBase[T]) GM() string { return "gm" }

type GOuter struct{ GBase[int] }

func CallD(d *D) string { return d.DM() }
`,
		"demo/demo.go": `package demo

import (
	"example.com/unit/saver"
	"example.com/unit/store"
)

type Local struct{}

func (l *Local) Save() string { return "local" }

func ViaValue() string {
	s := store.NewStore()
	return s.Save()
}

func ViaWithErr() string {
	s, err := store.NewWithErr()
	_ = err
	return s.Save()
}

func ViaVarDecl() string {
	var s saver.Saver = store.New()
	return s.Save()
}

type Holder struct {
	st *store.Store
}

func ViaConcreteField(h *Holder) string {
	return h.st.Save()
}

type Wrap struct {
	store.Store
}

func ViaEmbeddedField(w *Wrap) string {
	return w.Store.Save()
}
`,
		"demo2/demo2.go": `package demo2

import "example.com/unit/store"

// Make is not a constructor (not New-named), so no type is proven;
// the legacy varPkg path — s came from store — still resolves the
// call into the store package.
func ViaMake() string {
	s := store.Make()
	return s.Save()
}
`,
	}
}

// refuseFiles: interfaces whose implementations the index must not
// guess at — two implementers for Dual, and a Solo implementer no
// constructor builds.
func refuseFiles() map[string]string {
	return map[string]string{
		"go.mod": "module example.com/refuse\n\ngo 1.21\n",
		"ifaces/ifaces.go": `package ifaces

type Dual interface{ Do() string }

type Solo interface{ Go() string }
`,
		"impls/impls.go": `package impls

type A struct{}

func NewA() *A { return &A{} }

func (a *A) Do() string { return "a" }

type B struct{}

func (b *B) Do() string { return "b" }

type C struct{}

func (c *C) Go() string { return "c" }
`,
		"use/use.go": `package use

import "example.com/refuse/ifaces"

func UseDual(d ifaces.Dual) string { return d.Do() }

func UseSolo(s ifaces.Solo) string { return s.Go() }
`,
	}
}

func hasCall(t *testing.T, p *Project, caller, callee string) bool {
	t.Helper()
	fn := p.FuncByID(caller)
	if fn == nil {
		t.Fatalf("no such function %s", caller)
	}
	for _, c := range fn.Calls {
		if c == callee {
			return true
		}
	}
	return false
}

func wantCall(t *testing.T, p *Project, caller, callee string) {
	t.Helper()
	if !hasCall(t, p, caller, callee) {
		t.Errorf("%s should call %s; calls: %v", caller, callee, p.FuncByID(caller).Calls)
	}
}

func wantNoCall(t *testing.T, p *Project, caller, callee string) {
	t.Helper()
	if hasCall(t, p, caller, callee) {
		t.Errorf("%s must not call %s; calls: %v", caller, callee, p.FuncByID(caller).Calls)
	}
}

func TestConstructorTable(t *testing.T) {
	dir := writeMeasureFixture(t, unitFiles())
	p, err := analyzeDir(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	store := p.typeByKey[typeKey{"store", "Store"}]
	cache := p.typeByKey[typeKey{"store", "Cache"}]
	if store == nil || cache == nil {
		t.Fatal("fixture types missing")
	}
	for id, want := range map[string]*Type{
		"store.New":        store, // New returning *Store
		"store.NewStore":   store, // New<Type> returning Store by value
		"store.NewWithErr": store, // the (T, error) form
		"store.NewThing":   cache, // the return type rules, not the name
	} {
		if got := p.ctorTable[id]; got != want {
			t.Errorf("ctorTable[%s] = %v, want %s", id, got, want.Name)
		}
	}
	for _, id := range []string{"store.Make", "store.NewSaver"} {
		if _, ok := p.ctorTable[id]; ok {
			t.Errorf("ctorTable must not contain %s (not a New-named constructor / no struct product)", id)
		}
	}
	if !p.ctorBuilds[store] || !p.ctorBuilds[cache] {
		t.Error("ctorBuilds should hold Store and Cache")
	}
}

func TestMethodSetPromotion(t *testing.T) {
	dir := writeMeasureFixture(t, unitFiles())
	p, err := analyzeDir(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	typ := func(name string) *Type {
		t.Helper()
		tt := p.typeByKey[typeKey{"shapes", name}]
		if tt == nil {
			t.Fatalf("no shapes.%s", name)
		}
		return tt
	}
	hit := func(t *Type, ptr bool, name string) (setHit, bool) {
		h, ok := p.methodSet(t, ptr)[name]
		return h, ok
	}

	// Value embedding: the value set gets value-receiver methods
	// only; the pointer set gets both.
	outerV := typ("OuterV")
	if h, ok := hit(outerV, false, "VM"); !ok || h.fn == nil || h.fn.ID != "shapes.Inner.VM" || h.depth != 1 {
		t.Errorf("OuterV value set VM = %+v, %v; want Inner.VM at depth 1", h, ok)
	}
	if _, ok := hit(outerV, false, "PM"); ok {
		t.Error("OuterV value set must not contain the pointer-receiver PM")
	}
	if h, ok := hit(outerV, true, "PM"); !ok || h.fn == nil || h.fn.ID != "shapes.Inner.PM" {
		t.Errorf("OuterV pointer set PM = %+v, %v; want Inner.PM", h, ok)
	}
	// Pointer embedding contributes the pointer set to both.
	outerP := typ("OuterP")
	if h, ok := hit(outerP, false, "PM"); !ok || h.fn == nil || h.fn.ID != "shapes.Inner.PM" {
		t.Errorf("OuterP value set PM = %+v, %v; want Inner.PM (pointer field)", h, ok)
	}
	// Nested embedding accumulates depth.
	if h, ok := hit(typ("Top"), true, "VM"); !ok || h.fn == nil || h.depth != 2 {
		t.Errorf("Top VM = %+v, %v; want Inner.VM at depth 2", h, ok)
	}
	// An own method shadows the promoted one.
	if h, ok := hit(typ("Shad"), true, "VM"); !ok || h.fn == nil || h.fn.ID != "shapes.Shad.VM" || h.depth != 0 {
		t.Errorf("Shad VM = %+v, %v; want Shad.VM at depth 0", h, ok)
	}
	// Diamond: the same method at the same depth via two paths is
	// ambiguous — present in the set with no target.
	if h, ok := hit(typ("D"), true, "DM"); !ok || h.fn != nil {
		t.Errorf("D DM = %+v, %v; want an ambiguous (nil-fn) hit", h, ok)
	}
	// Unequal depths are not ambiguous: the shallowest path wins.
	if h, ok := hit(typ("E"), true, "DM"); !ok || h.fn == nil || h.fn.ID != "shapes.Base.DM" || h.depth != 2 {
		t.Errorf("E DM = %+v, %v; want Base.DM at depth 2", h, ok)
	}
	// An embedded interface's method at depth 1 shadows a concrete
	// promotion at depth 2 — and has no concrete target itself.
	s := typ("S")
	if h, ok := hit(s, true, "M"); !ok || h.fn != nil {
		t.Errorf("S M = %+v, %v; want an interface (nil-fn) hit", h, ok)
	}
	if got := p.lookupMethod(s, "M"); got != nil {
		t.Errorf("lookupMethod(S, M) = %s, want nil", got.ID)
	}
	// Generic embedding resolves through the instantiation.
	if got := p.lookupMethod(typ("GOuter"), "GM"); got == nil || got.ID != "shapes.GBase.GM" {
		t.Errorf("lookupMethod(GOuter, GM) = %v, want shapes.GBase.GM", got)
	}
	// ...and the ambiguous diamond produces no call edge at all.
	fn := p.FuncByID("shapes.CallD")
	if fn == nil {
		t.Fatal("no shapes.CallD")
	}
	if len(fn.Calls) != 0 {
		t.Errorf("CallD (ambiguous selector) must have no call edges, got %v", fn.Calls)
	}
}

func TestConstructorDispatch(t *testing.T) {
	dir := writeMeasureFixture(t, unitFiles())
	p, err := analyzeDir(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	wantCall(t, p, "demo.ViaValue", "store.Store.Save")   // NewStore() Store by value
	wantCall(t, p, "demo.ViaWithErr", "store.Store.Save") // (T, error) tuple form
	wantCall(t, p, "demo2.ViaMake", "store.Store.Save")   // not a constructor: the legacy package path still lands it
	wantCall(t, p, "demo.ViaVarDecl", "store.Store.Save") // var s Saver = store.New()
	wantNoCall(t, p, "demo.ViaVarDecl", "demo.Local.Save")
	wantCall(t, p, "demo.ViaConcreteField", "store.Store.Save") // field of concrete type
	wantCall(t, p, "demo.ViaEmbeddedField", "store.Store.Save") // field that IS an embedded type
}

func TestInterfaceDispatchRefusals(t *testing.T) {
	dir := writeMeasureFixture(t, refuseFiles())
	p, err := analyzeDir(dir, true)
	if err != nil {
		t.Fatal(err)
	}
	// Two implementers: no guess at either.
	wantNoCall(t, p, "use.UseDual", "impls.A.Do")
	wantNoCall(t, p, "use.UseDual", "impls.B.Do")
	// One implementer, but no constructor builds it: no guess.
	wantNoCall(t, p, "use.UseSolo", "impls.C.Go")
}

// TestHeuristicBrokenModule is the load-broken fixture: one package
// imports a module that cannot resolve, so refineTypes bails
// (p.Typed stays false) and the edges below can only come from the
// heuristic pass.
func TestHeuristicBrokenModule(t *testing.T) {
	dir := writeMeasureFixture(t, brokenFiles())
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if p.Typed {
		t.Fatal("fixture must defeat the typed refinement (unresolvable import)")
	}
	if len(p.ParseErr) != 0 {
		t.Fatalf("analyzer itself should still parse everything, got %v", p.ParseErr)
	}
	// Constructor collision: the dispatch lands on Store, not on
	// the caller's same-named bait or the store package's Cache.
	wantCall(t, p, "app.ViaCtor", "store.Store.Save")
	wantNoCall(t, p, "app.ViaCtor", "app.Local.Save")
	wantCall(t, p, "store.SaveNew", "store.Store.Save")
	wantNoCall(t, p, "store.SaveNew", "store.Cache.Save")
	// Interface dispatch through a parameter and through a field.
	wantCall(t, p, "app.ViaIfaceParam", "store.Store.Save")
	wantCall(t, p, "app.ViaField", "store.Store.Save")
	wantCall(t, p, "app.ViaFieldLocal", "store.Store.Save")
	// Cross-package embedded promotion.
	wantCall(t, p, "app.ViaChild", "base.Base.Helper")
	if !strings.Contains(p.FuncByID("app.ViaChild").PkgDir, "app") {
		t.Fatal("fixture sanity: ViaChild should live in app")
	}
}
