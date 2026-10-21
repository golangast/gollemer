package analyze

import (
	"strings"
	"testing"
)

func diffFixture() (oldP, newP *Project) {
	mk := func(funcs ...*Func) *Project {
		p := &Project{Root: "/x", Packages: []*Package{{Name: "demo", Funcs: funcs}}}
		p.byID = map[string]*Func{}
		for _, f := range funcs {
			p.byID[f.ID] = f
		}
		return p
	}
	keep := &Func{ID: "demo.Keep", Name: "Keep", File: "a.go", Line: 1, EndLine: 3,
		Sig: "func Keep() int", Src: "func Keep() int { return 1 }"}
	gone := &Func{ID: "demo.Gone", Name: "Gone", File: "a.go", Line: 5, EndLine: 7,
		Sig: "func Gone()", Src: "func Gone() {}", Callers: []string{"demo.Keep"}}
	changing := &Func{ID: "demo.Counter.Add", Name: "Add", Receiver: "*Counter", File: "c.go", Line: 3, EndLine: 6,
		Sig: "func (c *Counter) Add(n int)", Src: "func (c *Counter) Add(n int) { c.total += n }"}
	oldP = mk(keep, gone, changing)

	keep2 := *keep
	changing2 := &Func{ID: "demo.Counter.Add", Name: "Add", Receiver: "*Counter", File: "c.go", Line: 3, EndLine: 7,
		Sig:     "func (c *Counter) Add(n int, strict bool) error",
		Src:     "func (c *Counter) Add(n int, strict bool) error { c.mu.Lock(); defer c.mu.Unlock(); c.total += n; return nil }",
		Callers: []string{"demo.Keep"}}
	fresh := &Func{ID: "demo.Store", Name: "Store", Exported: true, File: "b.go", Line: 1, EndLine: 4,
		Sig: "func Store(v int)", Src: "func Store(v int) {}"}
	newP = mk(&keep2, changing2, fresh)
	return oldP, newP
}

func TestDiffProjectsMeaningfulOnly(t *testing.T) {
	oldP, newP := diffFixture()
	events := DiffProjects(oldP, newP)
	byKind := map[string]WatchEvent{}
	for _, e := range events {
		byKind[e.Kind] = e
	}
	if _, ok := byKind["removed-func"]; !ok {
		t.Errorf("Gone had a caller — its removal is an event: %+v", events)
	}
	sig, ok := byKind["sig-changed"]
	if !ok || !strings.Contains(sig.Detail, "1 caller") {
		t.Errorf("Add's signature change should be an event naming the callers at stake: %+v", sig)
	}
	beh, ok := byKind["behavior-changed"]
	if !ok || !strings.Contains(beh.Detail, "locks") || !strings.Contains(beh.Detail, "can fail") {
		t.Errorf("Add gained locks + can-fail: %+v", beh)
	}
	if _, ok := byKind["new-func"]; !ok {
		t.Errorf("Store appearing should be an event: %+v", events)
	}
	if _, ok := byKind["untested-new"]; !ok {
		t.Errorf("Store is exported with no test — should be flagged: %+v", events)
	}
	// No change, no events.
	if ev := DiffProjects(oldP, oldP); len(ev) != 0 {
		t.Errorf("identical snapshots must be silent, got %+v", ev)
	}
}
