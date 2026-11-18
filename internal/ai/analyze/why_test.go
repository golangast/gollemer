package analyze

import (
	"strings"
	"testing"
)

func whyFixture(t *testing.T) (*Project, *Func, *Func) {
	t.Helper()
	add := &Func{ID: "demo.Counter.Add", Name: "Add", Receiver: "*Counter", File: "c.go", Line: 5, EndLine: 9,
		Sig:      "func (c *Counter) Add(n int)",
		Src:      "func (c *Counter) Add(n int) { c.mu.Lock(); defer c.mu.Unlock(); c.total += n }",
		Callers:  []string{"demo.main", "demo.TestAdd"},
		Doc:      "Add increases the total by n.", Exported: true}
	get := &Func{ID: "demo.Counter.Get", Name: "Get", Receiver: "*Counter", File: "c.go", Line: 11, EndLine: 16,
		Sig:     "func (c *Counter) Get() (int, error)",
		Src:     "func (c *Counter) Get() (int, error) { c.mu.RLock(); defer c.mu.RUnlock(); return c.total, nil }",
		Callers: []string{"demo.main"}}
	mainFn := &Func{ID: "demo.main", Name: "main", IsMain: true, File: "main.go", Line: 3, EndLine: 6,
		Calls: []string{"demo.Counter.Add", "demo.Counter.Get"}}
	testFn := &Func{ID: "demo.TestAdd", Name: "TestAdd", IsTest: true, File: "c_test.go", Line: 3, EndLine: 5,
		Calls: []string{"demo.Counter.Add"}}
	p := &Project{Root: t.TempDir(), Packages: []*Package{{Name: "demo", Funcs: []*Func{add, get, mainFn, testFn}}}}
	p.byID = map[string]*Func{add.ID: add, get.ID: get, mainFn.ID: mainFn, testFn.ID: testFn}
	return p, add, get
}

func TestWhyTextReasonsFromStructure(t *testing.T) {
	p, add, get := whyFixture(t)

	whyAdd := p.WhyText(add)
	for _, want := range []string{"Why Counter.Add", "Stated by the author", "LOCKS", "shared state", "2 methods"} {
		if !strings.Contains(whyAdd, want) {
			t.Errorf("WhyText(Add) missing %q:\n%s", want, whyAdd)
		}
	}
	whyGet := p.WhyText(get)
	for _, want := range []string{"READ lock", "RETURNS AN ERROR", "not in the code"} {
		if !strings.Contains(whyGet, want) {
			t.Errorf("WhyText(Get) missing %q:\n%s", want, whyGet)
		}
	}
	if strings.Contains(whyGet, "Stated by the author") {
		t.Errorf("Get has no doc comment; WhyText must not claim a stated reason:\n%s", whyGet)
	}
}

func TestCompareTextNamesTheDifference(t *testing.T) {
	p, add, get := whyFixture(t)
	out := p.CompareText(add, get)
	for _, want := range []string{"Counter.Add  vs  Counter.Get", "Behavior:", "In short:", "can fail"} {
		if !strings.Contains(out, want) {
			t.Errorf("CompareText missing %q:\n%s", want, out)
		}
	}
}
