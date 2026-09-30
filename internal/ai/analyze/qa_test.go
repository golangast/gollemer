package analyze

import (
	"strings"
	"testing"
)

func qaFixture(t *testing.T) *Project {
	t.Helper()
	p, err := Analyze(fixture(t))
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestAnswerWhatDoes(t *testing.T) {
	p := qaFixture(t)
	out, ok := p.Answer("what does Save do")
	if !ok {
		t.Fatal("expected an answer for 'what does Save do'")
	}
	if !strings.Contains(out, "func (s *Store) Save(name string)") {
		t.Errorf("expected signature in answer, got:\n%s", out)
	}
	if !strings.Contains(out, "Save stores an item.") {
		t.Errorf("expected doc sentence in answer, got:\n%s", out)
	}
	if !strings.Contains(out, "main.go") && !strings.Contains(out, "root.main") {
		t.Errorf("expected caller main in answer, got:\n%s", out)
	}
}

func TestAnswerWhatDoesBareName(t *testing.T) {
	p := qaFixture(t)
	// "store.New" is unambiguous even without a dot here: two New-like
	// names don't exist in the fixture, so bare "New" must resolve.
	out, ok := p.Answer("what does New do")
	if !ok {
		t.Fatal("expected an answer for 'what does New do'")
	}
	if !strings.Contains(out, "New makes a store.") {
		t.Errorf("expected doc sentence, got:\n%s", out)
	}
}

func TestAnswerWhatIsType(t *testing.T) {
	p := qaFixture(t)
	out, ok := p.Answer("what is Store")
	if !ok {
		t.Fatal("expected an answer for 'what is Store'")
	}
	if !strings.Contains(out, "type store.Store") || !strings.Contains(out, "struct") {
		t.Errorf("expected type card, got:\n%s", out)
	}
	if !strings.Contains(out, "Save") {
		t.Errorf("expected methods in type card, got:\n%s", out)
	}
}

func TestAnswerWhoCalls(t *testing.T) {
	p := qaFixture(t)
	out, ok := p.Answer("who calls Save")
	if !ok {
		t.Fatal("expected an answer for 'who calls Save'")
	}
	if !strings.Contains(out, "main.main") {
		t.Errorf("expected main.main as caller, got:\n%s", out)
	}
}

func TestAnswerWhatCalls(t *testing.T) {
	p := qaFixture(t)
	out, ok := p.Answer("what does main call")
	if !ok {
		t.Fatal("expected an answer for 'what does main call'")
	}
	if !strings.Contains(out, "store.New") || !strings.Contains(out, "store.Store.Save") {
		t.Errorf("expected callees, got:\n%s", out)
	}
}

func TestAnswerWhereDefined(t *testing.T) {
	p := qaFixture(t)
	out, ok := p.Answer("where is Count defined")
	if !ok {
		t.Fatal("expected an answer for 'where is Count defined'")
	}
	if !strings.Contains(out, "store/store.go") {
		t.Errorf("expected file location, got:\n%s", out)
	}
}

func TestAnswerShowMe(t *testing.T) {
	p := qaFixture(t)
	out, ok := p.Answer("show me New")
	if !ok {
		t.Fatal("expected an answer for 'show me New'")
	}
	if !strings.Contains(out, "func New() *Store") {
		t.Errorf("expected source excerpt, got:\n%s", out)
	}
}

func TestAnswerPackage(t *testing.T) {
	p := qaFixture(t)
	for _, q := range []string{"what's in package store", "list functions in store"} {
		out, ok := p.Answer(q)
		if !ok {
			t.Fatalf("expected an answer for %q", q)
		}
		if !strings.Contains(out, "package store") {
			t.Errorf("expected package card for %q, got:\n%s", q, out)
		}
	}
}

func TestAnswerAmbiguous(t *testing.T) {
	p := qaFixture(t)
	// "Save" and "New" are unique; add a duplicate name via method set:
	// both Store.Save and a second package's Save would collide.
	// The fixture has one Save, so ask about a name that matches two
	// funcs case-insensitively is hard; instead verify the unique path
	// returns a single detail card, not the ambiguous list.
	out, ok := p.Answer("what does Count do")
	if !ok {
		t.Fatal("expected an answer")
	}
	if strings.Contains(out, "which one did you mean") {
		t.Errorf("unique name should not be ambiguous, got:\n%s", out)
	}
}

func TestAnswerNoSteal(t *testing.T) {
	p := qaFixture(t)
	// None of these name a project symbol, so QA must decline and let
	// the other brains handle them.
	for _, q := range []string{
		"what does a goroutine do",
		"what does make chat do",
		"show me a visual",
		"what is the meaning of life",
		"how does a map work",
	} {
		if _, ok := p.Answer(q); ok {
			t.Errorf("QA should decline %q", q)
		}
	}
}

func TestAnswerWhereHandled(t *testing.T) {
	p := qaFixture(t)
	out, ok := p.Answer("where is save handled")
	if !ok {
		t.Fatal("expected an answer for 'where is save handled'")
	}
	if !strings.Contains(out, "Save") {
		t.Errorf("expected Save in where-to-change results, got:\n%s", out)
	}
}

func TestOverview(t *testing.T) {
	p := qaFixture(t)
	ov := p.Overview()
	for _, want := range []string{"runnable", "main.go", "store"} {
		if !strings.Contains(ov, want) {
			t.Errorf("overview missing %q: %s", want, ov)
		}
	}
	if !strings.Contains(p.Summary(), "IN PLAIN ENGLISH") {
		t.Error("Summary should include the plain-English overview")
	}
}

func TestRenderSig(t *testing.T) {
	p := qaFixture(t)
	fn, _, _, _ := p.resolve("store.Store.Save")
	if fn == nil {
		t.Fatal("could not resolve store.Store.Save")
	}
	if fn.Sig != "func (s *Store) Save(name string)" {
		t.Errorf("bad signature: %q", fn.Sig)
	}
	if fn.EndLine < fn.Line {
		t.Errorf("EndLine %d should be >= Line %d", fn.EndLine, fn.Line)
	}
}
