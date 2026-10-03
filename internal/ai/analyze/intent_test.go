package analyze

import (
	"strings"
	"testing"
)

func TestVerbPhrase(t *testing.T) {
	for name, want := range map[string][2]string{
		"DeleteFile":      {"deletes a file", "delete a file"},
		"CopyOrCloneFile": {"copies or clones a file", "copy or clone a file"},
		"Run":             {"runs", "run"},
		"Get":             {"gets", "get"},
		"ServeHTTP":       {"serves an http", "serve an http"},
		"CreateSnapshot":  {"creates a snapshot", "create a snapshot"},
		"CheckForWin":     {"checks for win", "check for win"},
		"Backup":          {"backs up", "back up"},
	} {
		want3, wantInf := want[0], want[1]
		if got := verbPhrase(name, true); got != want3 {
			t.Errorf("verbPhrase(%q, true) = %q, want %q", name, got, want3)
		}
		if got := verbPhrase(name, false); got != wantInf {
			t.Errorf("verbPhrase(%q, false) = %q, want %q", name, got, wantInf)
		}
	}
}

func TestThirdPerson(t *testing.T) {
	for inf, want := range map[string]string{
		"copy": "copies", "watch": "watches", "delete": "deletes",
		"fix": "fixes", "run": "runs", "back up": "backs up",
	} {
		if got := thirdPerson(inf); got != want {
			t.Errorf("thirdPerson(%q) = %q, want %q", inf, got, want)
		}
	}
}

func TestTestIntent(t *testing.T) {
	if got := testIntent("TestCopyDir_nested"); got != "Tests CopyDir (nested)." {
		t.Errorf("testIntent = %q", got)
	}
	if got := testIntent("TestSave"); got != "Tests Save." {
		t.Errorf("testIntent = %q", got)
	}
}

func TestIntentEval(t *testing.T) {
	p := understandFixture(t)
	byName := map[string]*Func{}
	for _, fn := range p.byID {
		byName[fn.Name] = fn
	}
	for name, want := range map[string]string{
		// Verb + direct effects + the alias guard.
		"Copy": "Copies. Writes files (os.WriteFile) and reads files (os.ReadFile).",
		"Run":  "Runs. Checks the `dry` flag.",
	} {
		if got := byName[name].Intent(); got != want {
			t.Errorf("%s.Intent() = %q, want %q", name, got, want)
		}
	}
	// Doc comment wins over synthesis.
	if got := byName["Greet"].Intent(); got != "Greet returns a greeting for the given name." {
		t.Errorf("Greet.Intent() = %q", got)
	}
}

func TestIntentQA(t *testing.T) {
	p := understandFixture(t)
	for q, want := range map[string]string{
		"what is Copy for":       "Copies. Writes files (os.WriteFile) and reads files (os.ReadFile).",
		"why does Run call Copy": "calls backup.Copy to copy.",
	} {
		out, ok := p.Answer(q)
		if !ok {
			t.Errorf("%q: not handled", q)
			continue
		}
		if !strings.Contains(out, want) {
			t.Errorf("%q: missing %q:\n%s", q, want, out)
		}
	}
	// A call that doesn't exist says so honestly.
	if out, ok := p.Answer("why does Copy call Run"); !ok || !strings.Contains(out, "doesn't call") {
		t.Errorf("why does Copy call Run: got %q, ok=%v", out, ok)
	}
}
