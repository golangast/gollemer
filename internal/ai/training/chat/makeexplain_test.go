package chat

import (
	"bufio"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The ## docs are parsed into explanations per target, with "#"
// continuation lines joined in.
func TestParseMakeDocs(t *testing.T) {
	mk := "## eval: score every brain on its fixed eval suite\neval:\n\tgo test ./...\n\n## smarter: the one-command upgrade\n#   continued description here\nsmarter:\n\tgo run main.go\n\nplain:\n\techo hi\n"
	makeTargetDocs = map[string]string{}
	makeTargetRecipes = map[string][]string{}
	defer func() { makeTargetDocs = map[string]string{}; makeTargetRecipes = map[string][]string{} }()
	parseMakeDocs(mk)

	if got := makeTargetDocs["eval"]; got != "score every brain on its fixed eval suite" {
		t.Errorf("eval doc = %q", got)
	}
	if got := makeTargetDocs["smarter"]; !strings.Contains(got, "continued description here") {
		t.Errorf("smarter doc should join continuation lines, got %q", got)
	}
	if _, ok := makeTargetDocs["plain"]; ok {
		t.Errorf("plain should have no doc entry")
	}
	if len(makeTargetRecipes["eval"]) == 0 || !strings.Contains(makeTargetRecipes["eval"][0], "go test") {
		t.Errorf("eval recipe not captured: %v", makeTargetRecipes["eval"])
	}
}

// The explain patterns catch the natural phrasings and nothing else.
func TestExplainMakeRe(t *testing.T) {
	ok := map[string]string{
		"explain make eval":        "eval",
		"Explain make train-go":    "train-go",
		"what does make eval do":   "eval",
		"what does make eval do?":  "eval",
		"what is make chat":        "chat",
		"what's make flow":         "flow",
		"  explain make eval  ":    "eval",
	}
	for in, want := range ok {
		m := explainMakeRe.FindStringSubmatch(in)
		if m == nil || strings.ToLower(m[1]) != want {
			t.Errorf("explainMakeRe(%q) = %v, want %q", in, m, want)
		}
	}
	bad := []string{
		"",
		"explain make",
		"explain makefiles",
		"what is make-believe",
		"explain this",
		"run make eval",
		"how do i run the evals",
	}
	for _, in := range bad {
		if m := explainMakeRe.FindStringSubmatch(in); m != nil {
			t.Errorf("explainMakeRe(%q) matched %v, want no match", in, m)
		}
	}
}

// Direct "run make" matches only the bare command shape.
func TestDirectRunMakeRe(t *testing.T) {
	ok := []string{"run make eval", "RUN MAKE EVAL", "  run make train-go  "}
	for _, in := range ok {
		if m := directRunMakeRe.FindStringSubmatch(in); m == nil {
			t.Errorf("directRunMakeRe(%q) did not match", in)
		}
	}
	bad := []string{
		"",
		"run make",
		"please run make eval",
		"run make eval && echo hi",
		"explain make eval",
		"make eval",
	}
	for _, in := range bad {
		if m := directRunMakeRe.FindStringSubmatch(in); m != nil {
			t.Errorf("directRunMakeRe(%q) matched, want no match", in)
		}
	}
}

// isMakeTarget knows every allowlist target plus the excluded chat
// entry points.
func TestIsMakeTarget(t *testing.T) {
	old := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{"eval": true}
	defer func() { makeTargetsAllowlist = old }()
	for _, want := range []string{"eval", "chat", "debug-chat"} {
		if !isMakeTarget(want) {
			t.Errorf("isMakeTarget(%q) = false, want true", want)
		}
	}
	if isMakeTarget("nope") {
		t.Errorf("isMakeTarget(nope) = true, want false")
	}
}

// The repo's own Makefile yields docs for the documented targets.
func TestRepoMakefileDocs(t *testing.T) {
	oldDocs, oldRecipes, oldAllow := makeTargetDocs, makeTargetRecipes, makeTargetsAllowlist
	makeTargetDocs = map[string]string{}
	makeTargetRecipes = map[string][]string{}
	makeTargetsAllowlist = map[string]bool{}
	defer func() { makeTargetDocs, makeTargetRecipes, makeTargetsAllowlist = oldDocs, oldRecipes, oldAllow }()
	initMakeAllowlist(repoRoot(t))
	for _, target := range []string{"eval", "chat", "smarter", "explain"} {
		if makeTargetDocs[target] == "" {
			t.Errorf("repo Makefile: no doc parsed for %q", target)
		}
	}
	if len(makeTargetRecipes["eval"]) == 0 {
		t.Errorf("repo Makefile: no recipe parsed for eval")
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	// Walk up until a Makefile is found (test runs in the chat dir).
	for {
		if _, err := os.Stat(filepath.Join(dir, "Makefile")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("repo root with Makefile not found")
		}
		dir = parent
	}
}

// tryExplainMake prints the doc + recipe for a known target (the run
// offer no-ops in tests: stdin is not a terminal).
func TestTryExplainMakeOutput(t *testing.T) {
	dir := t.TempDir()
	mk := "## eval: score every brain on its fixed eval suite\neval:\n\tgo test ./...\n"
	if err := os.WriteFile(filepath.Join(dir, "Makefile"), []byte(mk), 0644); err != nil {
		t.Fatal(err)
	}
	oldDocs, oldRecipes, oldAllow := makeTargetDocs, makeTargetRecipes, makeTargetsAllowlist
	makeTargetDocs = map[string]string{}
	makeTargetRecipes = map[string][]string{}
	makeTargetsAllowlist = map[string]bool{}
	defer func() { makeTargetDocs, makeTargetRecipes, makeTargetsAllowlist = oldDocs, oldRecipes, oldAllow }()
	initMakeAllowlist(dir)

	out := captureStdout(t, func() {
		sc := bufio.NewScanner(strings.NewReader(""))
		if !tryExplainMake("explain make eval", sc, dir, NewConversation()) {
			t.Error("tryExplainMake did not handle 'explain make eval'")
		}
	})
	if !strings.Contains(out, "score every brain") {
		t.Errorf("explain output missing doc:\n%s", out)
	}
	if !strings.Contains(out, "go test ./...") {
		t.Errorf("explain output missing recipe:\n%s", out)
	}
}

// tryDirectRunMake answers an unknown target plainly instead of running.
func TestTryDirectRunMakeUnknown(t *testing.T) {
	oldAllow := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{"eval": true}
	defer func() { makeTargetsAllowlist = oldAllow }()
	out := captureStdout(t, func() {
		sc := bufio.NewScanner(strings.NewReader(""))
		if !tryDirectRunMake("run make nope", sc, ".", NewConversation()) {
			t.Error("tryDirectRunMake did not handle 'run make nope'")
		}
	})
	if !strings.Contains(out, "don't know a make target") {
		t.Errorf("unknown target should be refused plainly:\n%s", out)
	}
}

func captureStdout(t *testing.T, f func()) string {
	t.Helper()
	old := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stdout = w
	f()
	w.Close()
	os.Stdout = old
	data, _ := io.ReadAll(r)
	return string(data)
}

// tryBareMakeTarget fires only on a single token naming a real target.
func TestTryBareMakeTarget(t *testing.T) {
	oldAllow := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{"eval": true, "explain": true}
	defer func() { makeTargetsAllowlist = oldAllow }()
	capture := func(in string) (bool, string) {
		var handled bool
		out := captureStdout(t, func() {
			sc := bufio.NewScanner(strings.NewReader(""))
			handled = tryBareMakeTarget(in, sc, ".", NewConversation())
		})
		return handled, out
	}
	for _, in := range []string{"eval", "EVAL", "eval?", "  explain  "} {
		handled, out := capture(in)
		if !handled || !strings.Contains(out, "run make") {
			t.Errorf("tryBareMakeTarget(%q) = %v, %q; want handled with run offer", in, handled, out)
		}
	}
	for _, in := range []string{"", "help me", "run make eval", "notatarget", "evaluations"} {
		if handled, _ := capture(in); handled {
			t.Errorf("tryBareMakeTarget(%q) handled it, want false", in)
		}
	}
}

// "how do i run the evals" must reach the makefile brain, not social.
func TestRouteEvalToMakefile(t *testing.T) {
	for _, in := range []string{
		"how do i run the evals",
		"run the evals",
		"explain eval",
		"eval",
	} {
		if got := routeDomain(in); got != MakefileDomain {
			t.Errorf("routeDomain(%q) = %q, want %q", in, got, MakefileDomain)
		}
	}
}
