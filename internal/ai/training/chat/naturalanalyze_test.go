package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The reported failure (John's screenshot, 4e77ef9 build): in the
// gollemer folder he typed "what does https://github.com/ugorji/go
// do" and got a summary of gollemer itself — "Let me look at the
// project in this folder first…". The intent classifier claimed
// the line as summarize (0.864) inside tryIntentDispatch, which
// runs before tryDeterministicAnswer — the only place
// tryNaturalAnalyze was hooked in — so the NL repo routing never
// saw the line. The other question shapes were scattered across
// analyze/walkthrough classifications, and "explain <url>" /
// "how does <url> work" matched no intent word in the detector at
// all. The fix claims repo-URL questions deterministically in
// tryIntentDispatch, before classification (see clone.go).

// seedRepoCache plants a tiny ugorji-shaped repo in the
// codebase-maps cache under a fake HOME, so ensureGitHubRepo finds
// it "already cloned" and no network or git is involved.
func seedRepoCache(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	root := filepath.Join(home, "workspace", "codebase-maps", "ugorji-go")
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module github.com/ugorji/go\n\ngo 1.21\n")
	write("codec/codec.go", `package codec

// FrobnicateZanzibar encodes a value the zanzibar way.
func FrobnicateZanzibar() int { return 42 }

// EncodeAll encodes every value the zanzibar way.
func EncodeAll() int { return FrobnicateZanzibar() }
`)
	return root
}

// decoyProject is the project the chat is "in" — the one the bug
// answered about. Its folder name is distinctive so its report is
// recognizable on sight.
func decoyProject(t *testing.T) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "decoyproj")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/decoy\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	src := `package decoyproj

// DecoyMarker does decoy things.
func DecoyMarker() int { return 1 }
`
	if err := os.WriteFile(filepath.Join(root, "decoy.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// Every question shape about a repo URL must be answered about
// THAT repo — with the real classifier loaded, proving the claim
// happens before it — and the repo becomes the current project.
func TestRepoURLQuestionShapes(t *testing.T) {
	withRealIntentModel(t)
	cacheRoot := seedRepoCache(t)
	lines := []string{
		"what does https://github.com/ugorji/go do", // the screenshot line
		"what is https://github.com/ugorji/go",
		"how does https://github.com/ugorji/go work",
		"explain https://github.com/ugorji/go",
		"tell me about https://github.com/ugorji/go",
		"analyze https://github.com/ugorji/go", // the already-working family
		"what does github.com/ugorji/go do",    // scheme-less form
	}
	for _, line := range lines {
		withPickState(t)
		withEditsProject(t, decoyProject(t))
		var handled bool
		out := captureStdout(t, func() { handled = tryIntentDispatch(line, nil) })
		if !handled {
			t.Errorf("%q not handled, output:\n%s", line, out)
			continue
		}
		if !strings.Contains(out, "ugorji-go") || !strings.Contains(out, "FrobnicateZanzibar") {
			t.Errorf("%q not answered about the URL's repo, output:\n%s", line, out)
		}
		if strings.Contains(out, "decoyproj") || strings.Contains(out, "DecoyMarker") {
			t.Errorf("%q answered about the current folder instead, output:\n%s", line, out)
		}
		if lastAnalyzeRoot != cacheRoot {
			t.Errorf("%q: current project root = %q, want the cached repo %q", line, lastAnalyzeRoot, cacheRoot)
		}
	}
}

// The detector itself: repo-root URL + intent words, nothing else.
func TestRepoQuestionURLDetection(t *testing.T) {
	accept := []string{
		"what does https://github.com/ugorji/go do",
		"what does https://github.com/ugorji/go do?",
		"analyze https://github.com/ugorji/go.",
		"look at https://github.com/ugorji/go.git",
		"take a look at this repo: https://github.com/ugorji/go",
		"check out (https://github.com/ugorji/go)",
		"how does https://github.com/ugorji/go/ work",
	}
	for _, line := range accept {
		url, ok := repoQuestionURL(line)
		if !ok || url != "https://github.com/ugorji/go" {
			t.Errorf("repoQuestionURL(%q) = (%q, %v), want the ugorji repo URL", line, url, ok)
		}
	}
	reject := []string{
		// Issue/PR/commit URLs belong to the where flow.
		"where for https://github.com/ugorji/go/issues/12",
		"look at https://github.com/ugorji/go/pull/3",
		"explain https://github.com/ugorji/go/commit/abc123",
		// A page inside the repo is not the repo root.
		"what does https://github.com/ugorji/go/blob/main/codec/codec.go do",
		"analyze https://github.com/ugorji/go/tree/main",
		// Not a GitHub repo URL at all.
		"what does https://go.dev/doc do",
		"what does github.com/ugorji do",
		// A bare mention names no question.
		"https://github.com/ugorji/go",
		"my code lives at https://github.com/ugorji/go btw",
		// A dotted symbol is not a URL (symbolpick's territory).
		"what does codec.decoderBincBytes.mapEnd do",
	}
	for _, line := range reject {
		if url, ok := repoQuestionURL(line); ok {
			t.Errorf("repoQuestionURL(%q) = %q, want no claim", line, url)
		}
		if _, ok := tryNaturalAnalyze(line); ok {
			t.Errorf("tryNaturalAnalyze(%q) claimed a line it must leave alone", line)
		}
	}
}

// An issue URL line must not be claimed by the repo-URL dispatch:
// it falls through to the intent layer's normal handling (the
// where flow owns it in the chat loops).
func TestRepoURLDispatchLeavesIssueURLs(t *testing.T) {
	withPickState(t)
	seedRepoCache(t)
	if tryRepoURLDispatch("where for https://github.com/ugorji/go/issues/12") {
		t.Error("issue URL claimed by repo-URL dispatch")
	}
	if tryRepoURLDispatch("what does codec.decoderBincBytes.mapEnd do") {
		t.Error("dotted-symbol line claimed by repo-URL dispatch")
	}
}
