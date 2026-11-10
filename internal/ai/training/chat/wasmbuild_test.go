package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The build grammar, at parse: the verb is "build" and the
// wasm word must be there — the full form, the short form
// (no "app"), articles optional, the folder tail in the
// family's shapes, politeness stripped.
func TestParseBuildWasmAppClaims(t *testing.T) {
	rows := []struct {
		line      string
		folder    string
		hasFolder bool
	}{
		{"build the wasm app in jim", "jim", true},
		{"build the wasm app", "", false},
		{"build a wasm app in jim", "jim", true},
		{"build wasm app in jim/temp", "jim/temp", true},
		{"build the wasm in jim", "jim", true},
		{"build the wasm", "", false},
		{"build the webassembly app in jim", "jim", true},
		{"build the wasm app in the folder jim", "jim", true},
		{"build the wasm app into jim", "jim", true},
		{"can you build the wasm app in jim", "jim", true},
		{"please build the wasm app in jim", "jim", true},
		{"build the wasm app in jim.", "jim", true},
	}
	for _, tc := range rows {
		folder, hasFolder, ok := parseBuildWasmApp(tc.line)
		if !ok {
			t.Errorf("%q: no claim", tc.line)
			continue
		}
		if folder != tc.folder || hasFolder != tc.hasFolder {
			t.Errorf("%q: parsed folder=%q hasFolder=%v, want %q/%v", tc.line, folder, hasFolder, tc.folder, tc.hasFolder)
		}
	}
}

// The shapes the build grammar must leave alone: above all
// the creation verbs that are not "build" (they belong to
// the create claim outright), the book's "build a …" lines
// (no wasm word), and the neighbours' sentences.
func TestParseBuildWasmAppDeclines(t *testing.T) {
	for _, line := range []string{
		"build the app in jim",
		"build a server that serves index.html",
		"build a server in jim",
		"build a database connection",
		"add a wasm app in jim",
		"create a wasm app in jim",
		"make a wasm app in jim",
		"put a wasm app in jim",
		"rebuild the wasm app in jim",
		"build the wasm app in jim extra",
		"build the wasm app in jim/main.go",
		"build the wasm app at jim/temp/index.html",
		"build the wasm app in",
		"build",
		"what is wasm",
		"when the button is clicked, show the card in jim",
		"add a card to jim",
	} {
		if folder, hasFolder, ok := parseBuildWasmApp(line); ok {
			t.Errorf("%q claimed as a wasm build (folder=%q hasFolder=%v)", line, folder, hasFolder)
		}
	}
}

// The resolution gate: the claim fires only on a folder
// that already holds a wasm app. A folder with a plain
// program in it, and a folder that isn't there at all, are
// declined silently — nothing printed, nothing run — and
// the line falls through to the create claim, whose
// behaviour on those lines is exactly what it was before
// this claim existed (the two-step for a missing folder,
// the already-there refusal for a taken main.go).
func TestBuildWasmAppResolutionGate(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jake/main.go": "package main\n\nfunc main() {}\n",
	}))
	profile := &UserProfile{}
	for _, line := range []string{"build the wasm app in jake", "build the wasm app in ghost", "build the wasm app"} {
		var handled bool
		out := captureStdout(t, func() { handled = runBuildWasmApp(line, yes, profile) })
		if handled {
			t.Errorf("%q: the build claim fired without an app in place:\n%s", line, out)
		}
		if out != "" {
			t.Errorf("%q: the build claim printed on a miss:\n%s", line, out)
		}
	}
	// Fall-through, the missing folder: the create claim's
	// two-step, in its own words.
	out := captureStdout(t, func() { claimChain("build the wasm app in ghost", yes, profile) })
	if !strings.Contains(out, "there's no folder ghost in this project") || !strings.Contains(out, "create a folder ghost") {
		t.Errorf("missing-folder fall-through is not the create two-step:\n%s", out)
	}
	if strings.Contains(out, "Build it?") {
		t.Errorf("a build gate was offered for a missing folder:\n%s", out)
	}
	// Fall-through, the plain folder: the create claim
	// refuses on the taken file, and nothing is built.
	out = captureStdout(t, func() { claimChain("build the wasm app in jake", yes, profile) })
	if !strings.Contains(out, "already there") {
		t.Errorf("plain-folder fall-through is not the create refusal:\n%s", out)
	}
	if _, err := os.Stat("jake/main.wasm"); !os.IsNotExist(err) {
		t.Error("a main.wasm appeared in a folder that holds no wasm app")
	}
}

// The folderless sentence with no app in the session
// declines too — and through the chain it meets the create
// claim's folder question, exactly as it did before.
func TestBuildWasmAppFolderlessColdSession(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("build the wasm app", yes, profile) })
	if !strings.Contains(out, "which folder should the app live in?") || !hasPendingQuestion() {
		t.Errorf("the folderless build sentence did not fall through to the create question:\n%s", out)
	}
}

// The session reading and the gate: with an app in place
// (created through the chain, so the session touched it),
// the folderless sentence reads as that app — the reading
// printed — and the gate names the shared build command
// and the folder. A "no" builds nothing.
func TestBuildWasmAppSessionReadingAndGate(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	var prompts []string
	ask := func(p string) (string, error) {
		prompts = append(prompts, p)
		return "n", nil
	}
	var handled bool
	out := captureStdout(t, func() { handled = runBuildWasmApp("build the wasm app", ask, profile) })
	if !handled {
		t.Fatalf("the folderless build sentence was not claimed with an app in the session:\n%s", out)
	}
	if !strings.Contains(out, "Reading the app as the wasm app in jim — the last wasm app this session touched.") {
		t.Errorf("the session reading was not printed:\n%s", out)
	}
	if !strings.Contains(out, "OK — not built.") {
		t.Errorf("the declined gate was not answered plainly:\n%s", out)
	}
	if len(prompts) != 1 || !strings.Contains(prompts[0], "Build it? "+analyze.WasmBuildCommand) || !strings.Contains(prompts[0], "(in jim)") {
		t.Errorf("gate prompt = %v, want the shared build command and the folder", prompts)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "main.wasm")); !os.IsNotExist(err) {
		t.Error("a main.wasm appeared after a declined gate")
	}
	// The named form's gate names the same command and
	// folder.
	prompts = nil
	out = captureStdout(t, func() { handled = runBuildWasmApp("build the wasm app in jim", ask, profile) })
	if !handled || len(prompts) != 1 || !strings.Contains(prompts[0], analyze.WasmBuildCommand) || !strings.Contains(prompts[0], "(in jim)") {
		t.Errorf("named-form gate wrong: handled=%v prompts=%v out:\n%s", handled, prompts, out)
	}
}

// The end-to-end build, for real (the analyze suite's
// TestWasmAppBuildsAsWasm discipline): the app is created
// through the chain, the build sentence compiles it with
// the documented command, and main.wasm is on disk, fresh
// from the run. A second build over the artifact is a
// rebuild, not an already-there refusal. The one skip: a
// machine where the app's pinned Go 1.27 toolchain can't
// be fetched at all — the build then fails at the fetch,
// in the open, and there is nothing here to prove (this
// package's TestMain isolates HOME, so the analyzer's
// wasm_exec.js-copy guard can't see the real module cache
// from inside these tests; the fetch failure is the
// honest signal available).
func TestBuildWasmAppEndToEnd(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	out := captureStdout(t, func() { claimChain("build the wasm app in jim", yes, profile) })
	if !strings.Contains(out, "Built —") && strings.Contains(out, "download") && strings.Contains(out, "go1.27") {
		t.Skipf("the Go 1.27 toolchain isn't reachable from here:\n%s", out)
	}
	if !strings.Contains(out, "Built — the wasm app in jim compiled") || !strings.Contains(out, "is new from this build") {
		t.Fatalf("the build's success was not narrated from the artifact:\n%s", out)
	}
	st, err := os.Stat(filepath.Join(root, "jim", "main.wasm"))
	if err != nil || st.Size() < 100_000 {
		t.Fatalf("main.wasm is missing or implausibly small: %v", err)
	}
	t.Logf("main.wasm: %d bytes", st.Size())
	out = captureStdout(t, func() { claimChain("build the wasm app in jim", yes, profile) })
	if !strings.Contains(out, "Built — the wasm app in jim compiled") {
		t.Errorf("the rebuild was not a build:\n%s", out)
	}
}

// The failure path: break the app's Go (a file the
// recognition doesn't read, so the folder still verifies
// as an app) and the build sentence reports the compiler's
// own words as a build failure — never a success — and no
// artifact appears.
func TestBuildWasmAppCompilerFailure(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	writeFile(t, root, "jim/vector.go", "//go:build js && wasm\n\npackage main\n\nfunc AddVectors(a, b []float32) []float32 {\n\treturn a +\n}\n")
	out := captureStdout(t, func() { claimChain("build the wasm app in jim", yes, profile) })
	if !strings.Contains(out, "It didn't build") || !strings.Contains(out, "vector.go") {
		t.Errorf("the compiler failure was not reported in the compiler's words:\n%s", out)
	}
	if strings.Contains(out, "Built —") {
		t.Errorf("a failed build was dressed as success:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "main.wasm")); !os.IsNotExist(err) {
		t.Error("a main.wasm appeared from a failed build")
	}
}
