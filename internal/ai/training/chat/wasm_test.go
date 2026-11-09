package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The wasm-app grammar, at parse: the wasm and webassembly
// words claim with their folder, articles optional, the
// folder word tolerated.
func TestParseWasmAppClaims(t *testing.T) {
	rows := []struct {
		line   string
		folder string
	}{
		{"add a wasm app in jim", "jim"},
		{"add a webassembly app in jim", "jim"},
		{"add wasm app in jim", "jim"},
		{"create a wasm app in jim/temp", "jim/temp"},
		{"make a wasm app into jim", "jim"},
		{"add a wasm app in the folder jim", "jim"},
		{"add a wasm app", ""},
		{"add a webassembly app", ""},
	}
	for _, tc := range rows {
		got, ok := parseWasmApp(tc.line)
		if !ok {
			t.Errorf("%q: no claim", tc.line)
			continue
		}
		if got.folder != tc.folder {
			t.Errorf("%q: parsed folder %q, want %q", tc.line, got.folder, tc.folder)
		}
	}
}

// The shapes the wasm grammar must leave alone — above all
// the neighbours' sentences: a templated page, a component,
// the book's .go lines, and the app-less wasm questions.
func TestParseWasmAppDeclines(t *testing.T) {
	for _, line := range []string{
		"add a templated page called home in jim",
		"add a card to jim",
		"add a wasm app in jim extra",
		"add a wasm in jim",
		"add an app in jim",
		"what is wasm",
		"add a server in jim",
		"add a wasm app at jim/temp/index.html",
	} {
		if got, ok := parseWasmApp(line); ok {
			t.Errorf("%q claimed as a wasm ask (%+v)", line, got)
		}
	}
}

// The component grammar's app targets, at parse: a folder
// or the main.go inside one claims as an app target, with
// the component read against the same registry. The page
// targets and other .go targets never match here — the two
// parses are disjoint.
func TestParseComponentWasmTarget(t *testing.T) {
	rows := []struct {
		line      string
		component string
		target    string
		goFile    bool
	}{
		{"add a card to jim", "card", "jim", false},
		{"add a card to jim/main.go", "card", "jim/main.go", true},
		{"add a switch to main.go", "switch", "main.go", true},
		{"add a navigation bar to jim", "navigation-bar", "jim", false},
		{"add card to jim", "card", "jim", false},
		{"add a carousel to jim", "", "jim", false},
	}
	for _, tc := range rows {
		got, ok := parseComponentWasmTarget(tc.line)
		if !ok {
			t.Errorf("%q: no claim", tc.line)
			continue
		}
		if got.component != tc.component || got.target != tc.target || got.goFile != tc.goFile {
			t.Errorf("%q: parsed %+v, want component=%q target=%q goFile=%v", tc.line, got, tc.component, tc.target, tc.goFile)
		}
		// Disjointness: the standard parse declines every
		// app target, today and after — the two parses
		// never both own a sentence.
		if _, ok := parseComponent(tc.line); ok {
			t.Errorf("%q: the standard component parse claimed an app target", tc.line)
		}
	}
	for _, line := range []string{
		"add a card to jim/temp/index.html",
		"add a card to the page",
		"add a card to jim/other.go",
		"add a button",
		"what components can i add",
	} {
		if got, ok := parseComponentWasmTarget(line); ok {
			t.Errorf("%q claimed as a wasm component target (%+v)", line, got)
		}
	}
}

// John's flow, end to end in one fixture project: the
// folder is created through the chat, the app lands in it,
// and components join the app — a card by its folder, a
// button after it in ask order, a switch by its main.go —
// while a repeat card is the already-there answer and
// nothing about the host project breaks.
func TestWasmAppJohnsFlow(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The wasm app is in jim.") {
		t.Errorf("app landing confirmation missing:\n%s", out)
	}
	if !strings.Contains(out, "It builds with Go 1.27") || !strings.Contains(out, analyze.WasmBuildCommand) {
		t.Errorf("app confirmation does not state the build:\n%s", out)
	}
	for _, rel := range []string{"jim/go.mod", "jim/main.go", "jim/vector.go", "jim/index.html"} {
		if _, err := os.Stat(filepath.Join(root, rel)); err != nil {
			t.Errorf("the app is missing %s", rel)
		}
	}
	main := readBack(t, root, "jim/main.go")
	if !strings.Contains(main, "debug.SetGCPercent(50)") {
		t.Error("the landed main.go lost the GOGC line")
	}

	out = captureStdout(t, func() {
		claimChain("add a card to jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The card is in the wasm app in jim.") {
		t.Errorf("card landing confirmation missing:\n%s", out)
	}
	comp := readBack(t, root, "jim/components.go")
	if !strings.Contains(comp, "func cardComponent(doc js.Value) js.Value") {
		t.Errorf("the card builder did not land:\n%s", comp)
	}
	main = readBack(t, root, "jim/main.go")
	if !strings.Contains(main, `app.Call("appendChild", cardComponent(doc))`) {
		t.Errorf("the card's mount line did not land:\n%s", main)
	}
	index := readBack(t, root, "jim/index.html")
	if strings.Count(index, ".card {") != 1 {
		t.Error("the card's styles were not carried into the host page exactly once")
	}

	// Repeat: the already-there answer, files untouched.
	before := readBack(t, root, "jim/components.go")
	out = captureStdout(t, func() {
		claimChain("add a card to jim", yes, profile)
	})
	if !strings.Contains(out, "already has a card") {
		t.Errorf("repeat card was not the already-there answer:\n%s", out)
	}
	if readBack(t, root, "jim/components.go") != before {
		t.Error("the repeat card changed components.go")
	}

	// A button accumulates after the card, and a switch
	// named by the app's main.go lands too.
	out = captureStdout(t, func() {
		claimChain("add a button to jim", yes, profile)
		claimChain("add a switch to jim/main.go", yes, profile)
	})
	if !strings.Contains(out, "Done — The button is in the wasm app in jim.") || !strings.Contains(out, "Done — The switch is in the wasm app in jim.") {
		t.Errorf("accumulating confirmations missing:\n%s", out)
	}
	main = readBack(t, root, "jim/main.go")
	ci, bi, si := strings.Index(main, "cardComponent(doc)"), strings.Index(main, "buttonComponent(doc)"), strings.Index(main, "switchComponent(doc)")
	if !(0 <= ci && ci < bi && bi < si) {
		t.Errorf("mounts are not in ask order:\n%s", main)
	}
}

// A component sentence aimed at a folder that holds no
// wasm app is not this piece's: the claim declines it
// untouched, and a main.go without the wasm tag is left
// exactly as it was.
func TestComponentWasmTargetDeclinesNonApp(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jake/main.go": "package main\n\nfunc main() {}\n",
	}))
	if runComponent("add a card to jake", yes) {
		t.Error("runComponent claimed a card for a folder with no app in it")
	}
	if runComponent("add a card to jake/main.go", yes) {
		t.Error("runComponent claimed a card for a plain main.go")
	}
	if _, err := os.Stat(filepath.Join(root, "jake", "components.go")); !os.IsNotExist(err) {
		t.Error("a components.go appeared in a folder that holds no app")
	}
	if got := readBack(t, root, "jake/main.go"); got != "package main\n\nfunc main() {}\n" {
		t.Errorf("the plain main.go changed:\n%s", got)
	}
}

// The folder question: a wasm sentence with no folder asks
// one question, and the one-word answer lands the app in
// that folder.
func TestWasmAppFolderQuestion(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/.keep": "",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a wasm app", yes, profile)
	})
	if !strings.Contains(out, "which folder should the app live in?") || !hasPendingQuestion() {
		t.Fatalf("the folderless ask was not claimed with a question:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The wasm app is in jim.") {
		t.Errorf("the folder answer did not land the app:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "main.go")); err != nil {
		t.Error("the app did not land in jim")
	}
}

// The missing folder is the family's two-step, in the wasm
// piece's own words.
func TestWasmAppMissingFolder(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(nil))
	out := captureStdout(t, func() {
		claimChain("add a wasm app in ghost", yes, &UserProfile{})
	})
	if !strings.Contains(out, "there's no folder ghost in this project") || !strings.Contains(out, "create a folder ghost") {
		t.Errorf("missing-folder answer is not the two-step:\n%s", out)
	}
}
