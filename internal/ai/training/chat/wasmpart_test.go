package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The parts grammar, at parse: both parts, either order,
// single parts, the optional "wasm app" middle, articles
// and the folder word tolerated.
func TestParseWasmPartClaims(t *testing.T) {
	rows := []struct {
		line   string
		header bool
		footer bool
		folder string
	}{
		{"add a header and footer to the wasm app in jim", true, true, "jim"},
		{"add a footer and header to the wasm app in jim", true, true, "jim"},
		{"add a header to the wasm app in jim", true, false, "jim"},
		{"add a footer to the wasm app in jim", false, true, "jim"},
		{"add a header to jim", true, false, "jim"},
		{"add header and footer in jim", true, true, "jim"},
		{"create a header to the wasm app in the folder jim", true, false, "jim"},
		{"add a header and footer to the wasm app", true, true, ""},
		{"add a header", true, false, ""},
	}
	for _, tc := range rows {
		got, ok := parseWasmPart(tc.line)
		if !ok {
			t.Errorf("%q: no claim", tc.line)
			continue
		}
		if got.header != tc.header || got.footer != tc.footer || got.folder != tc.folder {
			t.Errorf("%q: parsed %+v, want header=%v footer=%v folder=%q", tc.line, got, tc.header, tc.footer, tc.folder)
		}
	}
	// The canonical sentence (the clarify completion's
	// rebuild) is exact.
	if got := (wasmPartAsk{header: true, footer: true, folder: "jim"}).canonical(); got != "add a header and footer to the wasm app in jim" {
		t.Errorf("canonical both = %q", got)
	}
	if got := (wasmPartAsk{header: true, folder: "jim"}).canonical(); got != "add a header to the wasm app in jim" {
		t.Errorf("canonical header = %q", got)
	}
}

// The shapes the parts grammar must leave alone: the
// neighbours' sentences, above all the component asks
// whose names merely resemble a part.
func TestParseWasmPartDeclines(t *testing.T) {
	for _, line := range []string{
		"add a navigation bar to the header in jim",
		"add a card to jim",
		"add a header to jim/main.go",
		"add a header and footer to the wasm app in jim extra",
		"add a header and a footer to the wasm app in jim",
		"add a templated page called home in jim",
		"what is a header",
		"add a wasm app in jim",
	} {
		if got, ok := parseWasmPart(line); ok {
			t.Errorf("%q claimed as a parts ask (%+v)", line, got)
		}
	}
}

// The part-target component grammar, at parse — and its
// disjointness from the two component parses it sits
// beside: neither of them may own these sentences.
func TestParseComponentPartTarget(t *testing.T) {
	rows := []struct {
		line      string
		component string
		called    string
		part      string
		folder    string
	}{
		{"add a navigation bar to the header in jim", "navigation-bar", "", "header", "jim"},
		{"add a chip to the footer in jim", "chip", "", "footer", "jim"},
		{"add a chip to the header of the wasm app in jim", "chip", "", "header", "jim"},
		{"add a button called save to the footer in jim", "button", "save", "footer", "jim"},
		{"add a carousel to the header in jim", "", "", "header", "jim"},
		{"add a card to the footer in jim/temp", "card", "", "footer", "jim/temp"},
	}
	for _, tc := range rows {
		got, ok := parseComponentPartTarget(tc.line)
		if !ok {
			t.Errorf("%q: no claim", tc.line)
			continue
		}
		if got.component != tc.component || got.called != tc.called || got.part != tc.part || got.folder != tc.folder {
			t.Errorf("%q: parsed %+v, want component=%q called=%q part=%q folder=%q", tc.line, got, tc.component, tc.called, tc.part, tc.folder)
		}
		if _, ok := parseComponentWasmTarget(tc.line); ok {
			t.Errorf("%q: the app component parse also claimed a part target", tc.line)
		}
		if _, ok := parseComponent(tc.line); ok {
			t.Errorf("%q: the page component parse also claimed a part target", tc.line)
		}
	}
	for _, line := range []string{
		"add a card to jim",
		"add a card to the page",
		"add a card to jim/temp/index.html",
		"add a card to the header in jim/main.go",
		"add a card to the header",
		"add a card to the headers in jim",
	} {
		if got, ok := parseComponentPartTarget(line); ok {
			t.Errorf("%q claimed as a part component target (%+v)", line, got)
		}
	}
}

// The parts flow, end to end through the claim chain:
// the app lands, its parts land with their confirmation,
// a repeat is the already-there answer with files
// untouched, and a component into the header plus an
// event on a body component keep working with the parts
// in place.
func TestWasmPartFlow(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
		claimChain("add a card to jim", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("add a header and footer to the wasm app in jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The header and the footer are in the wasm app in jim") {
		t.Errorf("parts landing confirmation missing:\n%s", out)
	}
	for _, rel := range []string{"jim/header/header.go", "jim/footer/footer.go"} {
		if _, err := os.Stat(filepath.Join(root, rel)); err != nil {
			t.Errorf("the part file %s is missing", rel)
		}
	}
	main := readBack(t, root, "jim/main.go")
	hi, bi, fi := strings.Index(main, "header.Header(doc)"), strings.Index(main, "mountComponents(doc, app)"), strings.Index(main, "footer.Footer(doc)")
	if !(0 <= hi && hi < bi && bi < fi) {
		t.Errorf("the parts do not mount around the body:\n%s", main)
	}

	before := readBack(t, root, "jim/main.go")
	out = captureStdout(t, func() {
		claimChain("add a header and footer to the wasm app in jim", yes, profile)
	})
	if !strings.Contains(out, "already has a header and a footer") {
		t.Errorf("repeat parts was not the already-there answer:\n%s", out)
	}
	if readBack(t, root, "jim/main.go") != before {
		t.Error("the repeat parts ask changed main.go")
	}

	// A component into the header, then an event on a
	// body component whose target lives in the footer.
	out = captureStdout(t, func() {
		claimChain("add a navigation bar to the header in jim", yes, profile)
		claimChain("add a chip to the footer in jim", yes, profile)
		claimChain("add a button called save to jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The navigation bar is in the header of the wasm app in jim.") {
		t.Errorf("part component confirmation missing:\n%s", out)
	}
	if !strings.Contains(out, "Done — The chip is in the footer of the wasm app in jim.") {
		t.Errorf("footer component confirmation missing:\n%s", out)
	}
	hcomp := readBack(t, root, "jim/header/components.go")
	if !strings.Contains(hcomp, "func navigationBarComponent(") {
		t.Errorf("the navigation bar did not land in the header's components:\n%s", hcomp)
	}
	out = captureStdout(t, func() {
		claimChain("when the save button is clicked, show the chip in jim", yes, profile)
	})
	if !strings.Contains(out, "Done —") || !strings.Contains(out, "show the chip") {
		t.Errorf("event on a part component did not land:\n%s", out)
	}
	events := readBack(t, root, "jim/events.go")
	if !strings.Contains(events, `showElement(doc, "chip-1")`) {
		t.Errorf("the event does not act on the footer chip by its id:\n%s", events)
	}
}

// The session reading: with an app touched this session,
// the folderless parts sentence resolves to it, reading
// printed. And the claim declines, untouched, over a
// folder that holds no app.
func TestWasmPartSessionAndDecline(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jake/main.go": "package main\n\nfunc main() {}\n",
	}))
	profile := &UserProfile{}
	if runWasmPart("add a header to jake", yes) {
		t.Error("runWasmPart claimed a header for a folder with no app in it")
	}
	if _, err := os.Stat(filepath.Join(root, "jake", "header")); !os.IsNotExist(err) {
		t.Error("a header folder appeared in a project folder that holds no app")
	}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("add a footer to the wasm app", yes, profile)
	})
	if !strings.Contains(out, "Reading the app as the wasm app in jim") {
		t.Errorf("the session reading was not printed:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "footer", "footer.go")); err != nil {
		t.Errorf("the footer did not land by the session reading: %v", err)
	}
}

// The folder question: the sentence that names the app
// but no folder — and no app in the session — asks one
// question, and the one-word answer lands the parts in
// that folder (the wasm family's clarify pattern).
func TestWasmPartFolderQuestion(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/.keep": "",
	}))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("add a wasm app in jim", yes, profile)
	})
	// A fresh working context over the same project: no
	// app in the session, so the folderless sentence must
	// ask, not guess.
	resetWorkContext()
	out := captureStdout(t, func() {
		claimChain("add a header and footer to the wasm app", yes, profile)
	})
	if !strings.Contains(out, "which folder does the app live in?") || !hasPendingQuestion() {
		t.Fatalf("the folderless ask was not claimed with a question:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("jim", yes, profile)
	})
	if !strings.Contains(out, "Done — The header and the footer are in the wasm app in jim") {
		t.Errorf("the folder answer did not land the parts:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "header", "header.go")); err != nil {
		t.Error("the header did not land in jim after the answer")
	}
}

// PART C, pinned: a file created inside a part's folder
// takes the part's package — the folder's own package
// when the part is there (package header, from
// header.go), and the folder's own name when the folder
// stands empty inside the app. The create's own compile
// check is the verdict; when the app's pinned toolchain
// can't be fetched from here at all, the check fails at
// the fetch and there is nothing to pin (the build
// suite's skip discipline).
func TestCreateFileIntoWasmPartPackage(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
		claimChain("add a header to the wasm app in jim", yes, profile)
		claimChain("create a folder jim/side", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("create a file notes.go in folder jim/header", yes, profile)
	})
	if strings.Contains(out, "download") && strings.Contains(out, "go1.27") {
		t.Skipf("the Go 1.27 toolchain isn't reachable from here:\n%s", out)
	}
	if !strings.Contains(out, "Done — created jim/header/notes.go.") || !strings.Contains(out, "it builds") {
		t.Errorf("the create inside the part did not land verified:\n%s", out)
	}
	if got := readBack(t, root, "jim/header/notes.go"); got != "package header\n" {
		t.Errorf("notes.go = %q, want the part's own package clause", got)
	}
	out = captureStdout(t, func() {
		claimChain("create a file extra.go in folder jim/side", yes, profile)
	})
	if !strings.Contains(out, "Done — created jim/side/extra.go.") {
		t.Errorf("the create in the app's empty subfolder did not land:\n%s", out)
	}
	if got := readBack(t, root, "jim/side/extra.go"); got != "package side\n" {
		t.Errorf("extra.go = %q, want the folder's own name as its package", got)
	}
}
