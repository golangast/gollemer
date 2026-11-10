package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// isolateWasmExplainedStore points the first-run store
// at a throwaway file (the receipts tests' discipline)
// and returns its path.
func isolateWasmExplainedStore(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), ".gollemer_wasm_explained.json")
	old := wasmExplainedStorePathOverride
	wasmExplainedStorePathOverride = path
	t.Cleanup(func() { wasmExplainedStorePathOverride = old })
	return path
}

// The explanation sentence, at parse: the family's
// folder tail, the session form, and the neighbours it
// must leave alone.
func TestParseExplainWasmApp(t *testing.T) {
	rows := []struct {
		line   string
		folder string
		has    bool
		ok     bool
	}{
		{"explain the wasm app in jim", "jim", true, true},
		{"explain the webassembly app in jim", "jim", true, true},
		{"explain the wasm app", "", false, true},
		{"explain wasm app in the folder jim", "jim", true, true},
		{"explain the wasm app in jim/main.go", "", false, false},
		{"explain the app in jim", "", false, false},
		{"explain the wasm", "", false, false},
		{"what is the wasm app in jim", "", false, false},
	}
	for _, tc := range rows {
		folder, has, ok := parseExplainWasmApp(tc.line)
		if ok != tc.ok || (ok && (folder != tc.folder || has != tc.has)) {
			t.Errorf("%q = (%q, %v, %v), want (%q, %v, %v)", tc.line, folder, has, ok, tc.folder, tc.has, tc.ok)
		}
	}
}

// The walk, on demand through the chain: every file the
// fixture app actually has gets its line, naming the
// actual components and the actual event; the files it
// does NOT have (wasm_exec.js can't be copied under this
// package's isolated HOME; main.wasm isn't built) get
// the honest treatment — no glue line, and the not-built
// line instead of a built one.
func TestExplainWasmAppOnDemand(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
		claimChain("add a card to jim", yes, profile)
		claimChain("add a button called save to jim", yes, profile)
		claimChain("add a header and footer to the wasm app in jim", yes, profile)
		claimChain("add a navigation bar to the header in jim", yes, profile)
		claimChain("when the save button is clicked, set the card's words to 'Saved' in jim", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("explain the wasm app in jim", yes, profile)
	})
	for _, want := range []string{
		"The wasm app in jim, file by file:",
		"jim/go.mod — the app's own module",
		"pins Go 1.27",
		"jim/main.go — the program:",
		"mounts the header, then the home page, then the app's components, then the footer",
		"jim/home.go — the home page: the welcome heading and the vector demo's sums",
		"jim/vector.go — the vector unit",
		"jim/components.go — the app's own components",
		"a card, a button called save",
		"jim/events.go — the events the app answers",
		"when the save button is clicked, set the card's words to 'Saved'",
		"jim/header/ — the header part, its own package (package header)",
		"a navigation bar",
		"jim/footer/ — the footer part, its own package (package footer)",
		"jim/index.html — the host page",
		"The app isn't built yet — there's no main.wasm in jim",
		"\"build the wasm app in jim\"",
		"\"serve the wasm app in jim\"",
		"http://localhost:8080/static/",
		"When the page opens:",
		"What you can say for this app:",
		"add a navigation bar to the header in jim",
		"when the save button is clicked, set the card's words to 'Saved'",
		"the when-sentence is the wire between them",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the explanation is missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "wasm_exec.js — the Go runtime's glue") {
		t.Errorf("the explanation describes a wasm_exec.js the fixture app does not have:\n%s", out)
	}
	// No app in the folder: the honest refusal.
	out = captureStdout(t, func() {
		claimChain("explain the wasm app in jake", yes, profile)
	})
	if !strings.Contains(out, "there's no folder jake in this project") {
		t.Errorf("missing-folder explanation answer wrong:\n%s", out)
	}
}

// THE DRIFT DISCIPLINE: the walk names exactly what
// exists. A bare skeleton gets no components, events, or
// parts lines; removing a file removes its line; a
// main.wasm appearing flips the not-built line to the
// built one, with its size.
func TestExplainWasmAppDrift(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	text := wasmAppExplanation(root, "jim")
	for _, absent := range []string{"components.go —", "events.go —", "header/ —", "footer/ —", "mounts the header"} {
		if strings.Contains(text, absent) {
			t.Errorf("the bare app's explanation claims %q:\n%s", absent, text)
		}
	}
	if !strings.Contains(text, "mounts the home page, then the app's components") {
		t.Errorf("the bare app's main line is wrong:\n%s", text)
	}
	// A removed file's line disappears.
	if err := os.Remove(filepath.Join(root, "jim", "vector.go")); err != nil {
		t.Fatal(err)
	}
	text = wasmAppExplanation(root, "jim")
	if strings.Contains(text, "vector.go —") {
		t.Errorf("the removed vector.go still gets a line:\n%s", text)
	}
	// main.wasm appears: built, with its size.
	if err := os.WriteFile(filepath.Join(root, "jim", "main.wasm"), []byte("fake-wasm-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	text = wasmAppExplanation(root, "jim")
	if !strings.Contains(text, "jim/main.wasm — the built app, 15 bytes") {
		t.Errorf("the built line is missing or wrong:\n%s", text)
	}
	if strings.Contains(text, "isn't built yet") {
		t.Errorf("the built app still reads as not built:\n%s", text)
	}
}

// The first-run tour: the first build carries the
// explanation; the second does not; the on-demand
// sentence answers anyway. The store remembers per
// project + app, in the sibling file, 0600.
func TestWasmAppFirstRunExplanation(t *testing.T) {
	storePath := isolateWasmExplainedStore(t)
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
	if !strings.Contains(out, "Built — the wasm app in jim compiled") {
		t.Fatalf("the first build did not build:\n%s", out)
	}
	if !strings.Contains(out, "Since this is the first time the app in jim has run here") ||
		!strings.Contains(out, "The wasm app in jim, file by file:") {
		t.Errorf("the first build did not carry the tour:\n%s", out)
	}
	if !wasmAppExplainedSeen(root, "jim") {
		t.Error("the store does not remember the tour for this app")
	}
	st, err := os.Stat(storePath)
	if err != nil {
		t.Fatalf("the store was not written: %v", err)
	}
	if perm := st.Mode().Perm(); perm != 0o600 {
		t.Errorf("the store's permissions = %o, want 600", perm)
	}
	out = captureStdout(t, func() { claimChain("build the wasm app in jim", yes, profile) })
	if !strings.Contains(out, "Built — the wasm app in jim compiled") {
		t.Fatalf("the second build did not build:\n%s", out)
	}
	if strings.Contains(out, "first time the app in jim has run here") {
		t.Errorf("the second build repeated the tour:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("explain the wasm app in jim", yes, profile) })
	if !strings.Contains(out, "The wasm app in jim, file by file:") || !strings.Contains(out, "jim/main.wasm — the built app") {
		t.Errorf("the on-demand explanation after builds is wrong:\n%s", out)
	}
	// A different app in the same project is a different
	// memory: unseen until its own first run.
	if wasmAppExplainedSeen(root, "jake") {
		t.Error("an app that never ran reads as explained")
	}
}

// The store failing is silent: an unwritable path never
// blocks the tour itself, and "seen" reads false rather
// than erroring the caller.
func TestWasmExplainedStoreFailureSilent(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := wasmExplainedStorePathOverride
	wasmExplainedStorePathOverride = filepath.Join(blocker, ".gollemer_wasm_explained.json")
	t.Cleanup(func() { wasmExplainedStorePathOverride = old })
	if wasmAppExplainedSeen(dir, "jim") {
		t.Error("a broken store read as seen")
	}
	markWasmAppExplained(dir, "jim") // must not panic or block
	out := captureStdout(t, func() {
		maybeExplainWasmAppFirstRun("gollemer [runner]>", dir, "jim")
	})
	if !strings.Contains(out, "first time the app in jim has run here") {
		t.Errorf("a broken store suppressed the tour:\n%s", out)
	}
}

// The two-folder picture (John's serving shape): when a
// server for the app really exists in server/ — its file
// there AND its /static/ registration naming this folder —
// the walk gives server/main.go its line and closes with
// the arrangement in plain words. Without it, the walk
// invents no server: no webserver line, and the how-to
// names the server the serve sentence would create.
func TestExplainWasmAppServerFolder(t *testing.T) {
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	writeFile(t, dir, "jim/main.wasm", "fake wasm bytes")

	unserved := wasmAppExplanation(dir, "jim")
	if strings.Contains(unserved, "— the Go webserver") ||
		strings.Contains(unserved, "two folders, two jobs") {
		t.Errorf("the unserved walk invented a server:\n%s", unserved)
	}
	if !strings.Contains(unserved, "gives the app a dedicated Go webserver in a server folder") {
		t.Errorf("the unserved walk lost its how-to-serve:\n%s", unserved)
	}

	writeFile(t, dir, "server/main.go", "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"jim\"))))\n\thttp.ListenAndServe(\":8080\", nil)\n}\n")
	served := wasmAppExplanation(dir, "jim")
	for _, want := range []string{
		"server/main.go — the Go webserver, in its own folder at the project root",
		"hands the app's files to the browser at /static/",
		"two folders, two jobs — jim is the program that runs in the browser; the server folder is the Go program that hands it over",
		"the app answers at http://localhost:8080/static/",
	} {
		if !strings.Contains(served, want) {
			t.Errorf("the served walk is missing %q:\n%s", want, served)
		}
	}
	// The on-demand sentence carries the same line.
	out := captureStdout(t, func() { claimChain("explain the wasm app in jim", yes, profile) })
	if !strings.Contains(out, "server/main.go — the Go webserver") {
		t.Errorf("the on-demand explanation lost the server line:\n%s", out)
	}
}
