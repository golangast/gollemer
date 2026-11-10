package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The home-page sentence, at parse: the claim requires
// the words "home page to the wasm app" — the signature
// that keeps it clear of the page family, whose "home
// page" lines (a bare folder target, no target at all)
// are pinned as NOT this claim's.
func TestParseWasmHome(t *testing.T) {
	rows := []struct {
		line   string
		folder string
		ok     bool
	}{
		{"add a home page to the wasm app in jim", "jim", true},
		{"add a home page to the wasm app", "", true},
		{"create a home page to the webassembly app in the folder jim", "jim", true},
		{"add the home page to the wasm app in jim", "jim", true},
		{"put a home page to the wasm app into jim", "jim", true},
		{"add a home page to jim", "", false},
		{"add a home page in jim", "", false},
		{"add a home page", "", false},
		{"create a homepage", "", false},
		{"create a homepage in jim", "", false},
		{"add a home page to the app in jim", "", false},
		{"add a header to the wasm app in jim", "", false},
		{"add a home page to the wasm app in jim/main.go", "", false},
		{"add a home page to the wasm app in jim please", "", false},
	}
	for _, tc := range rows {
		q, ok := parseWasmHome(tc.line)
		if ok != tc.ok || (ok && q.folder != tc.folder) {
			t.Errorf("%q = (%q, %v), want (%q, %v)", tc.line, q.folder, ok, tc.folder, tc.ok)
		}
	}
}

// legacyChatWasmMain is a wasm app's main.go as the
// skeleton wrote it before the home page was a file of
// its own — the welcome inline — for the claim tests
// that start from an older app.
const legacyChatWasmMain = `//go:build js && wasm

package main

import (
	"fmt"
	"runtime/debug"
	"syscall/js"
)

func main() {
	debug.SetGCPercent(50)

	doc := js.Global().Get("document")
	app := doc.Call("getElementById", "app")

	heading := doc.Call("createElement", "h1")
	heading.Set("className", "headline")
	heading.Set("textContent", "TODO: your app's name")
	app.Call("appendChild", heading)

	// The vector demo (vector.go): two rows of numbers the
	// simd unit added, printed on the page as the proof it
	// ran.
	sums := AddVectors(
		[]float32{1, 2, 3, 4, 5, 6, 7, 8},
		[]float32{10, 20, 30, 40, 50, 60, 70, 80},
	)
	demo := doc.Call("createElement", "p")
	demo.Set("textContent", fmt.Sprintf("The vector demo added the two rows of numbers: %v", sums))
	app.Call("appendChild", demo)

	mountComponents(doc, app)

	select {}
}

func mountComponents(doc, app js.Value) {
	// gollemer:components-begin
	// gollemer:components-end
}
`

// The claim's outcomes through the chain: a fresh app
// (born with its home page) answers already-there; an
// older app — its main reverted to the inline skeleton
// and home.go removed — gets the move, and the moved app
// still builds.
func TestWasmHomeClaimOutcomes(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
	})
	out := captureStdout(t, func() {
		claimChain("add a home page to the wasm app in jim", yes, profile)
	})
	if !strings.Contains(out, "already has a home page") {
		t.Errorf("a fresh app should answer already-there:\n%s", out)
	}
	// Age the app: the inline skeleton back, home.go gone.
	writeFile(t, root, "jim/main.go", legacyChatWasmMain)
	if err := os.Remove(filepath.Join(root, "jim", "home.go")); err != nil {
		t.Fatal(err)
	}
	out = captureStdout(t, func() {
		claimChain("add a home page to the wasm app in jim", yes, profile)
	})
	if !strings.Contains(out, "home page") {
		t.Errorf("the move did not run:\n%s", out)
	}
	home := readBack(t, root, "jim/home.go")
	if !strings.Contains(home, "func Home(doc, app js.Value)") {
		t.Errorf("home.go was not created with the Home builder:\n%s", home)
	}
	main := readBack(t, root, "jim/main.go")
	if !strings.Contains(main, "Home(doc, app)") || strings.Contains(main, "TODO: your app's name") || strings.Contains(main, `"fmt"`) {
		t.Errorf("main did not trade the block for the mount:\n%s", main)
	}
	// The moved app compiles — the build sentence is the
	// proof the move left a working program.
	out = captureStdout(t, func() { claimChain("build the wasm app in jim", yes, profile) })
	if !strings.Contains(out, "Built —") && strings.Contains(out, "download") && strings.Contains(out, "go1.27") {
		t.Skipf("the Go 1.27 toolchain isn't reachable from here:\n%s", out)
	}
	if !strings.Contains(out, "Built — the wasm app in jim compiled") {
		t.Errorf("the moved app did not build:\n%s", out)
	}
	// And now the sentence answers already-there.
	out = captureStdout(t, func() {
		claimChain("add a home page to the wasm app in jim", yes, profile)
	})
	if !strings.Contains(out, "already has a home page") {
		t.Errorf("the moved app should answer already-there:\n%s", out)
	}
}

// The readout, at creation: the whole starter list for a
// fresh app — the parts sentence present (no parts yet),
// the into-the-header and home-page sentences absent
// (no header part; the app is born with its home page) —
// and the worked example that connects a button to a
// card.
func TestWasmCommandsReadoutAtCreation(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() { claimChain("create a folder jim", yes, profile) })
	out := captureStdout(t, func() { claimChain("add a wasm app in jim", yes, profile) })
	for _, want := range []string{
		"What you can say for this app:",
		"add a card to jim",
		"add a button called save to jim",
		"when the save button is clicked, show the card",
		"add a header and footer to the wasm app in jim",
		"build the wasm app in jim",
		"serve the wasm app in jim",
		"add a full stack app called notes in jim with fields title and body",
		"explain the wasm app in jim",
		"\"add a button called save to jim\" makes the button",
		"\"add a card to jim\" makes the card it acts on",
		"\"when the save button is clicked, set the card's words to 'Saved'\" connects them",
		"the when-sentence is the wire between them",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the creation readout is missing %q:\n%s", want, out)
		}
	}
	for _, absent := range []string{
		"add a navigation bar to the header in jim",
		"add a home page to the wasm app in jim",
	} {
		if strings.Contains(out, absent) {
			t.Errorf("the creation readout names %q, which this app's state rules out:\n%s", absent, out)
		}
	}
}

// THE READOUT'S DRIFT DISCIPLINE: each state-aware line
// appears and vanishes with the state it names — parts
// landed take the parts sentence away and bring the
// into-the-header one; a missing home.go brings the
// home-page sentence. The explanation closes with the
// same readout (one generator, both triggers).
func TestWasmCommandsReadoutStates(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, profile)
		claimChain("add a wasm app in jim", yes, profile)
		claimChain("add a header and footer to the wasm app in jim", yes, profile)
	})
	got := wasmAppCommandsReadout(root, "jim")
	if strings.Contains(got, "add a header and footer to the wasm app in jim") {
		t.Errorf("the parts sentence survived the parts landing:\n%s", got)
	}
	if !strings.Contains(got, "add a navigation bar to the header in jim") {
		t.Errorf("the into-the-header sentence is missing with a header present:\n%s", got)
	}
	if strings.Contains(got, "add a home page to the wasm app in jim") {
		t.Errorf("the home sentence appeared while home.go exists:\n%s", got)
	}
	if err := os.Remove(filepath.Join(root, "jim", "home.go")); err != nil {
		t.Fatal(err)
	}
	got = wasmAppCommandsReadout(root, "jim")
	if !strings.Contains(got, "add a home page to the wasm app in jim") {
		t.Errorf("the home sentence is missing with home.go gone:\n%s", got)
	}
	// The worked example prints in every state.
	if !strings.Contains(got, "the when-sentence is the wire between them") {
		t.Errorf("the worked example went missing:\n%s", got)
	}
	// The explanation closes with the readout.
	walk := wasmAppExplanation(root, "jim")
	open := strings.Index(walk, "When the page opens:")
	commands := strings.Index(walk, "What you can say for this app:")
	if open < 0 || commands < 0 || commands < open {
		t.Errorf("the explanation does not close with the readout:\n%s", walk)
	}
}
