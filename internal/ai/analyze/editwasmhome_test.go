package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// legacyWasmMain is a wasm app's main.go exactly as the
// skeleton wrote it before the home page became a file of
// its own: the welcome heading and the vector demo built
// inline, above the components mount.
const legacyWasmMain = `//go:build js && wasm

// This app runs in the browser: Go compiled to WebAssembly
// (a .wasm file), building its page through syscall/js. It
// lives in its own module — the go.mod beside it pins Go
// 1.27 — so the project around it keeps its own Go version,
// and the js && wasm build tag above keeps these files out
// of every ordinary build. The one build that includes
// them is the browser build its host page names:
//
//	GOEXPERIMENT=simd GOOS=js GOARCH=wasm go build -o main.wasm .

package main

import (
	"fmt"
	"runtime/debug"
	"syscall/js"
)

func main() {
	// The garbage collector's GOGC setting, turned down in
	// code: a browser page has no environment to pass flags
	// through, so the app sets the flag itself, first thing.
	// 50 collects at half the usual heap growth — a smaller
	// footprint in the tab, at a little CPU. Raise the
	// number here if the app would rather trade memory for
	// speed.
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

	// A wasm program whose main returns is finished, and
	// the browser takes the page down with it — so block
	// here forever and let the page live.
	select {}
}

// mountComponents puts the components the chat adds to
// this app on the page, in the order they were asked for:
// ask for one ("add a card to this app's folder") and its
// mount line lands between the markers. Leave the markers
// where they are.
func mountComponents(doc, app js.Value) {
	// gollemer:components-begin
	// gollemer:components-end
}
`

// wasmHomeProject plants a project holding a wasm app in
// jim whose main is the given content, with the app's
// other birth files beside it.
func wasmHomeProject(t *testing.T, main string, extra map[string]string) (string, *Project) {
	t.Helper()
	files := map[string]string{
		"go.mod":         "module example.com/host\n\ngo 1.21\n",
		"doc.go":         "package host\n",
		"jim/go.mod":     "module jim\n\ngo 1.27.0\n",
		"jim/main.go":    main,
		"jim/vector.go":  wasmAppVector(),
		"jim/index.html": wasmAppIndex(),
	}
	for k, v := range extra {
		files[k] = v
	}
	return serveProject(t, files)
}

func readWasmFile(t *testing.T, dir, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// The move: a legacy app's inline welcome lifts out of
// main into home.go, and the resulting main is byte-for-
// byte the main the skeleton writes today — a moved app
// and a fresh app are the same app.
func TestPlanWasmHomeMovesInlineWelcome(t *testing.T) {
	dir, proj := wasmHomeProject(t, legacyWasmMain, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("home plan refused: %q", refusal)
	}
	byRel := map[string]EditFile{}
	for _, f := range plan.Files {
		byRel[f.Rel] = f
	}
	if len(plan.Files) != 2 || byRel["jim/home.go"].Rel == "" || len(byRel["jim/home.go"].Before) != 0 {
		t.Fatalf("the move should create home.go and edit main.go, got %v", plan.Files)
	}
	applyPlan(t, dir, plan)
	if got := readWasmFile(t, dir, "jim/home.go"); got != wasmAppHome() {
		t.Errorf("home.go is not the skeleton's home file:\n%s", got)
	}
	if got := readWasmFile(t, dir, "jim/main.go"); got != wasmAppMain() {
		t.Errorf("a moved app's main differs from a fresh app's main:\n%s", got)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("the home plan does not verify against the re-analyzed project")
	}
}

// The states around the move, each answered for what it
// is: already-there, a squatter file, a mount that is
// missing its file, a file that is missing its mount.
func TestPlanWasmHomeStates(t *testing.T) {
	// Already there: home.go present and mounted.
	_, proj := wasmHomeProject(t, wasmAppMain(), map[string]string{"jim/home.go": wasmAppHome()})
	if _, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"}); !strings.Contains(refusal, "already has a home page") {
		t.Errorf("already-there refusal wrong: %q", refusal)
	}
	// A squatter home.go (no Home builder) is never
	// overwritten.
	_, proj = wasmHomeProject(t, legacyWasmMain, map[string]string{"jim/home.go": "//go:build js && wasm\n\npackage main\n"})
	if _, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"}); !strings.Contains(refusal, "doesn't carry the Home builder") {
		t.Errorf("squatter refusal wrong: %q", refusal)
	}
	// Mount without file: only the file is created.
	dir, proj := wasmHomeProject(t, wasmAppMain(), nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("create-only plan refused: %q", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "jim/home.go" || len(plan.Files[0].Before) != 0 {
		t.Fatalf("create-only should write just home.go, got %v", plan.Files)
	}
	applyPlan(t, dir, plan)
	// File without mount: only the mount lands.
	unmounted := strings.Replace(legacyWasmMain, wasmLegacyHomeBlock, "", 1)
	unmounted = strings.Replace(unmounted, "\t\"fmt\"\n", "", 1)
	dir, proj = wasmHomeProject(t, unmounted, map[string]string{"jim/home.go": wasmAppHome()})
	plan, refusal = proj.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("mount-only plan refused: %q", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "jim/main.go" {
		t.Fatalf("mount-only should edit just main.go, got %v", plan.Files)
	}
	applyPlan(t, dir, plan)
	main := readWasmFile(t, dir, "jim/main.go")
	hi, mi := strings.Index(main, "Home(doc, app)"), strings.Index(main, "mountComponents(doc, app)")
	if hi < 0 || mi < 0 || hi > mi {
		t.Errorf("the mount did not land before the components:\n%s", main)
	}
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("the mount-only plan does not verify")
	}
}

// The refusals: no app, and a welcome edited past the
// standard block — the move never rewrites by guess.
func TestPlanWasmHomeRefusals(t *testing.T) {
	_, proj := wasmHomeProject(t, legacyWasmMain, nil)
	// (the fixture's app at jim answers the happy path in
	// the tests above; here the folder without an app)
	_, proj2 := serveProject(t, map[string]string{
		"go.mod":         "module example.com/host\n\ngo 1.21\n",
		"doc.go":         "package host\n",
		"jim/index.html": "<html></html>\n",
	})
	_ = proj
	if _, refusal := proj2.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"}); !strings.Contains(refusal, "there's no wasm app in jim yet") {
		t.Errorf("no-app refusal wrong: %q", refusal)
	}
	edited := strings.Replace(legacyWasmMain, `heading.Set("textContent", "TODO: your app's name")`, `heading.Set("textContent", "My App")`, 1)
	_, proj = wasmHomeProject(t, edited, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"})
	if plan != nil || !strings.Contains(refusal, "isn't the standard block") {
		t.Errorf("edited-welcome refusal wrong: plan=%v refusal=%q", plan != nil, refusal)
	}
}

// A legacy app that already has its header part: the
// part mounted AFTER the inline welcome (the old parts
// anchor was the components call), so the move — which
// puts Home exactly where the block stood — keeps the
// app's render order: home, header, components. (A fresh
// app, or one that gains its parts after its home page,
// orders header above home — the parts test below pins
// that side.) The part's import survives the fmt
// import's removal.
func TestPlanWasmHomeWithHeaderPart(t *testing.T) {
	main := strings.Replace(legacyWasmMain, "\t\"fmt\"\n", "\t\"fmt\"\n\t\"jim/header\"\n", 1)
	main = strings.Replace(main, "\tmountComponents(doc, app)", "\tapp.Call(\"appendChild\", header.Header(doc))\n\tmountComponents(doc, app)", 1)
	dir, proj := wasmHomeProject(t, main, map[string]string{
		"jim/header/header.go": "//go:build js && wasm\n\npackage header\n",
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmHome, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("home plan with a header part refused: %q", refusal)
	}
	applyPlan(t, dir, plan)
	got := readWasmFile(t, dir, "jim/main.go")
	hi := strings.Index(got, "header.Header(doc)")
	ho := strings.Index(got, "Home(doc, app)")
	mi := strings.Index(got, "mountComponents(doc, app)")
	if hi < 0 || ho < 0 || mi < 0 || !(ho < hi && hi < mi) {
		t.Errorf("page order changed by the move (home=%d header=%d components=%d) — it must render as it did:\n%s", ho, hi, mi, got)
	}
	if !strings.Contains(got, `"jim/header"`) || strings.Contains(got, `"fmt"`) {
		t.Errorf("imports wrong after the move — the part's import stays, fmt goes:\n%s", got)
	}
}

// The parts planner on a fresh (new-skeleton) app: the
// header's mount lands above the home chunk, not between
// its comment and its line — page order header, home,
// components, footer.
func TestPlanWasmPartHeaderMountsAboveHome(t *testing.T) {
	dir, proj := wasmHomeProject(t, wasmAppMain(), map[string]string{"jim/home.go": wasmAppHome()})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})
	if refusal != "" || plan == nil {
		t.Fatalf("parts plan refused: %q", refusal)
	}
	applyPlan(t, dir, plan)
	got := readWasmFile(t, dir, "jim/main.go")
	hi := strings.Index(got, "header.Header(doc)")
	hc := strings.Index(got, "// The home page (home.go):")
	ho := strings.Index(got, "Home(doc, app)")
	mi := strings.Index(got, "mountComponents(doc, app)")
	fi := strings.Index(got, "footer.Footer(doc)")
	if !(hi < hc && hc < ho && ho < mi && mi < fi) {
		t.Errorf("mount order wrong (header=%d homecomment=%d home=%d components=%d footer=%d):\n%s", hi, hc, ho, mi, fi, got)
	}
}
