package analyze

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// plantWasmExec plants a fake Go install tree in a module
// cache: golang.org/toolchain@…-go<version>.linux-amd64
// holding a sentinel wasm_exec.js, so the lookup's
// highest-1.27.x rule is pinned on any machine.
func plantWasmExec(t *testing.T, modcache, version, sentinel string) {
	t.Helper()
	dir := filepath.Join(modcache, "golang.org", "toolchain@v0.0.1-go"+version+".linux-amd64", "lib", "wasm")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "wasm_exec.js"), []byte(sentinel), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The skeleton's plan shape: five new files in the folder —
// the app's own module pinning Go 1.27, the main with the
// GOGC line and the mount markers, the simd unit, the host
// page — and the wasm_exec.js copied from the highest Go
// 1.27 in the module cache (planted here, so the bytes are
// pinned exactly).
func TestPlanWasmAppShape(t *testing.T) {
	modcache := t.TempDir()
	plantWasmExec(t, modcache, "1.27.1", "// wasm_exec 1.27.1\n")
	plantWasmExec(t, modcache, "1.27.9", "// wasm_exec 1.27.9 sentinel\n")
	t.Setenv("GOMODCACHE", modcache)
	dir, proj := serveProject(t, map[string]string{
		"go.mod":    "module example.com/host\n\ngo 1.21\n",
		"doc.go":    "package host\n",
		"jim/.keep": "",
	})
	// The folder needs to exist as a folder: a placeholder
	// stands for the (empty) folder the chat's create made.
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("wasm app refused: %q", refusal)
	}
	byRel := map[string]string{}
	for _, f := range plan.Files {
		if len(f.Before) != 0 {
			t.Errorf("%s: the skeleton edits an existing file", f.Rel)
		}
		byRel[f.Rel] = string(f.After)
	}
	if got := byRel["jim/go.mod"]; got != "module jim\n\ngo 1.27.0\n" {
		t.Errorf("go.mod = %q, want the app's own module pinning Go 1.27.0", got)
	}
	main := byRel["jim/main.go"]
	for _, want := range []string{
		"//go:build js && wasm", `"syscall/js"`, `"runtime/debug"`,
		"debug.SetGCPercent(50)", "GOGC setting, turned down",
		"// gollemer:components-begin", "// gollemer:components-end",
		"AddVectors(", "select {}",
	} {
		if !strings.Contains(main, want) {
			t.Errorf("main.go is missing %q", want)
		}
	}
	vector := byRel["jim/vector.go"]
	for _, want := range []string{`"simd"`, "func AddVectors(", "simd.LoadFloat32s", "simd.LoadFloat32sPart", "GOEXPERIMENT=simd"} {
		if !strings.Contains(vector, want) {
			t.Errorf("vector.go is missing %q", want)
		}
	}
	index := byRel["jim/index.html"]
	for _, want := range []string{WasmBuildCommand, `src="wasm_exec.js"`, `fetch("main.wasm")`, `id="app"`, MaterialStyleMarker} {
		if !strings.Contains(index, want) {
			t.Errorf("index.html is missing %q", want)
		}
	}
	if got := byRel["jim/wasm_exec.js"]; got != "// wasm_exec 1.27.9 sentinel\n" {
		t.Errorf("wasm_exec.js = %q, want the highest planted 1.27.x copy", got)
	}
	notes := strings.Join(plan.Notes, "\n")
	for _, want := range []string{"Go 1.27", "SetGCPercent(50)", WasmBuildCommand, "Go 1.27.9 install"} {
		if !strings.Contains(notes, want) {
			t.Errorf("plan notes are missing %q:\n%s", want, notes)
		}
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatalf("the project no longer analyzes with the app in it: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Error("the skeleton plan does not verify against the re-analyzed project")
	}
}

// With no Go 1.27 install anywhere the lookup can see, the
// plan still lands — without wasm_exec.js — and says so in
// the open, naming the exact manual copy.
func TestPlanWasmAppWithoutExecJS(t *testing.T) {
	t.Setenv("GOMODCACHE", t.TempDir())
	t.Setenv("HOME", t.TempDir())
	_, proj := serveProject(t, map[string]string{
		"go.mod":    "module example.com/host\n\ngo 1.21\n",
		"doc.go":    "package host\n",
		"jim/.keep": "",
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("wasm app refused: %q", refusal)
	}
	for _, f := range plan.Files {
		if f.Rel == "jim/wasm_exec.js" {
			t.Fatal("the plan carries a wasm_exec.js it could not have found")
		}
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "wasm_exec.js is NOT in this plan") || !strings.Contains(notes, "lib/wasm/wasm_exec.js") {
		t.Errorf("the manual-copy note is missing or vague:\n%s", notes)
	}
}

// The skeleton's refusals: the project root, a missing
// folder, and any file already there — named, with nothing
// planned.
func TestPlanWasmAppRefusals(t *testing.T) {
	base := map[string]string{
		"go.mod": "module example.com/host\n\ngo 1.21\n",
		"doc.go": "package host\n",
	}
	_, proj := serveProject(t, base)
	if _, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "."}); !strings.Contains(refusal, "folder of its own") {
		t.Errorf("root refusal = %q", refusal)
	}
	if _, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "ghost"}); !strings.Contains(refusal, "create a folder ghost") {
		t.Errorf("missing-folder refusal = %q", refusal)
	}
	_, proj2 := serveProject(t, map[string]string{
		"go.mod":      "module example.com/host\n\ngo 1.21\n",
		"doc.go":      "package host\n",
		"jim/main.go": "package main\n",
	})
	plan, refusal := proj2.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "jim"})
	if plan != nil || !strings.Contains(refusal, "jim/main.go is already there") {
		t.Errorf("existing-file refusal = plan %v, %q", plan != nil, refusal)
	}
}

// A card into the app: the builder joins components.go
// (created with the shared helper), the mount line lands
// between main.go's markers, and the card's styles join
// the host page exactly once. A button after it accumulates
// in ask order; a second card is the already-there answer.
func TestPlanComponentWasmCardLands(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":    "module example.com/host\n\ngo 1.21\n",
		"doc.go":    "package host\n",
		"jim/.keep": "",
	})
	app, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "jim"})
	if refusal != "" {
		t.Fatalf("skeleton refused: %q", refusal)
	}
	applyPlan(t, dir, app)
	proj, _ = Analyze(dir)

	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("card into the app refused: %q", refusal)
	}
	byRel := map[string]string{}
	for _, f := range plan.Files {
		byRel[f.Rel] = string(f.After)
	}
	comp := byRel["jim/components.go"]
	for _, want := range []string{"//go:build js && wasm", "func el(doc js.Value", "func cardComponent(doc js.Value) js.Value", "TODO: the card's heading"} {
		if !strings.Contains(comp, want) {
			t.Errorf("components.go is missing %q", want)
		}
	}
	if strings.Contains(comp, "FuncOf") || strings.Contains(comp, "addEventListener") {
		t.Error("the builder carries an event hook — v1 renders and mounts only")
	}
	main := byRel["jim/main.go"]
	begin := strings.Index(main, "// gollemer:components-begin")
	mount := strings.Index(main, `app.Call("appendChild", cardComponent(doc))`)
	end := strings.Index(main, "// gollemer:components-end")
	if !(begin >= 0 && begin < mount && mount < end) {
		t.Errorf("the card's mount line is not between the markers:\n%s", main)
	}
	index := byRel["jim/index.html"]
	cardComp, _ := MaterialComponentByID("card")
	if strings.Count(index, ".card {") != 1 || !strings.Contains(index, cardComp.CSS) {
		t.Error("the card's styles did not join the host page exactly once, whole")
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatalf("the project no longer analyzes with the component in it: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Error("the card plan does not verify against the re-analyzed project")
	}

	// The same card again: already there, no plan.
	plan2, refusal := fresh.PlanEdit(EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	if plan2 != nil || !strings.Contains(refusal, "already has a card") {
		t.Errorf("second card = plan %v, refusal %q", plan2 != nil, refusal)
	}
	// A button accumulates after the card, in ask order,
	// and carries no styles twice.
	plan3, refusal := fresh.PlanEdit(EditRequest{Kind: EditComponent, Component: "button", WasmFolder: "jim"})
	if refusal != "" || plan3 == nil {
		t.Fatalf("button into the app refused: %q", refusal)
	}
	byRel3 := map[string]string{}
	for _, f := range plan3.Files {
		byRel3[f.Rel] = string(f.After)
	}
	// The host page is in this plan too: the buttons chunk
	// is new there (the variables already ride from the
	// skeleton), and it must land exactly once.
	if _, touched := byRel3["jim/index.html"]; !touched {
		t.Error("the button's plan left the host page out — its styles chunk had somewhere to go")
	}
	comp3 := byRel3["jim/components.go"]
	if strings.Index(comp3, "func cardComponent") > strings.Index(comp3, "func buttonComponent") {
		t.Error("the button's builder did not land after the card's")
	}
	main3 := byRel3["jim/main.go"]
	if strings.Index(main3, "cardComponent(doc)") > strings.Index(main3, "buttonComponent(doc)") {
		t.Error("the mounts are not in ask order")
	}
	applyPlan(t, dir, plan3)
	indexAfter := readFile(t, dir, "jim/index.html")
	if strings.Count(indexAfter, ".btn {") != 1 || strings.Count(indexAfter, ".card {") != 1 || strings.Count(indexAfter, "--primary: #6750A4") != 1 {
		t.Error("styles were not carried exactly once across the two components")
	}
}

// The wasm component refusals: no app in the folder, a
// main.go that is not a wasm main, and an app whose mount
// markers were taken out by hand.
func TestPlanComponentWasmRefusals(t *testing.T) {
	mk := func(files map[string]string) *Project {
		base := map[string]string{
			"go.mod": "module example.com/host\n\ngo 1.21\n",
			"doc.go": "package host\n",
		}
		for rel, src := range files {
			base[rel] = src
		}
		_, proj := serveProject(t, base)
		return proj
	}
	if _, refusal := mk(nil).PlanEdit(EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"}); !strings.Contains(refusal, "add a wasm app in jim") {
		t.Errorf("no-app refusal = %q", refusal)
	}
	plain := mk(map[string]string{"jim/main.go": "package main\n\nfunc main() {}\n"})
	if _, refusal := plain.PlanEdit(EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"}); !strings.Contains(refusal, "isn't a wasm app's main file") {
		t.Errorf("not-wasm refusal = %q", refusal)
	}
	stripped := wasmAppMain()
	stripped = strings.Replace(stripped, "// gollemer:components-begin\n", "", 1)
	noMarkers := mk(map[string]string{
		"jim/main.go":    stripped,
		"jim/index.html": wasmAppIndex(),
		"jim/go.mod":     "module jim\n\ngo 1.27.0\n",
	})
	if _, refusal := noMarkers.PlanEdit(EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"}); !strings.Contains(refusal, "components block") {
		t.Errorf("no-markers refusal = %q", refusal)
	}
}

// Every registry component has a builder, and each builder
// carries its HTML twin's classes — the two forms of a
// component cannot drift apart in what they look like.
func TestWasmBuildersCoverTheRegistry(t *testing.T) {
	for _, c := range MaterialComponents {
		builder, ok := wasmComponentBuilder(c.ID)
		if !ok {
			t.Errorf("component %s has no wasm builder", c.ID)
			continue
		}
		fn := wasmBuilderName(c.ID)
		if !strings.Contains(builder, "func "+fn+"(doc js.Value) js.Value") {
			t.Errorf("builder for %s does not declare %s", c.ID, fn)
		}
		if strings.Contains(builder, "FuncOf") || strings.Contains(builder, "addEventListener") {
			t.Errorf("builder for %s carries an event hook — v1 renders only", c.ID)
		}
	}
}

// THE WASM SMOKE: the skeleton and two components, built
// with the documented command in the app folder, produce a
// real main.wasm — and the host project's own build still
// passes with the app sitting inside it (the nested module
// keeps the app's files out of it).
func TestWasmAppBuildsAsWasm(t *testing.T) {
	if _, _, ok := findWasmExecJS(); !ok {
		t.Skip("no Go 1.27 install on this machine — the app pins a toolchain that isn't here")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH")
	}
	dir, proj := serveProject(t, map[string]string{
		"go.mod":    "module example.com/host\n\ngo 1.21\n",
		"doc.go":    "package host\n",
		"jim/.keep": "",
	})
	app, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "jim"})
	if refusal != "" {
		t.Fatalf("skeleton refused: %q", refusal)
	}
	applyPlan(t, dir, app)
	proj, _ = Analyze(dir)
	for _, comp := range []string{"card", "switch"} {
		plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: comp, WasmFolder: "jim"})
		if refusal != "" {
			t.Fatalf("%s into the app refused: %q", comp, refusal)
		}
		applyPlan(t, dir, plan)
		proj, _ = Analyze(dir)
	}
	// The host build, with the app inside the project.
	buildFixture(t, dir)
	// The app build, with the documented command.
	cmd := exec.Command(goBin, "build", "-o", "main.wasm", ".")
	cmd.Dir = filepath.Join(dir, "jim")
	cmd.Env = append(os.Environ(), "GOEXPERIMENT=simd", "GOOS=js", "GOARCH=wasm")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("the documented wasm build failed: %v\n%s", err, out)
	}
	st, err := os.Stat(filepath.Join(dir, "jim", "main.wasm"))
	if err != nil || st.Size() < 100_000 {
		t.Fatalf("main.wasm is missing or implausibly small: %v", err)
	}
	t.Logf("main.wasm: %d bytes", st.Size())
}

func readFile(t *testing.T, dir, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, rel))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
