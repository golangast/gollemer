package analyze

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// The parts plan, fresh: two new packages in the app —
// header/header.go and footer/footer.go, each with its
// exported builder, its own markers, and its root builder
// stamped with its id — main.go importing both by the
// app's own module path and mounting header before the
// body components call and footer after it, the body
// markers untouched, and the two root kinds' styles
// joining the host page once.
func TestPlanWasmPartShape(t *testing.T) {
	dir, proj := wasmAppProject(t)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})
	if refusal != "" || plan == nil {
		t.Fatalf("parts refused: %q", refusal)
	}
	byRel := map[string]string{}
	for _, f := range plan.Files {
		byRel[f.Rel] = string(f.After)
	}
	header := byRel["jim/header/header.go"]
	for _, want := range []string{
		"//go:build js && wasm", "package header", "func Header(doc js.Value) js.Value {",
		"root := appBarComponent(doc)", "mountHeaderComponents(doc, root)",
		"// gollemer:header-components-begin", "// gollemer:header-components-end",
		`bar.Set("id", "app-bar-1")`, "app-bar-title",
	} {
		if !strings.Contains(header, want) {
			t.Errorf("header.go is missing %q", want)
		}
	}
	footer := byRel["jim/footer/footer.go"]
	for _, want := range []string{
		"//go:build js && wasm", "package footer", "func Footer(doc js.Value) js.Value {",
		"root := footerComponent(doc)", "mountFooterComponents(doc, root)",
		"// gollemer:footer-components-begin", "// gollemer:footer-components-end",
		`foot.Set("id", "footer-1")`, "site-footer",
	} {
		if !strings.Contains(footer, want) {
			t.Errorf("footer.go is missing %q", want)
		}
	}
	main := byRel["jim/main.go"]
	hdrMount := strings.Index(main, `app.Call("appendChild", header.Header(doc))`)
	bodyMount := strings.Index(main, "mountComponents(doc, app)")
	ftrMount := strings.Index(main, `app.Call("appendChild", footer.Footer(doc))`)
	if !(0 <= hdrMount && hdrMount < bodyMount && bodyMount < ftrMount) {
		t.Errorf("mount order wrong: header %d, body %d, footer %d\n%s", hdrMount, bodyMount, ftrMount, main)
	}
	for _, want := range []string{`"jim/header"`, `"jim/footer"`, "// gollemer:components-begin", "// gollemer:components-end"} {
		if !strings.Contains(main, want) {
			t.Errorf("main.go is missing %q", want)
		}
	}
	if strings.Index(main, `"jim/footer"`) > strings.Index(main, `"jim/header"`) {
		t.Errorf("imports not in sorted order:\n%s", main)
	}
	index := byRel["jim/index.html"]
	appBar, _ := MaterialComponentByID("app-bar")
	foot, _ := MaterialComponentByID("footer")
	if !strings.Contains(index, appBar.CSS) || !strings.Contains(index, foot.CSS) {
		t.Errorf("the parts' styles did not join the host page")
	}
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})
	_ = proj
	// The instances read the parts as parts: roots first
	// and last, in page order.
	insts := proj.WasmAppComponents("jim")
	if len(insts) != 2 || insts[0].Kind != "app-bar" || insts[0].Part != "header" || !insts[0].Root ||
		insts[1].Kind != "footer" || insts[1].Part != "footer" || !insts[1].Root {
		t.Errorf("instances after parts = %+v", insts)
	}
}

// One part at a time; asking again is the already-there
// answer; the combined ask with one part present lands
// the other and notes the one that's there.
func TestPlanWasmPartSingleAndAlreadyThere(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true})
	if _, err := os.Stat(filepath.Join(dir, "jim", "header", "header.go")); err != nil {
		t.Fatalf("header.go not landed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "jim", "footer")); !os.IsNotExist(err) {
		t.Fatalf("the footer landed with a header-only ask")
	}
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true}); plan != nil || !strings.Contains(refusal, "already has a header") {
		t.Errorf("re-ask header = plan %v, refusal %q", plan != nil, refusal)
	}
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})
	if refusal != "" || plan == nil {
		t.Fatalf("combined ask with header present refused: %q", refusal)
	}
	noted := false
	for _, n := range plan.Notes {
		if strings.Contains(n, "The header is already in the app") {
			noted = true
		}
	}
	if !noted {
		t.Errorf("the plan does not note the header already there: %v", plan.Notes)
	}
	applyPlan(t, dir, plan)
	proj2, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if plan, refusal := proj2.PlanEdit(EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true}); plan != nil || !strings.Contains(refusal, "already has a header and a footer") {
		t.Errorf("re-ask both = plan %v, refusal %q", plan != nil, refusal)
	}
}

// THE MOVE: a body app bar and a body footer move into
// their parts — builder text (and id) preserved in the
// part file, mount lines gone from main.go, builders gone
// from components.go, the other body component untouched,
// nothing rendering twice. A named app bar keeps its
// name through the move.
func TestPlanWasmPartMovesBodyComponents(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "app-bar", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "footer", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})

	header := readFile(t, dir, "jim/header/header.go")
	if !strings.Contains(header, "func appBarComponent(doc js.Value) js.Value {") || !strings.Contains(header, `bar.Set("id", "app-bar-1")`) {
		t.Errorf("the app bar did not move into header.go with its stamp:\n%s", header)
	}
	footer := readFile(t, dir, "jim/footer/footer.go")
	if !strings.Contains(footer, "func footerComponent(doc js.Value) js.Value {") || !strings.Contains(footer, `foot.Set("id", "footer-1")`) {
		t.Errorf("the footer did not move into footer.go with its stamp:\n%s", footer)
	}
	comp := readFile(t, dir, "jim/components.go")
	if strings.Contains(comp, "appBarComponent") || strings.Contains(comp, "footerComponent") {
		t.Errorf("the moved builders are still in components.go:\n%s", comp)
	}
	if !strings.Contains(comp, "func cardComponent(") {
		t.Errorf("the card was disturbed by the moves:\n%s", comp)
	}
	main := readFile(t, dir, "jim/main.go")
	if strings.Contains(main, "appBarComponent(doc)") || strings.Contains(main, "footerComponent(doc)") {
		t.Errorf("the moved mount lines are still in main.go:\n%s", main)
	}
	if !strings.Contains(main, "cardComponent(doc)") {
		t.Errorf("the card's mount line was disturbed:\n%s", main)
	}
	insts := proj.WasmAppComponents("jim")
	var kinds []string
	for _, in := range insts {
		kinds = append(kinds, in.Kind+"@"+in.Part)
	}
	if strings.Join(kinds, ",") != "app-bar@header,card@,footer@footer" {
		t.Errorf("instance places after the moves = %v", kinds)
	}

	// A named app bar moves with its name.
	dir2, proj2 := wasmAppProject(t)
	proj2 = wasmAdd(t, dir2, proj2, EditRequest{Kind: EditComponent, Component: "app-bar", ComponentName: "top", WasmFolder: "jim"})
	proj2 = wasmAdd(t, dir2, proj2, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true})
	header2 := readFile(t, dir2, "jim/header/header.go")
	if !strings.Contains(header2, "func topAppBarComponent(doc js.Value) js.Value {") || !strings.Contains(header2, `bar.Set("id", "top")`) ||
		!strings.Contains(header2, "root := topAppBarComponent(doc)") {
		t.Errorf("the named app bar did not move with its name:\n%s", header2)
	}
	insts2 := proj2.WasmAppComponents("jim")
	if len(insts2) != 1 || insts2[0].Name != "top" || insts2[0].ID != "top" || !insts2[0].Root {
		t.Errorf("named move instances = %+v", insts2)
	}
}

// A move that can't be read safely refuses the part
// plainly: the body mounts an app bar whose builder is
// NOT in components.go (a hand edit) — the plan refuses
// and writes nothing. A file squatting on the part's
// path refuses the same honest way.
func TestPlanWasmPartMoveUnsafeRefuses(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "app-bar", WasmFolder: "jim"})
	// Hand-break the app: the builder leaves components.go
	// while its mount line stays in main.go.
	compAbs := filepath.Join(dir, "jim", "components.go")
	if err := os.WriteFile(compAbs, []byte(wasmComponentsHeader), 0o644); err != nil {
		t.Fatal(err)
	}
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true}); plan != nil ||
		!strings.Contains(refusal, "in a shape I can lift out whole") {
		t.Errorf("unsafe move = plan %v, refusal %q", plan != nil, refusal)
	}
	if _, err := os.Stat(filepath.Join(dir, "jim", "header")); !os.IsNotExist(err) {
		t.Errorf("the refused move wrote a header folder")
	}

	dir2, proj2 := wasmAppProject(t)
	if err := os.MkdirAll(filepath.Join(dir2, "jim", "header"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir2, "jim", "header", "header.go"), []byte("package header\n\n// mine\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if plan, refusal := proj2.PlanEdit(EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true}); plan != nil ||
		!strings.Contains(refusal, "isn't a header part") {
		t.Errorf("squatted part path = plan %v, refusal %q", plan != nil, refusal)
	}
}

// Components into parts: the builder joins the part's
// components.go, the mount lands inside the part's
// markers, styles join the host page once. Already-there
// is per part; a name taken anywhere in the app is taken.
func TestPlanComponentWasmPart(t *testing.T) {
	dir, proj := wasmAppProject(t)
	// No part yet: the two-step refusal names the part ask.
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "chip", WasmFolder: "jim", ComponentPart: "header"}); plan != nil ||
		!strings.Contains(refusal, "has no header yet") || !strings.Contains(refusal, `"add a header to the wasm app in jim"`) {
		t.Errorf("component into missing part = plan %v, refusal %q", plan != nil, refusal)
	}
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "navigation-bar", WasmFolder: "jim", ComponentPart: "header"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "chip", WasmFolder: "jim", ComponentPart: "footer"})

	hcomp := readFile(t, dir, "jim/header/components.go")
	if !strings.Contains(hcomp, "package header") || !strings.Contains(hcomp, "func navigationBarComponent(") || !strings.Contains(hcomp, `nav.Set("id", "navigation-bar-1")`) {
		t.Errorf("the navigation bar did not land in the header's components.go:\n%s", hcomp)
	}
	if strings.Contains(hcomp, "func el(") {
		t.Errorf("the part's components.go redefines the part's el helper:\n%s", hcomp)
	}
	header := readFile(t, dir, "jim/header/header.go")
	begin := strings.Index(header, "// gollemer:header-components-begin")
	end := strings.Index(header, "// gollemer:header-components-end")
	mount := strings.Index(header, `root.Call("appendChild", navigationBarComponent(doc))`)
	if !(begin >= 0 && begin < mount && mount < end) {
		t.Errorf("the nav mount is not inside the header markers:\n%s", header)
	}
	fcomp := readFile(t, dir, "jim/footer/components.go")
	if !strings.Contains(fcomp, "func chipComponent(") || !strings.Contains(fcomp, `c.Set("id", "chip-1")`) {
		t.Errorf("the chip did not land in the footer's components.go:\n%s", fcomp)
	}
	// Per-part already-there: a card in the body is not a
	// card in the header — but a second one in the header
	// is already there.
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim", ComponentPart: "header"})
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim", ComponentPart: "header"}); plan != nil ||
		!strings.Contains(refusal, "already has a card in it") {
		t.Errorf("second card in header = plan %v, refusal %q", plan != nil, refusal)
	}
	// A name taken in the body is taken in a part.
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"})
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "chip", ComponentName: "save", WasmFolder: "jim", ComponentPart: "footer"}); plan != nil ||
		!strings.Contains(refusal, "already has a component called save") {
		t.Errorf("body name reused in a part = plan %v, refusal %q", plan != nil, refusal)
	}
	// The header's own root is its app bar: an unnamed app
	// bar asked into the header is already there.
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "app-bar", WasmFolder: "jim", ComponentPart: "header"}); plan != nil ||
		!strings.Contains(refusal, "already has a app bar in it") && !strings.Contains(refusal, "already has an app bar in it") {
		t.Errorf("app bar into the header = plan %v, refusal %q", plan != nil, refusal)
	}
}

// Events across parts: a body button's click shows a
// chip inside the header; a switch inside the footer
// toggles a body card; the header's root app bar is a
// valid target. Ids are the ones the parts stamped.
func TestPlanWasmEventAcrossParts(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "chip", WasmFolder: "jim", ComponentPart: "header"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "switch", WasmFolder: "jim", ComponentPart: "footer"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})

	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceName: "save", EventName: "click",
		EventAction: "show", EventTargetKind: "chip",
	})
	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "switch", EventName: "change",
		EventAction: "toggle", EventTargetKind: "card",
	})
	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceName: "save", EventName: "click",
		EventAction: "hide", EventTargetKind: "app-bar",
	})
	events := readFile(t, dir, "jim/events.go")
	for _, want := range []string{
		"func onSaveClickShowChip1", "func onSwitch1ChangeToggleCard1", "func onSaveClickHideAppBar1",
		`showElement(doc, "chip-1")`, `toggleElement(doc, "card-1")`, `hideElement(doc, "app-bar-1")`,
	} {
		if !strings.Contains(events, want) {
			t.Errorf("events.go is missing %q:\n%s", want, events)
		}
	}
	// main.go mounts the parts BEFORE registerEvents, so
	// the handlers find their components on the page.
	main := readFile(t, dir, "jim/main.go")
	reg := strings.Index(main, "registerEvents(doc)")
	if !(strings.Index(main, "header.Header(doc)") < reg && strings.Index(main, "footer.Footer(doc)") < reg) {
		t.Errorf("parts do not mount before registerEvents:\n%s", main)
	}
}

// THE PARTS SMOKE: an app with parts, components in its
// body and both parts, and an event across them, built
// with the documented command, produces a real
// main.wasm — and the host project's own build still
// passes with the app inside it.
func TestWasmPartAppBuildsAsWasm(t *testing.T) {
	if _, _, ok := findWasmExecJS(); !ok {
		t.Skip("no Go 1.27 install on this machine — the app pins a toolchain that isn't here")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH")
	}
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true, WasmPartFooter: true})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "navigation-bar", WasmFolder: "jim", ComponentPart: "header"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "chip", WasmFolder: "jim", ComponentPart: "footer"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceName: "save", EventName: "click",
		EventAction: "hide", EventTargetKind: "chip",
	})
	buildFixture(t, dir)
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
	t.Logf("main.wasm with parts: %d bytes", st.Size())
}

// extractWasmBuilder: the middle of three builders lifts
// out whole — comment, stamp, closing brace — and the
// file that remains is one the piece could have written.
func TestExtractWasmBuilder(t *testing.T) {
	_, proj := wasmAppProject(t)
	_ = proj
	content := wasmComponentsHeader
	for _, kind := range []string{"button", "card", "chip"} {
		b, ok := wasmComponentBuilder(kind, wasmBuilderName(kind), kind+"-1")
		if !ok {
			t.Fatalf("no builder for %s", kind)
		}
		if content != wasmComponentsHeader {
			content = strings.TrimRight(content, "\n") + "\n\n"
		}
		content += b
	}
	block, rest, ok := extractWasmBuilder(content, "cardComponent")
	if !ok {
		t.Fatal("the card builder did not lift")
	}
	if !strings.Contains(block, "// A card: a heading") || !strings.Contains(block, `card.Set("id", "card-1")`) || !strings.HasSuffix(block, "}\n") {
		t.Errorf("the lifted block is wrong:\n%s", block)
	}
	if strings.Contains(rest, "cardComponent") || !strings.Contains(rest, "buttonComponent") || !strings.Contains(rest, "chipComponent") {
		t.Errorf("the remaining file is wrong:\n%s", rest)
	}
	if strings.Contains(rest, "\n\n\n") {
		t.Errorf("the remaining file has a doubled blank line:\n%s", rest)
	}
	if _, _, ok := extractWasmBuilder(content, "nopeComponent"); ok {
		t.Errorf("a missing builder reported a lift")
	}
}
