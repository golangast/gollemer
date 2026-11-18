package analyze

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// wasmAppProject builds a fixture project with a wasm app
// in jim and returns the project re-analyzed after each
// landing, the way the spine sees it.
func wasmAppProject(t *testing.T) (string, *Project) {
	t.Helper()
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
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatalf("re-analyze after the skeleton: %v", err)
	}
	return dir, proj
}

// wasmAdd plans and applies one edit, failing the test on
// a refusal, and returns the re-analyzed project.
func wasmAdd(t *testing.T, dir string, proj *Project, req EditRequest) *Project {
	t.Helper()
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("edit %+v refused: %q", req.Kind, refusal)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatalf("re-analyze: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Fatalf("edit %+v does not verify", req.Kind)
	}
	return fresh
}

// One click event: the handler joins a NEW events.go (with
// the keep-alive list and the helpers), its registration
// lands between the register markers, and main.go gains
// the registerEvents call right after the mounts. A change
// event after it accumulates — handler and registration in
// ask order — and main.go is not touched again. The same
// event asked twice is the already-there answer.
func TestPlanWasmEventLandsAndAccumulates(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "switch", WasmFolder: "jim"})

	plan, refusal := proj.PlanEdit(EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "button", EventName: "click",
		EventAction: "show", EventTargetKind: "card",
	})
	if refusal != "" || plan == nil {
		t.Fatalf("click event refused: %q", refusal)
	}
	byRel := map[string]string{}
	for _, f := range plan.Files {
		byRel[f.Rel] = string(f.After)
	}
	events := byRel["jim/events.go"]
	for _, want := range []string{
		"//go:build js && wasm",
		"var eventFuncs []js.Func",
		"garbage-collected",
		"func showElement(doc js.Value, id string)",
		"func hideElement(doc js.Value, id string)",
		"func toggleElement(doc js.Value, id string)",
		"// gollemer:events-begin", "// gollemer:events-end",
		"// When the button is clicked, show the card",
		"func onButton1ClickShowCard1(doc js.Value) {",
		`el := doc.Call("getElementById", "button-1")`,
		"js.FuncOf(func(this js.Value, args []js.Value) interface{} {",
		`showElement(doc, "card-1")`,
		"eventFuncs = append(eventFuncs, fn)",
		`el.Call("addEventListener", "click", fn)`,
	} {
		if !strings.Contains(events, want) {
			t.Errorf("events.go is missing %q:\n%s", want, events)
		}
	}
	main := byRel["jim/main.go"]
	mountAt := strings.Index(main, "mountComponents(doc, app)")
	regAt := strings.Index(main, "\tregisterEvents(doc)\n")
	if !(mountAt >= 0 && regAt > mountAt) {
		t.Errorf("registerEvents is not called right after the mounts:\n%s", main)
	}
	if _, touched := byRel["jim/components.go"]; touched {
		t.Error("the plan rewrote components.go — the builders already carried their ids")
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatalf("re-analyze after the event: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Error("the event plan does not verify against the re-analyzed project")
	}

	// A change event accumulates: registration after the
	// first, handler after the first, main.go untouched.
	plan2, refusal := fresh.PlanEdit(EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "switch", EventName: "change",
		EventAction: "toggle", EventTargetKind: "card",
	})
	if refusal != "" || plan2 == nil {
		t.Fatalf("change event refused: %q", refusal)
	}
	var events2 string
	for _, f := range plan2.Files {
		if f.Rel == "jim/main.go" {
			t.Error("the second event touched main.go — the register call was already there")
		}
		if f.Rel == "jim/events.go" {
			events2 = string(f.After)
		}
	}
	first := strings.Index(events2, "\n\tonButton1ClickShowCard1(doc)\n")
	second := strings.Index(events2, "\n\tonSwitch1ChangeToggleCard1(doc)\n")
	if !(0 <= first && first < second) {
		t.Errorf("registrations are not in ask order:\n%s", events2)
	}
	if !strings.Contains(events2, `el.Call("addEventListener", "change", fn)`) || !strings.Contains(events2, `toggleElement(doc, "card-1")`) {
		t.Errorf("the switch handler is missing its pieces:\n%s", events2)
	}
	applyPlan(t, dir, plan2)
	fresh, _ = Analyze(dir)

	// The same event again: already there, no plan.
	plan3, refusal := fresh.PlanEdit(EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "button", EventName: "click",
		EventAction: "show", EventTargetKind: "card",
	})
	if plan3 != nil || !strings.Contains(refusal, "already does that") {
		t.Errorf("repeat event = plan %v, refusal %q", plan3 != nil, refusal)
	}
}

// The set-words event on a named button: the handler sets
// the card's text, the name is the id it looks the button
// up by, and a repeat with different words is a named
// conflict — not a second handler fighting the first.
func TestPlanWasmEventSetWords(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})

	req := EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceName: "save", EventName: "click",
		EventAction: "setwords", EventTargetKind: "card", EventWords: "Saved",
	}
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("set-words event refused: %q", refusal)
	}
	var events string
	for _, f := range plan.Files {
		if f.Rel == "jim/events.go" {
			events = string(f.After)
		}
	}
	for _, want := range []string{
		"func onSaveClickSetCard1Words(doc js.Value) {",
		`el := doc.Call("getElementById", "save")`,
		`target := doc.Call("getElementById", "card-1")`,
		`target.Set("textContent", "Saved")`,
		"// When the save button is clicked, set the card's words to 'Saved'",
	} {
		if !strings.Contains(events, want) {
			t.Errorf("events.go is missing %q:\n%s", want, events)
		}
	}
	applyPlan(t, dir, plan)
	fresh, _ := Analyze(dir)
	if !plan.Verify(fresh) {
		t.Error("the set-words plan does not verify")
	}
	// The same words again: already there.
	if plan2, refusal := fresh.PlanEdit(req); plan2 != nil || !strings.Contains(refusal, "already does that") {
		t.Errorf("same-words repeat = plan %v, refusal %q", plan2 != nil, refusal)
	}
	// Different words: the conflict, naming what it sets now.
	other := req
	other.EventWords = "Done"
	if plan3, refusal := fresh.PlanEdit(other); plan3 != nil || !strings.Contains(refusal, "already answers that event by setting the card's words to 'Saved'") {
		t.Errorf("different-words repeat = plan %v, refusal %q", plan3 != nil, refusal)
	}
}

// The id backfill: an app whose components were written
// before builders carried ids (hand-planted in the old
// shape — the chip in its oldest, return-line shape) gets
// the stamps when an event first reaches for its
// components, with exactly the positional ids the
// instance reading assigned them.
func TestPlanWasmEventBackfillsLegacyIDs(t *testing.T) {
	oldChip := "// A chip: one small labeled pill.\nfunc chipComponent(doc js.Value) js.Value {\n\treturn el(doc, \"span\", \"chip\", \"TODO: the chip's words\")\n}\n"
	main := wasmAppMain()
	main = strings.Replace(main, "// gollemer:components-begin\n",
		"// gollemer:components-begin\n\tapp.Call(\"appendChild\", buttonComponent(doc))\n\tapp.Call(\"appendChild\", chipComponent(doc))\n", 1)
	comp := wasmComponentsHeader
	if b, ok := wasmComponentBuilderTemplate("button"); ok {
		comp += b
	} else {
		t.Fatal("no button template")
	}
	comp += "\n" + oldChip
	dir, proj := serveProject(t, map[string]string{
		"go.mod":            "module example.com/host\n\ngo 1.21\n",
		"doc.go":            "package host\n",
		"jim/go.mod":        wasmAppGoMod("jim"),
		"jim/main.go":       main,
		"jim/index.html":    wasmAppIndex(),
		"jim/components.go": comp,
	})
	plan, refusal := proj.PlanEdit(EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "button", EventName: "click",
		EventAction: "toggle", EventTargetKind: "chip",
	})
	if refusal != "" || plan == nil {
		t.Fatalf("event into a legacy app refused: %q", refusal)
	}
	var newComp string
	for _, f := range plan.Files {
		if f.Rel == "jim/components.go" {
			newComp = string(f.After)
		}
	}
	if newComp == "" {
		t.Fatal("the plan left components.go out — the backfill is missing")
	}
	for _, want := range []string{
		`b.Set("id", "button-1")`,
		"c := el(doc, \"span\", \"chip\", \"TODO: the chip's words\")",
		`c.Set("id", "chip-1")`,
		"return c",
	} {
		if !strings.Contains(newComp, want) {
			t.Errorf("backfilled components.go is missing %q:\n%s", want, newComp)
		}
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "had no id yet") {
		t.Errorf("the backfill note is missing:\n%s", notes)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatalf("re-analyze after the backfill: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Error("the backfill plan does not verify")
	}
}

// Component naming, at the planner: named instances are
// distinct (their own builders, their name as the id),
// they accumulate beside the kind's one unnamed instance
// (whose id counts them), a taken name is refused across
// kinds, and the unnamed duplicate keeps its exact
// already-there answer.
func TestPlanComponentWasmNamedInstances(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"})
	comp := readFile(t, dir, "jim/components.go")
	for _, want := range []string{"func saveButtonComponent(doc js.Value) js.Value", `b.Set("id", "save")`} {
		if !strings.Contains(comp, want) {
			t.Errorf("components.go is missing %q:\n%s", want, comp)
		}
	}
	main := readFile(t, dir, "jim/main.go")
	if !strings.Contains(main, `app.Call("appendChild", saveButtonComponent(doc))`) {
		t.Errorf("the named mount line is missing:\n%s", main)
	}

	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "cancel", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", WasmFolder: "jim"})
	comp = readFile(t, dir, "jim/components.go")
	if !strings.Contains(comp, `b.Set("id", "button-3")`) {
		t.Errorf("the unnamed third button's id does not count its named siblings:\n%s", comp)
	}
	// The unnamed duplicate: the shipped already-there answer.
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "button", WasmFolder: "jim"}); plan != nil || !strings.Contains(refusal, "already has a button in it — nothing to add.") {
		t.Errorf("unnamed duplicate = plan %v, refusal %q", plan != nil, refusal)
	}
	// The same name again, and the name on another kind.
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"}); plan != nil || !strings.Contains(refusal, "already has a button called save") {
		t.Errorf("same name, same kind = plan %v, refusal %q", plan != nil, refusal)
	}
	if plan, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, Component: "card", ComponentName: "save", WasmFolder: "jim"}); plan != nil || !strings.Contains(refusal, "already has a component called save") {
		t.Errorf("same name, other kind = plan %v, refusal %q", plan != nil, refusal)
	}
	// The instances read back exactly as they landed.
	insts := proj.WasmAppComponents("jim")
	if len(insts) != 3 {
		t.Fatalf("instances = %+v, want 3", insts)
	}
	want := []WasmComponentInstance{
		{Kind: "button", Name: "save", ID: "save", Builder: "saveButtonComponent"},
		{Kind: "button", Name: "cancel", ID: "cancel", Builder: "cancelButtonComponent"},
		{Kind: "button", Name: "", ID: "button-3", Builder: "buttonComponent"},
	}
	for i, w := range want {
		if insts[i] != w {
			t.Errorf("instance %d = %+v, want %+v", i, insts[i], w)
		}
	}
}

// The events refusals: each closed set and each missing
// piece named in words a user can act on.
func TestPlanWasmEventRefusals(t *testing.T) {
	_, proj := wasmAppProject(t)
	if _, refusal := proj.PlanEdit(EditRequest{
		Kind: EditWasmEvent, WasmFolder: "ghost",
		EventSourceKind: "button", EventName: "click", EventAction: "show", EventTargetKind: "card",
	}); !strings.Contains(refusal, "add a wasm app in ghost") {
		t.Errorf("no-app refusal = %q", refusal)
	}

	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "switch", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "chip", WasmFolder: "jim"})
	ev := func(src, name, event, action, tgt string) EditRequest {
		return EditRequest{
			Kind: EditWasmEvent, WasmFolder: "jim",
			EventSourceKind: src, EventSourceName: name, EventName: event,
			EventAction: action, EventTargetKind: tgt,
		}
	}
	// Before any button exists: the missing-source answer.
	if plan, refusal := proj.PlanEdit(ev("button", "", "click", "show", "card")); plan != nil || !strings.Contains(refusal, "has no button in it yet") {
		t.Errorf("missing button = plan %v, refusal %q", plan != nil, refusal)
	}
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "cancel", WasmFolder: "jim"})
	rows := []struct {
		req  EditRequest
		want string
	}{
		{ev("", "ghost", "click", "show", "card"), "no component called ghost"},    // a name the app doesn't have
		{ev("chip", "", "click", "show", "card"), "doesn't take a click"},          // click outside its set
		{ev("switch", "", "click", "show", "card"), "has no click to answer"},      // the switch's own event named
		{ev("button", "save", "change", "show", "card"), "doesn't change in this"}, // change outside its set
		{ev("button", "save", "click", "setwords", "switch"), "only a button's or a card's words can be set"},
		{ev("button", "", "click", "show", "card"), "more than one button"},          // two named buttons, no name in the sentence
		{ev("button", "save", "click", "show", "footer"), "has no footer in it yet"}, // the target isn't there
	}
	for _, tc := range rows {
		if plan, refusal := proj.PlanEdit(tc.req); plan != nil || !strings.Contains(refusal, tc.want) {
			t.Errorf("req %+v = plan %v, refusal %q — want refusal containing %q", tc.req, plan != nil, refusal, tc.want)
		}
	}
}

// THE EVENTS SMOKE: an app with a named button, a card,
// and a switch, plus two events, built with the documented
// command, produces a real main.wasm — and the host
// project's own build still passes with the app inside it.
func TestWasmEventAppBuildsAsWasm(t *testing.T) {
	if _, _, ok := findWasmExecJS(); !ok {
		t.Skip("no Go 1.27 install on this machine — the app pins a toolchain that isn't here")
	}
	goBin, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go toolchain not on PATH")
	}
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "button", ComponentName: "save", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "card", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, Component: "switch", WasmFolder: "jim"})
	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceName: "save", EventName: "click",
		EventAction: "setwords", EventTargetKind: "card", EventWords: "Saved",
	})
	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "switch", EventName: "change",
		EventAction: "toggle", EventTargetKind: "card",
	})
	events := readFile(t, dir, "jim/events.go")
	for _, want := range []string{"js.FuncOf", "addEventListener", "func onSaveClickSetCard1Words", "func onSwitch1ChangeToggleCard1"} {
		if !strings.Contains(events, want) {
			t.Errorf("events.go is missing %q", want)
		}
	}
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
	t.Logf("main.wasm with events: %d bytes", st.Size())
}
