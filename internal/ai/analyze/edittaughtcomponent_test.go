package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The taught-component planner: a builder template the
// chat cut from the user's own example (its name and id
// slots unfilled), landed the registry's way — marker,
// stamp, mount, already-there — in the body and in a part,
// and wired into events by the same instances reading the
// registry kinds use.

const taughtPriceTagTemplate = `func {{name}}(doc js.Value) js.Value {
	tag := el(doc, "div", "price-tag", "")
	tag.Set("id", "{{id}}")
	price := el(doc, "span", "", "$0")
	tag.Call("appendChild", price)
	return tag
}`

func taughtCompReq() EditRequest {
	return EditRequest{
		Kind:                    EditTaughtComponent,
		WasmFolder:              "jim",
		TaughtKindPhrase:        "price tag",
		TaughtComponentTemplate: taughtPriceTagTemplate,
		TaughtComponentRootVar:  "tag",
		TaughtComponentStyled:   true,
		TaughtNote:              "From your example PriceTag in proto/components.go, taught Nov 10, 2026.",
	}
}

func TestTaughtComponentNamesAndMarkers(t *testing.T) {
	if got := wasmTaughtBuilderName("price tag"); got != "priceTagComponent" {
		t.Errorf("wasmTaughtBuilderName = %q", got)
	}
	if got := wasmTaughtNamedBuilderName("total", "price tag"); got != "totalPriceTagComponent" {
		t.Errorf("wasmTaughtNamedBuilderName = %q", got)
	}
	if got := wasmTaughtKindSlug("price tag"); got != "price-tag" {
		t.Errorf("wasmTaughtKindSlug = %q", got)
	}
	if got := wasmTaughtKindCamel("sale banner"); got != "SaleBanner" {
		t.Errorf("wasmTaughtKindCamel = %q", got)
	}
	content := "// gollemer:taught-component kind=\"price tag\" root=tag\nfunc priceTagComponent(doc js.Value) js.Value {\n\treturn tag\n}\n\nfunc cardComponent(doc js.Value) js.Value {\n\treturn card\n}\n"
	marks := taughtComponentMarkers(content)
	if len(marks) != 1 || marks["priceTagComponent"].Kind != "price tag" || marks["priceTagComponent"].RootVar != "tag" {
		t.Errorf("markers = %+v", marks)
	}
	rendered := renderTaughtBuilder(taughtCompReq(), "priceTagComponent", "price-tag-1")
	for _, want := range []string{
		"// gollemer:taught-component kind=\"price tag\" root=tag",
		"func priceTagComponent(doc js.Value) js.Value {",
		"tag.Set(\"id\", \"price-tag-1\")",
	} {
		if !strings.Contains(rendered, want) {
			t.Errorf("rendered builder is missing %q:\n%s", want, rendered)
		}
	}
	if strings.Contains(rendered, "{{") {
		t.Errorf("rendered builder still carries a slot:\n%s", rendered)
	}
}

func TestPlanTaughtComponentLandsInBody(t *testing.T) {
	dir, proj := wasmAppProject(t)
	plan, refusal := proj.PlanEdit(taughtCompReq())
	if refusal != "" || plan == nil {
		t.Fatalf("taught component refused: %q", refusal)
	}
	if len(plan.Files) != 2 {
		t.Fatalf("plan files = %d, want the components file and main only (no styles travel): %v", len(plan.Files), plan.Files)
	}
	byRel := map[string]string{}
	for _, f := range plan.Files {
		byRel[f.Rel] = string(f.After)
	}
	comp := byRel["jim/components.go"]
	for _, want := range []string{
		"// gollemer:taught-component kind=\"price tag\" root=tag",
		"func priceTagComponent(doc js.Value) js.Value {",
		"tag.Set(\"id\", \"price-tag-1\")",
		"price := el(doc, \"span\", \"\", \"$0\")",
	} {
		if !strings.Contains(comp, want) {
			t.Errorf("components.go is missing %q:\n%s", want, comp)
		}
	}
	if main := byRel["jim/main.go"]; !strings.Contains(main, "app.Call(\"appendChild\", priceTagComponent(doc))") {
		t.Errorf("main.go does not mount the taught builder:\n%s", main)
	}
	if !strings.Contains(plan.Summary, "Put a price tag in the wasm app in jim") {
		t.Errorf("summary = %q", plan.Summary)
	}
	notes := strings.Join(plan.Notes, "\n")
	for _, want := range []string{
		"From your example PriceTag in proto/components.go, taught Nov 10, 2026.",
		"no styles travel with a taught component in this version",
	} {
		if !strings.Contains(notes, want) {
			t.Errorf("notes are missing %q:\n%s", want, notes)
		}
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatalf("re-analyze: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Error("the taught landing does not verify")
	}
	insts := fresh.WasmAppComponents("jim")
	var found *WasmComponentInstance
	for i := range insts {
		if insts[i].Builder == "priceTagComponent" {
			found = &insts[i]
		}
	}
	if found == nil {
		t.Fatalf("no instance read back for the taught builder: %+v", insts)
	}
	if !found.Taught || found.Kind != "price tag" || found.ID != "price-tag-1" || found.Name != "" || found.Part != "" {
		t.Errorf("taught instance = %+v", *found)
	}
	// The same unnamed ask again: already there.
	if _, refusal := fresh.PlanEdit(taughtCompReq()); !strings.Contains(refusal, "already has a price tag in it — nothing to add") {
		t.Errorf("the repeat unnamed refusal = %q", refusal)
	}
}

func TestPlanTaughtComponentNamedAndDeclares(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, taughtCompReq()) // the unnamed one, id price-tag-1
	req := taughtCompReq()
	req.ComponentName = "total"
	proj = wasmAdd(t, dir, proj, req)
	var named *WasmComponentInstance
	insts := proj.WasmAppComponents("jim")
	for i := range insts {
		if insts[i].Builder == "totalPriceTagComponent" {
			named = &insts[i]
		}
	}
	if named == nil || named.Name != "total" || named.ID != "total" || !named.Taught {
		t.Fatalf("named instance = %+v", named)
	}
	// The same name for the same kind: already there.
	if _, refusal := proj.PlanEdit(req); !strings.Contains(refusal, "already has a price tag called total — nothing to add") {
		t.Errorf("the repeat named refusal = %q", refusal)
	}
	// The id the unnamed instance carries is taken as an
	// id, not giveable to another instance.
	byID := taughtCompReq()
	byID.ComponentName = "price-tag-1"
	if _, refusal := proj.PlanEdit(byID); !strings.Contains(refusal, "whose id is price-tag-1") {
		t.Errorf("the id-collision refusal = %q", refusal)
	}
	// No template: the planner's own honesty.
	bare := taughtCompReq()
	bare.TaughtComponentTemplate = ""
	if _, refusal := proj.PlanEdit(bare); !strings.Contains(refusal, "needs a taught component to add") {
		t.Errorf("the no-template refusal = %q", refusal)
	}
}

func TestPlanTaughtComponentUnnamedDeclaresRefusal(t *testing.T) {
	dir, proj := wasmAppProject(t)
	// His own unmounted prototype sits in the app under
	// the very name the unnamed instance would take. (The
	// skeleton creates no components.go — the first
	// component landing does — so the prototype's file is
	// written whole, header and all.)
	comp := filepath.Join(dir, "jim", "components.go")
	src := wasmComponentsHeader + "func priceTagComponent(doc js.Value) js.Value {\n\ttag := el(doc, \"div\", \"price-tag\", \"\")\n\ttag.Set(\"id\", \"pt-0\")\n\treturn tag\n}\n"
	if err := os.WriteFile(comp, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	var err error
	proj, err = Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, refusal := proj.PlanEdit(taughtCompReq())
	for _, want := range []string{"already declares priceTagComponent", "give the price tag a name"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("the declares refusal %q is missing %q", refusal, want)
		}
	}
}

func TestPlanTaughtComponentInPart(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditWasmPart, WasmFolder: "jim", WasmPartHeader: true})
	req := taughtCompReq()
	req.ComponentPart = "header"
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		t.Fatalf("taught part component refused: %q", refusal)
	}
	byRel := map[string]string{}
	for _, f := range plan.Files {
		byRel[f.Rel] = string(f.After)
	}
	if comp := byRel["jim/header/components.go"]; !strings.Contains(comp, "func priceTagComponent(doc js.Value) js.Value {") ||
		!strings.Contains(comp, "// gollemer:taught-component kind=\"price tag\" root=tag") {
		t.Errorf("the header's components.go does not carry the taught builder:\n%s", comp)
	}
	if hdr := byRel["jim/header/header.go"]; !strings.Contains(hdr, "root.Call(\"appendChild\", priceTagComponent(doc))") {
		t.Errorf("the header does not mount the taught builder:\n%s", hdr)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatalf("re-analyze: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Error("the taught part landing does not verify")
	}
	var partInst *WasmComponentInstance
	insts := fresh.WasmAppComponents("jim")
	for i := range insts {
		if insts[i].Taught {
			partInst = &insts[i]
		}
	}
	if partInst == nil || partInst.Part != "header" || partInst.Kind != "price tag" || partInst.ID != "price-tag-1" {
		t.Fatalf("part instance = %+v", partInst)
	}
	// The same unnamed ask in the same part: already
	// there. In the body it is still free to land.
	if _, refusal := fresh.PlanEdit(req); !strings.Contains(refusal, "already has a price tag in it") {
		t.Errorf("the repeat part refusal = %q", refusal)
	}
	if _, refusal := fresh.PlanEdit(taughtCompReq()); refusal != "" {
		t.Errorf("the body add after the part add refused: %q", refusal)
	}
}

// A taught component in events: a click source by its
// whole contract (a root element with an id), a show
// target like any component, and the two honest refusals —
// changing is not its event, and its words are its own.
func TestPlanTaughtComponentEvents(t *testing.T) {
	dir, proj := wasmAppProject(t)
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, WasmFolder: "jim", Component: "button", ComponentName: "save"})
	proj = wasmAdd(t, dir, proj, EditRequest{Kind: EditComponent, WasmFolder: "jim", Component: "card"})
	proj = wasmAdd(t, dir, proj, taughtCompReq())

	// Taught source, click, registry target — while the
	// kind has one instance, the bare kind names it.
	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "price tag", EventName: "click",
		EventAction: "show", EventTargetKind: "card",
	})
	named := taughtCompReq()
	named.ComponentName = "total"
	proj = wasmAdd(t, dir, proj, named)
	// Registry source, taught target — by name.
	proj = wasmAdd(t, dir, proj, EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "button", EventSourceName: "save", EventName: "click",
		EventAction: "show", EventTargetKind: "price tag", EventTargetName: "total",
	})
	events, err := os.ReadFile(filepath.Join(dir, "jim", "events.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"doc.Call(\"getElementById\", \"price-tag-1\")",
		"showElement(doc, \"total\")",
	} {
		if !strings.Contains(string(events), want) {
			t.Errorf("events.go is missing %q:\n%s", want, events)
		}
	}
	proj, err = Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	// Changing is not a taught component's event (asked
	// of the named one — the bare kind, with two instances
	// in the app, is the ambiguity answer's business).
	if _, refusal := proj.PlanEdit(EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "price tag", EventSourceName: "total", EventName: "change",
		EventAction: "show", EventTargetKind: "card",
	}); !strings.Contains(refusal, "a price tag doesn't change in this set") {
		t.Errorf("the change refusal = %q", refusal)
	}
	// Its words are its own: the set-words refusal the
	// grammar gives every kind outside the settable set.
	if _, refusal := proj.PlanEdit(EditRequest{
		Kind: EditWasmEvent, WasmFolder: "jim",
		EventSourceKind: "button", EventSourceName: "save", EventName: "click",
		EventAction: "setwords", EventTargetKind: "price tag", EventTargetName: "total", EventWords: "Cheap",
	}); !strings.Contains(refusal, "only a button's or a card's words can be set") {
		t.Errorf("the set-words refusal = %q", refusal)
	}
}
