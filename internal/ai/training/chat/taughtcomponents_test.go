package chat

import (
	"path/filepath"
	"strings"
	"testing"
)

// Taught components: the teach seat's second cut of
// example — a DOM builder taught from his own project,
// added to a wasm app like a registry kind, wired into
// events, listed as his, and forgotten by the seat's own
// shapes. The fixture's example builder lives in
// proto/components.go under the js && wasm tag (excluded
// from the host build, parsed everywhere), with its own
// el helper standing in for an app's.

// taughtBuilderFile is the fixture's example: a builder
// in the registry's shape — el-made root, the id stamp
// under the line that creates it, a class on the root.
const taughtBuilderFile = `//go:build js && wasm

package proto

import "syscall/js"

func el(doc js.Value, tag, class string, texts ...string) js.Value {
	e := doc.Call("createElement", tag)
	if class != "" {
		e.Set("className", class)
	}
	for _, text := range texts {
		e.Call("appendChild", doc.Call("createTextNode", text))
	}
	return e
}

// PriceTag is a small price display.
func PriceTag(doc js.Value) js.Value {
	tag := el(doc, "div", "price-tag", "")
	tag.Set("id", "pt-1")
	price := el(doc, "span", "", "$0")
	tag.Call("appendChild", price)
	return tag
}
`

const taughtBuilderShortElFile = `//go:build js && wasm

package proto

import "syscall/js"

func ShortEl(doc js.Value) js.Value {
	tag := el(doc, "div", "price-tag")
	tag.Set("id", "pt-1")
	return tag
}
`

const taughtBuilderNoStampFile = `//go:build js && wasm

package proto

import "syscall/js"

func NoStamp(doc js.Value) js.Value {
	tag := el(doc, "div", "price-tag", "")
	price := el(doc, "span", "", "$0")
	tag.Call("appendChild", price)
	return tag
}
`

const taughtBuilderBadShapeFile = `//go:build js && wasm

package proto

import "syscall/js"

func BadShape(doc js.Value) {
	tag := el(doc, "div", "price-tag", "")
	tag.Set("id", "pt-1")
	_ = tag
}
`

const taughtBuilderVarStampFile = `//go:build js && wasm

package proto

import "syscall/js"

func VarStamp(doc js.Value) js.Value {
	tag := el(doc, "div", "price-tag", "")
	myID := "pt-1"
	tag.Set("id", myID)
	return tag
}
`

const taughtBuilderRawRootFile = `//go:build js && wasm

package proto

import "syscall/js"

func RawRoot(doc js.Value) js.Value {
	tag := doc.Call("createElement", "div")
	tag.Set("id", "pt-1")
	return tag
}
`

// taughtComponentFixture builds the static-assets fixture
// project with the proto builders in it and points the
// taught store at a throwaway file outside the project.
func taughtComponentFixture(t *testing.T) (root, storePath string) {
	t.Helper()
	root = staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"proto/components.go": taughtBuilderFile,
		"proto/nostamp.go":    taughtBuilderNoStampFile,
		"proto/badshape.go":   taughtBuilderBadShapeFile,
		"proto/varstamp.go":   taughtBuilderVarStampFile,
		"proto/rawroot.go":    taughtBuilderRawRootFile,
		"proto/shortel.go":    taughtBuilderShortElFile,
	}))
	storePath = filepath.Join(t.TempDir(), ".gollemer_taught.json")
	old := taughtStorePathOverride
	taughtStorePathOverride = storePath
	t.Cleanup(func() { taughtStorePathOverride = old })
	return root, storePath
}

const teachPriceTagLine = "learn this component as a price tag from PriceTag in proto/components.go"

func taughtComponentsInStore(t *testing.T) []taughtComponent {
	t.Helper()
	d, err := loadTaughtKinds()
	if err != nil {
		t.Fatalf("store load: %v", err)
	}
	return d.Components
}

// --- grammar -----------------------------------------------------------------

func TestParseTeachComponentAsk(t *testing.T) {
	cases := []struct {
		line, kind, name, file string
	}{
		{"learn this component as a price tag from PriceTag in proto/components.go", "price tag", "PriceTag", "proto/components.go"},
		{"learn a component as a banner from Banner", "banner", "Banner", ""},
		{"teach a component as a price tag from PriceTag in proto/components.go", "price tag", "PriceTag", "proto/components.go"},
		{"teach you a component as a sale banner from SaleBanner in shop/ui.go", "sale banner", "SaleBanner", "shop/ui.go"},
		{"teach this component as a price tag from PriceTag", "price tag", "PriceTag", ""},
		{"learn an component as a price tag from PriceTag", "price tag", "PriceTag", ""},
	}
	for _, c := range cases {
		ta, ok := parseTeachComponentAsk(c.line)
		if !ok {
			t.Errorf("%q: not claimed", c.line)
			continue
		}
		if normalizeKindPhrase(ta.kind) != c.kind || ta.name != c.name || ta.file != c.file {
			t.Errorf("%q: got {kind:%q name:%q file:%q}, want {%q %q %q}", c.line, ta.kind, ta.name, ta.file, c.kind, c.name, c.file)
		}
	}
	// The neighbours stay the declaration seat's (or no
	// one's): the bare teach shapes, kind teaches without
	// the component word, adds, questions.
	for _, line := range []string{
		"learn this as a price tag from PriceTag in proto/components.go",
		"learn a sidebar page from Jake in jim/main.go",
		"teach you a repository from Store in store/store.go",
		"learn a component from PriceTag in proto/components.go",
		"add a price tag to jim",
		"what components can i add",
	} {
		if ta, ok := parseTeachComponentAsk(line); ok {
			t.Errorf("parseTeachComponentAsk(%q) claimed %+v — not its sentence", line, ta)
		}
	}
	// The declaration grammar reads the component shapes
	// as a kind named "component as a …" — which is why
	// the component parse is seated first in runTaughtKinds.
	if _, ok := parseTeachAsk("teach a component as a price tag from PriceTag in proto/components.go"); !ok {
		t.Error("the declaration parse was expected to (mis)read the component shape — the seat order note is stale")
	}
}

func TestParseTaughtComponentAdds(t *testing.T) {
	rows := []struct {
		line, kind, called, target string
	}{
		{"add a price tag to jim", "price tag", "", "jim"},
		{"add a price tag called total to jim", "price tag", "total", "jim"},
		{"add a price tag to jim/main.go", "price tag", "", "jim/main.go"},
		{"Add a big red sale banner thing to shop/web", "big red sale banner thing", "", "shop/web"},
		{"add the price tag into jim", "price tag", "", "jim"},
	}
	for _, tc := range rows {
		got, ok := parseTaughtComponentWasm(tc.line)
		if !ok {
			t.Errorf("parseTaughtComponentWasm(%q) declined", tc.line)
			continue
		}
		if got.kind != tc.kind || got.called != tc.called || got.target != tc.target {
			t.Errorf("parseTaughtComponentWasm(%q) = %+v, want kind %q called %q target %q", tc.line, got, tc.kind, tc.called, tc.target)
		}
	}
	for _, line := range []string{
		"add a price tag",                         // no target
		"add a price tag to jim/other.go",         // another .go target: the book keeps it
		"add a price tag to jim/temp/index.html",  // a page target: the standard parse's
		"add a price tag called to jim",           // no name after "called"
		"add a price tag called total due to jim", // words after the name
		"add a price tag to the header in jim",    // the part shape, not this one
	} {
		if got, ok := parseTaughtComponentWasm(line); ok {
			t.Errorf("parseTaughtComponentWasm(%q) claimed %+v — want a decline", line, got)
		}
	}
	part, ok := parseTaughtComponentPart("add a price tag to the header in jim")
	if !ok || part.kind != "price tag" || part.part != "header" || part.folder != "jim" {
		t.Errorf("part ask = %+v, %v", part, ok)
	}
	part, ok = parseTaughtComponentPart("add a price tag called total to the footer of the wasm app in jim")
	if !ok || part.kind != "price tag" || part.called != "total" || part.part != "footer" || part.folder != "jim" {
		t.Errorf("part ask = %+v, %v", part, ok)
	}
	if got, ok := parseTaughtComponentPart("add a price tag to jim"); ok {
		t.Errorf("the app shape claimed as a part ask: %+v", got)
	}
}

// --- the full arc --------------------------------------------------------------

func TestTeachComponentFullArc(t *testing.T) {
	root, _ := taughtComponentFixture(t)
	profile := &UserProfile{}

	out := captureStdout(t, func() {
		if !claimChain(teachPriceTagLine, yes, profile) {
			t.Error("the teach sentence was not claimed")
		}
	})
	for _, want := range []string{
		"Here's what I learned from PriceTag in proto/components.go:",
		"The kind: \"price tag\"",
		"\"add a price tag called <name> to <app folder>\" adds a named one",
		"no styles travel with a taught component in this first version",
		"Saved as yours",
		"\"forget the price tag\" to remove it",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the teach summary did not say %q:\n%s", want, out)
		}
	}
	comps := taughtComponentsInStore(t)
	if len(comps) != 1 {
		t.Fatalf("store components = %d, want 1", len(comps))
	}
	entry := comps[0]
	if entry.Kind != "price tag" || entry.Example != "PriceTag" || entry.RootVar != "tag" || !entry.Styled || entry.File != "proto/components.go" {
		t.Errorf("stored component = %+v", entry)
	}
	for _, want := range []string{"func {{name}}(doc js.Value) js.Value {", "tag.Set(\"id\", \"{{id}}\")"} {
		if !strings.Contains(entry.Template, want) {
			t.Errorf("the stored template is missing %q:\n%s", want, entry.Template)
		}
	}
	if strings.Contains(entry.Template, "PriceTag") {
		t.Errorf("the stored template still names the example:\n%s", entry.Template)
	}

	for _, line := range []string{
		"create a folder jim",
		"add a wasm app in jim",
		"add a button called save to jim",
		"add a card to jim",
	} {
		if !claimChain(line, yes, profile) {
			t.Fatalf("%q was not claimed", line)
		}
	}

	// Add it, unnamed: the registry landing's shape, with
	// the marker and the positional id.
	out = captureStdout(t, func() {
		if !claimChain("add a price tag to jim", yes, profile) {
			t.Error("the add sentence was not claimed")
		}
	})
	for _, want := range []string{
		"Done — The price tag is in the wasm app in jim",
		"I built the project and checked that it's really there.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the add landing did not say %q:\n%s", want, out)
		}
	}
	comp := readBack(t, root, "jim/components.go")
	for _, want := range []string{
		"// gollemer:taught-component kind=\"price tag\" root=tag",
		"func priceTagComponent(doc js.Value) js.Value {",
		"tag.Set(\"id\", \"price-tag-1\")",
	} {
		if !strings.Contains(comp, want) {
			t.Errorf("jim/components.go is missing %q:\n%s", want, comp)
		}
	}
	if main := readBack(t, root, "jim/main.go"); !strings.Contains(main, "app.Call(\"appendChild\", priceTagComponent(doc))") {
		t.Errorf("jim/main.go does not mount the taught builder:\n%s", main)
	}

	// Again, unnamed: already there, nothing rewritten.
	before := readBack(t, root, "jim/components.go")
	out = captureStdout(t, func() { claimChain("add a price tag to jim", yes, profile) })
	if !strings.Contains(out, "already has a price tag in it — nothing to add") {
		t.Errorf("the repeat was not the already-there answer:\n%s", out)
	}
	if after := readBack(t, root, "jim/components.go"); after != before {
		t.Error("the repeat rewrote components.go")
	}

	// As an event SOURCE while it is the only one of its
	// kind: the bare kind names it.
	out = captureStdout(t, func() {
		if !claimChain("when the price tag is clicked, show the card in jim", yes, profile) {
			t.Error("the taught-source event was not claimed")
		}
	})
	if !strings.Contains(out, "Done — The event is in the wasm app in jim: when the price tag is clicked, show the card.") {
		t.Errorf("the taught-source landing is wrong:\n%s", out)
	}

	// Named: its own builder, the name as its id.
	out = captureStdout(t, func() {
		if !claimChain("add a price tag called total to jim", yes, profile) {
			t.Error("the named add was not claimed")
		}
	})
	if !strings.Contains(out, "Done — The price tag called total is in the wasm app in jim") {
		t.Errorf("the named landing is wrong:\n%s", out)
	}
	comp = readBack(t, root, "jim/components.go")
	if !strings.Contains(comp, "func totalPriceTagComponent(doc js.Value) js.Value {") || !strings.Contains(comp, "tag.Set(\"id\", \"total\")") {
		t.Errorf("the named builder did not land with its name as its id:\n%s", comp)
	}

	// As an event TARGET, by its name in front of its kind.
	out = captureStdout(t, func() {
		if !claimChain("when the save button is clicked, show the total price tag in jim", yes, profile) {
			t.Error("the taught-target event was not claimed")
		}
	})
	if !strings.Contains(out, "Done — The event is in the wasm app in jim: when the save button is clicked, show the total price tag.") {
		t.Errorf("the taught-target landing is wrong:\n%s", out)
	}

	// Its words are its own: the set-words refusal.
	out = captureStdout(t, func() {
		if !claimChain("when the save button is clicked, set the total price tag's words to 'Cheap' in jim", yes, profile) {
			t.Error("the set-words sentence was not claimed")
		}
	})
	if !strings.Contains(out, "only a button's or a card's words can be set") {
		t.Errorf("the set-words refusal is wrong:\n%s", out)
	}
	// Changing is not its event.
	out = captureStdout(t, func() {
		if !claimChain("when the total price tag changes, show the card in jim", yes, profile) {
			t.Error("the change sentence was not claimed")
		}
	})
	if !strings.Contains(out, "a price tag doesn't change in this set") {
		t.Errorf("the change refusal is wrong:\n%s", out)
	}

	// Discovery lists it as his.
	out = captureStdout(t, func() {
		if !claimChain("what components can i add", yes, profile) {
			t.Error("the discovery question was not claimed")
		}
	})
	for _, want := range []string{"Yours, taught in this project:", "your price tag (taught)"} {
		if !strings.Contains(out, want) {
			t.Errorf("the discovery listing is missing %q:\n%s", want, out)
		}
	}

	// Forget it, and the add is the honest not-known answer.
	out = captureStdout(t, func() {
		if !claimChain("forget the price tag", yes, profile) {
			t.Error("the forget sentence was not claimed")
		}
	})
	if !strings.Contains(out, "Forgotten: your \"price tag\" component") {
		t.Errorf("the forget answer is wrong:\n%s", out)
	}
	if comps := taughtComponentsInStore(t); len(comps) != 0 {
		t.Errorf("store components after forget = %d, want 0", len(comps))
	}
	out = captureStdout(t, func() { claimChain("add a price tag to jim", yes, profile) })
	if !strings.Contains(out, "There's no component called \"price tag\" in the set I can add.") {
		t.Errorf("the after-forget add was not the honest not-known answer:\n%s", out)
	}
}

// --- refusals ------------------------------------------------------------------

func TestTeachComponentRefusals(t *testing.T) {
	taughtComponentFixture(t)
	profile := &UserProfile{}
	cases := []struct {
		line string
		want string
	}{
		{"learn this component as a banner from NoStamp in proto/nostamp.go",
			"never sets its root element's id the registry way"},
		{"learn this component as a banner from BadShape in proto/badshape.go",
			"doesn't have a component builder's shape"},
		{"learn this component as a banner from VarStamp in proto/varstamp.go",
			"sets its root's id, but not to a plain string"},
		{"learn this component as a banner from RawRoot in proto/rawroot.go",
			"never builds a root element with the app's el helper"},
		{"learn this component as a banner from ShortEl in proto/shortel.go",
			"calls the app's el helper with 3 arguments"},
		{"learn this component as a card from PriceTag in proto/components.go",
			"already a built-in component"},
		{"learn this component as a banner from Nope in proto/components.go",
			"I couldn't find a function named \"Nope\""},
		{"learn this component as a component part from PriceTag in proto/components.go",
			"can't contain the word"},
	}
	for _, c := range cases {
		out := captureStdout(t, func() {
			if !claimChain(c.line, yes, profile) {
				t.Errorf("%q was not claimed", c.line)
			}
		})
		if !strings.Contains(out, "Not learning that:") || !strings.Contains(out, c.want) {
			t.Errorf("%q: the refusal did not say %q:\n%s", c.line, c.want, out)
		}
	}
	if comps := taughtComponentsInStore(t); len(comps) != 0 {
		t.Errorf("refusals saved components: %+v", comps)
	}

	// Teach it for real, then again: the seat's
	// forget-first guidance.
	captureStdout(t, func() { claimChain(teachPriceTagLine, yes, profile) })
	out := captureStdout(t, func() { claimChain(teachPriceTagLine, yes, profile) })
	for _, want := range []string{"you've already taught me a \"price tag\" component in this project", "forget the price tag"} {
		if !strings.Contains(out, want) {
			t.Errorf("the re-teach refusal did not say %q:\n%s", want, out)
		}
	}
	if comps := taughtComponentsInStore(t); len(comps) != 1 {
		t.Errorf("store components = %d, want the original 1", len(comps))
	}
}

// One phrase, one seat: a component can't take a taught
// piece's name, and a piece can't take a component's.
func TestTeachComponentOnePhraseOneSeat(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"proto/components.go": taughtBuilderFile,
		"jim/main.go":         staticJimMain,
	}))
	storePath := filepath.Join(t.TempDir(), ".gollemer_taught.json")
	old := taughtStorePathOverride
	taughtStorePathOverride = storePath
	t.Cleanup(func() { taughtStorePathOverride = old })
	_ = root
	profile := &UserProfile{}

	captureStdout(t, func() { claimChain(teachPriceTagLine, yes, profile) })
	out := captureStdout(t, func() { claimChain("learn a price tag from Jake in jim/main.go", yes, profile) })
	if !strings.Contains(out, "you've already taught me a component called \"price tag\"") {
		t.Errorf("the piece-over-component refusal is wrong:\n%s", out)
	}
	// The decl seat's use guidance for the component's name
	// points at the app, not at another teach.
	out = captureStdout(t, func() { claimChain("add a price tag named Thing to jim/main.go", yes, profile) })
	if !strings.Contains(out, "is a component you taught me, not a piece") {
		t.Errorf("the use-guidance answer is wrong:\n%s", out)
	}
}

// The part landing through the chat: the header of the
// app, its own components file.
func TestTeachComponentPartFlow(t *testing.T) {
	root, _ := taughtComponentFixture(t)
	profile := &UserProfile{}
	captureStdout(t, func() { claimChain(teachPriceTagLine, yes, profile) })
	for _, line := range []string{
		"create a folder jim",
		"add a wasm app in jim",
		"add a header to the wasm app in jim",
	} {
		if !claimChain(line, yes, profile) {
			t.Fatalf("%q was not claimed", line)
		}
	}
	out := captureStdout(t, func() {
		if !claimChain("add a price tag to the header in jim", yes, profile) {
			t.Error("the part add was not claimed")
		}
	})
	if !strings.Contains(out, "Done — The price tag is in the header of the wasm app in jim") {
		t.Errorf("the part landing is wrong:\n%s", out)
	}
	comp := readBack(t, root, "jim/header/components.go")
	if !strings.Contains(comp, "func priceTagComponent(doc js.Value) js.Value {") ||
		!strings.Contains(comp, "// gollemer:taught-component kind=\"price tag\" root=tag") {
		t.Errorf("the header's components.go does not carry the taught builder:\n%s", comp)
	}
}

// Discovery without any taught components is the registry
// listing, unchanged — no "yours" group.
func TestTeachComponentDiscoveryRegistryOnly(t *testing.T) {
	taughtComponentFixture(t)
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("what components can i add", yes, profile) })
	if strings.Contains(out, "(taught)") || strings.Contains(out, "Yours, taught") {
		t.Errorf("the registry-only listing grew a taught group:\n%s", out)
	}
	if !strings.Contains(out, "The components you can add to a page") {
		t.Errorf("the registry listing is missing:\n%s", out)
	}
}

// The manage surfaces treat components as the seat's own:
// the listing names them in their own group, and the
// forget-all shape clears them with the kinds.
func TestTeachComponentManageSurfaces(t *testing.T) {
	taughtComponentFixture(t)
	profile := &UserProfile{}
	captureStdout(t, func() { claimChain(teachPriceTagLine, yes, profile) })
	out := captureStdout(t, func() { claimChain("what have i taught you", yes, profile) })
	for _, want := range []string{"What you've taught me:", "\"price tag\" (component)", "from your example PriceTag in proto/components.go"} {
		if !strings.Contains(out, want) {
			t.Errorf("the taught listing is missing %q:\n%s", want, out)
		}
	}
	out = captureStdout(t, func() { claimChain("forget everything i taught you", yes, profile) })
	if !strings.Contains(out, "\"price tag\" (component, from PriceTag in proto/components.go)") {
		t.Errorf("the forget-all answer is wrong:\n%s", out)
	}
	if comps := taughtComponentsInStore(t); len(comps) != 0 {
		t.Errorf("store components after forget-all = %d, want 0", len(comps))
	}
	out = captureStdout(t, func() { claimChain("forget the price tag", yes, profile) })
	if !strings.Contains(out, "you haven't taught me anything yet") {
		t.Errorf("the forget-after-empty answer is wrong:\n%s", out)
	}
}

// A component taught in ANOTHER project does not answer
// here: the store keeps kinds per project, and this
// project's discovery and adds see none of it.
func TestTeachComponentProjectScoped(t *testing.T) {
	_, storePath := taughtComponentFixture(t)
	profile := &UserProfile{}
	captureStdout(t, func() { claimChain(teachPriceTagLine, yes, profile) })
	_ = storePath

	// A second project, same store: no taught group in
	// discovery, and the add falls to the not-known answer.
	other := staticAssetsFixture(t, staticAssetsFiles(nil))
	_ = other
	out := captureStdout(t, func() { claimChain("what components can i add", yes, profile) })
	if strings.Contains(out, "(taught)") {
		t.Errorf("another project's component listed here:\n%s", out)
	}
	claimChain("create a folder jim", yes, profile)
	claimChain("add a wasm app in jim", yes, profile)
	out = captureStdout(t, func() { claimChain("add a price tag to jim", yes, profile) })
	if !strings.Contains(out, "There's no component called \"price tag\" in the set I can add.") {
		t.Errorf("the cross-project add was not the honest answer:\n%s", out)
	}
	// Forgetting by name still reaches it from here — one
	// phrase is one seat, wherever it was taught.
	out = captureStdout(t, func() { claimChain("forget the price tag", yes, profile) })
	if !strings.Contains(out, "Forgotten: your \"price tag\" component") {
		t.Errorf("the cross-project forget is wrong:\n%s", out)
	}
}
