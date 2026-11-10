package chat

import (
	"strings"
	"testing"
)

// The event sentence's accepted shapes, pinned at the
// parse: every one of these lines is claimed by
// parseWasmEvent and renders back (through the planner's
// own sentence renderer) as the canonical line shown.
func TestParseWasmEventClaims(t *testing.T) {
	rows := []struct {
		line string
		want string // the canonical rendering
	}{
		{"when the button is clicked, show the card in jim", "when the button is clicked, show the card in jim"},
		{"when the save button is clicked, set the card's words to 'Saved' in jim/web", "when the save button is clicked, set the card's words to 'Saved' in jim/web"},
		{"when save is clicked, hide the card", "when the save is clicked, hide the card"}, // a name alone renders as itself until the app resolves its kind
		{"when the switch is flipped, toggle the card in jim", "when the switch is flipped, toggle the card in jim"},
		{"when the text field changes, show the footer in jim", "when the text field changes, show the footer in jim"},
		{"When the button is pressed, show the card.", "when the button is clicked, show the card"},
		{"when the text field is changed, hide the card in jim/main.go", "when the text field changes, hide the card in jim/main.go"},
		{"when the card is clicked, toggle the card in jim", "when the card is clicked, toggle the card in jim"},
		{`when the button is clicked, set the card's words to "Saved" in jim`, "when the button is clicked, set the card's words to 'Saved' in jim"},
		{"when the button is clicked, set the card's words to Saved", "when the button is clicked, set the card's words to 'Saved'"},
		{"when the navigation bar is clicked, show the card", "when the navigation bar is clicked, show the card"},
		{"when the tonal button is clicked, hide the outlined button", "when the tonal button is clicked, hide the outlined button"},
	}
	for _, tc := range rows {
		q, ok := parseWasmEvent(tc.line)
		if !ok {
			t.Errorf("parseWasmEvent(%q) declined — want a claim", tc.line)
			continue
		}
		if got := q.canonical(); got != tc.want {
			t.Errorf("parseWasmEvent(%q).canonical() = %q, want %q", tc.line, got, tc.want)
		}
		if q.source.unknown != "" || q.target.unknown != "" {
			t.Errorf("parseWasmEvent(%q) left an unknown reference: %+v", tc.line, q)
		}
	}
	// The fields underneath two of the shapes.
	q, ok := parseWasmEvent("when the save button is clicked, set the card's words to 'Saved' in jim/web")
	if !ok || q.source.kind != "button" || q.source.name != "save" || q.event != "click" ||
		q.action != "setwords" || q.words != "Saved" || q.target.kind != "card" || !q.hasFolder || q.folder != "jim/web" {
		t.Errorf("set-words ask = %+v, %v", q, ok)
	}
	// A reference the registry doesn't know is carried as
	// unknown — the flow names it, the parse doesn't guess.
	q, ok = parseWasmEvent("when the magic box is clicked, show the card in jim")
	if !ok || q.source.unknown != "magic box" {
		t.Errorf("magic-box ask = %+v, %v — want unknown 'magic box'", q, ok)
	}
}

// The event grammar's declines: neighbours it must never
// take — component adds, the app sentence, commands,
// questions, and near-miss event shapes.
func TestParseWasmEventDeclines(t *testing.T) {
	for _, line := range []string{
		"add a card to jim",
		"add a wasm app in jim",
		"add a button called save to jim",
		"when is dinner ready",
		"run jim/main.go",
		"what components can i add",
		"when the button is clicked show the card",                   // no comma
		"when the button is clicked, dance the card",                 // an action outside the set
		"when the button is clicked, set the card's text to 'Saved'", // text, not words
		"when the button is clicked,",                                // no action at all
		"when the button explodes, show the card",                    // an event outside the set
		"show the card when the button is clicked",                   // the shape opens with "when"
	} {
		if got, ok := parseWasmEvent(line); ok {
			t.Errorf("parseWasmEvent(%q) claimed %+v — not its sentence", line, got)
		}
	}
}

// The naming prerequisite at the parse: "called <name>"
// joins the wasm component sentence — and malformed uses
// of it decline the line instead of guessing, exactly the
// lines the plain grammar declined before names existed.
func TestParseComponentWasmTargetCalled(t *testing.T) {
	rows := []struct {
		line              string
		component, called string
		target            string
	}{
		{"add a button called save to jim", "button", "save", "jim"},
		{"add a text field called query to jim/main.go", "text-field", "query", "jim/main.go"},
		{"Add a card called welcome to jim/web", "card", "welcome", "jim/web"},
		{"add a card to jim", "card", "", "jim"}, // the unnamed shape, unchanged
	}
	for _, tc := range rows {
		got, ok := parseComponentWasmTarget(tc.line)
		if !ok {
			t.Errorf("parseComponentWasmTarget(%q) declined", tc.line)
			continue
		}
		if got.component != tc.component || got.called != tc.called || got.target != tc.target {
			t.Errorf("parseComponentWasmTarget(%q) = %+v, want component %q called %q target %q", tc.line, got, tc.component, tc.called, tc.target)
		}
	}
	for _, line := range []string{
		"add a button called to jim",                      // no name after "called"
		"add a button called save now to jim",             // two words after "called"
		"add a called button to jim",                      // "called" is not a component
		"add a button called save to jim/temp/index.html", // pages take no names
		"add a button called the card to jim",             // a grammar word is not a name
	} {
		if got, ok := parseComponentWasmTarget(line); ok {
			t.Errorf("parseComponentWasmTarget(%q) claimed %+v — want a decline", line, got)
		}
	}
	// The page grammar is untouched by names: the same
	// sentence aimed at a page parses exactly as before —
	// the component word read as the name, honestly.
	if got, ok := parseComponent("add a button called save to jim/temp/index.html"); !ok || got.component != "" {
		t.Errorf("parseComponent on a page with 'called' = %+v, %v — want the standing unknown-component read", got, ok)
	}
}

// John's flow, end to end through the claim chain: name a
// button, wire the sentence, repeat it (already there),
// ask without a folder (the session's app answers), and
// leave the wire ambiguous (the clarifying question asks,
// and the one-word answer completes it).
func TestWasmEventJohnsFlow(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(nil))
	profile := &UserProfile{}
	for _, line := range []string{
		"create a folder jim",
		"add a wasm app in jim",
		"add a button called save to jim",
		"add a card to jim",
		"add a switch to jim",
	} {
		if !claimChain(line, yes, profile) {
			t.Fatalf("%q was not claimed", line)
		}
	}
	const setWords = "when the save button is clicked, set the card's words to 'Saved' in jim"
	out := captureStdout(t, func() {
		if !claimChain(setWords, yes, profile) {
			t.Error("the set-words sentence was not claimed")
		}
	})
	for _, want := range []string{
		"Done — The event is in the wasm app in jim: when the save button is clicked, set the card's words to 'Saved'.",
		"I built the project and checked that it's really there.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the landing did not say %q:\n%s", want, out)
		}
	}
	events := readBack(t, root, "jim/events.go")
	for _, want := range []string{
		"func onSaveClickSetCard1Words(doc js.Value) {",
		"eventFuncs = append(eventFuncs, fn)",
		"func registerEvents(doc js.Value) {",
		"onSaveClickSetCard1Words(doc)",
	} {
		if !strings.Contains(events, want) {
			t.Errorf("events.go is missing %q:\n%s", want, events)
		}
	}
	if main := readBack(t, root, "jim/main.go"); !strings.Contains(main, "mountComponents(doc, app)\n\tregisterEvents(doc)") {
		t.Errorf("main.go does not register the events after the mounts:\n%s", main)
	}

	out = captureStdout(t, func() {
		if !claimChain("when the switch is flipped, toggle the card in jim", yes, profile) {
			t.Error("the switch sentence was not claimed")
		}
	})
	if !strings.Contains(out, "Done — The event is in the wasm app in jim: when the switch is flipped, toggle the card.") {
		t.Errorf("the switch landing's confirmation is wrong:\n%s", out)
	}

	// The same event again: already there, byte-identical file.
	before := readBack(t, root, "jim/events.go")
	out = captureStdout(t, func() { claimChain(setWords, yes, profile) })
	if !strings.Contains(out, "already does that") {
		t.Errorf("the repeat was not the already-there answer:\n%s", out)
	}
	if after := readBack(t, root, "jim/events.go"); after != before {
		t.Error("the repeat rewrote events.go")
	}

	// No folder: the session's app answers, its reading printed.
	out = captureStdout(t, func() {
		if !claimChain("when the save button is clicked, hide the card", yes, profile) {
			t.Error("the folder-less sentence was not claimed")
		}
	})
	for _, want := range []string{
		"Reading the app as the wasm app in jim — the last wasm app this session touched.",
		"Done — The event is in the wasm app in jim: when the save button is clicked, hide the card.",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("the folder-less landing did not say %q:\n%s", want, out)
		}
	}

	// Two buttons and no name: the clarifying question.
	if !claimChain("add a button called cancel to jim", yes, profile) {
		t.Fatal("the second named button was not claimed")
	}
	out = captureStdout(t, func() {
		if !claimChain("when the button is clicked, show the card in jim", yes, profile) {
			t.Error("the ambiguous sentence was not claimed")
		}
	})
	if !strings.Contains(out, "more than one button") || !hasPendingQuestion() {
		t.Errorf("the ambiguity did not become the clarifying question:\n%s", out)
	}
	out = captureStdout(t, func() {
		if !claimChain("cancel", yes, profile) {
			t.Error("the clarifying answer was not claimed")
		}
	})
	if !strings.Contains(out, "Done — The event is in the wasm app in jim: when the cancel button is clicked, show the card.") {
		t.Errorf("the clarified landing's confirmation is wrong:\n%s", out)
	}
	if events := readBack(t, root, "jim/events.go"); !strings.Contains(events, "func onCancelClickShowCard1(doc js.Value)") {
		t.Errorf("the clarified handler is missing:\n%s", events)
	}
}

// The events piece's honest edges: no folder anywhere, a
// folder of plain pages (script-free, by design), a
// missing folder, a folder with no app yet, a component
// the set doesn't know, and a component the app doesn't
// have.
func TestWasmEventBoundaries(t *testing.T) {
	profile := &UserProfile{}

	staticAssetsFixture(t, staticAssetsFiles(nil))
	out := captureStdout(t, func() { claimChain("when the button is clicked, show the card", yes, profile) })
	if !strings.Contains(out, "doesn't name the app's folder") {
		t.Errorf("the folder-less, session-less answer:\n%s", out)
	}

	staticAssetsFixture(t, staticAssetsFiles(map[string]string{"jim/index.html": "<!doctype html>\n<html><body>hi</body></html>\n"}))
	out = captureStdout(t, func() { claimChain("when the button is clicked, show the card in jim", yes, profile) })
	if !strings.Contains(out, "plain pages stay script-free") {
		t.Errorf("the pages answer:\n%s", out)
	}

	staticAssetsFixture(t, staticAssetsFiles(nil))
	out = captureStdout(t, func() { claimChain("when the button is clicked, show the card in ghost", yes, profile) })
	if !strings.Contains(out, "there's no folder ghost") || !strings.Contains(out, "create a folder ghost") {
		t.Errorf("the missing-folder answer:\n%s", out)
	}

	staticAssetsFixture(t, staticAssetsFiles(map[string]string{"jim/.keep": ""}))
	out = captureStdout(t, func() { claimChain("when the button is clicked, show the card in jim", yes, profile) })
	if !strings.Contains(out, "there's no wasm app in jim yet") || !strings.Contains(out, "add a wasm app in jim") {
		t.Errorf("the no-app-yet answer:\n%s", out)
	}
	out = captureStdout(t, func() { claimChain("when the magic box is clicked, show the card in jim", yes, profile) })
	if !strings.Contains(out, `There's no component called "magic box"`) {
		t.Errorf("the unknown-component answer:\n%s", out)
	}

	staticAssetsFixture(t, staticAssetsFiles(nil))
	claimChain("create a folder jim", yes, profile)
	claimChain("add a wasm app in jim", yes, profile)
	out = captureStdout(t, func() { claimChain("when the button is clicked, show the card in jim", yes, profile) })
	if !strings.Contains(out, "Not doing that edit:") || !strings.Contains(out, "has no button in it yet") {
		t.Errorf("the app-without-the-component answer:\n%s", out)
	}
}
