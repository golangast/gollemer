package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The component grammar, at parse: every registry form
// claims with its component and target — typed file or
// "the page" — including the material adjective and the
// alias forms ("top bar", "fab"). An unknown name still
// matches the shape, with no component: the flow answers
// with the list.
func TestParseComponentClaims(t *testing.T) {
	rows := []struct {
		line      string
		component string
		file      string
		thePage   bool
	}{
		{"add a button to jim/temp/index.html", "button", "jim/temp/index.html", false},
		{"add a material button to jim/temp/index.html", "button", "jim/temp/index.html", false},
		{"add a tonal button to jim/temp/index.html", "tonal-button", "jim/temp/index.html", false},
		{"add an outlined button to jim/temp/index.html", "outlined-button", "jim/temp/index.html", false},
		{"add a text button to jim/temp/index.html", "text-button", "jim/temp/index.html", false},
		{"add a card to jim/temp/index.html", "card", "jim/temp/index.html", false},
		{"add an app bar to jim/temp/index.html", "app-bar", "jim/temp/index.html", false},
		{"add a top bar to jim/temp/index.html", "app-bar", "jim/temp/index.html", false},
		{"add a material app bar to jim/temp/index.html", "app-bar", "jim/temp/index.html", false},
		{"add a floating button to jim/temp/index.html", "fab", "jim/temp/index.html", false},
		{"add a fab to jim/temp/index.html", "fab", "jim/temp/index.html", false},
		{"add a chip to jim/temp/index.html", "chip", "jim/temp/index.html", false},
		{"add a text field to jim/temp/index.html", "text-field", "jim/temp/index.html", false},
		{"add a switch to jim/temp/index.html", "switch", "jim/temp/index.html", false},
		{"add a card to the page", "card", "", true},
		{"add a switch to the page", "switch", "", true},
		{"add a button to the page", "button", "", true},
		// The dropped article (John live, 2026-11-09: "add
		// card to jim/temp/index.html" was declined to the
		// can't-assemble answer while "add a card…" landed):
		// the same sentences with no a/an/the claim with the
		// same slots.
		{"add card to jim/temp/index.html", "card", "jim/temp/index.html", false},
		{"add button to the page", "button", "", true},
		{"add switch to the page", "switch", "", true},
		{"add text field to jim/temp/index.html", "text-field", "jim/temp/index.html", false},
		{"add app bar to jim/temp/index.html", "app-bar", "jim/temp/index.html", false},
		{"add material button to jim/temp/index.html", "button", "jim/temp/index.html", false},
		{"add tonal button to jim/temp/index.html", "tonal-button", "jim/temp/index.html", false},
		// An unknown name matches the shape with no
		// component — the flow answers with what exists.
		{"add a carousel to jim/temp/index.html", "", "jim/temp/index.html", false},
		{"add carousel to jim/temp/index.html", "", "jim/temp/index.html", false},
	}
	for _, tc := range rows {
		got, ok := parseComponent(tc.line)
		if !ok {
			t.Errorf("%q: no claim", tc.line)
			continue
		}
		if got.component != tc.component || got.file != tc.file || got.thePage != tc.thePage {
			t.Errorf("%q: parsed %+v, want component=%q file=%q thePage=%v", tc.line, got, tc.component, tc.file, tc.thePage)
		}
	}
}

// The shapes the component piece must leave alone — above
// all the .go target, which keeps the concept book's
// meaning exactly as before.
func TestParseComponentDeclines(t *testing.T) {
	for _, line := range []string{
		"add a button to main.go",
		"add card to main.go", // the dropped article opens no .go door either
		"add a card to notes.txt",
		"add a button",
		"add card",
		"add a server in jim",
		"what components can i add",
		"what components are there",
		"serve static files",
		"add a string to jim/main.go",
	} {
		if got, ok := parseComponent(line); ok {
			t.Errorf("%q claimed as a component ask (%+v)", line, got)
		}
	}
}

// The discovery answer is the registry: every component's
// words, its alias notes, and one example sentence each —
// and both of John's shapes claim it.
func TestComponentDiscoveryAnswer(t *testing.T) {
	text := componentListText()
	for _, c := range analyze.MaterialComponents {
		if !strings.Contains(text, c.Words) {
			t.Errorf("discovery answer is missing %q", c.Words)
		}
		if !strings.Contains(text, "\"add a "+c.Words+" to jim/temp/index.html\"") {
			t.Errorf("discovery answer has no example sentence for %q", c.Words)
		}
		if c.Note != "" && !strings.Contains(text, c.Note) {
			t.Errorf("discovery answer lost %s's note %q", c.Words, c.Note)
		}
	}
	for _, line := range []string{
		"what components can I add", "what components are there",
		// The singular (John live, 2026-11-09: the symbol
		// explainer answered it by looking "add" up as a
		// function name) and the page forms.
		"what component can I add", "what component can you add",
		"what can I add to the page", "what can I add to my page",
	} {
		out := captureStdout(t, func() {
			if !tryComponentList(line) {
				t.Errorf("%q did not claim the discovery answer", line)
			}
		})
		if !strings.Contains(out, "text field") || !strings.Contains(out, "switch") {
			t.Errorf("discovery answer for %q is incomplete:\n%s", line, out)
		}
	}
}

// The discovery claim leaves symbol questions to the
// symbol explainer: they were never in the closed set, and
// the singular/page additions did not widen it toward them.
func TestComponentDiscoveryLeavesSymbolQuestions(t *testing.T) {
	for _, line := range []string{
		"what does Add do",
		"what does chat.routeDomain do",
		"what component is a button",
	} {
		if tryComponentList(line) {
			t.Errorf("tryComponentList(%q) claimed a line that is not its", line)
		}
	}
}

// John's sentences end to end, starting from a basic page:
// a button lands with its styles, the Done line names the
// component, and "the page" then resolves to the same file
// for a card.
func TestComponentButtonLandsAndThePageResolves(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": analyze.WebpageContent("basic", "index"),
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a button to jim/temp/index.html", yes, profile)
	})
	if !strings.Contains(out, "Done — The button is in jim/temp/index.html.") {
		t.Errorf("button landing confirmation missing:\n%s", out)
	}
	page := readBack(t, root, "jim/temp/index.html")
	for _, want := range []string{analyze.MaterialStyleMarker, "--primary: #6750A4", ".btn {", `class="btn btn-filled"`} {
		if !strings.Contains(page, want) {
			t.Errorf("page after the button is missing %q", want)
		}
	}
	// "the page" now means that file: a card joins it, the
	// reading printed, styles still carried exactly once.
	out = captureStdout(t, func() {
		claimChain("add a card to the page", yes, profile)
	})
	if !strings.Contains(out, "Reading \"the page\" as jim/temp/index.html") {
		t.Errorf("the page reading was not printed:\n%s", out)
	}
	if !strings.Contains(out, "Done — The card is in jim/temp/index.html.") {
		t.Errorf("card landing confirmation missing:\n%s", out)
	}
	page = readBack(t, root, "jim/temp/index.html")
	if strings.Count(page, "--primary: #6750A4") != 1 || strings.Count(page, ".btn {") != 1 || strings.Count(page, ".card {") != 1 {
		t.Errorf("styles were not carried exactly once:\n%s", page)
	}
	// A second button adds the snippet and no CSS.
	out = captureStdout(t, func() {
		claimChain("add a button to jim/temp/index.html", yes, profile)
	})
	page = readBack(t, root, "jim/temp/index.html")
	if strings.Count(page, `class="btn btn-filled"`) != 2 {
		t.Error("the second button did not land")
	}
	if strings.Count(page, "--primary: #6750A4") != 1 || strings.Count(page, ".btn {") != 1 {
		t.Error("the second button carried the styles again")
	}
}

// John's two live lines from 2026-11-09, end to end: the
// dropped-article add lands the card in the named page,
// and the singular discovery line answers with the list —
// the two answers he did not get.
func TestComponentJohnsDroppedArticleAndSingularLines(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": analyze.WebpageContent("basic", "index"),
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add card to jim/temp/index.html", yes, profile)
	})
	if !strings.Contains(out, "Done — The card is in jim/temp/index.html.") {
		t.Errorf("dropped-article landing confirmation missing:\n%s", out)
	}
	if page := readBack(t, root, "jim/temp/index.html"); !strings.Contains(page, `class="card"`) {
		t.Errorf("the card did not land:\n%s", page)
	}
	out = captureStdout(t, func() {
		if !claimChain("what component can i add", yes, profile) {
			t.Error("the chain declined the singular discovery line")
		}
	})
	if !strings.Contains(out, "The components you can add to a page") || !strings.Contains(out, "text field") {
		t.Errorf("singular discovery answer incomplete:\n%s", out)
	}
}

// The remaining pinned sentences: a text field into a typed
// file, and a switch to "the page" in a session that just
// made the page through the chat itself.
func TestComponentTextFieldAndSwitchToThePage(t *testing.T) {
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": analyze.WebpageContent("responsive", "index"),
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a text field to jim/temp/index.html", yes, profile)
	})
	if !strings.Contains(out, "Done — The text field is in jim/temp/index.html.") {
		t.Errorf("text field confirmation missing:\n%s", out)
	}
	if page := readBack(t, root, "jim/temp/index.html"); !strings.Contains(page, `class="field"`) || !strings.Contains(page, ".field label {") {
		t.Errorf("the text field and its styles did not land:\n%s", page)
	}

	root = staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/keep.css": "body {}\n",
	}))
	out = captureStdout(t, func() {
		claimChain("add a basic webpage to jim/temp/index.html", yes, profile)
		claimChain("add a switch to the page", yes, profile)
	})
	if !strings.Contains(out, "Reading \"the page\" as jim/temp/index.html") {
		t.Errorf("the page reading was not printed:\n%s", out)
	}
	if !strings.Contains(out, "Done — The switch is in jim/temp/index.html.") {
		t.Errorf("switch confirmation missing:\n%s", out)
	}
	if page := readBack(t, root, "jim/temp/index.html"); !strings.Contains(page, `class="switch"`) {
		t.Errorf("the switch did not land:\n%s", page)
	}
}

// "the page" with no page in the session: honest guidance,
// never a guess — and nothing is written.
func TestComponentThePageWithoutContext(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/keep.css": "body {}\n",
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		if !claimChain("add a card to the page", yes, profile) {
			t.Error("the component claim declined the no-context page ask")
		}
	})
	if !strings.Contains(out, "I don't have a page from this session") {
		t.Errorf("no-context guidance missing:\n%s", out)
	}
}

// The refusals through the chain: an unknown component
// answers with the registry set, a missing file and an
// empty file both point at the webpage piece first, and a
// page with no </body> names the structure it lacks.
func TestComponentRefusalsThroughChain(t *testing.T) {
	profile := &UserProfile{}
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": analyze.WebpageContent("basic", "index"),
		"jim/temp/empty.html": "",
		"jim/temp/nobody.html": "<html>\n<head></head>\n<p>hi</p>\n</html>\n",
	}))
	out := captureStdout(t, func() {
		claimChain("add a carousel to jim/temp/index.html", yes, profile)
	})
	if !strings.Contains(out, "no component called \"carousel\"") {
		t.Errorf("unknown-component refusal missing:\n%s", out)
	}
	for _, c := range analyze.MaterialComponents {
		if !strings.Contains(out, c.Words) {
			t.Errorf("unknown-component refusal does not list %q:\n%s", c.Words, out)
		}
	}
	if got := readBack(t, root, "jim/temp/index.html"); got != analyze.WebpageContent("basic", "index") {
		t.Error("the unknown-component refusal wrote to the page")
	}
	out = captureStdout(t, func() {
		claimChain("add a button to jim/temp/nope.html", yes, profile)
	})
	if !strings.Contains(out, "add a webpage to jim/temp/nope.html") {
		t.Errorf("missing-file two-step missing:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("add a button to jim/temp/empty.html", yes, profile)
	})
	if !strings.Contains(out, "is empty") {
		t.Errorf("empty-file guidance missing:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimChain("add a button to jim/temp/nobody.html", yes, profile)
	})
	if !strings.Contains(out, "no </body>") {
		t.Errorf("no-body guidance missing:\n%s", out)
	}
}

// A button into a material page through the chain adds no
// CSS: the file afterwards is the template page with one
// more button before </body>.
func TestComponentButtonIntoMaterialPage(t *testing.T) {
	material := analyze.WebpageContent("material", "index")
	root := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/temp/index.html": material,
	}))
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a button to jim/temp/index.html", yes, profile)
	})
	if !strings.Contains(out, "Done — The button is in jim/temp/index.html.") {
		t.Errorf("confirmation missing:\n%s", out)
	}
	snippet, _ := analyze.MaterialComponentSnippet("button", "")
	want := strings.Replace(material, "</body>", snippet+"</body>", 1)
	if got := readBack(t, root, "jim/temp/index.html"); got != want {
		t.Errorf("the material page changed by more than the snippet")
	}
}
