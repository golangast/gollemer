package chat

import (
	"strings"
	"testing"
)

// The phrasing layer (attempt 2, arm B seated): classify +
// extract maps messy sentences onto canonical ones, validated
// through the real parsers, and each piece executes through
// the deterministic write claims with its own gate. These
// tests pin the route on the real trained classifier, the
// deterministic-hit protection (grammar-served lines never
// consult the layer), the messy single and John's compound
// landing end to end, the declined-gate stop, and none-class
// lines falling through untouched.

// withPhrasingModel points the classifier seams at the real
// trained artifact (the production loader resolves it through
// the project root, which tests aim at scratch fixtures).
func withPhrasingModel(t *testing.T) {
	t.Helper()
	m, err := LoadIntentModel("../../../../data/phrasing_model.json")
	if err != nil {
		t.Fatalf("load phrasing model: %v", err)
	}
	oldC, oldT := phrasingClassifier, phrasingTopFamily
	phrasingClassifier = func(line string) (string, bool) {
		fam, _, ok := m.ClassifyIntent(line)
		return fam, ok
	}
	phrasingTopFamily = func(line string) string {
		fam, _, _ := m.Predict(line)
		return fam
	}
	t.Cleanup(func() { phrasingClassifier, phrasingTopFamily = oldC, oldT })
}

func TestPhrasingRouteShapes(t *testing.T) {
	withPhrasingModel(t)
	cases := []struct {
		line string
		want []string
	}{
		{"add instantiation of struct Jim to handler Jake and have it print the results",
			[]string{"add instantiation of struct Jim to handler Jake", `add a print line saying "the results" in Jake to main.go`}},
		{"kindly register userCount in Metrics",
			[]string{"register userCount in Metrics"}},
		{"hey gollemer, add a var to vault.go",
			[]string{"add a variable to vault.go"}}, // the book's canonical wording
		{"could you stick a for loop in main to main.go",
			[]string{"add a for loop in main to main.go"}},
	}
	for _, c := range cases {
		got, ok := PhrasingRoute(c.line)
		if !ok {
			t.Errorf("PhrasingRoute(%q) not claimed", c.line)
			continue
		}
		if strings.Join(got, " ## ") != strings.Join(c.want, " ## ") {
			t.Errorf("PhrasingRoute(%q) = %q, want %q", c.line, got, c.want)
		}
	}
}

func TestPhrasingRouteRefusesNoneClass(t *testing.T) {
	withPhrasingModel(t)
	for _, line := range []string{
		"write a function that reverses a string",
		"what is a goroutine",
		"how do I sort a slice in Go",
		"add a function named jake that returns a string", // the boundary's own line
		"tell me a joke",
	} {
		if pieces, ok := PhrasingRoute(line); ok {
			t.Errorf("PhrasingRoute(%q) claimed %q — none-class lines must fall through", line, pieces)
		}
		if runPhrasingLayer(line, yes) {
			t.Errorf("runPhrasingLayer(%q) claimed a none-class line", line)
		}
	}
}

// phrasingFixture: Jim lives in jim.go, Jake (and main) in
// main.go — the shape John's compound needs: the instantiate
// piece resolves project-wide, the print piece targets
// main.go's Jake.
func phrasingFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/phrasing\n\ngo 1.21\n")
	writeFile(t, dir, "jim.go", `package main

type Jim struct {
	Name  string
	Count int
}
`)
	writeFile(t, dir, "main.go", `package main

func Jake() {
	println("ok")
}

func main() { Jake() }
`)
	return dir
}

// The deterministic-hit protection: lines a grammar serves
// are claimed before the layer is ever consulted, and the
// consultation count proves it.
func TestPhrasingProtectionGrammarLinesNeverConsult(t *testing.T) {
	withPhrasingModel(t)
	dir := phrasingFixture(t)
	withChatSession(t, dir, true)

	consults := 0
	old := phrasingProvider
	phrasingProvider = func(line string) ([]string, bool) {
		consults++
		return PhrasingRoute(line)
	}
	t.Cleanup(func() { phrasingProvider = old })

	profile := &UserProfile{}
	for _, line := range []string{
		"add a for loop in main to main.go",                 // the book's own grammar
		"add instantiation of struct Jim to handler Jake",   // instantiate's own grammar
		"add a print line saying \"hi\" in main to main.go", // the book's print grammar
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through the chain:\n%s", line, out)
		}
	}
	if consults != 0 {
		t.Errorf("grammar-served lines consulted the phrasing layer %d times, want 0", consults)
	}

	// A messy line the grammars cannot read DOES consult it,
	// exactly once, and lands through it.
	line := "could you stick a switch statement in Jake to main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("messy line %q fell through:\n%s", line, out)
	}
	if consults != 1 {
		t.Errorf("messy line consulted the layer %d times, want 1", consults)
	}
	if !strings.Contains(out, "phrasing layer") {
		t.Errorf("messy line's answer does not name the phrasing layer:\n%s", out)
	}
	got := readBack(t, dir, "main.go")
	if !strings.Contains(got, "switch") {
		t.Errorf("the switch statement did not land in main.go:\n%s", got)
	}
}

// John's compound, end to end: both pieces land inside Jake,
// each through its own gated flow, and the answer says the
// phrasing layer read the line and shows both canonicals.
func TestPhrasingJohnsCompoundEndToEnd(t *testing.T) {
	withPhrasingModel(t)
	dir := phrasingFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	line := "add instantiation of struct Jim to handler Jake and have it print the results"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("John's compound fell through the chain:\n%s", out)
	}
	for _, want := range []string{
		"phrasing layer",
		"add instantiation of struct Jim to handler Jake",
		`add a print line saying "the results" in Jake to main.go`,
		"built the project and checked",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("compound output missing %q:\n%s", want, out)
		}
	}
	got := readBack(t, dir, "main.go")
	jakeAt := strings.Index(got, "func Jake()")
	jimAt := strings.Index(got, "jim := Jim{")
	printAt := strings.Index(got, `fmt.Println("the results")`)
	mainAt := strings.Index(got, "func main()")
	if !(jakeAt >= 0 && jimAt > jakeAt && printAt > jakeAt && jimAt < mainAt && printAt < mainAt) {
		t.Errorf("both pieces should land inside Jake:\n%s", got)
	}
}

// A declined gate on piece one stops the compound: piece two
// is never offered, nothing lands, and the stop is said.
func TestPhrasingCompoundDeclineStopsRest(t *testing.T) {
	withPhrasingModel(t)
	dir := phrasingFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	asks := 0
	noThenYes := func(string) (string, error) {
		asks++
		if asks == 1 {
			return "n", nil
		}
		return "y", nil
	}
	line := "add instantiation of struct Jim to handler Jake and have it print the results"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, noThenYes, profile) })
	if !handled {
		t.Fatalf("compound fell through:\n%s", out)
	}
	if asks != 1 {
		t.Errorf("gate asked %d times, want exactly 1 (the second piece must never be offered)", asks)
	}
	if !strings.Contains(out, "Stopping there") {
		t.Errorf("the stop was not said plainly:\n%s", out)
	}
	got := readBack(t, dir, "main.go")
	if strings.Contains(got, "jim := Jim{") || strings.Contains(got, "Println") {
		t.Errorf("a declined compound wrote something:\n%s", got)
	}
}
