package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Journeys for the richer follow-ups (followup.go): the
// last landed piece repeats from "another one" / "the same
// but for X" by rebuilding its own canonical sentence and
// running it through its own claim — never by replaying
// state. All journeys run the real claim chain (claimChain),
// with the vocabulary stores isolated (vocabFixture).

func followupProject(t *testing.T, files map[string]string) string {
	t.Helper()
	root, _ := vocabFixture(t, files)
	return root
}

const followupMainGo = `package main

func main() {}
`

// A concept declaration repeated bare asks its name through
// the clarifying machinery; the one-word answer completes
// the repeat through the book's own claim.
func TestFollowUpConceptDeclarationNameQuestion(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod":      "module example.com/followup\n\ngo 1.21\n",
		"doc.go":      "package followup\n",
		"jim/main.go": followupMainGo,
	})
	out := vocabRun(t, "add a struct named Profile to jim/main.go")
	if !strings.Contains(out, "Here's the edit I assembled") {
		t.Fatalf("the original piece did not land:\n%s", out)
	}
	out = vocabRun(t, "another one")
	if !strings.Contains(out, "what should I call it?") {
		t.Errorf("the bare follow-up asked no name question:\n%s", out)
	}
	if !hasPendingQuestion() {
		t.Fatal("the follow-up set no pending question")
	}
	out = vocabRun(t, "Settings")
	if !strings.Contains(out, `I read that as: "add a struct named Settings to jim/main.go".`) {
		t.Errorf("the answer did not complete as the rebuilt sentence:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	for _, want := range []string{"type Profile struct", "type Settings struct"} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q:\n%s", want, got)
		}
	}
	// A follow-up to the follow-up: the context moved on to
	// the new piece, and a named repeat lands directly.
	out = vocabRun(t, "another one named Extra in jim/main.go")
	if !strings.Contains(out, `I read that as: "add a struct named Extra to jim/main.go".`) {
		t.Errorf("the named follow-up did not run:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "type Extra struct") {
		t.Errorf("the chained follow-up did not land:\n%s", got)
	}
}

// A target swap keeps the piece's name: "the same but for
// X" is the same piece, somewhere else.
func TestFollowUpTargetSwapKeepsName(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod":         "module example.com/followup\n\ngo 1.21\n",
		"doc.go":         "package followup\n",
		"jim/main.go":    followupMainGo,
		"store/store.go": "package store\n",
	})
	vocabRun(t, "add a struct named Profile to jim/main.go")
	out := vocabRun(t, "the same but for store/profile.go")
	if !strings.Contains(out, `I read that as: "add a struct named Profile to store/profile.go".`) {
		t.Errorf("the swap did not run as the rebuilt sentence:\n%s", out)
	}
	if got := readBack(t, root, "store/profile.go"); !strings.Contains(got, "type Profile struct") {
		t.Errorf("the swapped piece did not land in store/profile.go:\n%s", got)
	}
	// And the same name at the ORIGINAL target refuses as a
	// fact — the repeat faces the same checks as the full
	// sentence: Profile is already declared in that package.
	out = vocabRun(t, "the same but for jim/main.go")
	if !strings.Contains(out, "Profile already exists in package main") {
		t.Errorf("the same-package swap did not refuse as a fact:\n%s", out)
	}
}

// A statement repeats with its saying carried; a name given
// to a statement is answered honestly, never dropped.
func TestFollowUpStatementRepeat(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod":      "module example.com/followup\n\ngo 1.21\n",
		"doc.go":      "package followup\n",
		"jim/main.go": followupMainGo,
	})
	vocabRun(t, `add a print line in main saying "hello there" to jim/main.go`)
	out := vocabRun(t, "another one")
	if !strings.Contains(out, `saying "hello there"`) {
		t.Errorf("the repeat dropped the saying:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); strings.Count(got, `fmt.Println("hello there")`) != 2 {
		t.Errorf("the statement did not land twice:\n%s", got)
	}
	out = vocabRun(t, "another one named Foo")
	if !strings.Contains(out, "is a statement") || !strings.Contains(out, "doesn't take a name") {
		t.Errorf("the named statement follow-up was not answered honestly:\n%s", out)
	}
}

// A taught piece repeats through the teach seat's own use
// claim, template and all.
func TestFollowUpTaughtRepeat(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod":      "module example.com/followup\n\ngo 1.21\n",
		"doc.go":      "package followup\n",
		"jim/main.go": followupMainGo,
	})
	vocabRun(t,
		"add a form handler named Jake to jim/main.go",
		"learn this as a signup form from Jake in jim/main.go",
		"add a signup form named Signup to jim/signup.go",
	)
	out := vocabRun(t, "another one named Signin to jim/signup.go")
	if !strings.Contains(out, `I read that as: "add a signup form named Signin to jim/signup.go".`) {
		t.Errorf("the taught follow-up did not run:\n%s", out)
	}
	if got := readBack(t, root, "jim/signup.go"); !strings.Contains(got, "func Signin(") {
		t.Errorf("the taught repeat did not land:\n%s", got)
	}
}

// A webserver repeats at a swapped target.
func TestFollowUpServerSwap(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod":      "module example.com/followup\n\ngo 1.21\n",
		"doc.go":      "package followup\n",
		"jim/main.go": followupMainGo,
		"api/main.go": followupMainGo,
	})
	vocabRun(t, "add a webserver to api/main.go")
	out := vocabRun(t, "the same but for jim/main.go")
	if !strings.Contains(out, `I read that as: "add a webserver to jim/main.go".`) {
		t.Errorf("the server swap did not run:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "ListenAndServe") {
		t.Errorf("the webserver did not land in jim/main.go:\n%s", got)
	}
}

// A wiring repeats by name: bare, it asks which handler
// through the clarifying machinery, and the answer wires
// that handler at its own default route.
func TestFollowUpWireRepeat(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod": "module example.com/followup\n\ngo 1.21\n",
		"doc.go": "package followup\n",
		"jim/main.go": `package main

import "net/http"

func main() {
	http.ListenAndServe(":8080", nil)
}

func Jake(w http.ResponseWriter, r *http.Request) {}
func Jim(w http.ResponseWriter, r *http.Request) {}
`,
	})
	vocabRun(t, "connect the Jake handler to the webserver")
	out := vocabRun(t, "another one")
	if !strings.Contains(out, "Another handler to wire in — what's its name?") {
		t.Errorf("the bare wire follow-up asked no question:\n%s", out)
	}
	out = vocabRun(t, "Jim")
	if !strings.Contains(out, `I read that as: "connect the Jim handler to the webserver".`) {
		t.Errorf("the wire answer did not complete as the rebuilt sentence:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	for _, want := range []string{`http.HandleFunc("/jake", Jake)`, `http.HandleFunc("/jim", Jim)`} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q:\n%s", want, got)
		}
	}
}

// A page kind repeats at a swapped target with its handler
// name kept.
func TestFollowUpPageSwap(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod": "module example.com/followup\n\ngo 1.21\n",
		"doc.go": "package followup\n",
	})
	vocabRun(t,
		"create a file site/main.go with a webserver",
		"create a file web/main.go with a webserver",
		"create a homepage in site",
	)
	out := vocabRun(t, "the same but for web")
	if !strings.Contains(out, `I read that as: "create a homepage named Home in web".`) {
		t.Errorf("the page swap did not run:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "web", "index.html")); err != nil {
		t.Errorf("the swapped homepage's page did not land: %v", err)
	}
	if got := readBack(t, root, "web/main.go"); !strings.Contains(got, "func Home(") {
		t.Errorf("the swapped homepage's handler did not land:\n%s", got)
	}
}

// A created file repeats as a creation: bare, it asks the
// new file's name with a target question; named, it lands
// in the same folder directly.
func TestFollowUpCreateRepeat(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod": "module example.com/followup\n\ngo 1.21\n",
		"doc.go": "package followup\n",
	})
	vocabRun(t, "create a folder jim", "create a file jim/notes.go")
	out := vocabRun(t, "another one")
	if !strings.Contains(out, "Another file — what should it be called?") {
		t.Errorf("the bare create follow-up asked no question:\n%s", out)
	}
	if !hasPendingQuestion() {
		t.Fatal("the create follow-up set no pending question")
	}
	vocabRun(t, "todo.go")
	if _, err := os.Stat(filepath.Join(root, "jim", "todo.go")); err != nil {
		t.Errorf("the answered creation did not land: %v", err)
	}
	vocabRun(t, "another one named list.go")
	if _, err := os.Stat(filepath.Join(root, "jim", "list.go")); err != nil {
		t.Errorf("the named creation did not land: %v", err)
	}
}

// Composition: a follow-up's target slot takes a taught
// name, with the reading printed.
func TestFollowUpAliasComposition(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod":      "module example.com/followup\n\ngo 1.21\n",
		"doc.go":      "package followup\n",
		"jim/main.go": followupMainGo,
		"api/main.go": followupMainGo,
	})
	vocabRun(t,
		"call api/main.go the api file",
		"add a struct named Profile to jim/main.go",
	)
	out := vocabRun(t, "another one named Copy in the api file")
	if !strings.Contains(out, `Reading "the api file" as api/main.go — your name for it.`) {
		t.Errorf("the composition printed no reading:\n%s", out)
	}
	if got := readBack(t, root, "api/main.go"); !strings.Contains(got, "type Copy struct") {
		t.Errorf("the composed follow-up did not land in api/main.go:\n%s", got)
	}
}

// "the other file": with exactly two files in play, a
// concept sentence aimed at it lands in the OTHER one, with
// the reading printed.
func TestOtherFileReference(t *testing.T) {
	root := followupProject(t, map[string]string{
		"go.mod": "module example.com/followup\n\ngo 1.21\n",
		"doc.go": "package followup\n",
		"jim/a.go": `package main

func main() {}
`,
		"jim/b.go": `package main

func Helper() {}
`,
	})
	vocabRun(t,
		"add a print line in main to jim/a.go",
		"add a struct named Marker to jim/b.go",
	)
	out := vocabRun(t, "add a print line to the other file")
	if !strings.Contains(out, `Reading "the other file" as jim/a.go — the other file from this session's work.`) {
		t.Errorf("the other-file reference printed no reading:\n%s", out)
	}
	if got := readBack(t, root, "jim/a.go"); strings.Count(got, "fmt.Println") != 2 {
		t.Errorf("the other-file piece did not land in jim/a.go:\n%s", got)
	}
	if got := readBack(t, root, "jim/b.go"); strings.Contains(got, "fmt.Println") {
		t.Errorf("the other-file piece landed in the wrong file:\n%s", got)
	}
}

// A piece with no rebuildable sentence (a delegate) gets an
// honest answer naming it — never a fabricated repeat.
func TestFollowUpNotRebuildable(t *testing.T) {
	_ = followupProject(t, map[string]string{
		"go.mod":         "module example.com/followup\n\ngo 1.21\n",
		"jim/main.go":    followupMainGo,
		"store/store.go": "package store\n\nfunc Load() string { return \"x\" }\n",
	})
	vocabRun(t, "add a function in jim/main.go that calls store.Load")
	out := vocabRun(t, "another one")
	if !strings.Contains(out, "can't repeat from a follow-up") || !strings.Contains(out, "function that calls another") {
		t.Errorf("the unrebuildable piece was not answered honestly:\n%s", out)
	}
}

// The pins: with no session context, follow-up lines claim
// nothing and print nothing — byte-identical to before the
// feature existed.
func TestFollowUpNoContextSilent(t *testing.T) {
	_ = followupProject(t, map[string]string{
		"go.mod":      "module example.com/followup\n\ngo 1.21\n",
		"doc.go":      "package followup\n",
		"jim/main.go": followupMainGo,
	})
	for _, line := range []string{"another one", "another one named Extra in jim/main.go", "the same but for jim/main.go"} {
		var handled bool
		out := captureStdout(t, func() { handled = runFollowUp(line, yes) })
		if handled || out != "" {
			t.Errorf("%q with no context: handled=%v out=%q, want silent decline", line, handled, out)
		}
	}
	// And the spoken-target pass leaves a follow-up line
	// untouched even when a name it mentions is taught.
	vocabRun(t, "call jim/main.go the api file")
	out := captureStdout(t, func() {
		if got := rewriteSpokenTargets("another one in the api file"); got != "another one in the api file" {
			t.Errorf("the pass rewrote a follow-up line: %q", got)
		}
	})
	if out != "" {
		t.Errorf("the pass printed for a follow-up line:\n%s", out)
	}
}
