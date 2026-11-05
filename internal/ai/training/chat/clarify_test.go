package chat

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// Journeys for the clarifying questions (clarify.go): an
// ask that used to end in guidance prose now asks ONE
// question, and the next line — a name, a number, a path,
// a distinguishing word — completes the ORIGINAL ask
// through its own claim and gate. A line that is not an
// answer clears the question and is processed exactly as
// if none had been asked. All journeys run the real claim
// chain (claimChain, addserver_session_test.go), the same
// order as the unified chat loop.

// clarifyPhrasing points the classifier seams at the real
// trained artifact, like withPhrasingModel — but resolved
// from this file's path, because a journey test builds
// several fixtures and t.Chdir moves the working directory
// the relative-path loader depends on.
func clarifyPhrasing(t *testing.T) {
	t.Helper()
	_, file, _, _ := runtime.Caller(0)
	m, err := LoadIntentModel(filepath.Join(filepath.Dir(file), "..", "..", "..", "..", "data", "phrasing_model.json"))
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

func clarifyFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, content := range files {
		writeFile(t, root, rel, content)
	}
	clarifyPhrasing(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	return root
}

func clarifyRun(t *testing.T, lines ...string) string {
	t.Helper()
	profile := &UserProfile{}
	return captureStdout(t, func() {
		for _, line := range lines {
			claimChain(line, yes, profile)
		}
	})
}

const clarifyWireMain = `package main

import "net/http"

func main() {
	http.ListenAndServe(":8080", nil)
}
`

const clarifyWireHandlers = `package main

import "net/http"

func Jake(w http.ResponseWriter, r *http.Request) {}
func Jim(w http.ResponseWriter, r *http.Request) {}
`

func clarifyWireFixture(t *testing.T) string {
	t.Helper()
	return clarifyFixture(t, map[string]string{
		"main.go":     clarifyWireMain,
		"handlers.go": clarifyWireHandlers,
	})
}

// The headline journey: a bare wiring ask with two
// unregistered handlers asks which one; the NAME answer
// completes the original ask and the registration lands
// behind the normal gate.
func TestClarifyWireNameAnswerLands(t *testing.T) {
	root := clarifyWireFixture(t)
	out := clarifyRun(t, "connect the handler to the webserver")
	for _, want := range []string{
		"More than one handler isn't registered yet — which one should I wire in?",
		"1. Jake (handlers.go:",
		"2. Jim (handlers.go:",
		"Nothing's been written",
		"Answer with the name or the number",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("question missing %q:\n%s", want, out)
		}
	}
	if !hasPendingQuestion() {
		t.Fatal("the bare wiring ask set no pending question")
	}
	out = clarifyRun(t, "Jake")
	if !strings.Contains(out, `http.HandleFunc("/jake", Jake)`) && !strings.Contains(readBack(t, root, "main.go"), `http.HandleFunc("/jake", Jake)`) {
		t.Errorf("the name answer did not wire Jake:\n%s\n--- main.go:\n%s", out, readBack(t, root, "main.go"))
	}
	if hasPendingQuestion() {
		t.Error("the completed ask left a question pending")
	}
}

// The NUMBER answer picks from the same list.
func TestClarifyWireNumberAnswerLands(t *testing.T) {
	root := clarifyWireFixture(t)
	clarifyRun(t, "connect the handler to the webserver")
	clarifyRun(t, "2")
	if got := readBack(t, root, "main.go"); !strings.Contains(got, `http.HandleFunc("/jim", Jim)`) {
		t.Errorf("the number answer did not wire Jim:\n%s", got)
	}
}

// A full valid ask while a question is pending is NOT an
// answer: it clears the question and is processed as
// itself — nothing from the held ask happens.
func TestClarifyQuestionFallsThroughToFullAsk(t *testing.T) {
	root := clarifyWireFixture(t)
	clarifyRun(t, "connect the handler to the webserver")
	out := clarifyRun(t, "add a struct named Profile to main.go")
	if got := readBack(t, root, "main.go"); !strings.Contains(got, "type Profile struct") {
		t.Errorf("the full ask did not land:\n%s\n--- out:\n%s", got, out)
	}
	if got := readBack(t, root, "main.go"); strings.Contains(got, "HandleFunc(\"/jake\"") {
		t.Errorf("the held wiring ask completed behind a full ask:\n%s", got)
	}
	if hasPendingQuestion() {
		t.Error("the full ask left the stale question pending")
	}
}

// The full original sentence the guidance used to spell
// still works while a question is pending.
func TestClarifyWireFullSentenceStillWorks(t *testing.T) {
	root := clarifyWireFixture(t)
	clarifyRun(t, "connect the handler to the webserver")
	clarifyRun(t, "connect the Jake handler to the webserver")
	if got := readBack(t, root, "main.go"); !strings.Contains(got, `http.HandleFunc("/jake", Jake)`) {
		t.Errorf("the full sentence did not wire Jake:\n%s", got)
	}
	if hasPendingQuestion() {
		t.Error("the full sentence left a question pending")
	}
}

// A fact never converts: when the answered handler's route
// is already taken, the completion refuses with the
// conflict — and offers no further question.
func TestClarifyWireConflictStaysRefusal(t *testing.T) {
	root := clarifyFixture(t, map[string]string{
		"main.go": `package main

import "net/http"

func Other(w http.ResponseWriter, r *http.Request) {}

func main() {
	http.HandleFunc("/jake", Other)
	http.ListenAndServe(":8080", nil)
}
`,
		"handlers.go": clarifyWireHandlers,
	})
	clarifyRun(t, "connect the handler to the webserver")
	out := clarifyRun(t, "Jake")
	if !strings.Contains(out, "already taken") {
		t.Errorf("the conflict answer did not refuse with the taken route:\n%s", out)
	}
	if hasPendingQuestion() {
		t.Error("a conflict refusal set a question")
	}
	if got := readBack(t, root, "main.go"); strings.Contains(got, `"/jake", Jake`) {
		t.Errorf("the conflicted registration was written:\n%s", got)
	}
}

// The database two-sentence choice, answered by ONE word
// each way.
func TestClarifyDatabaseWordAnswers(t *testing.T) {
	files := map[string]string{"jim/main.go": "package main\n\nfunc main() {}\n"}

	root := clarifyFixture(t, files)
	out := clarifyRun(t, "add a database to jim/main.go")
	for _, want := range []string{
		`"database" names two different pieces here`,
		"1. database connection function",
		"2. inline database piece",
		`Answer "connection" or "inline", or the number, or the full sentence`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("database question missing %q:\n%s", want, out)
		}
	}
	clarifyRun(t, "connection")
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("the \"connection\" answer did not land the declaration:\n%s", got)
	}

	root = clarifyFixture(t, files)
	clarifyRun(t, "add a database to jim/main.go")
	clarifyRun(t, "inline")
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "sql.Open(") {
		t.Errorf("the \"inline\" answer did not land the statement:\n%s", got)
	}
}

// The ambiguous-piece choice (handler), answered by its
// distinguishing word.
func TestClarifyAmbiguousPieceWordAnswer(t *testing.T) {
	root := clarifyFixture(t, map[string]string{"main.go": "package main\n\nfunc main() {}\n"})
	clarifyRun(t, "add handler Jim")
	clarifyRun(t, "form")
	if _, err := os.Stat(filepath.Join(root, "handlers", "jim.go")); err != nil {
		// The form handler's folder-default landing: wherever
		// it landed, its source must exist somewhere.
		found := false
		filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() && strings.HasSuffix(path, ".go") {
				if b, rerr := os.ReadFile(path); rerr == nil && strings.Contains(string(b), "func Jim(") {
					found = true
				}
			}
			return nil
		})
		if !found {
			t.Errorf("the \"form\" answer landed no Jim handler under %s", root)
		}
	}
}

// Missing target: the question offers the tracked file as
// a numbered suggestion; the number completes the ask.
func TestClarifyNoTargetSuggestionNumber(t *testing.T) {
	root := clarifyFixture(t, map[string]string{
		"main.go":     "package main\n\nfunc main() {}\n",
		"jim/main.go": "package main\n\nfunc main() {}\n",
	})
	clarifyRun(t, "create a file jim/db.go")
	out := clarifyRun(t, "add a struct named Profile")
	for _, want := range []string{
		"which file should the struct go in?",
		"1. jim/db.go",
		"Answer with a file or folder",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("no-target question missing %q:\n%s", want, out)
		}
	}
	clarifyRun(t, "1")
	if got := readBack(t, root, "jim/db.go"); !strings.Contains(got, "type Profile struct") {
		t.Errorf("the number answer did not land the struct in the suggested file:\n%s", got)
	}
}

// Missing target, answered by typing the path outright.
func TestClarifyNoTargetPathAnswer(t *testing.T) {
	root := clarifyFixture(t, map[string]string{
		"main.go":     "package main\n\nfunc main() {}\n",
		"jim/main.go": "package main\n\nfunc main() {}\n",
	})
	clarifyRun(t, "create a file jim/db.go")
	clarifyRun(t, "add a struct named Profile")
	clarifyRun(t, "jim/db.go")
	if got := readBack(t, root, "jim/db.go"); !strings.Contains(got, "type Profile struct") {
		t.Errorf("the path answer did not land the struct:\n%s", got)
	}
}

// An unnamed declaration asks for its name; one identifier
// completes it.
func TestClarifyUnnamedDeclarationNameAnswer(t *testing.T) {
	root := clarifyFixture(t, map[string]string{"jim/main.go": "package main\n\nfunc main() {}\n"})
	out := clarifyRun(t, "add a struct to jim/main.go")
	if !strings.Contains(out, "needs a name — what should I call it?") {
		t.Errorf("unnamed question missing:\n%s", out)
	}
	clarifyRun(t, "Profile")
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "type Profile struct") {
		t.Errorf("the name answer did not land the struct:\n%s", got)
	}
}

// Ambiguous same-name files (session context): the answer
// by path completes into the chosen file; the answer by
// number can pick the ROOT namesake too.
func TestClarifyCtxAmbiguousAnswers(t *testing.T) {
	files := map[string]string{
		"main.go":     "package main\n\nfunc main() {}\n",
		"jim/main.go": "package main\n\nfunc main() {}\n",
		"db.go":       "package main\n\nfunc Run() {}\n",
	}
	// setup tracks jim/db.go in the session context and
	// gives it the host function the print ask names.
	setup := func(t *testing.T) string {
		root := clarifyFixture(t, files)
		clarifyRun(t, "create a file jim/db.go")
		clarifyRun(t, "add a function named Run to jim/db.go")
		return root
	}

	root := setup(t)
	out := clarifyRun(t, "add a print line in Run to db.go")
	for _, want := range []string{
		`"db.go" names more than one file here`,
		"1. jim/db.go",
		"2. db.go (a file at the project root)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("ctx question missing %q:\n%s", want, out)
		}
	}
	clarifyRun(t, "jim/db.go")
	if got := readBack(t, root, "jim/db.go"); !strings.Contains(got, "fmt.Println") {
		t.Errorf("the path answer did not land the print in jim/db.go:\n%s", got)
	}
	if got := readBack(t, root, "db.go"); strings.Contains(got, "fmt.Println") {
		t.Errorf("the print landed in the root namesake too:\n%s", got)
	}

	root = setup(t)
	clarifyRun(t, "add a print line in Run to db.go")
	clarifyRun(t, "2")
	if got := readBack(t, root, "db.go"); !strings.Contains(got, "fmt.Println") {
		t.Errorf("the number answer did not land the print in the root db.go:\n%s", got)
	}
}

// A folder with two package-main files: the declaration
// ask asks which file; the path answer completes it.
func TestClarifyMultiFolderAnswers(t *testing.T) {
	files := map[string]string{
		"jim/a.go": "package main\n\nfunc main() {}\n",
		"jim/b.go": "package main\n\nfunc helper() {}\n",
	}
	root := clarifyFixture(t, files)
	out := clarifyRun(t, "add a struct named Profile to jim")
	for _, want := range []string{
		"jim/ has more than one main-package file — which one should the struct go in?",
		"1. jim/a.go",
		"2. jim/b.go",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("multi question missing %q:\n%s", want, out)
		}
	}
	clarifyRun(t, "jim/b.go")
	if got := readBack(t, root, "jim/b.go"); !strings.Contains(got, "type Profile struct") {
		t.Errorf("the path answer did not land the struct in jim/b.go:\n%s", got)
	}

	// The statement sibling, by number.
	root = clarifyFixture(t, files)
	clarifyRun(t, "add a print line in main to jim")
	clarifyRun(t, "1")
	if got := readBack(t, root, "jim/a.go"); !strings.Contains(got, "fmt.Println") {
		t.Errorf("the number answer did not land the print in jim/a.go:\n%s", got)
	}

	// The webserver sibling, by base name.
	root = clarifyFixture(t, files)
	clarifyRun(t, "add a webserver to jim")
	clarifyRun(t, "a.go")
	if got := readBack(t, root, "jim/a.go"); !strings.Contains(got, "ListenAndServe") {
		t.Errorf("the base-name answer did not land the server in jim/a.go:\n%s", got)
	}
}

// Serve-html's dangling target, completed by a file name.
func TestClarifyServeDanglingTarget(t *testing.T) {
	root := clarifyFixture(t, map[string]string{"main.go": "package main\n\nfunc main() {}\n"})
	out := clarifyRun(t, "serve index.html to the file")
	if !strings.Contains(out, "which file should the wiring go in?") {
		t.Errorf("serve question missing:\n%s", out)
	}
	clarifyRun(t, "main.go")
	if got := readBack(t, root, "main.go"); !strings.Contains(got, "index.html") {
		t.Errorf("the file answer did not wire the page:\n%s", got)
	}
	if _, err := os.Stat(filepath.Join(root, "index.html")); err != nil {
		t.Errorf("the page was not created: %v", err)
	}
}

// The page family's dangling "named", completed by a name.
func TestClarifyPageNameMissing(t *testing.T) {
	root := clarifyFixture(t, map[string]string{"main.go": "package main\n\nfunc main() {}\n"})
	out := clarifyRun(t, "create a form page named")
	if !strings.Contains(out, "what should I call it?") {
		t.Errorf("page name question missing:\n%s", out)
	}
	clarifyRun(t, "Signup")
	if got := readBack(t, root, "main.go"); !strings.Contains(got, "func Signup(") {
		t.Errorf("the name answer landed no Signup handler:\n%s", got)
	}
	if _, err := os.Stat(filepath.Join(root, "form.html")); err != nil {
		t.Errorf("the form page was not created: %v", err)
	}
}

// With no question pending, the answer claim is invisible:
// it claims nothing and prints nothing.
func TestClarifyNoQuestionIsInvisible(t *testing.T) {
	clarifyFixture(t, map[string]string{"main.go": "package main\n\nfunc main() {}\n"})
	out := captureStdout(t, func() {
		if runClarifyAnswer("Jake", yes) {
			t.Error("runClarifyAnswer claimed a line with no question pending")
		}
		if runClarifyAnswer("jim/main.go", yes) {
			t.Error("runClarifyAnswer claimed a path with no question pending")
		}
	})
	if out != "" {
		t.Errorf("the answer claim printed with no question pending: %q", out)
	}
}

// Re-asking the same bare ask while its question is
// pending is not an answer: the question is asked again,
// nothing is wired.
func TestClarifyReaskIsNotAnAnswer(t *testing.T) {
	root := clarifyWireFixture(t)
	clarifyRun(t, "connect the handler to the webserver")
	clarifyRun(t, "connect the handler to the webserver")
	if !hasPendingQuestion() {
		t.Error("the re-asked question vanished")
	}
	if got := readBack(t, root, "main.go"); strings.Contains(got, "HandleFunc(\"/jake\"") {
		t.Errorf("the re-ask wired a handler:\n%s", got)
	}
	clearPendingQuestion()
}
