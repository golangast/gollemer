package chat

import (
	"strings"
	"testing"
)

// Journeys for the partner-style landing confirmations
// (confirm.go): every landing family's Done line is rendered
// from the transaction's own record — the piece and its file,
// the route a wiring added, the follow-up line a handler
// still needs — and nothing prints on a gate-out or a
// refusal. All journeys run the real claim chain.

const confirmGoMod = "module example.com/confirm\n\ngo 1.21\n"

const confirmServerMain = `package main

import "net/http"

func main() {
	http.ListenAndServe(":8080", nil)
}
`

const confirmPlainMain = `package main

func main() {}
`

const confirmCustomServerMain = `package main

import "net/http"

func main() {
	http.ListenAndServe(":8080", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
}
`

const confirmOneHandler = `package main

import "net/http"

func Jake(w http.ResponseWriter, r *http.Request) {}
`

func confirmFixture(t *testing.T, files map[string]string) string {
	t.Helper()
	if _, ok := files["go.mod"]; !ok {
		files["go.mod"] = confirmGoMod
	}
	return clarifyFixture(t, files)
}

func confirmRun(t *testing.T, lines ...string) string {
	t.Helper()
	profile := &UserProfile{}
	return captureStdout(t, func() {
		for _, line := range lines {
			claimChain(line, yes, profile)
		}
	})
}

// A named wiring lands and the confirmation names the
// handler, its own file, and the route it now answers at —
// the plan's record, not the sentence's words.
func TestConfirmWireNamed(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go":     confirmServerMain,
		"handlers.go": confirmOneHandler,
	})
	out := confirmRun(t, "connect the Jake handler to the webserver")
	if !strings.Contains(out, "Done — Jake is in handlers.go, registered at /jake. I built the project and checked that it's really there.") {
		t.Errorf("wire confirmation missing or wrong:\n%s", out)
	}
	if strings.Contains(out, ", at ") {
		t.Errorf("the machinery lead survived:\n%s", out)
	}
}

// A bare wiring ask completed through the clarifying
// question (item 1) lands through the same spine and gets
// the same confirmation — inheritance, not a special case.
func TestConfirmWireBareViaClarifyingAnswer(t *testing.T) {
	clarifyWireFixture(t)
	out := confirmRun(t, "connect the handler to the webserver", "Jake")
	if !strings.Contains(out, "Done — Jake is in handlers.go, registered at /jake.") {
		t.Errorf("clarified wire confirmation missing or wrong:\n%s", out)
	}
}

// The headline journey: a handler lands in a project that
// has a server, and the confirmation names the exact wire
// follow-up; typing it lands the wiring, whose confirmation
// names the route.
func TestConfirmHandlerDeclNamesWireFollowUp(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go": confirmServerMain,
	})
	out := confirmRun(t, "add a form handler named Jake to main.go")
	want := `Done — The form handler Jake is in main.go. I built the project and checked that it's really there. It isn't wired to the server yet — "connect the Jake handler to the webserver" will do it.`
	if !strings.Contains(out, want) {
		t.Errorf("handler confirmation missing its wire follow-up:\nwant %q in:\n%s", want, out)
	}
	out = confirmRun(t, "connect the Jake handler to the webserver")
	if !strings.Contains(out, "Done — Jake is in main.go, registered at /jake.") {
		t.Errorf("followed wire confirmation missing or wrong:\n%s", out)
	}
}

// A handler that lands where there is no server at all gets
// the honest prerequisite step — the add-a-webserver line,
// verified against the planner before it is promised.
func TestConfirmHandlerDeclWithoutServerNamesAddServer(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go": confirmPlainMain,
	})
	out := confirmRun(t, "add a form handler named Jake to main.go")
	want := `Done — The form handler Jake is in main.go. I built the project and checked that it's really there. It has no server of its own yet — "add a webserver to main.go" will add one.`
	if !strings.Contains(out, want) {
		t.Errorf("handler confirmation missing its add-server step:\nwant %q in:\n%s", want, out)
	}
}

// A custom server is a fact, not a missing slot: wiring
// into it is hand work, so the confirmation names no
// follow-up line at all rather than promising one the
// planner would refuse.
func TestConfirmHandlerDeclCustomServerStaysSilent(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go": confirmCustomServerMain,
	})
	out := confirmRun(t, "add a form handler named Jake to main.go")
	if !strings.Contains(out, "Done — The form handler Jake is in main.go.") {
		t.Fatalf("handler confirmation missing:\n%s", out)
	}
	if strings.Contains(out, "will do it") || strings.Contains(out, "will add one") {
		t.Errorf("the confirmation promised a follow-up the planner refuses:\n%s", out)
	}
}

// A page composed with its registration: the confirmation
// names the page, the handler, and the route it answers at.
func TestConfirmPageComposedNamesRoute(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go": confirmServerMain,
	})
	out := confirmRun(t, "create a form page to main.go")
	if !strings.Contains(out, "Done — The form page is in form.html — Form answers at /form.") {
		t.Errorf("page confirmation missing or wrong:\n%s", out)
	}
}

// A page with no server to compose into lands page and
// handler, and the confirmation says the handler's next
// step in the same verified shape as a handler landing.
func TestConfirmPageWithoutServer(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go": confirmPlainMain,
	})
	out := confirmRun(t, "create a form page to main.go")
	want := `Done — The form page is in form.html, and its handler Form is in main.go. I built the project and checked that it's really there. It has no server of its own yet — "add a webserver to main.go" will add one.`
	if !strings.Contains(out, want) {
		t.Errorf("serverless page confirmation missing or wrong:\nwant %q in:\n%s", want, out)
	}
}

// Serve-html on an existing program: file and page, from
// the plan's record.
func TestConfirmServeHTML(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go":    confirmServerMain,
		"index.html": "<html><body>hi</body></html>\n",
	})
	out := confirmRun(t, "serve an html file to main.go")
	if !strings.Contains(out, "Done — main.go now serves index.html.") {
		t.Errorf("serve confirmation missing or wrong:\n%s", out)
	}
}

// A webserver spliced into an existing main, and one
// created as a whole new program file.
func TestConfirmAddServer(t *testing.T) {
	confirmFixture(t, map[string]string{
		"main.go": confirmPlainMain,
	})
	out := confirmRun(t, "add a webserver to main.go")
	if !strings.Contains(out, "Done — main.go has a main that serves HTTP now.") {
		t.Errorf("add-server (existing file) confirmation missing or wrong:\n%s", out)
	}

	// A file holding only a bare package clause becomes the
	// whole program — the created-whole grammar.
	confirmFixture(t, map[string]string{
		"doc.go":      "package confirm\n",
		"srv/main.go": "package main\n",
	})
	out = confirmRun(t, "add a webserver to srv/main.go")
	if !strings.Contains(out, "Done — srv/main.go is a webserver program now.") {
		t.Errorf("add-server (bare-clause file) confirmation missing or wrong:\n%s", out)
	}
}

// A concept statement keeps the plan's verified clause as
// its first sentence — it is already a plain fact — with
// the spine's verification sentence after it and the old
// ", at <site>" machinery lead gone.
func TestConfirmConceptStatementKeepsClause(t *testing.T) {
	confirmFixture(t, map[string]string{
		"doc.go":      "package confirm\n",
		"app/main.go": "package main\n\nfunc main() {\n}\n",
	})
	out := confirmRun(t, "add a print line in main to app/main.go")
	if !strings.Contains(out, "Done — main.go now has the print line in main. I built the project and checked that it's really there.") {
		t.Errorf("statement confirmation missing or wrong:\n%s", out)
	}
	if strings.Contains(out, ", at app/main.go") {
		t.Errorf("the machinery lead survived:\n%s", out)
	}
}

// A taught kind lands with the same name-first grammar,
// its kind phrase the user's own words. The fixture's
// project has no server the new handler could join and its
// file is a library package, so no next step is named.
func TestConfirmTaughtDecl(t *testing.T) {
	taughtFixture(t)
	if handled, out := taughtRun(t, "learn this as a signup form from Jake in jim/main.go"); !handled {
		t.Fatalf("teach line not handled:\n%s", out)
	}
	handled, out := taughtRun(t, "add a signup form named Signup to site/site.go")
	if !handled {
		t.Fatalf("taught use line not handled:\n%s", out)
	}
	// The taught fixture does not chdir, so LinkPath renders
	// long; the sentence's head and the file tail are the facts.
	if !strings.Contains(out, "Done — The signup form Signup is in ") || !strings.Contains(out, "site/site.go.") {
		t.Errorf("taught confirmation missing or wrong:\n%s", out)
	}
	if strings.Contains(out, "will do it") || strings.Contains(out, "will add one") {
		t.Errorf("the taught confirmation promised a step that cannot land:\n%s", out)
	}
}

// Make-and-call's clause names both files — it stands.
func TestConfirmMakeCall(t *testing.T) {
	confirmFixture(t, map[string]string{
		"doc.go":   "package confirm\n",
		"jim/a.go": "package main\n",
		"jim/b.go": "package main\n\nfunc main() {}\n",
	})
	out := confirmRun(t, "make a function named Foo in jim/a.go and call it in jim/b.go")
	if !strings.Contains(out, "Done — Foo now exists in jim/a.go and is called from jim/b.go.") {
		t.Errorf("make-and-call confirmation missing or wrong:\n%s", out)
	}
}

// A follow-up landing (item 2) repeats through the original
// claim, so its confirmation is the family's own — the new
// piece's name and file, nothing replayed.
func TestConfirmFollowUpLanding(t *testing.T) {
	followupProject(t, map[string]string{
		"go.mod":      "module example.com/followup\n\ngo 1.21\n",
		"doc.go":      "package followup\n",
		"jim/main.go": followupMainGo,
	})
	out := vocabRun(t, "add a struct named Profile to jim/main.go")
	if !strings.Contains(out, "Done — The struct Profile is in jim/main.go.") {
		t.Fatalf("the original landing's confirmation missing or wrong:\n%s", out)
	}
	out = vocabRun(t, "another one named Settings")
	if !strings.Contains(out, "Done — The struct Settings is in jim/main.go.") {
		t.Errorf("the follow-up landing's confirmation missing or wrong:\n%s", out)
	}
}

// A delegate with the wrapper named in the sentence: the
// confirmation names the wrapper, its file, and the source
// it calls. A gate-out prints no confirmation.
func TestConfirmDelegate(t *testing.T) {
	dir := delegateChatFixture(t)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		runDelegate("add a function Double in lib/other.go that calls Local", yes)
	})
	// LinkPath renders relative to the process cwd, so the
	// file's path is long in tests; its tail is the fact.
	if !strings.Contains(out, "Done — Double is in ") || !strings.Contains(out, "lib/other.go — it calls Local.") {
		t.Errorf("delegate confirmation missing or wrong:\n%s", out)
	}
	out = captureStdout(t, func() {
		runDelegate("add a function Triple in lib/other.go that calls Local", no)
	})
	if strings.Contains(out, "Done —") || strings.Contains(out, "it calls Local") {
		t.Errorf("a gated-out delegate printed a confirmation:\n%s", out)
	}
}

// The gate is the gate: a declined spine edit and a refused
// wiring both print no confirmation at all.
func TestConfirmGateOutAndRefusalPrintNothing(t *testing.T) {
	root := confirmFixture(t, map[string]string{
		"main.go": confirmPlainMain,
	})
	profile := &UserProfile{}
	out := captureStdout(t, func() { claimChain("add a struct named Profile to main.go", no, profile) })
	if !strings.Contains(out, "Not applied") {
		t.Fatalf("the gate-out did not print its own answer:\n%s", out)
	}
	if strings.Contains(out, "Done —") {
		t.Errorf("a gated-out edit printed a confirmation:\n%s", out)
	}

	// No server anywhere: the wiring planner refuses, and a
	// refusal is not a landing.
	writeFile(t, root, "handlers.go", confirmOneHandler)
	out = confirmRun(t, "connect the Jake handler to the webserver")
	if strings.Contains(out, "Done —") {
		t.Errorf("a refused wiring printed a confirmation:\n%s", out)
	}
}

// Create's shapes: the folder line is a plain confirmation
// now; a webserver starter names the line that starts it; a
// plain file keeps its verification sentence.
func TestConfirmCreateShapes(t *testing.T) {
	confirmFixture(t, map[string]string{
		"doc.go": "package confirm\n",
	})
	out := confirmRun(t, "create a folder named jim")
	if !strings.Contains(out, "Done — created jim/.") {
		t.Errorf("folder confirmation missing or wrong:\n%s", out)
	}
	if strings.Contains(out, "I checked — it's there") {
		t.Errorf("the folder line kept its machinery tail:\n%s", out)
	}

	out = confirmRun(t, "create a file webtest/main.go with a webserver")
	want := `Done — created webtest/ and webtest/main.go. It's the webserver starter and it builds — "run the program webtest/main.go" will start it.`
	if !strings.Contains(out, want) {
		t.Errorf("webserver creation confirmation missing or wrong:\nwant %q in:\n%s", want, out)
	}

	out = confirmRun(t, "create a file notes.txt")
	if !strings.Contains(out, "Done — created notes.txt. I checked — it's there.") {
		t.Errorf("plain file confirmation missing or wrong:\n%s", out)
	}
}

// With no go.mod at the project root there is no build to
// run, and the confirmation says which check stood in for
// it — the one disclosure the old machinery line carried
// that a partner still owes.
func TestConfirmNoGoModDisclosure(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "main.go", confirmPlainMain)
	clarifyPhrasing(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	out := confirmRun(t, "add a struct named Profile to main.go")
	if !strings.Contains(out, "Done — The struct Profile is in main.go.") {
		t.Fatalf("no-go.mod confirmation missing or wrong:\n%s", out)
	}
	if !strings.Contains(out, "I checked by re-reading the project — there's no go.mod at its root, so no build to run.") {
		t.Errorf("the no-build disclosure is missing:\n%s", out)
	}
}
