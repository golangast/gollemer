package chat

import (
	"strings"
	"testing"
)

// Created-target compounds (2026-10-31): the phrasing layer's
// second piece lands inside the function the first piece
// creates — "make a handler named Login in web.go and add a
// switch in it". The placement is expressed three ways (the
// created name repeated, a pronoun, silence) and all three
// route to the same two canonical pieces: the classifier
// flags the shape, routeCreatedPiece binds the unbound slot
// deterministically, and each piece executes through its own
// gate against the re-analyzed project — executeEditPlan
// re-reads the project after every applied edit, so piece
// two's planner sees the function piece one just wrote.
// These tests pin the three forms at the route, the shapes
// that must NOT bind (a lone pronoun; a pronoun behind a
// statement or a type), and John's handler + switch landing
// end to end.

func TestPhrasingCreatedTargetRouteForms(t *testing.T) {
	withPhrasingModel(t)
	cases := []struct {
		line string
		want []string
	}{
		// The pronoun form — John's example, named and filed.
		{"make a form handler named Login in web.go and add a switch in it",
			[]string{"add a form handler named Login to web.go",
				"add a switch statement in Login to web.go"}},
		// The created name repeated.
		{"make a function called Bake in main.go and put a for loop in Bake",
			[]string{"add a function named Bake to main.go",
				"add a for loop in Bake to main.go"}},
		// Silence: no placement words on the second piece at all.
		{"make a middleware named Shield in web.go and add a database query",
			[]string{"add a middleware named Shield to web.go",
				"add a database query in Shield to web.go"}},
		// The pronoun binds the function; the file rides along.
		{"add a function named Bake to main.go and add a switch in it to main.go",
			[]string{"add a function named Bake to main.go",
				"add a switch statement in Bake to main.go"}},
		// A print piece inside the created handler.
		{"make a form handler named Login in web.go and add a print line saying \"all done\" in it",
			[]string{"add a form handler named Login to web.go",
				"add a print line saying \"all done\" in Login to web.go"}},
		// "inside it", behind a recursive function.
		{"add a recursive function named Walk to tree.go and add a type switch inside it",
			[]string{"add a recursive function named Walk to tree.go",
				"add a type switch in Walk to tree.go"}},
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

// The shapes that must not bind: without a function-creating
// first piece there is nothing for a pronoun (or a silence)
// to point at, and the layer must leave the line unclaimed —
// a lone pronoun piece most of all.
func TestPhrasingCreatedTargetRefusals(t *testing.T) {
	withPhrasingModel(t)
	for _, line := range []string{
		"add a switch in it",      // the lone pronoun: no creator exists
		"add a switch inside it",  // the same, the other pronoun spelling
		"add a for loop in there", // and the third
		"add a struct named Box to main.go and add a switch in it",         // a type is not a home
		"add an interface named Box to main.go and add a switch",           // silence behind a type
		"add a for loop to main.go and add a switch in it",                 // a statement creates nothing
		"add a switch statement in Login to web.go and add a switch in it", // still nothing created
	} {
		if pieces, ok := PhrasingRoute(line); ok {
			t.Errorf("PhrasingRoute(%q) claimed %q — no function was created for the second piece to land in", line, pieces)
		}
		if runPhrasingLayer(line, yes) {
			t.Errorf("runPhrasingLayer(%q) claimed a line with nothing to bind to", line)
		}
	}
}

// createdFixture: an empty web.go beside a main.go — the
// handler (and everything that lands in it) is created by the
// compound itself.
func createdFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/created\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"ok\")\n}\n")
	writeFile(t, dir, "web.go", "package main\n")
	return dir
}

// John's example end to end: the handler is created by piece
// one and the switch lands INSIDE it by piece two — which
// only works if piece two plans against the project piece one
// left behind.
func TestPhrasingCreatedTargetEndToEnd(t *testing.T) {
	withPhrasingModel(t)
	dir := createdFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	line := "make a form handler named Login in web.go and add a switch in it"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("created-target compound fell through the chain:\n%s", out)
	}
	for _, want := range []string{
		"phrasing layer",
		"add a form handler named Login to web.go",
		"add a switch statement in Login to web.go",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("compound output missing %q:\n%s", want, out)
		}
	}
	got := readBack(t, dir, "web.go")
	loginAt := strings.Index(got, "func Login(w http.ResponseWriter, r *http.Request)")
	switchAt := strings.Index(got, "switch")
	if loginAt < 0 {
		t.Fatalf("the handler was not created in web.go:\n%s", got)
	}
	if switchAt < loginAt {
		t.Errorf("the switch did not land inside the created handler:\n%s", got)
	}
}

// The upstream theft (2026-10-31, John's live session): a
// statement-first compound was captured by the concept claim
// before the phrasing layer ever saw it — parseConcept joined
// the whole tail after "to" into one target
// ("jim/main.go and add a print line in main to jim/main.go")
// and the shared resolver refused the compound as a single
// ask naming a file nobody typed. The class fix: a file
// target in these grammars is ONE path token (the concept
// grammar and parseAddServer read their tails the same way);
// a longer tail is another ask, and the claim declines it.
// John's exact line now falls through to the phrasing layer
// and lands both pieces in jim/main.go's main.
func TestPhrasingCompoundNotStolenUpstream(t *testing.T) {
	withPhrasingModel(t)
	stolen := "add a for loop in main to jim/main.go and add a print line in main to jim/main.go"
	if req, ok := parseConcept(stolen); ok {
		t.Errorf("parseConcept claimed the compound with target %q — a multi-piece line is not one ask", req.target)
	}
	if got, ok := parseAddServer("add a webserver to jim/main.go and add a for loop in main to jim/main.go"); ok {
		t.Errorf("parseAddServer claimed a compound with target %q — the same tail-join capture", got)
	}
	// The single-ask forms keep their claims and targets.
	if req, ok := parseConcept("add a for loop in main to jim/main.go"); !ok || req.target != "jim/main.go" {
		t.Errorf("parseConcept lost the single ask: ok=%v target=%q", ok, req.target)
	}
	if got, ok := parseAddServer("add a webserver to jim/main.go"); !ok || got != "jim/main.go" {
		t.Errorf("parseAddServer lost the single ask: ok=%v target=%q", ok, got)
	}

	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/stolen\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	writeFile(t, dir, "jim/main.go", "package main\n\nfunc main() {\n\tprintln(\"jim\")\n}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(stolen, yes, profile) })
	if !handled {
		t.Fatalf("John's line fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "phrasing layer") {
		t.Errorf("John's line was not answered through the phrasing layer:\n%s", out)
	}
	got := readBack(t, dir, "jim/main.go")
	mainAt := strings.Index(got, "func main()")
	if mainAt < 0 || !strings.Contains(got[mainAt:], "for ") || !strings.Contains(got[mainAt:], "fmt.Println") {
		t.Errorf("both pieces should land inside jim/main.go's main:\n%s", got)
	}
}

// The shared-target compound: the file is stated once, at
// the end, for both pieces — John's second live line. The
// distribution is route work (routeSharedPiece): the piece
// missing its target is retried with the compound's one file
// appended, and only the file distributes — a piece must
// state its own function, or be a declaration, to take it.
func TestPhrasingSharedTargetEndToEnd(t *testing.T) {
	withPhrasingModel(t)
	line := "please add a for loop in main and a print line in main to jim/main.go"
	want := []string{"add a for loop in main to jim/main.go", "add a print line in main to jim/main.go"}
	got, ok := PhrasingRoute(line)
	if !ok {
		t.Fatalf("PhrasingRoute(%q) not claimed", line)
	}
	if strings.Join(got, " ## ") != strings.Join(want, " ## ") {
		t.Fatalf("PhrasingRoute(%q) = %q, want %q", line, got, want)
	}

	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/shared\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	writeFile(t, dir, "jim/main.go", "package main\n\nfunc main() {\n\tprintln(\"jim\")\n}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("shared-target line fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "phrasing layer") {
		t.Errorf("shared-target line was not answered through the phrasing layer:\n%s", out)
	}
	body := readBack(t, dir, "jim/main.go")
	mainAt := strings.Index(body, "func main()")
	if mainAt < 0 || !strings.Contains(body[mainAt:], "for ") || !strings.Contains(body[mainAt:], "fmt.Println") {
		t.Errorf("both pieces should land inside jim/main.go's main:\n%s", body)
	}

	// The restraint: a bare statement piece stating no home
	// does not take the shared file as an "in main" reading.
	if pieces, ok := PhrasingRoute("add a for loop in main to jim/main.go and add a switch"); ok {
		t.Errorf("PhrasingRoute claimed %q — a home-less statement piece must not inherit a function", pieces)
	}
}

// The named-repeat and omitted forms land too: a function and
// its for loop, a session handler and its print line.
func TestPhrasingCreatedTargetNamedAndOmittedLand(t *testing.T) {
	withPhrasingModel(t)
	dir := createdFixture(t)
	withChatSession(t, dir, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("make a function called Bake in main.go and put a for loop in Bake", yes, profile)
	})
	if !handled {
		t.Fatalf("named-repeat compound fell through:\n%s", out)
	}
	got := readBack(t, dir, "main.go")
	bakeAt := strings.Index(got, "func Bake()")
	if bakeAt < 0 {
		t.Fatalf("the function was not created in main.go:\n%s", got)
	}
	if forAt := strings.Index(got[bakeAt:], "for "); forAt < 0 {
		t.Errorf("the for loop did not land inside the created function:\n%s", got)
	}

	out = captureStdout(t, func() {
		handled = claimChain("make a session handler named Gate in web.go and add a print line saying \"all done\"", yes, profile)
	})
	if !handled {
		t.Fatalf("omitted-form compound fell through:\n%s", out)
	}
	got = readBack(t, dir, "web.go")
	gateAt := strings.Index(got, "func Gate(w http.ResponseWriter, r *http.Request)")
	if gateAt < 0 {
		t.Fatalf("the session handler was not created in web.go:\n%s", got)
	}
	if printAt := strings.Index(got[gateAt:], `fmt.Println("all done")`); printAt < 0 {
		t.Errorf("the print line did not land inside the created handler:\n%s", got)
	}
}
