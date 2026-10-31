package chat

import (
	"strings"
	"testing"
)

// The write-sentence boundary (writeboundary.go): the two lines
// from the 2026-10-25 20:11 live screenshot, claimed at the end
// of the chain and answered by the composer — the brain salad
// they originally got ("salutations! even a way, dusty shelf
// kind." / "time, don'T mood book, and an extra early!") is what
// NEVER reaching a brain means here, pinned by handled=true at
// the boundary (the chain's last stop before tryIntentDispatch,
// the only road onward to the brains) and by the answers' words.

// The fall-through evidence, encoded: every line the boundary
// serves with GUIDANCE parses false in every earlier parser —
// the near-misses are real gaps, not refusals duplicated from
// the shapes. (Case is absent on purpose: its under-specified
// forms already parse as assembled edits, so the planner's own
// refusals own them; field-to-a-STRUCT is likewise claimed —
// only field-to-a-FILE falls through.)
func TestWriteBoundaryGuidanceLinesFallThroughEveryShape(t *testing.T) {
	for _, line := range []string{
		"add a function to jim/main.go",
		"add a function in jim/main.go",
		"add a field to jim/main.go",
		"add a parameter to Fetch",
		"add a server to jim/main.go",
		"add print line to jim/main.go",
		"add an interface to jim/main.go",
		"add a method to jim/main.go",
	} {
		if _, ok := parseAssembledEdit(line); ok {
			t.Errorf("parseAssembledEdit(%q) claimed — the boundary must not duplicate a shape", line)
		}
		if _, ok := parseRenameSentence(line); ok {
			t.Errorf("parseRenameSentence(%q) claimed", line)
		}
		if _, _, ok := parseBuildGoal(line); ok {
			t.Errorf("parseBuildGoal(%q) claimed", line)
		}
		if _, ok := parseCreate(line); ok {
			t.Errorf("parseCreate(%q) claimed", line)
		}
		if _, ok := parseBoundary(line); ok {
			t.Errorf("parseBoundary(%q) claimed", line)
		}
		if _, ok := parseAddServer(line); ok {
			t.Errorf("parseAddServer(%q) claimed", line)
		}
		if _, _, _, ok := parseDelegate(line); ok {
			t.Errorf("parseDelegate(%q) claimed", line)
		}
		if _, ok := parseWriteBoundary(line); !ok {
			t.Errorf("parseWriteBoundary(%q) did not claim a proven fall-through", line)
		}
	}
}

// The parse table: kinds and targets for the claimed family.
func TestParseWriteBoundaryClaims(t *testing.T) {
	cases := []struct {
		line string
		kind writeBoundaryKind
	}{
		{"add print line to jim/main.go", wbPrintLine},
		{"add a print line to jim/main.go", wbPrintLine},
		{"put a println in jim/main.go", wbPrintLine},
		{"add a function to jim/main.go", wbFunction},
		{"add a function in jim/main.go", wbFunction},
		{"add a function to jim", wbFunction},
		{"insert a function into jim/main.go", wbFunction},
		{"write a function in jim/main.go", wbFunction},
		{"create a function in jim/main.go", wbFunction},
		{"add a field to jim/main.go", wbField},
		{"add a parameter to Fetch", wbParameter},
		{"add a server to jim/main.go", wbServer},
		{"add an interface to jim/main.go", wbGeneric},
		{"add a method to jim/main.go", wbGeneric},
		{"add a struct to jim/main.go", wbGeneric},
		{"add an import to jim/main.go", wbGeneric},
		{"add a handler to jim/main.go", wbGeneric},
		{"add a statement to jim/main.go", wbGeneric},
		{"add a loop to jim/main.go", wbGeneric},
		{"add frobnicate to jim/main.go", wbGeneric}, // a file target is evidence enough
	}
	for _, c := range cases {
		req, ok := parseWriteBoundary(c.line)
		if !ok {
			t.Errorf("parseWriteBoundary(%q) not claimed", c.line)
			continue
		}
		if req.kind != c.kind {
			t.Errorf("parseWriteBoundary(%q) kind = %q, want %q", c.line, req.kind, c.kind)
		}
	}
	if req, ok := parseWriteBoundary("add a function to jim/main.go"); !ok || req.target != "jim/main.go" || !req.fileTgt {
		t.Errorf("function line target = %+v, %v — want jim/main.go as a file target", req, ok)
	}
	if req, ok := parseWriteBoundary("add an interface to jim/main.go"); !ok || req.object != "an interface" {
		t.Errorf("interface line object = %+v, %v — want the object as typed", req, ok)
	}
}

// The calibration table: lines the boundary must NEVER claim —
// questions, other brains' sentences (generation asks, the
// getter/test/comment family), ordinary chat, and non-writes.
func TestParseWriteBoundaryNonClaims(t *testing.T) {
	for _, line := range []string{
		"how do i add a function to jim/main.go", // a question — never verb-led
		"how do I add a case to a switch",
		"what does add mean",
		"why would i add a field to Config",
		"can you add a function to jim/main.go",
		"write a function that reverses a string", // gocode generation (brain-owned)
		"write a function that sorts ints",
		"write a hello world program in go",
		"add a function that reverses a string", // generation shape, no target
		"create a function for my app",          // create|write need a file target
		"add a getter for Timeout",              // other brains' (edits_test.go)
		"add a test for Describe",
		"add a comment to Describe",
		"add it to the list of things to do", // no code object, no file
		"add me on discord thanks",
		"hello how are you",
		"what does Describe do",
		"tour",
	} {
		if req, ok := parseWriteBoundary(line); ok {
			t.Errorf("parseWriteBoundary(%q) claimed as %+v — that line is not the boundary's", line, req)
		}
		if runWriteBoundary(line) {
			t.Errorf("runWriteBoundary(%q) claimed a non-write line", line)
		}
	}
}

// A boundary-owned function line through the whole
// deterministic chain, in his session state — jim/ created
// through the create feature, the project analyzed first —
// ending at the boundary with the composer's words, writing
// nothing. (His exact lines are no longer boundary cases: "add
// print line to jim/main.go" is served by the concept book, and
// "add a function to jim/main.go" gets the composer's nameless
// guidance — one claim earlier each; their chain coverage lives
// in conceptbook_test.go. The boundary keeps the function asks
// the composer doesn't claim — insert|write verbs like this
// one — and its answer names the same three working forms.)
func TestWriteBoundaryLiveSessionLines(t *testing.T) {
	cases := []struct {
		line string
		want []string
	}{
		{"insert a function into jim/main.go", []string{
			"A function needs a shape I can assemble",
			`"add a function named <Name> to jim/main.go"`,
			`"add a function in jim/main.go that calls <something that exists>"`,
			`"copy <existing function> as <new name>"`,
		}},
	}
	for _, analyzed := range []bool{true, false} {
		for _, c := range cases {
			root := liveSessionRoot(t)
			withChatSession(t, root, analyzed)
			profile := &UserProfile{}
			createJimThroughTheChain(t, profile)
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(c.line, yes, profile) })
			if !handled {
				t.Errorf("analyzed=%v: %q fell through the whole claim chain:\n%s", analyzed, c.line, out)
				continue
			}
			requireNoAnalyzeOutput(t, c.line, out)
			if !strings.Contains(out, "gollemer [composer]>") {
				t.Errorf("analyzed=%v: %q was not answered by the composer:\n%s", analyzed, c.line, out)
			}
			for _, w := range c.want {
				if !strings.Contains(out, w) {
					t.Errorf("analyzed=%v: %q answer does not say %q:\n%s", analyzed, c.line, w, out)
				}
			}
			if got := readBack(t, root, "jim/main.go"); got != "package main\n" {
				t.Errorf("analyzed=%v: %q wrote to jim/main.go — a boundary answer never writes:\n%s", analyzed, c.line, got)
			}
		}
	}
}

// The near-miss answers name the working form, each with the
// user's own target or subject filled in.
func TestWriteBoundaryNearMissAnswers(t *testing.T) {
	cases := []struct {
		line string
		want string
	}{
		{"add a field to jim/main.go", `jim/main.go alone doesn't tell me which struct should carry it`},
		{"add a field to jim/main.go", `"add a Timeout field of type int to Config"`},
		{"insert a function into jim/main.go", "A function needs a shape I can assemble"},
		{"insert a function into jim/main.go", `"add a function named <Name> to jim/main.go"`},
		{"add a parameter to Fetch", `"add a context parameter to <Func>"`},
		{"add a parameter to Fetch", `"add a <type> parameter named <name> to <Func> using <expression>"`},
		{"add a server to jim/main.go", `The piece's name here is webserver: "add a webserver to jim/main.go"`},
		{"add an interface to jim/main.go", "I can't add an interface yet"},
		{"add a method to jim/main.go", "I can't add a method yet"},
		{"add an import to jim/main.go", "I can't add an import yet"},
	}
	for _, c := range cases {
		var handled bool
		out := captureStdout(t, func() { handled = runWriteBoundary(c.line) })
		if !handled {
			t.Errorf("runWriteBoundary(%q) not claimed", c.line)
			continue
		}
		if !strings.Contains(out, c.want) {
			t.Errorf("runWriteBoundary(%q) answer does not say %q:\n%s", c.line, c.want, out)
		}
		if !strings.Contains(out, "gollemer [composer]>") {
			t.Errorf("runWriteBoundary(%q) not composer-tagged:\n%s", c.line, out)
		}
	}
}

// Ordering: every currently servable write sentence still reaches
// its own shape through the chain — the boundary's words appear
// in NONE of their answers, and each is handled before it.
func TestWriteBoundaryOrderingBattery(t *testing.T) {
	boundaryPhrases := []string{"needs a shape I can assemble", "I can't add", "loose statement isn't one of the pieces", "The piece's name here is webserver"}
	check := func(t *testing.T, line, out string, handled bool) {
		t.Helper()
		if !handled {
			t.Errorf("%q fell through the claim chain:\n%s", line, out)
		}
		for _, p := range boundaryPhrases {
			if strings.Contains(out, p) {
				t.Errorf("%q was answered by the boundary (%q) instead of its shape:\n%s", line, p, out)
			}
		}
	}

	// Assembled edits + creations + webserver, on the live-session
	// fixture (create lines and the server fill included).
	t.Run("edits and creations keep their lines", func(t *testing.T) {
		root := liveSessionRoot(t)
		withChatSession(t, root, true)
		profile := &UserProfile{}
		createJimThroughTheChain(t, profile)
		for _, line := range []string{
			"add a webserver to jim/main.go", // fills jim/main.go with the component
			"create a folder named extra",
		} {
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
			check(t, line, out, handled)
		}
	})

	// The assembled-edit family against its own fixture, gated
	// "no" so nothing is written: each is claimed by its shape
	// (its refusal or its gate), never by the boundary.
	t.Run("assembled family keeps its lines", func(t *testing.T) {
		dir := editsChatFixture(t)
		withEditsProject(t, dir)
		profile := &UserProfile{}
		for _, line := range []string{
			"add a plum case to Describe",
			"add a Timeout field of type int to Config",
			"add a function like Greet named GreetLoudly",
			"add a context parameter to StepOK",
			"register thanks in Greetings",
			"rename Describe to Explain",
		} {
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(line, no, profile) })
			check(t, line, out, handled)
		}
	})

	// The delegate against its fixture: the plan shows, gated no.
	t.Run("delegate keeps its line", func(t *testing.T) {
		dir := delegateChatFixture(t)
		withServerProject(t, dir)
		profile := &UserProfile{}
		line := "add a function in app/main.go that calls Load"
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, no, profile) })
		check(t, line, out, handled)
		if !strings.Contains(out, "delegate I assembled") {
			t.Errorf("delegate line did not reach the delegate flow:\n%s", out)
		}
	})
}

// The chain leaves questions and ordinary lines untouched: none
// of these is claimed by ANY deterministic claim, the boundary
// included — they fall to the intent layer and the brains
// exactly as before.
func TestWriteBoundaryChainLeavesQuestionsAlone(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	for _, line := range []string{
		"how do i add a function to jim/main.go",
		"how do I add a case to a switch",
		"add it to the list of things to do",
		"add a getter for Timeout",
		"add a test for Describe",
		"add a comment to Describe",
		"write a function that reverses a string",
		"add a function that reverses a string",
	} {
		var handled bool
		captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if handled {
			t.Errorf("claimChain(%q) claimed a line that belongs past the deterministic chain", line)
		}
	}
}

// The named-clause claims (2026-10-30): a name slot before a
// "that …" clause makes the line a declaration being specified,
// claimed even with no target — John's live line and its kin.
// Each is a real fall-through: every earlier parser refuses it
// (the concept grammar's slot zone can't spend the clause, the
// delegate wants "that calls", and so on), so the boundary is
// the only deterministic answer it can get.
func TestParseWriteBoundaryNamedClauseClaims(t *testing.T) {
	cases := []struct {
		line    string
		kind    writeBoundaryKind
		name    string
		returns bool
	}{
		{"add a function named jake that returns a string", wbFunction, "jake", true},
		{"add a function name is jake that returns a string", wbFunction, "jake", true},
		{"add a function named jake that sorts numbers", wbFunction, "jake", false},
		{"add a struct named Config that holds settings", wbGeneric, "Config", false},
		{"add an interface named Reader that reads bytes", wbGeneric, "Reader", false},
		{"put a struct name is Config that holds settings", wbGeneric, "Config", false},
	}
	for _, c := range cases {
		if _, ok := parseAssembledEdit(c.line); ok {
			t.Errorf("parseAssembledEdit(%q) claimed — not a boundary fall-through", c.line)
		}
		if _, ok := parseRenameSentence(c.line); ok {
			t.Errorf("parseRenameSentence(%q) claimed", c.line)
		}
		if _, _, ok := parseBuildGoal(c.line); ok {
			t.Errorf("parseBuildGoal(%q) claimed", c.line)
		}
		if _, ok := parseCreate(c.line); ok {
			t.Errorf("parseCreate(%q) claimed", c.line)
		}
		if _, ok := parseBoundary(c.line); ok {
			t.Errorf("parseBoundary(%q) claimed", c.line)
		}
		if _, ok := parseAddServer(c.line); ok {
			t.Errorf("parseAddServer(%q) claimed", c.line)
		}
		if _, _, _, ok := parseDelegate(c.line); ok {
			t.Errorf("parseDelegate(%q) claimed", c.line)
		}
		if _, ok := parseConcept(c.line); ok {
			t.Errorf("parseConcept(%q) claimed — the concept book owns it, not the boundary", c.line)
		}
		req, ok := parseWriteBoundary(c.line)
		if !ok {
			t.Errorf("parseWriteBoundary(%q) did not claim a named-clause line", c.line)
			continue
		}
		if req.kind != c.kind {
			t.Errorf("parseWriteBoundary(%q) kind = %q, want %q", c.line, req.kind, c.kind)
		}
		if req.declName != c.name {
			t.Errorf("parseWriteBoundary(%q) declName = %q, want %q", c.line, req.declName, c.name)
		}
		if req.returnsClause != c.returns {
			t.Errorf("parseWriteBoundary(%q) returnsClause = %v, want %v", c.line, req.returnsClause, c.returns)
		}
		if req.hasTarget || req.target != "" {
			t.Errorf("parseWriteBoundary(%q) target = %q, %v — want none", c.line, req.target, req.hasTarget)
		}
	}
}

// The named-clause answers: the working scaffold sentence with
// the user's own name filled in, and — for a returns clause —
// the plain returns truth (not assembled yet; the scaffold
// comes with no return; the body is the user's to fill in).
func TestWriteBoundaryNamedClauseAnswers(t *testing.T) {
	cases := []struct {
		line    string
		want    []string
		notWant []string
	}{
		{"add a function named jake that returns a string", []string{
			`"add a function named jake to main.go"`,
			"return types aren't one of the pieces I can assemble",
			"the scaffold comes with no return and the body is yours to fill in",
		}, nil},
		{"add a function named jake that sorts numbers", []string{
			`"add a function named jake to main.go"`,
			"isn't one of the pieces I can assemble",
		}, []string{"return type"}},
		{"add a struct named Config that holds settings", []string{
			"I can't add a struct as asked",
			`"add a struct named Config to main.go"`,
		}, nil},
		{"add an interface named Reader that reads bytes", []string{
			`"add an interface named Reader to main.go"`,
		}, nil},
	}
	for _, c := range cases {
		var handled bool
		out := captureStdout(t, func() { handled = runWriteBoundary(c.line) })
		if !handled {
			t.Errorf("runWriteBoundary(%q) not claimed", c.line)
			continue
		}
		if !strings.Contains(out, "gollemer [composer]>") {
			t.Errorf("runWriteBoundary(%q) not composer-tagged:\n%s", c.line, out)
		}
		for _, w := range c.want {
			if !strings.Contains(out, w) {
				t.Errorf("runWriteBoundary(%q) answer does not say %q:\n%s", c.line, w, out)
			}
		}
		for _, w := range c.notWant {
			if strings.Contains(out, w) {
				t.Errorf("runWriteBoundary(%q) answer says %q — the clause was not a returns clause:\n%s", c.line, w, out)
			}
		}
	}
}

// The calibration holds around the new claim: a that-clause
// line WITHOUT a name slot is still generation's, a name inside
// the clause doesn't count, and the create|write file guard is
// unchanged (the name slot didn't widen the verbs).
func TestParseWriteBoundaryNamedNonClaims(t *testing.T) {
	for _, line := range []string{
		"add a function that sorts numbers",                 // generation shape, no name slot
		"add a function that returns a struct named Point",  // the name is the clause's own
		"write a function named jake that returns a string", // write still needs a file target
		"create a function named jake that returns a string",
	} {
		if req, ok := parseWriteBoundary(line); ok {
			t.Errorf("parseWriteBoundary(%q) claimed as %+v — that line is not the boundary's", line, req)
		}
		if runWriteBoundary(line) {
			t.Errorf("runWriteBoundary(%q) claimed a line past the calibration", line)
		}
	}
}

// John's exact live line through the whole deterministic chain:
// the boundary answers it — composer tag, his own name in the
// scaffold sentence, the returns note — and no brain ever sees
// it (handled=true at the chain's last deterministic stop; the
// salad it originally got, "salutations! …", is the marker of
// the road not taken). The concept-served twin line, same name
// WITH a file, still belongs to the concept book: gated "no",
// none of the boundary's words in its answer.
func TestWriteBoundaryNamedClauseLiveLine(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)

	line := "add a function named jake that returns a string"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("%q fell through the whole claim chain:\n%s", line, out)
	}
	if !strings.Contains(out, "gollemer [composer]>") {
		t.Errorf("%q was not answered by the composer:\n%s", line, out)
	}
	for _, w := range []string{
		`"add a function named jake to main.go"`,
		"return types aren't one of the pieces I can assemble",
		"the body is yours to fill in",
	} {
		if !strings.Contains(out, w) {
			t.Errorf("%q answer does not say %q:\n%s", line, w, out)
		}
	}
	if strings.Contains(out, "salutations") {
		t.Errorf("%q reached a brain — the salad it got live:\n%s", line, out)
	}

	served := "add a function named jake to jim/main.go"
	out = captureStdout(t, func() { handled = claimChain(served, no, profile) })
	if !handled {
		t.Fatalf("%q fell through the whole claim chain:\n%s", served, out)
	}
	for _, p := range []string{"needs a shape I can assemble", "I can't add", "return types aren't"} {
		if strings.Contains(out, p) {
			t.Errorf("%q was answered by the boundary (%q) instead of the concept book:\n%s", served, p, out)
		}
	}
	if got := readBack(t, root, "jim/main.go"); got != "package main\n" {
		t.Errorf("gated-no concept line wrote to jim/main.go:\n%s", got)
	}
}
