package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The capability seat (capability.go): the whole line is
// a general capability question, or the seat declines.
// These tests pin both halves, plus the drift deal: every
// example the catalog prints must still parse through the
// claim that serves it.

func TestCapabilityClaimShapes(t *testing.T) {
	for _, line := range []string{
		"what can you do",
		"what can you do?",
		"What can you do?",
		"what can you do for me",
		"what are my options",
		"what are my options?",
		"what are you able to do",
		"what do you do",
		"what can i do",
		"what can i do here",
		"what can i ask you",
		"what should i ask",
		"how do i use you",
		"how do i use this",
		"help",
		"help!",
		"help me",
		"what else can you do",
		// One leading conversational filler is allowed.
		"hey what can you do",
		"hey, what can you do?",
		"ok help",
		"okay, what are my options",
	} {
		if !isCapabilityLine(line) {
			t.Errorf("isCapabilityLine(%q) = false, want claim", line)
		}
	}
}

// The near-miss pins: a line with a task in it, an advice
// shape, or another claim's exact line must decline, so
// its own claim (or the what-now guidance) answers it.
func TestCapabilityDeclines(t *testing.T) {
	for _, line := range []string{
		// Task-shaped: the task keeps its own claim.
		"help me add a server",
		"help me add a server in main.go",
		"help me fix this error",
		"what can you do about the server",
		"what can you do with this file",
		"what can you do to help me",
		"can you add a server in main.go",
		"can you help me",
		"what can i do about this error",
		"what can i do to fix this",
		// Advice shapes: what-now territory, not a catalog.
		"what do i do",
		"what do I do?",
		"what should i do",
		"what should i do now",
		// Other claims' exact lines.
		"help all",
		"full help",
		"teach me",
		"start",
		"next",
		"brains",
		"what do you remember",
		"what have you built here",
		"what's my dog's name",
		"what did i just say",
		// Not questions about capabilities at all.
		"what can you say",
		"what do you know",
		"how do i add a server",
		"do something",
		"",
	} {
		if isCapabilityLine(line) {
			t.Errorf("isCapabilityLine(%q) = true, want decline", line)
		}
	}
}

// The claim prints the catalog once, records the
// exchange, and the rendered answer carries every group
// and every entry's example.
func TestCapabilityClaimPrintsCatalog(t *testing.T) {
	conv := NewConversation()
	var handled bool
	out := captureStdout(t, func() { handled = tryCapabilityAnswer("what can you do?", conv) })
	if !handled {
		t.Fatal("tryCapabilityAnswer declined its own shape")
	}
	plain := stripANSI(out)
	if !strings.Contains(plain, "Here's what I can do.") {
		t.Errorf("answer lost its opening:\n%s", plain)
	}
	for _, g := range capabilityCatalog {
		if !strings.Contains(plain, g.title) {
			t.Errorf("answer lost group %q", g.title)
		}
		for _, e := range g.entries {
			if !strings.Contains(plain, "\""+e.example+"\"") {
				t.Errorf("answer lost example %q", e.example)
			}
		}
	}
	if !strings.Contains(plain, "Try: add a server in main.go") {
		t.Errorf("answer lost its closing invitation:\n%s", plain)
	}
	if !strings.Contains(plain, "\"help all\"") {
		t.Errorf("answer lost the help-all pointer:\n%s", plain)
	}
	if got := len(conv.Turns()); got != 2 {
		t.Errorf("conversation holds %d turns, want 2 (user + reply)", got)
	}
	// A task line through the claim itself declines and
	// prints nothing.
	out = captureStdout(t, func() { handled = tryCapabilityAnswer("help me add a server", conv) })
	if handled || out != "" {
		t.Errorf("task line: handled=%v out=%q, want decline in silence", handled, out)
	}
}

// capabilityExampleGates maps every example the catalog
// prints to the real gate of the claim that serves it:
// the claim's own parse/recognizer, at the level the
// codebase's claim tests work at. A gate returning false
// means the sentence no longer lands — the catalog entry
// must shrink or go.
func capabilityExampleGates(t *testing.T) map[string]func(string) bool {
	t.Helper()
	parse := func(ok func(string) bool) func(string) bool { return ok }
	return map[string]func(string) bool{
		"add a server in main.go": parse(func(s string) bool { _, ok := parseAddServer(s); return ok }),
		"serve index.html in main.go": parse(func(s string) bool {
			_, ok := parseServeHTML(s)
			return ok
		}),
		"add a form handler named Jake to main.go": parse(func(s string) bool {
			req, ok := parseConcept(s)
			return ok && req.entry.ID == "form-handler"
		}),
		"add a struct named Profile to main.go": parse(func(s string) bool {
			req, ok := parseConcept(s)
			return ok && req.entry.ID == "struct"
		}),
		"add a Name field of type string to Profile": parse(func(s string) bool {
			_, ok := parseAssembledEdit(s)
			return ok
		}),
		"create a homepage": parse(func(s string) bool {
			q, ok := parsePageFamily(s)
			return ok && q.kind == "home"
		}),
		"connect the Jake handler to the webserver": parse(func(s string) bool {
			_, ok := parseWireHandler(s)
			return ok
		}),
		"add a database connection function named Connect to db.go": parse(func(s string) bool {
			req, ok := parseConcept(s)
			return ok && req.entry.ID == "database-connect"
		}),
		"create a folder jim": parse(func(s string) bool { _, ok := parseCreate(s); return ok }),
		"rename the function Foo to Bar": parse(func(s string) bool {
			_, ok := parseRenameSentence(s)
			return ok
		}),
		"add a function in app/main.go that calls store.Load": parse(func(s string) bool {
			_, _, _, ok := parseDelegate(s)
			return ok
		}),
		"add instantiation of struct Jim to handler Jake": parse(func(s string) bool {
			_, ok := parseInstantiate(s)
			return ok
		}),
		"what is a goroutine":      parse(func(s string) bool { _, ok := LookupGoKnowledge(s); return ok }),
		"what have you built here": parse(func(s string) bool { return receiptQueryRe.MatchString(s) }),
		"what time is it": parse(func(s string) bool {
			var handled bool
			captureStdout(t, func() { handled = tryTalkStateAnswer(s, NewConversation()) })
			return handled
		}),
		"./main.go:10:2: undefined: Foo": parse(func(s string) bool {
			_, ok := tryErrorHelp(s)
			if !ok {
				return false
			}
			_, _, ok = parseCompilerError(s)
			return ok
		}),
		"my dog is Bella": parse(func(s string) bool {
			// Capture is a hook, not a claim: the gate is
			// the fact grammar itself.
			_, _, ok := parseFactCapture(s)
			return ok
		}),
		"what do you remember": parse(func(s string) bool {
			isolateMemoryStore(t)
			var handled bool
			captureStdout(t, func() { handled = tryMemoryCommand(s, NewConversation()) })
			return handled
		}),
		"learn this as a signup form from Jake in jim/main.go": parse(func(s string) bool {
			_, ok := parseTeachAsk(s)
			return ok
		}),
		"call jim/main.go the api file": parse(func(s string) bool {
			_, ok := parseVocabTeach(s)
			return ok
		}),
		"another one": parse(func(s string) bool {
			// The follow-up claim's gate is its shape; the
			// claim itself then requires a landed piece in
			// the session, which is the claim's own honest
			// precondition, not a catalog promise.
			_, ok := parseFollowUp(s)
			return ok
		}),
		"run the program in main.go": parse(func(s string) bool {
			shape, _ := parseProgramLine(s)
			return shape == shapeRunProgram
		}),
		"run make test": parse(func(s string) bool {
			oldAllow := makeTargetsAllowlist
			makeTargetsAllowlist = map[string]bool{"test": true}
			defer func() { makeTargetsAllowlist = oldAllow }()
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "Makefile"), []byte("test:\n\t@true\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			r := NewLineReader(strings.NewReader("n\n"))
			var handled bool
			captureStdout(t, func() { handled = tryDirectRunMake(s, r, dir, NewConversation()) })
			return handled
		}),
	}
}

// TestCapabilityCatalogExamplesClaimed is the drift test:
// every example in the catalog walks through its owning
// claim's real gate. A catalog entry with no gate mapped
// here fails too — a new promise must bring its proof.
func TestCapabilityCatalogExamplesClaimed(t *testing.T) {
	gates := capabilityExampleGates(t)
	seen := map[string]bool{}
	for _, g := range capabilityCatalog {
		for _, e := range g.entries {
			seen[e.example] = true
			gate, ok := gates[e.example]
			if !ok {
				t.Errorf("catalog example %q has no drift gate — map it in capabilityExampleGates", e.example)
				continue
			}
			if !gate(e.example) {
				t.Errorf("catalog example %q is no longer claimed by its owner — the catalog must shrink", e.example)
			}
		}
	}
	for ex := range gates {
		if !seen[ex] {
			t.Errorf("drift gate %q is not a catalog example anymore — remove it or restore the entry", ex)
		}
	}
}

// The catalog's prose names siblings the one printed
// example doesn't show (the other five page kinds, the
// function and interface rows, the rqlite row, the uses
// and make-and-call forms, the file form of create). Pin
// those wordings too, so the prose cannot outlive the
// machinery it names.
func TestCapabilityCatalogProseSiblingsClaimed(t *testing.T) {
	for line, kind := range map[string]string{
		"create a form page":       "form",
		"create an about page":     "about",
		"create a contact page":    "contact",
		"create an item list page": "list",
		"create a details page":    "details",
	} {
		q, ok := parsePageFamily(line)
		if !ok || q.kind != kind {
			t.Errorf("page prose sibling %q: ok=%v kind=%q, want %q", line, ok, q.kind, kind)
		}
	}
	for line, id := range map[string]string{
		"add a function named Validate to main.go": "function",
		"add an interface named Reader to main.go": "interface",
		"add an rqlite connection to main.go":      "rqlite-connect",
		"add a database connection to main.go":     "database",
	} {
		req, ok := parseConcept(line)
		if !ok || req.entry.ID != id {
			t.Errorf("book prose sibling %q: ok=%v id=%q, want %q", line, ok, req.entry.ID, id)
		}
	}
	if _, _, _, _, _, ok := parseUses("in web.go add a function that uses LoadUser from store.go"); !ok {
		t.Error("uses sibling no longer parses")
	}
	if _, _, _, _, _, ok := parseMakeCall("make a function named Validate in api/validate.go and call it in api/main.go"); !ok {
		t.Error("make-and-call sibling no longer parses")
	}
	if _, ok := parseCreate("create a file jim/main.go"); !ok {
		t.Error("create-file sibling no longer parses")
	}
	// The make-explanation half of the ASK entry's prose:
	// "explain make <target>" is claimed by tryExplainMake
	// for any target spelling (an unknown target gets the
	// claim's own plain answer, never a fall-through).
	{
		r := NewLineReader(strings.NewReader(""))
		var handled bool
		captureStdout(t, func() { handled = tryExplainMake("explain make test", r, ".", NewConversation()) })
		if !handled {
			t.Error("tryExplainMake no longer claims 'explain make test'")
		}
	}
	// The fact entry names job, birthday, and learning
	// beside the dog: their capture shapes hold too.
	for _, line := range []string{"my job is teaching", "my birthday is March 5", "I'm learning Go", "I live in Omaha"} {
		if _, _, ok := parseFactCapture(line); !ok {
			t.Errorf("fact prose sibling %q no longer parses", line)
		}
	}
}
