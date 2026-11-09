package chat

// The capability seat: a deterministic answer to "what
// can you do?" — the chat describing its own options from
// its own machinery.
//
// Why this exists: the conversational-feel work gave the
// chat memory (facts, receipts, session context), but when
// a user asked the plain question — "what can you do?" —
// there was no real answer. The beginner help owns a
// three-situations list aimed at understanding a project
// (start / tour / teach me), and in the single-domain
// loops it keeps that answer; but the unified chat's main
// work — building pieces, wiring, pages, remembering —
// was nowhere in it, and the shapes the beginner help does
// not own ("what are my options", "what can you do for
// me") fell through to the social model, which improvised
// a plausible list that could promise anything. A
// capability list a model improvises will drift and lie;
// this seat's list is DATA tied to the claims themselves.
//
// The claim owns GENERAL capability questions only: the
// whole line must be the question (one leading hey/ok is
// allowed, trailing punctuation is not part of the line).
// Anything with a task in it — "help me add a server",
// "what can you do about the server", "what can you do
// with this file" — declines and keeps its own claim, and
// the advice shapes ("what do I do", "what should I do")
// decline too: those belong to the what-now guidance, not
// to a catalog. In the unified loop the claim sits
// directly ahead of the beginner five: the three shapes
// the beginner help also matches ("help", "what can you
// do", "what do you do") get this fuller answer here,
// while the beginner surfaces themselves — "start",
// "teach me", "help all" and its phrase catalog, the tour —
// keep their claims untouched, and the closing line below
// points at them so nothing the help offered is lost.
//
// Drift-proofing is the mechanism that makes the catalog
// honest: every entry carries ONE sayable example, and
// the drift test (capability_test.go) walks each example
// through the real parse gate of the claim that owns it —
// the concept book's matcher, the page family's parser,
// the fact grammar, and so on. If a future change removes
// or renames a piece, its example stops parsing, the test
// fails, and the catalog must shrink with the machinery.
// Nothing unbuilt is listed: no move mechanism, no model
// proposer (an experiment the user never sees promised),
// nothing from the tokenizer work.

import (
	"fmt"
	"strings"
)

// capabilityShapes are the general capability questions,
// normalized (see normalizeCapabilityLine). The set is
// exact on purpose: a shape is owned whole-line or not
// at all.
var capabilityShapes = map[string]bool{
	"what can you do":         true,
	"what can you do for me":  true,
	"what are my options":     true,
	"what are you able to do": true,
	"what do you do":          true,
	"what can i do":           true,
	"what can i do here":      true,
	"what can i ask you":      true,
	"what should i ask":       true,
	"how do i use you":        true,
	"how do i use this":       true,
	"help":                    true,
	"help me":                 true,
	"what else can you do":    true,
}

// normalizeCapabilityLine folds a typed line to the form
// the shape set is written in: lowercase, whitespace
// collapsed, trailing sentence punctuation dropped, and
// one leading conversational filler ("hey", "ok",
// "okay") removed. It never rewrites the middle of the
// line — a task phrase anywhere in it keeps the line out
// of the set.
func normalizeCapabilityLine(line string) string {
	s := strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(line))), " ")
	s = strings.TrimRight(s, ".!?")
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, ' '); i > 0 {
		switch strings.TrimSuffix(s[:i], ",") {
		case "hey", "ok", "okay":
			s = strings.TrimSpace(s[i+1:])
		}
	}
	return s
}

// isCapabilityLine reports whether the whole line is a
// general capability question this seat owns.
func isCapabilityLine(line string) bool {
	return capabilityShapes[normalizeCapabilityLine(line)]
}

// capabilityEntry is one catalog line: what the chat can
// do, in plain words, and ONE sentence that does it —
// the drift test's handle on the entry.
type capabilityEntry struct {
	text    string
	example string
}

// capabilityGroup is one headed block of the answer.
type capabilityGroup struct {
	title   string
	entries []capabilityEntry
}

// capabilityCatalog is the whole promise, and it is a
// promise: every entry names a shipped, dependable
// ability, and every example parses through the claim
// that serves it (pinned by TestCapabilityCatalogExamplesClaimed).
// Wording names only what the serving claim covers — the
// field entry, for one, is the composer's own shape
// ("add a Name field of type string to Profile"); the
// "called"-style field sentence the phrasing layer
// mishandles is deliberately not promised here.
var capabilityCatalog = []capabilityGroup{
	{title: "BUILD — new pieces for your project", entries: []capabilityEntry{
		{"Add a server to a file", "add a server in main.go"},
		{"Serve one of your HTML pages from it", "serve index.html in main.go"},
		{"Put a webpage into an HTML file, from a template you pick (basic, responsive, with assets, two columns, three columns, grid, or rows)", "add a webpage to jim/temp/index.html"},
		{"Serve a folder's pages, styles, scripts, and images from your server at /static/", "serve static files"},
		{"Add a form handler", "add a form handler named Jake to main.go"},
		{"Add a struct, a function, an interface, or another basic piece by name", "add a struct named Profile to main.go"},
		{"Add a field to one of your structs", "add a Name field of type string to Profile"},
		{"Create a whole page with its handler wired up — a form page, a homepage, an about page, a contact page, an item list, or a details page", "create a homepage"},
		{"Wire a handler into your server", "connect the Jake handler to the webserver"},
		{"Add a database connection as its own function, in the file you choose — plain SQL, or rqlite when you ask for it", "add a database connection function named Connect to db.go"},
		{"Create a file or a folder", "create a folder jim"},
	}},
	{title: "CHANGE — edits to what's already there", entries: []capabilityEntry{
		{"Rename a function, a type, a variable, or a field everywhere it appears", "rename the function Foo to Bar"},
		{"Write a function in one file that calls code from another", "add a function in app/main.go that calls store.Load"},
		{"Make a value of one of your structs inside a function", "add instantiation of struct Jim to handler Jake"},
	}},
	{title: "ASK — answers from your project and my notes, never invented", entries: []capabilityEntry{
		{"Tell you what a Go word means, or what a make target does", "what is a goroutine"},
		{"Explain how to do a common Go task", "how do I serve static files"},
		{"Show you what I've built or changed in this project, from the receipts", "what have you built here"},
		{"Answer from the clock and the project state — the time, the branch, what changed", "what time is it"},
		{"Translate a pasted Go error into plain words, and offer the fix when I can work it out", "./main.go:10:2: undefined: Foo"},
	}},
	{title: "REMEMBER — what I keep for you", entries: []capabilityEntry{
		{"Remember facts you mention in passing — your name, your dog, where you live, your job, your favorites, your birthday, what you're learning — and answer them back exactly", "my dog is Bella"},
		{"Show you everything I remember, or forget one thing when you name it", "what do you remember"},
		{"Learn a new kind of piece from one example in your own project, then build it like my built-in kinds", "learn this as a signup form from Jake in jim/main.go"},
		{"Use your own names for your files and folders", "call jim/main.go the api file"},
		{"Repeat the last piece we landed, with a new name or in the other file", "another one"},
	}},
	{title: "RUN — your code, run here, only with your yes", entries: []capabilityEntry{
		{"Run your program and show you what it printed", "run the program in main.go"},
		{"Run one of this project's make targets", "run make test"},
	}},
}

// capabilityAnswer renders the catalog: an opening that
// says what the list is, the groups, and a close that
// hands the beginner surfaces back their pointers and
// ends on one concrete invitation.
func capabilityAnswer() string {
	var b strings.Builder
	b.WriteString("Here's what I can do. Every line below is a real sentence you can say to me, and every change waits behind a plan and your yes:")
	for _, g := range capabilityCatalog {
		b.WriteString("\n\n" + g.title + ":\n")
		for i, e := range g.entries {
			if i > 0 {
				b.WriteString("\n")
			}
			fmt.Fprintf(&b, "  %s: \"%s\"", e.text, e.example)
		}
	}
	b.WriteString("\n\nThat's the whole list — say something outside it and I'll tell you plainly instead of guessing. For the guided walk of a project, say \"start\"; the beginner's full phrase list is \"help all\".")
	b.WriteString("\n\nTry: add a server in main.go")
	return b.String()
}

// tryCapabilityAnswer is the seat's claim in the unified
// loop, seated directly ahead of the beginner five —
// after run-the-program and the terminal commands, whose
// lines are tasks, and before the intent layer and every
// model, which must never improvise this list. It mirrors
// the conversational claims beside it (fact recall, the
// state answers): print once, record the exchange, done.
func tryCapabilityAnswer(line string, conv *Conversation) bool {
	if !isCapabilityLine(line) {
		return false
	}
	out := capabilityAnswer()
	fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", TalkDomain)), out)
	if conv != nil {
		conv.AddUser(line)
		conv.AddReply(out, TalkDomain, false)
	}
	return true
}
