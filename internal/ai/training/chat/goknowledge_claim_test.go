package chat

import (
	"path/filepath"
	"strings"
	"testing"
)

// The ahead-of-the-gate claim (strongGoKnowledgeAnswer): pins for
// both directions. The claim exists because the fb1f988 router
// flips conversationally-wrapped Go-concept questions to social;
// the pins exist because the RAW LookupGoKnowledge gate fires on
// purely social lines (measured 2026-11-05 — see the rawHit
// column, asserted below so the reason for the tightening stays
// executable, not folklore).

func TestConceptQuestionSubject(t *testing.T) {
	cases := []struct {
		line string
		want string
		ok   bool
	}{
		{"explain it to me simply: what is a channel?", "channel", true},
		{"what is a worker pool, simply?", "worker pool", true},
		{"tell me simply, what is a waitgroup?", "waitgroup", true},
		{"what is the context package, in simple terms?", "context package", true},
		{"in plain words, what is a mutex?", "mutex", true},
		{"simply put, what are channels for?", "channels", true},
		{"what is a channel?", "channel", true},
		{"what is a channel in go?", "channel", true},
		// Not concept questions: no nucleus, whatever the vocabulary.
		{"i changed the channel last night", "", false},
		{"i can't wait for the weekend", "", false},
		{"can we switch the meeting to friday?", "", false},
		{"did you capture that moment?", "", false},
		{"hello", "", false},
	}
	for _, c := range cases {
		got, ok := conceptQuestionSubject(c.line)
		if ok != c.ok || got != c.want {
			t.Errorf("conceptQuestionSubject(%q) = %q, %v; want %q, %v", c.line, got, ok, c.want, c.ok)
		}
	}
}

func TestStrongGoKnowledgeAnswerHits(t *testing.T) {
	cases := []struct {
		line string
		want string // substring of the curated body
	}{
		// The four lines the retrained router flipped to social.
		{"explain it to me simply: what is a channel?", "Channels move values"},
		{"what is a worker pool, simply?", "bounds concurrency"},
		{"tell me simply, what is a waitgroup?", "sync.WaitGroup waits"},
		{"what is the context package, in simple terms?", "context.Context carries cancellation"},
		// The two general-Go probes that never flipped.
		{"in plain words, what is a mutex?", "sync.Mutex guards"},
		{"simply put, what are channels for?", "Channels move values"},
		// Bare and lightly-wrapped forms of the same family.
		{"what is a channel?", "Channels move values"},
		{"what is golang?", "compiled programming language"},
		{"what is a sync mutex?", "sync.Mutex guards"},
		{"what is a goroutine?", "lightweight thread"},
	}
	for _, c := range cases {
		body, ok := strongGoKnowledgeAnswer(c.line)
		if !ok {
			t.Errorf("strongGoKnowledgeAnswer(%q) declined, want claim", c.line)
			continue
		}
		if !strings.Contains(body, c.want) {
			t.Errorf("strongGoKnowledgeAnswer(%q) = %q, want substring %q", c.line, body, c.want)
		}
	}
}

func TestStrongGoKnowledgeAnswerDeclinesSocial(t *testing.T) {
	// rawHit records whether the RAW LookupGoKnowledge gate fires
	// on the full line. Every rawHit=true line here is a theft the
	// ungated consult would have committed; the strong claim must
	// decline all of them anyway.
	cases := []struct {
		line   string
		rawHit bool
	}{
		// The task's named families: social uses of channel / go /
		// wait.
		{"i changed the channel last night", true},
		{"the movie is on channel 4 tonight", true},
		{"that quote is out of context", true},
		{"i can't wait for the weekend", true},
		{"please wait for me by the door", true},
		{"the wait at the restaurant was so long", true},
		{"let me go get the groceries", true},
		{"i have to go run to the store", true},
		// The phrase-prefix trap: "what is go" ⊂ "what is going".
		{"what is going on?", true},
		// Single distinctive words in social statements.
		{"i am sorting my photos from the trip", true},
		{"the box contains all my old photos", true},
		{"the characters in this book are wonderful", true},
		{"we had a big argument last night", true},
		{"thanks for the pointers, that helped a lot", true},
		{"i keep locking myself out of the house", true},
		{"i hope you recover soon", true},
		{"the fields were green and the sky was clear", true},
		{"for the record, i did call her back", true},
		{"there is constant noise from the street", true},
		{"a piece of string and some tape", true},
		{"please select a date for the party", true},
		{"we are testing the recipe tonight", true},
		{"the formatting of my resume is off", true},
		{"i am printing photos for the wall", true},
		{"i keep logging into the wrong account", true},
		{"she received an inheritance from her aunt", true},
		{"i need closure on this", true},
		{"the cleanup took all day saturday", true},
		{"that's so generic, try again", true},
		{"we should implement the plan tomorrow", true},
		{"my method is simple but it works", true},
		{"i looked it up in the dictionary", true},
		// The >=2-keyword-hit tier on ordinary mail talk.
		{"i will send it when i receive the photos", true},
		// Questions, but not concept questions the base explains:
		// no what-is nucleus, or a subject the base's winning entry
		// cannot explain word for word (the coverage audit).
		{"can we switch the meeting to friday?", true},
		{"did you capture that moment?", true},
		{"what is the point of all this waiting?", true},
		{"what is a string quartet?", true},
		{"what is a function of the heart?", true},
		// Genuine what-is questions about non-Go subjects.
		{"what is love?", false},
		{"what's your name?", false},
		{"what is a good gift for my mom?", false},
		{"what is the weather like today?", false},
		// The talk probes' own social class.
		{"hello", false},
		{"how are you?", false},
		{"tell me a joke", false},
		{"good morning", false},
		{"i'm bored", false},
		{"thanks!", false},
		{"are you a robot?", false},
	}
	for _, c := range cases {
		if _, raw := LookupGoKnowledge(c.line); raw != c.rawHit {
			t.Errorf("LookupGoKnowledge(%q) raw hit = %v, want %v (the recorded gate measurement moved — re-audit the claim)", c.line, raw, c.rawHit)
		}
		if body, ok := strongGoKnowledgeAnswer(c.line); ok {
			t.Errorf("strongGoKnowledgeAnswer(%q) claimed %q, want decline (social line must fall through)", c.line, body)
		}
	}
}

// The intent layer's guard: with the SHIPPED classifier live, the
// one flipped line it steals ("tell me simply, what is a
// waitgroup?" -> what_now at confidence 1.000, measured) must fall
// through tryIntentDispatch untouched, while a genuine what-now
// line still dispatches — the guard closes the theft without
// closing the intent layer's own territory.
func TestIntentLayerLeavesStrongConceptQuestions(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	oldRoot := unifiedProjectRoot
	unifiedProjectRoot = repoRoot
	defer func() { unifiedProjectRoot = oldRoot }()

	m := defaultIntentModel()
	if m == nil {
		t.Skip("intent model unavailable in this test process (its lazy loader was already consumed without a data root); the guard is pinned by the isolated leg and the live smoke")
	}
	// The theft is real, not hypothetical: assert the shipped
	// classifier's verdict so a future model that stops misfiring
	// makes this test say so instead of silently going vacuous.
	if intent, _, ok := m.ClassifyIntent("tell me simply, what is a waitgroup?"); !ok || intent != "what_now" {
		t.Fatalf("shipped classifier verdict for the waitgroup line = %q, ok=%v; want the recorded what_now theft", intent, ok)
	}
	for _, line := range []string{
		"tell me simply, what is a waitgroup?",
		"explain it to me simply: what is a channel?",
		"what is a worker pool, simply?",
		"what is the context package, in simple terms?",
	} {
		if tryIntentDispatch(line, nil) {
			t.Errorf("intent layer claimed %q, want fall-through to the deterministic [go] seat", line)
		}
	}
	// The intent layer's own territory still dispatches.
	var handled bool
	out := captureStdout(t, func() { handled = tryIntentDispatch("im stuck", nil) })
	if !handled {
		t.Errorf("intent layer declined %q, want the what_now dispatch to survive the guard", "im stuck")
	}
	if !strings.Contains(out, "Nothing's open yet") && !strings.Contains(out, "tour") {
		t.Errorf("what_now dispatch for %q printed %q, want the beginner guidance", "im stuck", out)
	}
}

// The claim through the real seat: tryDeterministicAnswer with the
// route the NEW router actually returns for the flipped lines
// (social) must claim them with the deterministic [go] answer, and
// must leave purely social lines unhandled for the social arm.
func TestStrongClaimThroughDeterministicSeat(t *testing.T) {
	dir := t.TempDir()
	conv := NewConversation()
	reader := NewLineReader(strings.NewReader(""))

	flips := []struct{ line, want string }{
		{"explain it to me simply: what is a channel?", "Channels move values"},
		{"what is a worker pool, simply?", "bounds concurrency"},
		{"tell me simply, what is a waitgroup?", "sync.WaitGroup waits"},
		{"what is the context package, in simple terms?", "context.Context carries cancellation"},
	}
	for _, f := range flips {
		var handled bool
		out := captureStdout(t, func() {
			handled = tryDeterministicAnswer(f.line, SocialDomain, conv, reader, dir)
		})
		if !handled {
			t.Errorf("tryDeterministicAnswer(%q, social) not handled, want the [go] claim", f.line)
			continue
		}
		if !strings.Contains(out, "[go]") || !strings.Contains(out, f.want) {
			t.Errorf("tryDeterministicAnswer(%q, social) printed %q, want [go] answer containing %q", f.line, out, f.want)
		}
	}

	for _, line := range []string{
		"i changed the channel last night",
		"i can't wait for the weekend",
		"let me go get the groceries",
		"what is going on?",
		"hello",
	} {
		var handled bool
		captureStdout(t, func() {
			handled = tryDeterministicAnswer(line, SocialDomain, conv, reader, dir)
		})
		if handled {
			t.Errorf("tryDeterministicAnswer(%q, social) handled a purely social line, want fall-through", line)
		}
	}
}
