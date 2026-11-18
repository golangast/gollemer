package chat

import (
	"fmt"
	"strings"
	"testing"
)

// historyfix_test.go — pins for the history-leak fix
// (historyhook.go): a turn a deterministic seat claims must land
// in the session transcript exactly once, in order, and a later
// social turn must be composed with it. The claims here are fakes
// shaped like the real seats (print + return true; some record
// themselves, like the memory/capability/state/fact seats do).

func hookFor(conv *Conversation, line *string) *claimHook {
	return newClaimHook(conv, line)
}

// (a) A claimed turn appears in history exactly once: the hook
// records the user line and a claimed reply derived from the
// seat's own printed output.
func TestHookRecordsClaimedTurnExactlyOnce(t *testing.T) {
	conv := NewConversation()
	line := "what is a defer"
	hook := hookFor(conv, &line)
	claim := func() bool {
		fmt.Printf("%s %s\n", cTag("gollemer [go]>"), "a defer statement runs when the surrounding function returns.")
		return true
	}
	var passthrough string
	out := captureStdout(t, func() {
		if !hook.run(claim) {
			t.Error("hook.run(claim) = false, want true")
		}
	})
	passthrough = out
	if !strings.Contains(passthrough, "a defer statement runs") {
		t.Errorf("the seat's output must still reach stdout live, got %q", passthrough)
	}
	turns := conv.Turns()
	if len(turns) != 2 {
		t.Fatalf("turns = %d, want exactly 2 (one exchange): %+v", len(turns), turns)
	}
	if turns[0].Speaker != "you" || turns[0].Text != line {
		t.Errorf("turn 0 = %+v, want the user's line", turns[0])
	}
	if turns[1].Speaker != "gollemer" || !turns[1].Claimed || turns[1].Domain != ClaimedDomain {
		t.Errorf("turn 1 = %+v, want a claimed hook entry", turns[1])
	}
	if turns[1].Text != "a defer statement runs when the surrounding function returns." {
		t.Errorf("entry text = %q, want the seat's own answer without its tag", turns[1].Text)
	}
	// A second claimed turn lands exactly once more.
	if !hook.run(claim) {
		t.Fatal("second claim declined")
	}
	if len(conv.Turns()) != 4 {
		t.Fatalf("after two claims: %d turns, want 4", len(conv.Turns()))
	}
}

// (a, self-recording seat) A seat that records its own exchange
// (the memory/capability/state/fact pattern) must NOT be
// double-recorded by the hook.
func TestHookDoesNotDoubleRecordSelfRecordingSeat(t *testing.T) {
	conv := NewConversation()
	line := "what do you remember"
	hook := hookFor(conv, &line)
	claim := func() bool {
		fmt.Printf("%s %s\n", cTag("gollemer [talk]>"), "here is what i remember.")
		conv.AddUser(line)
		conv.AddReply("here is what i remember.", TalkDomain, false)
		return true
	}
	captureStdout(t, func() {
		if !hook.run(claim) {
			t.Fatal("claim declined")
		}
	})
	turns := conv.Turns()
	if len(turns) != 2 {
		t.Fatalf("turns = %d, want exactly the seat's own 2: %+v", len(turns), turns)
	}
	if turns[1].Claimed || turns[1].Domain != TalkDomain {
		t.Errorf("the seat's own reply must stand untouched, got %+v", turns[1])
	}
}

// (a, lone reply) A seat that recorded only its reply gets its
// user line inserted before it — the exchange completes, in
// order, with the seat's reply text untouched.
func TestHookCompletesLoneReply(t *testing.T) {
	conv := NewConversation()
	line := "/analyze"
	hook := hookFor(conv, &line)
	claim := func() bool {
		conv.AddReply("the project has 3 packages.", GoAnalyzeDomain, false)
		return true
	}
	captureStdout(t, func() {
		if !hook.run(claim) {
			t.Fatal("claim declined")
		}
	})
	turns := conv.Turns()
	if len(turns) != 2 || turns[0].Speaker != "you" || turns[0].Text != line ||
		turns[1].Speaker != "gollemer" || turns[1].Text != "the project has 3 packages." {
		t.Fatalf("turns = %+v, want [you: /analyze, gollemer: the project…]", turns)
	}
}

// (a, lone user) A seat that recorded only the user line gets
// the claimed reply appended from its own output.
func TestHookCompletesLoneUserTurn(t *testing.T) {
	conv := NewConversation()
	line := "some seat line"
	hook := hookFor(conv, &line)
	claim := func() bool {
		conv.AddUser(line)
		fmt.Printf("%s %s\n", cTag("gollemer [talk]>"), "the seat's answer.")
		return true
	}
	captureStdout(t, func() {
		if !hook.run(claim) {
			t.Fatal("claim declined")
		}
	})
	turns := conv.Turns()
	if len(turns) != 2 || turns[1].Text != "the seat's answer." || !turns[1].Claimed {
		t.Fatalf("turns = %+v, want the user line + a claimed entry with the seat's answer", turns)
	}
}

// (b) An unclaimed line leaves no trace from the hook: the
// conversational tail records it exactly once, as before. The
// claim's printed output still passes through.
func TestHookUnclaimedLeavesNoTrace(t *testing.T) {
	conv := NewConversation()
	line := "just chatting"
	hook := hookFor(conv, &line)
	out := captureStdout(t, func() {
		if hook.run(func() bool {
			fmt.Println("a seat thinking out loud, then declining")
			return false
		}) {
			t.Error("hook.run(declining claim) = true, want false")
		}
	})
	if !strings.Contains(out, "thinking out loud") {
		t.Errorf("a declining claim's output must pass through, got %q", out)
	}
	if conv.Len() != 0 {
		t.Fatalf("conv has %d turns after a declined claim, want 0", conv.Len())
	}
	// The tail's own recording (unchanged path) lands once.
	conv.AddUser(line)
	conv.AddReply("hi there.", SocialDomain, false)
	if conv.Len() != 2 {
		t.Fatalf("tail recording: %d turns, want 2", conv.Len())
	}
}

// (c) A social follow-up after a claimed turn is composed WITH
// the claimed turn present, in the standing format.
func TestClaimedTurnComposesSocialInput(t *testing.T) {
	conv := NewConversation()
	line := "what is a defer"
	hook := hookFor(conv, &line)
	captureStdout(t, func() {
		if !hook.run(func() bool {
			fmt.Printf("%s %s\n", cTag("gollemer [go]>"), "A defer statement runs when the surrounding function returns.")
			return true
		}) {
			t.Fatal("claim declined")
		}
	})
	got := conv.SocialInput("why does that matter")
	want := "before you said what is a defer . before i said a defer statement runs when the surrounding function returns. . now you say why does that matter"
	if got != want {
		t.Errorf("SocialInput after a claimed turn:\n got %q\nwant %q", got, want)
	}
}

// (c, write claim) A follow-up after a landed edit composes with
// the confirmation the seat printed — the landing's own sentence.
func TestClaimedWriteComposesWithConfirmation(t *testing.T) {
	conv := NewConversation()
	line := "add a server in main.go"
	hook := hookFor(conv, &line)
	captureStdout(t, func() {
		if !hook.run(func() bool {
			fmt.Println("plan: add a listener and a handler")
			fmt.Print("write it? [y/n] ")
			fmt.Printf("%s %s\n", cTag("gollemer [composer]>"),
				"Done — added a server in main.go. I built the project and checked that it's really there.")
			return true
		}) {
			t.Fatal("claim declined")
		}
	})
	turns := conv.Turns()
	if len(turns) != 2 {
		t.Fatalf("turns = %+v, want one exchange", turns)
	}
	wantEntry := "Done — added a server in main.go. I built the project and checked that it's really there."
	if turns[1].Text != wantEntry {
		t.Errorf("entry = %q, want the confirmation line %q", turns[1].Text, wantEntry)
	}
	got := conv.SocialInput("why did you put it there")
	want := "before you said add a server in main.go . before i said done — added a server in main.go. i built the project and checked that it's really there. . now you say why did you put it there"
	if got != want {
		t.Errorf("SocialInput after a write claim:\n got %q\nwant %q", got, want)
	}
}

// (d) Ordering is preserved across mixed claimed and unclaimed
// sequences.
func TestHookMixedSequenceOrdering(t *testing.T) {
	conv := NewConversation()
	line := ""
	hook := hookFor(conv, &line)
	claimWith := func(answer string) func() bool {
		return func() bool {
			fmt.Printf("%s %s\n", cTag("gollemer [go]>"), answer)
			return true
		}
	}
	captureStdout(t, func() {
		line = "what is a goroutine"
		if !hook.run(claimWith("a goroutine is a lightweight thread.")) {
			t.Fatal("claim 1 declined")
		}
		// An unclaimed model turn, recorded by the tail as today.
		conv.AddUser("hello")
		conv.AddReply("hey there!", SocialDomain, false)
		line = "what is a channel"
		if !hook.run(claimWith("a channel passes values between goroutines.")) {
			t.Fatal("claim 3 declined")
		}
	})
	turns := conv.Turns()
	wantSpeakers := []string{"you", "gollemer", "you", "gollemer", "you", "gollemer"}
	wantTexts := []string{
		"what is a goroutine", "a goroutine is a lightweight thread.",
		"hello", "hey there!",
		"what is a channel", "a channel passes values between goroutines.",
	}
	if len(turns) != len(wantTexts) {
		t.Fatalf("turns = %d, want %d: %+v", len(turns), len(wantTexts), turns)
	}
	for i := range wantTexts {
		if turns[i].Speaker != wantSpeakers[i] || turns[i].Text != wantTexts[i] {
			t.Errorf("turn %d = %s %q, want %s %q", i, turns[i].Speaker, turns[i].Text, wantSpeakers[i], wantTexts[i])
		}
	}
	if !turns[1].Claimed || turns[3].Claimed || !turns[5].Claimed {
		t.Errorf("claimed flags = %v/%v/%v, want true/false/true",
			turns[1].Claimed, turns[3].Claimed, turns[5].Claimed)
	}
}

// The recorded user line is the line as it stands when the claim
// fires: the unified loop's spoken-target pass rewrites the line
// mid-chain, and the tail would record the rewritten text.
func TestHookRecordsLineAsItStands(t *testing.T) {
	conv := NewConversation()
	line := "add a server in the api file"
	hook := hookFor(conv, &line)
	captureStdout(t, func() {
		if hook.run(func() bool { return false }) {
			t.Fatal("first claim should decline")
		}
		line = "add a server in jim/main.go" // the spoken-target rewrite
		if !hook.run(func() bool {
			fmt.Printf("%s %s\n", cTag("gollemer [composer]>"), "Done — added a server in jim/main.go.")
			return true
		}) {
			t.Fatal("second claim declined")
		}
	})
	turns := conv.Turns()
	if len(turns) != 2 || turns[0].Text != "add a server in jim/main.go" {
		t.Fatalf("turns = %+v, want the rewritten line recorded", turns)
	}
}

// Self-recorded TalkDomain replies (fact, state, capability)
// keep today's behavior: they do NOT compose into the model's
// input — facts stay out of talk context by design.
func TestTalkDomainRepliesDoNotCompose(t *testing.T) {
	conv := NewConversation()
	conv.AddUser("what is my dog's name")
	conv.AddReply("your dog's name is bella.", TalkDomain, false)
	if got := conv.SocialInput("is she a good girl"); got != "is she a good girl" {
		t.Errorf("TalkDomain history must not compose, got %q", got)
	}
}

// claimedEntryText: the compaction rules, case by case.
func TestClaimedEntryText(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			"tagged answer",
			"gollemer [go]> A defer statement runs when the function returns.\n",
			"A defer statement runs when the function returns.",
		},
		{
			"confirmation wins over the plan",
			"plan: add a server\ngollemer [composer]> Done — added a server in main.go. I built the project.\n",
			"Done — added a server in main.go. I built the project.",
		},
		{
			"fixed confirmation",
			"gollemer [composer]> Fixed — the import is added.\n",
			"Fixed — the import is added.",
		},
		{
			"untagged output passes through collapsed",
			"first line\nsecond   line\n",
			"first line second line",
		},
		{
			"gate prompt remnant trimmed",
			"gollemer [composer]> here is the plan, write it? [y/n]",
			"here is the plan, write it?",
		},
		{
			"ansi around the tag",
			"\x1b[36mgollemer [go]>\x1b[0m plain answer.\n",
			"plain answer.",
		},
	}
	for _, tc := range cases {
		if got := claimedEntryText(tc.raw); got != tc.want {
			t.Errorf("%s:\n got %q\nwant %q", tc.name, got, tc.want)
		}
	}
	// The cap: a long answer truncates on a word boundary.
	long := "gollemer [go]> " + strings.Repeat("word ", 100) + "\n"
	got := claimedEntryText(long)
	if len(got) > claimedEntryMax+3 || !strings.HasSuffix(got, "...") {
		t.Errorf("cap: len=%d, suffix=%q; want <= %d chars ending in ...", len(got), got[len(got)-3:], claimedEntryMax+3)
	}
}
