package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The fact-memory seat (factmemory.go): deterministic
// capture of stated facts into the long-term store, exact
// recall from the store, and management through the
// existing memory commands. All tests isolate the store
// under a temp home (memory_state_test.go's seam).

func TestFactCaptureShapes(t *testing.T) {
	for _, tc := range []struct {
		line, key, value, answer string
	}{
		{"my name is John", "name", "John", "Your name is John."},
		{"my name is Mary Jane", "name", "Mary Jane", "Your name is Mary Jane."},
		{"my dog is Bella", "pet:dog", "Bella", "Your dog's name is Bella."},
		{"my dog's name is Bella", "pet:dog", "Bella", "Your dog's name is Bella."},
		{"my dog is called Bella", "pet:dog", "Bella", "Your dog's name is Bella."},
		{"my cat is named Milo", "pet:cat", "Milo", "Your cat's name is Milo."},
		{"my pet is called Biscuit", "pet:pet", "Biscuit", "Your pet's name is Biscuit."},
		{"my wife's name is Ana", "family:wife", "Ana", "Your wife's name is Ana."},
		{"my husband is called Tom", "family:husband", "Tom", "Your husband's name is Tom."},
		{"my son's name is Eli", "family:son", "Eli", "Your son's name is Eli."},
		{"my friend's name is Sam", "family:friend", "Sam", "Your friend's name is Sam."},
		{"I live in Omaha", "city", "Omaha", "You live in Omaha."},
		{"i live in crestwood", "city", "crestwood", "You live in crestwood."},
		{"I live in New York", "city", "New York", "You live in New York."},
		{"I work as a nurse", "job", "a nurse", "You work as a nurse."},
		{"my job is teaching", "job", "teaching", "Your job is teaching."},
		{"my favorite color is forest green", "favorite:color", "forest green", "Your favorite color is forest green."},
		{"my favorite food is pizza", "favorite:food", "pizza", "Your favorite food is pizza."},
		{"my birthday is March 5", "birthday", "March 5", "Your birthday is March 5."},
		{"my birthday is in June", "birthday", "in June", "Your birthday is in June."},
		{"my birthday is 5/12", "birthday", "5/12", "Your birthday is 5/12."},
		{"I'm learning Go", "learning", "Go", "You're learning Go."},
		{"i am learning the piano", "learning", "the piano", "You're learning the piano."},
		{"actually my dog is Rex", "pet:dog", "Rex", "Your dog's name is Rex."},
	} {
		key, f, ok := parseFactCapture(tc.line)
		if !ok {
			t.Errorf("parseFactCapture(%q) declined, want key %q", tc.line, tc.key)
			continue
		}
		if key != tc.key || f.Value != tc.value || f.Answer != tc.answer {
			t.Errorf("parseFactCapture(%q) = %q, %+v — want key %q, value %q, answer %q",
				tc.line, key, f, tc.key, tc.value, tc.answer)
		}
	}
}

// The near-miss pins are the heart of the seat: a statement
// the grammar cannot read as a fact stores NOTHING and the
// hook stays silent, so the line is plain conversation.
func TestFactCaptureNearMissesStoreNothing(t *testing.T) {
	isolateMemoryStore(t)
	for _, line := range []string{
		"my dog is tired",
		"my dog is very sleepy",
		"my dog is a good boy",
		"my dog is happy today", // extra word + state word
		"my cat is sick",
		"I live in interesting times",
		"I live in a small town",
		"I live in the woods",
		"i live in new york", // multi-word place, lowercase as typed (v1 declines)
		"my favorite thing about her is her laugh",
		"my favorite food is pizza is great",
		"my job is hard",
		"my job is stressful",
		"my birthday is tomorrow",
		"i'm learning to let go",
		"my wife is a doctor", // family bare-is is not a name shape
		"my wife is tired",
		"her name is bella", // third person: conversation, not a capture
		"his name is rex",
		"my name is legion for we are many",
		"my cat is called",              // no value
		"what's my dog's name",          // a question, never a capture
		"I prefer bananas while coding", // the pre-existing ordinary-line pin
		"hello there",
		"i am sad today",
	} {
		if _, _, ok := parseFactCapture(line); ok {
			t.Errorf("parseFactCapture(%q) claimed a fact, want decline", line)
		}
		if note := captureFactStatement(line); note != "" {
			t.Errorf("captureFactStatement(%q) = %q, want silence", line, note)
		}
	}
	d, err := loadLongTermMemory()
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Facts) != 0 {
		t.Errorf("near-miss lines stored facts: %+v", d.Facts)
	}
	if len(d.Preferences) != 0 || len(d.Projects) != 0 {
		t.Errorf("near-miss lines touched the rest of the store: %+v", d)
	}
}

func TestFactCaptureDisclosureAndRecall(t *testing.T) {
	isolateMemoryStore(t)
	if note := captureFactStatement("my dog is Bella"); note != "I'll remember: your dog's name is Bella." {
		t.Errorf("disclosure = %q", note)
	}
	if note := captureFactStatement("I live in Omaha"); note != "I'll remember: you live in Omaha." {
		t.Errorf("disclosure = %q", note)
	}
	if note := captureFactStatement("my favorite color is forest green"); note != "I'll remember: your favorite color is forest green." {
		t.Errorf("disclosure = %q", note)
	}
	for _, tc := range []struct{ line, want string }{
		{"what's my dog's name?", "Your dog's name is Bella."},
		{"what is my dog called", "Your dog's name is Bella."},
		{"where do I live?", "You live in Omaha."},
		{"what's my favorite color?", "Your favorite color is forest green."},
		{"what's my name?", "You haven't told me your name yet."},
		{"when is my birthday?", "You haven't told me your birthday yet."},
		{"what do I do for work?", "You haven't told me what you do for work yet."},
		{"who is my wife?", "You haven't told me your wife's name yet."},
		{"what am I learning?", "You haven't told me what you're learning yet."},
	} {
		ans, ok := factRecallAnswer(tc.line)
		if !ok || ans != tc.want {
			t.Errorf("factRecallAnswer(%q) = %q, %v — want %q, true", tc.line, ans, ok, tc.want)
		}
	}
	// Distinct keys coexist: dog + cat + two favorites.
	captureFactStatement("my cat is called Milo")
	captureFactStatement("my favorite food is pizza")
	for _, tc := range []struct{ line, want string }{
		{"what's my cat's name?", "Your cat's name is Milo."},
		{"what's my favorite food?", "Your favorite food is pizza."},
		{"what's my dog's name?", "Your dog's name is Bella."},
	} {
		ans, ok := factRecallAnswer(tc.line)
		if !ok || ans != tc.want {
			t.Errorf("factRecallAnswer(%q) = %q, %v — want %q, true", tc.line, ans, ok, tc.want)
		}
	}
}

// Recall declines narrowly: a line that merely resembles a
// fact shape keeps falling through, untouched.
func TestFactRecallDeclines(t *testing.T) {
	for _, line := range []string{
		"what's my dog dreaming about",
		"what's my dog's name and age",
		"where do I live-stream",
		"where do I live stream",
		"what is the capital of france",
		"what's your name?",
		"what is your name",
		"what do I do",
		"what have you built here",
		"what do you remember",
		"who is my neighbor",
		"hello",
	} {
		if ans, ok := factRecallAnswer(line); ok {
			t.Errorf("factRecallAnswer(%q) claimed %q, want decline", line, ans)
		}
	}
}

func TestFactCorrectionOverwrites(t *testing.T) {
	isolateMemoryStore(t)
	captureFactStatement("my dog is Bella")
	note := captureFactStatement("actually my dog is Rex")
	if note != "I'll remember: your dog's name is Rex (updated from Bella)." {
		t.Errorf("correction disclosure = %q", note)
	}
	ans, ok := factRecallAnswer("what's my dog's name?")
	if !ok || ans != "Your dog's name is Rex." {
		t.Errorf("recall after correction = %q, %v", ans, ok)
	}
	// A same-value restatement is a plain disclosure, no update note.
	note = captureFactStatement("my dog is Rex")
	if note != "I'll remember: your dog's name is Rex." {
		t.Errorf("restatement disclosure = %q", note)
	}
}

// The loop-level claim: seated, prints under the talk tag,
// and records the exchange like the state arm does.
func TestFactRecallClaimAndSeatArm(t *testing.T) {
	isolateMemoryStore(t)
	root := talkFixtureProject(t)
	withChatSession(t, root, true)
	conv := NewConversation()
	captureFactStatement("my wife's name is Ana")

	var handled bool
	out := captureStdout(t, func() { handled = tryFactRecallAnswer("who is my wife?", conv) })
	if !handled || !strings.Contains(out, "gollemer [talk]> Your wife's name is Ana.") {
		t.Errorf("loop claim did not answer from the store:\n%s", out)
	}
	out = captureStdout(t, func() { handled = tryFactRecallAnswer("what's my dog's name?", conv) })
	if !handled || !strings.Contains(out, "You haven't told me your dog's name yet.") {
		t.Errorf("loop claim did not answer honestly for an unstored fact:\n%s", out)
	}

	// The seat's own arm answers ahead of the social arm
	// (this responder has no social handle loaded, so only
	// a deterministic arm can answer at all).
	installTalkResponder(t, &localTalkResponder{})
	handled, out = talkAskSeat(t, NewConversation(), "who is my wife?", false)
	if !handled || !strings.Contains(out, "Your wife's name is Ana.") {
		t.Errorf("seat arm did not answer the fact question:\n%s", out)
	}
}

// Management: the recall listing grows the facts section,
// forget shapes reach facts through the shared machinery,
// project forget leaves global facts alone, forget
// everything takes them.
func TestFactManagementThroughMemoryCommands(t *testing.T) {
	isolateMemoryStore(t)
	root := talkFixtureProject(t)
	withChatSession(t, root, true)
	conv := NewConversation()
	captureFactStatement("my dog is Bella")
	captureFactStatement("my birthday is March 5")
	if handled, out := memoryAsk(t, conv, "remember: I prefer tabs over spaces"); !handled {
		t.Fatalf("preference capture failed:\n%s", out)
	}

	handled, out := memoryAsk(t, conv, "what do you remember")
	if !handled || !strings.Contains(out, "Facts about you:") ||
		!strings.Contains(out, "your dog's name: Bella") ||
		!strings.Contains(out, "your birthday: March 5") ||
		!strings.Contains(out, "I prefer tabs over spaces") {
		t.Errorf("recall listing lacks the facts section:\n%s", out)
	}

	handled, out = memoryAsk(t, conv, "forget my dog's name")
	if !handled || !strings.Contains(out, "Forgotten: your dog's name (was Bella).") {
		t.Errorf("fact forget did not confirm the exact removal:\n%s", out)
	}
	if ans, _ := factRecallAnswer("what's my dog's name?"); ans != "You haven't told me your dog's name yet." {
		t.Errorf("forgotten fact still answers: %q", ans)
	}
	d, _ := loadLongTermMemory()
	if len(d.Facts) != 1 || len(d.Preferences) != 1 {
		t.Fatalf("fact forget touched other store content: %+v", d)
	}

	handled, out = memoryAsk(t, conv, "forget this project")
	if !handled || !strings.Contains(out, "your facts") {
		t.Errorf("project forget did not disclose facts surviving:\n%s", out)
	}
	d, _ = loadLongTermMemory()
	if len(d.Facts) != 1 {
		t.Errorf("forget-this-project erased a global fact: %+v", d.Facts)
	}

	handled, out = memoryAsk(t, conv, "forget everything")
	if !handled || !strings.Contains(out, "all facts about you") {
		t.Errorf("forget-everything did not disclose its full scope:\n%s", out)
	}
	d, _ = loadLongTermMemory()
	if len(d.Facts) != 0 || len(d.Preferences) != 0 {
		t.Errorf("store survived forget-everything: %+v", d)
	}
}

// Forget discipline for facts, two pins. (1) An exact
// phrase match outranks substring swallowing: "forget my
// name" removes the name fact even though "name" is a
// substring of the wife fact's label, and "forget my
// wife's name" removes the wife fact even though it
// contains the name fact's whole phrase. (2) A genuinely
// ambiguous query (two favorites, shared wording) lists
// the candidates and changes nothing.
func TestFactForgetExactWinsAndAmbiguityLists(t *testing.T) {
	isolateMemoryStore(t)
	root := talkFixtureProject(t)
	withChatSession(t, root, true)
	conv := NewConversation()
	captureFactStatement("my name is John")
	captureFactStatement("my wife's name is Ana")
	captureFactStatement("my favorite color is forest green")
	captureFactStatement("my favorite food is pizza")

	handled, out := memoryAsk(t, conv, "forget my name")
	if !handled || !strings.Contains(out, "Forgotten: your name (was John).") {
		t.Errorf("exact name forget did not remove exactly the name fact:\n%s", out)
	}
	d, _ := loadLongTermMemory()
	if len(d.Facts) != 3 {
		t.Fatalf("name forget removed the wrong facts: %+v", d.Facts)
	}
	if _, ok := d.Facts["family:wife"]; !ok {
		t.Errorf("name forget swallowed the wife fact: %+v", d.Facts)
	}

	handled, out = memoryAsk(t, conv, "forget my favorite")
	if !handled || !strings.Contains(out, "more than one stored fact") ||
		!strings.Contains(out, "your favorite color") || !strings.Contains(out, "your favorite food") ||
		!strings.Contains(out, "Nothing was forgotten") {
		t.Errorf("ambiguous favorite forget did not list and hold:\n%s", out)
	}
	d, _ = loadLongTermMemory()
	if len(d.Facts) != 3 {
		t.Errorf("ambiguous forget changed facts: %+v", d.Facts)
	}

	handled, out = memoryAsk(t, conv, "forget my wife's name")
	if !handled || !strings.Contains(out, "Forgotten: your wife's name (was Ana).") {
		t.Errorf("specific fact forget failed:\n%s", out)
	}
	handled, out = memoryAsk(t, conv, "forget my favorite color")
	if !handled || !strings.Contains(out, "Forgotten: your favorite color (was forest green).") {
		t.Errorf("specific favorite forget failed:\n%s", out)
	}
	d, _ = loadLongTermMemory()
	if len(d.Facts) != 1 {
		t.Fatalf("specific forgets removed the wrong facts: %+v", d.Facts)
	}
	if _, ok := d.Facts["favorite:food"]; !ok {
		t.Errorf("the food fact did not survive the specific forgets: %+v", d.Facts)
	}
}

// Schema discipline: a store file written before facts
// existed (no "facts" key) loads cleanly as an empty set,
// and a fact save into it keeps the rest of the file.
func TestFactStoreSchemaMigratesOnLoad(t *testing.T) {
	home := isolateMemoryStore(t)
	legacy := []byte("{\n  \"version\": 1,\n  \"preferences\": [\n    \"I prefer tabs\"\n  ]\n}\n")
	if err := os.WriteFile(filepath.Join(home, ".gollemer_memory.json"), legacy, 0o600); err != nil {
		t.Fatal(err)
	}
	d, err := loadLongTermMemory()
	if err != nil {
		t.Fatal(err)
	}
	if d.Facts == nil || len(d.Facts) != 0 || len(d.Preferences) != 1 {
		t.Fatalf("legacy store did not load with an empty fact set: %+v", d)
	}
	captureFactStatement("my dog is Bella")
	d, err = loadLongTermMemory()
	if err != nil {
		t.Fatal(err)
	}
	if len(d.Preferences) != 1 || d.Facts["pet:dog"].Value != "Bella" {
		t.Errorf("fact save into a legacy store lost content: %+v", d)
	}
}

// The multiturn harness's two deterministic cells, at unit
// level: the intent layer's dispatch must veto its own
// what_now verdict for a transcript-recall line (the
// measured theft), so the line falls through to
// Conversation.Recall — which then answers with the exact
// quote the harness pins. A genuine what-now line still
// dispatches.
func TestIntentDispatchVetoLeavesTranscriptRecall(t *testing.T) {
	conv := NewConversation()
	conv.AddUser("hello there")
	conv.AddReply("hey there! how have you been lately?", SocialDomain, false)

	if dispatchIntent("what_now", "what did I just say", nil) {
		t.Error("dispatch claimed a transcript-recall line, want the veto to fall through")
	}
	if dispatchIntent("what_now", "what did you just say", nil) {
		t.Error("dispatch claimed a transcript-recall line, want the veto to fall through")
	}
	ans, ok := conv.Recall("what did I just say")
	if !ok || ans != `you said: "hello there"` {
		t.Errorf("Recall(what did I just say) = %q, %v — want the harness's exact quote", ans, ok)
	}
	ans, ok = conv.Recall("what did you just say")
	if !ok || !strings.HasPrefix(ans, "i said: ") {
		t.Errorf("Recall(what did you just say) = %q, %v — want the harness's prefix", ans, ok)
	}

	var handled bool
	out := captureStdout(t, func() { handled = dispatchIntent("what_now", "im stuck", nil) })
	if !handled || !strings.Contains(out, "tour") {
		t.Errorf("genuine what_now dispatch changed: handled=%v out=%q", handled, out)
	}
}
