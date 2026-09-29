package chat

import "testing"

// Recall questions must be answered from the transcript, deterministically.
func TestConversationRecall(t *testing.T) {
	c := NewConversation()
	c.AddUser("hello there")
	c.AddReply("hey! how is your day going?", SocialDomain, false)

	if got, ok := c.Recall("what did I just say"); !ok || got != `you said: "hello there"` {
		t.Errorf("recall you said: got %q, ok=%v", got, ok)
	}
	if got, ok := c.Recall("what did you just say"); !ok || got != `i said: "hey! how is your day going?"` {
		t.Errorf("recall i said: got %q, ok=%v", got, ok)
	}
	if got, ok := c.Recall("repeat that"); !ok || got != `i said: "hey! how is your day going?"` {
		t.Errorf("repeat that: got %q, ok=%v", got, ok)
	}
	// Recall replies are skipped by later recall: "what did you say"
	// quotes the last real reply, never its own earlier answer.
	c.AddUser("what did you just say")
	c.AddReply(`i said: "hey! how is your day going?"`, SocialDomain, true)
	if got, ok := c.Recall("say that again"); !ok || got != `i said: "hey! how is your day going?"` {
		t.Errorf("recall skips recall replies: got %q, ok=%v", got, ok)
	}
}

// The last gocli command must be recallable for "what was the last command".
func TestConversationRecallCommand(t *testing.T) {
	c := NewConversation()
	c.AddUser("check if the dependencies are updated")
	c.AddReply("go mod tidy", GoCliDomain, false)
	c.AddUser("hello")
	c.AddReply("hey there!", SocialDomain, false)
	if got, ok := c.Recall("what was the last command"); !ok || got != "the last command was: go mod tidy" {
		t.Errorf("recall command: got %q, ok=%v", got, ok)
	}
}

// Empty transcript: recall admits it has nothing yet instead of inventing.
func TestConversationRecallEmpty(t *testing.T) {
	c := NewConversation()
	for _, q := range []string{"what did you say", "repeat that", "what was the last command", "what did I just say"} {
		got, ok := c.Recall(q)
		if !ok || got == "" {
			t.Errorf("empty recall %q: got %q, ok=%v", q, got, ok)
		}
	}
}

// Near-misses must NOT trigger recall: they are new questions, not memory.
func TestConversationRecallNearMiss(t *testing.T) {
	c := NewConversation()
	c.AddUser("hello")
	c.AddReply("hey!", SocialDomain, false)
	for _, q := range []string{
		"what did you say your name was",
		"repeat that video",
		"what did I say about the meeting tomorrow",
		"can you repeat that for the third time please now",
	} {
		if _, ok := c.Recall(q); ok {
			t.Errorf("near-miss %q wrongly triggered recall", q)
		}
	}
}

// History prefix: the previous social exchange is included; a non-social
// previous exchange is not (code in a chat model's context is noise).
func TestConversationSocialInput(t *testing.T) {
	c := NewConversation()
	if got := c.SocialInput("hello"); got != "hello" {
		t.Errorf("no history: got %q", got)
	}
	c.AddUser("hello")
	c.AddReply("hey there!", SocialDomain, false)
	want := "before you said hello . before i said hey there! . now you say how are you"
	if got := c.SocialInput("how are you"); got != want {
		t.Errorf("history prefix:\n got %q\nwant %q", got, want)
	}

	c2 := NewConversation()
	c2.AddUser("write a function")
	c2.AddReply("func add(a int, b int) int { return a + b }", GocodeDomain, false)
	if got := c2.SocialInput("thanks"); got != "thanks" {
		t.Errorf("non-social history must not prefix: got %q", got)
	}
}

// Transcript is bounded.
func TestConversationTrim(t *testing.T) {
	c := NewConversation()
	for i := 0; i < 100; i++ {
		c.AddUser("hi")
	}
	if c.Len() > 30 {
		t.Errorf("transcript not trimmed: %d turns", c.Len())
	}
}
