package chat

import (
	"fmt"
	"regexp"
	"strings"
)

// A Turn is one side of one chat exchange: something the user said or
// something gollemer said. Recall replies (answers to "what did you just
// say") are flagged so later recall skips past them to the last real reply.
type Turn struct {
	Speaker string // "you" or "gollemer"
	Text    string // raw text as typed/printed (not lowercased)
	Domain  string // routing domain of a gollemer turn, "" for user turns
	Recall  bool   // true when this turn only quoted the transcript
}

// Conversation is gollemer's session memory: a rolling transcript of the
// current chat. It is deliberately session-only — nothing is written to
// disk — so "remembering" never outlives the conversation it belongs to.
type Conversation struct {
	turns    []Turn
	maxTurns int
}

// NewConversation starts an empty session transcript.
func NewConversation() *Conversation {
	return &Conversation{maxTurns: 30}
}

// AddUser records a user message.
func (c *Conversation) AddUser(text string) {
	c.turns = append(c.turns, Turn{Speaker: "you", Text: text})
	c.trim()
}

// AddReply records a gollemer reply. recall marks replies that merely
// quoted the transcript (answers to recall questions).
func (c *Conversation) AddReply(text, domain string, recall bool) {
	c.turns = append(c.turns, Turn{Speaker: "gollemer", Text: text, Domain: domain, Recall: recall})
	c.trim()
}

// Clear forgets the whole session transcript.
func (c *Conversation) Clear() {
	c.turns = nil
}

// Len reports how many turns are remembered.
func (c *Conversation) Len() int {
	return len(c.turns)
}

// Turns returns the remembered transcript, oldest first.
func (c *Conversation) Turns() []Turn {
	return c.turns
}

func (c *Conversation) trim() {
	if len(c.turns) > c.maxTurns {
		c.turns = c.turns[len(c.turns)-c.maxTurns:]
	}
}

// lastUserText returns the most recent user message, "" when none.
func (c *Conversation) lastUserText() string {
	for i := len(c.turns) - 1; i >= 0; i-- {
		if c.turns[i].Speaker == "you" {
			return c.turns[i].Text
		}
	}
	return ""
}

// lastReply returns the most recent non-recall gollemer reply, "" when none.
// Recall replies are skipped so "what did you say" never quotes its own
// earlier answer back at the user.
func (c *Conversation) lastReply() string {
	for i := len(c.turns) - 1; i >= 0; i-- {
		if c.turns[i].Speaker == "gollemer" && !c.turns[i].Recall {
			return c.turns[i].Text
		}
	}
	return ""
}

// lastCommand returns the most recent gocli reply (an exact command), ""
// when none has been given this session.
func (c *Conversation) lastCommand() string {
	for i := len(c.turns) - 1; i >= 0; i-- {
		if c.turns[i].Speaker == "gollemer" && c.turns[i].Domain == GoCliDomain && !c.turns[i].Recall {
			return c.turns[i].Text
		}
	}
	return ""
}

// lastExchange returns the previous user message and the gollemer reply
// that followed it, for history-conditioned generation. Both must exist;
// the caller decides which domains may contribute context.
func (c *Conversation) lastExchange() (user, reply string, replyDomain string, ok bool) {
	ri := -1
	for i := len(c.turns) - 1; i >= 0; i-- {
		if c.turns[i].Speaker == "gollemer" && !c.turns[i].Recall {
			ri = i
			break
		}
	}
	if ri <= 0 {
		return "", "", "", false
	}
	for i := ri - 1; i >= 0; i-- {
		if c.turns[i].Speaker == "you" {
			return c.turns[i].Text, c.turns[ri].Text, c.turns[ri].Domain, true
		}
	}
	return "", "", "", false
}

// SocialInput builds the encoder input for a conversational turn. When the
// previous exchange was also social, it is prefixed so the model can use
// the context ("why?", "tell me more", pronouns); otherwise the message
// stands alone. The social model is trained on exactly this format —
// "before you said ... . before i said ... . now you say ..." — so the
// markers are ordinary vocabulary, not special tokens.
func (c *Conversation) SocialInput(current string) string {
	user, reply, domain, ok := c.lastExchange()
	if !ok || domain != SocialDomain {
		return current
	}
	return fmt.Sprintf("before you said %s . before i said %s . now you say %s",
		strings.ToLower(user), strings.ToLower(reply), current)
}

// Recall answers questions about the conversation itself by quoting the
// transcript verbatim. It returns the reply text and true when the input
// is a recall question; deterministic lookup, never the neural model.
func (c *Conversation) Recall(input string) (string, bool) {
	switch {
	case recallYouSaid.MatchString(input):
		if t := c.lastUserText(); t != "" {
			return fmt.Sprintf("you said: %q", t), true
		}
		return "we are just getting started, you have not said anything yet", true
	case recallISaid.MatchString(input):
		if t := c.lastReply(); t != "" {
			return fmt.Sprintf("i said: %q", t), true
		}
		return "i have not said anything yet", true
	case recallCommand.MatchString(input):
		if t := c.lastCommand(); t != "" {
			return fmt.Sprintf("the last command was: %s", t), true
		}
		return "i have not given you a command yet this session", true
	}
	return "", false
}

// Recall questions are anchored full matches so ordinary chat that merely
// contains the words ("repeat that video", "what did you say your name
// was") never triggers them.
var recallYouSaid = regexp.MustCompile(`(?i)^\s*(what did i (just |)say|what did i ask( you|)|what was my (last |)question|what have i (just |)said)\s*[?.!]*\s*$`)
var recallISaid = regexp.MustCompile(`(?i)^\s*(what did you (just |)say|repeat that|say that again|what was your (last |)answer|what did you (just |)tell me|can you repeat that)\s*[?.!]*\s*$`)
var recallCommand = regexp.MustCompile(`(?i)^\s*(what was the last command( you gave me|)|repeat the last command)\s*[?.!]*\s*$`)
