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
	// Claimed marks a gollemer turn the history hook recorded for a
	// deterministic seat that claimed the exchange (historyhook.go):
	// the seat printed its answer but never recorded it, so the hook
	// wrote this compact entry from the seat's own output. Claimed
	// exchanges count as context for SocialInput composition, like
	// social exchanges do.
	Claimed bool
}

// Conversation is gollemer's session memory: a rolling transcript of the
// current chat. It is deliberately session-only — nothing is written to
// disk — so "remembering" never outlives the conversation it belongs to.
type Conversation struct {
	turns    []Turn
	maxTurns int
	// records counts every turn ever added (monotonic; the trim
	// does not move it). The history hook diffs it around a claim
	// to learn what the claim recorded itself, so a claimed
	// exchange lands exactly once.
	records int
}

// NewConversation starts an empty session transcript.
func NewConversation() *Conversation {
	return &Conversation{maxTurns: 30}
}

// recordCount reports how many turns have ever been added.
func (c *Conversation) recordCount() int { return c.records }

// AddUser records a user message.
func (c *Conversation) AddUser(text string) {
	c.turns = append(c.turns, Turn{Speaker: "you", Text: text})
	c.records++
	c.trim()
}

// AddReply records a gollemer reply. recall marks replies that merely
// quoted the transcript (answers to recall questions).
func (c *Conversation) AddReply(text, domain string, recall bool) {
	c.turns = append(c.turns, Turn{Speaker: "gollemer", Text: text, Domain: domain, Recall: recall})
	c.records++
	c.trim()
}

// AddClaimedReply records the history hook's entry for a claimed
// exchange: a gollemer turn whose text is the compact rendering of
// what the claiming seat printed (see historyhook.go). The user's
// line is recorded separately (AddUser) just before it.
func (c *Conversation) AddClaimedReply(text string) {
	c.turns = append(c.turns, Turn{Speaker: "gollemer", Text: text, Domain: ClaimedDomain, Claimed: true})
	c.records++
	c.trim()
}

// AddUserBeforeLastReply inserts a user turn immediately before the
// final turn when that turn is a gollemer reply — completing an
// exchange whose reply was recorded without its user line. With no
// trailing gollemer turn it appends, like AddUser.
func (c *Conversation) AddUserBeforeLastReply(text string) {
	t := Turn{Speaker: "you", Text: text}
	if n := len(c.turns); n > 0 && c.turns[n-1].Speaker == "gollemer" {
		c.turns = append(c.turns, Turn{})
		copy(c.turns[n:], c.turns[n-1:])
		c.turns[n-1] = t
	} else {
		c.turns = append(c.turns, t)
	}
	c.records++
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
// the caller decides which domains may contribute context. claimed
// reports whether the reply turn is a history-hook entry for a
// deterministic claim (see historyhook.go).
func (c *Conversation) lastExchange() (user, reply string, replyDomain string, claimed bool, ok bool) {
	ri := -1
	for i := len(c.turns) - 1; i >= 0; i-- {
		if c.turns[i].Speaker == "gollemer" && !c.turns[i].Recall {
			ri = i
			break
		}
	}
	if ri <= 0 {
		return "", "", "", false, false
	}
	for i := ri - 1; i >= 0; i-- {
		if c.turns[i].Speaker == "you" {
			return c.turns[i].Text, c.turns[ri].Text, c.turns[ri].Domain, c.turns[ri].Claimed, true
		}
	}
	return "", "", "", false, false
}

// SocialInput builds the encoder input for a conversational turn. When the
// previous exchange was also social, it is prefixed so the model can use
// the context ("why?", "tell me more", pronouns); otherwise the message
// stands alone. The social model is trained on exactly this format —
// "before you said ... . before i said ... . now you say ..." — so the
// markers are ordinary vocabulary, not special tokens.
//
// A previous exchange a deterministic seat claimed contributes context
// the same way, through its hook-recorded entry: the thread the model
// inherits is unbroken even when the last turn never reached a model.
// Self-recorded deterministic replies (fact, state, capability —
// TalkDomain, recorded by their own seats in full) do NOT compose:
// facts stay out of the model's input by design, and the capability
// catalog is far beyond the composed format's length.
func (c *Conversation) SocialInput(current string) string {
	// The typo normalizer (normalize.go) repairs the user's
	// current words before they are composed — the only
	// place a line is reshaped on its way to the social
	// model, and only ever toward a word the model knows.
	current = normalizeSocialCurrent(current)
	user, reply, domain, claimed, ok := c.lastExchange()
	if !ok || (domain != SocialDomain && !claimed) {
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

// isTranscriptRecallLine reports whether the line is one of
// the anchored transcript-recall shapes Recall owns. The
// intent layer's dispatch veto (intent.go) uses it: the
// classifier reads "what did I just say" as a beginner
// intent, and the deterministic recall behind the layer
// must still get its line.
func isTranscriptRecallLine(input string) bool {
	return recallYouSaid.MatchString(input) || recallISaid.MatchString(input) || recallCommand.MatchString(input)
}

// Recall questions are anchored full matches so ordinary chat that merely
// contains the words ("repeat that video", "what did you say your name
// was") never triggers them.
var recallYouSaid = regexp.MustCompile(`(?i)^\s*(what did i (just |)say|what did i ask( you|)|what was my (last |)question|what have i (just |)said)\s*[?.!]*\s*$`)
var recallISaid = regexp.MustCompile(`(?i)^\s*(what did you (just |)say|repeat that|say that again|what was your (last |)answer|what did you (just |)tell me|can you repeat that)\s*[?.!]*\s*$`)
var recallCommand = regexp.MustCompile(`(?i)^\s*(what was the last command( you gave me|)|repeat the last command)\s*[?.!]*\s*$`)
