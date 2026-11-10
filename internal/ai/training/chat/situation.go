package chat

// The situation line: a compressed context payload the chat
// builds deterministically and prepends to the social model's
// composed input (SocialInputSituated), so a multi-turn input
// carries more than the single previous exchange SocialInput
// can hold. John directed this addition 2026-11-07 for the
// general multi-turn round: the model should know how to keep
// context, and the line is the part of the context that is
// true regardless of what the last exchange happened to be.
//
// Shape — one bracketed line, in the composed format's own
// plain-word style (the tokenizer splits the brackets and the
// dots as single tokens, like the " . " separators):
//
//	[ situation : evening . a few exchanges in . you told me about your dog and your city ]
//
// Every segment is derived from state the chat actually has,
// and nothing is invented:
//
//   - the part of the day, from the clock (the same clock the
//     state block in memory_state.go reads);
//   - how deep into the conversation this turn sits, counted
//     from the session transcript itself;
//   - the TOPICS of the long-term fact store — the categories
//     of what the user has told the chat (a dog, a city, a
//     job), read from the store keys. Fact VALUES never ride
//     the line: values belong to the fact-memory seat, which
//     answers for them exactly; the line names only that a
//     topic exists, so the model can stay on a subject without
//     being handed words to echo or invent from. Fact kinds
//     with no clean topic phrase (the user's name — names are
//     the fact seat's whole job — and free-form kinds) are
//     skipped, honestly, rather than phrased badly.
//
// The line is prepended ONLY when SocialInput composes — a
// first turn stands bare, exactly as the single-turn corpus
// taught, so single-turn behavior is untouched. The training
// pairs of the multi-turn round carry the same line in the
// same shape (composed by the capture probe through this very
// builder, with the session's assigned situation), so train
// and serve match by construction.

import (
	"strings"
	"time"
)

// SituationLine builds the bracketed situation line for a
// composed social input. exchanges is the number of completed
// exchanges already in the transcript (the composed input's
// own precondition — with no previous exchange there is no
// composed input and no line, so exchanges < 1 returns "").
// topics carries fact-topic phrases ("your dog") as built by
// situationFactTopics; the topics segment is omitted when
// there are none.
func SituationLine(now time.Time, exchanges int, topics []string) string {
	if exchanges < 1 {
		return ""
	}
	parts := []string{"[ situation : " + daypartWord(now), depthPhrase(exchanges)}
	if len(topics) > 0 {
		parts = append(parts, "you told me about "+joinTopics(topics))
	}
	return strings.Join(parts, " . ") + " ]"
}

// daypartWord names the part of the day from the clock:
// night in the small hours, then morning, afternoon, evening.
// The words are the greeting words the corpus already uses.
func daypartWord(now time.Time) string {
	switch h := now.Hour(); {
	case h >= 5 && h < 12:
		return "morning"
	case h >= 12 && h < 17:
		return "afternoon"
	case h >= 17 && h < 22:
		return "evening"
	default:
		return "night"
	}
}

// depthPhrase names how deep into the conversation the
// transcript has run, in three buckets coarse enough to be
// stable across a whole session's middle.
func depthPhrase(exchanges int) string {
	switch {
	case exchanges <= 1:
		return "just started chatting"
	case exchanges <= 4:
		return "a few exchanges in"
	default:
		return "deep in conversation"
	}
}

// joinTopics renders the topic list as a spoken list:
// "your dog", "your dog and your city",
// "your dog, your city and your job".
func joinTopics(topics []string) string {
	switch len(topics) {
	case 0:
		return ""
	case 1:
		return topics[0]
	case 2:
		return topics[0] + " and " + topics[1]
	default:
		return strings.Join(topics[:len(topics)-1], ", ") + " and " + topics[len(topics)-1]
	}
}

// situationTopicPhrase maps one long-term fact store key to
// the topic phrase the line may name, or "" when the kind has
// no clean phrase. Keys carry kind and subject ("pet:dog",
// "family:wife", "favorite:color", "city", "job",
// "birthday"); the phrase names the subject as something the
// user has, never its value. "name" and "learning" and any
// unknown kind return "" — the name is the fact seat's to
// answer for, and an unphraseable kind is skipped rather than
// mangled into the line.
func situationTopicPhrase(key string) string {
	kind, subject, hasSubject := strings.Cut(key, ":")
	subject = strings.TrimSpace(subject)
	cleanSubject := subject != "" && !strings.ContainsAny(subject, " \t")
	switch kind {
	case "pet":
		if hasSubject && cleanSubject {
			return "your " + subject
		}
	case "family":
		if hasSubject && cleanSubject {
			return "your " + subject
		}
	case "favorite":
		if hasSubject && cleanSubject {
			return "your favorite " + subject
		}
	case "city":
		return "your city"
	case "job":
		return "your job"
	case "birthday":
		return "your birthday"
	}
	return ""
}

// situationFactTopics reads the current project's slice of
// the long-term memory store and returns the topic phrases
// for the facts it holds: sorted by store key (the store's
// own deterministic order), duplicates removed, capped at
// three so the line stays a line. A missing or empty store
// yields no topics — the line simply goes without the
// segment.
func situationFactTopics() []string {
	d, _ := loadMemorySlice(memoryCurrentRoot())
	if len(d.Facts) == 0 {
		return nil
	}
	var topics []string
	seen := map[string]bool{}
	for _, key := range sortedFactKeys(d.Facts) {
		phrase := situationTopicPhrase(key)
		if phrase == "" || seen[phrase] {
			continue
		}
		seen[phrase] = true
		topics = append(topics, phrase)
	}
	if len(topics) > 3 {
		topics = topics[:3]
	}
	return topics
}

// exchangeCount counts the completed exchanges in the
// transcript: a user turn answered by a non-recall gollemer
// turn. Recall turns quote the transcript rather than answer
// it, so they never complete an exchange — the same rule
// lastExchange composes by.
func (c *Conversation) exchangeCount() int {
	n := 0
	seenUser := false
	for _, t := range c.turns {
		switch {
		case t.Speaker == "you":
			seenUser = true
		case t.Speaker == "gollemer" && !t.Recall && seenUser:
			n++
			seenUser = false
		}
	}
	return n
}

// SocialInputSituated is SocialInput with the situation line
// prepended when the input composes. The gate is SocialInput's
// own — a previous social (or history-hook claimed) exchange
// must exist — so a bare first turn, and a turn after a
// non-social exchange, stay byte-identical to SocialInput.
// The line is built from the conversation's own transcript
// depth, the caller's clock, and the caller's fact topics.
func (c *Conversation) SocialInputSituated(current string, now time.Time, topics []string) string {
	_, _, domain, claimed, ok := c.lastExchange()
	if !ok || (domain != SocialDomain && !claimed) {
		return current
	}
	line := SituationLine(now, c.exchangeCount(), topics)
	if line == "" {
		return c.SocialInput(current)
	}
	return line + " " + c.SocialInput(current)
}
