package chat

// Clarifying questions instead of dead ends (2026-11-05).
//
// When a chat ask can't fire because information is MISSING
// or AMBIGUOUS — several handlers fit "the handler", the
// sentence named no file, two files share the bare name the
// ask used — the chat used to print guidance prose and stop:
// a dead end the user had to decode and re-type in full.
// Now those seats ask ONE short question, hold the pending
// ask in session state (here, beside sessionctx), and the
// user's next line — if it answers the question (a name, a
// file, a number, a candidate's word) — completes the
// ORIGINAL ask through the same plan display, [y/n] gate,
// transaction and build verification as if it had been
// typed in full. Conversation, not forms.
//
// The mechanics, pinned:
//
//   - One pending question at a time, session-scoped,
//     never on disk, gone when the session ends. Setting a
//     new one replaces the old (a completion may itself end
//     at another question — the dialogue continues).
//   - The claim sits at the head of the loop's content
//     claims, ahead of learn mode, the edit claims, the
//     proposer, the phrasing layer, the boundary, and the
//     brains: first, so a non-answer clears the question
//     before ANY claim can act on the line (no stale
//     question completes an old ask after the user has
//     plainly moved on); and it cannot steal another flow's
//     answer, because an answer is one bare candidate word,
//     number, path, or identifier — never a sentence — and
//     the state-answering flows' setting lines pass through
//     here first, clearing any question, so the two are
//     never pending at once. With no question pending it
//     returns false without printing a byte: behavior is
//     byte-identical.
//   - This state is distinct from the intent layer's own
//     pendingClarify ("did you mean…?"): that one is
//     answered only by lines that reach the intent layer at
//     the loop's tail; this one is tried earlier, in the
//     edit chain. If both were ever outstanding, this — the
//     most recently asked — is tried first, and its
//     stopwords (yes/no included) hand an intent answer
//     through untouched.
//   - An answer must BE an answer: for a pick question, the
//     whole line is a candidate's number, name, or
//     distinguishing word; for a target question, one bare
//     path-shaped token; for a name question, one
//     identifier. A full valid ask therefore never parses
//     as an answer — it clears the question and is processed
//     exactly as if none had been asked, and the FULL
//     ORIGINAL SENTENCE the guidance spelled still works
//     the same way. No traps, no swallowed commands.
//   - A pending question is not a gate bypass and not a
//     fact override: the completion re-runs the seat's own
//     claim on the completed sentence, so every fact is
//     re-read and re-checked at completion time. Refusals
//     that state FACTS (a route conflict, a name collision,
//     a file that exists, no server to wire into) never
//     become questions at all — the seats only convert
//     their missing/ambiguous outcomes — and a completion
//     that meets such a fact refuses exactly as the full
//     sentence would have.

import (
	"fmt"
	"go/token"
	"strconv"
	"strings"
)

// clarifyKind is what shape of answer a pending question
// takes.
type clarifyKind int

const (
	// clarifyPick: one of a counted candidate set — answer
	// with a number, the candidate's label, or one of its
	// distinguishing words.
	clarifyPick clarifyKind = iota
	// clarifyTarget: a file or folder the ask didn't name —
	// answer with one bare path-shaped token; a numbered
	// suggestion may also be offered (the tracked file).
	clarifyTarget
	// clarifyName: a declaration's missing name — answer
	// with one identifier.
	clarifyName
)

// clarifyCandidate is one numbered choice a question
// offers. Label is what's picked (and the first answer
// word); Desc rides in parentheses; Words are the extra
// distinguishing words that also pick it.
type clarifyCandidate struct {
	Label string
	Desc  string
	Words []string
}

// pendingQuestion is the one held ask. complete runs the
// completion for a chosen candidate (pick, or a target
// suggestion picked by number/name); completeText runs it
// for a free-form target or name answer. Exactly one of
// the two is set per kind: pick questions carry complete,
// target questions carry completeText and may ALSO carry
// complete for their suggestions, name questions carry
// completeText.
type pendingQuestion struct {
	kind         clarifyKind
	cands        []clarifyCandidate
	complete     func(idx int, ask editAsk) bool
	completeText func(text string, ask editAsk) bool
}

var pendingEditQ *pendingQuestion

// setPendingQuestion holds q as the session's one pending
// question, replacing any previous one.
func setPendingQuestion(q *pendingQuestion) {
	pendingEditQ = q
}

// clearPendingQuestion forgets the pending question, if
// any. Called when the session's project changes (through
// resetWorkContext): a question belongs to the project and
// conversation that asked it.
func clearPendingQuestion() {
	pendingEditQ = nil
}

// hasPendingQuestion reports whether a question is being
// held — the seam tests assert state through.
func hasPendingQuestion() bool {
	return pendingEditQ != nil
}

// printClarifyIntro renders a question's opening line under
// the composer's tag.
func printClarifyIntro(tag, intro string) {
	fmt.Printf("%s %s\n", tag, intro)
}

// printClarifyCandidates renders the numbered candidate
// list, one per line.
func printClarifyCandidates(cands []clarifyCandidate) {
	for i, c := range cands {
		if c.Desc != "" {
			fmt.Printf("  %d. %s (%s)\n", i+1, c.Label, c.Desc)
		} else {
			fmt.Printf("  %d. %s\n", i+1, c.Label)
		}
	}
}

// printClarifyFooter renders how to answer — and that
// anything else simply moves on.
func printClarifyFooter(how string) {
	fmt.Printf("%s — or say anything else and we'll move on.\n", how)
}

// askPickQuestion prints a pick question (intro, numbered
// candidates, the how-to-answer footer) and holds it. The
// seats supply the words; this renders and holds.
func askPickQuestion(tag, intro, how string, cands []clarifyCandidate, complete func(idx int, ask editAsk) bool) {
	printClarifyIntro(tag, intro)
	printClarifyCandidates(cands)
	printClarifyFooter(how)
	setPendingQuestion(&pendingQuestion{kind: clarifyPick, cands: cands, complete: complete})
}

// clarifyAnswerText normalizes a would-be answer line for
// matching: trimmed, with the sentence punctuation the
// grammars tolerate stripped.
func clarifyAnswerText(line string) string {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	return strings.TrimSpace(s)
}

// clarifyStopwords are one-word lines that are never an
// answer to a target/name question — they keep their own
// meanings (or the boundary's) even mid-question.
var clarifyStopwords = map[string]bool{
	"the": true, "a": true, "an": true, "to": true, "in": true,
	"yes": true, "no": true, "y": true, "n": true,
	"stop": true, "quit": true, "exit": true, "cancel": true,
	"done": true, "installed": true,
}

// clarifyTargetToken reports whether a normalized line is
// one bare target-shaped token: a single word that names a
// .go file, carries a path, or is dot-free the way a folder
// name is. A sentence never qualifies (it has spaces or a
// verb's shape), which is what makes a full ask win over a
// pending question every time.
func clarifyTargetToken(s string) bool {
	if s == "" || strings.ContainsAny(s, " \t") {
		return false
	}
	if clarifyStopwords[strings.ToLower(s)] {
		return false
	}
	digits := true
	for _, r := range s {
		switch {
		case r == '_' || r == '-' || r == '.' || r == '/':
			digits = false
		case 'a' <= r && r <= 'z' || 'A' <= r && r <= 'Z':
			digits = false
		case '0' <= r && r <= '9':
		default:
			return false
		}
	}
	if digits {
		return false // a bare number is a pick answer or nothing, never a target
	}
	return strings.HasSuffix(strings.ToLower(s), ".go") ||
		strings.Contains(s, "/") || !strings.Contains(s, ".")
}

// clarifyNameToken reports whether a normalized line is one
// identifier that could name a declaration.
func clarifyNameToken(s string) bool {
	if s == "" || strings.ContainsAny(s, " \t") {
		return false
	}
	if clarifyStopwords[strings.ToLower(s)] {
		return false
	}
	return token.IsIdentifier(s)
}

// matchClarifyPick reads a normalized line as a pick
// answer: a candidate number, a candidate's label, or one
// of its distinguishing words — the WHOLE line, never a
// containment.
func matchClarifyPick(q *pendingQuestion, s string) (int, bool) {
	if n, err := strconv.Atoi(s); err == nil {
		if n >= 1 && n <= len(q.cands) {
			return n - 1, true
		}
		return 0, false
	}
	for i, c := range q.cands {
		if strings.EqualFold(s, c.Label) {
			return i, true
		}
		for _, w := range c.Words {
			if strings.EqualFold(s, w) {
				return i, true
			}
		}
	}
	return 0, false
}

// runClarifyAnswer is the chain claim's body: with a
// question pending, the line is first tried as its answer.
// An answer completes the held ask (consumed, whatever the
// completion's own outcome — plan, refusal, or declined
// gate). Anything else clears the question and returns
// false, so the line is processed exactly as if no question
// had been asked.
func runClarifyAnswer(line string, ask editAsk) bool {
	q := pendingEditQ
	if q == nil {
		return false
	}
	s := clarifyAnswerText(line)
	switch q.kind {
	case clarifyPick:
		if idx, ok := matchClarifyPick(q, s); ok {
			clearPendingQuestion()
			return q.complete(idx, ask)
		}
	case clarifyTarget:
		if idx, ok := matchClarifyPick(q, s); ok && q.complete != nil {
			clearPendingQuestion()
			return q.complete(idx, ask)
		}
		if clarifyTargetToken(s) && q.completeText != nil {
			clearPendingQuestion()
			return q.completeText(s, ask)
		}
	case clarifyName:
		if clarifyNameToken(s) && q.completeText != nil {
			clearPendingQuestion()
			return q.completeText(s, ask)
		}
	}
	clearPendingQuestion()
	return false
}

// tryClarifyAnswer is the chat-loop entry: it tries the
// line as the pending question's answer, on the chat's
// line reader.
func tryClarifyAnswer(line string, r *LineReader) bool {
	return runClarifyAnswer(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}
