package chat

import (
	"bytes"
	"io"
	"os"
	"strings"
)

// historyhook.go — the history leak fix (John's spec, 2026-11-07).
//
// The leak: both chat loops run their deterministic claim chain
// BEFORE the conversational tail records the user's line, so a
// turn a seat claimed (a grammar, a taught kind, the book, an
// edit, errorfix guidance…) never entered the session transcript
// at all. The next turn reached the social model bare — no
// "before you said / before i said" — however continuous the
// conversation had actually been.
//
// The fix is one centralized interceptor, not per-seat patches:
// every claim in both loops' chains runs through claimHook.run.
// The hook runs the claim under a stdout tee (the seat's output
// reaches the terminal exactly as before; a copy is buffered),
// then lands the exchange in the Conversation exactly once:
//
//   - a seat that recorded itself (the memory, capability,
//     talk-state and fact seats — user line + reply, detected by
//     the Conversation's records counter) is left alone;
//   - a seat that recorded only its reply gets its user line
//     inserted before that reply;
//   - a seat that recorded only the user line gets the claimed
//     reply appended;
//   - a seat that recorded nothing gets the user line plus a
//     claimed reply whose text is derived from the claim's OWN
//     captured output (claimedEntryText) — the seat's words,
//     compacted, never an invented paraphrase.
//
// The hook writes ONLY transcript turns. Receipts, the fact
// capture hook (which lives in the tail, on unclaimed lines, as
// before), session-context snapshots and every store behave
// exactly as they did.

// ClaimedDomain is the transcript tag for hook-recorded entries:
// the reply was a deterministic seat's, recorded by the hook.
// Like TalkDomain it is a transcript tag only, never a routing
// domain.
const ClaimedDomain = "claimed"

// claimHook is one turn's interceptor. line is a pointer: the
// unified loop rewrites the line mid-chain (the spoken-target
// pass), and the recorded user turn must be the line as it stood
// when the claim fired — the same text the tail would record.
type claimHook struct {
	conv *Conversation
	line *string
}

func newClaimHook(conv *Conversation, line *string) *claimHook {
	return &claimHook{conv: conv, line: line}
}

// run executes one deterministic claim and, when it fires, makes
// sure its exchange is in the conversation exactly once before
// returning. A claim that does not fire changes nothing.
func (h *claimHook) run(claim func() bool) bool {
	before := h.conv.recordCount()
	handled, out := teeStdout(claim)
	if !handled {
		return false
	}
	switch added := h.conv.recordCount() - before; {
	case added >= 2:
		// The seat recorded the exchange itself (user + reply).
	case added == 1 && len(h.conv.turns) > 0 && h.conv.turns[len(h.conv.turns)-1].Speaker == "you":
		// The seat recorded only the user line.
		h.conv.AddClaimedReply(claimedEntryText(out))
	case added == 1:
		// The seat recorded only its reply: complete the
		// exchange by putting the user line before it.
		h.conv.AddUserBeforeLastReply(*h.line)
	default:
		// The seat recorded nothing: the hook records the
		// whole exchange from the claim's own output.
		h.conv.AddUser(*h.line)
		h.conv.AddClaimedReply(claimedEntryText(out))
	}
	return true
}

// teeStdout runs f with os.Stdout replaced by a tee: everything f
// prints still reaches the real stdout live (prompts, gates and
// streamed output behave exactly as before), and a bounded copy
// (head + tail) is kept for the hook's entry. Capture failure
// degrades to running f bare — recording must never break a turn.
func teeStdout(f func() bool) (handled bool, captured string) {
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		return f(), ""
	}
	buf := &captureBuffer{}
	done := make(chan struct{})
	go func() {
		_, _ = io.Copy(io.MultiWriter(orig, buf), r)
		close(done)
	}()
	os.Stdout = w
	func() {
		defer func() { os.Stdout = orig }()
		handled = f()
	}()
	os.Stdout = orig
	_ = w.Close()
	<-done
	_ = r.Close()
	return handled, buf.String()
}

// captureBuffer keeps the head and tail of a claim's output (the
// entry derivation needs the answer's opening and, for write
// claims, the confirmation at the end) without buffering a whole
// streamed program output.
type captureBuffer struct {
	head bytes.Buffer
	tail bytes.Buffer
	n    int
}

const captureKeep = 8192

func (b *captureBuffer) Write(p []byte) (int, error) {
	b.n += len(p)
	if b.head.Len() < captureKeep {
		room := captureKeep - b.head.Len()
		if room >= len(p) {
			b.head.Write(p)
		} else {
			b.head.Write(p[:room])
		}
	}
	b.tail.Write(p)
	if b.tail.Len() > captureKeep {
		keep := b.tail.Bytes()[b.tail.Len()-captureKeep:]
		b.tail.Reset()
		b.tail.Write(keep)
	}
	return len(p), nil
}

// String returns the captured output: the whole thing when it
// fit, else head + a gap marker + tail.
func (b *captureBuffer) String() string {
	if b.n <= captureKeep {
		return b.head.String()
	}
	return b.head.String() + "\n...\n" + b.tail.String()
}

// claimedEntryMax bounds a hook entry: long enough to carry the
// seat's answer, short enough to sit in the composed social
// input's middle slot alongside it.
const claimedEntryMax = 280

// claimedEntryText derives a claimed exchange's transcript entry
// from the claim's own printed output, deterministically:
//
//   - a standing confirmation ("Done — " / "Fixed — ") is the
//     landing's own factual sentence, so the entry is that line;
//   - otherwise the entry is the seat's first tagged answer
//     block (or its whole output when it printed untagged);
//   - the "gollemer […]>" tag prefix is rendering, not reply
//     text, and is stripped — as it is for every recorded reply;
//   - whitespace is collapsed, a trailing "[y/n]" gate-prompt
//     remnant is trimmed, and the result is capped at
//     claimedEntryMax chars on a word boundary.
func claimedEntryText(raw string) string {
	s := stripANSI(raw)
	for _, marker := range []string{"Done — ", "Fixed — "} {
		if i := strings.LastIndex(s, marker); i >= 0 {
			end := strings.IndexByte(s[i:], '\n')
			if end < 0 {
				return compactEntry(s[i:])
			}
			return compactEntry(s[i : i+end])
		}
	}
	if i := strings.Index(s, "gollemer ["); i >= 0 {
		s = s[i:]
		if j := strings.Index(s, "]> "); j >= 0 {
			s = s[j+3:]
		}
	}
	return compactEntry(s)
}

// compactEntry collapses whitespace and caps one entry text.
func compactEntry(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	s = strings.TrimSuffix(s, "[y/n]")
	s = strings.TrimSpace(s)
	if len(s) <= claimedEntryMax {
		return s
	}
	cut := s[:claimedEntryMax]
	if i := strings.LastIndex(cut, " "); i > 0 {
		cut = cut[:i]
	}
	return cut + "..."
}
