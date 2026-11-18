package chat

import (
	"fmt"
	"os"
	"time"
	"unicode/utf8"
)

// pace.go — PACED OUTPUT. A wall of text appearing all at once is a
// wall nobody reads (John, 2026-10-21: "have it write a little slower
// so it isnt just a giant print out blocks"). In an interactive
// session, big multi-line answers TYPE OUT — a few characters at a
// time, at a reading pace — instead of dumping. Slowing the rate
// alone wasn't enough: emitting whole lines with a pause between
// them still landed as blocks (John, 2026-10-25: "can you have it
// print out in typewriter style not just sentence by sentence"), so
// each line's pause is now spent in small pieces across the line
// itself and the characters flow. Short output (a line or two)
// prints instantly — pacing a one-liner would just be latency.
//
// The mechanism is deliberately small: printPaced renders a whole
// string through the pause logic; the handful of places that print
// LARGE answers (story, tour stops, guided stops, lessons, uexplain)
// call it. Everything else keeps printing directly, unpaced.
//
// Rules that keep it honest:
//   - Interactive terminals only (piped/scripted output never paces).
//   - A burst budget: one answer may add at most paceBurstBudget of
//     delay, so a 300-line dump streams its head and flows the rest
//     instead of holding the user hostage.
//   - User control: "/speed" toggles pacing in chat; GOLLEMER_PACE=off
//     disables it from the environment.

// paceOn is the live switch (flipped by /speed). Starts from the env.
var paceOn = os.Getenv("GOLLEMER_PACE") != "off"

// paceInteractive is set by each chat loop when its LineReader is on
// a real terminal; piped sessions never pace.
var paceInteractive bool

// paceBurstBudget caps the total delay one printed answer may add.
const paceBurstBudget = 15 * time.Second

// paceMinLines is the smallest output that paces at all: one or two
// lines are a reply, not a wall.
const paceMinLines = 3

// paceDelayFor is the total pause a line of the given visible
// width earns: a flat beat plus time proportional to the line's
// length — 200ms + 20ms per character, capped at 2.4s — which
// lands a typical multi-line answer at roughly 50 characters
// per second, a comfortable reading pace. (The previous table,
// 25/55/90ms by width, measured ≈1,600 chars/sec; the user's
// verdict was "it types way too fast".) The pause is no longer
// taken in one lump after the line: pacePlan spends it across
// the line's typewriter chunks, so the line's total is exactly
// this and only the granularity changed. One pacing source:
// every paced surface flows through printPaced and this table.
func paceDelayFor(visible int) time.Duration {
	d := 200*time.Millisecond + time.Duration(visible)*20*time.Millisecond
	if d > 2400*time.Millisecond {
		d = 2400 * time.Millisecond
	}
	return d
}

// paceSleep is the pause hook (a variable so tests can count pauses
// without actually waiting).
var paceSleep = time.Sleep

// setSpeed flips the pacing switch and says where it landed.
// Called by the /speed family in the chat's beginner dispatcher.
func setSpeed(on bool) {
	paceOn = on
	if on {
		fmt.Println("Pacing on — long answers type out at a reading pace instead of dumping. Say \"speed off\" for full speed.")
	} else {
		fmt.Println("Pacing off — everything prints at full speed. Say \"speed on\" to bring the streaming back.")
	}
}

// printPaced prints s (which may contain many lines) as a
// typewriter stream: small pieces of text, each followed by its
// share of the pause, while the burst budget lasts. When pacing
// is off, non-interactive, or the text is short, it prints in
// one write. The pieces partition s exactly — paced and unpaced
// output are byte-identical; only the timing differs.
func printPaced(s string) {
	s = decorateOutput(s)
	if !paceOn || !paceInteractive || s == "" {
		printRaw(s)
		return
	}
	lines := splitLines(s)
	if len(lines) < paceMinLines {
		printRaw(s)
		return
	}
	for _, p := range pacePlan(lines) {
		printRaw(p.text)
		if p.sleep > 0 {
			paceSleep(p.sleep)
		}
	}
}

// paceChunkRunes is the visible width of one typewriter piece:
// small enough that text visibly flows character by character,
// large enough that the write rate stays trivial (at the
// reading pace, roughly twenty writes a second).
const paceChunkRunes = 3

// pacePiece is one emission unit of a paced print: the text to
// write, then the pause to take after it.
type pacePiece struct {
	text  string
	sleep time.Duration
}

// pacePlan turns lines into the typewriter emission plan. Each
// line but the last earns paceDelayFor's pause, subject to the
// same burst-budget rule as always: a line whose pause would
// overspend the budget gets none (its text flows whole, at full
// speed — the budget never drops or speeds up content, it just
// stops pausing), and a line that fits spends its whole pause
// across its own chunks. Per-line totals therefore match the
// old line-at-a-time schedule exactly; the plan only changes
// how finely each pause is spent.
func pacePlan(lines []string) []pacePiece {
	var pieces []pacePiece
	var spent time.Duration
	for i, ln := range lines {
		var d time.Duration
		if i < len(lines)-1 {
			d = paceDelayFor(visibleLen(ln))
			if spent+d > paceBurstBudget {
				d = 0 // budget spent: the rest flows at full speed
			} else {
				spent += d
			}
		}
		pieces = append(pieces, paceLinePieces(ln, d)...)
	}
	return pieces
}

// paceLinePieces splits one line into chunks of at most
// paceChunkRunes visible runes and deals the line's pause d
// across them in proportion to their widths — the last chunk
// takes the remainder, so the chunks' pauses sum to exactly d.
// A line with no pause to spend (the last line, or the budget
// is gone) stays a single whole piece: chunking text that will
// not pause would only multiply writes for no visible effect.
func paceLinePieces(ln string, d time.Duration) []pacePiece {
	chunks := paceChunks(ln)
	if d <= 0 || len(chunks) <= 1 {
		return []pacePiece{{text: ln, sleep: d}}
	}
	total := visibleLen(ln)
	pieces := make([]pacePiece, len(chunks))
	var dealt time.Duration
	for i, c := range chunks {
		var sleep time.Duration
		if i < len(chunks)-1 {
			if total > 0 {
				sleep = d * time.Duration(visibleLen(c)) / time.Duration(total)
			}
			dealt += sleep
		} else {
			sleep = d - dealt // the last chunk takes what is left
		}
		pieces[i] = pacePiece{text: c, sleep: sleep}
	}
	return pieces
}

// paceChunks cuts s into consecutive pieces of at most
// paceChunkRunes visible runes. Cuts land only on rune
// boundaries; an ANSI escape sequence counts as zero width and
// is never split — it travels whole inside whichever chunk is
// open when it appears (the same scanning stripANSI uses). The
// pieces partition s: joined, they are s byte for byte.
func paceChunks(s string) []string {
	var chunks []string
	start, width := 0, 0
	for i := 0; i < len(s); {
		if s[i] == 0x1b && i+1 < len(s) && s[i+1] == '[' {
			j := i + 2
			for j < len(s) && !(s[j] >= '@' && s[j] <= '~') {
				j++
			}
			i = j + 1
			if i > len(s) {
				i = len(s)
			}
			continue
		}
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
		width++
		if width == paceChunkRunes {
			chunks = append(chunks, s[start:i])
			start, width = i, 0
		}
	}
	if start < len(s) {
		chunks = append(chunks, s[start:])
	}
	return chunks
}

// printRaw is the unpaced write printPaced falls back to.
func printRaw(s string) {
	os.Stdout.WriteString(s)
}

// splitLines splits s keeping each line's trailing newline.
func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i+1])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
