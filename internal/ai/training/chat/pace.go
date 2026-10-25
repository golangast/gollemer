package chat

import (
	"fmt"
	"os"
	"time"
)

// pace.go — PACED OUTPUT. A wall of text appearing all at once is a
// wall nobody reads (John, 2026-10-21: "have it write a little slower
// so it isnt just a giant print out blocks"). In an interactive
// session, big multi-line answers are revealed line by line with a
// breath between lines, so they STREAM in instead of dumping. Short
// output (a line or two) prints instantly — pacing a one-liner would
// just be latency.
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

// paceDelayFor is the pause after a line of the given visible
// width: a flat beat plus time proportional to the line's
// length — 200ms + 20ms per character, capped at 2.4s — which
// lands a typical multi-line answer at roughly 50 characters
// per second, a comfortable reading pace. (The previous table,
// 25/55/90ms by width, measured ≈1,600 chars/sec; the user's
// verdict was "it types way too fast".) One pacing source:
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
		fmt.Println("Pacing on — long answers stream in line by line instead of dumping. Say \"speed off\" for full speed.")
	} else {
		fmt.Println("Pacing off — everything prints at full speed. Say \"speed on\" to bring the streaming back.")
	}
}

// printPaced prints s (which may contain many lines), pausing briefly
// between lines while the burst budget lasts. When pacing is off,
// non-interactive, or the text is short, it prints in one write.
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
	var spent time.Duration
	for i, ln := range lines {
		printRaw(ln)
		if i == len(lines)-1 {
			continue
		}
		d := paceDelayFor(visibleLen(ln))
		if spent+d > paceBurstBudget {
			continue // budget spent: the rest flows at full speed
		}
		spent += d
		paceSleep(d)
	}
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
