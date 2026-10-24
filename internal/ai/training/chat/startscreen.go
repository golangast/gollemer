package chat

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/pkg/termui"
)

// startscreen.go — the unified chat's whole launch face since the
// 2026-10-24 readability pass: ONE termui panel, printed instantly,
// then a short capability block that types itself out beneath it
// through the normal pacing. The name in the border, a three-line
// gopher, one line of what this is, three starter lines, a status-
// chip row — and, when the local stores have something TRUE to add
// for this project, his own starters and one continuity line.
// Everything else the old startup used to dump — the phrase catalog,
// the teach-me pitch, profile trivia — lives behind "help" or a
// single line here. Print less.

// capabilityLines is what Gollemer can do, one short line per
// family, in beginner words — derived from the README's "What
// Gollemer is for" so the product describes itself the same way
// everywhere. Generic product truth, not personal data: it shows
// on every launch, fresh install included.
var capabilityLines = []string{
	"Reads Go code and explains it in plain English — what it does, where to start.",
	"Finds where a change belongs; edits happen only with your yes — renames and error fixes too.",
	"Answers questions about the code you're in — what calls what, where data flows.",
	"Teaches Go from this project itself — a guided tour and short lessons on your real code.",
}

// capabilityText renders the capability block (a Prose unit).
func capabilityText() string {
	return termui.Prose(strings.Join(capabilityLines, "\n"))
}

// genericStarters are the stock starter lines, in order — the fill
// for any slot no true personal starter claims.
var genericStarters = []string{
	"start — a six-stop guided tour of this project",
	"ask in plain words — \"what does this program do\"",
	"help — everything you can say · /quit — leave",
}

// startPersonal is everything the local stores can truthfully add
// to THIS launch's screen for THIS project. It is derived
// READ-ONLY: rendering never writes the profile, never advances
// the Friend's WatchRead cursor, and never consumes a pending
// suggestion — the flows that own those state changes (the guided
// walk, the Friend's notes, the Watcher digest) behave exactly as
// before. Silence is the spec: a slot with nothing true behind it
// stays generic or stays empty.
type startPersonal struct {
	starters    []string // personal starter bodies, priority order
	continuity  string   // at most one line; "" = nothing to say
	teachResume bool     // a starter resumes the training tour
	guidedTour  bool     // a starter resumes the guided walk
}

// startPersonalFor reads the three stores (profile, Friend
// notebook, Watcher log — small local JSON reads only) and works
// out this project's personal starters and continuity line.
func startPersonalFor(root string) startPersonal {
	var out startPersonal
	prof := LoadProfile()
	fm := friendLoad()

	// Starter priority: his most-run command, then a guided-walk
	// bookmark for this project, then a half-finished training
	// tour. At most three personal starters; generic lines fill
	// whatever slots remain (see startScreenParts).
	if cmd := mostRunCommand(prof); cmd != "" {
		body := "your most-run command — " + cmd
		if strings.HasPrefix(cmd, "make ") {
			// "run make <target>" is the phrase chat acts on.
			body = "run " + cmd + " — your most-run command"
		}
		out.starters = append(out.starters, body)
	}
	if next, ok := guidedBookmarkFor(fm, root); ok {
		out.starters = append(out.starters, fmt.Sprintf(
			"start — continue the tour · you left off at stop %d of %d",
			next+1, len(guideCurriculum)))
		out.guidedTour = true
	}
	if total := len(lessonList()); fm.Taught > 0 && fm.Taught < total {
		out.starters = append(out.starters, fmt.Sprintf(
			"teach me — pick up the training tour · lesson %d of %d",
			fm.Taught+1, total))
		out.teachResume = true
	}
	if len(out.starters) > 3 {
		out.starters = out.starters[:3]
	}

	out.continuity = startContinuity(root, prof, fm)
	return out
}

// mostRunCommand returns his most-run recorded command — the
// profile's FrequentCommands is one global list (the schema has no
// per-project split to read). A command must recur (count ≥ 2) to
// count as a habit; ties break toward the most recently used.
func mostRunCommand(p *UserProfile) string {
	best := CommandStat{}
	for _, s := range p.FrequentCommands {
		if s.Count < 2 {
			continue
		}
		if s.Count > best.Count ||
			(s.Count == best.Count && s.LastUsed.After(best.LastUsed)) {
			best = s
		}
	}
	return truncateRunes(strings.TrimSpace(best.Cmd), 56)
}

// guidedBookmarkFor finds this project's guided-walk bookmark in
// the Friend's notebook: the stored value is the next stop's index
// (the same arithmetic handleStart resumes with), present only
// while the walk is unfinished.
func guidedBookmarkFor(m *friendMemory, root string) (int, bool) {
	for key, v := range m.Guided {
		if v > 0 && v < len(guideCurriculum) && sameProjectRoot(key, root) {
			return v, true
		}
	}
	return 0, false
}

// projectMemoryFor finds this project's record in the profile.
func projectMemoryFor(p *UserProfile, root string) *ProjectMemory {
	for key, pm := range p.Projects {
		if pm != nil && sameProjectRoot(key, root) {
			return pm
		}
	}
	return nil
}

// startContinuity picks AT MOST ONE continuity line, by truth
// priority: (a) Watcher changes for this project since the Friend
// last read them, (b) this project's recorded focus, (c) the
// Friend's welcome-back note when its own rules say it is due.
func startContinuity(root string, prof *UserProfile, fm *friendMemory) string {
	if line := watchContinuity(root, fm.WatchRead); line != "" {
		return line
	}
	if pm := projectMemoryFor(prof, root); pm != nil &&
		strings.TrimSpace(pm.Focus) != "" {
		return "last time: " + truncateRunes(strings.TrimSpace(pm.Focus), 72)
	}
	// (c) mirrors friendBeginSession's due rules exactly — a new
	// day, a remembered topic, and nothing already queued (the
	// one-pending rule). Display only: the Friend still queues and
	// speaks its own note through its own flow.
	if friendPending == "" && fm.LastDay != "" &&
		fm.LastDay != friendToday() && fm.LastTopic != "" {
		return "welcome back — last time you were deep in " +
			truncateRunes(fm.LastTopic, 40) + "."
	}
	return ""
}

// watchContinuity counts the distinct functions the Watcher has
// logged for this project since the Friend's read cursor — the
// same unread set friendWatchNotes digests. The "since" day is
// the cursor's day, or the oldest unread event's when the Friend
// has never read the log.
func watchContinuity(root, cursor string) string {
	funcs := map[string]bool{}
	var earliest time.Time
	for _, e := range readWatchLog() {
		if !sameProjectRoot(e.Root, root) {
			continue
		}
		if cursor != "" && e.T <= cursor {
			continue
		}
		funcs[e.Func] = true
		if t, err := time.Parse(time.RFC3339, e.T); err == nil &&
			(earliest.IsZero() || t.Before(earliest)) {
			earliest = t
		}
	}
	if len(funcs) == 0 {
		return ""
	}
	since := ""
	if t, err := time.Parse(time.RFC3339, cursor); err == nil {
		since = t.Format("Jan 2")
	} else if !earliest.IsZero() {
		since = earliest.Format("Jan 2")
	}
	noun := "functions"
	if len(funcs) == 1 {
		noun = "function"
	}
	if since != "" {
		return fmt.Sprintf("%d %s changed here since %s.", len(funcs), noun, since)
	}
	return fmt.Sprintf("%d %s changed here since you were last here.", len(funcs), noun)
}

// sameProjectRoot reports whether two path spellings name the same
// project folder (relative vs absolute spellings of one root).
func sameProjectRoot(a, b string) bool {
	if a == b {
		return true
	}
	ca, cb := absClean(a), absClean(b)
	return ca != "" && ca == cb
}

func absClean(p string) string {
	if p == "" {
		return ""
	}
	abs, err := filepath.Abs(p)
	if err != nil {
		return filepath.Clean(p)
	}
	return abs
}

// truncateRunes shortens s to at most n runes, with an ellipsis.
func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}

// startScreenParts composes the launch face: the panel (printed
// instantly, whole) and the capability block (paced beneath it on
// a terminal, printed whole when piped). Two parts because a box
// must never be caught half-drawn: the panel lands complete, and
// only the prose below it streams.
func startScreenParts(projectRoot string) (panel, capabilities string) {
	folder := filepath.Base(projectRoot)
	if folder == "" || folder == "." {
		folder = "this folder"
	}
	pers := startPersonalFor(projectRoot)

	bodies := append([]string{}, pers.starters...)
	for _, g := range genericStarters {
		if len(bodies) >= 3 {
			break
		}
		if pers.guidedTour && strings.HasPrefix(g, "start —") {
			continue // the resume starter IS the start line, personalized
		}
		bodies = append(bodies, g)
	}

	var lines []string
	lines = append(lines, "  ( o o )   a small Go companion — reads your code,")
	lines = append(lines, "   \\___/    explains it in plain English")
	lines = append(lines, "")
	for _, b := range bodies {
		lines = append(lines, "  "+termui.Accent.Apply(termui.SubMark)+" "+b)
	}
	// The teach-me line keeps its offer conditions — but when a
	// starter already resumes the tour, the offer would say the
	// same thing twice, so it stands down.
	if teachOfferDue() && !pers.teachResume {
		lines = append(lines, "  "+cHead("✦")+" new here? say "+cSay("\"teach me\"")+" — two minutes, learn by doing")
	}
	if pers.continuity != "" {
		lines = append(lines, "  "+termui.Muted.Apply(pers.continuity))
	}
	lines = append(lines, "")
	lines = append(lines, "  "+termui.Badge(folder, termui.Accent)+"  "+
		termui.Badge(fmt.Sprintf("%d brains ready", len(brainRegistry)), termui.Good))
	return termui.Panel("GOLLEMER", strings.Join(lines, "\n"), termui.Bright),
		capabilityText()
}

// startScreenText renders the whole launch face as one string
// (pure, so tests can pin its shape; plain mode carries all of it,
// in order, with no delays).
func startScreenText(projectRoot string) string {
	panel, caps := startScreenParts(projectRoot)
	if caps == "" {
		return panel
	}
	return panel + "\n" + caps
}

// printStartScreen shows the launch face once: the panel lands in
// one write, then the capability block types itself out beneath
// it through the chat's pacing (printPaced — instant and complete
// when piped or pacing is off).
func printStartScreen(projectRoot string) {
	panel, caps := startScreenParts(projectRoot)
	fmt.Println(panel)
	if caps != "" {
		printPaced(caps + "\n")
	}
}
