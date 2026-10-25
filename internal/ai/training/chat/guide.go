package chat

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/pkg/termui"
)

// guide.go — THE GUIDED START. A beginner does not arrive with a
// question; they arrive lost. Instead of a catalog to choose from, the
// guide DRIVES: a fixed curriculum over the open project — the story,
// the hardest-working function, the key words, a look at it running,
// the team (including the Friend), and the first concept to learn —
// one stop at a time, advanced with "next", interruptible with any
// question, resumable across sessions because the Friend's notebook
// remembers how far each project got.
//
// The curriculum is per-project: the Friend stop is about
// Gollemer's own cast, so it runs only when the project being
// walked IS gollemer. A tour of any other project has five stops
// and never detours into how this tool works. The stops share
// the project tour's header ("Stop N of M — title"), and only
// one walk of any kind is ever in flight: starting one ends the
// others, so "next" can never wander between walks.

type guideStep struct {
	kind  string // story | example | words | run | friend | concept
	title string
	brain string // who gets the credit on the coordinator's scoreboard
}

var guideCurriculum = []guideStep{
	{"story", "The story — what this program is and does", "analyst"},
	{"example", "The engine room — its hardest-working function, used for real", "analyst"},
	{"words", "The words — the names everything else is built from", "lexicon"},
	{"run", "See it move", "runner"},
	{"friend", "The quiet member of the team — the Friend", "friend"},
	{"concept", "Your first concept to learn", "teacher"},
}

var (
	guideActive bool
	guideIdx    int // index of the NEXT step to show
	guideRoot   string
	guideSteps  []guideStep // the curriculum of the walk in flight
)

// gollemerModule is this tool's own module path: the walk over
// gollemer itself is the self-tour, the one place stops about
// Gollemer's own cast belong.
const gollemerModule = "github.com/golangast/gollemer"

// guideStepsFor returns the walk for a project: the full
// curriculum for gollemer itself, the same list without the
// Friend stop for any other project.
func guideStepsFor(module string) []guideStep {
	if module == gollemerModule {
		return guideCurriculum
	}
	steps := make([]guideStep, 0, len(guideCurriculum)-1)
	for _, s := range guideCurriculum {
		if s.kind != "friend" {
			steps = append(steps, s)
		}
	}
	return steps
}

// activeSteps is the in-flight curriculum, falling back to the
// full list when no walk has set one.
func activeSteps() []guideStep {
	if guideSteps != nil {
		return guideSteps
	}
	return guideCurriculum
}

// guideStepCount quotes a project's walk length before any walk
// starts (the start screen's resume line): the root's own
// go.mod decides which curriculum the project gets.
func guideStepCount(root string) int {
	if data, err := os.ReadFile(filepath.Join(root, "go.mod")); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "module ") {
				if strings.TrimSpace(strings.TrimPrefix(line, "module ")) == gollemerModule {
					return len(guideCurriculum)
				}
				return len(guideCurriculum) - 1
			}
		}
	}
	return len(guideCurriculum)
}

// handleStart begins (or resumes) the guided walk for the open project.
func handleStart(restart bool) {
	if !ensureProject() {
		return
	}
	p := lastAnalyzeProject
	guideRoot = p.Root
	guideSteps = guideStepsFor(p.Module)
	// One walk at a time: starting the guided walk ends any
	// project tour or teach session in flight, so "next" can
	// only ever advance this walk. Both walks keep their place
	// in the Friend's notebook, so "tour" and "start" resume
	// where they left off — nothing is lost by switching.
	tourStops, tourIdx = nil, 0
	teachActive = false
	guideActive = true
	start := 0
	if !restart {
		if saved := friendLoad().Guided[p.Root]; saved > 0 && saved < len(guideSteps) {
			start = saved
		}
	}
	guideIdx = start
	if start > 0 {
		fmt.Printf("Welcome back — you left the guided walk at stop %d of %d for this project. Picking it up there. (Say \"start over\" to redo it from the top.)\n", start+1, len(guideSteps))
	} else if len(guideSteps) == len(guideCurriculum) {
		fmt.Println("I'll drive. Six short stops: the story of this project, its hardest-working function, its key words, a look at it running, the team behind the answers — and your first concept to learn.")
		fmt.Println("Just say \"next\" to move on. Ask questions anytime — they don't break the walk. \"stop guiding\" ends it.")
	} else {
		fmt.Println("I'll drive. Five short stops: the story of this project, its hardest-working function, its key words, a look at it running — and your first concept to learn.")
		fmt.Println("Just say \"next\" to move on. Ask questions anytime — they don't break the walk. \"stop guiding\" ends it.")
	}
	guideShow()
}

// guideShow prints the current step and advances the bookmark. The
// Friend remembers the bookmark (per project) across sessions.
// Stops stay short: the capped stops show their first screen and
// point at the command that shows the rest (the story stop is
// the one exception — it is the walk's reason to exist).
func guideShow() {
	p := lastAnalyzeProject
	steps := activeSteps()
	if p == nil || guideIdx >= len(steps) {
		guideFinish()
		return
	}
	step := steps[guideIdx]
	printBand(fmt.Sprintf("Stop %d of %d — %s", guideIdx+1, len(steps), step.title))
	switch step.kind {
	case "story":
		printPaced(p.Summary() + "\n" + p.Story() + "\n")
	case "example":
		tf := topFuncName(p)
		if tf == "" {
			fmt.Println("This project is small enough that the story above is most of it — no single function dominates.")
			break
		}
		fn := findFuncByQuery(p, tf)
		if fn == nil {
			fmt.Println("No function stands out yet.")
			break
		}
		fmt.Printf("Meet %s — called from %d place(s), more than anything else here. This is the engine room:\n\n", fn.Name, len(fn.Callers))
		ex := p.ExamplesText(fn)
		if n := strings.Count(ex, "\n"); n > 12 {
			ex = firstLines(ex, 12) + "\n… the whole example: say \"/example " + fn.Name + "\"."
		}
		fmt.Println(ex)
		friendNoteFunc(fn.Name)
	case "words":
		g := p.GlossaryText()
		if n := strings.Count(g, "\n"); n > 10 {
			g = firstLines(g, 10) + "\n… the rest of the dictionary: say \"/glossary\" for all of it."
		}
		fmt.Println(g)
	case "run":
		fmt.Println("Reading about code only goes so far — say \"run it\" and I'll start this project here (make target, entry point, or its tests) and narrate what happens, 60 seconds at most.")
		fmt.Println("Not ready? Say \"next\" and we'll skip the show.")
	case "friend":
		fmt.Println(decorateOutput(guideFriendText()))
	case "concept":
		fmt.Println("The concepts worth meeting first — the ones the program itself starts from:")
		fmt.Println(beginnerConceptsText(p))
	}
	recordBrain(step.brain, true, "guided stop: "+step.kind)
	guideIdx++
	m := friendLoad()
	if m.Guided == nil {
		m.Guided = map[string]int{}
	}
	m.Guided[guideRoot] = guideIdx
	friendSave()
	if guideIdx < len(steps) {
		fmt.Println("(say \"next\" for the next stop)")
	}
}

// beginnerConceptsText renders the concepts first screen: at
// most four concepts in beginner order (analyze.Beginner
// Concepts), each at most two lines — the concept itself and at
// most one Go-idea link, the primary one — with the rest of the
// path one question away, never deleted.
func beginnerConceptsText(p *analyze.Project) string {
	all := p.BeginnerConcepts(0)
	if len(all) == 0 {
		return "No exported types stood out — this project's concepts live in its functions. The story above is the way in."
	}
	screen := all
	if len(screen) > 4 {
		screen = screen[:4]
	}
	var b strings.Builder
	for i, c := range screen {
		fmt.Fprintf(&b, "%s %s%s\n", termui.Gold.Apply(fmt.Sprintf("%d.", i+1)),
			termui.Bright.Apply(c.Name), termui.Muted.Apply(" — "+c.OneLiner))
		if len(c.Features) > 0 {
			f := c.Features[0]
			fmt.Fprintf(&b, "   %s %s.\n", termui.Muted.Apply("Go idea it uses:"), f[0]+" ("+f[1]+")")
		}
	}
	if len(all) > len(screen) {
		fmt.Fprintf(&b, "… and %d more, in the order they build on each other — say \"what should I learn first\" for the whole path.\n", len(all)-len(screen))
	}
	return strings.TrimRight(b.String(), "\n")
}

// guideNext advances the walk; returns false when no walk is active.
func guideNext() bool {
	if !guideActive {
		return false
	}
	if guideIdx >= len(activeSteps()) {
		guideFinish()
		return true
	}
	guideShow()
	return true
}

func guideFinish() {
	guideActive = false
	guideSteps = nil
	m := friendLoad()
	if m.Guided != nil {
		delete(m.Guided, guideRoot)
		friendSave()
	}
	fmt.Println("\nThat's the guided walk — you know the story, the engine room, the words, and your first concept.")
	fmt.Println("From here it's yours: ask anything in your own words.")
	fmt.Println("  stuck?            \"what should I do now\"")
	fmt.Println("  curious?          \"why does <name> lock\"  ·  \"compare <name> and <name>\"")
	fmt.Println("  the whole menu?   \"help all\"")
	fmt.Println("And the Friend is watching for patterns — \"any suggestions\" hears its notebook.")
}

func guideStop() {
	if guideActive {
		guideActive = false
		fmt.Println("Guided walk ended — your place is saved in the Friend's notebook, so \"start\" picks it back up. Ask me anything.")
	} else {
		fmt.Println("No guided walk is running. Say \"start\" whenever you want the tour with training wheels.")
	}
}

// guideFriendText is the Friend's stop on the walk: what it is, what
// it can do, why it is quiet — and, for a returning user, what it
// already knows about them.
func guideFriendText() string {
	var b strings.Builder
	b.WriteString("Every brain so far answers questions. The Friend does something else: it watches how YOU work.\n\n")
	b.WriteString("What it can do:\n")
	b.WriteString("• It remembers — which functions you ask about, which brains help you, where you circle back — in a notebook on this machine (~/.gollemer_friend.json). It survives between sessions and is never uploaded.\n")
	b.WriteString("• It notices patterns: the same function asked three times, a brain failing you twice in a row, deep digging with no map, a project explored but never run.\n")
	b.WriteString("• It speaks rarely, on purpose — one thought, then at least ten turns of quiet — because a helper that nags is a helper you mute.\n")
	b.WriteString("• It reads the Watcher's log, so what changed in your code while you were away can greet you when you return.\n")

	m := friendLoad()
	var tops []string
	type kv struct {
		k string
		v int
	}
	var asks []kv
	for k, v := range m.FuncAsks {
		asks = append(asks, kv{k, v})
	}
	sort.Slice(asks, func(i, j int) bool { return asks[i].v > asks[j].v })
	for i := 0; i < len(asks) && i < 2; i++ {
		tops = append(tops, fmt.Sprintf("%s (%d×)", asks[i].k, asks[i].v))
	}
	if len(tops) > 0 {
		fmt.Fprintf(&b, "\nIts notebook on you so far: %d session(s) together; you return to %s. Say \"any suggestions\" and it reads the notebook aloud.", m.Sessions, strings.Join(tops, ", "))
	} else {
		b.WriteString("\nIts notebook on you is empty so far — it fills in as you work. Say \"any suggestions\" anytime to hear what it has.")
	}
	return b.String()
}

func firstLines(s string, n int) string {
	lines := strings.Split(s, "\n")
	if len(lines) > n {
		lines = lines[:n]
	}
	return strings.Join(lines, "\n")
}
