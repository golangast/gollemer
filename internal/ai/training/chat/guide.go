package chat

import (
	"fmt"
	"sort"
	"strings"
)

// guide.go — THE GUIDED START. A beginner does not arrive with a
// question; they arrive lost. Instead of a catalog to choose from, the
// guide DRIVES: a fixed curriculum over the open project — the story,
// the hardest-working function, the key words, a look at it running,
// the team (including the Friend), and the first concept to learn —
// one stop at a time, advanced with "next", interruptible with any
// question, resumable across sessions because the Friend's notebook
// remembers how far each project got.

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
)

// handleStart begins (or resumes) the guided walk for the open project.
func handleStart(restart bool) {
	if !ensureProject() {
		return
	}
	p := lastAnalyzeProject
	guideRoot = p.Root
	guideActive = true
	start := 0
	if !restart {
		if saved := friendLoad().Guided[p.Root]; saved > 0 && saved < len(guideCurriculum) {
			start = saved
		}
	}
	guideIdx = start
	if start > 0 {
		fmt.Printf("Welcome back — you left the guided walk at stop %d of %d for this project. Picking it up there. (Say \"start over\" to redo it from the top.)\n", start+1, len(guideCurriculum))
	} else {
		fmt.Println("I'll drive. Six short stops: the story of this project, its hardest-working function, its key words, a look at it running, the team behind the answers — and your first concept to learn.")
		fmt.Println("Just say \"next\" to move on. Ask questions anytime — they don't break the walk. \"stop guiding\" ends it.")
	}
	guideShow()
}

// guideShow prints the current step and advances the bookmark. The
// Friend remembers the bookmark (per project) across sessions.
func guideShow() {
	p := lastAnalyzeProject
	if p == nil || guideIdx >= len(guideCurriculum) {
		guideFinish()
		return
	}
	step := guideCurriculum[guideIdx]
	fmt.Printf("\n--- Guided stop %d of %d: %s ---\n", guideIdx+1, len(guideCurriculum), step.title)
	switch step.kind {
	case "story":
		fmt.Println(p.Summary())
		fmt.Println(p.Story())
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
		fmt.Println(p.ExamplesText(fn))
		friendNoteFunc(fn.Name)
	case "words":
		fmt.Println(p.GlossaryText())
	case "run":
		fmt.Println("Reading about code only goes so far — say \"run it\" and I'll start this project here (make target, entry point, or its tests) and narrate what happens, 60 seconds at most.")
		fmt.Println("Not ready? Say \"next\" and we'll skip the show.")
	case "friend":
		fmt.Println(guideFriendText())
	case "concept":
		fmt.Println("Concepts in this project, in the order they build on each other — here is your first one:")
		fmt.Println(firstLines(p.LearnPathText(), 10))
	}
	recordBrain(step.brain, true, "guided stop: "+step.kind)
	guideIdx++
	m := friendLoad()
	if m.Guided == nil {
		m.Guided = map[string]int{}
	}
	m.Guided[guideRoot] = guideIdx
	friendSave()
	if guideIdx < len(guideCurriculum) {
		fmt.Println("(say \"next\" for the next stop)")
	}
}

// guideNext advances the walk; returns false when no walk is active.
func guideNext() bool {
	if !guideActive {
		return false
	}
	if guideIdx >= len(guideCurriculum) {
		guideFinish()
		return true
	}
	guideShow()
	return true
}

func guideFinish() {
	guideActive = false
	m := friendLoad()
	if m.Guided != nil {
		delete(m.Guided, guideRoot)
		friendSave()
	}
	fmt.Println("\nThat's the guided walk — you know the story, the engine room, the words, and the team.")
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
