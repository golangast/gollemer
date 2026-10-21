package chat

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// teach.go — THE TRAINING TOUR. Most people will never read the
// README (John, 2026-10-21), so gollemer trains its user itself:
// six short lessons, each one following the same rhythm —
//
//	SAY   one plain sentence about what this skill is
//	SHOW  gollemer does it, live, on the project already open
//	TRY   the user says the phrase themselves and watches it work
//
// "next" skips the practice rep; any other question is answered
// normally and the lesson waits (the guide's interrupt rule). The
// Friend's notebook remembers how many lessons are done, so the
// tour resumes across sessions and, once finished, is offered
// no more. Triggers: "teach me", "training", "tutorial".

type lesson struct {
	title   string
	say     string // what this skill is, in one breath
	tryHint string // the phrase to practice
	brain   string
	try     func(low string) bool
	show    func()
}

var (
	teachActive    bool
	teachIdx       int // next lesson to present
	teachPresented bool
	teachPending   func() // deferred note, flushed at the next turn
)

// lessonList builds the curriculum against the currently open
// project (the demo when nothing else is), so the SHOW steps can
// name real functions.
func lessonList() []lesson {
	p := lastAnalyzeProject
	top := topFuncName(p)
	second := secondFuncName(p, top)
	return []lesson{
		{
			title:   "Ask what a project does",
			say:     "You never have to read a README here. Ask me what a project does and I'll tell you its story in plain English — what it is, what it does, where the important parts live.",
			tryHint: "what does this program do",
			brain:   "analyst",
			try:     func(low string) bool { return strings.Contains(low, "what does") },
			show: func() {
				printPaced(p.Summary() + "\n" + p.Story() + "\n")
			},
		},
		{
			title:   "Ask about one function",
			say:     "Pick any function name and ask what it does. I'll show you it being used for real — actual call sites from the code, not a guess.",
			tryHint: "example " + top,
			brain:   "analyst",
			try: func(low string) bool {
				if top == "" {
					return false
				}
				return strings.Contains(low, strings.ToLower(top)) &&
					(strings.Contains(low, "example") || strings.Contains(low, "what does"))
			},
			show: func() {
				fn := findFuncByQuery(p, top)
				if fn == nil {
					fmt.Println("This project is small — the story above is most of it. The next lessons still apply.")
					return
				}
				fmt.Printf("Meet %s — called from %d place(s), more than anything else here:\n\n", fn.Name, len(fn.Callers))
				printPaced(p.ExamplesText(fn))
				friendNoteFunc(fn.Name)
			},
		},
		{
			title:   "Ask why, and compare",
			say:     "Two power moves: ask WHY a function does something (why it locks, why it can fail) and I'll answer from the code itself — and ask me to compare two functions and I'll lay them side by side.",
			tryHint: "compare " + top + " and " + second,
			brain:   "reasoner",
			try: func(low string) bool {
				return strings.Contains(low, "why") || strings.Contains(low, "compare")
			},
			show: func() {
				a, b := findFuncByQuery(p, top), findFuncByQuery(p, second)
				if a != nil && b != nil {
					printPaced(p.CompareText(a, b) + "\n")
				} else {
					fmt.Println("This project has a single dominant function, so there's nothing fair to compare it with yet. The phrase still works anywhere: \"compare X and Y\".")
				}
			},
		},
		{
			title:   "See it run",
			say:     "Reading about code only goes so far. Say \"run it\" and I'll start the project right here — asking first, always — and narrate what happens in plain words.",
			tryHint: "run it",
			brain:   "runner",
			try:     func(low string) bool { return strings.Contains(low, "run it") },
			show: func() {
				fmt.Println("I won't run it during the lesson — the point is the phrase is yours to fire whenever you're ready. It works on this project whenever you say it.")
			},
		},
		{
			title:   "Get unstuck",
			say:     "When you're lost, don't reach for a manual. Say \"what should I do now\" and I'll name the next step from where you stand. Paste any Go error and I'll translate it into plain words.",
			tryHint: "what should I do now",
			brain:   "coordinator",
			try:     func(low string) bool { return strings.Contains(low, "what should") || strings.Contains(low, "stuck") },
			show: func() {
				handleWhatNow()
			},
		},
		{
			title:   "Meet the team",
			say:     "Last one. A crew of small specialists answers you — ask \"brains\" to meet them and \"how are your brains doing\" for the scoreboard. And the Friend quietly remembers how you work, so \"any suggestions\" gets smarter over time.",
			tryHint: "any suggestions",
			brain:   "friend",
			try: func(low string) bool {
				return strings.Contains(low, "suggestion") || low == "brains" || strings.Contains(low, "brains doing")
			},
			show: func() {
				printBrains()
			},
		},
	}
}

// secondFuncName picks a comparison partner: the next most-called
// function after the top one.
func secondFuncName(p *analyze.Project, top string) string {
	best, bestN := "", -1
	if p == nil {
		return "it"
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest || fn.Name == top || fn.Name == "" {
				continue
			}
			if n := len(fn.Callers); n > bestN {
				best, bestN = fn.Name, n
			}
		}
	}
	if best == "" {
		return top
	}
	return best
}

// handleTeach starts (or resumes) the training tour. Saying it
// while a tour is already running restarts from the top — the
// resume message promises exactly that.
func handleTeach() {
	if !ensureProject() {
		return
	}
	if teachActive {
		teachActive = true
		teachIdx = 0
		teachPresented = false
		teachPending = nil
		fmt.Println("Restarting the training tour from the top.")
		teachShow()
		return
	}
	teachActive = true
	teachIdx = 0
	if done := friendLoad().Taught; done > 0 && done < len(lessonList()) {
		teachIdx = done
		fmt.Printf("Welcome back — you'd finished %d of %d lessons. Picking up at lesson %d. (Say \"teach me\" again anytime to restart.)\n", done, len(lessonList()), done+1)
	} else if done >= len(lessonList()) {
		fmt.Println("You've finished the training tour before — running it again as a refresher, from the top.")
	} else {
		fmt.Println("Welcome to the training tour. Six tiny lessons, each one: I show you a thing I can do, then YOU say the phrase and watch it work. No reading, I promise.")
		fmt.Println("Say \"next\" to skip a practice rep, ask anything anytime (it won't break the tour), and \"stop teaching\" ends it.")
	}
	teachShow()
}

// teachShow presents the current lesson: band, SAY, SHOW, TRY.
func teachShow() {
	lessons := lessonList()
	if teachIdx >= len(lessons) {
		teachFinish()
		return
	}
	l := lessons[teachIdx]
	printBand(fmt.Sprintf("Lesson %d of %d — %s", teachIdx+1, len(lessons), l.title))
	fmt.Println(l.say)
	fmt.Println()
	l.show()
	fmt.Printf("\n%s Your turn — say: %s\n", cHead("TRY IT:"), cSay("\""+l.tryHint+"\""))
	teachPresented = true
	recordBrain(l.brain, true, "lesson: "+l.title)
}

// teachBefore runs at the top of every handled line (the wrapper
// calls it before dispatch). It flushes the deferred note from a
// completed rep, then checks this line against the current lesson's
// practice rep. A match prints the "watch it work" line NOW and
// queues the completion note for the next turn — the real answer,
// produced by the normal handlers, lands in between.
func teachBefore(line string) {
	if teachPending != nil {
		note := teachPending
		teachPending = nil
		note()
	}
	if !teachActive || !teachPresented || teachIdx >= len(lessonList()) {
		return
	}
	low := strings.Trim(strings.ToLower(strings.TrimSpace(line)), ".!? ")
	switch low {
	case "next", "next lesson", "stop teaching", "teach me":
		return // control phrases are not reps
	}
	l := lessonList()[teachIdx]
	if !l.try(low) {
		return
	}
	fmt.Printf("%s That's exactly it — watch it work:\n\n", cGood("✓"))
	teachIdx++
	teachPresented = false
	done, total := teachIdx, len(lessonList())
	m := friendLoad()
	m.Taught = done
	friendSave()
	if done < total {
		teachPending = func() {
			fmt.Printf("%s Lesson %d of %d done. Say %s for the next one — or just keep asking me things.\n",
				cGood("✓"), done, total, cSay("\"next lesson\""))
		}
	} else {
		teachPending = teachFinish
	}
}

// teachNextLesson presents the next lesson; a lesson that was
// presented but whose rep was skipped counts as done.
func teachNextLesson() bool {
	if !teachActive {
		return false
	}
	if teachPresented {
		teachIdx++
		m := friendLoad()
		m.Taught = teachIdx
		friendSave()
		teachPresented = false
	}
	if teachIdx >= len(lessonList()) {
		teachFinish()
		return true
	}
	teachShow()
	return true
}

func teachFinish() {
	teachActive = false
	teachPresented = false
	m := friendLoad()
	m.Taught = len(lessonList())
	friendSave()
	printBand("Training complete")
	fmt.Println("That's the whole tour — you can ask what a project does, interrogate any function,")
	fmt.Println("run things, get unstuck, and check on the team. Everything else is just asking")
	fmt.Println("in your own words: \"help all\" has the full menu when you want it.")
	fmt.Println("And the Friend is watching for patterns — \"any suggestions\" hears its notebook.")
	recordBrain("teacher", true, "training tour finished")
}

func teachStop() {
	if teachActive {
		teachActive = false
		teachPresented = false
		fmt.Println("Training paused — the Friend's notebook remembers your lessons, so \"teach me\" picks up where you left off.")
	} else {
		fmt.Println("No training tour is running. Say \"teach me\" whenever you want it.")
	}
}

// offerTeach prints the first-run invitation: shown at startup only
// until the tour is finished (or well begun), because a beginner who
// won't read a README might still take a two-minute tour.
func offerTeach() {
	m := friendLoad()
	if m.Taught >= len(lessonList()) {
		return
	}
	if m.Sessions > 3 && len(m.FuncAsks) > 0 {
		return // clearly not a beginner anymore
	}
	fmt.Printf("\n%s New here? Say %s and I'll teach you what I can do by showing you — two minutes, no reading.\n",
		cHead("✦"), cSay("\"teach me\""))
}
