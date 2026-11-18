package chat

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/pkg/termui"
)

// learn.go — LEARN MODE. One paced flow that teaches the open (or
// named) project end to end, composed entirely from pieces the chat
// already has: the analyzer's story, its reading guide, the
// entry-point walkthrough, the learn-path concepts (with Advocates
// links), and a short check whose questions are generated from
// analyzer ground truth — call relations, dataflow sources, the
// entry point — and judged against those same facts. Questions the
// user misses are written to the Friend's notebook so a later
// feature can bring them back.
//
// The interaction follows the guided tour's rules (guide.go): one
// stop at a time, "next" advances, questions asked mid-flow are
// answered normally and the flow waits, and an exit phrase ends it
// gracefully. The claim lives in the unified loop before the
// beginner commands and the intent classifier, so while learn mode
// is active its "next" and its check answers can never fall through
// to a brain.

var learnStageTitles = []string{
	"The story — what this program is and does",
	"The map — where to start",
	"The walkthrough — the entry point, in plain words",
	"The concepts — the ideas in this code",
	"Check yourself — a few questions",
}

// learnCheck is one comprehension question generated from analyzer
// facts: the prompt, every name that counts as right (short or
// dotted), and the correct fact in plain words for a miss.
type learnCheck struct {
	concept  string // what the question is about: a function, a value, "the entry point"
	question string
	accept   []string
	fact     string
}

var (
	learnActive  bool
	learnStage   int // index of the NEXT stage to show
	learnRoot    string
	learnChecks  []learnCheck
	learnCheckAt = -1 // question awaiting an answer; -1 outside the check stage
	learnMissed  int  // misses this run, for the closing line
)

// tryLearnCommand is learn mode's claim in the unified loop. It
// returns true when the line belonged to the flow: a start shape, a
// "next" or exit phrase while active, or an answer while a check
// question is waiting.
func tryLearnCommand(line string) bool {
	if target, ok := parseLearnStart(line); ok {
		startLearn(target)
		return true
	}
	if !learnActive {
		return false
	}
	low := strings.Trim(strings.ToLower(strings.TrimSpace(line)), ".!? ")
	if isLearnExit(low) {
		stopLearn()
		return true
	}
	if learnCheckAt >= 0 {
		switch low {
		case "next", "skip", "skip it", "pass":
			skipLearnCheck()
			return true
		}
		if strings.HasPrefix(line, "/") || isHelpLine(line) || isHelpAllLine(line) {
			return false // a command or help renders; the question waits
		}
		judgeLearnCheck(line)
		return true
	}
	if low == "next" {
		learnShow()
		return true
	}
	return false // any other line dispatches normally; the flow waits
}

// parseLearnStart matches the entry shapes: "learn" (the current
// project), "learn <github url>", "learn <folder>" when the folder
// exists, and the /learn spellings. It deliberately does NOT match
// /learn-path or /learn-pieces (older, separate features), nor their
// sentence forms "learn path" / "learn pieces", nor a target that
// resolves to nothing — those lines keep their existing owners.
func parseLearnStart(line string) (string, bool) {
	t := strings.TrimSpace(line)
	switch strings.ToLower(t) {
	case "learn", "/learn", "learn mode", "start learning":
		return "", true
	}
	for _, pre := range []string{"learn ", "/learn "} {
		rest, ok := cutPrefixFold(t, pre)
		if !ok {
			continue
		}
		rest = strings.TrimSpace(rest)
		rl := strings.ToLower(rest)
		if rl == "" || rl == "path" || rl == "pieces" || strings.HasPrefix(rl, "-") {
			return "", false
		}
		if githubURL.MatchString(rest) || learnPathExists(rest) {
			return rest, true
		}
		return "", false
	}
	return "", false
}

// learnPathExists reports whether rest names a local directory.
func learnPathExists(rest string) bool {
	path := rest
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return false
		}
		path = filepath.Join(home, strings.TrimPrefix(path, "~"))
	}
	fi, err := os.Stat(path)
	return err == nil && fi.IsDir()
}

func isLearnExit(low string) bool {
	switch low {
	case "stop learning", "stop learn", "end learn", "end learning",
		"quit learning", "done learning", "that's enough", "thats enough", "enough":
		return true
	}
	return false
}

// startLearn resolves the flow's project and shows the first stage.
func startLearn(target string) {
	p := learnResolveProject(target)
	if p == nil {
		return // the resolver already said why, honestly
	}
	learnActive = true
	learnStage = 0
	learnRoot = lastAnalyzeRoot
	if learnRoot == "" {
		learnRoot = p.Root
	}
	learnChecks = nil
	learnCheckAt = -1
	learnMissed = 0
	name := p.Module
	if name == "" {
		name = filepath.Base(p.Root)
	}
	fmt.Printf("Let's learn %s — five short stops: its story, the map, a walkthrough of the entry point, the concepts in the code, and a few questions to check it stuck.\n", name)
	fmt.Println("Say \"next\" to move on. Ask questions anytime — they don't break the flow. \"stop learning\" ends it.")
	learnShow()
}

// learnResolveProject finds the project the flow teaches: the named
// target (resolved/cloned and analyzed exactly like analyze), else
// the open project, else the folder the chat is standing in when it
// holds Go code. Unlike ensureProject it never falls back to the
// bundled demo — teaching a stand-in project as if it were yours
// would be a lie; with nothing real to teach it says so.
func learnResolveProject(target string) *analyze.Project {
	if target != "" {
		root, err := resolveAnalyzeRoot(target)
		if err != nil {
			fmt.Printf("I couldn't read that codebase: %v\n", err)
			return nil
		}
		sp := termui.Spin("Reading this project…")
		p, err := analyze.Analyze(root)
		sp.Stop("")
		if err != nil {
			fmt.Printf("I couldn't analyze %s: %v\n", root, err)
			return nil
		}
		setCurrentProject(root, p)
		return p
	}
	if lastAnalyzeProject != nil {
		return lastAnalyzeProject
	}
	if root := startFolder(); root != "" && hasGoFiles(root) {
		fmt.Println("Let me look at the project in this folder first…")
		if analyzeQuiet(root) {
			return lastAnalyzeProject
		}
		return nil
	}
	fmt.Println("No project is open yet, and this folder doesn't hold one — open a project first: say \"analyze <github link or folder>\" (or \"learn <github link>\"), or run me inside one, and then we'll learn it together.")
	return nil
}

// learnShow prints the current stage and moves the bookmark.
func learnShow() {
	p := lastAnalyzeProject
	if p == nil || learnStage >= len(learnStageTitles) {
		learnFinish()
		return
	}
	printBand(fmt.Sprintf("Learn %d of %d — %s", learnStage+1, len(learnStageTitles), learnStageTitles[learnStage]))
	switch learnStage {
	case 0: // the story
		printPaced(p.Summary() + "\n" + p.Story() + "\n")
		recordBrain("analyst", true, "learn: story")
	case 1: // the map
		fmt.Print(p.ReadingGuide())
		recordBrain("analyst", true, "learn: map")
	case 2: // the walkthrough
		learnShowWalkthrough(p)
		recordBrain("analyst", true, "learn: walkthrough")
	case 3: // the concepts
		fmt.Print(learnConceptsText(p))
		recordBrain("teacher", true, "learn: concepts")
	case 4: // the check
		recordBrain("teacher", true, "learn: check")
		learnChecks = buildLearnChecks(p)
		if len(learnChecks) == 0 {
			fmt.Println("There's nothing solid enough in this project to quiz you on yet — no entry point, no clear call chains, no tracked data. The four stops above are the lesson.")
			learnFinish()
			return
		}
		learnCheckAt = 0
		learnAskCheck()
	}
	learnStage++
	if learnActive && learnCheckAt < 0 && learnStage < len(learnStageTitles) {
		fmt.Println("(say \"next\" for the next stop)")
	}
}

// learnShowWalkthrough walks the main entry point in plain words
// through the same machinery "walk me through X" uses. A library
// has no entry point to walk; the stage says so and the flow goes on.
func learnShowWalkthrough(p *analyze.Project) {
	eps := p.EntryPoints()
	if len(eps) == 0 {
		fmt.Println("This project has no entry point — it's a library, so nothing \"runs first\". The map above is the way in: start from the exported API of the root package.")
		return
	}
	fn := eps[0]
	if len(eps) > 1 {
		fmt.Printf("This project has %d entry points — we'll walk the first, %s in %s:\n\n", len(eps), fn.Display(), p.LinkPath(fn.File))
	} else {
		fmt.Printf("The program starts at %s in %s — here it is in plain words:\n\n", fn.Display(), p.LinkPath(fn.File))
	}
	if out, ok := p.Answer("walk me through " + fn.Display()); ok {
		fmt.Println(out)
		return
	}
	if src := p.SourceOf(fn, 40); src != "" {
		fmt.Println("The plain-words walkthrough couldn't be built for this one — here's the entry point itself:")
		fmt.Println(src)
		return
	}
	fmt.Println("I couldn't read the entry point's source back — the map above still shows where it lives.")
}

// learnConceptsText renders the concepts stage: the beginner
// screen (analyze.BeginnerConcepts — front-door order, at most
// four concepts, one Go-idea link each), plus learn mode's own
// teaching extras per concept: an Advocates link when the
// resources matcher has the topic, and a tiny example from the
// project's own code where the machinery provides one.
func learnConceptsText(p *analyze.Project) string {
	// The beginner screen, same as the tour's concept stop:
	// beginner-usefulness order (the program's own front door
	// first), at most four concepts, one Go-idea link each.
	// The full dependency-ordered path stays one question
	// away ("what should I learn first").
	concepts := p.BeginnerConcepts(0)
	if len(concepts) == 0 {
		return "No exported types to order into a path — this project's concepts live in its functions, not its types. The map above is the way in, and \"what does <name> do\" answers for any function.\n"
	}
	const maxShown = 4
	var b strings.Builder
	b.WriteString("The ideas this project is built from — starting with the ones the program itself starts from:\n\n")
	shown := concepts
	extra := 0
	if len(shown) > maxShown {
		extra = len(shown) - maxShown
		shown = shown[:maxShown]
	}
	for i, c := range shown {
		fmt.Fprintf(&b, "%s %s%s\n", termui.Gold.Apply(fmt.Sprintf("%d.", i+1)),
			termui.Bright.Apply(c.Name), termui.Muted.Apply(" — "+c.OneLiner))
		if len(c.BuildsOn) > 0 {
			fmt.Fprintf(&b, "   %s %s.\n", termui.Muted.Apply("Builds on:"), strings.Join(c.BuildsOn, ", "))
		}
		if len(c.Features) > 0 {
			f := c.Features[0]
			fmt.Fprintf(&b, "   %s %s.\n", termui.Muted.Apply("Go idea it uses:"), f[0]+" ("+f[1]+")")
		}
		if link := learnAdvocatesLink(c); link != nil {
			fmt.Fprintf(&b, "   %s %s — %s\n", termui.Muted.Apply("From your Advocates list:"), link.Label, link.URL)
		}
		b.WriteString(learnConceptExample(p, c))
	}
	if extra > 0 {
		fmt.Fprintf(&b, "\n… and %d more — say %s for the whole path.\n", extra, cSay("\"what should I learn first\""))
	}
	return b.String()
}

// learnAdvocatesLink finds the user's own reading on a concept: the
// resources matcher tried on the concept's Go features first (the
// words the Advocates doc is indexed by), then the concept name,
// with the same qualify bar the resource packs use (score ≥ 3).
// With no snapshot loaded, the built-in seed list answers instead.
func learnAdvocatesLink(c analyze.LearnConcept) *advocatesEntry {
	var goals []string
	for _, f := range c.Features {
		goals = append(goals, f[0])
	}
	if c.Type != nil {
		goals = append(goals, c.Type.Name)
	}
	if doc := advocates(); doc != nil {
		for _, g := range goals {
			matches := matchTopics(doc, g)
			rankMatches(matches)
			for _, m := range matches {
				if m.score >= 3 && len(m.entries) > 0 {
					e := m.entries[0]
					return &e
				}
			}
		}
		return nil
	}
	for _, g := range goals {
		if seed := matchSeed(g); len(seed) > 0 {
			e := seed[0]
			return &e
		}
	}
	return nil
}

// learnConceptExample draws a tiny example from the project's own
// code: the first lines of the concept's most-called method. No
// method, no example — the line is simply absent.
func learnConceptExample(p *analyze.Project, c analyze.LearnConcept) string {
	if c.Type == nil {
		return ""
	}
	var best *analyze.Func
	for _, fn := range p.AllFuncs() {
		if fn.IsTest || fn.PkgDir != c.Type.PkgDir || strings.Trim(fn.Receiver, "*") != c.Type.Name {
			continue
		}
		if best == nil || len(fn.Callers) > len(best.Callers) {
			best = fn
		}
	}
	if best == nil {
		return ""
	}
	src := p.SourceOf(best, 5)
	if src == "" {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "   %s %s:\n", termui.Muted.Apply("In this project —"), best.Display())
	for _, l := range strings.Split(strings.TrimRight(src, "\n"), "\n") {
		b.WriteString("     " + l + "\n")
	}
	return b.String()
}

// --- The check --------------------------------------------------------

// buildLearnChecks generates 2–3 comprehension questions from
// analyzer ground truth about this project: what runs first (the
// entry point), who calls the engine room (call relations), and
// where a tracked value gets its data (the varflow index). A kind
// with no facts behind it is simply not asked.
func buildLearnChecks(p *analyze.Project) []learnCheck {
	var checks []learnCheck
	// 1. The entry point.
	if eps := p.EntryPoints(); len(eps) > 0 {
		e := eps[0]
		checks = append(checks, learnCheck{
			concept:  "the entry point",
			question: "What runs first when this program starts?",
			accept:   []string{"main", e.Display()},
			fact:     fmt.Sprintf("func main runs first — it's at %s:%d, and everything else is reached from there.", p.LinkPath(e.File), e.Line),
		})
	}
	// 2. Who calls the engine room (the most-called function).
	for _, fn := range p.TopFuncs(5) {
		if fn.IsMain || len(fn.Callers) == 0 {
			continue
		}
		var callers []*analyze.Func
		for _, id := range fn.Callers {
			if c := p.FuncByID(id); c != nil && !c.IsTest {
				callers = append(callers, c)
			}
		}
		if len(callers) == 0 {
			continue
		}
		accept := make([]string, 0, 2*len(callers))
		for _, c := range callers {
			accept = append(accept, c.Name, c.Display())
		}
		first := callers[0]
		fact := fmt.Sprintf("%s calls %s (%s:%d)", first.Display(), fn.Name, p.LinkPath(first.File), first.Line)
		if len(callers) > 1 {
			fact += fmt.Sprintf(" — %d functions call it in all, and any one of them is a right answer", len(callers))
		}
		fact += "."
		checks = append(checks, learnCheck{
			concept:  fn.Name,
			question: fmt.Sprintf("Name one function that calls %s.", fn.Name),
			accept:   accept,
			fact:     fact,
		})
		break
	}
	// 3. Where a tracked value gets its data (the varflow facts).
	// Setters inside test files are scaffolding, not the project's
	// dataflow, so they never anchor a question.
	for _, f := range p.VarFacts() {
		if f.Kind != "var" {
			continue
		}
		var setters []analyze.DataSetter
		for _, s := range f.Setters {
			if !strings.HasSuffix(s.File, "_test.go") {
				setters = append(setters, s)
			}
		}
		if len(setters) == 0 {
			continue
		}
		accept := make([]string, 0, 2*len(setters))
		for _, s := range setters {
			accept = append(accept, s.Short, s.Func)
		}
		s := setters[0]
		checks = append(checks, learnCheck{
			concept:  f.Name,
			question: fmt.Sprintf("Where does %s get its data — name one function that sets it.", f.Name),
			accept:   accept,
			fact:     fmt.Sprintf("%s gets its data in %s (%s:%d).", f.Name, s.Func, p.LinkPath(s.File), s.Line),
		})
		break
	}
	// Fallback: what the entry point calls, when the project gave
	// us fewer than two questions from the kinds above.
	if len(checks) < 2 {
		if eps := p.EntryPoints(); len(eps) > 0 {
			e := eps[0]
			var callees []*analyze.Func
			for _, id := range e.Calls {
				if c := p.FuncByID(id); c != nil && !c.IsTest {
					callees = append(callees, c)
				}
			}
			if len(callees) > 0 {
				accept := make([]string, 0, 2*len(callees))
				for _, c := range callees {
					accept = append(accept, c.Name, c.Display())
				}
				first := callees[0]
				checks = append(checks, learnCheck{
					concept:  "the entry point",
					question: "As the program starts, main calls several functions — name one.",
					accept:   accept,
					fact:     fmt.Sprintf("main calls %s (%s:%d) as it starts.", first.Display(), p.LinkPath(first.File), first.Line),
				})
			}
		}
	}
	if len(checks) > 3 {
		checks = checks[:3]
	}
	return checks
}

// learnAskCheck prints the current check question.
func learnAskCheck() {
	q := learnChecks[learnCheckAt]
	fmt.Printf("\n%s\n%s\n", cHead(fmt.Sprintf("Question %d of %d", learnCheckAt+1, len(learnChecks))), q.question)
	fmt.Println("(answer with the name — or \"next\" to skip it, \"stop learning\" to stop)")
}

// judgeLearnCheck scores an answer line against the waiting
// question's facts: right gets a brief confirmation; wrong gets the
// correct fact, plainly, and the Friend notes the miss.
func judgeLearnCheck(line string) {
	q := learnChecks[learnCheckAt]
	if learnAnswerRight(line, q) {
		fmt.Printf("%s Right — %s\n", cGood("✓"), q.fact)
	} else {
		fmt.Printf("Not quite — %s\n", q.fact)
		friendNoteMiss(q.concept, q.question, q.fact, learnRoot)
		learnMissed++
	}
	learnNextCheck()
}

// skipLearnCheck moves past the waiting question without judging
// it: the fact is shown for the record, nothing is recorded.
func skipLearnCheck() {
	q := learnChecks[learnCheckAt]
	fmt.Printf("Skipped — for the record: %s\n", q.fact)
	learnNextCheck()
}

func learnNextCheck() {
	learnCheckAt++
	if learnCheckAt >= len(learnChecks) {
		learnFinish()
		return
	}
	learnAskCheck()
}

// learnAnswerRight judges an answer against a check's accepted
// names, case-insensitively: the function's short name or its
// dotted name, as the whole answer or as a word inside it.
func learnAnswerRight(answer string, q learnCheck) bool {
	a := learnNormAnswer(answer)
	if a == "" {
		return false
	}
	tokens := map[string]bool{}
	for _, tok := range strings.FieldsFunc(a, func(r rune) bool {
		return r == ' ' || r == ',' || r == ';' || r == ':'
	}) {
		tokens[strings.Trim(tok, ".()")] = true
	}
	for _, acc := range q.accept {
		n := learnNormAnswer(acc)
		if n == "" {
			continue
		}
		if a == n {
			return true
		}
		short := n
		if i := strings.LastIndex(n, "."); i >= 0 {
			short = n[i+1:]
		}
		if tokens[short] {
			return true
		}
		if strings.Contains(n, ".") && strings.Contains(a, n) {
			return true
		}
	}
	return false
}

// learnNormAnswer flattens an answer (or an accepted name) for
// comparison: lowercase, no surrounding punctuation or "the"/"func"
// furniture, single spaces.
func learnNormAnswer(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, ".!?\"'` ")
	for {
		trimmed := s
		for _, pre := range []string{"the ", "func ", "function "} {
			trimmed = strings.TrimPrefix(trimmed, pre)
		}
		if trimmed == s {
			break
		}
		s = trimmed
	}
	return strings.Join(strings.Fields(s), " ")
}

// learnFinish closes the flow and hands back to free chat.
func learnFinish() {
	if !learnActive {
		return
	}
	learnActive = false
	learnCheckAt = -1
	name := "this project"
	if learnRoot != "" {
		name = filepath.Base(learnRoot)
	}
	fmt.Println()
	printBand("Learn — done")
	fmt.Printf("That's %s: the story, the map, the entry point, the concepts, and the check.\n", name)
	if learnMissed > 0 {
		fmt.Printf("The Friend noted the %d you missed, so they can come back another day.\n", learnMissed)
	}
	fmt.Println("From here it's yours: ask me anything about this code — or try a small edit: tell me what you'd like to change and I'll plan it with you.")
	recordBrain("teacher", true, "learn: finished")
}

// stopLearn ends the flow early, gracefully.
func stopLearn() {
	learnActive = false
	learnCheckAt = -1
	fmt.Println("Learn mode ended — the stops you've seen stay seen, and the Friend kept any misses. Say \"learn\" anytime to run it again, or just ask me anything.")
}

// learnResetForTest restores pristine flow state (tests only).
func learnResetForTest() {
	learnActive = false
	learnStage = 0
	learnRoot = ""
	learnChecks = nil
	learnCheckAt = -1
	learnMissed = 0
}
