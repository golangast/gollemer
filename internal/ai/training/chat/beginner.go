package chat

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// Beginner-five commands (2026-10-20): /example, /tour (+ next),
// /glossary, /watch, /learn-path. All read from the last analyzed
// project; the analyzer builds the content, this file only dispatches
// and, for /watch, runs the program under the usual confirmation.

// Tour state: stops are pre-rendered at /tour time; bare "next"
// advances only while a tour is active, so it can never hijack an
// unrelated conversation.
var (
	tourStops []string
	tourIdx   int
)

// handleBeginnerCommand dispatches the beginner five. It returns true
// when the line was one of them (handled or refused with a message).
// handleBeginnerCommand wraps the implementation so the COORDINATOR can
// record which brain each beginner command exercised.
func handleBeginnerCommand(line string, profile *UserProfile) bool {
	handled := handleBeginnerCommandImpl(line, profile)
	if handled {
		friendTurn++
		defer friendMaybeSpeak()
		// Map the CANONICAL form: callers hand us raw sentences
		// ("give me a tour") whose translation happens inside Impl —
		// keying off the raw line left tours and runs unrecorded.
		line = translateBeginnerSentence(line)
		key := ""
		switch {
		case strings.HasPrefix(line, "/tour"), strings.HasPrefix(line, "/example"),
			line == "next", line == "/summarize":
			key = "analyst"
		case line == "/glossary":
			key = "lexicon"
		case line == "/learn-path":
			key = "teacher"
		case line == "/watch":
			key = "runner"
	case line == "/tell-more":
			key = "memory"
		case line == "/what-now", line == "/brains", line == "/brainstatus", line == "/whybrain":
			key = "coordinator"
		}
		recordBrain(key, true, "sentence: "+line)
	}
	return handled
}

func handleBeginnerCommandImpl(line string, profile *UserProfile) bool {
	if isHelpLine(line) {
		printHelp()
		return true
	}
	line = translateBeginnerSentence(line)
	needProject := func() bool { return ensureProject() }
	switch {
	case line == "/brains":
		printBrains()
		return true
	case line == "/brainstatus":
		printBrainStatus()
		return true
	case line == "/whybrain":
		explainLastBrain()
		return true
	case line == "/tell-more":
		handleTellMore(profile)
		return true
	case line == "/what-now":
		handleWhatNow()
		return true
	case line == "/summarize":
		handleSummarize()
		return true
	case strings.HasPrefix(line, "/example "):
		if !needProject() {
			return true
		}
		fname := strings.TrimSpace(strings.TrimPrefix(line, "/example "))
		fn := findFuncByQuery(lastAnalyzeProject, fname)
		if fn == nil {
			fmt.Printf("Function %q not found.", fname)
			if near := nearFuncNames(lastAnalyzeProject, fname); len(near) > 0 {
				fmt.Printf(" Close matches: %s.", strings.Join(near, ", "))
			}
			fmt.Println()
			return true
		}
		fmt.Printf("\n%s", lastAnalyzeProject.ExamplesText(fn))
		lastActionKind, lastSubject = "example", fname
		return true
	case line == "/glossary":
		if !needProject() {
			return true
		}
		fmt.Printf("\n%s", lastAnalyzeProject.GlossaryText())
		lastActionKind = "glossary"
		return true
	case line == "/learn-path":
		if !needProject() {
			return true
		}
		fmt.Printf("\n%s", lastAnalyzeProject.LearnPathText())
		lastActionKind = "learn_path"
		return true
	case isTourLine(line):
		if arg := tourLineArg(line); arg != "" && !analyzeForTour(arg) {
			return true
		}
		if !needProject() {
			return true
		}
		tourStops = lastAnalyzeProject.TourStops()
		tourIdx = 0
		tourDone = false
		lastActionKind = "tour"
		printTourStop()
		return true
	case line == "next" && tourStops != nil:
		tourIdx++
		if tourIdx >= len(tourStops) {
			tourStops = nil
			tourDone = true
			fmt.Println("End of the tour — that was the last stop.")
			return true
		}
		printTourStop()
		return true
	case line == "/watch":
		if !needProject() {
			return true
		}
		runWatchIt(profile)
		return true
	}
	return false
}

// --- Sentence forms -------------------------------------------------------
// The beginner five are sentence-first (John, 2026-10-20: "why cant it
// just be a sentence"): "give me a tour", "run it", "what should I learn
// first" all work; the /slash forms remain as shortcuts. Translation
// rewrites a recognized sentence into its slash form and lets the
// switch above do the work, so both spellings share one code path.

// cutPrefixFold is strings.CutPrefix with case-insensitive matching;
// the returned rest keeps its original case (function names need it).
func cutPrefixFold(s, prefix string) (string, bool) {
	if len(s) >= len(prefix) && strings.EqualFold(s[:len(prefix)], prefix) {
		return strings.TrimSpace(s[len(prefix):]), true
	}
	return "", false
}

// looksLikeProjectRef reports whether s reads as a GitHub URL or a
// filesystem path, so "tour https://…" can carry its project along.
func looksLikeProjectRef(s string) bool {
	return strings.Contains(s, "github.com") || strings.Contains(s, "/") ||
		strings.HasPrefix(s, "~") || strings.HasSuffix(s, ".git")
}

func translateBeginnerSentence(line string) string {
	t := strings.TrimSpace(line)
	low := strings.Trim(strings.ToLower(t), ".!? ")
	// A tour that names its project: "tour <url>", "/tour <url>".
	if rest, ok := cutPrefixFold(t, "tour "); ok && looksLikeProjectRef(rest) {
		return "/tour " + rest
	}
	switch low {
	case "tour", "the tour", "give me a tour", "give me the tour",
		"take the tour", "start the tour", "tour this", "tour it",
		"tour this project", "show me around",
		"walk me through the project", "walk me through this project",
		"walk me through the repo", "walk me through this repo":
		return "/tour"
	case "glossary", "the glossary", "project glossary", "the dictionary",
		"project dictionary", "key terms", "what are the key terms",
		"what do the words mean", "what do these words mean",
		"define the types", "define the main types", "what are the main types":
		return "/glossary"
	case "brains", "show brains", "show me your brains", "what brains do you have",
		"list your brains", "list brains", "your brains":
		return "/brains"
	case "brain status", "how are your brains doing", "how are the brains doing",
		"are your brains working", "brain scoreboard":
		return "/brainstatus"
	case "why did you do that", "why did you answer that way", "who answered that",
		"which brain was that", "why that brain", "why did you use that brain":
		return "/whybrain"
	case "learn path", "learning path", "the learn path", "the learning path",
		"what should i learn first", "what do i learn first",
		"where should i start learning", "learning order",
		"what order should i learn", "what order should i learn this in":
		return "/learn-path"
	case "watch", "watch it", "watch it run", "run it", "run this",
		"run the project", "run this project", "run the program",
		"see it run", "let me see it run", "try running it":
		return "/watch"
	}
	switch low {
	case "tell me more", "more", "go on", "continue", "keep going", "what else",
		"more info", "tell me more about that", "tell me more about it":
		return "/tell-more"
	case "what now", "what to do now", "what should i do now", "what next",
		"whats next", "what do i do", "what should i do", "im stuck", "im lost",
		"i dont know what to do", "how do i use this", "what do i type":
		return "/what-now"
	case "what does this program do", "what does this app do", "what does this code do",
		"what does this project do", "what does it do", "summarize this project",
		"what is this program", "what is this for":
		return "/summarize"
	}
	if rest, ok := cutPrefixFold(t, "show me an example of "); ok {
		return "/example " + rest
	}
	if rest, ok := cutPrefixFold(t, "example of "); ok {
		return "/example " + rest
	}
	if rest, ok := cutPrefixFold(t, "example "); ok {
		return "/example " + rest
	}
	if rest, ok := cutPrefixFold(t, "how is "); ok && strings.HasSuffix(strings.ToLower(rest), " used") {
		return "/example " + strings.TrimSpace(rest[:len(rest)-len(" used")])
	}
	return line
}

// isTourLine matches the canonical tour forms after translation:
// "/tour" and "/tour <project>".
func isTourLine(line string) bool {
	return line == "/tour" || strings.HasPrefix(line, "/tour ")
}

// tourLineArg returns the project reference in "/tour <ref>", or "".
func tourLineArg(line string) string {
	return strings.TrimSpace(strings.TrimPrefix(line, "/tour"))
}

// analyzeForTour resolves and analyzes a project reference for
// "tour <project>", printing one line instead of the full report —
// the tour itself is the introduction.
func analyzeForTour(ref string) bool {
	root, err := resolveAnalyzeRoot(ref)
	if err != nil {
		fmt.Printf("I couldn't read that codebase: %v\n", err)
		return false
	}
	fmt.Println("[reading the codebase — parsing Go files...]")
	if !analyzeQuiet(root) {
		return false
	}
	fmt.Printf("Got it — %s. Starting the tour.\n", root)
	return true
}

// ensureProject makes sure SOME project is loaded (on-ramp rule: a
// default beats a refusal): the analyzed one, else the folder the
// chat started in, else the bundled demo.
func ensureProject() bool {
	if lastAnalyzeProject != nil {
		return true
	}
	if root := startFolder(); root != "" && hasGoFiles(root) {
		fmt.Println("Let me look at the project in this folder first…")
		if analyzeQuiet(root) {
			return true
		}
	}
	if demo := demoProjectDir(); demo != "" {
		fmt.Println("No project analyzed yet — I'll use the tiny built-in example (pocketcounter). Say 'tour <github link>' any time for a real project.")
		return analyzeQuiet(demo)
	}
	fmt.Println("First let me look at a project — say: analyze <github link or folder>")
	return false
}

// analyzeQuiet analyzes root and makes it the current project,
// without printing the full report.
func analyzeQuiet(root string) bool {
	p, err := analyze.Analyze(root)
	if err != nil {
		fmt.Printf("I couldn't analyze %s: %v\n", root, err)
		return false
	}
	lastAnalyzeRoot = root
	lastAnalyzeProject = p
	return true
}

// startFolder is the directory the chat was started in: the unified
// project root when set, else the process working directory.
func startFolder() string {
	if unifiedProjectRoot != "" {
		return unifiedProjectRoot
	}
	wd, err := os.Getwd()
	if err != nil {
		return ""
	}
	return wd
}

// hasGoFiles reports whether dir holds Go code (checked two levels
// deep, enough for a project root with cmd/ or examples/ layouts).
func hasGoFiles(dir string) bool {
	found := false
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if strings.Count(strings.TrimPrefix(path, dir), string(os.PathSeparator)) > 2 {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") {
			found = true
			return filepath.SkipAll
		}
		return nil
	})
	return found
}

// demoProjectDir finds the bundled pocketcounter example, which
// ships inside the repo at examples/pocketcounter.
func demoProjectDir() string {
	cands := []string{}
	for _, base := range []string{unifiedProjectRoot, startFolder()} {
		if base != "" {
			cands = append(cands, filepath.Join(base, "examples", "pocketcounter"))
		}
	}
	// Walk up from the working directory: covers `go run .` from a
	// repo subfolder and the chat package's own tests.
	if wd, err := os.Getwd(); err == nil {
		for d := wd; ; d = filepath.Dir(d) {
			cands = append(cands, filepath.Join(d, "examples", "pocketcounter"))
			if filepath.Dir(d) == d {
				break
			}
		}
	}
	for _, d := range cands {
		if fi, err := os.Stat(d); err == nil && fi.IsDir() {
			return d
		}
	}
	return ""
}

// --- Help -----------------------------------------------------------------

// isHelpLine matches the ways a stuck beginner asks what exists.
func isHelpLine(line string) bool {
	switch strings.Trim(strings.ToLower(strings.TrimSpace(line)), ".!? ") {
	case "help", "?", "menu", "commands", "what can you do",
		"what can i say", "what can i ask", "what do you do":
		return true
	}
	return false
}

// printSayThis is the startup hint: the few sentences a beginner
// needs in the first minute, shown by the chat itself.
func printSayThis() {
	fmt.Println("Say things like:")
	fmt.Println("  tour <github link>      — walk a project, one stop at a time (then: next)")
	fmt.Println("  what does this program do — the plain-English version")
	fmt.Println("  what are the main types — the project's words, in plain English")
	fmt.Println("  run it                  — watch a project run")
	fmt.Println("  what should I do now    — whenever you're stuck")
	fmt.Println("  help                    — everything you can say")
}

// printHelp is the 'help' answer: everything sayable, grouped by
// intent, sentences first and slashes as shortcuts.
func printHelp() {
	fmt.Println(`
Things you can say:

Learn a project:
  tour <github link>          walk it one stop at a time (then say: next)
  give me a tour              tour the project already open
  what does this program do   the plain-English summary
  what are the main types     its key words, one plain line each
  what should I learn first   its ideas, in the order to learn them
  example Names               see a function used for real (any name works)
  run it                      watch the project run, narrated

When you're stuck:
  what should I do now        I'll look at where you are and name the next step
  any suggestions             the Friend speaks — patterns it noticed in how you work
  tell me more                go one step deeper into whatever we just did
  help                        this list

Ask about code:
  analyze <github link or folder>
  what does Names do
  why does Names lock          the reasons the code itself shows
  compare Add and Total        two functions, side by side
  where for <github issue link>   where to edit for an issue

The team behind the answers:
  (another terminal) gollemer -watch <folder>   the Watcher diffs your code as you save; the Friend reports what mattered
  brains                       who does the work here, and their honest limits
  how are your brains doing    the scoreboard — who answered, who failed
  why did you do that          I'll name the brain and the recorded reason

Fix and write:
  paste any Go error          I'll translate it into plain words
  /fix undefined: foo         (shortcut for the same idea)
  write a function that reverses a string

Or just talk — ask me anything. Say 'exit' to leave.`)
}

func printTourStop() {
	fmt.Printf("\n--- Stop %d of %d ---\n%s\n", tourIdx+1, len(tourStops), tourStops[tourIdx])
	if tourIdx < len(tourStops)-1 {
		fmt.Println("(say 'next' for the next stop)")
	}
}

// makefileTargets lists the target names in dir's Makefile (using the
// shared makeTargetLine pattern from the chat package).
func makefileTargets(dir string) map[string]bool {
	out := map[string]bool{}
	data, err := os.ReadFile(filepath.Join(dir, "Makefile"))
	if err != nil {
		return out
	}
	for _, l := range strings.Split(string(data), "\n") {
		if m := makeTargetLine.FindStringSubmatch(l); m != nil {
			out[m[1]] = true
		}
	}
	return out
}

// runWatchIt is "watch it work": figure out how the analyzed project
// runs, confirm, run it (no shell, 60s cap), and narrate the outcome
// in plain words.
func runWatchIt(profile *UserProfile) {
	p := lastAnalyzeProject
	root := lastAnalyzeRoot
	if root == "" {
		root = p.Root
	}
	var argv []string
	dir := root
	why := ""
	// 1. A run-ish Makefile target in the project itself.
	targets := makefileTargets(root)
	for _, t := range []string{"run", "demo", "example", "start", "serve"} {
		if targets[t] {
			argv = []string{"make", t}
			why = fmt.Sprintf("the project's own Makefile has a %q target", t)
			break
		}
	}
	// 2. The entry point: go run in the main package's directory.
	if argv == nil {
		if eps := p.EntryPoints(); len(eps) > 0 {
			dir = filepath.Join(root, eps[0].PkgDir)
			argv = []string{"go", "run", "."}
			why = "that's the program's entry point"
		}
	}
	// 3. A library: watch its tests work instead.
	isTest := false
	if argv == nil {
		argv = []string{"go", "test", "./..."}
		why = "it's a library — no program to run, so watch its tests instead"
		isTest = true
	}
	if !confirmAction(fmt.Sprintf("Run it? %s (%s)", strings.Join(argv, " "), why), profile) {
		fmt.Println("OK — not run.")
		return
	}
	fmt.Printf("[running: %s — up to 60 seconds]\n", strings.Join(argv, " "))
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = dir
	outBytes, err := cmd.CombinedOutput()
	out := string(outBytes)
	timedOut := ctx.Err() == context.DeadlineExceeded
	// Show the program's own output, capped.
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if out == "" {
		lines = nil
	}
	const maxLines = 50
	shown := lines
	if len(shown) > maxLines {
		shown = shown[:maxLines]
	}
	for _, l := range shown {
		fmt.Println(l)
	}
	if len(lines) > maxLines {
		fmt.Printf("[… %d more lines not shown]\n", len(lines)-maxLines)
	}
	// Narrate.
	switch {
	case timedOut:
		fmt.Println("\nIt ran for a full minute without finishing, so I stopped it. Programs that do that are usually servers or tools waiting for input — that's normal for them, not an error.")
	case err != nil:
		fmt.Printf("\nIt stopped with an error (%v). The last lines above are the program's own words — paste one into /fix <error> and I'll explain it.\n", err)
	case isTest:
		passed := 0
		for _, l := range lines {
			if strings.HasPrefix(l, "ok ") {
				passed++
			}
		}
		fmt.Printf("\nIt works: %d package(s) of tests passed, none failed. Those tests are the project's safety net — /example shows them in action.\n", passed)
	default:
		if len(lines) == 0 {
			fmt.Println("\nIt ran fine and printed nothing — some programs do their work quietly (files, network, a database) instead of talking to the screen.")
		} else {
			fmt.Println("\nIt ran fine. What you see above is the program talking: that's its real output, not a description of it.")
		}
	}
}

// unknownCommandHint turns the old dead-end "Unknown command" into a
// nudge: the closest known phrase when there is one, plus example
// sentences either way.
func unknownCommandHint(line string) string {
	known := []string{
		"tour", "give me a tour", "run it", "example", "glossary", "learn-path",
		"what are the main types", "what should i learn first", "analyze", "where", "help",
		"/tour", "/example", "/glossary", "/watch", "/learn-path", "/analyze", "/fix",
		"/whatis", "/review", "/comment", "/find", "/arch", "/test", "/why-import",
		"/profile", "/history", "/forget", "/quit",
	}
	w := strings.ToLower(strings.TrimSpace(line))
	best, bestD := "", 3 // a suggestion needs edit distance <= 2
	for _, k := range known {
		if d := levenshtein(w, k); d < bestD {
			best, bestD = k, d
		}
	}
	if best != "" {
		return fmt.Sprintf("I don't know that one — did you mean '%s'? Say 'help' for everything you can say.", best)
	}
	return "I don't know that one. Say things like 'tour <github link>', 'what are the main types', or 'run it' — or say 'help' for the full list."
}

// --- Session thread: tell me more / what now -----------------------------
// Complete beginners ask three things over and over: "tell me
// more", "what do I do now", and "what does this program do". The
// first two are about the CONVERSATION, not the code, so the chat
// remembers the last thing it did and continues from there.

var (
	lastActionKind string // e.g. "example", "tour", "review"
	lastSubject    string // the function/type last discussed
	tourDone       bool
)

// topFuncName is the project's engine room: its most-called
// function, used for concrete suggestions ("try 'example New'").
func topFuncName(p *analyze.Project) string {
	if p == nil {
		return ""
	}
	if tops := p.TopFuncs(1); len(tops) > 0 {
		return tops[0].Name
	}
	return ""
}

// handleWhatNow answers "what should I do now" from the actual
// session state, with one concrete next step.
func handleWhatNow() {
	if lastAnalyzeProject == nil {
		fmt.Println("Nothing's open yet. Easiest first step: say 'tour' and I'll walk you through a tiny built-in example — or 'tour <github link>' for a real project.")
		return
	}
	if tourStops != nil {
		fmt.Printf("You're mid-tour — stop %d of %d. Say 'next' to keep going, or ask me anything about what you've seen.\n", tourIdx+1, len(tourStops))
		return
	}
	if !tourDone {
		fmt.Println("You haven't taken the tour of this project yet — say 'give me a tour'. It's the fastest way to get your bearings.")
		return
	}
	if tf := topFuncName(lastAnalyzeProject); tf != "" {
		fmt.Printf("Try this: say 'example %s' to see the busiest function in the project used for real. Or say 'run it' to watch the project run.\n", tf)
		return
	}
	fmt.Println("Say 'run it' to watch the project run, or 'what are the main types' for its key words.")
}

// handleTellMore continues whatever the last answer was, one step
// deeper along a natural chain: example → walkthrough → review →
// test; glossary → learn-path → tour; anything else → tour/next.
func handleTellMore(profile *UserProfile) {
	if tourStops != nil {
		handleBeginnerCommand("next", profile)
		return
	}
	subject := lastSubject
	switch lastActionKind {
	case "example":
		if subject != "" && ensureProject() {
			fmt.Printf("More on %s — how it works, step by step:\n", subject)
			if out, ok := lastAnalyzeProject.Answer("walk me through " + subject); ok {
				fmt.Println(out)
				lastActionKind = "walkthrough"
				return
			}
		}
	case "whatis", "show_func":
		if subject != "" {
			lastActionKind = ""
			fmt.Printf("More on %s — seeing it used for real:\n", subject)
			handleBeginnerCommand("/example "+subject, profile)
			return
		}
	case "walkthrough":
		if subject != "" && ensureProject() {
			fmt.Printf("More on %s — a quick review:\n%s\n", subject, reviewFunc(subject, lastAnalyzeProject))
			lastActionKind = "review"
			return
		}
	case "review":
		if subject != "" && ensureProject() {
			fmt.Printf("More on %s — a test you could start from:\n%s\n", subject, generateTestSkeleton(subject, lastAnalyzeProject))
			lastActionKind = "test_scaffold"
			return
		}
	case "glossary":
		if ensureProject() {
			fmt.Printf("More of the map — what to learn, in order:\n%s", lastAnalyzeProject.LearnPathText())
			lastActionKind = "learn_path"
			return
		}
	case "learn_path", "arch", "summarize", "analyze":
		handleBeginnerCommand("/tour", profile)
		return
	}
	// Nothing to continue: the best "more" is a guided next step.
	handleWhatNow()
}

// handleSummarize answers "what does this program do" with just
// the plain-English parts of the report.
func handleSummarize() {
	if !ensureProject() {
		return
	}
	fmt.Printf("\n%s\n%s", lastAnalyzeProject.Summary(), lastAnalyzeProject.Story())
	lastActionKind = "summarize"
}
