package chat

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
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
func handleBeginnerCommand(line string, profile *UserProfile) bool {
	needProject := func() bool {
		if lastAnalyzeProject == nil {
			fmt.Println("Run /analyze first.")
			return false
		}
		return true
	}
	switch {
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
		return true
	case line == "/glossary":
		if !needProject() {
			return true
		}
		fmt.Printf("\n%s", lastAnalyzeProject.GlossaryText())
		return true
	case line == "/learn-path":
		if !needProject() {
			return true
		}
		fmt.Printf("\n%s", lastAnalyzeProject.LearnPathText())
		return true
	case line == "/tour":
		if !needProject() {
			return true
		}
		tourStops = lastAnalyzeProject.TourStops()
		tourIdx = 0
		printTourStop()
		return true
	case line == "next" && tourStops != nil:
		tourIdx++
		if tourIdx >= len(tourStops) {
			tourStops = nil
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
