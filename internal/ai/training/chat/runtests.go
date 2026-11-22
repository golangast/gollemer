package chat

// Run the tests (2026-11-21, wave 1 of the six-piece
// program): "run the tests" in the unified chat runs the
// project's own Go tests and explains the outcome in plain
// words. Before this claim, the sentence had no owner: it
// fell to the intent layer and the makefile brain, which
// could only offer a make target — the project's tests ran,
// if at all, as raw terminal output the chat never read.
//
// This claim is the deterministic seat, beside
// run-the-program (whose gate, root rule, and [runner]
// voice it shares):
//
//   - The grammar is whole-line and imperative only: "run
//     the tests", "run tests", "test the project", and the
//     subtree spellings "run the tests in <folder>" / "run
//     the tests for <folder>". Questions — "do the tests
//     pass?", "why did the test fail" — are not shapes here
//     and fall through untouched; explaining a failure the
//     user pastes or points at is a later wave's seat.
//   - The run waits behind the run family's gate (the
//     profile's 'do all' directive auto-yeses, announced,
//     exactly as for a program run). On a no, nothing runs.
//   - The runner is argv only, no shell ever: `go test
//     ./... -count=1` at the project root, or
//     `go test ./<folder>/... -count=1` for a subtree. The
//     only part of the user's line that reaches argv is the
//     folder segment, resolved against the project root and
//     refused if it climbs out of the project — the
//     run-program resolver's rule. The run is bounded
//     (testRunTimeout); on timeout the process is killed
//     and the report says so, over whatever results were
//     already in.
//   - The report is parsed from real `go test` output:
//     package results, `--- FAIL` lines with their first
//     assertion line, `[build failed]` blocks with their
//     compiler errors, and panics read to the first frame
//     in the project's own code. Compiler errors get a
//     plain-words gloss for the common shapes (undefined,
//     unused import/variable, type mismatch, missing
//     return, syntax); unknown shapes are quoted plainly.
//     A failing test that resolves in the project index
//     earns one clause of context from its own doc comment
//     (or its subject's). Nothing here invents a fix, and
//     no edit is offered — the report ends by naming where
//     to start, and stops.

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// testRunTimeout bounds one `go test` run. A beginner's
// project tests should take seconds; four minutes is the
// honest ceiling before the chat says "didn't finish"
// instead of waiting forever.
const testRunTimeout = 4 * time.Minute

// testOutputCap bounds the captured output the parser
// reads. Real failure output is kilobytes; the cap exists
// so a runaway test printing forever cannot eat the chat's
// memory — past it, output is dropped, not the process.
const testOutputCap = 4 << 20

// ---------------------------------------------------------------------
// Parsing the sentence.

// parseRunTests classifies one chat line for this claim.
// target is the folder as typed ("" = the whole project).
// The shapes are the brief's, whole-line: anything else —
// questions above all — is not this claim's.
func parseRunTests(line string) (string, bool) {
	fields := strings.Fields(strings.TrimSpace(line))
	if len(fields) == 0 {
		return "", false
	}
	fold := func(s string) string {
		return strings.ToLower(strings.TrimRight(s, ".!?"))
	}
	fields = stripProgramSofteners(fields, fold)
	if len(fields) == 0 {
		return "", false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = fold(f)
	}
	// run [the] tests [in|for <folder>]
	if low[0] == "run" {
		rest := low[1:]
		if len(rest) > 0 && rest[0] == "the" {
			rest = rest[1:]
		}
		if len(rest) == 1 && rest[0] == "tests" {
			return "", true
		}
		if len(rest) == 3 && rest[0] == "tests" && (rest[1] == "in" || rest[1] == "for") {
			return strings.TrimRight(fields[len(fields)-1], ".!?"), true
		}
		return "", false
	}
	// test the project
	if low[0] == "test" && len(low) == 3 && low[1] == "the" && low[2] == "project" {
		return "", true
	}
	return "", false
}

// ---------------------------------------------------------------------
// The claim.

// tryRunTests is the chat-loop entry, seated beside
// run-the-program and the Linux commands.
func tryRunTests(line string, r *LineReader, profile *UserProfile) bool {
	return runRunTests(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	}, profile)
}

// runRunTests carries one line for this claim. A parsed
// shape always claims the line — the shape is unambiguous,
// so every failure to run is answered honestly by name
// rather than falling through to a brain that would guess.
func runRunTests(line string, ask editAsk, profile *UserProfile) bool {
	target, ok := parseRunTests(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [runner]>")
	rootAbs, rootOK := runProgramRoot(tag)
	if !rootOK {
		return true
	}
	pattern := "./..."
	scope := "the whole project"
	folder := ""
	if target != "" {
		rel := filepath.ToSlash(strings.TrimSpace(strings.TrimPrefix(target, "./")))
		rel = strings.TrimSuffix(rel, "/")
		if rel == "" || rel == "." {
			rel = ""
		}
		if rel != "" {
			abs, refusal := runProgramResolve(rootAbs, rel)
			if refusal != "" {
				fmt.Printf("%s %s\n", tag, refusal)
				recordBrain("runner", false, "test run refused: "+refusal)
				return true
			}
			st, err := os.Stat(abs)
			if err != nil || !st.IsDir() {
				fmt.Printf("%s There's no folder called %s in this project. %s\n", tag, rel, testFolderListText(rootAbs))
				recordBrain("runner", false, "test run refused: no folder "+rel)
				return true
			}
			folder = rel
			pattern = "./" + rel + "/..."
			scope = "the tests in " + rel + "/"
		}
	}
	prompt := fmt.Sprintf("Run it? go test %s (%s)", pattern, scope)
	if !runProgramGate(ask, profile, prompt) {
		fmt.Printf("%s OK — not run.\n", tag)
		recordBrain("runner", true, "test run declined")
		return true
	}
	fmt.Printf("%s Running go test %s now — I'll read the results when it finishes.\n", tag, pattern)
	out, timedOut, startErr := executeGoTest(rootAbs, pattern)
	if startErr != "" {
		fmt.Printf("%s I couldn't run the tests: %s.\n", tag, startErr)
		recordBrain("runner", false, "test run failed to start: "+startErr)
		return true
	}
	rep := parseGoTestOutput(out, testModulePath(rootAbs), rootAbs)
	rep.TimedOut = timedOut
	fmt.Print(renderGoTestReport(rep, folder))
	if rep.FailedPackages() > 0 || rep.TimedOut || rep.NoResults {
		recordBrain("runner", false, "tests reported: "+rep.shortReason())
	} else {
		recordBrain("runner", true, "tests passed")
	}
	return true
}

// testFolderListText names the project's top-level folders
// that hold Go files, for the no-such-folder answer: what
// exists, so the user can pick from real names.
func testFolderListText(rootAbs string) string {
	entries, err := os.ReadDir(rootAbs)
	if err != nil {
		return ""
	}
	var folders []string
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, ".") || name == "vendor" {
			continue
		}
		sub, err := os.ReadDir(filepath.Join(rootAbs, name))
		if err != nil {
			continue
		}
		for _, s := range sub {
			if !s.IsDir() && strings.HasSuffix(s.Name(), ".go") {
				folders = append(folders, name)
				break
			}
		}
	}
	switch {
	case len(folders) == 0:
		return "This project has no folders with Go files in them."
	case len(folders) == 1:
		return "The folder with Go files here is: " + folders[0] + "."
	default:
		return "The folders with Go files here are: " + strings.Join(folders, ", ") + "."
	}
}

// ---------------------------------------------------------------------
// The runner: argv only, bounded, output capped.

// cappedTestBuffer is an io.Writer that keeps at most
// testOutputCap bytes and drops the rest.
type cappedTestBuffer struct {
	buf  bytes.Buffer
	over bool
}

func (c *cappedTestBuffer) Write(p []byte) (int, error) {
	room := testOutputCap - c.buf.Len()
	switch {
	case room <= 0:
		c.over = true
	case len(p) > room:
		c.buf.Write(p[:room])
		c.over = true
	default:
		c.buf.Write(p)
	}
	return len(p), nil
}

// executeGoTest runs `go test <pattern> -count=1` in
// rootAbs and returns the captured output, whether the
// run hit the timeout, and — only when the process could
// not start at all — why.
func executeGoTest(rootAbs, pattern string) (string, bool, string) {
	ctx, cancel := context.WithTimeout(context.Background(), testRunTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "test", pattern, "-count=1")
	cmd.Dir = rootAbs
	var buf cappedTestBuffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	if err != nil {
		if ee, ok := err.(*exec.Error); ok {
			return "", false, "I can't find the go command on this machine (" + ee.Err.Error() + ")"
		}
	}
	return buf.buf.String(), ctx.Err() == context.DeadlineExceeded, ""
}

// testModulePath reads the module line from rootAbs/go.mod
// ("" when there is none) — the parser uses it to turn a
// package path into the folder its files live in.
func testModulePath(rootAbs string) string {
	b, err := os.ReadFile(filepath.Join(rootAbs, "go.mod"))
	if err != nil {
		return ""
	}
	for _, ln := range strings.Split(string(b), "\n") {
		f := strings.Fields(ln)
		if len(f) == 2 && f[0] == "module" {
			return strings.Trim(f[1], "\"")
		}
	}
	return ""
}

// ---------------------------------------------------------------------
// Parsing `go test` output.

// goTestReport is one parsed run.
type goTestReport struct {
	Packages  []*goTestPackage
	TimedOut  bool
	NoResults bool   // nothing parsed — go itself failed
	GoError   string // the first meaningful output line, when NoResults
}

type goTestPackage struct {
	Path      string // import path, as printed
	Dir       string // project-relative folder ("" = root)
	Status    string // "ok", "fail", "buildfail", "notests"
	Failures  []*goTestFailure
	BuildErrs []*goBuildError
	Panic     *goTestPanic
}

type goTestFailure struct {
	Test    string
	File    string // base name, as printed; Dir joins at render
	Line    int
	Message string
}

type goBuildError struct {
	File    string // project-relative path, as printed
	Line    int
	Col     int
	Message string
}

type goTestPanic struct {
	Message string
	Func    string // base name of the chosen frame's function
	File    string // project-relative when the frame is the project's
	Line    int
}

type panicFrame struct {
	Func string
	File string
	Line int
}

// FailedPackages counts the packages that did not pass.
func (r *goTestReport) FailedPackages() int {
	n := 0
	for _, p := range r.Packages {
		if p.Status == "fail" || p.Status == "buildfail" {
			n++
		}
	}
	return n
}

// RanPackages counts the packages that actually ran tests
// (passed or failed — "no test files" packages ran none).
func (r *goTestReport) RanPackages() int {
	n := 0
	for _, p := range r.Packages {
		if p.Status == "ok" || p.Status == "fail" || p.Status == "buildfail" {
			n++
		}
	}
	return n
}

func (r *goTestReport) shortReason() string {
	switch {
	case r.TimedOut:
		return "timed out"
	case r.NoResults:
		return "no results"
	default:
		return fmt.Sprintf("%d packages failed", r.FailedPackages())
	}
}

var (
	reTestFailLine  = regexp.MustCompile(`^--- FAIL: (\S+)`)
	reTestFileLine  = regexp.MustCompile(`^(\S+\.go):(\d+):\s*(.*)$`)
	reBuildErrLine  = regexp.MustCompile(`^(\S+\.go):(\d+):(\d+):\s*(.*)$`)
	rePkgOK         = regexp.MustCompile(`^ok\s+(\S+)`)
	rePkgNoTests    = regexp.MustCompile(`^\?\s+(\S+)\s+\[no test files\]`)
	rePkgFail       = regexp.MustCompile(`^FAIL\s+(\S+)(?:\s+\[(.*)\])?`)
	rePanicLine     = regexp.MustCompile(`^panic: (.*)$`)
	rePanicFuncLine = regexp.MustCompile(`^(\S+)\(`)
	rePanicFileLine = regexp.MustCompile(`^\s+(\S+\.go):(\d+)\s+\+0x`)
	reHashLine      = regexp.MustCompile(`^#\s+(\S+)`)
)

// parseGoTestOutput reads one run's captured output into a
// report. Package output arrives as contiguous blocks, so
// pending failures, build errors, and a panic attach to
// the next package result line that names their package.
func parseGoTestOutput(out, modulePath, rootAbs string) *goTestReport {
	rep := &goTestReport{}
	var pendFailures []*goTestFailure
	var pendBuild []*goBuildError
	var pendPanic *goTestPanic
	var pendFrames []panicFrame
	var curFailure *goTestFailure
	inPanic := false

	flush := func(pkg *goTestPackage) {
		pkg.Failures = pendFailures
		pkg.BuildErrs = pendBuild
		if pendPanic != nil {
			fr := choosePanicFrame(pendFrames, rootAbs)
			pendPanic.Func, pendPanic.File, pendPanic.Line = fr.Func, fr.File, fr.Line
			pkg.Panic = pendPanic
		}
		pendFailures, pendBuild, pendPanic, pendFrames = nil, nil, nil, nil
		curFailure = nil
		inPanic = false
	}

	for _, raw := range strings.Split(out, "\n") {
		line := strings.TrimRight(raw, " \t\r")
		trimmed := strings.TrimSpace(line)
		switch {
		case reHashLine.MatchString(line):
			// A build/vet block header ("# pkg"): the
			// file:line:col lines that follow are its errors.
		case rePkgOK.MatchString(line):
			m := rePkgOK.FindStringSubmatch(line)
			pkg := &goTestPackage{Path: m[1], Status: "ok", Dir: testPkgDir(m[1], modulePath)}
			rep.Packages = append(rep.Packages, pkg)
			pendFailures, pendBuild, pendPanic, pendFrames = nil, nil, nil, nil
			curFailure, inPanic = nil, false
		case rePkgNoTests.MatchString(line):
			m := rePkgNoTests.FindStringSubmatch(line)
			rep.Packages = append(rep.Packages, &goTestPackage{Path: m[1], Status: "notests", Dir: testPkgDir(m[1], modulePath)})
		case rePkgFail.MatchString(line):
			m := rePkgFail.FindStringSubmatch(line)
			pkg := &goTestPackage{Path: m[1], Status: "fail", Dir: testPkgDir(m[1], modulePath)}
			if m[2] != "" || len(pendBuild) > 0 {
				pkg.Status = "buildfail"
			}
			flush(pkg)
			rep.Packages = append(rep.Packages, pkg)
		case trimmed == "FAIL" || trimmed == "PASS":
			// The run's own summary line: not a package.
		case reTestFailLine.MatchString(trimmed):
			m := reTestFailLine.FindStringSubmatch(trimmed)
			curFailure = &goTestFailure{Test: m[1]}
			pendFailures = append(pendFailures, curFailure)
		case rePanicLine.MatchString(line):
			m := rePanicLine.FindStringSubmatch(line)
			msg := strings.TrimSuffix(m[1], " [recovered]")
			pendPanic = &goTestPanic{Message: msg}
			inPanic = true
		case inPanic && rePanicFileLine.MatchString(line):
			m := rePanicFileLine.FindStringSubmatch(line)
			ln, _ := strconv.Atoi(m[2])
			if len(pendFrames) > 0 {
				pendFrames[len(pendFrames)-1].File = m[1]
				pendFrames[len(pendFrames)-1].Line = ln
			}
		case inPanic && rePanicFuncLine.MatchString(line) && !strings.HasPrefix(line, "goroutine "):
			m := rePanicFuncLine.FindStringSubmatch(line)
			pendFrames = append(pendFrames, panicFrame{Func: m[1]})
		case reBuildErrLine.MatchString(line):
			m := reBuildErrLine.FindStringSubmatch(line)
			ln, _ := strconv.Atoi(m[2])
			col, _ := strconv.Atoi(m[3])
			pendBuild = append(pendBuild, &goBuildError{
				File: strings.TrimPrefix(m[1], "./"),
				Line: ln, Col: col, Message: m[4],
			})
		case curFailure != nil && curFailure.File == "" && !inPanic && (strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t")):
			// An indented line under a failing test: the
			// first file:line one is its assertion.
			if m := reTestFileLine.FindStringSubmatch(trimmed); m != nil && !strings.HasPrefix(m[3], "+0x") {
				ln, _ := strconv.Atoi(m[2])
				curFailure.File = m[1]
				curFailure.Line = ln
				curFailure.Message = m[3]
			}
		}
	}
	if len(rep.Packages) == 0 {
		rep.NoResults = true
		for _, ln := range strings.Split(out, "\n") {
			if s := strings.TrimSpace(ln); s != "" {
				rep.GoError = s
				break
			}
		}
	}
	return rep
}

// testPkgDir turns an import path into its project-relative
// folder via the module path ("" when it can't — the report
// then renders bare file names rather than inventing one).
func testPkgDir(pkgPath, modulePath string) string {
	if modulePath == "" || !strings.HasPrefix(pkgPath, modulePath) {
		return ""
	}
	return strings.TrimPrefix(pkgPath[len(modulePath):], "/")
}

// choosePanicFrame picks the frame the report names: the
// first one in the project's own code (its file lives under
// the root), else the first frame of all.
func choosePanicFrame(frames []panicFrame, rootAbs string) panicFrame {
	for _, fr := range frames {
		if fr.File == "" {
			continue
		}
		if rootAbs != "" && strings.HasPrefix(fr.File, rootAbs+string(filepath.Separator)) {
			rel, err := filepath.Rel(rootAbs, fr.File)
			if err == nil {
				return panicFrame{Func: baseFuncName(fr.Func), File: filepath.ToSlash(rel), Line: fr.Line}
			}
		}
	}
	for _, fr := range frames {
		if fr.File != "" {
			return panicFrame{Func: baseFuncName(fr.Func), File: fr.File, Line: fr.Line}
		}
	}
	return panicFrame{}
}

// baseFuncName shortens a frame's qualified function
// (example.com/mod/api.Boom) to its bare name (Boom).
func baseFuncName(qualified string) string {
	if i := strings.LastIndex(qualified, "."); i >= 0 {
		return qualified[i+1:]
	}
	return qualified
}

// ---------------------------------------------------------------------
// The plain-words gloss for compiler errors.

// goTestBuildGloss translates one compiler message's shape
// into plain words ("" when the shape is not one of the
// known ones — the message itself is quoted either way).
func goTestBuildGloss(msg string) string {
	lower := strings.ToLower(msg)
	switch {
	case strings.Contains(lower, "undefined:"):
		return "that name doesn't exist in this package — a typo, or something not yet written or imported"
	case strings.Contains(lower, "imported and not used"):
		return "Go won't compile a file that imports a package it never uses — use the import or take it out"
	case strings.Contains(lower, "declared and not used"):
		return "Go won't compile a variable that is set but never used"
	case strings.Contains(lower, "cannot use") || strings.Contains(lower, "mismatched types"):
		return "a value of one type is being used where a different type is needed"
	case strings.Contains(lower, "missing return"):
		return "the function promises a result, but one path through it reaches the end without returning one"
	case strings.Contains(lower, "no required module provides package"):
		return "an import names a package this module doesn't have yet"
	case strings.Contains(lower, "expected") && strings.Contains(lower, "found"):
		return "something is missing or out of place — a bracket, a comma, or a keyword"
	}
	return ""
}

// ---------------------------------------------------------------------
// The index clause: what a failing test is, from its own doc.

// testContextClause returns one clause of context for a
// failing test from the project index: the first sentence
// of the test's own doc comment, or of the function it
// names (TestAdd → Add). The subject is read in the
// failing test's own package first — a same-named function
// in another package is a different function, and its doc
// would describe the wrong code; only when the test's own
// package has no such function does any namesake speak.
// "" when nothing resolves — the report never invents
// context.
func testContextClause(testName, dir string) string {
	p := ensureTestIndex()
	if p == nil {
		return ""
	}
	var testFn, subjectHere, subjectAny *analyze.Func
	subjectName := strings.TrimPrefix(testName, "Test")
	if i := strings.Index(subjectName, "/"); i >= 0 {
		subjectName = subjectName[:i]
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.Name == testName && testFn == nil {
				testFn = fn
			}
			if subjectName != "" && subjectName != testName && fn.Name == subjectName && !fn.IsTest {
				if subjectAny == nil {
					subjectAny = fn
				}
				if testFuncDir(fn) == dir && subjectHere == nil {
					subjectHere = fn
				}
			}
		}
	}
	subject := subjectHere
	if subject == nil {
		subject = subjectAny
	}
	doc := ""
	if testFn != nil {
		doc = testFn.Doc
	}
	if doc == "" && subject != nil {
		doc = subject.Doc
	}
	return firstSentence(doc)
}

// testFuncDir is a function file's project-relative folder
// in the report's terms ("" for the project root).
func testFuncDir(fn *analyze.Func) string {
	d := filepath.Dir(filepath.FromSlash(fn.File))
	if d == "." {
		return ""
	}
	return filepath.ToSlash(d)
}

// firstSentence renders a doc comment's first sentence
// without its trailing period ("" for an empty comment).
func firstSentence(doc string) string {
	doc = strings.Join(strings.Fields(doc), " ")
	if doc == "" {
		return ""
	}
	if i := strings.Index(doc, ". "); i > 0 {
		doc = doc[:i]
	}
	return strings.TrimSuffix(doc, ".")
}

// ensureTestIndex loads (or refreshes) the current
// project's index the neighbours' way — the funcedit
// resolution's pattern — for context clauses only. Nil
// means no index to read; the report proceeds without
// clauses rather than failing.
func ensureTestIndex() *analyze.Project {
	rootRaw := creationRoot()
	if rootRaw == "" {
		return nil
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			return nil
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		return nil
	}
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		p, err := analyze.Analyze(rootAbs)
		switch {
		case err == nil:
			setCurrentProject(rootRaw, p)
		case strings.Contains(err.Error(), "no Go packages found"):
			setCurrentProject(rootRaw, &analyze.Project{Root: rootAbs})
		default:
			return nil
		}
	} else if !ensureFreshProject() {
		return nil
	}
	return lastAnalyzeProject
}

// ---------------------------------------------------------------------
// Rendering the report.

// renderGoTestReport renders the parsed run in the chat's
// plain voice. folder is the subtree the user named (""
// = the whole project), used by the green and no-tests
// lines to name their scope.
func renderGoTestReport(rep *goTestReport, folder string) string {
	tag := cTag("gollemer [runner]>")
	var b strings.Builder
	if rep.TimedOut {
		fmt.Fprintf(&b, "%s The tests didn't finish — I stopped them after %s, so this is only part of the story.", tag, testRunTimeout)
		if rep.FailedPackages() == 0 {
			b.WriteString(" Nothing had reported a failure yet.\n")
			return b.String()
		}
		b.WriteString(" Here's what had already failed by then:\n")
		writeFailureBlocks(&b, rep)
		return b.String()
	}
	if rep.NoResults {
		if rep.GoError != "" {
			errLine := rep.GoError
			if len(errLine) > 200 {
				errLine = errLine[:200] + "…"
			}
			fmt.Fprintf(&b, "%s I couldn't run the tests: %s\n", tag, errLine)
		} else {
			fmt.Fprintf(&b, "%s I couldn't run the tests — go test printed nothing at all.\n", tag)
		}
		return b.String()
	}
	failed := rep.FailedPackages()
	if failed == 0 {
		ran := rep.RanPackages()
		noTests := 0
		for _, p := range rep.Packages {
			if p.Status == "notests" {
				noTests++
			}
		}
		switch {
		case ran == 0:
			fmt.Fprintf(&b, "%s There are no tests to run%s — %s, and not one of them has a test file yet.\n",
				tag, testScopePhrase(folder), plural(len(rep.Packages), "package", "packages"))
		default:
			fmt.Fprintf(&b, "%s All the tests%s pass — %s ran their tests and every one is green.",
				tag, testScopePhrase(folder), plural(ran, "package", "packages"))
			if noTests > 0 {
				verb := "have"
				if noTests == 1 {
					verb = "has"
				}
				fmt.Fprintf(&b, " %s %s no test files, so nothing ran in them.", plural(noTests, "other package", "other packages"), verb)
			}
			b.WriteString("\n")
		}
		return b.String()
	}
	fmt.Fprintf(&b, "%s The tests ran — %s failed.\n", tag,
		fmt.Sprintf("%d of %d %s", failed, rep.RanPackages(), pluralWord(rep.RanPackages(), "package", "packages")))
	writeFailureBlocks(&b, rep)
	b.WriteString(testStartPointer(rep))
	return b.String()
}

// writeFailureBlocks renders one block per failed package:
// its name, then each failure on its own line.
func writeFailureBlocks(b *strings.Builder, rep *goTestReport) {
	for _, p := range rep.Packages {
		if p.Status != "fail" && p.Status != "buildfail" {
			continue
		}
		b.WriteString("\n")
		if p.Status == "buildfail" {
			fmt.Fprintf(b, "%s doesn't compile:\n", p.Path)
			for _, be := range p.BuildErrs {
				fmt.Fprintf(b, "  BUILD %s:%d:%d: %s", be.File, be.Line, be.Col, strings.TrimSuffix(be.Message, "."))
				if gloss := goTestBuildGloss(be.Message); gloss != "" {
					fmt.Fprintf(b, " — in plain words: %s", gloss)
				}
				b.WriteString(".\n")
			}
			if len(p.BuildErrs) == 0 {
				b.WriteString("  The build failed, but the output names no single error line.\n")
			}
			continue
		}
		fmt.Fprintf(b, "%s:\n", p.Path)
		panickedTest := ""
		if p.Panic != nil {
			for _, f := range p.Failures {
				if f.Message == "" {
					panickedTest = f.Test
				}
			}
		}
		wrote := false
		for _, f := range p.Failures {
			if p.Panic != nil && f.Test == panickedTest && f.Message == "" {
				continue // the panic line below speaks for it
			}
			wrote = true
			if f.File != "" {
				fmt.Fprintf(b, "  %s failed at %s:%d: %s", f.Test, testJoinFile(p.Dir, f.File), f.Line, strings.TrimSuffix(f.Message, "."))
			} else {
				fmt.Fprintf(b, "  %s failed", f.Test)
			}
			if clause := testContextClause(f.Test, p.Dir); clause != "" {
				fmt.Fprintf(b, ". (%s.)\n", clause)
			} else {
				b.WriteString(".\n")
			}
		}
		if p.Panic != nil {
			wrote = true
			name := panickedTest
			if name == "" {
				name = "A test"
			}
			fmt.Fprintf(b, "  %s panicked: %s.", name, strings.TrimSuffix(p.Panic.Message, "."))
			if p.Panic.Func != "" && p.Panic.File != "" {
				fmt.Fprintf(b, " The first frame in your code is %s, at %s:%d.", p.Panic.Func, p.Panic.File, p.Panic.Line)
			}
			b.WriteString("\n")
		}
		if !wrote {
			b.WriteString("  Its tests failed without naming one test — the test program itself stopped early.\n")
		}
	}
}

// testStartPointer ends a failure report by naming where
// to start: the first failure, honestly — never a fix.
func testStartPointer(rep *goTestReport) string {
	for _, p := range rep.Packages {
		if p.Status == "buildfail" && len(p.BuildErrs) > 0 {
			be := p.BuildErrs[0]
			return fmt.Sprintf("\nStart with %s:%d — nothing runs until the build passes.\n", be.File, be.Line)
		}
		if p.Status == "fail" {
			if p.Panic != nil && p.Panic.File != "" {
				return fmt.Sprintf("\nStart with %s in %s — that's where the panic surfaced.\n", p.Panic.Func, p.Panic.File)
			}
			for _, f := range p.Failures {
				if f.File != "" {
					return fmt.Sprintf("\nStart with %s in %s — it's the first failure.\n", f.Test, testJoinFile(p.Dir, f.File))
				}
				return fmt.Sprintf("\nStart with %s in %s — it's the first failure.\n", f.Test, p.Path)
			}
			return fmt.Sprintf("\nStart with %s — it's the first package that failed.\n", p.Path)
		}
	}
	return ""
}

// testJoinFile joins a package folder and a printed base
// name into the project-relative path the report renders.
func testJoinFile(dir, base string) string {
	if dir == "" {
		return base
	}
	return dir + "/" + base
}

// testScopePhrase names the run's scope in the green
// lines: " in api/" for a subtree, "" for the whole project.
func testScopePhrase(folder string) string {
	if folder == "" {
		return ""
	}
	return " in " + folder + "/"
}

// plural renders "1 package" / "3 packages".
func plural(n int, one, many string) string {
	return fmt.Sprintf("%d %s", n, pluralWord(n, one, many))
}

func pluralWord(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}
