package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// learnFixture is a small real project with one of everything the
// flow teaches from: an entry point, a call chain, a package-level
// variable set inside a function, and an exported type using a
// mutex (a learn-path concept with a Go feature).
func learnFixture(t *testing.T) string {
	t.Helper()
	root := filepath.Join(t.TempDir(), "learnproj")
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/learnproj\n\ngo 1.21\n")
	write("main.go", `package main

import "fmt"

// AnswerCount is how many answers have been loaded.
var AnswerCount int

func main() {
	LoadAnswers()
	fmt.Println(Compute(21))
}

// LoadAnswers loads the answers.
func LoadAnswers() { AnswerCount = 42 }

// Compute doubles a number.
func Compute(n int) int { return n * 2 }
`)
	write("counter.go", `package main

import "sync"

// Counter counts things safely.
type Counter struct {
	mu sync.Mutex
	n  int
}

// Inc adds one.
func (c *Counter) Inc() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.n++
}

// Value reads the count.
func (c *Counter) Value() int { return c.n }
`)
	return root
}

// withLearnState isolates the learn-flow globals and the Friend
// store for a test.
func withLearnState(t *testing.T) {
	t.Helper()
	learnResetForTest()
	friendResetForTest()
	t.Cleanup(func() {
		learnResetForTest()
		friendResetForTest()
	})
}

// The start shapes: bare "learn", URL targets, existing folders —
// and just as important, the shapes learn must NOT take from their
// existing owners (/learn-path, /learn-pieces, unresolvable words).
func TestParseLearnStart(t *testing.T) {
	dir := t.TempDir()
	accept := map[string]string{
		"learn":                               "",
		"Learn":                               "",
		"/learn":                              "",
		"learn mode":                          "",
		"learn https://github.com/ugorji/go":  "https://github.com/ugorji/go",
		"learn github.com/ugorji/go":          "github.com/ugorji/go",
		"/learn https://github.com/ugorji/go": "https://github.com/ugorji/go",
		"learn " + dir:                        dir,
	}
	for line, want := range accept {
		got, ok := parseLearnStart(line)
		if !ok || got != want {
			t.Errorf("parseLearnStart(%q) = (%q, %v), want (%q, true)", line, got, ok, want)
		}
	}
	reject := []string{
		"learn path",                // the /learn-path sentence form
		"learn pieces",              // the /learn-pieces family
		"/learn-path",               // the older feature itself
		"/learn-pieces",             // the older feature itself
		"learning",                  // not the bare word
		"learn about goroutines",    // no resolvable target
		"learn nonexistent-xyz-123", // no such folder
		"what should I learn first", // the learn-path sentence form
	}
	for _, line := range reject {
		if got, ok := parseLearnStart(line); ok {
			t.Errorf("parseLearnStart(%q) = (%q, true), want no claim", line, got)
		}
	}
}

// Judging tolerance: short name or dotted name, any case, alone or
// inside a sentence; near-misses and other names are wrong.
func TestLearnAnswerRight(t *testing.T) {
	q := learnCheck{
		question: "Name one function that calls Compute.",
		accept:   []string{"main", "main.main", "LoadAnswers", "main.LoadAnswers"},
	}
	right := []string{"main", "Main", "main.main", "func main", "the main function",
		"LoadAnswers", "loadanswers", "main.LoadAnswers", "it's LoadAnswers", "LoadAnswers."}
	for _, a := range right {
		if !learnAnswerRight(a, q) {
			t.Errorf("learnAnswerRight(%q) = false, want true", a)
		}
	}
	wrong := []string{"Compute", "the moon", "", "domain", "mainline", "Counter"}
	for _, a := range wrong {
		if learnAnswerRight(a, q) {
			t.Errorf("learnAnswerRight(%q) = true, want false", a)
		}
	}
	entry := learnCheck{question: "What runs first?", accept: []string{"main", "main.main"}}
	if !learnAnswerRight("main", entry) || !learnAnswerRight("func main", entry) {
		t.Error("entry question should accept main / func main")
	}
	if learnAnswerRight("Compute", entry) {
		t.Error("entry question accepted Compute")
	}
}

// The check questions are generated from the fixture's REAL facts:
// every accepted name is checked back against the analyzer's own
// indexes (call graph, varflow), never taken on the generator's word.
func TestLearnChecksFromFacts(t *testing.T) {
	root := learnFixture(t)
	p, err := analyze.Analyze(root)
	if err != nil {
		t.Fatal(err)
	}
	checks := buildLearnChecks(p)
	if len(checks) < 2 || len(checks) > 3 {
		t.Fatalf("got %d checks, want 2-3: %+v", len(checks), checks)
	}
	byConcept := map[string]learnCheck{}
	for _, c := range checks {
		byConcept[c.concept] = c
	}
	// Entry point: the expected answer is the fixture's real main.
	entry, ok := byConcept["the entry point"]
	if !ok {
		t.Fatalf("no entry-point question in %+v", checks)
	}
	eps := p.EntryPoints()
	if len(eps) != 1 || eps[0].Name != "main" {
		t.Fatalf("fixture entry points = %v", eps)
	}
	if !learnAnswerRight("main", entry) {
		t.Error("entry question does not accept main")
	}
	if !strings.Contains(entry.fact, "main.go:") {
		t.Errorf("entry fact lacks the real location: %q", entry.fact)
	}
	// Call question: the subject is a real function and every
	// accepted caller name belongs to a function the analyzer's
	// call graph lists as a caller of it.
	var callQ *learnCheck
	for i := range checks {
		if strings.HasPrefix(checks[i].question, "Name one function that calls ") {
			callQ = &checks[i]
		}
	}
	if callQ == nil {
		t.Fatalf("no call-relation question in %+v", checks)
	}
	subject := strings.TrimSuffix(strings.TrimPrefix(callQ.question, "Name one function that calls "), ".")
	var subjFn *analyze.Func
	for _, fn := range p.AllFuncs() {
		if fn.Name == subject {
			subjFn = fn
		}
	}
	if subjFn == nil {
		t.Fatalf("call question subject %q is not a fixture function", subject)
	}
	callerNames := map[string]bool{}
	for _, id := range subjFn.Callers {
		if c := p.FuncByID(id); c != nil {
			callerNames[c.Name] = true
			callerNames[c.Display()] = true
		}
	}
	for _, acc := range callQ.accept {
		if !callerNames[acc] {
			t.Errorf("call question accepts %q, which the call graph does not list as a caller of %s", acc, subject)
		}
	}
	// Dataflow question: the accepted setters are exactly the
	// functions VarFacts says assign the variable.
	var dataQ *learnCheck
	for i := range checks {
		if strings.HasPrefix(checks[i].question, "Where does ") {
			dataQ = &checks[i]
		}
	}
	if dataQ == nil {
		t.Fatalf("no dataflow question in %+v", checks)
	}
	setters := map[string]bool{}
	foundVar := false
	for _, f := range p.VarFacts() {
		if f.Name == dataQ.concept {
			foundVar = true
			for _, s := range f.Setters {
				setters[s.Short] = true
				setters[s.Func] = true
			}
		}
	}
	if !foundVar || len(setters) == 0 {
		t.Fatalf("VarFacts has no setters for %q, yet a dataflow question was generated", dataQ.concept)
	}
	for _, acc := range dataQ.accept {
		if !setters[acc] {
			t.Errorf("dataflow question accepts %q, which VarFacts does not list as a setter of %s", acc, dataQ.concept)
		}
	}
	if !learnAnswerRight("LoadAnswers", *dataQ) {
		t.Errorf("dataflow question should accept its real setter LoadAnswers: %+v", *dataQ)
	}
}

// The whole flow, driven line by line the way the unified loop
// drives it: stages in order, right answers confirmed, a wrong
// answer judged AND remembered by the Friend, and the flow exits
// into free chat.
func TestLearnModeEndToEnd(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	withLearnState(t)
	root := learnFixture(t)
	withEditsProject(t, root)

	drive := func(line string) (bool, string) {
		var handled bool
		out := captureStdout(t, func() { handled = tryLearnCommand(line) })
		return handled, out
	}

	handled, out := drive("learn")
	if !handled || !learnActive {
		t.Fatalf("learn not claimed/active, output:\n%s", out)
	}
	if !strings.Contains(out, "Let's learn") || !strings.Contains(out, "Learn 1 of 5") {
		t.Errorf("start output missing intro/first stage:\n%s", out)
	}
	// The story names the fixture's own facts.
	if !strings.Contains(out, "example.com/learnproj") {
		t.Errorf("story stage not about the fixture:\n%s", out)
	}
	// Mid-flow, an unrelated line is NOT the flow's (the guide's
	// interrupt rule): it falls through and the flow waits.
	if handled, _ := drive("hello there"); handled {
		t.Error("unrelated line claimed by the flow mid-stage")
	}
	// Map.
	handled, out = drive("next")
	if !handled || !strings.Contains(out, "Learn 2 of 5") || !strings.Contains(out, "WHERE TO START") {
		t.Errorf("map stage wrong:\n%s", out)
	}
	// Walkthrough.
	handled, out = drive("next")
	if !handled || !strings.Contains(out, "Learn 3 of 5") || !strings.Contains(out, "main") {
		t.Errorf("walkthrough stage wrong:\n%s", out)
	}
	// Concepts: the fixture's Counter, with its mutex feature.
	handled, out = drive("next")
	if !handled || !strings.Contains(out, "Learn 4 of 5") || !strings.Contains(out, "Counter") {
		t.Errorf("concepts stage wrong:\n%s", out)
	}
	if !strings.Contains(out, "mutexes") {
		t.Errorf("concepts stage missing the mutex feature:\n%s", out)
	}
	// Into the check: question 1 is the entry point.
	handled, out = drive("next")
	if !handled || !strings.Contains(out, "Learn 5 of 5") || !strings.Contains(out, "Question 1 of") {
		t.Errorf("check stage did not open:\n%s", out)
	}
	if len(learnChecks) < 2 {
		t.Fatalf("only %d check questions generated", len(learnChecks))
	}
	if learnChecks[0].concept != "the entry point" {
		t.Fatalf("question 1 concept = %q, want the entry point", learnChecks[0].concept)
	}
	// Right answer: confirmed, nothing recorded.
	handled, out = drive("main")
	if !handled || !strings.Contains(out, "Right") {
		t.Errorf("right answer not confirmed:\n%s", out)
	}
	if n := len(friendLoad().Misses); n != 0 {
		t.Errorf("a right answer recorded %d misses", n)
	}
	// Question 2: answer with one of its accepted names (white-box:
	// the generator's own fact, verified against the analyzer in
	// TestLearnChecksFromFacts).
	q2 := learnChecks[1]
	handled, out = drive(q2.accept[0])
	if !handled || !strings.Contains(out, "Right") {
		t.Errorf("question 2 right answer (%q) not confirmed:\n%s", q2.accept[0], out)
	}
	if n := len(friendLoad().Misses); n != 0 {
		t.Errorf("two right answers recorded %d misses", n)
	}
	// Question 3 (the last): a deliberately wrong answer is judged
	// wrong, the correct fact is stated, and the Friend records it.
	q3 := learnChecks[2]
	handled, out = drive("the moon did it")
	if !handled || !strings.Contains(out, "Not quite") {
		t.Errorf("wrong answer not judged:\n%s", out)
	}
	if !strings.Contains(out, q3.fact) {
		t.Errorf("wrong-answer reply missing the correct fact %q:\n%s", q3.fact, out)
	}
	misses := friendLoad().Misses
	if len(misses) != 1 {
		t.Fatalf("Friend recorded %d misses, want exactly 1: %+v", len(misses), misses)
	}
	m := misses[0]
	if m.Concept != q3.concept || m.Question != q3.question || m.Fact != q3.fact {
		t.Errorf("miss recorded wrong: %+v (question was %+v)", m, q3)
	}
	if m.Project != root {
		t.Errorf("miss project = %q, want the fixture root %q", m.Project, root)
	}
	if m.Day == "" {
		t.Error("miss has no day recorded")
	}
	// The flow is over: closing line, back to free chat.
	if learnActive {
		t.Error("flow still active after the last question")
	}
	if !strings.Contains(out, "ask me anything") {
		t.Errorf("closing line missing the hand-back:\n%s", out)
	}
	if handled, _ := drive("next"); handled {
		t.Error("'next' still claimed after the flow ended")
	}
}

// Skipping a question shows the fact but records nothing, and the
// exit phrase ends the flow gracefully at any point.
func TestLearnSkipAndStop(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	withLearnState(t)
	root := learnFixture(t)
	withEditsProject(t, root)
	drive := func(line string) (bool, string) {
		var handled bool
		out := captureStdout(t, func() { handled = tryLearnCommand(line) })
		return handled, out
	}
	drive("learn")
	for i := 0; i < 4; i++ {
		drive("next")
	}
	if learnCheckAt < 0 {
		t.Fatal("check stage not reached")
	}
	handled, out := drive("next") // skip question 1
	if !handled || !strings.Contains(out, "Skipped") {
		t.Errorf("skip not handled:\n%s", out)
	}
	if n := len(friendLoad().Misses); n != 0 {
		t.Errorf("a skip recorded %d misses", n)
	}
	handled, out = drive("stop learning")
	if !handled || learnActive {
		t.Errorf("stop learning did not end the flow:\n%s", out)
	}
	if !strings.Contains(out, "Learn mode ended") {
		t.Errorf("stop output missing the graceful end:\n%s", out)
	}
}

// No project anywhere: the honest line, never a brain, never the
// bundled demo dressed up as the user's project.
func TestLearnNoProjectHonestLine(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	withLearnState(t)
	empty := t.TempDir()
	oldProj, oldRoot, oldURoot := lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot
	t.Cleanup(func() { lastAnalyzeProject, lastAnalyzeRoot, unifiedProjectRoot = oldProj, oldRoot, oldURoot })
	lastAnalyzeProject, lastAnalyzeRoot = nil, ""
	unifiedProjectRoot = empty // a folder with no Go code in it
	var handled bool
	out := captureStdout(t, func() { handled = tryLearnCommand("learn") })
	if !handled {
		t.Fatalf("bare learn not claimed, output:\n%s", out)
	}
	if learnActive {
		t.Error("flow started with no project")
	}
	if !strings.Contains(out, "open a project first") {
		t.Errorf("missing the honest line:\n%s", out)
	}
	if strings.Contains(out, "pocketcounter") || strings.Contains(out, "salutations") {
		t.Errorf("no-project learn fell back to the demo or a brain:\n%s", out)
	}
}

// "learn <url>" claims the repo deterministically — with the real
// classifier loaded, against a decoy current project — and the
// repo becomes the flow's project.
func TestLearnURLClaimsRepo(t *testing.T) {
	withRealIntentModel(t)
	cacheRoot := seedRepoCache(t)
	withLearnState(t)
	withEditsProject(t, decoyProject(t))
	var handled bool
	out := captureStdout(t, func() {
		handled = tryLearnCommand("learn https://github.com/ugorji/go")
	})
	if !handled || !learnActive {
		t.Fatalf("learn <url> not claimed, output:\n%s", out)
	}
	if learnRoot != cacheRoot || lastAnalyzeRoot != cacheRoot {
		t.Errorf("flow project = %q/%q, want the cached repo %q", learnRoot, lastAnalyzeRoot, cacheRoot)
	}
	if strings.Contains(out, "DecoyMarker") || strings.Contains(out, "salutations") {
		t.Errorf("learn answered about the decoy or a brain:\n%s", out)
	}
	if !strings.Contains(out, "Learn 1 of 5") {
		t.Errorf("flow did not open on the repo:\n%s", out)
	}
	captureStdout(t, func() { tryLearnCommand("stop learning") })
}

// The older features keep their lines: learn never swallows
// /learn-path or /learn-pieces, and /learn-path still renders.
func TestLearnLeavesOlderFeaturesAlone(t *testing.T) {
	withLearnState(t)
	root := learnFixture(t)
	withEditsProject(t, root)
	for _, line := range []string{"/learn-path", "/learn-pieces", "learn path", "learn pieces"} {
		if tryLearnCommand(line) {
			t.Errorf("tryLearnCommand claimed %q, which belongs to an older feature", line)
		}
	}
	out := captureStdout(t, func() {
		if !handleBeginnerCommand("/learn-path", nil) {
			t.Error("/learn-path no longer handled")
		}
	})
	if !strings.Contains(out, "Learn this project in this order") || !strings.Contains(out, "Counter") {
		t.Errorf("/learn-path output changed:\n%s", out)
	}
}
