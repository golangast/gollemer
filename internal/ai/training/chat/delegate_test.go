package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// delegateChatFixture is the end-to-end stage: a store package
// to delegate FROM, an app (package main) and a lib package to
// delegate INTO.
func delegateChatFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/delegatechat\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package delegatechat\n")
	writeFile(t, dir, "store/store.go", `package store

type Item struct{ ID int }

func Load(id int) (Item, error) { return Item{ID: id}, nil }
`)
	writeFile(t, dir, "app/main.go", "package main\n\nfunc main() {}\n")
	writeFile(t, dir, "lib/lib.go", "package lib\n\nfunc Local(n int) int { return n * 2 }\n")
	writeFile(t, dir, "lib/other.go", "package lib\n")
	return dir
}

// countDelegateBackups walks dir for leftover backup files.
func countDelegateBackups(t *testing.T, dir string) int {
	t.Helper()
	n := 0
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(d.Name(), ".gollemer-backup") {
			n++
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func TestParseDelegateForms(t *testing.T) {
	claims := []struct {
		line, file, source, wrapper string
	}{
		{"add a function in app/main.go that calls Load", "app/main.go", "Load", ""},
		{"add a function LoadItem in app/main.go that calls store.Load", "app/main.go", "store.Load", "LoadItem"},
		{"wrap Load in app/main.go", "app/main.go", "Load", ""},
		{"add a function in the app/main.go that calls Load.", "app/main.go", "Load", ""},
		{"Add a function in lib/other.go that calls Local", "lib/other.go", "Local", ""},
	}
	for _, c := range claims {
		file, source, wrapper, ok := parseDelegate(c.line)
		if !ok {
			t.Errorf("parseDelegate(%q) not claimed", c.line)
			continue
		}
		if file != c.file || source != c.source || wrapper != c.wrapper {
			t.Errorf("parseDelegate(%q) = (%q, %q, %q), want (%q, %q, %q)", c.line, file, source, wrapper, c.file, c.source, c.wrapper)
		}
	}
	for _, line := range []string{
		"add a webserver to jim/main.go",               // add-server's
		"create a file in folder jim with a webserver", // creations'
		"create a folder named jim",                    // creations'
		"i want to build a webserver",                  // resources'
		"rename Describe to Explain",                   // rename's
		"add a function like Greet named GreetLoudly",  // the clone shape's
		"what does Load do",
		"add a function in app/main.go", // no source named
		"wrap it up",
	} {
		if _, _, _, ok := parseDelegate(line); ok {
			t.Errorf("parseDelegate(%q) claimed a line that isn't a delegate", line)
		}
	}
}

// The neighboring write shapes keep their own lines: each line
// below is claimed by its owner and NOT by the delegate parser
// (proven above); here the owners speak for themselves.
func TestDelegateNeighboursKeepTheirLines(t *testing.T) {
	if _, ok := parseAddServer("add a webserver to jim/main.go"); !ok {
		t.Error("add-server lost its line")
	}
	if _, ok := parseCreate("create a folder named jim"); !ok {
		t.Error("creations lost its line")
	}
	if _, _, ok := parseBuildGoal("i want to build a webserver"); !ok {
		t.Error("resources lost its line")
	}
	if _, ok := parseRenameSentence("rename Describe to Explain"); !ok {
		t.Error("rename lost its line")
	}
	if _, ok := parseAssembledEdit("add a function like Greet named GreetLoudly"); !ok {
		t.Error("the clone shape lost its line")
	}
}

func TestDelegateCrossPackageEndToEnd(t *testing.T) {
	dir := delegateChatFixture(t)
	withServerProject(t, dir)
	// A non-default mode, to prove modes survive the write.
	if err := os.Chmod(filepath.Join(dir, "app", "main.go"), 0o755); err != nil {
		t.Fatal(err)
	}
	var handled bool
	out := captureStdout(t, func() {
		handled = runDelegate("add a function in app/main.go that calls Load", yes)
	})
	if !handled {
		t.Fatal("delegate line not handled")
	}
	// The plan showed the provenance before the gate. (LinkPath
	// renders relative to the process cwd, so in tests the path
	// is long; its tail is the source fact.)
	if !strings.Contains(out, "built from ") || !strings.Contains(out, "store/store.go:5 — func Load(id int) (Item, error)") {
		t.Errorf("plan output missing the provenance line:\n%s", out)
	}
	if !strings.Contains(out, "Done —") {
		t.Errorf("no success outcome:\n%s", out)
	}
	main := readBack(t, dir, "app/main.go")
	for _, want := range []string{`"example.com/delegatechat/store"`, "func Load(id int) (store.Item, error)", "return store.Load(id)"} {
		if !strings.Contains(main, want) {
			t.Errorf("applied file missing %q:\n%s", want, main)
		}
	}
	if fi, err := os.Stat(filepath.Join(dir, "app", "main.go")); err != nil || fi.Mode().Perm() != 0o755 {
		t.Errorf("main.go mode after a successful delegate = %v (err %v), want 0755 preserved", fi.Mode().Perm(), err)
	}
	stub := readBack(t, dir, "app/main_test.go")
	if !strings.Contains(stub, "func TestLoad(t *testing.T)") || !strings.Contains(stub, "TODO") {
		t.Errorf("test skeleton wrong:\n%s", stub)
	}
	if n := countDelegateBackups(t, dir); n != 0 {
		t.Errorf("%d backup files left after a successful delegate", n)
	}
	if _, ok := buildPackageDir(filepath.Join(dir, "app")); !ok {
		t.Error("app package doesn't build after the delegate")
	}
}

func TestDelegateDeclineKeepsBytes(t *testing.T) {
	dir := delegateChatFixture(t)
	withServerProject(t, dir)
	before := readBack(t, dir, "app/main.go")
	out := captureStdout(t, func() {
		runDelegate("add a function in app/main.go that calls Load", no)
	})
	if !strings.Contains(out, "Not applied") {
		t.Errorf("decline should say so:\n%s", out)
	}
	if got := readBack(t, dir, "app/main.go"); got != before {
		t.Errorf("declined delegate changed the file:\n%s", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "app", "main_test.go")); !os.IsNotExist(err) {
		t.Error("declined delegate created the test skeleton")
	}
	if n := countDelegateBackups(t, dir); n != 0 {
		t.Errorf("%d backup files left after a decline", n)
	}
}

func TestDelegateSamePackageEndToEnd(t *testing.T) {
	dir := delegateChatFixture(t)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		runDelegate("add a function Double in lib/other.go that calls Local", yes)
	})
	if !strings.Contains(out, "Done —") {
		t.Fatalf("same-package delegate didn't land:\n%s", out)
	}
	other := readBack(t, dir, "lib/other.go")
	if !strings.Contains(other, "func Double(n int) int") || !strings.Contains(other, "return Local(n)") {
		t.Errorf("same-package wrapper wrong:\n%s", other)
	}
	if strings.Contains(other, "example.com/delegatechat/lib") {
		t.Errorf("same-package delegate imported its own package:\n%s", other)
	}
	if stub := readBack(t, dir, "lib/other_test.go"); !strings.Contains(stub, "func TestDouble(t *testing.T)") {
		t.Errorf("same-package skeleton wrong:\n%s", stub)
	}
	// Unnamed, the same request is the honest refusal.
	before := readBack(t, dir, "lib/other.go")
	out = captureStdout(t, func() {
		runDelegate("add a function in lib/other.go that calls Local", yes)
	})
	if !strings.Contains(out, "needs its own name") {
		t.Errorf("unnamed same-package delegate should be refused plainly:\n%s", out)
	}
	if got := readBack(t, dir, "lib/other.go"); got != before {
		t.Error("refused same-package delegate changed the file")
	}
}

func TestDelegateMissingAndTypeRefusals(t *testing.T) {
	dir := delegateChatFixture(t)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		if !runDelegate("add a function in app/main.go that calls Nope", yes) {
			t.Error("missing-symbol line fell through instead of being claimed")
		}
	})
	if !strings.Contains(out, `can't find a function called "Nope"`) {
		t.Errorf("missing-symbol refusal wrong:\n%s", out)
	}
	// A type's name gets the functions-only refusal — claimed,
	// never a fall-through to a brain.
	out = captureStdout(t, func() {
		if !runDelegate("add a function in app/main.go that calls Item", yes) {
			t.Error("type-name line fell through instead of being claimed")
		}
	})
	if !strings.Contains(out, "not a function") {
		t.Errorf("type-name refusal wrong:\n%s", out)
	}
	if strings.Contains(readBack(t, dir, "app/main.go"), "func Load") {
		t.Error("a refusal wrote code")
	}
}

// The bounded retry, end to end: the target already imports a
// DIFFERENT package whose declared name is also "store" (dir
// legacy/), so the primary's plain import of the real store
// collides — the exact import/qualification failure class the
// one retry exists for. The plan's alternate (the source under
// the storepkg alias, added alongside) must win, be reported,
// and fire exactly once.
func TestDelegateRetryOnQualifierCollision(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/retry\n\ngo 1.21\n")
	writeFile(t, dir, "doc.go", "package retry\n")
	writeFile(t, dir, "store/store.go", "package store\n\nfunc Load(id int) (int, error) { return id, nil }\n")
	writeFile(t, dir, "legacy/legacy.go", "package store\n\nfunc Legacy() int { return 7 }\n")
	writeFile(t, dir, "app/main.go", `package main

import (
	"fmt"

	"example.com/retry/legacy"
)

func main() { fmt.Println(store.Legacy()) }
`)
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		runDelegate("add a function in app/main.go that calls Load", yes)
	})
	if !strings.Contains(out, "Done —") {
		t.Fatalf("retry fixture delegate didn't land:\n%s", out)
	}
	if !strings.Contains(out, "the retry the plan named") || !strings.Contains(out, "storepkg") {
		t.Errorf("outcome doesn't report the retry and the form that won:\n%s", out)
	}
	main := readBack(t, dir, "app/main.go")
	for _, want := range []string{`storepkg "example.com/retry/store"`, "return storepkg.Load(id)", "store.Legacy()"} {
		if !strings.Contains(main, want) {
			t.Errorf("retried file missing %q:\n%s", want, main)
		}
	}
	if n := strings.Count(main, `storepkg "`); n != 1 {
		t.Errorf("alias import appears %d times — the retry must fire exactly once:\n%s", n, main)
	}
	if _, ok := buildPackageDir(filepath.Join(dir, "app")); !ok {
		t.Error("app package doesn't build after the retry")
	}
	if n := countDelegateBackups(t, dir); n != 0 {
		t.Errorf("%d backup files left after the retry", n)
	}
}

// A pre-broken sibling in the target package doesn't block a
// good delegate: the verdict is baseline-relative, and the
// outcome says the red predates the edit.
func TestDelegateBaselineRedSibling(t *testing.T) {
	dir := delegateChatFixture(t)
	writeFile(t, dir, "app/other.go", "package main\n\nfunc broken() int { return DoesNotExist }\n")
	withServerProject(t, dir)
	out := captureStdout(t, func() {
		runDelegate("add a function in app/main.go that calls Load", yes)
	})
	if !strings.Contains(out, "Done —") {
		t.Fatalf("delegate into a baseline-red package didn't land:\n%s", out)
	}
	if !strings.Contains(out, "had 1 build error(s) before this edit") {
		t.Errorf("outcome doesn't own the pre-existing red:\n%s", out)
	}
	if !strings.Contains(readBack(t, dir, "app/main.go"), "return store.Load(id)") {
		t.Error("wrapper missing after a baseline-red success")
	}
	if n := countDelegateBackups(t, dir); n != 0 {
		t.Errorf("%d backup files left", n)
	}
}

// The mirror image: a delegate that would ADD a new error is
// caught and restored. The collision is invisible to the
// planner — the colliding file lands after the project was
// analyzed (a stale index, the way a file edited in another
// window would be) — so only the baseline-relative verdict can
// catch it. Mode and bytes must both come back.
func TestDelegateNewErrorCaughtAndRestored(t *testing.T) {
	dir := delegateChatFixture(t)
	withEditsProject(t, dir) // analyzed BEFORE the colliding file exists
	writeFile(t, dir, "app/load.go", `package main

import "example.com/delegatechat/store"

func Load(id int) (store.Item, error) { return store.Load(id) }
`)
	mainAbs := filepath.Join(dir, "app", "main.go")
	if err := os.Chmod(mainAbs, 0o755); err != nil {
		t.Fatal(err)
	}
	before := readBack(t, dir, "app/main.go")
	out := captureStdout(t, func() {
		runDelegate("add a function in app/main.go that calls Load", yes)
	})
	if !strings.Contains(out, "restored your files from the on-disk backups") {
		t.Fatalf("new-error delegate wasn't rolled back:\n%s", out)
	}
	if strings.Contains(out, "Done —") {
		t.Errorf("rolled-back delegate claims success:\n%s", out)
	}
	if got := readBack(t, dir, "app/main.go"); got != before {
		t.Errorf("restored file differs from the original:\n%s", got)
	}
	if fi, err := os.Stat(mainAbs); err != nil || fi.Mode().Perm() != 0o755 {
		t.Errorf("mode after restore = %v (err %v), want 0755", fi.Mode().Perm(), err)
	}
	if _, err := os.Stat(filepath.Join(dir, "app", "main_test.go")); !os.IsNotExist(err) {
		t.Error("rolled-back delegate left its test skeleton behind")
	}
	if n := countDelegateBackups(t, dir); n != 0 {
		t.Errorf("%d backup files left after a rollback", n)
	}
}

// A leftover backup beside the target stops the edit before
// anything is written — it may be the only copy of an original.
func TestDelegateLeftoverBackupRefusal(t *testing.T) {
	dir := delegateChatFixture(t)
	writeFile(t, dir, "app/.main.go.gollemer-backup", "someone's original")
	withServerProject(t, dir)
	before := readBack(t, dir, "app/main.go")
	out := captureStdout(t, func() {
		runDelegate("add a function in app/main.go that calls Load", yes)
	})
	if !strings.Contains(out, "already a leftover backup") {
		t.Errorf("leftover backup didn't stop the edit:\n%s", out)
	}
	if got := readBack(t, dir, "app/main.go"); got != before {
		t.Error("edit proceeded despite the leftover backup")
	}
	if got := readBack(t, dir, "app/.main.go.gollemer-backup"); got != "someone's original" {
		t.Error("the leftover backup itself was touched")
	}
}

// The crash-recovery scan: leftovers are named with the exact
// restore command (mode included when the sidecar recorded
// one), and .git is not scanned.
func TestLeftoverBackupNotices(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, ".doc.go.gollemer-backup", "orig doc")
	writeFile(t, dir, ".doc.go.gollemer-backup.mode", "0644\n")
	writeFile(t, dir, "app/.main.go.gollemer-backup", "orig main")
	writeFile(t, dir, ".git/.x.go.gollemer-backup", "not ours to report")
	notices := leftoverBackupNotices(dir)
	if len(notices) != 2 {
		t.Fatalf("notices = %d, want 2:\n%s", len(notices), strings.Join(notices, "\n"))
	}
	joined := strings.Join(notices, "\n")
	if !strings.Contains(joined, "doc.go") || !strings.Contains(joined, "cp ") || !strings.Contains(joined, "chmod 0644") {
		t.Errorf("doc.go notice missing the restore command with its mode:\n%s", joined)
	}
	for _, n := range notices {
		if strings.Contains(n, "main.go") && strings.Contains(n, "chmod") {
			t.Errorf("main.go notice invents a mode its sidecar never recorded:\n%s", n)
		}
		if strings.Contains(n, ".git") {
			t.Errorf("notice reaches into .git:\n%s", n)
		}
	}
}

// The retry classifier and the build verdict, on canned compiler
// output (the strings a real collision prints).
func TestDelegateRetryClassifier(t *testing.T) {
	collision := "# example.com/retry/app\n" +
		"./main.go:5:2: store redeclared in this block\n" +
		"./main.go:5:2: \"example.com/retry/store\" imported and not used\n" +
		"./main.go:12:10: undefined: store.Load\n"
	if !delegateRetryApplies(collision, map[string]bool{}, "app/main.go", "Load") {
		t.Error("the import-collision output isn't retry-eligible")
	}
	typeErr := "./main.go:12:10: cannot use \"x\" (untyped string constant) as int value in argument to store.Load\n"
	if delegateRetryApplies(typeErr, map[string]bool{}, "app/main.go", "Load") {
		t.Error("a type error in the wrapper is retry-eligible — it must not be")
	}
	sibling := "./other.go:3:20: undefined: DoesNotExist\n"
	if delegateRetryApplies(sibling, map[string]bool{}, "app/main.go", "Load") {
		t.Error("an error in another file is retry-eligible — it must not be")
	}
	if delegateRetryApplies("", map[string]bool{}, "app/main.go", "Load") {
		t.Error("empty output is retry-eligible — it must not be")
	}

	// The verdict: clean passes; a subset of the baseline with
	// nothing pointing at the delegate passes; a new key fails;
	// an unparseable failure with no keys anywhere fails (the
	// degenerate guard); a baseline key whose line names the
	// generated call fails.
	if !delegateBuildPasses("", true, nil, nil, "app/main.go", "Load") {
		t.Error("a clean build doesn't pass")
	}
	base := buildErrorKeys(sibling)
	if !delegateBuildPasses(sibling, false, base, base, "app/main.go", "Load") {
		t.Error("baseline-only errors don't pass")
	}
	withNew := sibling + typeErr
	if delegateBuildPasses(withNew, false, buildErrorKeys(withNew), base, "app/main.go", "Load") {
		t.Error("a new error key passes — it must not")
	}
	if delegateBuildPasses("panic: something", false, map[string]bool{}, map[string]bool{}, "app/main.go", "Load") {
		t.Error("a keyless failure passes — the degenerate guard must fail it")
	}
	refs := "./main.go:12:10: undefined: store.Load\n"
	refsKeys := buildErrorKeys(refs)
	if delegateBuildPasses(refs, false, refsKeys, refsKeys, "app/main.go", "Load") {
		t.Error("an error naming the generated call passes — it must not")
	}
}
