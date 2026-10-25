package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const freshnessNotice = "You changed files outside the chat since I last read this project — I re-read it before planning."

// stampPastSnapshot makes an externally rewritten fixture file
// unambiguously newer than the freshness baseline, independent
// of the filesystem's mtime granularity.
func stampPastSnapshot(t *testing.T, dir, rel string) {
	t.Helper()
	if lastAnalyzeSnapshot == nil {
		t.Fatal("no freshness snapshot to stamp against")
	}
	state, ok := lastAnalyzeSnapshot.files[filepath.ToSlash(rel)]
	if !ok {
		t.Fatalf("snapshot lacks %s", rel)
	}
	when := state.modTime.Add(2 * time.Second)
	if err := os.Chtimes(filepath.Join(dir, rel), when, when); err != nil {
		t.Fatal(err)
	}
}

// An outside signature change is re-read before the delegate
// plans: the plan and the applied wrapper carry the NEW
// signature, not the one the session analyzed first.
func TestFreshnessDelegatePlansFromNewSignature(t *testing.T) {
	dir := delegateChatFixture(t)
	withEditsProject(t, dir)
	writeFile(t, dir, "store/store.go", `package store

type Item struct{ ID int }

func Load(id string) (Item, error) { return Item{ID: len(id)}, nil }
`)
	stampPastSnapshot(t, dir, "store/store.go")

	var handled bool
	out := captureStdout(t, func() {
		handled = runDelegate("add a function in app/main.go that calls Load", yes)
	})
	if !handled {
		t.Fatal("delegate line not handled")
	}
	if !strings.Contains(out, freshnessNotice) {
		t.Fatalf("stale signature did not trigger the re-read notice:\n%s", out)
	}
	if !strings.Contains(out, "func Load(id string) (store.Item, error)") {
		t.Fatalf("delegate plan does not use the new signature:\n%s", out)
	}
	main := readBack(t, dir, "app/main.go")
	if !strings.Contains(main, "func Load(id string) (store.Item, error)") || !strings.Contains(main, "return store.Load(id)") {
		t.Fatalf("applied wrapper does not use the new signature:\n%s", main)
	}
}

// A deleted snapshotted file and a newly appeared .go file each
// make the index stale: the next edit line re-reads once, says
// so, and only then plans (here, declined at the gate).
func TestFreshnessDeletedAndNewFilesTriggerReRead(t *testing.T) {
	t.Run("deleted file", func(t *testing.T) {
		dir := editsChatFixture(t)
		withEditsProject(t, dir)
		beforeSnap := lastAnalyzeSnapshot
		if err := os.Remove(filepath.Join(dir, "other", "other.go")); err != nil {
			t.Fatal(err)
		}
		out := captureStdout(t, func() {
			runAssembledEdit("add a plum case to Describe", no)
		})
		if !strings.Contains(out, freshnessNotice) {
			t.Fatalf("deleted file did not trigger the re-read:\n%s", out)
		}
		if lastAnalyzeSnapshot == beforeSnap {
			t.Fatal("snapshot was not refreshed after the deletion-triggered re-read")
		}
		if !strings.Contains(out, "Here's the edit I assembled") {
			t.Fatalf("the request did not proceed against the fresh index:\n%s", out)
		}
	})

	t.Run("new file", func(t *testing.T) {
		dir := editsChatFixture(t)
		withEditsProject(t, dir)
		beforeSnap := lastAnalyzeSnapshot
		writeFile(t, dir, "shop/extra.go", "package shop\n\n// Extra is new.\nfunc Extra() int { return 1 }\n")
		out := captureStdout(t, func() {
			runAssembledEdit("add a plum case to Describe", no)
		})
		if !strings.Contains(out, freshnessNotice) {
			t.Fatalf("new file did not trigger the re-read:\n%s", out)
		}
		if lastAnalyzeSnapshot == beforeSnap {
			t.Fatal("snapshot was not refreshed after the new-file re-read")
		}
		if !strings.Contains(out, "Here's the edit I assembled") {
			t.Fatalf("the request did not proceed against the fresh index:\n%s", out)
		}
	})
}

// The common case stays silent: no outside change means no
// notice, no re-read, and the same snapshot object throughout.
func TestFreshnessQuietWhenNothingChanged(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	beforeSnap := lastAnalyzeSnapshot
	beforeProj := lastAnalyzeProject
	out := captureStdout(t, func() {
		runAssembledEdit("add a plum case to Describe", no)
	})
	if strings.Contains(out, "outside the chat") {
		t.Fatalf("fresh index produced a stale notice:\n%s", out)
	}
	if lastAnalyzeSnapshot != beforeSnap {
		t.Fatal("snapshot was replaced even though nothing changed")
	}
	if lastAnalyzeProject != beforeProj {
		t.Fatal("project was re-analyzed even though nothing changed")
	}
}

// Writes the chat makes itself refresh the baseline through the
// spine's post-edit re-read — including the scaffold file the
// first edit creates — so the very next edit line sees a fresh
// index and stays silent.
func TestFreshnessOwnWritesDoNotTriggerNotice(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	out := captureStdout(t, func() {
		if !runAssembledEdit("add a plum case to Describe", yes) {
			t.Error("first (own) edit not claimed")
		}
	})
	if !strings.Contains(out, "Done —") {
		t.Fatalf("first (own) edit did not apply:\n%s", out)
	}
	out = captureStdout(t, func() {
		if !runAssembledEdit("register thanks in Greetings", yes) {
			t.Error("second edit not claimed")
		}
	})
	if strings.Contains(out, "outside the chat") {
		t.Fatalf("the chat's own write tripped the freshness guard:\n%s", out)
	}
	if !strings.Contains(out, "Done —") {
		t.Fatalf("second edit did not apply:\n%s", out)
	}
}

// A file the create feature itself writes joins the snapshot, so
// creating a .go file in the open project does not read as an
// outside change at the next edit line.
func TestFreshnessOwnCreationDoesNotTriggerNotice(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	out := captureStdout(t, func() {
		if !runCreate("create a file extra.go in folder shop", yes) {
			t.Error("create line not claimed")
		}
	})
	if strings.Contains(out, "outside the chat") {
		t.Fatalf("creation in a fresh project produced a stale notice:\n%s", out)
	}
	if got := readBack(t, dir, "shop/extra.go"); got != "package shop\n" {
		t.Fatalf("created file wrong: %q", got)
	}
	out = captureStdout(t, func() {
		runAssembledEdit("add a plum case to Describe", no)
	})
	if strings.Contains(out, "outside the chat") {
		t.Fatalf("the chat's own creation tripped the freshness guard:\n%s", out)
	}
}

// When the outside change leaves the tree unreadable, the guard
// stops the write honestly: the failure is named, no plan is
// shown, and nothing is planned from the stale index.
func TestFreshnessUnreadableTreeStopsTheWrite(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/freshfail\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\n// Pick chooses a letter.\nfunc Pick(kind string) string {\n\tswitch kind {\n\tcase \"a\":\n\t\treturn \"A\"\n\tdefault:\n\t\treturn \"?\"\n\t}\n}\n")
	withEditsProject(t, dir)
	broken := "package main\n\nfunc Pick( {\n"
	writeFile(t, dir, "main.go", broken)
	stampPastSnapshot(t, dir, "main.go")

	asked := false
	ask := func(string) (string, error) {
		asked = true
		return "y", nil
	}
	var handled bool
	out := captureStdout(t, func() {
		handled = runAssembledEdit("add a b case to Pick", ask)
	})
	if !handled {
		t.Fatal("edit line not claimed")
	}
	if asked {
		t.Fatal("a stopped write reached the [y/n] gate")
	}
	if !strings.Contains(out, "couldn't re-read it") || !strings.Contains(out, "won't plan from the old index") {
		t.Fatalf("stale-stop was not honest about the failed re-read:\n%s", out)
	}
	if strings.Contains(out, "Here's the edit I assembled") {
		t.Fatalf("a plan was shown from the stale index:\n%s", out)
	}
	if got := readBack(t, dir, "main.go"); got != broken {
		t.Fatalf("the stopped write touched the file:\n%s", got)
	}
}

// A file the analyzer already knows it cannot parse is part of
// the baseline, not a perpetual "new file": a project carrying a
// broken .go file from the start must not re-read (and announce)
// on every write.
func TestFreshnessParseErrorFileIsBaselined(t *testing.T) {
	dir := editsChatFixture(t)
	writeFile(t, dir, "other/broken.go", "package other\n\nfunc Broken( {\n")
	withEditsProject(t, dir)
	if _, ok := lastAnalyzeSnapshot.files["other/broken.go"]; !ok {
		t.Fatal("snapshot does not baseline the parse-error file")
	}
	beforeSnap := lastAnalyzeSnapshot
	out := captureStdout(t, func() {
		runAssembledEdit("add a plum case to Describe", no)
	})
	if strings.Contains(out, "outside the chat") {
		t.Fatalf("a known parse-error file tripped the freshness guard:\n%s", out)
	}
	if lastAnalyzeSnapshot != beforeSnap {
		t.Fatal("snapshot was replaced even though nothing changed")
	}
}

// A rolled-back edit restores the exact bytes the index was
// built from; the restore's mtime bump is the chat's own doing,
// so the next write line stays silent.
func TestFreshnessRolledBackEditDoesNotTriggerNotice(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	out := captureStdout(t, func() {
		if !runAssembledEdit("add a other.ModeMissing case to ModeName", yes) {
			t.Error("build-failing edit not claimed")
		}
	})
	if !strings.Contains(out, "restored your files") {
		t.Fatalf("the edit did not roll back as expected:\n%s", out)
	}
	out = captureStdout(t, func() {
		runAssembledEdit("add a plum case to Describe", no)
	})
	if strings.Contains(out, "outside the chat") {
		t.Fatalf("the chat's own rollback tripped the freshness guard:\n%s", out)
	}
}

// Read-only questions are out of the invariant's scope: they keep
// answering from the session's index without the guard firing,
// even after an outside change. (Any later WRITE still re-reads.)
func TestFreshnessReadOnlyQuestionIsNotGuarded(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	beforeProj := lastAnalyzeProject
	beforeSnap := lastAnalyzeSnapshot
	writeFile(t, dir, "shop/shop.go", strings.Replace(readBack(t, dir, "shop/shop.go"), "a fruit", "a tasty fruit", 1))
	out, ok := tryCodebaseQuestion("what does Describe do")
	if !ok {
		t.Fatal("codebase question not answered")
	}
	if !strings.Contains(out, "Describe") {
		t.Fatalf("answer does not cover Describe:\n%s", out)
	}
	if lastAnalyzeProject != beforeProj || lastAnalyzeSnapshot != beforeSnap {
		t.Fatal("a read-only question triggered a re-read — the guard is write-only")
	}
}
