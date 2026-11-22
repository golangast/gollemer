package chat

// The function-edit claim (funcedit.go): the two live
// failures from John's 2026-11-21 session. "remove the
// function named jake" — a project sentence about a
// function created minutes earlier — fell through every
// claim to the talk brain, which answered a social
// greeting; "have the function jake return a string"
// reached the [go] brain and got a lecture on string
// comparison. Both operations are real landings now,
// planned in the analyzer (editfuncedit.go) and carried by
// the write spine: these tests pin the LANDINGS end to end
// (gate yes writes and builds, gate no writes nothing, the
// plan warns about callers, main is refused at plan time,
// the conservative change-return refusals hold, a failed
// build restores the file byte-identical and writes no
// receipt), the gate discipline (an unresolved name falls
// through silently, a question is never claimed), the
// ambiguity question (which now lands after the pick), and
// the neighbouring file-delete refusal byte-for-byte —
// removing a function lands; removing a FILE is still the
// user's alone.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// funcEditJakeRoot is John's test shape: a module whose
// jake/main.go holds func jake beside func main.
var funcEditJakeRoot = map[string]string{
	"go.mod":       "module example.com/funceditjake\n\ngo 1.21\n",
	"doc.go":       "package funceditjake\n",
	"jake/main.go": "package main\n\nfunc jake() {}\n\nfunc main() {}\n",
}

const (
	funcEditRemoveDone = "Done — jake is removed from jake/main.go. I built the project and checked that it's really there."
	funcEditReturnDone = "Done — jake in jake/main.go now returns string. I built the project and checked that it's really there."
)

func TestFuncEditRemoveLands(t *testing.T) {
	isolateReceiptStore(t)
	dir := t.TempDir()
	for rel, body := range funcEditJakeRoot {
		writeFile(t, dir, rel, body)
	}
	t.Chdir(dir)
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("remove the function named jake", yes, nil) {
			t.Error("the remove line was not claimed")
		}
	})
	if !strings.Contains(out, "Here's the edit I assembled from this project's own code:") ||
		!strings.Contains(out, "Remove main.jake from jake/main.go") {
		t.Errorf("the removal plan was not shown:\n%s", out)
	}
	if !strings.Contains(out, "jake isn't called from anywhere else in this project — nothing else uses it.") {
		t.Errorf("the plan did not say jake has no callers:\n%s", out)
	}
	if !strings.Contains(out, funcEditRemoveDone) {
		t.Errorf("the landing did not print the partner confirmation %q:\n%s", funcEditRemoveDone, out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != "package main\n\nfunc main() {}\n" {
		t.Errorf("jake/main.go after the removal:\n%q\nwant jake gone and main kept", got)
	}
	if _, ok := lastReceiptOfKind(mustReceipts(t, dir), "remove_func"); !ok {
		t.Error("a landed removal wrote no receipt")
	}
}

func TestFuncEditReturnLands(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range funcEditJakeRoot {
		writeFile(t, dir, rel, body)
	}
	t.Chdir(dir)
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("have the function jake return a string", yes, nil) {
			t.Error("the return line was not claimed")
		}
	})
	if !strings.Contains(out, "func jake() becomes func jake() string") {
		t.Errorf("the plan did not show the signature change:\n%s", out)
	}
	if !strings.Contains(out, funcEditReturnDone) {
		t.Errorf("the landing did not print the partner confirmation %q:\n%s", funcEditReturnDone, out)
	}
	want := "package main\n\nfunc jake() string {\n\treturn \"\"\n}\n\nfunc main() {}\n"
	if got := readBack(t, dir, "jake/main.go"); got != want {
		t.Errorf("jake/main.go after the change:\n%q\nwant %q", got, want)
	}
}

func TestFuncEditRemoveGateNoWritesNothing(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range funcEditJakeRoot {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("remove the function named jake", no, nil) {
			t.Error("the remove line was not claimed")
		}
	})
	if !strings.Contains(out, "Not applied — every file is exactly as it was.") {
		t.Errorf("a no at the gate did not print the not-applied line:\n%s", out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != funcEditJakeRoot["jake/main.go"] {
		t.Errorf("a no at the gate still wrote:\n%q", got)
	}
}

// funcEditCallerRoot: main calls jake, so removing jake is
// a breaking edit — the plan must say so before the gate,
// and a yes must end in a build failure and a rollback.
var funcEditCallerRoot = map[string]string{
	"go.mod":       "module example.com/funceditcaller\n\ngo 1.21\n",
	"doc.go":       "package funceditcaller\n",
	"jake/main.go": "package main\n\nfunc main() {\n\tjake()\n}\n\nfunc jake() {}\n",
}

func TestFuncEditRemoveWithCallersWarns(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range funcEditCallerRoot {
		writeFile(t, dir, rel, body)
	}
	t.Chdir(dir)
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("remove the function named jake", no, nil) {
			t.Error("the remove line was not claimed")
		}
	})
	if !strings.Contains(out, "Heads up: jake is called from 1 place — main (jake/main.go:3). Removing it breaks those calls.") {
		t.Errorf("the plan did not warn about the caller before the gate:\n%s", out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != funcEditCallerRoot["jake/main.go"] {
		t.Errorf("a no at the gate still wrote:\n%q", got)
	}
}

func TestFuncEditRemoveCalledFunctionRollsBack(t *testing.T) {
	isolateReceiptStore(t)
	dir := t.TempDir()
	for rel, body := range funcEditCallerRoot {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	before := readBack(t, dir, "jake/main.go")
	out := captureStdout(t, func() {
		if !claimChain("remove the function named jake", yes, nil) {
			t.Error("the remove line was not claimed")
		}
	})
	if !strings.Contains(out, "That edit doesn't build, so I restored your files from the on-disk backups") {
		t.Errorf("the failed build did not report the rollback honestly:\n%s", out)
	}
	if strings.Contains(out, "Done —") {
		t.Errorf("a rolled-back edit printed a confirmation:\n%s", out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != before {
		t.Errorf("the rollback did not restore the file byte-identical:\nbefore %q\nafter  %q", before, got)
	}
	if _, ok := lastReceiptOfKind(mustReceipts(t, dir), "remove_func"); ok {
		t.Error("a rolled-back removal still wrote a receipt")
	}
}

func TestFuncEditRemoveMainRefused(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range funcEditJakeRoot {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("remove the function named main", yes, nil) {
			t.Error("the remove-main line was not claimed")
		}
	})
	if !strings.Contains(out, "Not doing that edit:") ||
		!strings.Contains(out, "main is the program's entry point") {
		t.Errorf("removing main was not refused at plan time:\n%s", out)
	}
	if strings.Contains(out, "Apply this edit?") {
		t.Errorf("the refusal still reached the gate:\n%s", out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != funcEditJakeRoot["jake/main.go"] {
		t.Errorf("the refusal still wrote:\n%q", got)
	}
}

func TestFuncEditReturnTodoBodyInsertsZero(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":       "module example.com/funcedittodo\n\ngo 1.21\n",
		"doc.go":       "package funcedittodo\n",
		"jake/main.go": "package main\n\nfunc jake() {\n\t// TODO: write jake's body\n}\n\nfunc main() {}\n",
	}
	for rel, body := range files {
		writeFile(t, dir, rel, body)
	}
	t.Chdir(dir)
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("have the function jake return a string", yes, nil) {
			t.Error("the return line was not claimed")
		}
	})
	if !strings.Contains(out, funcEditReturnDone) {
		t.Errorf("the landing did not print the partner confirmation:\n%s", out)
	}
	want := "package main\n\nfunc jake() string {\n\t// TODO: write jake's body\n\treturn \"\"\n}\n\nfunc main() {}\n"
	if got := readBack(t, dir, "jake/main.go"); got != want {
		t.Errorf("jake/main.go after the change:\n%q\nwant %q", got, want)
	}
}

func TestFuncEditReturnExistingResultsDeclines(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":       "module example.com/funceditresults\n\ngo 1.21\n",
		"doc.go":       "package funceditresults\n",
		"jake/main.go": "package main\n\nfunc jake() int {\n\treturn 0\n}\n\nfunc main() {}\n",
	}
	for rel, body := range files {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("have the function jake return a string", yes, nil) {
			t.Error("the return line was not claimed")
		}
	})
	if !strings.Contains(out, "Not doing that edit:") ||
		!strings.Contains(out, "already returns int") {
		t.Errorf("a function with results was not declined at plan time:\n%s", out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != files["jake/main.go"] {
		t.Errorf("the decline still wrote:\n%q", got)
	}
}

func TestFuncEditReturnInBodyDeclines(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":       "module example.com/funceditret\n\ngo 1.21\n",
		"doc.go":       "package funceditret\n",
		"jake/main.go": "package main\n\nfunc jake() {\n\tprintln(\"working\")\n\treturn\n}\n\nfunc main() {}\n",
	}
	for rel, body := range files {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("have the function jake return a string", yes, nil) {
			t.Error("the return line was not claimed")
		}
	})
	if !strings.Contains(out, "Not doing that edit:") ||
		!strings.Contains(out, "already has a return statement in its body (line 5)") {
		t.Errorf("a body with a return was not declined at plan time:\n%s", out)
	}
	if got := readBack(t, dir, "jake/main.go"); got != files["jake/main.go"] {
		t.Errorf("the decline still wrote:\n%q", got)
	}
}

func TestFuncEditReturnWithResultsInFixture(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":       "module example.com/funceditres\n\ngo 1.21\n",
		"doc.go":       "package funceditres\n",
		"jake/main.go": "package main\n\nfunc jake() int { return 0 }\n\nfunc main() {}\n",
	}
	for rel, body := range files {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("have jake return a string", yes, nil) {
			t.Error("the return line was not claimed")
		}
	})
	if !strings.Contains(out, "jake") {
		t.Errorf("the answer did not name the function:\n%s", out)
	}
}

func TestFuncEditMethodLands(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":         "module example.com/funceditmethod\n\ngo 1.21\n",
		"doc.go":         "package funceditmethod\n",
		"store/store.go": "package store\n\ntype Store struct{}\n\nfunc (s *Store) Count() int {\n\treturn 0\n}\n\nfunc (s *Store) Name() string {\n\treturn \"s\"\n}\n",
	}
	for rel, body := range files {
		writeFile(t, dir, rel, body)
	}
	t.Chdir(dir)
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("remove the function named Store.Count", yes, nil) {
			t.Error("the method line was not claimed")
		}
	})
	if !strings.Contains(out, "Done — Count is removed from store/store.go. I built the project and checked that it's really there.") {
		t.Errorf("the method removal did not land with its confirmation:\n%s", out)
	}
	want := "package store\n\ntype Store struct{}\n\nfunc (s *Store) Name() string {\n\treturn \"s\"\n}\n"
	if got := readBack(t, dir, "store/store.go"); got != want {
		t.Errorf("store/store.go after the removal:\n%q\nwant %q", got, want)
	}
}

func TestFuncEditUnresolvedNameFallsThrough(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range funcEditJakeRoot {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	for _, line := range []string{
		"remove the function named nosuch",
		"delete the function nosuch",
		"have the function nosuch return a string",
		"make nosuch return an int",
	} {
		if claimChain(line, yes, nil) {
			t.Errorf("%q was claimed for a name that resolves nowhere", line)
		}
	}
}

func TestFuncEditQuestionVetoes(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range funcEditJakeRoot {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	// Factual questions about the very same function are
	// the go brain's, never this claim's.
	for _, line := range []string{
		"what does jake return",
		"does jake return a string",
		"how do I remove the function jake",
		"can I delete the function jake",
		"why does jake return nothing",
	} {
		if claimChain(line, yes, nil) {
			t.Errorf("question %q was claimed by the function-edit catch", line)
		}
	}
}

func TestFuncEditAmbiguousAsksThenLands(t *testing.T) {
	dir := t.TempDir()
	files := map[string]string{
		"go.mod":    "module example.com/funceditamb\n\ngo 1.21\n",
		"doc.go":    "package funceditamb\n",
		"a/a.go":    "package a\n\nfunc jake() {}\n",
		"b/b.go":    "package b\n\nfunc jake() {}\n",
		"c/main.go": "package main\n\nfunc main() {}\n",
	}
	for rel, body := range files {
		writeFile(t, dir, rel, body)
	}
	t.Chdir(dir)
	withChatSession(t, dir, true)
	out := captureStdout(t, func() {
		if !claimChain("remove the function named jake", yes, nil) {
			t.Error("the ambiguous line was not claimed")
		}
	})
	if !strings.Contains(out, "There's more than one function named jake — which one do you mean? Nothing's been written.") {
		t.Errorf("the ambiguity question was not asked:\n%s", out)
	}
	if got := readBack(t, dir, "a/a.go"); got != files["a/a.go"] {
		t.Errorf("the question alone wrote to a/a.go:\n%q", got)
	}
	out = captureStdout(t, func() {
		if !claimChain("1", yes, nil) {
			t.Error("the pick was not claimed")
		}
	})
	if !strings.Contains(out, "Done — jake is removed from a/a.go. I built the project and checked that it's really there.") {
		t.Errorf("the picked removal did not land:\n%s", out)
	}
	if got := readBack(t, dir, "a/a.go"); got != "package a\n" {
		t.Errorf("a/a.go after the removal:\n%q\nwant just the package clause", got)
	}
	if got := readBack(t, dir, "b/b.go"); got != files["b/b.go"] {
		t.Errorf("the other jake was touched:\n%q", got)
	}
}

// TestFuncEditFileDeleteRefusalUntouched pins the
// neighbouring boundary: removing a FUNCTION lands now,
// but "remove the file" is still the composer's refusal,
// word for word — deleting files stays the user's alone.
func TestFuncEditFileDeleteRefusalUntouched(t *testing.T) {
	dir := t.TempDir()
	for rel, body := range funcEditJakeRoot {
		writeFile(t, dir, rel, body)
	}
	withChatSession(t, dir, true)
	want := "I can't delete files or folders — deleting is yours alone. In your terminal, the command is: rm jake/main.go. What I can do: create a file or folder here behind a [y/n] gate, and explain any command in plain words before you run it."
	out := captureStdout(t, func() {
		if !claimChain("remove the file jake/main.go", yes, nil) {
			t.Error("the file-delete line was not claimed by its own claim")
		}
	})
	if !strings.Contains(out, want) {
		t.Errorf("the file-delete refusal changed:\n%s\nwant it to contain %q", out, want)
	}
	if _, err := os.Stat(filepath.Join(dir, "jake", "main.go")); err != nil {
		t.Errorf("the file-delete line removed the file: %v", err)
	}
}
