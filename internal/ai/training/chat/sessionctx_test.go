package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The session working context (sessionctx.go): the chat
// remembers what it is building — the working folder, the
// last file it created or edited, the last piece landed —
// and bare follow-ups resolve against that work, out loud.
//
// The canonical journey is John's own 2026-11-02 session:
// four lines building one database package in
// jake/database/, which the context-less chat made him
// spell out in full every time.

func ctxNo(string) (string, error) { return "n", nil }

// seedCtxProject is the journey's ground: a module with a
// root program, nothing else.
func seedCtxProject(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/sessionctx\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {}\n")
	return dir
}

func ctxRun(t *testing.T, line string, ask editAsk, profile *UserProfile) (bool, string) {
	t.Helper()
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, ask, profile) })
	return handled, out
}

func ctxAbsent(t *testing.T, root, rel string) {
	t.Helper()
	if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); !os.IsNotExist(err) {
		t.Errorf("%s exists — it should never have been created (stat err: %v)", rel, err)
	}
}

// The canonical journey, end to end through the claim
// chain: the .db file, the package file, then two bare
// follow-ups that only the working context can settle —
// each landing in jake/database/, each printing its
// reading with the full path and the reason.
func TestSessionCtxJourney(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}

	// Step 1: the database file itself (the starter form
	// makes its folders).
	handled, out := ctxRun(t, "create file jake/database/database.db", yes, profile)
	if !handled || !strings.Contains(out, "Done — created") {
		t.Fatalf("step 1 not created:\n%s", out)
	}
	if workCtx.lastFile != "jake/database/database.db" || workCtx.folder != "jake/database" {
		t.Fatalf("step 1 context = %+v", workCtx)
	}

	// Step 2: the package file, spelled in full.
	handled, out = ctxRun(t, "create file jake/database/database.go", yes, profile)
	if !handled || !strings.Contains(out, "Done — created") {
		t.Fatalf("step 2 not created:\n%s", out)
	}
	if got := readBack(t, root, "jake/database/database.go"); got != "package database\n" {
		t.Fatalf("step 2 content = %q", got)
	}
	if workCtx.lastFile != "jake/database/database.go" || workCtx.pkg != "database" || workCtx.lastHow != "created" {
		t.Fatalf("step 2 context = %+v", workCtx)
	}

	// Step 3: the connection, aimed by bare name only. The
	// reading is printed, the plan names the full path, and
	// Connect lands in the file he just created — never at
	// the project root, where the bare name points today
	// without context.
	handled, out = ctxRun(t, "add a database connection function named Connect to database.go", yes, profile)
	if !handled {
		t.Fatalf("step 3 fell through:\n%s", out)
	}
	if !strings.Contains(out, `Reading "database.go" as jake/database/database.go — the file you just created.`) {
		t.Errorf("step 3 did not print the reading:\n%s", out)
	}
	if !strings.Contains(out, "jake/database/database.go") || !strings.Contains(out, "built the project and checked") {
		t.Errorf("step 3 did not land through the spine:\n%s", out)
	}
	if got := readBack(t, root, "jake/database/database.go"); !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("step 3 did not land Connect in the working file:\n%s", got)
	}
	ctxAbsent(t, root, "database.go")
	if workCtx.piece != "database connect" || workCtx.pieceName != "Connect" || workCtx.lastHow != "edited" {
		t.Errorf("step 3 piece context = %+v", workCtx)
	}

	// Step 4: a second file for the same package, named
	// bare — the folder reading puts it beside the work.
	handled, out = ctxRun(t, "add the file db.go", yes, profile)
	if !handled {
		t.Fatalf("step 4 fell through:\n%s", out)
	}
	if !strings.Contains(out, `Reading "db.go" as jake/database/db.go — the folder you're working in.`) {
		t.Errorf("step 4 did not print the folder reading:\n%s", out)
	}
	if got := readBack(t, root, "jake/database/db.go"); got != "package database\n" {
		t.Errorf("step 4 content = %q", got)
	}
	ctxAbsent(t, root, "db.go")
}

// The pronoun tails: each reads as the file just created
// and lands the same plan the spelled-out line lands.
func TestSessionCtxPronounTails(t *testing.T) {
	for _, tail := range []string{"to it", "to that file", "to the same file"} {
		root := seedCtxProject(t)
		withChatSession(t, root, true)
		t.Chdir(root)
		profile := &UserProfile{}
		if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
			t.Fatalf("%s: folder create fell through:\n%s", tail, out)
		}
		if handled, out := ctxRun(t, "create file jake/database/database.go", yes, profile); !handled {
			t.Fatalf("%s: file create fell through:\n%s", tail, out)
		}
		line := "add a database connection function named Connect " + tail
		handled, out := ctxRun(t, line, yes, profile)
		if !handled {
			t.Fatalf("%q fell through:\n%s", line, out)
		}
		phrase := strings.TrimPrefix(tail, "to ")
		if !strings.Contains(out, "Reading \""+phrase+"\" as jake/database/database.go — the file you just created.") {
			t.Errorf("%q did not print its reading:\n%s", line, out)
		}
		if got := readBack(t, root, "jake/database/database.go"); !strings.Contains(got, "func Connect() (*sql.DB, error)") {
			t.Errorf("%q did not land Connect:\n%s", line, got)
		}
		ctxAbsent(t, root, "database.go")
	}
}

// No context, no reading: the pronoun lines keep today's
// behavior byte for byte — "to it" meets the concept
// resolver's refusal, and "to that file" is not the
// concept grammar's at all (the two-token tail declines),
// so the line reaches the write boundary, which misreads
// "that file" as a behaviour clause — today's answer,
// unchanged.
func TestSessionCtxPronounNoContext(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}

	handled, out := ctxRun(t, "add a database connection function named Connect to it", yes, profile)
	if !handled {
		t.Fatalf("\"to it\" fell through without context:\n%s", out)
	}
	if !strings.Contains(out, "there's no folder it in this project") {
		t.Errorf("\"to it\" without context lost today's refusal:\n%s", out)
	}
	if strings.Contains(out, "Reading ") {
		t.Errorf("\"to it\" without context printed a reading:\n%s", out)
	}

	handled, out = ctxRun(t, "add a database connection function named Connect to that file", yes, profile)
	if !handled {
		t.Fatalf("\"to that file\" fell through without context:\n%s", out)
	}
	if !strings.Contains(out, "I can't add a function with that extra clause") {
		t.Errorf("\"to that file\" without context lost today's boundary answer:\n%s", out)
	}
	if strings.Contains(out, "Reading ") {
		t.Errorf("\"to that file\" without context printed a reading:\n%s", out)
	}
	ctxAbsent(t, root, "database.go")
}

// Two files fit: the bare name matches the tracked file
// AND a namesake stands at the project root. The answer
// is guidance naming both, with the ask spelled once in
// full — nothing is planned, gated, or written.
func TestSessionCtxAmbiguousBareName(t *testing.T) {
	root := seedCtxProject(t)
	writeFile(t, root, "database.go", "package main\n")
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	if handled, out := ctxRun(t, "create file jake/database/database.go", yes, profile); !handled {
		t.Fatalf("file create fell through:\n%s", out)
	}
	beforeRoot := readBack(t, root, "database.go")
	beforeWork := readBack(t, root, "jake/database/database.go")

	handled, out := ctxRun(t, "add a database connection function named Connect to database.go", yes, profile)
	if !handled {
		t.Fatalf("the ambiguous line fell through:\n%s", out)
	}
	for _, want := range []string{
		`"database.go" names more than one file here`,
		"jake/database/database.go (the file you just created)",
		"database.go (a file at the project root)",
		"I won't guess which one you mean",
		`"add a database connection function named Connect to jake/database/database.go"`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "Apply this edit?") {
		t.Errorf("the ambiguous line reached a gate:\n%s", out)
	}
	if readBack(t, root, "database.go") != beforeRoot || readBack(t, root, "jake/database/database.go") != beforeWork {
		t.Errorf("the ambiguous line wrote something")
	}
}

// A cold session's bare name keeps today's root-relative
// behavior exactly: the declaration creates database.go at
// the project root, and no reading is printed.
func TestSessionCtxNoContextBareNameToday(t *testing.T) {
	root := seedCtxProject(t)
	if err := os.MkdirAll(filepath.Join(root, "jake", "database"), 0o755); err != nil {
		t.Fatal(err)
	}
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}

	handled, out := ctxRun(t, "add a database connection function named Connect to database.go", yes, profile)
	if !handled {
		t.Fatalf("the cold bare-name line fell through:\n%s", out)
	}
	if strings.Contains(out, "Reading ") {
		t.Errorf("a cold session printed a reading:\n%s", out)
	}
	if got := readBack(t, root, "database.go"); !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("the cold line did not create the root file as today:\n%s", got)
	}
	ctxAbsent(t, root, "jake/database/database.go")
}

// An explicit target with context live: no reading is
// printed — the assumption note belongs to resolved asks
// only — and the line lands exactly as it does today.
func TestSessionCtxExplicitTargetNoNote(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	if handled, out := ctxRun(t, "create file jake/database/database.go", yes, profile); !handled {
		t.Fatalf("file create fell through:\n%s", out)
	}
	handled, out := ctxRun(t, "add a database connection function named Connect to jake/database/database.go", yes, profile)
	if !handled {
		t.Fatalf("the explicit line fell through:\n%s", out)
	}
	if strings.Contains(out, "Reading ") {
		t.Errorf("an explicit target printed a reading:\n%s", out)
	}
	if got := readBack(t, root, "jake/database/database.go"); !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("the explicit line did not land Connect:\n%s", got)
	}
}

// The folder candidate: the bare name is a DIFFERENT file
// that already exists in the working folder — it resolves
// there, with the folder reason, not to the tracked file
// and not to the root.
func TestSessionCtxFolderCandidate(t *testing.T) {
	root := seedCtxProject(t)
	if err := os.MkdirAll(filepath.Join(root, "jake", "database"), 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, root, "jake/database/other.go", "package database\n")
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create file jake/database/database.go", yes, profile); !handled {
		t.Fatalf("setup create fell through:\n%s", out)
	}
	handled, out := ctxRun(t, "add a struct named Profile to other.go", yes, profile)
	if !handled {
		t.Fatalf("the folder-candidate line fell through:\n%s", out)
	}
	if !strings.Contains(out, `Reading "other.go" as jake/database/other.go — a file in the folder you're working in.`) {
		t.Errorf("the folder candidate did not print its reading:\n%s", out)
	}
	if got := readBack(t, root, "jake/database/other.go"); !strings.Contains(got, "type Profile struct") {
		t.Errorf("the struct did not land in the folder's file:\n%s", got)
	}
	ctxAbsent(t, root, "other.go")
}

// A resolved target can still refuse — the reading prints
// first, and the refusal speaks about the resolved file.
func TestSessionCtxResolvedRefusalIsVisible(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	if handled, out := ctxRun(t, "create file jake/database/database.go", yes, profile); !handled {
		t.Fatalf("file create fell through:\n%s", out)
	}
	before := readBack(t, root, "jake/database/database.go")
	handled, out := ctxRun(t, "add a database query in main to database.go", yes, profile)
	if !handled {
		t.Fatalf("the resolved-refusal line fell through:\n%s", out)
	}
	if !strings.Contains(out, `Reading "database.go" as jake/database/database.go — the file you just created.`) {
		t.Errorf("the refusal did not print the reading:\n%s", out)
	}
	if !strings.Contains(out, "jake/database/database.go declares no functions") {
		t.Errorf("the refusal is not about the resolved file:\n%s", out)
	}
	if readBack(t, root, "jake/database/database.go") != before {
		t.Errorf("the refusal changed the file")
	}
	ctxAbsent(t, root, "database.go")
}

// Refusals and declined gates note nothing: the context
// still describes the last work that actually landed.
func TestSessionCtxRefusalsNoteNothing(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	if handled, out := ctxRun(t, "create file jake/database/database.go", yes, profile); !handled {
		t.Fatalf("file create fell through:\n%s", out)
	}
	want := workCtx

	// A refusal: a statement into the function-less file
	// (the no-host guidance).
	if handled, out := ctxRun(t, "add a database to jake/database/database.go", yes, profile); !handled {
		t.Fatalf("the refusal line fell through:\n%s", out)
	}
	if workCtx != want {
		t.Errorf("a refusal moved the context:\nbefore %+v\nafter  %+v", want, workCtx)
	}

	// A declined gate: the plan is offered and refused.
	handled, out := ctxRun(t, "add a struct named Profile to jake/database/database.go", ctxNo, profile)
	if !handled || !strings.Contains(out, "Not applied") {
		t.Fatalf("the declined line did not decline:\n%s", out)
	}
	if workCtx != want {
		t.Errorf("a declined gate moved the context:\nbefore %+v\nafter  %+v", want, workCtx)
	}
	if got := readBack(t, root, "jake/database/database.go"); strings.Contains(got, "Profile") {
		t.Errorf("the declined edit landed anyway:\n%s", got)
	}
}

// A bare creation right after a folder creation — before
// any file exists — reads in the new folder: the folder
// context alone carries it.
func TestSessionCtxCreateAfterFolderCreate(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	if workCtx.lastFile != "" || workCtx.folder != "jake/database" {
		t.Fatalf("folder context = %+v", workCtx)
	}
	handled, out := ctxRun(t, "add the file db.go", yes, profile)
	if !handled {
		t.Fatalf("the bare create fell through:\n%s", out)
	}
	if !strings.Contains(out, `Reading "db.go" as jake/database/db.go — the folder you're working in.`) {
		t.Errorf("the bare create did not print the folder reading:\n%s", out)
	}
	if got := readBack(t, root, "jake/database/db.go"); got != "package database\n" {
		t.Errorf("the bare create landed wrong: %q", got)
	}
	ctxAbsent(t, root, "db.go")
}

// The create family's own bare names keep their shipped
// root anchoring even with the context live: the folder
// reading belongs to the add-verb follow-up form, and a
// bare "create store.go" lands at the root with no
// reading printed — the bare-file package-rules contract,
// holding inside a session too.
func TestSessionCtxCreateVerbStaysRooted(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	if handled, out := ctxRun(t, "create file jake/database/database.go", yes, profile); !handled {
		t.Fatalf("file create fell through:\n%s", out)
	}
	handled, out := ctxRun(t, "create store.go", yes, profile)
	if !handled {
		t.Fatalf("the create-verb line fell through:\n%s", out)
	}
	if strings.Contains(out, "Reading ") {
		t.Errorf("a create-verb bare name printed a reading:\n%s", out)
	}
	if got := readBack(t, root, "store.go"); got != "package main\n" {
		t.Errorf("store.go = %q, want the root fallback, package main", got)
	}
	ctxAbsent(t, root, "jake/database/store.go")
}

// The context belongs to its project: another project's
// root makes it inert, setCurrentProject to a different
// project clears it, and a same-project re-read keeps it.
func TestSessionCtxProjectSwitch(t *testing.T) {
	root := seedCtxProject(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	profile := &UserProfile{}
	if handled, out := ctxRun(t, "create a folder jake/database", yes, profile); !handled {
		t.Fatalf("folder create fell through:\n%s", out)
	}
	abs, ok := claimRootAbs(root)
	if !ok || !workContextLive(abs) {
		t.Fatalf("context not live for its own root (%q, %v)", abs, ok)
	}
	other := seedCtxProject(t)
	otherAbs, _ := claimRootAbs(other)
	if workContextLive(otherAbs) {
		t.Errorf("context claims another project's root")
	}
	// A same-project re-read (the freshness guard's move)
	// keeps the context.
	if !analyzeQuiet(root) {
		t.Fatal("re-analyze failed")
	}
	if workCtx.folder != "jake/database" {
		t.Errorf("a same-project re-read cleared the context: %+v", workCtx)
	}
	// A real switch clears it.
	if !analyzeQuiet(other) {
		t.Fatal("other analyze failed")
	}
	if workCtx != (workContext{}) {
		t.Errorf("a project switch kept the old context: %+v", workCtx)
	}
}

// Unit pins for the readers: shapes that must never read,
// and the root-namesake spoiler.
func TestSessionCtxReadersUnit(t *testing.T) {
	root := seedCtxProject(t)
	writeFile(t, root, "jake/database/database.go", "package database\n")
	abs, _ := claimRootAbs(root)
	workCtx = workContext{root: abs, folder: "jake/database", lastFile: "jake/database/database.go", lastHow: "created"}
	defer resetWorkContext()

	// A path-shaped target never reads.
	if r, n, g := resolveContextTarget("jake/database/database.go", "x", abs); r != "" || n != "" || g != "" {
		t.Errorf("path target read: %q %q %q", r, n, g)
	}
	// A folder target never reads (files only).
	if r, n, g := resolveContextTarget("jake", "x", abs); r != "" || n != "" || g != "" {
		t.Errorf("folder target read: %q %q %q", r, n, g)
	}
	// A name nothing in the context fits does not read.
	if r, n, g := resolveContextTarget("missing.go", "x", abs); r != "" || n != "" || g != "" {
		t.Errorf("unfitting name read: %q %q %q", r, n, g)
	}
	// The pronoun rewrite leaves non-tail lines alone.
	for _, line := range []string{
		"add a function named Greet to main.go",
		"add a print line in main to jim/main.go",
		"what does it do",
	} {
		if out, note := resolvePronounTail(line, abs); note != "" || out != line {
			t.Errorf("resolvePronounTail(%q) = %q, %q — want untouched", line, out, note)
		}
	}
	// The create reader needs a folder context and a bare
	// name; at the root it stays silent.
	if f, n := resolveCreateFolder("db.go", abs); f != "jake/database" || n == "" {
		t.Errorf("create reader = %q, %q", f, n)
	}
	workCtx.folder = ""
	if f, n := resolveCreateFolder("db.go", abs); f != "" || n != "" {
		t.Errorf("create reader at root = %q, %q — want silent", f, n)
	}
}
