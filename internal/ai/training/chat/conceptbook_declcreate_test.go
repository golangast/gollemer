package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The declaration-create build (2026-11-01): a concept
// DECLARATION aimed at a named file that doesn't exist yet,
// in a folder that does, creates the file as part of the
// plan — where it used to refuse with the webserver claim's
// remedy ("say \"create a file api/profile.go with a
// webserver\""), the wrong piece and, in a package-main
// folder, a second func main waiting to happen. These tests
// pin the chat half end to end through the claim chain: the
// workout's failing shapes (paces steps 5–7), the pins that
// statements and folder targets keep today's answers, and
// the gorqlite gate's now-keepable promise.

// declCreateRoot is the fixture: the live-session module
// plus an api/ folder holding a package-main program file —
// the workout project's shape from step 1 on.
func declCreateRoot(t *testing.T) string {
	t.Helper()
	root := liveSessionRoot(t)
	api := filepath.Join(root, "api")
	if err := os.MkdirAll(api, 0o755); err != nil {
		t.Fatal(err)
	}
	writeFile(t, api, "main.go", `package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "hi")
	})
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("stopped:", err)
	}
}
`)
	return root
}

// Step 6's first failing line: the struct lands in a created
// api/profile.go, package main read from the folder's files.
func TestDeclCreateStructToNewFile(t *testing.T) {
	root := declCreateRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a struct named Profile to api/profile.go", yes, profile)
	})
	if !handled {
		t.Fatalf("struct-to-new-file fell through the chain:\n%s", out)
	}
	if strings.Contains(out, "with a webserver") {
		t.Errorf("the webserver remedy leaked into a declaration answer:\n%s", out)
	}
	if !strings.Contains(out, "Create ") || !strings.Contains(out, "a new file in the api folder (package main)") ||
		!strings.Contains(out, "with the struct Profile, from the concept book") {
		t.Errorf("plan does not show the file being created:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	got := readBack(t, root, "api/profile.go")
	for _, want := range []string{"package main", "type Profile struct"} {
		if !strings.Contains(got, want) {
			t.Errorf("api/profile.go missing %q:\n%s", want, got)
		}
	}
}

// Step 6's other failing declaration: a plain function into
// a created file beside the program.
func TestDeclCreateFunctionToNewFile(t *testing.T) {
	root := declCreateRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a function named Validate to api/validate.go", yes, profile)
	})
	if !handled {
		t.Fatalf("function-to-new-file fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	got := readBack(t, root, "api/validate.go")
	if !strings.Contains(got, "func Validate() {") {
		t.Errorf("api/validate.go missing the declaration:\n%s", got)
	}
}

// The named-connection sibling (c13a650) composed with
// creation: the ask resolves to the database-connect
// declaration, and the declaration creates its named file.
func TestDeclCreateNamedConnectionNewFile(t *testing.T) {
	root := declCreateRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a database connection named Connet to api/db.go", yes, profile)
	})
	if !handled {
		t.Fatalf("named connection to a new file fell through:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	got := readBack(t, root, "api/db.go")
	for _, want := range []string{"package main", "func Connet() (*sql.DB, error)"} {
		if !strings.Contains(got, want) {
			t.Errorf("api/db.go missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "func main(") {
		t.Errorf("api/db.go grew a fabricated main:\n%s", got)
	}
}

// A missing folder names the real first step — never the
// webserver remedy — and offers no gate.
func TestDeclCreateMissingFolderRefusal(t *testing.T) {
	root := declCreateRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a struct named Profile to nowhere/profile.go", yes, profile)
	})
	if !handled {
		t.Fatalf("missing-folder declaration fell through the chain:\n%s", out)
	}
	want := "Not adding that: there's no folder nowhere in this project — say \"create a folder nowhere\" first, then ask again."
	if !strings.Contains(out, want) {
		t.Errorf("missing-folder refusal = the answer below, want it to contain %q:\n%s", want, out)
	}
	if strings.Contains(out, "webserver") {
		t.Errorf("the webserver remedy leaked into the refusal:\n%s", out)
	}
	if strings.Contains(out, "Apply this edit?") {
		t.Errorf("a refusal offered a gate:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "nowhere")); !os.IsNotExist(err) {
		t.Errorf("the refusal created the folder anyway")
	}
}

// Statements keep refusing — creating files is a
// declaration's licence only — but since the
// folder-guidance fix (2026-11-03) the refusal is rendered
// in the statement family's own words: the spelled
// creation step, then the same ask again. (It used to serve
// the webserver claim's remedy verbatim; that wording now
// belongs to webserver asks only.)
func TestDeclCreateStatementPinMissingFile(t *testing.T) {
	root := declCreateRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a print line in main to api/new.go", yes, profile)
	})
	if !handled {
		t.Fatalf("statement to a missing file fell through the chain:\n%s", out)
	}
	want := "Not adding that: there's no api/new.go in this project — a print line lives inside a function in a file that exists. Say \"create a file api/new.go\" first, then \"add a print line in main to api/new.go\" and it lands."
	if !strings.Contains(out, want) {
		t.Errorf("statement refusal changed shape, want it to contain %q:\n%s", want, out)
	}
	if strings.Contains(out, "webserver") {
		t.Errorf("the webserver remedy leaked into a statement answer:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "api", "new.go")); !os.IsNotExist(err) {
		t.Errorf("the statement refusal created the file anyway")
	}
}

// Step 5's broken promise, healed: the gate holds an ask
// for store/db.go (store/ exists, the file doesn't), the
// first answer says the file will be created when the plan
// lands, and after the install the re-offer creates it.
func TestDeclCreateGorqliteReofferCreatesFile(t *testing.T) {
	resetPendingGorqlite(t)
	root := declCreateRoot(t)
	if err := os.MkdirAll(filepath.Join(root, "store"), 0o755); err != nil {
		t.Fatal(err)
	}
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a gorqlite connection named Rq to store/db.go", yes, profile)
	})
	if !handled {
		t.Fatalf("gorqlite ask fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "go get github.com/rqlite/gorqlite") {
		t.Errorf("gate guidance lacks the go get line:\n%s", out)
	}
	if !strings.Contains(out, "store/db.go doesn't exist yet — it will be created as part of the plan when it lands.") {
		t.Errorf("gate's first answer doesn't promise the file's creation:\n%s", out)
	}
	if strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("a plan was offered despite the missing dependency:\n%s", out)
	}
	if pendingGorqlite == nil {
		t.Fatal("the gated ask was not held in the session")
	}

	seedGorqliteStub(t, root)
	out = captureStdout(t, func() { handled = claimChain("installed", yes, profile) })
	if !handled {
		t.Fatalf("the installed follow-up fell through the chain:\n%s", out)
	}
	if !strings.Contains(out, "Here's the edit I assembled") {
		t.Errorf("the original plan was not re-offered:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
	got := readBack(t, root, "store/db.go")
	for _, want := range []string{"package store", "func Rq() (*gorqlite.Connection, error)", `"github.com/rqlite/gorqlite"`} {
		if !strings.Contains(got, want) {
			t.Errorf("store/db.go missing %q:\n%s", want, got)
		}
	}
}

// The gate's creation sentence is scoped: an existing target
// file doesn't earn it, a statement row doesn't earn it, and
// a missing folder doesn't earn it (there the re-offer's
// honest answer is the create-a-folder step).
func TestDeclCreateGateNoteScoped(t *testing.T) {
	// Existing file: gate fires, no creation sentence.
	resetPendingGorqlite(t)
	root := declCreateRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	out := captureStdout(t, func() {
		claimChain("add a gorqlite connection named Rq to api/main.go", yes, profile)
	})
	if !strings.Contains(out, "go get github.com/rqlite/gorqlite") {
		t.Fatalf("gate did not fire for the existing-file ask:\n%s", out)
	}
	if strings.Contains(out, "will be created as part of the plan") {
		t.Errorf("existing-file gate answer promised a creation:\n%s", out)
	}

	// Statement row to a missing file: gate fires, no
	// creation sentence — statements never create files.
	resetPendingGorqlite(t)
	out = captureStdout(t, func() {
		claimChain("add a gorqlite query in main to api/new.go", yes, profile)
	})
	if !strings.Contains(out, "go get github.com/rqlite/gorqlite") {
		t.Fatalf("gate did not fire for the statement ask:\n%s", out)
	}
	if strings.Contains(out, "will be created as part of the plan") {
		t.Errorf("statement gate answer promised a creation:\n%s", out)
	}

	// Missing folder: gate fires, no creation sentence; the
	// re-offer after the install refuses with the folder
	// step, in the open.
	resetPendingGorqlite(t)
	out = captureStdout(t, func() {
		claimChain("add a gorqlite connection named Rq to nowhere/db.go", yes, profile)
	})
	if !strings.Contains(out, "go get github.com/rqlite/gorqlite") {
		t.Fatalf("gate did not fire for the missing-folder ask:\n%s", out)
	}
	if strings.Contains(out, "will be created as part of the plan") {
		t.Errorf("missing-folder gate answer promised a creation:\n%s", out)
	}
	seedGorqliteStub(t, root)
	out = captureStdout(t, func() { claimChain("installed", yes, profile) })
	want := "there's no folder nowhere in this project — say \"create a folder nowhere\" first, then ask again"
	if !strings.Contains(out, want) {
		t.Errorf("missing-folder re-offer answer doesn't name the folder step %q:\n%s", want, out)
	}
	if _, err := os.Stat(filepath.Join(root, "nowhere")); !os.IsNotExist(err) {
		t.Errorf("the re-offer created the folder anyway")
	}
}

// Step 7's compound, end to end: the first piece is a
// declaration to a new file (it creates api/welcome.go),
// and the statement piece then lands inside the handler the
// first piece just wrote.
func TestDeclCreateCompoundFirstPieceCreates(t *testing.T) {
	withPhrasingModel(t)
	root := declCreateRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("make a form handler named Welcome in api/welcome.go and add a switch in it", yes, profile)
	})
	if !handled {
		t.Fatalf("compound fell through the chain:\n%s", out)
	}
	got := readBack(t, root, "api/welcome.go")
	for _, want := range []string{"package main", "func Welcome(", "switch"} {
		if !strings.Contains(got, want) {
			t.Errorf("api/welcome.go missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(out, "there's no api/welcome.go in this project") {
		t.Errorf("the compound still hit the missing-file refusal:\n%s", out)
	}
}
