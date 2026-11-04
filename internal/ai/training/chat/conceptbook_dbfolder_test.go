package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The database folder-guidance fix (2026-11-03): the
// ambiguous-piece guidance's own suggested sentences used to
// dead-end in the webserver claim's refusals whenever the
// target folder had no Go host file — John's live jim/db
// session: jim/db held only database.db, the guidance said
// "add a database connection function named Connect to
// jim/db", and the answer was "jim/db/ has no main program
// to add a server to — the working move is \"create a file
// jim/db/main.go with a webserver\"". These tests pin the
// whole folder matrix for the database-connect declaration,
// the statement family's own refusals with their spelled
// follow-ups fed back until they land, and the non-theft
// pins in both directions.

// dbFolderRoot is the live-session module with jim/db in
// one of the matrix states (files written verbatim; a nil
// entry creates an empty file).
func dbFolderRoot(t *testing.T, files map[string]string) string {
	t.Helper()
	root := liveSessionRoot(t)
	for rel, src := range files {
		if src == "" {
			if err := os.MkdirAll(filepath.Join(root, filepath.Dir(rel)), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, rel), nil, 0o644); err != nil {
				t.Fatal(err)
			}
			continue
		}
		writeFile(t, root, rel, src)
	}
	return root
}

const dbDeclLine = "add a database connection function named Connect to jim/db"

// Cells that already worked keep their answers: a folder
// with a main.go, or with a single package-main file, hosts
// the declaration in that file.
func TestDBFolderDeclHostCellsUnchanged(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		host  string
	}{
		{"main.go", map[string]string{"jim/db/main.go": "package main\n\nfunc main() {}\n"}, "jim/db/main.go"},
		{"one main-package file", map[string]string{"jim/db/app.go": "package main\n\nfunc main() {}\n"}, "jim/db/app.go"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := dbFolderRoot(t, c.files)
			withChatSession(t, root, true)
			profile := &UserProfile{}
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(dbDeclLine, yes, profile) })
			if !handled {
				t.Fatalf("folder-host declaration fell through:\n%s", out)
			}
			if !strings.Contains(out, "built the project and checked") {
				t.Errorf("spine build/verify not reported:\n%s", out)
			}
			got := readBack(t, root, c.host)
			if !strings.Contains(got, "func Connect() (*sql.DB, error)") {
				t.Errorf("Connect did not land in the host file %s:\n%s", c.host, got)
			}
			if _, err := os.Stat(filepath.Join(root, "jim", "db", "connect.go")); !os.IsNotExist(err) {
				t.Errorf("a host cell grew a separate connect.go anyway")
			}
		})
	}
}

// The fixed cells: a folder with no package-main file —
// only library files, or no Go files at all — takes the
// declaration as its own file in the folder, named for the
// declaration, under the folder's package rules.
func TestDBFolderDeclCreatesOwnFile(t *testing.T) {
	cases := []struct {
		name    string
		files   map[string]string
		wantPkg string
	}{
		{"empty of Go files", map[string]string{"jim/db/database.db": ""}, "package db"},
		{"library files only", map[string]string{"jim/db/store.go": "package store\n\nfunc Open() {}\n"}, "package store"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			root := dbFolderRoot(t, c.files)
			withChatSession(t, root, true)
			profile := &UserProfile{}
			var handled bool
			out := captureStdout(t, func() { handled = claimChain(dbDeclLine, yes, profile) })
			if !handled {
				t.Fatalf("no-host declaration fell through:\n%s", out)
			}
			if strings.Contains(out, "webserver") || strings.Contains(out, "add a server") {
				t.Errorf("the webserver remedy leaked into a database answer:\n%s", out)
			}
			if !strings.Contains(out, "built the project and checked") {
				t.Errorf("spine build/verify not reported:\n%s", out)
			}
			got := readBack(t, root, "jim/db/connect.go")
			for _, want := range []string{c.wantPkg, "func Connect() (*sql.DB, error)"} {
				if !strings.Contains(got, want) {
					t.Errorf("jim/db/connect.go missing %q:\n%s", want, got)
				}
			}
		})
	}
}

// A missing folder refuses with the create family's first
// step — and the spelled step, followed by the same ask,
// lands.
func TestDBFolderDeclMissingFolderThenLands(t *testing.T) {
	root := dbFolderRoot(t, nil)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(dbDeclLine, yes, profile) })
	if !handled {
		t.Fatalf("missing-folder declaration fell through:\n%s", out)
	}
	want := "Not adding that: there's no folder jim/db in this project — say \"create a folder jim/db\" first, then ask again."
	if !strings.Contains(out, want) {
		t.Errorf("missing-folder refusal = the answer below, want it to contain %q:\n%s", want, out)
	}
	if strings.Contains(out, "webserver") {
		t.Errorf("the webserver remedy leaked into the refusal:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("create a folder jim/db", yes, profile) })
	if !handled {
		t.Fatalf("the spelled create step fell through:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain(dbDeclLine, yes, profile) })
	if !handled {
		t.Fatalf("the re-ask fell through:\n%s", out)
	}
	got := readBack(t, root, "jim/db/connect.go")
	if !strings.HasPrefix(got, "package db\n") || !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("the re-ask did not create jim/db/connect.go with Connect:\n%s", got)
	}
}

// Several package-main files is a genuine choice, refused
// in declaration words — the example sentence is the ask's
// own shape, not the webserver claim's.
func TestDBFolderDeclMultiMainRefusal(t *testing.T) {
	root := dbFolderRoot(t, map[string]string{
		"jim/db/a.go": "package main\n\nfunc A() {}\n",
		"jim/db/b.go": "package main\n\nfunc B() {}\n",
	})
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(dbDeclLine, yes, profile) })
	if !handled {
		t.Fatalf("multi-main declaration fell through:\n%s", out)
	}
	want := "Not adding that: jim/db/ has more than one main-package file (a.go, b.go) — say which one, like \"add a database connect named Connect to jim/db/a.go\"."
	if !strings.Contains(out, want) {
		t.Errorf("multi-main refusal = the answer below, want it to contain %q:\n%s", want, out)
	}
	if strings.Contains(out, "webserver") {
		t.Errorf("the webserver claim's example leaked into the refusal:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "db", "connect.go")); !os.IsNotExist(err) {
		t.Errorf("the refusal created connect.go anyway")
	}
}

// The guidance's second sentence in John's state: the
// statement aimed at a main.go that doesn't exist yet gets
// the statement family's own two-step — and feeding the
// spelled sentences back lands the piece.
func TestDBFolderStatementMissingFileTwoStepLands(t *testing.T) {
	root := dbFolderRoot(t, map[string]string{"jim/db/database.db": ""})
	withChatSession(t, root, true)
	profile := &UserProfile{}
	const line = "add a database in main to jim/db/main.go"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("missing-file statement fell through:\n%s", out)
	}
	want := "Not adding that: there's no jim/db/main.go in this project — a database lives inside a function in a file that exists. Say \"create a file jim/db/main.go\" first, then \"add a database in main to jim/db/main.go\" and it lands."
	if !strings.Contains(out, want) {
		t.Errorf("missing-file statement refusal = the answer below, want it to contain %q:\n%s", want, out)
	}
	if strings.Contains(out, "webserver") {
		t.Errorf("the webserver remedy leaked into a database answer:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(root, "jim", "db", "main.go")); !os.IsNotExist(err) {
		t.Fatalf("the refusal created the file anyway")
	}

	out = captureStdout(t, func() { handled = claimChain("create a file jim/db/main.go", yes, profile) })
	if !handled {
		t.Fatalf("the spelled create step fell through:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("the re-ask fell through:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported on the re-ask:\n%s", out)
	}
	got := readBack(t, root, "jim/db/main.go")
	if !strings.Contains(got, "sql.Open") {
		t.Errorf("the database statement did not land in the created main.go:\n%s", got)
	}
}

// The statement aimed at the folder itself, with no host:
// the same creation step, then the folder form of the ask
// lands.
func TestDBFolderStatementFolderNoHostTwoStepLands(t *testing.T) {
	root := dbFolderRoot(t, map[string]string{"jim/db/database.db": ""})
	withChatSession(t, root, true)
	profile := &UserProfile{}
	const line = "add a database in main to jim/db"
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("folder statement fell through:\n%s", out)
	}
	want := "Not adding that: jim/db/ has no main program to put the database in — the working move is \"create a file jim/db/main.go\", then \"add a database in main to jim/db\" and it lands."
	if !strings.Contains(out, want) {
		t.Errorf("folder statement refusal = the answer below, want it to contain %q:\n%s", want, out)
	}
	out = captureStdout(t, func() { handled = claimChain("create a file jim/db/main.go", yes, profile) })
	if !handled {
		t.Fatalf("the spelled create step fell through:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain(line, yes, profile) })
	if !handled {
		t.Fatalf("the re-ask fell through:\n%s", out)
	}
	got := readBack(t, root, "jim/db/main.go")
	if !strings.Contains(got, "sql.Open") {
		t.Errorf("the database statement did not land through the folder form:\n%s", got)
	}
}

// A library folder can't take the statement and can't grow
// a program file beside its files: the refusal says so in
// statement words and spells no dead-end step.
func TestDBFolderStatementLibraryFolderRefusal(t *testing.T) {
	root := dbFolderRoot(t, map[string]string{"jim/db/store.go": "package store\n\nfunc Open() {}\n"})
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a database in main to jim/db", yes, profile) })
	if !handled {
		t.Fatalf("library-folder statement fell through:\n%s", out)
	}
	want := "Not adding that: jim/db/ has no main program to put the database in — its Go files are package store, and a database lives inside a function in a program (package main) file. Point the ask at a program file instead."
	if !strings.Contains(out, want) {
		t.Errorf("library-folder refusal = the answer below, want it to contain %q:\n%s", want, out)
	}
	if strings.Contains(out, "webserver") {
		t.Errorf("the webserver remedy leaked into the refusal:\n%s", out)
	}
}

// John's journey, end to end through the chain: the
// guidance fires in the jim/db state, and BOTH of its
// suggested sentences now reach a landing (the declaration
// directly; this test follows the declaration — the
// statement journey is pinned above).
func TestDBFolderGuidanceJourneyLands(t *testing.T) {
	root := dbFolderRoot(t, map[string]string{"jim/db/database.db": ""})
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add database to jim/db", yes, profile) })
	if !handled {
		t.Fatalf("the bare database ask fell through:\n%s", out)
	}
	for _, want := range []string{
		"\"add a database connection function named Connect to jim/db\"",
		"\"add a database in main to jim/db/main.go\"",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("guidance missing its sentence %q:\n%s", want, out)
		}
	}
	out = captureStdout(t, func() { handled = claimChain(dbDeclLine, yes, profile) })
	if !handled {
		t.Fatalf("the guidance's first sentence fell through:\n%s", out)
	}
	got := readBack(t, root, "jim/db/connect.go")
	if !strings.Contains(got, "func Connect() (*sql.DB, error)") {
		t.Errorf("the guidance's first sentence did not land Connect in jim/db:\n%s", got)
	}
}

// Non-theft, both directions: the webserver claim keeps
// its own answers for its own sentences in the very state
// that used to lend them out.
func TestDBFolderServerLinesKeepServerWords(t *testing.T) {
	root := dbFolderRoot(t, map[string]string{"jim/db/database.db": ""})
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a webserver to jim/db", yes, profile) })
	if !handled {
		t.Fatalf("the webserver ask fell through:\n%s", out)
	}
	want := "Not adding that: jim/db/ has no main program to add a server to — the working move is \"create a file jim/db/main.go with a webserver\"."
	if !strings.Contains(out, want) {
		t.Errorf("webserver refusal changed shape, want it to contain %q:\n%s", want, out)
	}
}
