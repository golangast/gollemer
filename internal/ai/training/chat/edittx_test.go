package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// countTxLeftovers walks dir for transaction debris: backup
// files and the manifest.
func countTxLeftovers(t *testing.T, dir string) (backups int, manifest bool) {
	t.Helper()
	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if strings.HasSuffix(d.Name(), ".gollemer-backup") {
			backups++
		}
		if d.Name() == txManifestName {
			manifest = true
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return backups, manifest
}

// The transaction machinery itself, as units (the successor of
// the delegate-era backup round-trip test, on the shared
// helper): the manifest is written and armed, the backup holds
// the original, bytes and mode round-trip through a forced
// rollback, a created file is removed by it, and both a
// completed rollback and a commit leave a clean slate.
func TestEditTxRoundTrip(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "app/main.go", "package main\n")
	abs := filepath.Join(dir, "app", "main.go")
	if err := os.Chmod(abs, 0o751); err != nil {
		t.Fatal(err)
	}
	files := []analyze.EditFile{
		{Rel: "app/main.go", After: []byte("package main\n\n// changed\n")},
		{Rel: "app/main_test.go", After: []byte("package main\n")},
	}
	tx, err := beginEditTx(dir, files)
	if err != nil {
		t.Fatalf("beginEditTx: %v", err)
	}
	m, _, present, merr := readTxManifest(dir)
	if merr != nil || !present {
		t.Fatalf("manifest after begin: present=%v err=%v", present, merr)
	}
	if m.State != txStateArmed || len(m.Files) != 2 {
		t.Fatalf("manifest after begin = %+v, want state %q with both files", m, txStateArmed)
	}
	if got := readBack(t, dir, "app/.main.go.gollemer-backup"); got != "package main\n" {
		t.Errorf("backup holds %q, want the original bytes", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "app", ".main.go.gollemer-backup.mode")); !os.IsNotExist(err) {
		t.Error("a .mode sidecar was written — the manifest supersedes sidecars")
	}
	// A second transaction over the live one refuses: the
	// manifest is an earlier edit's recovery record.
	if _, err := beginEditTx(dir, files[:1]); err == nil || !strings.Contains(err.Error(), txManifestName) {
		t.Errorf("second beginEditTx = %v, want the interrupted-transaction refusal", err)
	}
	if _, werr, _ := tx.writeAll(files); werr != nil {
		t.Fatalf("writeAll: %v", werr)
	}
	if got := readBack(t, dir, "app/main.go"); !strings.Contains(got, "changed") {
		t.Fatalf("apply didn't write:\n%s", got)
	}
	if unrestored := tx.rollback(); len(unrestored) != 0 {
		t.Fatalf("rollback left unrestored: %v", unrestored)
	}
	if got := readBack(t, dir, "app/main.go"); got != "package main\n" {
		t.Errorf("restored bytes = %q, want the original", got)
	}
	if fi, err := os.Stat(abs); err != nil || fi.Mode().Perm() != 0o751 {
		t.Errorf("restored mode = %v (err %v), want 0751", fi.Mode().Perm(), err)
	}
	if _, err := os.Stat(filepath.Join(dir, "app", "main_test.go")); !os.IsNotExist(err) {
		t.Error("rollback left the created file behind")
	}
	if n, man := countTxLeftovers(t, dir); n != 0 || man {
		t.Errorf("after rollback: %d backups, manifest=%v — want a clean slate", n, man)
	}
	// A committed transaction leaves a clean slate too, and
	// keeps the edit.
	tx2, err := beginEditTx(dir, files)
	if err != nil {
		t.Fatalf("beginEditTx after rollback: %v", err)
	}
	if _, werr, _ := tx2.writeAll(files); werr != nil {
		t.Fatalf("writeAll (commit round): %v", werr)
	}
	tx2.commit()
	if n, man := countTxLeftovers(t, dir); n != 0 || man {
		t.Errorf("after commit: %d backups, manifest=%v — want a clean slate", n, man)
	}
	if got := readBack(t, dir, "app/main.go"); !strings.Contains(got, "changed") {
		t.Error("commit discarded the applied edit")
	}
}

// Ordering, proven by a forced failure: the manifest is armed
// and BOTH backups are on disk before the first target write; a
// second file whose write is forced to fail leaves the first
// file restored byte- and mode-identical from its backup, and
// the unrestorable file is named, keeps its backup, and stays
// in a trimmed manifest a later rollback can finish.
func TestEditTxOrderingForcedSecondFailure(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "a.go", "aaa original\n")
	writeFile(t, dir, "b.go", "bbb original\n")
	aAbs := filepath.Join(dir, "a.go")
	if err := os.Chmod(aAbs, 0o755); err != nil {
		t.Fatal(err)
	}
	files := []analyze.EditFile{
		{Rel: "a.go", After: []byte("aaa edited\n")},
		{Rel: "b.go", After: []byte("bbb edited\n")},
	}
	tx, err := beginEditTx(dir, files)
	if err != nil {
		t.Fatalf("beginEditTx: %v", err)
	}
	// Before any target write: manifest armed, both backups
	// present with the originals, targets untouched.
	m, _, present, _ := readTxManifest(dir)
	if !present || m.State != txStateArmed {
		t.Fatalf("manifest before writes = %+v (present %v), want armed", m, present)
	}
	if got := readBack(t, dir, ".a.go.gollemer-backup"); got != "aaa original\n" {
		t.Errorf("a's backup before writes = %q", got)
	}
	if got := readBack(t, dir, ".b.go.gollemer-backup"); got != "bbb original\n" {
		t.Errorf("b's backup before writes = %q", got)
	}
	if got := readBack(t, dir, "a.go"); got != "aaa original\n" {
		t.Fatalf("a target changed before its write:\n%s", got)
	}
	if err := tx.write(files[0]); err != nil {
		t.Fatalf("write a: %v", err)
	}
	// Sabotage b's target: a directory now sits where the file
	// was, so both the write and the restore must fail.
	bAbs := filepath.Join(dir, "b.go")
	if err := os.Remove(bAbs); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(bAbs, 0o755); err != nil {
		t.Fatal(err)
	}
	if _, werr, failedRel := tx.writeAll(files[1:]); werr == nil || failedRel != "b.go" {
		t.Fatalf("sabotaged write = (%v, %q), want a failure on b.go", werr, failedRel)
	}
	// The manifest is still armed at the moment of failure.
	if m, _, _, _ := readTxManifest(dir); m == nil || m.State != txStateArmed {
		t.Fatalf("manifest at failure = %+v, want armed", m)
	}
	unrestored := tx.rollback()
	if len(unrestored) != 1 || unrestored[0] != "b.go" {
		t.Fatalf("unrestored = %v, want [b.go]", unrestored)
	}
	if got := readBack(t, dir, "a.go"); got != "aaa original\n" {
		t.Errorf("a after rollback = %q, want the original bytes", got)
	}
	if fi, err := os.Stat(aAbs); err != nil || fi.Mode().Perm() != 0o755 {
		t.Errorf("a mode after rollback = %v (err %v), want 0755", fi.Mode().Perm(), err)
	}
	if _, err := os.Stat(filepath.Join(dir, ".a.go.gollemer-backup")); !os.IsNotExist(err) {
		t.Error("a's backup survived its successful restore")
	}
	if _, err := os.Stat(filepath.Join(dir, ".b.go.gollemer-backup")); err != nil {
		t.Error("b's backup was removed even though b could not be restored")
	}
	m, _, present, _ = readTxManifest(dir)
	if !present || len(m.Files) != 1 || m.Files[0].Path != "b.go" {
		t.Fatalf("trimmed manifest = %+v (present %v), want just b.go", m, present)
	}
	// Unsabotage and roll back again: now the set completes
	// and the slate is clean.
	if err := os.Remove(bAbs); err != nil {
		t.Fatal(err)
	}
	if unrestored := tx.rollback(); len(unrestored) != 0 {
		t.Fatalf("second rollback left unrestored: %v", unrestored)
	}
	if got := readBack(t, dir, "b.go"); got != "bbb original\n" {
		t.Errorf("b after second rollback = %q, want the original", got)
	}
	if n, man := countTxLeftovers(t, dir); n != 0 || man {
		t.Errorf("after the completed rollback: %d backups, manifest=%v — want a clean slate", n, man)
	}
}

// Begin's two refusals: a leftover backup beside a target (the
// delegate's shipped rule) and a live transaction manifest.
func TestEditTxBeginRefusals(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "x.go", "package x\n")
	writeFile(t, dir, ".x.go.gollemer-backup", "someone's original")
	files := []analyze.EditFile{{Rel: "x.go", After: []byte("package x\n\n// edited\n")}}
	if _, err := beginEditTx(dir, files); err == nil || !strings.Contains(err.Error(), "already a leftover backup") {
		t.Errorf("beginEditTx over a leftover backup = %v, want the leftover refusal", err)
	}
	if got := readBack(t, dir, ".x.go.gollemer-backup"); got != "someone's original" {
		t.Error("the refusal touched the leftover backup")
	}
	if _, _, present, _ := readTxManifest(dir); present {
		t.Error("the refusal left a manifest behind")
	}
}

// (b)+(f) through the assembled-edits spine: a build-failing
// edit restores its target's exact mode from the on-disk
// backup, and both that rollback and a successful edit leave
// neither backups nor a manifest behind.
func TestEditsSpineTxModeAndCleanSlate(t *testing.T) {
	dir := editsChatFixture(t)
	withEditsProject(t, dir)
	shopAbs := filepath.Join(dir, "shop", "shop.go")
	if err := os.Chmod(shopAbs, 0o755); err != nil {
		t.Fatal(err)
	}
	before := readBack(t, dir, "shop/shop.go")
	if !runAssembledEdit("add a other.ModeMissing case to ModeName", yes) {
		t.Fatal("edit sentence not claimed")
	}
	if after := readBack(t, dir, "shop/shop.go"); after != before {
		t.Fatalf("build failure must restore the original bytes:\n%s", after)
	}
	if fi, err := os.Stat(shopAbs); err != nil || fi.Mode().Perm() != 0o755 {
		t.Errorf("mode after spine rollback = %v (err %v), want 0755 restored from the backup", fi.Mode().Perm(), err)
	}
	if n, man := countTxLeftovers(t, dir); n != 0 || man {
		t.Errorf("after spine rollback: %d backups, manifest=%v — want a clean slate", n, man)
	}

	// Success: the mode survives the write itself, and the
	// commit leaves no trace.
	dir2 := editsChatFixture(t)
	withEditsProject(t, dir2)
	shop2 := filepath.Join(dir2, "shop", "shop.go")
	if err := os.Chmod(shop2, 0o751); err != nil {
		t.Fatal(err)
	}
	if !runAssembledEdit("add a plum case to Describe", yes) {
		t.Fatal("edit sentence not claimed")
	}
	if after := readBack(t, dir2, "shop/shop.go"); !strings.Contains(after, `case "plum":`) {
		t.Fatalf("successful edit missing its case:\n%s", after)
	}
	if fi, err := os.Stat(shop2); err != nil || fi.Mode().Perm() != 0o751 {
		t.Errorf("mode after spine success = %v (err %v), want 0751 preserved", fi.Mode().Perm(), err)
	}
	if n, man := countTxLeftovers(t, dir2); n != 0 || man {
		t.Errorf("after spine success: %d backups, manifest=%v — want a clean slate", n, man)
	}
}

// (d) rename: a build-failing rename rolls both files of the
// set back from their on-disk backups with no trace left, and
// a leftover backup beside a target refuses the rename before
// anything is written.
func TestRenameTxRollbackAndRefusal(t *testing.T) {
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/readrename2\n\ngo 1.21\n")
	write("lib/lib.go", "package lib\n\ntype Source struct{}\n\nfunc (Source) Read(p []byte) (int, error) { return 0, nil }\n")
	write("app/app.go", "package app\n\nimport (\n\t\"io\"\n\n\t\"example.com/readrename2/lib\"\n)\n\nvar _ io.Reader = lib.Source{}\n")
	withEditsProject(t, dir)
	libBefore := readBack(t, dir, "lib/lib.go")
	appBefore := readBack(t, dir, "app/app.go")
	if !runRenameEdit("rename Read to ReadAll", yes) {
		t.Fatal("rename sentence not claimed")
	}
	if got := readBack(t, dir, "lib/lib.go"); got != libBefore {
		t.Errorf("build-failing rename was not rolled back:\n%s", got)
	}
	if got := readBack(t, dir, "app/app.go"); got != appBefore {
		t.Errorf("app file changed by a rolled-back rename:\n%s", got)
	}
	if n, man := countTxLeftovers(t, dir); n != 0 || man {
		t.Errorf("after rename rollback: %d backups, manifest=%v — want a clean slate", n, man)
	}

	// A leftover backup beside the target stops the rename
	// cold, in transaction terms.
	dir2 := editsChatFixture(t)
	withEditsProject(t, dir2)
	writeFile(t, dir2, "shop/.shop.go.gollemer-backup", "someone's original")
	shopBefore := readBack(t, dir2, "shop/shop.go")
	out := captureStdout(t, func() {
		runRenameEdit("rename Describe to Explain", yes)
	})
	if !strings.Contains(out, "already a leftover backup") {
		t.Errorf("leftover backup didn't stop the rename:\n%s", out)
	}
	if got := readBack(t, dir2, "shop/shop.go"); got != shopBefore {
		t.Error("rename proceeded despite the leftover backup")
	}
	if got := readBack(t, dir2, "shop/.shop.go.gollemer-backup"); got != "someone's original" {
		t.Error("the leftover backup itself was touched")
	}
}

// (d) error→fix: a leftover backup beside the target refuses
// the fix in transaction terms, and a fix that breaks the
// build rolls back from the on-disk backup with no trace.
func TestErrorFixTxRefusalAndRollback(t *testing.T) {
	dir := errorfixChatFixture(t)
	withEditsProject(t, dir)
	writeFile(t, dir, ".broken_import.go.gollemer-backup", "someone's original")
	before := readBack(t, dir, "broken_import.go")
	out := captureStdout(t, func() {
		if !runErrorFix(`./broken_import.go:3:1: imported and not used: "os"`, yes) {
			t.Error("fix line fell through instead of being claimed")
		}
	})
	if !strings.Contains(out, "already a leftover backup") {
		t.Errorf("leftover backup didn't stop the fix:\n%s", out)
	}
	if got := readBack(t, dir, "broken_import.go"); got != before {
		t.Error("fix proceeded despite the leftover backup")
	}
	if got := readBack(t, dir, ".broken_import.go.gollemer-backup"); got != "someone's original" {
		t.Error("the leftover backup itself was touched")
	}

	// Rollback from disk, no trace left.
	dir2 := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir2, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/chaterr3\n\ngo 1.21\n")
	write("greet.go", "package main\n\nfunc Greet() string { return \"hi\" }\n\nfunc main() { println(Greet()) }\n")
	write("spell.go", "package main\n\nfunc Spell() string {\n\treturn Greeet(\"extra\")\n}\n")
	withEditsProject(t, dir2)
	spellBefore := readBack(t, dir2, "spell.go")
	out = captureStdout(t, func() {
		if !runErrorFix(`./spell.go:4:9: undefined: Greeet`, yes) {
			t.Error("spelling fix not offered (it should be — and then rolled back)")
		}
	})
	if !strings.Contains(out, "restored your files from the on-disk backups") {
		t.Errorf("rollback didn't speak in on-disk terms:\n%s", out)
	}
	if after := readBack(t, dir2, "spell.go"); after != spellBefore {
		t.Fatalf("a fix that broke the build was not rolled back:\n%s", after)
	}
	if n, man := countTxLeftovers(t, dir2); n != 0 || man {
		t.Errorf("after error-fix rollback: %d backups, manifest=%v — want a clean slate", n, man)
	}
}

// --- chat-start recovery -----------------------------------------------------

// seedArmedTx leaves dir looking like a chat died mid-edit: an
// armed manifest, an edited file whose original (bytes + mode)
// sits in its backup, and a created file still standing.
func seedArmedTx(t *testing.T, dir string) {
	t.Helper()
	writeFile(t, dir, "keep.go", "package seed\n\n// edited\n")
	if err := os.Chmod(filepath.Join(dir, "keep.go"), 0o644); err != nil {
		t.Fatal(err)
	}
	writeFile(t, dir, ".keep.go.gollemer-backup", "package seed\n\n// original\n")
	writeFile(t, dir, "new.go", "package seed\n\n// created by the edit\n")
	m := &txManifest{
		ID:    "tx-seed-1",
		State: txStateArmed,
		Files: []txFileRecord{
			{Path: "keep.go", Mode: "0755", Existed: true},
			{Path: "new.go", Existed: false},
		},
	}
	seedTxManifest(t, dir, m)
}

func seedTxManifest(t *testing.T, dir string, m *txManifest) {
	t.Helper()
	tx := &editTx{root: dir, manifestPath: filepath.Join(dir, txManifestName), manifest: m}
	if err := tx.saveManifest(); err != nil {
		t.Fatal(err)
	}
}

// TTY + the default answer: the originals come back (bytes and
// mode), the created file goes, and the slate is clean.
func TestStartupRecoveryArmedYes(t *testing.T) {
	dir := t.TempDir()
	seedArmedTx(t, dir)
	asked := false
	ask := func(string) (string, error) { asked = true; return "", nil } // empty = the default yes
	out := captureStdout(t, func() {
		runStartupRecovery(dir, true, ask)
	})
	if !asked {
		t.Error("a TTY start never offered the restore")
	}
	if !strings.Contains(out, "tx-seed-1") || !strings.Contains(out, "Restored") {
		t.Errorf("recovery didn't announce and confirm:\n%s", out)
	}
	if got := readBack(t, dir, "keep.go"); got != "package seed\n\n// original\n" {
		t.Errorf("keep.go = %q, want the backup's bytes", got)
	}
	if fi, err := os.Stat(filepath.Join(dir, "keep.go")); err != nil || fi.Mode().Perm() != 0o755 {
		t.Errorf("keep.go mode = %v (err %v), want the manifest's 0755", fi.Mode().Perm(), err)
	}
	if _, err := os.Stat(filepath.Join(dir, "new.go")); !os.IsNotExist(err) {
		t.Error("the created file survived the restore")
	}
	if n, man := countTxLeftovers(t, dir); n != 0 || man {
		t.Errorf("after recovery: %d backups, manifest=%v — want a clean slate", n, man)
	}
}

// TTY + no: everything stays exactly in place, and the manual
// restore commands are printed, never run.
func TestStartupRecoveryArmedNo(t *testing.T) {
	dir := t.TempDir()
	seedArmedTx(t, dir)
	ask := func(string) (string, error) { return "n", nil }
	out := captureStdout(t, func() {
		runStartupRecovery(dir, true, ask)
	})
	if !strings.Contains(out, "cp ") || !strings.Contains(out, "chmod 0755") {
		t.Errorf("the manual restore commands weren't printed:\n%s", out)
	}
	if got := readBack(t, dir, "keep.go"); !strings.Contains(got, "edited") {
		t.Errorf("a declined restore changed the file:\n%s", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "new.go")); err != nil {
		t.Error("a declined restore removed the created file")
	}
	if n, man := countTxLeftovers(t, dir); n != 1 || !man {
		t.Errorf("after a declined recovery: %d backups, manifest=%v — want both left in place", n, man)
	}
}

// Piped: never prompt — the announcement and the manual
// commands print, and the files stay as they lie.
func TestStartupRecoveryArmedPiped(t *testing.T) {
	dir := t.TempDir()
	seedArmedTx(t, dir)
	ask := func(string) (string, error) {
		t.Error("a piped start must never prompt")
		return "", nil
	}
	out := captureStdout(t, func() {
		runStartupRecovery(dir, false, ask)
	})
	if !strings.Contains(out, "tx-seed-1") || !strings.Contains(out, "cp ") {
		t.Errorf("piped recovery didn't announce with commands:\n%s", out)
	}
	if got := readBack(t, dir, "keep.go"); !strings.Contains(got, "edited") {
		t.Errorf("a piped start restored without being asked:\n%s", got)
	}
	if n, man := countTxLeftovers(t, dir); n != 1 || !man {
		t.Errorf("after a piped recovery: %d backups, manifest=%v — want both left in place", n, man)
	}
}

// A "begun" leftover never touched a file: on a TTY the offer
// clears the partial backups; piped, it prints the clear
// commands and leaves them.
func TestStartupRecoveryBegun(t *testing.T) {
	seed := func(t *testing.T) string {
		dir := t.TempDir()
		writeFile(t, dir, "x.go", "package seed\n")
		writeFile(t, dir, ".x.go.gollemer-backup", "package seed\n")
		seedTxManifest(t, dir, &txManifest{
			ID:    "tx-seed-2",
			State: txStateBegun,
			Files: []txFileRecord{{Path: "x.go", Mode: "0644", Existed: true}},
		})
		return dir
	}
	dir := seed(t)
	out := captureStdout(t, func() {
		runStartupRecovery(dir, true, yes)
	})
	if !strings.Contains(out, "never started writing") || !strings.Contains(out, "Cleared") {
		t.Errorf("begun recovery didn't explain and clear:\n%s", out)
	}
	if n, man := countTxLeftovers(t, dir); n != 0 || man {
		t.Errorf("after clearing a begun transaction: %d backups, manifest=%v", n, man)
	}
	if got := readBack(t, dir, "x.go"); got != "package seed\n" {
		t.Errorf("clearing a begun transaction touched the file:\n%s", got)
	}

	dir2 := seed(t)
	out = captureStdout(t, func() {
		runStartupRecovery(dir2, false, yes)
	})
	if !strings.Contains(out, "rm -f") {
		t.Errorf("piped begun recovery didn't print the clear commands:\n%s", out)
	}
	if n, man := countTxLeftovers(t, dir2); n != 1 || !man {
		t.Errorf("piped begun recovery acted: %d backups, manifest=%v — want both left in place", n, man)
	}
}

// No manifest at all: legacy sidecar pairs (the shipped
// delegate build's leftovers) get the printed notice — command
// and mode from the sidecar — and nothing is executed.
func TestStartupRecoveryLegacyOrphans(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, ".doc.go.gollemer-backup", "orig doc")
	writeFile(t, dir, ".doc.go.gollemer-backup.mode", "0640\n")
	writeFile(t, dir, "app/.main.go.gollemer-backup", "orig main")
	ask := func(string) (string, error) {
		t.Error("legacy orphans must never prompt")
		return "", nil
	}
	out := captureStdout(t, func() {
		runStartupRecovery(dir, true, ask)
	})
	if !strings.Contains(out, "[leftover backup]") || !strings.Contains(out, "chmod 0640") {
		t.Errorf("legacy notices missing or modeless:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "doc.go")); !os.IsNotExist(err) {
		t.Error("a legacy notice executed its restore")
	}
	if got := readBack(t, dir, ".doc.go.gollemer-backup"); got != "orig doc" {
		t.Error("the legacy backup itself was touched")
	}
}

// A manifest that can't be parsed is announced and left
// exactly in place — never deleted, never guessed at.
func TestStartupRecoveryUnreadableManifest(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, txManifestName, "{not json")
	out := captureStdout(t, func() {
		runStartupRecovery(dir, true, yes)
	})
	if !strings.Contains(out, "couldn't read it") {
		t.Errorf("unreadable manifest wasn't announced as such:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, txManifestName)); err != nil {
		t.Error("the unreadable manifest was removed")
	}
}
