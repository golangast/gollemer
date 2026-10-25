package chat

// Edit transactions: the one write path every plan-based
// overwrite rides. Before the delegate shape (0710f98), the
// write spines restored from in-memory Before bytes only — a
// chat killed mid-apply left a partially written project with
// no on-disk recovery record. The delegate proved the stronger
// shape (a disk backup beside every touched file); this file
// generalizes it to every plan application — the assembled
// edits, the webserver, the rename, the error fix, and the
// delegate — and gives the transaction a manifest, so recovery
// never has to guess what a leftover backup means.
//
// The lifecycle:
//
//   - beginEditTx first writes the transaction manifest
//     (.gollemer-transaction.json in the project root: a
//     transaction id, a state, and for every file in the set
//     its project-relative path, its permission mode, and
//     whether it existed) with state "begun". Then it writes
//     every backup (.<name>.gollemer-backup beside the file,
//     0600 — the delegate's shipped naming; modes live in the
//     manifest, not in sidecars). Only when ALL backups are on
//     disk does the manifest flip to "armed" — and only an
//     armed transaction's caller may write targets. The
//     ordering is the point: a crash anywhere leaves either an
//     untouched project with partial backups ("begun") or a
//     fully recoverable one ("armed").
//   - A manifest already in the root when a transaction begins
//     refuses the edit outright — a previous crash's leftover
//     is never overwritten; chat start offers its recovery. A
//     backup already beside a target refuses the same way (the
//     delegate's shipped rule): it may be the only copy of an
//     earlier original.
//   - commit (verified success) deletes the backups and the
//     manifest: clean slate. rollback restores the whole set
//     from the backups (bytes + mode; files recorded absent
//     are deleted), then deletes the backups and the manifest.
//     A file whose restore fails keeps its backup, the
//     manifest is trimmed to just the unrestored files, and
//     the file is named in the return value — so backups on
//     disk can ONLY mean an interrupted run, which is what
//     makes the chat-start recovery below unambiguous.
//   - Chat start (startupEditRecovery) reads the manifest: an
//     "armed" leftover is announced with its transaction id
//     and file list; on a terminal the user is prompted
//     "Restore the original files? [Y/n]" (restore is the
//     default); declining — or any piped session — leaves
//     everything in place and prints the exact manual restore
//     commands, never executed. A "begun" leftover never
//     touched a file: the offer is to clear its partial
//     backups. Backups with NO manifest (the shipped delegate
//     build's legacy sidecar pairs, or a trimmed remainder)
//     keep the older treatment: a printed restore command,
//     never executed, never a prompt — no manifest, no certain
//     state.

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// txManifestName is the one transaction record per project,
// written in the project root.
const txManifestName = ".gollemer-transaction.json"

// Transaction states, in the only order they can occur.
const (
	txStateBegun = "begun" // manifest written, backups incomplete — no target was written
	txStateArmed = "armed" // all backups complete — targets may be (or have been) written
)

// txFileRecord is one file's line in the manifest: enough to
// restore it without any other record.
type txFileRecord struct {
	Path    string `json:"path"`           // project-relative, slash form
	Mode    string `json:"mode,omitempty"` // permission bits, "%04o" — existing files only
	Existed bool   `json:"existed"`        // false: the edit created it; restoring deletes it
}

// txManifest is the on-disk transaction record.
type txManifest struct {
	ID    string         `json:"id"`
	State string         `json:"state"`
	Files []txFileRecord `json:"files"`
}

// editTxEntry is one file's place in a live transaction.
type editTxEntry struct {
	rel         string
	abs         string
	bakAbs      string
	created     bool // the file did not exist when the transaction began
	mode        os.FileMode
	wroteBackup bool // this transaction wrote the backup (safe to remove on cleanup)
	restored    bool // a rollback already put this entry back; its backup is gone
}

// editTx is one plan application's write transaction.
type editTx struct {
	root         string
	manifestPath string
	manifest     *txManifest
	entries      []*editTxEntry
}

// backupPathsFor names a file's backup (the shipped delegate
// naming) — the mode sidecar path of the legacy build is the
// same string with ".mode" appended.
func backupPathsFor(abs string) (bak, mode string) {
	bak = filepath.Join(filepath.Dir(abs), "."+filepath.Base(abs)+".gollemer-backup")
	return bak, bak + ".mode"
}

// writeSyncFile writes a file and syncs it before closing, so a
// crash can't leave a record that names bytes it doesn't hold.
func writeSyncFile(path string, data []byte, perm os.FileMode) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

// saveManifest persists the transaction record (human-readable
// JSON: it is a recovery file, meant to be inspectable).
func (tx *editTx) saveManifest() error {
	data, err := json.MarshalIndent(tx.manifest, "", "  ")
	if err != nil {
		return err
	}
	return writeSyncFile(tx.manifestPath, append(data, '\n'), 0o644)
}

// readTxManifest loads the transaction record at root. present
// is false (with no error) when no transaction is recorded.
func readTxManifest(root string) (m *txManifest, path string, present bool, err error) {
	path = filepath.Join(root, txManifestName)
	data, rerr := os.ReadFile(path)
	if os.IsNotExist(rerr) {
		return nil, path, false, nil
	}
	if rerr != nil {
		return nil, path, true, rerr
	}
	m = &txManifest{}
	if jerr := json.Unmarshal(data, m); jerr != nil {
		return nil, path, true, jerr
	}
	return m, path, true, nil
}

// beginEditTx opens the transaction for a plan's file set: the
// manifest is written first, every existing file's backup is on
// disk before it returns, and the manifest is "armed" only
// after the last backup lands. On any failure the pieces this
// call itself wrote are removed again (a crash can't run that
// cleanup — which is exactly the case the manifest survives to
// describe) and the error says why.
func beginEditTx(root string, files []analyze.EditFile) (*editTx, error) {
	manifestPath := filepath.Join(root, txManifestName)
	if _, err := os.Stat(manifestPath); err == nil {
		return nil, fmt.Errorf("there's an interrupted edit transaction in this project (%s) — a chat session ended mid-edit, and its files' originals are still in the backups beside them. Start the chat here and you'll be offered the restore, or clear %s by hand first, then ask again", txManifestName, txManifestName)
	}
	// Gather the set and check the per-file refusal (a leftover
	// backup beside a target) BEFORE anything is written.
	tx := &editTx{root: root, manifestPath: manifestPath}
	seen := map[string]bool{}
	for _, f := range files {
		if seen[f.Rel] {
			continue // one entry per file, however the plan lists it
		}
		seen[f.Rel] = true
		abs := filepath.Join(root, f.Rel)
		bakAbs, _ := backupPathsFor(abs)
		e := &editTxEntry{rel: f.Rel, abs: abs, bakAbs: bakAbs}
		if _, err := os.Stat(bakAbs); err == nil {
			return nil, fmt.Errorf("there's already a leftover backup for %s (%s) — an earlier edit may not have finished. Restore it or remove it first, then ask again", f.Rel, filepath.Base(bakAbs))
		}
		fi, err := os.Stat(abs)
		if os.IsNotExist(err) {
			e.created = true
		} else if err != nil {
			return nil, fmt.Errorf("couldn't read %s to back it up: %v", f.Rel, err)
		} else {
			e.mode = fi.Mode().Perm()
		}
		tx.entries = append(tx.entries, e)
	}
	// The manifest goes down first, state "begun"...
	tx.manifest = &txManifest{
		ID:    fmt.Sprintf("tx-%d-%d", time.Now().UnixNano(), os.Getpid()),
		State: txStateBegun,
	}
	for _, e := range tx.entries {
		rec := txFileRecord{Path: e.rel, Existed: !e.created}
		if !e.created {
			rec.Mode = fmt.Sprintf("%04o", e.mode)
		}
		tx.manifest.Files = append(tx.manifest.Files, rec)
	}
	if err := tx.saveManifest(); err != nil {
		return nil, fmt.Errorf("couldn't write the transaction record (%s): %v", txManifestName, err)
	}
	fail := func(err error) (*editTx, error) {
		for _, e := range tx.entries {
			if e.wroteBackup {
				_ = os.Remove(e.bakAbs)
			}
		}
		_ = os.Remove(manifestPath)
		return nil, err
	}
	// ...then every backup...
	for _, e := range tx.entries {
		if e.created {
			continue
		}
		data, err := os.ReadFile(e.abs)
		if err != nil {
			return fail(fmt.Errorf("couldn't read %s to back it up: %v", e.rel, err))
		}
		if err := writeSyncFile(e.bakAbs, data, 0o600); err != nil {
			return fail(fmt.Errorf("couldn't write the backup for %s: %v", e.rel, err))
		}
		e.wroteBackup = true
	}
	// ...and only a fully backed-up set is armed.
	tx.manifest.State = txStateArmed
	if err := tx.saveManifest(); err != nil {
		return fail(fmt.Errorf("couldn't write the transaction record (%s): %v", txManifestName, err))
	}
	return tx, nil
}

// entry finds the transaction entry for a plan file.
func (tx *editTx) entry(rel string) *editTxEntry {
	for _, e := range tx.entries {
		if e.rel == rel {
			return e
		}
	}
	return nil
}

// write applies one plan file inside the transaction,
// preserving the mode the file had when the transaction began
// (a file the transaction creates gets 0644). The delegate's
// bounded retry writes its alternate form through here too —
// the same transaction, so a retry failure still restores the
// pre-edit original, never the first attempt's output.
func (tx *editTx) write(f analyze.EditFile) error {
	mode := os.FileMode(0o644)
	existing := false
	if e := tx.entry(f.Rel); e != nil && !e.created {
		mode = e.mode
		existing = true
	}
	abs := filepath.Join(tx.root, f.Rel)
	if err := os.WriteFile(abs, f.After, mode); err != nil {
		return err
	}
	if existing {
		// WriteFile only applies its mode to new files; make
		// the preserved mode certain on existing ones too.
		_ = os.Chmod(abs, mode)
	}
	return nil
}

// writeAll applies the files in order. On failure it reports
// how many were written and which one failed; the caller rolls
// the transaction back.
func (tx *editTx) writeAll(files []analyze.EditFile) (written int, err error, failedRel string) {
	for _, f := range files {
		if werr := tx.write(f); werr != nil {
			return written, werr, f.Rel
		}
		written++
	}
	return written, nil, ""
}

// commit closes a verified transaction: the backups and the
// manifest are deleted — clean slate. Backups on disk after a
// commit can only mean an interruption, never a success.
func (tx *editTx) commit() {
	for _, e := range tx.entries {
		if e.wroteBackup {
			_ = os.Remove(e.bakAbs)
		}
	}
	_ = os.Remove(tx.manifestPath)
}

// restore puts one entry back from its on-disk backup — bytes
// and recorded mode; a file the transaction created is removed.
func (e *editTxEntry) restore() error {
	if e.created {
		if err := os.Remove(e.abs); err != nil && !os.IsNotExist(err) {
			return err
		}
		return nil
	}
	data, err := os.ReadFile(e.bakAbs)
	if err != nil {
		return err
	}
	if err := os.WriteFile(e.abs, data, e.mode); err != nil {
		return err
	}
	return os.Chmod(e.abs, e.mode)
}

// rollback restores every file in the set from its on-disk
// backup, best effort across the whole set: one file's failed
// restore doesn't stop the others. Restored files' backups are
// removed; when the whole set is back, the manifest goes too.
// A file that could NOT be restored keeps its backup, the
// manifest is trimmed to just the unrestored files (still
// "armed", so chat-start recovery can finish exactly those),
// and the file is named in the return value (project-relative
// paths, in plan order) so the caller's message can say so
// instead of claiming everything is as it was.
func (tx *editTx) rollback() (unrestored []string) {
	var failed []*editTxEntry
	for _, e := range tx.entries {
		if e.restored {
			continue // an earlier rollback already put it back
		}
		if err := e.restore(); err != nil {
			failed = append(failed, e)
			unrestored = append(unrestored, e.rel)
			continue
		}
		e.restored = true
		// The restored bytes are the ones the session's index was
		// built from; re-baseline the freshness snapshot so the
		// rollback's own mtime bump doesn't read as an outside
		// change at the next write.
		noteSnapshotRestoredFile(tx.root, e.rel, e.created)
		if e.wroteBackup {
			_ = os.Remove(e.bakAbs)
		}
	}
	if len(failed) == 0 {
		_ = os.Remove(tx.manifestPath)
		return nil
	}
	keep := map[string]bool{}
	for _, e := range failed {
		keep[e.rel] = true
	}
	var recs []txFileRecord
	for _, rec := range tx.manifest.Files {
		if keep[rec.Path] {
			recs = append(recs, rec)
		}
	}
	tx.manifest.Files = recs
	_ = tx.saveManifest()
	return unrestored
}

// rollbackOutcome renders the tail of a rollback message: the
// exact-restoration clause when the rollback put the whole set
// back, or the honest naming of what it could not restore —
// those files' backups and the transaction record are still in
// place, and the next chat start will offer the restore.
func rollbackOutcome(unrestored []string, link func(string) string) string {
	if len(unrestored) == 0 {
		return "everything is exactly as it was"
	}
	names := make([]string, 0, len(unrestored))
	for _, rel := range unrestored {
		names = append(names, link(rel))
	}
	return "but I could NOT put back " + strings.Join(names, ", ") +
		" from its on-disk backup — that file may not be as it was. Its backup and the transaction record are still in place, and the next chat start will offer the restore"
}

// --- chat-start recovery -----------------------------------------------------

// resolveStartRoot applies the chat-start root rule: the launch
// folder as given, else the folder the chat is standing in,
// made absolute.
func resolveStartRoot(projectRoot string) string {
	root := projectRoot
	if root == "" {
		root = startFolder()
	}
	if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	return root
}

// startupEditRecovery is RunRealChat's first act, for every
// chat mode: an interrupted edit's manifest gets its recovery
// (prompted on a terminal, manual commands when piped), and
// manifest-less leftover backups get their printed notices.
func startupEditRecovery(projectRoot string) {
	runStartupRecovery(resolveStartRoot(projectRoot), stdinIsTerminal(), ttyAsk)
}

// runStartupRecovery is startupEditRecovery's testable core:
// root is resolved, interactive says whether a prompt may be
// asked at all, and ask reads one answer line.
func runStartupRecovery(root string, interactive bool, ask editAsk) {
	m, mpath, present, rerr := readTxManifest(root)
	if !present {
		printLeftoverNotices(root, nil)
		return
	}
	if rerr != nil {
		// A record we can't read is never deleted and never
		// guessed at: say so, and let the generic scan name
		// whatever backups it can see.
		fmt.Printf("[interrupted edit] There's a transaction record from an interrupted edit in this project (%s), but I couldn't read it (%v) — I'm leaving it exactly as it is. Have a look at it by hand; the backups it names are still beside their files.\n", mpath, rerr)
		printLeftoverNotices(root, nil)
		return
	}
	covered := map[string]bool{}
	for _, f := range m.Files {
		bak, _ := backupPathsFor(filepath.Join(root, filepath.FromSlash(f.Path)))
		covered[bak] = true
	}
	if m.State == txStateBegun {
		recoverBegunTx(root, m, mpath, interactive, ask)
	} else {
		recoverArmedTx(root, m, mpath, interactive, ask)
	}
	// Legacy orphans the manifest doesn't cover still get
	// their printed notices.
	printLeftoverNotices(root, covered)
}

// announceTx prints the shared head of a recovery: what was
// interrupted, and the files it touched.
func announceTx(m *txManifest) {
	fmt.Printf("[interrupted edit] A chat session ended during an edit (transaction %s) in this project, before it could finish or undo itself. The edit touched:\n", m.ID)
	for _, f := range m.Files {
		suffix := ""
		if !f.Existed {
			suffix = " (created by the edit)"
		}
		fmt.Printf("  - %s%s\n", f.Path, suffix)
	}
}

// promptYes asks a [Y/n] question: the default (an empty line,
// or any answer that isn't an explicit no) is yes — the restore
// is the safe side of an interrupted edit.
func promptYes(ask editAsk, question string) bool {
	ans, err := ask(question)
	if err != nil {
		return false // a prompt we couldn't read is not a yes
	}
	switch strings.ToLower(strings.TrimSpace(ans)) {
	case "n", "no":
		return false
	}
	return true
}

// recoverBegunTx handles a "begun" leftover: the edit never
// wrote a target (targets wait for "armed"), so recovery is
// housekeeping — clear the partial backups and the record.
func recoverBegunTx(root string, m *txManifest, mpath string, interactive bool, ask editAsk) {
	announceTx(m)
	fmt.Println("That transaction never started writing — no file was changed by it; only partial backups were left behind.")
	clear := func() {
		for _, f := range m.Files {
			bak, _ := backupPathsFor(filepath.Join(root, filepath.FromSlash(f.Path)))
			_ = os.Remove(bak)
		}
		_ = os.Remove(mpath)
	}
	if interactive && promptYes(ask, "Clear the partial backups and the transaction record? [Y/n] ") {
		clear()
		fmt.Println("[interrupted edit] Cleared — the partial backups and the transaction record are gone, and no file was ever changed by that edit.")
		return
	}
	fmt.Println("Leaving them in place. To clear them by hand, run:")
	for _, f := range m.Files {
		bak, _ := backupPathsFor(filepath.Join(root, filepath.FromSlash(f.Path)))
		fmt.Printf("  rm -f %q\n", bak)
	}
	fmt.Printf("  rm -f %q\n", mpath)
}

// recoverArmedTx handles an "armed" leftover: every original
// is in its backup, and some targets may carry the edit. On a
// terminal the restore is offered (default yes); otherwise the
// exact manual commands are printed, never executed.
func recoverArmedTx(root string, m *txManifest, mpath string, interactive bool, ask editAsk) {
	announceTx(m)
	if interactive && promptYes(ask, "Restore the original files? [Y/n] ") {
		unrestored := restoreTxFiles(root, m)
		if len(unrestored) == 0 {
			fmt.Println("[interrupted edit] Restored — every file is back exactly as it was, and the transaction record is cleared.")
			return
		}
		fmt.Printf("[interrupted edit] I restored what I could, but I could NOT put back %s — the backup for it is missing or unreadable, so that file may not be as it was. The transaction record stays in place, naming it.\n", strings.Join(unrestored, ", "))
		return
	}
	fmt.Println("Leaving everything in place. To restore the originals by hand, run:")
	for _, cmd := range manualRestoreCommands(root, m, mpath) {
		fmt.Printf("  %s\n", cmd)
	}
}

// restoreTxFiles puts every file in the manifest back from its
// backup — bytes and the manifest's recorded mode; files the
// edit created are removed. Restored files' backups are
// removed; when the whole set is back, the manifest is deleted.
// A file that could not be restored keeps its backup, the
// manifest is trimmed to just those files, and it is named in
// the return value.
func restoreTxFiles(root string, m *txManifest) (unrestored []string) {
	var failed []txFileRecord
	for _, f := range m.Files {
		abs := filepath.Join(root, filepath.FromSlash(f.Path))
		bak, _ := backupPathsFor(abs)
		restored := false
		if !f.Existed {
			err := os.Remove(abs)
			restored = err == nil || os.IsNotExist(err)
		} else if data, err := os.ReadFile(bak); err == nil {
			mode := os.FileMode(0o644)
			if v, perr := strconv.ParseUint(f.Mode, 8, 32); perr == nil {
				mode = os.FileMode(v)
			}
			restored = os.WriteFile(abs, data, mode) == nil && os.Chmod(abs, mode) == nil
		}
		if !restored {
			failed = append(failed, f)
			unrestored = append(unrestored, f.Path)
			continue
		}
		_ = os.Remove(bak)
	}
	mpath := filepath.Join(root, txManifestName)
	if len(failed) == 0 {
		_ = os.Remove(mpath)
		return nil
	}
	trimmed := &txManifest{ID: m.ID, State: txStateArmed, Files: failed}
	if data, err := json.MarshalIndent(trimmed, "", "  "); err == nil {
		_ = writeSyncFile(mpath, append(data, '\n'), 0o644)
	}
	return unrestored
}

// manualRestoreCommands renders the exact terminal commands
// that would restore an armed transaction by hand — printed,
// never executed.
func manualRestoreCommands(root string, m *txManifest, mpath string) []string {
	var cmds []string
	for _, f := range m.Files {
		abs := filepath.Join(root, filepath.FromSlash(f.Path))
		if !f.Existed {
			cmds = append(cmds, fmt.Sprintf("rm %q", abs))
			continue
		}
		bak, _ := backupPathsFor(abs)
		cmd := fmt.Sprintf("cp %q %q", bak, abs)
		if f.Mode != "" {
			cmd += fmt.Sprintf(" && chmod %s %q", f.Mode, abs)
		}
		cmd += fmt.Sprintf(" && rm %q", bak)
		cmds = append(cmds, cmd)
	}
	cmds = append(cmds, fmt.Sprintf("rm %q", mpath))
	return cmds
}

// ttyAsk reads one answer line from the terminal, a byte at a
// time — deliberately not a buffered reader, so it can never
// swallow input meant for the chat loop that follows.
func ttyAsk(prompt string) (string, error) {
	fmt.Print(prompt)
	var b []byte
	one := make([]byte, 1)
	for {
		n, err := os.Stdin.Read(one)
		if n > 0 {
			if one[0] == '\n' {
				break
			}
			b = append(b, one[0])
		}
		if err != nil {
			return string(b), err
		}
	}
	return string(b), nil
}

// --- legacy leftover notices -------------------------------------------------

// leftoverBackupNotices scans root for leftover backups
// (.gollemer-backup files) NOT covered by a transaction
// manifest and renders one notice per file, naming the file
// each backup belongs to and the exact terminal command that
// restores it. This is the shipped delegate build's recovery
// surface: the chat prints them at start; it never runs the
// restore itself — the command is the user's to run. exclude
// holds absolute backup paths a manifest already accounts for
// (nil excludes nothing).
func leftoverBackupNotices(root string) []string {
	return leftoverBackupNoticesExcept(root, nil)
}

func leftoverBackupNoticesExcept(root string, exclude map[string]bool) []string {
	if root == "" {
		return nil
	}
	var notices []string
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if name := d.Name(); name == ".git" || name == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		name := d.Name()
		if !strings.HasPrefix(name, ".") || !strings.HasSuffix(name, ".gollemer-backup") {
			return nil
		}
		if exclude[path] {
			return nil
		}
		origName := strings.TrimSuffix(strings.TrimPrefix(name, "."), ".gollemer-backup")
		origAbs := filepath.Join(filepath.Dir(path), origName)
		mode := ""
		if mb, rerr := os.ReadFile(path + ".mode"); rerr == nil {
			mode = strings.TrimSpace(string(mb))
		}
		cmd := fmt.Sprintf("cp %q %q", path, origAbs)
		if mode != "" {
			cmd += fmt.Sprintf(" && chmod %s %q", mode, origAbs)
		}
		cmd += fmt.Sprintf(" && rm %q", path)
		if mode != "" {
			cmd += fmt.Sprintf(" %q", path+".mode")
		}
		notices = append(notices, fmt.Sprintf("[leftover backup] %s has an unfinished-edit backup (%s) from an earlier session. To put the original back: %s", origAbs, path, cmd))
		return nil
	})
	sort.Strings(notices)
	return notices
}

// printLeftoverNotices prints the legacy notices for root,
// skipping any backup a manifest already accounts for.
func printLeftoverNotices(root string, covered map[string]bool) {
	for _, n := range leftoverBackupNoticesExcept(root, covered) {
		fmt.Println(n)
	}
}
