// Multi-file patch transactions.
//
// ApplyPatchSet validates a set of file changes in the runner sandbox
// and commits them to disk only if every affected package builds and
// passes `go test`. Either all files validate and are committed, or
// nothing on disk changes.
package runner

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// PatchSet is a set of file changes applied transactionally. Keys are
// module-relative paths ("pkg/foo/bar.go"); values are the complete
// new file contents.
type PatchSet struct {
	Files map[string]string
}

// Add records a file change in the set.
func (p *PatchSet) Add(path, content string) {
	if p.Files == nil {
		p.Files = make(map[string]string)
	}
	p.Files[path] = content
}

// Len returns the number of files in the set.
func (p PatchSet) Len() int {
	return len(p.Files)
}

// ApplyPatchSet validates patches in an isolated sandbox and commits
// them to targetDir only if `go test ./...` passes for every package.
//
//   - Validation failure: the result (with parsed errors) is returned
//     and nothing on disk is touched.
//   - Infrastructure failure (bad targetDir, path escape, write error):
//     an error is returned. A commit that fails midway rolls back the
//     files already written, so the tree is never left half-patched.
//
// Candidate paths are jailed to targetDir by the same rules as
// ValidateGeneratedCode.
func ApplyPatchSet(targetDir string, patches PatchSet) (*ExecutionResult, error) {
	if targetDir == "" {
		return nil, fmt.Errorf("runner: empty targetDir")
	}
	if len(patches.Files) == 0 {
		return nil, fmt.Errorf("runner: empty patch set")
	}
	names := make([]string, 0, len(patches.Files))
	for name := range patches.Files {
		names = append(names, name)
	}
	sort.Strings(names)

	fmt.Printf("[tx] validating %d patched file(s) in sandbox...\n", len(names))
	res, err := ValidateGeneratedCode(targetDir, patches.Files)
	if err != nil {
		return nil, fmt.Errorf("runner: patch validation: %w", err)
	}
	if !res.Passed {
		fmt.Printf("[tx] validation FAILED with %d error(s); nothing committed\n", len(res.Errors))
		return res, nil
	}
	if err := commitPatches(targetDir, patches.Files, names); err != nil {
		return nil, err
	}
	fmt.Printf("[tx] committed %d file(s) to %s\n", len(names), targetDir)
	return res, nil
}

// fileBackup records a file's pre-commit state for rollback.
type fileBackup struct {
	existed bool
	data    []byte
	mode    os.FileMode
}

// commitPatches writes every patched file to targetDir. On any failure
// it restores the files already written (and removes newly created
// ones), leaving the tree as it was.
func commitPatches(targetDir string, files map[string]string, names []string) error {
	backups := make(map[string]fileBackup, len(names))
	var written []string
	rollback := func() {
		// Best effort; nothing sensible to do with a rollback
		// failure beyond reporting the original error.
		for _, name := range written {
			b := backups[name]
			path := filepath.Join(targetDir, name)
			if b.existed {
				_ = os.WriteFile(path, b.data, b.mode)
			} else {
				_ = os.Remove(path)
			}
		}
	}
	for _, name := range names {
		// Re-jail here: validation already rejected bad paths, but
		// the commit writes outside the sandbox, so check again
		// rather than trusting call order.
		path, err := safeJoin(targetDir, name)
		if err != nil {
			rollback()
			return fmt.Errorf("runner: commit %q: %w", name, err)
		}
		var b fileBackup
		if data, rerr := os.ReadFile(path); rerr == nil {
			mode := os.FileMode(0o644)
			if info, serr := os.Stat(path); serr == nil {
				mode = info.Mode().Perm()
			}
			b = fileBackup{existed: true, data: data, mode: mode}
		}
		backups[name] = b
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			rollback()
			return fmt.Errorf("runner: commit %q: create dir: %w", name, err)
		}
		mode := os.FileMode(0o644)
		if b.existed {
			mode = b.mode
		}
		if err := os.WriteFile(path, []byte(files[name]), mode); err != nil {
			rollback()
			return fmt.Errorf("runner: commit %q: write: %w", name, err)
		}
		written = append(written, name)
	}
	return nil
}
