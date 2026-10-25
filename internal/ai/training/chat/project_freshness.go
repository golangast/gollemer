package chat

// The index-freshness invariant for write paths.
//
// analyze.Analyze reads a project once and the session then plans
// edits from that index. If the user changes Go files in another
// editor mid-session, the index's symbol facts — signatures and
// caller lists — no longer describe the tree, while the target
// files' bytes are still read fresh at plan time. Planning from
// those stale facts can produce a confident plan for code that is
// no longer there.
//
// Every time a project becomes the session's current project, this
// file records the mtime and size of every .go file the analyzer
// put in its package list. Before a write shape is selected,
// ensureFreshProject compares the tree with that snapshot: a
// changed, missing, or newly appeared .go file means the index is
// stale, so the project is re-read once and the original request
// proceeds against the fresh facts. This is deliberately a stat
// check, not a content hash: an outside edit that preserves both
// mtime and size exactly is outside the invariant's stated reach.

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// projectFileState is the freshness fingerprint of one analyzed
// .go file: modification time and size, exactly as os.Stat saw
// them just after the analysis that read the file.
type projectFileState struct {
	modTime time.Time
	size    int64
}

// projectTreeSnapshot is the freshness baseline for the current
// project: its root and the state of every .go file in the
// analyzer's package file lists, keyed by slash-relative path.
type projectTreeSnapshot struct {
	root  string
	files map[string]projectFileState
}

// lastAnalyzeSnapshot is the baseline belonging to
// lastAnalyzeProject. setCurrentProject is the only way production
// code installs a current project, so the two never drift apart.
var lastAnalyzeSnapshot *projectTreeSnapshot

// setCurrentProject makes p the session's current project and
// records its freshness baseline. root is the display/session root
// string the caller has always stored in lastAnalyzeRoot; the
// snapshot itself is taken against p.Root, the path Analyze read.
func setCurrentProject(root string, p *analyze.Project) {
	lastAnalyzeRoot = root
	lastAnalyzeProject = p
	if p == nil {
		lastAnalyzeSnapshot = nil
		return
	}
	if lastAnalyzeRoot == "" {
		lastAnalyzeRoot = p.Root
	}
	lastAnalyzeSnapshot = snapshotProjectTree(p)
}

// snapshotProjectTree stats every .go file the analyzer included
// in p's packages — and every file it skipped with a parse error.
// The skipped files matter: they are part of the tree the index
// describes (as known-unreadable), so leaving them out would make
// the new-file sweep report them as "new" on every single write,
// forever. Files that vanish in the instant between the analysis
// and the stat are simply absent from the baseline, which makes
// the next freshness check treat the tree as changed and re-read
// it — the safe direction.
func snapshotProjectTree(p *analyze.Project) *projectTreeSnapshot {
	if p == nil {
		return nil
	}
	snap := &projectTreeSnapshot{root: p.Root, files: map[string]projectFileState{}}
	add := func(rel string) {
		rel = filepath.ToSlash(rel)
		if _, seen := snap.files[rel]; seen {
			return
		}
		fi, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(rel)))
		if err != nil {
			return
		}
		snap.files[rel] = projectFileState{modTime: fi.ModTime(), size: fi.Size()}
	}
	for _, pkg := range p.Packages {
		for _, rel := range pkg.Files {
			add(rel)
		}
	}
	// ParseErr entries are "rel/path.go: reason" — the path is
	// everything up to the .go file's own ": " separator.
	for _, pe := range p.ParseErr {
		rel := pe
		if idx := strings.Index(rel, ".go: "); idx >= 0 {
			rel = rel[:idx+len(".go")]
		}
		add(rel)
	}
	return snap
}

// sameAnalyzedRoot reports whether two path strings name the same
// directory, resolving symlinks when both sides resolve.
func sameAnalyzedRoot(a, b string) bool {
	if a == b {
		return true
	}
	if a == "" || b == "" {
		return false
	}
	ra, ea := filepath.EvalSymlinks(a)
	rb, eb := filepath.EvalSymlinks(b)
	if ea == nil && eb == nil {
		return ra == rb
	}
	aa, erra := filepath.Abs(a)
	bb, errb := filepath.Abs(b)
	return erra == nil && errb == nil && filepath.Clean(aa) == filepath.Clean(bb)
}

// currentProjectStale reports whether the tree under the current
// project differs from the freshness baseline: any snapshotted
// file changed or missing, or any new .go file the baseline does
// not know. A missing baseline (or one for another root) is stale
// by definition — the guard re-reads rather than trust facts it
// cannot date.
func currentProjectStale() bool {
	p := lastAnalyzeProject
	if p == nil {
		return false
	}
	snap := lastAnalyzeSnapshot
	if snap == nil || !sameAnalyzedRoot(snap.root, p.Root) {
		return true
	}
	if _, err := os.Stat(p.Root); err != nil {
		return true
	}
	for rel, want := range snap.files {
		fi, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(rel)))
		if err != nil {
			return true
		}
		if fi.Size() != want.size || !fi.ModTime().Equal(want.modTime) {
			return true
		}
	}
	// New-file sweep: walk the tree the way Analyze does (hidden
	// directories, .git, and vendor excluded) and flag any .go
	// file the baseline lacks.
	stale := false
	walkErr := filepath.WalkDir(p.Root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			if path == p.Root {
				return err
			}
			return nil // unreadable entry: Analyze skips those too
		}
		if d.IsDir() {
			if path != p.Root {
				name := d.Name()
				if name == ".git" || name == "vendor" || strings.HasPrefix(name, ".") {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".go") {
			return nil
		}
		rel, rerr := filepath.Rel(p.Root, path)
		if rerr != nil {
			return nil
		}
		if _, ok := snap.files[filepath.ToSlash(rel)]; !ok {
			stale = true
			return filepath.SkipAll
		}
		return nil
	})
	return stale || walkErr != nil
}

// ensureFreshProject is the write paths' guard: with no project
// loaded it defers to ensureProject; with a fresh index it costs
// only the stat sweep and prints nothing; with a stale index it
// re-reads the project once, says so in one line, and lets the
// caller's request proceed against the fresh facts. When the
// re-read fails — the tree was left mid-edit, or the root is gone
// — it says why and returns false: no write path may plan from
// facts it knows are out of date.
func ensureFreshProject() bool {
	if lastAnalyzeProject == nil {
		return ensureProject()
	}
	if !currentProjectStale() {
		return true
	}
	target := lastAnalyzeProject.Root
	if target == "" {
		target = lastAnalyzeRoot
	}
	fresh, err := analyze.Analyze(target)
	if err != nil {
		fmt.Printf("You changed files outside the chat since I last read this project, but I couldn't re-read it: %v — I won't plan from the old index.\n", err)
		return false
	}
	root := lastAnalyzeRoot
	if root == "" {
		root = target
	}
	setCurrentProject(root, fresh)
	fmt.Println("You changed files outside the chat since I last read this project — I re-read it before planning.")
	return true
}

// noteSnapshotRestoredFile re-baselines one file a transaction
// rollback just put back. The restored bytes are the bytes the
// current index was built from, so the index is still accurate —
// only the mtime moved, and without this note the next write
// would report the chat's own rollback as an outside change.
// Only files the baseline already knows are touched: a file the
// snapshot never held stays unrecorded, so the sweep still sees
// it. A file the transaction created and the rollback deleted
// simply leaves the baseline.
func noteSnapshotRestoredFile(root, rel string, created bool) {
	snap := lastAnalyzeSnapshot
	if snap == nil || !sameAnalyzedRoot(snap.root, root) {
		return
	}
	rel = filepath.ToSlash(rel)
	if _, ok := snap.files[rel]; !ok {
		return
	}
	if created {
		delete(snap.files, rel)
		return
	}
	fi, err := os.Stat(filepath.Join(snap.root, filepath.FromSlash(rel)))
	if err != nil {
		return
	}
	snap.files[rel] = projectFileState{modTime: fi.ModTime(), size: fi.Size()}
}

// noteOwnCreatedFile records a .go file this chat itself just
// created in the current project's snapshot, so the creation does
// not read as an outside change at the next write. Creations never
// re-analyze, so without this the new file would trip the new-file
// sweep even though the chat knows exactly where it came from.
func noteOwnCreatedFile(rel string) {
	p := lastAnalyzeProject
	snap := lastAnalyzeSnapshot
	if p == nil || snap == nil || !sameAnalyzedRoot(snap.root, p.Root) {
		return
	}
	rel = filepath.ToSlash(rel)
	fi, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(rel)))
	if err != nil {
		return
	}
	snap.files[rel] = projectFileState{modTime: fi.ModTime(), size: fi.Size()}
}
