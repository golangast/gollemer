package chat

// Memory and system-state awareness for the local talk seat.
//
// There are three distinct layers here, and they are kept
// distinct on purpose:
//
//   - System state is assembled fresh for every talk context:
//     the clock, the operating system, the project the chat is
//     in, the branch named by .git/HEAD, and the Go files that
//     differ from the freshness snapshot. It is never stored.
//   - Short-term memory is a rolling, in-process summary of
//     this session (built / decided / open). It is updated at
//     the session-context landing hooks and dies with the
//     process.
//   - Long-term memory is a chat-owned store outside every
//     project tree. The storage seam is deliberately small:
//     Load and Save a value. The JSON file backend is complete
//     and is the default. An rqlite backend is not wired: the
//     shipped gorqlite gate guards generated book code in a
//     user's project; it does not configure a database, or a
//     connection file, for the chat's own store. Wiring one
//     would need new configuration UX beyond that gate, so
//     this build ships the seam and the JSON backend rather
//     than a half-connected database.
//
// Long-term captures are explicit only. A line must use one
// of the remembered shapes below; ordinary conversation is
// never mined for preferences. Every successful capture is
// echoed. Stored session summaries are data for recall only:
// nothing in this file feeds training.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// ----- long-term store -----

type longTermSessionSummary struct {
	EndedAt time.Time `json:"ended_at"`
	Text    string    `json:"text"`
	Built   []string  `json:"built,omitempty"`
	Decided []string  `json:"decided,omitempty"`
	Open    []string  `json:"open,omitempty"`
}

type longTermProjectMemory struct {
	Notes     []string                 `json:"notes,omitempty"`
	Summaries []longTermSessionSummary `json:"summaries,omitempty"`
}

type longTermMemoryData struct {
	Version     int                               `json:"version"`
	Preferences []string                          `json:"preferences,omitempty"`
	Projects    map[string]*longTermProjectMemory `json:"projects,omitempty"`
}

func emptyLongTermMemoryData() longTermMemoryData {
	return longTermMemoryData{Version: 1, Projects: map[string]*longTermProjectMemory{}}
}

func normalizeLongTermMemoryData(d longTermMemoryData) longTermMemoryData {
	if d.Version == 0 {
		d.Version = 1
	}
	if d.Projects == nil {
		d.Projects = map[string]*longTermProjectMemory{}
	}
	for root, p := range d.Projects {
		if p == nil {
			d.Projects[root] = &longTermProjectMemory{}
		}
	}
	return d
}

// memoryStore is the whole storage seam: load one value or
// save one value. Operations live in the service functions
// below so a second backend cannot drift semantically from
// the JSON backend.
type memoryStore interface {
	Load() (longTermMemoryData, error)
	Save(longTermMemoryData) error
}

type memoryBackendKind string

const (
	memoryBackendJSON   memoryBackendKind = "json"
	memoryBackendRqlite memoryBackendKind = "rqlite"
)

var (
	activeMemoryBackend     = memoryBackendJSON
	memoryStorePathOverride string
)

var errRqliteMemoryBackendNotWired = errors.New("rqlite memory backend is not wired: the existing gorqlite gate guards book code in a user's project and does not configure a chat-owned store or choose its connection file")

func openMemoryStore() (memoryStore, error) {
	return openMemoryStoreForBackend(activeMemoryBackend)
}

func openMemoryStoreForBackend(kind memoryBackendKind) (memoryStore, error) {
	switch kind {
	case memoryBackendJSON:
		return jsonMemoryStore{path: memoryStorePath()}, nil
	case memoryBackendRqlite:
		return nil, errRqliteMemoryBackendNotWired
	default:
		return nil, fmt.Errorf("unknown memory backend %q", kind)
	}
}

func memoryStorePath() string {
	if memoryStorePathOverride != "" {
		return memoryStorePathOverride
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ".gollemer_memory.json"
	}
	return filepath.Join(home, ".gollemer_memory.json")
}

type jsonMemoryStore struct{ path string }

func (s jsonMemoryStore) Load() (longTermMemoryData, error) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return emptyLongTermMemoryData(), nil
	}
	if err != nil {
		return emptyLongTermMemoryData(), err
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return emptyLongTermMemoryData(), nil
	}
	var out longTermMemoryData
	if err := json.Unmarshal(data, &out); err != nil {
		return emptyLongTermMemoryData(), err
	}
	return normalizeLongTermMemoryData(out), nil
}

func (s jsonMemoryStore) Save(d longTermMemoryData) error {
	d = normalizeLongTermMemoryData(d)
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".gollemer-memory-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, s.path)
}

func loadLongTermMemory() (longTermMemoryData, error) {
	store, err := openMemoryStore()
	if err != nil {
		return emptyLongTermMemoryData(), err
	}
	return store.Load()
}

func saveLongTermMemory(d longTermMemoryData) error {
	store, err := openMemoryStore()
	if err != nil {
		return err
	}
	if err := store.Save(d); err != nil {
		return err
	}
	memoryRecall.loaded = false
	return nil
}

func appendUniqueString(xs []string, x string) []string {
	for _, old := range xs {
		if old == x {
			return xs
		}
	}
	return append(xs, x)
}

func projectMemoryForWrite(d *longTermMemoryData, root string) *longTermProjectMemory {
	root = canonicalMemoryRoot(root)
	if p := d.Projects[root]; p != nil {
		return p
	}
	p := &longTermProjectMemory{}
	d.Projects[root] = p
	return p
}

func addLongTermPreference(text string) error {
	d, err := loadLongTermMemory()
	if err != nil {
		return err
	}
	d.Preferences = appendUniqueString(d.Preferences, text)
	return saveLongTermMemory(d)
}

func addLongTermProjectNote(root, text string) error {
	d, err := loadLongTermMemory()
	if err != nil {
		return err
	}
	p := projectMemoryForWrite(&d, root)
	p.Notes = appendUniqueString(p.Notes, text)
	return saveLongTermMemory(d)
}

func addLongTermSessionSummary(root string, s longTermSessionSummary) error {
	d, err := loadLongTermMemory()
	if err != nil {
		return err
	}
	p := projectMemoryForWrite(&d, root)
	p.Summaries = append(p.Summaries, s)
	if len(p.Summaries) > 20 {
		p.Summaries = append([]longTermSessionSummary{}, p.Summaries[len(p.Summaries)-20:]...)
	}
	return saveLongTermMemory(d)
}

// ----- roots and recall -----

func canonicalMemoryRoot(root string) string {
	if strings.TrimSpace(root) == "" {
		return ""
	}
	if resolved, err := filepath.EvalSymlinks(root); err == nil {
		return filepath.Clean(resolved)
	}
	if abs, err := filepath.Abs(root); err == nil {
		return filepath.Clean(abs)
	}
	return filepath.Clean(root)
}

func memoryCurrentRoot() string {
	if workCtx.root != "" {
		return canonicalMemoryRoot(workCtx.root)
	}
	if lastAnalyzeProject != nil && lastAnalyzeProject.Root != "" {
		return canonicalMemoryRoot(lastAnalyzeProject.Root)
	}
	if lastAnalyzeRoot != "" {
		return canonicalMemoryRoot(lastAnalyzeRoot)
	}
	return canonicalMemoryRoot(unifiedProjectRoot)
}

type memoryRecallState struct {
	root      string
	data      longTermMemoryData
	loaded    bool
	announced map[string]bool
}

var memoryRecall = memoryRecallState{announced: map[string]bool{}}

func resetMemoryRecall() {
	announced := memoryRecall.announced
	memoryRecall = memoryRecallState{announced: announced}
	if memoryRecall.announced == nil {
		memoryRecall.announced = map[string]bool{}
	}
}

func loadMemorySlice(root string) (longTermMemoryData, *longTermProjectMemory) {
	root = canonicalMemoryRoot(root)
	if root == "" {
		return emptyLongTermMemoryData(), nil
	}
	if memoryRecall.loaded && memoryRecall.root == root {
		return memoryRecall.data, memoryRecall.data.Projects[root]
	}
	d, err := loadLongTermMemory()
	if err != nil {
		// Recall must never break an answer. State and memory
		// commands surface store errors on their own paths;
		// context assembly proceeds with an empty slice.
		d = emptyLongTermMemoryData()
	}
	memoryRecall.root = root
	memoryRecall.data = d
	memoryRecall.loaded = true
	return d, d.Projects[root]
}

func latestLongTermSummary(p *longTermProjectMemory) (longTermSessionSummary, bool) {
	if p == nil {
		return longTermSessionSummary{}, false
	}
	for i := len(p.Summaries) - 1; i >= 0; i-- {
		if strings.TrimSpace(p.Summaries[i].Text) != "" && p.Summaries[i].Text != "No pieces landed." {
			return p.Summaries[i], true
		}
	}
	return longTermSessionSummary{}, false
}

// beginMemorySession loads this project's slice and returns
// the one continuity line for a previous session, once per
// root in this process. No previous summary, no line.
func beginMemorySession(root string) string {
	root = canonicalMemoryRoot(root)
	_, p := loadMemorySlice(root)
	if root == "" || memoryRecall.announced[root] {
		return ""
	}
	s, ok := latestLongTermSummary(p)
	if !ok {
		return ""
	}
	memoryRecall.announced[root] = true
	return "Last time here: " + s.Text
}

func memoryContextText(root string) string {
	d, p := loadMemorySlice(root)
	var parts []string
	if len(d.Preferences) > 0 {
		parts = append(parts, "User preferences: "+strings.Join(d.Preferences, "; "))
	}
	if p != nil {
		if len(p.Notes) > 0 {
			parts = append(parts, "Project notes: "+strings.Join(p.Notes, "; "))
		}
		if s, ok := latestLongTermSummary(p); ok {
			parts = append(parts, "Last session here: "+s.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// ----- short-term session summary -----

type sessionMemorySummary struct {
	root    string
	built   []string
	decided []string
	open    []string
}

var shortTermMemory sessionMemorySummary

func resetShortTermMemory() { shortTermMemory = sessionMemorySummary{} }

func sessionMemoryForRoot(root string) {
	root = canonicalMemoryRoot(root)
	if root == "" {
		return
	}
	if shortTermMemory.root != "" && shortTermMemory.root != root {
		shortTermMemory = sessionMemorySummary{}
	}
	shortTermMemory.root = root
}

func noteSessionBuilt(root, description string) {
	sessionMemoryForRoot(root)
	shortTermMemory.built = appendUniqueString(shortTermMemory.built, description)
}

func noteSessionDecision(root, description string) {
	sessionMemoryForRoot(root)
	shortTermMemory.decided = appendUniqueString(shortTermMemory.decided, description)
}

func noteSessionOpen(root, description string) {
	sessionMemoryForRoot(root)
	shortTermMemory.open = appendUniqueString(shortTermMemory.open, description)
}

func resolveSessionOpen(root, needle string) {
	sessionMemoryForRoot(root)
	needle = strings.ToLower(needle)
	kept := shortTermMemory.open[:0]
	for _, item := range shortTermMemory.open {
		if !strings.Contains(strings.ToLower(item), needle) {
			kept = append(kept, item)
		}
	}
	shortTermMemory.open = kept
}

func sessionSummaryText() string {
	var parts []string
	if len(shortTermMemory.built) > 0 {
		parts = append(parts, "Built: "+strings.Join(shortTermMemory.built, "; "))
	}
	if len(shortTermMemory.decided) > 0 {
		parts = append(parts, "Decided: "+strings.Join(shortTermMemory.decided, "; "))
	}
	if len(shortTermMemory.open) > 0 {
		parts = append(parts, "Open: "+strings.Join(shortTermMemory.open, "; "))
	}
	return strings.Join(parts, ". ")
}

func sessionSummaryHasContent() bool {
	return len(shortTermMemory.built)+len(shortTermMemory.decided)+len(shortTermMemory.open) > 0
}

// finishMemorySession stores the rolling summary for the
// project it belongs to and returns the non-silent echo to
// print on the way out. An empty session writes no summary.
func finishMemorySession() string {
	if !sessionSummaryHasContent() {
		return ""
	}
	root := shortTermMemory.root
	if root == "" {
		root = memoryCurrentRoot()
	}
	text := sessionSummaryText()
	s := longTermSessionSummary{
		EndedAt: time.Now(),
		Text:    text,
		Built:   append([]string{}, shortTermMemory.built...),
		Decided: append([]string{}, shortTermMemory.decided...),
		Open:    append([]string{}, shortTermMemory.open...),
	}
	if err := addLongTermSessionSummary(root, s); err != nil {
		return ""
	}
	resetShortTermMemory()
	return "I'll remember this session: " + text + "."
}

// ----- system state -----

type systemState struct {
	Now         time.Time
	OS          string
	Arch        string
	Root        string
	Folder      string
	Branch      string
	Changed     []string
	ChangeNote  string
	Preferences []string
	Session     string
}

func buildSystemState(root string) systemState {
	root = canonicalMemoryRoot(root)
	folder := "."
	if workContextLive(root) && workCtx.folder != "" {
		folder = workCtx.folder
	}
	d, _ := loadMemorySlice(root)
	changed, note := changedGoFilesSinceRead(root)
	return systemState{
		Now:         time.Now(),
		OS:          runtime.GOOS,
		Arch:        runtime.GOARCH,
		Root:        root,
		Folder:      folder,
		Branch:      gitBranchForRoot(root),
		Changed:     changed,
		ChangeNote:  note,
		Preferences: append([]string{}, d.Preferences...),
		Session:     sessionSummaryText(),
	}
}

func (s systemState) Text() string {
	changed := "none"
	if len(s.Changed) > 0 {
		changed = strings.Join(s.Changed, "; ")
	}
	prefs := "none stored"
	if len(s.Preferences) > 0 {
		prefs = strings.Join(s.Preferences, "; ")
	}
	session := s.Session
	if session == "" {
		session = "nothing has landed yet"
	}
	return fmt.Sprintf("System state: time %s; OS %s/%s; project root %s; working folder %s; git branch %s; changed Go files since the last project read: %s (%s); user preferences: %s; this session: %s.",
		s.Now.Format("Monday, January 2, 2006 at 3:04:05 PM MST"), s.OS, s.Arch, s.Root, s.Folder, s.Branch, changed, s.ChangeNote, prefs, session)
}

func gitBranchForRoot(root string) string {
	if root == "" {
		return "unknown"
	}
	dir := root
	for {
		gitPath := filepath.Join(dir, ".git")
		gitDir := ""
		if fi, err := os.Stat(gitPath); err == nil {
			if fi.IsDir() {
				gitDir = gitPath
			} else {
				data, err := os.ReadFile(gitPath)
				if err == nil {
					text := strings.TrimSpace(string(data))
					if strings.HasPrefix(text, "gitdir:") {
						gitDir = strings.TrimSpace(strings.TrimPrefix(text, "gitdir:"))
						if !filepath.IsAbs(gitDir) {
							gitDir = filepath.Join(dir, gitDir)
						}
					}
				}
			}
		}
		if gitDir != "" {
			if branch, ok := readGitHeadBranch(gitDir); ok {
				return branch
			}
			return "unknown"
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "unknown"
		}
		dir = parent
	}
}

func readGitHeadBranch(gitDir string) (string, bool) {
	data, err := os.ReadFile(filepath.Join(gitDir, "HEAD"))
	if err != nil {
		// In a linked worktree HEAD can live under the common
		// dir named by commondir.
		common, cerr := os.ReadFile(filepath.Join(gitDir, "commondir"))
		if cerr != nil {
			return "", false
		}
		commonDir := strings.TrimSpace(string(common))
		if !filepath.IsAbs(commonDir) {
			commonDir = filepath.Join(gitDir, commonDir)
		}
		data, err = os.ReadFile(filepath.Join(commonDir, "HEAD"))
		if err != nil {
			return "", false
		}
	}
	head := strings.TrimSpace(string(data))
	if strings.HasPrefix(head, "ref:") {
		ref := strings.TrimSpace(strings.TrimPrefix(head, "ref:"))
		return strings.TrimPrefix(ref, "refs/heads/"), true
	}
	if len(head) >= 7 {
		return "detached at " + head[:7], true
	}
	return "", false
}

// changedGoFilesSinceRead is deliberately narrower than git
// status. The frame's verified baseline is the freshness
// snapshot of analyzed Go files, so this reports modified,
// deleted, and newly appeared Go files against that baseline
// and names the limitation in ChangeNote.
func changedGoFilesSinceRead(root string) ([]string, string) {
	const note = "Go files checked against the chat's last project read; this is not a full git status"
	snap := lastAnalyzeSnapshot
	if snap == nil || lastAnalyzeProject == nil || !sameAnalyzedRoot(snap.root, root) {
		return nil, "no project-read baseline is available, so changed files cannot be listed honestly"
	}
	var changed []string
	for rel, want := range snap.files {
		fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			changed = append(changed, "deleted: "+rel)
			continue
		}
		if fi.Size() != want.size || !fi.ModTime().Equal(want.modTime) {
			changed = append(changed, "modified: "+rel)
		}
	}
	_ = filepath.WalkDir(root, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			if path == root {
				return err
			}
			return nil
		}
		if entry.IsDir() {
			if path != root {
				name := entry.Name()
				if name == ".git" || name == "vendor" || strings.HasPrefix(name, ".") {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if !strings.HasSuffix(entry.Name(), ".go") {
			return nil
		}
		rel, rerr := filepath.Rel(root, path)
		if rerr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if _, ok := snap.files[rel]; !ok {
			changed = append(changed, "new: "+rel)
		}
		return nil
	})
	sort.Strings(changed)
	return changed, note
}

// ----- deterministic state answers -----

var (
	memoryTimeRe    = regexp.MustCompile(`(?i)^\s*(what(?:'s| is) the (current )?(time|date|day)( today)?|what time is it|what day is it|what(?:'s| is) today'?s (date|day)|current time|current date)\s*[?.!]*\s*$`)
	memoryOSRe      = regexp.MustCompile(`(?i)^\s*(what (operating system|os|system|environment) (am i|are we) (on|in)|what os is this|where are we working|what project are we in|what folder are we in)\s*[?.!]*\s*$`)
	memoryBranchRe  = regexp.MustCompile(`(?i)^\s*(what branch (am i|are we) on|which branch is this|what(?:'s| is) the (git )?branch)\s*[?.!]*\s*$`)
	memoryChangedRe = regexp.MustCompile(`(?i)^\s*(what(?:'s| is) (my |the )?(git status|status)|what changed|what files changed|is the (tree|working tree) dirty|are there any changes)\s*[?.!]*\s*$`)
	memoryDoneRe    = regexp.MustCompile(`(?i)^\s*(what have we (made|built|done)|what did we (just )?(make|build|do)|what have we done so far)\s*[?.!]*\s*$`)
	memoryOpenRe    = regexp.MustCompile(`(?i)^\s*(what(?:'s| is) (still )?open|what threads are open|what is left open)\s*[?.!]*\s*$`)
	memoryDecidedRe = regexp.MustCompile(`(?i)^\s*(what have we decided|what did we decide)\s*[?.!]*\s*$`)
)

func isTalkStateLine(line string) bool {
	return memoryTimeRe.MatchString(line) || memoryOSRe.MatchString(line) ||
		memoryBranchRe.MatchString(line) || memoryChangedRe.MatchString(line) ||
		memoryDoneRe.MatchString(line) || memoryOpenRe.MatchString(line) ||
		memoryDecidedRe.MatchString(line)
}

func talkStateAnswer(ctx TalkContext, line string) (string, bool) {
	s := ctx.State
	switch {
	case memoryTimeRe.MatchString(line):
		lower := strings.ToLower(line)
		if strings.Contains(lower, "date") || strings.Contains(lower, "day") {
			return "It's " + s.Now.Format("Monday, January 2, 2006") + ".", true
		}
		return "It's " + s.Now.Format("3:04:05 PM MST on Monday, January 2, 2006") + ".", true
	case memoryOSRe.MatchString(line):
		lower := strings.ToLower(line)
		if strings.Contains(lower, "folder") {
			if s.Folder == "." {
				return fmt.Sprintf("We're working in the project root folder (%s).", s.Root), true
			}
			return fmt.Sprintf("We're working in the %s folder (%s).", s.Folder, filepath.Join(s.Root, filepath.FromSlash(s.Folder))), true
		}
		if strings.Contains(lower, "project") {
			return fmt.Sprintf("We're in the project at %s, working folder %s.", s.Root, s.Folder), true
		}
		return fmt.Sprintf("You're on %s (%s). This chat is running in %s, and the working folder is %s.", s.OS, s.Arch, s.Root, s.Folder), true
	case memoryBranchRe.MatchString(line):
		if s.Branch == "unknown" {
			return "I couldn't read a branch from this project's .git/HEAD.", true
		}
		return fmt.Sprintf("You're on branch %s.", s.Branch), true
	case memoryChangedRe.MatchString(line):
		if strings.HasPrefix(s.ChangeNote, "no project-read") {
			return "I don't have a project-read baseline yet, so I can't honestly list changed files. I can read the branch from .git, but I won't call that a git status.", true
		}
		if len(s.Changed) == 0 {
			return "No Go files have changed since I last read this project. " + s.ChangeNote + ".", true
		}
		return "Changed Go files since I last read this project: " + strings.Join(s.Changed, "; ") + ". " + s.ChangeNote + ".", true
	case memoryDoneRe.MatchString(line):
		if len(shortTermMemory.built) > 0 {
			return "So far this session: Built: " + strings.Join(shortTermMemory.built, "; ") + ".", true
		}
		if ctx.WorkLive && ctx.Work.piece != "" {
			what := ctx.Work.piece
			if ctx.Work.pieceName != "" {
				what += " " + ctx.Work.pieceName
			}
			return fmt.Sprintf("The last thing that landed was %s in %s.", what, ctx.Work.lastFile), true
		}
		return "Nothing has landed yet this session.", true
	case memoryOpenRe.MatchString(line):
		if len(shortTermMemory.open) == 0 {
			return "No open threads are being held in this session.", true
		}
		return "Open this session: " + strings.Join(shortTermMemory.open, "; ") + ".", true
	case memoryDecidedRe.MatchString(line):
		if len(shortTermMemory.decided) == 0 {
			return "No explicit preferences or notes have been captured in this session.", true
		}
		return "Decided this session: " + strings.Join(shortTermMemory.decided, "; ") + ".", true
	}
	return "", false
}

// ----- explicit memory commands -----

func memoryReply(line string, conv *Conversation, text string) bool {
	fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", TalkDomain)), text)
	if conv != nil {
		conv.AddUser(line)
		conv.AddReply(text, TalkDomain, false)
	}
	return true
}

func cleanMemoryText(s string) string {
	s = strings.Join(strings.Fields(strings.TrimSpace(s)), " ")
	return strings.TrimSuffix(s, ".")
}

func isPreferenceText(s string) bool {
	lower := strings.ToLower(s)
	for _, prefix := range []string{"i prefer", "i like", "i want", "i use", "my preference is", "my name is", "call me"} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

var (
	rememberColonRe  = regexp.MustCompile(`(?i)^\s*(?:please\s+)?remember\s*:\s*(.+?)\s*$`)
	rememberThatRe   = regexp.MustCompile(`(?i)^\s*(?:please\s+)?remember that\s+(.+?)\s*$`)
	rememberPlainRe  = regexp.MustCompile(`(?i)^\s*(?:please\s+)?remember\s+(.+?)\s*$`)
	projectNoteRe    = regexp.MustCompile(`(?i)^\s*(?:please\s+)?(?:note(?: for this project)?|remember (?:a note )?for this project)\s*:\s*(.+?)\s*$`)
	noteThatRe       = regexp.MustCompile(`(?i)^\s*(?:please\s+)?note that\s+(.+?)\s*$`)
	callMeRe         = regexp.MustCompile(`(?i)^\s*(?:please\s+)?call me\s+(.+?)\s*$`)
	memoryRecallRe   = regexp.MustCompile(`(?i)^\s*(what do you remember(?: about (me|this project))?|show me what you remember|what have you remembered)\s*[?.!]*\s*$`)
	forgetAllRe      = regexp.MustCompile(`(?i)^\s*forget everything\s*[!.]*\s*$`)
	forgetProjectRe  = regexp.MustCompile(`(?i)^\s*forget this project\s*[!.]*\s*$`)
	forgetPrefixedRe = regexp.MustCompile(`(?i)^\s*forget my\s+(.+?)\s*[!.]*\s*$`)
)

func parseMemoryCapture(line string) (scope, text string, ok bool) {
	if m := projectNoteRe.FindStringSubmatch(line); m != nil {
		return "project", cleanMemoryText(m[1]), true
	}
	// "note that <fact>" — the colon note's spoken form
	// (widened 2026-11-05; stored as the project's note,
	// same as the colon shape).
	if m := noteThatRe.FindStringSubmatch(line); m != nil {
		return "project", cleanMemoryText(m[1]), true
	}
	if m := callMeRe.FindStringSubmatch(line); m != nil {
		return "preference", "call me " + cleanMemoryText(m[1]), true
	}
	if m := rememberColonRe.FindStringSubmatch(line); m != nil {
		text = cleanMemoryText(m[1])
		if isPreferenceText(text) {
			return "preference", text, true
		}
		return "project", text, true
	}
	if m := rememberThatRe.FindStringSubmatch(line); m != nil {
		text = cleanMemoryText(m[1])
		if isPreferenceText(text) {
			return "preference", text, true
		}
		return "project", text, true
	}
	if m := rememberPlainRe.FindStringSubmatch(line); m != nil {
		text = cleanMemoryText(m[1])
		if isPreferenceText(text) {
			return "preference", text, true
		}
	}
	return "", "", false
}

func renderMemoryRecall(root string) (string, error) {
	d, err := loadLongTermMemory()
	if err != nil {
		return "", err
	}
	root = canonicalMemoryRoot(root)
	p := d.Projects[root]
	var b strings.Builder
	b.WriteString("Here's what I remember.\nYour preferences:")
	if len(d.Preferences) == 0 {
		b.WriteString("\n- none stored")
	} else {
		for _, pref := range d.Preferences {
			b.WriteString("\n- " + pref)
		}
	}
	fmt.Fprintf(&b, "\nFor this project (%s):", root)
	if p == nil || (len(p.Notes) == 0 && len(p.Summaries) == 0) {
		b.WriteString("\n- no notes or past session summaries stored")
	} else {
		for _, note := range p.Notes {
			b.WriteString("\n- Project note: " + note)
		}
		if s, ok := latestLongTermSummary(p); ok {
			b.WriteString("\n- Last session: " + s.Text)
		}
	}
	b.WriteString("\nThat's everything stored for you and this project.")
	return b.String(), nil
}

func normalizeMemoryMatch(s string) string {
	s = strings.ToLower(s)
	s = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == ' ' {
			return r
		}
		return ' '
	}, s)
	return strings.Join(strings.Fields(s), " ")
}

func forgetLongTermPreference(query string) (removed string, candidates []string, err error) {
	q := normalizeMemoryMatch(query)
	q = strings.TrimPrefix(q, "preference for ")
	q = strings.TrimPrefix(q, "preference ")
	q = strings.TrimSuffix(q, " preference")
	if q == "" || q == "preference" {
		d, lerr := loadLongTermMemory()
		if lerr != nil {
			return "", nil, lerr
		}
		return "", d.Preferences, nil
	}
	d, err := loadLongTermMemory()
	if err != nil {
		return "", nil, err
	}
	var matches []int
	for i, pref := range d.Preferences {
		np := normalizeMemoryMatch(pref)
		if np == q || strings.Contains(np, q) || strings.Contains(q, np) {
			matches = append(matches, i)
		}
	}
	if len(matches) == 0 {
		return "", nil, nil
	}
	if len(matches) > 1 {
		for _, i := range matches {
			candidates = append(candidates, d.Preferences[i])
		}
		return "", candidates, nil
	}
	i := matches[0]
	removed = d.Preferences[i]
	d.Preferences = append(d.Preferences[:i], d.Preferences[i+1:]...)
	return removed, nil, saveLongTermMemory(d)
}

// tryMemoryCommand claims only the explicit long-term
// memory shapes: captures, recall, and forgets. It returns
// false for every ordinary line, so no preference is ever
// inferred from conversation.
func tryMemoryCommand(line string, conv *Conversation) bool {
	root := memoryCurrentRoot()
	if forgetAllRe.MatchString(line) {
		if err := saveLongTermMemory(emptyLongTermMemoryData()); err != nil {
			return memoryReply(line, conv, "I couldn't update my memory store: "+err.Error())
		}
		resetMemoryRecall()
		return memoryReply(line, conv, "Forgotten: everything I had stored — all preferences, all project notes, and all session summaries. Nothing else was touched.")
	}
	if forgetProjectRe.MatchString(line) {
		d, err := loadLongTermMemory()
		if err != nil {
			return memoryReply(line, conv, "I couldn't update my memory store: "+err.Error())
		}
		delete(d.Projects, root)
		if err := saveLongTermMemory(d); err != nil {
			return memoryReply(line, conv, "I couldn't update my memory store: "+err.Error())
		}
		return memoryReply(line, conv, fmt.Sprintf("Forgotten: this project's notes and session summaries (%s). Your preferences and other projects were not touched.", root))
	}
	if m := forgetPrefixedRe.FindStringSubmatch(line); m != nil {
		removed, candidates, err := forgetLongTermPreference(m[1])
		if err != nil {
			return memoryReply(line, conv, "I couldn't update my memory store: "+err.Error())
		}
		if removed != "" {
			return memoryReply(line, conv, fmt.Sprintf("Forgotten: your preference %q. Nothing else was touched.", removed))
		}
		if len(candidates) == 1 {
			return memoryReply(line, conv, fmt.Sprintf("I have one stored preference: %q. Say \"forget my %s\" with its words and I'll remove only that one.", candidates[0], candidates[0]))
		}
		if len(candidates) > 1 {
			return memoryReply(line, conv, "That matches more than one stored preference — "+strings.Join(candidates, "; ")+". Nothing was forgotten; say which one you mean.")
		}
		return memoryReply(line, conv, "I don't have a stored preference matching that. Nothing was forgotten.")
	}
	if memoryRecallRe.MatchString(line) {
		text, err := renderMemoryRecall(root)
		if err != nil {
			return memoryReply(line, conv, "I couldn't read my memory store: "+err.Error())
		}
		return memoryReply(line, conv, text)
	}
	if scope, text, ok := parseMemoryCapture(line); ok {
		var err error
		if scope == "preference" {
			err = addLongTermPreference(text)
			if err == nil {
				noteSessionDecision(root, "preference: "+text)
			}
			if err != nil {
				return memoryReply(line, conv, "I couldn't save that to my memory store: "+err.Error())
			}
			return memoryReply(line, conv, "I'll remember: "+text+".")
		}
		err = addLongTermProjectNote(root, text)
		if err == nil {
			noteSessionDecision(root, "project note: "+text)
		}
		if err != nil {
			return memoryReply(line, conv, "I couldn't save that to my memory store: "+err.Error())
		}
		return memoryReply(line, conv, "I'll remember: "+text+" (for this project).")
	}
	return false
}
