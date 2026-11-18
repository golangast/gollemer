package chat

// The session's working context: what we're building, where,
// and what just landed.
//
// John's live failure (2026-11-02) was four lines spent
// building one thing — a database package in jake/database/ —
// with the chat treating each line as a stranger: every line
// had to name its file in full, and the refusals ignored the
// work in progress. The session already carried fragments of
// context before this file — the current project
// (lastAnalyzeRoot/lastAnalyzeProject, installed by
// setCurrentProject), the gorqlite gate's one held ask
// (pendingGorqlite), the freshness snapshot's record of the
// chat's own creations (noteOwnCreatedFile) — but nothing
// remembered WHAT WAS JUST MADE. This file is that memory,
// built on the same pattern as the held ask: plain
// package-level session state, never on disk.
//
// What is tracked (and only this):
//
//   - the project root the context belongs to — context is
//     inert for any other root, and setCurrentProject clears
//     it when the session's project actually changes (a cd,
//     a clone, a fresh analyze of somewhere else);
//   - the working folder: the folder of the file the work
//     just touched, or a folder the chat just created — with
//     its package name read off the disk at note time;
//   - the last file the chat created or successfully edited;
//   - the last piece landed there: a plain-words kind (the
//     book's canonical name for a concept piece) and its
//     name when it has one;
//   - the file the landing before that touched, which of a
//     file or a folder landed most recently, and — for the
//     families whose sentences rebuild exactly — the last
//     piece in rebuildable form: the richer follow-ups'
//     state (followup.go), and the vocabulary's "this"
//     (vocab.go).
//
// Notes land ONLY on success: at the shared edit spine's
// commit, at runCreate's two Done points, and at the rename /
// uses / delegate / error-fix commits. A refusal, a declined
// gate, and a rolled-back edit note nothing — the context
// describes work that is really on disk, never work that was
// merely discussed.
//
// Resolution (resolveContextTarget / resolvePronounTail /
// create's folder reading) is deliberately narrow, because
// silent guessing is this product's cardinal sin:
//
//   - a pronoun tail ("to it", "to that file", "to the same
//     file") reads as the one tracked file;
//   - a bare .go file name reads as the tracked file when
//     its name matches, or as an existing file of that name
//     in the working folder;
//   - a bare file name in the add-verb creation form
//     ("add the file db.go") reads as the working folder
//     (a folder context is singular, so this reading is
//     never ambiguous) — create|make bare names keep the
//     create family's shipped root anchoring;
//   - when a DIFFERENT file of the same name also exists at
//     the project root, two files fit and the answer is a
//     clarifying question naming both — never a coin flip;
//   - when nothing in the context fits, nothing here engages
//     at all and the line keeps today's behavior, byte for
//     byte (a cold session's bare name still resolves
//     root-relative, an uncontexted "to it" still meets the
//     resolver's refusal).
//
// And every resolution is VISIBLE: the claim prints the
// reading — the full resolved path and why — before its plan
// or refusal, so the assumption can be seen and corrected.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// workContext is the session's memory of the work in
// progress. The zero value is the cold session: no context,
// nothing resolves.
type workContext struct {
	root      string // project root (absolute) this context belongs to
	folder    string // slash-relative working folder; "" = the project root
	pkg       string // the working folder's package name, when known
	lastFile  string // slash-relative path of the last file created or edited
	lastHow   string // "created" | "edited": how lastFile got there
	piece     string // plain-words kind of the last piece landed
	pieceName string // the last piece's name, when it has one ("Connect")

	// The richer follow-ups' state (followup.go, 2026-11-05):
	// the file the landing BEFORE lastFile touched (the
	// session's second file — "the other file" has a referent
	// only when exactly these two are in play), which of a
	// file or a folder landed most recently ("this" in the
	// vocabulary's teach sentences, vocab.go), and the last
	// piece's rebuildable snapshot when its family has one.
	prevFile string // slash-relative; "" = no second file in play
	prevHow  string // "created" | "edited": how prevFile got there
	lastKind string // "file" | "folder": what landed most recently
	landed   *landedPiece

	// The wasm events piece's context (wasmevent.go): the
	// folder of the last wasm app this session created or
	// added a component or an event to — the app an event
	// sentence with no folder named resolves to, with the
	// reading printed. "" = no app touched here.
	wasmFolder string
}

// landedPiece is the last landed piece in rebuildable form:
// everything a follow-up ("another one", "the same but for
// X") needs to reconstruct the piece's own canonical
// sentence and re-run its own claim, so the repeat faces
// every fact check, gate, transaction and build
// verification the original did. Only the families whose
// sentences rebuild exactly carry a snapshot — the concept
// book, taught kinds, the webserver, serve-html, wiring,
// the page family, and plain file creations. Everything
// else (the assembled edits, delegates, uses, renames,
// make-and-call's two-file compound, instantiations,
// model-proposed declarations) leaves landed nil: the
// follow-up claim then answers honestly that this piece
// has no sentence it can rebuild, and never fabricates one.
type landedPiece struct {
	family string // "concept" | "taught" | "server" | "serve" | "wire" | "page" | "create"
	file   string // the piece's primary file, slash-relative ("" for wire)

	// concept: the book entry's ID (the entry itself is
	// re-read from the loaded book at rebuild time), the
	// declaration's name ("" = the piece takes none), and
	// the statement slots that are part of the piece's
	// shape. The line-phrase placement is deliberately NOT
	// carried: a repeat lands where a fresh ask would put
	// it, not on the original's line.
	conceptID string
	name      string
	text      string // the {text} saying slot, as planned ("" = the row's default)
	inFunc    string // the statement's host function slot
	like      string // the {methods} source type slot

	// taught: the kind phrase (the name rides in name).
	kindPhrase string

	// serve: the page the recipe serves (slash-relative).
	html string

	// wire: the handler's name rides in name; the route is
	// deliberately NOT carried — it was the first handler's
	// address, not the piece's shape, so a repeat wires at
	// the new handler's own default route.

	// page: the page kind and its handler's name (in name).
	pageKind string

	// create: the created file's folder and the starter the
	// creation carried ("" = a plain empty file).
	folder     string
	createKind string
}

var workCtx workContext

// resetWorkContext forgets the work in progress. Called by
// setCurrentProject when the session's project changes.
func resetWorkContext() {
	workCtx = workContext{}
	receiptResumePending = receiptResumeState{}
	clearPendingQuestion()
	resetShortTermMemory()
	resetMemoryRecall()
}

// workContextLive reports whether the tracked context
// belongs to the project rooted at rootAbs.
func workContextLive(rootAbs string) bool {
	return workCtx.root != "" && sameAnalyzedRoot(workCtx.root, rootAbs)
}

// noteLandedFile records a file the chat just created or
// successfully edited, with the piece that landed in it. The
// working folder follows the file, and its package is read
// from the folder's own Go files.
func noteLandedFile(rootAbs, rel, how, piece, name string) {
	rel = filepath.ToSlash(rel)
	if workCtx.lastFile != "" && workCtx.lastFile != rel {
		workCtx.prevFile = workCtx.lastFile
		workCtx.prevHow = workCtx.lastHow
	}
	workCtx.root = rootAbs
	workCtx.lastFile = rel
	workCtx.lastHow = how
	workCtx.lastKind = "file"
	workCtx.piece = piece
	workCtx.pieceName = name
	workCtx.folder = slashDir(rel)
	workCtx.pkg = folderPackageOf(rootAbs, workCtx.folder)
	desc := piece
	if name != "" {
		desc += " " + name
	}
	if desc != "" {
		desc += " in " + rel
	} else {
		desc = rel
	}
	noteSessionBuilt(rootAbs, desc)
}

// noteLandedFolder records a folder the chat just created:
// the working folder moves there, but the last FILE stands —
// creating a folder touches no file.
func noteLandedFolder(rootAbs, rel string) {
	rel = filepath.ToSlash(rel)
	workCtx.root = rootAbs
	workCtx.folder = rel
	workCtx.lastKind = "folder"
	workCtx.pkg = folderPackageOf(rootAbs, rel)
	noteSessionBuilt(rootAbs, "folder "+rel)
}

// slashDir is the slash-relative folder of a slash-relative
// file path; "" for a file at the project root.
func slashDir(rel string) string {
	if i := strings.LastIndex(rel, "/"); i >= 0 {
		return rel[:i]
	}
	return ""
}

// folderPackageOf reads a folder's package name from the Go
// files on disk (the create family's helper): exactly one
// package is a fact; zero or several is "" — unknown, never
// guessed.
func folderPackageOf(rootAbs, folder string) string {
	dirAbs := rootAbs
	if folder != "" {
		dirAbs = filepath.Join(rootAbs, filepath.FromSlash(folder))
	}
	if pkgs := folderPackages(dirAbs); len(pkgs) == 1 {
		return pkgs[0]
	}
	return ""
}

// claimRootAbs settles a claim's root to an absolute path
// the way the claims themselves do (symlinks resolved,
// absolute fallback), but silently: "" when it can't be
// settled, and the caller simply skips the context readings
// — the claim's own root block will speak for the failure
// exactly as it does today.
func claimRootAbs(rootRaw string) (string, bool) {
	if abs, err := filepath.EvalSymlinks(rootRaw); err == nil {
		return abs, true
	}
	abs, err := filepath.Abs(rootRaw)
	if err != nil {
		return "", false
	}
	return abs, true
}

// noteSpineLanded records a landed edit-spine plan in the
// working context. The primary file is the request's own
// target (every spine caller names it there); a request
// that named none falls back to the plan's first file. A
// plan file with no original bytes was created by the plan
// (a declaration creating its named file, the serve-html
// program); anything else was edited. The piece is the
// book's canonical name for a concept piece, the request
// kind otherwise; the declaration name rides along when
// there is one.
func noteSpineLanded(req analyze.EditRequest, plan *analyze.EditPlan, root string) {
	if plan == nil {
		return
	}
	rel := req.File
	if rel == "" && len(plan.Files) > 0 {
		rel = plan.Files[0].Rel
	}
	if rel == "" {
		return
	}
	how := "edited"
	for _, f := range plan.Files {
		if f.Rel == rel && len(f.Before) == 0 {
			how = "created"
		}
	}
	piece := string(req.Kind)
	name := req.ConceptName
	if req.Kind == analyze.EditConcept && req.Concept != nil {
		piece = req.Concept.Canonical()
	}
	if req.Kind == analyze.EditTaughtDecl {
		piece = req.TaughtKindPhrase
		name = req.TaughtName
	}
	noteLandedFile(root, rel, how, piece, name)
	// The wasm family's landings move the session's wasm
	// app with them: the app created, a component into it,
	// or an event into it — the folder an event sentence
	// with no folder of its own resolves to.
	if req.WasmFolder != "" && (req.Kind == analyze.EditWasmApp || req.Kind == analyze.EditComponent || req.Kind == analyze.EditTaughtComponent || req.Kind == analyze.EditWasmEvent || req.Kind == analyze.EditWasmPart) {
		workCtx.wasmFolder = req.WasmFolder
	}
	noteSpineSnapshot(req, rel)
}

// landedSnapshotFor builds a landed piece in rebuildable
// form for the follow-ups (followup.go): the families whose
// canonical sentences rebuild exactly from the request the
// spine just carried. Every other kind returns nil — the
// follow-up claim answers those honestly rather than
// inventing a sentence. The receipts store (receipts.go)
// serializes this same snapshot, so an in-session follow-up
// and a cross-session one rebuild from the same facts.
func landedSnapshotFor(req analyze.EditRequest, rel string) *landedPiece {
	switch req.Kind {
	case analyze.EditConcept:
		if req.Concept == nil {
			return nil
		}
		return &landedPiece{
			family:    "concept",
			file:      rel,
			conceptID: req.Concept.ID,
			name:      req.ConceptName,
			text:      req.ConceptText,
			inFunc:    req.ConceptFunc,
			like:      req.ConceptLike,
		}
	case analyze.EditTaughtDecl:
		return &landedPiece{
			family:     "taught",
			file:       rel,
			kindPhrase: req.TaughtKindPhrase,
			name:       req.TaughtName,
		}
	case analyze.EditAddServer:
		return &landedPiece{family: "server", file: rel}
	case analyze.EditServeHTML:
		return &landedPiece{family: "serve", file: rel, html: req.ServeHTML}
	case analyze.EditWireHandler:
		return &landedPiece{family: "wire", name: req.WireName}
	case analyze.EditPage:
		return &landedPiece{
			family:   "page",
			file:     rel,
			pageKind: req.PageKind,
			name:     req.PageName,
		}
	}
	return nil
}

// noteSpineSnapshot records the landed piece in rebuildable
// form for the follow-ups. Make-and-call carries no
// snapshot (rebuilding only its concept half would drop the
// call), but its SECOND file is noted as the session's other
// file: the two files of that one landing are exactly the
// pair "the other file" speaks of.
func noteSpineSnapshot(req analyze.EditRequest, rel string) {
	workCtx.landed = landedSnapshotFor(req, rel)
	if req.Kind == analyze.EditMakeCall && req.MakeCallTarget != "" && req.MakeCallTarget != rel {
		workCtx.prevFile = filepath.ToSlash(req.MakeCallTarget)
		workCtx.prevHow = "edited"
	}
}

// noteCreateLanded records a plain file creation (the create
// family's own Done point, beside its noteLandedFile call) in
// rebuildable form: the follow-up's new instance is another
// creation of the same kind — a plain file, or the webserver
// starter when the creation carried one.
func noteCreateLanded(rel, kind string) {
	rel = filepath.ToSlash(rel)
	workCtx.landed = &landedPiece{
		family:     "create",
		file:       rel,
		folder:     slashDir(rel),
		createKind: kind,
	}
}

// notePlanLanded records a landed plan from one of the
// flows that run their own transaction (rename, uses,
// delegate, error fix): the plan's first file stands as
// the file the work touched, edited by the named flow.
func notePlanLanded(root string, plan *analyze.EditPlan, piece string) {
	if plan == nil || len(plan.Files) == 0 {
		return
	}
	noteLandedFile(root, plan.Files[0].Rel, "edited", piece, "")
	// These flows (rename, delegate, uses, error fix) carry
	// no rebuildable snapshot: a follow-up after one answers
	// honestly rather than guessing a sentence.
	workCtx.landed = nil
}

// otherFileCandidate returns the session's other file: the
// file the landing before the last one touched, when the
// session holds exactly two distinct files and both are
// still on disk. "the other file" (followup.go, vocab.go)
// has a referent only in that state — one file in play, or
// a file gone from disk, and the phrase resolves to
// nothing, never to a guess.
func otherFileCandidate() (string, bool) {
	if workCtx.root == "" || workCtx.prevFile == "" || workCtx.prevFile == workCtx.lastFile {
		return "", false
	}
	for _, rel := range []string{workCtx.lastFile, workCtx.prevFile} {
		st, err := os.Stat(filepath.Join(workCtx.root, filepath.FromSlash(rel)))
		if err != nil || st.IsDir() {
			return "", false
		}
	}
	return workCtx.prevFile, true
}

// lastFileReason is the why of a reading that resolved to
// the tracked file, in the note's voice.
func (c workContext) lastFileReason() string {
	if c.lastHow == "created" {
		return "the file you just created"
	}
	return "the file you just edited"
}

// ctxCandidate is one file the context offers for a bare
// name, with the reason it fits.
type ctxCandidate struct {
	path   string // slash-relative resolved path
	reason string
}

// resolveContextTarget settles a concept ask's bare target —
// one file name, no folder — against the working context.
// It returns exactly one of:
//
//   - resolved != "": the target read as resolved, with the
//     note line that says so (the caller prints it before
//     its plan or refusal);
//   - cands != nil: two files fit and neither may be
//     picked — the caller asks which (clarify.go) and
//     writes nothing until the user answers;
//   - both empty: the context offers nothing for this
//     target and the ask keeps today's behavior untouched.
//
// A file of the same name at the project root never WINS
// silently — root-relative is what the bare name already
// means today — but when a context file also fits, the root
// namesake is a second fit and the answer is a question.
func resolveContextTarget(typed, line string, rootAbs string) (resolved, note string, cands []ctxCandidate) {
	if typed == "" || strings.Contains(typed, "/") ||
		!strings.HasSuffix(strings.ToLower(typed), ".go") {
		return "", "", nil
	}
	if !workContextLive(rootAbs) {
		return "", "", nil
	}
	seen := map[string]bool{}
	add := func(path, reason string) {
		if path == "" || seen[path] {
			return
		}
		if st, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(path))); err != nil || st.IsDir() {
			return
		}
		seen[path] = true
		cands = append(cands, ctxCandidate{path: path, reason: reason})
	}
	if workCtx.lastFile != "" && filepath.Base(workCtx.lastFile) == typed {
		add(workCtx.lastFile, workCtx.lastFileReason())
	}
	if workCtx.folder != "" {
		add(workCtx.folder+"/"+typed, "a file in the folder you're working in")
	}
	if len(cands) == 0 {
		return "", "", nil
	}
	// The root namesake: a second fit, never a silent winner.
	rootFit := false
	if st, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(typed))); err == nil && !st.IsDir() && !seen[typed] {
		rootFit = true
	}
	if rootFit || len(cands) > 1 {
		all := append([]ctxCandidate{}, cands...)
		if rootFit {
			all = append(all, ctxCandidate{path: typed, reason: "a file at the project root"})
		}
		return "", "", all
	}
	c := cands[0]
	return c.path, fmt.Sprintf("Reading %q as %s — %s.", typed, c.path, c.reason), nil
}

// spliceTypedTarget rewrites the ask's typed target (its
// last occurrence in the line) to a full resolved path —
// the rewrite a clarifying answer completes the ask with:
// "add a print line in main to db.go" answered with the
// tracked file becomes the same line with jim/db.go in the
// tail, which the concept grammar then reads exactly as it
// reads the spelled-out line. False when the typed target
// doesn't appear in the line at all (the caller then keeps
// its standing answer rather than inventing a rewrite).
func spliceTypedTarget(line, typed, replacement string) (string, bool) {
	idx := strings.LastIndex(line, typed)
	if idx < 0 {
		return "", false
	}
	return line[:idx] + replacement + line[idx+len(typed):], true
}

// rePronounTail matches the three pronoun tails at the very
// end of a line: "to it", "to that file", "to the same file"
// (the article optional on the last, the preposition any of
// the concept grammar's target openers).
var rePronounTail = regexp.MustCompile(`(?i)(^|\s)(to|into|in)\s+((?:the\s+)?(?:that\s+file|same\s+file)|it)\s*$`)

// resolvePronounTail rewrites a concept line's pronoun tail
// to the tracked file, when the context holds one: "add a
// database connection function named Connect to it" becomes
// the same line with the full path in the tail, so the
// concept grammar — untouched — parses it exactly as it
// parses the spelled-out line. The second return is the
// reading's note line ("" when nothing was rewritten: no
// pronoun tail, no live context, or no tracked file — in
// every such case the line keeps today's behavior verbatim).
func resolvePronounTail(line string, rootAbs string) (string, string) {
	if !workContextLive(rootAbs) || workCtx.lastFile == "" {
		return line, ""
	}
	t := strings.TrimSpace(line)
	t = strings.TrimSuffix(t, ".")
	t = strings.TrimSuffix(t, "?")
	t = strings.TrimSpace(t)
	m := rePronounTail.FindStringSubmatchIndex(t)
	if m == nil {
		return line, ""
	}
	prep := t[m[4]:m[5]]
	phrase := t[m[6]:m[7]]
	rewritten := t[:m[0]] + mSpace(t, m) + prep + " " + workCtx.lastFile
	note := fmt.Sprintf("Reading %q as %s — %s.", phrase, workCtx.lastFile, workCtx.lastFileReason())
	return rewritten, note
}

// mSpace preserves the single separator the tail match
// consumed before the preposition (group 1): a space mid-line,
// nothing at the line's start.
func mSpace(t string, m []int) string {
	if m[2] < 0 {
		return ""
	}
	return t[m[2]:m[3]]
}

// resolveCreateFolder reads the add-verb form's bare file
// name — no folder asked, none in the name — against the
// working folder. (The add form only: "add the file …" is
// the follow-up spelling, while create|make bare names
// keep the create family's shipped root anchoring — the
// bare-file package rules pin a bare "create store.go" to
// the root, in a session or out of one.) The folder
// context is singular (there is exactly one folder the
// session is working in), so this reading is never
// ambiguous; a root namesake does not compete, because a
// creation never targets an existing file — if the name
// is taken in the working folder, the create flow's own
// already-there refusal says so with the full path.
// Returns the folder to create in and the note line, or
// "" when the context offers nothing (no live context, or
// the work is at the project root — today's behavior
// either way).
func resolveCreateFolder(file string, rootAbs string) (folder, note string) {
	if file == "" || strings.Contains(file, "/") || !workContextLive(rootAbs) || workCtx.folder == "" {
		return "", ""
	}
	resolved := workCtx.folder + "/" + file
	return workCtx.folder, fmt.Sprintf("Reading %q as %s — the folder you're working in.", file, resolved)
}
