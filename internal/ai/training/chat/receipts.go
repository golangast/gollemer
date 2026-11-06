package chat

// Durable landing receipts and cross-session work-context
// resume (Week 4 transfer package, 2026-11-05).
//
// A landing used to leave only two traces: the session's
// transient work context (sessionctx.go) and a prose summary
// in the long-term memory store. Both were the wrong shape
// for the question "what exactly did the chat change here,
// and can a new session pick the work back up?" — the first
// dies with the process, the second is prose, not facts.
//
// Receipts are the verified landing record, kept as data:
// one receipt per successful landing, written at the same
// hooks that note the session context, sourced ONLY from
// the executed plan and the landing's own verification
// record — never from the user's sentence and never from a
// model. A refusal, a declined gate, and a rolled-back edit
// return before those hooks, so they write no receipt.
//
// The store is a sibling of the taught and vocabulary
// stores: ~/.gollemer_receipts.json, deliberately NOT a
// section of the memory store, so forgetting preferences,
// names, taught pieces, or project memory can never erase
// build history (and forgetting build history touches
// nothing else). 0600, written atomically, keyed per
// project root. History is capped per project (see
// landingReceiptCap): the newest receipts are the resume
// facts and the honest answer to "what have you built
// here"; older ones drop off the end.
//
// Resume is context, never execution: when a session sets
// its project, the newest receipts rebuild the work
// context — last file and how it got there, the file
// before it, the working folder and its package, the last
// piece and its rebuildable snapshot where its family has
// one — so follow-ups and the session summary work across
// a restart. Restoring writes no project file and replays
// no effect. A receipt whose file is no longer on disk is
// dropped from the restored context, out loud in the
// continuity line, rather than resolved against a ghost.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// landingReceiptCap bounds one project's stored history:
// the newest 50 landings are kept; older receipts are
// dropped when a new one is appended.
const landingReceiptCap = 50

// The receipt's verification words: a build ran against
// the landing and passed its check, or the landing was
// verified by re-reading the project (no go.mod to build
// against, or a creation whose check is existence/parse).
const (
	receiptVerificationBuilt   = "built-and-passed"
	receiptVerificationReRead  = "re-read-only"
	receiptVerificationBuiltW  = "built and passed"
	receiptVerificationReReadW = "checked by re-reading the project"
)

// landingReceiptFile is one file an executed plan touched,
// with how it got there: created by the plan (it had no
// original bytes) or edited in place.
type landingReceiptFile struct {
	Path   string `json:"path"`
	Change string `json:"change"` // "created" | "edited"
}

// landingReceiptSnapshot is a landed piece in rebuildable
// form — landedPiece (sessionctx.go) made serializable, so
// a follow-up in a later session rebuilds the piece's
// canonical sentence exactly as an in-session follow-up
// does. Only the families whose sentences rebuild exactly
// carry one; every other landing leaves it nil and a
// follow-up after it answers honestly, as today.
type landingReceiptSnapshot struct {
	Family     string `json:"family"`
	File       string `json:"file,omitempty"`
	ConceptID  string `json:"concept_id,omitempty"`
	Name       string `json:"name,omitempty"`
	Text       string `json:"text,omitempty"`
	InFunc     string `json:"in_func,omitempty"`
	Like       string `json:"like,omitempty"`
	KindPhrase string `json:"kind_phrase,omitempty"`
	HTML       string `json:"html,omitempty"`
	PageKind   string `json:"page_kind,omitempty"`
	Folder     string `json:"folder,omitempty"`
	CreateKind string `json:"create_kind,omitempty"`
}

// landingReceipt is one verified landing, as data.
type landingReceipt struct {
	At           time.Time               `json:"at"`
	Kind         string                  `json:"kind"` // the plan's kind ("concept", "wire_handler", "create_file", a fix class, ...)
	Name         string                  `json:"name,omitempty"`
	Piece        string                  `json:"piece,omitempty"` // plain-words kind, the session context's own words
	PrimaryFile  string                  `json:"primary_file,omitempty"`
	Folder       string                  `json:"folder,omitempty"` // folder landings: the folder created / the working folder a file landing implies is derived, never stored twice
	Files        []landingReceiptFile    `json:"files,omitempty"`
	Route        string                  `json:"route,omitempty"` // wiring (and page) landings only
	Verification string                  `json:"verification"`    // receiptVerificationBuilt | receiptVerificationReRead
	Verify       string                  `json:"verify"`          // the plan's VerifyDesc (create family: its created line, from its plan)
	Snapshot     *landingReceiptSnapshot `json:"snapshot,omitempty"`
	OtherFile    string                  `json:"other_file,omitempty"` // make-and-call: the landing's second file, the session's "other file"
}

type landingReceiptData struct {
	Version  int                         `json:"version"`
	Projects map[string][]landingReceipt `json:"projects,omitempty"`
}

func emptyLandingReceiptData() landingReceiptData {
	return landingReceiptData{Version: 1, Projects: map[string][]landingReceipt{}}
}

func normalizeLandingReceiptData(d landingReceiptData) landingReceiptData {
	if d.Version == 0 {
		d.Version = 1
	}
	if d.Projects == nil {
		d.Projects = map[string][]landingReceipt{}
	}
	return d
}

// receiptStore is the storage seam, the sibling stores'
// shape: load one value or save one value.
type receiptStore interface {
	Load() (landingReceiptData, error)
	Save(landingReceiptData) error
}

var receiptStorePathOverride string

func openReceiptStore() receiptStore { return jsonReceiptStore{path: receiptStorePath()} }

func receiptStorePath() string {
	if receiptStorePathOverride != "" {
		return receiptStorePathOverride
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ".gollemer_receipts.json"
	}
	return filepath.Join(home, ".gollemer_receipts.json")
}

type jsonReceiptStore struct{ path string }

func (s jsonReceiptStore) Load() (landingReceiptData, error) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return emptyLandingReceiptData(), nil
	}
	if err != nil {
		return emptyLandingReceiptData(), err
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return emptyLandingReceiptData(), nil
	}
	var out landingReceiptData
	if err := json.Unmarshal(data, &out); err != nil {
		return emptyLandingReceiptData(), err
	}
	return normalizeLandingReceiptData(out), nil
}

func (s jsonReceiptStore) Save(d landingReceiptData) error {
	d = normalizeLandingReceiptData(d)
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".gollemer-receipts-*.tmp")
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

func loadLandingReceipts() (landingReceiptData, error) { return openReceiptStore().Load() }

func saveLandingReceipts(d landingReceiptData) error { return openReceiptStore().Save(d) }

// landingReceiptsForRoot returns one project's receipts,
// oldest first, as stored. Root comparison canonicalizes
// both sides the memory store's way, so a project reached
// by two spellings finds its own history.
func landingReceiptsForRoot(root string) ([]landingReceipt, error) {
	root = canonicalMemoryRoot(root)
	if root == "" {
		return nil, nil
	}
	d, err := loadLandingReceipts()
	if err != nil {
		return nil, err
	}
	for stored, recs := range d.Projects {
		if canonicalMemoryRoot(stored) == root {
			return append([]landingReceipt{}, recs...), nil
		}
	}
	return nil, nil
}

// appendLandingReceipt appends one receipt to a project's
// history and enforces the cap (oldest dropped).
func appendLandingReceipt(root string, rec landingReceipt) error {
	root = canonicalMemoryRoot(root)
	if root == "" || rec.Kind == "" {
		return nil
	}
	if rec.At.IsZero() {
		rec.At = time.Now()
	}
	rec.PrimaryFile = filepath.ToSlash(rec.PrimaryFile)
	rec.Folder = filepath.ToSlash(rec.Folder)
	rec.OtherFile = filepath.ToSlash(rec.OtherFile)
	for i := range rec.Files {
		rec.Files[i].Path = filepath.ToSlash(rec.Files[i].Path)
	}
	d, err := loadLandingReceipts()
	if err != nil {
		return err
	}
	key := root
	for stored := range d.Projects {
		if canonicalMemoryRoot(stored) == root {
			key = stored
			break
		}
	}
	recs := append(d.Projects[key], rec)
	if len(recs) > landingReceiptCap {
		recs = append([]landingReceipt{}, recs[len(recs)-landingReceiptCap:]...)
	}
	d.Projects[key] = recs
	return saveLandingReceipts(d)
}

// recordLandingReceipt is the landing hooks' write: a
// receipt is auxiliary to the landing itself (the Done
// line's contract is the build verification, already
// passed), so a store failure never reaches the user's
// landing — and never prints over it.
func recordLandingReceipt(root string, rec landingReceipt) {
	_ = appendLandingReceipt(root, rec)
}

// clearLandingReceipts removes one project's receipts and
// reports how many there were. Other projects are not
// touched, and no other store is opened.
func clearLandingReceipts(root string) (int, error) {
	root = canonicalMemoryRoot(root)
	if root == "" {
		return 0, nil
	}
	d, err := loadLandingReceipts()
	if err != nil {
		return 0, err
	}
	for stored := range d.Projects {
		if canonicalMemoryRoot(stored) == root {
			n := len(d.Projects[stored])
			delete(d.Projects, stored)
			if err := saveLandingReceipts(d); err != nil {
				return 0, err
			}
			return n, nil
		}
	}
	return 0, nil
}

// ----- building receipts from landing records ------------------------------

func receiptVerification(built bool) string {
	if built {
		return receiptVerificationBuilt
	}
	return receiptVerificationReRead
}

func receiptFilesFromPlan(plan *analyze.EditPlan) []landingReceiptFile {
	if plan == nil {
		return nil
	}
	out := make([]landingReceiptFile, 0, len(plan.Files))
	for _, f := range plan.Files {
		change := "edited"
		if len(f.Before) == 0 {
			change = "created"
		}
		out = append(out, landingReceiptFile{Path: filepath.ToSlash(f.Rel), Change: change})
	}
	return out
}

func receiptSnapshotFromLanded(lp *landedPiece) *landingReceiptSnapshot {
	if lp == nil {
		return nil
	}
	return &landingReceiptSnapshot{
		Family:     lp.family,
		File:       lp.file,
		ConceptID:  lp.conceptID,
		Name:       lp.name,
		Text:       lp.text,
		InFunc:     lp.inFunc,
		Like:       lp.like,
		KindPhrase: lp.kindPhrase,
		HTML:       lp.html,
		PageKind:   lp.pageKind,
		Folder:     lp.folder,
		CreateKind: lp.createKind,
	}
}

func landedFromReceiptSnapshot(s *landingReceiptSnapshot) *landedPiece {
	if s == nil || s.Family == "" {
		return nil
	}
	return &landedPiece{
		family:     s.Family,
		file:       s.File,
		conceptID:  s.ConceptID,
		name:       s.Name,
		text:       s.Text,
		inFunc:     s.InFunc,
		like:       s.Like,
		kindPhrase: s.KindPhrase,
		html:       s.HTML,
		pageKind:   s.PageKind,
		folder:     s.Folder,
		createKind: s.CreateKind,
	}
}

// spineReceiptName reads the name a spine landing carries,
// from the request the spine executed — the landing record,
// never the sentence that asked for it.
func spineReceiptName(req analyze.EditRequest) string {
	switch req.Kind {
	case analyze.EditConcept:
		return req.ConceptName
	case analyze.EditTaughtDecl:
		return req.TaughtName
	case analyze.EditProposedDecl:
		return req.ProposedName
	case analyze.EditWireHandler:
		return req.WireName
	case analyze.EditPage:
		return req.PageName
	case analyze.EditAddFunc:
		return req.NewName
	case analyze.EditAddField:
		return req.FieldName
	case analyze.EditRename:
		return req.RenameTo
	case analyze.EditDelegate:
		return req.WrapperName
	case analyze.EditUse:
		return req.UseName
	case analyze.EditMakeCall:
		return req.ConceptName
	case analyze.EditInstantiate:
		return req.InstVar
	case analyze.EditAddParam:
		return req.SigName
	}
	return ""
}

// spineReceiptPiece is noteSpineLanded's own piece rule:
// the book's canonical name for a concept piece, the
// taught phrase for a taught piece, the request kind
// otherwise.
func spineReceiptPiece(req analyze.EditRequest) string {
	if req.Kind == analyze.EditConcept && req.Concept != nil {
		return req.Concept.Canonical()
	}
	if req.Kind == analyze.EditTaughtDecl {
		return req.TaughtKindPhrase
	}
	return string(req.Kind)
}

// recordSpineReceipt writes the edit spine's receipt, at
// the spine's own landing hook (edits.go, beside
// noteSpineLanded). The wiring route and name come from
// the one registration the plan added — diffed against
// the pre-write snapshot, the confirmation's own source.
func recordSpineReceipt(req analyze.EditRequest, plan *analyze.EditPlan, root string, built bool, beforeRegs []analyze.WireRegistration, fresh *analyze.Project) {
	if plan == nil {
		return
	}
	rec := landingReceipt{
		Kind:         string(req.Kind),
		Name:         spineReceiptName(req),
		Piece:        spineReceiptPiece(req),
		PrimaryFile:  landingPrimaryRel(req, plan),
		Files:        receiptFilesFromPlan(plan),
		Verification: receiptVerification(built),
		Verify:       plan.VerifyDesc,
		Snapshot:     receiptSnapshotFromLanded(landedSnapshotFor(req, landingPrimaryRel(req, plan))),
	}
	switch req.Kind {
	case analyze.EditWireHandler:
		if reg, ok := newPlanRegistration(beforeRegs, fresh, plan); ok {
			rec.Route = reg.Route
			if rec.Name == "" {
				rec.Name = reg.Name
			}
		}
	case analyze.EditPage:
		rec.Route = req.PageRoute
	case analyze.EditMakeCall:
		if req.MakeCallTarget != "" && req.MakeCallTarget != rec.PrimaryFile {
			rec.OtherFile = filepath.ToSlash(req.MakeCallTarget)
		}
	}
	recordLandingReceipt(root, rec)
}

// planReceiptName settles a plan-flow landing's name: the
// caller's own fact when it has one, else the executed
// plan's verified clause (a delegate/use plan names its
// new function there — the planner derived it, and the
// clause is the record of that derivation).
func planReceiptName(kind, provided string, plan *analyze.EditPlan) string {
	if provided != "" {
		return provided
	}
	if plan == nil {
		return ""
	}
	if kind == analyze.EditDelegate || kind == analyze.EditUse {
		if i := strings.Index(plan.VerifyDesc, " now exists in "); i > 0 {
			return plan.VerifyDesc[:i]
		}
	}
	return ""
}

// recordPlanReceipt writes a receipt for one of the flows
// that run their own transaction beside notePlanLanded
// (delegate, uses, rename, error fix).
func recordPlanReceipt(root string, plan *analyze.EditPlan, kind, name, piece string, built bool) {
	if plan == nil {
		return
	}
	primary := ""
	if len(plan.Files) > 0 {
		primary = plan.Files[0].Rel
	}
	recordLandingReceipt(root, landingReceipt{
		Kind:         kind,
		Name:         planReceiptName(kind, name, plan),
		Piece:        piece,
		PrimaryFile:  primary,
		Files:        receiptFilesFromPlan(plan),
		Verification: receiptVerification(built),
		Verify:       plan.VerifyDesc,
	})
}

// recordCreateFileReceipt writes the create family's file
// receipt, beside its noteLandedFile / noteCreateLanded
// calls. A .go file's check was a build (or a compile of
// the file itself); a bare-clause parse check and a
// non-.go file's existence check are re-reads, and the
// receipt says which.
func recordCreateFileReceipt(root string, plan *createPlan, req createReq) {
	if plan == nil || plan.file == nil {
		return
	}
	// The verification create.go actually ran, from the
	// plan: a Go file was compile-checked (checkGoFile)
	// UNLESS it is a bare package-main clause, which gets
	// the parse-only check; a non-Go file is verified by
	// existence alone.
	verification := receiptVerificationReRead
	if plan.isGo && !(plan.bareClause && plan.pkg == "main") {
		verification = receiptVerificationBuilt
	}
	recordLandingReceipt(root, landingReceipt{
		Kind:         "create_file",
		Piece:        "file",
		PrimaryFile:  plan.file.rel,
		Files:        []landingReceiptFile{{Path: plan.file.rel, Change: "created"}},
		Verification: verification,
		Verify:       "created " + plan.file.rel,
		Snapshot: receiptSnapshotFromLanded(&landedPiece{
			family:     "create",
			file:       plan.file.rel,
			folder:     slashDir(plan.file.rel),
			createKind: req.kind,
		}),
	})
}

// recordCreateFolderReceipt writes the create family's
// folder receipt, beside its noteLandedFolder call: a
// folder landing touches no file, so Files is empty and
// the folder itself is the record.
func recordCreateFolderReceipt(root string, item createItem) {
	recordLandingReceipt(root, landingReceipt{
		Kind:         "create_folder",
		Piece:        "folder",
		Folder:       item.rel,
		Verification: receiptVerificationReRead,
		Verify:       "created folder " + item.rel,
	})
}

// ----- resume ----------------------------------------------------------------

// receiptResumeState is the pending continuity disclosure
// for a context restored from receipts: what was restored,
// held until beginMemorySession speaks it, once per root
// per process (receiptResumeAnnounced).
type receiptResumeState struct {
	root string
	text string
}

var (
	receiptResumePending   receiptResumeState
	receiptResumeAnnounced = map[string]bool{}
)

// restoreWorkContextFromReceipts rebuilds the session's
// work context from a project's receipts. It runs when a
// session sets its project and holds no live context for
// it (project_freshness.go's setCurrentProject). It never
// writes a project file and never replays an effect: the
// receipts are replayed as CONTEXT TRANSITIONS only, in
// landing order, through the same state the note hooks
// maintain — and any referent that is no longer on disk
// is skipped and named in the disclosure, so nothing
// downstream can resolve against a ghost.
func restoreWorkContextFromReceipts(rootAbs string) {
	root := canonicalMemoryRoot(rootAbs)
	if root == "" || workContextLive(root) {
		return
	}
	recs, err := landingReceiptsForRoot(root)
	if err != nil || len(recs) == 0 {
		return
	}
	restored := workContext{root: root}
	var stale []string
	var builtDescs []string
	noteBuilt := func(desc string) {
		if desc != "" {
			builtDescs = appendUniqueString(builtDescs, desc)
		}
	}
	for _, rec := range recs {
		if rec.Kind == "create_folder" {
			folder := filepath.ToSlash(rec.Folder)
			if folder == "" {
				continue
			}
			st, serr := os.Stat(filepath.Join(root, filepath.FromSlash(folder)))
			if serr != nil || !st.IsDir() {
				stale = appendUniqueString(stale, folder+"/")
				continue
			}
			restored.folder = folder
			restored.lastKind = "folder"
			restored.pkg = folderPackageOf(root, folder)
			noteBuilt("folder " + folder)
			continue
		}
		primary := filepath.ToSlash(rec.PrimaryFile)
		if primary == "" {
			continue
		}
		st, serr := os.Stat(filepath.Join(root, filepath.FromSlash(primary)))
		if serr != nil || st.IsDir() {
			stale = appendUniqueString(stale, primary)
			continue
		}
		how := "edited"
		for _, f := range rec.Files {
			if f.Path == primary && f.Change != "" {
				how = f.Change
			}
		}
		if restored.lastFile != "" && restored.lastFile != primary {
			restored.prevFile = restored.lastFile
			restored.prevHow = restored.lastHow
		}
		restored.lastFile = primary
		restored.lastHow = how
		restored.lastKind = "file"
		restored.folder = slashDir(primary)
		restored.pkg = folderPackageOf(root, restored.folder)
		restored.piece = rec.Piece
		restored.pieceName = rec.Name
		restored.landed = landedFromReceiptSnapshot(rec.Snapshot)
		if rec.OtherFile != "" && rec.OtherFile != primary {
			if ost, oerr := os.Stat(filepath.Join(root, filepath.FromSlash(rec.OtherFile))); oerr == nil && !ost.IsDir() {
				restored.prevFile = rec.OtherFile
				restored.prevHow = "edited"
				for _, f := range rec.Files {
					if f.Path == rec.OtherFile && f.Change != "" {
						restored.prevHow = f.Change
					}
				}
			}
		}
		desc := rec.Piece
		if rec.Name != "" {
			desc += " " + rec.Name
		}
		if desc != "" {
			desc += " in " + primary
		} else {
			desc = primary
		}
		noteBuilt(desc)
	}
	if restored.lastFile == "" && restored.lastKind == "" {
		// Everything the receipts named is gone: restore
		// nothing, but still disclose the drop.
		if len(stale) > 0 {
			receiptResumePending = receiptResumeState{
				root: root,
				text: "I found build receipts here, but dropped " + strings.Join(stale, ", ") +
					" from the restored context — " + stalePronoun(len(stale)) + " not in the project anymore, and I won't resolve against what's gone.",
			}
		}
		return
	}
	workCtx = restored
	// The short-term built summary rides the restore, the
	// way noteLandedFile feeds it in a live session.
	for _, desc := range builtDescs {
		noteSessionBuilt(root, desc)
	}
	receiptResumePending = receiptResumeState{root: root, text: receiptResumeText(restored, stale)}
}

func stalePronoun(n int) string {
	if n == 1 {
		return "it's"
	}
	return "they're"
}

// receiptResumeText renders the disclosure for a restored
// context: what the session picked back up, and what it
// dropped because the disk no longer holds it.
func receiptResumeText(c workContext, stale []string) string {
	var main string
	switch {
	case c.lastKind == "folder" && c.lastFile != "":
		main = fmt.Sprintf("the working folder is %s, and the last piece was %s in %s", c.folder, receiptPieceDesc(c.piece, c.pieceName), c.lastFile)
	case c.lastKind == "folder":
		main = fmt.Sprintf("the working folder is %s", c.folder)
	case c.piece != "":
		main = fmt.Sprintf("the last piece was %s in %s (%s)", receiptPieceDesc(c.piece, c.pieceName), c.lastFile, c.lastHow)
	default:
		main = fmt.Sprintf("the last file worked on was %s (%s)", c.lastFile, c.lastHow)
	}
	if c.prevFile != "" && c.lastKind == "file" {
		main += fmt.Sprintf(", with %s before it", c.prevFile)
	}
	text := "Resumed from the build receipts: " + main + "."
	if len(stale) > 0 {
		text += fmt.Sprintf(" I dropped %s from that context — %s not in the project anymore.", strings.Join(stale, ", "), stalePronoun(len(stale)))
	}
	return text
}

func receiptPieceDesc(piece, name string) string {
	if name == "" {
		return "the " + piece
	}
	return "the " + piece + " " + name
}

// receiptResumeLine speaks the pending restore disclosure,
// once per root per process, for beginMemorySession to
// fold into the continuity line.
func receiptResumeLine(root string) string {
	root = canonicalMemoryRoot(root)
	if root == "" || receiptResumeAnnounced[root] {
		return ""
	}
	if receiptResumePending.root != root || receiptResumePending.text == "" {
		return ""
	}
	receiptResumeAnnounced[root] = true
	return receiptResumePending.text
}

// ----- query -----------------------------------------------------------------

func receiptVerificationWords(v string) string {
	if v == receiptVerificationBuilt {
		return receiptVerificationBuiltW
	}
	return receiptVerificationReReadW
}

// renderLandingReceipts answers "what have you built
// here" from the receipts alone: the exact list, most
// recent first, with dates. No receipts is an honest
// nothing-yet, in this claim's own words.
func renderLandingReceipts(root string) (string, error) {
	recs, err := landingReceiptsForRoot(root)
	if err != nil {
		return "", err
	}
	if len(recs) == 0 {
		return "Nothing yet — no landing in this project has left a receipt, so I haven't built or changed anything here that I can show you.", nil
	}
	var b strings.Builder
	b.WriteString("Here's what I've built or changed in this project, from the landing receipts (newest first):")
	for i := len(recs) - 1; i >= 0; i-- {
		r := recs[i]
		what := r.Verify
		if what == "" {
			what = r.Kind
		}
		fmt.Fprintf(&b, "\n%d. %s — %s.", len(recs)-i, r.At.Format("Jan 2, 2006 at 3:04 PM"), what)
		if len(r.Files) > 0 {
			parts := make([]string, 0, len(r.Files))
			for _, f := range r.Files {
				parts = append(parts, fmt.Sprintf("%s (%s)", f.Path, f.Change))
			}
			fmt.Fprintf(&b, " Files: %s.", strings.Join(parts, ", "))
		} else if r.Folder != "" {
			fmt.Fprintf(&b, " Folder: %s/.", r.Folder)
		}
		if r.Route != "" {
			fmt.Fprintf(&b, " Route: %s.", r.Route)
		}
		fmt.Fprintf(&b, " Verification: %s.", receiptVerificationWords(r.Verification))
	}
	return b.String(), nil
}

// tryReceiptForget claims the build-history forget in the
// unified loop, seated directly ahead of the teach seat:
// the taught store's generic "forget the <kind>" grammar
// would otherwise read "build history" as a kind phrase
// and answer for a store the user never meant (caught by
// the live smoke, 2026-11-05 — the state-answer seat sits
// after the teach seat, so the line never reached it). The
// shape is the one exact anchored line (receiptForgetRe,
// memory_state.go), so nothing else can be caught here;
// the same answer also stays seated in the state-answer
// family for contexts that reach it directly.
func tryReceiptForget(line string) bool {
	if !receiptForgetRe.MatchString(line) {
		return false
	}
	root := memoryCurrentRoot()
	if root == "" {
		root = workCtx.root
	}
	fmt.Printf("%s %s\n", cTag("gollemer [talk]>"), forgetLandingReceiptsText(root))
	recordBrain("talk", true, "build history forgotten")
	return true
}

// forgetLandingReceiptsText clears one project's receipts
// and echoes exactly what was cleared.
func forgetLandingReceiptsText(root string) string {
	n, err := clearLandingReceipts(root)
	if err != nil {
		return "I couldn't update the build-history store (" + receiptStorePath() + "): " + err.Error()
	}
	if n == 0 {
		return "There was no build history for this project to forget — no landing here has left a receipt."
	}
	word := "receipts"
	if n == 1 {
		word = "receipt"
	}
	return fmt.Sprintf("Forgotten: this project's build history — %d landing %s for %s. Other projects' receipts, and everything else I keep, stay as they are.", n, word, canonicalMemoryRoot(root))
}
