package chat

// Rename everywhere: the chat face of the Composer's second write
// shape. "rename X to Y" is parsed HERE, deterministically, beside
// the assembled edits — a sentence that would rewrite the user's
// files is never left for the intent classifier or a neural model.
//
// The analyzer resolves the name (analyze.RenameCandidates); a
// unique target becomes a plan from the project's type information
// (analyze.PlanRenameTarget) and rides the same gate as every other
// edit: exact diff, [y/n], apply, go build, re-analyze, verify —
// any failure restores the original bytes. An ambiguous name prints
// a numbered pick list and arms pendingRenamePick, following the
// pendingFuncPick conventions: one outstanding list at a time, a
// bare number answers it, any other line clears it, and the pick is
// re-validated against the current project before it plans.
//
// Accepted forms (trailing "." or "?" optional):
//
//	rename <X> to <Y>
//	rename the <X> to <Y>
//	rename [the] function|method|type|variable|constant|field <X> to <Y>
//	… any of those, plus "in <package-or-file>" at the end

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var reRename = regexp.MustCompile(`(?i)^rename (?:the )?(?:(function|func|method|type|struct|variable|var|constant|const|field) )?([A-Za-z_][\w.]*) to ([A-Za-z_]\w*)(?: in ([A-Za-z_][\w./-]*))?$`)

// parseRenameSentence turns one chat line into a rename request.
// The second return is false for anything that is not a rename
// request — including questions about renaming ("how do I rename…"),
// which belong to other brains.
func parseRenameSentence(line string) (analyze.EditRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	m := reRename.FindStringSubmatch(s)
	if m == nil {
		return analyze.EditRequest{}, false
	}
	return analyze.EditRequest{
		Kind:        analyze.EditRename,
		RenameKind:  m[1],
		RenameFrom:  m[2],
		RenameTo:    m[3],
		RenameScope: m[4],
	}, true
}

// pendingRename is the one outstanding rename disambiguation: the
// candidates in printed order, the new name, and the request that
// produced them (for re-validation when the pick is answered).
type pendingRename struct {
	cands   []analyze.RenameTarget
	newName string
	req     analyze.EditRequest
}

var pendingRenamePick *pendingRename

// tryRenameEdit is the chat-loop entry: it claims rename sentences
// (and numeric answers to a rename pick list) and runs the gate
// flow on the chat's line reader.
func tryRenameEdit(line string, r *LineReader) bool {
	return runRenameEdit(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runRenameEdit carries one rename from sentence (or pick) to a
// verified change, an honest refusal, or a clean "not applied". It
// returns true when the line was rename business, claimed or not.
func runRenameEdit(line string, ask editAsk) bool {
	if pendingRenamePick != nil {
		if handled := resolveRenamePick(line, ask); handled {
			return true
		}
	}
	req, ok := parseRenameSentence(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	hadProject := lastAnalyzeProject != nil
	if !ensureFreshProject() {
		if !hadProject {
			fmt.Printf("%s I need a project to rename in — analyze one first (or start me inside one).\n", tag)
		}
		return true
	}
	proj := lastAnalyzeProject
	cands := proj.RenameCandidates(req)
	switch {
	case len(cands) == 0:
		_, refusal := proj.PlanEdit(req) // reuse the refusal wording
		if refusal == "" {
			refusal = "I couldn't find that name in this project."
		}
		fmt.Printf("%s Not renaming: %s\n", tag, refusal)
		recordBrain("composer", false, "rename refused: not found")
		return true
	case len(cands) > 1:
		fmt.Printf("%s %q could mean more than one thing here — which one should become %s?\n", tag, req.RenameFrom, req.RenameTo)
		for i, c := range cands {
			fmt.Printf(" %d. the %s %s (%s:%d)\n", i+1, c.KindWord(), c.Display(), proj.LinkPath(c.File), c.Line)
		}
		fmt.Println("Reply with the number, or say it again with the package or receiver.")
		pendingRenamePick = &pendingRename{cands: cands, newName: req.RenameTo, req: req}
		setFuncPick(nil) // the freshest list owns numeric replies
		recordBrain("composer", true, "rename pick offered")
		return true
	}
	gateRenamePlan(proj, cands[0], req.RenameTo, ask)
	return true
}

// resolveRenamePick gives a pending rename list first claim on a
// numeric reply — the same contract as resolveFuncPick: a line that
// is no answer clears the list and is processed normally (handled
// is false), and a pick is re-validated against the current project
// before it plans.
func resolveRenamePick(line string, ask editAsk) (handled bool) {
	pk := pendingRenamePick
	if pk == nil {
		return false
	}
	n, err := strconv.Atoi(strings.TrimSpace(line))
	if err != nil || n < 1 || n > len(pk.cands) {
		pendingRenamePick = nil
		return false
	}
	pendingRenamePick = nil
	tag := cTag("gollemer [composer]>")
	hadProject := lastAnalyzeProject != nil
	if !ensureFreshProject() {
		if !hadProject {
			fmt.Printf("%s I need the project that list came from — analyze one first.\n", tag)
		}
		return true
	}
	proj := lastAnalyzeProject
	chosen := pk.cands[n-1]
	still := false
	for _, c := range proj.RenameCandidates(pk.req) {
		if c == chosen {
			still = true
			break
		}
	}
	if !still {
		fmt.Printf("%s That name isn't there anymore — the project changed since the list was printed. Ask me again and I'll re-check.\n", tag)
		recordBrain("composer", false, "rename pick stale")
		return true
	}
	gateRenamePlan(proj, chosen, pk.newName, ask)
	return true
}

// gateRenamePlan shows one rename plan behind the [y/n] gate and
// runs the spine: apply, build, re-analyze, verify — restoring the
// original bytes on any failure after applying.
func gateRenamePlan(proj *analyze.Project, target analyze.RenameTarget, newName string, ask editAsk) {
	tag := cTag("gollemer [composer]>")
	plan, refusal := proj.PlanRenameTarget(target, newName)
	if refusal != "" || plan == nil {
		if refusal == "" {
			refusal = "I couldn't work that rename out safely from this project's type information."
		}
		fmt.Printf("%s Not renaming: %s\n", tag, refusal)
		recordBrain("composer", false, "rename refused")
		return
	}
	fmt.Printf("%s Here's the rename, worked out from the project's type information — every place the name appears:\n\n%s\n\n%s\n", tag, plan.Summary, plan.Diff())
	ans, err := ask("Apply this rename? [y/n] ")
	if err != nil || (strings.ToLower(strings.TrimSpace(ans)) != "y" && strings.ToLower(strings.TrimSpace(ans)) != "yes") {
		fmt.Printf("%s Not applied — every file is exactly as it was.\n", tag)
		recordBrain("composer", true, "rename declined at the gate")
		return
	}
	root := proj.Root
	if root == "" {
		root = lastAnalyzeRoot
	}
	// The write transaction: on-disk backups of the whole set
	// before anything is written (edittx.go).
	tx, berr := beginEditTx(root, plan.Files)
	if berr != nil {
		fmt.Printf("%s Not renaming: %s\n", tag, berr)
		recordBrain("composer", false, "rename refused: backup")
		return
	}
	_, werr, wrel := tx.writeAll(plan.Files)
	if werr != nil {
		unrestored := tx.rollback()
		fmt.Printf("%s I couldn't write %s (%v) — so I restored your files from the on-disk backups — %s.\n", tag, proj.LinkPath(wrel), werr, rollbackOutcome(unrestored, proj.LinkPath))
		recordBrain("composer", false, "rename write failed; restored")
		return
	}
	built := false
	if _, serr := os.Stat(filepath.Join(root, "go.mod")); serr == nil {
		out, bok := buildProject(root)
		if !bok {
			unrestored := tx.rollback()
			fmt.Printf("%s That rename doesn't build, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
			fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(out), explainBuildError(out))
			recordBrain("composer", false, "rename rolled back: build failed")
			return
		}
		built = true
	}
	fresh, aerr := analyze.Analyze(root)
	if aerr != nil || fresh == nil || !plan.Verify(fresh) {
		unrestored := tx.rollback()
		fmt.Printf("%s I applied the rename but couldn't verify the change is really there, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
		recordBrain("composer", false, "rename rolled back: verify failed")
		return
	}
	tx.commit()
	setCurrentProject(root, fresh)
	how := "built the project and checked"
	if !built {
		how = "re-read the project and checked (no go.mod at its root, so no build to run)"
	}
	fmt.Printf("%s Done — %s, at %s. I %s that it's really there.\n", tag, plan.VerifyDesc, plan.Site, how)
	recordBrain("composer", true, "rename applied")
}
