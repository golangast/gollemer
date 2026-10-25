package chat

// Error -> fix: when a pasted compiler error has exactly one
// mechanical repair (analyze.PlanErrorFix), the translation errhelp
// already printed is followed by the repair itself, offered behind
// the same [y/n] gate as every other edit. The flow mirrors the
// edits spine — apply, build, verify, restore on failure — with one
// difference forced by the situation: the project was ALREADY
// broken when the user pasted the error, so "the build passes"
// can't be the success test. Instead the flow takes a baseline
// build before applying, and afterwards the fix counts as taken
// only when the pasted error is gone from the new build output and
// no error appeared that wasn't in the baseline. Remaining
// pre-existing errors are reported, not treated as failure — one
// error at a time.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	reCompilerLine     = regexp.MustCompile(`^(.+?\.go):(\d+)(?::(\d+))?:\s+(.+)$`)
	reErrUnusedImportA = regexp.MustCompile(`^imported and not used: "([^"]+)"`)
	reErrUnusedImportB = regexp.MustCompile(`^"([^"]+)" imported and not used`)
	reErrUnusedVarA    = regexp.MustCompile(`^declared and not used: ([A-Za-z_]\w*)`)
	reErrUnusedVarB    = regexp.MustCompile(`^([A-Za-z_]\w*) declared and not used`)
	reErrUndefined     = regexp.MustCompile(`^undefined: ([A-Za-z_][\w.]*)$`)
	reErrMissingReturn = regexp.MustCompile(`^missing return at end of function`)
)

// classifyErrorMessage maps the message half of a compiler line to
// a fix class and the name it involves. Classes whose repair is
// not unique and mechanical (type mismatches, argument counts,
// everything else) report false — those keep the translation-only
// behavior they have today.
func classifyErrorMessage(msg string) (class, name string, ok bool) {
	msg = strings.TrimSpace(msg)
	if m := reErrUnusedImportA.FindStringSubmatch(msg); m != nil {
		return analyze.ErrorFixUnusedImport, m[1], true
	}
	if m := reErrUnusedImportB.FindStringSubmatch(msg); m != nil {
		return analyze.ErrorFixUnusedImport, m[1], true
	}
	if m := reErrUnusedVarA.FindStringSubmatch(msg); m != nil {
		return analyze.ErrorFixUnusedVar, m[1], true
	}
	if m := reErrUnusedVarB.FindStringSubmatch(msg); m != nil {
		return analyze.ErrorFixUnusedVar, m[1], true
	}
	if m := reErrUndefined.FindStringSubmatch(msg); m != nil {
		return analyze.ErrorFixUndefinedName, m[1], true
	}
	if reErrMissingReturn.MatchString(msg) {
		return analyze.ErrorFixMissingReturn, "", true
	}
	return "", "", false
}

// parseCompilerError parses one pasted compiler line ("./x.go:5:2:
// message") into a fix request plus the file path AS PASTED — the
// caller resolves it against the project it knows.
func parseCompilerError(line string) (analyze.ErrorFixRequest, string, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimPrefix(s, "vet: ")
	m := reCompilerLine.FindStringSubmatch(s)
	if m == nil {
		return analyze.ErrorFixRequest{}, "", false
	}
	class, name, ok := classifyErrorMessage(m[4])
	if !ok {
		return analyze.ErrorFixRequest{}, "", false
	}
	ln, _ := strconv.Atoi(m[2])
	col := 0
	if m[3] != "" {
		col, _ = strconv.Atoi(m[3])
	}
	return analyze.ErrorFixRequest{Class: class, Line: ln, Col: col, Name: name}, m[1], true
}

// resolveErrorFile turns a pasted path into a project-relative one:
// absolute paths are relativized, "./" is stripped, and a path that
// only matches the tail of one project file (the user built from
// inside a subpackage) resolves to that file when it's unique.
func resolveErrorFile(proj *analyze.Project, raw string) (string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false
	}
	var all []string
	inIndex := map[string]bool{}
	for _, pkg := range proj.Packages {
		for _, f := range pkg.Files {
			inIndex[f] = true
			all = append(all, f)
		}
	}
	cand := raw
	if filepath.IsAbs(cand) {
		rel, err := filepath.Rel(proj.Root, cand)
		if err != nil || strings.HasPrefix(rel, "..") {
			return "", false
		}
		cand = rel
	}
	cand = filepath.ToSlash(strings.TrimPrefix(cand, "./"))
	if inIndex[cand] {
		return cand, true
	}
	// On disk under the root counts too: the index may predate the
	// file, and the planner re-reads and re-checks it anyway.
	if !strings.HasPrefix(cand, "..") {
		if fi, err := os.Stat(filepath.Join(proj.Root, cand)); err == nil && !fi.IsDir() {
			return cand, true
		}
	}
	var suffix []string
	for _, f := range all {
		if strings.HasSuffix(f, "/"+cand) {
			suffix = append(suffix, f)
		}
	}
	if len(suffix) == 1 {
		return suffix[0], true
	}
	base := filepath.Base(cand)
	var byBase []string
	for _, f := range all {
		if filepath.Base(f) == base {
			byBase = append(byBase, f)
		}
	}
	if len(byBase) == 1 {
		return byBase[0], true
	}
	return "", false
}

// errorKey identifies one build error independent of position —
// removing lines above shifts line numbers, so positions can't be
// part of a before/after comparison.
func errorKey(rel, class, name string) string {
	return rel + "|" + class + "|" + name
}

// buildErrorKeys reduces `go build` output to the set of its errors,
// keyed by file + class + name (or file + raw message for shapes
// the fix classes don't cover).
func buildErrorKeys(out string) map[string]bool {
	keys := map[string]bool{}
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		m := reCompilerLine.FindStringSubmatch(ln)
		if m == nil {
			continue
		}
		rel := filepath.ToSlash(strings.TrimPrefix(m[1], "./"))
		if class, name, ok := classifyErrorMessage(m[4]); ok {
			keys[errorKey(rel, class, name)] = true
		} else {
			keys[rel+"|raw|"+strings.TrimSpace(m[4])] = true
		}
	}
	return keys
}

// tryErrorFix is the chat-loop entry: after a pasted error's
// translation has printed, offer its mechanical fix (when it has
// exactly one) behind the [y/n] gate.
func tryErrorFix(line string, r *LineReader) {
	runErrorFix(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runErrorFix offers and, on yes, applies one error's mechanical
// fix. It returns true when a fix was offered (whatever the user
// answered); a pasted error with no unique mechanical fix returns
// false and leaves the translation standing alone.
func runErrorFix(line string, ask editAsk) bool {
	req, rawFile, ok := parseCompilerError(line)
	if !ok {
		return false
	}
	if !ensureFreshProject() {
		return false
	}
	proj := lastAnalyzeProject
	rel, ok := resolveErrorFile(proj, rawFile)
	if !ok {
		return false
	}
	req.File = rel
	plan, _ := proj.PlanErrorFix(req)
	if plan == nil {
		return false
	}
	tag := cTag("gollemer [composer]>")
	fmt.Printf("%s I can fix that one mechanically — here's the exact change:\n\n%s\n\n%s\n", tag, plan.Summary, plan.Diff())
	ans, err := ask("Apply this fix? [y/n] ")
	if err != nil || (strings.ToLower(strings.TrimSpace(ans)) != "y" && strings.ToLower(strings.TrimSpace(ans)) != "yes") {
		fmt.Printf("%s Not applied — every file is exactly as it was.\n", tag)
		recordBrain("composer", true, "error fix declined at the gate: "+req.Class)
		return true
	}
	root := proj.Root
	if root == "" {
		root = lastAnalyzeRoot
	}
	hasMod := false
	if _, serr := os.Stat(filepath.Join(root, "go.mod")); serr == nil {
		hasMod = true
	}
	var baseline map[string]bool
	if hasMod {
		out, _ := buildProject(root)
		baseline = buildErrorKeys(out)
	}
	// The write transaction: an on-disk backup of the file set
	// before anything is written (edittx.go).
	tx, berr := beginEditTx(root, plan.Files)
	if berr != nil {
		fmt.Printf("%s Not doing that fix: %s\n", tag, berr)
		recordBrain("composer", false, "error fix refused: backup")
		return true
	}
	_, werr, wrel := tx.writeAll(plan.Files)
	if werr != nil {
		unrestored := tx.rollback()
		fmt.Printf("%s I couldn't write %s (%v) — so I restored your files from the on-disk backups — %s.\n", tag, proj.LinkPath(wrel), werr, rollbackOutcome(unrestored, proj.LinkPath))
		recordBrain("composer", false, "error fix write failed; restored")
		return true
	}
	if hasMod {
		out, bok := buildProject(root)
		after := buildErrorKeys(out)
		targetKey := errorKey(req.File, req.Class, req.Name)
		newErrs := 0
		for k := range after {
			if !baseline[k] {
				newErrs++
			}
		}
		// The fix took iff its error is gone, nothing new appeared,
		// and the build isn't failing for some output-less reason.
		if after[targetKey] || newErrs > 0 || (!bok && len(after) == 0) {
			unrestored := tx.rollback()
			fmt.Printf("%s That fix didn't take, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
			fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(out), explainBuildError(out))
			recordBrain("composer", false, "error fix rolled back: build failed")
			return true
		}
		fresh, aerr := analyze.Analyze(root)
		if aerr != nil || fresh == nil || !plan.Verify(fresh) {
			unrestored := tx.rollback()
			fmt.Printf("%s I applied the fix but couldn't verify it is really there, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
			recordBrain("composer", false, "error fix rolled back: verify failed")
			return true
		}
		tx.commit()
		setCurrentProject(root, fresh)
		if bok {
			fmt.Printf("%s Fixed — %s. The project builds clean now.\n", tag, plan.VerifyDesc)
		} else {
			fmt.Printf("%s Fixed — %s. The build still reports %d other error(s) — paste the next one and I'll take that too.\n", tag, plan.VerifyDesc, len(after))
		}
		recordBrain("composer", true, "error fix applied: "+req.Class)
		return true
	}
	// No go.mod at the root: the structural verify is the check,
	// mirroring the edits spine.
	fresh, aerr := analyze.Analyze(root)
	if aerr != nil || fresh == nil || !plan.Verify(fresh) {
		unrestored := tx.rollback()
		fmt.Printf("%s I applied the fix but couldn't verify it is really there, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
		recordBrain("composer", false, "error fix rolled back: verify failed")
		return true
	}
	tx.commit()
	setCurrentProject(root, fresh)
	fmt.Printf("%s Fixed — %s (no go.mod at the project root, so there was no build to check against).\n", tag, plan.VerifyDesc)
	recordBrain("composer", true, "error fix applied: "+req.Class)
	return true
}
