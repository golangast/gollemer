package chat

// Cross-file delegate: the chat face of the seventh assembled
// shape. "Add a function in app/main.go that calls store.Load"
// writes a wrapper into one file from a REAL signature the
// analyzer read in another — behind the same [y/n] gate as every
// write, with the weakness list of the classic agent loop
// engineered out one by one:
//
//   - no broken file on failure: before anything is applied, the
//     original bytes AND permission mode of every touched file go
//     to a disk backup beside it (.gollemer-backup), so even a
//     crashed chat leaves a recoverable original — and the next
//     chat start says so, with the exact restore command.
//   - no lying verdict: the outcome is baseline-relative (the
//     errorfix rule) and judged on the affected package only —
//     never a whole-suite run, never blamed for red that predates
//     the edit, never certified by an empty test suite.
//   - no open-ended retry: at most ONE, in the alternate form the
//     plan already named and the gate already covered, and only
//     for an import/qualification-class build failure.
//   - no clean-tree demand: the backup is per-file; dirty trees
//     and untracked targets are the normal case.
//
// The proposer is the deterministic assembler over the analyzer
// index (analyze.planDelegate) — no model emits patch text
// anywhere in this path.

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	reDelegateNamed = regexp.MustCompile(`(?i)^add a function ([A-Za-z_]\w*) in (.+?) that calls ([A-Za-z_][\w.]*)$`)
	reDelegateAnon  = regexp.MustCompile(`(?i)^add a function in (.+?) that calls ([A-Za-z_][\w.]*)$`)
	reDelegateWrap  = regexp.MustCompile(`(?i)^wrap ([A-Za-z_][\w.]*) in (.+)$`)
)

// parseDelegate reads one chat line as a cross-file delegate
// request: the target file as typed, the source function as
// typed, and the wrapper's name ("" when the sentence doesn't
// give one). The second return is false for anything outside the
// grammar.
func parseDelegate(line string) (file, source, wrapper string, ok bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	if m := reDelegateNamed.FindStringSubmatch(s); m != nil {
		return cleanTypedFile(m[2]), m[3], m[1], true
	}
	if m := reDelegateAnon.FindStringSubmatch(s); m != nil {
		return cleanTypedFile(m[1]), m[2], "", true
	}
	if m := reDelegateWrap.FindStringSubmatch(s); m != nil {
		return cleanTypedFile(m[2]), m[1], "", true
	}
	return "", "", "", false
}

// cleanTypedFile tidies a file name as typed in a sentence.
func cleanTypedFile(s string) string {
	s = strings.TrimSpace(s)
	if low := strings.ToLower(s); strings.HasPrefix(low, "the ") {
		s = strings.TrimSpace(s[len("the "):])
	}
	return strings.TrimSuffix(s, "/")
}

// tryDelegate is the chat-loop entry: it claims delegate
// sentences and runs the whole flow on the chat's line reader.
func tryDelegate(line string, r *LineReader) bool {
	return runDelegate(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runDelegate carries one delegate request from sentence to
// verified change (or an honest refusal / a clean "not applied").
func runDelegate(line string, ask editAsk) bool {
	typedFile, sourceName, wrapperName, ok := parseDelegate(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "delegate refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "delegate refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "delegate refused: root missing")
		return true
	}
	// The fence and the existence checks: this is an edit, so
	// the target must already be there, inside the root.
	abs, rel, refusal := serverTargetResolve(rootAbs, typedFile)
	if refusal != "" {
		fmt.Printf("%s Not doing that delegate: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "delegate refused: target")
		return true
	}
	st, serr := os.Stat(abs)
	switch {
	case serr != nil:
		fmt.Printf("%s Not doing that delegate: there's no %s in this project — a delegate goes into a file that already exists. (For a brand-new file, say \"create a file %s\" first.)\n", tag, typedFile, typedFile)
		recordBrain("composer", false, "delegate refused: target missing")
		return true
	case st.IsDir():
		fmt.Printf("%s Not doing that delegate: %s is a folder — name the .go file the function should live in.\n", tag, typedFile)
		recordBrain("composer", false, "delegate refused: target is a folder")
		return true
	case !strings.HasSuffix(strings.ToLower(typedFile), ".go"):
		fmt.Printf("%s Not doing that delegate: %s isn't a Go file.\n", tag, typedFile)
		recordBrain("composer", false, "delegate refused: target not Go")
		return true
	case strings.HasSuffix(strings.ToLower(typedFile), "_test.go"):
		fmt.Printf("%s Not doing that delegate: delegates don't go in test files — the test scaffold this shape writes is the only test-file change it makes.\n", tag)
		recordBrain("composer", false, "delegate refused: target is a test file")
		return true
	}
	// Analyze the target's root quietly, the add-server way
	// (never ensureProject: its demo fallback would aim the edit
	// at a stand-in project).
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "delegate refused: project unreadable")
			return true
		}
	}
	proj := lastAnalyzeProject
	plan, refusal := proj.PlanEdit(analyze.EditRequest{
		Kind:         analyze.EditDelegate,
		DelegateFile: rel,
		SourceName:   sourceName,
		WrapperName:  wrapperName,
	})
	if refusal != "" || plan == nil {
		if refusal == "" {
			refusal = "I couldn't assemble that delegate from this project's own code."
		}
		fmt.Printf("%s Not doing that delegate: %s\n", tag, refusal)
		recordBrain("composer", false, "delegate refused: plan")
		return true
	}
	fmt.Printf("%s Here's the delegate I assembled from this project's own code:\n\n%s\n%s\n", tag, plan.Summary, plan.Provenance)
	for _, n := range plan.Notes {
		fmt.Printf("%s\n", n)
	}
	fmt.Printf("\n%s\n", plan.Diff())
	ans, err := ask("Apply this edit? [y/n] ")
	if err != nil || (strings.ToLower(strings.TrimSpace(ans)) != "y" && strings.ToLower(strings.TrimSpace(ans)) != "yes") {
		fmt.Printf("%s Not applied — every file is exactly as it was.\n", tag)
		recordBrain("composer", true, "delegate declined at the gate")
		return true
	}

	tgtPkg := delegateTargetPkg(proj, rel)
	if tgtPkg == nil {
		fmt.Printf("%s I lost track of %s's package — nothing was written.\n", tag, rel)
		recordBrain("composer", false, "delegate aborted: package lost")
		return true
	}
	pkgDirAbs := filepath.Join(rootAbs, tgtPkg.Dir)

	hasMod := false
	if _, serr := os.Stat(filepath.Join(rootAbs, "go.mod")); serr == nil {
		hasMod = true
	}

	// The baseline: the target package's build error keys,
	// captured before the edit so the verdict can be relative —
	// and its tests, when the build is green, so "still green"
	// means something.
	var baseKeys map[string]bool
	baseBuildOK := false
	baseTestsRun, baseTestsOK := false, false
	baseTestCompileOK := false
	if hasMod {
		baseOut, bok := buildPackageDir(pkgDirAbs)
		baseBuildOK = bok
		baseKeys = buildErrorKeys(baseOut)
		if bok && tgtPkg.Tests > 0 {
			_, tok := testPackageDir(pkgDirAbs)
			baseTestsRun, baseTestsOK = true, tok
		}
		if delegatePlanTouchesTests(plan) || tgtPkg.Tests > 0 {
			_, cok := testCompilePackageDir(pkgDirAbs)
			baseTestCompileOK = cok
		}
	}

	// Disk backups for every file the plan touches, BEFORE
	// applying. A leftover backup from an earlier session is a
	// refusal, not an overwrite: it may be the only copy of the
	// original.
	backups, berr := writeDelegateBackups(rootAbs, plan.Files)
	if berr != nil {
		discardDelegateBackups(backups)
		fmt.Printf("%s Not doing that delegate: %s\n", tag, berr)
		recordBrain("composer", false, "delegate refused: backup")
		return true
	}

	// Apply the primary form, preserving each existing file's
	// permission mode.
	if werr := applyDelegateFiles(rootAbs, plan.Files, backups); werr != nil {
		restoreDelegateBackups(backups)
		fmt.Printf("%s I couldn't write the edit (%v) — so I restored everything from the on-disk backups; your files are exactly as they were.\n", tag, werr)
		recordBrain("composer", false, "delegate write failed; restored")
		return true
	}

	retried := false
	if hasMod {
		afterOut, afterOK := buildPackageDir(pkgDirAbs)
		afterKeys := buildErrorKeys(afterOut)
		okVerdict := delegateBuildPasses(afterOut, afterOK, afterKeys, baseKeys, rel, sourceName)
		if !okVerdict && len(plan.AltFiles) > 0 && delegateRetryApplies(afterOut, baseKeys, rel, sourceName) {
			// The one bounded retry: the alternate form named
			// in the plan replaces fileA. The backup still
			// holds the pre-edit original.
			if werr := applyDelegateFiles(rootAbs, plan.AltFiles[:1], backups[:1]); werr == nil {
				altOut, altOK := buildPackageDir(pkgDirAbs)
				altKeys := buildErrorKeys(altOut)
				if delegateBuildPasses(altOut, altOK, altKeys, baseKeys, rel, sourceName) {
					retried = true
					afterOut, afterOK = altOut, altOK
				} else {
					restoreDelegateBackups(backups)
					fmt.Printf("%s That delegate doesn't build — I tried the retry form named in the plan too, and it didn't build either, so I restored your files from the on-disk backups; everything is exactly as it was.\n", tag)
					fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(altOut), explainBuildError(altOut))
					recordBrain("composer", false, "delegate rolled back: build failed after retry")
					return true
				}
			}
		}
		if !okVerdict && !retried {
			restoreDelegateBackups(backups)
			fmt.Printf("%s That delegate doesn't build, so I restored your files from the on-disk backups — everything is exactly as it was.\n", tag)
			fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(afterOut), explainBuildError(afterOut))
			recordBrain("composer", false, "delegate rolled back: build failed")
			return true
		}
		// The test files must at least compile — the stub is
		// ours — and when the baseline tests were green they
		// must still be: that verdict rides on the package's
		// pre-existing tests, never on the stub. Both checks are
		// baseline-relative like the build verdict: a package
		// whose test files didn't compile before the edit isn't
		// failed for staying that way.
		if delegatePlanTouchesTests(plan) || tgtPkg.Tests > 0 {
			if baseTestsOK {
				if tout, tok := testPackageDir(pkgDirAbs); !tok {
					restoreDelegateBackups(backups)
					fmt.Printf("%s That delegate builds, but the %s package's tests fail with it in place, so I restored your files from the on-disk backups.\n", tag, tgtPkg.Name)
					fmt.Printf("The tests said: %s\n", tailLines(tout, 12))
					recordBrain("composer", false, "delegate rolled back: package tests failed")
					return true
				}
			} else if baseTestCompileOK {
				if cout, cok := testCompilePackageDir(pkgDirAbs); !cok {
					restoreDelegateBackups(backups)
					fmt.Printf("%s That delegate builds, but the package's test files don't compile with it in place, so I restored your files from the on-disk backups.\n", tag)
					fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(cout), explainBuildError(cout))
					recordBrain("composer", false, "delegate rolled back: test files don't compile")
					return true
				}
			}
		}
	}

	// Re-analyze and verify the wrapper really exists and calls
	// the source — the spine's structural check.
	fresh, aerr := analyze.Analyze(rootAbs)
	if aerr != nil || fresh == nil || !plan.Verify(fresh) {
		restoreDelegateBackups(backups)
		fmt.Printf("%s I applied the delegate but couldn't verify it's really there, so I restored your files from the on-disk backups — everything is exactly as it was.\n", tag)
		recordBrain("composer", false, "delegate rolled back: verify failed")
		return true
	}
	lastAnalyzeProject = fresh
	discardDelegateBackups(backups)

	how := "built the " + tgtPkg.Name + " package and checked"
	if !hasMod {
		how = "re-read the project and checked (no go.mod at its root, so no build to run)"
	}
	fmt.Printf("%s Done — %s, at %s. I %s that it's really there.\n", tag, plan.VerifyDesc, plan.Site, how)
	if retried {
		fmt.Printf("Heads up: the plain form didn't build, so what's in your file is the retry the plan named: %s.\n", plan.AltDesc)
	}
	if hasMod && !baseBuildOK {
		fmt.Printf("One more thing: the %s package had %d build error(s) before this edit and still does — none of them came from this delegate; I left them exactly as they were.\n", tgtPkg.Name, len(baseKeys))
	}
	if baseTestsRun && !baseTestsOK {
		fmt.Printf("Its tests were already failing before this edit, so I rested the verdict on the build, not the tests.\n")
	}
	note := "delegate applied: " + sourceName
	if retried {
		note += " (retry form)"
	}
	recordBrain("composer", true, note)
	return true
}

// delegateTargetPkg finds the analyzed package owning rel.
func delegateTargetPkg(proj *analyze.Project, rel string) *analyze.Package {
	for _, pkg := range proj.Packages {
		for _, f := range pkg.Files {
			if f == rel {
				return pkg
			}
		}
	}
	return nil
}

// delegatePlanTouchesTests reports whether the plan writes a
// _test.go file (the skeleton).
func delegatePlanTouchesTests(plan *analyze.EditPlan) bool {
	for _, f := range plan.Files {
		if strings.HasSuffix(f.Rel, "_test.go") {
			return true
		}
	}
	return false
}

// delegateBuildPasses is the baseline-relative verdict for one
// build of the affected package: the build passes outright, OR
// it adds no error key the baseline didn't have and no error
// line points at the generated code. A build that fails with no
// parseable error keys at all can't be judged — that's a
// failure, not a pass (the errorfix guard).
func delegateBuildPasses(out string, buildOK bool, afterKeys, baseKeys map[string]bool, targetRel, sourceName string) bool {
	if buildOK {
		return true
	}
	if len(afterKeys) == 0 && len(baseKeys) == 0 {
		return false
	}
	for k := range afterKeys {
		if !baseKeys[k] {
			return false
		}
	}
	return !buildOutputReferencesDelegate(out, targetRel, sourceName)
}

var (
	reErrUndefinedQual = regexp.MustCompile(`undefined: [A-Za-z_]\w*\.[A-Za-z_]\w*`)
	reErrQualUndefined = regexp.MustCompile(`[A-Za-z_]\w*\.[A-Za-z_]\w* undefined`)
	reErrRedeclared    = regexp.MustCompile(`^[A-Za-z_]\w* redeclared in this block$`)
)

// delegateRetryApplies reports whether a failed build is the
// import/qualification class the one bounded retry exists for:
// every error line that is NEW (not a baseline key) must read as
// a qualification failure on the target file — the source's
// qualified name undefined, its import redeclared or unused, and
// the note lines those errors trail — and at least one line must
// be a real qualification error, not a note.
func delegateRetryApplies(out string, baseKeys map[string]bool, targetRel, sourceName string) bool {
	targetBase := filepath.Base(targetRel)
	real := false
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
		msg := strings.TrimSpace(m[4])
		var key string
		if class, name, okk := classifyErrorMessage(msg); okk {
			key = errorKey(rel, class, name)
		} else {
			key = rel + "|raw|" + msg
		}
		if baseKeys[key] {
			continue // pre-existing: not the retry's business
		}
		if rel != targetBase {
			return false
		}
		switch {
		case reErrUndefinedQual.MatchString(msg) && strings.Contains(msg, "."+lastSegment(sourceName)):
			real = true
		case reErrQualUndefined.MatchString(msg) && strings.Contains(msg, "."+lastSegment(sourceName)):
			real = true
		case reErrRedeclared.MatchString(msg):
			real = true
		case strings.Contains(msg, "imported") && (strings.Contains(msg, "not used") || strings.Contains(msg, "imported as")):
			real = true
		case strings.HasPrefix(msg, "other declaration of "):
			// The note line a redeclaration trails — noise, not
			// a disqualifier.
		default:
			return false
		}
	}
	return real
}

// buildOutputReferencesDelegate reports whether any error line
// points at the generated code: an error in the target file
// naming the qualified call the wrapper makes.
func buildOutputReferencesDelegate(out, targetRel, sourceName string) bool {
	targetBase := filepath.Base(targetRel)
	needle := "." + lastSegment(sourceName)
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		m := reCompilerLine.FindStringSubmatch(ln)
		if m == nil {
			continue
		}
		if filepath.ToSlash(strings.TrimPrefix(m[1], "./")) != targetBase {
			continue
		}
		if strings.Contains(m[4], needle) {
			return true
		}
	}
	return false
}

// lastSegment returns the part of a typed name after the last dot.
func lastSegment(s string) string {
	if i := strings.LastIndex(s, "."); i >= 0 {
		return s[i+1:]
	}
	return s
}

// --- package build / test runners (affected package only) -------------------

// buildPackageDir compiles exactly one package (the directory),
// discarding the binary — never ./..., never the whole module.
func buildPackageDir(dir string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "build", "-o", os.DevNull, ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// testPackageDir runs exactly one package's tests, bounded by
// the runner's 60-second precedent.
func testPackageDir(dir string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "test", "-count=1", ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// testCompilePackageDir compiles the package's test files
// without running any test, so a broken scaffold can never slip
// in under a verdict that rested on the plain build.
func testCompilePackageDir(dir string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "test", "-count=1", "-run", "^$", ".")
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// tailLines renders the last n lines of command output.
func tailLines(out string, n int) string {
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// --- disk backups ------------------------------------------------------------

// delegateBackup is one plan file's on-disk safety copy: the
// original bytes at .<name>.gollemer-backup beside the file, and
// the original permission mode in a .mode sidecar next to it, so
// a restore is byte- and mode-exact even across a crashed chat.
type delegateBackup struct {
	rel     string
	abs     string
	bakAbs  string
	modeAbs string
	created bool // the file did not exist before the edit (a created scaffold)
	mode    os.FileMode
}

// backupPathsFor names a file's backup and mode sidecar.
func backupPathsFor(abs string) (bak, mode string) {
	bak = filepath.Join(filepath.Dir(abs), "."+filepath.Base(abs)+".gollemer-backup")
	return bak, bak + ".mode"
}

// writeDelegateBackups writes the backup for every file in
// files, from the bytes on disk right now. A file with a backup
// already beside it stops the whole edit: that backup may be the
// only copy of an earlier original.
func writeDelegateBackups(root string, files []analyze.EditFile) ([]*delegateBackup, error) {
	var out []*delegateBackup
	for _, f := range files {
		abs := filepath.Join(root, f.Rel)
		bakAbs, modeAbs := backupPathsFor(abs)
		b := &delegateBackup{rel: f.Rel, abs: abs, bakAbs: bakAbs, modeAbs: modeAbs}
		if _, err := os.Stat(bakAbs); err == nil {
			return out, fmt.Errorf("there's already a leftover backup for %s (%s) — an earlier edit may not have finished. Restore it or remove it first, then ask again", f.Rel, filepath.Base(bakAbs))
		}
		fi, err := os.Stat(abs)
		if os.IsNotExist(err) {
			b.created = true
			out = append(out, b)
			continue
		}
		if err != nil {
			return out, fmt.Errorf("couldn't read %s to back it up: %v", f.Rel, err)
		}
		data, err := os.ReadFile(abs)
		if err != nil {
			return out, fmt.Errorf("couldn't read %s to back it up: %v", f.Rel, err)
		}
		b.mode = fi.Mode().Perm()
		if err := os.WriteFile(bakAbs, data, 0o600); err != nil {
			return out, fmt.Errorf("couldn't write the backup for %s: %v", f.Rel, err)
		}
		if err := os.WriteFile(modeAbs, []byte(fmt.Sprintf("%04o\n", b.mode)), 0o600); err != nil {
			return out, fmt.Errorf("couldn't write the backup's mode record for %s: %v", f.Rel, err)
		}
		out = append(out, b)
	}
	return out, nil
}

// applyDelegateFiles writes the given plan files, preserving
// each existing file's recorded permission mode (the spine's
// writer flattens everything to 0644; this shape restores modes
// everywhere, including the success path).
func applyDelegateFiles(root string, files []analyze.EditFile, backups []*delegateBackup) error {
	byRel := map[string]*delegateBackup{}
	for _, b := range backups {
		byRel[b.rel] = b
	}
	for _, f := range files {
		mode := os.FileMode(0o644)
		if b := byRel[f.Rel]; b != nil && !b.created {
			mode = b.mode
		}
		if err := os.WriteFile(filepath.Join(root, f.Rel), f.After, mode); err != nil {
			return err
		}
		if b := byRel[f.Rel]; b != nil && !b.created {
			// WriteFile only applies its mode to new files; make
			// the preserved mode certain on existing ones too.
			_ = os.Chmod(filepath.Join(root, f.Rel), mode)
		}
	}
	return nil
}

// restoreDelegateBackups puts every file back from its on-disk
// backup — bytes and mode — removes files the edit created, and
// removes the backups themselves. Best effort by design: it
// runs on paths this flow just wrote.
func restoreDelegateBackups(backups []*delegateBackup) {
	for _, b := range backups {
		if b.created {
			_ = os.Remove(b.abs)
		} else if data, err := os.ReadFile(b.bakAbs); err == nil {
			_ = os.WriteFile(b.abs, data, b.mode)
			_ = os.Chmod(b.abs, b.mode)
		}
		_ = os.Remove(b.bakAbs)
		_ = os.Remove(b.modeAbs)
	}
}

// discardDelegateBackups removes the backups after a successful
// edit — the originals have been replaced on purpose.
func discardDelegateBackups(backups []*delegateBackup) {
	for _, b := range backups {
		_ = os.Remove(b.bakAbs)
		_ = os.Remove(b.modeAbs)
	}
}

// --- crash recovery notices --------------------------------------------------

// leftoverBackupNotices scans root for unfinished-edit backups
// (.gollemer-backup files) and renders one notice per file,
// naming the file each backup belongs to and the exact terminal
// command that restores it. The chat prints them at start; it
// never runs the restore itself — the command is the user's to
// run, the boundary-answer style used everywhere else.
func leftoverBackupNotices(root string) []string {
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
