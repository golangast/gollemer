package chat

// Cross-file use: the chat face of the assembled shape beside
// the delegate. "In web.go add a function that uses LoadUser
// from store.go" writes a NEW function into web.go whose body
// is a first use of a symbol the project already has — for a
// function, a call with starting values the assembler derives
// from its real parameters and handling for its real returns;
// for a type, a value built from its real definition. The
// delegate next door is the passthrough (the new function's
// signature IS the source's); this shape's function has its
// own empty signature and consumes the call inside its body.
//
// The flow is the delegate's, because the guarantees are the
// same: the shared disk-backup transaction, a baseline-relative
// verdict on the one affected package, at most ONE retry in
// the plan's named alternate form for an import/qualification
// failure, and a structural verify before the outcome is
// reported. The proposer is the deterministic assembler over
// the analyzer index (analyze.planUse) — no model emits patch
// text anywhere in this path.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	// The non-lead named form carries the name bare
	// ("add a function Report in …") or behind the book's
	// introducer ("add a function named Report in …") —
	// the hybrid. The named-connection rule taught the book
	// that a name belongs to a declaration; this shape IS a
	// declaration ask, so the "named" form routes here too
	// instead of falling through to the boundary's
	// extra-clause refusal, whose forms never mention the
	// uses sentence that works.
	reUsesNamedLong = regexp.MustCompile(`(?i)^add a function (?:named )?([A-Za-z_]\w*) in (.+?) that uses (?:the )?(type )?([A-Za-z_][\w.]*)(?: from (.+))?$`)
	reUsesAnonLong  = regexp.MustCompile(`(?i)^add a function in (.+?) that uses (?:the )?(type )?([A-Za-z_][\w.]*)(?: from (.+))?$`)
	reUsesLeadNamed = regexp.MustCompile(`(?i)^in (.+?),? add a function (?:named )?([A-Za-z_]\w*) that uses (?:the )?(type )?([A-Za-z_][\w.]*)(?: from (.+))?$`)
	reUsesLeadAnon  = regexp.MustCompile(`(?i)^in (.+?),? add a function that uses (?:the )?(type )?([A-Za-z_][\w.]*)(?: from (.+))?$`)
)

// parseUses reads one chat line as a cross-file use request:
// the target file as typed, the referenced symbol as typed
// (bare or pkg.Name), the file the sentence says the symbol
// lives in ("" when unstated), the new function's name (""
// when the sentence doesn't give one), and whether the "type"
// designator forces the symbol to resolve as a type. The
// second return is false for anything outside the grammar —
// including every "that calls" sentence, which is the
// delegate's.
func parseUses(line string) (file, symbol, fromFile, name string, forceType bool, ok bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	if m := reUsesNamedLong.FindStringSubmatch(s); m != nil {
		return cleanTypedFile(m[2]), m[4], cleanTypedFile(m[5]), m[1], m[3] != "", true
	}
	if m := reUsesLeadNamed.FindStringSubmatch(s); m != nil {
		return cleanTypedFile(m[1]), m[4], cleanTypedFile(m[5]), m[2], m[3] != "", true
	}
	if m := reUsesAnonLong.FindStringSubmatch(s); m != nil {
		return cleanTypedFile(m[1]), m[3], cleanTypedFile(m[4]), "", m[2] != "", true
	}
	if m := reUsesLeadAnon.FindStringSubmatch(s); m != nil {
		return cleanTypedFile(m[1]), m[3], cleanTypedFile(m[4]), "", m[2] != "", true
	}
	return "", "", "", "", false, false
}

// tryUses is the chat-loop entry: it claims use sentences and
// runs the whole flow on the chat's line reader.
func tryUses(line string, r *LineReader) bool {
	return runUses(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runUses carries one use request from sentence to verified
// change (or an honest refusal / a clean "not applied") — the
// delegate's flow, in the use's own words.
func runUses(line string, ask editAsk) bool {
	typedFile, symbol, fromFile, newName, forceType, ok := parseUses(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "use refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "use refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "use refused: root missing")
		return true
	}
	// The fence and the existence checks: this is an edit, so
	// the target must already be there, inside the root.
	abs, rel, refusal := serverTargetResolve(rootAbs, typedFile)
	if refusal != "" {
		fmt.Printf("%s Not doing that use: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "use refused: target")
		return true
	}
	st, serr := os.Stat(abs)
	switch {
	case serr != nil:
		fmt.Printf("%s Not doing that use: there's no %s in this project — a use goes into a file that already exists. (For a brand-new file, say \"create a file %s\" first.)\n", tag, typedFile, typedFile)
		recordBrain("composer", false, "use refused: target missing")
		return true
	case st.IsDir():
		fmt.Printf("%s Not doing that use: %s is a folder — name the .go file the function should live in.\n", tag, typedFile)
		recordBrain("composer", false, "use refused: target is a folder")
		return true
	case !strings.HasSuffix(strings.ToLower(typedFile), ".go"):
		fmt.Printf("%s Not doing that use: %s isn't a Go file.\n", tag, typedFile)
		recordBrain("composer", false, "use refused: target not Go")
		return true
	case strings.HasSuffix(strings.ToLower(typedFile), "_test.go"):
		fmt.Printf("%s Not doing that use: new functions don't go in test files — the test scaffold this shape writes is the only test-file change it makes.\n", tag)
		recordBrain("composer", false, "use refused: target is a test file")
		return true
	}
	// Analyze the target's root quietly, the add-server way.
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "use refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "use refused: stale project unreadable")
		return true
	}
	proj := lastAnalyzeProject
	plan, refusal := proj.PlanEdit(analyze.EditRequest{
		Kind:        analyze.EditUse,
		UseFile:     rel,
		UseSymbol:   symbol,
		UseFromFile: fromFile,
		UseName:     newName,
		UseType:     forceType,
	})
	if refusal != "" || plan == nil {
		if refusal == "" {
			refusal = "I couldn't assemble that use from this project's own code."
		}
		fmt.Printf("%s Not doing that use: %s\n", tag, refusal)
		recordBrain("composer", false, "use refused: plan")
		return true
	}
	fmt.Printf("%s Here's the use I assembled from this project's own code:\n\n%s\n%s\n", tag, plan.Summary, plan.Provenance)
	for _, n := range plan.Notes {
		fmt.Printf("%s\n", n)
	}
	fmt.Printf("\n%s\n", plan.Diff())
	ans, err := ask("Apply this edit? [y/n] ")
	if err != nil || (strings.ToLower(strings.TrimSpace(ans)) != "y" && strings.ToLower(strings.TrimSpace(ans)) != "yes") {
		fmt.Printf("%s Not applied — every file is exactly as it was.\n", tag)
		recordBrain("composer", true, "use declined at the gate")
		return true
	}

	tgtPkg := delegateTargetPkg(proj, rel)
	if tgtPkg == nil {
		fmt.Printf("%s I lost track of %s's package — nothing was written.\n", tag, rel)
		recordBrain("composer", false, "use aborted: package lost")
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
	// means something. (The verdict helpers are the delegate's:
	// they are parameterized by target file and symbol, and the
	// failure class they judge — build errors the edit added —
	// is the same for both shapes.)
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

	// The shared write transaction (edittx.go): disk backups
	// for every file the plan touches, BEFORE applying.
	tx, berr := beginEditTx(rootAbs, plan.Files)
	if berr != nil {
		fmt.Printf("%s Not doing that use: %s\n", tag, berr)
		recordBrain("composer", false, "use refused: backup")
		return true
	}

	// Apply the primary form, preserving each existing file's
	// permission mode.
	if _, werr, _ := tx.writeAll(plan.Files); werr != nil {
		unrestored := tx.rollback()
		fmt.Printf("%s I couldn't write the edit (%v) — so I restored everything from the on-disk backups — %s.\n", tag, werr, rollbackOutcome(unrestored, proj.LinkPath))
		recordBrain("composer", false, "use write failed; restored")
		return true
	}

	retried := false
	if hasMod {
		afterOut, afterOK := buildPackageDir(pkgDirAbs)
		afterKeys := buildErrorKeys(afterOut)
		okVerdict := delegateBuildPasses(afterOut, afterOK, afterKeys, baseKeys, rel, symbol)
		if !okVerdict && len(plan.AltFiles) > 0 && delegateRetryApplies(afterOut, baseKeys, rel, symbol) {
			// The one bounded retry: the alternate form named
			// in the plan replaces the target file — inside the
			// SAME transaction, so a retry failure still
			// restores the pre-edit original, never the first
			// attempt.
			if werr := tx.write(plan.AltFiles[0]); werr == nil {
				altOut, altOK := buildPackageDir(pkgDirAbs)
				altKeys := buildErrorKeys(altOut)
				if delegateBuildPasses(altOut, altOK, altKeys, baseKeys, rel, symbol) {
					retried = true
					afterOut, afterOK = altOut, altOK
				} else {
					unrestored := tx.rollback()
					fmt.Printf("%s That use doesn't build — I tried the retry form named in the plan too, and it didn't build either, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
					fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(altOut), explainBuildError(altOut))
					recordBrain("composer", false, "use rolled back: build failed after retry")
					return true
				}
			}
		}
		if !okVerdict && !retried {
			unrestored := tx.rollback()
			fmt.Printf("%s That use doesn't build, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
			fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(afterOut), explainBuildError(afterOut))
			recordBrain("composer", false, "use rolled back: build failed")
			return true
		}
		// The test files must at least compile — the stub is
		// ours — and when the baseline tests were green they
		// must still be: baseline-relative, like the build
		// verdict, resting on the package's pre-existing tests,
		// never on the stub.
		if delegatePlanTouchesTests(plan) || tgtPkg.Tests > 0 {
			if baseTestsOK {
				if tout, tok := testPackageDir(pkgDirAbs); !tok {
					unrestored := tx.rollback()
					fmt.Printf("%s That use builds, but the %s package's tests fail with it in place, so I restored your files from the on-disk backups — %s.\n", tag, tgtPkg.Name, rollbackOutcome(unrestored, proj.LinkPath))
					fmt.Printf("The tests said: %s\n", tailLines(tout, 12))
					recordBrain("composer", false, "use rolled back: package tests failed")
					return true
				}
			} else if baseTestCompileOK {
				if cout, cok := testCompilePackageDir(pkgDirAbs); !cok {
					unrestored := tx.rollback()
					fmt.Printf("%s That use builds, but the package's test files don't compile with it in place, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
					fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(cout), explainBuildError(cout))
					recordBrain("composer", false, "use rolled back: test files don't compile")
					return true
				}
			}
		}
	}

	// Re-analyze and verify the new function really exists and
	// uses the symbol — the spine's structural check.
	fresh, aerr := analyze.Analyze(rootAbs)
	if aerr != nil || fresh == nil || !plan.Verify(fresh) {
		unrestored := tx.rollback()
		fmt.Printf("%s I applied the use but couldn't verify it's really there, so I restored your files from the on-disk backups — %s.\n", tag, rollbackOutcome(unrestored, proj.LinkPath))
		recordBrain("composer", false, "use rolled back: verify failed")
		return true
	}
	setCurrentProject(rootAbs, fresh)
	tx.commit()
	notePlanLanded(rootAbs, plan, "function that uses a symbol")
	recordPlanReceipt(rootAbs, plan, analyze.EditUse, newName, "function that uses a symbol", hasMod)

	how := "built the " + tgtPkg.Name + " package and checked"
	if !hasMod {
		how = "re-read the project and checked (no go.mod at its root, so no build to run)"
	}
	fmt.Printf("%s Done — %s, at %s. I %s that it's really there.\n", tag, plan.VerifyDesc, plan.Site, how)
	if retried {
		fmt.Printf("Heads up: the plain form didn't build, so what's in your file is the retry the plan named: %s.\n", plan.AltDesc)
	}
	if hasMod && !baseBuildOK {
		fmt.Printf("One more thing: the %s package had %d build error(s) before this edit and still does — none of them came from this use; I left them exactly as they were.\n", tgtPkg.Name, len(baseKeys))
	}
	if baseTestsRun && !baseTestsOK {
		fmt.Printf("Its tests were already failing before this edit, so I rested the verdict on the build, not the tests.\n")
	}
	note := "use applied: " + symbol
	if retried {
		note += " (retry form)"
	}
	recordBrain("composer", true, note)
	return true
}
