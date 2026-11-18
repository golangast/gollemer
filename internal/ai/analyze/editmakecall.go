package analyze

// Make-and-call: ONE ask, two files. "Make a function named
// Validate in api/validate.go and call it in api/main.go"
// declares a NEW plain function in the first file and writes
// a call to it in the second — in one plan, behind one gate,
// in one transaction, with the spine's whole-project build
// as the verdict and both files rolling back together.
//
// The two halves are the family's existing machinery, joined:
//
//   - The callee is the concept book's function row, planned
//     by planConcept itself (req.Concept/ConceptName/File
//     ride along), so every declaration rule — the name
//     collision refusal, the package conflict refusals, a new
//     file created in an existing folder, the empty-scaffold
//     honesty note — is the declaration's own, verbatim.
//   - The caller is assembled from the callee's PLANNED
//     declaration: the callee plan's After bytes are parsed
//     back and the declaration read from them, exactly as it
//     will sit on disk, and the uses machinery
//     (newDelegateGen + usesGen.funcBody) renders the calling
//     code against that signature — starting values derived
//     for its parameters, its results handled in the body.
//     Nothing is read from a file the plan hasn't written.
//
// Two caller forms: a NEW function in the target file (the
// uses shape, named by the sentence), or the bare call
// inserted at the end of an EXISTING function in the target
// file — the sentence's host, or the package's main when the
// sentence names none. No alternate/retry form is offered:
// the spine's one build verdict covers both files, and a
// failure restores both, so there is no second form to name.

import (
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
)

// planMakeCall assembles the joined plan: the concept
// planner's declaration plan for the callee, plus the caller
// side assembled against that plan's own output.
func (p *Project) planMakeCall(req EditRequest) (*EditPlan, string) {
	entry := req.Concept
	if entry == nil || entry.Kind != ConceptDeclaration {
		return nil, "that edit needs the concept book's function piece, and none was named."
	}
	name := strings.TrimSpace(req.ConceptName)
	calleeRel := strings.TrimSpace(req.File)
	targetRel := strings.TrimSpace(req.MakeCallTarget)
	if calleeRel == "" || targetRel == "" {
		return nil, "that edit needs both files named — the one the new function goes in and the one that calls it."
	}
	if calleeRel == targetRel {
		return nil, "the function and its caller are aimed at the same file — that is an ordinary one-file ask, not this shape."
	}
	if why, blocked := p.fileBlocked(targetRel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(targetRel), why)
	}
	if strings.HasSuffix(targetRel, "_test.go") {
		return nil, fmt.Sprintf("%s is a test file — new functions don't go there.", p.LinkPath(targetRel))
	}

	// Half one: the callee, planned by the declaration
	// machinery itself. Its refusals ARE this plan's
	// refusals (the name already exists, the package
	// conflicts, the folder is missing) — they surface
	// verbatim, before anything about the caller is read.
	calleePlan, refusal := p.planConcept(EditRequest{
		Kind:        EditConcept,
		File:        calleeRel,
		Concept:     entry,
		ConceptName: name,
	})
	if refusal != "" || calleePlan == nil {
		if refusal == "" {
			refusal = "I couldn't assemble the new function's declaration."
		}
		return nil, refusal
	}
	if len(calleePlan.Files) != 1 {
		return nil, "internal error in my assembler: the declaration plan touched more than its one file — I haven't shown you a plan. Nothing was written."
	}
	calleeFile := calleePlan.Files[0]

	// The planned declaration, read back from the plan's own
	// bytes: this AST is the signature source for the caller,
	// the same discipline finishUsePlan uses when it parses
	// its new function back out of its assembled primary.
	calleeFset, calleeAST, err := parseGoSrc(calleeRel, calleeFile.After)
	if err != nil || calleeAST == nil {
		return nil, "internal error in my assembler: the planned declaration doesn't parse — I haven't shown you a plan. Nothing was written."
	}
	var calleeDecl *ast.FuncDecl
	for _, d := range calleeAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == name {
			calleeDecl = fd
			break
		}
	}
	if calleeDecl == nil {
		return nil, fmt.Sprintf("internal error in my assembler: the planned file has no function named %s — I haven't shown you a plan. Nothing was written.", name)
	}
	srcPkgName := calleeAST.Name.Name
	srcDir := filepath.ToSlash(filepath.Dir(calleeRel))
	if srcDir == "." {
		srcDir = ""
	}
	srcLine := insertedLine(calleeFile.After, "func "+name)
	sig := "func " + name + "()"
	if calleeDecl.Type != nil && calleeDecl.Type.Params != nil && len(calleeDecl.Type.Params.List) > 0 {
		// The book's function row is parameterless, but the
		// caller is assembled from whatever the plan declares:
		// keep the provenance honest if that ever changes.
		sig = "func " + name + "(…)"
	}
	src := &Func{
		Name:     name,
		Pkg:      srcPkgName,
		PkgDir:   srcDir,
		File:     calleeRel,
		Line:     srcLine,
		Sig:      sig,
		Exported: ast.IsExported(name),
	}

	// Post-declaration bytes by file: the caller side reads
	// its target as the callee plan leaves it (the two sides
	// name different files, but the target's PACKAGE may hold
	// the callee file — a host main can live there).
	postBytes := map[string][]byte{calleeRel: calleeFile.After}

	// The target side's facts: its package, its current (or
	// post-declaration) bytes, its parsed form.
	tgtPkg := p.pkgOfFile(targetRel)
	if tgtPkg == nil {
		return nil, fmt.Sprintf("%s isn't part of the project I analyzed — it may sit in a folder the analyzer skips, or under its own go.mod. I can only write into files the index knows.", p.LinkPath(targetRel))
	}
	tgtSrc, tgtFset, tgtAST, refusal := p.makeCallTargetState(targetRel, postBytes)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(tgtSrc) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(targetRel))
	}

	cross := srcDir != tgtPkg.Dir
	if cross {
		if srcPkgName == "main" {
			return nil, fmt.Sprintf("%s will live in package main — a program, and programs can't be imported, so nothing in another package can call it. Put the new function in a library package, or the caller in the same package.", name)
		}
		if !src.Exported {
			return nil, fmt.Sprintf("%s isn't exported (its name starts lowercase), and %s is a different package — Go only lets other packages call exported names. Name it with a capital letter (like %s), or put the caller in %s instead.", name, tgtPkg.Name, titleFirst(name), srcPkgName)
		}
	}

	// Half two: the caller, in the sentence's form.
	var callerFile *EditFile
	var callerSite string
	var callerTail string // the summary's calling-side clause
	var verifyFile, verifyFunc string
	var extraNotes []string
	if caller := strings.TrimSpace(req.MakeCallCaller); caller != "" {
		ef, site, notes, refusal := p.makeCallNamedCaller(req, src, calleeDecl, calleeFset, calleeAST, targetRel, tgtPkg, tgtSrc, tgtFset, tgtAST, cross, caller)
		if refusal != "" {
			return nil, refusal
		}
		callerFile, callerSite, extraNotes = ef, site, notes
		callerTail = fmt.Sprintf("the caller is a new function %s in %s, whose body calls %s", caller, p.LinkPath(targetRel), name)
		verifyFile, verifyFunc = targetRel, caller
	} else {
		ef, hostRel, hostName, site, notes, refusal := p.makeCallHostedCall(req, src, calleeDecl, targetRel, tgtPkg, tgtSrc, tgtFset, tgtAST, cross, postBytes)
		if refusal != "" {
			return nil, refusal
		}
		callerFile, callerSite, extraNotes = ef, site, notes
		callerTail = fmt.Sprintf("the call lands at the end of %s in %s", hostName, p.LinkPath(hostRel))
		verifyFile, verifyFunc = hostRel, hostName
	}

	// Join the files: the callee's first; the caller's
	// replaces any same-file entry (its After was assembled
	// against the post-declaration bytes, its Before stays
	// the file's true original).
	files := []EditFile{calleeFile}
	merged := false
	for i := range files {
		if files[i].Rel == callerFile.Rel {
			orig := files[i].Before
			files[i] = *callerFile
			files[i].Before = orig
			files[i].Diff = unifiedDiff(files[i].Rel, orig, files[i].After)
			merged = true
		}
	}
	if !merged {
		files = append(files, *callerFile)
	}

	notes := append([]string{}, calleePlan.Notes...)
	notes = append(notes, extraNotes...)
	notes = append(notes, "One plan covers both files: if either side fails to build, both files go back exactly as they were.")

	summary := fmt.Sprintf("Make %s in %s and call it in %s: %s lands in %s, and %s.",
		name, p.LinkPath(calleeRel), p.LinkPath(targetRel), name, p.LinkPath(calleeRel), callerTail)
	verifyDesc := fmt.Sprintf("%s now exists in %s and is called from %s", name, p.LinkPath(calleeRel), p.LinkPath(verifyFile))
	calleeVerify := calleePlan.verify
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       callerSite,
		Files:      files,
		Provenance: fmt.Sprintf("built from the declaration this plan writes — %s (%s:%d)", sig, p.LinkPath(calleeRel), srcLine),
		Notes:      notes,
		verify: func(after *Project) bool {
			if calleeVerify != nil && !calleeVerify(after) {
				return false
			}
			if calleeVerify == nil {
				// planConcept's plans always carry a verify;
				// fall back to the declaration check itself.
				found := false
				for _, pk := range after.Packages {
					for _, fn := range pk.Funcs {
						if fn.Receiver == "" && fn.Name == name && fn.File == calleeRel {
							found = true
						}
					}
				}
				if !found {
					return false
				}
			}
			return verifyDelegateCall(after.Root, verifyFile, verifyFunc, name)
		},
	}, ""
}

// makeCallTargetState reads the caller side's file as the
// plan leaves it: the callee plan's After bytes when the file
// is the callee's own, the disk bytes otherwise. A target the
// index knows but the disk lacks (it would be created by the
// callee plan only when it IS the callee file) refuses in the
// uses family's voice.
func (p *Project) makeCallTargetState(rel string, postBytes map[string][]byte) ([]byte, *token.FileSet, *ast.File, string) {
	if after, ok := postBytes[rel]; ok {
		fset, fileAST, err := parseGoSrc(rel, after)
		if err != nil || fileAST == nil {
			return nil, nil, nil, "internal error in my assembler: the planned file doesn't parse — I haven't shown you a plan. Nothing was written."
		}
		return after, fset, fileAST, ""
	}
	src, fset, fileAST, refusal := p.readProjectFile(rel)
	if refusal != "" {
		if strings.Contains(refusal, "no such file") || strings.Contains(refusal, "not exist") {
			return nil, nil, nil, fmt.Sprintf("there's no %s in this project — the calling side goes into a file that already exists. (For a brand-new file, say \"create a file %s\" first.)", p.LinkPath(rel), rel)
		}
		return nil, nil, nil, refusal
	}
	return src, fset, fileAST, ""
}

// makeCallNamedCaller assembles the new caller function in
// the target file, through the uses machinery: the delegate
// rendering context is built over the PLANNED declaration
// (its file's post-plan AST is the signature source), and
// usesGen.funcBody renders the body — the call with starting
// values for the declaration's parameters and its results
// handled, exactly as a use of an existing function renders.
func (p *Project) makeCallNamedCaller(req EditRequest, src *Func, srcDecl *ast.FuncDecl, srcFset *token.FileSet, srcAST *ast.File, targetRel string, tgtPkg *Package, tgtSrc []byte, tgtFset *token.FileSet, tgtAST *ast.File, cross bool, caller string) (*EditFile, string, []string, string) {
	hint := fmt.Sprintf("\"add a function %s in %s that uses %s\"", caller, targetRel, src.Name)
	newName, refusal := p.useFuncName(tgtPkg, targetRel, src.Name, src.Pkg, caller, hint)
	if refusal != "" {
		return nil, "", nil, refusal
	}
	if cross == false && newName == src.Name {
		// useFuncName reads the pre-plan index: a caller named
		// exactly like the callee, in the callee's own
		// package, would collide with the declaration this
		// very plan writes.
		return nil, "", nil, fmt.Sprintf("%s is the new function's own name, and package %s can't hold two of them — pick a different name for the caller.", src.Name, tgtPkg.Name)
	}
	g, refusal := p.newDelegateGen(src, srcDecl, srcFset, srcAST, tgtAST, tgtFset, tgtPkg, cross, newName, targetRel, tgtSrc)
	if refusal != "" {
		return nil, "", nil, refusal
	}
	g.needed = map[string]string{}
	res, refusal := analyzeUseResults(g, srcDecl)
	if refusal != "" {
		return nil, "", nil, refusal
	}
	u := &usesGen{g: g, pkgName: src.Pkg}
	var probeArgs []useArg
	var probeVariadic string
	u.bodyFor = func(qual string) (string, usesExpect, string) {
		text, exp, args, variadic, ref := u.funcBody(qual, src, res)
		if ref == "" {
			probeArgs, probeVariadic = args, variadic
		}
		return text, exp, ref
	}
	if _, _, ref := u.bodyFor(u.formQual(false)); ref != "" {
		return nil, "", nil, ref
	}
	ef, refusal := u.formFile(false)
	if refusal != "" {
		return nil, "", nil, refusal
	}
	var notes []string
	if len(probeArgs) > 0 {
		var pairs []string
		for _, a := range probeArgs {
			pairs = append(pairs, a.label)
		}
		notes = append(notes, fmt.Sprintf("The call starts %s off with placeholder values for its parameters (%s) — they're the types' zero values, there to be replaced.", src.Name, strings.Join(pairs, ", ")))
	} else if probeVariadic == "" {
		notes = append(notes, fmt.Sprintf("The call passes %s no arguments — the declaration this plan writes takes none.", src.Name))
	}
	if probeVariadic != "" {
		notes = append(notes, fmt.Sprintf("%s is variadic in %s — the call passes it none of those; add the ones you need in the body.", probeVariadic, src.Name))
	}
	site := fmt.Sprintf("%s:%d", p.LinkPath(targetRel), insertedLine(ef.After, "func "+newName))
	return ef, site, notes, ""
}

// makeCallHostedCall inserts the bare call at the end of an
// existing function's body: the sentence's host, or — when
// the sentence named none — the target package's main, in the
// target file first and then in any file of its package. The
// returned EditFile is the HOST's file, which may be the
// callee's own (post-declaration bytes) or the target's.
func (p *Project) makeCallHostedCall(req EditRequest, src *Func, srcDecl *ast.FuncDecl, targetRel string, tgtPkg *Package, tgtSrc []byte, tgtFset *token.FileSet, tgtAST *ast.File, cross bool, postBytes map[string][]byte) (*EditFile, string, string, string, []string, string) {
	hostWanted := strings.TrimSpace(req.MakeCallHost)
	explicit := hostWanted != ""
	if !explicit {
		hostWanted = "main"
	}

	// Candidate host files, in order: the target file, then
	// the rest of its package (main may live beside the
	// target rather than in it).
	type hostFile struct {
		rel  string
		src  []byte
		fset *token.FileSet
		ast  *ast.File
	}
	var candidates []hostFile
	seen := map[string]bool{}
	addCandidate := func(rel string, src []byte, fset *token.FileSet, fileAST *ast.File) {
		if seen[rel] {
			return
		}
		seen[rel] = true
		candidates = append(candidates, hostFile{rel, src, fset, fileAST})
	}
	addCandidate(targetRel, tgtSrc, tgtFset, tgtAST)
	for _, f := range tgtPkg.Files {
		if seen[f] {
			continue
		}
		src, fset, fileAST, refusal := p.makeCallTargetState(f, postBytes)
		if refusal != "" {
			continue
		}
		addCandidate(f, src, fset, fileAST)
	}

	var hostDecl *ast.FuncDecl
	var hostIn hostFile
	var methodHost bool
	for _, c := range candidates {
		for _, d := range c.ast.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Name.Name != hostWanted {
				continue
			}
			if fd.Recv != nil {
				methodHost = true
				continue
			}
			hostDecl, hostIn = fd, c
			break
		}
		if hostDecl != nil {
			break
		}
	}
	if hostDecl == nil {
		switch {
		case methodHost:
			return nil, "", "", "", nil, fmt.Sprintf("%s in %s is a method — it runs through a value of its type, and a bare call to %s inside it is a shape I can't place honestly. Name a plain function instead.", hostWanted, p.LinkPath(targetRel), src.Name)
		case explicit:
			return nil, "", "", "", nil, fmt.Sprintf("there's no function %s in %s (or its package) — the call needs a function that already exists to land in. Name one that exists, or give the caller its own function instead: \"make a function named %s in %s and add a function named <Caller> in %s that calls it\".", hostWanted, p.LinkPath(targetRel), src.Name, req.File, targetRel)
		default:
			return nil, "", "", "", nil, fmt.Sprintf("there's no main in %s or its package to hold the call, and I won't invent a caller — say which function should call %s: \"make a function named %s in %s and call it from <Func> in %s\", or give the caller its own function: \"make a function named %s in %s and add a function named <Caller> in %s that calls it\".", p.LinkPath(targetRel), src.Name, src.Name, req.File, targetRel, src.Name, req.File, targetRel)
		}
	}
	if hostDecl.Body == nil {
		return nil, "", "", "", nil, fmt.Sprintf("%s in %s has no body — it's a declaration without one (an assembly or external function), so there's nowhere for the call to land.", hostWanted, p.LinkPath(hostIn.rel))
	}

	// The call's text, qualified across packages the way the
	// uses route qualifies: the host file's existing import
	// of the callee's package when it has one, its declared
	// name otherwise, with the import added by the plan.
	call := src.Name + "()"
	var specs []string
	if cross {
		srcPath := p.Module
		if src.PkgDir != "" {
			srcPath = p.Module + "/" + src.PkgDir
		}
		if srcPath == "" || strings.HasPrefix(srcPath, "/") {
			return nil, "", "", "", nil, "I couldn't work out the new function's package import path (the project has no go.mod module line I can read), so I can't write the import honestly."
		}
		qual := src.Pkg
		existing := ""
		hasImport := false
		for _, is := range hostIn.ast.Imports {
			_, eff := importSpecOf(is)
			path, _ := strconv.Unquote(is.Path.Value)
			if path == srcPath {
				hasImport = true
				existing = eff
			}
		}
		if hasImport {
			if existing == "." || existing == "_" {
				return nil, "", "", "", nil, fmt.Sprintf("%s imports %s with a %s import, which gives the package no name I can call through — give that import its plain name first, then ask again.", p.LinkPath(hostIn.rel), srcPath, map[string]string{".": "dot", "_": "blank"}[existing])
			}
			qual = existing
		} else {
			specs = append(specs, delegateImportSpec(srcPath, qual, src.Pkg))
		}
		call = qual + "." + src.Name + "()"
	}

	// Insert at the end of the host's body: on its own line,
	// just before the closing brace's line — or, for a body
	// written on one line, by opening the braces up.
	edits := conceptImportEdits(hostIn.ast, hostIn.fset, hostIn.src, specs)
	lbraceLine := lineOfPos(hostIn.fset, hostDecl.Body.Lbrace)
	rbraceLine := lineOfPos(hostIn.fset, hostDecl.Body.Rbrace)
	if lbraceLine == rbraceLine {
		edits = append(edits, textEdit{
			start: offOf(hostIn.fset, hostDecl.Body.Lbrace) + 1,
			end:   offOf(hostIn.fset, hostDecl.Body.Rbrace),
			text:  "\n\t" + call + "\n",
		})
	} else {
		at := lineStartOffset(hostIn.src, rbraceLine)
		edits = append(edits, textEdit{start: at, end: at, text: "\t" + call + "\n"})
	}
	after := applyTextEdits(hostIn.src, edits)
	ef := finishFile(hostIn.rel, hostIn.src, after)

	// Self-check: the host still exists and now calls the
	// callee, in the assembled bytes.
	if !makeCallHostedInBytes(after, hostWanted, src.Name) {
		return nil, "", "", "", nil, "internal error in my assembler: the assembled file doesn't show the call inside its host — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written."
	}
	site := fmt.Sprintf("%s:%d", p.LinkPath(hostIn.rel), insertedLine(after, call))
	var notes []string
	if hostIn.rel != targetRel {
		notes = append(notes, fmt.Sprintf("The %s that holds the call lives in %s, not %s — the call landed there, in the same package.", hostWanted, p.LinkPath(hostIn.rel), p.LinkPath(targetRel)))
	}
	return &ef, hostIn.rel, hostWanted, site, notes, ""
}

// makeCallHostedInBytes reports whether the named host
// function in these bytes calls the callee — the in-memory
// twin of verifyDelegateCall, for the self-check before a
// plan exists.
func makeCallHostedInBytes(after []byte, host, callee string) bool {
	fset, fileAST, err := parseGoSrc("makecall.go", after)
	if err != nil || fileAST == nil {
		return false
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != host || fd.Body == nil {
			continue
		}
		found := false
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				txt := exprText(fset, call.Fun)
				if txt == callee || strings.HasSuffix(txt, "."+callee) {
					found = true
				}
			}
			return !found
		})
		return found
	}
	return false
}
