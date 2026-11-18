package analyze

// The scaffold beyond the delegate: the extension that lets
// the other assembled shapes whose edit produces or modifies
// a function with an index-known signature carry the same
// table-driven scaffold, through the same generator — no
// second type table, no second honesty standard.
//
//   - EditAddFunc (clone): the new function is a real function
//     with the sibling's real signature, appended to the
//     sibling's own file — scaffold it exactly as the delegate
//     scaffolds its wrapper, aimed at the NEW name.
//   - EditAddCase: the scaffold aims at the function holding
//     the switch, as it stands after the edit (a case edit
//     never changes a signature). Its header and plan note say
//     the scaffold accompanies the new case; the generated
//     cases exercise the function's inputs generally — no
//     case pretends to pin the new branch's behavior.
//   - EditAddField: the scaffold aims at one function the
//     index can single out for the struct — its New<Type>
//     constructor when it has one, else its most-called method
//     (fieldScaffoldTarget). When the index singles out
//     nothing, there is no scaffold and the plan says why: a
//     scaffold needs a function to call, and inventing a
//     target would be fiction.
//
// The webserver shape, creations, rename, register, and the
// error fixes get no scaffold and no note: none of them
// produces a function a scaffold could honestly aim at (the
// webserver's only function is main serving ListenAndServe —
// a generated test against it would block, and the template
// is already the known quantity).

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strconv"
	"strings"
)

// scaffoldTarget is one function a scaffold can aim at: its
// index entry, its declaration re-read from its own file, and
// the names the scaffold uses for it.
type scaffoldTarget struct {
	fn      *Func
	decl    *ast.FuncDecl
	fset    *token.FileSet
	fileAST *ast.File
	name    string // the name the test calls (the clone's NEW name)
	display string // prose name: "Parse", "Config.Save"

	// entry, when non-empty, says why the called function is
	// the program's entry under the name the test would use
	// ("main is the program's entry point") — such a target
	// gets a note, never a scaffold, because calling it would
	// run the program. Callers set it for the function AS
	// CALLED: a clone of init under a fresh name is an
	// ordinary function and leaves it empty, while a clone
	// NAMED main sets it.
	entry string
}

// entryReason says why a function the test would call under
// this name is the program's entry, or "" when it isn't.
func entryReason(name string) string {
	switch name {
	case "main":
		return "main is the program's entry point"
	case "init":
		return "init runs on its own before main"
	}
	return ""
}

// scaffoldDisplay names a function for prose: a method is
// "Config.Save", a plain function its bare name.
func scaffoldDisplay(fn *Func) string {
	if fn.Receiver != "" {
		return strings.Trim(fn.Receiver, "*") + "." + fn.Name
	}
	return fn.Name
}

// scaffoldTargetFor re-reads fn's declaration from its file —
// the planners' own line-match pattern.
func (p *Project) scaffoldTargetFor(fn *Func) (scaffoldTarget, string) {
	_, fset, fileAST, refusal := p.readProjectFile(fn.File)
	if refusal != "" || fileAST == nil {
		return scaffoldTarget{}, fmt.Sprintf("I couldn't re-read %s in %s to build one.", fn.Display(), p.LinkPath(fn.File))
	}
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == fn.Name && lineOfPos(fset, fd.Pos()) == fn.Line {
			return scaffoldTarget{
				fn: fn, decl: fd, fset: fset, fileAST: fileAST,
				name: fn.Name, display: scaffoldDisplay(fn),
				entry: entryReason(fn.Name),
			}, ""
		}
	}
	return scaffoldTarget{}, fmt.Sprintf("I couldn't re-read %s in %s to build one.", fn.Display(), p.LinkPath(fn.File))
}

// receiverTypeParams returns the type parameter names a
// method declaration's receiver carries — in Go they are
// written at the receiver itself (func (s Set[T]) Add), so the
// declaration holds them; no type lookup is needed.
func receiverTypeParams(decl *ast.FuncDecl) []string {
	if decl.Recv == nil || len(decl.Recv.List) == 0 {
		return nil
	}
	var out []string
	collect := func(e ast.Expr) {
		switch t := e.(type) {
		case *ast.IndexExpr:
			if id, ok := t.Index.(*ast.Ident); ok {
				out = append(out, id.Name)
			}
		case *ast.IndexListExpr:
			for _, ix := range t.Indices {
				if id, ok := ix.(*ast.Ident); ok {
					out = append(out, id.Name)
				}
			}
		}
	}
	e := decl.Recv.List[0].Type
	if star, ok := e.(*ast.StarExpr); ok {
		e = star.X
	}
	collect(e)
	return out
}

// scaffoldGenFor builds the rendering context the scaffold
// builder needs for one function under test in its own
// package — a delegateGen with cross=false, the function's own
// file as both the signature's source context and (until the
// builder swaps in the test file's situation) the target.
// planDelegate's gen is the same struct aimed across
// packages; same-package targets need no re-qualification,
// only the honest rendering of the function's own signature.
func (p *Project) scaffoldGenFor(tgt scaffoldTarget) *delegateGen {
	g := &delegateGen{
		p:          p,
		src:        tgt.fn,
		srcDecl:    tgt.decl,
		srcFset:    tgt.fset,
		srcAST:     tgt.fileAST,
		cross:      false,
		wrapper:    tgt.name,
		typeParams: map[string]bool{},
		srcTop:     map[string]bool{},
		srcImports: map[string]string{},
		tgtImports: map[string]string{},
	}
	if sp := p.pkgByDir(tgt.fn.PkgDir); sp != nil {
		for _, fn := range sp.Funcs {
			if fn.Receiver == "" {
				g.srcTop[fn.Name] = true
			}
		}
		for _, t := range sp.Types {
			g.srcTop[t.Name] = true
		}
		for _, v := range sp.Vars {
			g.srcTop[v.Name] = true
		}
		g.tgtPkg = sp
	}
	if tgt.decl.Type.TypeParams != nil {
		for _, f := range tgt.decl.Type.TypeParams.List {
			for _, n := range f.Names {
				g.typeParams[n.Name] = true
			}
		}
	}
	for _, n := range receiverTypeParams(tgt.decl) {
		g.typeParams[n] = true
		g.recvTypeParams = append(g.recvTypeParams, n)
	}
	for _, is := range tgt.fileAST.Imports {
		_, eff := importSpecOf(is)
		path, _ := strconv.Unquote(is.Path.Value)
		g.srcImports[eff] = path
		g.tgtImports[path] = eff
	}
	return g
}

// attachScaffold builds tgt's scaffold and appends it to the
// plan — the file after the edit's own files, the note after
// any notes already present. A nil file with a note is the
// honest-omission path (generic target, name already taken,
// self-check failure); an entry-point target (tgt.entry set)
// gets its note here because every caller shares the reason.
func (p *Project) attachScaffold(plan *EditPlan, tgt scaffoldTarget, pres scaffoldPresentation) {
	if tgt.entry != "" {
		plan.Notes = append(plan.Notes, fmt.Sprintf("No test scaffold added: %s — a test can't call it like an ordinary function, so there's nothing honest for a scaffold to aim at.", tgt.entry))
		return
	}
	tgtPkg := p.pkgByDir(tgt.fn.PkgDir)
	if tgtPkg == nil {
		return // an indexed function always has its package; silence over fiction
	}
	testRel := scaffoldTestRel(tgtPkg, tgt.fn.File)
	for _, f := range plan.Files {
		if f.Rel == testRel {
			plan.Notes = append(plan.Notes, fmt.Sprintf("No test scaffold added: this edit already rewrites %s, and stacking a scaffold onto the same file in the same plan could clobber it.", testRel))
			return
		}
	}
	g := p.scaffoldGenFor(tgt)
	ef, note := g.planScaffold(tgtPkg, tgt.fn.File, pres)
	if ef != nil {
		plan.Files = append(plan.Files, *ef)
	}
	if note != "" {
		plan.Notes = append(plan.Notes, note)
	}
}

// fieldScaffoldTarget picks the one function an add-field
// scaffold can honestly aim at for struct t, or explains why
// there is none. The rule, first match wins:
//
//  1. The struct's constructor — a plain function named
//     New<Type> in the struct's own package. It is the
//     canonical way the struct gets built, and the field
//     addition changes what it builds.
//  2. The struct's most-called method, by the index's caller
//     counts — outright winner only. A single method wins by
//     being the only candidate; a tie at the top singles out
//     nothing, and the note says so rather than guess.
func (p *Project) fieldScaffoldTarget(t *Type) (target *Func, why string, reason string) {
	var ctor *Func
	var methods []*Func
	for _, pkg := range p.Packages {
		if pkg.Dir != t.PkgDir {
			continue
		}
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			if fn.Receiver == "" && fn.Name == "New"+t.Name && ctor == nil {
				ctor = fn
			}
			if strings.Trim(fn.Receiver, "*") == t.Name {
				methods = append(methods, fn)
			}
		}
	}
	if ctor != nil {
		return ctor, fmt.Sprintf("%s's constructor", t.Name), ""
	}
	if len(methods) == 0 {
		return nil, "", fmt.Sprintf("%s has no New%s constructor and no methods — a scaffold needs one function that builds or uses the struct, and I won't invent a target.", t.Name, t.Name)
	}
	if len(methods) == 1 {
		return methods[0], fmt.Sprintf("%s's only method", t.Name), ""
	}
	best := -1
	var winners []*Func
	for _, m := range methods {
		switch n := len(m.Callers); {
		case n > best:
			best, winners = n, []*Func{m}
		case n == best:
			winners = append(winners, m)
		}
	}
	if len(winners) == 1 {
		return winners[0], fmt.Sprintf("%s's most-called method (%d callers in the index, more than any other)", t.Name, best), ""
	}
	names := make([]string, 0, len(winners))
	for _, w := range winners {
		names = append(names, w.Name)
	}
	sort.Strings(names)
	return nil, "", fmt.Sprintf("several of %s's methods (%s) are called equally often (%d callers each), so the index can't single one out — a scaffold needs one function to aim at, and I won't guess.", t.Name, strings.Join(names, ", "), best)
}
