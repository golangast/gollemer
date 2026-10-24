package analyze

// Error -> fix: the mechanical half of the error help. errhelp
// already translates a pasted compiler error into plain words; this
// planner asks the narrower question — is there exactly ONE
// mechanical edit that repairs this error in this project? When
// there is, it becomes an ordinary EditPlan and rides the same
// diff -> [y/n] -> build -> verify spine as every other edit. When
// there isn't (type mismatches, argument counts, anything with two
// defensible repairs), the planner has nothing and the translation
// stands alone, exactly as before.
//
// The classes, each decided on whether the correct edit is unique:
//
//	(a) imported and not used      -> remove the import. Unique.
//	(b) declared and not used      -> remove the declaration, only
//	    when the initializer cannot do anything (no calls, no
//	    channel receives) and nothing later assigns to the name.
//	(c) undefined: X, one near     -> fix the spelling at the error
//	    match in scope                site. Unique by construction.
//	(d) undefined: X               -> X is a project package's name
//	    (add the import), or the one exported symbol named X in
//	    another project package (add the import, qualify the use).
//	(e) missing return             -> append the zero-value return
//	    the signature promises; bare `return` when every result is
//	    named. Signatures whose zero can't be derived exactly
//	    (types defined outside the project, generics) are refused.

import (
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Error fix classes the chat's compiler-line parser can request.
const (
	ErrorFixUnusedImport  = "unused_import"
	ErrorFixUnusedVar     = "unused_var"
	ErrorFixUndefinedName = "undefined_name"
	ErrorFixMissingReturn = "missing_return"
)

// ErrorFixRequest is one parsed compiler error line.
type ErrorFixRequest struct {
	Class string
	File  string // project-relative path of the file the error names
	Line  int
	Col   int
	Name  string // import path (unused_import) or symbol name
}

// PlanErrorFix assembles the unique mechanical repair for one
// compiler error, or reports (nil, "") when this error has none —
// the chat stays silent then and the translation stands alone. A
// non-empty refusal means the class applied but couldn't be made
// safe; the chat treats it the same (silence), it exists for tests
// and debugging.
func (p *Project) PlanErrorFix(req ErrorFixRequest) (*EditPlan, string) {
	switch req.Class {
	case ErrorFixUnusedImport:
		return p.planUnusedImport(req)
	case ErrorFixUnusedVar:
		return p.planUnusedVar(req)
	case ErrorFixUndefinedName:
		return p.planUndefinedName(req)
	case ErrorFixMissingReturn:
		return p.planMissingReturn(req)
	}
	return nil, ""
}

// errorFixTarget loads and checks the file every class works on:
// it must be a project file and not generated or vendored.
func (p *Project) errorFixTarget(req ErrorFixRequest) ([]byte, *token.FileSet, *ast.File, string) {
	if req.File == "" {
		return nil, nil, nil, ""
	}
	if why, blocked := p.fileBlocked(req.File); blocked {
		return nil, nil, nil, fmt.Sprintf("the error is in %s: %s", p.LinkPath(req.File), why)
	}
	src, fset, fileAST, refusal := p.readProjectFile(req.File)
	if refusal != "" {
		return nil, nil, nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, nil, nil, fmt.Sprintf("the error is in %s, a generated file (marked DO NOT EDIT)", p.LinkPath(req.File))
	}
	return src, fset, fileAST, ""
}

// pkgOfFile finds the analyzed package that owns a project file.
func (p *Project) pkgOfFile(rel string) *Package {
	for _, pkg := range p.Packages {
		for _, f := range pkg.Files {
			if f == rel {
				return pkg
			}
		}
	}
	// Fall back to the directory: a file the walk skipped (a parse
	// error when the index was built) still belongs somewhere.
	dir := filepath.ToSlash(filepath.Dir(rel))
	if dir == "." {
		dir = ""
	}
	for _, pkg := range p.Packages {
		if pkg.Dir == dir {
			return pkg
		}
	}
	return nil
}

// removeDeclLines builds the edit that deletes whole lines
// [firstLine, lastLine] of src.
func removeLinesEdit(src []byte, firstLine, lastLine int) textEdit {
	return textEdit{
		start: lineStartOffset(src, firstLine),
		end:   lineStartOffset(src, lastLine+1),
		text:  "",
	}
}

// insertImportEdits writes the splice that adds ONE import spec to
// a file, following the shapes planImportEdits uses: into the
// paren block when there is one, wrapping a lone import into a
// block, or a fresh block after the package clause.
func insertImportEdits(fset *token.FileSet, fileAST *ast.File, src []byte, spec string) []textEdit {
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.IMPORT {
			continue
		}
		if gd.Lparen.IsValid() {
			rbraceLine := lineOfPos(fset, gd.Rparen)
			return []textEdit{{
				start: lineStartOffset(src, rbraceLine),
				end:   lineStartOffset(src, rbraceLine),
				text:  "\t" + spec + "\n",
			}}
		}
		declLine := lineOfPos(fset, gd.Pos())
		declEnd := lineOfPos(fset, gd.End()-1)
		old := strings.Join(strings.Split(string(src), "\n")[declLine-1:declEnd], "\n")
		trimmed := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(old), "import"))
		block := "import (\n\t" + trimmed + "\n\t" + spec + "\n)"
		return []textEdit{{
			start: lineStartOffset(src, declLine),
			end:   lineStartOffset(src, declEnd+1),
			text:  block + "\n",
		}}
	}
	pkgLine := lineOfPos(fset, fileAST.Name.Pos())
	return []textEdit{{
		start: lineStartOffset(src, pkgLine+1),
		end:   lineStartOffset(src, pkgLine+1),
		text:  "\nimport (\n\t" + spec + "\n)\n",
	}}
}

// importSpecText renders an import spec for a project package,
// naming it explicitly when the package's real name differs from
// its path's last element.
func importSpecText(pkg *Package, path string) string {
	base := path[strings.LastIndex(path, "/")+1:]
	if pkg != nil && pkg.Name != "" && pkg.Name != base {
		return pkg.Name + " " + strconv.Quote(path)
	}
	return strconv.Quote(path)
}

// fileImports returns a file's imports as (path -> effective name).
func fileImports(fileAST *ast.File) map[string]string {
	out := map[string]string{}
	for _, is := range fileAST.Imports {
		spec, eff := importSpecOf(is)
		_ = spec
		path, _ := strconv.Unquote(is.Path.Value)
		out[path] = eff
	}
	return out
}

// verifyFileAfter re-reads one file of the re-analyzed project and
// hands it to a check. Used by every error-fix Verify closure.
func verifyFileAfter(after *Project, rel string, check func(fset *token.FileSet, f *ast.File) bool) bool {
	_, fset, fileAST, refusal := (&Project{Root: after.Root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	return check(fset, fileAST)
}

// --- (a) imported and not used ---------------------------------------------

func (p *Project) planUnusedImport(req ErrorFixRequest) (*EditPlan, string) {
	src, fset, fileAST, refusal := p.errorFixTarget(req)
	if refusal != "" {
		return nil, refusal
	}
	var target *ast.ImportSpec
	var decl *ast.GenDecl
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.IMPORT {
			continue
		}
		for _, spec := range gd.Specs {
			is, ok := spec.(*ast.ImportSpec)
			if !ok {
				continue
			}
			if path, _ := strconv.Unquote(is.Path.Value); path == req.Name {
				target, decl = is, gd
			}
		}
	}
	if target == nil {
		return nil, fmt.Sprintf("there is no import of %q in %s — the code may have changed since that error was copied", req.Name, p.LinkPath(req.File))
	}
	var edit textEdit
	if decl.Lparen.IsValid() && len(decl.Specs) > 1 {
		edit = removeLinesEdit(src, lineOfPos(fset, target.Pos()), lineOfPos(fset, target.End()-1))
	} else {
		edit = removeLinesEdit(src, lineOfPos(fset, decl.Pos()), lineOfPos(fset, decl.End()-1))
	}
	after := applyTextEdits(src, []textEdit{edit})
	ef := finishFile(req.File, src, after)
	path, rel := req.Name, req.File
	return &EditPlan{
		Summary:    fmt.Sprintf("Remove the unused import %q from %s — nothing in the file uses it, and Go refuses to compile until imports match reality.", path, p.LinkPath(rel)),
		VerifyDesc: fmt.Sprintf("the unused import %q is gone from %s", path, rel),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), lineOfPos(fset, target.Pos())),
		Files:      []EditFile{ef},
		verify: func(afterP *Project) bool {
			return verifyFileAfter(afterP, rel, func(_ *token.FileSet, f *ast.File) bool {
				_, stillThere := fileImports(f)[path]
				return !stillThere
			})
		},
	}, ""
}

// --- (b) declared and not used ----------------------------------------------

// pureExpr reports whether evaluating e cannot do anything: no
// calls, no channel receives. A function literal is pure to CREATE
// (its body doesn't run), so the check doesn't descend into one.
func pureExpr(e ast.Expr) bool {
	switch x := e.(type) {
	case *ast.CallExpr:
		return false
	case *ast.FuncLit:
		return true
	case *ast.UnaryExpr:
		if x.Op == token.ARROW {
			return false
		}
		return pureExpr(x.X)
	case *ast.BinaryExpr:
		return pureExpr(x.X) && pureExpr(x.Y)
	case *ast.ParenExpr:
		return pureExpr(x.X)
	case *ast.SelectorExpr:
		return pureExpr(x.X)
	case *ast.IndexExpr:
		return pureExpr(x.X) && pureExpr(x.Index)
	case *ast.IndexListExpr:
		if !pureExpr(x.X) {
			return false
		}
		for _, ix := range x.Indices {
			if !pureExpr(ix) {
				return false
			}
		}
		return true
	case *ast.SliceExpr:
		return pureExpr(x.X) &&
			(x.Low == nil || pureExpr(x.Low)) &&
			(x.High == nil || pureExpr(x.High)) &&
			(x.Max == nil || pureExpr(x.Max))
	case *ast.StarExpr:
		return pureExpr(x.X)
	case *ast.TypeAssertExpr:
		return pureExpr(x.X)
	case *ast.CompositeLit:
		for _, elt := range x.Elts {
			if kv, ok := elt.(*ast.KeyValueExpr); ok {
				if !pureExpr(kv.Value) {
					return false
				}
				continue
			}
			if !pureExpr(elt) {
				return false
			}
		}
		return true
	}
	return true
}

// laterAssignments reports whether anything after position `from`
// in the function assigns to (or increments) the name — those uses
// would break if the declaration vanished. Re-declarations (:=) in
// inner scopes create a new variable and don't count.
func laterAssignments(body *ast.BlockStmt, name string, from token.Pos) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found || n == nil {
			return false
		}
		switch x := n.(type) {
		case *ast.AssignStmt:
			if x.Pos() > from && x.Tok != token.DEFINE {
				for _, lhs := range x.Lhs {
					if id, ok := lhs.(*ast.Ident); ok && id.Name == name {
						found = true
					}
				}
			}
		case *ast.IncDecStmt:
			if x.Pos() > from {
				if id, ok := x.X.(*ast.Ident); ok && id.Name == name {
					found = true
				}
			}
		}
		return true
	})
	return found
}

// stmtOwnLines checks that [startOff, endOff) is the only thing on
// its lines besides whitespace and a trailing comment, so deleting
// whole lines deletes exactly the statement.
func stmtOwnLines(src []byte, fset *token.FileSet, n ast.Node) bool {
	startLine := lineOfPos(fset, n.Pos())
	endLine := lineOfPos(fset, n.End()-1)
	lineStart := lineStartOffset(src, startLine)
	if strings.TrimSpace(string(src[lineStart:offOf(fset, n.Pos())])) != "" {
		return false
	}
	lineEnd := lineStartOffset(src, endLine+1)
	rest := strings.TrimSpace(string(src[offOf(fset, n.End()):lineEnd]))
	return rest == "" || strings.HasPrefix(rest, "//")
}

func (p *Project) planUnusedVar(req ErrorFixRequest) (*EditPlan, string) {
	src, fset, fileAST, refusal := p.errorFixTarget(req)
	if refusal != "" {
		return nil, refusal
	}
	// The enclosing function: the smallest FuncDecl spanning the
	// error line.
	var fn *ast.FuncDecl
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		if lineOfPos(fset, fd.Pos()) <= req.Line && req.Line <= lineOfPos(fset, fd.End()-1) {
			if fn == nil || fd.Pos() > fn.Pos() {
				fn = fd
			}
		}
	}
	if fn == nil {
		return nil, "that error doesn't point inside a function in this file — unused-variable errors only happen to locals, so there's nothing safe to remove"
	}
	cannot := func(why string) (*EditPlan, string) {
		return nil, fmt.Sprintf("I won't remove %s automatically: %s", req.Name, why)
	}
	var removeEdit textEdit
	var declEnd token.Pos
	found := false
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		if found {
			return false
		}
		switch x := n.(type) {
		case *ast.AssignStmt:
			if x.Tok != token.DEFINE || lineOfPos(fset, x.Pos()) != req.Line {
				return true
			}
			idx := -1
			for i, lhs := range x.Lhs {
				if id, ok := lhs.(*ast.Ident); ok && id.Name == req.Name {
					idx = i
				}
			}
			if idx < 0 {
				return true
			}
			found = true
			if len(x.Lhs) != 1 || len(x.Rhs) != 1 {
				refusal = "it is declared together with other values, and removing just this one would change what the others get"
				return false
			}
			if !pureExpr(x.Rhs[0]) {
				refusal = "its value comes from a call or a channel, which might do something — deleting the line could change what the program does"
				return false
			}
			if !stmtOwnLines(src, fset, x) {
				refusal = "it shares its line with other code, so removing it cleanly isn't mechanical"
				return false
			}
			declEnd = x.End()
			removeEdit = removeLinesEdit(src, lineOfPos(fset, x.Pos()), lineOfPos(fset, x.End()-1))
		case *ast.GenDecl:
			if x.Tok != token.VAR {
				return true
			}
			for _, spec := range x.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok || lineOfPos(fset, vs.Pos()) != req.Line {
					continue
				}
				match := false
				for _, nm := range vs.Names {
					if nm.Name == req.Name {
						match = true
					}
				}
				if !match {
					continue
				}
				found = true
				if len(vs.Names) != 1 {
					refusal = "it is declared together with other names on one line"
					return false
				}
				if len(vs.Values) > 1 {
					refusal = "its declaration has more values than names, which isn't a shape I can remove safely"
					return false
				}
				if len(vs.Values) == 1 && !pureExpr(vs.Values[0]) {
					refusal = "its value comes from a call or a channel, which might do something — deleting the line could change what the program does"
					return false
				}
				declEnd = vs.End()
				if x.Lparen.IsValid() && len(x.Specs) > 1 {
					if !stmtOwnLines(src, fset, vs) {
						refusal = "it shares its line with other code, so removing it cleanly isn't mechanical"
						return false
					}
					removeEdit = removeLinesEdit(src, lineOfPos(fset, vs.Pos()), lineOfPos(fset, vs.End()-1))
				} else {
					if !stmtOwnLines(src, fset, x) {
						refusal = "it shares its line with other code, so removing it cleanly isn't mechanical"
						return false
					}
					removeEdit = removeLinesEdit(src, lineOfPos(fset, x.Pos()), lineOfPos(fset, x.End()-1))
				}
				return false
			}
		}
		return true
	})
	if refusal != "" {
		return cannot(refusal)
	}
	if !found {
		return nil, fmt.Sprintf("I couldn't find the declaration of %s at %s:%d — the code may have changed since that error was copied", req.Name, p.LinkPath(req.File), req.Line)
	}
	if laterAssignments(fn.Body, req.Name, declEnd) {
		return cannot("it is assigned again later in the function, so the declaration is load-bearing — deleting it would break those lines")
	}
	after := applyTextEdits(src, []textEdit{removeEdit})
	ef := finishFile(req.File, src, after)
	name, rel := req.Name, req.File
	fnStartLine := lineOfPos(fset, fn.Pos())
	return &EditPlan{
		Summary:    fmt.Sprintf("Remove the declaration of %s at %s:%d — it is created there and never read, and its value can't do anything on its own.", name, p.LinkPath(rel), req.Line),
		VerifyDesc: fmt.Sprintf("the unused variable %s is gone from %s", name, rel),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), req.Line),
		Files:      []EditFile{ef},
		verify: func(afterP *Project) bool {
			return verifyFileAfter(afterP, rel, func(fset2 *token.FileSet, f *ast.File) bool {
				for _, d := range f.Decls {
					fd, ok := d.(*ast.FuncDecl)
					if !ok || fd.Body == nil || lineOfPos(fset2, fd.Pos()) != fnStartLine {
						continue
					}
					stillThere := false
					ast.Inspect(fd.Body, func(n ast.Node) bool {
						switch x := n.(type) {
						case *ast.AssignStmt:
							if x.Tok == token.DEFINE {
								for _, lhs := range x.Lhs {
									if id, ok := lhs.(*ast.Ident); ok && id.Name == name {
										stillThere = true
									}
								}
							}
						case *ast.ValueSpec:
							for _, nm := range x.Names {
								if nm.Name == name {
									stillThere = true
								}
							}
						}
						return !stillThere
					})
					return !stillThere
				}
				return false
			})
		},
	}, ""
}

// --- (c) + (d) undefined: X ---------------------------------------------------

var builtinNames = []string{
	"append", "cap", "clear", "close", "complex", "copy", "delete",
	"imag", "len", "make", "max", "min", "new", "panic", "print",
	"println", "real", "recover",
	"any", "bool", "byte", "comparable", "error",
	"complex64", "complex128", "float32", "float64",
	"int", "int8", "int16", "int32", "int64",
	"rune", "string",
	"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
	"true", "false", "iota", "nil",
}

// nearName reports whether cand is a plausible intended spelling of
// typed: same name in different case, or a small edit distance.
func nearName(typed, cand string) bool {
	if typed == cand {
		return false
	}
	if strings.EqualFold(typed, cand) {
		return true
	}
	if len(typed) < 3 || len(cand) < 3 {
		return false
	}
	d := levenshtein(strings.ToLower(typed), strings.ToLower(cand))
	if len(typed) <= 4 || len(cand) <= 4 {
		return d <= 1
	}
	return d <= 2
}

// identSite locates an identifier occurrence in a parsed file.
type identSite struct {
	id       *ast.Ident
	start    int
	end      int
	isSel    bool // the .Name of a selector (a member, not a name in scope)
	isDef    bool
	encloses ast.Node // innermost FuncDecl or FuncLit containing it
}

// findIdentSite finds the identifier the compiler error points at:
// exact line+column first, then the only same-named identifier on
// the line.
func findIdentSite(fset *token.FileSet, fileAST *ast.File, name string, line, col int) *identSite {
	var stack []ast.Node
	var enclosing ast.Node
	var exact, onLine *identSite
	countOnLine := 0
	ast.Inspect(fileAST, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return false
		}
		stack = append(stack, n)
		switch n.(type) {
		case *ast.FuncDecl, *ast.FuncLit:
			enclosing = n
		}
		id, ok := n.(*ast.Ident)
		if !ok || id.Name != name {
			return true
		}
		pos := fset.Position(id.Pos())
		if pos.Line != line {
			return true
		}
		site := &identSite{
			id:    id,
			start: offOf(fset, id.Pos()),
			end:   offOf(fset, id.End()),
		}
		if len(stack) >= 2 {
			if sel, ok := stack[len(stack)-2].(*ast.SelectorExpr); ok && sel.Sel == id {
				site.isSel = true
			}
		}
		// Was this ident defined here? It's a Def if it names a
		// declaration rather than a use — detectable syntactically
		// for the shapes that matter (its parent declares it).
		site.encloses = enclosing
		countOnLine++
		onLine = site
		if pos.Column == col {
			exact = site
		}
		return true
	})
	if exact != nil {
		return exact
	}
	if countOnLine == 1 {
		return onLine
	}
	return nil
}

// funcScopeNames collects the names in scope inside one function:
// parameters, results, the receiver, and every local declared in
// its own body (nested closures keep their own scope and are not
// descended into).
func funcScopeNames(fn ast.Node) map[string]bool {
	out := map[string]bool{}
	var ft *ast.FuncType
	var body *ast.BlockStmt
	switch f := fn.(type) {
	case *ast.FuncDecl:
		ft = f.Type
		body = f.Body
		if f.Recv != nil {
			for _, field := range f.Recv.List {
				for _, nm := range field.Names {
					out[nm.Name] = true
				}
			}
		}
	case *ast.FuncLit:
		ft = f.Type
		body = f.Body
	}
	if ft != nil {
		for _, list := range []*ast.FieldList{ft.Params, ft.Results} {
			if list == nil {
				continue
			}
			for _, field := range list.List {
				for _, nm := range field.Names {
					out[nm.Name] = true
				}
			}
		}
	}
	if body == nil {
		return out
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncLit:
			return false // a closure's locals are its own scope
		case *ast.AssignStmt:
			if x.Tok == token.DEFINE {
				for _, lhs := range x.Lhs {
					if id, ok := lhs.(*ast.Ident); ok {
						out[id.Name] = true
					}
				}
			}
		case *ast.ValueSpec:
			for _, nm := range x.Names {
				out[nm.Name] = true
			}
		case *ast.RangeStmt:
			if id, ok := x.Key.(*ast.Ident); ok {
				out[id.Name] = true
			}
			if id, ok := x.Value.(*ast.Ident); ok {
				out[id.Name] = true
			}
		}
		return true
	})
	return out
}

func (p *Project) planUndefinedName(req ErrorFixRequest) (*EditPlan, string) {
	if strings.Contains(req.Name, ".") {
		return nil, "that undefined name is a package member (pkg.Name) — fixing those mechanically isn't in this build"
	}
	src, fset, fileAST, refusal := p.errorFixTarget(req)
	if refusal != "" {
		return nil, refusal
	}
	site := findIdentSite(fset, fileAST, req.Name, req.Line, req.Col)
	if site == nil {
		return nil, fmt.Sprintf("I couldn't find %q at %s:%d — the code may have changed since that error was copied", req.Name, p.LinkPath(req.File), req.Line)
	}
	if site.isSel {
		return nil, "that name is a field or method after a dot — its fix depends on the value's type, which isn't mechanical"
	}
	filePkg := p.pkgOfFile(req.File)

	// (c) spelling: gather every name visible at the site.
	scope := map[string]string{} // name -> plain-words provenance
	addScope := func(name, desc string) {
		if _, dup := scope[name]; !dup {
			scope[name] = desc
		}
	}
	if site.encloses != nil {
		for name := range funcScopeNames(site.encloses) {
			addScope(name, "a variable in this function")
		}
	}
	if filePkg != nil {
		for _, fn := range filePkg.Funcs {
			if fn.Receiver == "" {
				addScope(fn.Name, fmt.Sprintf("a function in this package (%s:%d)", p.LinkPath(fn.File), fn.Line))
			}
		}
		for _, ty := range filePkg.Types {
			addScope(ty.Name, fmt.Sprintf("a type in this package (%s:%d)", p.LinkPath(ty.File), ty.Line))
		}
		for _, v := range filePkg.Vars {
			addScope(v.Name, fmt.Sprintf("a package-level name in this package (%s:%d)", p.LinkPath(v.File), v.Line))
		}
	}
	for _, eff := range fileImports(fileAST) {
		addScope(eff, "the name of a package this file imports")
	}
	for _, b := range builtinNames {
		addScope(b, "a Go builtin")
	}
	var near []string
	for name := range scope {
		if nearName(req.Name, name) {
			near = append(near, name)
		}
	}
	sort.Strings(near)
	if len(near) == 1 {
		fixed := near[0]
		edits := []textEdit{{start: site.start, end: site.end, text: fixed}}
		after := applyTextEdits(src, edits)
		ef := finishFile(req.File, src, after)
		oldName, rel := req.Name, req.File
		beforeCount := countIdents(fileAST, oldName)
		fixedBefore := countIdents(fileAST, fixed)
		return &EditPlan{
			Summary:    fmt.Sprintf("Fix the spelling at %s:%d — %s isn't defined, but %s is %s, and it's the only close match in scope.", p.LinkPath(rel), req.Line, oldName, fixed, scope[fixed]),
			VerifyDesc: fmt.Sprintf("%s at %s:%d now reads %s", oldName, rel, req.Line, fixed),
			Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), req.Line),
			Files:      []EditFile{ef},
			verify: func(afterP *Project) bool {
				return verifyFileAfter(afterP, rel, func(_ *token.FileSet, f *ast.File) bool {
					return countIdents(f, oldName) == beforeCount-1 && countIdents(f, fixed) == fixedBefore+1
				})
			},
		}, ""
	}
	if len(near) > 1 {
		return nil, fmt.Sprintf("%s could be a misspelling of %s — more than one close match, so there is no single mechanical fix", req.Name, strings.Join(near, " or "))
	}

	// (d) the import fixes. (d1) the name IS a project package.
	imports := fileImports(fileAST)
	var pkgNameMatches []*Package
	for _, pkg := range p.Packages {
		if pkg.Name == req.Name && pkg != filePkg {
			pkgNameMatches = append(pkgNameMatches, pkg)
		}
	}
	// (d2) the name is an exported symbol of exactly one other
	// project package.
	symPkgs := map[*Package]bool{}
	for _, pkg := range p.Packages {
		if pkg == filePkg {
			continue
		}
		for _, fn := range pkg.Funcs {
			if fn.Receiver == "" && fn.Exported && fn.Name == req.Name {
				symPkgs[pkg] = true
			}
		}
		for _, ty := range pkg.Types {
			if ty.Exported && ty.Name == req.Name {
				symPkgs[pkg] = true
			}
		}
		for _, v := range pkg.Vars {
			if v.Exported && v.Name == req.Name {
				symPkgs[pkg] = true
			}
		}
	}
	switch {
	case len(pkgNameMatches) == 1 && len(symPkgs) == 0:
		pkg := pkgNameMatches[0]
		return p.planAddImportFix(req, src, fset, fileAST, site, pkg, imports, false)
	case len(pkgNameMatches) == 0 && len(symPkgs) == 1:
		var pkg *Package
		for pkg = range symPkgs {
		}
		return p.planAddImportFix(req, src, fset, fileAST, site, pkg, imports, true)
	case len(pkgNameMatches)+len(symPkgs) > 1:
		return nil, fmt.Sprintf("%s matches more than one package or symbol in this module, so adding an import isn't a single mechanical fix", req.Name)
	}
	return nil, fmt.Sprintf("no near match and no project package provides %s — nothing mechanical to offer", req.Name)
}

func countIdents(f *ast.File, name string) int {
	n := 0
	ast.Inspect(f, func(node ast.Node) bool {
		if id, ok := node.(*ast.Ident); ok && id.Name == name {
			n++
		}
		return true
	})
	return n
}

// planAddImportFix builds the (d) plans: import the package, and
// (for symbol matches) qualify the use at the error site.
func (p *Project) planAddImportFix(req ErrorFixRequest, src []byte, fset *token.FileSet, fileAST *ast.File, site *identSite, pkg *Package, imports map[string]string, qualify bool) (*EditPlan, string) {
	path := p.pkgPathOf(pkg.Dir)
	if _, already := imports[path]; already {
		return nil, fmt.Sprintf("%s is already imported in this file, so the missing import isn't the problem", path)
	}
	for otherPath, eff := range imports {
		if eff == pkg.Name && otherPath != path {
			return nil, fmt.Sprintf("this file already uses the name %s for %s, so importing %s under that name would collide", pkg.Name, otherPath, path)
		}
	}
	edits := insertImportEdits(fset, fileAST, src, importSpecText(pkg, path))
	var summary, verifyDesc string
	if qualify {
		edits = append(edits, textEdit{start: site.start, end: site.end, text: pkg.Name + "." + req.Name})
		summary = fmt.Sprintf("At %s:%d, %s isn't defined — but package %s (%s) has an exported %s. Add the import and use it as %s.%s there.",
			p.LinkPath(req.File), req.Line, req.Name, pkg.Name, path, req.Name, pkg.Name, req.Name)
		verifyDesc = fmt.Sprintf("%s is imported and the use at %s:%d reads %s.%s", path, req.File, req.Line, pkg.Name, req.Name)
	} else {
		summary = fmt.Sprintf("At %s:%d, %s isn't defined — %s is the name of this project's %s package (%s), and this file never imports it. Add the import.",
			p.LinkPath(req.File), req.Line, req.Name, req.Name, pkg.Name, path)
		verifyDesc = fmt.Sprintf("%s is imported by %s", path, req.File)
	}
	after := applyTextEdits(src, edits)
	ef := finishFile(req.File, src, after)
	rel, symName, pkgName := req.File, req.Name, pkg.Name
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), req.Line),
		Files:      []EditFile{ef},
		verify: func(afterP *Project) bool {
			return verifyFileAfter(afterP, rel, func(_ *token.FileSet, f *ast.File) bool {
				if _, ok := fileImports(f)[path]; !ok {
					return false
				}
				if !qualify {
					return true
				}
				found := false
				ast.Inspect(f, func(n ast.Node) bool {
					if sel, ok := n.(*ast.SelectorExpr); ok && sel.Sel.Name == symName {
						if id, ok := sel.X.(*ast.Ident); ok && id.Name == pkgName {
							found = true
						}
					}
					return !found
				})
				return found
			})
		},
	}, ""
}

// --- (e) missing return -------------------------------------------------------

// zeroKind classifies a result type by the zero its signature needs.
func (p *Project) zeroKind(fset *token.FileSet, fileAST *ast.File, filePkg *Package, e ast.Expr, depth int) string {
	if depth > 5 {
		return ""
	}
	switch x := e.(type) {
	case *ast.Ident:
		switch x.Name {
		case "string":
			return "string"
		case "bool":
			return "bool"
		case "error", "any", "comparable":
			return "nilable"
		case "int", "int8", "int16", "int32", "int64",
			"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
			"byte", "rune", "float32", "float64", "complex64", "complex128":
			return "numeric"
		}
		ty := p.lookupType(x.Name, filePkg)
		if ty == nil {
			return ""
		}
		switch ty.Kind {
		case "struct":
			return "struct"
		case "interface":
			return "nilable"
		}
		under := p.typeSpecExpr(ty)
		if under == nil {
			return ""
		}
		return p.zeroKind(fset, fileAST, filePkg, under, depth+1)
	case *ast.StarExpr:
		return "nilable"
	case *ast.ArrayType:
		if x.Len == nil {
			return "nilable"
		}
		return "array"
	case *ast.MapType, *ast.ChanType, *ast.FuncType, *ast.InterfaceType:
		return "nilable"
	case *ast.ParenExpr:
		return p.zeroKind(fset, fileAST, filePkg, x.X, depth+1)
	case *ast.SelectorExpr:
		qual, ok := x.X.(*ast.Ident)
		if !ok {
			return ""
		}
		// Resolve qualifier -> import path -> project package.
		impPath := importPathForQualifier(fileAST, qual.Name)
		if impPath == "" {
			return ""
		}
		pkg := p.pkgByPath(impPath)
		if pkg == nil {
			return "" // a type from outside the module: kind unknowable here
		}
		ty := p.lookupTypeIn(pkg, x.Sel.Name)
		if ty == nil {
			return ""
		}
		switch ty.Kind {
		case "struct":
			return "struct"
		case "interface":
			return "nilable"
		}
		return ""
	case *ast.IndexExpr:
		return p.zeroKind(fset, fileAST, filePkg, x.X, depth+1)
	case *ast.IndexListExpr:
		return p.zeroKind(fset, fileAST, filePkg, x.X, depth+1)
	}
	return ""
}

// importPathForQualifier maps an import qualifier in a file to its
// path, honoring explicit import names.
func importPathForQualifier(fileAST *ast.File, qual string) string {
	for _, is := range fileAST.Imports {
		_, eff := importSpecOf(is)
		if eff == qual {
			path, _ := strconv.Unquote(is.Path.Value)
			return path
		}
	}
	return ""
}

// pkgByPath finds a project package by its module import path.
func (p *Project) pkgByPath(path string) *Package {
	for _, pkg := range p.Packages {
		if p.pkgPathOf(pkg.Dir) == path {
			return pkg
		}
	}
	return nil
}

// lookupTypeIn finds a named type in one package.
func (p *Project) lookupTypeIn(pkg *Package, name string) *Type {
	for _, ty := range pkg.Types {
		if ty.Name == name {
			return ty
		}
	}
	return nil
}

// typeSpecExpr re-reads a project type's declaration and returns its
// underlying type expression (for defined-type zero derivation).
func (p *Project) typeSpecExpr(ty *Type) ast.Expr {
	_, _, fileAST, refusal := p.readProjectFile(ty.File)
	if refusal != "" || fileAST == nil {
		return nil
	}
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range gd.Specs {
			if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == ty.Name {
				return ts.Type
			}
		}
	}
	return nil
}

func (p *Project) planMissingReturn(req ErrorFixRequest) (*EditPlan, string) {
	src, fset, fileAST, refusal := p.errorFixTarget(req)
	if refusal != "" {
		return nil, refusal
	}
	var fn *ast.FuncDecl
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		if lineOfPos(fset, fd.Body.Rbrace) == req.Line || lineOfPos(fset, fd.End()-1) == req.Line {
			fn = fd
			break
		}
	}
	if fn == nil {
		return nil, fmt.Sprintf("I couldn't find the function ending at %s:%d — the code may have changed since that error was copied", p.LinkPath(req.File), req.Line)
	}
	if fn.Type.Results == nil || len(fn.Type.Results.List) == 0 {
		return nil, "that function returns nothing, so a missing-return fix doesn't apply to it"
	}
	if fn.Type.TypeParams != nil && len(fn.Type.TypeParams.List) > 0 {
		return nil, "that function is generic — the zero values of type parameters aren't a mechanical fix, so I won't guess"
	}
	filePkg := p.pkgOfFile(req.File)
	allNamed := true
	total := 0
	for _, field := range fn.Type.Results.List {
		n := len(field.Names)
		if n == 0 {
			allNamed = false
			n = 1
		}
		total += n
	}
	var retText string
	if allNamed {
		retText = "return"
	} else {
		var zeros []string
		for _, field := range fn.Type.Results.List {
			kind := p.zeroKind(fset, fileAST, filePkg, field.Type, 0)
			if kind == "" {
				return nil, fmt.Sprintf("I can't derive a safe zero value for the result type %s — it is defined outside this project or in a shape I can't resolve exactly", exprText(fset, field.Type))
			}
			var zero string
			switch kind {
			case "numeric":
				zero = "0"
			case "string":
				zero = `""`
			case "bool":
				zero = "false"
			case "nilable":
				zero = "nil"
			case "struct", "array":
				zero = exprText(fset, field.Type) + "{}"
			}
			n := len(field.Names)
			if n == 0 {
				n = 1
			}
			for i := 0; i < n; i++ {
				zeros = append(zeros, zero)
			}
		}
		retText = "return " + strings.Join(zeros, ", ")
	}
	funcIndent := lineIndent(src, lineOfPos(fset, fn.Pos()))
	rbraceOff := offOf(fset, fn.Body.Rbrace)
	insert := funcIndent + "\t" + retText + "\n" + funcIndent
	if rbraceOff == 0 || src[rbraceOff-1] != '\n' {
		// The brace shares its line with code: open a fresh line
		// for the return first.
		insert = "\n" + insert
	}
	after := applyTextEdits(src, []textEdit{{start: rbraceOff, end: rbraceOff, text: insert}})
	ef := finishFile(req.File, src, after)

	var resultTypes []string
	for _, field := range fn.Type.Results.List {
		resultTypes = append(resultTypes, exprText(fset, field.Type))
	}
	rel, fnName := req.File, fn.Name.Name
	fnLine := lineOfPos(fset, fn.Pos())
	return &EditPlan{
		Summary: fmt.Sprintf("Add the missing return at the end of %s (%s:%d) — its signature promises %s back, and one path reaches the end returning nothing. The fix: %s.",
			fnName, p.LinkPath(rel), fnLine, strings.Join(resultTypes, ", "), retText),
		VerifyDesc: fmt.Sprintf("%s now ends with %s", fnName, retText),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), fnLine),
		Files:      []EditFile{ef},
		verify: func(afterP *Project) bool {
			return verifyFileAfter(afterP, rel, func(fset2 *token.FileSet, f *ast.File) bool {
				for _, d := range f.Decls {
					fd, ok := d.(*ast.FuncDecl)
					if !ok || fd.Body == nil || fd.Name.Name != fnName || lineOfPos(fset2, fd.Pos()) != fnLine {
						continue
					}
					if len(fd.Body.List) == 0 {
						return false
					}
					_, isRet := fd.Body.List[len(fd.Body.List)-1].(*ast.ReturnStmt)
					return isRet
				}
				return false
			})
		},
	}, ""
}
