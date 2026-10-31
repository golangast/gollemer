package analyze

// Instantiation (editinstantiate.go): "add instantiation of
// struct Jim to handler Jake" — a statement of the form
// <var> := <Type>{...} at the end of a function that already
// exists, for a type that already exists. Both sides resolve
// against the project's own index (the analyzer is the
// disambiguator, per the spine): the object word must be a
// TYPE the project declares, the target a FUNCTION it
// declares, and the literal is assembled from the type's real
// definition — struct literals are field-named, with each
// field's zero value derived by the decl/type-pool
// conventions (zeroLiteralFor). What those conventions cannot
// derive, and unexported fields another package may not set,
// are left out of the literal and named in the plan's notes —
// the add-field honesty rules. Interfaces and the types a
// literal cannot make (channels, funcs, defined basics) refuse
// with the reason. The statement lands through the same plan
// machinery as the concept book's statement rows: end of the
// named function's body, imports added when the type lives in
// another package, AST self-check, and a verify closure — one
// insertion path, no second inserter.

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"strings"
)

// resolveInstantiateType finds the project type a sentence
// named — bare ("Jim") or qualified ("store.Jim"). Matching is
// the concept book's "like" discipline: exact case wins,
// case-insensitive is forgiven, and a name declared in more
// than one package refuses with the candidates listed.
func (p *Project) resolveInstantiateType(name string) (ty *Type, pkg *Package, refusal string) {
	qual, base := "", name
	if i := strings.LastIndex(name, "."); i > 0 {
		qual, base = name[:i], name[i+1:]
	}
	type cand struct {
		ty  *Type
		pkg *Package
	}
	var exact, folded []cand
	for _, pk := range p.Packages {
		if qual != "" && !strings.EqualFold(pk.Name, qual) {
			continue
		}
		for _, t := range pk.Types {
			switch {
			case t.Name == base:
				exact = append(exact, cand{t, pk})
			case strings.EqualFold(t.Name, base):
				folded = append(folded, cand{t, pk})
			}
		}
	}
	pool := exact
	if len(pool) == 0 {
		pool = folded
	}
	switch {
	case len(pool) == 1:
		return pool[0].ty, pool[0].pkg, ""
	case len(pool) == 0:
		if qual != "" {
			return nil, nil, fmt.Sprintf("package %s has no type named %s that I can find — an instantiation starts from a type the project already declares.", qual, base)
		}
		return nil, nil, fmt.Sprintf("I can't find a type named %q anywhere in this project — an instantiation starts from a type that already exists, so add the type first (\"add a struct named %s to <file>\") and ask again.", name, name)
	default:
		var where []string
		for _, c := range pool {
			where = append(where, fmt.Sprintf("%s in package %s (%s)", c.ty.Name, c.pkg.Name, p.LinkPath(c.ty.File)))
		}
		return nil, nil, fmt.Sprintf("the name %s is declared in more than one package: %s. Say which one, like %q.", base, strings.Join(where, "; "), pool[0].pkg.Name+"."+pool[0].ty.Name)
	}
}

// resolveInstantiateFunc finds the function a sentence named
// as the statement's home. Test functions are not handlers and
// never candidates. When several functions share the name, the
// one living in the type's own package wins if it is the only
// one there; anything else refuses with the candidates.
func (p *Project) resolveInstantiateFunc(name string, tyPkg *Package) (fn *Func, refusal string) {
	var exact, folded []*Func
	var allNames []string
	seenName := map[string]bool{}
	for _, pk := range p.Packages {
		for _, f := range pk.Funcs {
			if f.IsTest {
				continue
			}
			if !seenName[f.Name] {
				seenName[f.Name] = true
				allNames = append(allNames, f.Name)
			}
			switch {
			case f.Name == name:
				exact = append(exact, f)
			case strings.EqualFold(f.Name, name):
				folded = append(folded, f)
			}
		}
	}
	pool := exact
	if len(pool) == 0 {
		pool = folded
	}
	switch {
	case len(pool) == 0:
		list := "none it declares"
		if len(allNames) > 0 {
			if len(allNames) > 8 {
				allNames = allNames[:8]
			}
			list = strings.Join(allNames, ", ")
		}
		return nil, fmt.Sprintf("I can't find a function named %q in this project — the statement has to land in a function that already exists (the functions here include: %s).", name, list)
	case len(pool) == 1:
		return pool[0], ""
	}
	if tyPkg != nil {
		var inPkg []*Func
		for _, f := range pool {
			if f.PkgDir == tyPkg.Dir {
				inPkg = append(inPkg, f)
			}
		}
		if len(inPkg) == 1 {
			return inPkg[0], ""
		}
	}
	var where []string
	for _, f := range pool {
		disp := f.Name
		if f.Receiver != "" {
			disp = f.Receiver + "." + f.Name
		}
		where = append(where, fmt.Sprintf("%s in %s", disp, p.LinkPath(f.File)))
	}
	return nil, fmt.Sprintf("the name %s belongs to more than one function: %s — I can't tell which one should hold the new statement.", name, strings.Join(where, "; "))
}

// instantiateTypeSpec re-reads a type's declaration and
// returns its TypeSpec (the typeSpecExpr pattern, keeping the
// spec itself: the planner needs the type parameters and the
// underlying expression, not just the expression).
func (p *Project) instantiateTypeSpec(ty *Type) (*ast.TypeSpec, *token.FileSet) {
	_, fset, fileAST, refusal := p.readProjectFile(ty.File)
	if refusal != "" || fileAST == nil {
		return nil, nil
	}
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.TYPE {
			continue
		}
		for _, spec := range gd.Specs {
			if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == ty.Name {
				return ts, fset
			}
		}
	}
	return nil, nil
}

// instantiateKindOK reports whether a type's underlying kind
// is one a literal can make: a struct, or a defined
// slice/map/array type. Generic types are out — their type
// arguments are the user's to choose.
func (p *Project) instantiateKindOK(ty *Type) bool {
	spec, _ := p.instantiateTypeSpec(ty)
	if spec == nil || (spec.TypeParams != nil && len(spec.TypeParams.List) > 0) {
		return false
	}
	switch spec.Type.(type) {
	case *ast.StructType, *ast.ArrayType, *ast.MapType:
		return true
	}
	return false
}

// CanInstantiate is the chat layer's gate for the PLAIN
// sentence form ("add a Jim in Jake", no instance wording):
// the claim holds only when both sides resolve uniquely and
// the type is one a literal can make. The instance-worded
// form skips this gate and lets planInstantiate's refusals
// answer in the family's own voice.
func (p *Project) CanInstantiate(typeName, funcName string) bool {
	ty, tyPkg, refusal := p.resolveInstantiateType(typeName)
	if refusal != "" || ty == nil {
		return false
	}
	if !p.instantiateKindOK(ty) {
		return false
	}
	fn, refusal := p.resolveInstantiateFunc(funcName, tyPkg)
	if refusal != "" || fn == nil {
		return false
	}
	if _, blocked := p.fileBlocked(fn.File); blocked {
		return false
	}
	return true
}

// planInstantiate assembles the instantiation statement and
// its insertion at the end of the target function's body.
func (p *Project) planInstantiate(req EditRequest) (*EditPlan, string) {
	typeName := strings.TrimSpace(req.InstType)
	funcName := strings.TrimSpace(req.InstFunc)
	if typeName == "" || funcName == "" {
		return nil, "that edit needs both a type to make and a function to put it in."
	}
	ty, tyPkg, refusal := p.resolveInstantiateType(typeName)
	if refusal != "" {
		return nil, refusal
	}
	fn, refusal := p.resolveInstantiateFunc(funcName, tyPkg)
	if refusal != "" {
		return nil, refusal
	}
	rel := fn.File
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(rel), why)
	}
	src, fset, fileAST, refusal := p.readProjectFile(rel)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(rel))
	}
	spec, specFset := p.instantiateTypeSpec(ty)
	if spec == nil {
		return nil, fmt.Sprintf("I couldn't re-read %s's declaration in %s, so I can't build its literal honestly.", ty.Name, p.LinkPath(ty.File))
	}
	if spec.TypeParams != nil && len(spec.TypeParams.List) > 0 {
		return nil, fmt.Sprintf("%s is a generic type — it needs its type arguments in brackets (like %s[int]) before anything can be made of it, and I can't pick those for you.", ty.Name, ty.Name)
	}

	// The target FuncDecl in the file's own AST — matched by
	// name and receiver, the pair Go itself treats as one
	// declaration.
	var decl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if ok && fd.Name.Name == fn.Name && instantiateRecvBase(fd) == instantiateRecvBaseName(fn.Receiver) {
			decl = fd
			break
		}
	}
	if decl == nil {
		return nil, fmt.Sprintf("I couldn't find %s in %s any more — the file changed under me; ask me to look at the project again, then ask again.", fn.Name, p.LinkPath(rel))
	}
	if decl.Body == nil {
		return nil, fmt.Sprintf("%s in %s has no body — it's a declaration without code, so there's nowhere to put the statement.", fn.Name, p.LinkPath(rel))
	}

	cross := ty.PkgDir != fn.PkgDir
	typeRef := ty.Name
	var specs []string
	if cross {
		if p.Module == "" {
			return nil, "I couldn't work out the type's package import path (the project has no go.mod module line I can read), so I can't write the import honestly."
		}
		path := p.pkgPathOf(tyPkg.Dir)
		typeRef = tyPkg.Name + "." + ty.Name
		// A different package already imported under the same
		// name would make the qualification name the wrong type.
		for _, is := range fileAST.Imports {
			_, eff := importSpecOf(is)
			if eff == tyPkg.Name {
				if ip, _ := strconv.Unquote(is.Path.Value); ip != path {
					return nil, fmt.Sprintf("%s already imports a different package named %s, so I can't refer to %s there honestly.", p.LinkPath(rel), tyPkg.Name, typeRef)
				}
			}
		}
		specs = missingImportSpecs(fileAST, []string{strconv.Quote(path)})
	}

	// The literal, from the type's real definition.
	var literal string
	var notes []string
	switch e := spec.Type.(type) {
	case *ast.InterfaceType:
		return nil, fmt.Sprintf("%s is an interface — you can't make one directly, only a concrete type that satisfies it. Name one of your own structs (or another concrete type) instead.", ty.Name)
	case *ast.StructType:
		lit, ns := p.instantiateStructLiteral(ty, e, specFset, typeRef, cross, fileAST.Name.Name)
		literal, notes = lit, ns
	case *ast.ArrayType, *ast.MapType:
		literal = typeRef + "{}"
	default:
		return nil, instantiateNonLiteral(ty, spec, specFset)
	}

	// The variable's name: the sentence's, or derived from the
	// type. A collision with a name the function already uses
	// refuses — the spine never renames silently.
	varName := strings.TrimSpace(req.InstVar)
	if varName != "" && !token.IsIdentifier(varName) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a variable.", varName)
	}
	derived := varName == ""
	if derived {
		varName = lowerFirstName(ty.Name)
	}
	if instantiateLocalNames(decl)[varName] {
		if derived {
			return nil, fmt.Sprintf("%s already uses the name %q itself — say \"named <another name>\" at the end of the sentence to pick the new variable's name.", fn.Name, varName)
		}
		return nil, fmt.Sprintf("%s already uses the name %q itself — pick a different one with \"named <name>\".", fn.Name, varName)
	}

	stmt := varName + " := " + literal + "\n_ = " + varName
	notes = append(notes, fmt.Sprintf("The \"_ = %s\" line after the statement keeps the build green until %s uses %s — delete it once it does.", varName, fn.Name, varName))
	firstLine := strings.TrimSpace(strings.SplitN(stmt, "\n", 2)[0])

	var edits []textEdit
	edits = append(edits, conceptImportEdits(fileAST, fset, src, specs)...)
	indent := "\t"
	if len(decl.Body.List) > 0 {
		indent = lineIndent(src, lineOfPos(fset, decl.Body.List[0].Pos()))
		if indent == "" {
			indent = "\t"
		}
	}
	lbraceLine := lineOfPos(fset, decl.Body.Lbrace)
	rbraceLine := lineOfPos(fset, decl.Body.Rbrace)
	if rbraceLine == lbraceLine {
		edits = append(edits, textEdit{
			start: offOf(fset, decl.Body.Rbrace),
			end:   offOf(fset, decl.Body.Rbrace),
			text:  "\n" + indentConceptLines(stmt, indent) + "\n" + lineIndent(src, lineOfPos(fset, decl.Pos())),
		})
	} else {
		edits = append(edits, textEdit{
			start: lineStartOffset(src, rbraceLine),
			end:   lineStartOffset(src, rbraceLine),
			text:  indentConceptLines(stmt, indent) + "\n",
		})
	}
	ef := finishFile(rel, src, applyTextEdits(src, edits))
	if bad := checkInstantiate(ef.After, fn, firstLine); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	plan := &EditPlan{
		Summary:    fmt.Sprintf("Add %s at the end of %s in %s — an instantiation of %s, assembled from its definition in %s", firstLine, fn.Name, p.LinkPath(rel), ty.Name, p.LinkPath(ty.File)),
		VerifyDesc: fmt.Sprintf("%s now creates a %s", fn.Name, ty.Name),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, firstLine)),
		Files:      []EditFile{ef},
		Notes:      notes,
		verify: func(after *Project) bool {
			_, fset2, ast2, refusal := after.readProjectFile(rel)
			if refusal != "" || ast2 == nil {
				return false
			}
			return checkInstantiateIn(fset2, ast2, fn, firstLine) == ""
		},
	}
	return plan, ""
}

// instantiateStructLiteral renders a field-named struct
// literal from the type's own declaration, fields in
// declaration order. A field is left out — and named in a
// note — when it is unexported and the target function lives
// in another package (it cannot be set there, the add-field
// honesty rule), or when zeroLiteralFor cannot derive its
// zero value. Omitted fields still take their zero values;
// the literal stays honest either way.
func (p *Project) instantiateStructLiteral(owner *Type, st *ast.StructType, fset *token.FileSet, typeRef string, cross bool, targetPkg string) (string, []string) {
	type fieldLine struct{ name, zero string }
	var lines []fieldLine
	var hidden, underivable []string
	unnamed := 0
	for _, f := range st.Fields.List {
		ftype := exprText(fset, f.Type)
		if len(f.Names) == 0 {
			name := embeddedFieldName(f.Type)
			if name == "" {
				unnamed++
				continue
			}
			if cross && !ast.IsExported(name) {
				hidden = append(hidden, name)
				continue
			}
			if z := zeroLiteralFor(p, owner, ftype); z != "" {
				lines = append(lines, fieldLine{name, z})
			} else {
				underivable = append(underivable, name)
			}
			continue
		}
		for _, n := range f.Names {
			if cross && !n.IsExported() {
				hidden = append(hidden, n.Name)
				continue
			}
			if z := zeroLiteralFor(p, owner, ftype); z != "" {
				lines = append(lines, fieldLine{n.Name, z})
			} else {
				underivable = append(underivable, n.Name)
			}
		}
	}
	var notes []string
	if len(hidden) > 0 {
		notes = append(notes, fmt.Sprintf("Left out of the literal: %s — unexported fields of %s, which package %s can't set; they keep their zero values.", strings.Join(hidden, ", "), owner.Name, targetPkg))
	}
	if len(underivable) > 0 {
		notes = append(notes, fmt.Sprintf("Left out of the literal: %s — I can't write a zero value for their types, so they keep their zero values.", strings.Join(underivable, ", ")))
	}
	if unnamed > 0 {
		notes = append(notes, "One embedded field has no name I could set it by in a literal, so it keeps its zero value.")
	}
	if len(lines) == 0 {
		return typeRef + "{}", notes
	}
	var b strings.Builder
	b.WriteString(typeRef + "{\n")
	for _, l := range lines {
		b.WriteString("\t" + l.name + ": " + l.zero + ",\n")
	}
	b.WriteString("}")
	return b.String(), notes
}

// instantiateNonLiteral refuses a type a literal cannot make,
// naming what the type is built on and how one is actually
// made — the honest answer instead of a broken plan.
func instantiateNonLiteral(ty *Type, spec *ast.TypeSpec, fset *token.FileSet) string {
	switch spec.Type.(type) {
	case *ast.ChanType:
		return fmt.Sprintf("%s is a channel type — a channel is made with make(%s), not with a %s{} literal. The concept book has the channel pieces, if you want one inside the function.", ty.Name, ty.Name, ty.Name)
	case *ast.FuncType:
		return fmt.Sprintf("%s is a function type — a value of it is made by writing a function, not with a literal.", ty.Name)
	case *ast.StarExpr:
		return fmt.Sprintf("%s is a pointer type — a pointer starts from a value whose address you take, not from a literal.", ty.Name)
	default:
		return fmt.Sprintf("%s is built on %s — a literal like %s{} isn't how you make one; %s(0)-style conversion of a value is. (The literal forms cover structs and defined slice, map and array types.)", ty.Name, exprText(fset, spec.Type), ty.Name, ty.Name)
	}
}

// instantiateLocalNames collects the names a function already
// uses for itself — receiver, parameters, results, and every
// local declared in its body — so a new variable never
// silently shadows or collides.
func instantiateLocalNames(decl *ast.FuncDecl) map[string]bool {
	names := map[string]bool{}
	add := func(fl *ast.FieldList) {
		if fl == nil {
			return
		}
		for _, f := range fl.List {
			for _, n := range f.Names {
				names[n.Name] = true
			}
		}
	}
	add(decl.Recv)
	add(decl.Type.Params)
	add(decl.Type.Results)
	if decl.Body == nil {
		return names
	}
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.GenDecl:
			if n.Tok == token.VAR || n.Tok == token.CONST {
				for _, s := range n.Specs {
					if vs, ok := s.(*ast.ValueSpec); ok {
						for _, nm := range vs.Names {
							names[nm.Name] = true
						}
					}
				}
			}
		case *ast.AssignStmt:
			if n.Tok == token.DEFINE {
				for _, lhs := range n.Lhs {
					if id, ok := lhs.(*ast.Ident); ok {
						names[id.Name] = true
					}
				}
			}
		case *ast.RangeStmt:
			if n.Tok == token.DEFINE {
				for _, e := range []ast.Expr{n.Key, n.Value} {
					if id, ok := e.(*ast.Ident); ok {
						names[id.Name] = true
					}
				}
			}
		}
		return true
	})
	return names
}

// instantiateRecvBase is a FuncDecl's receiver base name as
// the index spells it ("Store" for both Store and *Store
// receivers), "" for a plain function.
func instantiateRecvBase(fd *ast.FuncDecl) string {
	if fd.Recv == nil || len(fd.Recv.List) == 0 {
		return ""
	}
	return embeddedFieldName(fd.Recv.List[0].Type)
}

// instantiateRecvBaseName normalizes an index Receiver string
// ("*Store", "store.Store") to its base name.
func instantiateRecvBaseName(recv string) string {
	recv = strings.TrimSpace(strings.TrimPrefix(recv, "*"))
	if i := strings.LastIndex(recv, "."); i >= 0 {
		recv = recv[i+1:]
	}
	return recv
}

// lowerFirstName derives a variable name from a type name:
// the type's name with its first letter lowered.
func lowerFirstName(name string) string {
	if name == "" {
		return name
	}
	r := []rune(name)
	if r[0] >= 'A' && r[0] <= 'Z' {
		r[0] += 'a' - 'A'
	}
	return string(r)
}

// checkInstantiate is the assembled-bytes self-check: the
// target function must now contain the statement.
func checkInstantiate(after []byte, fn *Func, firstLine string) string {
	fset, fileAST, err := parseGoSrc("instantiate.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	return checkInstantiateIn(fset, fileAST, fn, firstLine)
}

func checkInstantiateIn(fset *token.FileSet, fileAST *ast.File, fn *Func, firstLine string) string {
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != fn.Name || fd.Body == nil {
			continue
		}
		if instantiateRecvBase(fd) != instantiateRecvBaseName(fn.Receiver) {
			continue
		}
		for _, stmt := range fd.Body.List {
			if strings.HasPrefix(stmtText(fset, stmt), strings.TrimSpace(firstLine)) {
				return ""
			}
		}
		return fmt.Sprintf("%s doesn't contain the statement after assembly", fn.Name)
	}
	return fmt.Sprintf("the assembled file has no function named %s", fn.Name)
}
