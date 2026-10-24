package analyze

// Rename everywhere: the Composer's second write shape. A rename is
// never a text substitution — golang.org/x/tools/refactor/rename was
// evaluated first and rejected on its own evidence: its package doc
// calls it "the obsolete implementation of the deleted gorename" that
// "has not worked properly since the advent of Go modules", it loads
// through the GOPATH-era go/loader, and it rewrites files itself,
// which the plan -> diff -> [y/n] spine cannot allow.
//
// So the sites come from go/packages + go/types instead (the same
// loader typed.go already uses): the target is resolved to a
// types.Object key, and every *ast.Ident in every loaded package —
// definition and uses, test files and external test packages
// included — whose object carries that key is a site. Object
// identity is what makes the correctness bar hold: a same-named
// local in another scope is a different object and is never touched,
// and a use the type checker can see cannot be missed. Files the
// load cannot see (other platforms' build tags, nested modules) are
// guarded separately below: if one of them mentions the name where
// it could mean this symbol, the rename refuses rather than guess.
//
// Everything here is planning only, like edits.go: nothing writes.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
)

// RenameTarget is one resolved symbol a rename could act on. The
// chat layer lists these for disambiguation and hands one back to
// PlanRenameTarget once the user picks.
type RenameTarget struct {
	Kind   string // "func", "method", "type", "var", "const", "field"
	Name   string
	Pkg    string // package name
	PkgDir string // package directory, relative to the project root
	Recv   string // methods: receiver base type; fields: owning struct
	File   string // declaration file, relative to the root
	Line   int
}

// Display renders the target the way pick lists print symbols:
// pkg.Name, or pkg.Recv.Name for methods and fields.
func (t RenameTarget) Display() string {
	if t.Recv != "" {
		return t.Pkg + "." + t.Recv + "." + t.Name
	}
	return t.Pkg + "." + t.Name
}

// KindWord renders a target kind the way a person would say it.
func (t RenameTarget) KindWord() string {
	switch t.Kind {
	case "func":
		return "function"
	case "method":
		return "method"
	case "var":
		return "variable"
	case "const":
		return "constant"
	}
	return t.Kind
}

const renameCanDo = "Rename works on functions, methods, types, package-level variables and constants, and struct fields."

// normalizeRenameKind folds the sentence's kind words into the
// internal kind vocabulary.
func normalizeRenameKind(k string) string {
	switch strings.ToLower(strings.TrimSpace(k)) {
	case "function", "func":
		return "func"
	case "method":
		return "method"
	case "type", "struct":
		return "type"
	case "variable", "var":
		return "var"
	case "constant", "const":
		return "const"
	case "field":
		return "field"
	}
	return ""
}

// recvBaseName reduces a rendered receiver ("*Widget", "Widget",
// "List[T]") to the base type name.
func recvBaseName(recv string) string {
	r := strings.TrimSpace(strings.Trim(recv, "*"))
	if i := strings.Index(r, "["); i >= 0 {
		r = r[:i]
	}
	return r
}

// RenameCandidates resolves a rename request's "from" phrase against
// the analyzer index and returns every symbol it could mean, after
// the kind hint and the "in <pkg/file>" scope filter. Resolution is
// case-insensitive and accepts bare names ("Save"), Recv.Name, and
// pkg.Name forms — the same forgiveness as findProjectFuncs.
func (p *Project) RenameCandidates(req EditRequest) []RenameTarget {
	q := strings.TrimSpace(req.RenameFrom)
	if q == "" {
		return nil
	}
	kind := normalizeRenameKind(req.RenameKind)
	var out []RenameTarget
	add := func(t RenameTarget) {
		if kind != "" && t.Kind != kind {
			return
		}
		if req.RenameScope != "" && !renameScopeMatches(t, req.RenameScope) {
			return
		}
		for _, o := range out {
			if o == t {
				return
			}
		}
		out = append(out, t)
	}
	dotted := strings.Contains(q, ".")
	lastSeg := q
	if i := strings.LastIndex(q, "."); i >= 0 {
		lastSeg = q[i+1:]
	}
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			t := RenameTarget{
				Name: fn.Name, Pkg: fn.Pkg, PkgDir: fn.PkgDir,
				File: fn.File, Line: fn.Line, Kind: "func",
			}
			if fn.Receiver != "" {
				t.Kind = "method"
				t.Recv = recvBaseName(fn.Receiver)
			}
			switch {
			case !dotted && strings.EqualFold(q, fn.Name):
				add(t)
			case dotted && (strings.EqualFold(q, fn.Display()) ||
				(fn.Receiver != "" && strings.EqualFold(q, recvBaseName(fn.Receiver)+"."+fn.Name)) ||
				strings.EqualFold(q, fn.Pkg+"."+fn.Name)):
				add(t)
			}
		}
		for _, ty := range pkg.Types {
			if (!dotted && strings.EqualFold(q, ty.Name)) ||
				(dotted && strings.EqualFold(q, ty.Pkg+"."+ty.Name)) {
				add(RenameTarget{Kind: "type", Name: ty.Name, Pkg: ty.Pkg, PkgDir: ty.PkgDir, File: ty.File, Line: ty.Line})
			}
			if ty.Kind == "struct" {
				for _, f := range ty.Fields {
					if (!dotted && strings.EqualFold(q, f)) ||
						(dotted && (strings.EqualFold(q, ty.Name+"."+f) ||
							strings.EqualFold(q, ty.Pkg+"."+ty.Name+"."+f) ||
							strings.EqualFold(lastSeg, f) && strings.EqualFold(q[:len(q)-len(lastSeg)-1], ty.Pkg))) {
						line := ty.Line
						if ty.FieldLines != nil && ty.FieldLines[f] > 0 {
							line = ty.FieldLines[f]
						}
						add(RenameTarget{Kind: "field", Name: f, Pkg: ty.Pkg, PkgDir: ty.PkgDir, Recv: ty.Name, File: ty.File, Line: line})
					}
				}
			}
		}
	}
	for _, v := range p.allVars() {
		if (!dotted && strings.EqualFold(q, v.Name)) ||
			(dotted && strings.EqualFold(q, v.Pkg+"."+v.Name)) {
			k := "var"
			if v.Const {
				k = "const"
			}
			add(RenameTarget{Kind: k, Name: v.Name, Pkg: v.Pkg, PkgDir: v.PkgDir, File: v.File, Line: v.Line})
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Display() != out[j].Display() {
			return out[i].Display() < out[j].Display()
		}
		if out[i].File != out[j].File {
			return out[i].File < out[j].File
		}
		return out[i].Line < out[j].Line
	})
	return out
}

// renameScopeMatches applies the "in <pkg/file>" disambiguator: the
// scope may name a package, a package directory, a file, or the tail
// of a file's path.
func renameScopeMatches(t RenameTarget, scope string) bool {
	scope = strings.TrimSpace(scope)
	return t.Pkg == scope || t.PkgDir == scope || t.File == scope ||
		filepathBase(t.File) == scope || strings.HasSuffix(t.File, "/"+scope)
}

// planRename is the PlanEdit path for EditRename: unique target ->
// plan; none or several -> an honest refusal (the chat layer offers
// the numbered pick for "several" before ever calling this).
func (p *Project) planRename(req EditRequest) (*EditPlan, string) {
	cands := p.RenameCandidates(req)
	switch {
	case len(cands) == 0:
		if req.RenameScope != "" {
			return nil, fmt.Sprintf("I couldn't find %q in %s — nothing there matches that name. %s", req.RenameFrom, req.RenameScope, renameCanDo)
		}
		return nil, fmt.Sprintf("I couldn't find anything named %q in this project to rename. %s", req.RenameFrom, renameCanDo)
	case len(cands) > 1:
		var b strings.Builder
		fmt.Fprintf(&b, "%q could mean more than one thing here:\n", req.RenameFrom)
		for _, c := range cands {
			fmt.Fprintf(&b, "  - the %s %s (%s:%d)\n", c.KindWord(), c.Display(), p.LinkPath(c.File), c.Line)
		}
		b.WriteString("Say which one with its package or receiver — or add \"in <package>\" — and I'll do it.")
		return nil, b.String()
	}
	return p.PlanRenameTarget(cands[0], req.RenameTo)
}

// --- the typed load -------------------------------------------------------

// objKey identifies a package-level object or a field across the
// several package instances a test-aware load produces (the plain
// package, its test variant, its external test package each
// type-check fresh, so pointers differ while the key is stable).
type objKey struct {
	pkgPath string
	recv    string // receiver/struct base name; "" for package-level
	name    string
	kind    string // "func", "type", "var", "const", "field"
}

// keyOfPkgPath builds the import path of a project package dir.
func (p *Project) pkgPathOf(dir string) string {
	if dir == "" {
		return p.Module
	}
	return p.Module + "/" + filepath.ToSlash(dir)
}

// keyOfTypesObj maps a types.Object to its key. The second return is
// false for objects a rename never targets (locals, labels, imports,
// builtins). Field owners are NOT resolved here — a *types.Var field
// doesn't know its struct; the traversal fills recv from context.
func keyOfTypesObj(obj types.Object) (objKey, bool) {
	if obj == nil || obj.Pkg() == nil {
		return objKey{}, false
	}
	switch o := obj.(type) {
	case *types.Func:
		if o.Signature() == nil || o.Signature().Recv() == nil {
			return objKey{pkgPath: obj.Pkg().Path(), name: o.Name(), kind: "func"}, true
		}
		return objKey{pkgPath: obj.Pkg().Path(), recv: namedTypeBase(o.Signature().Recv().Type()), name: o.Name(), kind: "func"}, true
	case *types.TypeName:
		return objKey{pkgPath: obj.Pkg().Path(), name: o.Name(), kind: "type"}, true
	case *types.Const:
		return objKey{pkgPath: obj.Pkg().Path(), name: o.Name(), kind: "const"}, true
	case *types.Var:
		if o.IsField() {
			return objKey{pkgPath: obj.Pkg().Path(), name: o.Name(), kind: "field"}, true
		}
		if o.Parent() == o.Pkg().Scope() {
			return objKey{pkgPath: obj.Pkg().Path(), name: o.Name(), kind: "var"}, true
		}
	}
	return objKey{}, false
}

// namedTypeBase reduces a type to its named base ("*Widget" and
// "List[T]" both give the declared name), "" when unnamed.
func namedTypeBase(t types.Type) string {
	for {
		if pt, ok := t.(*types.Pointer); ok {
			t = pt.Elem()
			continue
		}
		break
	}
	if n, ok := t.(*types.Named); ok {
		return n.Obj().Name()
	}
	return ""
}

// loadForRename loads the whole module with type info, test files
// included. Any type error in a module package is a refusal: sites
// are only provably complete when the type checker saw everything.
func (p *Project) loadForRename() ([]*packages.Package, string) {
	if p.Module == "" {
		return nil, "this project has no go.mod, so I can't load it with the compiler's type information — renaming safely means finding every use exactly, and that takes type info."
	}
	cfg := &packages.Config{
		Mode:  packages.NeedName | packages.NeedFiles | packages.NeedSyntax | packages.NeedTypes | packages.NeedTypesInfo | packages.NeedImports | packages.NeedDeps,
		Dir:   p.Root,
		Tests: true,
	}
	pkgs, err := packages.Load(cfg, "./...")
	if err != nil {
		return nil, fmt.Sprintf("I couldn't load the project with type information (%v), so I can't rename safely.", err)
	}
	for _, pkg := range pkgs {
		if !p.inModulePath(pkg.PkgPath) {
			continue
		}
		for _, e := range pkg.Errors {
			return nil, fmt.Sprintf("The project doesn't type-check right now — %s. Fix that first; renaming needs a clean type-check so every use of the name can be found exactly.", e.Msg)
		}
	}
	return pkgs, ""
}

func (p *Project) inModulePath(pkgPath string) bool {
	return pkgPath == p.Module || strings.HasPrefix(pkgPath, p.Module+"/")
}

// pkgInstanceDir gives a loaded package instance's directory
// relative to the root, from its own file list (PkgPath arithmetic
// lies for test variants and external test packages).
func (p *Project) pkgInstanceDir(pkg *packages.Package) string {
	for _, f := range pkg.GoFiles {
		rel, err := filepath.Rel(p.Root, f)
		if err == nil && !strings.HasPrefix(rel, "..") {
			return filepath.Dir(rel)
		}
	}
	return ""
}

// findTargetObj locates the target's types.Object in whichever
// loaded instance declares it, and the package scope instances used
// for collision checks.
func findTargetObj(pkgs []*packages.Package, t RenameTarget, wantPath string) (types.Object, []*packages.Package) {
	var instances []*packages.Package
	for _, pkg := range pkgs {
		if pkg.Types == nil {
			continue
		}
		// An instance "of the target package" declares the same
		// import path — test variants share it.
		if pkg.PkgPath == wantPath {
			instances = append(instances, pkg)
		}
	}
	for _, pkg := range instances {
		scope := pkg.Types.Scope()
		switch t.Kind {
		case "func", "type", "var", "const":
			if obj := scope.Lookup(t.Name); obj != nil {
				if key, ok := keyOfTypesObj(obj); ok && key.kind == t.Kind {
					return obj, instances
				}
			}
		case "method":
			if tn, ok := scope.Lookup(t.Recv).(*types.TypeName); ok {
				if named, ok := tn.Type().(*types.Named); ok {
					for i := 0; i < named.NumMethods(); i++ {
						if m := named.Method(i); m.Name() == t.Name {
							return m, instances
						}
					}
				}
			}
		case "field":
			if tn, ok := scope.Lookup(t.Recv).(*types.TypeName); ok {
				if st, ok := tn.Type().Underlying().(*types.Struct); ok {
					for i := 0; i < st.NumFields(); i++ {
						if f := st.Field(i); f.Name() == t.Name {
							return f, instances
						}
					}
				}
			}
		}
	}
	return nil, instances
}

// renameSite is one identifier occurrence to rewrite.
type renameSite struct {
	rel        string
	start, end int
}

// collectRenameSites walks every loaded package's syntax with a
// parent stack (fields need their owner from context) and records
// each identifier whose object carries the target key.
func collectRenameSites(p *Project, pkgs []*packages.Package, want objKey) ([]renameSite, []string) {
	var sites []renameSite
	seen := map[string]bool{}
	var external []string
	extSeen := map[string]bool{}
	for _, pkg := range pkgs {
		if pkg.TypesInfo == nil || pkg.Fset == nil {
			continue
		}
		info := pkg.TypesInfo
		for _, file := range pkg.Syntax {
			var stack []ast.Node
			ast.Inspect(file, func(n ast.Node) bool {
				if n == nil {
					stack = stack[:len(stack)-1]
					return false
				}
				stack = append(stack, n)
				id, ok := n.(*ast.Ident)
				if !ok {
					return true
				}
				obj := info.Defs[id]
				if obj == nil {
					obj = info.Uses[id]
				}
				key, ok := keyOfTypesObj(obj)
				if ok && key.kind == "field" {
					key.recv = fieldOwnerName(info, stack)
					if key.recv == "" {
						ok = false
					}
				}
				if !ok || key != want {
					// Embedded fields are dual citizens: the
					// identifier declares the field AND names its
					// type (go/types records only the field), so a
					// type rename must rewrite it too.
					if tk, tok2 := embeddedTypeKey(info, stack, id); tok2 && tk == want {
						// fall through and record the site
					} else {
						return true
					}
				}
				tf := pkg.Fset.File(id.Pos())
				if tf == nil {
					return true
				}
				abs := tf.Name()
				// The generated test main calls test functions by
				// name; it is regenerated from current source on
				// every run, so it follows a rename on its own.
				if filepath.Base(abs) == "_testmain.go" {
					return true
				}
				rel, err := filepath.Rel(p.Root, abs)
				if err != nil || strings.HasPrefix(rel, "..") {
					if !extSeen[abs] {
						extSeen[abs] = true
						external = append(external, abs)
					}
					return true
				}
				rel = filepath.ToSlash(rel)
				s := renameSite{rel: rel, start: tf.Offset(id.Pos()), end: tf.Offset(id.End())}
				k := fmt.Sprintf("%s|%d|%d", rel, s.start, s.end)
				if !seen[k] {
					seen[k] = true
					sites = append(sites, s)
				}
				return true
			})
		}
	}
	sort.Slice(sites, func(i, j int) bool {
		if sites[i].rel != sites[j].rel {
			return sites[i].rel < sites[j].rel
		}
		return sites[i].start < sites[j].start
	})
	return sites, external
}

// embeddedTypeKey reports the type key an embedded-field identifier
// also carries: in `struct { Base }` and `struct { *Base }`, the
// identifier is the field's declaration and a use of the type.
func embeddedTypeKey(info *types.Info, stack []ast.Node, id *ast.Ident) (objKey, bool) {
	if len(stack) < 2 {
		return objKey{}, false
	}
	embedded := false
	switch parent := stack[len(stack)-2].(type) {
	case *ast.Field:
		embedded = len(parent.Names) == 0 && parent.Type == ast.Expr(id)
	case *ast.StarExpr:
		if len(stack) >= 3 {
			if f, ok := stack[len(stack)-3].(*ast.Field); ok {
				embedded = len(f.Names) == 0 && f.Type == ast.Expr(parent)
			}
		}
	}
	if !embedded {
		return objKey{}, false
	}
	t := info.TypeOf(id)
	for {
		if pt, ok := t.(*types.Pointer); ok {
			t = pt.Elem()
			continue
		}
		break
	}
	if named, ok := t.(*types.Named); ok {
		return keyOfTypesObj(named.Obj())
	}
	return objKey{}, false
}

// fieldOwnerName resolves the struct a field identifier belongs to
// from its syntactic context: a selector's X type, a composite
// literal's type, or the struct declaration itself.
func fieldOwnerName(info *types.Info, stack []ast.Node) string {
	// stack's top is the identifier itself.
	for i := len(stack) - 2; i >= 0 && i >= len(stack)-4; i-- {
		switch parent := stack[i].(type) {
		case *ast.SelectorExpr:
			return namedTypeBase(info.TypeOf(parent.X))
		case *ast.KeyValueExpr:
			// Keep walking up to the composite literal.
			continue
		case *ast.CompositeLit:
			return namedTypeBase(info.TypeOf(parent))
		case *ast.Field:
			// The declaration: the owning TypeSpec is above.
			for j := i - 1; j >= 0; j-- {
				if ts, ok := stack[j].(*ast.TypeSpec); ok {
					return ts.Name.Name
				}
			}
			return ""
		case *ast.StructType:
			continue
		default:
			return ""
		}
	}
	return ""
}

// --- the planner ------------------------------------------------------------

// PlanRenameTarget assembles the rename of one resolved target into
// an EditPlan, or refuses with a user-ready reason.
func (p *Project) PlanRenameTarget(t RenameTarget, newName string) (*EditPlan, string) {
	if !token.IsIdentifier(newName) {
		return nil, fmt.Sprintf("%q isn't a name Go allows — a name is letters, digits and underscores and can't start with a digit or be a Go keyword.", newName)
	}
	if newName == t.Name {
		return nil, fmt.Sprintf("It's already named %s — nothing to rename.", t.Name)
	}
	switch {
	case t.Kind == "func" && t.Name == "main":
		return nil, "main is the program's entry point — renaming it would leave the program with no way to start."
	case t.Kind == "func" && t.Name == "init":
		return nil, "init runs because of its exact name — renaming it would turn it into a function nothing calls."
	case t.Kind == "method" && (t.Name == "Error" || t.Name == "String"):
		return nil, fmt.Sprintf("%s is special: that exact name is how Go recognizes the %s interface, which lives outside this project — renaming it here would silently stop the type from satisfying it.", t.Name, map[string]string{"Error": "error", "String": "fmt.Stringer"}[t.Name])
	}
	if t.Kind == "field" && p.fieldIsEmbedded(t) {
		return nil, fmt.Sprintf("%s is an embedded field — its name IS the type name, so renaming it means renaming the type. Say \"rename type %s to %s\" if that's what you mean.", t.Name, t.Name, newName)
	}
	if why, blocked := p.fileBlocked(t.File); blocked {
		return nil, fmt.Sprintf("I won't rename %s — it is declared in %s: %s.", t.Display(), p.LinkPath(t.File), why)
	}
	if src, err := os.ReadFile(filepath.Join(p.Root, t.File)); err == nil && fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't rename %s — it is declared in %s, a generated file (marked DO NOT EDIT).", t.Display(), p.LinkPath(t.File))
	}

	pkgs, refusal := p.loadForRename()
	if refusal != "" {
		return nil, refusal
	}
	wantPath := p.pkgPathOf(t.PkgDir)
	obj, instances := findTargetObj(pkgs, t, wantPath)
	if obj == nil {
		return nil, fmt.Sprintf("I found %s in my index but not in the type-checked project — the code may have changed since I read it. Analyze the project again and retry.", t.Display())
	}
	want, ok := keyOfTypesObj(obj)
	if !ok {
		return nil, fmt.Sprintf("I couldn't pin down %s in the type-checked project, so I won't rename it blind.", t.Display())
	}
	if t.Kind == "field" {
		want.recv = t.Recv
	}
	if refusal := p.renameCollision(t, newName, obj, instances, pkgs); refusal != "" {
		return nil, refusal
	}

	sites, external := collectRenameSites(p, pkgs, want)
	if len(external) > 0 {
		return nil, fmt.Sprintf("Something outside this project (%s) also uses %s, and I can't rewrite code that isn't here — renaming would leave that behind.", external[0], t.Name)
	}
	if len(sites) == 0 {
		return nil, fmt.Sprintf("I couldn't find any uses of %s in the type-checked project — the code may have changed since I read it. Analyze again and retry.", t.Display())
	}
	// Every file the rename touches must itself be editable, and
	// the file set the load covered is recorded for the guard below.
	covered := map[string]bool{}
	for _, pkg := range pkgs {
		for _, f := range pkg.GoFiles {
			if rel, err := filepath.Rel(p.Root, f); err == nil && !strings.HasPrefix(rel, "..") {
				covered[filepath.ToSlash(rel)] = true
			}
		}
	}
	byFile := map[string][]renameSite{}
	for _, s := range sites {
		byFile[s.rel] = append(byFile[s.rel], s)
	}
	for rel := range byFile {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("A use of %s lives in %s: %s — renaming would have to edit it too, so I won't.", t.Name, p.LinkPath(rel), why)
		}
	}
	if refusal := p.renameUncoveredGuard(t, covered); refusal != "" {
		return nil, refusal
	}

	var files []EditFile
	var counts []string
	for _, rel := range sortedSiteFiles(byFile, t.File) {
		src, err := readFileBytes(filepath.Join(p.Root, rel))
		if err != nil {
			return nil, fmt.Sprintf("I couldn't read %s: %v", p.LinkPath(rel), err)
		}
		if fileBytesBlocked(src) {
			return nil, fmt.Sprintf("A use of %s lives in %s, a generated file (marked DO NOT EDIT) — renaming would have to edit it too, so I won't.", t.Name, p.LinkPath(rel))
		}
		var edits []textEdit
		for _, s := range byFile[rel] {
			if s.end > len(src) || string(src[s.start:s.end]) != t.Name {
				return nil, fmt.Sprintf("%s changed since I read it — analyze the project again and retry the rename.", p.LinkPath(rel))
			}
			edits = append(edits, textEdit{start: s.start, end: s.end, text: newName})
		}
		files = append(files, finishFile(rel, src, applyTextEdits(src, edits)))
		counts = append(counts, fmt.Sprintf("%s (%d)", p.LinkPath(rel), len(byFile[rel])))
	}

	oldDisplay, newDisplay := t.Display(), (RenameTarget{Kind: t.Kind, Name: newName, Pkg: t.Pkg, PkgDir: t.PkgDir, Recv: t.Recv}).Display()
	total := len(sites)
	summary := fmt.Sprintf("Rename the %s %s to %s — %d place(s) across %d file(s): %s.",
		t.KindWord(), oldDisplay, newName, total, len(files), strings.Join(counts, ", "))
	siteLine := t.Line
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s is now %s", oldDisplay, newDisplay),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(t.File), siteLine),
		Files:      files,
		verify: func(after *Project) bool {
			return renameVerify(after, t, newName)
		},
	}, ""
}

func sortedSiteFiles(byFile map[string][]renameSite, decl string) []string {
	var rels []string
	for rel := range byFile {
		rels = append(rels, rel)
	}
	sort.Slice(rels, func(i, j int) bool {
		if rels[i] == decl {
			return true
		}
		if rels[j] == decl {
			return false
		}
		return rels[i] < rels[j]
	})
	return rels
}

// renameVerify checks the re-analyzed project: the new name exists
// in the target's home, and the old name no longer does.
func renameVerify(after *Project, t RenameTarget, newName string) bool {
	switch t.Kind {
	case "func", "method":
		foundNew, foundOld := false, false
		for _, pkg := range after.Packages {
			if pkg.Dir != t.PkgDir {
				continue
			}
			for _, fn := range pkg.Funcs {
				if recvBaseName(fn.Receiver) != t.Recv {
					continue
				}
				if fn.Name == newName {
					foundNew = true
				}
				if fn.Name == t.Name {
					foundOld = true
				}
			}
		}
		return foundNew && !foundOld
	case "type":
		foundNew, foundOld := false, false
		for _, pkg := range after.Packages {
			if pkg.Dir != t.PkgDir {
				continue
			}
			for _, ty := range pkg.Types {
				if ty.Name == newName {
					foundNew = true
				}
				if ty.Name == t.Name {
					foundOld = true
				}
			}
		}
		return foundNew && !foundOld
	case "var", "const":
		foundNew, foundOld := false, false
		for _, v := range after.allVars() {
			if v.PkgDir != t.PkgDir {
				continue
			}
			if v.Name == newName {
				foundNew = true
			}
			if v.Name == t.Name {
				foundOld = true
			}
		}
		return foundNew && !foundOld
	case "field":
		for _, pkg := range after.Packages {
			if pkg.Dir != t.PkgDir {
				continue
			}
			for _, ty := range pkg.Types {
				if ty.Name != t.Recv {
					continue
				}
				hasNew, hasOld := false, false
				for _, f := range ty.Fields {
					if f == newName {
						hasNew = true
					}
					if f == t.Name {
						hasOld = true
					}
				}
				return hasNew && !hasOld
			}
		}
	}
	return false
}

// fieldIsEmbedded reports whether the target field is an embedded
// one (e.g. `type A struct { B }`). The analyzer records a declared
// type for every NAMED field in FieldTypes; an embedded field is
// exactly the one with no entry — its "name" is its type's name.
func (p *Project) fieldIsEmbedded(t RenameTarget) bool {
	for _, pkg := range p.Packages {
		if pkg.Dir != t.PkgDir {
			continue
		}
		for _, ty := range pkg.Types {
			if ty.Name == t.Recv && ty.Kind == "struct" {
				_, named := ty.FieldTypes[t.Name]
				return !named
			}
		}
	}
	return false
}

// renameCollision runs the pre-checks that keep a rename from
// creating a name clash the diff wouldn't obviously show.
func (p *Project) renameCollision(t RenameTarget, newName string, obj types.Object, instances, allPkgs []*packages.Package) string {
	switch t.Kind {
	case "func", "type", "var", "const":
		for _, pkg := range instances {
			if pkg.Types == nil {
				continue
			}
			if clash := pkg.Types.Scope().Lookup(newName); clash != nil && clash != obj {
				pos := pkg.Fset.Position(clash.Pos())
				return fmt.Sprintf("%s is already taken in package %s — there's a %s with that name at %s:%d. Pick a different new name.",
					newName, t.Pkg, typesObjKindName(clash), p.LinkPath(filepath.ToSlash(mustRel(p.Root, pos.Filename))), pos.Line)
			}
		}
	case "method":
		for _, pkg := range instances {
			if pkg.Types == nil {
				continue
			}
			tn, ok := pkg.Types.Scope().Lookup(t.Recv).(*types.TypeName)
			if !ok {
				continue
			}
			named, ok := tn.Type().(*types.Named)
			if !ok {
				continue
			}
			if sel := types.NewMethodSet(types.NewPointer(named)).Lookup(pkg.Types, newName); sel != nil {
				return fmt.Sprintf("%s already has a method named %s — the two would collide. Pick a different new name.", t.Recv, newName)
			}
			if sel := types.NewMethodSet(named).Lookup(pkg.Types, newName); sel != nil {
				return fmt.Sprintf("%s already has a method named %s — the two would collide. Pick a different new name.", t.Recv, newName)
			}
			if st, ok := named.Underlying().(*types.Struct); ok {
				for i := 0; i < st.NumFields(); i++ {
					if st.Field(i).Name() == newName {
						return fmt.Sprintf("%s already has a field named %s — a type can't have a field and a method with the same name. Pick a different new name.", t.Recv, newName)
					}
				}
			}
			break
		}
		// A method that satisfies a project interface can't be
		// renamed alone: the interface keeps the old name, and
		// every place the type is used AS that interface would
		// silently (or loudly) break.
		if refusal := p.renameInterfaceGuard(t, instances, allPkgs); refusal != "" {
			return refusal
		}
	case "field":
		for _, pkg := range instances {
			if pkg.Types == nil {
				continue
			}
			tn, ok := pkg.Types.Scope().Lookup(t.Recv).(*types.TypeName)
			if !ok {
				continue
			}
			named, ok := tn.Type().(*types.Named)
			if !ok {
				continue
			}
			fields := map[string]bool{}
			collectStructFieldNames(named.Underlying(), fields, map[types.Type]bool{})
			if fields[newName] {
				return fmt.Sprintf("The %s struct already has a field named %s (its own or promoted from an embedded one). Pick a different new name.", t.Recv, newName)
			}
			if sel := types.NewMethodSet(types.NewPointer(named)).Lookup(pkg.Types, newName); sel != nil {
				return fmt.Sprintf("%s already has a method named %s — a type can't have a field and a method with the same name. Pick a different new name.", t.Recv, newName)
			}
			break
		}
	}
	return ""
}

// renameInterfaceGuard refuses a method rename when the receiver
// type implements a project interface that declares the old name —
// the interface is a second "owner" of that name inside the project.
func (p *Project) renameInterfaceGuard(t RenameTarget, instances, allPkgs []*packages.Package) string {
	// The receiver's named type, from the target instances.
	var named *types.Named
	for _, pkg := range instances {
		if pkg.Types == nil {
			continue
		}
		if tn, ok := pkg.Types.Scope().Lookup(t.Recv).(*types.TypeName); ok {
			if n, ok := tn.Type().(*types.Named); ok {
				named = n
				break
			}
		}
	}
	if named == nil {
		return ""
	}
	// All project packages' scopes: any interface declaring the
	// method that this type satisfies blocks the rename.
	for _, pkg := range allPkgs {
		if pkg.Types == nil || !p.inModulePath(pkg.PkgPath) {
			continue
		}
		scope := pkg.Types.Scope()
		for _, name := range scope.Names() {
			tn, ok := scope.Lookup(name).(*types.TypeName)
			if !ok {
				continue
			}
			iface, ok := tn.Type().Underlying().(*types.Interface)
			if !ok {
				continue
			}
			ms := types.NewMethodSet(tn.Type())
			if ms.Lookup(tn.Pkg(), t.Name) == nil {
				continue
			}
			if types.Implements(named, iface) || types.Implements(types.NewPointer(named), iface) {
				return fmt.Sprintf("%s.%s is part of the %s interface (in package %s) — %s satisfies it, so renaming just this method would break that promise. Renaming interface methods isn't something rename does yet.",
					t.Recv, t.Name, tn.Name(), tn.Pkg().Name(), t.Recv)
			}
		}
	}
	return ""
}

// collectStructFieldNames gathers a struct's field names, following
// embedded fields into their structs (promoted names included).
func collectStructFieldNames(t types.Type, out map[string]bool, seen map[types.Type]bool) {
	st, ok := t.Underlying().(*types.Struct)
	if !ok || seen[t] {
		return
	}
	seen[t] = true
	for i := 0; i < st.NumFields(); i++ {
		f := st.Field(i)
		out[f.Name()] = true
		if f.Embedded() {
			collectStructFieldNames(f.Type(), out, seen)
		}
	}
}

func typesObjKindName(obj types.Object) string {
	switch obj.(type) {
	case *types.Func:
		return "function"
	case *types.TypeName:
		return "type"
	case *types.Const:
		return "constant"
	case *types.Var:
		return "variable"
	}
	return "declaration"
}

func mustRel(root, abs string) string {
	rel, err := filepath.Rel(root, abs)
	if err != nil {
		return abs
	}
	return rel
}

// renameUncoveredGuard scans project files the typed load did NOT
// cover — build-tagged for other platforms, other GOOS/GOARCH
// suffixes, nested modules — that could reference the target: same
// package directory, or importing the target's package. If such a
// file uses the old name where it isn't a local shadowing it, the
// rename refuses: an unverifiable use is a miss waiting to happen.
func (p *Project) renameUncoveredGuard(t RenameTarget, covered map[string]bool) string {
	wantPath := p.pkgPathOf(t.PkgDir)
	errFound := ""
	_ = filepath.WalkDir(p.Root, func(path string, d os.DirEntry, err error) error {
		if err != nil || errFound != "" {
			return err
		}
		if d.IsDir() {
			name := d.Name()
			if name == "vendor" || name == ".git" || (strings.HasPrefix(name, ".") && path != p.Root) {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, rerr := filepath.Rel(p.Root, path)
		if rerr != nil {
			return nil
		}
		rel = filepath.ToSlash(rel)
		if covered[rel] {
			return nil
		}
		sameDir := filepath.ToSlash(filepath.Dir(rel)) == t.PkgDir || (t.PkgDir == "" && filepath.Dir(rel) == ".")
		if !sameDir && !fileImportsPath(path, wantPath) {
			return nil
		}
		if fileMentionsSymbol(path, t.Name) {
			errFound = fmt.Sprintf("One catch: %s mentions %s, but that file isn't part of this platform's build (build tags or a nested module), so I can't check its uses exactly — I won't rename blind. Rename the use there by hand, or bring the file into the build, and retry.",
				p.LinkPath(rel), t.Name)
		}
		return nil
	})
	return errFound
}

// fileImportsPath reports whether a Go file imports the given path.
func fileImportsPath(path, wantPath string) bool {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
	if err != nil {
		return false
	}
	for _, is := range f.Imports {
		pth, _ := strconv.Unquote(is.Path.Value)
		if pth == wantPath {
			return true
		}
	}
	return false
}

// fileMentionsSymbol reports whether a file uses the name as an
// identifier anywhere except as the declaration site of a local
// shadow (a parameter, a := target, a local var or range variable
// with the same spelling — those declare a different thing). Uses
// OF such a local still count: without type info for this file its
// uses can't be told apart from the target's, and a rare false
// refusal from the guard beats a silent missed site.
func fileMentionsSymbol(path, name string) bool {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return false
	}
	localDecl := map[*ast.Ident]bool{}
	markFieldNames := func(fl *ast.FieldList) {
		if fl == nil {
			return
		}
		for _, f := range fl.List {
			for _, nm := range f.Names {
				localDecl[nm] = true
			}
		}
	}
	ast.Inspect(file, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncType:
			markFieldNames(x.Params)
			markFieldNames(x.Results)
		case *ast.AssignStmt:
			if x.Tok == token.DEFINE {
				for _, lhs := range x.Lhs {
					if id, ok := lhs.(*ast.Ident); ok {
						localDecl[id] = true
					}
				}
			}
		case *ast.ValueSpec:
			// Only specs inside functions are locals; a file-scope
			// walk can't tell, so treat ValueSpecs under a FuncDecl
			// via the second pass below instead. Marking here would
			// hide the declaration of a same-package variant of the
			// target itself (foo_windows.go defining the symbol for
			// another platform) — exactly what the guard exists for.
		case *ast.RangeStmt:
			if id, ok := x.Key.(*ast.Ident); ok {
				localDecl[id] = true
			}
			if id, ok := x.Value.(*ast.Ident); ok {
				localDecl[id] = true
			}
		}
		return true
	})
	// Local var specs (var x = ...) inside function bodies.
	for _, d := range file.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Body == nil {
			continue
		}
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			if vs, ok := n.(*ast.ValueSpec); ok {
				for _, nm := range vs.Names {
					localDecl[nm] = true
				}
			}
			return true
		})
	}
	found := false
	ast.Inspect(file, func(n ast.Node) bool {
		if found {
			return false
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == name && !localDecl[id] {
			found = true
		}
		return true
	})
	return found
}
