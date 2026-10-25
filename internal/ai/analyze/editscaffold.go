package analyze

// The delegate's test scaffold: the rich version of the old
// TODO stub. Where the stub was an empty table with a "fill me
// in" comment, the scaffold is a complete table-driven test for
// the wrapper, assembled from the same index signature the
// wrapper itself was built from:
//
//   - the args struct has one field per parameter, typed exactly
//     as the wrapper's signature renders them;
//   - the cases' INPUTS are generated from a deterministic type
//     table (scaffoldValueFor): a "zero values" case, plus one
//     named edge case per parameter where an honest edge value
//     can be constructed from the type facts alone;
//   - the EXPECTATIONS are never generated. want/wantErr are
//     present in the table but hold only type-shaped placeholder
//     values under explicit TODO comments — the assembler knows
//     the types, not the answers — and the test SKIPS itself
//     (t.Skip as its first act) until the reader fills the
//     expectations in and deletes the skip. That is the honest
//     resolution of the scaffold's central tension: it must
//     compile as part of the package's tests (a scaffold that
//     doesn't compile would poison the user's test runs), yet
//     its placeholder expectations must never fail a suite and
//     never silently pass as if they were checks. Skipped is
//     the truthful state: present, runnable, visibly unfinished.
//
// The scaffold began as a courtesy artifact of the delegate
// shape (5f64d54); the extension generalized the entry point
// (planScaffold, gens built by scaffoldGenFor in
// editscaffold_attach.go) so the clone, add-case, and
// add-field shapes scaffold their functions through the same
// machinery — one generator, one type table, one honesty
// standard. It is never verification evidence: the chat's
// verdict rests on the build and the package's pre-existing
// tests, the bounded retry never touches it, and nothing in
// this file refines an expectation from an observed result.

import (
	"fmt"
	"go/ast"
	"sort"
	"strconv"
	"strings"
)

// scaffoldParam is one wrapper parameter as the scaffold sees
// it: the name and rendered type the wrapper itself uses, plus
// the type expression (in the source file's context) the value
// table generates from.
type scaffoldParam struct {
	name     string
	typ      string // args-struct field type (a slice type for a variadic parameter)
	expr     ast.Expr
	ctx      scaffoldCtx
	variadic bool
	receiver bool // a method's receiver: args-struct field 0, never a call argument
}

// scaffoldResult is one wrapper result: its rendered type and
// whether it is the error result.
type scaffoldResult struct {
	typ     string
	expr    ast.Expr
	isError bool
}

// scaffoldCtx is the context a type expression was written in:
// the imports of its file (qualifier -> path) and the package
// directory its local names belong to. Values for a struct's
// fields are generated in the context of the file that declares
// the struct, not the source function's file.
type scaffoldCtx struct {
	imports map[string]string
	pkgDir  string
}

// scaffoldPresentation is how one edit shape presents its
// scaffold: the header's opening sentence(s), the function's
// prose name for the plan note, an extra note sentence saying
// what edit the scaffold accompanies ("" when the scaffolded
// function IS the edit, as for the delegate and the clone), and
// the noun the header's honesty line verifies nothing with.
// The zero value is the delegate's presentation.
type scaffoldPresentation struct {
	intro      string // header opening, %s = the test's name; lines split on \n
	display    string // prose name of the function under test
	accompany  string // extra plan-note sentence, or ""
	verifyNoun string // "the delegate" / "this edit"
}

// scaffoldBuilder assembles one scaffold. It renders types
// through the delegate generator (so qualification and import
// collection match the wrapper's), with the generator's target
// import map swapped for the TEST file's situation.
type scaffoldBuilder struct {
	g    *delegateGen
	qual string // the source package's qualifier as the test file uses it
	pres scaffoldPresentation

	// recvName is the args-struct field holding a method
	// target's receiver ("" for a plain function): the run
	// loop's call is tc.args.<recvName>.<Method>(...).
	recvName string

	params  []scaffoldParam
	results []scaffoldResult

	needCancelHelper bool
	needSrcImport    bool     // the rendered code references the source package's qualifier
	todos            []string // assembler TODO notes, first-mention order
	todoSeen         map[string]bool

	// collected is the import set (path -> qualifier) rendering
	// registered in the generator, captured before the
	// generator's own maps are restored.
	collected map[string]string
}

func (b *scaffoldBuilder) todo(msg string) {
	if !b.todoSeen[msg] {
		b.todoSeen[msg] = true
		b.todos = append(b.todos, msg)
	}
}

func (b *scaffoldBuilder) hasError() bool {
	for _, r := range b.results {
		if r.isError {
			return true
		}
	}
	return false
}

func (b *scaffoldBuilder) valueResults() []scaffoldResult {
	var out []scaffoldResult
	for _, r := range b.results {
		if !r.isError {
			out = append(out, r)
		}
	}
	return out
}

// --- the plan's second file --------------------------------------------------

// scaffoldTestRel names the test file a scaffold for a
// function in targetRel lands in: the target file's own base
// name, in the target package's directory.
func scaffoldTestRel(tgtPkg *Package, targetRel string) string {
	base := targetRel
	if i := strings.LastIndex(base, "/"); i >= 0 {
		base = base[i+1:]
	}
	testRel := strings.TrimSuffix(base, ".go") + "_test.go"
	if tgtPkg.Dir != "" {
		testRel = tgtPkg.Dir + "/" + testRel
	}
	return testRel
}

// delegateScaffold is the delegate shape's use of the shared
// scaffolder: the wrapper is the edit, so the presentation is
// the zero one (delegate wording throughout).
func (g *delegateGen) delegateScaffold(tgtPkg *Package, targetRel string) (*EditFile, string) {
	return g.planScaffold(tgtPkg, targetRel, scaffoldPresentation{display: g.wrapper})
}

// planScaffold builds the scaffold EditFile for the plan:
// appended to the package's natural _test.go (the target's own
// base name) or created there. Returns nil, with a note, when
// the function is generic, the test already exists, or the
// assembled scaffold fails the self-check. Shared by every
// shape that scaffolds — the delegate (5f64d54) and, since the
// extension, the clone, add-case, and add-field planners,
// whose gens are built by scaffoldGenFor.
func (g *delegateGen) planScaffold(tgtPkg *Package, targetRel string, pres scaffoldPresentation) (*EditFile, string) {
	p := g.p
	testRel := scaffoldTestRel(tgtPkg, targetRel)
	stubName := "Test" + titleFirst(g.wrapper)
	if len(g.typeParamNames()) > 0 {
		// A generic function's test would have to pick concrete
		// type arguments the assembler has no basis to choose —
		// a scaffold in generic terms doesn't compile, and an
		// instantiated one would be invented. Omit it, plainly.
		return nil, fmt.Sprintf("No test scaffold added: %s is generic, and its test would need concrete type arguments — that's a choice for whoever writes the test.", pres.display)
	}
	accompany := ""
	if pres.accompany != "" {
		accompany = " " + pres.accompany
	}
	note := fmt.Sprintf("Plus a test scaffold (%s in %s) — a full table-driven test whose case inputs are generated from %s's parameter types (zero values, plus a named edge case per parameter where an honest one exists).%s Its expected results are TODO placeholders, and the test skips itself until you fill them in — it is not part of how the edit is verified.", stubName, testRel, pres.display, accompany)

	existing, tfset, testAST, _ := p.readProjectFile(testRel)
	if testAST != nil {
		for _, d := range testAST.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == stubName {
				return nil, fmt.Sprintf("No test scaffold added: %s already has a %s.", testRel, stubName)
			}
		}
	}

	b := &scaffoldBuilder{g: g, pres: pres, todoSeen: map[string]bool{}}

	// Render for the test file: the generator's import
	// resolution consults the target file's imports, but the
	// scaffold lives in the test file — swap in the test file's
	// own import situation (create: none; append: its declared
	// imports), so qualifiers in the scaffold match the file it
	// lands in and only genuinely missing imports are added.
	savedTgt := g.tgtImports
	savedNeeded := g.needed
	testImports := map[string]string{} // path -> effective qualifier
	if testAST != nil {
		for _, is := range testAST.Imports {
			_, eff := importSpecOf(is)
			path, _ := strconv.Unquote(is.Path.Value)
			testImports[path] = eff
		}
	}
	g.tgtImports = map[string]string{}
	for path, eff := range testImports {
		g.tgtImports[path] = eff
	}
	g.needed = map[string]string{}
	b.qual = g.formQual(false)
	if eff, ok := testImports[g.srcPath]; ok && g.cross {
		b.qual = eff
	}

	b.collectSignature()
	text := b.render(stubName)
	b.collected = g.needed
	g.tgtImports = savedTgt
	g.needed = savedNeeded
	var specs []string
	for path := range b.extraImports() {
		if _, already := testImports[path]; !already {
			specs = append(specs, path)
		}
	}

	if bad := selfCheckScaffold(text.body, stubName, b); bad != "" {
		return nil, fmt.Sprintf("No test scaffold added: the one I assembled failed my own checks (%s), so I left it out rather than hand you a broken test file.", bad)
	}
	if len(b.todos) > 0 {
		note += " Assembler TODOs in it: " + strings.Join(b.todos, "; ") + "."
	}

	sort.Strings(specs)
	var specTxt []string
	for _, path := range specs {
		specTxt = append(specTxt, scaffoldImportSpec(path, b.importQual(path)))
	}

	if testAST != nil {
		// Append to the existing test file, adding the imports
		// the scaffold needs and the file lacks.
		var edits []textEdit
		if len(specTxt) > 0 {
			edits = append(edits, insertImportSpecs(testAST, tfset, existing, specTxt)...)
		}
		body := text.body
		if text.helper != "" && !fileDeclaresFunc(testAST, "gollemerCancelledContext") {
			body = text.helper + "\n" + body
		}
		appendText := "\n" + body
		if len(existing) == 0 || existing[len(existing)-1] != '\n' {
			appendText = "\n" + appendText
		}
		edits = append(edits, textEdit{start: len(existing), end: len(existing), text: appendText})
		after := applyTextEdits(existing, edits)
		ef := finishDelegateFile(testRel, existing, after)
		if bad := selfCheckScaffoldFile(ef.After, stubName, b); bad != "" {
			return nil, fmt.Sprintf("No test scaffold added: appended to your existing %s it failed my own checks (%s), so I left it out rather than hand you a broken test file.", testRel, bad)
		}
		return &ef, note
	}
	// Create it: the target package's own clause, the import
	// block, any helpers, the scaffold.
	var sb strings.Builder
	sb.WriteString("package " + tgtPkg.Name + "\n\n")
	if len(specTxt) == 1 {
		sb.WriteString("import " + specTxt[0] + "\n\n")
	} else if len(specTxt) > 1 {
		sb.WriteString("import (\n")
		for _, s := range specTxt {
			sb.WriteString("\t" + s + "\n")
		}
		sb.WriteString(")\n\n")
	}
	if text.helper != "" {
		sb.WriteString(text.helper + "\n")
	}
	sb.WriteString(text.body)
	ef := finishDelegateFile(testRel, nil, []byte(sb.String()))
	if bad := selfCheckScaffoldFile(ef.After, stubName, b); bad != "" {
		return nil, fmt.Sprintf("No test scaffold added: the one I assembled failed my own checks (%s), so I left it out rather than hand you a broken test file.", bad)
	}
	return &ef, note
}

// importQual is the qualifier the scaffold's code uses for an
// import path: what rendering registered, else the path's last
// segment.
func (b *scaffoldBuilder) importQual(path string) string {
	if b.g.cross && path == b.g.srcPath {
		return b.qual
	}
	if q, ok := b.collected[path]; ok {
		return q
	}
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}

func (b *scaffoldBuilder) extraImports() map[string]bool {
	out := map[string]bool{"testing": true}
	for path := range b.collected {
		out[path] = true
	}
	if len(b.valueResults()) > 0 {
		out["reflect"] = true
	}
	if b.needCancelHelper {
		out["context"] = true
	}
	if b.needSrcImport && b.g.cross {
		out[b.g.srcPath] = true
	}
	return out
}

// scaffoldImportSpec renders one import spec, naming the
// qualifier explicitly when it isn't the path's last segment
// (the form the delegate's own imports use).
func scaffoldImportSpec(path, qual string) string {
	last := path
	if i := strings.LastIndex(path, "/"); i >= 0 {
		last = path[i+1:]
	}
	if qual == last {
		return strconv.Quote(path)
	}
	return qual + " " + strconv.Quote(path)
}

// collectSignature expands the source declaration into the
// scaffold's parameters and results, in wrapper order, with
// names exactly as the wrapper declares them. A method's
// receiver becomes args-struct field 0: the value table
// generates for it like any parameter, and the run loop calls
// through it.
func (b *scaffoldBuilder) collectSignature() {
	g := b.g
	srcCtx := scaffoldCtx{imports: g.srcImports, pkgDir: g.src.PkgDir}
	wparams := g.wrapperParams(b.qual)
	if g.srcDecl.Recv != nil && len(g.srcDecl.Recv.List) > 0 {
		rf := g.srcDecl.Recv.List[0]
		// The field's name is the scaffold's own choice (it is
		// just a struct field holding the receiver value): the
		// declaration's receiver name when it is usable and
		// free, else a deterministic fallback. (Go itself bars
		// a parameter sharing the receiver's name, so a clash
		// can only come from a generated argN placeholder.)
		used := map[string]bool{}
		for _, wp := range wparams {
			used[wp.name] = true
		}
		name := ""
		if len(rf.Names) > 0 {
			name = rf.Names[0].Name
		}
		if name == "" || name == "_" || used[name] {
			name = ""
			for _, cand := range []string{"recv", "receiver", "self"} {
				if !used[cand] {
					name = cand
					break
				}
			}
		}
		b.recvName = name
		b.params = append(b.params, scaffoldParam{
			name: name, typ: g.renderType(rf.Type, b.qual), expr: rf.Type, ctx: srcCtx, receiver: true,
		})
	}
	pi := 0
	if fl := g.srcDecl.Type.Params; fl != nil {
		for _, f := range fl.List {
			count := len(f.Names)
			if count == 0 {
				count = 1
			}
			for i := 0; i < count; i++ {
				wp := wparams[pi]
				pi++
				expr := f.Type
				typ := wp.typ
				if ell, isEll := f.Type.(*ast.Ellipsis); isEll {
					expr = &ast.ArrayType{Elt: ell.Elt}
					typ = g.renderType(expr, b.qual)
				}
				b.params = append(b.params, scaffoldParam{
					name: wp.name, typ: typ, expr: expr, ctx: srcCtx, variadic: wp.variadic,
				})
			}
		}
	}
	if fl := g.srcDecl.Type.Results; fl != nil {
		for _, f := range fl.List {
			count := len(f.Names)
			if count == 0 {
				count = 1
			}
			typ := g.renderType(f.Type, b.qual)
			for i := 0; i < count; i++ {
				b.results = append(b.results, scaffoldResult{
					typ: typ, expr: f.Type, isError: typ == "error",
				})
			}
		}
	}
}

// --- rendering ---------------------------------------------------------------

type scaffoldText struct {
	body   string
	helper string
}

// render produces the scaffold function (and the cancelled-
// context helper when a case uses it). The cases are rendered
// first: they are what decides the helper and the TODO list.
func (b *scaffoldBuilder) render(stubName string) scaffoldText {
	wrapper := b.g.wrapper
	cases := b.renderCases()

	var sb strings.Builder
	if b.pres.intro != "" {
		for _, ln := range strings.Split(fmt.Sprintf(b.pres.intro, stubName), "\n") {
			fmt.Fprintf(&sb, "// %s\n", ln)
		}
	} else {
		fmt.Fprintf(&sb, "// %s is a test scaffold assembled by gollemer next to the %s delegate.\n", stubName, wrapper)
	}
	verifyNoun := b.pres.verifyNoun
	if verifyNoun == "" {
		verifyNoun = "the delegate"
	}
	sb.WriteString("// The cases' inputs are generated from the parameter types — zero\n")
	sb.WriteString("// values, plus a named edge case per parameter where an honest edge\n")
	sb.WriteString("// value exists. The expected results (want, wantErr) are TODO\n")
	sb.WriteString("// placeholders: gollemer knows the types, not the answers, so the\n")
	sb.WriteString("// test skips itself until you fill them in. It is a courtesy\n")
	fmt.Fprintf(&sb, "// artifact — nothing verifies %s with it.\n", verifyNoun)
	if len(b.todos) > 0 {
		sb.WriteString("//\n// Assembler TODOs:\n")
		for _, td := range b.todos {
			fmt.Fprintf(&sb, "//   - %s\n", td)
		}
	}
	fmt.Fprintf(&sb, "func %s(t *testing.T) {\n", stubName)
	sb.WriteString("\t// Until the TODO expectations below are filled in, this\n")
	sb.WriteString("\t// scaffold skips — delete this line to run the cases.\n")
	sb.WriteString("\tt.Skip(\"scaffold: fill in the want/wantErr TODOs below, then delete this t.Skip line\")\n\n")

	// The args struct: one field per parameter.
	sb.WriteString("\ttype args struct {\n")
	for _, prm := range b.params {
		fmt.Fprintf(&sb, "\t\t%s %s\n", prm.name, prm.typ)
	}
	sb.WriteString("\t}\n")

	// The table: name, args, want (one per non-error result),
	// wantErr (when the signature returns an error).
	sb.WriteString("\ttests := []struct {\n")
	sb.WriteString("\t\tname string\n")
	sb.WriteString("\t\targs args\n")
	for i := range b.valueResults() {
		field := "want"
		if len(b.valueResults()) > 1 {
			field = fmt.Sprintf("want%d", i+1)
		}
		fmt.Fprintf(&sb, "\t\t%s %s\n", field, b.valueResults()[i].typ)
	}
	if b.hasError() {
		sb.WriteString("\t\twantErr bool\n")
	}
	sb.WriteString("\t}{\n")
	sb.WriteString(cases)
	sb.WriteString("\t}\n")

	// The run loop: the standard shape — call, error check,
	// result comparisons.
	sb.WriteString("\tfor _, tc := range tests {\n")
	sb.WriteString("\t\tt.Run(tc.name, func(t *testing.T) {\n")
	var lhs []string
	gotN := 0
	for _, r := range b.results {
		if r.isError {
			lhs = append(lhs, "err")
		} else {
			gotN++
			if len(b.valueResults()) == 1 {
				lhs = append(lhs, "got")
			} else {
				lhs = append(lhs, fmt.Sprintf("got%d", gotN))
			}
		}
	}
	var callArgs []string
	for _, prm := range b.params {
		if prm.receiver {
			continue // the receiver is the call's target, not an argument
		}
		a := "tc.args." + prm.name
		if prm.variadic {
			a += "..."
		}
		callArgs = append(callArgs, a)
	}
	callName := wrapper
	if b.recvName != "" {
		callName = "tc.args." + b.recvName + "." + wrapper
	}
	call := fmt.Sprintf("%s(%s)", callName, strings.Join(callArgs, ", "))
	if len(lhs) > 0 {
		fmt.Fprintf(&sb, "\t\t\t%s := %s\n", strings.Join(lhs, ", "), call)
	} else {
		fmt.Fprintf(&sb, "\t\t\t%s\n", call)
		sb.WriteString("\t\t\t// TODO: the call returns nothing to compare — check its effects here.\n")
	}
	if b.hasError() {
		sb.WriteString("\t\t\tif (err != nil) != tc.wantErr {\n")
		fmt.Fprintf(&sb, "\t\t\t\tt.Errorf(\"%s() error = %%v, wantErr %%v\", err, tc.wantErr)\n", wrapper)
		sb.WriteString("\t\t\t\treturn\n")
		sb.WriteString("\t\t\t}\n")
	}
	gotN = 0
	for range b.valueResults() {
		gotN++
		got, want := "got", "tc.want"
		if len(b.valueResults()) > 1 {
			got, want = fmt.Sprintf("got%d", gotN), fmt.Sprintf("tc.want%d", gotN)
		}
		fmt.Fprintf(&sb, "\t\t\tif !reflect.DeepEqual(%s, %s) {\n", got, want)
		fmt.Fprintf(&sb, "\t\t\t\tt.Errorf(\"%s() = %%v, want %%v\", %s, %s)\n", wrapper, got, want)
		sb.WriteString("\t\t\t}\n")
	}
	sb.WriteString("\t\t})\n")
	sb.WriteString("\t}\n")
	sb.WriteString("}\n")

	body := sb.String()
	if b.g.cross && strings.Contains(body, b.qual+".") {
		b.needSrcImport = true
	}
	helper := ""
	if b.needCancelHelper {
		helper = "// gollemerCancelledContext returns a context that has already\n" +
			"// been cancelled — the edge case for a context parameter.\n" +
			"func gollemerCancelledContext() context.Context {\n" +
			"\tctx, cancel := context.WithCancel(context.Background())\n" +
			"\tcancel()\n" +
			"\treturn ctx\n" +
			"}\n"
	}
	return scaffoldText{body: body, helper: helper}
}

// renderCases renders the table rows: the zero-values case,
// then one edge case per parameter that has an honest edge
// value.
func (b *scaffoldBuilder) renderCases() string {
	var sb strings.Builder
	// A failed value attempt may have registered imports for
	// types its literal would have used; snapshot the import
	// set around each attempt and roll back on failure, so a
	// discarded literal never leaves an unused import behind.
	snapshot := func() map[string]string {
		m := map[string]string{}
		for k, v := range b.g.needed {
			m[k] = v
		}
		return m
	}
	zeroAssigns := make([]string, len(b.params))
	zeroVals := make([]string, len(b.params))
	for i, prm := range b.params {
		snap := snapshot()
		if v, ok, _ := b.zeroValue(prm.expr, prm.ctx, 0); ok {
			zeroAssigns[i] = fmt.Sprintf("%s: %s,", prm.name, v)
			zeroVals[i] = v
		} else {
			b.g.needed = snap
			zeroAssigns[i] = fmt.Sprintf("// TODO: set %s yourself — no honest value could be generated for %s.", prm.name, prm.typ)
		}
	}
	writeCase := func(name string, assigns []string) {
		fmt.Fprintf(&sb, "\t\t{\n\t\t\tname: %s,\n\t\t\targs: args{\n", strconv.Quote(name))
		for _, a := range assigns {
			fmt.Fprintf(&sb, "\t\t\t\t%s\n", a)
		}
		sb.WriteString("\t\t\t},\n")
		for i, r := range b.valueResults() {
			field := "want"
			if len(b.valueResults()) > 1 {
				field = fmt.Sprintf("want%d", i+1)
			}
			snap := snapshot()
			if v, ok, _ := b.zeroValue(r.expr, scaffoldCtx{imports: b.g.srcImports, pkgDir: b.g.src.PkgDir}, 0); ok {
				fmt.Fprintf(&sb, "\t\t\t%s: %s, // TODO: the result you expect — this is only a placeholder so the file compiles.\n", field, v)
			} else {
				b.g.needed = snap
				fmt.Fprintf(&sb, "\t\t\t// TODO: fill in %s — no placeholder could be built for %s.\n", field, r.typ)
			}
		}
		if b.hasError() {
			sb.WriteString("\t\t\twantErr: false, // TODO: set true if this case should return an error.\n")
		}
		sb.WriteString("\t\t},\n")
	}
	writeCase("zero values", zeroAssigns)
	for i, prm := range b.params {
		snap := snapshot()
		ev, ok, reason := b.edgeValue(prm.expr, prm.ctx, 0)
		if !ok {
			b.g.needed = snap
			if reason != "" {
				b.todo(fmt.Sprintf("%s (%s): no edge case — %s", prm.name, prm.typ, reason))
			}
			continue
		}
		if len(b.params) == 1 && ev == zeroVals[0] {
			continue // a one-parameter edge identical to the zero case adds nothing
		}
		assigns := make([]string, len(b.params))
		copy(assigns, zeroAssigns)
		assigns[i] = fmt.Sprintf("%s: %s,", prm.name, ev)
		writeCase("edge: "+prm.name+" — "+edgeLabel(prm.expr), assigns)
	}
	return sb.String()
}

// edgeLabel names what kind of edge a parameter's edge case is
// — a description of the input class, never a claimed behavior.
func edgeLabel(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		switch t.Name {
		case "string":
			return "non-empty string"
		case "bool":
			return "true"
		case "rune":
			return "a letter"
		case "byte":
			return "largest byte"
		}
		if isSignedInt(t.Name) {
			return "negative"
		}
		if isUnsignedInt(t.Name) {
			return "one"
		}
		if isFloat(t.Name) {
			return "negative"
		}
		if t.Name == "any" {
			return "a string value"
		}
		return "populated"
	case *ast.StarExpr:
		return "populated"
	case *ast.ArrayType:
		if t.Len == nil {
			return "one element"
		}
		return "filled"
	case *ast.MapType:
		return "one entry"
	case *ast.ChanType:
		return "buffered channel"
	case *ast.SelectorExpr:
		if t.Sel.Name == "Context" {
			return "cancelled context"
		}
		return "populated"
	case *ast.StructType:
		return "populated"
	}
	return "edge value"
}
