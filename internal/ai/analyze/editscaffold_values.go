package analyze

// The scaffold's type table: deterministic generation of test
// INPUT values from a parameter's type alone. Two values are
// generated per parameter —
//
//	zero: the type's zero value (0, "", false, nil,
//	      context.Background(), T{}), always explicit where
//	      an honest literal exists;
//	edge: one named edge value from the user's own table
//	      (-1, "test-string", true, a one-element slice, a
//	      one-entry map, a populated struct, a cancelled
//	      context), only where it can be CONSTRUCTED from the
//	      type facts — never invented.
//
// A type the table can't build for (an interface parameter, a
// struct with unexported fields, a function parameter, a
// generic type parameter) yields no value and a reason; the
// scaffold then carries a TODO naming the parameter instead of
// a made-up value.

import (
	"go/ast"
	"strconv"
	"strings"
)

const scaffoldMaxDepth = 4

func isSignedInt(n string) bool {
	switch n {
	case "int", "int8", "int16", "int32", "int64":
		return true
	}
	return false
}

func isUnsignedInt(n string) bool {
	switch n {
	case "uint", "uint8", "uint16", "uint32", "uint64", "uintptr":
		return true
	}
	return false
}

func isFloat(n string) bool { return n == "float32" || n == "float64" }

func isComplex(n string) bool { return n == "complex64" || n == "complex128" }

func isBasic(n string) bool {
	return isSignedInt(n) || isUnsignedInt(n) || isFloat(n) || isComplex(n) ||
		n == "string" || n == "bool" || n == "byte" || n == "rune"
}

func basicZero(n string) string {
	switch n {
	case "string":
		return `""`
	case "bool":
		return "false"
	}
	return "0"
}

func basicEdge(n string) string {
	switch n {
	case "string":
		return `"test-string"`
	case "bool":
		return "true"
	case "byte":
		return "255" // the largest byte
	case "rune":
		return `'x'`
	}
	if isUnsignedInt(n) {
		return "1"
	}
	return "-1"
}

// --- lookups -----------------------------------------------------------------

// findProjectType resolves a named type in one project package.
func (b *scaffoldBuilder) findProjectType(pkgDir, name string) *Type {
	pkg := b.g.p.pkgByDir(pkgDir)
	if pkg == nil {
		return nil
	}
	for _, t := range pkg.Types {
		if t.Name == name {
			return t
		}
	}
	return nil
}

// typeSpec re-reads a project type's declaration: its TypeSpec
// and the context (imports, package) of the file declaring it.
func (b *scaffoldBuilder) typeSpec(t *Type) (*ast.TypeSpec, scaffoldCtx) {
	_, _, fAST, refusal := b.g.p.readProjectFile(t.File)
	if refusal != "" || fAST == nil {
		return nil, scaffoldCtx{}
	}
	ctx := scaffoldCtx{imports: map[string]string{}, pkgDir: t.PkgDir}
	for _, is := range fAST.Imports {
		_, eff := importSpecOf(is)
		path, _ := strconv.Unquote(is.Path.Value)
		ctx.imports[eff] = path
	}
	for _, d := range fAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range gd.Specs {
			if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == t.Name {
				return ts, ctx
			}
		}
	}
	return nil, scaffoldCtx{}
}

// qualifiedName renders a project type's name as the test file
// must write it: bare in its own package, qualified elsewhere
// (registering the import).
func (b *scaffoldBuilder) qualifiedName(t *Type) string {
	g := b.g
	if t.PkgDir == g.tgtPkg.Dir {
		return t.Name
	}
	if g.cross && t.PkgDir == g.src.PkgDir {
		return b.qual + "." + t.Name
	}
	path := g.p.Module
	if t.PkgDir != "" {
		path += "/" + t.PkgDir
	}
	g.needed[path] = t.Pkg
	return t.Pkg + "." + t.Name
}

// projectDirForPath maps an import path back to a project
// package directory, when the path is inside this module.
func (b *scaffoldBuilder) projectDirForPath(path string) (string, bool) {
	mod := b.g.p.Module
	if mod == "" {
		return "", false
	}
	if path == mod {
		return "", true
	}
	if strings.HasPrefix(path, mod+"/") {
		return strings.TrimPrefix(path, mod+"/"), true
	}
	return "", false
}

// isKnownInterface reports whether a standard-library type is
// a well-known interface — public knowledge, so its zero (nil)
// and its edge rule (no invented implementation) are honest
// without the definition living in this project.
func isKnownInterface(path, name string) bool {
	switch path {
	case "io":
		switch name {
		case "Reader", "Writer", "Closer", "Seeker", "ReadWriter", "ReadCloser",
			"WriteCloser", "ReadSeekCloser", "ByteReader", "ByteScanner",
			"RuneReader", "RuneScanner", "StringWriter":
			return true
		}
	case "fmt":
		return name == "Stringer"
	case "net/http":
		return name == "Handler"
	case "sort":
		return name == "Interface"
	}
	return false
}

// --- zero values ---------------------------------------------------------------

// zeroValue generates the explicit zero value for a type
// expression, or reports why none can be written honestly.
func (b *scaffoldBuilder) zeroValue(e ast.Expr, ctx scaffoldCtx, depth int) (string, bool, string) {
	if depth > scaffoldMaxDepth {
		return "", false, "the type nests too deeply to build a value for"
	}
	switch t := e.(type) {
	case *ast.Ident:
		if isBasic(t.Name) {
			return basicZero(t.Name), true, ""
		}
		switch t.Name {
		case "any", "error":
			return "nil", true, ""
		}
		if b.g.typeParams[t.Name] {
			return "", false, "it is a generic type parameter — the real type is the caller's choice"
		}
		typ := b.findProjectType(ctx.pkgDir, t.Name)
		if typ == nil {
			return "", false, "the type isn't in the project's index"
		}
		return b.namedZero(typ, b.qualifiedName(typ), depth)
	case *ast.SelectorExpr:
		id, ok := t.X.(*ast.Ident)
		if !ok {
			return "", false, "the type's shape couldn't be read"
		}
		path, isImport := ctx.imports[id.Name]
		if !isImport {
			return "", false, "the type's package couldn't be resolved"
		}
		if path == "context" && t.Sel.Name == "Context" {
			b.g.needed["context"] = "context"
			return "context.Background()", true, ""
		}
		rendered := b.g.renderType(e, b.qual)
		if isKnownInterface(path, t.Sel.Name) {
			return "nil", true, ""
		}
		if dir, isProj := b.projectDirForPath(path); isProj {
			if typ := b.findProjectType(dir, t.Sel.Name); typ != nil {
				return b.namedZero(typ, rendered, depth)
			}
		}
		// Standard-library types whose shape is public
		// knowledge; anything else's definition isn't in this
		// project, so no literal is offered for it.
		if path == "time" && t.Sel.Name == "Time" {
			return rendered + "{}", true, ""
		}
		if path == "time" && t.Sel.Name == "Duration" {
			return "0", true, ""
		}
		return "", false, "its definition isn't in this project, so no honest literal exists for it"
	case *ast.StarExpr, *ast.MapType, *ast.ChanType, *ast.FuncType, *ast.InterfaceType:
		return "nil", true, ""
	case *ast.ArrayType:
		if t.Len == nil {
			return "nil", true, ""
		}
		return b.g.renderType(e, b.qual) + "{}", true, ""
	case *ast.StructType:
		return b.g.renderType(e, b.qual) + "{}", true, ""
	case *ast.Ellipsis:
		return "nil", true, ""
	case *ast.ParenExpr:
		return b.zeroValue(t.X, ctx, depth)
	}
	return "", false, "no honest value could be generated for this type"
}

// namedZero is zeroValue for a named project type, written with
// the given qualified name.
func (b *scaffoldBuilder) namedZero(typ *Type, q string, depth int) (string, bool, string) {
	switch typ.Kind {
	case "struct":
		return q + "{}", true, ""
	case "interface":
		return "nil", true, ""
	}
	// A defined type over something else: only the honest
	// conversions are offered — over a basic type, or a
	// composite literal where the underlying type is one.
	spec, _ := b.typeSpec(typ)
	if spec == nil {
		return "", false, "the type's declaration couldn't be re-read"
	}
	switch u := spec.Type.(type) {
	case *ast.Ident:
		if isBasic(u.Name) {
			return q + "(" + basicZero(u.Name) + ")", true, ""
		}
	case *ast.ArrayType, *ast.MapType, *ast.StructType:
		return q + "{}", true, ""
	case *ast.StarExpr, *ast.ChanType, *ast.FuncType, *ast.InterfaceType:
		return "nil", true, ""
	}
	return "", false, "the type's underlying shape has no honest zero literal"
}

// --- edge values ---------------------------------------------------------------

// edgeValue generates one edge value for a type expression,
// from the type table, or reports why none exists.
func (b *scaffoldBuilder) edgeValue(e ast.Expr, ctx scaffoldCtx, depth int) (string, bool, string) {
	if depth > scaffoldMaxDepth {
		return "", false, "the type nests too deeply to build a value for"
	}
	switch t := e.(type) {
	case *ast.Ident:
		if isBasic(t.Name) {
			return basicEdge(t.Name), true, ""
		}
		switch t.Name {
		case "any":
			return `"test-string"`, true, ""
		case "error":
			return "", false, "an error value would have to be invented"
		}
		if b.g.typeParams[t.Name] {
			return "", false, "it is a generic type parameter — the real type is the caller's choice"
		}
		typ := b.findProjectType(ctx.pkgDir, t.Name)
		if typ == nil {
			return "", false, "the type isn't in the project's index"
		}
		return b.namedEdge(typ, b.qualifiedName(typ), depth)
	case *ast.SelectorExpr:
		id, ok := t.X.(*ast.Ident)
		if !ok {
			return "", false, "the type's shape couldn't be read"
		}
		path, isImport := ctx.imports[id.Name]
		if !isImport {
			return "", false, "the type's package couldn't be resolved"
		}
		if path == "context" && t.Sel.Name == "Context" {
			b.g.needed["context"] = "context"
			b.needCancelHelper = true
			return "gollemerCancelledContext()", true, ""
		}
		rendered := b.g.renderType(e, b.qual)
		if isKnownInterface(path, t.Sel.Name) {
			return "", false, "it is an interface — a test needs a real implementation, not an invented one"
		}
		if dir, isProj := b.projectDirForPath(path); isProj {
			if typ := b.findProjectType(dir, t.Sel.Name); typ != nil {
				return b.namedEdge(typ, rendered, depth)
			}
		}
		if path == "time" && t.Sel.Name == "Duration" {
			return strings.TrimSuffix(rendered, t.Sel.Name) + "Second", true, ""
		}
		return "", false, "its definition isn't in this project, so no honest edge value exists for it"
	case *ast.StarExpr:
		lit, ok, reason := b.compositeLiteral(t.X, ctx, depth+1)
		if !ok {
			return "", false, reason
		}
		return "&" + lit, true, ""
	case *ast.ArrayType:
		rendered := b.g.renderType(e, b.qual)
		if t.Len == nil {
			v, ok, reason := b.sampleValue(t.Elt, ctx, depth+1)
			if !ok {
				return "", false, reason
			}
			return rendered + "{" + v + "}", true, ""
		}
		lit, isLit := t.Len.(*ast.BasicLit)
		if !isLit {
			return "", false, "the array's length isn't a literal, so it can't be filled mechanically"
		}
		n, err := strconv.Atoi(lit.Value)
		if err != nil || n < 1 || n > 8 {
			return "", false, "the array's length isn't a literal, so it can't be filled mechanically"
		}
		v, ok, reason := b.sampleValue(t.Elt, ctx, depth+1)
		if !ok {
			return "", false, reason
		}
		parts := make([]string, n)
		for i := range parts {
			parts[i] = v
		}
		return rendered + "{" + strings.Join(parts, ", ") + "}", true, ""
	case *ast.MapType:
		k, ok, reason := b.sampleValue(t.Key, ctx, depth+1)
		if !ok {
			return "", false, reason
		}
		v, ok, reason := b.sampleValue(t.Value, ctx, depth+1)
		if !ok {
			return "", false, reason
		}
		return b.g.renderType(e, b.qual) + "{" + k + ": " + v + "}", true, ""
	case *ast.ChanType:
		return "make(" + b.g.renderType(e, b.qual) + ", 1)", true, ""
	case *ast.FuncType:
		return "", false, "a function value would have to be invented"
	case *ast.InterfaceType:
		if t.Methods == nil || len(t.Methods.List) == 0 {
			return `"test-string"`, true, ""
		}
		return "", false, "it is an interface — a test needs a real implementation, not an invented one"
	case *ast.StructType:
		return b.populatedStructLiteral(e, "", t, ctx, depth)
	case *ast.Ellipsis:
		return b.edgeValue(&ast.ArrayType{Elt: t.Elt}, ctx, depth)
	case *ast.ParenExpr:
		return b.edgeValue(t.X, ctx, depth)
	}
	return "", false, "no honest edge value could be generated for this type"
}

// namedEdge is edgeValue for a named project type.
func (b *scaffoldBuilder) namedEdge(typ *Type, q string, depth int) (string, bool, string) {
	switch typ.Kind {
	case "struct":
		spec, sctx := b.typeSpec(typ)
		if spec == nil {
			return "", false, "the type's declaration couldn't be re-read"
		}
		st, ok := spec.Type.(*ast.StructType)
		if !ok {
			return "", false, "the type's declaration couldn't be re-read"
		}
		return b.populatedStructLiteral(spec.Type, q, st, sctx, depth)
	case "interface":
		return "", false, "it is an interface — a test needs a real implementation, not an invented one"
	}
	spec, sctx := b.typeSpec(typ)
	if spec == nil {
		return "", false, "the type's declaration couldn't be re-read"
	}
	switch u := spec.Type.(type) {
	case *ast.Ident:
		if isBasic(u.Name) {
			return q + "(" + basicEdge(u.Name) + ")", true, ""
		}
	case *ast.ArrayType:
		if u.Len == nil {
			v, ok, reason := b.sampleValue(u.Elt, sctx, depth+1)
			if !ok {
				return "", false, reason
			}
			return q + "{" + v + "}", true, ""
		}
	case *ast.MapType:
		k, ok, reason := b.sampleValue(u.Key, sctx, depth+1)
		if !ok {
			return "", false, reason
		}
		v, ok, reason := b.sampleValue(u.Value, sctx, depth+1)
		if !ok {
			return "", false, reason
		}
		return q + "{" + k + ": " + v + "}", true, ""
	case *ast.StructType:
		return b.populatedStructLiteral(spec.Type, q, u, sctx, depth)
	}
	return "", false, "the type's underlying shape has no honest edge value"
}

// compositeLiteral builds a populated composite literal for a
// struct, slice, map, or array type expression — the value a
// pointer's edge case points at.
func (b *scaffoldBuilder) compositeLiteral(e ast.Expr, ctx scaffoldCtx, depth int) (string, bool, string) {
	switch e.(type) {
	case *ast.StructType, *ast.ArrayType, *ast.MapType:
		return b.edgeValue(e, ctx, depth)
	case *ast.Ident, *ast.SelectorExpr:
		// A named type: only a struct (or a composite defined
		// type) has a literal to point at.
		if v, ok, reason := b.edgeValue(e, ctx, depth); ok && strings.Contains(v, "{") {
			return v, true, ""
		} else if !ok {
			return "", false, reason
		}
		return "", false, "a populated value behind this pointer couldn't be built"
	}
	return "", false, "a populated value behind this pointer couldn't be built — a pointer to a basic value needs a variable to point at"
}

// sampleValue picks the value a populated container or struct
// field is filled with: the edge value where the table has one,
// else the zero value.
func (b *scaffoldBuilder) sampleValue(e ast.Expr, ctx scaffoldCtx, depth int) (string, bool, string) {
	if v, ok, _ := b.edgeValue(e, ctx, depth); ok {
		return v, true, ""
	}
	return b.zeroValue(e, ctx, depth)
}

// --- struct population ---------------------------------------------------------

type scaffoldField struct {
	name string
	expr ast.Expr
}

// structFieldsOf flattens a struct type's fields, deriving the
// literal name of embedded fields from their type.
func structFieldsOf(st *ast.StructType) []scaffoldField {
	var out []scaffoldField
	if st.Fields == nil {
		return nil
	}
	for _, f := range st.Fields.List {
		if len(f.Names) == 0 {
			out = append(out, scaffoldField{name: embeddedFieldName(f.Type), expr: f.Type})
			continue
		}
		for _, n := range f.Names {
			out = append(out, scaffoldField{name: n.Name, expr: f.Type})
		}
	}
	return out
}

func embeddedFieldName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.SelectorExpr:
		return t.Sel.Name
	case *ast.StarExpr:
		return embeddedFieldName(t.X)
	case *ast.IndexExpr:
		return embeddedFieldName(t.X)
	case *ast.IndexListExpr:
		return embeddedFieldName(t.X)
	}
	return ""
}

// populatedStructLiteral renders StructType{...} (or
// NamedType{...} when q names a defined type) with every field
// filled by its sample value. A field the test file's package
// cannot set — an unexported field of another package's
// struct — fails the whole literal, honestly.
func (b *scaffoldBuilder) populatedStructLiteral(typeExpr ast.Expr, q string, st *ast.StructType, ctx scaffoldCtx, depth int) (string, bool, string) {
	if depth > scaffoldMaxDepth {
		return "", false, "the type nests too deeply to build a value for"
	}
	head := q
	if head == "" {
		head = b.g.renderType(typeExpr, b.qual)
	}
	var parts []string
	for _, f := range structFieldsOf(st) {
		if f.name == "" {
			return "", false, "one of the struct's fields couldn't be named"
		}
		if !ast.IsExported(f.name) && ctx.pkgDir != b.g.tgtPkg.Dir {
			return "", false, "the struct has unexported fields, so code outside its package can't fill it in"
		}
		v, ok, _ := b.sampleValue(f.expr, ctx, depth+1)
		if !ok {
			return "", false, "one of the struct's fields has no honest value"
		}
		parts = append(parts, f.name+": "+v)
	}
	return head + "{" + strings.Join(parts, ", ") + "}", true, ""
}
