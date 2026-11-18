package decl

import (
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
)

// The generation pools. Every atom here (a type name, a field name, a
// whole type expression) has an injective lowercase form across the
// whole pool — the trainer lowercases, so two atoms sharing a
// lowercase form would be indistinguishable to the model and
// unrestorable for the compiler. AssertPoolInjectivity enforces it.

// LocalTypes are the project-type names: struct names, receiver types,
// embedded types, and the element types of composites.
var LocalTypes = []string{
	"User", "Account", "Config", "Cache", "Store", "Session", "Profile",
	"Product", "Order", "Address", "Client", "Server", "Worker", "Queue",
	"Buffer", "Record", "Report", "Metrics", "Settings", "Task", "Job",
	"Item", "Token", "Event", "Message", "Request", "Response", "Handler",
	"Catalog", "Invoice", "Project", "Device", "Channel", "Window",
}

// GenericLocals are local types that only appear instantiated.
var GenericLocals = map[string]string{
	"List": "type List[T any] struct{}",
	"Pair": "type Pair[K comparable, V any] struct{}",
}

// FieldNames are struct field names (exported style).
var FieldNames = []string{
	"ID", "Name", "Email", "Age", "Count", "Size", "Path", "Host",
	"Port", "Key", "Value", "Title", "Active", "Created", "Updated",
	"Score", "Price", "Total", "Items", "Tags", "Data", "Owner",
	"State", "Level", "Note", "Code", "Mode", "Text", "Body", "Kind",
	"Slug", "Rank", "Weight", "Height", "Width", "Depth", "Color",
	"Group", "Zone", "Limit", "Offset", "Secret", "Label", "Index",
}

// VerbNames are method and function names.
var VerbNames = []string{
	"Read", "Write", "Close", "Open", "Save", "Load", "Fetch", "Store",
	"Delete", "Update", "Create", "Validate", "Render", "Parse", "Encode",
	"Decode", "Start", "Stop", "Reset", "Flush", "Send", "Receive",
	"Handle", "Process", "Check", "Run", "Build", "Find", "List",
	"Count", "Sort", "Merge", "Split", "Join", "Copy", "Move", "Serve",
	"Dial", "Print", "Scan",
}

// ParamNames are parameter names (unexported style, as Go writes them).
var ParamNames = []string{
	"id", "name", "key", "value", "data", "item", "index", "count",
	"size", "path", "query", "limit", "offset", "flag", "mode", "text",
	"buf", "out", "first", "second", "target", "source", "filter",
	"option", "ctx", "req", "w", "r", "n", "msg", "raw", "opts",
}

// ReceiverNames are one/two-letter receiver names.
var ReceiverNames = []string{"s", "c", "h", "w", "r", "q", "b", "m", "t", "u", "p", "g"}

// BuiltinPool is the weighted working set of predeclared types for
// fields and parameters (error and any included; error is steered to
// result lists by the sampler).
var BuiltinPool = []string{
	"string", "int", "int64", "int32", "float64", "float32",
	"bool", "byte", "rune", "uint", "uint64", "any",
}

// ImportedPool is the working set of imported type references.
var ImportedPool = []string{
	"time.Time", "time.Duration", "context.Context", "json.RawMessage",
	"http.Handler", "http.ResponseWriter", "url.URL", "io.Reader",
	"io.Writer", "sql.DB",
}

// paramTypeForName steers a few parameter names to their idiomatic
// types when the sampler picks them, so generated signatures read like
// real Go instead of random pairings. It returns "" for no steering.
func paramTypeForName(name string) string {
	switch name {
	case "ctx":
		return "context.Context"
	case "w":
		return "http.ResponseWriter"
	case "r", "req":
		return "*http.Request"
	case "buf":
		return "[]byte"
	case "n":
		return "int"
	case "msg":
		return "string"
	case "raw":
		return "[]byte"
	}
	return ""
}

// RestoreMaps returns the lowercase -> canonical map for the TYPE
// namespace: local type names, generic locals, and builtins — the
// identifiers that can appear inside a type expression. Restoration
// is per-namespace on purpose: a field named ID and a parameter named
// id share the lowercase token "id" legitimately, and only the slot
// being restored decides which canonical form it means. Each
// namespace is internally injective (AssertPoolInjectivity).
func RestoreMaps() (atoms map[string]string) {
	atoms = map[string]string{}
	add := func(s string) {
		atoms[strings.ToLower(s)] = s
	}
	for _, s := range LocalTypes {
		add(s)
	}
	for s := range GenericLocals {
		add(s)
	}
	for _, s := range BuiltinPool {
		atoms[s] = s
	}
	add("error")
	return atoms
}

// FieldRestoreMap maps lowercase field names to canonical form.
func FieldRestoreMap() map[string]string {
	m := map[string]string{}
	for _, s := range FieldNames {
		m[strings.ToLower(s)] = s
	}
	return m
}

// VerbRestoreMap maps lowercase method/function names to canonical
// form.
func VerbRestoreMap() map[string]string {
	m := map[string]string{}
	for _, s := range VerbNames {
		m[strings.ToLower(s)] = s
	}
	return m
}

// AssertPoolInjectivity verifies that within each restore namespace
// no two distinct atoms share a lowercase form, and that every pool
// type expression canonicalizes. It returns the first violation.
func AssertPoolInjectivity() error {
	check := func(group []string) error {
		seen := map[string]string{}
		for _, atom := range group {
			low := strings.ToLower(atom)
			if prev, dup := seen[low]; dup && prev != atom {
				return &PoolError{Msg: "lowercase collision: " + prev + " vs " + atom}
			}
			seen[low] = atom
		}
		return nil
	}
	// The type namespace mixes locals, builtins, and generics.
	typeNS := append(append([]string{}, LocalTypes...), BuiltinPool...)
	for name := range GenericLocals {
		typeNS = append(typeNS, name)
	}
	if err := check(typeNS); err != nil {
		return err
	}
	for _, group := range [][]string{FieldNames, VerbNames, ParamNames, ReceiverNames} {
		if err := check(group); err != nil {
			return err
		}
	}
	for _, e := range ImportedPool {
		if _, err := MakeTypeRef(e); err != nil {
			return err
		}
	}
	return nil
}

// PoolError is a pool construction violation.
type PoolError struct{ Msg string }

func (e *PoolError) Error() string { return e.Msg }

// RestoreSpec maps a lowercase (model-world) spec back to canonical
// case, restoring each slot through its own namespace map (types,
// fields, verbs) and RestoreExpr for type expressions. Parameter and
// receiver names are lowercase in both worlds and restore as-is.
// ok=false when some atom is not restorable (a hallucinated name).
func RestoreSpec(s Spec) (Spec, bool) {
	typeAtoms := RestoreMaps()
	fieldAtoms := FieldRestoreMap()
	verbAtoms := VerbRestoreMap()
	fromMap := func(m map[string]string) func(string) (string, bool) {
		return func(n string) (string, bool) {
			if n == "" {
				return "", true
			}
			v, ok := m[n]
			return v, ok
		}
	}
	declName := fromMap(typeAtoms)
	if s.Kind == KindFunc {
		declName = func(n string) (string, bool) {
			if v, ok := verbAtoms[n]; ok {
				return v, true
			}
			// Constructors: New<Type>.
			if strings.HasPrefix(n, "new") && len(n) > 3 {
				if v, ok := typeAtoms[n[3:]]; ok {
					return "New" + v, true
				}
			}
			return "", false
		}
	}
	identity := func(n string) (string, bool) { return n, true }
	out := Spec{Kind: s.Kind}
	var ok bool
	if out.Name, ok = declName(s.Name); !ok {
		return Spec{}, false
	}
	restoreRef := func(r TypeRef) (TypeRef, bool) {
		expr, ok := RestoreExpr(r.Expr)
		if !ok {
			return TypeRef{}, false
		}
		ref, err := MakeTypeRef(expr)
		if err != nil {
			return TypeRef{}, false
		}
		return ref, true
	}
	for _, f := range s.Fields {
		nf := Field{Embedded: f.Embedded, Tag: f.Tag}
		if nf.Name, ok = fromMap(fieldAtoms)(f.Name); !ok {
			return Spec{}, false
		}
		if nf.Type, ok = restoreRef(f.Type); !ok {
			return Spec{}, false
		}
		out.Fields = append(out.Fields, nf)
	}
	for _, e := range s.Embeds {
		ne, ok2 := restoreRef(e)
		if !ok2 {
			return Spec{}, false
		}
		out.Embeds = append(out.Embeds, ne)
	}
	for _, m := range s.Methods {
		nm := Method{}
		if nm.Name, ok = fromMap(verbAtoms)(m.Name); !ok {
			return Spec{}, false
		}
		for _, p := range m.Params {
			np := Param{}
			if np.Name, ok = identity(p.Name); !ok {
				return Spec{}, false
			}
			if np.Type, ok = restoreRef(p.Type); !ok {
				return Spec{}, false
			}
			nm.Params = append(nm.Params, np)
		}
		for _, r := range m.Results {
			nr, ok2 := restoreRef(r)
			if !ok2 {
				return Spec{}, false
			}
			nm.Results = append(nm.Results, nr)
		}
		out.Methods = append(out.Methods, nm)
	}
	if s.Receiver != nil {
		nr := &Receiver{Pointer: s.Receiver.Pointer}
		if nr.Name, ok = identity(s.Receiver.Name); !ok {
			return Spec{}, false
		}
		if nr.Type, ok = fromMap(typeAtoms)(s.Receiver.Type); !ok {
			return Spec{}, false
		}
		out.Receiver = nr
	}
	for _, p := range s.Params {
		np := Param{}
		if np.Name, ok = identity(p.Name); !ok {
			return Spec{}, false
		}
		if np.Type, ok = restoreRef(p.Type); !ok {
			return Spec{}, false
		}
		out.Params = append(out.Params, np)
	}
	for _, r := range s.Results {
		nr, ok2 := restoreRef(r)
		if !ok2 {
			return Spec{}, false
		}
		out.Results = append(out.Results, nr)
	}
	c, err := Canonicalize(out)
	if err != nil {
		return Spec{}, false
	}
	return c, true
}

// selectorMembers maps a lowercase selector member (the part after the
// dot in time.Time) to its canonical case, derived from the imported
// pool plus *http.Request, which appears as a parameter type.
var selectorMembers = func() map[string]string {
	m := map[string]string{}
	for _, e := range append(append([]string{}, ImportedPool...), "http.Request") {
		parts := strings.SplitN(e, ".", 2)
		if len(parts) == 2 {
			m[strings.ToLower(parts[1])] = parts[1]
		}
	}
	return m
}()

// RestoreExpr restores a lowercase type expression to canonical case.
// It parses the expression, rewrites identifiers through the atom map
// (selector members through the imported-pool map; package qualifiers
// are already lowercase and stay), and re-prints canonically.
// ok=false when an identifier is unknown to the pools.
func RestoreExpr(lowerExpr string) (string, bool) {
	parsed, err := parser.ParseExpr(lowerExpr)
	if err != nil {
		return "", false
	}
	atoms := RestoreMaps()
	if !rewriteIdents(parsed, atoms) {
		return "", false
	}
	var b strings.Builder
	if err := format.Node(&b, token.NewFileSet(), parsed); err != nil {
		return "", false
	}
	return b.String(), true
}

// rewriteIdents rewrites identifier case in place over the type
// expression node set. Selector qualifiers (package names) are left
// alone; their members go through selectorMembers.
func rewriteIdents(n ast.Expr, atoms map[string]string) bool {
	switch node := n.(type) {
	case *ast.Ident:
		v, ok := atoms[node.Name]
		if !ok {
			return false
		}
		node.Name = v
		return true
	case *ast.SelectorExpr:
		member, ok := selectorMembers[node.Sel.Name]
		if !ok {
			return false
		}
		node.Sel.Name = member
		return true // X is a package qualifier: already lowercase
	case *ast.StarExpr:
		return rewriteIdents(node.X, atoms)
	case *ast.ArrayType:
		if node.Len != nil {
			if !rewriteIdents(node.Len, atoms) {
				return false
			}
		}
		return rewriteIdents(node.Elt, atoms)
	case *ast.MapType:
		return rewriteIdents(node.Key, atoms) && rewriteIdents(node.Value, atoms)
	case *ast.ParenExpr:
		return rewriteIdents(node.X, atoms)
	case *ast.Ellipsis:
		return rewriteIdents(node.Elt, atoms)
	case *ast.ChanType:
		return rewriteIdents(node.Value, atoms)
	case *ast.IndexExpr:
		return rewriteIdents(node.X, atoms) && rewriteIdents(node.Index, atoms)
	case *ast.IndexListExpr:
		if !rewriteIdents(node.X, atoms) {
			return false
		}
		for _, ix := range node.Indices {
			if !rewriteIdents(ix, atoms) {
				return false
			}
		}
		return true
	case *ast.BasicLit:
		return true // array length literal
	case *ast.FuncType:
		for _, f := range node.Params.List {
			if !rewriteIdents(f.Type, atoms) {
				return false
			}
		}
		if node.Results != nil {
			for _, f := range node.Results.List {
				if !rewriteIdents(f.Type, atoms) {
					return false
				}
			}
		}
		return true
	case *ast.StructType:
		for _, f := range node.Fields.List {
			if !rewriteIdents(f.Type, atoms) {
				return false
			}
		}
		return true
	case *ast.InterfaceType:
		for _, f := range node.Methods.List {
			if !rewriteIdents(f.Type, atoms) {
				return false
			}
		}
		return true
	}
	return false
}
