// Package analysis performs symbolic path analysis and static safety
// verification on Go AST nodes before code execution.
//
// ProveSafety walks the AST with a branch-aware path walk: if, switch,
// and for statements fork the abstract program state (nil-ness of
// variables, index bound facts), early-return and panic branches prune
// paths, and joins merge them — so every check respects the path
// conditions established along the current execution path. A flat
// ast.Inspect pass cannot join path states, so it is used only for the
// preliminary nil-check collection sweep; the path walk itself is a
// recursive branch-tracking traversal. Three invariants are proved:
//
//   - nil safety: every selector (x.Y) and method call (x.Method()) is
//     checked against the path condition tree. A dereference of a
//     variable that is provably nil on the current path is a
//     CRITICAL_NIL_DEREFERENCE; a dereference of a variable that is
//     nil-checked somewhere in the function but not guarded on the
//     current path is a WARNING under the same category.
//   - resource leaks: every acquisition (os.Open, http.Get, db.Query,
//     or any call whose first non-error result implements io.Closer)
//     must have a matching defer x.Close() in the same scope, else a
//     RESOURCE_LEAK_WARNING.
//   - slice bounds: indexing a slice/array with a variable index needs
//     a bound fact on the current path (loop bound, range, or an
//     explicit i < len(arr) check), else a SLICE_BOUNDS_RISK.
//
// The analysis is intentionally soundy, not sound: it favors precision
// on common idioms (early-return nil guards, for-i-len loops) over
// theoretical completeness, and stays silent when it cannot decide.
package analysis

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strconv"
	"strings"
)

// Severities for Violation.
const (
	SeverityCritical = "CRITICAL"
	SeverityWarning  = "WARNING"
)

// Violation categories.
const (
	CategoryCriticalNilDereference = "CRITICAL_NIL_DEREFERENCE"
	CategoryResourceLeakWarning    = "RESOURCE_LEAK_WARNING"
	CategorySliceBoundsRisk        = "SLICE_BOUNDS_RISK"
)

// Remediation hints.
const (
	hintNilDeref = "Add an explicit nil check (if x == nil { ... }) before dereferencing x on this path."
	hintLeak     = "Add defer f.Close() in the same scope immediately after acquiring the resource."
	hintBounds   = "Guard the index with a bounds check (if i < len(arr)) or iterate with range."
)

// Violation is one failed safety property: what was proven unsafe,
// where, and the offending code.
type Violation struct {
	Severity    string `json:"severity"`    // "CRITICAL" or "WARNING"
	Category    string `json:"category"`    // e.g. "CRITICAL_NIL_DEREFERENCE"
	LineNumber  int    `json:"lineNumber"`  // 0 when no FileSet was supplied
	CodeSnippet string `json:"codeSnippet"` // short rendering of the offending expression
	Message     string `json:"message"`     // what failed and how to fix it
}

// SafetyReport is the outcome of proving one file: a 0.0-100.0 score
// (100 minus 25 per CRITICAL and 10 per WARNING), whether the file
// passed all critical safety proofs, and the violations found.
type SafetyReport struct {
	SafetyScore float64     `json:"safetyScore"`
	IsVerified  bool        `json:"isVerified"`
	Violations  []Violation `json:"violations"`
}

// ProveSafety verifies the safety invariants of file: nil/interface
// safety, resource-leak freedom, and slice/array boundary safety.
// info carries the type information from type-checking file (it may be
// nil, in which case type-dependent rules degrade gracefully rather
// than failing). Because token positions need a *token.FileSet to
// become line numbers, use ProveSafetyWithFileSet when line numbers
// matter; here LineNumber is 0.
func ProveSafety(file *ast.File, info *types.Info) (*SafetyReport, error) {
	return ProveSafetyWithFileSet(file, info, nil)
}

// ProveSafetyWithFileSet is ProveSafety with the FileSet the file was
// parsed with, so violations carry real line numbers.
func ProveSafetyWithFileSet(file *ast.File, info *types.Info, fset *token.FileSet) (*SafetyReport, error) {
	if file == nil {
		return nil, fmt.Errorf("analysis: nil file")
	}
	a := &analyzer{
		fset:       fset,
		info:       info,
		seen:       map[string]bool{},
		violations: []Violation{},
	}
	for _, decl := range file.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok || fn.Body == nil {
			continue
		}
		a.nilChecked = map[varKey]bool{}
		a.collectNilChecks(fn.Body)
		a.walkBlock(fn.Body, newPathEnv())
	}
	return a.report(), nil
}

// nilness is the abstract nil-state of a variable on one path.
type nilness uint8

const (
	nilUnknown nilness = iota
	nilIsNil
	nilNotNil
)

// varKey identifies a variable: by its types.Object when type
// information is available (precise under shadowing), otherwise by
// name.
type varKey struct {
	obj  types.Object
	name string
}

func (k varKey) String() string {
	if k.obj != nil {
		return fmt.Sprintf("obj:%p", k.obj)
	}
	return "name:" + k.name
}

// pathEnv is the abstract state carried along one execution path:
// nil-states per variable and bound facts of the form
// "<varkey><len(<array expr>)".
type pathEnv struct {
	nils  map[varKey]nilness
	facts map[string]bool
}

func newPathEnv() *pathEnv {
	return &pathEnv{nils: map[varKey]nilness{}, facts: map[string]bool{}}
}

func (e *pathEnv) clone() *pathEnv {
	c := newPathEnv()
	for k, v := range e.nils {
		c.nils[k] = v
	}
	for k, v := range e.facts {
		c.facts[k] = v
	}
	return c
}

// join merges branch environments into e: a nil-state survives only if
// every branch agrees, otherwise it becomes unknown; facts survive
// only if present in every branch.
func (e *pathEnv) join(branches ...*pathEnv) {
	if len(branches) == 0 {
		return
	}
	keys := map[varKey]struct{}{}
	for _, b := range branches {
		for k := range b.nils {
			keys[k] = struct{}{}
		}
	}
	joined := make(map[varKey]nilness, len(keys))
	for k := range keys {
		st := branches[0].nils[k]
		agree := true
		for _, b := range branches[1:] {
			if b.nils[k] != st {
				agree = false
				break
			}
		}
		if agree {
			joined[k] = st
		} else {
			joined[k] = nilUnknown
		}
	}
	e.nils = joined
	inter := map[string]bool{}
	for f := range branches[0].facts {
		ok := true
		for _, b := range branches[1:] {
			if !b.facts[f] {
				ok = false
				break
			}
		}
		if ok {
			inter[f] = true
		}
	}
	e.facts = inter
}

type analyzer struct {
	fset       *token.FileSet
	info       *types.Info
	nilChecked map[varKey]bool
	violations []Violation
	seen       map[string]bool
}

func (a *analyzer) report() *SafetyReport {
	score := 100.0
	critical := 0
	for _, v := range a.violations {
		if v.Severity == SeverityCritical {
			score -= 25
			critical++
		} else {
			score -= 10
		}
	}
	if score < 0 {
		score = 0
	}
	return &SafetyReport{
		SafetyScore: score,
		// Verified means every critical safety proof passed.
		// WARNINGs are risks to review, not proof failures.
		IsVerified: critical == 0,
		Violations: a.violations,
	}
}

func (a *analyzer) addViolation(sev, cat string, node ast.Node, pos token.Pos, subject, msg, hint string) {
	line := 0
	if a.fset != nil && pos.IsValid() {
		line = a.fset.Position(pos).Line
	}
	// Dedup on the raw position offset, not the line number: offsets are
	// unique per site even when no FileSet is available for line mapping.
	key := cat + "\x00" + strconv.Itoa(int(pos)) + "\x00" + subject
	if a.seen[key] {
		return
	}
	a.seen[key] = true
	if hint != "" {
		if !strings.HasSuffix(msg, ".") {
			msg += "."
		}
		msg += " " + hint
	}
	a.violations = append(a.violations, Violation{
		Severity:    sev,
		Category:    cat,
		LineNumber:  line,
		CodeSnippet: snippetOf(node),
		Message:     msg,
	})
}

// snippetOf renders a short source snippet identifying the offending
// code. It never fails: unrecognized nodes degrade to "".
func snippetOf(n ast.Node) string {
	if e, ok := n.(ast.Expr); ok {
		return exprText(e)
	}
	return ""
}

func (a *analyzer) vkey(id *ast.Ident) varKey {
	if id == nil {
		return varKey{}
	}
	if a.info != nil {
		if obj := a.info.ObjectOf(id); obj != nil {
			return varKey{obj: obj}
		}
	}
	return varKey{name: id.Name}
}

// collectNilChecks records every variable ever compared against nil in
// the function body. A dereference of such a variable outside a
// nil-guarded region is suspicious: the author knew nil was possible.
func (a *analyzer) collectNilChecks(body *ast.BlockStmt) {
	ast.Inspect(body, func(n ast.Node) bool {
		b, ok := n.(*ast.BinaryExpr)
		if !ok || (b.Op != token.EQL && b.Op != token.NEQ) {
			return true
		}
		var v ast.Expr
		switch {
		case isNilLit(b.X):
			v = b.Y
		case isNilLit(b.Y):
			v = b.X
		default:
			return true
		}
		if id, ok := v.(*ast.Ident); ok {
			a.nilChecked[a.vkey(id)] = true
		}
		return true
	})
}

func isNilLit(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	return ok && id.Name == "nil"
}

// exprText renders small expressions canonically for fact keys.
func exprText(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.SelectorExpr:
		if x := exprText(t.X); x != "" {
			return x + "." + t.Sel.Name
		}
	case *ast.CallExpr:
		if f := exprText(t.Fun); f != "" {
			return f + "(...)"
		}
	case *ast.IndexExpr:
		if x := exprText(t.X); x != "" {
			return x + "[" + exprText(t.Index) + "]"
		}
	case *ast.BasicLit:
		return t.Value
	}
	return ""
}

// boundFact builds the fact key recording that v < len(arr).
func boundFact(k varKey, arr string) string {
	return k.String() + "<len(" + arr + ")"
}

// invalidate drops facts mentioning k (e.g. after k is reassigned).
func (e *pathEnv) invalidate(k varKey) {
	prefix := k.String() + "<"
	for f := range e.facts {
		if strings.HasPrefix(f, prefix) {
			delete(e.facts, f)
		}
	}
}

// assume refines env with the knowledge that cond evaluates to truth.
func (a *analyzer) assume(cond ast.Expr, truth bool, env *pathEnv) {
	for {
		p, ok := cond.(*ast.ParenExpr)
		if !ok {
			break
		}
		cond = p.X
	}
	switch c := cond.(type) {
	case *ast.BinaryExpr:
		switch {
		case c.Op == token.EQL || c.Op == token.NEQ:
			var v ast.Expr
			assertsNil := c.Op == token.EQL
			switch {
			case isNilLit(c.X):
				v = c.Y
			case isNilLit(c.Y):
				v = c.X
			default:
				return
			}
			id, ok := v.(*ast.Ident)
			if !ok {
				return
			}
			st := nilNotNil
			if assertsNil == truth {
				st = nilIsNil
			}
			env.nils[a.vkey(id)] = st
		case (c.Op == token.LSS || c.Op == token.GTR) && truth:
			// i < len(arr)  or  len(arr) > i
			var v, arr ast.Expr
			if c.Op == token.LSS {
				v, arr = c.X, lenArg(c.Y)
			} else {
				v, arr = c.Y, lenArg(c.X)
			}
			id, ok := v.(*ast.Ident)
			if !ok || arr == nil {
				return
			}
			if text := exprText(arr); text != "" {
				env.facts[boundFact(a.vkey(id), text)] = true
			}
		case c.Op == token.LAND && truth:
			a.assume(c.X, true, env)
			a.assume(c.Y, true, env)
		case c.Op == token.LOR && !truth:
			a.assume(c.X, false, env)
			a.assume(c.Y, false, env)
		}
	case *ast.UnaryExpr:
		if c.Op == token.NOT {
			a.assume(c.X, !truth, env)
		}
	}
}

// lenArg returns the array argument if e is a len(...) call.
func lenArg(e ast.Expr) ast.Expr {
	c, ok := e.(*ast.CallExpr)
	if !ok || len(c.Args) != 1 {
		return nil
	}
	if id, ok := c.Fun.(*ast.Ident); !ok || id.Name != "len" {
		return nil
	}
	return c.Args[0]
}

// terminates reports whether s never falls through (return or panic).
func (a *analyzer) terminates(s ast.Stmt) bool {
	switch t := s.(type) {
	case *ast.ReturnStmt:
		return true
	case *ast.ExprStmt:
		if call, ok := t.X.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "panic" {
				return true
			}
		}
	case *ast.IfStmt:
		thenTerm := t.Body != nil && a.blockTerminates(t.Body)
		var elseTerm bool
		switch e := t.Else.(type) {
		case *ast.IfStmt:
			elseTerm = a.terminates(e)
		case *ast.BlockStmt:
			elseTerm = a.blockTerminates(e)
		}
		return thenTerm && elseTerm && t.Else != nil
	}
	return false
}

func (a *analyzer) blockTerminates(b *ast.BlockStmt) bool {
	if b == nil || len(b.List) == 0 {
		return false
	}
	return a.terminates(b.List[len(b.List)-1])
}

// walkBlock walks statements in order; it reports whether control
// never falls through the block.
func (a *analyzer) walkBlock(block *ast.BlockStmt, env *pathEnv) bool {
	if block == nil {
		return false
	}
	a.checkLeaks(block.List)
	for _, stmt := range block.List {
		if a.walkStmt(stmt, env) {
			return true
		}
	}
	return false
}

func (a *analyzer) walkStmt(s ast.Stmt, env *pathEnv) bool {
	if s == nil {
		return false
	}
	switch t := s.(type) {
	case *ast.ExprStmt:
		a.walkExpr(t.X, env)
		return a.terminates(t)
	case *ast.AssignStmt:
		a.walkAssign(t, env)
	case *ast.DeclStmt:
		a.walkDecl(t, env)
	case *ast.IncDecStmt:
		if id, ok := t.X.(*ast.Ident); ok {
			env.invalidate(a.vkey(id))
		}
		a.walkExpr(t.X, env)
	case *ast.ReturnStmt:
		for _, e := range t.Results {
			a.walkExpr(e, env)
		}
		return true
	case *ast.IfStmt:
		return a.walkIf(t, env)
	case *ast.ForStmt:
		a.walkFor(t, env)
	case *ast.RangeStmt:
		a.walkRange(t, env)
	case *ast.SwitchStmt:
		a.walkSwitch(t, env)
	case *ast.TypeSwitchStmt:
		a.walkTypeSwitch(t, env)
	case *ast.SelectStmt:
		for _, stmt := range t.Body.List {
			if cc, ok := stmt.(*ast.CommClause); ok {
				if cc.Comm != nil {
					a.walkStmt(cc.Comm, env)
				}
				a.walkBlock(&ast.BlockStmt{List: cc.Body}, env.clone())
			}
		}
	case *ast.BlockStmt:
		return a.walkBlock(t, env)
	case *ast.DeferStmt:
		a.walkExpr(t.Call, env)
	case *ast.GoStmt:
		a.walkExpr(t.Call, env)
	case *ast.SendStmt:
		a.walkExpr(t.Value, env)
	case *ast.LabeledStmt:
		return a.walkStmt(t.Stmt, env)
	}
	return false
}

func (a *analyzer) walkIf(s *ast.IfStmt, env *pathEnv) bool {
	if s.Init != nil {
		a.walkStmt(s.Init, env)
		a.checkLeaks(append([]ast.Stmt{s.Init}, s.Body.List...))
	}
	a.walkExpr(s.Cond, env)
	thenEnv := env.clone()
	elseEnv := env.clone()
	a.assume(s.Cond, true, thenEnv)
	a.assume(s.Cond, false, elseEnv)
	thenTerm := a.walkBlock(s.Body, thenEnv)
	var elseTerm bool
	if s.Else != nil {
		switch e := s.Else.(type) {
		case *ast.IfStmt:
			elseTerm = a.walkIf(e, elseEnv)
		case *ast.BlockStmt:
			elseTerm = a.walkBlock(e, elseEnv)
		}
	}
	switch {
	case thenTerm && elseTerm:
		return true
	case thenTerm:
		*env = *elseEnv
	case elseTerm:
		*env = *thenEnv
	default:
		env.join(thenEnv, elseEnv)
	}
	return false
}

func (a *analyzer) walkFor(s *ast.ForStmt, env *pathEnv) {
	if s.Init != nil {
		a.walkStmt(s.Init, env)
		if s.Body != nil {
			a.checkLeaks(append([]ast.Stmt{s.Init}, s.Body.List...))
		}
	}
	if s.Cond != nil {
		a.walkExpr(s.Cond, env)
	}
	bodyEnv := env.clone()
	if s.Cond != nil {
		a.assume(s.Cond, true, bodyEnv)
	}
	a.walkBlock(s.Body, bodyEnv)
	// Post-loop state stays the pre-loop state: the loop may not run.
}

func (a *analyzer) walkRange(s *ast.RangeStmt, env *pathEnv) {
	a.walkExpr(s.X, env)
	bodyEnv := env.clone()
	if key, ok := s.Key.(*ast.Ident); ok && key.Name != "_" && a.info != nil {
		if isSliceOrArray(a.info.TypeOf(s.X)) {
			if arr := exprText(s.X); arr != "" {
				bodyEnv.facts[boundFact(a.vkey(key), arr)] = true
			}
		}
	}
	a.walkBlock(s.Body, bodyEnv)
}

func (a *analyzer) walkSwitch(s *ast.SwitchStmt, env *pathEnv) {
	if s.Init != nil {
		a.walkStmt(s.Init, env)
	}
	if s.Tag != nil {
		a.walkExpr(s.Tag, env)
	}
	for _, stmt := range s.Body.List {
		cc, ok := stmt.(*ast.CaseClause)
		if !ok {
			continue
		}
		for _, e := range cc.List {
			a.walkExpr(e, env)
		}
		a.walkBlock(&ast.BlockStmt{List: cc.Body}, env.clone())
	}
}

func (a *analyzer) walkTypeSwitch(s *ast.TypeSwitchStmt, env *pathEnv) {
	if s.Init != nil {
		a.walkStmt(s.Init, env)
	}
	a.walkStmt(s.Assign, env)
	var vid *ast.Ident
	if as, ok := s.Assign.(*ast.AssignStmt); ok && len(as.Lhs) > 0 {
		vid, _ = as.Lhs[0].(*ast.Ident)
	}
	for _, stmt := range s.Body.List {
		cc, ok := stmt.(*ast.CaseClause)
		if !ok {
			continue
		}
		clauseEnv := env.clone()
		for _, e := range cc.List {
			a.walkExpr(e, env)
			if isNilLit(e) && vid != nil {
				clauseEnv.nils[a.vkey(vid)] = nilIsNil
			}
		}
		a.walkBlock(&ast.BlockStmt{List: cc.Body}, clauseEnv)
	}
}

func (a *analyzer) walkAssign(s *ast.AssignStmt, env *pathEnv) {
	for _, rhs := range s.Rhs {
		a.walkExpr(rhs, env)
	}
	multi := len(s.Rhs) == 1 && len(s.Lhs) > 1
	for i, lhs := range s.Lhs {
		id, ok := lhs.(*ast.Ident)
		if !ok || id.Name == "_" {
			continue
		}
		key := a.vkey(id)
		if multi {
			env.nils[key] = nilUnknown
			continue
		}
		var rhs ast.Expr
		if i < len(s.Rhs) {
			rhs = s.Rhs[i]
		}
		env.nils[key] = a.nilnessOf(rhs, env)
		env.invalidate(key)
	}
}

func (a *analyzer) walkDecl(d *ast.DeclStmt, env *pathEnv) {
	gd, ok := d.Decl.(*ast.GenDecl)
	if !ok {
		return
	}
	for _, spec := range gd.Specs {
		vs, ok := spec.(*ast.ValueSpec)
		if !ok {
			continue
		}
		for i, name := range vs.Names {
			if name.Name == "_" {
				continue
			}
			key := a.vkey(name)
			var rhs ast.Expr
			switch {
			case i < len(vs.Values):
				rhs = vs.Values[i]
			case len(vs.Values) == 1 && len(vs.Names) > 1:
				rhs = nil // a, b = f(): unmappable
			}
			if rhs == nil {
				// Zero value: pointers and interfaces start nil.
				t := a.identType(name)
				if isPointerType(t) || isInterfaceType(t) {
					env.nils[key] = nilIsNil
				} else {
					env.nils[key] = nilUnknown
				}
			} else {
				a.walkExpr(rhs, env)
				env.nils[key] = a.nilnessOf(rhs, env)
			}
			env.invalidate(key)
		}
	}
}

func (a *analyzer) nilnessOf(e ast.Expr, env *pathEnv) nilness {
	if e == nil {
		return nilUnknown
	}
	switch t := e.(type) {
	case *ast.Ident:
		if t.Name == "nil" {
			return nilIsNil
		}
		return env.nils[a.vkey(t)]
	case *ast.UnaryExpr:
		if t.Op == token.AND {
			return nilNotNil
		}
	case *ast.CallExpr:
		if id, ok := t.Fun.(*ast.Ident); ok && (id.Name == "new" || id.Name == "make") {
			return nilNotNil
		}
	case *ast.FuncLit:
		return nilNotNil
	case *ast.CompositeLit:
		return nilNotNil
	}
	return nilUnknown
}

// walkExpr visits every expression, checking dereferences and indices.
func (a *analyzer) walkExpr(e ast.Expr, env *pathEnv) {
	if e == nil {
		return
	}
	ast.Inspect(e, func(n ast.Node) bool {
		if n == nil {
			return true
		}
		switch t := n.(type) {
		case *ast.FuncLit:
			// Closures get a fresh function-like analysis.
			if t.Body != nil {
				saved := a.nilChecked
				a.nilChecked = map[varKey]bool{}
				a.collectNilChecks(t.Body)
				a.walkBlock(t.Body, newPathEnv())
				a.nilChecked = saved
			}
			return false
		case *ast.StarExpr:
			a.checkDeref(t.X, env, "pointer dereference")
		case *ast.SelectorExpr:
			switch {
			case isPointerType(a.exprType(t.X)):
				a.checkDeref(t.X, env, "pointer dereference")
			case isInterfaceType(a.exprType(t.X)):
				a.checkDeref(t.X, env, "interface method invocation")
			}
		case *ast.IndexExpr:
			a.checkIndex(t, env)
		}
		return true
	})
}

func (a *analyzer) checkDeref(x ast.Expr, env *pathEnv, what string) {
	id, ok := x.(*ast.Ident)
	if !ok || id.Name == "_" {
		return
	}
	key := a.vkey(id)
	switch env.nils[key] {
	case nilIsNil:
		a.addViolation(SeverityCritical, CategoryCriticalNilDereference, id, id.Pos(), id.Name,
			fmt.Sprintf("definite nil %s: %q is nil here", what, id.Name), hintNilDeref)
	case nilUnknown:
		if a.nilChecked[key] {
			a.addViolation(SeverityWarning, CategoryCriticalNilDereference, id, id.Pos(), id.Name,
				fmt.Sprintf("possible nil %s: %q is nil-checked in this function but not guarded on this path", what, id.Name), hintNilDeref)
		}
	}
}

func (a *analyzer) checkIndex(t *ast.IndexExpr, env *pathEnv) {
	if a.info == nil {
		return
	}
	if !isSliceOrArray(a.info.TypeOf(t.X)) {
		return // maps and other indexables are safe
	}
	switch idx := t.Index.(type) {
	case *ast.Ident:
		if idx.Name == "_" {
			return
		}
		need := boundFact(a.vkey(idx), exprText(t.X))
		if !env.facts[need] {
			a.addViolation(SeverityWarning, CategorySliceBoundsRisk, t, t.Pos(), idx.Name,
				fmt.Sprintf("unguarded slice index: no check ensures %s < len(%s)", idx.Name, exprText(t.X)), hintBounds)
		}
	case *ast.BasicLit:
		// Constant indices cannot be assessed without lengths; skip.
	default:
		a.addViolation(SeverityWarning, CategorySliceBoundsRisk, t, t.Pos(), "index",
			fmt.Sprintf("index expression cannot be statically bounded for %s", exprText(t.X)), hintBounds)
	}
}

// --- resource leak detection ---

// resourceMethods are method names whose results conventionally need
// closing (database/sql and friends).
var resourceMethods = map[string]bool{"Query": true, "QueryContext": true}

// acq records a resource acquisition site.
type acq struct {
	name string
	pos  token.Pos
	call *ast.CallExpr
}

// checkLeaks verifies that every resource acquired in stmts has a
// matching defer x.Close() in the same statement list.
func (a *analyzer) checkLeaks(stmts []ast.Stmt) {
	acquired := map[varKey]acq{}
	for _, s := range stmts {
		switch t := s.(type) {
		case *ast.AssignStmt:
			if len(t.Rhs) == 0 {
				continue
			}
			call, ok := t.Rhs[0].(*ast.CallExpr)
			if !ok {
				continue
			}
			a.recordAcq(call, t.Lhs, t.Pos(), acquired)
		case *ast.DeclStmt:
			gd, ok := t.Decl.(*ast.GenDecl)
			if !ok {
				continue
			}
			for _, spec := range gd.Specs {
				vs, ok := spec.(*ast.ValueSpec)
				if !ok || len(vs.Values) == 0 {
					continue
				}
				call, ok := vs.Values[0].(*ast.CallExpr)
				if !ok {
					continue
				}
				lhs := make([]ast.Expr, len(vs.Names))
				for i, n := range vs.Names {
					lhs[i] = n
				}
				a.recordAcq(call, lhs, vs.Pos(), acquired)
			}
		}
	}
	if len(acquired) == 0 {
		return
	}
	closed := map[varKey]bool{}
	for _, s := range stmts {
		ds, ok := s.(*ast.DeferStmt)
		if !ok {
			continue
		}
		a.markClosed(ds.Call, closed)
	}
	for key, ac := range acquired {
		if closed[key] {
			continue
		}
		if ac.name == "_" {
			a.addViolation(SeverityWarning, CategoryResourceLeakWarning, ac.call, ac.pos, "discarded",
				"resource result discarded: the acquired handle cannot be closed", hintLeak)
			continue
		}
		a.addViolation(SeverityWarning, CategoryResourceLeakWarning, ac.call, ac.pos, ac.name,
			fmt.Sprintf("resource %q acquired here may leak: no defer Close() in the same scope", ac.name), hintLeak)
	}
}

// recordAcq maps a resource-acquiring call to its receiving variable.
func (a *analyzer) recordAcq(call *ast.CallExpr, lhs []ast.Expr, pos token.Pos, acquired map[varKey]acq) {
	idx := a.resourceResultIndex(call)
	if idx < 0 || idx >= len(lhs) {
		return
	}
	id, ok := lhs[idx].(*ast.Ident)
	if !ok {
		return
	}
	if t := a.identType(id); isErrorType(t) {
		return
	}
	acquired[a.vkey(id)] = acq{name: id.Name, pos: pos, call: call}
}

// resourceResultIndex returns the result index holding the closable
// resource, or -1 if the call is not a resource acquisition.
func (a *analyzer) resourceResultIndex(call *ast.CallExpr) int {
	// Type-based: first non-error result with a Close method.
	if sig := a.callSignature(call); sig != nil {
		if res := sig.Results(); res != nil {
			for i := 0; i < res.Len(); i++ {
				rt := res.At(i).Type()
				if isErrorType(rt) {
					continue
				}
				if hasCloseMethod(rt) {
					return i
				}
			}
		}
	}
	// Known constructors whose result needs closing even though its
	// type lacks a Close method (http.Get: close resp.Body).
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return -1
	}
	name := sel.Sel.Name
	switch {
	case name == "Get" || name == "Post" || name == "Head":
		if a.isPkg(sel, "net/http") || identName(sel.X) == "http" {
			return 0
		}
	case resourceMethods[name]:
		return 0
	}
	return -1
}

// markClosed records variables closed by a deferred call:
// defer f.Close(), defer f.Body.Close(), or a deferred closure
// containing f.Close().
func (a *analyzer) markClosed(call *ast.CallExpr, closed map[varKey]bool) {
	mark := func(c *ast.CallExpr) {
		sel, ok := c.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Close" {
			return
		}
		if id := rootIdent(sel.X); id != nil {
			closed[a.vkey(id)] = true
		}
	}
	mark(call)
	if lit, ok := call.Fun.(*ast.FuncLit); ok && lit.Body != nil {
		ast.Inspect(lit.Body, func(n ast.Node) bool {
			if c, ok := n.(*ast.CallExpr); ok {
				mark(c)
			}
			return true
		})
	}
}

// rootIdent drills through selector chains to the base identifier:
// f.Body.Close -> f.
func rootIdent(e ast.Expr) *ast.Ident {
	for {
		switch t := e.(type) {
		case *ast.Ident:
			return t
		case *ast.SelectorExpr:
			e = t.X
		default:
			return nil
		}
	}
}

func identName(e ast.Expr) string {
	if id, ok := e.(*ast.Ident); ok {
		return id.Name
	}
	return ""
}

// --- type helpers (all nil-safe) ---

func (a *analyzer) exprType(e ast.Expr) types.Type {
	if a.info == nil || e == nil {
		return nil
	}
	return a.info.TypeOf(e)
}

func (a *analyzer) identType(id *ast.Ident) types.Type {
	if a.info == nil || id == nil {
		return nil
	}
	if t := a.info.TypeOf(id); t != nil {
		return t
	}
	if obj := a.info.ObjectOf(id); obj != nil {
		return obj.Type()
	}
	return nil
}

func (a *analyzer) callSignature(call *ast.CallExpr) *types.Signature {
	if a.info == nil || call == nil {
		return nil
	}
	sig, _ := a.info.TypeOf(call.Fun).(*types.Signature)
	return sig
}

// isPkg reports whether sel denotes a member of the given package path.
func (a *analyzer) isPkg(sel *ast.SelectorExpr, path string) bool {
	if a.info == nil {
		return false
	}
	obj := a.info.ObjectOf(sel.Sel)
	if obj == nil {
		return false
	}
	pkg := obj.Pkg()
	return pkg != nil && pkg.Path() == path
}

func isPointerType(t types.Type) bool {
	_, ok := t.(*types.Pointer)
	return ok
}

func isInterfaceType(t types.Type) bool {
	_, ok := t.(*types.Interface)
	return ok
}

func isSliceOrArray(t types.Type) bool {
	switch t.(type) {
	case *types.Slice, *types.Array:
		return true
	}
	return false
}

func isErrorType(t types.Type) bool {
	if t == nil {
		return false
	}
	errObj := types.Universe.Lookup("error")
	if errObj == nil {
		return false
	}
	iface, ok := errObj.Type().Underlying().(*types.Interface)
	if !ok {
		return false
	}
	return types.Implements(t, iface)
}

// hasCloseMethod reports whether the method set of t (or *t) contains
// a Close method.
func hasCloseMethod(t types.Type) bool {
	if t == nil {
		return false
	}
	recv := t
	if _, ok := t.(*types.Pointer); !ok {
		recv = types.NewPointer(t)
	}
	return types.NewMethodSet(recv).Lookup(nil, "Close") != nil
}

/*
Runnable example: run ProveSafety on a sample Go AST and print the
JSON safety report.

package main

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"

	"github.com/golangast/gollemer/pkg/analysis"
)

const sample = `package p

import "os"

type T struct{ X int }

func risky(p *T, ok bool) int {
	if p == nil {
		println("nil")
	}
	f, _ := os.Open("data.txt")
	_ = f
	if ok {
		return p.X
	}
	return 0
}
`

func main() {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "sample.go", sample, 0)
	if err != nil {
		panic(err)
	}
	info := &types.Info{
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
		Types: make(map[ast.Expr]types.TypeAndValue),
	}
	cfg := types.Config{Importer: importer.Default()}
	if _, err := cfg.Check("p", fset, []*ast.File{file}, info); err != nil {
		panic(err)
	}
	report, err := analysis.ProveSafetyWithFileSet(file, info, fset)
	if err != nil {
		panic(err)
	}
	out, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(out))
}
*/
