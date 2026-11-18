package analyze

// The goroutine-result piece (editgoroutine.go): "have the
// goroutine return a string" — give an existing anonymous
// goroutine a way to hand a value back to the function that
// started it.
//
// A goroutine cannot return a value in Go: a go statement
// throws any results away, because nothing is waiting to
// receive them. The language's own answer is a channel —
// the goroutine sends, the starter receives — and that is
// the whole transformation, assembled here from the file's
// real structure:
//
//	resultCh := make(chan string)   // before the go statement
//	go func() {
//		// …the goroutine's own body, untouched…
//		resultCh <- ""              // its last step: the send
//	}()
//	result := <-resultCh            // right after: the receive
//	_ = result
//
// The send carries the type's zero value under a TODO in
// the book's voice: which expression is "the result" is
// the user's knowledge, not something a planner may guess,
// so the piece wires the road and marks where the real
// value goes. The receive lands immediately after the go
// statement, so the function waits there until the
// goroutine sends — the waiting is what "returning" means
// here, and the plan's notes say so in plain words.
//
// V1 scope, held honestly: the goroutine must be a plain
// go func() { ... }() written out in the open — no
// parameters, no declared results, no send of its own yet.
// A go statement that calls a named function declines:
// its result would have to come from inside that function,
// which is a different edit. Several goroutines with no
// position named declines too (the chat layer turns that
// one into a question first; this refusal is the fallback
// that keeps the planner honest when called directly).

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

// goroutineResultZeros is the piece's closed type set: the
// result types a sentence may name, each with the zero
// value the send stands in with. A channel for one of the
// project's own types is a bigger piece — it is not
// silently approximated here.
var goroutineResultZeros = map[string]string{
	"string":  `""`,
	"int":     "0",
	"bool":    "false",
	"float64": "0",
}

// GoroutineResultZeroValue reports the zero-value literal
// a result type's send stands in with, and whether the
// type is one this piece builds for. The chat layer checks
// membership with it before doing any project work, so an
// unknown type is refused in the piece's own words at the
// claim, not discovered at the planner.
func GoroutineResultZeroValue(typ string) (string, bool) {
	z, ok := goroutineResultZeros[typ]
	return z, ok
}

// GoroutineResultTypes renders the closed set for refusal
// sentences, in the order the docs name it.
func GoroutineResultTypes() string {
	return "string, int, bool, or float64"
}

// goroutineSite is one go statement inside a target file,
// with the declaration that holds it.
type goroutineSite struct {
	decl   *ast.FuncDecl
	fnDisp string // "main", or "Store.Run" for a method
	goStmt *ast.GoStmt
	line   int
}

// goroutineFuncDecls settles the declarations a function
// name scopes to: every FuncDecl when no name was given;
// the name's matches otherwise (exact case wins, and a
// case-folded match is forgiven when the exact pool is
// empty — the instantiate discipline). found is false
// only when a name was given and nothing in the file
// carries it.
func goroutineFuncDecls(fileAST *ast.File, funcName string) (decls []*ast.FuncDecl, found bool) {
	var exact, folded []*ast.FuncDecl
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		switch {
		case funcName == "" || fd.Name.Name == funcName:
			exact = append(exact, fd)
		case strings.EqualFold(fd.Name.Name, funcName):
			folded = append(folded, fd)
		}
	}
	if funcName == "" || len(exact) > 0 {
		return exact, true
	}
	return folded, len(folded) > 0
}

// goroutineFuncDisp renders a declaration the way refusal
// and summary sentences name it: the bare name, or the
// receiver's base type and the name for a method.
func goroutineFuncDisp(fd *ast.FuncDecl) string {
	if rb := instantiateRecvBase(fd); rb != "" {
		return rb + "." + fd.Name.Name
	}
	return fd.Name.Name
}

// scanGoroutineSites collects one declaration set's go
// statements in source order (an outer goroutine ahead of
// any goroutine nested inside it — ast.Inspect's order).
func scanGoroutineSites(fset *token.FileSet, decls []*ast.FuncDecl) []goroutineSite {
	var sites []goroutineSite
	for _, d := range decls {
		if d.Body == nil {
			continue
		}
		disp := goroutineFuncDisp(d)
		ast.Inspect(d.Body, func(n ast.Node) bool {
			gs, ok := n.(*ast.GoStmt)
			if !ok {
				return true
			}
			sites = append(sites, goroutineSite{
				decl:   d,
				fnDisp: disp,
				goStmt: gs,
				line:   lineOfPos(fset, gs.Pos()),
			})
			return true
		})
	}
	return sites
}

// GoroutineResultSite is one go statement the chat layer
// may offer as a clarifying-question candidate: the
// function that holds it and its line in the file.
type GoroutineResultSite struct {
	Func string
	Line int
}

// GoroutineResultSites lists the go statements in a file
// (or in one named function in it) in source order — the
// chat layer's candidate source for its which-one
// question. It returns nil whenever the scope does not
// resolve (an unreadable file, a function the file does
// not declare): every other state is the planner's to
// answer, through executeEditPlan as usual.
func (p *Project) GoroutineResultSites(rel, funcName string) []GoroutineResultSite {
	_, fset, fileAST, refusal := p.readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return nil
	}
	decls, found := goroutineFuncDecls(fileAST, funcName)
	if funcName != "" && !found {
		return nil
	}
	sites := scanGoroutineSites(fset, decls)
	out := make([]GoroutineResultSite, len(sites))
	for i, s := range sites {
		out[i] = GoroutineResultSite{Func: s.fnDisp, Line: s.line}
	}
	return out
}

// goroutineSiteShape classifies one site: the func literal
// when the go statement runs one (nil when it calls
// something else — callee then names what it runs), and
// the three v1 shape facts about the literal.
func goroutineSiteShape(fset *token.FileSet, s goroutineSite) (lit *ast.FuncLit, callee string, params, results, sends bool) {
	call := s.goStmt.Call
	if call == nil {
		return nil, "", false, false, false
	}
	fl, ok := call.Fun.(*ast.FuncLit)
	if !ok {
		return nil, exprText(fset, call.Fun), false, false, false
	}
	params = fl.Type.Params != nil && len(fl.Type.Params.List) > 0
	results = fl.Type.Results != nil && len(fl.Type.Results.List) > 0
	if fl.Body != nil {
		ast.Inspect(fl.Body, func(n ast.Node) bool {
			if _, ok := n.(*ast.SendStmt); ok {
				sends = true
				return false
			}
			return true
		})
	}
	return fl, "", params, results, sends
}

// goroutineScopeWords names the resolved scope for
// sentences: a named function in its file, or the file.
func goroutineScopeWords(p *Project, req EditRequest) string {
	if req.FuncName != "" {
		return fmt.Sprintf("%s in %s", req.FuncName, p.LinkPath(req.File))
	}
	return p.LinkPath(req.File)
}

// planGoroutineResult assembles the channel transformation
// for one goroutine — see the file header.
func (p *Project) planGoroutineResult(req EditRequest) (*EditPlan, string) {
	rel := strings.TrimSpace(req.File)
	if rel == "" {
		return nil, "that edit needs a file to work in."
	}
	zero, known := goroutineResultZeros[req.GoroutineType]
	if !known {
		return nil, fmt.Sprintf("%q isn't a result type I can build this with yet — %s are the ones this piece knows. (A channel carrying one of your own types is a bigger piece.)", req.GoroutineType, GoroutineResultTypes())
	}
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
	decls, found := goroutineFuncDecls(fileAST, req.FuncName)
	if req.FuncName != "" && !found {
		return nil, fmt.Sprintf("there's no function named %q in %s — the goroutine has to live in a function that's already there.", req.FuncName, p.LinkPath(rel))
	}
	scope := goroutineScopeWords(p, req)
	sites := scanGoroutineSites(fset, decls)

	var site goroutineSite
	switch {
	case req.GoroutineOrdinal > 0:
		if req.GoroutineOrdinal > len(sites) {
			noun := "goroutines"
			if len(sites) == 1 {
				noun = "goroutine"
			}
			return nil, fmt.Sprintf("%s holds %d %s, so there is no %s one — the positions count from the first go statement in %s.", scope, len(sites), noun, ordinalWord(req.GoroutineOrdinal), scope)
		}
		site = sites[req.GoroutineOrdinal-1]
	case len(sites) == 0:
		if req.FuncName != "" {
			return nil, fmt.Sprintf("there's no goroutine in %s — this piece gives an existing goroutine its way to send a result back. Add one first (\"add a goroutine to %s\"), then ask again.", scope, req.FuncName)
		}
		return nil, fmt.Sprintf("there's no goroutine in %s — this piece gives an existing goroutine its way to send a result back. Add one first (\"add a goroutine to main\" if main is where it belongs), then ask again.", scope)
	case len(sites) == 1:
		site = sites[0]
	default:
		var b strings.Builder
		fmt.Fprintf(&b, "there's more than one goroutine in %s — say which one, by its position (\"the first goroutine in %s\"):", scope, sites[0].fnDisp)
		for i, s := range sites {
			fmt.Fprintf(&b, "\n  - the %s goroutine, in %s (%s:%d)", ordinalWord(i+1), s.fnDisp, p.LinkPath(rel), s.line)
		}
		return nil, b.String()
	}

	lit, callee, params, results, sends := goroutineSiteShape(fset, site)
	here := fmt.Sprintf("the goroutine in %s (%s:%d)", site.fnDisp, p.LinkPath(rel), site.line)
	switch {
	case lit == nil:
		return nil, fmt.Sprintf("%s runs %s — its result would have to come from inside %s, and changing that function is a different edit. This piece covers a goroutine written out as go func() { ... }() right there in %s.", here, callee, callee, site.fnDisp)
	case params:
		return nil, fmt.Sprintf("%s takes a value in — %s — and this piece covers the plain shape, go func() { ... }() with nothing passed in. It would need the channel wired by hand, or the goroutine reshaped to take nothing in first.", here, stmtText(fset, lit.Type))
	case results:
		return nil, fmt.Sprintf("%s already declares a result — %s — and a go statement throws a goroutine's results away, so as written they go nowhere. Take the result off the func literal and ask again, and I'll wire the channel version, the one that arrives.", here, stmtText(fset, lit.Type))
	case sends:
		return nil, fmt.Sprintf("%s already sends on a channel — it already has its way to pass a value back, so there's nothing for this piece to add.", here)
	}

	// The placement is line-shaped: the channel line goes
	// before the go statement's line and the receive after
	// its last line. A go statement sharing its line with
	// other code has no exact home for either — refuse
	// plainly rather than splice into the middle of a line.
	goLine := lineOfPos(fset, site.goStmt.Pos())
	goEndLine := lineOfPos(fset, site.goStmt.End()-1)
	if !goStmtLineIsolated(src, site.goStmt, fset) {
		return nil, fmt.Sprintf("%s shares its line with other code — I place the channel lines by line, so put the go statement on a line of its own and ask again.", here)
	}

	// The two new names: resultCh for the channel, result
	// for the received value — the first pair the enclosing
	// function does not already use, counted up
	// deterministically (resultCh2/result2, …).
	taken := instantiateLocalNames(site.decl)
	chName, resName := "", ""
	for i := 1; i <= 99; i++ {
		c, r := "resultCh", "result"
		if i > 1 {
			c, r = fmt.Sprintf("resultCh%d", i), fmt.Sprintf("result%d", i)
		}
		if !taken[c] && !taken[r] {
			chName, resName = c, r
			break
		}
	}
	if chName == "" {
		return nil, fmt.Sprintf("%s already uses every result name I would pick (result, result2, …) — rename one of them and ask again.", site.fnDisp)
	}

	indent := lineIndent(src, goLine)
	body := lit.Body
	innerIndent := indent + "\t"
	if body != nil && len(body.List) > 0 {
		if ind := lineIndent(src, lineOfPos(fset, body.List[len(body.List)-1].Pos())); ind != "" {
			innerIndent = ind
		}
	}
	todo := fmt.Sprintf("// TODO: the goroutine's real result goes here — %s is only the zero value, sent until the goroutine has one", zero)
	send := chName + " <- " + zero

	var edits []textEdit
	// The channel, declared directly before the go statement.
	edits = append(edits, textEdit{
		start: lineStartOffset(src, goLine),
		end:   lineStartOffset(src, goLine),
		text:  indent + chName + " := make(chan " + req.GoroutineType + ")\n",
	})
	// The send, as the goroutine body's last step.
	switch {
	case body == nil:
		return nil, fmt.Sprintf("%s has no body to send from.", here)
	case lineOfPos(fset, body.Lbrace) == lineOfPos(fset, body.Rbrace):
		edits = append(edits, textEdit{
			start: offOf(fset, body.Rbrace),
			end:   offOf(fset, body.Rbrace),
			text:  "\n" + innerIndent + todo + "\n" + innerIndent + send + "\n" + indent,
		})
	default:
		edits = append(edits, textEdit{
			start: lineStartOffset(src, lineOfPos(fset, body.Rbrace)),
			end:   lineStartOffset(src, lineOfPos(fset, body.Rbrace)),
			text:  innerIndent + todo + "\n" + innerIndent + send + "\n",
		})
	}
	// The receive, directly after the go statement.
	recvAt := lineStartOffset(src, goEndLine+1)
	recvText := indent + resName + " := <-" + chName + "\n" + indent + "_ = " + resName + "\n"
	if recvAt >= len(src) && len(src) > 0 && src[len(src)-1] != '\n' {
		recvText = "\n" + recvText
	}
	edits = append(edits, textEdit{start: recvAt, end: recvAt, text: recvText})

	ef := finishFile(rel, src, applyTextEdits(src, edits))
	recvBase := instantiateRecvBase(site.decl)
	if bad := checkGoroutineResult(ef.After, site.decl.Name.Name, recvBase, chName, resName, req.GoroutineType); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	notes := []string{
		"A goroutine can't hand a value back by returning — a go statement throws its results away. The channel is Go's way back: the goroutine sends, and the function that started it receives.",
		fmt.Sprintf("The receive sits right after the go statement, so %s waits there until the goroutine sends — the lines after it run with the value in hand.", site.fnDisp),
		fmt.Sprintf("For now the goroutine sends %s, the zero value for %s, under a TODO where the real result goes — swap in the real value when the goroutine works one out.", zero, req.GoroutineType),
		fmt.Sprintf("The \"_ = %s\" line after the receive keeps the build green until %s uses %s — delete it once it does.", resName, site.fnDisp, resName),
	}
	if goroutineBodyReturns(lit) {
		notes = append(notes, "One thing to know: the goroutine's body has a return in it — a path that returns early never reaches the send, and the receive would wait forever. Keep every path flowing to the send at the end.")
	}

	fnName := site.decl.Name.Name
	plan := &EditPlan{
		Summary:    fmt.Sprintf("Give the goroutine in %s in %s a way to return a %s: a channel (%s) declared before it, a send inside it, and the receive (%s := <-%s) right after it", site.fnDisp, p.LinkPath(rel), req.GoroutineType, chName, resName, chName),
		VerifyDesc: fmt.Sprintf("the goroutine in %s now sends its %s back on the %s channel, and %s receives it", site.fnDisp, req.GoroutineType, chName, site.fnDisp),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, chName+" := make(chan "+req.GoroutineType+")")),
		Files:      []EditFile{ef},
		Notes:      notes,
		verify: func(after *Project) bool {
			_, _, ast2, refusal := after.readProjectFile(rel)
			if refusal != "" || ast2 == nil {
				return false
			}
			return checkGoroutineResultIn(ast2, fnName, recvBase, chName, resName, req.GoroutineType) == ""
		},
	}
	return plan, ""
}

// ordinalWord renders a small position the way the
// sentences say it (first, second, third, …). Positions
// past fifth fall back to the numeral — the grammar's
// named positions stop there too.
func ordinalWord(n int) string {
	switch n {
	case 1:
		return "first"
	case 2:
		return "second"
	case 3:
		return "third"
	case 4:
		return "fourth"
	case 5:
		return "fifth"
	}
	return fmt.Sprintf("%dth", n)
}

// goStmtLineIsolated reports whether a go statement
// starts its line (only whitespace before it) and ends
// its line (only whitespace or a // comment after it) —
// the shape the line-anchored inserts need.
func goStmtLineIsolated(src []byte, gs *ast.GoStmt, fset *token.FileSet) bool {
	start := offOf(fset, gs.Pos())
	lineStart := start
	for lineStart > 0 && src[lineStart-1] != '\n' {
		lineStart--
	}
	for i := lineStart; i < start; i++ {
		if src[i] != ' ' && src[i] != '\t' {
			return false
		}
	}
	end := offOf(fset, gs.End())
	for i := end; i < len(src) && src[i] != '\n'; i++ {
		if src[i] == '/' && i+1 < len(src) && src[i+1] == '/' {
			return true // the rest of the line is a comment
		}
		if src[i] != ' ' && src[i] != '\t' && src[i] != '\r' {
			return false
		}
	}
	return true
}

// goroutineBodyReturns reports whether a goroutine's body
// holds a return statement anywhere — the early-exit
// shape the plan's honesty note names.
func goroutineBodyReturns(lit *ast.FuncLit) bool {
	found := false
	if lit == nil || lit.Body == nil {
		return false
	}
	ast.Inspect(lit.Body, func(n ast.Node) bool {
		if _, ok := n.(*ast.ReturnStmt); ok {
			found = true
			return false
		}
		return true
	})
	return found
}

// checkGoroutineResult is the assembled-bytes self-check:
// the target function must now hold the whole pattern —
// channel declared, go statement sending on it, receive
// after — in one statement list, in that order.
func checkGoroutineResult(after []byte, fnName, recvBase, chName, resName, typ string) string {
	_, fileAST, err := parseGoSrc("goroutine.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	return checkGoroutineResultIn(fileAST, fnName, recvBase, chName, resName, typ)
}

func checkGoroutineResultIn(fileAST *ast.File, fnName, recvBase, chName, resName, typ string) string {
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != fnName || fd.Body == nil {
			continue
		}
		if instantiateRecvBase(fd) != recvBase {
			continue
		}
		if goroutinePatternIn(fd.Body, chName, resName, typ) {
			return ""
		}
		return fmt.Sprintf("%s doesn't hold the channel pattern after assembly", fnName)
	}
	return fmt.Sprintf("the assembled file has no function named %s", fnName)
}

// goroutinePatternIn walks every statement list in a
// function body (blocks and case bodies alike) looking
// for the plan's trio in order within ONE list: the
// channel declaration, the go statement whose literal
// sends on the channel, and the receive.
func goroutinePatternIn(body *ast.BlockStmt, chName, resName, typ string) bool {
	found := false
	var lists [][]ast.Stmt
	ast.Inspect(body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.BlockStmt:
			lists = append(lists, n.List)
		case *ast.CaseClause:
			lists = append(lists, n.Body)
		}
		return true
	})
	for _, list := range lists {
		stage := 0
		for _, stmt := range list {
			switch stage {
			case 0:
				if isChanDecl(stmt, chName, typ) {
					stage = 1
				}
			case 1:
				if gs, ok := stmt.(*ast.GoStmt); ok && goLitSendsOn(gs, chName) {
					stage = 2
				}
			case 2:
				if isReceiveAssign(stmt, chName, resName) {
					found = true
				}
			}
			if found {
				return true
			}
		}
	}
	return false
}

// isChanDecl reports whether a statement is
// "<chName> := make(chan <typ>)".
func isChanDecl(stmt ast.Stmt, chName, typ string) bool {
	as, ok := stmt.(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
		return false
	}
	id, ok := as.Lhs[0].(*ast.Ident)
	if !ok || id.Name != chName {
		return false
	}
	call, ok := as.Rhs[0].(*ast.CallExpr)
	if !ok || len(call.Args) != 1 {
		return false
	}
	fn, ok := call.Fun.(*ast.Ident)
	if !ok || fn.Name != "make" {
		return false
	}
	ct, ok := call.Args[0].(*ast.ChanType)
	if !ok {
		return false
	}
	elt, ok := ct.Value.(*ast.Ident)
	return ok && elt.Name == typ
}

// goLitSendsOn reports whether a go statement runs a func
// literal whose body sends on the named channel.
func goLitSendsOn(gs *ast.GoStmt, chName string) bool {
	if gs.Call == nil {
		return false
	}
	fl, ok := gs.Call.Fun.(*ast.FuncLit)
	if !ok || fl.Body == nil {
		return false
	}
	found := false
	ast.Inspect(fl.Body, func(n ast.Node) bool {
		ss, ok := n.(*ast.SendStmt)
		if !ok {
			return true
		}
		if id, ok := ss.Chan.(*ast.Ident); ok && id.Name == chName {
			found = true
			return false
		}
		return true
	})
	return found
}

// isReceiveAssign reports whether a statement is
// "<resName> := <-<chName>".
func isReceiveAssign(stmt ast.Stmt, chName, resName string) bool {
	as, ok := stmt.(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 1 || len(as.Rhs) != 1 {
		return false
	}
	id, ok := as.Lhs[0].(*ast.Ident)
	if !ok || id.Name != resName {
		return false
	}
	un, ok := as.Rhs[0].(*ast.UnaryExpr)
	if !ok || un.Op != token.ARROW {
		return false
	}
	ch, ok := un.X.(*ast.Ident)
	return ok && ch.Name == chName
}
