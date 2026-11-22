package analyze

// Function edits (editfuncedit.go): the two operations the
// chat's function-edit claim (chat/funcedit.go) serves —
// taking one existing function OUT of its file, and giving
// a function that returns nothing a return type. Both are
// planned the house way: the target is resolved in the
// index (and pinned to the file the chat layer resolved,
// when it settled one), the edit is assembled from the
// file's real bytes by position, and the plan's Verify
// re-reads the applied file for the intended structure.
// Planning only — the chat's spine owns the gate, the
// write, the build, and the rollback.
//
// Remove is deliberately plain: the whole declaration —
// doc comment, signature, body — comes out by lines, with
// the seam tidied so no double blank line survives. The
// index's caller list is counted and named in the plan's
// notes BEFORE the gate, so the user decides with the
// impact in front of them; if they say yes and the build
// then fails on a caller this index missed, the spine
// restores the file and says so. Removing main is refused
// outright: a program with no main has no entry point.
//
// Change-return is deliberately narrow — it extends the
// signature machinery's discipline (editsig.go: resolve
// exactly, edit textually, refuse what can't be proven)
// to the one signature change that touches no call site.
// A function with NO results gains one, and its body
// gains the type's zero-value return so it builds. A
// function that already returns something is refused
// (changing an existing result changes every caller), and
// a body that already returns anywhere is refused (what
// that return should hand back is a judgment, not an
// assembly).

import (
	"fmt"
	"go/ast"
	"go/token"
	"strings"
)

// resolveFuncEditTarget settles the one function a
// remove/change-return request names: the index resolution
// every signature neighbour uses, narrowed to req.FuncFile
// when the chat layer pinned one. verb completes the
// not-found sentence ("so there's nothing to …").
func (p *Project) resolveFuncEditTarget(req EditRequest, verb string) (*Func, string) {
	matches := p.findProjectFuncs(req.FuncName)
	if req.FuncFile != "" {
		var pinned []*Func
		for _, fn := range matches {
			if fn.File == req.FuncFile {
				pinned = append(pinned, fn)
			}
		}
		matches = pinned
	}
	if len(matches) == 0 {
		return nil, fmt.Sprintf("I couldn't find a function named %q in this project, so there's nothing to %s.", req.FuncName, verb)
	}
	if len(matches) > 1 {
		return nil, p.funcCandidates("name the receiver or package", req.FuncName, matches)
	}
	return matches[0], ""
}

// funcDeclAtLine re-reads fn's file and returns its bytes
// plus the declaration the index line names — the shared
// first step of both planners, in planAddCase's shape.
func (p *Project) funcDeclAtLine(fn *Func) ([]byte, *token.FileSet, *ast.FuncDecl, string) {
	src, fset, fileAST, refusal := p.readProjectFile(fn.File)
	if refusal != "" {
		return nil, nil, nil, refusal
	}
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && lineOfPos(fset, fd.Pos()) == fn.Line {
			return src, fset, fd, ""
		}
	}
	return nil, nil, nil, fmt.Sprintf("I found %s in my index but couldn't re-read its declaration in %s.", fn.Display(), p.LinkPath(fn.File))
}

// --- remove a function -------------------------------------------------------

func (p *Project) planRemoveFunc(req EditRequest) (*EditPlan, string) {
	fn, refusal := p.resolveFuncEditTarget(req, "remove")
	if refusal != "" {
		return nil, refusal
	}
	if fn.IsMain {
		return nil, "main is the program's entry point — a program with no main has no way to start, so I won't take it out. If main should do something different, that's a change to what it does, not a removal."
	}
	if why, blocked := p.fileBlocked(fn.File); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(fn.File), why)
	}
	src, fset, decl, refusal := p.funcDeclAtLine(fn)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(fn.File))
	}
	startPos := decl.Pos()
	if decl.Doc != nil {
		startPos = decl.Doc.Pos()
	}
	startLine := lineOfPos(fset, startPos)
	endLine := lineOfPos(fset, decl.End()-1)
	after := removeSourceLines(src, startLine, endLine)
	ef := finishFile(fn.File, src, after)
	rel, name, recvBase := fn.File, fn.Name, recvBaseName(fn.Receiver)
	return &EditPlan{
		Summary:    fmt.Sprintf("Remove %s from %s — the whole declaration goes: doc comment, signature and body.", fn.Display(), p.LinkPath(fn.File)),
		VerifyDesc: fmt.Sprintf("%s is removed from %s", fn.Name, p.LinkPath(fn.File)),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(fn.File), startLine),
		Files:      []EditFile{ef},
		Notes:      []string{p.removeFuncCallersNote(fn)},
		verify: func(after *Project) bool {
			return verifyFuncAbsent(after.Root, rel, name, recvBase)
		},
	}, ""
}

// removeFuncCallersNote is the plan's impact line, from the
// index's caller list (BuildGraph's heuristic edges): who
// calls this function, named before the gate so the user
// decides with the breakage in view. The function's own
// recursive calls are not callers for this purpose — they
// leave with the declaration.
func (p *Project) removeFuncCallersNote(fn *Func) string {
	var callers []*Func
	for _, id := range fn.Callers {
		cf := p.FuncByID(id)
		if cf == nil || cf.ID == fn.ID {
			continue
		}
		callers = append(callers, cf)
	}
	if len(callers) == 0 {
		return fmt.Sprintf("%s isn't called from anywhere else in this project — nothing else uses it.", fn.Name)
	}
	var places []string
	for i, cf := range callers {
		if i == 4 {
			places = append(places, fmt.Sprintf("and %d more", len(callers)-4))
			break
		}
		places = append(places, fmt.Sprintf("%s (%s:%d)", cf.Name, p.LinkPath(cf.File), cf.Line))
	}
	noun := "place"
	if len(callers) != 1 {
		noun = "places"
	}
	return fmt.Sprintf("Heads up: %s is called from %d %s — %s. Removing it breaks those calls. If you say yes and the build then fails, I'll put your files back exactly as they were and tell you.", fn.Name, len(callers), noun, strings.Join(places, ", "))
}

// removeSourceLines deletes the 1-based inclusive line
// range [startLine, endLine] from src and tidies the seam:
// when the lines on both sides of the cut are blank, one
// of them goes too, and the result ends with exactly one
// newline. gofmt (finishFile) settles the rest.
func removeSourceLines(src []byte, startLine, endLine int) []byte {
	lines := strings.Split(string(src), "\n")
	if startLine < 1 {
		startLine = 1
	}
	if endLine > len(lines) {
		endLine = len(lines)
	}
	if startLine > endLine || startLine > len(lines) {
		return src
	}
	cut := make([]string, 0, len(lines)-(endLine-startLine+1))
	cut = append(cut, lines[:startLine-1]...)
	cut = append(cut, lines[endLine:]...)
	if seam := startLine - 1; seam > 0 && seam < len(cut) &&
		strings.TrimSpace(cut[seam-1]) == "" && strings.TrimSpace(cut[seam]) == "" {
		cut = append(cut[:seam], cut[seam+1:]...)
	}
	out := strings.TrimRight(strings.Join(cut, "\n"), "\n")
	if out != "" {
		out += "\n"
	}
	return []byte(out)
}

// --- give a function a return type -------------------------------------------

func (p *Project) planChangeReturn(req EditRequest) (*EditPlan, string) {
	fn, refusal := p.resolveFuncEditTarget(req, "change")
	if refusal != "" {
		return nil, refusal
	}
	if fn.IsMain {
		return nil, "main's signature is fixed by the runtime — it can't return a value, so there's nothing to change."
	}
	if fn.IsInit {
		return nil, "init's signature is fixed by the runtime — it can't return a value, so there's nothing to change."
	}
	if why, blocked := p.fileBlocked(fn.File); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(fn.File), why)
	}
	src, fset, decl, refusal := p.funcDeclAtLine(fn)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(fn.File))
	}
	typ := strings.TrimSpace(req.ReturnType)
	if typ == "" {
		return nil, "That sentence names no return type — say what it should return, like \"have the function jake return a string\"."
	}
	if decl.Type.Results != nil && len(decl.Type.Results.List) > 0 {
		cur := strings.Join(strings.Fields(string(src[offOf(fset, decl.Type.Results.Pos()):offOf(fset, decl.Type.Results.End())])), " ")
		return nil, fmt.Sprintf("%s already returns %s — this edit only gives a return type to a function that has none. Changing a return type it already has would change what every caller receives, and I can't prove that safe.", fn.Display(), cur)
	}
	if decl.Body == nil {
		return nil, fmt.Sprintf("%s is declared without a body in %s, so there's nowhere for a return to go — I can't give it a return type honestly.", fn.Display(), p.LinkPath(fn.File))
	}
	if line, found := firstReturnLine(fset, decl); found {
		return nil, fmt.Sprintf("%s already has a return statement in its body (line %d) — giving it a result would mean deciding what that return hands back, and I can't prove that safe. This edit only covers bodies with no return yet.", fn.Display(), line)
	}
	zero, ok := zeroValueForType(typ)
	if !ok {
		return nil, fmt.Sprintf("I don't know a safe zero value for %s, so I can't write the return the new signature would need.", typ)
	}
	// Two textual splices, the signature machinery's way:
	// the type lands right after the parameter list's
	// closing paren, and the zero return lands just before
	// the body's closing brace — for an empty or TODO-only
	// body it becomes the only statement, for a body with
	// statements it is appended as the final one.
	sigInsertAt := offOf(fset, decl.Type.Params.Closing) + 1
	rbOff := offOf(fset, decl.Body.Rbrace)
	retText := "\treturn " + zero + "\n"
	if rbOff == 0 || src[rbOff-1] != '\n' {
		retText = "\n" + retText
	}
	after := applyTextEdits(src, []textEdit{
		{start: sigInsertAt, end: sigInsertAt, text: " " + typ},
		{start: rbOff, end: rbOff, text: retText},
	})
	ef := finishFile(fn.File, src, after)
	headStart := offOf(fset, decl.Pos())
	head := strings.TrimRight(string(src[headStart:offOf(fset, decl.Body.Pos())]), " \t\r\n")
	newHead := head
	if relIns := sigInsertAt - headStart; relIns >= 0 && relIns <= len(head) {
		newHead = head[:relIns] + " " + typ + head[relIns:]
	}
	bodyWords := "its body gains `return " + zero + "` as its last statement"
	if len(decl.Body.List) == 0 {
		bodyWords = "its empty body gains `return " + zero + "`"
	}
	rel, name, recvBase := fn.File, fn.Name, recvBaseName(fn.Receiver)
	return &EditPlan{
		Summary: fmt.Sprintf("Change %s in %s to return %s: %s becomes %s, and %s so it builds.",
			fn.Name, p.LinkPath(fn.File), typ,
			strings.Join(strings.Fields(head), " "), strings.Join(strings.Fields(newHead), " "), bodyWords),
		VerifyDesc: fmt.Sprintf("%s in %s now returns %s", fn.Name, p.LinkPath(fn.File), typ),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(fn.File), lineOfPos(fset, decl.Pos())),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			return verifyChangeReturn(after.Root, rel, name, recvBase, typ)
		},
	}, ""
}

// firstReturnLine finds the first return statement anywhere
// in a function body — nested closures included, on
// purpose: the change-return rule is "any return at all",
// the conservative reading.
func firstReturnLine(fset *token.FileSet, decl *ast.FuncDecl) (int, bool) {
	line := 0
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		if line != 0 {
			return false
		}
		if rs, ok := n.(*ast.ReturnStmt); ok {
			line = lineOfPos(fset, rs.Pos())
			return false
		}
		return true
	})
	return line, line != 0
}

// zeroValueForType names the zero value a body can return
// for the type as typed: "" for string, false for bool, 0
// for every numeric kind, nil for error and for the
// pointer/slice/map/chan/func/interface shapes, and T{}
// for a named type (a dotted name only when its final
// segment is exported — time.Duration; anything else
// couldn't be built here anyway). The second return is
// false when no safe zero can be named; the build after
// applying remains the backstop for the shapes a name
// alone can't settle (a named type whose kind isn't a
// struct, say).
func zeroValueForType(typ string) (string, bool) {
	t := strings.TrimSpace(typ)
	switch {
	case t == "string":
		return `""`, true
	case t == "bool":
		return "false", true
	case zeroNumericTypes[t]:
		return "0", true
	case t == "error" || t == "any" || t == "interface{}" || t == "interface {}":
		return "nil", true
	case strings.HasPrefix(t, "*") || strings.HasPrefix(t, "[]") ||
		strings.HasPrefix(t, "map[") || strings.HasPrefix(t, "chan ") ||
		strings.HasPrefix(t, "<-chan ") || strings.HasPrefix(t, "func(") ||
		strings.HasPrefix(t, "func ") || strings.HasPrefix(t, "interface{") ||
		strings.HasPrefix(t, "interface {"):
		return "nil", true
	}
	last := t
	if i := strings.LastIndex(t, "."); i >= 0 {
		last = t[i+1:]
	}
	if last != "" && last[0] >= 'A' && last[0] <= 'Z' && isNamedTypeText(t) {
		return t + "{}", true
	}
	return "", false
}

// zeroNumericTypes are the predeclared kinds whose zero
// value is written 0.
var zeroNumericTypes = map[string]bool{
	"int": true, "int8": true, "int16": true, "int32": true, "int64": true,
	"uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
	"uintptr": true, "float32": true, "float64": true,
	"complex64": true, "complex128": true,
	"byte": true, "rune": true,
}

// isNamedTypeText reports whether t is shaped like a named
// type reference: identifier characters and dots only,
// starting with a letter or underscore.
func isNamedTypeText(t string) bool {
	if t == "" {
		return false
	}
	for i, r := range t {
		switch {
		case r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r == '_' || r == '.':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// --- verify helpers ------------------------------------------------------------

// findDeclInFile re-reads one file of a re-analyzed project
// and returns the declaration with the given name and
// receiver base ("" = a plain function), or nil.
func findDeclInFile(root, rel, name, recvBase string) (*ast.FuncDecl, *token.FileSet) {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return nil, nil
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != name {
			continue
		}
		base := ""
		if fd.Recv != nil && len(fd.Recv.List) > 0 {
			base = recvBaseName(exprText(fset, fd.Recv.List[0].Type))
		}
		if base == recvBase {
			return fd, fset
		}
	}
	return nil, nil
}

// verifyFuncAbsent proves a removal landed: the named
// function is no longer declared in its file.
func verifyFuncAbsent(root, rel, name, recvBase string) bool {
	fd, _ := findDeclInFile(root, rel, name, recvBase)
	return fd == nil
}

// verifyChangeReturn proves a change-return landed: the
// named function now declares exactly the one result type
// asked for, and its body returns.
func verifyChangeReturn(root, rel, name, recvBase, typ string) bool {
	fd, fset := findDeclInFile(root, rel, name, recvBase)
	if fd == nil || fd.Type.Results == nil || len(fd.Type.Results.List) == 0 {
		return false
	}
	var got []string
	for _, field := range fd.Type.Results.List {
		n := len(field.Names)
		if n == 0 {
			n = 1
		}
		for i := 0; i < n; i++ {
			got = append(got, stripWS(exprText(fset, field.Type)))
		}
	}
	if len(got) != 1 || got[0] != stripWS(typ) {
		return false
	}
	_, hasReturn := firstReturnLine(fset, fd)
	return hasReturn
}
