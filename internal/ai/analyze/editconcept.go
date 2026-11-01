package analyze

// The concept book: names of code mapped to their basic code, as
// DATA. The composer used to grow one hand-written grammar per
// code kind (a case, a field, a webserver, a delegate…), and any
// sentence phrased between the grammars fell through the seams —
// the user's word for the fix (2026-10-25): "cant there be
// training for names of code to basic code so that it can know
// generally what I want". So the kinds live in a book instead:
// data/concepts.json at the repo root, one row per concept, and
// ONE planner (planConcept) that assembles any row by its KIND —
// statement or declaration — binding the slots the sentence
// supplied ({text}, {name}, {methods}) from the sentence and
// from the analyzer index. A new concept is a new row, not new
// grammar code; the data-driven proof is a test that registers a
// third concept ("for loop") purely as a JSON row and drives it
// end to end.
//
// This is recall-style data consumed deterministically, like the
// makefile brain's pairs — never a model. A concept the book
// doesn't know is not this planner's: the chat's write-sentence
// boundary answers for it honestly.
//
// A book row (JSON, in data/concepts.json under "concepts"):
//
//	{
//	  "id":       "print-line",               // unique, stable
//	  "names":    ["print line", "println"],  // phrasings; the
//	                                          // first is canonical
//	  "kind":     "statement",                // or "declaration"
//	  "template": "fmt.Println(\"{text}\")",  // the basic code,
//	                                          // with slots
//	  "imports":  ["fmt"],                    // optional
//	  "note":     "…"                         // optional: a plan
//	                                          // note the row's
//	                                          // plans always
//	                                          // carry (runtime
//	                                          // honesty the code
//	                                          // alone can't say)
//	}
//
// Slots: {text} is the saying slot — it is substituted with the
// Go-escaped CONTENT of a string literal (the template carries
// the quotes, as the print line's does); it defaults to empty.
// {name} is a declaration's name (required for declarations).
// {methods} is a declaration's method block, filled from a real
// type's method set when the sentence says "like <Type>", else
// a TODO comment — methods are never invented.
// {request} is a statement's handler slot: the planner binds
// it from the target function's REAL *http.Request parameter
// (whatever the user named it), so a template using it can
// only land inside a handler — any other target is a refusal
// naming the handlers the file does have. A request value is
// never invented either.
// {response} is the same slot's other half (the security
// sweep, 2026-11-01): it binds from the target function's
// REAL http.ResponseWriter parameter, so a template that
// must write the answer — set a cookie, refuse a request —
// can only land inside a handler too. The writer is bound
// from the signature, never invented, exactly as the
// request is.

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Concept kinds: the two assembly classes a book row can belong
// to. A statement goes inside a function body; a declaration is
// appended at file scope.
const (
	ConceptStatement   = "statement"
	ConceptDeclaration = "declaration"
)

// ConceptEntry is one row of the concept book.
type ConceptEntry struct {
	ID       string   `json:"id"`
	Names    []string `json:"names"`
	Kind     string   `json:"kind"`
	Template string   `json:"template"`
	Imports  []string `json:"imports,omitempty"`
	Note     string   `json:"note,omitempty"`
}

// Canonical is the concept's first (display) name.
func (e *ConceptEntry) Canonical() string {
	if e == nil || len(e.Names) == 0 {
		return ""
	}
	return e.Names[0]
}

// conceptBookFile is the on-disk shape of data/concepts.json.
// Rows are kept raw so one malformed row can be skipped without
// losing the rest of the book.
type conceptBookFile struct {
	Source   string            `json:"source"`
	Concepts []json.RawMessage `json:"concepts"`
}

// ParseConceptBook reads a concept book. A row that doesn't
// parse, lacks an id/name/template, names an unknown kind, or
// repeats an earlier row's id is SKIPPED (and counted) — a bad
// row never breaks the book, and a wholly unreadable file simply
// yields no entries.
func ParseConceptBook(blob []byte) (entries []ConceptEntry, skipped int) {
	var file conceptBookFile
	if err := json.Unmarshal(blob, &file); err != nil {
		return nil, 0
	}
	seen := map[string]bool{}
	for _, raw := range file.Concepts {
		var e ConceptEntry
		if err := json.Unmarshal(raw, &e); err != nil {
			skipped++
			continue
		}
		e.ID = strings.TrimSpace(e.ID)
		e.Template = strings.TrimSpace(e.Template)
		var names []string
		for _, n := range e.Names {
			if n = strings.Join(strings.Fields(strings.ToLower(n)), " "); n != "" {
				names = append(names, n)
			}
		}
		e.Names = names
		if e.ID == "" || len(e.Names) == 0 || e.Template == "" ||
			(e.Kind != ConceptStatement && e.Kind != ConceptDeclaration) || seen[e.ID] {
			skipped++
			continue
		}
		seen[e.ID] = true
		entries = append(entries, e)
	}
	return entries, skipped
}

// planConcept assembles one book concept into req.File.
// Everything the sentence couldn't supply honestly is a refusal
// in the family's voice; everything assembled is self-checked by
// re-parsing before the plan exists, and the spine's build
// verdict is the backstop after the gate.
func (p *Project) planConcept(req EditRequest) (*EditPlan, string) {
	entry := req.Concept
	if entry == nil {
		return nil, "that edit needs a concept from the book, and none was named."
	}
	rel := strings.TrimSpace(req.File)
	if rel == "" {
		return nil, fmt.Sprintf("that edit needs a file to add the %s to.", entry.Canonical())
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
	switch entry.Kind {
	case ConceptStatement:
		return p.planConceptStatement(req, entry, rel, src, fset, fileAST)
	case ConceptDeclaration:
		return p.planConceptDeclaration(req, entry, rel, src, fset, fileAST)
	}
	return nil, fmt.Sprintf("the book's %q row names a kind I can't assemble (%q).", entry.ID, entry.Kind)
}

// conceptImportEdits returns the text edits adding every import
// spec the entry (or a copied method set) needs that the file
// lacks, via the family's one import inserter.
func conceptImportEdits(fileAST *ast.File, fset *token.FileSet, src []byte, specs []string) []textEdit {
	if len(specs) == 0 {
		return nil
	}
	sort.Strings(specs)
	return insertImportSpecs(fileAST, fset, src, specs)
}

// missingImportSpecs filters candidate specs (full import specs
// like `"fmt"` or `http "net/http"`) to those whose path the
// file doesn't already import.
func missingImportSpecs(fileAST *ast.File, specs []string) []string {
	var out []string
	seen := map[string]bool{}
	for _, spec := range specs {
		path := spec
		if i := strings.LastIndex(spec, " "); i >= 0 {
			path = spec[i+1:]
		}
		path, err := strconv.Unquote(strings.TrimSpace(path))
		if err != nil || seen[path] || hasImportPath(fileAST, path) {
			continue
		}
		seen[path] = true
		out = append(out, spec)
	}
	return out
}

// --- line placement ----------------------------------------------------------
//
// A sentence may name the line its piece should land on or by
// — "add a print line in main to jim/main.go on line 12", "add
// a for loop before line 8 in main to jim/main.go" (the chat
// grammar lives in the concept composer; EditRequest.PlaceLine
// / PlaceBefore carry it here). The number counts the target
// file's lines as the user sees them BEFORE the edit, 1-based,
// and insertion happens only at whole-statement boundaries
// inside the named function (for statements) or between
// top-level declarations (for declarations): a statement is
// never split, and a line inside a multi-line statement
// anchors to that statement's end. The plan quotes the anchor
// — the number AND the line's current text — so the user sees
// where the piece lands before the gate, and landing is
// verified structurally: the new statement/declaration must
// sit at the resolved index in the re-parsed body/file, on top
// of the usual presence checks and the spine's build verdict.

// srcLines splits src into its lines as the user sees them:
// line n (1-based) is lines[n-1], newline stripped. A trailing
// newline ends the last line rather than starting a new one.
func srcLines(src []byte) []string {
	if len(src) == 0 {
		return nil
	}
	lines := strings.Split(string(src), "\n")
	if lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// declLineSpan returns the 1-based first and last lines a
// top-level declaration occupies, as the user sees them: a
// leading doc comment travels with its declaration (the AST
// attaches it there), and the last line is the one holding
// the declaration's final token.
func declLineSpan(fset *token.FileSet, d ast.Decl) (int, int) {
	start := lineOfPos(fset, d.Pos())
	switch d := d.(type) {
	case *ast.FuncDecl:
		if d.Doc != nil {
			start = lineOfPos(fset, d.Doc.Pos())
		}
	case *ast.GenDecl:
		if d.Doc != nil {
			start = lineOfPos(fset, d.Doc.Pos())
		}
	}
	return start, lineOfPos(fset, d.End()-1)
}

// declDescribe names a top-level declaration in beginner
// words for the placement refusals: "the struct Profile",
// "the function helper", "a const block", "the import block".
func declDescribe(d ast.Decl) string {
	switch d := d.(type) {
	case *ast.FuncDecl:
		if d.Recv != nil && len(d.Recv.List) > 0 {
			return fmt.Sprintf("the method %s", d.Name.Name)
		}
		return fmt.Sprintf("the function %s", d.Name.Name)
	case *ast.GenDecl:
		switch d.Tok {
		case token.IMPORT:
			return "the import block"
		case token.CONST:
			return "a const block"
		case token.VAR:
			return "a var block"
		case token.TYPE:
			for _, spec := range d.Specs {
				if ts, ok := spec.(*ast.TypeSpec); ok {
					switch ts.Type.(type) {
					case *ast.StructType:
						return fmt.Sprintf("the struct %s", ts.Name.Name)
					case *ast.InterfaceType:
						return fmt.Sprintf("the interface %s", ts.Name.Name)
					}
					return fmt.Sprintf("the type %s", ts.Name.Name)
				}
			}
			return "a type declaration"
		}
	}
	return "a declaration"
}

// placeQuote renders the anchor a plan quotes for a line
// placement: the line number and the actual current text of
// that line, so the user sees where the piece lands before
// the gate. A blank line has no text to quote; it is named
// as one instead.
func placeQuote(lines []string, n int, before bool) string {
	word := "after"
	if before {
		word = "before"
	}
	text := ""
	if n >= 1 && n <= len(lines) {
		text = strings.TrimSpace(lines[n-1])
	}
	if text == "" {
		return fmt.Sprintf("%s line %d (a blank line)", word, n)
	}
	return fmt.Sprintf("%s line %d: `%s`", word, n, text)
}

// linePastEnd is the shared refusal for a line number beyond
// the file's last line.
func (p *Project) linePastEnd(rel string, n, count int) string {
	return fmt.Sprintf("line %d is past the end of %s — the file has %d lines, so there's no line %d to aim at.", n, p.LinkPath(rel), count, n)
}

func capFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// articleOf picks a/an for a concept's canonical name — the
// planner-side copy of the composer's rule.
func articleOf(name string) string {
	if name != "" && strings.ContainsRune("aeiou", rune(name[0])) {
		return "an"
	}
	return "a"
}

// stmtPlacement is a resolved line placement inside one
// function body: idx is the index in the body's CURRENT
// statement list before which the new statement is inserted
// (len(list) = the end of the body). atLbrace / atRbrace mark
// the bodies whose edge statement shares a line with the
// brace itself (a one-line body, statements crowded onto the
// opening line, a closing brace on the last statement's
// line): there is no line boundary to insert at there, so
// the insertion rides the brace's own offset. quote is the
// anchor text the plan shows.
type stmtPlacement struct {
	idx      int
	atLbrace bool
	atRbrace bool
	quote    string
}

// resolveStmtPlacement validates req's line phrase against
// the target file and resolves it to an insertion point in
// target's body. piece names what is being placed, with its
// article ("a print line", "an instantiation of Jim"). Every
// refusal is in placement terms.
func (p *Project) resolveStmtPlacement(rel string, src []byte, fset *token.FileSet, fileAST *ast.File, target *ast.FuncDecl, piece string, req EditRequest) (stmtPlacement, string) {
	n := req.PlaceLine
	lines := srcLines(src)
	if n > len(lines) {
		return stmtPlacement{}, p.linePastEnd(rel, n, len(lines))
	}
	tStart, tEnd := declLineSpan(fset, target)
	contained := false
	for _, d := range fileAST.Decls {
		start, end := declLineSpan(fset, d)
		if n < start || n > end {
			continue
		}
		contained = true
		if fd, ok := d.(*ast.FuncDecl); ok {
			if fd != target {
				return stmtPlacement{}, fmt.Sprintf("line %d is inside %s, which runs lines %d–%d — not %s, which runs lines %d–%d. %s lands in one function: name %s instead, or aim at a line inside %s.",
					n, fd.Name.Name, start, end, target.Name.Name, tStart, tEnd, capFirst(piece), fd.Name.Name, target.Name.Name)
			}
		} else {
			return stmtPlacement{}, fmt.Sprintf("line %d is inside %s (lines %d–%d) — that's not inside a function, and %s can only land inside a function body. %s runs lines %d–%d; aim at a line in there.",
				n, declDescribe(d), start, end, piece, target.Name.Name, tStart, tEnd)
		}
	}
	if !contained {
		if n == lineOfPos(fset, fileAST.Name.Pos()) {
			return stmtPlacement{}, fmt.Sprintf("line %d is the package clause — %s lands inside a function, and %s runs lines %d–%d. Aim at a line in there.",
				n, piece, target.Name.Name, tStart, tEnd)
		}
		return stmtPlacement{}, fmt.Sprintf("line %d isn't inside any function — it sits between the file's declarations. %s runs lines %d–%d; aim at a line in there.",
			n, target.Name.Name, tStart, tEnd)
	}

	quote := placeQuote(lines, n, req.PlaceBefore)
	body := target.Body
	list := body.List
	lbraceLine := lineOfPos(fset, body.Lbrace)
	rbraceLine := lineOfPos(fset, body.Rbrace)
	if len(list) == 0 {
		return stmtPlacement{idx: 0, atRbrace: lbraceLine == rbraceLine, quote: quote}, ""
	}
	starts := make([]int, len(list))
	ends := make([]int, len(list))
	for i, s := range list {
		starts[i] = lineOfPos(fset, s.Pos())
		ends[i] = lineOfPos(fset, s.End()-1)
	}
	var idx int
	if req.PlaceBefore {
		// Before the statement starting at line N; a line
		// inside a statement anchors before that statement,
		// and a line on no statement (blank/comment) anchors
		// before the following one.
		idx = len(list)
		for i := range list {
			if ends[i] >= n {
				idx = i
				break
			}
		}
	} else {
		// After the complete statement line N belongs to; a
		// line on no statement anchors after the preceding
		// one.
		idx = len(list)
		for i := range list {
			if starts[i] > n {
				idx = i
				break
			}
			idx = i + 1
		}
	}
	pl := stmtPlacement{idx: idx, quote: quote}
	switch {
	case idx < len(list) && starts[idx] <= lbraceLine:
		pl.atLbrace = true
	case idx == len(list) && rbraceLine <= ends[len(list)-1]:
		pl.atRbrace = true
	}
	return pl, ""
}

// declPlacement is a resolved line placement at file scope:
// the byte offset at which the new declaration's text is
// inserted (len(src) when it lands at the end of the file)
// and the index in the file's CURRENT declaration list the
// new declaration will occupy.
type declPlacement struct {
	offset int
	idx    int
	quote  string
}

// resolveDeclPlacement validates req's line phrase at file
// scope and resolves it to a landing between declarations:
// a line where a declaration starts anchors before OR after
// the whole declaration; a line where one ends anchors after
// it; a line strictly inside a declaration's span refuses —
// a declaration is never split. A line on no declaration
// (blank/comment, between declarations) anchors to the
// neighbouring declarations, the statement rule at file
// scope. piece names what is being placed, with its article
// ("a struct").
func (p *Project) resolveDeclPlacement(rel string, src []byte, fset *token.FileSet, fileAST *ast.File, piece string, req EditRequest) (declPlacement, string) {
	n := req.PlaceLine
	lines := srcLines(src)
	if n > len(lines) {
		return declPlacement{}, p.linePastEnd(rel, n, len(lines))
	}
	quote := placeQuote(lines, n, req.PlaceBefore)
	decls := fileAST.Decls
	afterDecl := func(i int) declPlacement {
		_, end := declLineSpan(fset, decls[i])
		off := len(src)
		if end < len(lines) {
			off = lineStartOffset(src, end+1)
		}
		return declPlacement{offset: off, idx: i + 1, quote: quote}
	}
	beforeDecl := func(i int) declPlacement {
		start, _ := declLineSpan(fset, decls[i])
		return declPlacement{offset: lineStartOffset(src, start), idx: i, quote: quote}
	}
	insideRefusal := func(i int) string {
		start, end := declLineSpan(fset, decls[i])
		return fmt.Sprintf("line %d is inside %s (lines %d–%d) — %s lands between declarations, never inside one. Aim at line %d, where it starts, or line %d, where it ends.",
			n, declDescribe(decls[i]), start, end, piece, start, end)
	}
	for i, d := range decls {
		start, end := declLineSpan(fset, d)
		switch {
		case n == start:
			if req.PlaceBefore {
				return beforeDecl(i), ""
			}
			return afterDecl(i), ""
		case n == end && end > start:
			if !req.PlaceBefore {
				return afterDecl(i), ""
			}
			return declPlacement{}, insideRefusal(i)
		case n > start && n < end:
			return declPlacement{}, insideRefusal(i)
		}
	}
	if len(decls) == 0 {
		return declPlacement{}, fmt.Sprintf("line %d can't anchor %s yet — %s has no declarations at all to place it against. Add it without a line number first.", n, piece, p.LinkPath(rel))
	}
	if n == lineOfPos(fset, fileAST.Name.Pos()) {
		first, _ := declLineSpan(fset, decls[0])
		return declPlacement{}, fmt.Sprintf("line %d is the package clause — %s lands between declarations, and the first declaration starts at line %d. Aim at the line where a declaration starts or ends.", n, piece, first)
	}
	// No declaration occupies the line: the preceding /
	// following neighbours anchor it, exactly as a
	// blank/comment line inside a function body does.
	prev, next := -1, -1
	for i, d := range decls {
		start, end := declLineSpan(fset, d)
		if end < n {
			prev = i
		}
		if start > n && next < 0 {
			next = i
		}
	}
	if req.PlaceBefore {
		if next >= 0 {
			return beforeDecl(next), ""
		}
		return afterDecl(prev), ""
	}
	if prev >= 0 {
		return afterDecl(prev), ""
	}
	return beforeDecl(next), ""
}

// stmtAtIndex reports whether the statement at index idx in
// fd's body is the one whose first line matches firstLine —
// the structural position check a line placement adds on top
// of the presence checks.
func stmtAtIndex(fset *token.FileSet, fd *ast.FuncDecl, idx int, firstLine string) bool {
	if fd.Body == nil || idx < 0 || idx >= len(fd.Body.List) {
		return false
	}
	return strings.HasPrefix(conceptStmtText(fset, fd.Body.List[idx]), stmtLineKey(firstLine))
}

// --- statement concepts ------------------------------------------------------

// requestParamName returns the name of fd's *http.Request
// parameter — the value a handler-shaped statement template's
// {request} slot binds. The qualifier is the file's local
// name for net/http (its import alias, else "http"). An
// unnamed request parameter can't be referenced from the
// body, so it doesn't count.
func requestParamName(fset *token.FileSet, fileAST *ast.File, fd *ast.FuncDecl) string {
	if fd.Type == nil || fd.Type.Params == nil {
		return ""
	}
	qual := "http"
	for _, imp := range fileAST.Imports {
		if p, err := strconv.Unquote(imp.Path.Value); err == nil && p == "net/http" {
			if imp.Name != nil && imp.Name.Name != "_" && imp.Name.Name != "." {
				qual = imp.Name.Name
			}
			break
		}
	}
	for _, field := range fd.Type.Params.List {
		text := strings.Join(strings.Fields(exprText(fset, field.Type)), "")
		if text != qual+".Request" && text != "*"+qual+".Request" {
			continue
		}
		if len(field.Names) > 0 {
			return field.Names[0].Name
		}
	}
	return ""
}

// responseParamName returns the name of fd's
// http.ResponseWriter parameter — the value a handler-shaped
// statement template's {response} slot binds. The qualifier
// rule matches requestParamName's; the writer type is an
// interface, so it never carries a star. A nameless writer
// can't be referenced from the body, so it doesn't count.
func responseParamName(fset *token.FileSet, fileAST *ast.File, fd *ast.FuncDecl) string {
	if fd.Type == nil || fd.Type.Params == nil {
		return ""
	}
	qual := "http"
	for _, imp := range fileAST.Imports {
		if p, err := strconv.Unquote(imp.Path.Value); err == nil && p == "net/http" {
			if imp.Name != nil && imp.Name.Name != "_" && imp.Name.Name != "." {
				qual = imp.Name.Name
			}
			break
		}
	}
	for _, field := range fd.Type.Params.List {
		text := strings.Join(strings.Fields(exprText(fset, field.Type)), "")
		if text != qual+".ResponseWriter" {
			continue
		}
		if len(field.Names) > 0 {
			return field.Names[0].Name
		}
	}
	return ""
}

// handlerFuncNames lists the file's functions that receive an
// http.Request — the honest shortlist a {request} refusal can
// point at.
func handlerFuncNames(fset *token.FileSet, fileAST *ast.File) []string {
	var out []string
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && requestParamName(fset, fileAST, fd) != "" {
			out = append(out, fd.Name.Name)
		}
	}
	return out
}

// planConceptStatement puts a rendered statement at the end of a
// function body: the function the sentence named with "in
// <Func>", else main. A program file with no main yet grows one
// around the statement (the add-server precedent); a named
// function that doesn't exist is a refusal listing the functions
// the file does have.
func (p *Project) planConceptStatement(req EditRequest, entry *ConceptEntry, rel string, src []byte, fset *token.FileSet, fileAST *ast.File) (*EditPlan, string) {
	stmt := renderConceptTemplate(entry, req, "")
	if strings.TrimSpace(stmt) == "" {
		return nil, fmt.Sprintf("the book's %q template renders to nothing — the row needs fixing before I can assemble it.", entry.ID)
	}
	firstLine := strings.TrimSpace(strings.SplitN(stmt, "\n", 2)[0])

	// The entry's own imports, before anything else is decided:
	// they ride whichever insertion follows.
	var specs []string
	for _, path := range entry.Imports {
		specs = append(specs, strconv.Quote(path))
	}
	specs = missingImportSpecs(fileAST, specs)

	wantFunc := strings.TrimSpace(req.ConceptFunc)
	explicit := wantFunc != ""
	if !explicit {
		wantFunc = "main"
	}
	var decl, folded *ast.FuncDecl
	var funcNames []string
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		funcNames = append(funcNames, fd.Name.Name)
		switch {
		case fd.Name.Name == wantFunc:
			decl = fd
		case strings.EqualFold(fd.Name.Name, wantFunc):
			if folded == nil {
				folded = fd
			}
		}
	}
	if decl == nil {
		decl = folded // the family's case-insensitive forgiveness
	}

	var edits []textEdit
	edits = append(edits, conceptImportEdits(fileAST, fset, src, specs)...)
	var note string
	var place *stmtPlacement
	growMain := false
	resolvedFunc := wantFunc
	// A {request} template can only live in a handler: the
	// refusal that says so, pointing at the handlers the file
	// does have (or at the form handler that would start one).
	needsRequest := strings.Contains(entry.Template, "{request}")
	handlerRefusal := func() string {
		handlers := handlerFuncNames(fset, fileAST)
		if len(handlers) == 0 {
			return fmt.Sprintf("the %s reads the values on the request a handler receives, and %s has no handler — a function taking an http.Request — to put it in. Add a form handler first, then say \"add a %s in <Handler> to %s\".",
				entry.Canonical(), p.LinkPath(rel), entry.Canonical(), p.LinkPath(rel))
		}
		return fmt.Sprintf("the %s reads the values on the request a handler receives, so it belongs in a handler — the handlers in %s are: %s. Say \"add a %s in %s to %s\".",
			entry.Canonical(), p.LinkPath(rel), conceptNameList(handlers), entry.Canonical(), handlers[0], p.LinkPath(rel))
	}
	// A {response} template can only live in a handler too:
	// the refusal that says so, in the same shape, pointing
	// at the same honest shortlist.
	needsResponse := strings.Contains(entry.Template, "{response}")
	handlerRefusalResponse := func() string {
		handlers := handlerFuncNames(fset, fileAST)
		if len(handlers) == 0 {
			return fmt.Sprintf("the %s writes to the response a handler sends, and %s has no handler — a function taking an http.ResponseWriter — to put it in. Add a form handler first, then say \"add a %s in <Handler> to %s\".",
				entry.Canonical(), p.LinkPath(rel), entry.Canonical(), p.LinkPath(rel))
		}
		return fmt.Sprintf("the %s writes to the response a handler sends, so it belongs in a handler — the handlers in %s are: %s. Say \"add a %s in %s to %s\".",
			entry.Canonical(), p.LinkPath(rel), conceptNameList(handlers), entry.Canonical(), handlers[0], p.LinkPath(rel))
	}
	if decl == nil {
		switch {
		case explicit:
			return nil, fmt.Sprintf("%s has no function named %q to put the %s in — the functions there are: %s. Name one of those, like \"add a %s in %s to %s\".",
				p.LinkPath(rel), wantFunc, entry.Canonical(), conceptNameList(funcNames), entry.Canonical(), conceptFirstOr(funcNames, "main"), p.LinkPath(rel))
		case fileAST.Name.Name != "main":
			return nil, fmt.Sprintf("%s is package %s, so it has no main to put the %s in — name a function, like \"add a %s in %s to %s\" (the functions there are: %s).",
				p.LinkPath(rel), fileAST.Name.Name, entry.Canonical(), entry.Canonical(), conceptFirstOr(funcNames, "main"), p.LinkPath(rel), conceptNameList(funcNames))
		case needsRequest:
			// No main to grow: a main could never receive the
			// request this piece reads.
			return nil, handlerRefusal()
		case needsResponse:
			// No main to grow: a main has no response to
			// write to either.
			return nil, handlerRefusalResponse()
		case req.PlaceLine > 0:
			// A line placement needs a function that exists:
			// a main grown on the spot would land at the end
			// of the file, and no line the user can see is
			// inside it yet.
			return nil, fmt.Sprintf("%s has no main yet — a new main would be created at the end of the file, so there's no line inside main for line %d to name. Add the %s without a line number first, then place the next one by line.", p.LinkPath(rel), req.PlaceLine, entry.Canonical())
		default:
			// No main yet in a program file: the statement gets
			// a main to live in, appended whole (below, after
			// the imports — an append and a no-imports insert
			// can share the same offset in a bare-clause file,
			// and spliced in the wrong order the main would land
			// before the import block, which Go rejects).
			growMain = true
			note = fmt.Sprintf("%s had no main, so the %s got a new main to live in.", p.LinkPath(rel), entry.Canonical())
		}
	} else {
		if decl.Body == nil {
			return nil, fmt.Sprintf("%s in %s has no body — it's a declaration without code, so there's nowhere to put the %s.", decl.Name.Name, p.LinkPath(rel), entry.Canonical())
		}
		resolvedFunc = decl.Name.Name
		if needsRequest {
			// Bind {request} from the target function's real
			// signature — the parameter's own name, whatever
			// the user called it. No request parameter, no
			// landing: the values this piece reads exist only
			// on a handler's request.
			param := requestParamName(fset, fileAST, decl)
			if param == "" {
				return nil, fmt.Sprintf("%s in %s doesn't receive an http.Request. %s", decl.Name.Name, p.LinkPath(rel), handlerRefusal())
			}
			stmt = strings.ReplaceAll(stmt, "{request}", param)
			firstLine = strings.TrimSpace(strings.SplitN(stmt, "\n", 2)[0])
		}
		if needsResponse {
			// Bind {response} the same way: the writer
			// parameter's own name, from the same signature.
			// No writer parameter, no landing: the answer
			// this piece writes exists only on a handler's
			// response.
			param := responseParamName(fset, fileAST, decl)
			if param == "" {
				return nil, fmt.Sprintf("%s in %s doesn't receive an http.ResponseWriter. %s", decl.Name.Name, p.LinkPath(rel), handlerRefusalResponse())
			}
			stmt = strings.ReplaceAll(stmt, "{response}", param)
			firstLine = strings.TrimSpace(strings.SplitN(stmt, "\n", 2)[0])
		}
		indent := "\t"
		if len(decl.Body.List) > 0 {
			indent = lineIndent(src, lineOfPos(fset, decl.Body.List[0].Pos()))
			if indent == "" {
				indent = "\t"
			}
		}
		if req.PlaceLine > 0 {
			pl, refusal := p.resolveStmtPlacement(rel, src, fset, fileAST, decl,
				articleOf(entry.Canonical())+" "+entry.Canonical(), req)
			if refusal != "" {
				return nil, refusal
			}
			place = &pl
		}
		lbraceLine := lineOfPos(fset, decl.Body.Lbrace)
		rbraceLine := lineOfPos(fset, decl.Body.Rbrace)
		switch {
		case place != nil && place.atLbrace:
			// The anchor statement shares the opening line:
			// the new statement rides the brace's own offset,
			// opening the line up.
			edits = append(edits, textEdit{
				start: offOf(fset, decl.Body.Lbrace) + 1,
				end:   offOf(fset, decl.Body.Lbrace) + 1,
				text:  "\n" + indentConceptLines(stmt, indent) + "\n" + lineIndent(src, lineOfPos(fset, decl.Pos())),
			})
		case place != nil && place.atRbrace:
			// The closing brace shares a line with code (a
			// one-line body, or a crowded last line): the new
			// statement rides the brace's own offset.
			edits = append(edits, textEdit{
				start: offOf(fset, decl.Body.Rbrace),
				end:   offOf(fset, decl.Body.Rbrace),
				text:  "\n" + indentConceptLines(stmt, indent) + "\n" + lineIndent(src, lineOfPos(fset, decl.Pos())),
			})
		case place != nil && place.idx < len(decl.Body.List):
			// Before the anchor statement, at its line's start.
			atLine := lineOfPos(fset, decl.Body.List[place.idx].Pos())
			ind := lineIndent(src, atLine)
			if ind == "" {
				ind = indent
			}
			edits = append(edits, textEdit{
				start: lineStartOffset(src, atLine),
				end:   lineStartOffset(src, atLine),
				text:  indentConceptLines(stmt, ind) + "\n",
			})
		case rbraceLine == lbraceLine:
			// An empty one-line body "{}": open it up.
			edits = append(edits, textEdit{
				start: offOf(fset, decl.Body.Rbrace),
				end:   offOf(fset, decl.Body.Rbrace),
				text:  "\n" + indentConceptLines(stmt, indent) + "\n" + lineIndent(src, lineOfPos(fset, decl.Pos())),
			})
		default:
			edits = append(edits, textEdit{
				start: lineStartOffset(src, rbraceLine),
				end:   lineStartOffset(src, rbraceLine),
				text:  indentConceptLines(stmt, indent) + "\n",
			})
		}
	}

	var after []byte
	if growMain {
		base := applyTextEdits(src, edits)
		text := "\nfunc main() {\n" + indentConceptLines(stmt, "\t") + "\n}\n"
		if len(base) == 0 || base[len(base)-1] != '\n' {
			text = "\n" + text
		}
		after = append(append([]byte{}, base...), text...)
	} else {
		after = applyTextEdits(src, edits)
	}
	ef := finishFile(rel, src, after)

	// AST self-check: the assembled file must parse, and the
	// target function must now contain the rendered statement.
	if bad := selfCheckConceptStatement(ef.After, resolvedFunc, firstLine); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}
	// A placed statement must additionally sit at the
	// resolved index in the body — presence alone would let
	// a mis-anchored insert pass.
	if place != nil {
		if bad := selfCheckConceptStatementAt(ef.After, resolvedFunc, firstLine, place.idx); bad != "" {
			return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
		}
	}

	summary := fmt.Sprintf("Add a %s at the end of %s in %s, from the concept book: %s",
		entry.Canonical(), resolvedFunc, p.LinkPath(rel), firstLine)
	if place != nil {
		summary = fmt.Sprintf("Add a %s in %s in %s, %s, from the concept book: %s",
			entry.Canonical(), resolvedFunc, p.LinkPath(rel), place.quote, firstLine)
	}
	plan := &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s now has the %s in %s", filepath.Base(rel), entry.Canonical(), resolvedFunc),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, firstLine)),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			if !verifyConceptStatement(after.Root, rel, resolvedFunc, firstLine) {
				return false
			}
			if place == nil {
				return true
			}
			return verifyConceptStatementAt(after.Root, rel, resolvedFunc, firstLine, place.idx)
		},
	}
	if note != "" {
		plan.Notes = append(plan.Notes, note)
	}
	if entry.Note != "" {
		plan.Notes = append(plan.Notes, entry.Note)
	}
	return plan, ""
}

// --- declaration concepts ----------------------------------------------------

// planConceptDeclaration appends a rendered declaration at the
// end of the file. The {name} slot is REQUIRED — a declaration
// without a name is guidance, never a placeholder. The {methods}
// slot, when the template has one, binds from a real type's
// method set ("like <Type>") or stays an honest TODO.
func (p *Project) planConceptDeclaration(req EditRequest, entry *ConceptEntry, rel string, src []byte, fset *token.FileSet, fileAST *ast.File) (*EditPlan, string) {
	name := strings.TrimSpace(req.ConceptName)
	canon := entry.Canonical()
	if name == "" {
		article := "a"
		if canon != "" && strings.ContainsRune("aeiou", rune(canon[0])) {
			article = "an"
		}
		return nil, fmt.Sprintf("%s %s needs a name before I can add it — say \"add a %s named <Name> to %s\".", article, canon, canon, rel)
	}
	if !token.IsIdentifier(name) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a %s.", name, canon)
	}
	if pkg := p.pkgOfFile(rel); pkg != nil && p.pkgDeclares(pkg, name) {
		return nil, fmt.Sprintf("%s already exists in package %s (%s) — pick a different name for the %s.", name, pkg.Name, p.LinkPath(rel), canon)
	}

	methods := ""
	var notes []string
	var methodSpecs []string
	if strings.Contains(entry.Template, "{methods}") {
		switch {
		case strings.TrimSpace(req.ConceptLike) != "":
			lines, specs, refusal := p.conceptLikeMethods(req.ConceptLike)
			if refusal != "" {
				return nil, refusal
			}
			methods = strings.Join(lines, "\n")
			methodSpecs = specs
			notes = append(notes, fmt.Sprintf("The methods are %s's real method set, copied from the index — signatures verbatim, bodies left for the types that will satisfy %s.", req.ConceptLike, name))
		default:
			methods = "\t// TODO: add the methods this " + canon + " needs"
			notes = append(notes, fmt.Sprintf("The method block is a TODO — the methods are yours to add, and I never invent them. To copy a real type's methods instead, say \"add a %s named %s like <Type> to %s\".", canon, name, rel))
		}
	}

	decl := renderConceptTemplate(entry, req, methods)
	// Method lines arrive with their own tab indent; the
	// declaration pass normalizes any template line that
	// carries none (a hand-written row may not).
	decl = indentConceptDecl(decl)

	// A func-shaped row whose signature is EMPTY (the book's
	// function and recursion entries) assembles an EMPTY
	// scaffold: no parameters, no results, a TODO body. The
	// plan says so plainly, and names the two shapes that
	// produce a working function with a real signature — the
	// same honesty pattern as the interface's
	// never-invented-methods note, keyed on the rendered
	// declaration's shape, not on the row's id. Func rows WITH
	// a signature (the web family's handlers and middleware)
	// carry real parameters and bodies; the scaffold note would
	// misdescribe them, so it keys on the empty signature.
	if conceptScaffoldRe.MatchString(decl) {
		notes = append(notes, fmt.Sprintf("This is an empty scaffold — %s has a name and a body to fill in, but no parameters, no results, and nothing that does work yet. For a function with a real signature from the start, say \"add a function in %s that calls <Something>\" (it wraps one that exists) or \"add a function like <Existing> named <New>\" (it copies one).", name, rel))
	}

	var specs []string
	for _, path := range entry.Imports {
		specs = append(specs, strconv.Quote(path))
	}
	specs = append(specs, methodSpecs...)
	specs = missingImportSpecs(fileAST, specs)

	// A line placement resolves to a landing between
	// declarations before anything is spliced.
	var place *declPlacement
	if req.PlaceLine > 0 {
		pl, refusal := p.resolveDeclPlacement(rel, src, fset, fileAST, articleOf(canon)+" "+canon, req)
		if refusal != "" {
			return nil, refusal
		}
		place = &pl
	}

	var after []byte
	if place != nil && place.offset < len(src) {
		// Between declarations: the import edits and the
		// declaration's insertion compose over the original
		// bytes in one splice.
		edits := conceptImportEdits(fileAST, fset, src, specs)
		edits = append(edits, textEdit{start: place.offset, end: place.offset, text: decl + "\n\n"})
		after = applyTextEdits(src, edits)
	} else {
		// Imports first, the declaration appended after — the
		// same coincident-offset hazard as the statement
		// planner's grown main (a bare-clause file puts both
		// at EOF). A placement that resolved to the end of
		// the file lands here too: its anchor is the file's
		// last declaration, so appending IS the placement.
		base := applyTextEdits(src, conceptImportEdits(fileAST, fset, src, specs))
		text := "\n" + decl + "\n"
		if len(base) == 0 || base[len(base)-1] != '\n' {
			text = "\n" + text
		}
		after = append(append([]byte{}, base...), text...)
	}
	ef := finishFile(rel, src, after)

	// AST self-check: the file parses and now declares the name.
	if bad := selfCheckConceptDeclaration(ef.After, name); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}
	// A placed declaration must additionally sit at the
	// resolved index in the file's declarations.
	if place != nil {
		if bad := selfCheckConceptDeclarationAt(ef.After, name, place.idx); bad != "" {
			return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
		}
	}

	if entry.Note != "" {
		notes = append(notes, entry.Note)
	}

	summary := fmt.Sprintf("Add the %s %s to %s, from the concept book.", canon, name, p.LinkPath(rel))
	if strings.TrimSpace(req.ConceptLike) != "" && strings.Contains(entry.Template, "{methods}") {
		summary = fmt.Sprintf("Add the %s %s to %s, with %s's method set copied from this project's own code.", canon, name, p.LinkPath(rel), req.ConceptLike)
	}
	if place != nil {
		summary = fmt.Sprintf("Add the %s %s to %s, %s, from the concept book.", canon, name, p.LinkPath(rel), place.quote)
		if strings.TrimSpace(req.ConceptLike) != "" && strings.Contains(entry.Template, "{methods}") {
			summary = fmt.Sprintf("Add the %s %s to %s, %s, with %s's method set copied from this project's own code.", canon, name, p.LinkPath(rel), place.quote, req.ConceptLike)
		}
	}
	plan := &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s now declares the %s %s", filepath.Base(rel), canon, name),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, strings.SplitN(decl, "\n", 2)[0])),
		Files:      []EditFile{ef},
		Notes:      notes,
		verify: func(after *Project) bool {
			// A declaration row's product is a top-level name
			// in the target file — a TYPE for the interface
			// and struct rows, a plain FUNCTION for the
			// function row. Either form verifies the plan.
			for _, pk := range after.Packages {
				for _, t := range pk.Types {
					if t.Name == name && t.File == rel {
						return true
					}
				}
				for _, fn := range pk.Funcs {
					if fn.Receiver == "" && fn.Name == name && fn.File == rel {
						return true
					}
				}
			}
			return false
		},
	}
	return plan, ""
}

// conceptLikeMethods returns the method signatures of the type
// the sentence named with "like <Type>", verbatim from the
// index, plus the import specs those signatures lean on (copied
// from the files the methods live in — what the existing import
// machinery can prove; the build verdict is the backstop).
func (p *Project) conceptLikeMethods(like string) (lines []string, specs []string, refusal string) {
	like = strings.TrimSpace(like)
	var matches []*Type
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if strings.EqualFold(t.Name, like) || strings.EqualFold(t.Pkg+"."+t.Name, like) {
				matches = append(matches, t)
			}
		}
	}
	// Exact-case wins over the index's Q&A forgiveness.
	var exact []*Type
	lastSeg := like
	if i := strings.LastIndex(like, "."); i >= 0 {
		lastSeg = like[i+1:]
	}
	for _, t := range matches {
		if t.Name == lastSeg {
			exact = append(exact, t)
		}
	}
	if len(exact) > 0 {
		matches = exact
	}
	if len(matches) == 0 {
		return nil, nil, fmt.Sprintf("I couldn't find a type named %q in this project, so there's no method set to copy. Name a type that exists, or leave out the \"like\" and fill the methods in yourself.", like)
	}
	if len(matches) > 1 {
		var b strings.Builder
		fmt.Fprintf(&b, "%q is more than one type here:\n", like)
		for _, t := range matches {
			fmt.Fprintf(&b, "  - %s.%s (%s:%d)\n", t.Pkg, t.Name, p.LinkPath(t.File), t.Line)
		}
		b.WriteString("Say which one with its package — written as pkg.Name.")
		return nil, nil, b.String()
	}
	t := matches[0]

	sourceFiles := map[string]bool{t.File: true}
	if t.Kind == "interface" {
		lines = p.interfaceMethodLines(t)
	} else {
		var fns []*Func
		for _, fn := range p.AllFuncs() {
			if fn.IsTest || fn.Receiver == "" {
				continue
			}
			if strings.TrimPrefix(fn.Receiver, "*") == t.Name && fn.PkgDir == t.PkgDir {
				fns = append(fns, fn)
			}
		}
		sort.Slice(fns, func(i, j int) bool {
			if fns[i].File != fns[j].File {
				return fns[i].File < fns[j].File
			}
			return fns[i].Line < fns[j].Line
		})
		for _, fn := range fns {
			if line := methodSigLine(fn.Sig); line != "" {
				lines = append(lines, "\t"+line)
				sourceFiles[fn.File] = true
			}
		}
	}
	if len(lines) == 0 {
		return nil, nil, fmt.Sprintf("%s has no methods — an interface copied from it would be empty, and an empty interface is just interface{}. Give %s some methods first, or leave out the \"like\" and write the methods yourself.", t.Name, t.Name)
	}

	// Import specs for the qualifiers the copied signatures use,
	// copied from the files those signatures live in.
	quals := map[string]bool{}
	for _, ln := range lines {
		for _, m := range conceptQualRe.FindAllStringSubmatch(ln, -1) {
			quals[m[1]] = true
		}
	}
	specByQual := map[string]string{}
	for rel := range sourceFiles {
		_, _, sibAST, r := p.readProjectFile(rel)
		if r != "" || sibAST == nil {
			continue
		}
		for _, is := range sibAST.Imports {
			spec, eff := importSpecOf(is)
			if quals[eff] {
				specByQual[eff] = spec
			}
		}
	}
	for _, spec := range specByQual {
		specs = append(specs, spec)
	}
	sort.Strings(specs)
	return lines, specs, ""
}

var conceptQualRe = regexp.MustCompile(`([A-Za-z_]\w*)\.`)

// conceptScaffoldRe matches a rendered declaration that is a
// parameterless function scaffold: "func Name() {" — the shape
// whose plan note owns the emptiness.
var conceptScaffoldRe = regexp.MustCompile(`^func [A-Za-z_]\w*\(\) \{`)

// interfaceMethodLines renders an interface type's own declared
// methods (and embedded interfaces) as declaration lines, from
// its file's AST — the index records interface method NAMES,
// but the signatures live in the source.
func (p *Project) interfaceMethodLines(t *Type) []string {
	_, fset, fileAST, refusal := p.readProjectFile(t.File)
	if refusal != "" || fileAST == nil {
		return nil
	}
	_ = fset
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.TYPE {
			continue
		}
		for _, spec := range gd.Specs {
			ts, ok := spec.(*ast.TypeSpec)
			if !ok || ts.Name.Name != t.Name {
				continue
			}
			it, ok := ts.Type.(*ast.InterfaceType)
			if !ok || it.Methods == nil {
				return nil
			}
			var lines []string
			for _, m := range it.Methods.List {
				if len(m.Names) == 0 {
					lines = append(lines, "\t"+typeString(m.Type))
					continue
				}
				ft, ok := m.Type.(*ast.FuncType)
				if !ok {
					continue
				}
				line := "\t" + m.Names[0].Name + "("
				if ft.Params != nil {
					line += fieldListString(ft.Params.List)
				}
				line += ")"
				if ft.Results != nil && len(ft.Results.List) > 0 {
					res := ft.Results.List
					if len(res) == 1 && len(res[0].Names) == 0 {
						line += " " + typeString(res[0].Type)
					} else {
						line += " (" + fieldListString(res) + ")"
					}
				}
				lines = append(lines, line)
			}
			return lines
		}
	}
	return nil
}

// methodSigLine turns an index signature ("func (s *Store)
// Load(path string) (Item, error)") into the interface method
// line ("Load(path string) (Item, error)").
func methodSigLine(sig string) string {
	s := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(sig), "func "))
	if strings.HasPrefix(s, "(") {
		if i := strings.Index(s, ") "); i >= 0 {
			return strings.TrimSpace(s[i+2:])
		}
		return ""
	}
	return s
}

// renderConceptTemplate substitutes the bound slots into a book
// template: {text} with the Go-escaped content of the saying
// slot (empty by default — the print line's user-specified
// form), {name} with the declaration name, {methods} with the
// method block the planner bound (verbatim; lines carry their
// own indent).
func renderConceptTemplate(entry *ConceptEntry, req EditRequest, methods string) string {
	quoted := strconv.Quote(req.ConceptText)
	content := quoted[1 : len(quoted)-1]
	out := strings.ReplaceAll(entry.Template, "{text}", content)
	out = strings.ReplaceAll(out, "{name}", strings.TrimSpace(req.ConceptName))
	out = strings.ReplaceAll(out, "{methods}", methods)
	return out
}

// indentConceptLines prefixes every line of text with indent.
func indentConceptLines(text, indent string) string {
	lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
	for i, ln := range lines {
		if strings.TrimSpace(ln) == "" {
			continue
		}
		lines[i] = indent + ln
	}
	return strings.Join(lines, "\n")
}

// indentConceptDecl re-indents a rendered declaration: the
// template's first line stays flush, every later line gets one
// tab unless it already carries indent or is the closing brace.
func indentConceptDecl(decl string) string {
	lines := strings.Split(decl, "\n")
	for i := 1; i < len(lines); i++ {
		ln := lines[i]
		if strings.TrimSpace(ln) == "" || strings.TrimSpace(ln) == "}" ||
			strings.HasPrefix(ln, "\t") || strings.HasPrefix(ln, " ") {
			continue
		}
		lines[i] = "\t" + ln
	}
	return strings.Join(lines, "\n")
}

func conceptNameList(names []string) string {
	if len(names) == 0 {
		return "none — the file declares no functions"
	}
	return strings.Join(names, ", ")
}

func conceptFirstOr(names []string, fallback string) string {
	if len(names) > 0 {
		return names[0]
	}
	return fallback
}

// conceptStmtText renders a top-level statement for the
// containment checks below. When the assembler inserts a
// statement directly after a comment sitting at the end of the
// target body — the routing row's trailing TODO line is the
// shipped example — the parser attaches that comment to the
// inserted statement as its Doc whenever the statement is
// declaration-led (a var row, like the type switch), and the
// printer then leads with the comment instead of the
// statement's own first line, hiding it from the prefix test.
// The comment belongs to the file, not to the statement:
// render a declaration-led statement without its Doc. (Found
// 2026-10-31: routing then type switch in one main was refused
// even though the statement had landed, correctly placed.)
func conceptStmtText(fset *token.FileSet, stmt ast.Stmt) string {
	if ds, ok := stmt.(*ast.DeclStmt); ok {
		if gd, ok := ds.Decl.(*ast.GenDecl); ok && gd.Doc != nil {
			bare := *gd
			bare.Doc = nil
			return stmtText(fset, &ast.DeclStmt{Decl: &bare})
		}
	}
	return stmtText(fset, stmt)
}

// stmtLineKey canonicalizes a statement's first line the
// way stmtText canonicalizes the rendered side of the
// containment checks below: whitespace runs collapse to
// single spaces. Both sides go through the same
// canonicalization, so a template whose first line carries
// significant whitespace inside a string literal (the
// trim-whitespace row's padded sample) is judged on its
// tokens, exactly like every other row — the check still
// requires the whole first line, token for token, at the
// head of a statement in the target function.
func stmtLineKey(firstLine string) string {
	return strings.Join(strings.Fields(firstLine), " ")
}

// selfCheckConceptStatement re-parses the assembled file and
// confirms the target function now contains the rendered
// statement's first line.
func selfCheckConceptStatement(after []byte, funcName, firstLine string) string {
	fset, fileAST, err := parseGoSrc("concept.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != funcName || fd.Body == nil {
			continue
		}
		for _, stmt := range fd.Body.List {
			if strings.HasPrefix(conceptStmtText(fset, stmt), stmtLineKey(firstLine)) {
				return ""
			}
		}
		return fmt.Sprintf("%s doesn't contain the statement after assembly", funcName)
	}
	return fmt.Sprintf("the assembled file has no function named %s", funcName)
}

// selfCheckConceptDeclaration re-parses the assembled file and
// confirms it now declares the given name at top level — a type
// or a plain function, whichever form the row's template builds
// (the planner is generic by kind, and "declaration" covers
// both).
func selfCheckConceptDeclaration(after []byte, name string) string {
	_, fileAST, err := parseGoSrc("concept.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	for _, d := range fileAST.Decls {
		switch d := d.(type) {
		case *ast.GenDecl:
			if d.Tok != token.TYPE {
				continue
			}
			for _, spec := range d.Specs {
				if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == name {
					return ""
				}
			}
		case *ast.FuncDecl:
			if d.Recv == nil && d.Name.Name == name {
				return ""
			}
		}
	}
	return fmt.Sprintf("the assembled file doesn't declare %s", name)
}

// selfCheckConceptStatementAt is selfCheckConceptStatement
// with the position a line placement resolved: the rendered
// statement must sit at index idx in the target function's
// body, not merely somewhere inside it.
func selfCheckConceptStatementAt(after []byte, funcName, firstLine string, idx int) string {
	fset, fileAST, err := parseGoSrc("concept.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != funcName || fd.Body == nil {
			continue
		}
		if !stmtAtIndex(fset, fd, idx, firstLine) {
			return fmt.Sprintf("%s doesn't hold the statement at the placed position after assembly", funcName)
		}
		return ""
	}
	return fmt.Sprintf("the assembled file has no function named %s", funcName)
}

// selfCheckConceptDeclarationAt is selfCheckConceptDeclaration
// with the position a line placement resolved: the new
// declaration must sit at index idx in the file's top-level
// declarations, not merely somewhere among them.
func selfCheckConceptDeclarationAt(after []byte, name string, idx int) string {
	_, fileAST, err := parseGoSrc("concept.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	if idx < 0 || idx >= len(fileAST.Decls) {
		return fmt.Sprintf("the assembled file's declarations don't hold %s at the placed position", name)
	}
	switch d := fileAST.Decls[idx].(type) {
	case *ast.GenDecl:
		if d.Tok == token.TYPE {
			for _, spec := range d.Specs {
				if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == name {
					return ""
				}
			}
		}
	case *ast.FuncDecl:
		if d.Recv == nil && d.Name.Name == name {
			return ""
		}
	}
	return fmt.Sprintf("the assembled file's declarations don't hold %s at the placed position", name)
}

// verifyConceptStatementAt is the post-edit form of
// selfCheckConceptStatementAt: the placed statement must sit
// at the resolved index in the target function's body.
func verifyConceptStatementAt(root, rel, funcName, firstLine string, idx int) bool {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != funcName || fd.Body == nil {
			continue
		}
		return stmtAtIndex(fset, fd, idx, firstLine)
	}
	return false
}

// verifyConceptStatement re-reads the edited file and confirms
// the target function holds the statement — the
// verifyCaseExists pattern.
func verifyConceptStatement(root, rel, funcName, firstLine string) bool {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != funcName || fd.Body == nil {
			continue
		}
		for _, stmt := range fd.Body.List {
			if strings.HasPrefix(conceptStmtText(fset, stmt), stmtLineKey(firstLine)) {
				return true
			}
		}
		return false
	}
	return false
}
