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
//	  "imports":  ["fmt"]                     // optional
//	}
//
// Slots: {text} is the saying slot — it is substituted with the
// Go-escaped CONTENT of a string literal (the template carries
// the quotes, as the print line's does); it defaults to empty.
// {name} is a declaration's name (required for declarations).
// {methods} is a declaration's method block, filled from a real
// type's method set when the sentence says "like <Type>", else
// a TODO comment — methods are never invented.

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

// --- statement concepts ------------------------------------------------------

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
	growMain := false
	resolvedFunc := wantFunc
	if decl == nil {
		switch {
		case explicit:
			return nil, fmt.Sprintf("%s has no function named %q to put the %s in — the functions there are: %s. Name one of those, like \"add a %s in %s to %s\".",
				p.LinkPath(rel), wantFunc, entry.Canonical(), conceptNameList(funcNames), entry.Canonical(), conceptFirstOr(funcNames, "main"), p.LinkPath(rel))
		case fileAST.Name.Name != "main":
			return nil, fmt.Sprintf("%s is package %s, so it has no main to put the %s in — name a function, like \"add a %s in %s to %s\" (the functions there are: %s).",
				p.LinkPath(rel), fileAST.Name.Name, entry.Canonical(), entry.Canonical(), conceptFirstOr(funcNames, "main"), p.LinkPath(rel), conceptNameList(funcNames))
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
			// An empty one-line body "{}": open it up.
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

	summary := fmt.Sprintf("Add a %s at the end of %s in %s, from the concept book: %s",
		entry.Canonical(), resolvedFunc, p.LinkPath(rel), firstLine)
	plan := &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s now has the %s in %s", filepath.Base(rel), entry.Canonical(), resolvedFunc),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, firstLine)),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			return verifyConceptStatement(after.Root, rel, resolvedFunc, firstLine)
		},
	}
	if note != "" {
		plan.Notes = append(plan.Notes, note)
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

	var specs []string
	for _, path := range entry.Imports {
		specs = append(specs, strconv.Quote(path))
	}
	specs = append(specs, methodSpecs...)
	specs = missingImportSpecs(fileAST, specs)

	// Imports first, the declaration appended after — the same
	// coincident-offset hazard as the statement planner's
	// grown main (a bare-clause file puts both at EOF).
	base := applyTextEdits(src, conceptImportEdits(fileAST, fset, src, specs))
	text := "\n" + decl + "\n"
	if len(base) == 0 || base[len(base)-1] != '\n' {
		text = "\n" + text
	}
	after := append(append([]byte{}, base...), text...)
	ef := finishFile(rel, src, after)

	// AST self-check: the file parses and now declares the name.
	if bad := selfCheckConceptDeclaration(ef.After, name); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	summary := fmt.Sprintf("Add the %s %s to %s, from the concept book.", canon, name, p.LinkPath(rel))
	if strings.TrimSpace(req.ConceptLike) != "" && strings.Contains(entry.Template, "{methods}") {
		summary = fmt.Sprintf("Add the %s %s to %s, with %s's method set copied from this project's own code.", canon, name, p.LinkPath(rel), req.ConceptLike)
	}
	plan := &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s now declares the %s %s", filepath.Base(rel), canon, name),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, "type "+name+" ")),
		Files:      []EditFile{ef},
		Notes:      notes,
		verify: func(after *Project) bool {
			for _, pk := range after.Packages {
				for _, t := range pk.Types {
					if t.Name == name && t.File == rel {
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
			if strings.HasPrefix(stmtText(fset, stmt), strings.TrimSpace(firstLine)) {
				return ""
			}
		}
		return fmt.Sprintf("%s doesn't contain the statement after assembly", funcName)
	}
	return fmt.Sprintf("the assembled file has no function named %s", funcName)
}

// selfCheckConceptDeclaration re-parses the assembled file and
// confirms it now declares a type with the given name.
func selfCheckConceptDeclaration(after []byte, name string) string {
	_, fileAST, err := parseGoSrc("concept.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.TYPE {
			continue
		}
		for _, spec := range gd.Specs {
			if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == name {
				return ""
			}
		}
	}
	return fmt.Sprintf("the assembled file doesn't declare %s", name)
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
			if strings.HasPrefix(stmtText(fset, stmt), strings.TrimSpace(firstLine)) {
				return true
			}
		}
		return false
	}
	return false
}
