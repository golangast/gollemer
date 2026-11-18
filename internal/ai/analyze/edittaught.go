package analyze

// Taught declarations: the teach seat's writer. When the user
// teaches the chat a piece from an example in their own project
// ("learn a sidebar page from Jake in jim/main.go"), the chat
// layer stores the example's source as a template with the
// declaration's name as the only slot. A later ask ("add a
// sidebar page named Signup to site/main.go") arrives here as
// data — the template already renamed to the new name — exactly
// the way a concept-book row's rendered template or a proposed
// declaration arrives: assembled by this planner, gated, built
// and verified by the shared spine. The template is the user's
// own code, carried verbatim; the planner adds the imports it
// leans on, checks the name against the package, and refuses
// the states it cannot assemble honestly. Nothing here invents
// code: what lands is the example, renamed, and if the project
// then refuses to build (the example leaned on something its
// new home lacks), the spine restores and says so.

import (
	"fmt"
	"go/format"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// planTaughtDecl assembles one taught declaration into
// req.File: appended at file scope when the file exists, or
// created as a new file under the bare-file family's package
// rules when it doesn't — the two licences a concept-book
// declaration already holds.
func (p *Project) planTaughtDecl(req EditRequest) (*EditPlan, string) {
	text := strings.TrimSpace(req.TaughtText)
	if text == "" {
		return nil, "that edit needs a taught piece to add, and none was supplied."
	}
	rel := strings.TrimSpace(req.File)
	if rel == "" {
		return nil, "that edit needs a file to add the piece to."
	}
	name := strings.TrimSpace(req.TaughtName)
	if name == "" {
		return nil, "the taught piece has no name to add it under."
	}
	kind := strings.TrimSpace(req.TaughtKindPhrase)
	if kind == "" {
		kind = "piece"
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(rel), why)
	}
	if !token.IsIdentifier(name) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a declaration.", name)
	}
	if _, err := os.Stat(filepath.Join(p.Root, rel)); os.IsNotExist(err) {
		return p.planTaughtNamedFileCreate(req, text, rel, name, kind)
	}
	src, fset, fileAST, refusal := p.readProjectFile(rel)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(rel))
	}
	if pkg := p.pkgOfFile(rel); pkg != nil && p.pkgDeclares(pkg, name) {
		return nil, fmt.Sprintf("%s already exists in package %s (%s) — pick a different name for the %s.", name, pkg.Name, p.LinkPath(rel), kind)
	}

	// Imports the example leans on, added the concept family's
	// way: only the ones the file lacks.
	specs := missingImportSpecs(fileAST, splitImportSpecs(req.TaughtImports))

	// Imports first, the declaration appended after — the same
	// coincident-offset care as the concept declaration pass.
	base := applyTextEdits(src, conceptImportEdits(fileAST, fset, src, specs))
	appendix := "\n" + text + "\n"
	if len(base) == 0 || base[len(base)-1] != '\n' {
		appendix = "\n" + appendix
	}
	after := append(append([]byte{}, base...), appendix...)
	ef := finishFile(rel, src, after)

	// AST self-check: the assembled file parses and now
	// declares the taught name.
	if bad := selfCheckConceptDeclaration(ef.After, name); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	firstLine := strings.SplitN(text, "\n", 2)[0]
	var notes []string
	if req.TaughtNote != "" {
		notes = append(notes, req.TaughtNote)
	}
	plan := &EditPlan{
		Summary: fmt.Sprintf("Add the %s %s to %s — the piece you taught me, renamed from its example.", kind, name, p.LinkPath(rel)),
		Site:    fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, firstLine)),
		Files:   []EditFile{ef},
		Notes:   notes,
		verify: func(after *Project) bool {
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
	plan.VerifyDesc = fmt.Sprintf("%s now declares the %s %s", filepath.Base(rel), kind, name)
	return plan, ""
}

// planTaughtNamedFileCreate assembles the create plan for a
// taught declaration whose named target file is new. The
// package rules are the bare-file family's, exactly as the
// concept book's named-file create settles them
// (planConceptNamedFileCreate): the folder's own package when
// it has one, a mixing refusal when it has two, and the
// family's defaults (main.go / project root / the folder's own
// name) when it has none.
func (p *Project) planTaughtNamedFileCreate(req EditRequest, text, rel, name, kind string) (*EditPlan, string) {
	folder := filepath.ToSlash(filepath.Dir(rel))
	folderName := folder + " folder"
	dirKey := folder
	if folder == "." {
		folderName = "project folder"
		dirKey = ""
	}
	pkgs, folderExists := folderPackages(p.Root, folder)
	if !folderExists {
		return nil, fmt.Sprintf("there's no folder %s in this project — say \"create a folder %s\" first, then ask again", folder, folder)
	}
	pkgName := ""
	switch {
	case len(pkgs) > 1:
		names := make([]string, 0, len(pkgs))
		for n := range pkgs {
			names = append(names, n)
		}
		sort.Strings(names)
		var parts []string
		for _, n := range names {
			var files []string
			for _, f := range pkgs[n] {
				if folder == "." {
					files = append(files, p.LinkPath(f))
				} else {
					files = append(files, p.LinkPath(folder+"/"+f))
				}
			}
			parts = append(parts, fmt.Sprintf("package %s (%s)", n, strings.Join(files, ", ")))
		}
		return nil, fmt.Sprintf("the %s mixes %s — Go allows one package per folder, so I can't tell which package a new file there belongs to. Sort the folder out by hand, or name a file somewhere else for the %s.", folderName, strings.Join(parts, " and "), kind)
	case len(pkgs) == 1:
		for n := range pkgs {
			pkgName = n
		}
	default:
		switch {
		case filepath.Base(rel) == "main.go", folder == ".":
			pkgName = "main"
		case folderSegmentRe.MatchString(filepath.Base(folder)):
			pkgName = filepath.Base(folder)
		default:
			pkgName = "main"
		}
	}

	// A name collision anywhere in the target package refuses
	// with the family's wording, package-aware.
	if pkg := p.pkgByDir(dirKey); pkg != nil && p.pkgDeclares(pkg, name) {
		return nil, fmt.Sprintf("%s already exists in package %s (%s) — pick a different name for the %s.", name, pkg.Name, p.LinkPath(rel), kind)
	}

	// A created file named main.go is a program file, and the
	// bare-file family starts every new main.go with the empty
	// func main stub — so the created file carries the stub
	// BESIDE the taught declaration (never instead of it, and
	// never when the declaration IS func main). A package that
	// already has its main refuses, naming the file that holds
	// it — the family's rule, in its words.
	stubMain := ""
	var notes []string
	if req.TaughtNote != "" {
		notes = append(notes, req.TaughtNote)
	}
	isMainDecl := name == "main" && req.TaughtDeclKind == "function"
	if filepath.Base(rel) == "main.go" && !isMainDecl {
		if pkg := p.pkgByDir(dirKey); pkg != nil {
			for _, fn := range pkg.Funcs {
				if fn.Receiver == "" && fn.Name == "main" {
					return nil, fmt.Sprintf("%s already defines func main — that folder already has its main program, so another main.go can't be added", p.LinkPath(fn.File))
				}
			}
		}
		stubMain = "func main() {\n\t// TODO: write the body\n}"
		notes = append(notes, fmt.Sprintf("The file is named main.go, so it also carries the empty func main a new main.go starts with — the %s %s sits above it.", kind, name))
	}

	// The new file's whole content: package clause, imports,
	// declaration — assembled through the family's one import
	// inserter over a parsed skeleton, exactly as the concept
	// family's creates assemble theirs.
	base := "package " + pkgName + "\n\n" + text + "\n"
	if stubMain != "" {
		base = "package " + pkgName + "\n\n" + text + "\n\n" + stubMain + "\n"
	}
	_, skelAST, err := parseGoSrc(rel, []byte(base))
	if err != nil || skelAST == nil {
		return nil, fmt.Sprintf("internal error in my assembler: the %s %s doesn't parse as its own file — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", kind, name)
	}
	specs := missingImportSpecs(skelAST, splitImportSpecs(req.TaughtImports))
	after := []byte(base)
	if len(specs) > 0 {
		fset, ast2, err := parseGoSrc(rel, after)
		if err != nil || ast2 == nil {
			return nil, "internal error in my assembler: the assembled file doesn't parse — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written."
		}
		after = applyTextEdits(after, conceptImportEdits(ast2, fset, after, specs))
	}
	if f, err := format.Source(after); err == nil {
		after = f
	}
	if bad := selfCheckConceptDeclaration(after, name); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	if len(pkgs) == 1 && folder != "." && pkgName != folder {
		notes = append(notes, fmt.Sprintf("The %s folder's files are package %s, so the new file carries that package name, not the folder's.", folder, pkgName))
	}
	if len(pkgs) == 0 && folder != "." && pkgName == filepath.Base(folder) {
		notes = append(notes, fmt.Sprintf("The %s folder has no Go files yet, so the new file takes the folder's name as its package.", folder))
	}

	summary := fmt.Sprintf("Create %s — a new file in the %s (package %s) — with the %s %s, the piece you taught me.", p.LinkPath(rel), folderName, pkgName, kind, name)
	plan := &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s now declares the %s %s in package %s", filepath.Base(rel), kind, name, pkgName),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(after, strings.SplitN(text, "\n", 2)[0])),
		Files:      []EditFile{{Rel: rel, Before: nil, After: after, Diff: unifiedDiff(rel, nil, after)}},
		Notes:      notes,
		verify: func(after *Project) bool {
			pkg := after.pkgByDir(dirKey)
			if pkg == nil || pkg.Name != pkgName {
				return false
			}
			for _, t := range pkg.Types {
				if t.Name == name && t.File == rel {
					return true
				}
			}
			for _, fn := range pkg.Funcs {
				if fn.Receiver == "" && fn.Name == name && fn.File == rel {
					return true
				}
			}
			return false
		},
	}
	return plan, ""
}

// splitImportSpecs splits the request's space-joined import
// specs back into individual specs. A spec is a quoted path
// with an optional alias in front (db "database/sql"), so a
// plain Fields split would tear an aliased spec in two.
func splitImportSpecs(joined string) []string {
	var specs []string
	s := strings.TrimSpace(joined)
	for s != "" {
		end := importSpecEnd(s)
		if end <= 0 {
			break
		}
		specs = append(specs, strings.TrimSpace(s[:end]))
		s = strings.TrimSpace(s[end:])
	}
	return specs
}

// importSpecEnd returns the length of the first import spec
// in s: an optional alias (or _ / .) followed by the quoted
// path.
func importSpecEnd(s string) int {
	i := 0
	if s[0] != '"' {
		for i < len(s) && s[i] != ' ' && s[i] != '\t' {
			i++
		}
		for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
			i++
		}
	}
	if i >= len(s) || s[i] != '"' {
		return -1
	}
	j := i + 1
	for j < len(s) && s[j] != '"' {
		if s[j] == '\\' {
			j++
		}
		j++
	}
	if j >= len(s) {
		return -1
	}
	return j + 1
}
