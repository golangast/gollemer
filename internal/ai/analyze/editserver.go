package analyze

// The add-a-webserver edit: a sixth assembled shape, planned in
// edits.go's family but assembled here because it is the one
// shape whose "sibling" is a fixed component rather than code
// copied out of the project. "add a webserver to jim/main.go"
// splices the tiny net/http component below into an existing
// package-main file — behind the same PlanEdit spine as every
// other write: plan, diff, gate, build, verify, restore.

import (
	"fmt"
	"go/ast"
	"path/filepath"
	"strconv"
	"strings"
)

// The webserver component: ONE source of truth for the tiny
// net/http server, shared by both write paths that can produce
// it — the chat's gated creations write WebServerSource whole
// into a new file, and planAddServer below splices
// WebServerMainBody into an existing file's main. It used to live
// in the chat package as the creations starter; it moved here so
// the two can never drift into two different servers.

// WebServerMainBody is the statement block of the component's
// main: register one handler, announce, serve. Tab-indented, one
// statement group per line block, no trailing blank line — ready
// to sit between a func main's braces.
const WebServerMainBody = `	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "hello — your webserver is running")
	})
	fmt.Println("listening at http://localhost:8080 — press Ctrl+C to stop")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
`

// WebServerSource is the whole component as a standalone file:
// the package clause, its imports, and a main whose body is
// exactly WebServerMainBody.
const WebServerSource = `package main

import (
	"fmt"
	"net/http"
)

// A tiny webserver: every page gets one friendly line back.
// Run it with: go run main.go — then open http://localhost:8080
func main() {
` + WebServerMainBody + `}
`

// planAddServer assembles the edit for req.File. The file's own
// state decides the shape:
//
//	(a) a bare package clause becomes the whole component file;
//	(b) a main-less file grows a main carrying the component;
//	(c) an empty main is filled with the component's body.
//
// Three states are honest refusals instead: a non-empty main
// (splicing a server into running logic isn't safe — the refusal
// carries the component's exact lines to paste by hand), a
// library package (a webserver is a program), and a file that
// already serves. The component reuses the template's anonymous
// handler, so there is no handler name to collide with.
func (p *Project) planAddServer(req EditRequest) (*EditPlan, string) {
	rel := filepath.ToSlash(strings.TrimSpace(req.File))
	if rel == "" {
		return nil, "that edit needs a file to add the webserver to."
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

	// (e) A webserver is a program; programs live in package main.
	if fileAST.Name.Name != "main" {
		return nil, p.serverNeedsMainRefusal(rel, fileAST.Name.Name)
	}
	// (f) Already serving: net/http imported AND ListenAndServe
	// called — the component, or one like it, is already there.
	if hasImportPath(fileAST, "net/http") && callsListenAndServe(fileAST) {
		return nil, fmt.Sprintf("%s already has a webserver — it imports net/http and calls ListenAndServe. Nothing to add.", p.LinkPath(rel))
	}

	var mainDecl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "main" {
			mainDecl = fd
			break
		}
	}

	// (a) A bare package clause and nothing else: the file
	// becomes the whole component — byte for byte the server a
	// creation would write.
	if len(fileAST.Decls) == 0 && strings.TrimSpace(string(src)) == "package main" {
		ef := finishFile(rel, src, []byte(WebServerSource))
		return &EditPlan{
			Summary:    fmt.Sprintf("Turn %s into a working webserver — the whole tiny program: package main, one handler, listening at http://localhost:8080.", p.LinkPath(rel)),
			VerifyDesc: fmt.Sprintf("%s is a program that serves HTTP", filepath.Base(rel)),
			Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, "http.ListenAndServe")),
			Files:      []EditFile{ef},
			verify:     func(after *Project) bool { return verifyServerPresent(after.Root, rel) },
		}, ""
	}

	// (d) A main that already does things: refuse, handing over
	// the component's exact lines instead of splicing blindly.
	if mainDecl != nil && mainDecl.Body != nil && len(mainDecl.Body.List) > 0 {
		return nil, serverByHandRefusal(p.LinkPath(rel))
	}

	// Imports the component needs that the file lacks.
	var missing []string
	for _, path := range []string{"fmt", "net/http"} {
		if !hasImportPath(fileAST, path) {
			missing = append(missing, strconv.Quote(path))
		}
	}
	var edits []textEdit
	if len(missing) > 0 {
		edits = append(edits, insertImportSpecs(fileAST, fset, src, missing)...)
	}

	var summary string
	if mainDecl == nil {
		// (b) No main yet: append one carrying the component.
		text := "\n// A tiny webserver: every page gets one friendly line back.\n" +
			"// It listens at http://localhost:8080.\n" +
			"func main() {\n" + WebServerMainBody + "}\n"
		if !strings.HasSuffix(string(src), "\n") {
			text = "\n" + text
		}
		edits = append(edits, textEdit{start: len(src), end: len(src), text: text})
		summary = fmt.Sprintf("Add a webserver to %s: a new main() registers one handler and serves at http://localhost:8080.", p.LinkPath(rel))
	} else {
		// (c) An empty main: fill it with the component's body.
		edits = append(edits, textEdit{
			start: offOf(fset, mainDecl.Body.Lbrace) + 1,
			end:   offOf(fset, mainDecl.Body.Rbrace),
			text:  "\n" + WebServerMainBody,
		})
		summary = fmt.Sprintf("Add a webserver to %s: main() registers one handler and serves at http://localhost:8080.", p.LinkPath(rel))
	}
	after := applyTextEdits(src, edits)
	ef := finishFile(rel, src, after)
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("%s has a main that serves HTTP", filepath.Base(rel)),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, "http.ListenAndServe")),
		Files:      []EditFile{ef},
		verify:     func(after *Project) bool { return verifyServerPresent(after.Root, rel) },
	}, ""
}

// serverNeedsMainRefusal is state (e)'s answer: why not, and the
// nearest real capability in one breath — the project's own main
// file when the analyzer knows one, a fresh server file when it
// doesn't.
func (p *Project) serverNeedsMainRefusal(rel, pkgName string) string {
	base := fmt.Sprintf("%s is package %s — a webserver is a program, and programs live in package main.", p.LinkPath(rel), pkgName)
	if eps := p.EntryPoints(); len(eps) > 0 {
		mf := eps[0]
		return fmt.Sprintf("%s This project's program starts in %s — say \"add a webserver to %s\" and I'll put it there.", base, p.LinkPath(mf.File), mf.File)
	}
	return fmt.Sprintf("%s Say \"create a file main.go with a webserver\" and I'll write you a whole server file instead.", base)
}

// serverByHandRefusal is state (d)'s answer: main already does
// its own work, so the component's exact lines — the register
// block and the serve call, verbatim from the template — are
// handed over to paste, with the handler note.
func serverByHandRefusal(linkPath string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "main() in %s already does its own work, so I won't splice a server into running code. To add it by hand, paste these lines inside main (the handler answers every page with one friendly line; the file also needs \"fmt\" and \"net/http\" imported):\n\n", linkPath)
	b.WriteString(strings.TrimRight(WebServerMainBody, "\n"))
	return b.String()
}

// hasImportPath reports whether the file imports path (by path,
// whatever local name it goes by).
func hasImportPath(f *ast.File, path string) bool {
	for _, is := range f.Imports {
		if pth, _ := strconv.Unquote(is.Path.Value); pth == path {
			return true
		}
	}
	return false
}

// callsListenAndServe reports whether the file calls
// something.ListenAndServe anywhere.
func callsListenAndServe(f *ast.File) bool {
	found := false
	ast.Inspect(f, func(n ast.Node) bool {
		if found {
			return false
		}
		if call, ok := n.(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "ListenAndServe" {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

// verifyServerPresent re-reads the edited file and confirms it
// is now a program that serves: package main, a func main, and
// a ListenAndServe call — the verifyCaseExists pattern.
func verifyServerPresent(root, rel string) bool {
	_, _, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil || fileAST.Name.Name != "main" {
		return false
	}
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "main" {
			return callsListenAndServe(fileAST)
		}
	}
	return false
}
