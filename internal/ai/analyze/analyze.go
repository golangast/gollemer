// Package analyze reads a Go project off disk and builds a structural map
// of it: packages, exported API, a heuristic call graph, and the import
// graph. It exists so Gollemer can answer "how does this codebase work,
// where do I start, and where do I change things" about other people's
// Go code.
//
// It is pure stdlib (go/ast, go/parser) and strictly read-only: it never
// executes the analyzed code and never writes into the analyzed directory.
// Call edges are resolved heuristically (same-package names and method
// names, plus cross-package selectors through imports), which is enough
// to find the hot spots of a codebase but not a substitute for
// golang.org/x/tools on precise call graphs.
package analyze

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Project is the structural map of one Go codebase.
type Project struct {
	Root     string // absolute path analyzed
	Module   string // module path from go.mod, if found
	Packages []*Package
	Files    int
	Lines    int
	ParseErr []string // files skipped: "rel/path.go: reason"

	byID     map[string]*Func
	byMethod map[string][]*Func // method name -> methods (heuristic resolution)
	byPkgDir map[string]*Package

	// fieldReaders maps "pkg.Type.Field" -> func IDs that read the value,
	// directly or through threaded calls. Built by buildFieldReaders.
	fieldReaders map[string][]string
	// fieldWriters maps "pkg.Type.Field" -> func IDs that write the value
	// (direct only — a write is where the value is set). Built by
	// buildFieldWriters.
	fieldWriters map[string][]string
}

// Package is one directory of Go source.
type Package struct {
	Name     string   // package clause, e.g. "chat"
	Dir      string   // directory relative to Root ("" = Root)
	Doc      string   // package doc comment ("// Package chat ..."), first one found
	Files    []string // .go files, relative to Root
	Lines    int      // total lines across Files
	Imports  []string // every import path, sorted, unique
	Internal []string // imports that resolve to another project package
	Funcs    []*Func
	Types    []*Type
	HasMain  bool
	Tests    int // number of _test.go files

	aliasCache map[string]*Package // import qualifier -> project package
}

// Func is one function or method.
type Func struct {
	ID        string // Dir.Name or Dir.Recv.Name, unique within the project
	Name      string
	PkgDir    string
	Pkg       string
	Receiver  string // "" for plain functions
	Exported  bool
	IsMain    bool
	IsInit    bool
	IsTest    bool // declared in a _test.go file
	Doc       string
	File      string // relative to Root
	Line      int
	EndLine   int    // last line of the declaration (for source excerpts)
	Sig       string // rendered signature, e.g. "func (c *Chat) Add(a int) int"
	Params    int
	Calls     []string        // callee IDs (heuristic)
	Callers   []string        // filled by BuildGraph
	LoopCalls map[string]bool // callee IDs called inside a for/range loop
	Score     int             // importance, filled by Rank

	ExtCalls   []string // external calls, e.g. "os.Remove" (sorted, unique)
	Mutates    bool     // directly invokes a mutating external call
	MutatesAll bool     // transitively mutates: self or any callee does
	Guards     []Guard  // if-statements and the calls each one guards

	Effects    []string          // direct effect categories, e.g. "fs-write"
	EffectsAll []string          // transitive effect categories
	Reads      []string          // "pkg.Type.Field" values read via params
	Writes     []string          // "pkg.Type.Field" values written
	ParamTypes map[string]string // param/receiver name -> "pkg.Type"
	CallArgs   []CallArg         // call sites with identifier arguments

	localTypes map[string]string // local var name -> "pkg.Type"

	decl *ast.FuncDecl  // kept during scan for call extraction, then dropped
	fset *token.FileSet // shared with the scan; lets effects read positions
	pkg  *Package
}

// Type is one named type declaration.
type Type struct {
	Name      string
	PkgDir    string
	Pkg       string
	Kind      string // "struct", "interface", or "other"
	Exported  bool
	Doc       string
	Methods   []string // method names declared on it
	Fields    []string // struct field names
	ImplCount int      // interfaces only: heuristic implementer count
	File      string
	Line      int
}

// Analyze reads the Go project rooted at dir and returns its structural map.
func Analyze(dir string) (*Project, error) {
	if strings.TrimSpace(dir) == "" {
		return nil, fmt.Errorf("analyze: empty directory")
	}
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("analyze: %w", err)
	}
	fi, err := os.Stat(abs)
	if err != nil {
		return nil, fmt.Errorf("analyze: %w", err)
	}
	if !fi.IsDir() {
		return nil, fmt.Errorf("analyze: %s is not a directory", dir)
	}
	p := &Project{
		Root:     abs,
		byID:     map[string]*Func{},
		byMethod: map[string][]*Func{},
		byPkgDir: map[string]*Package{},
	}
	p.Module = findModule(abs)
	fset := token.NewFileSet()
	walkErr := filepath.WalkDir(abs, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // unreadable entry: skip, keep going
		}
		if d.IsDir() {
			name := d.Name()
			if name == ".git" || name == "vendor" || strings.HasPrefix(name, ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		rel, _ := filepath.Rel(abs, path)
		src, perr := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if perr != nil {
			p.ParseErr = append(p.ParseErr, rel+": "+firstLine(perr.Error()))
			return nil
		}
		p.addFile(rel, src, fset)
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("analyze: %w", walkErr)
	}
	if len(p.Packages) == 0 {
		return nil, fmt.Errorf("analyze: no Go packages found under %s", dir)
	}
	p.BuildGraph()
	p.linkInternal()
	p.Rank()
	return p, nil
}

// pkgFor returns the Package for dir, creating it on first use.
func (p *Project) pkgFor(dir, pkgName string) *Package {
	if pkg, ok := p.byPkgDir[dir]; ok {
		return pkg
	}
	pkg := &Package{Name: pkgName, Dir: dir}
	p.byPkgDir[dir] = pkg
	p.Packages = append(p.Packages, pkg)
	return pkg
}

// funcID builds the unique ID for a function or method.
func funcID(dir, recv, name string) string {
	base := dir
	if base == "" {
		base = "root"
	}
	base = strings.ReplaceAll(base, string(filepath.Separator), ".")
	if recv != "" {
		return base + "." + recv + "." + name
	}
	return base + "." + name
}

// addFile records one parsed file's declarations.
func (p *Project) addFile(rel string, src *ast.File, fset *token.FileSet) {
	dir := filepath.Dir(rel)
	if dir == "." {
		dir = ""
	}
	pkg := p.pkgFor(dir, src.Name.Name)
	pkg.Files = append(pkg.Files, rel)
	// Keep the most descriptive package doc: authors usually document
	// the package in one file and leave the rest bare.
	if d := docText(src.Doc); d != "" && len(d) > len(pkg.Doc) {
		pkg.Doc = d
	}
	p.Files++
	if tf := fset.File(src.Pos()); tf != nil {
		n := tf.LineCount()
		p.Lines += n
		pkg.Lines += n
	}
	isTestFile := strings.HasSuffix(rel, "_test.go")
	if isTestFile {
		pkg.Tests++
	}
	for _, imp := range src.Imports {
		path := strings.Trim(imp.Path.Value, `"`)
		if !contains(pkg.Imports, path) {
			pkg.Imports = append(pkg.Imports, path)
		}
	}
	for _, decl := range src.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			fn := &Func{
				Name:     d.Name.Name,
				PkgDir:   dir,
				Pkg:      src.Name.Name,
				Exported: ast.IsExported(d.Name.Name),
				IsTest:   isTestFile,
				Doc:      docText(d.Doc),
				File:     rel,
				Line:     fset.Position(d.Pos()).Line,
				EndLine:  fset.Position(d.End()).Line,
				decl:     d,
				fset:     fset,
				pkg:      pkg,
			}
			if d.Recv != nil && len(d.Recv.List) > 0 {
				fn.Receiver = exprName(d.Recv.List[0].Type)
			}
			fn.Sig = renderSig(d, fn.Receiver)
			if d.Type.Params != nil {
				for _, field := range d.Type.Params.List {
					n := len(field.Names)
					if n == 0 {
						n = 1
					}
					fn.Params += n
				}
			}
			fn.IsMain = fn.Receiver == "" && fn.Name == "main" && src.Name.Name == "main"
			fn.IsInit = fn.Receiver == "" && fn.Name == "init"
			fn.ID = funcID(dir, fn.Receiver, fn.Name)
			pkg.Funcs = append(pkg.Funcs, fn)
			if fn.IsMain {
				pkg.HasMain = true
			}
			p.byID[fn.ID] = fn
			if fn.Receiver != "" {
				p.byMethod[fn.Name] = append(p.byMethod[fn.Name], fn)
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				t := &Type{
					Name:     ts.Name.Name,
					PkgDir:   dir,
					Pkg:      src.Name.Name,
					Kind:     "other",
					Exported: ast.IsExported(ts.Name.Name),
					Doc:      docText(d.Doc),
					File:     rel,
					Line:     fset.Position(ts.Pos()).Line,
				}
				switch st := ts.Type.(type) {
				case *ast.StructType:
					t.Kind = "struct"
					t.Fields = fieldNames(st.Fields)
				case *ast.InterfaceType:
					t.Kind = "interface"
					t.Methods = fieldNames(st.Methods)
				}
				pkg.Types = append(pkg.Types, t)
			}
		}
	}
}

// finalize sorts everything for stable output and drops the ASTs.
func (p *Project) finalize() {
	sort.Slice(p.Packages, func(i, j int) bool { return p.Packages[i].Dir < p.Packages[j].Dir })
	for _, pkg := range p.Packages {
		sort.Strings(pkg.Files)
		sort.Strings(pkg.Imports)
		sort.Slice(pkg.Funcs, func(i, j int) bool { return pkg.Funcs[i].ID < pkg.Funcs[j].ID })
		sort.Slice(pkg.Types, func(i, j int) bool { return pkg.Types[i].Name < pkg.Types[j].Name })
		for _, fn := range pkg.Funcs {
			fn.decl = nil
		}
	}
}

// findModule reads the module path from the nearest go.mod at or above dir.
func findModule(dir string) string {
	for d := dir; ; d = filepath.Dir(d) {
		data, err := os.ReadFile(filepath.Join(d, "go.mod"))
		if err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "module "); ok {
					return strings.TrimSpace(rest)
				}
			}
			return ""
		}
		parent := filepath.Dir(d)
		if parent == d {
			return ""
		}
	}
}

// docText flattens a comment group to one line.
func docText(cg *ast.CommentGroup) string {
	if cg == nil {
		return ""
	}
	t := strings.TrimSpace(cg.Text())
	t = strings.Join(strings.Fields(t), " ")
	if len(t) > 220 {
		t = t[:217] + "..."
	}
	return t
}

// exprName renders a receiver/field type expression to a short name.
func exprName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return exprName(t.X)
	case *ast.IndexExpr: // generics: Box[T]
		return exprName(t.X)
	case *ast.IndexListExpr:
		return exprName(t.X)
	case *ast.SelectorExpr:
		return t.Sel.Name
	}
	return ""
}

// fieldNames collects declared field/method names from a field list.
func fieldNames(fl *ast.FieldList) []string {
	var out []string
	if fl == nil {
		return out
	}
	for _, f := range fl.List {
		for _, n := range f.Names {
			out = append(out, n.Name)
		}
		// Embedded field or interface method set member without a name.
		if len(f.Names) == 0 {
			if id := exprName(f.Type); id != "" {
				out = append(out, id)
			}
		}
	}
	return out
}

func contains(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// LinkPath renders a project-relative file path as a terminal-clickable
// link. When the analyzed project isn't the working directory — a fresh
// clone in ./example, a repo in ~/elsewhere — the path is rewritten
// relative to the working directory, so file:line links click through
// to the file. When the project is the working directory, the path is
// unchanged.
func (p *Project) LinkPath(rel string) string {
	if p.Root == "" || rel == "" {
		return rel
	}
	cwd, err := os.Getwd()
	if err != nil {
		return rel
	}
	if r, err := filepath.Rel(cwd, filepath.Join(p.Root, rel)); err == nil && r != "" {
		return r
	}
	return rel
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// taskAliases expands a few user words to the codebase's own vocabulary
// before matching. A starting set — extend as real usage shows gaps.
var taskAliases = map[string][]string{
	"brain": {"domain"}, // gollemer's "brains" are domain constants + router cases
}

// camelSplit splits "ServeHTTP" into ["Serve", "HTTP"], "userID" into
// ["user", "ID"]. Used to match task keywords against symbol names.
func camelSplit(s string) []string {
	var words []string
	start := 0
	runes := []rune(s)
	for i := 1; i <= len(runes); i++ {
		if i == len(runes) || (isUpper(runes[i]) && !isUpper(runes[i-1])) ||
			(isUpper(runes[i-1]) && i < len(runes) && !isUpper(runes[i])) {
			if i > start {
				words = append(words, string(runes[start:i]))
			}
			start = i
		}
	}
	return words
}

func isUpper(r rune) bool { return r >= 'A' && r <= 'Z' }

// wordSet tokenizes text into a lowercase word set.
func wordSet(s string) map[string]bool {
	set := map[string]bool{}
	for _, w := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	}) {
		if len(w) > 1 && !stopwords[w] {
			set[w] = true
		}
	}
	return set
}

var stopwords = map[string]bool{
	"the": true, "and": true, "for": true, "with": true, "that": true,
	"this": true, "from": true, "into": true, "are": true, "was": true,
	"have": true, "has": true, "will": true, "would": true, "should": true,
	"can": true, "its": true, "our": true, "your": true, "their": true,
	"they": true, "them": true, "then": true, "than": true, "when": true,
	"what": true, "where": true, "which": true, "how": true, "all": true,
	"any": true, "each": true, "new": true, "use": true, "used": true,
	"using": true, "also": true, "such": true, "like": true, "just": true,
	"only": true, "not": true, "but": true, "out": true, "about": true,
	"more": true, "some": true, "does": true, "did": true, "been": true,
	"being": true, "between": true, "both": true, "code": true, "golang": true,
	// Change-verbs carry no domain meaning in "where do I <verb> X" tasks;
	// dropping them keeps matches focused on the thing being changed.
	"add": true, "make": true, "get": true, "set": true, "put": true,
	"change": true, "update": true, "modify": true, "fix": true,
	"implement": true, "create": true, "build": true, "run": true,
	"handle": true, "support": true, "need": true, "needs": true, "want": true,
}
