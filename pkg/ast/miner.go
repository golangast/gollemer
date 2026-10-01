// Repository style mining: infer a codebase's coding conventions.
//
// InferRepositoryStyle inspects every package in the module that owns
// the CodebaseContext and distills how that repository writes Go:
// how it creates errors, which struct tags it favors, whether
// context.Context is systematic, and how densely it documents exported
// identifiers. The resulting RepoStyle feeds SystemPromptGuidelines,
// turning observations into plain-text instructions for code
// generation so new code matches the repo's habits.
//
// All classification uses type-checked ASTs (go/packages +
// go/types), not text search, so aliased imports (e.g. errpkg
// "github.com/pkg/errors") and dot imports resolve correctly. Only
// positively type-resolved references are counted, so packages with
// type errors contribute what resolved rather than noise.
package ast

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strconv"
	"strings"
)

// Error handling styles recognized by the miner.
const (
	// ErrorStyleFmtWrap: errors wrapped with fmt.Errorf and %w.
	ErrorStyleFmtWrap = "fmt_wrap"
	// ErrorStylePkgErrors: errors wrapped with github.com/pkg/errors.
	ErrorStylePkgErrors = "pkg_errors"
	// ErrorStyleStandard: errors created with the standard errors
	// package (errors.New / errors.Join) or returned plain.
	ErrorStyleStandard = "standard_errors"
)

// RepoStyle is the inferred coding conventions of one repository.
type RepoStyle struct {
	// ErrorHandlingStyle is one of "fmt_wrap", "pkg_errors",
	// "standard_errors".
	ErrorHandlingStyle string `json:"errorHandlingStyle"`
	// CommonStructTags lists struct tag keys by frequency, most
	// common first (e.g. ["json", "db", "validate"]).
	CommonStructTags []string `json:"commonStructTags"`
	// UsesContext reports whether a majority of exported functions
	// accept context.Context as their first parameter.
	UsesContext bool `json:"usesContext"`
	// DocCommentDensity is the fraction of exported identifiers
	// carrying a doc comment, in [0, 1].
	DocCommentDensity float64 `json:"docCommentDensity"`
}

// SystemPromptGuidelines converts the inferred conventions into
// plain-text system instructions for an LLM prompt.
func (s *RepoStyle) SystemPromptGuidelines() string {
	if s == nil {
		return ""
	}
	var b strings.Builder
	switch s.ErrorHandlingStyle {
	case ErrorStyleFmtWrap:
		b.WriteString("Always wrap errors using fmt.Errorf with %w to preserve the error chain.\n")
	case ErrorStylePkgErrors:
		b.WriteString("Always wrap errors using github.com/pkg/errors (errors.Wrap / errors.Wrapf).\n")
	default:
		b.WriteString("Create errors with the standard errors package (errors.New); return errors directly without a wrapping library.\n")
	}
	if len(s.CommonStructTags) > 0 {
		top := s.CommonStructTags
		if len(top) > 5 {
			top = top[:5]
		}
		fmt.Fprintf(&b, "Use these struct tags where appropriate: %s.\n", strings.Join(top, ", "))
	}
	if s.UsesContext {
		b.WriteString("Always accept ctx context.Context as the first argument of exported functions.\n")
	}
	switch {
	case s.DocCommentDensity >= 0.8:
		b.WriteString("Write a doc comment for every exported identifier, matching the repository's thorough documentation habit.\n")
	case s.DocCommentDensity >= 0.4:
		b.WriteString("Write doc comments for exported identifiers.\n")
	default:
		b.WriteString("Keep comments sparse, matching the repository's light documentation style.\n")
	}
	return b.String()
}

// InferRepositoryStyle mines the coding conventions of the module
// containing the CodebaseContext's source directory.
func InferRepositoryStyle(codeCtx *CodebaseContext) (*RepoStyle, error) {
	if codeCtx == nil {
		return nil, fmt.Errorf("miner: nil CodebaseContext")
	}
	if codeCtx.SourceDir == "" {
		return nil, fmt.Errorf("miner: CodebaseContext has no SourceDir (load it with LoadPackageContext)")
	}
	_, pkgs, err := loadModulePackages(codeCtx.SourceDir)
	if err != nil {
		return nil, fmt.Errorf("miner: %w", err)
	}
	m := &styleMiner{tagCounts: make(map[string]int)}
	for _, p := range pkgs {
		if p == nil || p.TypesInfo == nil || len(p.Syntax) == 0 {
			continue
		}
		for _, f := range p.Syntax {
			if f == nil {
				continue
			}
			m.visitFile(f, p.TypesInfo)
		}
	}
	return m.style(), nil
}

// styleMiner accumulates observations across a module's ASTs.
type styleMiner struct {
	fmtWrap, pkgErrors, stdErr int
	tagCounts                  map[string]int
	exportedFuncs, ctxFuncs    int
	exportedIdents, documented int
}

func (m *styleMiner) visitFile(f *ast.File, info *types.Info) {
	ast.Inspect(f, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncDecl:
			m.visitFuncDecl(n, info)
		case *ast.GenDecl:
			m.visitGenDecl(n)
		case *ast.StructType:
			m.visitStructType(n)
		case *ast.CallExpr:
			m.visitCallExpr(n, info)
		}
		return true
	})
}

// visitFuncDecl records context.Context usage and doc comments for
// exported functions and methods.
func (m *styleMiner) visitFuncDecl(d *ast.FuncDecl, info *types.Info) {
	if d == nil || d.Name == nil || !d.Name.IsExported() {
		return
	}
	m.exportedFuncs++
	m.exportedIdents++
	if d.Doc != nil {
		m.documented++
	}
	// The receiver is separate (d.Recv); Params[0] is the first real
	// argument, where the ctx convention lives.
	if d.Type != nil && d.Type.Params != nil && len(d.Type.Params.List) > 0 {
		if isContextType(d.Type.Params.List[0].Type, info) {
			m.ctxFuncs++
		}
	}
}

// visitGenDecl records doc comments for exported package-level types,
// consts, vars, and exported struct fields.
func (m *styleMiner) visitGenDecl(d *ast.GenDecl) {
	if d == nil {
		return
	}
	for _, spec := range d.Specs {
		switch s := spec.(type) {
		case *ast.TypeSpec:
			if s == nil || s.Name == nil || !s.Name.IsExported() {
				continue
			}
			m.exportedIdents++
			if s.Doc != nil || d.Doc != nil {
				m.documented++
			}
			if st, ok := s.Type.(*ast.StructType); ok {
				m.visitStructFieldsDocs(st)
			}
		case *ast.ValueSpec:
			if s == nil {
				continue
			}
			for _, name := range s.Names {
				if name == nil || !name.IsExported() {
					continue
				}
				m.exportedIdents++
				if s.Doc != nil || d.Doc != nil {
					m.documented++
				}
			}
		}
	}
}

// visitStructFieldsDocs counts exported struct fields and their doc
// comments (go/doc shows trailing line comments as field docs).
func (m *styleMiner) visitStructFieldsDocs(st *ast.StructType) {
	if st == nil || st.Fields == nil {
		return
	}
	for _, field := range st.Fields.List {
		if field == nil {
			continue
		}
		for _, name := range field.Names {
			if name == nil || !name.IsExported() {
				continue
			}
			m.exportedIdents++
			if field.Doc != nil || field.Comment != nil {
				m.documented++
			}
		}
	}
}

// visitStructType aggregates struct tag keys across the module.
func (m *styleMiner) visitStructType(st *ast.StructType) {
	if st == nil || st.Fields == nil {
		return
	}
	for _, field := range st.Fields.List {
		if field == nil || field.Tag == nil {
			continue
		}
		for _, key := range structTagKeys(field.Tag.Value) {
			m.tagCounts[key]++
		}
	}
}

// visitCallExpr classifies error creation/wrapping calls using
// type-checked package paths, so import aliases resolve correctly.
func (m *styleMiner) visitCallExpr(c *ast.CallExpr, info *types.Info) {
	if c == nil || info == nil {
		return
	}
	sel, ok := c.Fun.(*ast.SelectorExpr)
	if !ok || sel == nil || sel.Sel == nil {
		return
	}
	fn, ok := info.Uses[sel.Sel].(*types.Func)
	if !ok || fn == nil || fn.Pkg() == nil {
		return
	}
	switch {
	case fn.Pkg().Path() == "fmt" && fn.Name() == "Errorf":
		if callWraps(c) {
			m.fmtWrap++
		} else {
			m.stdErr++
		}
	case fn.Pkg().Path() == "errors" && (fn.Name() == "New" || fn.Name() == "Join"):
		m.stdErr++
	case fn.Pkg().Path() == "github.com/pkg/errors":
		m.pkgErrors++
	}
}

// style assembles the final RepoStyle from the observations.
func (m *styleMiner) style() *RepoStyle {
	tags := make([]string, 0, len(m.tagCounts))
	for key := range m.tagCounts {
		tags = append(tags, key)
	}
	sort.Slice(tags, func(i, j int) bool {
		if m.tagCounts[tags[i]] != m.tagCounts[tags[j]] {
			return m.tagCounts[tags[i]] > m.tagCounts[tags[j]]
		}
		return tags[i] < tags[j]
	})
	var density float64
	if m.exportedIdents > 0 {
		density = float64(m.documented) / float64(m.exportedIdents)
	}
	var usesCtx bool
	if m.exportedFuncs > 0 {
		usesCtx = float64(m.ctxFuncs)/float64(m.exportedFuncs) >= 0.5
	}
	return &RepoStyle{
		ErrorHandlingStyle: m.errorStyle(),
		CommonStructTags:   tags,
		UsesContext:        usesCtx,
		DocCommentDensity:  density,
	}
}

// errorStyle picks the winning style by plurality; ties prefer the
// stdlib-only style, so %w or pkg/errors are only recommended when
// they are the clear habit.
func (m *styleMiner) errorStyle() string {
	best, style := m.stdErr, ErrorStyleStandard
	if m.fmtWrap > best {
		best, style = m.fmtWrap, ErrorStyleFmtWrap
	}
	if m.pkgErrors > best {
		best, style = m.pkgErrors, ErrorStylePkgErrors
	}
	return style
}

// isContextType reports whether e denotes context.Context.
func isContextType(e ast.Expr, info *types.Info) bool {
	if e == nil || info == nil {
		return false
	}
	named, ok := info.TypeOf(e).(*types.Named)
	if !ok || named == nil || named.Obj() == nil || named.Obj().Pkg() == nil {
		return false
	}
	return named.Obj().Pkg().Path() == "context" && named.Obj().Name() == "Context"
}

// callWraps reports whether a fmt.Errorf call wraps with %w, based on
// a string-literal format argument.
func callWraps(c *ast.CallExpr) bool {
	if c == nil || len(c.Args) == 0 {
		return false
	}
	lit, ok := c.Args[0].(*ast.BasicLit)
	if !ok || lit.Kind != token.STRING {
		return false
	}
	s, err := strconv.Unquote(lit.Value)
	if err != nil {
		return false
	}
	return strings.Contains(s, "%w")
}

// structTagKeys parses the keys of a raw struct tag literal,
// e.g. `json:"name" db:"id"` -> ["json", "db"].
func structTagKeys(tag string) []string {
	unquoted, err := strconv.Unquote(tag)
	if err != nil {
		return nil
	}
	var keys []string
	for {
		unquoted = strings.TrimLeft(unquoted, " ")
		if unquoted == "" {
			break
		}
		i := strings.Index(unquoted, ":")
		if i <= 0 {
			break
		}
		key := unquoted[:i]
		rest := unquoted[i+1:]
		if !strings.HasPrefix(rest, "\"") {
			break
		}
		j := 1
		for j < len(rest) {
			if rest[j] == '\\' {
				j += 2
				continue
			}
			if rest[j] == '"' {
				break
			}
			j++
		}
		if j >= len(rest) {
			break
		}
		keys = append(keys, key)
		unquoted = rest[j+1:]
	}
	return keys
}

/*
Runnable example: mine the style of a target directory.

package main

import (
	"fmt"
	"log"

	"github.com/golangast/gollemer/pkg/ast"
)

func main() {
	codeCtx, err := ast.LoadPackageContext("./myproject")
	if err != nil {
		log.Fatal(err)
	}
	style, err := ast.InferRepositoryStyle(codeCtx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("error style: %s\n", style.ErrorHandlingStyle)
	fmt.Printf("struct tags: %v\n", style.CommonStructTags)
	fmt.Printf("uses context: %v\n", style.UsesContext)
	fmt.Printf("doc density:  %.2f\n", style.DocCommentDensity)
	fmt.Println("--- prompt guidelines ---")
	fmt.Print(style.SystemPromptGuidelines())
}
*/
