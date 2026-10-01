// Package ast provides type-resolved codebase metadata for Go code
// intelligence. It upgrades raw go/parser inspection to full type
// resolution using golang.org/x/tools/go/packages and go/types,
// producing a JSON-serializable CodebaseContext suitable as dynamic
// context for code generation.
//
// The intermediate representation keeps only what a generator needs:
// package identity, imports, struct shapes (fields, resolved types, JSON
// tags, doc comments) and interface shapes (method signatures). Method
// bodies, function implementations, and unexported locals are
// intentionally out of scope.
//
// NOTE ON DEPENDENCIES: this package requires golang.org/x/tools
// (the go/packages driver). That is an external module, not the
// standard library:
//
//	go get golang.org/x/tools
//
// The go/types resolution itself is standard library (go/types); only
// the package loading driver is external.
package ast

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/go/packages"
)

// FieldMeta describes a single struct field with its fully resolved type.
type FieldMeta struct {
	Name       string `json:"name"`
	TypeString string `json:"typeString"`
	IsExported bool   `json:"isExported"`
	JSONTag    string `json:"jsonTag,omitempty"`
	DocComment string `json:"docComment,omitempty"`
}

// StructMeta describes a named struct type declared in the package.
type StructMeta struct {
	Name       string      `json:"name"`
	DocComment string      `json:"docComment,omitempty"`
	Fields     []FieldMeta `json:"fields"`
}

// MethodMeta describes one interface method signature. Parameters and
// Results are rendered as "name type" when the parameter is named,
// otherwise as the bare type string.
type MethodMeta struct {
	Name       string   `json:"name"`
	Parameters []string `json:"parameters"`
	Results    []string `json:"results"`
	DocComment string   `json:"docComment,omitempty"`
}

// InterfaceMeta describes a named interface type declared in the package.
type InterfaceMeta struct {
	Name       string       `json:"name"`
	DocComment string       `json:"docComment,omitempty"`
	Methods    []MethodMeta `json:"methods"`
}

// CodebaseContext is the JSON-serializable intermediate representation
// of one loaded Go package. Imports maps import path to the local
// package name used in source (honoring explicit aliases).
type CodebaseContext struct {
	PackageName string            `json:"packageName"`
	ModulePath  string            `json:"modulePath,omitempty"`
	Imports     map[string]string `json:"imports"`
	Structs     []StructMeta      `json:"structs"`
	Interfaces  []InterfaceMeta   `json:"interfaces"`
	// SourceDir is the absolute directory the context was loaded from.
	// It is a local load hint (used by impact analysis to reload the
	// module), not portable metadata; omitted when empty.
	SourceDir string `json:"sourceDir,omitempty"`
}

// LoadPackageContext loads the Go package in dir with full type
// information and extracts its struct and interface metadata.
//
// The pattern is "." (exactly the package in dir); call once per
// directory for multi-package codebases. Package-level build or
// type-checking errors are collected from pkg.Errors and returned as a
// single descriptive error — no partial context is returned, so callers
// never silently generate from a broken type graph.
func LoadPackageContext(dir string) (*CodebaseContext, error) {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("ast: resolve dir %q: %w", dir, err)
	}
	cfg := &packages.Config{
		Mode: packages.NeedName |
			packages.NeedFiles |
			packages.NeedCompiledGoFiles |
			packages.NeedImports |
			packages.NeedTypes |
			packages.NeedTypesInfo |
			packages.NeedSyntax |
			packages.NeedModule,
		Dir: absDir,
	}
	pkgs, err := packages.Load(cfg, ".")
	if err != nil {
		return nil, fmt.Errorf("ast: packages.Load(%q): %w", dir, err)
	}
	if len(pkgs) == 0 {
		return nil, fmt.Errorf("ast: no packages found in %q", dir)
	}

	var errMsgs []string
	for _, p := range pkgs {
		if p == nil {
			continue
		}
		for _, e := range p.Errors {
			errMsgs = append(errMsgs, e.Msg)
		}
	}
	if len(errMsgs) > 0 {
		return nil, fmt.Errorf("ast: package load errors in %q: %s",
			dir, strings.Join(errMsgs, "; "))
	}

	pkg := pkgs[0]
	if pkg.Types == nil || pkg.TypesInfo == nil {
		return nil, fmt.Errorf("ast: type information unavailable for %q", dir)
	}

	aliases := importAliases(pkg.Syntax)
	for path, imp := range pkg.Imports {
		if imp == nil {
			continue
		}
		if _, ok := aliases[path]; !ok && imp.Name != "" {
			aliases[path] = imp.Name
		}
	}
	qualifier := makeQualifier(pkg.Types, aliases)
	docs := docComments(pkg.Syntax)

	var modulePath string
	if pkg.Module != nil {
		modulePath = pkg.Module.Path
	}

	ctx := &CodebaseContext{
		PackageName: pkg.Name,
		ModulePath:  modulePath,
		Imports:     aliases,
		Structs:     []StructMeta{},
		Interfaces:  []InterfaceMeta{},
		SourceDir:   absDir,
	}

	// TypesInfo.Defs records every defining identifier in the package,
	// which is exactly where named type declarations live. (TypesInfo.Uses
	// records references to those declarations; declaration metadata needs
	// only Defs.)
	for _, obj := range pkg.TypesInfo.Defs {
		tn, ok := obj.(*types.TypeName)
		if !ok || tn == nil {
			continue
		}
		// Only package-level declarations; skip local types, method
		// receivers' scopes, and any other nested scopes.
		if tn.Parent() != pkg.Types.Scope() {
			continue
		}
		named, ok := tn.Type().(*types.Named)
		if !ok || named == nil {
			continue
		}
		switch underlying := named.Underlying().(type) {
		case *types.Struct:
			ctx.Structs = append(ctx.Structs,
				extractStruct(tn, underlying, docs, qualifier))
		case *types.Interface:
			ctx.Interfaces = append(ctx.Interfaces,
				extractInterface(tn, underlying, docs, qualifier))
		}
	}

	// Defs is a map: sort for deterministic output.
	sort.Slice(ctx.Structs, func(i, j int) bool {
		return ctx.Structs[i].Name < ctx.Structs[j].Name
	})
	sort.Slice(ctx.Interfaces, func(i, j int) bool {
		return ctx.Interfaces[i].Name < ctx.Interfaces[j].Name
	})

	return ctx, nil
}

// extractStruct converts one *types.Struct declaration into metadata,
// resolving each field's type string (with pointer indicators and
// package qualifiers), JSON tag, and doc comment.
func extractStruct(tn *types.TypeName, st *types.Struct,
	docs map[token.Pos]string, qf types.Qualifier) StructMeta {

	sm := StructMeta{
		Name:       tn.Name(),
		DocComment: docs[tn.Pos()],
		Fields:     []FieldMeta{},
	}
	for i := 0; i < st.NumFields(); i++ {
		f := st.Field(i)
		if f == nil {
			continue
		}
		var typeStr string
		if ft := f.Type(); ft != nil {
			typeStr = types.TypeString(ft, qf)
		}
		sm.Fields = append(sm.Fields, FieldMeta{
			Name:       f.Name(),
			TypeString: typeStr,
			IsExported: f.Exported(),
			JSONTag:    reflect.StructTag(st.Tag(i)).Get("json"),
			DocComment: docs[f.Pos()],
		})
	}
	return sm
}

// extractInterface converts one *types.Interface declaration into
// metadata, rendering each method's full signature.
func extractInterface(tn *types.TypeName, iface *types.Interface,
	docs map[token.Pos]string, qf types.Qualifier) InterfaceMeta {

	im := InterfaceMeta{
		Name:       tn.Name(),
		DocComment: docs[tn.Pos()],
		Methods:    []MethodMeta{},
	}
	// NumMethods includes methods promoted from embedded interfaces.
	for i := 0; i < iface.NumMethods(); i++ {
		m := iface.Method(i)
		if m == nil {
			continue
		}
		sig, ok := m.Type().(*types.Signature)
		if !ok || sig == nil {
			continue
		}
		params, results := signatureStrings(sig, qf)
		im.Methods = append(im.Methods, MethodMeta{
			Name:       m.Name(),
			Parameters: params,
			Results:    results,
			DocComment: docs[m.Pos()],
		})
	}
	return im
}

// signatureStrings renders a signature's parameter and result lists.
// The final parameter of a variadic signature is rendered as "...T"
// rather than "[]T", matching how it is written in source.
func signatureStrings(sig *types.Signature, qf types.Qualifier) (params, results []string) {
	params = tupleStrings(sig.Params(), qf)
	if sig.Variadic() && sig.Params() != nil && sig.Params().Len() > 0 {
		last := sig.Params().At(sig.Params().Len() - 1)
		if last != nil {
			if slice, ok := last.Type().(*types.Slice); ok && slice != nil {
				var elem string
				if et := slice.Elem(); et != nil {
					elem = types.TypeString(et, qf)
				}
				name := ""
				if n := last.Name(); n != "" {
					name = n + " "
				}
				params[len(params)-1] = name + "..." + strings.TrimSpace(elem)
			}
		}
	}
	results = tupleStrings(sig.Results(), qf)
	return params, results
}

// tupleStrings renders a parameter or result tuple as "name type" for
// named entries and the bare type string for unnamed ones.
func tupleStrings(t *types.Tuple, qf types.Qualifier) []string {
	out := []string{}
	if t == nil {
		return out
	}
	for i := 0; i < t.Len(); i++ {
		v := t.At(i)
		if v == nil {
			continue
		}
		var typeStr string
		if vt := v.Type(); vt != nil {
			typeStr = types.TypeString(vt, qf)
		}
		if name := v.Name(); name != "" {
			out = append(out, name+" "+strings.TrimSpace(typeStr))
		} else {
			out = append(out, typeStr)
		}
	}
	return out
}

// makeQualifier returns a types.Qualifier that renders types from the
// loaded package unqualified and types from other packages with the
// local import name used in source (honoring explicit aliases such as
// `import js "encoding/json"`).
func makeQualifier(pkg *types.Package, aliases map[string]string) types.Qualifier {
	return func(other *types.Package) string {
		if other == nil || other == pkg {
			return ""
		}
		if name, ok := aliases[other.Path()]; ok && name != "" {
			return name
		}
		return other.Name()
	}
}

// importAliases maps import path to the local package name written in
// source, honoring explicit aliases. Blank ("_") and dot (".") imports
// have no usable qualifier and are skipped.
func importAliases(files []*ast.File) map[string]string {
	aliases := make(map[string]string)
	for _, f := range files {
		if f == nil {
			continue
		}
		for _, imp := range f.Imports {
			if imp == nil || imp.Path == nil {
				continue
			}
			path, err := strconv.Unquote(imp.Path.Value)
			if err != nil || path == "" {
				continue
			}
			var name string
			if imp.Name != nil {
				name = imp.Name.Name
			}
			if name == "" {
				name = path
				if i := strings.LastIndex(path, "/"); i >= 0 {
					name = path[i+1:]
				}
			}
			if name == "_" || name == "." {
				continue
			}
			aliases[path] = name
		}
	}
	return aliases
}

// docComments maps the position of each declared type name, struct field
// name, and interface method name to its doc comment text. Positions are
// token.Pos values, which match the Pos() reported by go/types objects
// for the same identifiers.
func docComments(files []*ast.File) map[token.Pos]string {
	docs := make(map[token.Pos]string)
	for _, f := range files {
		if f == nil {
			continue
		}
		ast.Inspect(f, func(n ast.Node) bool {
			switch n := n.(type) {
			case *ast.GenDecl:
				// A lone `// doc\ntype T struct{}` attaches the comment
				// to the GenDecl, not the TypeSpec. Inherit it here; the
				// TypeSpec case below overwrites with the spec's own doc
				// when one is present (pre-order: parent first).
				if n.Tok == token.TYPE && n.Doc != nil {
					if text := commentText(n.Doc); text != "" {
						for _, spec := range n.Specs {
							ts, ok := spec.(*ast.TypeSpec)
							if !ok || ts == nil || ts.Name == nil {
								continue
							}
							if _, exists := docs[ts.Name.Pos()]; !exists {
								docs[ts.Name.Pos()] = text
							}
						}
					}
				}
			case *ast.TypeSpec:
				if n.Name != nil {
					if text := commentText(n.Doc); text != "" {
						docs[n.Name.Pos()] = text
					}
				}
			case *ast.Field:
				text := commentText(n.Doc)
				if text == "" {
					text = commentText(n.Comment)
				}
				if text == "" {
					return true
				}
				if len(n.Names) == 0 {
					// Embedded field: key by the type expression's
					// position, which is what go/types reports for the
					// field object.
					docs[n.Type.Pos()] = text
				} else {
					for _, name := range n.Names {
						docs[name.Pos()] = text
					}
				}
			}
			return true
		})
	}
	return docs
}

// commentText returns the trimmed text of a comment group, or "".
func commentText(cg *ast.CommentGroup) string {
	if cg == nil {
		return ""
	}
	return strings.TrimSpace(cg.Text())
}

/*
Example: load a package and print its context as formatted JSON.

	package main

	import (
		"encoding/json"
		"fmt"
		"log"

		resolver "github.com/golangast/gollemer/pkg/ast"
	)

	func main() {
		ctx, err := resolver.LoadPackageContext(".")
		if err != nil {
			log.Fatalf("load: %v", err)
		}
		out, err := json.MarshalIndent(ctx, "", "  ")
		if err != nil {
			log.Fatalf("json: %v", err)
		}
		fmt.Println(string(out))
	}
*/
