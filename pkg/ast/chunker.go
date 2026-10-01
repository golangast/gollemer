// Semantic code chunking for vector embeddings.
//
// ChunkFile splits a parsed Go file into chunks along AST node
// boundaries — one chunk per function, method, struct, or interface —
// instead of arbitrary line windows. Each chunk carries formatted source,
// line bounds, doc comments, and the identifiers it references, so an
// embedding retriever can trace relationships between chunks.
//
// This file is pure AST (go/ast, go/token); it needs no type
// information. One honest limitation follows from that: without
// go/types, a local variable is indistinguishable from a package-level
// reference, so Dependencies may include locals. Pair with
// LoadPackageContext (resolver.go) when precise type edges are needed.
package ast

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/ast"
	"go/format"
	"go/token"
	"go/types"
	"sort"
	"strings"
)

// CodeChunk is one semantically coherent unit of Go source, sized for
// vector embedding.
//
// Kind is one of "struct", "interface", "function", or "method".
// Method chunks keep Kind "method"; the receiver type is attached to
// SymbolName in "Recv.Name" form ("(*Server).Start" for pointer
// receivers, "Server.Start" for value receivers).
//
// Dependencies holds the deduplicated, sorted names of identifiers
// referenced inside the chunk: custom types and functions it relates
// to. Excluded: the chunk's own symbol (a recursive call is a
// self-edge), receiver/parameter/result/field/local definitions, "_",
// and predeclared identifiers. Package qualifiers are kept ("json" in
// "json.Marshal"). Without type information, a shadowing local hides a
// same-named package-level reference.
type CodeChunk struct {
	ID           string   `json:"id"`
	FilePath     string   `json:"filePath"`
	Kind         string   `json:"kind"`
	SymbolName   string   `json:"symbolName"`
	DocComment   string   `json:"docComment,omitempty"`
	CodeContent  string   `json:"codeContent"`
	StartLine    int      `json:"startLine"`
	EndLine      int      `json:"endLine"`
	Dependencies []string `json:"dependencies"`
}

// ChunkFile splits a parsed Go file into semantic code chunks in file
// order. fset must be the token.FileSet the file was parsed with;
// filePath is recorded on every chunk and feeds the chunk ID hash.
func ChunkFile(fset *token.FileSet, file *ast.File, filePath string) ([]CodeChunk, error) {
	if fset == nil {
		return nil, fmt.Errorf("ast: ChunkFile: nil token.FileSet")
	}
	if file == nil {
		return nil, fmt.Errorf("ast: ChunkFile: nil *ast.File")
	}
	if filePath == "" {
		return nil, fmt.Errorf("ast: ChunkFile: empty filePath")
	}

	chunks := []CodeChunk{}
	var firstErr error
	ast.Inspect(file, func(n ast.Node) bool {
		if firstErr != nil {
			return false
		}
		switch n := n.(type) {
		case *ast.FuncDecl:
			c, err := funcChunk(fset, n, filePath)
			if err != nil {
				firstErr = err
				return false
			}
			chunks = append(chunks, c)
			// A body cannot declare another chunk-level symbol.
			return false
		case *ast.GenDecl:
			if n.Tok != token.TYPE {
				return false
			}
			// A lone `// doc\ntype T struct{}` attaches the comment to
			// the GenDecl, not the TypeSpec; inherit it when the spec
			// has none of its own.
			genDoc := commentText(n.Doc)
			for _, spec := range n.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok || ts == nil {
					continue
				}
				c, chunked, err := typeChunk(fset, ts, genDoc, filePath)
				if err != nil {
					firstErr = err
					return false
				}
				if chunked {
					chunks = append(chunks, c)
				}
			}
			return false
		}
		return true
	})
	if firstErr != nil {
		return nil, firstErr
	}
	return chunks, nil
}

// funcChunk builds a "function" or "method" chunk from a top-level
// function declaration.
func funcChunk(fset *token.FileSet, fn *ast.FuncDecl, filePath string) (CodeChunk, error) {
	if fn.Name == nil {
		return CodeChunk{}, fmt.Errorf("ast: FuncDecl with nil name")
	}
	code, err := formatNode(fset, fn)
	if err != nil {
		return CodeChunk{}, err
	}

	kind := "function"
	symbol := fn.Name.Name
	// Names defined by the chunk itself are not dependencies: the
	// function name (a recursive call is a self-edge), the receiver
	// name and base type (the method's owner), parameter/result names,
	// and body locals. Name-based, so a shadowing local hides a
	// same-named package-level reference — rare and documented.
	excludeNames := map[string]bool{fn.Name.Name: true}
	if fn.Recv != nil && len(fn.Recv.List) > 0 && fn.Recv.List[0] != nil {
		kind = "method"
		symbol = methodSymbol(fn)
		fieldNamesInto(fn.Recv, excludeNames)
		if base := baseTypeName(fn.Recv.List[0].Type); base != "" {
			excludeNames[base] = true
		}
	}
	if fn.Type != nil {
		fieldNamesInto(fn.Type.Params, excludeNames)
		fieldNamesInto(fn.Type.Results, excludeNames)
	}
	for name := range localNames(fn) {
		excludeNames[name] = true
	}

	// Definitions are not dependencies: skip the function name,
	// receiver names, and parameter/result names by identity, so a
	// same-named type reference elsewhere still counts.
	skip := map[*ast.Ident]bool{fn.Name: true}
	if fn.Recv != nil {
		skipFieldIdents(skip, fn.Recv)
	}
	if fn.Type != nil {
		skipFieldIdents(skip, fn.Type.Params)
		skipFieldIdents(skip, fn.Type.Results)
	}

	c := CodeChunk{
		FilePath:     filePath,
		Kind:         kind,
		SymbolName:   symbol,
		DocComment:   commentText(fn.Doc),
		CodeContent:  code,
		StartLine:    fset.Position(fn.Pos()).Line,
		EndLine:      fset.Position(fn.End()).Line,
		Dependencies: collectIdents(fn, skip, excludeNames),
	}
	c.ID = chunkID(filePath, kind, symbol, code)
	return c, nil
}

// typeChunk builds a "struct" or "interface" chunk from a type spec.
// genDoc is the enclosing GenDecl's doc comment, used when the spec has
// none of its own. It reports chunked=false for type specs that are
// neither structs nor interfaces (e.g. `type Name string`), which carry
// no chunkable shape.
func typeChunk(fset *token.FileSet, ts *ast.TypeSpec, genDoc, filePath string) (CodeChunk, bool, error) {
	if ts.Name == nil {
		return CodeChunk{}, false, fmt.Errorf("ast: TypeSpec with nil name")
	}
	var kind string
	switch ts.Type.(type) {
	case *ast.StructType:
		kind = "struct"
	case *ast.InterfaceType:
		kind = "interface"
	default:
		return CodeChunk{}, false, nil
	}

	// Render the spec as a standalone `type Name ...` declaration so the
	// chunk is valid, readable Go on its own.
	wrapped := &ast.GenDecl{Tok: token.TYPE, Specs: []ast.Spec{ts}}
	code, err := formatNode(fset, wrapped)
	if err != nil {
		return CodeChunk{}, false, err
	}

	symbol := ts.Name.Name
	doc := commentText(ts.Doc)
	if doc == "" {
		doc = genDoc
	}
	skip := map[*ast.Ident]bool{ts.Name: true}
	switch t := ts.Type.(type) {
	case *ast.StructType:
		if t.Fields != nil {
			skipFieldIdents(skip, t.Fields)
		}
	case *ast.InterfaceType:
		if t.Methods != nil {
			skipFieldIdents(skip, t.Methods)
		}
	}

	c := CodeChunk{
		FilePath:     filePath,
		Kind:         kind,
		SymbolName:   symbol,
		DocComment:   doc,
		CodeContent:  code,
		StartLine:    fset.Position(ts.Pos()).Line,
		EndLine:      fset.Position(ts.End()).Line,
		Dependencies: collectIdents(ts, skip, map[string]bool{symbol: true}),
	}
	c.ID = chunkID(filePath, kind, symbol, code)
	return c, true, nil
}

// methodSymbol renders "(*Server).Start" for pointer receivers and
// "Server.Start" for value receivers (including generic receivers like
// "Box[T].Get", where the base name is used).
func methodSymbol(fn *ast.FuncDecl) string {
	if fn.Recv == nil || len(fn.Recv.List) == 0 || fn.Recv.List[0] == nil {
		return fn.Name.Name
	}
	base := baseTypeName(fn.Recv.List[0].Type)
	if base == "" {
		return fn.Name.Name
	}
	if _, ok := fn.Recv.List[0].Type.(*ast.StarExpr); ok {
		return "(*" + base + ")." + fn.Name.Name
	}
	return base + "." + fn.Name.Name
}

// baseTypeName strips pointer and generic type arguments to the bare
// type name: "*Server" -> "Server", "Box[T]" -> "Box".
func baseTypeName(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		if t == nil {
			return ""
		}
		return t.Name
	case *ast.StarExpr:
		if t == nil {
			return ""
		}
		return baseTypeName(t.X)
	case *ast.IndexExpr:
		if t == nil {
			return ""
		}
		return baseTypeName(t.X)
	case *ast.IndexListExpr:
		if t == nil {
			return ""
		}
		return baseTypeName(t.X)
	default:
		return ""
	}
}

// skipFieldIdents marks every defined name in a field list (receiver
// names, parameter names, struct field names, interface method names)
// so they are not mistaken for references. It recurses into nested
// function types, whose parameter and result names are definitions too
// (e.g. interface method signatures).
func skipFieldIdents(skip map[*ast.Ident]bool, fl *ast.FieldList) {
	if fl == nil {
		return
	}
	for _, f := range fl.List {
		if f == nil {
			continue
		}
		for _, name := range f.Names {
			if name != nil {
				skip[name] = true
			}
		}
		if ft, ok := f.Type.(*ast.FuncType); ok && ft != nil {
			skipFieldIdents(skip, ft.Params)
			skipFieldIdents(skip, ft.Results)
		}
	}
}

// fieldNamesInto collects every defined name in a field list by name,
// recursing into nested function types.
func fieldNamesInto(fl *ast.FieldList, into map[string]bool) {
	if fl == nil {
		return
	}
	for _, f := range fl.List {
		if f == nil {
			continue
		}
		for _, name := range f.Names {
			if name != nil && name.Name != "" {
				into[name.Name] = true
			}
		}
		if ft, ok := f.Type.(*ast.FuncType); ok && ft != nil {
			fieldNamesInto(ft.Params, into)
			fieldNamesInto(ft.Results, into)
		}
	}
}

// localNames gathers names defined inside a function body — short
// variable declarations (`n := 0`), range keys/values, and `var` / local
// `type` declarations — so uses of locals are not reported as
// dependencies.
func localNames(fn *ast.FuncDecl) map[string]bool {
	names := make(map[string]bool)
	if fn.Body == nil {
		return names
	}
	addName := func(id *ast.Ident) {
		if id != nil && id.Name != "" {
			names[id.Name] = true
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.AssignStmt:
			if n.Tok == token.DEFINE {
				for _, lhs := range n.Lhs {
					if id, ok := lhs.(*ast.Ident); ok {
						addName(id)
					}
				}
			}
		case *ast.RangeStmt:
			for _, e := range []ast.Expr{n.Key, n.Value} {
				if id, ok := e.(*ast.Ident); ok {
					addName(id)
				}
			}
		case *ast.DeclStmt:
			gd, ok := n.Decl.(*ast.GenDecl)
			if !ok || gd == nil {
				break
			}
			for _, spec := range gd.Specs {
				switch s := spec.(type) {
				case *ast.ValueSpec:
					if s == nil {
						break
					}
					for _, name := range s.Names {
						addName(name)
					}
				case *ast.TypeSpec:
					if s != nil {
						addName(s.Name)
					}
				}
			}
		}
		return true
	})
	return names
}

// collectIdents gathers the deduplicated, sorted names of identifiers
// referenced under node, excluding definitions (skip, by identity),
// the chunk's own names, "_" and predeclared identifiers.
func collectIdents(node ast.Node, skip map[*ast.Ident]bool, excludeNames map[string]bool) []string {
	seen := make(map[string]bool)
	var names []string
	ast.Inspect(node, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok || id == nil || skip[id] {
			return true
		}
		name := id.Name
		if name == "" || name == "_" || excludeNames[name] || isPredeclared(name) || seen[name] {
			return true
		}
		seen[name] = true
		names = append(names, name)
		return true
	})
	sort.Strings(names)
	return names
}

// isPredeclared reports whether name is a predeclared Go identifier
// (types, constants, or builtin functions).
func isPredeclared(name string) bool {
	return types.Universe.Lookup(name) != nil
}

// formatNode renders one AST node back to formatted Go source.
func formatNode(fset *token.FileSet, node ast.Node) (string, error) {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, node); err != nil {
		return "", fmt.Errorf("ast: format chunk: %w", err)
	}
	return strings.TrimSpace(buf.String()), nil
}

// chunkID is a deterministic 128-bit content hash: stable across runs
// and across line shifts, unique per file/kind/symbol/content.
func chunkID(filePath, kind, symbol, code string) string {
	h := sha256.New()
	h.Write([]byte(filePath))
	h.Write([]byte{0})
	h.Write([]byte(kind))
	h.Write([]byte{0})
	h.Write([]byte(symbol))
	h.Write([]byte{0})
	h.Write([]byte(code))
	return hex.EncodeToString(h.Sum(nil))[:32]
}

/*
Example: chunk a sample Go file and print each chunk's metadata.

	package main

	import (
		"fmt"
		"go/parser"
		"go/token"
		"log"

		chunker "github.com/golangast/gollemer/pkg/ast"
	)

	const sample = `package sample

	// Server serves requests.
	type Server struct {
		Addr string
	}

	// Start starts the server.
	func (s *Server) Start() error { return nil }

	// Hello says hello.
	func Hello(name string) string { return "hi " + name }
	`

	func main() {
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, "sample.go", sample, parser.ParseComments)
		if err != nil {
			log.Fatal(err)
		}
		// Keep the parsed *ast.File if you need it; ChunkFile only
		// needs the file and its file set.
		chunks, err := chunker.ChunkFile(fset, f, "sample.go")
		if err != nil {
			log.Fatal(err)
		}
		for _, c := range chunks {
			fmt.Printf("%-9s %-18s id=%s lines=%d-%d deps=%v\n",
				c.Kind, c.SymbolName, c.ID[:8], c.StartLine, c.EndLine, c.Dependencies)
		}
	}
*/
