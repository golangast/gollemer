// Package symbol implements a high-precision symbol reference graph for Go codebases.
// It provides LSIF/SCIP-like capabilities: Find Definitions, Find References,
// Find Implementations, and symbol-level dependency tracing.
package symbol

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// SymbolKind classifies a symbol in the codebase.
type SymbolKind int

const (
	KindUnknown SymbolKind = iota
	KindFunction
	KindMethod
	KindStruct
	KindInterface
	KindVariable
	KindConstant
	KindType
	KindField
	KindParameter
	KindPackage
	KindImport
)

func (k SymbolKind) String() string {
	switch k {
	case KindFunction:
		return "function"
	case KindMethod:
		return "method"
	case KindStruct:
		return "struct"
	case KindInterface:
		return "interface"
	case KindVariable:
		return "variable"
	case KindConstant:
		return "constant"
	case KindType:
		return "type"
	case KindField:
		return "field"
	case KindParameter:
		return "parameter"
	case KindPackage:
		return "package"
	case KindImport:
		return "import"
	default:
		return "unknown"
	}
}

// Symbol represents a single symbol in the codebase.
type Symbol struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	Kind       SymbolKind        `json:"kind"`
	File       string            `json:"file"`
	Line       int               `json:"line"`
	Column     int               `json:"column"`
	EndLine    int               `json:"end_line"`
	EndColumn  int               `json:"end_column"`
	Package    string            `json:"package"`
	Receiver   string            `json:"receiver,omitempty"`  // For methods
	Signature  string            `json:"signature,omitempty"` // Full type signature
	DocComment string            `json:"doc_comment,omitempty"`
	Exported   bool              `json:"exported"`
	References []*Reference      `json:"references,omitempty"`
	Children   []*Symbol         `json:"children,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// Reference represents a usage of a symbol at a specific location.
type Reference struct {
	File    string `json:"file"`
	Line    int    `json:"line"`
	Column  int    `json:"column"`
	EndLine int    `json:"end_line"`
	EndCol  int    `json:"end_col"`
	Context string `json:"context"` // Surrounding code snippet
	IsWrite bool   `json:"is_write"`
	IsTest  bool   `json:"is_test"`
}

// SymbolGraph is the complete index of all symbols and their relationships.
type SymbolGraph struct {
	mu      sync.RWMutex
	symbols map[string]*Symbol   // symbol ID -> Symbol
	byFile  map[string][]*Symbol // file path -> symbols
	byName  map[string][]*Symbol // symbol name -> symbols (for quick lookup)
	byPkg   map[string][]*Symbol // package -> symbols
	imports map[string][]string  // file -> imported packages
	edges   []*Edge              // symbol-to-symbol relationships
	rootDir string
}

// EdgeType describes the relationship between two symbols.
type EdgeType int

const (
	EdgeCalls      EdgeType = iota // function A calls function B
	EdgeImplements                 // struct A implements interface B
	EdgeExtends                    // type A extends type B (embedding)
	EdgeContains                   // file/package A contains symbol B
	EdgeReferences                 // symbol A references symbol B
	EdgeImports                    // file A imports package B
)

// Edge represents a relationship between two symbols.
type Edge struct {
	SourceID string   `json:"source_id"`
	TargetID string   `json:"target_id"`
	Type     EdgeType `json:"type"`
	File     string   `json:"file,omitempty"`
	Line     int      `json:"line,omitempty"`
}

// NewSymbolGraph creates an empty symbol graph.
func NewSymbolGraph(rootDir string) *SymbolGraph {
	return &SymbolGraph{
		symbols: make(map[string]*Symbol),
		byFile:  make(map[string][]*Symbol),
		byName:  make(map[string][]*Symbol),
		byPkg:   make(map[string][]*Symbol),
		imports: make(map[string][]string),
		edges:   make([]*Edge, 0),
		rootDir: rootDir,
	}
}

// IndexWorkspace scans all .go files in the workspace and builds the symbol graph.
// maxFiles limits the number of files to index (0 = unlimited).
func (sg *SymbolGraph) IndexWorkspace() error {
	return sg.IndexWorkspaceWithLimit(0)
}

// IndexWorkspaceWithLimit indexes the workspace with an optional file limit.
// maxFiles: maximum number of files to index (0 = unlimited).
func (sg *SymbolGraph) IndexWorkspaceWithLimit(maxFiles int) error {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	var goFiles []string
	err := filepath.Walk(sg.rootDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			// Skip hidden directories and vendor
			if strings.HasPrefix(info.Name(), ".") || info.Name() == "vendor" || info.Name() == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasSuffix(path, ".go") && !strings.HasSuffix(path, "_test.go") {
			goFiles = append(goFiles, path)
			if maxFiles > 0 && len(goFiles) >= maxFiles {
				return filepath.SkipDir
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("walk workspace: %w", err)
	}

	if maxFiles > 0 && len(goFiles) > maxFiles {
		goFiles = goFiles[:maxFiles]
	}

	fmt.Fprintf(os.Stderr, "   Indexing %d Go files...\n", len(goFiles))

	// First pass: parse all files and collect top-level symbols
	for i, file := range goFiles {
		if err := sg.indexFile(file); err != nil {
			fmt.Fprintf(os.Stderr, "   warning: indexing %s: %v\n", file, err)
		}
		if (i+1)%50 == 0 {
			fmt.Fprintf(os.Stderr, "   Progress: %d/%d files indexed\n", i+1, len(goFiles))
		}
	}

	fmt.Fprintf(os.Stderr, "   Resolving cross-file references...\n")

	// Second pass: resolve cross-file references
	sg.resolveReferences()

	return nil
}

// indexFile parses a single Go file and extracts all symbols.
func (sg *SymbolGraph) indexFile(filePath string) error {
	relPath, err := filepath.Rel(sg.rootDir, filePath)
	if err != nil {
		relPath = filePath
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return fmt.Errorf("parse %s: %w", filePath, err)
	}

	pkgName := f.Name.Name
	if f.Name.Name != "" {
		sg.byPkg[pkgName] = append(sg.byPkg[pkgName], &Symbol{
			ID:   "pkg:" + pkgName,
			Name: pkgName,
			Kind: KindPackage,
			File: relPath,
		})
	}

	// Record imports
	for _, imp := range f.Imports {
		impPath := strings.Trim(imp.Path.Value, "\"")
		sg.imports[relPath] = append(sg.imports[relPath], impPath)

		// Create import symbol
		impSym := &Symbol{
			ID:   fmt.Sprintf("import:%s:%s", relPath, impPath),
			Name: impPath,
			Kind: KindImport,
			File: relPath,
		}
		if imp.Name != nil {
			impSym.Name = imp.Name.Name + " " + impPath
		}
		sg.addSymbol(impSym)
	}

	// Extract top-level declarations
	for _, decl := range f.Decls {
		switch d := decl.(type) {
		case *ast.GenDecl:
			sg.indexGenDecl(fset, d, relPath, pkgName)
		case *ast.FuncDecl:
			sg.indexFuncDecl(fset, d, relPath, pkgName)
		}
	}

	return nil
}

// indexGenDecl indexes general declarations (const, var, type).
func (sg *SymbolGraph) indexGenDecl(fset *token.FileSet, d *ast.GenDecl, file, pkg string) {
	for _, spec := range d.Specs {
		switch s := spec.(type) {
		case *ast.TypeSpec:
			pos := fset.Position(s.Pos())
			endPos := fset.Position(s.End())
			kind := KindType
			var docComment string
			if s.Doc != nil {
				docComment = s.Doc.Text()
			}

			// Check if it's a struct or interface
			switch s.Type.(type) {
			case *ast.StructType:
				kind = KindStruct
			case *ast.InterfaceType:
				kind = KindInterface
			}

			sym := &Symbol{
				ID:         fmt.Sprintf("%s:%s:%s", pkg, s.Name.Name, file),
				Name:       s.Name.Name,
				Kind:       kind,
				File:       file,
				Line:       pos.Line,
				Column:     pos.Column,
				EndLine:    endPos.Line,
				EndColumn:  endPos.Column,
				Package:    pkg,
				Exported:   s.Name.IsExported(),
				DocComment: docComment,
				Children:   make([]*Symbol, 0),
			}

			// Index struct fields
			if st, ok := s.Type.(*ast.StructType); ok {
				for _, field := range st.Fields.List {
					for _, name := range field.Names {
						fieldPos := fset.Position(name.Pos())
						fieldSym := &Symbol{
							ID:       fmt.Sprintf("%s:%s.%s:%s", pkg, s.Name.Name, name.Name, file),
							Name:     name.Name,
							Kind:     KindField,
							File:     file,
							Line:     fieldPos.Line,
							Column:   fieldPos.Column,
							Package:  pkg,
							Exported: name.IsExported(),
						}
						sym.Children = append(sym.Children, fieldSym)
						sg.addSymbol(fieldSym)
					}
				}
			}

			// Index interface methods
			if it, ok := s.Type.(*ast.InterfaceType); ok {
				for _, method := range it.Methods.List {
					for _, name := range method.Names {
						methodPos := fset.Position(name.Pos())
						methodSym := &Symbol{
							ID:       fmt.Sprintf("%s:%s.%s:%s", pkg, s.Name.Name, name.Name, file),
							Name:     name.Name,
							Kind:     KindMethod,
							File:     file,
							Line:     methodPos.Line,
							Column:   methodPos.Column,
							Package:  pkg,
							Exported: name.IsExported(),
							Receiver: s.Name.Name,
						}
						sym.Children = append(sym.Children, methodSym)
						sg.addSymbol(methodSym)
					}
				}
			}

			sg.addSymbol(sym)

		case *ast.ValueSpec:
			kind := KindVariable
			if d.Tok == token.CONST {
				kind = KindConstant
			}
			for _, name := range s.Names {
				pos := fset.Position(name.Pos())
				sym := &Symbol{
					ID:       fmt.Sprintf("%s:%s:%s", pkg, name.Name, file),
					Name:     name.Name,
					Kind:     kind,
					File:     file,
					Line:     pos.Line,
					Column:   pos.Column,
					Package:  pkg,
					Exported: name.IsExported(),
				}
				sg.addSymbol(sym)
			}
		}
	}
}

// indexFuncDecl indexes function and method declarations.
func (sg *SymbolGraph) indexFuncDecl(fset *token.FileSet, d *ast.FuncDecl, file, pkg string) {
	pos := fset.Position(d.Pos())
	endPos := fset.Position(d.End())
	kind := KindFunction
	receiver := ""

	if d.Recv != nil && len(d.Recv.List) > 0 {
		kind = KindMethod
		// Extract receiver type name
		recvType := d.Recv.List[0].Type
		switch t := recvType.(type) {
		case *ast.Ident:
			receiver = t.Name
		case *ast.StarExpr:
			if ident, ok := t.X.(*ast.Ident); ok {
				receiver = "*" + ident.Name
			}
		}
	}

	var docComment string
	if d.Doc != nil {
		docComment = d.Doc.Text()
	}

	sym := &Symbol{
		ID:         fmt.Sprintf("%s:%s:%s", pkg, d.Name.Name, file),
		Name:       d.Name.Name,
		Kind:       kind,
		File:       file,
		Line:       pos.Line,
		Column:     pos.Column,
		EndLine:    endPos.Line,
		EndColumn:  endPos.Column,
		Package:    pkg,
		Receiver:   receiver,
		Exported:   d.Name.IsExported(),
		DocComment: docComment,
		Signature:  sg.formatFuncSignature(d),
	}

	sg.addSymbol(sym)
}

// formatFuncSignature creates a human-readable function signature.
func (sg *SymbolGraph) formatFuncSignature(d *ast.FuncDecl) string {
	var parts []string
	parts = append(parts, "func")
	if d.Recv != nil && len(d.Recv.List) > 0 {
		parts = append(parts, "("+types.ExprString(d.Recv.List[0].Type)+")")
	}
	parts = append(parts, d.Name.Name)
	parts = append(parts, types.ExprString(d.Type))
	return strings.Join(parts, " ")
}

// addSymbol adds a symbol to all lookup indices.
func (sg *SymbolGraph) addSymbol(sym *Symbol) {
	sg.symbols[sym.ID] = sym
	sg.byFile[sym.File] = append(sg.byFile[sym.File], sym)
	sg.byName[sym.Name] = append(sg.byName[sym.Name], sym)
}

// resolveReferences performs a second pass to find all symbol usages.
// Only resolves references for symbols that have been indexed, to save memory.
func (sg *SymbolGraph) resolveReferences() {
	// Build a set of known symbol names for quick lookup
	knownNames := make(map[string]bool)
	for _, sym := range sg.symbols {
		knownNames[sym.Name] = true
	}

	for file := range sg.byFile {
		sg.resolveFileReferences(file, knownNames)
	}
}

// resolveFileReferences finds all references in a single file.
func (sg *SymbolGraph) resolveFileReferences(filePath string, knownNames map[string]bool) {
	absPath := filepath.Join(sg.rootDir, filePath)
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, absPath, nil, 0)
	if err != nil {
		return
	}

	isTestFile := strings.HasSuffix(filePath, "_test.go")

	ast.Inspect(f, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.CallExpr:
			// Function/method calls - only resolve if the function name is known
			var funcName string
			switch fun := node.Fun.(type) {
			case *ast.Ident:
				funcName = fun.Name
			case *ast.SelectorExpr:
				funcName = fun.Sel.Name
			}
			if funcName != "" && knownNames[funcName] {
				pos := fset.Position(node.Pos())
				ref := &Reference{
					File:   filePath,
					Line:   pos.Line,
					Column: pos.Column,
					IsTest: isTestFile,
				}
				for _, sym := range sg.byName[funcName] {
					sym.References = append(sym.References, ref)
				}
			}
		case *ast.SelectorExpr:
			// Field/method access - only resolve if the field name is known
			if _, ok := node.X.(*ast.Ident); ok && knownNames[node.Sel.Name] {
				pos := fset.Position(node.Sel.Pos())
				ref := &Reference{
					File:   filePath,
					Line:   pos.Line,
					Column: pos.Column,
					IsTest: isTestFile,
				}
				for _, sym := range sg.byName[node.Sel.Name] {
					sym.References = append(sym.References, ref)
				}
			}
		case *ast.Ident:
			// Direct identifier references - only resolve if the name is known
			if node.Name != "_" && knownNames[node.Name] && !isDeclContext(node, f) {
				pos := fset.Position(node.Pos())
				ref := &Reference{
					File:   filePath,
					Line:   pos.Line,
					Column: pos.Column,
					IsTest: isTestFile,
				}
				for _, sym := range sg.byName[node.Name] {
					sym.References = append(sym.References, ref)
				}
			}
		}
		return true
	})
}

// extractContext gets the surrounding line of code for context.
func (sg *SymbolGraph) extractContext(fset *token.FileSet, f *ast.File, pos token.Pos) string {
	if fset == nil {
		return ""
	}
	position := fset.Position(pos)
	content, err := os.ReadFile(position.Filename)
	if err != nil {
		return ""
	}
	lines := strings.Split(string(content), "\n")
	if position.Line-1 >= 0 && position.Line-1 < len(lines) {
		return strings.TrimSpace(lines[position.Line-1])
	}
	return ""
}

// isDeclContext checks if an identifier is in a declaration context (not a reference).
func isDeclContext(n *ast.Ident, f *ast.File) bool {
	if n.Obj != nil {
		switch n.Obj.Kind {
		case ast.Typ, ast.Fun, ast.Var, ast.Con, ast.Lbl:
			return true
		}
	}
	return false
}

// ─── Query Methods ─────────────────────────────────────────────────────────────

// FindDefinitions returns all symbols that define the given name.
func (sg *SymbolGraph) FindDefinitions(name string) []*Symbol {
	sg.mu.RLock()
	defer sg.mu.RUnlock()
	return sg.byName[name]
}

// FindReferences returns all references to a symbol by name.
func (sg *SymbolGraph) FindReferences(name string) []*Reference {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	var refs []*Reference
	seen := make(map[string]bool)
	for _, sym := range sg.byName[name] {
		for _, ref := range sym.References {
			key := fmt.Sprintf("%s:%d:%d", ref.File, ref.Line, ref.Column)
			if !seen[key] {
				seen[key] = true
				refs = append(refs, ref)
			}
		}
	}
	return refs
}

// FindImplementations returns all types that implement a given interface.
func (sg *SymbolGraph) FindImplementations(interfaceName string) []*Symbol {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	var results []*Symbol
	// Find the interface symbol
	var iface *Symbol
	for _, sym := range sg.byName[interfaceName] {
		if sym.Kind == KindInterface {
			iface = sym
			break
		}
	}
	if iface == nil {
		return nil
	}

	// Collect interface method names
	ifaceMethods := make(map[string]bool)
	for _, child := range iface.Children {
		if child.Kind == KindMethod {
			ifaceMethods[child.Name] = true
		}
	}

	if len(ifaceMethods) == 0 {
		return nil
	}

	// Find structs that implement all interface methods.
	// Methods are stored as separate symbols with a Receiver field matching the struct name.
	for _, sym := range sg.symbols {
		if sym.Kind == KindStruct {
			implemented := make(map[string]bool)
			// Look through all methods in the same package that have this struct as receiver
			for _, method := range sg.symbols {
				if method.Kind == KindMethod && method.Package == sym.Package {
					// Check if receiver matches (with or without pointer)
					recv := method.Receiver
					if recv == sym.Name || recv == "*"+sym.Name {
						if ifaceMethods[method.Name] {
							implemented[method.Name] = true
						}
					}
				}
			}
			if len(implemented) == len(ifaceMethods) {
				results = append(results, sym)
			}
		}
	}

	return results
}

// FindCallers returns all functions that call a given function.
func (sg *SymbolGraph) FindCallers(funcName string) []*Symbol {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	callerSet := make(map[string]*Symbol)
	for _, sym := range sg.byName[funcName] {
		for _, ref := range sym.References {
			// Find the function that contains this reference
			for _, fileSym := range sg.byFile[ref.File] {
				if fileSym.Kind == KindFunction || fileSym.Kind == KindMethod {
					if ref.Line >= fileSym.Line && ref.Line <= fileSym.EndLine {
						callerSet[fileSym.ID] = fileSym
					}
				}
			}
		}
	}

	var callers []*Symbol
	for _, sym := range callerSet {
		callers = append(callers, sym)
	}
	sort.Slice(callers, func(i, j int) bool {
		return callers[i].ID < callers[j].ID
	})
	return callers
}

// GetSymbolsByFile returns all symbols in a given file.
func (sg *SymbolGraph) GetSymbolsByFile(file string) []*Symbol {
	sg.mu.RLock()
	defer sg.mu.RUnlock()
	return sg.byFile[file]
}

// GetSymbolsByPackage returns all symbols in a given package.
func (sg *SymbolGraph) GetSymbolsByPackage(pkg string) []*Symbol {
	sg.mu.RLock()
	defer sg.mu.RUnlock()
	return sg.byPkg[pkg]
}

// GetSymbol returns a specific symbol by ID.
func (sg *SymbolGraph) GetSymbol(id string) *Symbol {
	sg.mu.RLock()
	defer sg.mu.RUnlock()
	return sg.symbols[id]
}

// SearchSymbols searches for symbols matching a query.
func (sg *SymbolGraph) SearchSymbols(query string) []*Symbol {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	lower := strings.ToLower(query)
	var results []*Symbol
	for _, sym := range sg.symbols {
		if strings.Contains(strings.ToLower(sym.Name), lower) ||
			strings.Contains(strings.ToLower(sym.Package), lower) {
			results = append(results, sym)
		}
	}
	return results
}

// TraceCallGraph traces the full call graph from a starting function.
// Returns a map of function -> list of functions it calls.
func (sg *SymbolGraph) TraceCallGraph(funcName string, depth int) map[string][]string {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	result := make(map[string][]string)
	visited := make(map[string]bool)
	sg.traceCalls(funcName, depth, 0, visited, result)
	return result
}

func (sg *SymbolGraph) traceCalls(funcName string, maxDepth, currentDepth int, visited map[string]bool, result map[string][]string) {
	if currentDepth >= maxDepth || visited[funcName] {
		return
	}
	visited[funcName] = true

	var callees []string
	for _, sym := range sg.byName[funcName] {
		for _, ref := range sym.References {
			// Extract called function name from context
			callee := sg.extractCalledFunc(ref.Context)
			if callee != "" && callee != funcName {
				callees = append(callees, callee)
			}
		}
	}

	if len(callees) > 0 {
		result[funcName] = callees
		for _, callee := range callees {
			sg.traceCalls(callee, maxDepth, currentDepth+1, visited, result)
		}
	}
}

// extractCalledFunc extracts the function name being called from a code snippet.
func (sg *SymbolGraph) extractCalledFunc(context string) string {
	// Simple heuristic: look for patterns like "funcName(" or "pkg.FuncName("
	if idx := strings.Index(context, "("); idx > 0 {
		before := strings.TrimSpace(context[:idx])
		parts := strings.Fields(before)
		if len(parts) > 0 {
			last := parts[len(parts)-1]
			// Handle pkg.Func format
			if dotIdx := strings.LastIndex(last, "."); dotIdx >= 0 {
				return last[dotIdx+1:]
			}
			return last
		}
	}
	return ""
}

// Summary returns a human-readable summary of the symbol graph.
func (sg *SymbolGraph) Summary() string {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Symbol Graph Summary:\n"))
	sb.WriteString(fmt.Sprintf("  Total symbols: %d\n", len(sg.symbols)))
	sb.WriteString(fmt.Sprintf("  Files indexed: %d\n", len(sg.byFile)))
	sb.WriteString(fmt.Sprintf("  Packages: %d\n", len(sg.byPkg)))

	// Count by kind
	kindCount := make(map[SymbolKind]int)
	for _, sym := range sg.symbols {
		kindCount[sym.Kind]++
	}
	for kind, count := range kindCount {
		sb.WriteString(fmt.Sprintf("  %s: %d\n", kind, count))
	}

	return sb.String()
}

// ─── JSON Serialization ────────────────────────────────────────────────────────

// ExportJSON serializes the symbol graph to JSON.
func (sg *SymbolGraph) ExportJSON() ([]byte, error) {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	type exportData struct {
		Symbols []*Symbol `json:"symbols"`
		Edges   []*Edge   `json:"edges"`
	}

	data := exportData{
		Symbols: make([]*Symbol, 0, len(sg.symbols)),
		Edges:   sg.edges,
	}
	for _, sym := range sg.symbols {
		data.Symbols = append(data.Symbols, sym)
	}

	return json.MarshalIndent(data, "", "  ")
}

// ImportJSON deserializes the symbol graph from JSON.
func (sg *SymbolGraph) ImportJSON(data []byte) error {
	sg.mu.Lock()
	defer sg.mu.Unlock()

	type importData struct {
		Symbols []*Symbol `json:"symbols"`
		Edges   []*Edge   `json:"edges"`
	}

	var d importData
	if err := json.Unmarshal(data, &d); err != nil {
		return err
	}

	for _, sym := range d.Symbols {
		sg.addSymbol(sym)
	}
	sg.edges = d.Edges

	return nil
}
