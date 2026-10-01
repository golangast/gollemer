// Package memory implements a hybrid Graph + Vector memory indexer for
// Go codebases. It captures both semantic vector embeddings and
// relational AST graph dependencies so that retrieval can expand a
// semantic hit into its structural neighborhood.
//
// The pipeline is: semantic chunks (pkg/ast.CodeChunk) become CodeNodes;
// call sites, composite literals, and type references become typed
// edges; QueryContext then fuses a cosine-similarity vector search with
// a breadth-first graph walk to return seeds plus their structural
// context.
//
// Thread safety: a KnowledgeGraph must not be copied after first use;
// always work with *KnowledgeGraph. Reads (QueryContext) take a read
// lock; mutations (SetEmbedding) take a write lock. BuildGraph assembles
// the graph before it is shared, so no locking is needed during the
// build itself.
package memory

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"math"
	"sort"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"

	astpkg "github.com/golangast/gollemer/pkg/ast"
)

// Edge relations.
const (
	// RelationCalls: a function/method body calls another known symbol.
	RelationCalls = "CALLS"
	// RelationImplements: a struct's method set satisfies an interface.
	RelationImplements = "IMPLEMENTS"
	// RelationInstantiates: a function/method body builds a composite
	// literal of a known struct type.
	RelationInstantiates = "INSTANTIATES"
	// RelationDependsOn: a chunk references a known type (field,
	// parameter, result, or embedded type).
	RelationDependsOn = "DEPENDS_ON"
)

// CodeNode is one indexed unit of code: a semantic chunk plus its
// vector embedding.
type CodeNode struct {
	// ID is the chunk's deterministic content hash.
	ID          string    `json:"id"`
	FilePath    string    `json:"filePath"`
	Kind        string    `json:"kind"` // "struct", "interface", "function", "method"
	SymbolName  string    `json:"symbolName"`
	PackageName string    `json:"packageName"`
	DocComment  string    `json:"docComment,omitempty"`
	CodeContent string    `json:"codeContent"`
	Embedding   []float32 `json:"embedding,omitempty"`
}

// CodeEdge is one typed, directed relationship between nodes.
type CodeEdge struct {
	SourceID string `json:"sourceID"`
	TargetID string `json:"targetID"`
	Relation string `json:"relation"` // CALLS, IMPLEMENTS, INSTANTIATES, DEPENDS_ON
}

// KnowledgeGraph is the indexed codebase. Nodes and Edges carry the
// full directed, typed graph; AdjacencyList is the traversal index
// used by QueryContext: for every node, the sorted, deduplicated IDs
// of its neighbors along CALLS and IMPLEMENTS edges in both
// directions, so a seed expands to its callees, its callers, and its
// interface contracts alike.
//
// Do not copy a KnowledgeGraph after first use.
type KnowledgeGraph struct {
	mu            sync.RWMutex
	Nodes         map[string]CodeNode `json:"nodes"`
	Edges         []CodeEdge          `json:"edges"`
	AdjacencyList map[string][]string `json:"adjacencyList"`
}

// BuildGraph indexes codeCtx's package from chunks: every chunk becomes
// a CodeNode, and edges are derived from call sites (CALLS), composite
// literals (INSTANTIATES), type references (DEPENDS_ON), and
// go/types-checked interface satisfaction (IMPLEMENTS).
//
// IMPLEMENTS detection reloads the package's type information with
// go/types using codeCtx.SourceDir. If SourceDir is empty or the
// reload fails, the graph is still built — CALLS, INSTANTIATES, and
// DEPENDS_ON do not need type info — but no IMPLEMENTS edges are
// added. Method-call resolution is name-based with a same-receiver
// refinement; a call that is ambiguous across several receiver types
// is skipped rather than guessed.
func BuildGraph(codeCtx *astpkg.CodebaseContext, chunks []astpkg.CodeChunk) (*KnowledgeGraph, error) {
	if codeCtx == nil {
		return nil, fmt.Errorf("memory: nil CodebaseContext")
	}
	g := &KnowledgeGraph{
		Nodes:         make(map[string]CodeNode, len(chunks)),
		Edges:         []CodeEdge{},
		AdjacencyList: make(map[string][]string, len(chunks)),
	}
	b := &builder{
		g:          g,
		pkgName:    codeCtx.PackageName,
		funcs:      map[string]string{},
		methods:    map[string][]string{},
		methodRecv: map[string]string{},
		types:      map[string]string{},
		seenEdge:   map[string]bool{},
		imports:    map[string]bool{},
	}
	for _, path := range codeCtx.Imports {
		b.imports[path] = true
	}

	for _, c := range chunks {
		b.addNode(c)
	}
	for _, c := range chunks {
		b.edgesForChunk(c)
	}
	b.addImplementsEdges(codeCtx)
	b.buildAdjacency()
	return g, nil
}

// SetEmbedding attaches a vector embedding to a node. The slice is
// copied. It is a no-op for unknown IDs or a nil graph.
func (g *KnowledgeGraph) SetEmbedding(id string, emb []float32) {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	n, ok := g.Nodes[id]
	if !ok {
		return
	}
	cp := make([]float32, len(emb))
	copy(cp, emb)
	n.Embedding = cp
	g.Nodes[id] = n
}

// GetNode returns the node with the given ID.
func (g *KnowledgeGraph) GetNode(id string) (CodeNode, bool) {
	if g == nil {
		return CodeNode{}, false
	}
	g.mu.RLock()
	defer g.mu.RUnlock()
	n, ok := g.Nodes[id]
	return n, ok
}

// QueryContext performs hybrid retrieval over the graph:
//
//  1. Vector search: rank every node by cosine similarity between
//     queryEmbedding and the node's embedding; the topK become seeds.
//     Nodes without embeddings (or with mismatched dimensions) score 0.
//  2. Graph traversal: breadth-first search from the seeds up to
//     maxGraphDepth along CALLS and IMPLEMENTS edges (both directions).
//  3. Context synthesis: seeds first (in similarity rank order), then
//     discovered nodes in BFS visit order, deduplicated.
//
// It returns an error for a nil graph, an empty query embedding,
// non-positive topK, or negative maxGraphDepth.
func QueryContext(graph *KnowledgeGraph, queryEmbedding []float32, topK int, maxGraphDepth int) ([]CodeNode, error) {
	if graph == nil {
		return nil, fmt.Errorf("memory: nil graph")
	}
	if len(queryEmbedding) == 0 {
		return nil, fmt.Errorf("memory: empty query embedding")
	}
	if topK <= 0 {
		return nil, fmt.Errorf("memory: topK must be positive")
	}
	if maxGraphDepth < 0 {
		return nil, fmt.Errorf("memory: maxGraphDepth must be non-negative")
	}

	graph.mu.RLock()
	defer graph.mu.RUnlock()

	type scored struct {
		id  string
		sim float32
	}
	scoredNodes := make([]scored, 0, len(graph.Nodes))
	for id, n := range graph.Nodes {
		scoredNodes = append(scoredNodes, scored{id: id, sim: CosineSimilarity(queryEmbedding, n.Embedding)})
	}
	sort.Slice(scoredNodes, func(i, j int) bool {
		if scoredNodes[i].sim != scoredNodes[j].sim {
			return scoredNodes[i].sim > scoredNodes[j].sim
		}
		return scoredNodes[i].id < scoredNodes[j].id
	})
	if topK > len(scoredNodes) {
		topK = len(scoredNodes)
	}

	visited := make(map[string]bool, len(graph.Nodes))
	order := make([]string, 0, len(graph.Nodes))
	queue := make([]string, 0, topK)
	for _, s := range scoredNodes[:topK] {
		if !visited[s.id] {
			visited[s.id] = true
			order = append(order, s.id)
			queue = append(queue, s.id)
		}
	}
	for depth := 0; depth < maxGraphDepth && len(queue) > 0; depth++ {
		var next []string
		for _, id := range queue {
			for _, nb := range graph.AdjacencyList[id] {
				if !visited[nb] {
					visited[nb] = true
					order = append(order, nb)
					next = append(next, nb)
				}
			}
		}
		queue = next
	}

	out := make([]CodeNode, 0, len(order))
	for _, id := range order {
		if n, ok := graph.Nodes[id]; ok {
			out = append(out, n)
		}
	}
	return out, nil
}

// CosineSimilarity returns the cosine of the angle between a and b. It
// is 0 for dimension mismatches, empty vectors, and zero vectors, and
// the result is clamped to [-1, 1] against float error.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	s := dot / (math.Sqrt(na) * math.Sqrt(nb))
	if s > 1 {
		return 1
	}
	if s < -1 {
		return -1
	}
	return float32(s)
}

// builder holds the transient symbol tables used while building.
type builder struct {
	g          *KnowledgeGraph
	pkgName    string
	funcs      map[string]string   // function name -> node ID
	methods    map[string][]string // method name -> node IDs
	methodRecv map[string]string   // method node ID -> receiver base type name
	types      map[string]string   // type name -> node ID
	seenEdge   map[string]bool
	imports    map[string]bool // local import names, for external-call filtering
}

func (b *builder) addNode(c astpkg.CodeChunk) {
	n := CodeNode{
		ID:          c.ID,
		FilePath:    c.FilePath,
		Kind:        c.Kind,
		SymbolName:  c.SymbolName,
		PackageName: b.pkgName,
		DocComment:  c.DocComment,
		CodeContent: c.CodeContent,
	}
	b.g.Nodes[n.ID] = n
	switch c.Kind {
	case "function":
		b.funcs[c.SymbolName] = n.ID
	case "method":
		recv, name := splitMethodSymbol(c.SymbolName)
		b.methods[name] = append(b.methods[name], n.ID)
		b.methodRecv[n.ID] = recv
	case "struct", "interface":
		b.types[c.SymbolName] = n.ID
	}
}

func (b *builder) edgesForChunk(c astpkg.CodeChunk) {
	id, ok := b.nodeID(c)
	if !ok {
		return
	}
	// Type references become DEPENDS_ON edges for every chunk kind.
	for _, dep := range c.Dependencies {
		if tid, ok := b.types[dep]; ok {
			b.addEdge(id, tid, RelationDependsOn)
		}
	}
	if c.Kind != "function" && c.Kind != "method" {
		return
	}
	fn := parseFuncDecl(c.CodeContent)
	if fn == nil || fn.Body == nil {
		return
	}
	recvName, recvType := "", ""
	if fn.Recv != nil && len(fn.Recv.List) > 0 && fn.Recv.List[0] != nil {
		recvType = baseTypeName(fn.Recv.List[0].Type)
		if len(fn.Recv.List[0].Names) > 0 && fn.Recv.List[0].Names[0] != nil {
			recvName = fn.Recv.List[0].Names[0].Name
		}
	}
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		switch t := n.(type) {
		case *ast.CallExpr:
			b.callEdge(id, t, recvName, recvType)
		case *ast.CompositeLit:
			b.instantiateEdge(id, t)
		}
		return true
	})
}

func (b *builder) nodeID(c astpkg.CodeChunk) (string, bool) {
	_, ok := b.g.Nodes[c.ID]
	return c.ID, ok
}

func (b *builder) callEdge(srcID string, call *ast.CallExpr, recvName, recvType string) {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		// A type conversion (T(x)) is not a call.
		if _, isType := b.types[fun.Name]; isType {
			return
		}
		if tid, ok := b.funcs[fun.Name]; ok {
			b.addEdge(srcID, tid, RelationCalls)
		}
	case *ast.SelectorExpr:
		x, ok := fun.X.(*ast.Ident)
		if !ok || x == nil {
			return
		}
		if b.imports[x.Name] {
			return // qualified call into another package
		}
		b.methodCallEdge(srcID, x.Name, fun.Sel.Name, recvName, recvType)
	}
}

func (b *builder) methodCallEdge(srcID, recvVar, method, recvName, recvType string) {
	cands := b.methods[method]
	if len(cands) == 0 {
		return
	}
	// A call on the method's own receiver variable resolves precisely
	// to the same receiver type's method.
	if recvVar == recvName && recvType != "" {
		for _, cid := range cands {
			if b.methodRecv[cid] == recvType {
				b.addEdge(srcID, cid, RelationCalls)
				return
			}
		}
	}
	if len(cands) == 1 {
		b.addEdge(srcID, cands[0], RelationCalls)
	}
	// Otherwise the call is ambiguous across receiver types; skip
	// rather than guess.
}

func (b *builder) instantiateEdge(srcID string, lit *ast.CompositeLit) {
	if lit == nil || lit.Type == nil {
		return
	}
	if id, ok := lit.Type.(*ast.Ident); ok && id != nil {
		if tid, ok := b.types[id.Name]; ok {
			b.addEdge(srcID, tid, RelationInstantiates)
		}
	}
}

func (b *builder) addEdge(src, dst, rel string) {
	if src == "" || dst == "" {
		return
	}
	key := src + "\x00" + dst + "\x00" + rel
	if b.seenEdge[key] {
		return
	}
	b.seenEdge[key] = true
	b.g.Edges = append(b.g.Edges, CodeEdge{SourceID: src, TargetID: dst, Relation: rel})
}

// addImplementsEdges inserts IMPLEMENTS edges using real go/types
// interface satisfaction. The package's type information is reloaded
// from codeCtx.SourceDir; when that is unavailable the step is
// skipped and the graph keeps its other edges.
func (b *builder) addImplementsEdges(codeCtx *astpkg.CodebaseContext) {
	if codeCtx.SourceDir == "" {
		return
	}
	tpkg := loadTypesPackage(codeCtx.SourceDir)
	if tpkg == nil {
		return
	}
	scope := tpkg.Scope()
	if scope == nil {
		return
	}
	for _, sm := range codeCtx.Structs {
		sID, ok := b.types[sm.Name]
		if !ok {
			continue
		}
		tn, ok := scope.Lookup(sm.Name).(*types.TypeName)
		if !ok || tn == nil {
			continue
		}
		named, ok := tn.Type().(*types.Named)
		if !ok || named == nil {
			continue
		}
		ptr := types.NewPointer(named)
		for _, im := range codeCtx.Interfaces {
			iID, ok := b.types[im.Name]
			if !ok || iID == sID {
				continue
			}
			itn, ok := scope.Lookup(im.Name).(*types.TypeName)
			if !ok || itn == nil {
				continue
			}
			iface, ok := itn.Type().Underlying().(*types.Interface)
			if !ok || iface == nil {
				continue
			}
			if types.Implements(ptr, iface) {
				b.addEdge(sID, iID, RelationImplements)
			}
		}
	}
}

// buildAdjacency indexes CALLS and IMPLEMENTS edges for traversal, in
// both directions, with sorted deduplicated neighbor lists.
func (b *builder) buildAdjacency() {
	for id := range b.g.Nodes {
		b.g.AdjacencyList[id] = []string{}
	}
	for _, e := range b.g.Edges {
		if e.Relation != RelationCalls && e.Relation != RelationImplements {
			continue
		}
		b.g.AdjacencyList[e.SourceID] = append(b.g.AdjacencyList[e.SourceID], e.TargetID)
		b.g.AdjacencyList[e.TargetID] = append(b.g.AdjacencyList[e.TargetID], e.SourceID)
	}
	for id, nbs := range b.g.AdjacencyList {
		b.g.AdjacencyList[id] = dedupSorted(nbs)
	}
}

// loadTypesPackage reloads the type information of the package in dir.
// It returns nil when the package cannot be loaded cleanly.
func loadTypesPackage(dir string) *types.Package {
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedTypes,
		Dir:  dir,
	}
	pkgs, err := packages.Load(cfg, ".")
	if err != nil || len(pkgs) == 0 {
		return nil
	}
	p := pkgs[0]
	if p == nil || p.Types == nil || p.IllTyped || len(p.Errors) > 0 {
		return nil
	}
	return p.Types
}

// parseFuncDecl parses one chunk's formatted source back into a
// *ast.FuncDecl. Chunk content is a declaration fragment, so it is
// wrapped in a package clause first.
func parseFuncDecl(content string) *ast.FuncDecl {
	if strings.TrimSpace(content) == "" {
		return nil
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "chunk.go", "package chunk\n"+content, parser.SkipObjectResolution)
	if err != nil || f == nil {
		return nil
	}
	for _, d := range f.Decls {
		if fn, ok := d.(*ast.FuncDecl); ok && fn != nil {
			return fn
		}
	}
	return nil
}

// splitMethodSymbol splits "(*Server).Start" into ("Server", "Start").
func splitMethodSymbol(sym string) (recv, method string) {
	if i := strings.LastIndex(sym, ")."); i >= 0 {
		return strings.TrimPrefix(sym[:i], "(*"), sym[i+2:]
	}
	if j := strings.LastIndex(sym, "."); j >= 0 {
		return sym[:j], sym[j+1:]
	}
	return "", sym
}

// baseTypeName strips pointer and generic arguments: "*Server" and
// "Box[T]" both become "Server" and "Box".
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

func dedupSorted(in []string) []string {
	if len(in) == 0 {
		return in
	}
	cp := append([]string(nil), in...)
	sort.Strings(cp)
	out := cp[:0]
	for i, s := range cp {
		if i == 0 || s != cp[i-1] {
			out = append(out, s)
		}
	}
	return out
}

/*
Runnable example: build a hybrid graph for a sample package, attach
dummy embeddings, run a hybrid query, and print the expanded context.

package main

import (
	"fmt"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"

	astpkg "github.com/golangast/gollemer/pkg/ast"
	"github.com/golangast/gollemer/pkg/memory"
)

const greeterSrc = `package greeter

// Greeter greets people.
type Greeter struct{ Prefix string }

// Greet renders a greeting for name.
func (g *Greeter) Greet(name string) string { return g.Prefix + ", " + name }

// Friendly can greet.
type Friendly interface{ Greet(name string) string }

// Hello greets the world via g.
func Hello(g *Greeter) string { return g.Greet("world") }
`

func main() {
	dir, err := os.MkdirTemp("", "memdemo")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module example.com/greeter\n\ngo 1.26.0\n"), 0o644); err != nil {
		log.Fatal(err)
	}
	srcPath := filepath.Join(dir, "greeter.go")
	if err := os.WriteFile(srcPath, []byte(greeterSrc), 0o644); err != nil {
		log.Fatal(err)
	}

	ctx, err := astpkg.LoadPackageContext(dir)
	if err != nil {
		log.Fatal(err)
	}
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, srcPath, nil, parser.ParseComments)
	if err != nil {
		log.Fatal(err)
	}
	chunks, err := astpkg.ChunkFile(fset, f, srcPath)
	if err != nil {
		log.Fatal(err)
	}
	g, err := memory.BuildGraph(ctx, chunks)
	if err != nil {
		log.Fatal(err)
	}

	// Dummy 3-D embeddings, one per symbol.
	emb := map[string][]float32{
		"Greeter":  {0.1, 0.9, 0.0},
		"Greet":    {0.8, 0.2, 0.1},
		"Friendly": {0.0, 0.9, 0.1},
		"Hello":    {0.9, 0.1, 0.0},
	}
	for _, c := range chunks {
		name := c.SymbolName
		if i := len(name) - 1; i >= 0 {
			// method chunks look like "(*Greeter).Greet": use the method name.
			if j := lastDot(name); j >= 0 {
				name = name[j+1:]
			}
		}
		if e, ok := emb[name]; ok {
			g.SetEmbedding(c.ID, e)
		}
	}

	// Query near Hello: the seed expands to Greet (called) and Greeter
	// (depended on) through the graph walk.
	res, err := memory.QueryContext(g, []float32{0.9, 0.1, 0.0}, 1, 2)
	if err != nil {
		log.Fatal(err)
	}
	for _, n := range res {
		fmt.Printf("%-9s %-16s edges=%d\n", n.Kind, n.SymbolName, len(g.AdjacencyList[n.ID]))
	}
}

func lastDot(s string) int {
	for i := len(s) - 1; i >= 0; i-- {
		if s[i] == '.' {
			return i
		}
	}
	return -1
}
*/
