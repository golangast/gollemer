package memory

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sync"
	"testing"

	astpkg "github.com/golangast/gollemer/pkg/ast"
)

const indexerFixture = `package memtest

// Greeter greets people.
type Greeter struct {
	Prefix string
}

// Greet renders a greeting for name.
func (g *Greeter) Greet(name string) string {
	return g.Prefix + ", " + name
}

// Friendly can greet.
type Friendly interface {
	Greet(name string) string
}

// Hello greets the world via g.
func Hello(g *Greeter) string {
	return g.Greet("world")
}

// Shout greets loudly via Hello.
func Shout(g *Greeter) string {
	return Hello(g) + "!"
}

// NewGreeter builds a Greeter.
func NewGreeter(prefix string) *Greeter {
	return &Greeter{Prefix: prefix}
}
`

// fixtureGraph builds a temp module, chunks it, and builds the graph.
func fixtureGraph(t *testing.T) (*KnowledgeGraph, []astpkg.CodeChunk) {
	t.Helper()
	root := t.TempDir()
	write := func(rel, content string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/memtest\n\ngo 1.26.0\n")
	write("memtest.go", indexerFixture)

	ctx, err := astpkg.LoadPackageContext(root)
	if err != nil {
		t.Fatalf("LoadPackageContext: %v", err)
	}
	srcPath := filepath.Join(root, "memtest.go")
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, srcPath, nil, parser.ParseComments)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	chunks, err := astpkg.ChunkFile(fset, f, srcPath)
	if err != nil {
		t.Fatalf("ChunkFile: %v", err)
	}
	g, err := BuildGraph(ctx, chunks)
	if err != nil {
		t.Fatalf("BuildGraph: %v", err)
	}
	return g, chunks
}

func idBySymbol(g *KnowledgeGraph, symbol string) string {
	for id, n := range g.Nodes {
		if n.SymbolName == symbol {
			return id
		}
	}
	return ""
}

func edgeSet(g *KnowledgeGraph) map[string]bool {
	m := map[string]bool{}
	names := map[string]string{}
	for id, n := range g.Nodes {
		names[id] = n.SymbolName
	}
	for _, e := range g.Edges {
		m[names[e.SourceID]+" -"+e.Relation+"-> "+names[e.TargetID]] = true
	}
	return m
}

func TestBuildGraphNodes(t *testing.T) {
	g, _ := fixtureGraph(t)
	if len(g.Nodes) != 6 {
		t.Fatalf("nodes = %d, want 6", len(g.Nodes))
	}
	for _, sym := range []string{"Greeter", "(*Greeter).Greet", "Friendly", "Hello", "Shout", "NewGreeter"} {
		if idBySymbol(g, sym) == "" {
			t.Errorf("missing node for symbol %q", sym)
		}
	}
	n := g.Nodes[idBySymbol(g, "Hello")]
	if n.Kind != "function" || n.PackageName != "memtest" {
		t.Errorf("Hello node = %+v, want kind=function package=memtest", n)
	}
	if n.CodeContent == "" || n.DocComment == "" {
		t.Error("Hello node missing content or doc comment")
	}
	// IDs are deterministic content hashes.
	if len(n.ID) != 32 {
		t.Errorf("node ID %q is not a 32-char hash", n.ID)
	}
}

func TestBuildGraphEdges(t *testing.T) {
	g, _ := fixtureGraph(t)
	edges := edgeSet(g)
	want := []string{
		"Hello -CALLS-> (*Greeter).Greet",
		"Shout -CALLS-> Hello",
		"Greeter -IMPLEMENTS-> Friendly",
		"NewGreeter -INSTANTIATES-> Greeter",
		"Hello -DEPENDS_ON-> Greeter",
		"Shout -DEPENDS_ON-> Greeter",
		"NewGreeter -DEPENDS_ON-> Greeter",
	}
	for _, w := range want {
		if !edges[w] {
			t.Errorf("missing edge %q\n got: %v", w, edges)
		}
	}
}

func TestBuildGraphAdjacency(t *testing.T) {
	g, _ := fixtureGraph(t)
	hello := idBySymbol(g, "Hello")
	names := map[string]bool{}
	for _, nb := range g.AdjacencyList[hello] {
		names[g.Nodes[nb].SymbolName] = true
	}
	// Bidirectional: Hello reaches Greet (callee) and Shout (caller).
	if !names["(*Greeter).Greet"] || !names["Shout"] {
		t.Errorf("Hello adjacency = %v, want Greet and Shout", names)
	}
	// DEPENDS_ON edges are not traversal edges.
	if names["Greeter"] {
		t.Errorf("Hello adjacency should not include DEPENDS_ON neighbor Greeter: %v", names)
	}
	// Every node has an adjacency entry.
	for id := range g.Nodes {
		if _, ok := g.AdjacencyList[id]; !ok {
			t.Errorf("node %q missing from AdjacencyList", g.Nodes[id].SymbolName)
		}
	}
}

func TestBuildGraphNilContext(t *testing.T) {
	if _, err := BuildGraph(nil, nil); err == nil {
		t.Error("BuildGraph(nil, nil): expected error, got nil")
	}
}

func TestCosineSimilarity(t *testing.T) {
	cases := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, 2, 3}, []float32{1, 2, 3}, 1},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1},
		{"scaled", []float32{1, 1}, []float32{3, 3}, 1},
		{"mismatch", []float32{1, 2}, []float32{1, 2, 3}, 0},
		{"zero", []float32{0, 0}, []float32{1, 2}, 0},
		{"empty", nil, nil, 0},
		{"one-empty", []float32{1}, nil, 0},
	}
	for _, c := range cases {
		if got := CosineSimilarity(c.a, c.b); got != c.want {
			t.Errorf("%s: CosineSimilarity = %v, want %v", c.name, got, c.want)
		}
	}
	// Near-parallel vectors stay within [-1, 1].
	if got := CosineSimilarity([]float32{0.1, 0.2}, []float32{0.1, 0.2}); got < -1 || got > 1 {
		t.Errorf("out of range: %v", got)
	}
}

func TestQueryContextHybrid(t *testing.T) {
	g, chunks := fixtureGraph(t)
	emb := map[string][]float32{
		"Greeter":          {0.1, 0.9, 0.0},
		"(*Greeter).Greet": {0.8, 0.2, 0.1},
		"Friendly":         {0.0, 0.9, 0.1},
		"Hello":            {0.9, 0.1, 0.0},
		"Shout":            {0.85, 0.15, 0.05},
		"NewGreeter":       {0.2, 0.1, 0.9},
	}
	for _, c := range chunks {
		g.SetEmbedding(c.ID, emb[c.SymbolName])
	}

	// Query closest to Hello; depth 2 must pull in Greet (callee) and
	// Shout (caller) through the graph walk.
	res, err := QueryContext(g, []float32{0.9, 0.1, 0.0}, 1, 2)
	if err != nil {
		t.Fatalf("QueryContext: %v", err)
	}
	if len(res) == 0 || res[0].SymbolName != "Hello" {
		t.Fatalf("first result = %v, want Hello seed first", res)
	}
	got := map[string]bool{}
	for _, n := range res {
		got[n.SymbolName] = true
	}
	for _, want := range []string{"Hello", "(*Greeter).Greet", "Shout"} {
		if !got[want] {
			t.Errorf("expanded context missing %q: %v", want, got)
		}
	}
	if len(res) != 3 {
		t.Errorf("expanded context size = %d, want 3 (no duplicates)", len(res))
	}
}

func TestQueryContextDepthZero(t *testing.T) {
	g, chunks := fixtureGraph(t)
	for _, c := range chunks {
		g.SetEmbedding(c.ID, []float32{1, 0})
	}
	res, err := QueryContext(g, []float32{1, 0}, 2, 0)
	if err != nil {
		t.Fatalf("QueryContext: %v", err)
	}
	if len(res) != 2 {
		t.Errorf("depth-0 result size = %d, want exactly the 2 seeds", len(res))
	}
}

func TestQueryContextErrors(t *testing.T) {
	g, _ := fixtureGraph(t)
	for _, tc := range []struct {
		name string
		g    *KnowledgeGraph
		q    []float32
		topK int
		dep  int
	}{
		{"nil graph", nil, []float32{1}, 1, 1},
		{"empty query", g, nil, 1, 1},
		{"zero topK", g, []float32{1}, 0, 1},
		{"negative depth", g, []float32{1}, 1, -1},
	} {
		if _, err := QueryContext(tc.g, tc.q, tc.topK, tc.dep); err == nil {
			t.Errorf("%s: expected error, got nil", tc.name)
		}
	}
}

func TestQueryContextTopKClamped(t *testing.T) {
	g, _ := fixtureGraph(t)
	res, err := QueryContext(g, []float32{1, 0, 0}, 1000, 0)
	if err != nil {
		t.Fatalf("QueryContext: %v", err)
	}
	if len(res) != len(g.Nodes) {
		t.Errorf("clamped topK: got %d nodes, want %d", len(res), len(g.Nodes))
	}
}

func TestQueryContextConcurrent(t *testing.T) {
	g, chunks := fixtureGraph(t)
	for _, c := range chunks {
		g.SetEmbedding(c.ID, []float32{0.5, 0.5})
	}
	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if i%2 == 0 {
				g.SetEmbedding(chunks[0].ID, []float32{float32(i), 0})
			} else if _, err := QueryContext(g, []float32{1, 0}, 2, 1); err != nil {
				t.Errorf("QueryContext: %v", err)
			}
		}(i)
	}
	wg.Wait()
}
