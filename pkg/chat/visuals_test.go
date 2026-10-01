package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/pkg/memory"
)

func seqTestGraph() *memory.KnowledgeGraph {
	nodes := map[string]memory.CodeNode{
		"h": {ID: "h", SymbolName: "HandleOrder", Kind: "function", PackageName: "api", FilePath: "api/order.go",
			CodeContent: "func HandleOrder(id string) {\n\tValidate(id)\n\tsvc.Do(id)\n}"},
		"v": {ID: "v", SymbolName: "Validate", Kind: "function", PackageName: "api", FilePath: "api/order.go",
			CodeContent: "func Validate(id string) bool {\n\treturn id != \"\"\n}"},
		"s": {ID: "s", SymbolName: "Service.Do", Kind: "method", PackageName: "api", FilePath: "api/service.go",
			CodeContent: "func (s Service) Do(id string) {\n\ts.repo.Query(id)\n}"},
		"r": {ID: "r", SymbolName: "Repo.Query", Kind: "method", PackageName: "api", FilePath: "api/repo.go",
			CodeContent: "func (r Repo) Query(id string) {\n\tHandleOrder(id) // retry entry on failure (cycle)\n}"},
	}
	return &memory.KnowledgeGraph{
		Nodes: nodes,
		Edges: []memory.CodeEdge{
			{SourceID: "h", TargetID: "s", Relation: memory.RelationCalls},
			{SourceID: "h", TargetID: "v", Relation: memory.RelationCalls},
			{SourceID: "s", TargetID: "r", Relation: memory.RelationCalls},
			{SourceID: "r", TargetID: "h", Relation: memory.RelationCalls}, // cycle back
			{SourceID: "s", TargetID: "ghost", Relation: memory.RelationCalls},
		},
	}
}

func TestGenerateMermaidGraph(t *testing.T) {
	nodes := []memory.CodeNode{
		{ID: "a", SymbolName: "Cache", Kind: "interface"},
		{ID: "b", SymbolName: "Store", Kind: "struct"},
		{ID: "c", SymbolName: "HandleOrder", Kind: "function"},
	}
	edges := []memory.CodeEdge{
		{SourceID: "c", TargetID: "b", Relation: memory.RelationCalls},
		{SourceID: "b", TargetID: "a", Relation: memory.RelationImplements},
		{SourceID: "c", TargetID: "missing", Relation: memory.RelationCalls},
	}
	got := GenerateMermaidGraph(nodes, edges)
	if !strings.HasPrefix(got, "graph TD\n") {
		t.Errorf("missing graph TD header:\n%s", got)
	}
	for _, want := range []string{
		`n0(('Cache'))`,       // interface: double circle
		`n1['Store']`,         // struct: rectangle
		`n2(['HandleOrder'])`, // function: stadium
		"n2 -->|CALLS| n1",
		"n1 -->|IMPLEMENTS| n0",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("mermaid missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "missing") {
		t.Errorf("dangling edge was not skipped:\n%s", got)
	}
}

func TestBuildExecutionSequence(t *testing.T) {
	payload, err := BuildExecutionSequence(seqTestGraph(), "HandleOrder")
	if err != nil {
		t.Fatalf("BuildExecutionSequence: %v", err)
	}
	// DFS pre-order: h -> s -> r (r->h is a visited cycle) -> v.
	wantOrder := []string{"HandleOrder", "Service.Do", "Repo.Query", "Validate"}
	if len(payload.SequenceSteps) != len(wantOrder) {
		t.Fatalf("steps = %d, want %d", len(payload.SequenceSteps), len(wantOrder))
	}
	for i, want := range wantOrder {
		if !strings.Contains(payload.SequenceSteps[i].Title, want) {
			t.Errorf("step %d title = %q, want it to mention %q", i, payload.SequenceSteps[i].Title, want)
		}
		if payload.SequenceSteps[i].FilePath == "" {
			t.Errorf("step %d missing file path", i)
		}
		if payload.SequenceSteps[i].CodeSnippet == "" {
			t.Errorf("step %d missing code snippet", i)
		}
	}
	if !strings.Contains(payload.SequenceSteps[3].Subtitle, "leaf") {
		t.Errorf("leaf step subtitle = %q, want leaf note", payload.SequenceSteps[3].Subtitle)
	}
	if !strings.HasPrefix(payload.MermaidDiagram, "graph TD\n") {
		t.Error("mermaid diagram missing header")
	}
	if !strings.Contains(payload.MermaidDiagram, "|CALLS|") {
		t.Error("mermaid diagram missing CALLS edges")
	}
	if !strings.Contains(payload.BeginnerAnalogy, "recipe") {
		t.Errorf("analogy = %q, want the function/recipe analogy", payload.BeginnerAnalogy)
	}
	if payload.SafetyBadge.Summary == "" {
		t.Error("empty safety badge summary")
	}
}

func TestBuildExecutionSequenceEntryResolution(t *testing.T) {
	g := seqTestGraph()
	// Add two same-named methods to force ambiguity.
	g.Nodes["g1"] = memory.CodeNode{ID: "g1", SymbolName: "(*A).Get", Kind: "method", PackageName: "api", FilePath: "api/a.go", CodeContent: "func (a *A) Get() {}"}
	g.Nodes["g2"] = memory.CodeNode{ID: "g2", SymbolName: "(*B).Get", Kind: "method", PackageName: "api", FilePath: "api/b.go", CodeContent: "func (b *B) Get() {}"}

	if _, err := BuildExecutionSequence(nil, "HandleOrder"); err == nil {
		t.Error("nil graph: want error")
	}
	if _, err := BuildExecutionSequence(g, "  "); err == nil {
		t.Error("empty symbol: want error")
	}
	if _, err := BuildExecutionSequence(g, "Nope"); err == nil {
		t.Error("unknown symbol: want error")
	}
	if _, err := BuildExecutionSequence(g, "Get"); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Errorf("ambiguous fuzzy symbol: want ambiguity error, got %v", err)
	}
	// Exact method symbol resolves.
	p, err := BuildExecutionSequence(g, "(*A).Get")
	if err != nil {
		t.Fatalf("exact method symbol: %v", err)
	}
	if !strings.Contains(p.SequenceSteps[0].Title, "(*A).Get") {
		t.Errorf("resolved wrong node: %q", p.SequenceSteps[0].Title)
	}
	// Node ID resolves directly.
	if _, err := BuildExecutionSequence(g, "h"); err != nil {
		t.Errorf("node ID resolution: %v", err)
	}
}

func TestConceptAnalogy(t *testing.T) {
	cases := map[string]string{
		"interface": "A wall outlet (defines the plug shape)",
		"struct":    "A concrete appliance (implements the plug)",
		"channel":   "A conveyor belt moving data between goroutines",
	}
	for concept, want := range cases {
		if got := ConceptAnalogy(concept); got != want {
			t.Errorf("ConceptAnalogy(%q) = %q, want %q", concept, got, want)
		}
	}
	if got := ConceptAnalogy("frobnicate"); got == "" {
		t.Error("unknown concept: want a fallback analogy, got empty")
	}
}

func TestSafetyBadgeAllocMetrics(t *testing.T) {
	content := "import \"io\"\n\nfunc demo(x *int, c io.Closer) int {\n" +
		"\tif x == nil {\n\t\treturn 0\n\t}\n" +
		"\tdefer c.Close()\n" +
		"\tm := make(map[string]int)\n" +
		"\tp := new(int)\n" +
		"\ts := append([]int{}, 1)\n" +
		"\tt := struct{ A int }{A: 1}\n" +
		"\t_, _, _, _ = m, p, s, t\n" +
		"\treturn *x\n}"
	g := &memory.KnowledgeGraph{
		Nodes: map[string]memory.CodeNode{
			"d": {ID: "d", SymbolName: "demo", Kind: "function", PackageName: "p", FilePath: "p/demo.go", CodeContent: content},
		},
	}
	payload, err := BuildExecutionSequence(g, "demo")
	if err != nil {
		t.Fatalf("BuildExecutionSequence: %v", err)
	}
	m := payload.SafetyBadge.AllocMetrics
	if m.MakeCalls != 1 || m.NewCalls != 1 || m.Appends != 1 || m.CompositeLits != 2 {
		t.Errorf("alloc metrics = %+v, want {1 1 2 1}", m)
	}
	if payload.SafetyBadge.NullSafetyStatus != "clean" {
		t.Errorf("null safety status = %q, want clean", payload.SafetyBadge.NullSafetyStatus)
	}
	if !strings.Contains(payload.SafetyBadge.Summary, "allocation site(s)") {
		t.Errorf("badge summary = %q", payload.SafetyBadge.Summary)
	}
}
