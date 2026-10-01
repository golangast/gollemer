// Visual components for the chat interface: execution step sequences,
// Mermaid flowcharts, safety badges, and beginner analogies, all
// derived from the indexed codebase graph and Go AST structures.
//
// Everything here is standard library. The visual payload is built
// from memory.CodeNodes (AST chunks) and memory.CodeEdges, so it
// works on any graph produced by memory.BuildGraph without needing
// the original repository on disk.
package chat

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"sort"
	"strings"

	"github.com/golangast/gollemer/pkg/analysis"
	"github.com/golangast/gollemer/pkg/memory"
)

const (
	maxSequenceSteps = 50 // hard cap: keeps payloads readable
	maxSequenceDepth = 12 // hard cap: deep recursion still terminates
	snippetMaxLines  = 12
)

// StepItem is one step of an execution sequence, in call order.
type StepItem struct {
	Title       string `json:"title"`
	Subtitle    string `json:"subtitle"`
	CodeSnippet string `json:"codeSnippet"`
	FilePath    string `json:"filePath"`
}

// AllocMetrics counts static allocation sites found in the sequenced
// code: make/new calls, composite literals, and appends.
type AllocMetrics struct {
	MakeCalls     int `json:"makeCalls"`
	NewCalls      int `json:"newCalls"`
	CompositeLits int `json:"compositeLits"`
	Appends       int `json:"appends"`
}

// Total returns the overall number of allocation sites.
func (a AllocMetrics) Total() int {
	return a.MakeCalls + a.NewCalls + a.CompositeLits + a.Appends
}

// SafetyBadge summarizes symbolic safety over the sequenced code.
// NullSafetyStatus is "clean" (no violations), "warnings" (only
// WARNINGs), or "critical" (at least one CRITICAL violation).
type SafetyBadge struct {
	NullSafetyStatus string       `json:"nullSafetyStatus"`
	AllocMetrics     AllocMetrics `json:"allocMetrics"`
	Summary          string       `json:"summary"`
}

// VisualPayload is the complete visual answer for one entry point.
type VisualPayload struct {
	SequenceSteps   []StepItem  `json:"sequenceSteps"`
	MermaidDiagram  string      `json:"mermaidDiagram"`
	SafetyBadge     SafetyBadge `json:"safetyBadge"`
	BeginnerAnalogy string      `json:"beginnerAnalogy"`
}

// BuildExecutionSequence traverses CALLS edges in graph starting from
// entrySymbol and returns the ordered call hierarchy as visual steps,
// e.g. HTTP handler -> service -> repository -> database query.
//
// entrySymbol may be a node ID, an exact symbol name ("HandleOrder",
// "(*Store).Get"), or an unambiguous short name ("Get" resolves when
// only one symbol ends that way). Traversal is depth-first pre-order
// with a visited set, so cycles terminate; steps and depth are capped
// (see maxSequenceSteps/maxSequenceDepth).
func BuildExecutionSequence(graph *memory.KnowledgeGraph, entrySymbol string) (*VisualPayload, error) {
	if graph == nil {
		return nil, fmt.Errorf("chat: nil knowledge graph")
	}
	entry, err := resolveEntrySymbol(graph, entrySymbol)
	if err != nil {
		return nil, err
	}

	// CALLS adjacency, restricted to nodes present in the graph.
	calls := make(map[string][]string)
	for _, e := range graph.Edges {
		if e.Relation != memory.RelationCalls {
			continue
		}
		if _, ok := graph.Nodes[e.SourceID]; !ok {
			continue
		}
		if _, ok := graph.Nodes[e.TargetID]; !ok {
			continue
		}
		calls[e.SourceID] = append(calls[e.SourceID], e.TargetID)
	}
	for id := range calls {
		sort.Strings(calls[id])
		calls[id] = dedupStrings(calls[id])
	}

	// Iterative depth-first pre-order traversal with a visited set.
	var order []memory.CodeNode
	visited := map[string]bool{entry.ID: true}
	type frame struct {
		id       string
		depth    int
		childIdx int
	}
	stack := []frame{{id: entry.ID}}
	for len(stack) > 0 && len(order) < maxSequenceSteps {
		f := &stack[len(stack)-1]
		if f.childIdx == 0 {
			order = append(order, graph.Nodes[f.id])
		}
		if f.depth >= maxSequenceDepth || f.childIdx >= len(calls[f.id]) {
			stack = stack[:len(stack)-1]
			continue
		}
		child := calls[f.id][f.childIdx]
		f.childIdx++
		if visited[child] {
			continue
		}
		visited[child] = true
		stack = append(stack, frame{id: child, depth: f.depth + 1})
	}

	steps := make([]StepItem, 0, len(order))
	for i, n := range order {
		steps = append(steps, StepItem{
			Title:       fmt.Sprintf("Step %d: %s", i+1, displayName(n)),
			Subtitle:    stepSubtitle(n, calls[n.ID], graph),
			CodeSnippet: snippet(n.CodeContent, snippetMaxLines),
			FilePath:    n.FilePath,
		})
	}

	payload := &VisualPayload{
		SequenceSteps:   steps,
		MermaidDiagram:  GenerateMermaidGraph(order, sequenceEdges(calls, order)),
		SafetyBadge:     buildSafetyBadge(order),
		BeginnerAnalogy: analogyForNode(entry),
	}
	return payload, nil
}

// resolveEntrySymbol finds the graph node for an entry symbol: a node
// ID, an exact symbol name, or an unambiguous trailing name.
func resolveEntrySymbol(graph *memory.KnowledgeGraph, symbol string) (memory.CodeNode, error) {
	symbol = strings.TrimSpace(symbol)
	if symbol == "" {
		return memory.CodeNode{}, fmt.Errorf("chat: empty entry symbol")
	}
	if n, ok := graph.GetNode(symbol); ok {
		return n, nil
	}
	var exact, fuzzy []memory.CodeNode
	for _, n := range graph.Nodes {
		switch {
		case n.SymbolName == symbol:
			exact = append(exact, n)
		case strings.HasSuffix(n.SymbolName, "."+symbol) || strings.HasSuffix(n.SymbolName, ")"+symbol):
			fuzzy = append(fuzzy, n)
		}
	}
	if len(exact) == 1 {
		return exact[0], nil
	}
	if len(exact) > 1 {
		return memory.CodeNode{}, fmt.Errorf("chat: %q is ambiguous: %s", symbol, candidateList(exact))
	}
	if len(fuzzy) == 1 {
		return fuzzy[0], nil
	}
	if len(fuzzy) > 1 {
		return memory.CodeNode{}, fmt.Errorf("chat: %q is ambiguous: %s", symbol, candidateList(fuzzy))
	}
	return memory.CodeNode{}, fmt.Errorf("chat: no code found for %q", symbol)
}

func candidateList(ns []memory.CodeNode) string {
	names := make([]string, 0, len(ns))
	for _, n := range ns {
		loc := n.SymbolName
		if n.FilePath != "" {
			loc += " (" + n.FilePath + ")"
		}
		names = append(names, loc)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}

// sequenceEdges rebuilds CALLS edges among the visited nodes for the
// payload diagram.
func sequenceEdges(calls map[string][]string, order []memory.CodeNode) []memory.CodeEdge {
	in := make(map[string]bool, len(order))
	for _, n := range order {
		in[n.ID] = true
	}
	var out []memory.CodeEdge
	for _, n := range order {
		for _, dst := range calls[n.ID] {
			if in[dst] {
				out = append(out, memory.CodeEdge{SourceID: n.ID, TargetID: dst, Relation: memory.RelationCalls})
			}
		}
	}
	return out
}

// GenerateMermaidGraph formats nodes and edges as a Mermaid flowchart.
// Node shapes encode kind: rectangles for structs, double circles for
// interfaces, stadiums for functions/methods. Edges whose endpoints
// are not both in nodes are skipped; unknown relations are labeled
// as-is.
func GenerateMermaidGraph(nodes []memory.CodeNode, edges []memory.CodeEdge) string {
	var sb strings.Builder
	sb.WriteString("graph TD\n")
	ids := make(map[string]string, len(nodes))
	for i, n := range nodes {
		mid := fmt.Sprintf("n%d", i)
		ids[n.ID] = mid
		fmt.Fprintf(&sb, "    %s%s\n", mid, mermaidNodeShape(n))
	}
	for _, e := range edges {
		src, ok1 := ids[e.SourceID]
		dst, ok2 := ids[e.TargetID]
		if !ok1 || !ok2 {
			continue
		}
		rel := e.Relation
		if rel == "" {
			rel = "LINKS"
		}
		fmt.Fprintf(&sb, "    %s -->|%s| %s\n", src, mermaidSafe(rel), dst)
	}
	return sb.String()
}

func mermaidNodeShape(n memory.CodeNode) string {
	label := mermaidSafe(displayName(n))
	switch n.Kind {
	case "interface":
		return fmt.Sprintf("(('%s'))", label)
	case "struct":
		return fmt.Sprintf("['%s']", label)
	default:
		return fmt.Sprintf("(['%s'])", label)
	}
}

func mermaidSafe(s string) string {
	s = strings.ReplaceAll(s, "'", "")
	s = strings.ReplaceAll(s, "\"", "")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "{", "")
	s = strings.ReplaceAll(s, "}", "")
	return strings.TrimSpace(s)
}

// ConceptAnalogy translates a Go concept name into an everyday
// analogy.
func ConceptAnalogy(concept string) string {
	switch strings.ToLower(strings.TrimSpace(concept)) {
	case "interface":
		return "A wall outlet (defines the plug shape)"
	case "struct":
		return "A concrete appliance (implements the plug)"
	case "channel", "chan":
		return "A conveyor belt moving data between goroutines"
	case "goroutine":
		return "A second cook in the kitchen, working at the same time as you"
	case "method":
		return "A recipe card taped to one specific appliance — only that appliance can follow it"
	case "function", "func":
		return "A recipe card anyone in the kitchen can follow"
	case "pointer":
		return "A sticky note with a house's address — not the house itself"
	case "slice":
		return "A stretchy tray that holds a row of items and grows as you add more"
	case "array":
		return "A fixed-size egg carton — one slot per item, no more, no less"
	case "map":
		return "A labeled drawer cabinet — ask for a label, get the drawer"
	case "mutex":
		return "A bathroom door lock — one goroutine at a time"
	case "error":
		return "A note passed back saying what went wrong, if anything did"
	case "defer":
		return "A promise to tidy up that runs automatically when the function walks out the door"
	case "package":
		return "A labeled toolbox holding related tools"
	default:
		return "A building block of the program — follow the diagram arrows to see how it connects"
	}
}

// analogyForNode weaves the entry node's name into its kind analogy.
func analogyForNode(n memory.CodeNode) string {
	name := displayName(n)
	switch n.Kind {
	case "interface":
		return fmt.Sprintf("%q is an interface: %s. Any struct whose methods match that shape plugs straight in.", name, ConceptAnalogy("interface"))
	case "struct":
		return fmt.Sprintf("%q is a struct: %s. Its methods are where the behavior lives — follow the steps below.", name, ConceptAnalogy("struct"))
	case "method":
		return fmt.Sprintf("%s is a method: %s. The sequence below shows what it calls, in order.", name, ConceptAnalogy("method"))
	default:
		return fmt.Sprintf("%q is a function: %s. The sequence below shows what it calls, in order.", name, ConceptAnalogy("function"))
	}
}

// buildSafetyBadge analyzes each sequenced node's code: the symbolic
// prover for null safety, AST inspection for allocation sites.
func buildSafetyBadge(nodes []memory.CodeNode) SafetyBadge {
	badge := SafetyBadge{NullSafetyStatus: "clean"}
	files := map[string]bool{}
	var violations, criticals int
	var scoreSum float64
	var scored int
	for _, n := range nodes {
		if n.FilePath != "" {
			files[n.FilePath] = true
		}
		fs, alloc := analyzeNodeChunk(n)
		badge.AllocMetrics.MakeCalls += alloc.MakeCalls
		badge.AllocMetrics.NewCalls += alloc.NewCalls
		badge.AllocMetrics.CompositeLits += alloc.CompositeLits
		badge.AllocMetrics.Appends += alloc.Appends
		if !fs.ok {
			continue
		}
		violations += fs.violations
		criticals += fs.criticals
		scoreSum += fs.score
		scored++
	}
	switch {
	case criticals > 0:
		badge.NullSafetyStatus = "critical"
	case violations > 0:
		badge.NullSafetyStatus = "warnings"
	}
	avg := 0.0
	if scored > 0 {
		avg = scoreSum / float64(scored)
	}
	badge.Summary = fmt.Sprintf("%d step(s) across %d file(s): null safety %s (avg score %.0f), %d allocation site(s) spotted",
		len(nodes), len(files), badge.NullSafetyStatus, avg, badge.AllocMetrics.Total())
	return badge
}

// fileSafety is the shared per-unit safety result used by the chat
// server (whole files) and the visualizer (code chunks).
type fileSafety struct {
	score      float64
	nullChecks int
	defers     int
	violations int
	criticals  int
	ok         bool
}

// analyzeParsedFile runs the symbolic prover over one parsed file and
// counts explicit nil checks and deferred Close calls. Type checking
// is best-effort: type errors are swallowed and the prover degrades
// gracefully without type info.
func analyzeParsedFile(fset *token.FileSet, f *ast.File) fileSafety {
	info := &types.Info{
		Defs:  make(map[*ast.Ident]types.Object),
		Uses:  make(map[*ast.Ident]types.Object),
		Types: make(map[ast.Expr]types.TypeAndValue),
	}
	cfg := types.Config{Importer: importer.Default(), Error: func(error) {}}
	_, _ = cfg.Check("chat_safety", fset, []*ast.File{f}, info)
	report, err := analysis.ProveSafetyWithFileSet(f, info, fset)
	if err != nil {
		return fileSafety{}
	}
	var nullChecks, defers, criticals int
	ast.Inspect(f, func(n ast.Node) bool {
		switch t := n.(type) {
		case *ast.BinaryExpr:
			if (t.Op == token.EQL || t.Op == token.NEQ) && (isNilIdent(t.X) || isNilIdent(t.Y)) {
				nullChecks++
			}
		case *ast.DeferStmt:
			if isCloseCall(t.Call) {
				defers++
			}
		}
		return true
	})
	for _, v := range report.Violations {
		if v.Severity == analysis.SeverityCritical {
			criticals++
		}
	}
	return fileSafety{
		score:      report.SafetyScore,
		nullChecks: nullChecks,
		defers:     defers,
		violations: len(report.Violations),
		criticals:  criticals,
		ok:         true,
	}
}

// analyzeNodeChunk parses one node's code as a standalone snippet
// (best-effort) and returns its safety facts plus allocation-site
// counts.
func analyzeNodeChunk(n memory.CodeNode) (fileSafety, AllocMetrics) {
	var alloc AllocMetrics
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, n.SymbolName+".go", "package chatviz\n"+n.CodeContent, 0)
	if err != nil {
		return fileSafety{}, alloc
	}
	ast.Inspect(f, func(nd ast.Node) bool {
		switch t := nd.(type) {
		case *ast.CallExpr:
			if id, ok := t.Fun.(*ast.Ident); ok {
				switch id.Name {
				case "make":
					alloc.MakeCalls++
				case "new":
					alloc.NewCalls++
				case "append":
					alloc.Appends++
				}
			}
		case *ast.CompositeLit:
			alloc.CompositeLits++
		}
		return true
	})
	return analyzeParsedFile(fset, f), alloc
}

func isNilIdent(e ast.Expr) bool {
	id, ok := e.(*ast.Ident)
	return ok && id.Name == "nil"
}

// isCloseCall reports whether call is of the form x.Close() or
// x.Body.Close().
func isCloseCall(call *ast.CallExpr) bool {
	if call == nil {
		return false
	}
	sel, ok := call.Fun.(*ast.SelectorExpr)
	return ok && sel.Sel.Name == "Close"
}

func displayName(n memory.CodeNode) string {
	if n.SymbolName != "" {
		return n.SymbolName
	}
	if len(n.ID) > 12 {
		return n.ID[:12]
	}
	return n.ID
}

func kindWord(kind string) string {
	switch kind {
	case "struct":
		return "Struct"
	case "interface":
		return "Interface"
	case "method":
		return "Method"
	default:
		return "Function"
	}
}

func stepSubtitle(n memory.CodeNode, calleeIDs []string, graph *memory.KnowledgeGraph) string {
	var sb strings.Builder
	sb.WriteString(kindWord(n.Kind))
	if n.PackageName != "" {
		sb.WriteString(" in package " + n.PackageName)
	}
	if len(calleeIDs) == 0 {
		sb.WriteString(" · leaf — this is where the actual work happens")
		return sb.String()
	}
	names := make([]string, 0, len(calleeIDs))
	for _, id := range calleeIDs {
		if cn, ok := graph.GetNode(id); ok {
			names = append(names, displayName(cn))
		}
	}
	sort.Strings(names)
	const maxNames = 4
	shown := names
	more := ""
	if len(names) > maxNames {
		shown = names[:maxNames]
		more = fmt.Sprintf(" (+%d more)", len(names)-maxNames)
	}
	sb.WriteString(" · calls: " + strings.Join(shown, ", ") + more)
	return sb.String()
}

// snippet returns the first maxLines of content, truncated.
func snippet(content string, maxLines int) string {
	lines := strings.Split(strings.TrimSpace(content), "\n")
	if len(lines) > maxLines {
		lines = append(lines[:maxLines], "...")
	}
	s := strings.Join(lines, "\n")
	if len(s) > 1200 {
		s = s[:1197] + "..."
	}
	return s
}

func dedupStrings(in []string) []string {
	seen := make(map[string]bool, len(in))
	out := in[:0]
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

/*
Runnable example: build a tiny in-memory call graph (as if parsed from
AST chunks) and generate the visual payload for an entry point.

package main

import (
	"encoding/json"
	"fmt"

	"github.com/golangast/gollemer/pkg/chat"
	"github.com/golangast/gollemer/pkg/memory"
)

func main() {
	nodes := map[string]memory.CodeNode{
		"h": {ID: "h", SymbolName: "HandleOrder", Kind: "function", PackageName: "api", FilePath: "api/order.go",
			CodeContent: "func HandleOrder(id string) {\n\tsvc := NewOrderService()\n\torder, err := svc.Find(id)\n\t_ = order\n\t_ = err\n}"},
		"s": {ID: "s", SymbolName: "(*OrderService).Find", Kind: "method", PackageName: "api", FilePath: "api/service.go",
			CodeContent: "func (s *OrderService) Find(id string) (Order, error) {\n\treturn s.repo.Query(id)\n}"},
		"q": {ID: "q", SymbolName: "(*OrderRepo).Query", Kind: "method", PackageName: "api", FilePath: "api/repo.go",
			DocComment:  "Query fetches one order by id.",
			CodeContent: "func (r *OrderRepo) Query(id string) (Order, error) {\n\trows := make([]Order, 0, 1)\n\t_ = rows\n\treturn Order{}, nil\n}"},
	}
	graph := &memory.KnowledgeGraph{
		Nodes: nodes,
		Edges: []memory.CodeEdge{
			{SourceID: "h", TargetID: "s", Relation: memory.RelationCalls},
			{SourceID: "s", TargetID: "q", Relation: memory.RelationCalls},
		},
	}
	payload, err := chat.BuildExecutionSequence(graph, "HandleOrder")
	if err != nil {
		panic(err)
	}
	out, _ := json.MarshalIndent(payload, "", "  ")
	fmt.Println(string(out))
}
*/
