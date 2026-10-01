// Package chat serves an HTTP chat interface that lets beginner
// developers query a Go repository in plain English and receive an
// easy-to-understand explanation alongside an interactive visual
// diagram of the relevant code.
//
// NewServer indexes a Go module once (type-resolved package context
// via pkg/ast, hybrid Graph + Vector memory via pkg/memory). Each
// POST to /api/chat embeds the query, pulls the top-K seed nodes plus
// their depth-2 graph neighborhood via memory.QueryContext, and
// responds with:
//
//   - explanation: beginner-friendly prose describing what the
//     matched structs, interfaces, and functions do, how they connect,
//     and where to start reading.
//   - nodes/edges: a JSON diagram payload (structs, interfaces,
//     functions; CALLS, IMPLEMENTS, and USES relations).
//   - safetySummary: counts of explicit nil checks and deferred closes
//     plus the symbolic safety score from prover.ProveSafety.
//
// GET / serves a small interactive chat page that draws the diagram
// as SVG. Everything is standard library (net/http, encoding/json).
package chat

import (
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"net/http"
	"path/filepath"
	"sort"
	"strings"

	gast "github.com/golangast/gollemer/pkg/ast"
	"github.com/golangast/gollemer/pkg/memory"
)

const (
	defaultTopK    = 5
	maxTopK        = 20
	graphDepth     = 2 // BFS expansion over CALLS/IMPLEMENTS, both directions
	embedDim       = 256
	maxBodyBytes   = 1 << 20
	maxSafetyFiles = 8 // cap per-request safety analysis for latency
)

// ChatRequest is the /api/chat request body.
type ChatRequest struct {
	Query  string `json:"query"`
	Target string `json:"target,omitempty"` // optional file or package to focus, e.g. "pkg/memory" or "server.go"
	TopK   int    `json:"topK,omitempty"`
}

// VisualNode is one diagram node.
type VisualNode struct {
	ID      string `json:"id"`
	Label   string `json:"label"`
	Kind    string `json:"kind"` // "struct", "interface", "function"
	Package string `json:"package"`
}

// VisualEdge is one directed diagram edge.
type VisualEdge struct {
	Source   string `json:"source"`
	Target   string `json:"target"`
	Relation string `json:"relation"` // "CALLS", "IMPLEMENTS", "USES"
}

// SafetySummary aggregates symbolic safety facts over the files behind
// the answer. VerifiedNullChecks counts explicit nil comparisons
// (x == nil / x != nil); ResourceDefers counts defer f.Close()
// statements; SafetyScore is the mean prover.ProveSafety score.
type SafetySummary struct {
	FilesAnalyzed      int     `json:"filesAnalyzed"`
	VerifiedNullChecks int     `json:"verifiedNullChecks"`
	ResourceDefers     int     `json:"resourceDefers"`
	SafetyScore        float64 `json:"safetyScore"`
	Violations         int     `json:"violations"`
}

// ChatResponse is the /api/chat response body.
type ChatResponse struct {
	Explanation   string        `json:"explanation"`
	Nodes         []VisualNode  `json:"nodes"`
	Edges         []VisualEdge  `json:"edges"`
	SafetySummary SafetySummary `json:"safetySummary"`
}

// Server is a chat server bound to one indexed Go module.
type Server struct {
	dir   string
	graph *memory.KnowledgeGraph
	mux   *http.ServeMux
}

// NewServer indexes the Go module at dir and returns a chat server for
// it. dir must be (or be inside) a Go module; *_test.go, vendor/, and
// .git/ are skipped so diagrams stay focused on the real code.
func NewServer(dir string) (*Server, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, fmt.Errorf("chat: resolve dir: %w", err)
	}
	codeCtx, err := gast.LoadPackageContext(abs)
	if err != nil {
		return nil, fmt.Errorf("chat: load package context: %w", err)
	}
	chunks := indexChunks(abs, 1<<20)
	if len(chunks) == 0 {
		return nil, fmt.Errorf("chat: no Go code chunks indexed under %s", abs)
	}
	graph, err := memory.BuildGraph(codeCtx, chunks)
	if err != nil {
		return nil, fmt.Errorf("chat: build graph: %w", err)
	}
	for id := range graph.Nodes {
		node, ok := graph.GetNode(id)
		if !ok {
			continue
		}
		graph.SetEmbedding(id, memory.EmbedText(node.CodeContent+"\n"+node.DocComment, embedDim))
	}
	s := &Server{dir: abs, graph: graph, mux: http.NewServeMux()}
	s.mux.HandleFunc("/api/chat", s.handleChat)
	s.mux.HandleFunc("/", s.handleIndex)
	return s, nil
}

// Handler returns the server's HTTP handler for http.ListenAndServe.
func (s *Server) Handler() http.Handler { return s.mux }

// indexChunks parses every non-test .go file under dir into semantic
// chunks, skipping vendor/ and .git/. Unparseable files are skipped.
func indexChunks(dir string, max int) []gast.CodeChunk {
	var chunks []gast.CodeChunk
	_ = filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" || d.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		name := d.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			return nil
		}
		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			return nil
		}
		rel, _ := filepath.Rel(dir, path)
		cs, err := gast.ChunkFile(fset, f, rel)
		if err != nil {
			return nil
		}
		chunks = append(chunks, cs...)
		return nil
	})
	if len(chunks) > max {
		chunks = chunks[:max]
	}
	return chunks
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) handleIndex(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(indexHTML))
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "use POST"})
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid JSON body"})
		return
	}
	query := strings.TrimSpace(req.Query)
	if query == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "query is required"})
		return
	}
	topK := req.TopK
	if topK <= 0 {
		topK = defaultTopK
	}
	if topK > maxTopK {
		topK = maxTopK
	}
	resp, err := s.answer(query, strings.TrimSpace(req.Target), topK)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// answer runs retrieval, explanation, diagram, and safety analysis for
// one query.
func (s *Server) answer(query, target string, topK int) (*ChatResponse, error) {
	nodes, err := memory.QueryContext(s.graph, memory.EmbedText(query, embedDim), topK, graphDepth)
	if err != nil {
		return nil, fmt.Errorf("chat: query context: %w", err)
	}
	if target != "" {
		filtered := filterByTarget(nodes, target)
		if len(filtered) == 0 {
			return &ChatResponse{
				Explanation: fmt.Sprintf("I couldn't find anything under %q matching %q.\n\nTry asking without the target, or check the file/package name — the diagram below is empty because nothing matched.", target, query),
				Nodes:       []VisualNode{},
				Edges:       []VisualEdge{},
			}, nil
		}
		if len(filtered) < len(nodes) {
			nodes = filtered
		}
	}
	if len(nodes) == 0 {
		return &ChatResponse{
			Explanation: fmt.Sprintf("I couldn't find any code related to %q in this repository. Try different words — for example, name a struct, function, or file you are curious about.", query),
			Nodes:       []VisualNode{},
			Edges:       []VisualEdge{},
		}, nil
	}
	vnodes := toVisualNodes(nodes)
	vedges := s.visualEdges(vnodes)
	return &ChatResponse{
		Explanation:   s.explain(query, nodes, vedges),
		Nodes:         vnodes,
		Edges:         vedges,
		SafetySummary: s.safetySummary(nodes),
	}, nil
}

// filterByTarget keeps nodes under a file path, directory, package, or
// symbol name.
func filterByTarget(nodes []memory.CodeNode, target string) []memory.CodeNode {
	var out []memory.CodeNode
	for _, n := range nodes {
		if strings.Contains(n.FilePath, target) ||
			n.PackageName == target ||
			strings.Contains(n.SymbolName, target) {
			out = append(out, n)
		}
	}
	return out
}

// toVisualNodes converts code nodes to diagram nodes. The "method"
// kind folds into "function" (the receiver stays in the label, e.g.
// "(*Store).Get") to keep the diagram's kind set to
// struct/interface/function.
func toVisualNodes(nodes []memory.CodeNode) []VisualNode {
	out := make([]VisualNode, 0, len(nodes))
	for _, n := range nodes {
		kind := n.Kind
		if kind == "method" {
			kind = "function"
		}
		label := n.SymbolName
		if label == "" {
			label = n.ID
			if len(label) > 12 {
				label = label[:12]
			}
		}
		out = append(out, VisualNode{
			ID:      n.ID,
			Label:   label,
			Kind:    kind,
			Package: n.PackageName,
		})
	}
	return out
}

// mapRelation folds the indexer's relation set onto the diagram's
// CALLS/IMPLEMENTS/USES vocabulary: DEPENDS_ON and INSTANTIATES both
// mean "uses" in beginner terms.
func mapRelation(rel string) string {
	switch rel {
	case memory.RelationCalls:
		return "CALLS"
	case memory.RelationImplements:
		return "IMPLEMENTS"
	case "DEPENDS_ON", "INSTANTIATES":
		return "USES"
	default:
		return ""
	}
}

// visualEdges returns the diagram edges whose endpoints are all in the
// answer's node set, sorted for stable output.
func (s *Server) visualEdges(vnodes []VisualNode) []VisualEdge {
	in := make(map[string]bool, len(vnodes))
	for _, n := range vnodes {
		in[n.ID] = true
	}
	out := []VisualEdge{}
	for _, e := range s.graph.Edges {
		if !in[e.SourceID] || !in[e.TargetID] {
			continue
		}
		rel := mapRelation(e.Relation)
		if rel == "" {
			continue
		}
		out = append(out, VisualEdge{Source: e.SourceID, Target: e.TargetID, Relation: rel})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Source != out[j].Source {
			return out[i].Source < out[j].Source
		}
		if out[i].Target != out[j].Target {
			return out[i].Target < out[j].Target
		}
		return out[i].Relation < out[j].Relation
	})
	return out
}

// explain renders the beginner-friendly prose answer: an overview, a
// plain-English tour of the top matches, how they connect, and where
// to start reading.
func (s *Server) explain(query string, nodes []memory.CodeNode, edges []VisualEdge) string {
	var sb strings.Builder
	pkgs := map[string]bool{}
	for _, n := range nodes {
		pkgs[n.PackageName] = true
	}
	noun := "piece"
	if len(nodes) != 1 {
		noun = "pieces"
	}
	fmt.Fprintf(&sb, "I found %d %s of code related to %q", len(nodes), noun, query)
	switch len(pkgs) {
	case 1:
		for p := range pkgs {
			fmt.Fprintf(&sb, " in package %q", p)
		}
	default:
		names := make([]string, 0, len(pkgs))
		for p := range pkgs {
			names = append(names, p)
		}
		sort.Strings(names)
		fmt.Fprintf(&sb, " across packages %s", strings.Join(names, ", "))
	}
	sb.WriteString(".\n\n")

	labels := make(map[string]string, len(nodes))
	for _, n := range nodes {
		labels[n.ID] = n.SymbolName
	}
	limit := 3
	if len(nodes) < limit {
		limit = len(nodes)
	}
	for i := 0; i < limit; i++ {
		sb.WriteString(FormatBeginnerExplanation(nodes[i]))
		sb.WriteString("\n\n")
	}

	if len(edges) > 0 {
		sb.WriteString("How they fit together:\n")
		shown := 0
		for _, e := range edges {
			if shown >= 8 {
				break
			}
			src, ok1 := labels[e.Source]
			dst, ok2 := labels[e.Target]
			if !ok1 || !ok2 {
				continue
			}
			sb.WriteString("• " + describeEdge(src, dst, e.Relation) + "\n")
			shown++
		}
		sb.WriteString("\n")
	}

	var funcs []memory.CodeNode
	for _, n := range nodes {
		if n.Kind == "function" || n.Kind == "method" {
			funcs = append(funcs, n)
		}
	}
	if len(funcs) > 0 {
		sb.WriteString("Where to start reading (in this order):\n")
		for i, f := range funcs {
			if i >= 5 {
				break
			}
			fmt.Fprintf(&sb, "%d. %s  (%s)\n", i+1, f.SymbolName, f.FilePath)
		}
	}
	return sb.String()
}

func describeEdge(src, dst, rel string) string {
	switch rel {
	case "CALLS":
		return fmt.Sprintf("%s calls %s — reading %s will lead you into %s.", src, dst, src, dst)
	case "IMPLEMENTS":
		return fmt.Sprintf("%s implements the %s interface — %s is a cable that fits %s's socket.", src, dst, src, dst)
	default: // USES
		return fmt.Sprintf("%s uses %s — it depends on %s to do its job.", src, dst, dst)
	}
}

// FormatBeginnerExplanation translates one code node into
// beginner-friendly prose using everyday analogies.
func FormatBeginnerExplanation(node memory.CodeNode) string {
	name := node.SymbolName
	if name == "" {
		name = node.ID
	}
	var sb strings.Builder
	switch node.Kind {
	case "struct":
		fmt.Fprintf(&sb, "%q is a struct — think of it as a form that bundles related pieces of data under one name. ", name)
		sb.WriteString("The struct itself just holds data; the real behavior lives in its methods — the function nodes connected to it in the diagram. Start with those.")
	case "interface":
		fmt.Fprintf(&sb, "%q is an interface — like a plug socket on the wall. ", name)
		sb.WriteString("It doesn't do anything itself; it just describes a shape (a set of method signatures). Any struct whose methods match that shape automatically fits — that's what the IMPLEMENTS arrows in the diagram mean. When you see a function accept this interface as a parameter, it means \"give me any cable that fits this socket\".")
	case "method":
		fmt.Fprintf(&sb, "%s is a method — a recipe that belongs to a particular struct. ", name)
		sb.WriteString("The part in front (like (*Store) in (*Store).Get) tells you whose data it is allowed to work with. Read it when you want to know what that struct can do.")
	default: // "function" and anything else
		fmt.Fprintf(&sb, "%q is a function — a standalone recipe: you hand it some inputs, it runs its steps, and it hands back a result. ", name)
		sb.WriteString("Functions are the verbs of the program; follow the CALLS arrows out of one to see the steps it delegates.")
	}
	if doc := firstSentence(node.DocComment); doc != "" {
		sb.WriteString(" In the author's own words: \"" + doc + "\"")
	}
	return sb.String()
}

// firstSentence extracts the first sentence of a doc comment, stripped
// of comment markers.
func firstSentence(doc string) string {
	var words []string
	for _, line := range strings.Split(doc, "\n") {
		line = strings.TrimSpace(line)
		line = strings.TrimPrefix(line, "//")
		line = strings.TrimPrefix(line, "/*")
		line = strings.TrimSuffix(line, "*/")
		line = strings.TrimPrefix(line, "*")
		line = strings.TrimSpace(line)
		if line != "" {
			words = append(words, line)
		}
	}
	text := strings.Join(words, " ")
	if i := strings.Index(text, ". "); i >= 0 {
		text = text[:i+1]
	}
	text = strings.TrimSpace(text)
	if len(text) > 160 {
		text = text[:157] + "..."
	}
	return text
}

// safetySummary analyzes the distinct files behind the answer's nodes.
func (s *Server) safetySummary(nodes []memory.CodeNode) SafetySummary {
	seen := map[string]bool{}
	var files []string
	for _, n := range nodes {
		if n.FilePath == "" || seen[n.FilePath] {
			continue
		}
		seen[n.FilePath] = true
		files = append(files, n.FilePath)
		if len(files) >= maxSafetyFiles {
			break
		}
	}
	sum := SafetySummary{}
	var scoreSum float64
	for _, rel := range files {
		fs := analyzeFileSafety(filepath.Join(s.dir, rel))
		if !fs.ok {
			continue
		}
		sum.FilesAnalyzed++
		sum.VerifiedNullChecks += fs.nullChecks
		sum.ResourceDefers += fs.defers
		sum.Violations += fs.violations
		scoreSum += fs.score
	}
	if sum.FilesAnalyzed > 0 {
		sum.SafetyScore = scoreSum / float64(sum.FilesAnalyzed)
	}
	return sum
}

// analyzeFileSafety parses one file and runs the shared analyzer over
// it (see visuals.go).
func analyzeFileSafety(path string) fileSafety {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return fileSafety{}
	}
	return analyzeParsedFile(fset, f)
}

// indexHTML is the interactive chat page served at /. It posts
// questions to /api/chat and draws the returned diagram as SVG;
// clicking a node inspects it.
const indexHTML = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Gollemer Code Chat</title>
<style>
body{font-family:system-ui,sans-serif;max-width:920px;margin:2em auto;padding:0 1em;color:#222}
#log{border:1px solid #ddd;border-radius:8px;padding:1em;min-height:160px;max-height:320px;overflow-y:auto;margin-bottom:1em}
.msg{margin-bottom:1em}.q{font-weight:bold}.a{white-space:pre-wrap}
#diagram{border:1px solid #ddd;border-radius:8px;margin-top:1em;background:#fafafa}
form{display:flex;gap:.5em;margin-top:1em}input{flex:1;padding:.6em;font-size:1em}button{padding:.6em 1.2em;font-size:1em}
.node{cursor:pointer}.edge-label{font-size:10px;fill:#666}
#detail{margin-top:.5em;font-size:.9em;color:#444;min-height:1.4em}
.legend{font-size:.85em;color:#666;margin-top:.5em}
</style></head>
<body>
<h1>Gollemer Code Chat</h1>
<p>Ask about this Go repository in plain English. You get a beginner-friendly explanation plus a diagram of the code involved.</p>
<div id="log"></div>
<div id="detail">Click a circle in the diagram to inspect it.</div>
<svg id="diagram" width="880" height="420"></svg>
<div class="legend">Colors: <span style="color:#4a90d9">●</span> struct <span style="color:#9b59b6">●</span> interface <span style="color:#27ae60">●</span> function &nbsp;|&nbsp; Arrows: CALLS (solid), IMPLEMENTS (purple), USES (dashed)</div>
<form id="f"><input id="q" placeholder="e.g. how does saving work?" autocomplete="off"><button>Ask</button></form>
<script>
var svg=document.getElementById('diagram'),log=document.getElementById('log'),detail=document.getElementById('detail');
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function color(k){return k==='struct'?'#4a90d9':(k==='interface'?'#9b59b6':'#27ae60');}
function draw(nodes,edges){
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  var W=880,H=420,cx=W/2,cy=H/2,R=150,pos={},NS='http://www.w3.org/2000/svg';
  nodes.forEach(function(n,i){
    var a=2*Math.PI*i/Math.max(1,nodes.length)-Math.PI/2;
    pos[n.id]={x:cx+R*Math.cos(a),y:cy+R*Math.sin(a)};
  });
  edges.forEach(function(e){
    var p1=pos[e.source],p2=pos[e.target];if(!p1||!p2)return;
    var l=document.createElementNS(NS,'line');
    l.setAttribute('x1',p1.x);l.setAttribute('y1',p1.y);
    l.setAttribute('x2',p2.x);l.setAttribute('y2',p2.y);
    l.setAttribute('stroke',e.relation==='IMPLEMENTS'?'#9b59b6':'#999');
    l.setAttribute('stroke-width','2');
    if(e.relation==='USES')l.setAttribute('stroke-dasharray','5 4');
    svg.appendChild(l);
    var t=document.createElementNS(NS,'text');
    t.setAttribute('x',(p1.x+p2.x)/2+4);t.setAttribute('y',(p1.y+p2.y)/2-4);
    t.setAttribute('class','edge-label');t.textContent=e.relation;svg.appendChild(t);
  });
  nodes.forEach(function(n){
    var p=pos[n.id],g=document.createElementNS(NS,'g');
    g.setAttribute('class','node');
    var c=document.createElementNS(NS,'circle');
    c.setAttribute('cx',p.x);c.setAttribute('cy',p.y);c.setAttribute('r',36);
    c.setAttribute('fill',color(n.kind));c.setAttribute('opacity','0.88');
    g.appendChild(c);
    var t=document.createElementNS(NS,'text');
    t.setAttribute('x',p.x);t.setAttribute('y',p.y+4);
    t.setAttribute('text-anchor','middle');t.setAttribute('font-size','11');t.setAttribute('fill','#fff');
    var label=n.label.length>15?n.label.slice(0,14)+'…':n.label;
    t.textContent=label;g.appendChild(t);
    (function(node){
      g.addEventListener('click',function(){
        detail.textContent=node.kind+': '+node.label+' — package '+node.package;
      });
    })(n);
    svg.appendChild(g);
  });
}
document.getElementById('f').addEventListener('submit',function(ev){
  ev.preventDefault();
  var q=document.getElementById('q').value;if(!q.trim())return;
  log.innerHTML+='<div class="msg q">You: '+esc(q)+'</div>';
  document.getElementById('q').value='';
  fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:q})})
    .then(function(r){return r.json().then(function(d){return {status:r.status,body:d};});})
    .then(function(res){
      var d=res.body;
      if(res.status!==200){log.innerHTML+='<div class="msg a">Error: '+esc(d.error||'unknown')+'</div>';return;}
      log.innerHTML+='<div class="msg a">'+esc(d.explanation)+'</div>';
      log.scrollTop=log.scrollHeight;
      draw(d.nodes||[],d.edges||[]);
      var s=d.safetySummary||{};
      detail.textContent='Safety over '+s.filesAnalyzed+' file(s): '+(s.verifiedNullChecks||0)+' nil checks, '+(s.resourceDefers||0)+' deferred closes, score '+(s.safetyScore||0)+', '+(s.violations||0)+' violation(s). Click a circle to inspect it.';
    });
});
</script>
</body></html>`

/*
Runnable example: index a Go repository and start the chat server on
port 8080. Open http://localhost:8080 in a browser and ask something
like "how does saving work".

package main

import (
	"fmt"
	"net/http"

	"github.com/golangast/gollemer/pkg/chat"
)

func main() {
	srv, err := chat.NewServer("./myrepo") // any local Go module
	if err != nil {
		panic(err)
	}
	fmt.Println("gollemer chat: http://localhost:8080")
	if err := http.ListenAndServe(":8080", srv.Handler()); err != nil {
		panic(err)
	}
}
*/
