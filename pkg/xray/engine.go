// Package xray is an interactive "X-Ray Assistant" for Go codebases.
// It translates natural-language queries into idiomatic Go code,
// explains the result in plain English, maps Go primitives to
// real-world analogies, and renders structured visual execution
// traces from AST call graphs. Standard library only, plus the
// repo's own beginner (code templates) and analysis (safety prover)
// packages.
package xray

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"regexp"
	"sort"
	"strings"

	"github.com/golangast/gollemer/pkg/analysis"
	"github.com/golangast/gollemer/pkg/beginner"
)

// XRayRequest is one assistant query: what to build or trace, an
// optional package hint, and optional Go source to trace through.
type XRayRequest struct {
	Query         string `json:"query"`
	TargetPackage string `json:"targetPackage"`
	ContextCode   string `json:"contextCode"`
}

// ExecutionStep is one frame of a visual execution trace.
type ExecutionStep struct {
	StepNumber  int    `json:"stepNumber"`
	Title       string `json:"title"`
	Subtitle    string `json:"subtitle"`
	FilePath    string `json:"filePath"`
	Description string `json:"description"`
	CodeSnippet string `json:"codeSnippet"`
}

// SafetyBadge is a compact safety verdict for generated/traced code.
type SafetyBadge struct {
	NullSafetyStatus string `json:"nullSafetyStatus"` // clean, warnings, or critical
	AllocEstimate    string `json:"allocEstimate"`    // e.g. "low — 2 allocation sites"
}

// XRayResponse is the full assistant answer.
type XRayResponse struct {
	GeneratedCode       string          `json:"generatedCode"`
	BeginnerExplanation string          `json:"beginnerExplanation"`
	Analogy             string          `json:"analogy"`
	VisualSequence      []ExecutionStep `json:"visualSequence"`
	SafetyBadge         SafetyBadge     `json:"safetyBadge"`
}

// Engine is the X-Ray assistant. Construct with NewEngine; the
// package-level functions expose the same operations for callers
// that prefer plain functions.
type Engine struct{}

// NewEngine returns a ready-to-use X-Ray assistant.
func NewEngine() *Engine { return &Engine{} }

// SynthesizeWithXRay answers req: it either generates idiomatic Go
// from the natural-language query, or — when the query asks to trace
// a flow and ContextCode is supplied — traces the call path through
// the provided code.
func (e *Engine) SynthesizeWithXRay(ctx context.Context, req XRayRequest) (*XRayResponse, error) {
	return SynthesizeWithXRay(ctx, req)
}

// TraceExecutionPath renders the call hierarchy under entrySymbol as
// an ordered visual sequence.
func (e *Engine) TraceExecutionPath(file *ast.File, entrySymbol string) ([]ExecutionStep, error) {
	return TraceExecutionPath(file, entrySymbol)
}

// GetConceptAnalogy maps a Go primitive to a real-world mental model.
func (e *Engine) GetConceptAnalogy(goPrimitive string) string {
	return GetConceptAnalogy(goPrimitive)
}

// SynthesizeWithXRay is the package-level engine entry point; see
// Engine.SynthesizeWithXRay.
func SynthesizeWithXRay(ctx context.Context, req XRayRequest) (*XRayResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("xray: %w", err)
	}
	query := strings.TrimSpace(req.Query)
	if query == "" {
		return nil, fmt.Errorf("xray: empty query")
	}

	if isTraceQuery(query) {
		if strings.TrimSpace(req.ContextCode) == "" {
			return nil, fmt.Errorf("xray: tracing %q needs ContextCode with the Go source to trace through", query)
		}
		return synthesizeTrace(ctx, req)
	}
	return synthesizeCode(ctx, req)
}

var traceVerbs = regexp.MustCompile(`(?i)\b(trace|traces|tracing|flow|flows|call[\s-]?path|walk[\s-]?through|how does|how do)\b`)

// isTraceQuery reports whether the query asks to trace a flow rather
// than generate code.
func isTraceQuery(query string) bool { return traceVerbs.MatchString(query) }

// synthesizeCode generates Go from the query via the beginner
// template engine, then traces the generated program from main.
func synthesizeCode(ctx context.Context, req XRayRequest) (*XRayResponse, error) {
	code, explanation, err := beginner.GenerateGoFromCommand(req.Query)
	if err != nil {
		return nil, fmt.Errorf("xray: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("xray: %w", err)
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "generated.go", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("xray: generated code failed to parse: %w", err)
	}
	steps, err := tracePath(fset, "generated.go", file, "main")
	if err != nil {
		// Generated programs always declare main, but never fail the
		// whole response if tracing hiccups.
		steps = nil
	}
	return &XRayResponse{
		GeneratedCode:       code,
		BeginnerExplanation: explanation,
		Analogy:             GetConceptAnalogy(detectPrimitive(req.Query, code)),
		VisualSequence:      steps,
		SafetyBadge:         buildSafetyBadge(fset, file),
	}, nil
}

// synthesizeTrace parses req.ContextCode and traces the execution
// path for the requested flow.
func synthesizeTrace(ctx context.Context, req XRayRequest) (*XRayResponse, error) {
	name := strings.TrimSpace(req.TargetPackage)
	if name == "" {
		name = "context.go"
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, name, req.ContextCode, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("xray: could not parse ContextCode: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("xray: %w", err)
	}
	entry, err := resolveEntrySymbol(req.Query, declaredFunctions(file), localCallGraph(file))
	if err != nil {
		return nil, err
	}
	steps, err := tracePath(fset, name, file, entry)
	if err != nil {
		return nil, err
	}
	var formatted bytes.Buffer
	if err := format.Node(&formatted, fset, file); err != nil {
		return nil, fmt.Errorf("xray: could not format ContextCode: %w", err)
	}
	titles := make([]string, 0, len(steps))
	for _, s := range steps {
		titles = append(titles, strings.TrimPrefix(s.Title, fmt.Sprintf("Step %d: ", s.StepNumber)))
	}
	expl := fmt.Sprintf("The flow starts at %s and runs through %d step(s): %s. ",
		entry, len(steps), strings.Join(titles, " → "))
	expl += "Each step below shows what that function does and which functions it calls next, " +
		"so you can follow the request exactly the way the computer executes it."
	return &XRayResponse{
		GeneratedCode:       formatted.String(),
		BeginnerExplanation: expl,
		Analogy:             GetConceptAnalogy(detectPrimitive(req.Query, req.ContextCode)),
		VisualSequence:      steps,
		SafetyBadge:         buildSafetyBadge(fset, file),
	}, nil
}

// TraceExecutionPath inspects file with ast.Inspect and renders the
// function call hierarchy starting at entrySymbol as an ordered
// visual sequence (depth-first, pre-order). entrySymbol must name a
// declared function exactly (case-insensitive); anything else is an
// error listing the available functions.
//
// The file is re-parsed from its gofmt rendering so positions are
// self-consistent; FilePath is therefore "snippet.go". Callers that
// know the real filename should parse with parser.ParseFile
// themselves and use SynthesizeWithXRay, which preserves it.
func TraceExecutionPath(file *ast.File, entrySymbol string) ([]ExecutionStep, error) {
	if file == nil {
		return nil, fmt.Errorf("xray: nil file")
	}
	var buf bytes.Buffer
	probe := token.NewFileSet()
	if err := format.Node(&buf, probe, file); err != nil {
		return nil, fmt.Errorf("xray: could not render file: %w", err)
	}
	fset := token.NewFileSet()
	norm, err := parser.ParseFile(fset, "snippet.go", buf.Bytes(), parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("xray: could not re-parse file: %w", err)
	}
	return tracePath(fset, "snippet.go", norm, entrySymbol)
}

// tracePath is the shared tracer: fset must be the FileSet file was
// parsed with, and filename labels the FilePath of every step.
func tracePath(fset *token.FileSet, filename string, file *ast.File, entrySymbol string) ([]ExecutionStep, error) {
	funcs := map[string]*ast.FuncDecl{}
	var order []string
	ast.Inspect(file, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Body != nil {
			if _, seen := funcs[fn.Name.Name]; !seen {
				order = append(order, fn.Name.Name)
			}
			funcs[fn.Name.Name] = fn
		}
		return true
	})
	if len(order) == 0 {
		return nil, fmt.Errorf("xray: no functions declared in the code")
	}
	entry := ""
	for _, name := range order {
		if name == entrySymbol || strings.EqualFold(name, entrySymbol) {
			entry = name
			break
		}
	}
	if entry == "" {
		return nil, fmt.Errorf("xray: no function %q declared (have: %s)", entrySymbol, strings.Join(order, ", "))
	}

	// Call graph over locally declared functions. Selector calls
	// (svc.Find, fmt.Println) resolve by their base name, so a method
	// call links to a same-named local function when one exists.
	callsOf := localCallGraph(file)

	const maxSteps = 50
	var steps []ExecutionStep
	visited := map[string]bool{}
	var dfs func(name string)
	dfs = func(name string) {
		if visited[name] || len(steps) >= maxSteps {
			return
		}
		visited[name] = true
		fn := funcs[name]
		n := len(steps) + 1
		human := humanize(name)
		subtitle := "function"
		if fn.Recv != nil && len(fn.Recv.List) > 0 {
			subtitle = "method"
		}
		desc := firstSentence(docText(fn))
		if desc == "" {
			desc = human
			if calls := callsOf[name]; len(calls) > 0 {
				desc += " — calls " + strings.Join(calls, ", ")
			}
			desc += "."
		}
		steps = append(steps, ExecutionStep{
			StepNumber:  n,
			Title:       fmt.Sprintf("Step %d: %s", n, human),
			Subtitle:    subtitle,
			FilePath:    filename,
			Description: desc,
			CodeSnippet: snippetOf(fset, fn, 12),
		})
		for _, c := range callsOf[name] {
			dfs(c)
		}
	}
	dfs(entry)
	return steps, nil
}

// declaredFunctions lists top-level function names in declaration order.
func declaredFunctions(file *ast.File) []string {
	var names []string
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			names = append(names, fn.Name.Name)
		}
	}
	return names
}

var quotedSymbol = regexp.MustCompile("`([^`]+)`|\"([^\"]+)\"")

// resolveEntrySymbol picks the trace entry from the query: a
// quoted/backticked symbol first, then an exact name match, then the
// unique uncalled function (the flow's root), then main, then the
// first declared function.
func resolveEntrySymbol(query string, names []string, graph map[string][]string) (string, error) {
	if len(names) == 0 {
		return "", fmt.Errorf("xray: no functions declared in ContextCode")
	}
	if m := quotedSymbol.FindStringSubmatch(query); m != nil {
		want := m[1]
		if want == "" {
			want = m[2]
		}
		for _, n := range names {
			if n == want {
				return n, nil
			}
		}
		return "", fmt.Errorf("xray: no function %q in ContextCode (have: %s)", want, strings.Join(names, ", "))
	}
	lower := strings.ToLower(query)
	for _, n := range names {
		if strings.ToLower(n) == lower {
			return n, nil
		}
		for _, tok := range strings.FieldsFunc(lower, func(r rune) bool {
			return r < 'a' || r > 'z'
		}) {
			if tok == strings.ToLower(n) {
				return n, nil
			}
		}
	}
	if root, ok := uniqueRoot(graph); ok {
		return root, nil
	}
	for _, n := range names {
		if n == "main" {
			return n, nil
		}
	}
	return names[0], nil
}

// localCallGraph maps each declared function to the locally
// declared functions it calls. Selector calls (svc.Find,
// fmt.Println) resolve by their base name, so a method call links
// to a same-named local function when one exists.
func localCallGraph(file *ast.File) map[string][]string {
	funcs := map[string]*ast.FuncDecl{}
	for _, decl := range file.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Body != nil {
			funcs[fn.Name.Name] = fn
		}
	}
	graph := make(map[string][]string, len(funcs))
	for name, fn := range funcs {
		type hit struct {
			name string
			pos  token.Pos
		}
		seen := map[string]bool{}
		var calls []hit
		ast.Inspect(fn, func(n ast.Node) bool {
			call, ok := n.(*ast.CallExpr)
			if !ok {
				return true
			}
			var cname string
			switch f := call.Fun.(type) {
			case *ast.Ident:
				cname = f.Name
			case *ast.SelectorExpr:
				cname = f.Sel.Name
			}
			if cname == "" || seen[cname] {
				return true
			}
			if _, local := funcs[cname]; !local {
				return true
			}
			seen[cname] = true
			calls = append(calls, hit{cname, call.Pos()})
			return true
		})
		// Source order, so the trace reads like execution.
		sort.Slice(calls, func(i, j int) bool { return calls[i].pos < calls[j].pos })
		names := make([]string, 0, len(calls))
		for _, h := range calls {
			names = append(names, h.name)
		}
		graph[name] = names
	}
	return graph
}

// uniqueRoot returns the single function that no other declared
// function calls — the entry point of the flow.
func uniqueRoot(graph map[string][]string) (string, bool) {
	called := map[string]bool{}
	for _, calls := range graph {
		for _, c := range calls {
			called[c] = true
		}
	}
	var roots []string
	for name := range graph {
		if !called[name] {
			roots = append(roots, name)
		}
	}
	if len(roots) == 1 {
		return roots[0], true
	}
	return "", false
}

// docText returns the doc comment text of fn, or "".
func docText(fn *ast.FuncDecl) string {
	if fn.Doc == nil {
		return ""
	}
	return strings.TrimSpace(fn.Doc.Text())
}

// firstSentence returns the first sentence of s, or "".
func firstSentence(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if i := strings.Index(s, ". "); i >= 0 {
		return s[:i+1]
	}
	if strings.HasSuffix(s, ".") {
		return s
	}
	return s + "."
}

// snippetOf renders the first maxLines of fn's formatted source.
func snippetOf(fset *token.FileSet, fn *ast.FuncDecl, maxLines int) string {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, fn); err != nil {
		return ""
	}
	lines := strings.Split(buf.String(), "\n")
	if len(lines) > maxLines {
		lines = lines[:maxLines]
	}
	return strings.Join(lines, "\n")
}

// humanize turns a camelCase function name into words:
// parseJSONBody -> "Parse JSON body". All-caps runs (JSON, JWT,
// HTTP, DB, URL, ID) are preserved.
func humanize(name string) string {
	var words []string
	runes := []rune(name)
	start := 0
	flush := func(end int) {
		if end > start {
			words = append(words, string(runes[start:end]))
		}
		start = end
	}
	isUpper := func(r rune) bool { return r >= 'A' && r <= 'Z' }
	isLower := func(r rune) bool { return r >= 'a' && r <= 'z' }
	isDigit := func(r rune) bool { return r >= '0' && r <= '9' }
	for i := 1; i < len(runes); i++ {
		prev, cur := runes[i-1], runes[i]
		switch {
		case isLower(prev) && isUpper(cur):
			flush(i)
		case isUpper(prev) && isUpper(cur) && i+1 < len(runes) && isLower(runes[i+1]):
			flush(i)
		case (isDigit(prev) || isDigit(cur)) && isDigit(prev) != isDigit(cur) &&
			(isLower(prev) || isUpper(prev) || isLower(cur) || isUpper(cur)):
			flush(i)
		}
	}
	flush(len(runes))
	for i, w := range words {
		if i == 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
			continue
		}
		if w == strings.ToUpper(w) {
			continue // keep JSON, JWT, ...
		}
		words[i] = strings.ToLower(w)
	}
	return strings.Join(words, " ")
}

// countAllocSites counts static allocation sites: make/new/append
// calls and composite literals.
func countAllocSites(file *ast.File) int {
	n := 0
	ast.Inspect(file, func(x ast.Node) bool {
		switch t := x.(type) {
		case *ast.CallExpr:
			if id, ok := t.Fun.(*ast.Ident); ok {
				switch id.Name {
				case "make", "new", "append":
					n++
				}
			}
		case *ast.CompositeLit:
			n++
		}
		return true
	})
	return n
}

// buildSafetyBadge proves null-safety with the symbolic prover and
// estimates allocation pressure from static allocation sites.
func buildSafetyBadge(fset *token.FileSet, file *ast.File) SafetyBadge {
	status := "clean"
	if rep, err := analysis.ProveSafetyWithFileSet(file, nil, fset); err == nil && rep != nil {
		warn := false
		for _, v := range rep.Violations {
			if v.Severity == analysis.SeverityCritical {
				status = "critical"
				break
			}
			warn = true
		}
		if status != "critical" && warn {
			status = "warnings"
		}
	}
	sites := countAllocSites(file)
	level := "low"
	if sites > 10 {
		level = "high"
	} else if sites > 3 {
		level = "moderate"
	}
	return SafetyBadge{
		NullSafetyStatus: status,
		AllocEstimate:    fmt.Sprintf("%s — %d allocation site(s)", level, sites),
	}
}

// primitiveKeywords maps query/code keywords to analogy primitives,
// in priority order.
var primitiveKeywords = []struct {
	word      string
	primitive string
}{
	{"channel", "channel"},
	{"chan ", "channel"},
	{"goroutine", "goroutine"},
	{"interface", "interface"},
	{"struct", "struct"},
	{"mutex", "mutex"},
	{"defer", "defer"},
	{"slice", "slice"},
	{"map[", "map"},
	{"pointer", "pointer"},
	{"error", "error"},
	{"context", "context"},
}

// detectPrimitive picks the dominant Go primitive mentioned in the
// query or code, defaulting to "function".
func detectPrimitive(query, code string) string {
	hay := strings.ToLower(query + "\n" + code)
	for _, pk := range primitiveKeywords {
		if strings.Contains(hay, pk.word) {
			return pk.primitive
		}
	}
	return "function"
}

// GetConceptAnalogy maps a Go primitive to a real-world mental model.
// Lookup is case-insensitive; unknown primitives get a generic
// builder analogy rather than an empty string.
func GetConceptAnalogy(goPrimitive string) string {
	key := strings.ToLower(strings.TrimSpace(strings.TrimPrefix(goPrimitive, "*")))
	if a, ok := analogies[key]; ok {
		return a
	}
	return "A building block: Go gives you small, sharp tools that combine into larger programs — learn each one in isolation, then compose them."
}

var analogies = map[string]string{
	"interface": "A wall outlet spec: defines what plug shape is required without caring how the electricity is generated.",
	"struct":    "A concrete physical appliance: holds data and implements the plug.",
	"goroutine": "An independent worker at a factory line: lightweight and managed by Go's runtime scheduler.",
	"channel":   "A conveyor belt: moves data safely between workers without needing explicit locks.",
	"slice":     "A stretchy tray: grows with append and always knows how many items it holds.",
	"map":       "A labeled drawer cabinet: ask for a label, get the drawer; a missing label gives you an empty drawer.",
	"pointer":   "A sticky note with a house address: hand someone the note and they can repaint the actual house.",
	"defer":     "A promise to tidy up that runs automatically when the function walks out the door.",
	"mutex":     "A single bathroom key: only the worker holding it can enter the critical section.",
	"error":     "A receipt that says what went wrong: you must read it before continuing with your day.",
	"method":    "A recipe card taped to one appliance: only that appliance can follow it.",
	"function":  "A recipe card: named steps you can run anytime with different ingredients.",
	"package":   "A labeled toolbox: related tools live together, and other toolboxes can borrow them.",
	"context":   "A walkie-talkie passed down the call chain: anyone can announce 'stop, the caller gave up.'",
	"array":     "An egg carton: a fixed number of slots, each holding exactly one egg.",
	"select":    "A receptionist watching several doors: serves whichever visitor arrives first.",
	"waitgroup": "A teacher counting heads on a field trip: nobody leaves until every student is back on the bus.",
}

/*
Runnable example: instantiate the engine, answer a build request and
a trace request, and print the XRayResponse as JSON.

package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/golangast/gollemer/pkg/xray"
)

func main() {
	eng := xray.NewEngine()
	ctx := context.Background()

	build, err := eng.SynthesizeWithXRay(ctx, xray.XRayRequest{
		Query: "Create an HTTP server on port 8080 with a JSON health check",
	})
	if err != nil {
		panic(err)
	}
	out, _ := json.MarshalIndent(build, "", "  ")
	fmt.Println(string(out))

	trace, err := eng.SynthesizeWithXRay(ctx, xray.XRayRequest{
		Query: "Trace how user authentication flows from handler to database",
		ContextCode: `package auth

import "fmt"

func HandleLogin(user string) {
	fmt.Println("login attempt for", user)
	if ValidateJWT(user) {
		QueryDB(user)
	}
}

func ValidateJWT(user string) bool {
	fmt.Println("validating token for", user)
	return user != ""
}

func QueryDB(user string) {
	fmt.Println("querying database for", user)
}
`,
	})
	if err != nil {
		panic(err)
	}
	for _, s := range trace.VisualSequence {
		fmt.Printf("%s — %s\n", s.Title, s.Description)
	}
	fmt.Println("analogy:", eng.GetConceptAnalogy("channel"))
}
*/
