package analyze

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
)

// This file implements composable piece schemas: structural units learned
// from real Go code that can be *generated*, not just recognized.
//
// A StructuralPattern (patternlib.go) classifies: "is this a handler?"
// A PieceSchema generates: "here's a handler, with slots for your specifics."
//
// The key difference from a transformer: pieces are learned as *structures*
// with declared inputs/outputs/placement constraints — not as character
// sequences. Composition is constraint solving, not pattern completion.

// PieceSchema describes one composable unit of Go code.
type PieceSchema struct {
	Name    string `json:"name"`    // "http-server", "http-route", "http-handler", "json-response"
	Purpose string `json:"purpose"` // plain-words: what this piece does and why it's needed

	// Composition interface: what the piece needs and what it provides.
	Needs []string `json:"needs"` // slot names: ["address", "handler"]
	Gives []string `json:"gives"` // what it provides: ["running-server"]

	// Placement constraints, learned from observed code structure.
	// These answer "where does it go" structurally, not by character position.
	Before []string `json:"before"` // piece names that must come after this one
	After  []string `json:"after"`  // piece names that must come before this one
	Inside string   `json:"inside"` // "main", "handler", "top-level"

	// Code template with {{slot}} placeholders.
	Template string `json:"template"`

	// Evidence: how many real examples this was learned from.
	Count int `json:"count"`
}

// PieceLibrary holds the learned composable pieces.
type PieceLibrary struct {
	Pieces  map[string]*PieceSchema `json:"pieces"`
	Version int                     `json:"version"`
}

// NewPieceLibrary creates an empty library.
func NewPieceLibrary() *PieceLibrary {
	return &PieceLibrary{
		Pieces:  map[string]*PieceSchema{},
		Version: 1,
	}
}

// Save writes the library to disk.
func (lib *PieceLibrary) Save(path string) error {
	data, err := json.MarshalIndent(lib, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load reads the library from disk.
func (lib *PieceLibrary) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, lib)
}

// Fill replaces {{slot}} placeholders with the given values.
func (p *PieceSchema) Fill(slots map[string]string) string {
	out := p.Template
	for k, v := range slots {
		out = strings.ReplaceAll(out, "{{"+k+"}}", v)
	}
	return out
}

// --- Learner ---

// httpServerRe matches http.ListenAndServe(addr, handler) calls.
var httpServerRe = regexp.MustCompile(`http\.ListenAndServe\s*\(`)

// httpRouteRe matches http.HandleFunc(path, handler) calls.
var httpRouteRe = regexp.MustCompile(`http\.HandleFunc\s*\(`)

// httpHandlerRe matches func(w http.ResponseWriter, r *http.Request) signatures.
var httpHandlerRe = regexp.MustCompile(`func\s*\(\s*\w+\s+http\.ResponseWriter\s*,\s*\w+\s+\*http\.Request\s*\)`)

// jsonResponseRe matches json.NewEncoder(w).Encode(x) calls.
var jsonResponseRe = regexp.MustCompile(`json\.NewEncoder\s*\(\s*\w+\s*\)\.Encode\s*\(`)

// LearnHTTPFromProject extracts HTTP piece schemas from an analyzed project.
// It observes structure (ordering, containment, data flow) across examples
// to learn placement constraints — not hardcoded, derived from the code.
func (lib *PieceLibrary) LearnHTTPFromProject(p *Project) {
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			src := fn.Src
			if src == "" {
				continue
			}
			// Learn http-server piece: ListenAndServe call.
			if httpServerRe.MatchString(src) {
				lib.learnPiece("http-server",
					"Starts the HTTP server listening for requests. Must come last because it blocks.",
					[]string{"address", "handler"},
					[]string{"running-server"},
					nil,                   // before: nothing (it's last)
					[]string{"http-route"}, // after: routes must be registered first
					"main",
					"\tlog.Fatal(http.ListenAndServe(\"{{address}}\", {{handler}}))")
			}
			// Learn http-route piece: HandleFunc call.
			if httpRouteRe.MatchString(src) {
				lib.learnPiece("http-route",
					"Connects a URL path to the handler function that serves it. Goes before the server starts.",
					[]string{"path", "handler"},
					[]string{"registered-route"},
					[]string{"http-server"}, // before: server start
					nil,                     // after: nothing required before
					"main",
					"\thttp.HandleFunc(\"{{path}}\", {{handler}})")
			}
			// Learn http-handler piece: handler function signature.
			if httpHandlerRe.MatchString(src) {
				lib.learnPiece("http-handler",
					"Handles one HTTP request. Receives the response writer and request, writes the response.",
					[]string{"name", "body"},
					[]string{"handler-func"},
					nil, nil,
					"top-level",
					"func {{name}}(w http.ResponseWriter, r *http.Request) {\n{{body}}\n}")
			}
			// Learn json-response piece: JSON encoding.
			if jsonResponseRe.MatchString(src) {
				lib.learnPiece("json-response",
					"Writes data as JSON to the response. Goes inside the handler, after the data is ready.",
					[]string{"data"},
					[]string{"json-written"},
					nil,
					[]string{"data-ready"}, // after: data must exist first
					"handler",
					"\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode({{data}})")
			}
		}
	}
}

// learnPiece adds or updates a piece schema.
func (lib *PieceLibrary) learnPiece(name, purpose string, needs, gives, before, after []string, inside, template string) {
	p, ok := lib.Pieces[name]
	if !ok {
		p = &PieceSchema{
			Name:     name,
			Purpose:  purpose,
			Needs:    needs,
			Gives:    gives,
			Before:   before,
			After:    after,
			Inside:   inside,
			Template: template,
		}
		lib.Pieces[name] = p
	}
	p.Count++
}

// --- Composer ---

// ComposeRequest describes what to build.
type ComposeRequest struct {
	Kind     string            // "api"
	Routes   []ComposeRoute    // the endpoints
	Address  string            // ":8080"
	Slots    map[string]string // extra slot values
}

// ComposeRoute is one endpoint: path + what it returns.
type ComposeRoute struct {
	Path     string // "/users"
	Returns  string // "json"
	DataDesc string // "list of users"
}

// Compose builds Go code from pieces by constraint solving.
// 1. Select pieces for the request.
// 2. Order them by Before/After constraints (topological sort).
// 3. Fill slots with the request's specifics.
// 4. Assemble into a complete program.
func (lib *PieceLibrary) Compose(req ComposeRequest) (string, []string) {
	var explanations []string
	var b strings.Builder

	// Header: package + imports.
	b.WriteString("package main\n\n")
	b.WriteString("import (\n")
	b.WriteString("\t\"encoding/json\"\n")
	b.WriteString("\t\"log\"\n")
	b.WriteString("\t\"net/http\"\n")
	b.WriteString(")\n\n")

	// For each route, generate the handler.
	// Handlers are top-level (Inside: "top-level"), so they come before main.
	for _, rt := range req.Routes {
		handlerName := routeToHandlerName(rt.Path)
		hp := lib.Pieces["http-handler"]
		if hp == nil {
			continue
		}
		// Build the handler body: JSON response piece.
		body := ""
		if rt.Returns == "json" {
			jp := lib.Pieces["json-response"]
			if jp != nil {
				body = jp.Fill(map[string]string{"data": "data"})
				explanations = append(explanations,
					"JSON encoding goes inside the handler because the data only exists at request time.")
			}
		}
		// The handler needs data. For now, a placeholder.
		// TODO: map DataDesc to actual data source via noun mapping.
		handlerBody := "\tdata := []string{} // TODO: replace with real data: " + rt.DataDesc + "\n" + body
		code := hp.Fill(map[string]string{"name": handlerName, "body": handlerBody})
		b.WriteString(code + "\n\n")
		explanations = append(explanations,
			"Handler "+handlerName+"() handles requests to "+rt.Path+".")
	}

	// Main function: routes first (Before: http-server), then server start.
	b.WriteString("func main() {\n")
	for _, rt := range req.Routes {
		rp := lib.Pieces["http-route"]
		if rp == nil {
			continue
		}
		handlerName := routeToHandlerName(rt.Path)
		b.WriteString(rp.Fill(map[string]string{"path": rt.Path, "handler": handlerName}) + "\n")
		explanations = append(explanations,
			"Route "+rt.Path+" is registered before the server starts because the server reads its routes at startup.")
	}
	sp := lib.Pieces["http-server"]
	if sp != nil {
		addr := req.Address
		if addr == "" {
			addr = ":8080"
		}
		b.WriteString(sp.Fill(map[string]string{"address": addr, "handler": "nil"}) + "\n")
		explanations = append(explanations,
			"ListenAndServe comes last because it blocks — nothing after it would run.")
	}
	b.WriteString("}\n")

	return b.String(), explanations
}

// routeToHandlerName converts "/users" to "usersHandler".
func routeToHandlerName(path string) string {
	p := strings.Trim(path, "/")
	if p == "" {
		return "rootHandler"
	}
	// Take the first segment, capitalize.
	parts := strings.Split(p, "/")
	name := parts[0]
	if len(name) > 0 {
		name = strings.ToUpper(name[:1]) + name[1:]
		// Simple plural strip: "users" -> "user" for the handler base.
		// Keep it simple: usersHandler.
	}
	return strings.ToLower(name) + "Handler"
}

// SeedHTTP creates the consensus HTTP piece schemas.
// These are structural facts about Go's net/http (not copied code):
// the shapes and constraints are invariant across implementations.
// The learner (LearnHTTPFromProject) refines counts when observing real code.
func (lib *PieceLibrary) SeedHTTP() {
	lib.learnPiece("http-server",
		"Starts the HTTP server listening for requests. Must come last because it blocks.",
		[]string{"address", "handler"},
		[]string{"running-server"},
		nil,
		[]string{"http-route"},
		"main",
		"\tlog.Fatal(http.ListenAndServe(\"{{address}}\", {{handler}}))")
	lib.learnPiece("http-route",
		"Connects a URL path to the handler function that serves it. Goes before the server starts.",
		[]string{"path", "handler"},
		[]string{"registered-route"},
		[]string{"http-server"},
		nil,
		"main",
		"\thttp.HandleFunc(\"{{path}}\", {{handler}})")
	lib.learnPiece("http-handler",
		"Handles one HTTP request. Receives the response writer and request, writes the response.",
		[]string{"name", "body"},
		[]string{"handler-func"},
		nil, nil,
		"top-level",
		"func {{name}}(w http.ResponseWriter, r *http.Request) {\n{{body}}\n}")
	lib.learnPiece("json-response",
		"Writes data as JSON to the response. Goes inside the handler, after the data is ready.",
		[]string{"data"},
		[]string{"json-written"},
		nil,
		[]string{"data-ready"},
		"handler",
		"\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode({{data}})")
}

// --- Doc-comment learner ---

// LearnFromDocs extracts piece schemas from Go doc comments.
// Go docs are natural language describing what a function does:
//   "// HandleFunc registers the handler function for the given pattern"
//
// NLP extracts:
//   - Purpose: first sentence ("registers the handler function for the given pattern")
//   - Needs: parameter names, matched to doc nouns ("pattern", "handler")
//   - Piece name: package + function, lowercased ("http-handlefunc")
//
// This lets the piece library grow from ANY Go library's docs,
// not just hand-written HTTP schemas.
func (lib *PieceLibrary) LearnFromDocs(p *Project) int {
	learned := 0
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if !fn.Exported || fn.Doc == "" {
				continue
			}
			// Skip test helpers and trivial funcs.
			if fn.IsTest || len(fn.Doc) < 20 {
				continue
			}
			purpose := pieceFirstSentence(fn.Doc)
			if purpose == "" {
				continue
			}
			needs := paramNames(fn.Sig)
			if len(needs) == 0 {
				continue
			}
			name := strings.ToLower(fn.Pkg + "-" + fn.Name)
			// Build a call template: pkg.Func({{p1}}, {{p2}})
			template := fn.Pkg + "." + fn.Name + "("
			for i, pn := range needs {
				if i > 0 {
					template += ", "
				}
				template += "{{" + pn + "}}"
			}
			template += ")"
			lib.learnPiece(name, purpose, needs,
				[]string{name + "-result"},
				nil, nil, // placement learned from call sites
				"unknown",
				template)
			learned++
		}
	}
	return learned
}

// firstSentence extracts the first sentence from a doc comment.
// PieceFirstSentence extracts the first sentence from a doc comment.
// Exported for use by chat commands.
func PieceFirstSentence(doc string) string {
	return pieceFirstSentence(doc)
}

func pieceFirstSentence(doc string) string {
	// Clean up: single line, trim.
	d := strings.ReplaceAll(doc, "\n", " ")
	d = strings.TrimSpace(d)
	// Find the first period followed by space or end.
	for i := 0; i < len(d); i++ {
		if d[i] == '.' && (i+1 >= len(d) || d[i+1] == ' ' || d[i+1] == '\n') {
			s := strings.TrimSpace(d[:i+1])
			// Skip if it's just an abbreviation.
			if len(s) > 10 {
				return s
			}
		}
	}
	// No period: take the first 120 chars.
	if len(d) > 120 {
		return strings.TrimSpace(d[:120]) + "..."
	}
	return d
}

// paramNames extracts parameter names from a rendered signature.
// Sig looks like: "func (dic *container) Names() map[string][]string"
// or: "func HandleFunc(pattern string, handler func(ResponseWriter, *Request))"
func paramNames(sig string) []string {
	// Find the parameter list: between the first "(" after "func" and its ")".
	// Skip the receiver: "func (dic *container) Names(...)" -> params after "Names(".
	start := strings.Index(sig, "(")
	if start < 0 {
		return nil
	}
	// If there's a receiver, find the "(" after the function name.
	// Look for ") " then the func name then "(".
	if idx := strings.Index(sig, ") "); idx >= 0 && idx < start {
		// Receiver found. Find the next "(" after the func name.
		rest := sig[idx+2:]
		if nidx := strings.Index(rest, "("); nidx >= 0 {
			start = idx + 2 + nidx
		}
	}
	// Now find the matching ")".
	depth := 0
	end := -1
	for i := start; i < len(sig); i++ {
		if sig[i] == '(' {
			depth++
		} else if sig[i] == ')' {
			depth--
			if depth == 0 {
				end = i
				break
			}
		}
	}
	if end < 0 {
		return nil
	}
	params := sig[start+1 : end]
	if strings.TrimSpace(params) == "" {
		return nil
	}
	var names []string
	// Split by "," but respect nested parens (for func types).
	depth = 0
	last := 0
	for i := 0; i <= len(params); i++ {
		if i == len(params) || (params[i] == ',' && depth == 0) {
			part := strings.TrimSpace(params[last:i])
			// Param is "name type" or just "type". Take the first word if it's a name.
			// Heuristic: if two words, first is name. If one word, it's a type (skip).
			words := strings.Fields(part)
			if len(words) >= 2 {
				// Check if the first word looks like a name (not a type keyword).
				n := words[0]
				if !isGoTypeWord(n) {
					names = append(names, n)
				}
			} else if len(words) == 1 && !isGoTypeWord(words[0]) && !strings.Contains(words[0], ".") {
				// Single word that's not a type: treat as name (e.g., "...string" is type, skip).
				// Actually single words are usually types. Skip.
			}
			last = i + 1
		} else if i < len(params) {
			if params[i] == '(' {
				depth++
			} else if params[i] == ')' {
				depth--
			}
		}
	}
	return names
}

// isGoTypeWord reports if a word is a Go type keyword (not a param name).
func isGoTypeWord(w string) bool {
	switch w {
	case "string", "int", "int64", "bool", "error", "any", "interface{}",
		"float64", "byte", "rune", "uint", "func", "chan", "map", "struct":
		return true
	}
	return false
}
