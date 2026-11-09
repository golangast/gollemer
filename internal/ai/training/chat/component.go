package chat

// The component piece (John 2026-11-09, "add the component piece
// to the package"): one Material component added to an existing
// page by sentence — the material template's starter set, one
// component at a time:
//
//	add|create|make|build|put [a|an|the] [material]
//	    <component> to|in|into|at <file.html>
//	add|create|make|build|put [a|an|the] [material]
//	    <component> to the page
//
// <component> is a form from the registry in
// analyze.MaterialComponents (button — the filled one — tonal
// button, outlined button, text button, card, app bar / top
// bar, navigation bar / nav bar / navbar, floating button /
// fab, chip, text field, switch, footer): the
// grammar, the discovery answer, and the planner all read
// that one registry, so they cannot drift apart. "the page"
// resolves through the session context to the last webpage
// the session made or filled, with the reading printed; no
// page in context is honest guidance, never a guess.
//
// The shape claims with an answer even when the name is not
// a component ("add a carousel to …"): the refusal lists
// what exists, from the same registry. Seating keeps the
// neighbours safe: this claim sits after the webpage and
// serve-static claims in both loops, so "add a webpage" and
// the template wordings are already claimed before a name
// is read as a component. A .go target never matches the
// shape at all — the sentence declines untouched, and the
// book's pieces keep their targets exactly as before.
//
// The article is optional in the code as in the grammar
// above: John typed "add card to jim/temp/index.html" live
// (2026-11-09) and a parse that REQUIRED the article
// declined the line to the boundary's can't-assemble
// answer, where "add a card…" landed. Beginners drop
// articles; the shape never needed one.
//
// Assembly is the planner's (analyze.planComponent): the
// snippet lands just before </body>, and the Material
// styles travel with it exactly once — the variables and
// the component's own chunk go into the page's style block
// (created in <head> when the page has none) only when the
// markers say they are missing. The ask runs through the
// edit spine like every other piece: plan, gate,
// transaction, verify, confirmation.
//
// A target can also be a wasm app — its folder, or the
// main.go inside it ("add a card to jim", "add a card to
// jim/main.go"): the app's page is built by Go in the
// browser, so the component lands there as a Go DOM
// builder (parseComponentWasmTarget / runComponentWasm
// below, analyze.planComponentWasm). The routing is by
// content — the folder's main.go must carry the js && wasm
// build tag — and until that read says the folder holds an
// app, the sentence is not this piece's at all.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// componentAsk is one parsed component sentence: the
// registry id ("" when the shape matched but the name is
// not a component — the flow answers with the list), the
// name as typed, and the target: a typed .html file, or
// "the page".
type componentAsk struct {
	component string
	name      string
	file      string
	thePage   bool
}

func parseComponent(line string) (componentAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 4 {
		return componentAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return componentAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++ // the article slot — optional, never required
	}
	if i < len(fields) && low[i] == "material" {
		i++ // the material adjective: "add a material button"
	}
	// The component's name, up to the target's preposition
	// — one to three words ("button", "text field", "app
	// bar"), read against the registry below.
	j := i
	for j < len(fields) && low[j] != "to" && low[j] != "in" && low[j] != "into" && low[j] != "at" {
		j++
	}
	if j >= len(fields) || j == i || j-i > 3 {
		return componentAsk{}, false
	}
	out := componentAsk{name: strings.Join(low[i:j], " ")}
	rest := low[j+1:]
	switch {
	case len(rest) == 2 && rest[0] == "the" && rest[1] == "page":
		out.thePage = true
	case len(rest) == 1 && strings.HasSuffix(rest[0], ".html"):
		out.file = fields[j+1]
	default:
		// No target, a longer target clause, or a target
		// that is not an .html file: not this shape. A .go
		// target in particular declines untouched — the
		// book's pieces keep those sentences.
		return componentAsk{}, false
	}
	if comp, ok := analyze.MaterialComponentByForm(out.name); ok {
		out.component = comp.ID
	}
	return out, true
}

// componentListText is the discovery answer, generated
// from the registry: every component, its note, and one
// example sentence each — the same data the grammar
// matches, so the list cannot promise what doesn't parse.
func componentListText() string {
	var b strings.Builder
	b.WriteString("The components you can add to a page, one at a time — each is a piece of the material template, and its styles come with it the first time:\n")
	for _, c := range analyze.MaterialComponents {
		note := ""
		if c.Note != "" {
			note = " " + c.Note
		}
		fmt.Fprintf(&b, "  %s%s — \"add a %s to jim/temp/index.html\"\n", c.Words, note, c.Words)
	}
	b.WriteString("The component goes in just before the page's </body> — say one of those sentences with your own .html file, or \"to the page\" for the page you last made or filled here.")
	return b.String()
}

// tryComponentList claims the discovery shapes: "what
// components can I add", "what components are there", and
// their close spellings — a closed set, answered from the
// registry. The singular and the page forms are in the set
// because John typed "what component can i add" live
// (2026-11-09) and the line fell past the plural-only set
// to the intent layer, whose symbol route looked "add" up
// as a function name and answered with a pick list of Add
// functions. The claim's seat in both loops is ahead of
// that dispatch, so membership in this set is the whole
// protection — the symbol questions ("what does Add do")
// are not in it and keep their route untouched.
func tryComponentList(line string) bool {
	norm := strings.Join(strings.Fields(strings.ToLower(line)), " ")
	norm = strings.TrimSuffix(strings.TrimSuffix(norm, "?"), ".")
	switch norm {
	case "what components can i add", "what components are there",
		"what components can you add", "what components do you have",
		"what material components can i add", "what material components are there",
		"what component can i add", "what component can you add",
		"what can i add to the page", "what can i add to my page",
		"list components", "list the components":
		fmt.Printf("%s %s\n", cTag("gollemer [composer]>"), componentListText())
		return true
	}
	return false
}

// sessionPage resolves "the page": the session's last
// .html file the context still vouches for — the last file
// first, then the one before it — checked on disk under
// the project root. The reason rides along for the
// printed reading.
func sessionPage(rootAbs string) (rel, reason string, ok bool) {
	if !workContextLive(rootAbs) {
		return "", "", false
	}
	for _, cand := range []struct{ file, how string }{
		{workCtx.lastFile, workCtx.lastHow},
		{workCtx.prevFile, workCtx.prevHow},
	} {
		if cand.file == "" || !strings.HasSuffix(strings.ToLower(cand.file), ".html") {
			continue
		}
		info, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(cand.file)))
		if err != nil || info.IsDir() {
			continue
		}
		switch cand.how {
		case "created":
			reason = "the page you just created"
		case "edited":
			reason = "the page you just edited"
		default:
			reason = "the last page this session touched"
		}
		return cand.file, reason, true
	}
	return "", "", false
}

// tryComponent is the chat-loop entry: it claims component
// sentences and runs the flow on the chat's line reader.
func tryComponent(line string, r *LineReader) bool {
	return runComponent(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// componentWasmAsk is one parsed component sentence aimed
// at a wasm app: the registry id ("" when the name is not
// a component), the name as typed, and the target as typed
// — a folder, or the app's main.go inside one.
type componentWasmAsk struct {
	component string
	name      string
	target    string
	goFile    bool
}

// parseComponentWasmTarget reads the component shape with
// an app target — a folder ("add a card to jim") or the
// folder's main.go ("add a card to jim/main.go"). The page
// targets are the standard parse's and never match here,
// and any other .go target declines with them: the two
// parses are disjoint, so every sentence keeps exactly the
// claim it has today, and a folder or main.go target joins
// this piece only when the folder truly holds a wasm app
// (runComponentWasm reads the file before claiming).
func parseComponentWasmTarget(line string) (componentWasmAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 4 {
		return componentWasmAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return componentWasmAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	if i < len(fields) && low[i] == "material" {
		i++
	}
	j := i
	for j < len(fields) && low[j] != "to" && low[j] != "in" && low[j] != "into" && low[j] != "at" {
		j++
	}
	if j >= len(fields) || j == i || j-i > 3 {
		return componentWasmAsk{}, false
	}
	rest := low[j+1:]
	if len(rest) != 1 {
		return componentWasmAsk{}, false
	}
	out := componentWasmAsk{name: strings.Join(low[i:j], " "), target: fields[j+1]}
	switch tok := rest[0]; {
	case strings.HasSuffix(tok, ".html"):
		return componentWasmAsk{}, false // a page target: the standard parse's
	case tok == "main.go" || strings.HasSuffix(tok, "/main.go"):
		out.goFile = true
	case strings.HasSuffix(tok, ".go"):
		return componentWasmAsk{}, false // another .go target: the book keeps it
	}
	if comp, ok := analyze.MaterialComponentByForm(out.name); ok {
		out.component = comp.ID
	}
	return out, true
}

// runComponentWasm routes a folder/main.go component
// sentence to the app in that folder — but only when the
// folder truly holds one (its main.go carries the js &&
// wasm build tag and the syscall/js import). Until that
// read says yes it claims nothing and prints nothing, so
// a folder or main.go that is not a wasm app keeps the
// fate the sentence has always had further down the chain.
func runComponentWasm(q componentWasmAsk, ask editAsk) bool {
	rootRaw := creationRoot()
	if rootRaw == "" {
		return false
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			return false
		}
		rootAbs = abs
	}
	_, rel, refusal := fenceResolve(rootAbs, q.target)
	if refusal != "" {
		return false
	}
	folderRel := rel
	if q.goFile {
		folderRel = path.Dir(rel)
	}
	if !wasmAppFolderOf(rootAbs, folderRel) {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.component == "" {
		fmt.Printf("%s There's no component called %q in the set I can add.\n%s\n", tag, q.name, componentListText())
		recordBrain("composer", false, "component refused: unknown component")
		return true
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "component") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:       analyze.EditComponent,
		Component:  q.component,
		WasmFolder: folderRel,
	}, ask)
}

// runComponent carries one line from sentence to resolved
// page to the shared edit spine. It returns true when the
// line belonged to this claim at all.
func runComponent(line string, ask editAsk) bool {
	// A folder or main.go target first: it belongs to the
	// wasm mode when the folder holds an app, and to no one
	// here when it doesn't (runComponentWasm decides by
	// reading the app's main; the standard parse below
	// declines those targets untouched either way).
	if wq, ok := parseComponentWasmTarget(line); ok && runComponentWasm(wq, ask) {
		return true
	}
	q, ok := parseComponent(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.component == "" {
		fmt.Printf("%s There's no component called %q in the set I can add to a page.\n%s\n", tag, q.name, componentListText())
		recordBrain("composer", false, "component refused: unknown component")
		return true
	}
	words := analyze.MaterialComponentWords(q.component)
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "component")
	if !ok {
		return true
	}
	file := q.file
	if q.thePage {
		// A session path is already project-relative from
		// this session's own records (the serve-static
		// folder discipline) — no fence, but the reading is
		// printed, as everywhere a context target resolves.
		page, reason, found := sessionPage(rootAbs)
		if !found {
			fmt.Printf("%s I don't have a page from this session to add the %s to — \"the page\" means the webpage you last made or filled here. Name the file instead: \"add a %s to jim/temp/index.html\".\n", tag, words, words)
			recordBrain("composer", false, "component refused: no page in session")
			return true
		}
		fmt.Printf("Reading \"the page\" as %s — %s.\n", page, reason)
		file = page
	} else {
		_, rel, refusal := fenceResolve(rootAbs, file)
		if refusal != "" {
			fmt.Printf("%s Not adding that component: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "component refused: file outside project")
			return true
		}
		file = rel
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "component") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:      analyze.EditComponent,
		File:      file,
		Component: q.component,
	}, ask)
}
