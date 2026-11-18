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
// app, the sentence is not this piece's at all. In the app
// mode alone, the sentence can name its component —
// "add a button called save to jim" — a distinct instance
// with its own builder and its name as its element id,
// the identity the events piece (wasmevent.go) points at;
// the page modes take no name, exactly as before.
//
// A target can also be a PART of a wasm app (the parts
// piece, wasmpart.go): "add a navigation bar to the
// header in jim", "add a chip to the footer of the wasm
// app in jim" — the component lands inside the part's own
// package (parseComponentPartTarget /
// runComponentPartWasm below, analyze.planComponentWasmPart),
// gated on the same content read, so a folder that holds
// no app keeps the line's existing fate untouched.

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
		// His own taught components list in their own
		// group, after the registry's — only when this
		// project has any, so the registry answer is
		// byte-identical everywhere else.
		if taught := taughtComponentsCurrent(); len(taught) > 0 {
			fmt.Printf("Yours, taught in this project:\n")
			for _, c := range taught {
				fmt.Printf("  your %s (taught) — \"add a %s to <app folder>\"\n", c.Kind, c.Kind)
			}
		}
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
// a component), the name as typed, the "called <name>"
// name the sentence gave the component ("" when none —
// the events piece's prerequisite: a named component is a
// distinct instance an event can point at), and the target
// as typed — a folder, or the app's main.go inside one.
type componentWasmAsk struct {
	component string
	name      string
	called    string
	target    string
	goFile    bool
}

// parseComponentWasmTarget reads the component shape with
// an app target — a folder ("add a card to jim") or the
// folder's main.go ("add a card to jim/main.go") — with
// the app mode's one extra slot: "called <name>" after the
// component's words ("add a button called save to jim")
// names the instance, for the events piece to point at.
// The page targets are the standard parse's and never
// match here, and any other .go target declines with them:
// the two parses are disjoint, so every sentence keeps
// exactly the claim it has today, and a folder or main.go
// target joins this piece only when the folder truly
// holds a wasm app (runComponentWasm reads the file
// before claiming).
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
	if j >= len(fields) || j == i {
		return componentWasmAsk{}, false
	}
	// The optional "called <name>": the kind's words, then
	// "called", then exactly one name word — a name in any
	// other shape (none after "called", several, or a word
	// that can't be one) is not this sentence.
	seg := low[i:j]
	called := ""
	for k, w := range seg {
		if w != "called" {
			continue
		}
		if k == 0 || k != len(seg)-2 || !wasmComponentNameToken(seg[k+1]) {
			return componentWasmAsk{}, false
		}
		called = seg[k+1]
		seg = seg[:k]
		break
	}
	if len(seg) == 0 || len(seg) > 3 {
		return componentWasmAsk{}, false
	}
	rest := low[j+1:]
	if len(rest) != 1 {
		return componentWasmAsk{}, false
	}
	out := componentWasmAsk{name: strings.Join(seg, " "), called: called, target: fields[j+1]}
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
		Kind:          analyze.EditComponent,
		Component:     q.component,
		ComponentName: q.called,
		WasmFolder:    folderRel,
	}, ask)
}

// componentPartAsk is one parsed component sentence
// aimed at a part of a wasm app: the registry id (""
// when the name is not a component), the name as typed,
// the "called <name>" name ("" when none), the part
// ("header" | "footer"), and the app's folder as typed.
type componentPartAsk struct {
	component string
	name      string
	called    string
	part      string
	folder    string
}

// parseComponentPartTarget reads the component shape
// with a part target —
//
//	add [a|an|the] [material] <component> [called <name>]
//	    to the header|footer in <folder>
//	add … to the header|footer of the wasm app in <folder>
//
// The shape is disjoint from the page and app parses: it
// requires the words "to the header" or "to the footer"
// after the component's own words, and a folder — never
// a file — after them, so every sentence keeps exactly
// the claim it has today. Whether the sentence FIRES is
// the resolver's call (runComponentPartWasm reads the
// folder's main before claiming), the wasm mode's rule.
func parseComponentPartTarget(line string) (componentPartAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 7 {
		return componentPartAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return componentPartAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	if i < len(fields) && low[i] == "material" {
		i++
	}
	j := i
	for j < len(fields) && low[j] != "to" {
		j++
	}
	if j >= len(fields) || j == i {
		return componentPartAsk{}, false
	}
	seg := low[i:j]
	called := ""
	for k, w := range seg {
		if w != "called" {
			continue
		}
		if k == 0 || k != len(seg)-2 || !wasmComponentNameToken(seg[k+1]) {
			return componentPartAsk{}, false
		}
		called = seg[k+1]
		seg = seg[:k]
		break
	}
	if len(seg) == 0 || len(seg) > 3 {
		return componentPartAsk{}, false
	}
	rest := low[j+1:]
	if len(rest) < 4 || rest[0] != "the" || (rest[1] != "header" && rest[1] != "footer") {
		return componentPartAsk{}, false
	}
	out := componentPartAsk{name: strings.Join(seg, " "), called: called, part: rest[1]}
	rest = rest[2:]
	if len(rest) >= 3 && rest[0] == "of" && rest[1] == "the" && rest[2] == "wasm" {
		if len(rest) < 4 || rest[3] != "app" {
			return componentPartAsk{}, false
		}
		rest = rest[4:]
	}
	if len(rest) < 2 || (rest[0] != "in" && rest[0] != "into" && rest[0] != "at") {
		return componentPartAsk{}, false
	}
	rest = rest[1:]
	if len(rest) > 0 && rest[0] == "the" {
		rest = rest[1:]
	}
	if len(rest) > 0 && (rest[0] == "folder" || rest[0] == "directory") {
		rest = rest[1:]
	}
	if len(rest) != 1 {
		return componentPartAsk{}, false
	}
	tok := rest[0]
	if strings.HasSuffix(tok, ".html") || strings.HasSuffix(tok, ".go") {
		return componentPartAsk{}, false // a file target: not this shape
	}
	out.folder = fields[len(fields)-1]
	if comp, ok := analyze.MaterialComponentByForm(out.name); ok {
		out.component = comp.ID
	}
	return out, true
}

// runComponentPartWasm routes a part-target component
// sentence to the part of the app in that folder — but
// only when the folder truly holds a wasm app (its
// main.go carries the js && wasm build tag). Until that
// read says yes it claims nothing and prints nothing, so
// a folder that is not a wasm app keeps the fate the
// sentence has always had further down the chain — the
// page modes included.
func runComponentPartWasm(q componentPartAsk, ask editAsk) bool {
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
	_, rel, refusal := fenceResolve(rootAbs, q.folder)
	if refusal != "" {
		return false
	}
	if !wasmAppFolderOf(rootAbs, rel) {
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
		Kind:          analyze.EditComponent,
		Component:     q.component,
		ComponentName: q.called,
		WasmFolder:    rel,
		ComponentPart: q.part,
	}, ask)
}

// runComponent carries one line from sentence to resolved
// page to the shared edit spine. It returns true when the
// line belonged to this claim at all.
func runComponent(line string, ask editAsk) bool {
	// A taught component's sentences first, part then app:
	// they belong to this claim only when the phrase names
	// a component taught in this project AND the folder
	// holds a wasm app — until both readings say yes the
	// run side claims nothing and prints nothing, exactly
	// the registry modes' discipline, so every sentence
	// keeps the fate it has always had. They sit ahead of
	// the registry parses because a taught phrase is any
	// length, where the registry parses cap their segment
	// at three words and would decline (or misread) his.
	if pq, ok := parseTaughtComponentPart(line); ok && runTaughtComponentPart(pq, ask) {
		return true
	}
	if wq, ok := parseTaughtComponentWasm(line); ok && runTaughtComponentWasm(wq, ask) {
		return true
	}
	// A part target first ("add a chip to the header in
	// jim"): it belongs to the wasm mode when the folder
	// holds an app with that part, and to no one here when
	// it doesn't — the shape's "to the header/footer"
	// words are ones no page sentence carries.
	if pq, ok := parseComponentPartTarget(line); ok && runComponentPartWasm(pq, ask) {
		return true
	}
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
