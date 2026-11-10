package chat

// Taught components: the teach seat's component side
// (taughtkinds.go is the seat; this file is its second
// cut of example). Where a taught kind learns a
// declaration's shape from one function or type, a taught
// component learns a wasm component's shape from one DOM
// builder in the user's project: "learn this component as
// a price tag from PriceTag in jim/components.go". The
// builder contract is the registry's own, read off the
// builders the wasm piece ships (analyze/editwasm.go): a
// plain function taking the document value and returning
// the element it built — func <Name>(doc js.Value)
// js.Value — whose elements are made with the app's el
// helper in the helper's own shape, el(doc, tag, class,
// text), and whose root element is stamped with its id
// the registry way, <root>.Set("id", "..."), under the
// line that creates it. Events and mounts find a
// component by that id; a builder without the stamp is
// refused with exactly that said, and nothing is learned
// from it.
//
// Taught components live in the same store file as taught
// kinds (~/.gollemer_taught.json), in their own section,
// and FORGET and LIST treat them as the seat's own. Two
// differences from taught kinds, both deliberate: a
// component belongs to the project it was taught in (its
// builder leans on that app's helpers, so the store
// records the project and the kind answers only there),
// and its kind phrase may never equal a registry
// component's name — the registry wins its own names, at
// teach time, plainly.
//
// Adding one rides the component claim (component.go
// seats the taught parses ahead of the registry ones, the
// same silent-until-an-app discipline), and the planner
// (analyze.EditTaughtComponent) lands it exactly the way
// a registry component lands — body or part, named or
// positional, already-there identical — except styles: a
// taught builder's classes come with it, but no styles
// travel in this version, and the teach confirmation
// says so plainly when the example's builder uses classes.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// taughtComponent is one taught component kind: the
// builder template and the facts the add side, the
// listings, and the plan's provenance render from.
type taughtComponent struct {
	Kind     string    `json:"kind"`     // normalized phrase ("price tag")
	Display  string    `json:"display"`  // the phrase as he typed it
	Example  string    `json:"example"`  // the example builder's name
	Template string    `json:"template"` // source with the name and id slots
	RootVar  string    `json:"root_var"` // the variable the root element is kept in
	Styled   bool      `json:"styled,omitempty"`
	File     string    `json:"file"` // the example's file, relative to its project
	Root     string    `json:"root"` // the project the example was taught in
	TaughtAt time.Time `json:"taught_at"`
}

func (c taughtComponent) provenance() string {
	return fmt.Sprintf("From your example %s in %s, taught %s.", c.Example, c.File, taughtDate(c.TaughtAt))
}

// findTaughtComponent returns the stored component with
// this normalized phrase taught in this project, or nil.
func (d taughtKindData) findTaughtComponent(normalized, rootAbs string) *taughtComponent {
	for i := range d.Components {
		if d.Components[i].Kind == normalized && d.Components[i].Root == rootAbs {
			return &d.Components[i]
		}
	}
	return nil
}

// findAnyTaughtComponent returns a stored component with
// this normalized phrase in any project, or nil — the
// one-phrase-one-seat check's read.
func (d taughtKindData) findAnyTaughtComponent(normalized string) *taughtComponent {
	for i := range d.Components {
		if d.Components[i].Kind == normalized {
			return &d.Components[i]
		}
	}
	return nil
}

// taughtComponentsFor returns the components taught in
// this project, in store order.
func (d taughtKindData) taughtComponentsFor(rootAbs string) []taughtComponent {
	var out []taughtComponent
	for _, c := range d.Components {
		if c.Root == rootAbs {
			out = append(out, c)
		}
	}
	return out
}

// taughtComponentsCurrent reads the components taught in
// the project this session is working in — quietly: any
// failure (no project folder, an unreadable store) reads
// as none, for the surfaces that only extend when some
// exist (discovery, the event references).
func taughtComponentsCurrent() []taughtComponent {
	rootRaw := creationRoot()
	if rootRaw == "" {
		return nil
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			return nil
		}
		rootAbs = abs
	}
	data, err := loadTaughtKinds()
	if err != nil {
		return nil
	}
	return data.taughtComponentsFor(rootAbs)
}

// taughtComponentHere returns the component with this
// phrase taught in the current project, or nil — quietly,
// as taughtComponentsCurrent reads.
func taughtComponentHere(phrase string) *taughtComponent {
	kind := normalizeKindPhrase(phrase)
	for _, c := range taughtComponentsCurrent() {
		if c.Kind == kind {
			cc := c
			return &cc
		}
	}
	return nil
}

// ----- teach grammar ---------------------------------------------------------

// parseTeachComponentAsk reads the component teach
// shapes:
//
//	learn this component as a <kind> from <Name> [in <file>]
//	learn|teach [you] a|an component as a <kind> from <Name> [in <file>]
//	teach this component as a <kind> from <Name> [in <file>]
//
// The "component" word is the shape's spine: it is what
// seats this grammar ahead of the declaration teach
// grammar, whose kind phrase would otherwise swallow it.
// Anything else declines.
func parseTeachComponentAsk(line string) (teachAsk, bool) {
	fields, low := splitSentence(line)
	if len(fields) < 6 {
		return teachAsk{}, false
	}
	i := 0
	switch low[0] {
	case "learn":
		i = 1
	case "teach":
		i = 1
		if i < len(low) && low[i] == "you" {
			i++
		}
	default:
		return teachAsk{}, false
	}
	// The introducer: "this component as" or an article
	// before "component as".
	if i < len(low) && low[i] == "this" {
		i++
	} else if i < len(low) && (low[i] == "a" || low[i] == "an") {
		i++
	} else {
		return teachAsk{}, false
	}
	if i+1 >= len(low) || low[i] != "component" || low[i+1] != "as" {
		return teachAsk{}, false
	}
	i += 2
	if i < len(low) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	// The kind runs to the first "from" that leaves a
	// valid tail, exactly as the declaration grammar reads
	// its kind.
	for j := i + 1; j < len(low); j++ {
		if low[j] != "from" {
			continue
		}
		name, file, ok := parseExampleTail(fields, low, j+1)
		if !ok {
			continue
		}
		kind := strings.Join(fields[i:j], " ")
		if normalizeKindPhrase(kind) == "" {
			continue
		}
		return teachAsk{kind: kind, name: name, file: file}, true
	}
	return teachAsk{}, false
}

// ----- teach -----------------------------------------------------------------

// runTeachComponent carries one component teach sentence:
// the name guards, the builder's location and contract
// validation, the template cut, the summary, the save —
// runTeach's discipline throughout, with every refusal
// plain and nothing written.
func runTeachComponent(ta teachAsk) {
	tag := cTag("gollemer [composer]>")
	kind := normalizeKindPhrase(ta.kind)

	// The kind's name is the sentence's own contract, as
	// with a taught kind — plus this seat's own word: a
	// kind named with "component" would tangle the very
	// sentences that teach it.
	for _, w := range strings.Fields(kind) {
		if w == "named" || w == "from" || w == "component" {
			fmt.Printf("%s Not learning that: a kind's name can't contain the word %q — the sentences that teach and use it parse by that word, so it would tangle them. Pick a plainer name, like \"price tag\".\n", tag, w)
			recordBrain("composer", false, "taught component refused: kind word")
			return
		}
		if proposerKindWords[w] {
			fmt.Printf("%s Not learning that: %q is a declaration kind word the book and the proposer already parse by — a taught kind named with it would sit in front of their sentences. Pick a name of your own for the kind, like \"price tag\".\n", tag, w)
			recordBrain("composer", false, "taught component refused: kind word")
			return
		}
	}
	// Registry kinds win their names: a taught component
	// sits beside the built-ins, never over one.
	if comp, ok := analyze.MaterialComponentByForm(kind); ok {
		fmt.Printf("%s Not learning that: %q is already a built-in component I can add — your taught components sit beside the built-in ones, never over them, so pick a different name for the kind, like \"price tag\".\n", tag, analyze.MaterialComponentWords(comp.ID))
		recordBrain("composer", false, "taught component refused: kind is a registry component")
		return
	}
	if owner, taken := reservedKindNames()[kind]; taken {
		fmt.Printf("%s Not learning that: %q is already one of %s's pieces — your taught kinds sit beside them, never over them, so pick a different name for the kind, like \"price tag\".\n", tag, kind, owner)
		recordBrain("composer", false, "taught component refused: kind shadows a family")
		return
	}
	data, err := loadTaughtKinds()
	if err != nil {
		fmt.Printf("%s I couldn't read your taught-pieces store (%s): %v\n", tag, taughtStorePath(), err)
		recordBrain("composer", false, "taught component refused: store unreadable")
		return
	}
	// One phrase can't be both a piece and a component: a
	// later "add a <kind>" would have two seats claiming it.
	if old := data.findTaughtKind(kind); old != nil {
		fmt.Printf("%s Not learning that: you've already taught me a piece called %q — %s A name is one piece or one component, never both. If you want to teach me a component under this name, forget the piece first: say \"forget the %s\".\n", tag, kind, old.provenance(), kind)
		recordBrain("composer", false, "taught component refused: kind is a taught piece")
		return
	}

	rootRaw, rootAbs, ok := taughtProjectRoot()
	if !ok {
		recordBrain("composer", false, "taught component refused: no root")
		return
	}
	if old := data.findTaughtComponent(kind, rootAbs); old != nil {
		fmt.Printf("%s Not learning that: you've already taught me a %q component in this project — %s If you want to teach me a different one under this name, forget that one first: say \"forget the %s\".\n", tag, kind, old.provenance(), kind)
		recordBrain("composer", false, "taught component refused: kind already taught")
		return
	}
	if !taughtFreshProject(rootRaw, rootAbs) {
		recordBrain("composer", false, "taught component refused: project unreadable")
		return
	}

	src, fset, fileAST, fd, rel, refusal := locateComponentBuilder(rootAbs, ta)
	if refusal != "" {
		fmt.Printf("%s Not learning that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "taught component refused: example")
		return
	}
	rootVar, stampLit, styled, refusal := validateComponentBuilder(fd, fileAST, ta, rel)
	if refusal != "" {
		fmt.Printf("%s Not learning that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "taught component refused: builder shape")
		return
	}
	tmpl := buildComponentTemplate(src, fset, fd, ta.name, stampLit)
	if tmpl == "" {
		fmt.Printf("%s Not learning that: I couldn't cut a template from %s in %s — nothing was saved.\n", tag, ta.name, rel)
		recordBrain("composer", false, "taught component refused: template")
		return
	}

	entry := taughtComponent{
		Kind:     kind,
		Display:  strings.Join(strings.Fields(ta.kind), " "),
		Example:  ta.name,
		Template: tmpl,
		RootVar:  rootVar,
		Styled:   styled,
		File:     rel,
		Root:     rootAbs,
		TaughtAt: time.Now(),
	}

	// The summary prints BEFORE the save, runTeach's rule:
	// whether or not the save then succeeds, the words are
	// the truth about what would be kept.
	fmt.Printf("%s Here's what I learned from %s in %s:\n", tag, ta.name, rel)
	fmt.Printf("  The kind: %q — from here on, \"add a %s to <app folder>\" adds one to a wasm app here, and \"add a %s called <name> to <app folder>\" adds a named one.\n", kind, kind, kind)
	fmt.Printf("  The example: the builder %s, at %s:%d.\n", ta.name, rel, lineOf(fset, fd.Pos()))
	fmt.Printf("  What varies: the builder's name — each copy is stamped with its own — and the root element's id: a named %s carries its name as its id, an unnamed one its place among its kind (\"%s-1\"). And where it lands: the app's body, or its header or footer.\n", kind, strings.Join(strings.Fields(kind), "-"))
	fmt.Printf("  What stays fixed: everything else — the builder's code stays exactly as in %s.\n", ta.name)
	if styled {
		fmt.Printf("  Styles: %s gives its elements classes, and the classes come with it — but no styles travel with a taught component in this first version, so it shows styled only where the page's styles already cover those classes.\n", ta.name)
	}
	data.Components = append(data.Components, entry)
	if err := saveTaughtKinds(data); err != nil {
		fmt.Printf("  I couldn't save it to your store (%s): %v — nothing was kept.\n", taughtStorePath(), err)
		recordBrain("composer", false, "taught component refused: store unwritable")
		return
	}
	fmt.Printf("  Saved as yours: it lives in %s, and it belongs to this project — the one %s was taught in. Say \"forget the %s\" to remove it.\n", taughtStorePath(), ta.name, kind)
	recordBrain("composer", true, "taught component learned: "+kind)
}

// locateComponentBuilder resolves the teach sentence's
// builder to a plain function in the project, rendering
// every refusal in the teach family's words —
// locateExample's discipline, for builders.
func locateComponentBuilder(rootAbs string, ta teachAsk) (src []byte, fset *token.FileSet, fileAST *ast.File, fd *ast.FuncDecl, rel string, refusal string) {
	if ta.file != "" {
		abs, relT, fence := serverTargetResolve(rootAbs, ta.file)
		if fence != "" {
			return nil, nil, nil, nil, "", fence
		}
		st, err := os.Stat(abs)
		if err != nil || st.IsDir() {
			return nil, nil, nil, nil, "", fmt.Sprintf("there's no file %s in this project — name the file the builder lives in", strings.TrimSuffix(ta.file, "/"))
		}
		src, err := os.ReadFile(abs)
		if err != nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't read %s (%v)", relT, err)
		}
		fset := token.NewFileSet()
		fileAST, perr := parser.ParseFile(fset, relT, src, parser.ParseComments)
		if perr != nil || fileAST == nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("%s doesn't parse right now (%v), so I can't read %s's shape from it — fix the file and ask again", relT, perr, ta.name)
		}
		found, method := findBuilderDecl(fileAST, ta.name)
		if found == nil {
			if method != nil {
				return nil, nil, nil, nil, "", builderMethodRefusal(ta.name, method, relT)
			}
			return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't find a function named %q in %s — there's no builder with that name in that file", ta.name, relT)
		}
		return src, fset, fileAST, found, relT, ""
	}

	// No file named: the index searches the whole project,
	// and builders the index can't see (an app folder it
	// didn't reach) are searched for in the components.go
	// files themselves.
	var cands []exampleCandidate
	seen := map[string]bool{}
	if lastAnalyzeProject != nil {
		for _, pkg := range lastAnalyzeProject.Packages {
			for _, fn := range pkg.Funcs {
				if fn.Name != ta.name || fn.Receiver != "" || fn.IsTest || seen[fn.File] {
					continue
				}
				seen[fn.File] = true
				cands = append(cands, exampleCandidate{kind: "function", file: fn.File, line: fn.Line})
			}
		}
	}
	if len(cands) == 0 {
		for _, c := range walkComponentBuilders(rootAbs, ta.name) {
			if !seen[c.file] {
				seen[c.file] = true
				cands = append(cands, c)
			}
		}
	}
	switch {
	case len(cands) == 1:
		c := cands[0]
		src, err := os.ReadFile(filepath.Join(rootAbs, filepath.FromSlash(c.file)))
		if err != nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't read %s (%v)", c.file, err)
		}
		fset := token.NewFileSet()
		fileAST, perr := parser.ParseFile(fset, c.file, src, parser.ParseComments)
		if perr != nil || fileAST == nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("%s doesn't parse right now (%v), so I can't read %s's shape from it — fix the file and ask again", c.file, perr, ta.name)
		}
		found, method := findBuilderDecl(fileAST, ta.name)
		if found == nil {
			if method != nil {
				return nil, nil, nil, nil, "", builderMethodRefusal(ta.name, method, c.file)
			}
			return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't find a function named %q in %s", ta.name, c.file)
		}
		return src, fset, fileAST, found, c.file, ""
	case len(cands) > 1:
		var parts []string
		for _, c := range cands {
			parts = append(parts, fmt.Sprintf("the builder %s in %s", ta.name, c.file))
		}
		return nil, nil, nil, nil, "", fmt.Sprintf("more than one function is named %s — %s. Say which one you mean by naming its file, like \"learn this component as a %s from %s in %s\"", ta.name, strings.Join(parts, " and "), normalizeKindPhrase(ta.kind), ta.name, cands[0].file)
	default:
		return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't find a function named %q in this project — there's no builder with that name. If it doesn't exist yet, add it first — a builder in a wasm app, in the shape of the app's own components — then teach me from it", ta.name)
	}
}

// findBuilderDecl finds the plain function named name in a
// parsed file; a method of that name comes back separately,
// for its own refusal.
func findBuilderDecl(fileAST *ast.File, name string) (fd *ast.FuncDecl, method *ast.FuncDecl) {
	for _, d := range fileAST.Decls {
		f, ok := d.(*ast.FuncDecl)
		if !ok || f.Name.Name != name {
			continue
		}
		if f.Recv == nil || len(f.Recv.List) == 0 {
			return f, nil
		}
		if method == nil {
			method = f
		}
	}
	return nil, method
}

func builderMethodRefusal(name string, method *ast.FuncDecl, rel string) string {
	recv := ""
	if method.Recv != nil && len(method.Recv.List) > 0 {
		recv = strings.TrimPrefix(exprText(method.Recv.List[0].Type), "*")
	}
	return fmt.Sprintf("%s in %s is a method — it hangs on the type %s, so it isn't a builder on its own — and a component is taught from a plain builder function only, like the app's own components. Teach me from a plain function", name, rel, recv)
}

// walkComponentBuilders is the index fallback's search:
// the project's components.go files, near the root, parsed
// for a plain function of this name — the files a builder
// lives in, in the shape the wasm piece writes.
func walkComponentBuilders(rootAbs, name string) []exampleCandidate {
	var out []exampleCandidate
	_ = filepath.WalkDir(rootAbs, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		rel, rerr := filepath.Rel(rootAbs, p)
		if rerr != nil {
			return nil
		}
		if d.IsDir() {
			if rel != "." {
				if strings.HasPrefix(d.Name(), ".") || d.Name() == "vendor" || d.Name() == "node_modules" {
					return filepath.SkipDir
				}
				if strings.Count(filepath.ToSlash(rel), "/") >= 3 {
					return filepath.SkipDir
				}
			}
			return nil
		}
		if d.Name() != "components.go" {
			return nil
		}
		src, rerr := os.ReadFile(p)
		if rerr != nil {
			return nil
		}
		fset := token.NewFileSet()
		fa, perr := parser.ParseFile(fset, filepath.ToSlash(rel), src, parser.ParseComments)
		if perr != nil || fa == nil {
			return nil
		}
		if fd, _ := findBuilderDecl(fa, name); fd != nil {
			out = append(out, exampleCandidate{kind: "function", file: filepath.ToSlash(rel), line: lineOf(fset, fd.Pos())})
		}
		return nil
	})
	return out
}

// validateComponentBuilder reads one builder against the
// registry contract, derived from the builders the wasm
// piece ships: the signature, the imports it may lean on,
// the el calls in the helper's own four-argument shape,
// the root element made with that helper, and the root's
// id stamp — returning the root's variable, the stamp's
// string literal, and whether the builder gives its
// elements classes. Every failure is a refusal that says
// exactly what doesn't hold.
func validateComponentBuilder(fd *ast.FuncDecl, fileAST *ast.File, ta teachAsk, rel string) (rootVar string, stampLit *ast.BasicLit, styled bool, refusal string) {
	// The signature: one js.Value in, one js.Value out.
	params, results := 0, 0
	paramOK, resultOK := false, false
	if fd.Type != nil && fd.Type.Params != nil {
		for _, f := range fd.Type.Params.List {
			params += max(len(f.Names), 1)
			if exprText(f.Type) == "js.Value" {
				paramOK = true
			}
		}
	}
	if fd.Type != nil && fd.Type.Results != nil {
		for _, f := range fd.Type.Results.List {
			results += max(len(f.Names), 1)
			if exprText(f.Type) == "js.Value" {
				resultOK = true
			}
		}
	}
	if fd.Body == nil || params != 1 || results != 1 || !paramOK || !resultOK {
		return "", nil, false, fmt.Sprintf("%s in %s doesn't have a component builder's shape: a builder takes the document and hands back the element it built — func %s(doc js.Value) js.Value, the shape of the app's own components. Teach me from a builder in that shape", ta.name, rel, ta.name)
	}
	// The imports it may lean on: syscall/js and the file's
	// own package only. A builder travels into other apps
	// of this project; another package's names wouldn't
	// resolve there, so they are refused, not carried.
	locals := map[string]bool{}
	for _, sp := range fileAST.Imports {
		p, err := strconv.Unquote(sp.Path.Value)
		if err != nil {
			continue
		}
		local := importLocalName(p)
		if sp.Name != nil {
			local = sp.Name.Name
		}
		locals[local] = true
	}
	leansOn := ""
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		if leansOn != "" {
			return false
		}
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok && locals[id.Name] && id.Name != "js" {
				leansOn = id.Name
				return false
			}
		}
		return true
	})
	if leansOn != "" {
		return "", nil, false, fmt.Sprintf("%s leans on the %s package in %s, and a taught component's builder travels on its own — this first version teaches builders that lean only on syscall/js and the app's own helpers, like the built-in components do. Teach me from a builder in that shape", ta.name, leansOn, rel)
	}
	// The el calls: the app's helper takes four arguments
	// — el(doc, tag, class, text) — and a taught builder
	// lands against that helper exactly as it stands, so
	// every el call in the builder has to match its shape,
	// the way every built-in builder's calls do.
	shortEl := 0
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		if shortEl != 0 {
			return false
		}
		if call, ok := n.(*ast.CallExpr); ok {
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "el" && len(call.Args) != 4 {
				shortEl = len(call.Args)
				return false
			}
		}
		return true
	})
	if shortEl != 0 {
		return "", nil, false, fmt.Sprintf("%s calls the app's el helper with %d arguments in %s — the app's el takes four: el(doc, \"div\", \"\", \"words\") — the document, a tag, its classes, and its words — and a taught builder lands against that helper exactly as it stands. Fix the el calls to the helper's shape and teach me again", ta.name, shortEl, rel)
	}
	// The root: the first top-level element the builder
	// makes with the app's el helper, kept in a variable.
	rootAt := -1
	for i, stmt := range fd.Body.List {
		as, ok := stmt.(*ast.AssignStmt)
		if !ok || as.Tok != token.DEFINE {
			continue
		}
		for k, rhs := range as.Rhs {
			call, ok := rhs.(*ast.CallExpr)
			if !ok {
				continue
			}
			if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "el" && k < len(as.Lhs) {
				if lv, ok := as.Lhs[k].(*ast.Ident); ok {
					rootVar, rootAt = lv.Name, i
				}
			}
		}
		if rootVar != "" {
			break
		}
	}
	if rootVar == "" {
		return "", nil, false, fmt.Sprintf("%s never builds a root element with the app's el helper — a builder makes its root with el(doc, \"div\", ...), the line its id stamp hangs on, like every built-in component. Teach me from a builder that builds its root that way", ta.name)
	}
	// The stamp: after the root is made, a plain statement
	// setting its id to a written-out string. This is the
	// contract's load-bearing line — the app's events and
	// mounts find the component by that id.
	for _, stmt := range fd.Body.List[rootAt+1:] {
		es, ok := stmt.(*ast.ExprStmt)
		if !ok {
			continue
		}
		call, ok := es.X.(*ast.CallExpr)
		if !ok || len(call.Args) != 2 {
			continue
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "Set" {
			continue
		}
		if id, ok := sel.X.(*ast.Ident); !ok || id.Name != rootVar {
			continue
		}
		key, ok := call.Args[0].(*ast.BasicLit)
		if !ok || key.Kind != token.STRING {
			continue
		}
		if v, err := strconv.Unquote(key.Value); err != nil || v != "id" {
			continue
		}
		lit, ok := call.Args[1].(*ast.BasicLit)
		if !ok || lit.Kind != token.STRING {
			return "", nil, false, fmt.Sprintf("%s sets its root's id, but not to a plain string — the stamp is %s.Set(\"id\", \"...\") with the id written out, so each copy of the component can be stamped with its own id. Write the id as a plain string and teach me again", ta.name, rootVar)
		}
		stampLit = lit
		break
	}
	if stampLit == nil {
		return "", nil, false, fmt.Sprintf("%s never sets its root element's id the registry way: a builder stamps the element it built with %s.Set(\"id\", \"...\") right under the line that creates it, and the app finds the component by that id — its mounts and its events both read it. Add that line to %s and teach me again", ta.name, rootVar, ta.name)
	}
	// Classes: the root's class argument to el, or a
	// className set — the styles honesty reads this.
	ast.Inspect(fd.Body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		if id, ok := call.Fun.(*ast.Ident); ok && id.Name == "el" && len(call.Args) >= 3 {
			if lit, ok := call.Args[2].(*ast.BasicLit); ok && lit.Kind == token.STRING {
				if v, err := strconv.Unquote(lit.Value); err == nil && v != "" {
					styled = true
				}
			}
		}
		if sel, ok := call.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Set" && len(call.Args) == 2 {
			if lit, ok := call.Args[0].(*ast.BasicLit); ok && lit.Kind == token.STRING {
				if v, err := strconv.Unquote(lit.Value); err == nil && v == "className" {
					styled = true
				}
			}
		}
		return true
	})
	return rootVar, stampLit, styled, ""
}

// buildComponentTemplate cuts the stored template from
// the builder's source: the function's own text (its
// leading doc comment left behind — it names the example,
// not the kind), with every place the builder names
// itself replaced by the name slot and the stamp's id
// string replaced by the id slot — buildDeclTemplate's
// cut, for a builder. The planner stamps both slots per
// instance.
func buildComponentTemplate(src []byte, fset *token.FileSet, fd *ast.FuncDecl, name string, stampLit *ast.BasicLit) string {
	file := fset.File(fd.Pos())
	if file == nil {
		return ""
	}
	off := func(pos token.Pos) int { return file.Offset(pos) }
	start, end := off(fd.Pos()), off(fd.End())
	type slot struct {
		lo, hi int
		text   string
	}
	var slots []slot
	var stack []ast.Node
	ast.Inspect(fd, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return false
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == name && !identNamesAnother(stack, id) {
			slots = append(slots, slot{off(id.Pos()), off(id.End()), analyze.TaughtNameSlot})
		}
		stack = append(stack, n)
		return true
	})
	slots = append(slots, slot{off(stampLit.Pos()), off(stampLit.End()), strconv.Quote(analyze.TaughtIDSlot)})
	if len(slots) < 2 {
		return ""
	}
	sort.Slice(slots, func(a, b int) bool { return slots[a].lo > slots[b].lo })
	text := string(src[start:end])
	for _, s := range slots {
		text = text[:s.lo-start] + s.text + text[s.hi-start:]
	}
	return text
}

// ----- add grammar -----------------------------------------------------------

// taughtComponentWasmAsk is one parsed add sentence for a
// taught component aimed at a wasm app: the kind phrase
// (normalized), the "called <name>" name ("" when none),
// and the target as typed — a folder, or the app's
// main.go inside one.
type taughtComponentWasmAsk struct {
	kind   string
	called string
	target string
	goFile bool
}

// parseTaughtComponentWasm reads the component claim's
// app shape for a taught kind:
//
//	add [a|an|the] <kind words> [called <name>] to|in|into|at <folder|folder/main.go>
//
// It is read from the END — the target is the last word,
// the preposition before it — because a taught kind is a
// phrase of the user's own words, any length, where the
// registry parse caps its segment at three. Whether the
// phrase names a taught component of this project is the
// run side's question (and the run side stays silent
// until it does and the folder holds an app), so a line
// this parse reads keeps its fate everywhere else until
// then.
func parseTaughtComponentWasm(line string) (taughtComponentWasmAsk, bool) {
	fields, low := splitSentence(line)
	if len(fields) < 5 {
		return taughtComponentWasmAsk{}, false
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return taughtComponentWasmAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	last := len(fields) - 1
	if !isTargetPrep(low[last-1]) {
		return taughtComponentWasmAsk{}, false
	}
	out := taughtComponentWasmAsk{target: fields[last]}
	switch tok := low[last]; {
	case strings.HasSuffix(tok, ".html"):
		return taughtComponentWasmAsk{}, false // a page target: the standard parse's
	case tok == "main.go" || strings.HasSuffix(tok, "/main.go"):
		out.goFile = true
	case strings.HasSuffix(tok, ".go"):
		return taughtComponentWasmAsk{}, false // another .go target: the book keeps it
	}
	seg := low[i : last-1]
	// The part shape ("… to the header in <folder>") is
	// the part parse's, never this one's — the two parses
	// stay disjoint, as the registry's do.
	for k := 0; k+2 < len(seg); k++ {
		if seg[k] == "to" && seg[k+1] == "the" && (seg[k+2] == "header" || seg[k+2] == "footer") {
			return taughtComponentWasmAsk{}, false
		}
	}
	for k, w := range seg {
		if w != "called" {
			continue
		}
		if k == 0 || k != len(seg)-2 || !wasmComponentNameToken(seg[k+1]) {
			return taughtComponentWasmAsk{}, false
		}
		out.called = seg[k+1]
		seg = seg[:k]
		break
	}
	if len(seg) == 0 {
		return taughtComponentWasmAsk{}, false
	}
	out.kind = normalizeKindPhrase(strings.Join(seg, " "))
	if out.kind == "" {
		return taughtComponentWasmAsk{}, false
	}
	return out, true
}

// runTaughtComponentWasm routes a folder/main.go sentence
// naming a taught component of this project to the app in
// that folder — runComponentWasm's discipline exactly:
// silent until the phrase is his and the folder truly
// holds a wasm app, so every other sentence keeps its
// fate.
func runTaughtComponentWasm(q taughtComponentWasmAsk, ask editAsk) bool {
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
	data, err := loadTaughtKinds()
	if err != nil {
		return false
	}
	entry := data.findTaughtComponent(q.kind, rootAbs)
	if entry == nil {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "component") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:                    analyze.EditTaughtComponent,
		WasmFolder:              folderRel,
		ComponentName:           q.called,
		TaughtKindPhrase:        entry.Kind,
		TaughtComponentTemplate: entry.Template,
		TaughtComponentRootVar:  entry.RootVar,
		TaughtComponentStyled:   entry.Styled,
		TaughtNote:              entry.provenance(),
	}, ask)
}

// taughtComponentPartAsk is one parsed add sentence for a
// taught component aimed at a part of a wasm app.
type taughtComponentPartAsk struct {
	kind   string
	called string
	part   string
	folder string
}

// parseTaughtComponentPart reads the part shape for a
// taught kind:
//
//	add [a|an|the] <kind words> [called <name>]
//	    to the header|footer [of the wasm app] in|into|at <folder>
//
// parseComponentPartTarget's discipline, with the kind
// read as the whole segment before "to the header" — a
// taught phrase is any length.
func parseTaughtComponentPart(line string) (taughtComponentPartAsk, bool) {
	fields, low := splitSentence(line)
	if len(fields) < 7 {
		return taughtComponentPartAsk{}, false
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return taughtComponentPartAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	j := i
	for j < len(fields) && low[j] != "to" {
		j++
	}
	if j >= len(fields) || j == i {
		return taughtComponentPartAsk{}, false
	}
	seg := low[i:j]
	out := taughtComponentPartAsk{}
	for k, w := range seg {
		if w != "called" {
			continue
		}
		if k == 0 || k != len(seg)-2 || !wasmComponentNameToken(seg[k+1]) {
			return taughtComponentPartAsk{}, false
		}
		out.called = seg[k+1]
		seg = seg[:k]
		break
	}
	if len(seg) == 0 {
		return taughtComponentPartAsk{}, false
	}
	out.kind = normalizeKindPhrase(strings.Join(seg, " "))
	if out.kind == "" {
		return taughtComponentPartAsk{}, false
	}
	rest := low[j+1:]
	if len(rest) < 4 || rest[0] != "the" || (rest[1] != "header" && rest[1] != "footer") {
		return taughtComponentPartAsk{}, false
	}
	out.part = rest[1]
	rest = rest[2:]
	if len(rest) >= 4 && rest[0] == "of" && rest[1] == "the" && rest[2] == "wasm" && rest[3] == "app" {
		rest = rest[4:]
	}
	if len(rest) < 2 || (rest[0] != "in" && rest[0] != "into" && rest[0] != "at") {
		return taughtComponentPartAsk{}, false
	}
	rest = rest[1:]
	if len(rest) > 0 && rest[0] == "the" {
		rest = rest[1:]
	}
	if len(rest) > 0 && (rest[0] == "folder" || rest[0] == "directory") {
		rest = rest[1:]
	}
	if len(rest) != 1 {
		return taughtComponentPartAsk{}, false
	}
	tok := rest[0]
	if strings.HasSuffix(tok, ".html") || strings.HasSuffix(tok, ".go") {
		return taughtComponentPartAsk{}, false // a file target: not this shape
	}
	out.folder = fields[len(fields)-1]
	return out, true
}

// runTaughtComponentPart routes a part-target sentence
// naming a taught component of this project to the part
// of the app in that folder — runComponentPartWasm's
// discipline exactly, silent until the phrase is his and
// the folder holds an app.
func runTaughtComponentPart(q taughtComponentPartAsk, ask editAsk) bool {
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
	data, err := loadTaughtKinds()
	if err != nil {
		return false
	}
	entry := data.findTaughtComponent(q.kind, rootAbs)
	if entry == nil {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "component") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:                    analyze.EditTaughtComponent,
		WasmFolder:              rel,
		ComponentName:           q.called,
		ComponentPart:           q.part,
		TaughtKindPhrase:        entry.Kind,
		TaughtComponentTemplate: entry.Template,
		TaughtComponentRootVar:  entry.RootVar,
		TaughtComponentStyled:   entry.Styled,
		TaughtNote:              entry.provenance(),
	}, ask)
}

// ----- events ------------------------------------------------------------------

// taughtComponentRefKind settles an event reference's
// unknown words as a taught component's kind: the whole
// phrase is a kind taught in this project.
func taughtComponentRefKind(unknown string) (string, bool) {
	if unknown == "" {
		return "", false
	}
	if c := taughtComponentHere(unknown); c != nil {
		return c.Kind, true
	}
	return "", false
}

// taughtComponentRefNamed settles an event reference's
// unknown words as a name in front of a taught kind: "the
// total price tag" reads as the price tag called total.
func taughtComponentRefNamed(unknown string) (name, phrase string, ok bool) {
	words := strings.Fields(unknown)
	if len(words) < 2 || !wasmComponentNameToken(words[0]) {
		return "", "", false
	}
	if c := taughtComponentHere(strings.Join(words[1:], " ")); c != nil {
		return words[0], c.Kind, true
	}
	return "", "", false
}

// settleTaughtEventRef settles the one reference shape
// the registry parse can't read for a one-word taught
// kind: the bare name slot. A kind of one word ("banner")
// parses as a bare NAME — that reading stands when a
// mounted component really carries the name, and gives
// way to the kind reading when none does but a taught
// instance of that kind is mounted.
func settleTaughtEventRef(r wasmEventRef, instances []analyze.WasmComponentInstance) wasmEventRef {
	if r.kind != "" || r.name == "" {
		return r
	}
	for _, in := range instances {
		if in.Name == r.name || in.ID == r.name {
			return r
		}
	}
	for _, in := range instances {
		if in.Taught && in.Kind == r.name {
			return wasmEventRef{kind: in.Kind}
		}
	}
	return r
}
