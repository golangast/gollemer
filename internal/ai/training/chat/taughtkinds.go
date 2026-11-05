package chat

// The teach seat: how the chat learns a NEW piece without an
// engineer authoring one. Everything else the composer writes
// comes from the concept book — data, but data a person wrote
// into the product. A taught kind is the user's own data: he
// shows the chat ONE declaration in his project ("learn a
// signup form from Jake in jim/main.go"), the chat reads its
// structure the way the index reads everything — the AST, not
// statistics — and stores a template whose only slot is the
// declaration's name: every place the piece names itself (its
// declaration, its calls to itself, its references to its own
// type) renames consistently, and EVERYTHING else stays
// literal. A later "add a signup form named Signup to
// site/main.go" assembles that template through the same plan,
// gate, transaction and build verification as a book write
// (analyze.EditTaughtDecl), marked as from his example.
//
// The store is the user's, outside every project tree, under
// the memory build's discipline (61dd18c): a sibling file
// ~/.gollemer_taught.json — deliberately NOT a section of the
// memory store, so erasing preferences never erases his
// pieces and forgetting a piece never touches his memory —
// behind a small Load/Save seam, 0600, written atomically.
// Taught kinds are global across his projects (they are his,
// not a project's), they never shadow the concept book or the
// page family (a colliding kind name is refused at teach
// time), and capture is explicit only: the teach sentences
// below, never an inference from ordinary lines.
//
// v1 scope, stated plainly: Go declarations only (a plain
// function or a type — methods are refused in words), the
// example must already exist in the open project (no pasted
// snippets, no whole files, no HTML), and only the name and
// the placement vary — field names, literals and every other
// identifier stay exactly as the example has them. A taught
// piece assembled somewhere the example's assumptions don't
// hold fails the spine's build check like any other edit:
// full restore, the compiler's words, the template kept.

import (
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// ----- the store -----------------------------------------------------------

// taughtNameMarker is the slot stored in a template wherever
// the example declaration names itself. It reads plainly in
// the user's own store file, and the use side substitutes the
// new name for it before planning.
const taughtNameMarker = "{{name}}"

// taughtKind is one taught piece: the template and the facts
// the use side and the listings render from.
type taughtKind struct {
	Kind     string    `json:"kind"`              // normalized phrase ("signup form")
	Display  string    `json:"display"`           // the phrase as he typed it
	Example  string    `json:"example"`           // the example declaration's name
	DeclKind string    `json:"decl_kind"`         // "function" or "type"
	Template string    `json:"template"`          // source with taughtNameMarker slots
	Imports  []string  `json:"imports,omitempty"` // full import specs the decl leans on
	File     string    `json:"file"`              // the example's file, relative to its project
	Root     string    `json:"root"`              // the project the example was taught in
	TaughtAt time.Time `json:"taught_at"`
}

type taughtKindData struct {
	Version int          `json:"version"`
	Kinds   []taughtKind `json:"kinds,omitempty"`
}

func emptyTaughtKindData() taughtKindData { return taughtKindData{Version: 1} }

func normalizeTaughtKindData(d taughtKindData) taughtKindData {
	if d.Version == 0 {
		d.Version = 1
	}
	if d.Kinds == nil {
		d.Kinds = []taughtKind{}
	}
	return d
}

// taughtStore is the storage seam, the memory store's shape:
// load one value or save one value.
type taughtStore interface {
	Load() (taughtKindData, error)
	Save(taughtKindData) error
}

var taughtStorePathOverride string

func openTaughtStore() taughtStore { return jsonTaughtStore{path: taughtStorePath()} }

func taughtStorePath() string {
	if taughtStorePathOverride != "" {
		return taughtStorePathOverride
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ".gollemer_taught.json"
	}
	return filepath.Join(home, ".gollemer_taught.json")
}

type jsonTaughtStore struct{ path string }

func (s jsonTaughtStore) Load() (taughtKindData, error) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return emptyTaughtKindData(), nil
	}
	if err != nil {
		return emptyTaughtKindData(), err
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return emptyTaughtKindData(), nil
	}
	var out taughtKindData
	if err := json.Unmarshal(data, &out); err != nil {
		return emptyTaughtKindData(), err
	}
	return normalizeTaughtKindData(out), nil
}

func (s jsonTaughtStore) Save(d taughtKindData) error {
	d = normalizeTaughtKindData(d)
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".gollemer-taught-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, s.path)
}

func loadTaughtKinds() (taughtKindData, error) { return openTaughtStore().Load() }

func saveTaughtKinds(d taughtKindData) error { return openTaughtStore().Save(d) }

// findTaughtKind returns the stored kind with this normalized
// phrase, or nil.
func (d taughtKindData) findTaughtKind(normalized string) *taughtKind {
	for i := range d.Kinds {
		if d.Kinds[i].Kind == normalized {
			return &d.Kinds[i]
		}
	}
	return nil
}

// normalizeKindPhrase is the kind key everywhere: lowercase,
// single-spaced, punctuation-trimmed.
func normalizeKindPhrase(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(s))), " ")
}

// taughtDate renders a taught date the way the plans and
// listings say it: "Nov 3, 2026".
func taughtDate(t time.Time) string { return t.Format("Jan 2, 2006") }

func (k taughtKind) provenance() string {
	return fmt.Sprintf("From your example %s in %s, taught %s.", k.Example, k.File, taughtDate(k.TaughtAt))
}

// ----- the book and page families own their names --------------------------

// bookKindNames is every name the concept book answers to
// (already normalized by the book's loader) plus the page
// family's kind phrases: a taught kind may sit beside those
// families, never over them, so a kind phrase equal to one of
// these is refused at teach time with the owner named.
func reservedKindNames() map[string]string {
	out := map[string]string{}
	for _, e := range conceptEntries() {
		for _, n := range e.Names {
			out[normalizeKindPhrase(n)] = "the concept book"
		}
	}
	for _, n := range []string{
		"homepage", "home page", "form page", "handler with a form",
		"about page", "about us page", "contact page", "contact us page",
		"item list page", "list page", "details page", "item details page",
	} {
		out[n] = "the page family"
	}
	return out
}

// proposerKindWords are the bare declaration kind words the
// speccopy proposer's grammar keys on; a taught kind carrying
// one as a whole word would sit in front of that claim and is
// refused at teach time for the same never-shadow reason.
var proposerKindWords = map[string]bool{
	"struct": true, "interface": true, "function": true, "func": true, "method": true,
}

// ----- grammar ---------------------------------------------------------------

// splitSentence normalizes a line for the teach-seat grammars:
// trimmed, with sentence-final punctuation removed in a way
// that never eats a file name's ".go".
func splitSentence(line string) (fields []string, low []string) {
	s := strings.TrimSpace(line)
	s = strings.TrimRight(s, "!?")
	if !strings.HasSuffix(strings.ToLower(s), ".go") {
		s = strings.TrimRight(s, ".")
	}
	s = strings.TrimSpace(s)
	fields = strings.Fields(s)
	low = make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(strings.Trim(f, ",;:\"'()"))
	}
	return fields, low
}

// teachAsk is one teach sentence: the kind phrase as typed,
// the example's name (case kept), and the example's file as
// typed ("" = search the project).
type teachAsk struct {
	kind string
	name string
	file string
}

// parseTeachAsk reads the teach shapes:
//
//	learn this as a <kind> from <Name> [in <file>]
//	learn a|an <kind> from <Name> [in <file>]
//	teach you a|an <kind> from <Name> [in <file>]
//
// (the "you" is optional after teach; "this as" only after
// learn). Anything else declines.
func parseTeachAsk(line string) (teachAsk, bool) {
	fields, low := splitSentence(line)
	if len(fields) < 5 {
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
	// The introducer: "this as" or an article.
	if i+1 < len(low) && low[i] == "this" && low[i+1] == "as" {
		if low[0] != "learn" {
			return teachAsk{}, false
		}
		i += 2
		if i < len(low) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
			i++
		}
	} else if i < len(low) && (low[i] == "a" || low[i] == "an") {
		i++
	} else {
		return teachAsk{}, false
	}
	// The kind runs to the first "from" that leaves a valid
	// tail: <Name> or <Name> in [the] [file] <file>.
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

// parseExampleTail reads "<Name>" or "<Name> in [the] [file]
// <file>" starting at index i.
func parseExampleTail(fields, low []string, i int) (name, file string, ok bool) {
	if i >= len(fields) {
		return "", "", false
	}
	name = strings.Trim(fields[i], ",;:\"'()")
	if !token.IsIdentifier(name) {
		return "", "", false
	}
	rest := low[i+1:]
	if len(rest) == 0 {
		return name, "", true
	}
	if rest[0] != "in" {
		return "", "", false
	}
	rest = rest[1:]
	orig := fields[i+2:]
	if len(rest) > 0 && rest[0] == "the" {
		rest = rest[1:]
		orig = orig[1:]
	}
	if len(rest) > 0 && conceptDesignator(rest[0]) {
		rest = rest[1:]
		orig = orig[1:]
	}
	if len(rest) == 0 {
		return "", "", false
	}
	return name, strings.Join(orig, " "), true
}

// taughtUseAsk is one use sentence against the store.
type taughtUseAsk struct {
	kind    string // normalized kind phrase
	name    string // the new declaration's name ("" = the sentence named none)
	target  string // as typed ("" = the sentence named nowhere)
	named   bool   // the sentence carried the "named <Name>" slot at all
	shapeOK bool   // the sentence is the full add-a-kind shape
}

// parseTaughtUse reads the use shapes:
//
//	add|create|make|put|build [a|an|the] <kind> named|called <Name> [to|in|into [the] [file|folder] <target>]
//	add|create|make|put|build [a|an|the] <kind> [to|in|into ... <target>]   (no name: guidance when the kind is taught)
//
// The kind is whatever runs between the article and the
// "named" slot (or the target tail); whether it names a
// taught kind is the caller's question, not the parser's.
//
// (put|build joined the verbs and "called" the name slot
// 2026-11-05: both are the neighbouring claims' own words —
// the creations read "named|called" the same way — and a use
// sentence spelled with either parsed as an unknown kind
// before, the name swallowed into the kind phrase.)
func parseTaughtUse(line string) (taughtUseAsk, bool) {
	fields, low := splitSentence(line)
	if len(fields) < 3 {
		return taughtUseAsk{}, false
	}
	switch low[0] {
	case "add", "create", "make", "put", "build":
	default:
		return taughtUseAsk{}, false
	}
	i := 1
	if i < len(low) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	// The name slot, if the sentence carries one: "named
	// <Name>", or "called <Name>" — the creations' second
	// spelling of the same introducer.
	namedAt := -1
	for j := i; j < len(low); j++ {
		if low[j] == "named" || low[j] == "called" {
			namedAt = j
			break
		}
	}
	if namedAt >= 0 {
		if namedAt == i || namedAt+1 >= len(fields) {
			return taughtUseAsk{}, false
		}
		u := taughtUseAsk{named: true, shapeOK: true}
		u.kind = normalizeKindPhrase(strings.Join(fields[i:namedAt], " "))
		if u.kind == "" {
			return taughtUseAsk{}, false
		}
		u.name = strings.Trim(fields[namedAt+1], ",;:\"'()")
		target, ok := parseTargetTail(fields, low, namedAt+2)
		if !ok {
			return taughtUseAsk{}, false
		}
		u.target = target
		return u, true
	}
	// No name slot: the kind runs to the target tail or the
	// end of the sentence.
	end := len(low)
	for j := i; j+1 < len(low); j++ {
		if isTargetPrep(low[j]) {
			end = j
			break
		}
	}
	if end == i {
		return taughtUseAsk{}, false
	}
	u := taughtUseAsk{kind: normalizeKindPhrase(strings.Join(fields[i:end], " "))}
	if u.kind == "" {
		return taughtUseAsk{}, false
	}
	if end < len(low) {
		target, ok := parseTargetTail(fields, low, end)
		if !ok {
			return taughtUseAsk{}, false
		}
		u.target = target
	}
	return u, true
}

func isTargetPrep(w string) bool { return w == "to" || w == "in" || w == "into" }

// parseTargetTail reads a use sentence's target tail starting
// at the preposition index: "to|in|into [the] [file|folder]
// <target>". An absent tail (index past the end) is fine and
// yields "".
func parseTargetTail(fields, low []string, i int) (string, bool) {
	if i >= len(low) {
		return "", true
	}
	if !isTargetPrep(low[i]) {
		return "", false
	}
	i++
	if i < len(low) && low[i] == "the" {
		i++
	}
	if i < len(low) && conceptDesignator(low[i]) {
		i++
	}
	if i >= len(fields) {
		return "", false
	}
	return strings.Join(fields[i:], " "), true
}

// ----- reading the example ---------------------------------------------------

// exampleDecl is one declaration found in a parsed file: the
// node the template is cut from, its kind in words, and the
// facts the refusals and the summary render.
type exampleDecl struct {
	kind     string // "function", "type", "method", or "value"
	recvType string // methods only: the receiver's base type
	node     ast.Decl
	spec     *ast.TypeSpec // types only: the spec (node is its GenDecl)
	line     int
}

// findExampleDecl finds the top-level declaration named name
// in a parsed file: a plain function, a method, a type, or a
// package-level value, in that order.
func findExampleDecl(fset *token.FileSet, fileAST *ast.File, name string) *exampleDecl {
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && fd.Name.Name == name {
			ex := &exampleDecl{kind: "function", node: fd, line: lineOf(fset, fd.Pos())}
			if fd.Recv != nil && len(fd.Recv.List) > 0 {
				ex.kind = "method"
				ex.recvType = strings.TrimPrefix(exprText(fd.Recv.List[0].Type), "*")
			}
			return ex
		}
	}
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, sp := range gd.Specs {
			switch sp := sp.(type) {
			case *ast.TypeSpec:
				if sp.Name.Name == name {
					return &exampleDecl{kind: "type", node: gd, spec: sp, line: lineOf(fset, sp.Pos())}
				}
			case *ast.ValueSpec:
				for _, n := range sp.Names {
					if n.Name == name {
						return &exampleDecl{kind: "value", node: gd, line: lineOf(fset, sp.Pos())}
					}
				}
			}
		}
	}
	return nil
}

func lineOf(fset *token.FileSet, pos token.Pos) int {
	return fset.Position(pos).Line
}

func exprText(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + exprText(t.X)
	case *ast.SelectorExpr:
		return exprText(t.X) + "." + t.Sel.Name
	case *ast.IndexExpr:
		return exprText(t.X)
	case *ast.IndexListExpr:
		return exprText(t.X)
	}
	return ""
}

// buildDeclTemplate cuts the stored template from the
// example's source: the declaration's own text (its leading
// doc comment left behind — it names the example, not the
// piece), with every place the declaration names itself
// replaced by the slot marker. What counts as naming itself:
// the declaration's own name, and every identifier equal to
// it inside the declaration — a recursive call, a reference
// to the declared type — EXCEPT an identifier that names
// something else by construction: a selector's member
// (x.Name), a composite literal's field key, or a declared
// field/parameter name (a struct field that shares the type's
// name is a field, not a self-reference).
func buildDeclTemplate(src []byte, fset *token.FileSet, ex *exampleDecl, name string) string {
	file := fset.File(ex.node.Pos())
	if file == nil {
		return ""
	}
	off := func(pos token.Pos) int { return file.Offset(pos) }
	var walkRoot ast.Node = ex.node
	start, end := off(ex.node.Pos()), off(ex.node.End())
	prefix := ""
	if ex.kind == "type" {
		if gd, ok := ex.node.(*ast.GenDecl); ok && len(gd.Specs) > 1 {
			// One type out of a grouped declaration: the spec
			// alone, re-headed with its keyword.
			walkRoot = ex.spec
			start, end = off(ex.spec.Pos()), off(ex.spec.End())
			prefix = "type "
		}
	}
	var slots [][2]int
	var stack []ast.Node
	ast.Inspect(walkRoot, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return false
		}
		if id, ok := n.(*ast.Ident); ok && id.Name == name && !identNamesAnother(stack, id) {
			slots = append(slots, [2]int{off(id.Pos()), off(id.End())})
		}
		stack = append(stack, n)
		return true
	})
	if len(slots) == 0 {
		return ""
	}
	sort.Slice(slots, func(a, b int) bool { return slots[a][0] > slots[b][0] })
	text := string(src[start:end])
	for _, s := range slots {
		text = text[:s[0]-start] + taughtNameMarker + text[s[1]-start:]
	}
	return prefix + text
}

// identNamesAnother reports whether an identifier names
// something else by construction rather than referring to
// the package-level declaration: a selector's member
// (x.Name — the X is a different node and is judged on its
// own), a composite literal's field key, or a declared name
// in a field list (a struct field, parameter, result, or
// type parameter that happens to share the declaration's
// name — the field's TYPE, like the *Store in "Next *Store",
// is a different node and still counts as a self-reference).
func identNamesAnother(stack []ast.Node, id *ast.Ident) bool {
	if len(stack) == 0 {
		return false
	}
	switch parent := stack[len(stack)-1].(type) {
	case *ast.SelectorExpr:
		return parent.Sel == id
	case *ast.KeyValueExpr:
		return parent.Key == id
	case *ast.Field:
		for _, n := range parent.Names {
			if n == id {
				return true
			}
		}
	}
	return false
}

// declImportSpecs reconciles the example file's imports to
// the ones the example declaration itself leans on: a spec is
// carried when its local name is used inside the declaration
// (a method call on a package, a qualified type), and blank
// imports always travel (a driver registration the body may
// need without naming). Dot imports are not carried in v1 —
// their names can't be attributed to this declaration — and
// neither are imports only the file's other declarations use.
func declImportSpecs(src []byte, fset *token.FileSet, fileAST *ast.File, root ast.Node) []string {
	file := fset.File(fileAST.Pos())
	if file == nil {
		return nil
	}
	off := func(pos token.Pos) int { return file.Offset(pos) }
	used := map[string]bool{}
	ast.Inspect(root, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if id, ok := sel.X.(*ast.Ident); ok {
				used[id.Name] = true
			}
		}
		return true
	})
	var out []string
	for _, sp := range fileAST.Imports {
		path, err := strconv.Unquote(sp.Path.Value)
		if err != nil {
			continue
		}
		alias := ""
		if sp.Name != nil {
			alias = sp.Name.Name
		}
		carry := false
		switch alias {
		case "_":
			carry = true
		case ".":
			carry = false
		default:
			local := alias
			if local == "" {
				local = importLocalName(path)
			}
			carry = used[local]
		}
		if carry {
			out = append(out, string(src[off(sp.Pos()):off(sp.End())]))
		}
	}
	sort.Strings(out)
	return out
}

// importLocalName derives an import's local package name from
// its path the usual way: the last element, with a trailing
// version suffix (".v3") removed.
func importLocalName(path string) string {
	base := path
	if i := strings.LastIndex(path, "/"); i >= 0 {
		base = path[i+1:]
	}
	if i := strings.LastIndex(base, ".v"); i > 0 {
		if _, err := strconv.Atoi(base[i+2:]); err == nil {
			base = base[:i]
		}
	}
	return base
}

// ----- the claim ---------------------------------------------------------------

// tryTaughtKinds is the chat-loop entry: the teach seat's
// three sentences — teach, use, manage — claimed
// deterministically after the concept book and before the
// speccopy proposer (taught kinds are deterministic data, so
// they sit with the deterministic writers; the book keeps its
// own sentences because it sits earlier and because a taught
// kind may never shadow a book name).
func tryTaughtKinds(line string, r *LineReader) bool {
	return runTaughtKinds(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

func runTaughtKinds(line string, ask editAsk) bool {
	if handled := runTaughtManage(line); handled {
		return true
	}
	if ta, ok := parseTeachAsk(line); ok {
		runTeach(ta)
		return true
	}
	if u, ok := parseTaughtUse(line); ok {
		return runTaughtUse(line, u, ask)
	}
	return false
}

// taughtProjectRoot settles the project the teach seat reads
// and writes in, with runConcept's own refusals (the teach
// seat edits nothing itself, but a use ask writes through the
// spine, and a teach ask reads the project's code).
func taughtProjectRoot() (rootRaw, rootAbs string, ok bool) {
	tag := cTag("gollemer [composer]>")
	rootRaw = creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to work in — start me inside your project and ask again.\n", tag)
		return "", "", false
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd work in (%s).\n", tag, rootRaw)
			return "", "", false
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd work in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		return "", "", false
	}
	return rootRaw, rootAbs, true
}

// taughtFreshProject makes the analyzed index current, the
// concept flow's way.
func taughtFreshProject(rootRaw, rootAbs string) bool {
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		return analyzeQuiet(rootAbs)
	}
	return ensureFreshProject()
}

// ----- teach -----------------------------------------------------------------

func runTeach(ta teachAsk) {
	tag := cTag("gollemer [composer]>")
	kind := normalizeKindPhrase(ta.kind)

	// The kind's name is the sentence's own contract: it may
	// not contain the words the use grammar parses by, may
	// not shadow a family that already owns a name, and may
	// not repeat a kind already taught.
	for _, w := range strings.Fields(kind) {
		if w == "named" || w == "from" {
			fmt.Printf("%s Not learning that: a kind's name can't contain the word %q — the sentences that teach and use it parse by that word, so it would tangle them. Pick a plainer name, like \"signup form\".\n", tag, w)
			recordBrain("composer", false, "taught refused: kind word")
			return
		}
		if proposerKindWords[w] {
			fmt.Printf("%s Not learning that: %q is a declaration kind word the book and the proposer already parse by — a taught kind named with it would sit in front of their sentences. Pick a name of your own for the kind, like \"signup form\".\n", tag, w)
			recordBrain("composer", false, "taught refused: kind word")
			return
		}
	}
	if owner, taken := reservedKindNames()[kind]; taken {
		fmt.Printf("%s Not learning that: %q is already one of %s's pieces — your taught kinds sit beside them, never over them, so pick a different name for the kind, like \"signup form\".\n", tag, kind, owner)
		recordBrain("composer", false, "taught refused: kind shadows a family")
		return
	}
	data, err := loadTaughtKinds()
	if err != nil {
		fmt.Printf("%s I couldn't read your taught-pieces store (%s): %v\n", tag, taughtStorePath(), err)
		recordBrain("composer", false, "taught refused: store unreadable")
		return
	}
	if old := data.findTaughtKind(kind); old != nil {
		fmt.Printf("%s Not learning that: you've already taught me a %q — %s If you want to teach me a different one under this name, forget that one first: say \"forget the %s\".\n", tag, kind, old.provenance(), kind)
		recordBrain("composer", false, "taught refused: kind already taught")
		return
	}

	rootRaw, rootAbs, ok := taughtProjectRoot()
	if !ok {
		recordBrain("composer", false, "taught refused: no root")
		return
	}
	if !taughtFreshProject(rootRaw, rootAbs) {
		recordBrain("composer", false, "taught refused: project unreadable")
		return
	}

	src, fset, fileAST, ex, rel, refusal := locateExample(rootAbs, ta)
	if refusal != "" {
		fmt.Printf("%s Not learning that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "taught refused: example")
		return
	}

	tmpl := buildDeclTemplate(src, fset, ex, ta.name)
	if tmpl == "" {
		fmt.Printf("%s Not learning that: I couldn't cut a template from %s in %s — nothing was saved.\n", tag, ta.name, rel)
		recordBrain("composer", false, "taught refused: template")
		return
	}
	imports := declImportSpecs(src, fset, fileAST, templateRoot(ex))

	entry := taughtKind{
		Kind:     kind,
		Display:  strings.Join(strings.Fields(ta.kind), " "),
		Example:  ta.name,
		DeclKind: ex.kind,
		Template: tmpl,
		Imports:  imports,
		File:     rel,
		Root:     rootAbs,
		TaughtAt: time.Now(),
	}

	// The summary prints BEFORE the save, and whether or not
	// the save then succeeds the words below are the truth
	// about what would be kept.
	fmt.Printf("%s Here's what I learned from %s in %s:\n", tag, ta.name, rel)
	fmt.Printf("  The kind: %q — from here on, \"add a %s named <Name> to <file>\" assembles one.\n", kind, kind)
	fmt.Printf("  The example: the %s %s, at %s:%d.\n", ex.kind, ta.name, rel, ex.line)
	switch ex.kind {
	case "function":
		fmt.Printf("  What varies: the name — %s becomes the new name everywhere the piece names itself, in its declaration and in its calls to itself — and where it lands.\n", ta.name)
	default:
		fmt.Printf("  What varies: the name — %s becomes the new name everywhere the piece names itself, in its declaration and in its references to itself — and where it lands.\n", ta.name)
	}
	if len(imports) == 0 {
		fmt.Printf("  What stays fixed: everything else — the code stays exactly as in %s, and it leans on no imports.\n", ta.name)
	} else {
		fmt.Printf("  What stays fixed: everything else — the code stays exactly as in %s, and the imports it leans on (%s) travel with it.\n", ta.name, strings.Join(importPathsOf(imports), ", "))
	}
	data.Kinds = append(data.Kinds, entry)
	if err := saveTaughtKinds(data); err != nil {
		fmt.Printf("  I couldn't save it to your store (%s): %v — nothing was kept.\n", taughtStorePath(), err)
		recordBrain("composer", false, "taught refused: store unwritable")
		return
	}
	fmt.Printf("  Saved as yours: it lives in %s, outside every project, so it works in all of them. Say \"forget the %s\" to remove it.\n", taughtStorePath(), kind)
	recordBrain("composer", true, "taught learned: "+kind)
}

// templateRoot is the node the template (and its import
// reconciliation) is cut from: the whole declaration, or the
// one spec when the type sits in a grouped declaration.
func templateRoot(ex *exampleDecl) ast.Node {
	if ex.kind == "type" {
		if gd, ok := ex.node.(*ast.GenDecl); ok && len(gd.Specs) > 1 {
			return ex.spec
		}
	}
	return ex.node
}

// importPathsOf renders stored import specs as bare paths
// for display.
func importPathsOf(specs []string) []string {
	out := make([]string, 0, len(specs))
	for _, sp := range specs {
		if i := strings.LastIndex(sp, " "); i >= 0 {
			sp = sp[i+1:]
		}
		if p, err := strconv.Unquote(strings.TrimSpace(sp)); err == nil {
			out = append(out, p)
		}
	}
	return out
}

// locateExample resolves the teach sentence's example to a
// declaration in the project, rendering every refusal in the
// teach family's words. On success it returns the example
// file's bytes and parse (the template is cut from them), the
// declaration found, and its project-relative file.
func locateExample(rootAbs string, ta teachAsk) (src []byte, fset *token.FileSet, fileAST *ast.File, ex *exampleDecl, rel string, refusal string) {
	if ta.file != "" {
		abs, relT, fence := serverTargetResolve(rootAbs, ta.file)
		if fence != "" {
			return nil, nil, nil, nil, "", fence
		}
		st, err := os.Stat(abs)
		if err != nil || st.IsDir() {
			return nil, nil, nil, nil, "", fmt.Sprintf("there's no file %s in this project — name the file the example lives in", strings.TrimSuffix(ta.file, "/"))
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
		found := findExampleDecl(fset, fileAST, ta.name)
		if found == nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't find a declaration named %q in %s — there's no function or type with that name in that file", ta.name, relT)
		}
		if refusal := exampleKindRefusal(found, ta, relT); refusal != "" {
			return nil, nil, nil, nil, "", refusal
		}
		return src, fset, fileAST, found, relT, ""
	}

	// No file named: the index searches the whole project.
	var reals, methods []exampleCandidate
	for _, pkg := range lastAnalyzeProject.Packages {
		for _, fn := range pkg.Funcs {
			if fn.Name != ta.name || fn.IsTest {
				continue
			}
			if fn.Receiver == "" {
				reals = append(reals, exampleCandidate{kind: "function", file: fn.File, line: fn.Line})
			} else {
				methods = append(methods, exampleCandidate{kind: "method", recv: strings.TrimPrefix(fn.Receiver, "*"), file: fn.File, line: fn.Line})
			}
		}
		for _, t := range pkg.Types {
			if t.Name != ta.name || strings.HasSuffix(t.File, "_test.go") {
				continue
			}
			reals = append(reals, exampleCandidate{kind: "type", file: t.File, line: t.Line})
		}
	}
	switch {
	case len(reals) == 1:
		c := reals[0]
		src, err := os.ReadFile(filepath.Join(rootAbs, filepath.FromSlash(c.file)))
		if err != nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't read %s (%v)", c.file, err)
		}
		fset := token.NewFileSet()
		fileAST, perr := parser.ParseFile(fset, c.file, src, parser.ParseComments)
		if perr != nil || fileAST == nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("%s doesn't parse right now (%v), so I can't read %s's shape from it — fix the file and ask again", c.file, perr, ta.name)
		}
		found := findExampleDecl(fset, fileAST, ta.name)
		if found == nil {
			return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't find a declaration named %q in %s", ta.name, c.file)
		}
		return src, fset, fileAST, found, c.file, ""
	case len(reals) > 1:
		var parts []string
		for _, c := range reals {
			parts = append(parts, fmt.Sprintf("the %s %s in %s", c.kind, ta.name, c.file))
		}
		return nil, nil, nil, nil, "", fmt.Sprintf("more than one declaration is named %s — %s. Say which one you mean by naming its file, like \"learn a %s from %s in %s\"", ta.name, strings.Join(parts, " and "), normalizeKindPhrase(ta.kind), ta.name, reals[0].file)
	case len(methods) > 0:
		m := methods[0]
		return nil, nil, nil, nil, "", methodRefusalText(ta.name, m.recv, m.file)
	default:
		return nil, nil, nil, nil, "", fmt.Sprintf("I couldn't find a declaration named %q in this project — there's no function or type with that name. If it doesn't exist yet, add it first, then teach me from it", ta.name)
	}
}

type exampleCandidate struct {
	kind, recv, file string
	line             int
}

// exampleKindRefusal renders the refusal for a declaration
// whose kind v1 can't teach from (a method, a value), or ""
// when the example is teachable.
func exampleKindRefusal(ex *exampleDecl, ta teachAsk, rel string) string {
	switch ex.kind {
	case "method":
		return methodRefusalText(ta.name, ex.recvType, rel)
	case "value":
		return fmt.Sprintf("%s in %s is a package-level value, not a function or a type — this first version of teaching learns from a function or a type only", ta.name, rel)
	}
	return ""
}

func methodRefusalText(name, recv, rel string) string {
	return fmt.Sprintf("%s in %s is a method — it hangs on the type %s, so its shape isn't a piece on its own — and this first version of teaching learns from a plain function or a type only. Teach me from the type %s itself, or from a plain function", name, rel, recv, recv)
}

// ----- use ---------------------------------------------------------------------

func runTaughtUse(line string, u taughtUseAsk, ask editAsk) bool {
	tag := cTag("gollemer [composer]>")
	data, err := loadTaughtKinds()
	if err != nil {
		fmt.Printf("%s I couldn't read your taught-pieces store (%s): %v\n", tag, taughtStorePath(), err)
		recordBrain("composer", false, "taught use refused: store unreadable")
		return true
	}
	entry := data.findTaughtKind(u.kind)
	if entry == nil {
		// A complete use shape naming a kind nobody taught
		// (and no family owns) gets the honest answer here —
		// the alternative is the boundary's generic one, and
		// this claim can say the useful thing: teach me. The
		// shape guards keep it narrow: a "named" slot, a
		// multiword kind, and no word the book or the proposer
		// parses by.
		if u.shapeOK && unknownKindAnswerable(u.kind) {
			fmt.Printf("%s I don't know a %q — the concept book has no piece by that name, and you haven't taught me one. Teach me from an example in your project: say \"learn a %s from <Name> in <file>\", and I'll keep its shape as yours.\n", tag, u.kind, u.kind)
			recordBrain("composer", false, "taught use refused: unknown kind")
			return true
		}
		return false
	}
	kind := entry.Kind
	if !u.named || u.name == "" {
		fmt.Printf("%s Not adding that: a %s needs a name — say \"add a %s named <Name> to <file>\", and it lands as your example %s with the name changed.\n", tag, kind, kind, entry.Example)
		recordBrain("composer", false, "taught use refused: unnamed")
		return true
	}
	if u.target == "" {
		fmt.Printf("%s Not adding that: the sentence doesn't name the file or folder — say it like \"add a %s named %s to jim/main.go\" (a file) or \"add a %s named %s to jim\" (its folder).\n", tag, kind, u.name, kind, u.name)
		recordBrain("composer", false, "taught use refused: no target")
		return true
	}

	rootRaw, rootAbs, ok := taughtProjectRoot()
	if !ok {
		recordBrain("composer", false, "taught use refused: no root")
		return true
	}
	// Target resolution under the declaration family's shared
	// settle rules, with the kind's own phrase in the words.
	rel, refusal := resolveConceptDeclTarget(rootAbs, u.target, u.name, kind)
	if refusal != "" {
		fmt.Printf("%s Not adding that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "taught use refused: target")
		return true
	}
	if !taughtFreshProject(rootRaw, rootAbs) {
		recordBrain("composer", false, "taught use refused: project unreadable")
		return true
	}
	text := strings.ReplaceAll(entry.Template, taughtNameMarker, u.name)
	return executeEditPlan(analyze.EditRequest{
		Kind:             analyze.EditTaughtDecl,
		File:             rel,
		TaughtText:       text,
		TaughtName:       u.name,
		TaughtDeclKind:   entry.DeclKind,
		TaughtKindPhrase: kind,
		TaughtImports:    strings.Join(entry.Imports, " "),
		TaughtNote:       entry.provenance(),
	}, ask)
}

// unknownKindAnswerable reports whether an unknown kind in a
// complete use shape is this claim's to answer: multiword
// (taught kinds are phrases; the one-word unknowns belong to
// the families that parse single words), not a reserved name,
// and carrying no word the proposer parses by.
func unknownKindAnswerable(kind string) bool {
	words := strings.Fields(kind)
	if len(words) < 2 {
		return false
	}
	if _, taken := reservedKindNames()[kind]; taken {
		return false
	}
	for _, w := range words {
		if proposerKindWords[w] {
			return false
		}
	}
	return true
}

// ----- manage ------------------------------------------------------------------

// The manage shapes: listing and forgetting. "forget my …"
// stays the memory store's (its preferences); the teach
// store's forgets are "forget the <kind>" and the two all
// forms.
var (
	taughtListRe      = regexp.MustCompile(`(?i)^\s*(what have i taught you|what did i teach you|what i taught you|what have you learned from me|what did you learn from me|list what (i've|i have) taught you|show what (i've|i have) taught you|show me what (i've |i have |i )?taught you)\s*[?.!]*\s*$`)
	taughtForgetAllRe = regexp.MustCompile(`(?i)^\s*forget (what i taught you|everything i taught you|all i taught you|everything you learned from me)\s*[!.]*\s*$`)
	taughtForgetRe    = regexp.MustCompile(`(?i)^\s*forget the\s+(.+?)(?:\s+i taught you)?\s*[!.]*\s*$`)
)

func runTaughtManage(line string) bool {
	if taughtListRe.MatchString(line) {
		listTaughtKinds()
		return true
	}
	if taughtForgetAllRe.MatchString(line) {
		forgetAllTaughtKinds()
		return true
	}
	if m := taughtForgetRe.FindStringSubmatch(line); m != nil {
		forgetTaughtKind(normalizeKindPhrase(m[1]))
		return true
	}
	return false
}

func listTaughtKinds() {
	tag := cTag("gollemer [composer]>")
	data, err := loadTaughtKinds()
	if err != nil {
		fmt.Printf("%s I couldn't read your taught-pieces store (%s): %v\n", tag, taughtStorePath(), err)
		recordBrain("composer", false, "taught list refused: store unreadable")
		return
	}
	if len(data.Kinds) == 0 {
		fmt.Printf("%s You haven't taught me any pieces yet. Teach me one from a declaration in your project — say \"learn a sidebar page from Jake in jim/main.go\" — and I'll keep its shape as yours, in every project.\n", tag)
		recordBrain("composer", true, "taught listed: none")
		return
	}
	fmt.Printf("%s What you've taught me:\n", tag)
	for _, k := range data.Kinds {
		fmt.Printf("  %q — from your example %s (%s) in %s, taught %s.\n", k.Kind, k.Example, k.DeclKind, k.File, taughtDate(k.TaughtAt))
	}
	fmt.Printf("  Use one by its kind — \"add a <kind> named <Name> to <file>\" — or remove one with \"forget the <kind>\".\n")
	recordBrain("composer", true, "taught listed")
}

func forgetTaughtKind(kind string) {
	tag := cTag("gollemer [composer]>")
	data, err := loadTaughtKinds()
	if err != nil {
		fmt.Printf("%s I couldn't read your taught-pieces store (%s): %v\n", tag, taughtStorePath(), err)
		recordBrain("composer", false, "taught forget refused: store unreadable")
		return
	}
	for i := range data.Kinds {
		if data.Kinds[i].Kind == kind {
			gone := data.Kinds[i]
			data.Kinds = append(data.Kinds[:i], data.Kinds[i+1:]...)
			if err := saveTaughtKinds(data); err != nil {
				fmt.Printf("%s I couldn't update your taught-pieces store (%s): %v — nothing was forgotten.\n", tag, taughtStorePath(), err)
				recordBrain("composer", false, "taught forget refused: store unwritable")
				return
			}
			fmt.Printf("%s Forgotten: your %q — from your example %s in %s. I won't assemble it again.\n", tag, gone.Kind, gone.Example, gone.File)
			recordBrain("composer", true, "taught forgotten: "+kind)
			return
		}
	}
	if len(data.Kinds) == 0 {
		fmt.Printf("%s You haven't taught me a %q — you haven't taught me anything yet, so there's nothing to forget.\n", tag, kind)
	} else {
		names := make([]string, 0, len(data.Kinds))
		for _, k := range data.Kinds {
			names = append(names, fmt.Sprintf("%q", k.Kind))
		}
		fmt.Printf("%s You haven't taught me a %q — the pieces you have taught me are %s. Nothing was forgotten.\n", tag, kind, strings.Join(names, " and "))
	}
	recordBrain("composer", true, "taught forget: no such kind")
}

func forgetAllTaughtKinds() {
	tag := cTag("gollemer [composer]>")
	data, err := loadTaughtKinds()
	if err != nil {
		fmt.Printf("%s I couldn't read your taught-pieces store (%s): %v\n", tag, taughtStorePath(), err)
		recordBrain("composer", false, "taught forget refused: store unreadable")
		return
	}
	if len(data.Kinds) == 0 {
		fmt.Printf("%s You haven't taught me anything yet — nothing to forget.\n", tag)
		recordBrain("composer", true, "taught forget: none")
		return
	}
	parts := make([]string, 0, len(data.Kinds))
	for _, k := range data.Kinds {
		parts = append(parts, fmt.Sprintf("%q (from %s in %s)", k.Kind, k.Example, k.File))
	}
	n := len(data.Kinds)
	data.Kinds = []taughtKind{}
	if err := saveTaughtKinds(data); err != nil {
		fmt.Printf("%s I couldn't update your taught-pieces store (%s): %v — nothing was forgotten.\n", tag, taughtStorePath(), err)
		recordBrain("composer", false, "taught forget refused: store unwritable")
		return
	}
	word := "pieces"
	if n == 1 {
		word = "piece"
	}
	fmt.Printf("%s Forgotten: everything you taught me — %d %s: %s.\n", tag, n, word, strings.Join(parts, ", "))
	recordBrain("composer", true, "taught forgotten: all")
}
