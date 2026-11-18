package chat

// The full-stack piece (John 2026-11-09: "add something
// for restful and forms with gowasm and the go backend
// and the database" — approved 2026-11-10): one sentence
// grows one named thing through every layer at once —
//
//	add a full stack app called <thing> in <folder>
//	    with fields <field> [kind], <field> [kind] and …
//
// the database table and the REST handlers on the Go
// backend, and the form and the list in the wasm app
// (analyze/editfullstack.go grows the code; this file is
// the sentence). The folder must already hold a wasm app
// — the stack's form lives in it — so an app-less folder
// meets the family's two-step naming "add a wasm app in
// <folder>", never a silent app creation inside this ask.
//
// The composition reads the route's facts the
// serve-the-wasm-app way (ServeWasmRouteFacts) before
// planning: a /static/ conflict refuses in the
// serve-static family's own words, and the backend's
// home is resolved to the project's server in server/ or
// its root-level program — or, with neither, the planner
// creates the server in server/. The landing runs the
// one spine with one extra in-transaction check: the
// wasm app's own compile, because the root build can't
// see into the app's module and a stack whose form
// doesn't compile must not land (edits.go's
// executeEditPlanChecked).
//
// Field kinds are a closed set of three — text (the
// default, unnamed), number, yesno ("yes-no" as typed
// works too). The thing's name is used exactly as given
// for the route and the table; the planner's rules
// (analyze.StackFieldsProblem) are applied here first,
// so a name or field the stack can't grow gets its
// guidance before any plan is assembled.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// fullStackAsk is one parsed full-stack sentence: the
// thing as typed, the folder as typed ("" when the
// sentence doesn't name one), the fields clause as typed
// (kept whole, so a clarify completion rebuilds the
// sentence with its fields untouched), and the fields
// themselves once the clause parses.
type fullStackAsk struct {
	thing     string
	folder    string
	hasFolder bool
	fieldsRaw string
	fields    []analyze.StackField
	// fieldsState: "ok" when the clause parsed, "missing"
	// when the sentence carries no fields clause at all,
	// or the problem's own words when the clause is there
	// but malformed (a segment that isn't a name with at
	// most one kind word).
	fieldsState string
}

// parseFullStackFields reads the fields clause — a list
// of segments split on commas and on " and ", each
// segment a field name with at most one kind word after
// it. It returns the fields, or "" and the problem in
// words for a segment outside that shape. The semantic
// rules (the name's shape, id, duplicates, Go-name
// clashes) are analyze.StackFieldsProblem's, applied by
// the caller over the parsed list.
func parseFullStackFields(raw string) ([]analyze.StackField, string) {
	var out []analyze.StackField
	for _, commaPart := range strings.Split(raw, ",") {
		for _, seg := range strings.Split(commaPart, " and ") {
			toks := strings.Fields(seg)
			if len(toks) == 0 {
				continue
			}
			if len(toks) > 2 {
				return nil, fmt.Sprintf("%q isn't a field's shape — a field is its name with at most one kind word after it (\"pages number\"), and the list is fields split by commas and \"and\".", strings.TrimSpace(seg))
			}
			f := analyze.StackField{Name: toks[0], Kind: analyze.StackFieldText}
			if len(toks) == 2 {
				switch strings.ToLower(toks[1]) {
				case "text":
					f.Kind = analyze.StackFieldText
				case "number":
					f.Kind = analyze.StackFieldNumber
				case "yesno", "yes-no":
					f.Kind = analyze.StackFieldYesNo
				default:
					return nil, fmt.Sprintf("%q isn't a field kind this sentence knows — a field is text (the default, so the word can be left off), number, or yesno (\"yes-no\" works too).", toks[1])
				}
			}
			out = append(out, f)
		}
	}
	if len(out) == 0 {
		return nil, "the fields clause names no fields — a full stack needs at least one: the form, the table, and the JSON are all built from the field list."
	}
	return out, ""
}

// parseFullStack reads the sentence: a verb from the
// family's set, an optional article, "full stack" (or
// "full-stack" / "fullstack"), "app", "called", the
// thing, then the family's optional folder tail, then an
// optional "with fields …" tail in that order — the
// brief's one shape. Anything else on the line is not
// this sentence.
func parseFullStack(line string) (fullStackAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 5 {
		return fullStackAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return fullStackAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	switch {
	case i+1 < len(fields) && low[i] == "full" && low[i+1] == "stack":
		i += 2
	case i < len(fields) && (low[i] == "full-stack" || low[i] == "fullstack"):
		i++
	default:
		return fullStackAsk{}, false
	}
	if i >= len(fields) || low[i] != "app" {
		return fullStackAsk{}, false
	}
	i++
	if i >= len(fields) || low[i] != "called" {
		return fullStackAsk{}, false
	}
	i++
	if i >= len(fields) {
		return fullStackAsk{}, false
	}
	out := fullStackAsk{thing: fields[i], fieldsState: "missing"}
	i++
	if i < len(fields) && (low[i] == "in" || low[i] == "into" || low[i] == "at") {
		i++
		if i < len(fields) && low[i] == "the" {
			i++
		}
		if i < len(fields) && (low[i] == "folder" || low[i] == "directory") {
			i++
		}
		if i >= len(fields) {
			return fullStackAsk{}, false
		}
		tail := low[i]
		if strings.HasSuffix(tail, ".html") || strings.HasSuffix(tail, ".go") {
			return fullStackAsk{}, false
		}
		out.folder = fields[i]
		out.hasFolder = true
		i++
	}
	if i < len(fields) {
		if low[i] != "with" || i+1 >= len(fields) || low[i+1] != "fields" {
			return fullStackAsk{}, false
		}
		out.fieldsRaw = strings.Join(fields[i+2:], " ")
		i = len(fields)
		if out.fieldsRaw == "" {
			out.fieldsState = "the fields clause names no fields — a full stack needs at least one: the form, the table, and the JSON are all built from the field list."
		} else if parsed, problem := parseFullStackFields(out.fieldsRaw); problem != "" {
			out.fieldsState = problem
		} else {
			out.fields = parsed
			out.fieldsState = "ok"
		}
	}
	if i != len(fields) {
		return fullStackAsk{}, false
	}
	return out, true
}

// fullStackCanonical renders the sentence with its
// folder named and its fields clause as typed — the
// clarify completion's rebuild.
func (q fullStackAsk) canonical() string {
	s := "add a full stack app called " + q.thing
	if q.hasFolder {
		s += " in " + q.folder
	}
	if q.fieldsRaw != "" {
		s += " with fields " + q.fieldsRaw
	}
	return s
}

// tryFullStack is the chat-loop entry: it claims
// full-stack sentences and runs the flow on the chat's
// line reader.
func tryFullStack(line string, r *LineReader) bool {
	return runFullStack(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runFullStack carries one line for this claim. It
// returns false — the line untouched, nothing printed —
// for every line outside the claim's shape; a shaped
// line is claimed (true) whether it ends landed,
// refused, or asked about.
func runFullStack(line string, ask editAsk) bool {
	q, ok := parseFullStack(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")

	// The sentence's own facts first: the fields clause
	// and the names it carries. A stack can't be planned
	// over a clause that doesn't parse, and the user
	// deserves the guidance before any project question.
	if q.fieldsState == "missing" {
		fmt.Printf("%s Not growing that stack yet: the sentence names no fields — the form, the table, and the JSON are all built from the field list. The whole shape is: \"add a full stack app called %s in %s with fields title, pages number and done yes-no\" — one field per thing the stack keeps, a kind word (text, number, yesno) after a field when it isn't text.\n", tag, q.thing, folderWords(q))
		recordBrain("composer", false, "full stack question: fields unnamed")
		return true
	}
	if q.fieldsState != "ok" {
		fmt.Printf("%s Not growing that stack: %s\n", tag, q.fieldsState)
		recordBrain("composer", false, "full stack refused: fields clause")
		return true
	}
	if problem := analyze.StackFieldsProblem(q.thing, q.fields); problem != "" {
		fmt.Printf("%s Not growing that stack: %s\n", tag, problem)
		recordBrain("composer", false, "full stack refused: names")
		return true
	}

	rootRaw, rootAbs, rok := staticAssetsRoot(tag, "full stack")
	if !rok {
		return true
	}

	// The folder: named (fenced), the session's, or asked.
	var folderRel string
	switch {
	case q.hasFolder:
		_, rel, refusal := fenceResolve(rootAbs, q.folder)
		if refusal != "" {
			fmt.Printf("%s Not growing that stack: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "full stack refused: folder outside project")
			return true
		}
		folderRel = rel
	case workContextLive(rootAbs) && workCtx.wasmFolder != "":
		folderRel = workCtx.wasmFolder
		if wasmAppFolderOf(rootAbs, folderRel) {
			fmt.Printf("Reading the app as the wasm app in %s — the last wasm app this session touched.\n", folderRel)
		}
	default:
		printClarifyIntro(tag, "Not growing that stack yet: the sentence doesn't name the app's folder — which folder does the app live in?")
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				q2 := q
				q2.folder = text
				q2.hasFolder = true
				return runFullStack(q2.canonical(), ask2)
			},
		})
		recordBrain("composer", false, "full stack question: folder unnamed")
		return true
	}
	if folderRel == "." {
		fmt.Printf("%s Not growing that stack: the stack's form lives in a wasm app, and a wasm app needs a folder of its own — its go.mod makes the folder a module of its own, apart from the project's build, so the app can't be the project root. Name a folder: \"add a full stack app called %s in jim with fields %s\".\n", tag, q.thing, q.fieldsRaw)
		recordBrain("composer", false, "full stack refused: project root")
		return true
	}
	abs := filepath.Join(rootAbs, filepath.FromSlash(folderRel))
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		if err == nil {
			fmt.Printf("%s Not growing that stack: %s isn't a folder — a wasm app belongs to a folder, as its own module.\n", tag, folderRel)
		} else {
			fmt.Printf("%s Not growing that stack: there's no folder %s in this project for the app to live in — say \"create a folder %s\" first, then \"add a wasm app in %s\", then ask again.\n", tag, folderRel, folderRel, folderRel)
		}
		recordBrain("composer", false, "full stack refused: folder missing")
		return true
	}
	if !wasmAppFolderOf(rootAbs, folderRel) {
		fmt.Printf("%s Not growing that stack: there's no wasm app in %s yet — the stack's form and list live in the app, so say \"add a wasm app in %s\" first, then ask again.\n", tag, folderRel, folderRel)
		recordBrain("composer", false, "full stack refused: no app in folder")
		return true
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "full stack") {
		return true
	}

	// The route's facts, the serve sentence's way: a
	// /static/ conflict refuses in the family's words
	// before anything is planned, and the backend's home
	// is the project's server in server/ or its
	// root-level program — neither, and the planner
	// creates the server in server/.
	facts := lastAnalyzeProject.ServeWasmRouteFacts(folderRel)
	if facts.Conflict != "" {
		fmt.Printf("%s Not growing that stack: %s\n", tag, facts.Conflict)
		recordBrain("composer", false, "full stack refused: /static/ taken")
		return true
	}
	if facts.OtherFile != "" {
		fmt.Printf("%s %s also registers /static/ for %s in its own program; this stack grows on the project's server in server/ (or the root program) instead, and that registration is left exactly as it is.\n", tag, facts.OtherFile, folderRel)
	}
	target := facts.ServerFile
	if target == "" {
		target = facts.RootFile
	}

	// The landing: the one spine, with the wasm app's
	// compile as the in-transaction extra check — the
	// root build can't see into the app's own module,
	// and a stack whose form doesn't compile must not
	// land.
	appAbs := filepath.Join(rootAbs, filepath.FromSlash(folderRel))
	check := func() (string, string, bool) {
		out, err, timedOut := execWasmBuild(appAbs)
		what := "the wasm app in " + folderRel + " doesn't compile (" + analyze.WasmBuildCommand + " in " + folderRel + ")"
		if timedOut {
			what = "the wasm app in " + folderRel + " didn't finish compiling in time (" + analyze.WasmBuildCommand + " in " + folderRel + ")"
		}
		return what, out, err == nil
	}
	return executeEditPlanChecked(analyze.EditRequest{
		Kind:            analyze.EditFullStack,
		WasmFolder:      folderRel,
		StackThing:      q.thing,
		StackFields:     q.fields,
		StackServerFile: target,
	}, ask, check)
}

// folderWords names the folder a guidance line should
// show: the sentence's own, or the family's example when
// it named none.
func folderWords(q fullStackAsk) string {
	if q.hasFolder {
		return q.folder
	}
	return "jim"
}
