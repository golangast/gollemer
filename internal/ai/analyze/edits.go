package analyze

// Edits by assembly: the Composer's write side. A change request is
// never free-written — the analyzer locates the exact site, an edit
// is ASSEMBLED by copying a real sibling pattern from the same
// project (a case clause, a field line, a table entry, a whole
// function), and the result is a plan: before/after bytes plus a
// unified diff the chat shows behind a [y/n] gate. After applying,
// the chat builds the project and re-analyzes it; the plan's Verify
// then checks the intended structure really exists, and any failure
// restores the original bytes.
//
// Everything here is planning only — nothing in this file writes to
// disk. Planners return (nil, refusal) with a user-ready sentence
// when the edit can't be located or assembled honestly: ambiguous
// names get their candidates listed, missing sibling patterns are a
// refusal, never an invention.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Edit kinds the chat's sentence parser can request.
const (
	EditAddCase   = "add_case"
	EditAddField  = "add_field"
	EditRegister  = "register"
	EditAddFunc   = "add_func"
	EditRename    = "rename"
	EditAddServer = "add_server"
	EditDelegate  = "delegate"

	// Serve an html file (editservehtml.go): the composite
	// recipe — the page created when missing (never
	// overwritten) plus the wiring that serves it via
	// http.ServeFile, composing with a server the file already
	// runs and never adding a second ListenAndServe.
	EditServeHTML = "serve_html"

	// Wire a handler (editwire.go): register an existing
	// handler function with the project's existing webserver —
	// one HandleFunc line inserted just before the serve call
	// in the function that serves, on the default mux or the
	// file's own *http.ServeMux. Nothing is created and no
	// second server is ever added; every state that cannot be
	// wired honestly (no handler, no server, a taken path, a
	// custom server) is a refusal with the facts.
	EditWireHandler = "wire_handler"

	// Webpage (editwebpage.go): one plain webpage from a named
	// template, written into one .html file — created when
	// missing, filled when it exists but is empty, never
	// overwriting a page that already has content. No Go file
	// is touched: the page is the whole edit.
	EditWebpage = "webpage"

	// Serve static files (editservestatic.go): one file-server
	// registration on the project's existing webserver — route
	// /static/ answered by http.FileServer over one project
	// folder, inserted before the serve statement under the
	// wiring machinery's rules (conflicts read project-wide,
	// custom servers refused with the line handed over).
	EditServeStatic = "serve_static"

	// Serve a wasm app (editservewasm.go): the missing-server
	// half of the chat's "serve the wasm app in <folder>"
	// composition. When the project has NO webserver at all,
	// this plan creates the house server in server/main.go —
	// the WebServerSource component — already carrying the
	// /static/ file-server registration over the app's folder
	// (req.WasmFolder), so one gated plan leaves a program
	// that serves the app. When a server exists the chat
	// routes the ask to EditServeStatic instead and this
	// planner refuses; the running is the run claim's, never
	// the planner's.
	EditServeWasmApp = "serve_wasm_app"

	// Component (editcomponent.go): one Material component
	// from the registry in editmaterial.go, inserted just
	// before an existing .html page's </body>, with the
	// Material styles it needs carried into the page exactly
	// once. In a templated page's body file (a fragment with
	// no </body> of its own) the snippet lands above the
	// body's footer call instead, its styles joining the
	// shared header. No Go file is touched.
	EditComponent = "component"

	// Templated page scaffold (edittemplatepage.go): a page
	// split across three files in the named folder's
	// templates folder — a shared header and footer plus
	// this page's body — joined at serve time by
	// html/template, with the handler that renders it in a
	// new Go file in the folder's package. No route is
	// wired.
	EditTemplatedPage = "templated_page"

	// Scaffold section (edittemplatepage.go): a counted
	// columns-and-rows section inserted into a templated
	// page's BODY file, just before its footer call, with
	// the section's styles joining the shared header's
	// style block exactly once. The header and footer files
	// are otherwise untouched.
	EditScaffoldSection = "scaffold_section"

	// Wasm app (editwasm.go): a browser app in one folder —
	// Go compiled to WebAssembly. The plan creates the app's
	// own module (a nested go.mod pinning Go 1.27, so the
	// host project keeps its own Go version and the app's
	// files stay out of the host build), its main.go (the
	// page built through syscall/js, the garbage collector's
	// GOGC knob set low in code), a small simd vector unit,
	// the host page that loads the built .wasm, and the
	// wasm_exec.js copied from this machine's Go 1.27
	// install when one is found. Components asked into the
	// app ride the component kind with WasmFolder set: they
	// land as Go DOM builders, not HTML (editwasm.go's
	// planComponentWasm).
	EditWasmApp = "wasm_app"

	// Wasm event (editwasmevent.go): one event on a component
	// already mounted in a wasm app — a source component, its
	// event (a click, or a change), and what happens (another
	// component shown, hidden, or toggled, or a button's or
	// card's words set). The plan writes the handler into the
	// app's events.go as a js.FuncOf callback, registers it
	// after the mounts in main.go, stamps the id attributes
	// the handler looks its components up by, and keeps every
	// callback in the file's package-level list so the
	// garbage collector can never take one. The request's
	// component references are as typed (a registry kind plus
	// an optional "called" name); the planner resolves them
	// against the components the app actually has mounted.
	EditWasmEvent = "wasm_event"

	// Page family (editpage.go): a named page kind as a matched
	// page+handler pair — an .html page created with starter
	// content (never overwritten), the handler that serves it
	// (for form kinds, the concept book's form-handler piece
	// adapted so its GET branch serves the page), and the route
	// registration when a composable server exists in the
	// handler's package.
	EditPage = "page"

	// Cross-file use (edituses.go): a NEW function in a named
	// file whose body is a first use of a function or type that
	// already exists elsewhere in the project — the call (or
	// the value) assembled from the referenced symbol's real
	// signature/definition, starting values derived honestly,
	// returns handled in the body. The delegate's passthrough
	// wrapper is the neighboring shape, not this one.
	EditUse = "use"

	// Concept-book edits (editconcept.go): the entry the chat's
	// composer matched from the book, assembled by kind.
	EditConcept = "concept"

	// Make-and-call (editmakecall.go): ONE ask, two files — a
	// NEW plain function declared in one file (the concept
	// book's function row, planned by the concept planner) and
	// a call to it written into a second file in the same plan:
	// a new caller function assembled from the planned
	// declaration's own signature (the uses machinery, reading
	// the declaration this plan writes rather than one already
	// on disk), or the bare call inserted into an existing host
	// function. One gate, one transaction, whole-project build.
	EditMakeCall = "make_call"

	// Model-proposed declarations (editproposed.go): a fully
	// rendered declaration the chat's speccopy proposer produced
	// from the user's sentence, appended at file scope.
	EditProposedDecl = "proposed_decl"

	// Taught declarations (edittaught.go): a declaration the
	// user taught the chat from an example in their own project
	// (the teach seat) — the example's source with its name
	// slots already renamed by the chat layer to the new name,
	// appended at file scope or created as a new file, the way
	// a concept-book declaration lands. The template is the
	// user's own data, carried verbatim.
	EditTaughtDecl = "taught_decl"

	// Instantiation (editinstantiate.go): a statement making a
	// value of an existing project type, inserted at the end of
	// an existing project function.
	EditInstantiate = "instantiate"

	// Signature changes (editsig.go): the target function is
	// FuncName, resolved like add_case's.
	EditRenameParam   = "rename_param"
	EditReorderParams = "reorder_params"
	EditAddParam      = "add_param"

	// Goroutine result (editgoroutine.go): give an existing
	// anonymous goroutine — a go func() { ... }() written out
	// in a function — a way to hand a value back: a channel
	// declared before the go statement, a send at the end of
	// the goroutine's body, and the receive right after it.
	// A goroutine cannot return a value in Go; the channel
	// is the language's own answer, assembled here.
	EditGoroutineResult = "goroutine_result"
)

// EditRequest is one parsed change request.
type EditRequest struct {
	Kind string

	// add_case: the function holding the switch, the new case
	// expression as typed, and optionally the switch's subject to
	// pick among several switches in one function.
	FuncName string
	CaseExpr string
	SwitchOn string

	// add_field.
	StructName string
	FieldName  string
	FieldType  string
	AfterField string // optional anchor; "" = after the last field
	FieldTag   string // optional struct tag content (json:"name"); "" = none

	// register: a package-level table (map or basic-literal slice)
	// and the new entry's key.
	TableName string
	Key       string

	// add_func: copy SiblingName, call it NewName.
	SiblingName string
	NewName     string

	// rename: the symbol as typed (bare, Recv.Name, or pkg.Name),
	// the new name, an optional kind word ("function", "type",
	// "variable", "constant", "field", "method"), and an optional
	// package/file scope used to disambiguate ("in shop").
	RenameFrom  string
	RenameTo    string
	RenameKind  string
	RenameScope string

	// add_server: the existing .go file (project-relative slash
	// path) the webserver component goes into. The chat layer
	// resolves folder targets to a file before planning.
	File string

	// serve_html: the page to serve (project-relative slash
	// path, .html), resolved by the chat layer into the Go
	// file's folder (File) or below it. The planner derives
	// the ServeFile argument from the two paths.
	ServeHTML string

	// webpage: the page file (File, shared with add_server —
	// project-relative slash path, .html) and the template
	// ("basic" | "responsive" | "assets" | "two-columns" |
	// "three-columns" | "grid" | "rows" | "material"), both
	// resolved by the chat layer.
	WebpageTemplate string

	// webpage, counted grid: the grid template with exact
	// counts ("add template 2 columns 3 rows to …"). Both
	// zero = the plain flowing grid. WebpageGridRows may be
	// zero with WebpageGridCols set (the sentence named
	// columns only) — the planner defaults the rows to 2,
	// the same default the chat layer states.
	WebpageGridCols int
	WebpageGridRows int

	// component: the page file (File, as above) and the
	// component's registry id ("button", "card",
	// "text-field", …; editmaterial.go), resolved by the
	// chat layer.
	Component string

	// component in wasm mode: the "called <name>" name the
	// sentence gave the component ("add a button called
	// save to jim"), "" when it gave none. A named component
	// is a distinct instance of its kind — its own builder,
	// its name as its element id — where an unnamed one is
	// the kind's one unnamed instance, with a positional id
	// ("button-1"). Only the wasm mode reads this; the page
	// modes never see it (their grammar takes no name).
	ComponentName string

	// wasm_app, and component in wasm mode: the wasm app's
	// folder (project-relative slash path), resolved by the
	// chat layer. For the component kind, a non-empty
	// WasmFolder routes the ask to the app in that folder
	// (File stays empty — the app's files are derived from
	// the folder); empty means the page modes above. The
	// serve_wasm_app kind reads it as the app folder the new
	// server's /static/ registration serves.
	WasmFolder string

	// wasm_event: the event's source component and the
	// action's target component, each as typed — a registry
	// kind (EventSourceKind / EventTargetKind; "" when the
	// sentence named the component by name alone) and the
	// "called" name (EventSourceName / EventTargetName; ""
	// when the sentence gave none). EventName is the event
	// ("click" | "change"), EventAction what happens
	// ("show" | "hide" | "toggle" | "setwords"), and
	// EventWords the setwords text, as typed ("" for the
	// other actions). The planner resolves both references
	// against the app's mounted components.
	EventSourceKind string
	EventSourceName string
	EventName       string
	EventAction     string
	EventTargetKind string
	EventTargetName string
	EventWords      string

	// templated_page: the project folder the scaffold
	// belongs to (SiteFolder, project-relative slash path;
	// "." = the project root) and the page's name as typed
	// (SiteName — "home" makes templates/home.html and the
	// HomePage handler), both resolved by the chat layer.
	SiteFolder string
	SiteName   string

	// scaffold_section: the body file the section lands in
	// (SiteBody, project-relative — the chat layer resolves
	// "the body" to the folder's one scaffold body, asking
	// when there are several), and the section's counts
	// (SiteCols, SiteRows, each 1–6).
	SiteBody string
	SiteCols int
	SiteRows int

	// serve_static: the folder whose files the server serves
	// at /static/ (project-relative slash path; "." = the
	// project root), resolved by the chat layer.
	StaticFolder string

	// wire_handler: the handler's name as typed ("" = the
	// sentence named none — the planner resolves the project's
	// one unregistered handler, or refuses naming the
	// candidates), and the route as typed ("" = the default:
	// "/" + the handler's lowercased name). The planner finds
	// the serving file itself, so File stays empty.
	WireName  string
	WireRoute string

	// page: the page kind ("form"|"home"|"about"|"contact"|
	// "list"|"details"), the page file (PageHTML,
	// project-relative slash path, .html), the handler's name
	// (PageName, already defaulted by the chat layer), and the
	// route (PageRoute, already derived by the chat layer).
	// File (shared, above) is the handler's host .go file
	// (project-relative; may not exist yet — the plan creates
	// it). Concept (shared, below) carries the book's
	// form-handler entry for form/contact kinds, nil otherwise.
	PageKind  string
	PageHTML  string
	PageName  string
	PageRoute string

	// delegate: the existing .go file (project-relative slash
	// path) the wrapper is added to, the source function as typed
	// (bare name or pkg.Name), and the wrapper's name ("" = the
	// source's name when the target package doesn't declare it).
	DelegateFile string
	SourceName   string
	WrapperName  string

	// use: the existing .go file (project-relative slash path)
	// the new function is added to, the referenced symbol as
	// typed (bare name or pkg.Name — a function or a type), the
	// file the sentence says the symbol lives in ("" = not
	// stated; used to disambiguate, never to invent), the new
	// function's name ("" = derived: "Use" + the symbol's
	// name), and whether the sentence's "type" designator
	// forces the symbol to resolve as a type.
	UseFile     string
	UseSymbol   string
	UseFromFile string
	UseName     string
	UseType     bool

	// make_call: the callee's name (its declaration is the
	// concept request riding in Concept/ConceptName/File —
	// File is the callee's file, resolved by the chat layer
	// under the declaration rules), the caller's file
	// (MakeCallTarget, an existing .go file resolved by the
	// chat layer), the new caller function's name
	// (MakeCallCaller, "" = no new function), and the existing
	// host function the bare call lands in (MakeCallHost,
	// "" = the target package's main, found by the planner).
	MakeCallTarget string
	MakeCallCaller string
	MakeCallHost   string

	// Signature changes (the target is FuncName):
	//
	// rename_param: SigOld is the parameter's current name,
	// SigNew its new name.
	//
	// reorder_params: SigOld is parameter A; SigMove is "" for
	// a swap of A and SigNew, or "first"/"last" for moving A, or
	// "before"/"after" with SigNew as the anchor parameter.
	//
	// add_param: SigType is the new parameter's type as typed,
	// SigName its name ("" = the context form, whose type is
	// context.Context). SigUsing is the call-site argument,
	// verbatim, for every site; "" with the context type means
	// derive each site's own in-scope context. SigNoUsing marks
	// a typed form typed without its "using <expression>"
	// clause — a guidance refusal, not an edit.
	SigOld     string
	SigNew     string
	SigMove    string
	SigName    string
	SigType    string
	SigUsing   string
	SigNoUsing bool

	// concept: the book entry to assemble (editconcept.go), the
	// target file (File, shared with add_server), and the slots
	// the sentence bound:
	//
	// ConceptName is the declaration's {name} (required for
	// declarations); ConceptText is the statement's {text}
	// saying slot ("" = the default, e.g. fmt.Println(""));
	// ConceptFunc is the statement's target function ("" =
	// main); ConceptLike is the type whose real method set
	// fills a declaration's {methods} slot.
	Concept     *ConceptEntry
	ConceptName string
	ConceptText string
	ConceptFunc string
	ConceptLike string
	// ConceptSibling is the statement's unique declaration
	// sibling under the book's naming rule (a statement name
	// plus " function" carried by exactly one declaration
	// row), supplied by the chat layer — which owns the
	// loaded book — so a missing-host refusal can spell the
	// sibling's sentence for a target file that declares no
	// functions. Nil when the statement has no sibling.
	ConceptSibling *ConceptEntry
	// ConceptFolder is the folder-default trigger: the book
	// row's folder field, carried when the sentence named NO
	// file or folder for a declaration ask. The planner then
	// creates <folder>/<name>.go as the folder's own package
	// instead of editing req.File (which stays empty). A
	// request that names a file never sets it — the named
	// target always wins.
	ConceptFolder string

	// Line placement (editconcept.go): the sentence's line
	// phrase — "on line 12", "before line 8" — carried by the
	// planners that insert code: concept statements (inside
	// the named function's body), concept declarations
	// (between the file's top-level declarations), and
	// instantiation (the concept statement's sibling). The
	// number counts the target file's lines as the user sees
	// them BEFORE the edit, 1-based. PlaceLine == 0 means the
	// sentence carried no line phrase and placement is the
	// family's default (end of body / end of file). PlaceBefore
	// is "before line N"; false is the on/at/after reading.
	PlaceLine   int
	PlaceBefore bool

	// instantiate: the type as typed (bare "Jim" or qualified
	// "store.Jim"), the function as typed that should hold the
	// new statement, and the variable's name from the
	// sentence's named|called slot ("" = derived from the type
	// name). The planner resolves both names against the
	// project index — resolution is the analyzer's job.
	InstType string
	InstFunc string
	InstVar  string

	// proposed_decl: the declaration already rendered to Go
	// text by the chat layer (the deterministic decl renderer
	// over the model's parsed, case-restored spec), the name
	// it declares, its kind ("struct", "interface", "func"),
	// the receiver's base type for a method ("" for types and
	// plain functions), and the import paths its type
	// expressions reference, space-joined (a path never
	// contains a space; the planner adds the missing ones the
	// same way the concept planner does).
	ProposedText     string
	ProposedName     string
	ProposedKind     string
	ProposedReceiver string
	ProposedImports  string

	// goroutine_result: the result type the sentence named
	// (GoroutineType — one of the closed set in
	// editgoroutine.go: string, int, bool, float64) and,
	// when the sentence pinned one of several goroutines,
	// its 1-based position among the go statements in the
	// resolved scope — the named function (FuncName, shared
	// above), or the whole file (File, shared above) when
	// no function was named. GoroutineOrdinal == 0 means
	// the sentence named no position.
	GoroutineType    string
	GoroutineOrdinal int

	// taught_decl: the taught template's source, already
	// renamed by the chat layer to TaughtName (the new
	// declaration's name); TaughtDeclKind is "function" or
	// "type" (a taught declaration is never a method);
	// TaughtKindPhrase is the user's kind phrase ("sidebar
	// page"), used in the plan's words; TaughtImports is the
	// space-joined full import specs the example declaration
	// leans on (full specs, `db "database/sql"` form allowed);
	// TaughtNote is the plan's provenance line, rendered by
	// the chat layer from the store ("From your example Jake
	// in jim/main.go, taught Nov 3, 2026.").
	TaughtText       string
	TaughtName       string
	TaughtDeclKind   string
	TaughtKindPhrase string
	TaughtImports    string
	TaughtNote       string
}

// EditFile is one file's before/after pair inside a plan.
type EditFile struct {
	Rel    string // relative to the project root
	Before []byte
	After  []byte
	Diff   string // unified diff of Before -> After
}

// EditPlan is a fully assembled edit, not yet applied.
type EditPlan struct {
	Summary    string // plain words: what changes, and where
	VerifyDesc string // plain words: what must be true once applied
	Site       string // LinkPath(file):line of the main change, in the new text
	Files      []EditFile

	// Notes are extra plan lines shown before the gate: the
	// delegate's companions to its provenance line, and the
	// scaffold notes (present or honestly omitted) of every
	// shape that carries one.
	Notes []string

	// Delegate-only extras (every other planner leaves them
	// zero): the provenance line naming the source fact the
	// plan was built from, and the ONE alternate file set the
	// bounded retry may use when the primary form fails with an
	// import/qualification error, with its plain-words
	// description.
	Provenance string
	AltFiles   []EditFile
	AltDesc    string

	verify func(after *Project) bool
}

// Diff renders every file's unified diff, in plan order.
func (pl *EditPlan) Diff() string {
	var parts []string
	for _, f := range pl.Files {
		if f.Diff != "" {
			parts = append(parts, f.Diff)
		}
	}
	return strings.Join(parts, "\n")
}

// Verify reports whether the re-analyzed project shows the intended
// structure. Callers pass the project analyzed AFTER applying.
func (pl *EditPlan) Verify(after *Project) bool {
	if pl == nil || pl.verify == nil || after == nil {
		return false
	}
	return pl.verify(after)
}

// PlanEdit dispatches one request to its planner. A nil plan with a
// non-empty refusal means "not doing that, and here is why, in words
// the user can act on".
func (p *Project) PlanEdit(req EditRequest) (*EditPlan, string) {
	switch req.Kind {
	case EditAddCase:
		return p.planAddCase(req)
	case EditAddField:
		return p.planAddField(req)
	case EditRegister:
		return p.planRegister(req)
	case EditAddFunc:
		return p.planCloneFunc(req)
	case EditRename:
		return p.planRename(req)
	case EditAddServer:
		return p.planAddServer(req)
	case EditServeHTML:
		return p.planServeHTML(req)
	case EditWireHandler:
		return p.planWireHandler(req)
	case EditWebpage:
		return p.planWebpage(req)
	case EditServeStatic:
		return p.planServeStatic(req)
	case EditServeWasmApp:
		return p.planServeWasmApp(req)
	case EditComponent:
		return p.planComponent(req)
	case EditTemplatedPage:
		return p.planTemplatedPage(req)
	case EditScaffoldSection:
		return p.planScaffoldSection(req)
	case EditWasmApp:
		return p.planWasmApp(req)
	case EditWasmEvent:
		return p.planWasmEvent(req)
	case EditPage:
		return p.planPage(req)
	case EditDelegate:
		return p.planDelegate(req)
	case EditUse:
		return p.planUse(req)
	case EditMakeCall:
		return p.planMakeCall(req)
	case EditConcept:
		return p.planConcept(req)
	case EditProposedDecl:
		return p.planProposedDecl(req)
	case EditTaughtDecl:
		return p.planTaughtDecl(req)
	case EditInstantiate:
		return p.planInstantiate(req)
	case EditRenameParam, EditReorderParams, EditAddParam:
		return p.planSignature(req)
	case EditGoroutineResult:
		return p.planGoroutineResult(req)
	}
	return nil, "I don't know that kind of edit yet."
}

// --- shared planning machinery -------------------------------------------

// textEdit is one splice into a file's bytes.
type textEdit struct {
	start, end int
	text       string
	// isImport marks a pure insert produced by the import-edit
	// helpers (insertImportSpecs, insertImportEdits). When
	// several inserts share one offset, applyTextEdits composes
	// the marked ones first: an import block stranded after a
	// declaration is not a style issue, it is unparseable Go,
	// so the order is a property of the edit kind — never of
	// the order a planner happened to list its edits in.
	isImport bool
}

// applyTextEdits splices edits (offsets into src) into a new byte
// slice. Overlapping edits are a planner bug and drop the later
// one. Inserts that share one offset — a package-clause-only
// file anchors both its new import block and an appended
// declaration at EOF — are composed into a single splice at
// that offset first (see composeSameOffsetInserts), so no
// planner has to merge them by hand to keep its imports ahead
// of its declarations.
func applyTextEdits(src []byte, edits []textEdit) []byte {
	sort.SliceStable(edits, func(i, j int) bool { return edits[i].start > edits[j].start })
	edits = composeSameOffsetInserts(edits)
	out := append([]byte{}, src...)
	lastStart := len(src) + 1
	for _, e := range edits {
		if e.end > lastStart || e.start < 0 || e.end > len(src) || e.start > e.end {
			continue
		}
		next := append([]byte{}, out[:e.start]...)
		next = append(next, e.text...)
		next = append(next, out[e.end:]...)
		out = next
		lastStart = e.start
	}
	return out
}

// composeSameOffsetInserts merges each run of consecutive pure
// inserts (start == end) that share one offset into a single
// edit at that offset: import edits first, the rest in the
// order they were passed. The input is sorted by start
// descending with a stable sort, so a run is consecutive and
// still carries its passed order. Span replacements are left
// alone — two of those at one offset overlap, which the splice
// loop already treats as a planner bug.
func composeSameOffsetInserts(edits []textEdit) []textEdit {
	out := make([]textEdit, 0, len(edits))
	for i := 0; i < len(edits); {
		e := edits[i]
		if e.start != e.end {
			out = append(out, e)
			i++
			continue
		}
		var imports, rest strings.Builder
		j := i
		for j < len(edits) && edits[j].start == e.start && edits[j].end == e.end {
			if edits[j].isImport {
				imports.WriteString(edits[j].text)
			} else {
				rest.WriteString(edits[j].text)
			}
			j++
		}
		if j == i+1 {
			out = append(out, e)
		} else {
			out = append(out, textEdit{start: e.start, end: e.end, text: imports.String() + rest.String()})
		}
		i = j
	}
	return out
}

// finishFile turns edited bytes for one file into an EditFile: when
// the original was gofmt-clean, the result is gofmt'd too, so an
// assembled edit never fights the project's formatting; when it
// wasn't clean, the sibling-shaped text goes in as built.
func finishFile(rel string, before, after []byte) EditFile {
	if clean, err := format.Source(before); err == nil && bytes.Equal(clean, before) {
		if f, err := format.Source(after); err == nil {
			after = f
		}
	}
	return EditFile{
		Rel:    rel,
		Before: before,
		After:  after,
		Diff:   unifiedDiff(rel, before, after),
	}
}

// fileBlocked reports whether a file is off-limits for edits:
// generated code or vendored code.
func (p *Project) fileBlocked(rel string) (string, bool) {
	if rel == "vendor" || strings.HasPrefix(rel, "vendor/") || strings.Contains(rel, "/vendor/") {
		return "it lives under vendor/, which is other people's code", true
	}
	if p.genFiles[rel] {
		return "it is a generated file (marked DO NOT EDIT)", true
	}
	return "", false
}

// fileBytesBlocked checks raw bytes for the generated marker — the
// belt-and-braces version for files the index may not have walked.
func fileBytesBlocked(src []byte) bool {
	for _, line := range strings.Split(string(src), "\n") {
		if generatedMarker.MatchString(strings.TrimSpace(line)) {
			return true
		}
		if len(line) > 0 && line[0] != '/' && !strings.HasPrefix(line, "*") {
			// The marker lives in a leading comment block; stop at code.
			if strings.HasPrefix(strings.TrimSpace(line), "package ") {
				return false
			}
		}
	}
	return false
}

// parseGoSrc parses one in-memory Go source file.
func parseGoSrc(name string, src []byte) (*token.FileSet, *ast.File, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, name, src, parser.ParseComments)
	if err != nil {
		return nil, nil, err
	}
	return fset, f, nil
}

// readProjectFile reads one project file and parses it.
func (p *Project) readProjectFile(rel string) ([]byte, *token.FileSet, *ast.File, string) {
	src, err := readFileBytes(filepath.Join(p.Root, rel))
	if err != nil {
		return nil, nil, nil, fmt.Sprintf("I couldn't read %s: %v", p.LinkPath(rel), err)
	}
	fset, f, err := parseGoSrc(rel, src)
	if err != nil {
		return nil, nil, nil, fmt.Sprintf("I couldn't parse %s: %v", p.LinkPath(rel), err)
	}
	return src, fset, f, ""
}

func readFileBytes(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func offOf(fset *token.FileSet, pos token.Pos) int { return fset.Position(pos).Offset }
func lineOfPos(fset *token.FileSet, pos token.Pos) int {
	return fset.Position(pos).Line
}

// lineStartOffset returns the byte offset where the given 1-based
// line begins in src.
func lineStartOffset(src []byte, line int) int {
	off := 0
	for l := 1; l < line; l++ {
		i := bytes.IndexByte(src[off:], '\n')
		if i < 0 {
			return len(src)
		}
		off += i + 1
	}
	return off
}

// lineIndent returns the leading whitespace of a 1-based line.
func lineIndent(src []byte, line int) string {
	start := lineStartOffset(src, line)
	end := start
	for end < len(src) && (src[end] == ' ' || src[end] == '\t') {
		end++
	}
	return string(src[start:end])
}

// findProjectFuncs returns every function matching a user-typed
// reference (bare name, Recv.Name, or pkg.Name, case-insensitive),
// test functions excluded — the same forgiveness as the chat's
// findFuncByQuery, but collecting all matches so ambiguity can be
// reported instead of silently picking one.
func (p *Project) findProjectFuncs(q string) []*Func {
	q = strings.TrimSpace(q)
	if q == "" {
		return nil
	}
	lastSeg := q
	if i := strings.LastIndex(q, "."); i >= 0 {
		lastSeg = q[i+1:]
	}
	var exact, bare []*Func
	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			recvName := ""
			if fn.Receiver != "" {
				recvName = strings.Trim(fn.Receiver, "*") + "." + fn.Name
			}
			switch {
			case recvName != "" && strings.EqualFold(q, recvName),
				fn.Pkg != "" && strings.EqualFold(q, fn.Pkg+"."+fn.Name),
				!strings.Contains(q, ".") && strings.EqualFold(q, fn.Name):
				exact = append(exact, fn)
			case strings.EqualFold(lastSeg, fn.Name) && strings.Contains(q, "."):
				bare = append(bare, fn)
			}
		}
	}
	if len(exact) > 0 {
		return exact
	}
	return bare
}

// funcCandidates renders an ambiguity refusal for function matches.
func (p *Project) funcCandidates(verb, q string, fns []*Func) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%q could mean more than one function here:\n", q)
	for _, fn := range fns {
		fmt.Fprintf(&b, "  - %s (%s:%d)\n", fn.Display(), p.LinkPath(fn.File), fn.Line)
	}
	fmt.Fprintf(&b, "Say which one — %s — and I'll do it.", verb)
	return b.String()
}

// findStructType resolves a struct by name (or pkg.Name), collecting
// all matches so ambiguity can be reported.
func (p *Project) findStructTypes(q string) []*Type {
	q = strings.TrimSpace(q)
	var out []*Type
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind != "struct" {
				continue
			}
			if strings.EqualFold(q, t.Name) || strings.EqualFold(q, t.Pkg+"."+t.Name) {
				out = append(out, t)
			}
		}
	}
	return out
}

// stmtText renders any node with the file set, one line.
func stmtText(fset *token.FileSet, n ast.Node) string {
	var buf bytes.Buffer
	if err := format.Node(&buf, fset, n); err != nil {
		return ""
	}
	return strings.Join(strings.Fields(buf.String()), " ")
}

// --- add a case to a switch ------------------------------------------------

type switchSite struct {
	node    ast.Node // *ast.SwitchStmt or *ast.TypeSwitchStmt
	body    *ast.BlockStmt
	subject string // rendered subject, for messages and matching
	line    int
	cases   []*ast.CaseClause
	def     *ast.CaseClause // default clause, if any
}

func collectSwitches(fset *token.FileSet, fn *ast.FuncDecl) []*switchSite {
	var out []*switchSite
	ast.Inspect(fn.Body, func(n ast.Node) bool {
		var site *switchSite
		switch s := n.(type) {
		case *ast.SwitchStmt:
			site = &switchSite{node: s, body: s.Body, line: lineOfPos(fset, s.Pos())}
			switch {
			case s.Tag != nil:
				site.subject = exprText(fset, s.Tag)
			case s.Init != nil:
				site.subject = stmtText(fset, s.Init)
			default:
				site.subject = "the conditions"
			}
		case *ast.TypeSwitchStmt:
			site = &switchSite{node: s, body: s.Body, line: lineOfPos(fset, s.Pos())}
			site.subject = stmtText(fset, s.Assign)
			if site.subject == "" {
				site.subject = stmtText(fset, s.Init)
			}
		}
		if site != nil {
			for _, stmt := range site.body.List {
				if cc, ok := stmt.(*ast.CaseClause); ok {
					site.cases = append(site.cases, cc)
					if cc.List == nil {
						site.def = cc
					}
				}
			}
			out = append(out, site)
		}
		return true // keep walking: a switch nested in a case is its own site
	})
	return out
}

func (p *Project) planAddCase(req EditRequest) (*EditPlan, string) {
	matches := p.findProjectFuncs(req.FuncName)
	if len(matches) == 0 {
		return nil, fmt.Sprintf("I couldn't find a function named %q in this project, so there's no switch to add a case to.", req.FuncName)
	}
	if len(matches) > 1 {
		return nil, p.funcCandidates("name the receiver or package", req.FuncName, matches)
	}
	fn := matches[0]
	if why, blocked := p.fileBlocked(fn.File); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(fn.File), why)
	}
	src, fset, fileAST, refusal := p.readProjectFile(fn.File)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(fn.File))
	}
	var decl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && lineOfPos(fset, fd.Pos()) == fn.Line {
			decl = fd
			break
		}
	}
	if decl == nil || decl.Body == nil {
		return nil, fmt.Sprintf("I found %s but couldn't re-read its body in %s.", fn.Display(), p.LinkPath(fn.File))
	}
	switches := collectSwitches(fset, decl)
	if len(switches) == 0 {
		return nil, fmt.Sprintf("%s has no switch statement — there's nothing to add a case to. (I only add cases to switches that already exist.)", fn.Display())
	}
	var site *switchSite
	if req.SwitchOn != "" {
		for _, s := range switches {
			if strings.Contains(strings.ToLower(s.subject), strings.ToLower(req.SwitchOn)) {
				if site != nil {
					return nil, fmt.Sprintf("More than one switch in %s matches %q — say a longer piece of the thing it switches on.", fn.Display(), req.SwitchOn)
				}
				site = s
			}
		}
		if site == nil {
			var b strings.Builder
			fmt.Fprintf(&b, "No switch in %s switches on %q. The switches there are:\n", fn.Display(), req.SwitchOn)
			for _, s := range switches {
				fmt.Fprintf(&b, "  - the switch on %s (%s:%d)\n", s.subject, p.LinkPath(fn.File), s.line)
			}
			return nil, b.String()
		}
	} else if len(switches) == 1 {
		site = switches[0]
	} else {
		var b strings.Builder
		fmt.Fprintf(&b, "%s has %d switches — say which one, like \"add a case to the switch on %s in %s\":\n", fn.Display(), len(switches), switches[0].subject, fn.Name)
		for _, s := range switches {
			fmt.Fprintf(&b, "  - the switch on %s (%s:%d)\n", s.subject, p.LinkPath(fn.File), s.line)
		}
		return nil, b.String()
	}

	// The sibling: the last non-default case. No sibling, no assembly.
	var tmpl *ast.CaseClause
	for i := len(site.cases) - 1; i >= 0; i-- {
		if site.cases[i] != site.def {
			tmpl = site.cases[i]
			break
		}
	}
	if tmpl == nil {
		return nil, fmt.Sprintf("The switch on %s in %s has no case yet to copy — assembly needs a sibling pattern, and I won't invent one.", site.subject, fn.Display())
	}
	tmplExprs := make([]string, 0, len(tmpl.List))
	for _, e := range tmpl.List {
		tmplExprs = append(tmplExprs, exprText(fset, e))
	}

	// Shape the new case expression like the sibling's.
	newExpr, why := shapeCaseExpr(fset, tmpl.List[0], req.CaseExpr, site)
	if why != "" {
		return nil, why
	}
	for _, c := range site.cases {
		for _, e := range c.List {
			if exprText(fset, e) == newExpr {
				return nil, fmt.Sprintf("The switch on %s in %s already has a case for %s — nothing to add.", site.subject, fn.Display(), newExpr)
			}
		}
	}

	// Clone the template clause's lines: rebuild the header with the
	// new expression, keep the body, substituting the sibling's case
	// value where the body mentions it.
	srcLines := strings.Split(string(src), "\n")
	startLine := lineOfPos(fset, tmpl.Pos())
	endLine := lineOfPos(fset, tmpl.End()-1)
	if startLine < 1 || endLine > len(srcLines) || endLine < startLine {
		return nil, "I couldn't lift the sibling case out of the file cleanly, so I won't guess at it."
	}
	colonOff := offOf(fset, tmpl.Colon)
	headerLine := srcLines[startLine-1]
	colonInLine := colonOff - lineStartOffset(src, startLine)
	if colonInLine < 0 || colonInLine >= len(headerLine) {
		return nil, "That case's label spans more than one line — too unusual a shape for me to copy safely."
	}
	indent := lineIndent(src, startLine)
	restOfHeader := headerLine[colonInLine+1:]
	var newLines []string
	newLines = append(newLines, indent+"case "+newExpr+":"+substituteCaseRefs(restOfHeader, tmplExprs, newExpr))
	for _, ln := range srcLines[startLine:endLine] {
		newLines = append(newLines, substituteCaseRefs(ln, tmplExprs, newExpr))
	}

	// Insert before the default clause, else before the switch's
	// closing brace.
	insertLine := lineOfPos(fset, site.body.Rbrace)
	if site.def != nil {
		insertLine = lineOfPos(fset, site.def.Pos())
	}
	edits := []textEdit{{
		start: lineStartOffset(src, insertLine),
		end:   lineStartOffset(src, insertLine),
		text:  strings.Join(newLines, "\n") + "\n",
	}}

	// Imports the new expression needs, copied from a sibling file.
	impEdits, why := p.planImportEdits(fn, fileAST, fset, src, newExpr)
	if why != "" {
		return nil, why
	}
	edits = append(edits, impEdits...)

	after := applyTextEdits(src, edits)
	ef := finishFile(fn.File, src, after)

	// The new clause's line in the finished text: count lines up to
	// the inserted block within the final bytes.
	siteLine := insertedLine(ef.After, newLines[0])

	summary := fmt.Sprintf("Add a case for %s to the switch on %s in %s (%s), copied from the case for %s.",
		newExpr, site.subject, fn.Display(), p.LinkPath(fn.File), strings.Join(tmplExprs, ", "))
	subject, fname, expr := site.subject, fn.Name, newExpr
	rel := fn.File
	plan := &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the switch on %s in %s now has a case for %s", subject, fname, expr),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), siteLine),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			return verifyCaseExists(after.Root, rel, fname, expr)
		},
	}
	// The case edit's scaffold: aimed at the function holding
	// the switch, as it stands with the new case in place. The
	// cases exercise its inputs generally — no generated case
	// claims to pin the new branch's behavior; that case is
	// the reader's to add in the TODOs, and the header and
	// note both say so.
	display := scaffoldDisplay(fn)
	p.attachScaffold(plan, scaffoldTarget{
		fn: fn, decl: decl, fset: fset, fileAST: fileAST,
		name: fn.Name, display: display,
		entry: entryReason(fn.Name),
	}, scaffoldPresentation{
		display:    display,
		verifyNoun: "this edit",
		intro: "%s is a test scaffold assembled by gollemer for " + display + ", written\n" +
			"alongside the new " + newExpr + " case this edit adds to its switch. The\n" +
			"cases below exercise " + display + "'s inputs generally — add a case for\n" +
			"the new branch in the TODOs.",
		accompany: "It scaffolds " + display + " as it stands with the new case in place — " +
			"the cases exercise its inputs generally, and a case for the new " + newExpr + " branch is yours to add in the TODOs.",
	})
	return plan, ""
}

// shapeCaseExpr makes the user's case expression match the sibling
// case's kind: string switches take quoted words, constant switches
// take identifiers, and so on.
func shapeCaseExpr(fset *token.FileSet, tmpl ast.Expr, raw string, site *switchSite) (string, string) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", "Add a case for what? Name the value — like \"add a \\\"backoff\\\" case to Parse\"."
	}
	userExpr, err := parser.ParseExpr(raw)
	if err != nil {
		// A bare word for a string switch is the common case.
		if lit, ok := tmpl.(*ast.BasicLit); ok && lit.Kind == token.STRING {
			return strconv.Quote(raw), ""
		}
		return "", fmt.Sprintf("I couldn't read %q as a Go value for a case label.", raw)
	}
	_, isTypeSwitch := site.node.(*ast.TypeSwitchStmt)
	if isTypeSwitch {
		if _, ok := userExpr.(*ast.BasicLit); ok {
			return "", "That switch matches on types, so the case has to be a type name, not a value."
		}
		return raw, ""
	}
	switch t := tmpl.(type) {
	case *ast.BasicLit:
		switch t.Kind {
		case token.STRING:
			switch u := userExpr.(type) {
			case *ast.BasicLit:
				if u.Kind == token.STRING {
					return raw, ""
				}
			case *ast.Ident:
				return strconv.Quote(u.Name), ""
			}
			return "", "This switch matches on text, so the case needs a word or a quoted string."
		case token.INT, token.FLOAT, token.CHAR, token.IMAG:
			if _, ok := userExpr.(*ast.BasicLit); ok {
				return raw, ""
			}
			return "", "This switch matches on numbers, so the case needs a number."
		}
		return raw, ""
	case *ast.Ident, *ast.SelectorExpr:
		switch userExpr.(type) {
		case *ast.Ident, *ast.SelectorExpr:
			return raw, ""
		}
		return "", "This switch matches on named values (constants), so the case needs a name like the existing ones — not a literal."
	}
	return raw, ""
}

// substituteCaseRefs replaces mentions of the sibling's case values
// in a cloned body with the new value. String contents are replaced
// in both their plain and Title forms (retry -> Retry) so derived
// names follow the project's own habit.
func substituteCaseRefs(text string, tmplExprs []string, newExpr string) string {
	out := text
	for _, old := range tmplExprs {
		if old == "" || old == newExpr {
			continue
		}
		out = strings.ReplaceAll(out, old, newExpr)
		// Derived spellings: "retry" the string literal became
		// "backoff"; a Retry identifier in the body becomes Backoff.
		if oldLit, err := strconv.Unquote(old); err == nil {
			if newLit, err2 := strconv.Unquote(newExpr); err2 == nil && oldLit != newLit {
				out = strings.ReplaceAll(out, titleFirst(oldLit), titleFirst(newLit))
			}
		}
	}
	return out
}

func titleFirst(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// insertedLine finds the 1-based line of the first occurrence of
// marker in content (used to report where an insert landed after
// any gofmt pass).
func insertedLine(content []byte, marker string) int {
	text := string(content)
	i := strings.Index(text, strings.TrimSpace(marker))
	if i < 0 {
		return 1
	}
	return strings.Count(text[:i], "\n") + 1
}

// selectorRoots returns the qualifier identifiers an expression
// leans on: the X in X.Y, transitively.
func selectorRoots(e ast.Expr) []string {
	seen := map[string]bool{}
	var out []string
	ast.Inspect(e, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			x := sel.X
			for {
				if s2, ok := x.(*ast.SelectorExpr); ok {
					x = s2.X
					continue
				}
				break
			}
			if id, ok := x.(*ast.Ident); ok && !seen[id.Name] {
				seen[id.Name] = true
				out = append(out, id.Name)
			}
		}
		return true
	})
	return out
}

// importSpecOf renders one import spec and its effective package
// name (explicit name if written, else the path's last element).
func importSpecOf(is *ast.ImportSpec) (spec, effName string) {
	path, _ := strconv.Unquote(is.Path.Value)
	effName = path[strings.LastIndex(path, "/")+1:]
	if is.Name != nil {
		effName = is.Name.Name
		return is.Name.Name + " " + is.Path.Value, effName
	}
	return is.Path.Value, effName
}

// planImportEdits works out any import the new case expression
// needs, copied from another file in the same package. Returns a
// refusal string when a needed package is imported nowhere.
func (p *Project) planImportEdits(fn *Func, fileAST *ast.File, fset *token.FileSet, src []byte, newExpr string) ([]textEdit, string) {
	expr, err := parser.ParseExpr(newExpr)
	if err != nil {
		return nil, ""
	}
	roots := selectorRoots(expr)
	if len(roots) == 0 {
		return nil, ""
	}
	// What this file already has: imports, its own package name, and
	// every top-level name in the package.
	have := map[string]bool{fileAST.Name.Name: true}
	for _, is := range fileAST.Imports {
		_, eff := importSpecOf(is)
		have[eff] = true
	}
	var pkg *Package
	for _, pk := range p.Packages {
		if pk.Dir == fn.PkgDir && pk.Name == fn.Pkg {
			pkg = pk
			break
		}
	}
	if pkg != nil {
		for _, f2 := range pkg.Funcs {
			have[f2.Name] = true
		}
		for _, t := range pkg.Types {
			have[t.Name] = true
		}
		for _, v := range pkg.Vars {
			have[v.Name] = true
		}
	}
	var need []string
	for _, r := range roots {
		if !have[r] {
			need = append(need, r)
		}
	}
	if len(need) == 0 {
		return nil, ""
	}
	// Find each needed qualifier's import spec in a sibling file.
	specByRoot := map[string]string{}
	if pkg != nil {
		for _, rel := range pkg.Files {
			if rel == fn.File {
				continue
			}
			_, _, sibAST, refusal := p.readProjectFile(rel)
			if refusal != "" || sibAST == nil {
				continue
			}
			for _, is := range sibAST.Imports {
				spec, eff := importSpecOf(is)
				for _, r := range need {
					if eff == r {
						specByRoot[r] = spec
					}
				}
			}
		}
	}
	for _, r := range need {
		if _, ok := specByRoot[r]; !ok {
			return nil, fmt.Sprintf("That case needs the package %q, and nothing in this project imports it — I can only copy an import that already exists somewhere in the project.", r)
		}
	}
	// Insert the specs into this file's import declaration.
	var specs []string
	for _, r := range need {
		specs = append(specs, specByRoot[r])
	}
	sort.Strings(specs)
	return insertImportSpecs(fileAST, fset, src, specs), ""
}

// insertImportSpecs returns the text edits that add specs (full
// import specs like `"fmt"`) to a file's imports: into the
// existing block before its closing paren, wrapped around a lone
// import, or as a new block right after the package clause when
// the file imports nothing yet. The pure inserts are marked
// isImport so applyTextEdits composes them ahead of any
// declaration insert sharing their offset.
func insertImportSpecs(fileAST *ast.File, fset *token.FileSet, src []byte, specs []string) []textEdit {
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok || gd.Tok != token.IMPORT {
			continue
		}
		if gd.Lparen.IsValid() {
			// Insert before the closing paren's line.
			rbraceLine := lineOfPos(fset, gd.Rparen)
			return []textEdit{{
				start:    lineStartOffset(src, rbraceLine),
				end:      lineStartOffset(src, rbraceLine),
				text:     "\t" + strings.Join(specs, "\n\t") + "\n",
				isImport: true,
			}}
		}
		// A lone import: wrap it into a block.
		declLine := lineOfPos(fset, gd.Pos())
		declEnd := lineOfPos(fset, gd.End()-1)
		old := strings.Join(strings.Split(string(src), "\n")[declLine-1:declEnd], "\n")
		trimmed := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(old), "import"))
		block := "import (\n\t" + trimmed + "\n\t" + strings.Join(specs, "\n\t") + "\n)"
		return []textEdit{{
			start: lineStartOffset(src, declLine),
			end:   lineStartOffset(src, declEnd+1),
			text:  block + "\n",
		}}
	}
	// No imports at all: add a block right after the package clause.
	pkgLine := lineOfPos(fset, fileAST.Name.Pos())
	block := "\nimport (\n\t" + strings.Join(specs, "\n\t") + "\n)\n"
	return []textEdit{{
		start:    lineStartOffset(src, pkgLine+1),
		end:      lineStartOffset(src, pkgLine+1),
		text:     block,
		isImport: true,
	}}
}

// verifyCaseExists re-reads the edited file and confirms the switch
// in fnName now carries a case for expr.
func verifyCaseExists(root, rel, fnName, expr string) bool {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	for _, d := range fileAST.Decls {
		fd, ok := d.(*ast.FuncDecl)
		if !ok || fd.Name.Name != fnName || fd.Body == nil {
			continue
		}
		found := false
		ast.Inspect(fd.Body, func(n ast.Node) bool {
			var body *ast.BlockStmt
			switch s := n.(type) {
			case *ast.SwitchStmt:
				body = s.Body
			case *ast.TypeSwitchStmt:
				body = s.Body
			}
			if body != nil {
				for _, stmt := range body.List {
					if cc, ok := stmt.(*ast.CaseClause); ok {
						for _, e := range cc.List {
							if exprText(fset, e) == expr {
								found = true
							}
						}
					}
				}
			}
			return true
		})
		if found {
			return true
		}
	}
	return false
}

// --- add a struct field ----------------------------------------------------

// structTagRe is the backstop for a tag the sentence supplied:
// one key:"value" pair whose key is an identifier and whose
// value carries no quote, backtick, backslash, or space — the
// characters that could break out of the tag literal the
// planner renders. The chat grammar assembles tags only in
// this shape; anything else is refused here rather than
// written into the user's struct.
var structTagRe = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_-]*:"[A-Za-z0-9_,.\-=:+/]*"$`)

func (p *Project) planAddField(req EditRequest) (*EditPlan, string) {
	if strings.TrimSpace(req.FieldType) == "" {
		return nil, fmt.Sprintf("A field needs a type before I can assemble it — say it like \"add a %s field of type int to %s\".", req.FieldName, req.StructName)
	}
	if strings.Contains(" "+strings.ToLower(req.FieldType)+" ", " tag ") {
		// A tag phrase the grammar couldn't read (no value,
		// a value with spaces) lands inside the type span of
		// the plain form. A type can't hold a tag — refuse
		// with the working shape rather than rendering the
		// phrase as code.
		return nil, fmt.Sprintf("I read %q as the field's type, and a type can't hold a tag — say the tag as its own phrase, like \"add a %s field of type string with a json tag name to %s\" (one kind word, one value word).", req.FieldType, req.FieldName, req.StructName)
	}
	if req.FieldTag != "" && !structTagRe.MatchString(req.FieldTag) {
		return nil, fmt.Sprintf("That tag (%s) isn't one I can assemble — a tag I can write is one key and value, like \"with a json tag name\": letters, digits, and , . - = : in the value, no spaces or quotes.", req.FieldTag)
	}
	fieldLine := req.FieldName + " " + req.FieldType
	if req.FieldTag != "" {
		fieldLine += " `" + req.FieldTag + "`"
	}
	matches := p.findStructTypes(req.StructName)
	if len(matches) == 0 {
		return nil, fmt.Sprintf("I couldn't find a struct named %q in this project.", req.StructName)
	}
	if len(matches) > 1 {
		var b strings.Builder
		fmt.Fprintf(&b, "%q is more than one struct here:\n", req.StructName)
		for _, t := range matches {
			fmt.Fprintf(&b, "  - %s.%s (%s:%d)\n", t.Pkg, t.Name, p.LinkPath(t.File), t.Line)
		}
		b.WriteString("Say which one with its package — like \"the store.Config struct\" written as store.Config.")
		return nil, b.String()
	}
	t := matches[0]
	for _, f := range t.Fields {
		if strings.EqualFold(f, req.FieldName) {
			return nil, fmt.Sprintf("The %s struct already has a field named %s — nothing to add.", t.Name, req.FieldName)
		}
	}
	if why, blocked := p.fileBlocked(t.File); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(t.File), why)
	}
	src, fset, fileAST, refusal := p.readProjectFile(t.File)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(t.File))
	}
	// Find the struct's AST node.
	var structNode *ast.StructType
	var typeDecl *ast.TypeSpec
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range gd.Specs {
			if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == t.Name {
				if st, ok := ts.Type.(*ast.StructType); ok {
					structNode, typeDecl = st, ts
				}
			}
		}
	}
	if structNode == nil {
		return nil, fmt.Sprintf("I found %s in the index but couldn't re-read its declaration in %s.", t.Name, p.LinkPath(t.File))
	}

	// Anchor: the named field, else the last field.
	fields := structNode.Fields.List
	totalNames := 0
	for _, f := range fields {
		totalNames += len(f.Names)
	}
	anchorIdx := len(fields) - 1
	insertNameIdx := totalNames
	if req.AfterField != "" {
		anchorIdx = -1
		before := 0
		for i, f := range fields {
			// The new field line goes after the anchor's whole
			// declaration line, so its positional index is past
			// every name that line declares (A, B int counts two).
			for _, n := range f.Names {
				if strings.EqualFold(n.Name, req.AfterField) {
					anchorIdx = i
					insertNameIdx = before + len(f.Names)
				}
			}
			before += len(f.Names)
		}
		if anchorIdx < 0 {
			var names []string
			for _, f := range fields {
				for _, n := range f.Names {
					names = append(names, n.Name)
				}
			}
			return nil, fmt.Sprintf("The %s struct has no field named %q — its fields are: %s.", t.Name, req.AfterField, strings.Join(names, ", "))
		}
	}

	editsByFile := map[string][]textEdit{}
	srcByFile := map[string][]byte{t.File: src}
	// The field line itself.
	if anchorIdx >= 0 {
		anchor := fields[anchorIdx]
		endLine := lineOfPos(fset, anchor.End()-1)
		indent := lineIndent(src, lineOfPos(fset, anchor.Pos()))
		editsByFile[t.File] = append(editsByFile[t.File], textEdit{
			start: lineStartOffset(src, endLine+1),
			end:   lineStartOffset(src, endLine+1),
			text:  indent + fieldLine + "\n",
		})
	} else {
		// Empty struct: open the braces with one field inside.
		lbraceLine := lineOfPos(fset, structNode.Fields.Opening)
		indent := lineIndent(src, lineOfPos(fset, typeDecl.Pos())) + "\t"
		editsByFile[t.File] = append(editsByFile[t.File], textEdit{
			start: lineStartOffset(src, lbraceLine+1),
			end:   lineStartOffset(src, lbraceLine+1),
			text:  indent + fieldLine + "\n",
		})
	}

	// Positional (unkeyed) composite literals of this type anywhere
	// in the project would break unless the new field's slot gets a
	// zero value. Keyed literals are untouched. An unsafe zero value
	// with positional literals present is a refusal, not a gamble.
	zero := zeroLiteralFor(p, t, req.FieldType)
	literalFiles := 0
	scanned := map[string]bool{}
	for _, pkg := range p.Packages {
		for _, rel := range pkg.Files {
			if scanned[rel] {
				continue
			}
			scanned[rel] = true
			if why, blocked := p.fileBlocked(rel); blocked {
				_ = why
				continue
			}
			lsrc := srcByFile[rel]
			var lfset *token.FileSet
			var lfile *ast.File
			if lsrc != nil {
				lfset, lfile = fset, fileAST
			} else {
				var r string
				lsrc, lfset, lfile, r = p.readProjectFile(rel)
				if r != "" {
					continue
				}
				if fileBytesBlocked(lsrc) {
					continue
				}
				srcByFile[rel] = lsrc
			}
			_ = lfset
			ast.Inspect(lfile, func(n ast.Node) bool {
				lit, ok := n.(*ast.CompositeLit)
				if !ok {
					return true
				}
				if compositeTypeName(pkg, lit) != t.Pkg+"."+t.Name {
					return true
				}
				if len(lit.Elts) == 0 {
					return true // T{} takes the zero value on its own
				}
				if _, keyed := lit.Elts[0].(*ast.KeyValueExpr); keyed {
					return true
				}
				// A positional literal must name every field.
				if len(lit.Elts) != totalNames {
					refusal = fmt.Sprintf("A positional %s literal at %s doesn't line up with the struct's %d fields, so I can't safely extend it — I won't touch this.", t.Name, p.LinkPath(rel), totalNames)
					return false
				}
				if zero == "" {
					refusal = fmt.Sprintf("%s is built positionally (without field names) at %s, and I can't invent a safe zero value for a new field of type %s — adding the field would leave that literal broken.", t.Name, p.LinkPath(rel), req.FieldType)
					return false
				}
				literalFiles++
				if insertNameIdx < len(lit.Elts) {
					pos := lit.Elts[insertNameIdx].Pos()
					// Need the offset in THIS file's fset.
					off := offsetIn(lsrc, lfset, pos)
					editsByFile[rel] = append(editsByFile[rel], textEdit{start: off, end: off, text: zero + ", "})
				} else {
					off := offsetIn(lsrc, lfset, lit.Elts[len(lit.Elts)-1].End())
					editsByFile[rel] = append(editsByFile[rel], textEdit{start: off, end: off, text: ", " + zero})
				}
				return true
			})
			if refusal != "" {
				return nil, refusal
			}
		}
	}

	var files []EditFile
	for rel, edits := range editsByFile {
		after := applyTextEdits(srcByFile[rel], edits)
		files = append(files, finishFile(rel, srcByFile[rel], after))
	}
	sort.Slice(files, func(i, j int) bool {
		if files[i].Rel == t.File {
			return true
		}
		if files[j].Rel == t.File {
			return false
		}
		return files[i].Rel < files[j].Rel
	})

	summary := fmt.Sprintf("Add the field %s %s to the %s struct (%s:%d), following its existing fields.",
		req.FieldName, req.FieldType, t.Name, p.LinkPath(t.File), t.Line)
	if literalFiles > 0 {
		summary += fmt.Sprintf(" %d place(s) build %s positionally, so each gets a zero value (%s) in the new slot — those files are in the diff too.", literalFiles, t.Name, zero)
	}
	if req.FieldTag != "" {
		summary += " It carries the struct tag `" + req.FieldTag + "`."
	} else {
		hasTag := false
		for _, f := range fields {
			if f.Tag != nil {
				hasTag = true
			}
		}
		if hasTag {
			summary += " Note: the other fields carry struct tags and I did not copy one — add a tag for " + req.FieldName + " if this type is marshaled."
		}
	}
	pkgName, typeName, fieldName := t.Pkg, t.Name, req.FieldName
	fileRel, wantTag := t.File, req.FieldTag
	verifyDesc := fmt.Sprintf("the %s struct now has a %s field", typeName, fieldName)
	if wantTag != "" {
		verifyDesc += " carrying the tag `" + wantTag + "`"
	}
	siteLine := t.Line
	if len(files) > 0 {
		siteLine = insertedLine(files[0].After, req.FieldName+" "+req.FieldType)
	}
	plan := &EditPlan{
		Summary:    summary,
		VerifyDesc: verifyDesc,
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(t.File), siteLine),
		Files:      files,
		verify: func(after *Project) bool {
			found := false
			for _, pk := range after.Packages {
				for _, tt := range pk.Types {
					if tt.Pkg == pkgName && tt.Name == typeName {
						for _, f := range tt.Fields {
							if f == fieldName {
								found = true
							}
						}
					}
				}
			}
			if !found {
				return false
			}
			if wantTag == "" {
				return true
			}
			// A tagged field verifies on its tag too: the
			// field's presence alone wouldn't prove the tag
			// the sentence asked for actually landed.
			_, _, fa, refusal := after.readProjectFile(fileRel)
			if refusal != "" || fa == nil {
				return false
			}
			for _, d := range fa.Decls {
				gd, ok := d.(*ast.GenDecl)
				if !ok {
					continue
				}
				for _, spec := range gd.Specs {
					ts, ok := spec.(*ast.TypeSpec)
					if !ok || ts.Name.Name != typeName {
						continue
					}
					st, ok := ts.Type.(*ast.StructType)
					if !ok {
						continue
					}
					for _, f := range st.Fields.List {
						for _, n := range f.Names {
							if n.Name == fieldName {
								return f.Tag != nil && f.Tag.Value == "`"+wantTag+"`"
							}
						}
					}
				}
			}
			return false
		},
	}
	// The field edit's scaffold: a field has no signature to
	// scaffold against, so the scaffold aims at the one
	// function the index singles out for the struct — its
	// constructor, else its most-called method — as it stands
	// with the new field in place. When the index singles out
	// nothing, the plan carries the reason instead: inventing
	// a target would be fiction.
	if tf, why, reason := p.fieldScaffoldTarget(t); tf != nil {
		if tgt, rerr := p.scaffoldTargetFor(tf); rerr == "" {
			pres := scaffoldPresentation{
				display:    tgt.display,
				verifyNoun: "this edit",
				accompany:  fmt.Sprintf("The scaffold aims at %s, %s.", tgt.display, why),
			}
			if tf.Receiver == "" {
				pres.intro = "%s is a test scaffold assembled by gollemer for " + tgt.display + ", the\n" +
					"constructor of " + t.Name + ", written alongside the " + req.FieldName + " field this\n" +
					"edit adds to " + t.Name + "."
			} else {
				pres.intro = "%s is a test scaffold assembled by gollemer for " + tgt.display + ",\n" +
					"written alongside the " + req.FieldName + " field this edit adds to " + t.Name + "."
			}
			p.attachScaffold(plan, tgt, pres)
		} else {
			plan.Notes = append(plan.Notes, "No test scaffold added: "+rerr)
		}
	} else {
		plan.Notes = append(plan.Notes, "No test scaffold added: "+reason)
	}
	return plan, ""
}

// offsetIn converts a position from a parsed file to a byte offset.
func offsetIn(src []byte, fset *token.FileSet, pos token.Pos) int {
	_ = src
	return fset.Position(pos).Offset
}

// zeroLiteralFor renders a safe zero value for a new field type, or
// "" when no zero can be written with confidence: builtin scalars,
// nil-able kinds, arrays, and in-project struct types qualify.
func zeroLiteralFor(p *Project, owner *Type, typeText string) string {
	t := strings.TrimSpace(typeText)
	switch {
	case t == "string":
		return `""`
	case t == "bool":
		return "false"
	case strings.HasPrefix(t, "[]") && len(t) > 2:
		return "nil"
	case strings.HasPrefix(t, "map["), strings.HasPrefix(t, "*"),
		strings.HasPrefix(t, "chan "), strings.HasPrefix(t, "func("),
		strings.HasPrefix(t, "interface"):
		return "nil"
	case strings.HasPrefix(t, "["):
		return t + "{}"
	}
	switch t {
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
		"float32", "float64", "complex64", "complex128", "byte", "rune":
		return "0"
	}
	// A named type: {} is only valid for structs. In-project types
	// can be checked; anything external stays unsafe.
	var fromPkg *Package
	for _, pk := range p.Packages {
		if pk.Name == owner.Pkg && pk.Dir == owner.PkgDir {
			fromPkg = pk
			break
		}
	}
	if tt := p.lookupType(t, fromPkg); tt != nil && tt.Kind == "struct" {
		return t + "{}"
	}
	return ""
}

// --- register an entry in a table -------------------------------------------

func (p *Project) planRegister(req EditRequest) (*EditPlan, string) {
	var matches []*Var
	for _, v := range p.allVars() {
		if strings.EqualFold(v.Name, req.TableName) {
			matches = append(matches, v)
		}
	}
	if len(matches) == 0 {
		return nil, fmt.Sprintf("I couldn't find a package-level table named %q. (Tables are package-level variables built as literals — like `var %s = map[...]...`.)", req.TableName, req.TableName)
	}
	if len(matches) > 1 {
		var b strings.Builder
		fmt.Fprintf(&b, "%q is a variable in more than one package:\n", req.TableName)
		for _, v := range matches {
			fmt.Fprintf(&b, "  - %s.%s (%s:%d)\n", v.Pkg, v.Name, p.LinkPath(v.File), v.Line)
		}
		b.WriteString("I need the only one — rename or say which package's, and I'll retry.")
		return nil, b.String()
	}
	v := matches[0]
	if why, blocked := p.fileBlocked(v.File); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(v.File), why)
	}
	src, fset, fileAST, refusal := p.readProjectFile(v.File)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(v.File))
	}
	// Find the var's initializer literal.
	var lit *ast.CompositeLit
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range gd.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for i, n := range vs.Names {
				if n.Name == v.Name && i < len(vs.Values) {
					if cl, ok := unparen(vs.Values[i]).(*ast.CompositeLit); ok {
						lit = cl
					}
				}
			}
		}
	}
	if lit == nil {
		return nil, fmt.Sprintf("%s isn't a literal table — its value is computed by code, so there's no entry list for me to extend by copying a sibling.", v.Name)
	}

	if _, isMap := lit.Type.(*ast.MapType); isMap {
		return p.planMapEntry(req, v, src, fset, lit)
	}
	if at, isSlice := lit.Type.(*ast.ArrayType); isSlice && at.Len == nil {
		return p.planSliceEntry(req, v, src, fset, lit)
	}
	return nil, fmt.Sprintf("%s is a %s literal — I can only extend maps and simple value lists by assembly.", v.Name, exprText(fset, lit.Type))
}

func unparen(e ast.Expr) ast.Expr {
	for {
		if pe, ok := e.(*ast.ParenExpr); ok {
			e = pe.X
			continue
		}
		return e
	}
}

// literalInsert computes the splice that appends one entry to a
// composite literal: a fresh line when entries live on their own
// lines, an inline ", entry" when the literal is written on one line.
func literalInsert(fset *token.FileSet, src []byte, lit *ast.CompositeLit, entryText string) textEdit {
	last := lit.Elts[len(lit.Elts)-1]
	lastLine := lineOfPos(fset, last.End()-1)
	rbraceLine := lineOfPos(fset, lit.Rbrace)
	if rbraceLine > lastLine {
		indent := lineIndent(src, lineOfPos(fset, last.Pos()))
		return textEdit{
			start: lineStartOffset(src, lastLine+1),
			end:   lineStartOffset(src, lastLine+1),
			text:  indent + entryText + ",\n",
		}
	}
	off := offOf(fset, lit.Rbrace)
	return textEdit{start: off, end: off, text: ", " + entryText}
}

func (p *Project) planMapEntry(req EditRequest, v *Var, src []byte, fset *token.FileSet, lit *ast.CompositeLit) (*EditPlan, string) {
	mt := lit.Type.(*ast.MapType)
	keyType := exprText(fset, mt.Key)
	var keyText string
	switch keyType {
	case "string":
		keyText = strconv.Quote(req.Key)
	default:
		if isIntKind(keyType) {
			if _, err := strconv.ParseInt(strings.TrimSpace(req.Key), 10, 64); err != nil {
				return nil, fmt.Sprintf("The %s table is keyed by %s, so the entry name has to be a number — %q isn't.", v.Name, keyType, req.Key)
			}
			keyText = strings.TrimSpace(req.Key)
		} else {
			return nil, fmt.Sprintf("The %s table is keyed by %s — I can only assemble entries for string- or number-keyed tables.", v.Name, keyType)
		}
	}
	if len(lit.Elts) == 0 {
		return nil, fmt.Sprintf("The %s table is empty — there's no sibling entry to copy, and I won't invent the value shape.", v.Name)
	}
	var tmpl *ast.KeyValueExpr
	for _, elt := range lit.Elts {
		kv, ok := elt.(*ast.KeyValueExpr)
		if !ok {
			return nil, fmt.Sprintf("The %s table has entries that aren't key: value pairs — too unusual a shape to extend safely.", v.Name)
		}
		if exprText(fset, kv.Key) == keyText {
			return nil, fmt.Sprintf("%s is already in the %s table — nothing to add.", keyText, v.Name)
		}
		tmpl = kv
	}
	// Assemble the value from the sibling entry: copy it, then let
	// the key's own spellings follow the new key (retry -> backoff,
	// Retry -> Backoff), the way the project names its handlers.
	valueText := string(src[offOf(fset, tmpl.Value.Pos()):offOf(fset, tmpl.Value.End())])
	if oldContent, err := strconv.Unquote(exprText(fset, tmpl.Key)); err == nil {
		if oldContent != req.Key {
			valueText = strings.ReplaceAll(valueText, titleFirst(oldContent), titleFirst(req.Key))
			valueText = strings.ReplaceAll(valueText, oldContent, req.Key)
		}
	} else if oldText := exprText(fset, tmpl.Key); oldText != keyText {
		valueText = strings.ReplaceAll(valueText, oldText, keyText)
	}
	entry := keyText + ": " + valueText
	ins := literalInsert(fset, src, lit, entry)
	after := applyTextEdits(src, []textEdit{ins})
	ef := finishFile(v.File, src, after)
	siteLine := insertedLine(ef.After, entry)
	tableName, key, rel := v.Name, keyText, v.File
	return &EditPlan{
		Summary:    fmt.Sprintf("Add %s to the %s table (%s), copied from its last entry (%s).", keyText, v.Name, p.LinkPath(v.File), exprText(fset, tmpl.Key)),
		VerifyDesc: fmt.Sprintf("the %s table now has an entry for %s", tableName, key),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), siteLine),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			return verifyMapKey(after.Root, rel, tableName, key)
		},
	}, ""
}

func isIntKind(t string) bool {
	switch t {
	case "int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64", "uintptr", "byte", "rune":
		return true
	}
	return false
}

func (p *Project) planSliceEntry(req EditRequest, v *Var, src []byte, fset *token.FileSet, lit *ast.CompositeLit) (*EditPlan, string) {
	if len(lit.Elts) == 0 {
		return nil, fmt.Sprintf("The %s list is empty — there's no sibling entry to copy.", v.Name)
	}
	var kind token.Token
	for _, elt := range lit.Elts {
		bl, ok := elt.(*ast.BasicLit)
		if !ok {
			return nil, fmt.Sprintf("The %s list holds built values, not simple literals — I can only extend lists of plain words or numbers by copying an entry.", v.Name)
		}
		if kind == token.ILLEGAL {
			kind = bl.Kind
		} else if bl.Kind != kind {
			return nil, fmt.Sprintf("The %s list mixes different kinds of values — too unusual a shape to extend safely.", v.Name)
		}
		if exprText(fset, bl) == req.Key || (kind == token.STRING && bl.Value == strconv.Quote(req.Key)) {
			return nil, fmt.Sprintf("%q is already in the %s list — nothing to add.", req.Key, v.Name)
		}
	}
	var entry string
	switch kind {
	case token.STRING:
		entry = strconv.Quote(req.Key)
	case token.INT:
		if _, err := strconv.ParseInt(strings.TrimSpace(req.Key), 10, 64); err != nil {
			return nil, fmt.Sprintf("The %s list holds numbers, so the new entry has to be one too — %q isn't.", v.Name, req.Key)
		}
		entry = strings.TrimSpace(req.Key)
	default:
		return nil, fmt.Sprintf("The %s list holds %s literals — I only extend word and number lists by assembly.", v.Name, kind)
	}
	ins := literalInsert(fset, src, lit, entry)
	after := applyTextEdits(src, []textEdit{ins})
	ef := finishFile(v.File, src, after)
	siteLine := insertedLine(ef.After, entry)
	tableName, rel := v.Name, v.File
	return &EditPlan{
		Summary:    fmt.Sprintf("Add %s to the %s list (%s), following its existing entries.", entry, v.Name, p.LinkPath(v.File)),
		VerifyDesc: fmt.Sprintf("the %s list now contains %s", tableName, entry),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(rel), siteLine),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			return verifySliceHas(after.Root, rel, tableName, entry)
		},
	}, ""
}

// verifyMapKey re-reads a file and confirms key exists in the named
// var's map literal.
func verifyMapKey(root, rel, varName, keyText string) bool {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	lit := findVarLiteral(fileAST, varName)
	if lit == nil {
		return false
	}
	for _, elt := range lit.Elts {
		if kv, ok := elt.(*ast.KeyValueExpr); ok && exprText(fset, kv.Key) == keyText {
			return true
		}
	}
	return false
}

func verifySliceHas(root, rel, varName, entry string) bool {
	_, fset, fileAST, refusal := (&Project{Root: root}).readProjectFile(rel)
	if refusal != "" || fileAST == nil {
		return false
	}
	lit := findVarLiteral(fileAST, varName)
	if lit == nil {
		return false
	}
	for _, elt := range lit.Elts {
		if exprText(fset, elt) == entry {
			return true
		}
	}
	return false
}

func findVarLiteral(fileAST *ast.File, varName string) *ast.CompositeLit {
	for _, d := range fileAST.Decls {
		gd, ok := d.(*ast.GenDecl)
		if !ok {
			continue
		}
		for _, spec := range gd.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for i, n := range vs.Names {
				if n.Name == varName && i < len(vs.Values) {
					if cl, ok := unparen(vs.Values[i]).(*ast.CompositeLit); ok {
						return cl
					}
				}
			}
		}
	}
	return nil
}

// --- add a function by copying a sibling ------------------------------------

func (p *Project) planCloneFunc(req EditRequest) (*EditPlan, string) {
	matches := p.findProjectFuncs(req.SiblingName)
	if len(matches) == 0 {
		return nil, fmt.Sprintf("I couldn't find a function named %q to copy. Assembly copies a real function from this project — name one that exists.", req.SiblingName)
	}
	if len(matches) > 1 {
		return nil, p.funcCandidates("name the receiver or package", req.SiblingName, matches)
	}
	sib := matches[0]
	if sib.Name == "main" {
		return nil, "A program gets exactly one main — cloning it would never compile. Name a different function to copy."
	}
	if !token.IsIdentifier(req.NewName) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a function.", req.NewName)
	}
	// The new name must be free in the sibling's package (same
	// receiver shape for methods).
	for _, pkg := range p.Packages {
		if pkg.Dir != sib.PkgDir {
			continue
		}
		for _, fn := range pkg.Funcs {
			if fn.Name == req.NewName && fn.Receiver == sib.Receiver {
				return nil, fmt.Sprintf("%s already exists in that package (%s:%d) — pick a different new name.", req.NewName, p.LinkPath(fn.File), fn.Line)
			}
		}
	}
	if why, blocked := p.fileBlocked(sib.File); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(sib.File), why)
	}
	src, fset, fileAST, refusal := p.readProjectFile(sib.File)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(sib.File))
	}
	var decl *ast.FuncDecl
	for _, d := range fileAST.Decls {
		if fd, ok := d.(*ast.FuncDecl); ok && lineOfPos(fset, fd.Pos()) == sib.Line {
			decl = fd
			break
		}
	}
	if decl == nil {
		return nil, fmt.Sprintf("I found %s but couldn't re-read it in %s.", sib.Display(), p.LinkPath(sib.File))
	}
	// The block: doc comment lines directly above, plus the decl.
	srcLines := strings.Split(string(src), "\n")
	firstLine := sib.Line
	for firstLine-1 >= 1 {
		above := strings.TrimSpace(srcLines[firstLine-2])
		if strings.HasPrefix(above, "//") {
			firstLine--
			continue
		}
		break
	}
	blockStart := lineStartOffset(src, firstLine)
	nameOff, nameEnd := offOf(fset, decl.Name.Pos()), offOf(fset, decl.Name.End())
	declEnd := offOf(fset, decl.End())
	block := string(src[blockStart:declEnd])
	// Rename inside the block: the declaration name, and the doc
	// comment's leading mention of the old name (the Go convention).
	rel := nameOff - blockStart
	block = block[:rel] + req.NewName + block[nameEnd-blockStart:]
	docEnd := offOf(fset, decl.Pos()) - blockStart
	if docEnd > 0 {
		doc := block[:docEnd]
		if fixed := strings.Replace(doc, "// "+sib.Name, "// "+req.NewName, 1); fixed != doc {
			block = fixed + block[docEnd:]
		}
	}
	// Append at the end of the file, the /add-getter way.
	after := append([]byte{}, src...)
	if len(after) == 0 || after[len(after)-1] != '\n' {
		after = append(after, '\n')
	}
	after = append(after, '\n')
	siteLine := bytes.Count(after, []byte("\n")) + 1 + (sib.Line - firstLine)
	after = append(after, block...)
	after = append(after, '\n')
	ef := finishFile(sib.File, src, after)

	note := ""
	if strings.Count(block, sib.Name) > 1 {
		note = fmt.Sprintf(" The body is copied word for word — where it mentioned %s, it still does; check the diff for those spots.", sib.Name)
	}
	pkgDir, recv, newName, relFile := sib.PkgDir, sib.Receiver, req.NewName, sib.File
	plan := &EditPlan{
		Summary:    fmt.Sprintf("Add %s to %s, copied from %s (%s:%d).%s", req.NewName, p.LinkPath(sib.File), sib.Display(), p.LinkPath(sib.File), sib.Line, note),
		VerifyDesc: fmt.Sprintf("a function named %s now exists in %s", newName, relFile),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(relFile), siteLine),
		Files:      []EditFile{ef},
		verify: func(after *Project) bool {
			for _, pk := range after.Packages {
				if pk.Dir != pkgDir {
					continue
				}
				for _, fn := range pk.Funcs {
					if fn.Name == newName && fn.Receiver == recv && fn.File == relFile {
						return true
					}
				}
			}
			return false
		},
	}
	// The clone's scaffold: the new function has the sibling's
	// real signature, so it scaffolds exactly as the delegate's
	// wrapper does — aimed at the NEW name, from the sibling's
	// declaration.
	display := req.NewName
	if sib.Receiver != "" {
		display = strings.Trim(sib.Receiver, "*") + "." + req.NewName
	}
	p.attachScaffold(plan, scaffoldTarget{
		fn: sib, decl: decl, fset: fset, fileAST: fileAST,
		name: req.NewName, display: display,
		entry: entryReason(req.NewName),
	}, scaffoldPresentation{
		display:    display,
		verifyNoun: "this edit",
		intro: "%s is a test scaffold assembled by gollemer for " + display + ", the\n" +
			"function this edit copies from " + sib.Name + ".",
	})
	return plan, ""
}

// --- unified diff -------------------------------------------------------------

// unifiedDiff renders a git-style unified diff of two texts with
// three lines of context. It is deliberately small: line-based LCS,
// hunks merged when their context would touch.
func unifiedDiff(rel string, before, after []byte) string {
	a := strings.Split(string(before), "\n")
	b := strings.Split(string(after), "\n")
	if len(a) > 0 && a[len(a)-1] == "" {
		a = a[:len(a)-1]
	}
	if len(b) > 0 && b[len(b)-1] == "" {
		b = b[:len(b)-1]
	}
	ops := diffOps(a, b)
	// Group into hunks: a change plus up to 3 unchanged lines on
	// each side; two changes within 6 lines merge into one hunk.
	type hunk struct{ lo, hi int } // op indexes [lo, hi)
	var hunks []hunk
	for i, op := range ops {
		if op.kind == eq {
			continue
		}
		lo := i - 3
		if lo < 0 {
			lo = 0
		}
		hi := i + 4
		if hi > len(ops) {
			hi = len(ops)
		}
		if n := len(hunks); n > 0 && lo <= hunks[n-1].hi {
			if hi > hunks[n-1].hi {
				hunks[n-1].hi = hi
			}
		} else {
			hunks = append(hunks, hunk{lo, hi})
		}
	}
	if len(hunks) == 0 {
		return ""
	}
	var sb strings.Builder
	fmt.Fprintf(&sb, "--- a/%s\n+++ b/%s\n", rel, rel)
	// Line numbers in a and b at each op index.
	aLine := make([]int, len(ops)+1)
	bLine := make([]int, len(ops)+1)
	for i, op := range ops {
		aLine[i+1], bLine[i+1] = aLine[i], bLine[i]
		switch op.kind {
		case eq:
			aLine[i+1]++
			bLine[i+1]++
		case del:
			aLine[i+1]++
		case add:
			bLine[i+1]++
		}
	}
	for _, h := range hunks {
		aStart, bStart := aLine[h.lo]+1, bLine[h.lo]+1
		aCount, bCount := aLine[h.hi]-aLine[h.lo], bLine[h.hi]-bLine[h.lo]
		fmt.Fprintf(&sb, "@@ -%s +%s @@\n", spanText(aStart, aCount), spanText(bStart, bCount))
		for i := h.lo; i < h.hi; i++ {
			switch ops[i].kind {
			case eq:
				sb.WriteString(" " + a[ops[i].a] + "\n")
			case del:
				sb.WriteString("-" + a[ops[i].a] + "\n")
			case add:
				sb.WriteString("+" + b[ops[i].b] + "\n")
			}
		}
	}
	return strings.TrimRight(sb.String(), "\n")
}

func spanText(start, count int) string {
	if count == 1 {
		return strconv.Itoa(start)
	}
	if count == 0 {
		return strconv.Itoa(start-1) + ",0"
	}
	return fmt.Sprintf("%d,%d", start, count)
}

const (
	eq  = 0
	del = 1
	add = 2
)

type diffOp struct {
	kind int
	a, b int // line indexes into a / b (-1 when not applicable)
}

// diffOps computes an LCS-based edit script between two line lists.
func diffOps(a, b []string) []diffOp {
	n, m := len(a), len(b)
	// LCS lengths table.
	dp := make([][]int, n+1)
	for i := range dp {
		dp[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if a[i] == b[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else if dp[i+1][j] >= dp[i][j+1] {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1]
			}
		}
	}
	var ops []diffOp
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case a[i] == b[j]:
			ops = append(ops, diffOp{eq, i, j})
			i++
			j++
		case dp[i+1][j] >= dp[i][j+1]:
			ops = append(ops, diffOp{del, i, -1})
			i++
		default:
			ops = append(ops, diffOp{add, -1, j})
			j++
		}
	}
	for ; i < n; i++ {
		ops = append(ops, diffOp{del, i, -1})
	}
	for ; j < m; j++ {
		ops = append(ops, diffOp{add, -1, j})
	}
	return ops
}
