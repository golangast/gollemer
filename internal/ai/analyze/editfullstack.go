package analyze

// The full-stack piece (chat/fullstack.go; John's ask,
// 2026-11-09: "add something for restful and forms with
// gowasm and the go backend and the database"): one named
// thing grown through every layer in ONE plan —
//
//	backend   server/<thing>.go beside the serving program:
//	          a connection to the project's database, the
//	          table created at startup, and the REST
//	          handlers (list, read one, create, update,
//	          delete) at /api/<thing>; the serving
//	          program's main gains one setup call, and the
//	          /static/ registration for the app's folder
//	          when it isn't there yet — so page and API are
//	          served by the same server, same origin.
//	frontend  <app>/<thing>.go in the wasm app: a form (one
//	          labeled input per field, and a save button)
//	          and the list the server's rows render into,
//	          mounted in the body components flow like any
//	          other component, talking to the API with
//	          fetch through syscall/js.
//
// THE DATABASE is rqlite — the engine this project's own
// database pieces already speak (the concept book's rqlite
// rows, the gorqlite family): a database server (the
// rqlited program) holding the data in its own SQLite file
// and answering over HTTP at http://localhost:4001. The
// generated connection is the book rows' shape — an
// address and an HTTP client, statements sent to
// /db/execute and /db/query with their values beside them,
// never inside the SQL text. There is deliberately no
// other engine and no new dependency: the backend file is
// standard library only, like the HTTP rows it follows.
// The rows live in the node, so they survive this server
// stopping and starting again; the plan says the node's
// part out loud (Notes), because a server started with no
// node up serves the page but cannot reach the data.
//
// The thing's name is used exactly as given — for the
// route segment, the table, and the generated names. No
// pluralizing, no renaming: "called notes" is /api/notes
// and the notes table; "called note" would be /api/note.

import (
	"fmt"
	"go/format"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// StackField is one field of a full-stack thing, as the
// sentence gave it: its name (used exactly — the JSON
// name, the column, the input's id) and its kind, one of
// the StackField* words. An unnamed kind in the sentence
// is text, the chat layer's default, applied before the
// request is built.
type StackField struct {
	Name string
	Kind string
}

// The field kinds the full-stack grammar knows: a line of
// text, a number, and a yes-or-no.
const (
	StackFieldText   = "text"
	StackFieldNumber = "number"
	StackFieldYesNo  = "yesno"
)

// stackNameRe is the shape a thing or field name must
// have: one word of letters, digits, and underscores,
// starting with a letter — it has to work as a SQL
// identifier, a route segment, and the root of the
// generated Go names, all three, with no quoting games.
var stackNameRe = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*$`)

// stackTitle upper-cases a name's first letter for the
// generated Go names (notes → Notes, in setupNotes and
// the NotesRow family). Names are validated ASCII-first-
// letter before this is ever called.
func stackTitle(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// stackFieldColumn is a field kind's column type in the
// table: text stays text, a number is a REAL (any JSON
// number fits), a yes-or-no is an INTEGER holding 0 or 1
// (SQLite has no boolean of its own).
func stackFieldColumn(kind string) string {
	switch kind {
	case StackFieldNumber:
		return "REAL"
	case StackFieldYesNo:
		return "INTEGER"
	default:
		return "TEXT"
	}
}

// stackFieldGoType is a field kind's Go type in the row
// struct — the JSON shape the handlers trade in.
func stackFieldGoType(kind string) string {
	switch kind {
	case StackFieldNumber:
		return "float64"
	case StackFieldYesNo:
		return "bool"
	default:
		return "string"
	}
}

// StackFieldsProblem words the first thing wrong with a
// thing/fields pair, or "" when the pair can be planned.
// The claim's parser applies the same rules for its
// guidance lines; the planner re-checks because a plan is
// the last word before files move.
func StackFieldsProblem(thing string, fields []StackField) string {
	if !stackNameRe.MatchString(thing) {
		return fmt.Sprintf("%q can't name a stack: a thing's name is one word of letters, digits, and underscores, starting with a letter — it becomes the /api/ address, the table's name, and the root of the generated code's names, all at once.", thing)
	}
	if len(fields) == 0 {
		return "a full stack needs at least one field — the form, the table, and the JSON are all built from the field list."
	}
	seen := map[string]bool{}
	goNames := map[string]string{}
	for _, f := range fields {
		if !stackNameRe.MatchString(f.Name) {
			return fmt.Sprintf("%q can't name a field: a field's name is one word of letters, digits, and underscores, starting with a letter.", f.Name)
		}
		if strings.EqualFold(f.Name, "id") {
			return "no field can be called id — every stack's table grows its own id column (the row's number, the one the /api/<thing>/<id> addresses use), so the name is taken before the sentence starts."
		}
		switch f.Kind {
		case StackFieldText, StackFieldNumber, StackFieldYesNo:
		default:
			return fmt.Sprintf("%q isn't a field kind this sentence knows — a field is text (the default), number, or yesno (\"yes-no\" works too).", f.Kind)
		}
		if seen[f.Name] {
			return fmt.Sprintf("the field %s is named twice — a thing's fields are one list, and each name in it is one column, one input, and one JSON name.", f.Name)
		}
		seen[f.Name] = true
		if other, clash := goNames[stackTitle(f.Name)]; clash {
			return fmt.Sprintf("the fields %s and %s would both become the Go name %s in the generated row type — a thing's fields need names that stay apart when their first letter is capitalized.", other, f.Name, stackTitle(f.Name))
		}
		goNames[stackTitle(f.Name)] = f.Name
	}
	return ""
}

// fullStackSetupName is the generated setup function's
// name for a thing: setupNotes for "notes" — the one call
// the serving program's main gains.
func fullStackSetupName(thing string) string { return "setup" + stackTitle(thing) }

// planFullStack assembles the whole stack for req's thing
// and fields. The serving program the backend joins is
// req.StackServerFile (resolved by the chat's composition
// from the serve-the-wasm-app facts), or — empty — the
// plan creates the server in server/, carrying the
// /static/ registration, exactly the serve sentence's
// arrangement: the app in its folder, its webserver in
// server/, two folders with one job each.
func (p *Project) planFullStack(req EditRequest) (*EditPlan, string) {
	thing := strings.TrimSpace(req.StackThing)
	fields := req.StackFields
	if problem := StackFieldsProblem(thing, fields); problem != "" {
		return nil, problem
	}
	folder := path.Clean(filepath.ToSlash(strings.TrimSpace(req.WasmFolder)))
	if folder == "" || folder == "." {
		return nil, "that edit needs the folder holding the wasm app the stack's form joins."
	}

	// ---- the app side: the folder must hold the app the
	// frontend joins, with the components block the mount
	// lands in and the host page the styles join.
	if st, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(folder))); err != nil || !st.IsDir() {
		return nil, fmt.Sprintf("there's no folder %s in this project for the app to live in — say \"create a folder %s\" first, then \"add a wasm app in %s\", then ask again.", folder, folder, folder)
	}
	_, appMainRel, _, appIndexRel, _ := wasmAppFiles(folder)
	appMainBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(appMainRel)))
	if err != nil {
		return nil, fmt.Sprintf("there's no wasm app in %s yet — the stack's form and list live in the app, so say \"add a wasm app in %s\" first, then ask again.", folder, folder)
	}
	appMain := string(appMainBefore)
	if !wasmAppMainContent(appMain) {
		return nil, fmt.Sprintf("%s isn't a wasm app's main file — it doesn't carry the js && wasm build tag a wasm app's main wears — so the %s stack has no app to join. Say \"add a wasm app in %s\" to make one.", p.LinkPath(appMainRel), thing, folder)
	}
	begin := strings.Index(appMain, "// gollemer:components-begin")
	end := strings.Index(appMain, "// gollemer:components-end")
	if begin < 0 || end < 0 || end < begin {
		return nil, fmt.Sprintf("%s doesn't have the components block the stack's section mounts into — the two gollemer:components marker comments in its mount function — so the %s form has nowhere honest to land. Nothing was written.", p.LinkPath(appMainRel), thing)
	}
	appIndexBefore, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(appIndexRel)))
	if err != nil {
		return nil, fmt.Sprintf("%s isn't there — the host page whose style block the stack's styles join is missing, so this isn't the app the wasm piece makes. Nothing was written.", p.LinkPath(appIndexRel))
	}
	appIndex := string(appIndexBefore)

	// ---- where the backend lives: beside the serving
	// program the composition resolved, or in the server
	// the plan creates in server/.
	createServer := req.StackServerFile == ""
	serverRel := req.StackServerFile
	if createServer {
		serverRel = path.Join(ServeWasmServerFolder, "main.go")
	}
	serverDir := path.Dir(serverRel)
	backendRel := path.Join(serverDir, thing+".go")
	frontendRel := path.Join(folder, thing+".go")
	for _, rel := range []string{backendRel, serverRel, frontendRel, appMainRel, appIndexRel} {
		if why, blocked := p.fileBlocked(rel); blocked {
			return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(rel), why)
		}
	}

	// ---- already-there, read off the project as it
	// stands: every layer present is the family's
	// nothing-to-add; some layers present is named, never
	// silently completed or overwritten.
	setupName := fullStackSetupName(thing)
	backendBefore, backendErr := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(backendRel)))
	frontendBefore, frontendErr := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(frontendRel)))
	pkgText := p.stackPackageText(serverDir)
	setupPresent := strings.Contains(pkgText, "func "+setupName+"(")
	routePresent := strings.Contains(pkgText, strconv.Quote("/api/"+thing))
	var there []string
	if backendErr == nil {
		there = append(there, p.LinkPath(backendRel))
	}
	if frontendErr == nil {
		there = append(there, p.LinkPath(frontendRel))
	}
	if setupPresent {
		there = append(there, "the "+setupName+" call in "+p.LinkPath(serverRel))
	}
	if routePresent {
		there = append(there, "the /api/"+thing+" registration")
	}
	backendThere := backendErr == nil
	frontendThere := frontendErr == nil
	switch {
	case backendThere && frontendThere && setupPresent && routePresent:
		return nil, fmt.Sprintf("the %s stack is already there — %s holds its handlers and its table's setup, %s holds its form and list, and /api/%s is registered. Nothing to add. Ask for a different thing (\"add a full stack app called …\") to grow another one beside it.", thing, p.LinkPath(backendRel), p.LinkPath(frontendRel), thing)
	case len(there) > 0:
		return nil, fmt.Sprintf("part of the %s stack is already there — %s — but not the whole of it, and I won't half-grow a stack over pieces I didn't write. Clear those pieces away by hand, or ask for the stack under a different thing's name.", thing, strings.Join(there, ", "))
	}
	// A builder with the section's name but no file of the
	// thing's own: the name is taken inside the app.
	if !frontendThere {
		for _, in := range p.WasmAppComponents(folder) {
			if in.ID == thing {
				return nil, fmt.Sprintf("the wasm app in %s already has a component whose id is %s — the %s section would carry the same id, and two elements with one id can't be told apart on the page. Pick another name for the thing.", folder, thing, thing)
			}
		}
	}

	// ---- the serving file: created (the house server,
	// the static registration already in it) or extended
	// (the setup call, and the registration when the file
	// doesn't carry it, both just before it serves).
	var serverBefore []byte
	var serverAfter []byte
	staticAdded := false
	if createServer {
		if _, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(serverRel))); err == nil {
			return nil, fmt.Sprintf("there's already a %s in this project, but nothing in it serves — no ListenAndServe anywhere — so it isn't a server the %s stack can ride on, and I won't overwrite what's yours.", p.LinkPath(serverRel), thing)
		}
		if st, err := os.Stat(filepath.Join(p.Root, ServeWasmServerFolder)); err == nil && !st.IsDir() {
			return nil, fmt.Sprintf("%s is a file, not a folder — the new server needs a folder named %s of its own to live in, and I won't move what's yours.", p.LinkPath(ServeWasmServerFolder), ServeWasmServerFolder)
		}
		if pkg, clash := p.stackFolderPackageClash(ServeWasmServerFolder); clash {
			return nil, fmt.Sprintf("the %s folder already holds %s code (package %s) — a server there has to be package main, so the %s stack's server can't move in beside it. Nothing was written.", ServeWasmServerFolder, pkg, pkg, thing)
		}
		content := serveWasmServerSource(folder)
		anchor := "\tif err := http.ListenAndServe"
		if !strings.Contains(content, anchor) {
			return nil, "the house server's source doesn't carry the serve statement the setup call belongs before — nothing was written."
		}
		content = strings.Replace(content, anchor, "\t"+setupName+"()\n"+anchor, 1)
		serverAfter = []byte(content)
		staticAdded = true
	} else {
		src, sErr := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(serverRel)))
		if sErr != nil {
			return nil, fmt.Sprintf("there's no %s for the %s stack's setup call to join — the server the sentence resolved isn't there. Nothing was written.", p.LinkPath(serverRel), thing)
		}
		serverBefore = src
		if fileBytesBlocked(src) {
			return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(serverRel))
		}
		w := p.wireScan()
		var site *wireSite
		for _, s := range w.sites {
			if s.file == serverRel {
				site = s
				break
			}
		}
		if site == nil {
			return nil, fmt.Sprintf("there's no webserver in %s this plan can add the %s stack to — the file holds no serve statement in this project's own module. Nothing was written.", p.LinkPath(serverRel), thing)
		}
		if site.kind == siteCustom {
			regLine := staticRegLine(site, folder)
			return nil, fmt.Sprintf("%s already runs a server in its own way — %s — so lines added for the default mux would never be reached, and I won't add a second server. To grow the %s stack there, add these two lines where the server is set up, before it starts serving:\n\n%s\n%s()", p.LinkPath(serverRel), site.how, thing, regLine, setupName)
		}
		// The /static/ route's state, read with the
		// serve-static family's own probe: already serving
		// this folder (in this file or another) means no
		// line is added; taken by anything else refuses in
		// that family's words — one path never points at
		// two folders, and the stack doesn't take it.
		served, _, routeRefusal := p.staticRouteState(w, folder)
		if !served && routeRefusal != "" {
			return nil, routeRefusal
		}
		var insert strings.Builder
		indent := lineIndent(site.wf.src, lineOfPos(site.wf.fset, site.stmt.Pos()))
		if !served {
			insert.WriteString(indent + staticRegLine(site, folder) + "\n")
			staticAdded = true
		}
		insert.WriteString(indent + setupName + "()\n")
		insertAt := lineStartOffset(site.wf.src, lineOfPos(site.wf.fset, site.stmt.Pos()))
		serverAfter = applyTextEdits(site.wf.src, []textEdit{{start: insertAt, end: insertAt, text: insert.String()}})
	}

	// ---- the frontend mount and the styles: the section
	// mounts between the body markers in ask order, like
	// any component; its styles join the host page once.
	endLine := strings.LastIndex(appMain[:end], "\n") + 1
	newAppMain := appMain[:endLine] + "\tapp.Call(\"appendChild\", " + thing + "Section(doc))\n" + appMain[endLine:]
	newIndex := appIndex
	cssAdded := false
	if !strings.Contains(appIndex, stackCSSMarker) {
		styleEnd := strings.Index(appIndex, "</style>")
		if styleEnd < 0 {
			return nil, fmt.Sprintf("%s has no style block for the stack's styles to join — it isn't the app's host page I recognise.", p.LinkPath(appIndexRel))
		}
		newIndex = appIndex[:styleEnd] + stackCSS + appIndex[styleEnd:]
		cssAdded = true
	}

	backendSource := stackBackendSource(thing, fields)
	frontendSource := stackFrontendSource(thing, fields)
	files := []EditFile{
		finishFile(backendRel, backendBefore, []byte(backendSource)),
		finishFile(serverRel, serverBefore, serverAfter),
		finishFile(frontendRel, frontendBefore, []byte(frontendSource)),
		finishFile(appMainRel, appMainBefore, []byte(newAppMain)),
	}
	if cssAdded {
		files = append(files, finishFile(appIndexRel, appIndexBefore, []byte(newIndex)))
	}

	fieldWords := make([]string, len(fields))
	for i, f := range fields {
		fieldWords[i] = f.Name + " (" + f.Kind + ")"
	}
	summary := fmt.Sprintf("Grow the whole %s stack in one plan. The backend: %s gains %s — a connection to the project's rqlite database, the %s table (id plus %s) created when the server starts, and the REST handlers at /api/%s — list, read one, create, update, delete, JSON in and out — registered by one new line in %s, %s(). The app's side: %s joins the app — the %s form (one input per field and a save button) and the list under it — mounted in %s with the app's other components, sending and reading through those same /api/%s addresses. The server serves the app's page from the same program, so the form's requests stay on the one origin.",
		thing, p.LinkPath(serverRel), p.LinkPath(backendRel), thing, strings.Join(fieldWords, ", "), thing, p.LinkPath(serverRel), setupName, p.LinkPath(frontendRel), thing, p.LinkPath(appMainRel), thing)
	if staticAdded {
		summary += fmt.Sprintf(" The /static/ registration serving the %s folder joins %s in the same plan.", folder, p.LinkPath(serverRel))
	}
	if cssAdded {
		summary += fmt.Sprintf(" The section's styles join the host page's style block in %s.", p.LinkPath(appIndexRel))
	}
	return &EditPlan{
		Summary:    summary,
		VerifyDesc: fmt.Sprintf("the %s stack is in place: the /api/%s handlers and the %s table's setup in %s, the form and list mounted in the wasm app in %s", thing, thing, thing, p.LinkPath(serverRel), folder),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(serverRel), insertedLine(serverAfter, setupName+"()")),
		Files:      files,
		Notes: []string{
			fmt.Sprintf("The database is rqlite — the engine this project's database pieces already speak. The %s table lives in an rqlite node's own SQLite file, reached at http://localhost:4001, and the rows survive this server stopping and starting (they are the node's, not the server's). The node has to be running for the data to answer: with no node up, the server still starts and the page still loads, but the list can't load and a save can't land — start one with \"rqlited -node-id=1 data/\" in your own terminal (ask \"how do I run rqlite\" and the chat walks through it), then start the server.", thing),
			"Nothing here logs anybody in — v1 of a stack has no accounts and no passwords; anyone who can open the page can read and change the rows. The form's checks are the simple kind: an empty text field is refused in the page, before anything is sent.",
			fmt.Sprintf("After this lands: \"build the wasm app in %s\" puts the form into the app, and \"serve the wasm app in %s\" runs the server that answers both the page and the API.", folder, folder),
		},
		verify: func(after *Project) bool {
			b, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(backendRel)))
			if err != nil || string(b) != backendSource {
				return false
			}
			s, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(serverRel)))
			if err != nil || !strings.Contains(string(s), setupName+"()") {
				return false
			}
			f, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(frontendRel)))
			if err != nil || string(f) != frontendSource {
				return false
			}
			m, err := os.ReadFile(filepath.Join(after.Root, filepath.FromSlash(appMainRel)))
			if err != nil || !strings.Contains(string(m), thing+"Section(doc)") {
				return false
			}
			return true
		},
	}, ""
}

// stackPackageText reads every .go file of one package
// folder (project-relative; "." is the root) into one
// text, for presence probes (a setup function, a route
// registration) that belong to the package, not to one
// file. Test files are skipped: a stack is runtime code.
func (p *Project) stackPackageText(dir string) string {
	abs := filepath.Join(p.Root, filepath.FromSlash(dir))
	entries, err := os.ReadDir(abs)
	if err != nil {
		return ""
	}
	var b strings.Builder
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		if data, err := os.ReadFile(filepath.Join(abs, name)); err == nil {
			b.Write(data)
			b.WriteByte('\n')
		}
	}
	return b.String()
}

// stackFolderPackageClash reports the package a folder's
// existing .go files declare, when it isn't package main:
// a created server (package main) can't move in beside
// library code. "" when the folder holds no .go files or
// only package main.
func (p *Project) stackFolderPackageClash(dir string) (string, bool) {
	abs := filepath.Join(p.Root, filepath.FromSlash(dir))
	entries, err := os.ReadDir(abs)
	if err != nil {
		return "", false
	}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(abs, name))
		if err != nil {
			continue
		}
		for _, ln := range strings.Split(string(data), "\n") {
			ln = strings.TrimSpace(ln)
			if strings.HasPrefix(ln, "package ") {
				if pkg := strings.Fields(ln)[1]; pkg != "main" {
					return pkg, true
				}
				break
			}
		}
	}
	return "", false
}

// ---- the generated backend -------------------------------------------

// stackBackendSource renders <thing>.go, the server-side
// half: the store (the rqlite connection), the row type,
// the setup that creates the table and registers the
// handlers, and the two handlers themselves. Standard
// library only, in the concept book's rqlite shape —
// statements and their values travel apart, and every
// answer is JSON written the one way.
func stackBackendSource(thing string, fields []StackField) string {
	title := stackTitle(thing)
	store := thing + "Store"
	rowType := title + "Row"
	var b strings.Builder
	b.WriteString("package main\n\n")
	b.WriteString("import (\n\t\"bytes\"\n\t\"encoding/json\"\n\t\"errors\"\n\t\"fmt\"\n\t\"io\"\n\t\"net/http\"\n\t\"strconv\"\n\t\"strings\"\n\t\"time\"\n)\n\n")
	fmt.Fprintf(&b, `// The %s part of this server: a table in the project's
// database and the /api/%s handlers that read and write
// it — grown by the full-stack sentence ("add a full
// stack app called %s ..."). The serving program's main
// calls %s before it serves; everything else here is
// %s' own.
//
// The database is rqlite: a database server (the rqlited
// program) that holds the data in its own SQLite file
// and answers over HTTP at the address below. The table
// is created when this server starts, if it isn't there
// already, and the rows live in the node — stopping this
// server doesn't touch them. No node running means no
// data answers: start one first (rqlited -node-id=1
// data/), and ask the chat "how do I run rqlite" for the
// whole walk-through.

`, thing, thing, thing, fullStackSetupName(thing), thing)
	fmt.Fprintf(&b, `// %s is the %s table's connection to the node: the
// node's address and the HTTP client that talks to it.
// It is this stack's one door to the database — every
// handler below goes through it, the way the concept
// book's rqlite rows do.
type %s struct {
	addr   string
	client *http.Client
}

// open%sStore makes the connection: the node's address, and
// a client with a patience limit, so a node that isn't
// answering fails a request instead of hanging it.
func open%sStore() *%s {
	return &%s{
		addr:   "http://localhost:4001",
		client: &http.Client{Timeout: 10 * time.Second},
	}
}

`, store, thing, store, title, title, store, store)
	// The row type.
	fmt.Fprintf(&b, `// %s is one row of the %s table, in Go and in JSON at
// once: the field names are the JSON names, exactly as
// the sentence gave them.
type %s struct {
	ID int64 `+"`json:\"id\"`"+`
`, rowType, thing, rowType)
	for _, f := range fields {
		fmt.Fprintf(&b, "\t%s %s `json:\"%s\"`\n", stackTitle(f.Name), stackFieldGoType(f.Kind), f.Name)
	}
	b.WriteString("}\n\n")
	// Write result + the two store verbs.
	fmt.Fprintf(&b, `// %sWriteResult is what a write reports back: the id
// it gave a new row, and how many rows it touched.
type %sWriteResult struct {
	LastInsertID int64
	RowsAffected int64
}

// exec runs one statement that changes data — CREATE
// TABLE, INSERT, UPDATE, DELETE — with its values
// beside it: rqlite receives the values apart from the
// SQL, so a value stays data and can never change what
// the statement does. The node's answer says what the
// write did — or carries the statement's own error,
// which can arrive even on an HTTP 200, so it is
// checked, never assumed.
func (s *%s) exec(statement string, values ...any) (%sWriteResult, error) {
	payload, err := json.Marshal([]any{append([]any{statement}, values...)})
	if err != nil {
		return %sWriteResult{}, err
	}
	resp, err := s.client.Post(s.addr+"/db/execute", "application/json", bytes.NewReader(payload))
	if err != nil {
		return %sWriteResult{}, err
	}
	defer resp.Body.Close()
	var answer struct {
		Results []struct {
			LastInsertID int64  `+"`json:\"last_insert_id\"`"+`
			RowsAffected int64  `+"`json:\"rows_affected\"`"+`
			Error        string `+"`json:\"error\"`"+`
		} `+"`json:\"results\"`"+`
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return %sWriteResult{}, err
	}
	if err := json.Unmarshal(body, &answer); err != nil {
		return %sWriteResult{}, s.nodeWords(body)
	}
	if len(answer.Results) == 0 {
		return %sWriteResult{}, errors.New("the node answered without a result")
	}
	if r := answer.Results[0]; r.Error != "" {
		return %sWriteResult{}, errors.New(r.Error)
	} else {
		return %sWriteResult{LastInsertID: r.LastInsertID, RowsAffected: r.RowsAffected}, nil
	}
}

// query runs one SELECT and reads the rows back as
// %s values: the node names the columns it sends, and
// each value lands in its field by that name. (For a
// read that must never be stale, the node's address
// takes ?level=strong — on one node, a plain read
// already sees every finished write.)
func (s *%s) query(statement string, values ...any) ([]%s, error) {
	payload, err := json.Marshal([]any{append([]any{statement}, values...)})
	if err != nil {
		return nil, err
	}
	resp, err := s.client.Post(s.addr+"/db/query", "application/json", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var answer struct {
		Results []struct {
			Columns []string `+"`json:\"columns\"`"+`
			Values  [][]any  `+"`json:\"values\"`"+`
			Error   string   `+"`json:\"error\"`"+`
		} `+"`json:\"results\"`"+`
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(body, &answer); err != nil {
		return nil, s.nodeWords(body)
	}
	if len(answer.Results) == 0 {
		return nil, errors.New("the node answered without a result")
	}
	if answer.Results[0].Error != "" {
		return nil, errors.New(answer.Results[0].Error)
	}
	rows := []%s{}
	for _, values := range answer.Results[0].Values {
		var row %s
		for i, col := range answer.Results[0].Columns {
			v := values[i]
			switch col {
			case "id":
				row.ID = %sInt(v)
`, thing, thing, store, thing, thing, thing, thing, thing, thing, thing, thing, rowType, store, rowType, rowType, rowType, thing)
	for _, f := range fields {
		switch f.Kind {
		case StackFieldText:
			fmt.Fprintf(&b, "\t\t\tcase %q:\n\t\t\t\trow.%s, _ = v.(string)\n", f.Name, stackTitle(f.Name))
		case StackFieldNumber:
			fmt.Fprintf(&b, "\t\t\tcase %q:\n\t\t\t\trow.%s = %sFloat(v)\n", f.Name, stackTitle(f.Name), thing)
		case StackFieldYesNo:
			fmt.Fprintf(&b, "\t\t\tcase %q:\n\t\t\t\trow.%s = %sInt(v) != 0\n", f.Name, stackTitle(f.Name), thing)
		}
	}
	fmt.Fprintf(&b, `			}
		}
		rows = append(rows, row)
	}
	return rows, nil
}

// %sInt reads one value the node sent back as an
// integer: JSON numbers all arrive as float64, and a
// missing value (a NULL in the table) reads as zero.
func %sInt(v any) int64 {
	f, _ := v.(float64)
	return int64(f)
}

// %sFloat reads one value the node sent back as a
// number, the same way.
func %sFloat(v any) float64 {
	f, _ := v.(float64)
	return f
}

// %sJSON writes one JSON answer: the status, the
// content type a JSON reader expects, and the value
// encoded — the one way this stack answers, so every
// answer has the same shape.
func %sJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

// %sFail answers 500 with the store's trouble in
// words: at this stack's scale the likeliest cause is
// the rqlite node not running, and the answer says so.
func %sFail(w http.ResponseWriter, err error) {
	%sJSON(w, http.StatusInternalServerError, map[string]string{"error": "the database didn't answer: " + err.Error() + " — is the rqlite node running? Start one with: rqlited -node-id=1 data/"})
}

`, thing, thing, thing, thing, thing, thing, thing, thing, thing)
	fmt.Fprintf(&b, `// nodeWords words a node answer that wasn't the JSON
// this code reads: a node still starting up answers in
// plain text ("leader not found" while it elects its
// leader), and its own words are worth more than a
// decode error. It is a method on the store so two
// stacks in one package each use their own store's.
func (s *%s) nodeWords(body []byte) error {
	text := strings.TrimSpace(string(body))
	if len(text) > 120 {
		text = text[:120]
	}
	if text == "" {
		return errors.New("the node answered with an empty reply this code couldn't read")
	}
	return fmt.Errorf("the node answered %%q instead of the result this code reads", text)
}

`, store)
	// The setup.
	cols := make([]string, len(fields))
	for i, f := range fields {
		cols[i] = f.Name + " " + stackFieldColumn(f.Kind)
	}
	fmt.Fprintf(&b, `// %s connects the %s stack: it opens the store,
// creates the %s table if it isn't there yet, and
// registers the handlers. The serving program's main
// calls it before serving.
func %s() {
	store := open%sStore()
	// A node started at the same moment as this server
	// can need a few seconds to settle — it elects its
	// leader before it takes writes — so the create is
	// tried a few times before the warning below gives
	// up on it: starting the node and the server
	// together is the normal way to run this stack,
	// not an error.
	var err error
	for tries := 0; tries < 5; tries++ {
		if _, err = store.exec("CREATE TABLE IF NOT EXISTS %s (id INTEGER PRIMARY KEY, %s)"); err == nil {
			break
		}
		time.Sleep(time.Second)
	}
	if err != nil {
		fmt.Printf("%s: the table wasn't created — %%v. The database is an rqlite node at %%s, and it has to be running before this server starts: start one with \"rqlited -node-id=1 data/\", then start this server again.\n", err, store.addr)
	}
	http.HandleFunc("/api/%s", store.%sCollection)
	http.HandleFunc("/api/%s/", store.%sRecord)
}

`, fullStackSetupName(thing), thing, thing, fullStackSetupName(thing), title, thing, strings.Join(cols, ", "), thing, thing, thing, thing, thing)
	// The collection handler.
	selectCols := "id"
	for _, f := range fields {
		selectCols += ", " + f.Name
	}
	placeholders := make([]string, len(fields))
	insertCols := make([]string, len(fields))
	for i, f := range fields {
		placeholders[i] = "?"
		insertCols[i] = f.Name
	}
	fmt.Fprintf(&b, `// %sCollection answers /api/%s itself: GET lists
// every row as a JSON array, in id order; POST makes a
// new row from the JSON body and answers with the row
// as it landed, id included.
func (s *%s) %sCollection(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		rows, err := s.query("SELECT %s FROM %s ORDER BY id")
		if err != nil {
			%sFail(w, err)
			return
		}
		%sJSON(w, http.StatusOK, rows)
	case http.MethodPost:
		var in %s
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
			%sJSON(w, http.StatusBadRequest, map[string]string{"error": "the body wasn't JSON in the %s shape: " + err.Error()})
			return
		}
`, thing, thing, store, thing, selectCols, thing, thing, thing, rowType, thing, thing)
	writeInsertArgs(&b, thing, fields, "in")
	fmt.Fprintf(&b, `		res, err := s.exec("INSERT INTO %s (%s) VALUES (%s)", %s)
		if err != nil {
			%sFail(w, err)
			return
		}
		rows, err := s.query("SELECT %s FROM %s WHERE id = ?", res.LastInsertID)
		if err != nil || len(rows) == 0 {
			if err == nil {
				err = errors.New("the write landed but the row couldn't be read back")
			}
			%sFail(w, err)
			return
		}
		%sJSON(w, http.StatusCreated, rows[0])
	default:
		w.Header().Set("Allow", "GET, POST")
		%sJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": r.Method + " doesn't belong on /api/%s — it lists with GET and grows with POST"})
	}
}

`, thing, strings.Join(insertCols, ", "), strings.Join(placeholders, ", "), insertArgs(fields, "in"), thing, selectCols, thing, thing, thing, thing, thing)
	// The record handler.
	setClauses := make([]string, len(fields))
	for i, f := range fields {
		setClauses[i] = f.Name + " = ?"
	}
	fmt.Fprintf(&b, `// %sRecord answers /api/%s/<id> — one row, named by
// the number the table gave it: GET reads it, PUT
// replaces it from the JSON body and answers with the
// row as it landed, DELETE removes it. An id that
// names no row is a 404, in JSON like every answer.
func (s *%s) %sRecord(w http.ResponseWriter, r *http.Request) {
	idText := strings.TrimPrefix(r.URL.Path, "/api/%s/")
	id, err := strconv.ParseInt(idText, 10, 64)
	if err != nil || id <= 0 {
		%sJSON(w, http.StatusNotFound, map[string]string{"error": "there is no %s row with the id " + idText})
		return
	}
	switch r.Method {
	case http.MethodGet:
		rows, err := s.query("SELECT %s FROM %s WHERE id = ?", id)
		if err != nil {
			%sFail(w, err)
			return
		}
		if len(rows) == 0 {
			%sJSON(w, http.StatusNotFound, map[string]string{"error": "there is no %s row with the id " + idText})
			return
		}
		%sJSON(w, http.StatusOK, rows[0])
	case http.MethodPut:
		var in %s
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
			%sJSON(w, http.StatusBadRequest, map[string]string{"error": "the body wasn't JSON in the %s shape: " + err.Error()})
			return
		}
`, thing, thing, store, thing, thing, thing, thing, selectCols, thing, thing, thing, thing, thing, rowType, thing, thing)
	writeInsertArgs(&b, thing, fields, "in")
	fmt.Fprintf(&b, `		res, err := s.exec("UPDATE %s SET %s WHERE id = ?", %s, id)
		if err != nil {
			%sFail(w, err)
			return
		}
		if res.RowsAffected == 0 {
			%sJSON(w, http.StatusNotFound, map[string]string{"error": "there is no %s row with the id " + idText})
			return
		}
		rows, err := s.query("SELECT %s FROM %s WHERE id = ?", id)
		if err != nil || len(rows) == 0 {
			if err == nil {
				err = errors.New("the write landed but the row couldn't be read back")
			}
			%sFail(w, err)
			return
		}
		%sJSON(w, http.StatusOK, rows[0])
	case http.MethodDelete:
		res, err := s.exec("DELETE FROM %s WHERE id = ?", id)
		if err != nil {
			%sFail(w, err)
			return
		}
		if res.RowsAffected == 0 {
			%sJSON(w, http.StatusNotFound, map[string]string{"error": "there is no %s row with the id " + idText})
			return
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		w.Header().Set("Allow", "GET, PUT, DELETE")
		%sJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": r.Method + " doesn't belong on one %s row — it is read with GET, replaced with PUT, and removed with DELETE"})
	}
}
`, thing, strings.Join(setClauses, ", "), insertArgs(fields, "in"), thing, thing, thing, selectCols, thing, thing, thing, thing, thing, thing, thing, thing, thing)
	return stackFormatted(b.String())
}

// writeInsertArgs writes the yes-or-no conversions a
// write needs ahead of its exec call: a yesno field's
// bool becomes the 0 or 1 its INTEGER column holds, in a
// variable named for the field, so the exec call itself
// reads as a plain list of the row's values.
func writeInsertArgs(b *strings.Builder, thing string, fields []StackField, recv string) {
	for _, f := range fields {
		if f.Kind != StackFieldYesNo {
			continue
		}
		fmt.Fprintf(b, "\t\t%sValue := 0\n\t\tif %s.%s {\n\t\t\t%sValue = 1\n\t\t}\n", f.Name, recv, stackTitle(f.Name), f.Name)
	}
}

// insertArgs renders the exec call's value list for a
// row held in the variable recv: text and number fields
// pass as they are; a yesno field passes its converted
// <field>Value (writeInsertArgs declares it above).
func insertArgs(fields []StackField, recv string) string {
	parts := make([]string, len(fields))
	for i, f := range fields {
		if f.Kind == StackFieldYesNo {
			parts[i] = f.Name + "Value"
		} else {
			parts[i] = recv + "." + stackTitle(f.Name)
		}
	}
	return strings.Join(parts, ", ")
}

// ---- the generated frontend ------------------------------------------

// stackCSSMarker is the comment the stack's style chunk
// opens with in the host page — the once-only test for
// the styles, the way each component's chunk is its own
// test in the component piece.
const stackCSSMarker = "/* full-stack sections"

// stackCSS is the stack section's style chunk: the form
// over the list, in the host page's own measures. It
// scopes everything under .stack- so it composes with
// whatever component styles the page already carries.
const stackCSS = `
/* full-stack sections: a thing's form over the list of
   its rows, as the full-stack sentence grows them. */
.stack-section {
	margin: 24px 0;
}
.stack-form .field {
	margin: 0 0 12px;
}
.stack-form label {
	display: block;
	margin: 0 0 4px;
}
.stack-list {
	margin: 16px 0 0;
}
.stack-row {
	display: flex;
	align-items: center;
	gap: 12px;
	padding: 8px 0;
	border-bottom: 1px solid #999;
}
.stack-row-text {
	flex: 1;
}
.stack-status {
	min-height: 1.2em;
}
`

// stackFrontendSource renders <app>/<thing>.go, the
// browser-side half: the section builder (the form and
// the list), the fetch plumbing, and the render. It is
// self-contained — its own element helper, its own
// callback keep-alive list — because it must compile in
// an app that may have no components.go or events.go
// yet. Everything it says to the server goes to relative
// /api/<thing> addresses: page and API are served by the
// same server, so there is no cross-origin question.
func stackFrontendSource(thing string, fields []StackField) string {
	title := stackTitle(thing)
	var b strings.Builder
	b.WriteString("//go:build js && wasm\n\npackage main\n\n")
	b.WriteString("import (\n\t\"strconv\"\n\t\"strings\"\n\t\"syscall/js\"\n)\n\n")
	fmt.Fprintf(&b, `// The %s section of this app: the form that adds a %s
// row and the list that shows the rows the server
// holds — grown by the full-stack sentence ("add a
// full stack app called %s ..."). The data lives on
// the server this app is served by: the form sends new
// rows to /api/%s and the list reads them back from
// there, in JSON. The addresses are relative — page
// and API come from the same server, so there is no
// cross-origin question to answer.
//
// The section is its own file on purpose: it compiles
// and reads without the app's components or events
// files, and a second stack ("called tasks", say)
// grows a sibling file of its own beside this one.

// %sFuncs holds this section's callbacks for the
// page's whole life — the events file's lesson: a
// js.FuncOf callback the Go side lets go of is
// garbage-collected, and a collected callback silently
// stops answering in the browser.
var %sFuncs []js.Func

// %sKeep keeps one callback alive for the page's life
// and hands it back, so a registration reads as one
// line.
func %sKeep(fn js.Func) js.Func {
	%sFuncs = append(%sFuncs, fn)
	return fn
}

// %sEl makes one element: its tag, its class when it
// has one, and its text when it has some — this
// section's own copy of the components file's helper,
// so the section stands on its own.
func %sEl(doc js.Value, tag, class, text string) js.Value {
	e := doc.Call("createElement", tag)
	if class != "" {
		e.Set("className", class)
	}
	if text != "" {
		e.Set("textContent", text)
	}
	return e
}

`, thing, thing, thing, thing, thing, thing, thing, thing, thing, thing, thing, thing)
	// The section builder.
	fmt.Fprintf(&b, `// %sSection builds the whole section: the form (one
// labeled input per field, and the save button) over
// the list the server's rows render into. Building it
// also asks the server for the current rows, so the
// list fills as soon as the answer comes back.
func %sSection(doc js.Value) js.Value {
	wrap := %sEl(doc, "section", "stack-section", "")
	wrap.Set("id", %q)
	wrap.Call("appendChild", %sEl(doc, "h2", "", %q))

	form := %sEl(doc, "form", "stack-form", "")
`, thing, thing, thing, thing, thing, title, thing)
	for _, f := range fields {
		label := stackTitle(f.Name)
		switch f.Kind {
		case StackFieldText, StackFieldNumber:
			inputType := "text"
			if f.Kind == StackFieldNumber {
				inputType = "number"
			}
			fmt.Fprintf(&b, `	%sField := %sEl(doc, "div", "field", "")
	%sField.Call("appendChild", %sEl(doc, "label", "", %q))
	%sInput := %sEl(doc, "input", "", "")
	%sInput.Set("type", %q)
	%sInput.Set("id", %q)
	%sField.Call("appendChild", %sInput)
	form.Call("appendChild", %sField)
`, f.Name, thing, f.Name, thing, label, f.Name, thing, f.Name, inputType, f.Name, thing+"-"+f.Name, f.Name, f.Name, f.Name)
		case StackFieldYesNo:
			fmt.Fprintf(&b, `	%sField := %sEl(doc, "div", "field", "")
	%sInput := %sEl(doc, "input", "", "")
	%sInput.Set("type", "checkbox")
	%sInput.Set("id", %q)
	%sField.Call("appendChild", %sInput)
	%sField.Call("appendChild", doc.Call("createTextNode", %q))
	form.Call("appendChild", %sField)
`, f.Name, thing, f.Name, thing, f.Name, f.Name, thing+"-"+f.Name, f.Name, f.Name, f.Name, " "+label, f.Name)
		}
	}
	fmt.Fprintf(&b, `	saveButton := %sEl(doc, "button", "btn btn-filled", "Save")
	saveButton.Set("type", "submit")
	form.Call("appendChild", saveButton)
	form.Call("addEventListener", "submit", %sKeep(js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) > 0 {
			args[0].Call("preventDefault")
		}
		%sSave(doc)
		return nil
	})))
	wrap.Call("appendChild", form)

	statusLine := %sEl(doc, "p", "stack-status", "")
	statusLine.Set("id", %q)
	wrap.Call("appendChild", statusLine)

	listBox := %sEl(doc, "div", "stack-list", "")
	listBox.Set("id", %q)
	wrap.Call("appendChild", listBox)

	%sRefresh(doc)
	return wrap
}

// %sSay writes one line into the section's status
// line — the form's own voice for "the title is
// empty", "the server didn't answer", and the rest.
// An empty line clears it.
func %sSay(doc js.Value, words string) {
	doc.Call("getElementById", %q).Set("textContent", words)
}

// %sFetch sends one request to this app's own server
// and hands the answered response to fn. A request
// that never arrives (the server isn't running) is
// caught and said out loud in the status line instead
// of failing silently.
func %sFetch(doc js.Value, method, path, body string, fn func(resp js.Value)) {
	opts := js.Global().Get("Object").New()
	opts.Set("method", method)
	if body != "" {
		headers := js.Global().Get("Object").New()
		headers.Set("Content-Type", "application/json")
		opts.Set("headers", headers)
		opts.Set("body", body)
	}
	caught := %sKeep(js.FuncOf(func(this js.Value, args []js.Value) any {
		%sSay(doc, "The server didn't answer — it may not be running. Serve the app and try again.")
		return nil
	}))
	js.Global().Call("fetch", path, opts).Call("then", %sKeep(js.FuncOf(func(this js.Value, args []js.Value) any {
		if len(args) > 0 {
			fn(args[0])
		}
		return nil
	}))).Call("catch", caught)
}

// %sRefresh asks the server for the rows and renders
// them, replacing whatever the list currently shows.
func %sRefresh(doc js.Value) {
	%sFetch(doc, "GET", "/api/%s", "", func(resp js.Value) {
		if !resp.Get("ok").Bool() {
			%sSay(doc, "The list didn't load — the server answered with an error. Is the database node running?")
			return
		}
		resp.Call("json").Call("then", %sKeep(js.FuncOf(func(this js.Value, args []js.Value) any {
			if len(args) > 0 {
				%sRender(doc, args[0])
			}
			return nil
		})))
	})
}

`, thing, thing, thing, thing, thing+"-status", thing, thing+"-list", thing, thing, thing, thing+"-status", thing, thing, thing, thing, thing, thing, thing, thing, thing, thing, thing, thing)
	// The render.
	fmt.Fprintf(&b, `// %sRender rebuilds the list from the rows the
// server sent: one row element per row — its fields
// as text (set as text, never as HTML, so a row's
// words can never become page markup) and a delete
// button that knows the row's id.
func %sRender(doc js.Value, rows js.Value) {
	list := doc.Call("getElementById", %q)
	for list.Get("firstChild").Truthy() {
		list.Call("removeChild", list.Get("firstChild"))
	}
	if rows.Length() == 0 {
		list.Call("appendChild", %sEl(doc, "p", "", "Nothing here yet — add the first one with the form."))
		return
	}
	for i := 0; i < rows.Length(); i++ {
		row := rows.Index(i)
		id := row.Get("id").Int()
`, thing, thing, thing+"-list", thing)
	for _, f := range fields {
		switch f.Kind {
		case StackFieldText:
			fmt.Fprintf(&b, "\t\t%sText := row.Get(%q).String()\n", f.Name, f.Name)
		case StackFieldNumber:
			fmt.Fprintf(&b, "\t\t%sText := strconv.FormatFloat(row.Get(%q).Float(), 'f', -1, 64)\n", f.Name, f.Name)
		case StackFieldYesNo:
			fmt.Fprintf(&b, "\t\t%sText := \"no\"\n\t\tif row.Get(%q).Bool() {\n\t\t\t%sText = \"yes\"\n\t\t}\n", f.Name, f.Name, f.Name)
		}
	}
	parts := make([]string, len(fields))
	for i, f := range fields {
		parts[i] = f.Name + "Text"
	}
	fmt.Fprintf(&b, `		text := strings.Join([]string{%s}, " · ")
		rowEl := %sEl(doc, "div", "stack-row", "")
		rowEl.Call("appendChild", %sEl(doc, "span", "stack-row-text", text))
		del := %sEl(doc, "button", "btn btn-text", "Delete")
		del.Set("type", "button")
		del.Call("addEventListener", "click", %sKeep(js.FuncOf(func(this js.Value, args []js.Value) any {
			%sDelete(doc, id)
			return nil
		})))
		rowEl.Call("appendChild", del)
		list.Call("appendChild", rowEl)
	}
}

`, strings.Join(parts, ", "), thing, thing, thing, thing, thing)
	// The save.
	fmt.Fprintf(&b, `// %sSave reads the form and sends the new row. The
// one check is the simple kind: an empty text field
// is refused here, in the page, and nothing is sent.
func %sSave(doc js.Value) {
`, thing, thing)
	for _, f := range fields {
		switch f.Kind {
		case StackFieldText:
			fmt.Fprintf(&b, `	%sValue := strings.TrimSpace(doc.Call("getElementById", %q).Get("value").String())
	if %sValue == "" {
		%sSay(doc, "The %s is empty — fill it in first; nothing was sent.")
		return
	}
`, f.Name, thing+"-"+f.Name, f.Name, thing, f.Name)
		case StackFieldNumber:
			fmt.Fprintf(&b, `	%sText := doc.Call("getElementById", %q).Get("value").String()
	// An empty number box reads as 0 — the page's
	// simple check is for text; a number always has a
	// value to send.
	%sValue, _ := strconv.ParseFloat(%sText, 64)
`, f.Name, thing+"-"+f.Name, f.Name, f.Name)
		case StackFieldYesNo:
			fmt.Fprintf(&b, `	%sValue := doc.Call("getElementById", %q).Get("checked").Bool()
`, f.Name, thing+"-"+f.Name)
		}
	}
	fmt.Fprintf(&b, `	payload := js.Global().Get("Object").New()
`)
	for _, f := range fields {
		fmt.Fprintf(&b, "\tpayload.Set(%q, %sValue)\n", f.Name, f.Name)
	}
	fmt.Fprintf(&b, `	%sFetch(doc, "POST", "/api/%s", js.Global().Get("JSON").Call("stringify", payload).String(), func(resp js.Value) {
		if !resp.Get("ok").Bool() {
			%sSay(doc, "The server didn't save that — its answer said no. Nothing was added.")
			return
		}
`, thing, thing, thing)
	for _, f := range fields {
		if f.Kind == StackFieldYesNo {
			fmt.Fprintf(&b, "\t\tdoc.Call(\"getElementById\", %q).Set(\"checked\", false)\n", thing+"-"+f.Name)
		} else {
			fmt.Fprintf(&b, "\t\tdoc.Call(\"getElementById\", %q).Set(\"value\", \"\")\n", thing+"-"+f.Name)
		}
	}
	fmt.Fprintf(&b, `		%sSay(doc, "")
		%sRefresh(doc)
	})
}

// %sDelete removes one row by id and renders the
// list again from the server's answer — the list is
// always the server's truth, never a local guess.
func %sDelete(doc js.Value, id int) {
	%sFetch(doc, "DELETE", "/api/%s/"+strconv.Itoa(id), "", func(resp js.Value) {
		%sRefresh(doc)
	})
}
`, thing, thing, thing, thing, thing, thing, thing)
	return stackFormatted(b.String())
}

// stackFormatted returns generated source gofmt-clean:
// the spine formats created files as it writes them and
// the plan's verify compares the landed bytes with the
// generated source, so the two must be the same bytes.
// A source that doesn't format (a generator bug) goes
// out as built — the build checks downstream will say
// so, in the open, rather than the format error hiding
// here.
func stackFormatted(src string) string {
	if f, err := format.Source([]byte(src)); err == nil {
		return string(f)
	}
	return src
}
