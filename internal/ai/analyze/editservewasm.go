package analyze

// The missing-server half of the chat's "serve the wasm app
// in <folder>" composition (chat/servewasm.go). Serving an
// app takes a server; when the project already has one, the
// route is the serve-static piece's edit and this planner is
// never asked. When the project has NO webserver — nothing
// calls ListenAndServe — this plan creates the one that
// serves the app: the house server component
// (WebServerSource, the same source the add-a-webserver edit
// and the gated creations write) in a new server/main.go,
// already carrying the /static/ file-server registration
// over the app's folder, inserted before the serve statement
// by the serve-static piece's own rule and rendered by its
// own line builder. One plan, one gate: after it lands, the
// project has a program that serves the app at
// http://localhost:8080/static/ — the component's own
// address, read from its source, not a new assumption.
//
// The conflicts are the serve-static piece's, read off the
// same scan (staticRouteState): a /static/ route held by
// anything else refuses with those words, and a route
// already serving this folder is its nothing-to-add. A
// server that exists but is taken (server/main.go already
// there, serving nothing this scan can see) is never
// overwritten.
//
// The scan this family reads is the MODULE-VISIBLE one
// (moduleVisibleWireWorld below): a registration or a
// server inside a nested module — a folder under the root
// carrying its own go.mod — belongs to that module, not to
// this project, which can neither build nor run it. Such a
// registration is neither "already serving" nor a conflict
// for this sentence (John's own project proved it: a
// /static/ registration inside paces/project, a nested
// module, sent the first version of this sentence running
// a file its project could not build). And the sentence
// serves from exactly two homes — server/, the dedicated
// webserver folder at the root, or a program at the root
// itself: a registration serving the app from any other
// program is named, never silently run or extended
// (ServeWasmRouteFacts).

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// ServeWasmServerFolder is the folder the created server
// lives in when "serve the wasm app" makes one: a folder
// named "server" at the project root, holding the one
// main.go the run claim then runs by that folder's name.
const ServeWasmServerFolder = "server"

// fileInProjectModule reports whether the project-relative
// file belongs to the project's own module: no go.mod may
// exist in any directory between the project root and the
// file's folder — the file's own folder included — other
// than the root's own go.mod. A file behind such a wall is
// a different module's code: this project's toolchain
// cannot build it and its programs are not this project's
// to run, so the serve-the-wasm-app sentence treats its
// servers and registrations as if they were not there.
func (p *Project) fileInProjectModule(rel string) bool {
	dir := path.Dir(rel)
	for dir != "." && dir != "/" && dir != "" {
		if _, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(dir), "go.mod")); err == nil {
			return false
		}
		dir = path.Dir(dir)
	}
	return true
}

// moduleVisibleWireWorld is the wire scan restricted to the
// project's own module: the registrations and serving sites
// whose files pass fileInProjectModule. The rest of the
// world (files, handlers) is shared untouched — the
// serve-static planner and the wiring family keep reading
// the full scan exactly as before; only the
// serve-the-wasm-app family's own reads use this view.
func (p *Project) moduleVisibleWireWorld(w *wireWorld) *wireWorld {
	out := *w
	out.regs = nil
	for _, r := range w.regs {
		if p.fileInProjectModule(r.file) {
			out.regs = append(out.regs, r)
		}
	}
	out.sites = nil
	for _, s := range w.sites {
		if p.fileInProjectModule(s.file) {
			out.sites = append(out.sites, s)
		}
	}
	return &out
}

// isServeWasmHome reports whether a file is one of the two
// homes the serve-the-wasm-app sentence serves an app
// from: the dedicated server in server/, or a program at
// the project root. The arrangement is the understandable
// one — the app in its own folder, its webserver in
// server/, two folders with one job each — so a
// registration living in any other program, however
// visible, is not this sentence's server.
func isServeWasmHome(file string) bool {
	dir := path.Dir(file)
	return dir == "." || dir == ServeWasmServerFolder
}

// ServeWasmRouteFacts is the /static/ route as the
// serve-the-wasm-app sentence reads it — over the
// module-visible scan only, and split by WHERE the serving
// registration lives. At most one of ServedFile, OtherFile
// and Conflict is set.
type ServeWasmRouteFacts struct {
	// ServedFile: a module-visible registration already
	// serves the folder from one of the sentence's homes
	// (server/ or a root-level program) — that file is the
	// program to run.
	ServedFile string
	// OtherFile/OtherLine: a module-visible registration
	// serves the folder, but from a folder deeper in the
	// tree that hosts its own program. The sentence does
	// not run that program and does not add lines to it;
	// it serves the app from server/ instead and names
	// this registration first. (A registration in a library
	// file — no server in its folder — is the project's
	// own route instead: ServedFile, as it always was.)
	OtherFile string
	OtherLine int
	// Conflict: the family's conflict words when a
	// module-visible /static/ is held for a different
	// folder or by another handler — the shipped refusal,
	// unchanged.
	Conflict string
	// ServerFile: a module-visible serving site in
	// server/main.go — the server the sentence extends
	// when the route is not serving yet. "" when server/
	// holds no server.
	ServerFile string
	// RootFile: a module-visible serving site in a program
	// at the project root, the other home. "" when none.
	RootFile string
}

// ServeWasmRouteFacts reads the route facts for folder
// (project-relative slash path) the way the
// serve-the-wasm-app sentence acts on them.
func (p *Project) ServeWasmRouteFacts(folder string) ServeWasmRouteFacts {
	w := p.moduleVisibleWireWorld(p.wireScan())
	var facts ServeWasmRouteFacts
	for _, s := range w.sites {
		switch path.Dir(s.file) {
		case ServeWasmServerFolder:
			if facts.ServerFile == "" {
				facts.ServerFile = s.file
			}
		case ".":
			if facts.RootFile == "" {
				facts.RootFile = s.file
			}
		}
	}
	served, file, refusal := p.staticRouteState(w, folder)
	switch {
	case served && (isServeWasmHome(file) || !wireFolderHostsServer(w, file)):
		// Serving from one of the two homes — or from a
		// library file, whose registration rides in whatever
		// program of this project calls it: already serving,
		// in the project's own code, exactly the shipped
		// answer (the claim runs the file, as it always has).
		facts.ServedFile = file
	case served:
		// Serving from a folder that hosts its own program
		// deeper in the tree: that program is not this
		// sentence's server.
		facts.OtherFile = file
		for _, r := range w.regs {
			if r.file == file && r.route == StaticRoute {
				facts.OtherLine = r.line
				break
			}
		}
	default:
		facts.Conflict = refusal
	}
	return facts
}

// wireFolderHostsServer reports whether the folder holding
// file contains a serving site of its own — the file's
// registration belongs to a program living in that folder,
// not to a library other programs call.
func wireFolderHostsServer(w *wireWorld, file string) bool {
	dir := path.Dir(file)
	for _, s := range w.sites {
		if path.Dir(s.file) == dir {
			return true
		}
	}
	return false
}

// serveWasmServerSource renders the created server's whole
// file: the house component, with the /static/ registration
// for folder spliced in immediately before its serve
// statement — the serve-static piece's insertion rule,
// applied to the component's fixed text.
func serveWasmServerSource(folder string) string {
	reg := staticRegLineFor("http", "http", folder)
	return strings.Replace(WebServerSource,
		"\tif err := http.ListenAndServe",
		"\t"+reg+"\n\tif err := http.ListenAndServe", 1)
}

// planServeWasmApp assembles the create-a-server plan for
// req.WasmFolder (the app folder the new server serves).
func (p *Project) planServeWasmApp(req EditRequest) (*EditPlan, string) {
	folder := filepath.ToSlash(strings.TrimSpace(req.WasmFolder))
	if folder == "" {
		return nil, "that edit needs the folder holding the wasm app to serve."
	}
	if st, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(folder))); err != nil || !st.IsDir() {
		return nil, fmt.Sprintf("there's no folder %s in this project to serve — a wasm app lives in a folder, so say \"create a folder %s\" first, then \"add a wasm app in %s\", then ask again", folder, folder, folder)
	}

	// The route and the servers are read over the
	// module-visible scan (moduleVisibleWireWorld): a server
	// or a registration behind a nested module's go.mod is
	// not this project's to run or extend, so it neither
	// blocks this plan nor answers for the route.
	w := p.moduleVisibleWireWorld(p.wireScan())
	served, servedFile, refusal := p.staticRouteState(w, folder)
	switch {
	case served && (isServeWasmHome(servedFile) || !wireFolderHostsServer(w, servedFile)):
		// Already serving from server/, the root program,
		// or a library file whose registration rides in this
		// project's own programs: the family's
		// nothing-to-add, unchanged.
		return nil, refusal
	case !served && refusal != "":
		// /static/ held for another folder, or by another
		// handler: the family's conflict words, unchanged.
		return nil, refusal
	}
	// served from another module-visible program deeper in
	// the tree — one whose own folder hosts its server —
	// is NOT a block: the chat names that registration and
	// this plan gives the app its own server in server/
	// anyway.
	for _, s := range w.sites {
		if path.Dir(s.file) == ServeWasmServerFolder || path.Dir(s.file) == "." {
			// A server already lives in one of the two homes
			// this sentence serves from; the chat routes that
			// case to the serve-static edit on that server.
			// This plan only assembles when neither home has
			// a server to add the route to.
			return nil, fmt.Sprintf("this project already has a webserver — the one in %s — so no new server is needed: the %s folder's /static/ registration belongs on that server (the \"serve static files\" edit adds it there).", p.LinkPath(s.file), folder)
		}
	}

	serverRel := ServeWasmServerFolder + "/main.go"
	if why, blocked := p.fileBlocked(serverRel); blocked {
		return nil, fmt.Sprintf("I won't write %s — %s.", p.LinkPath(serverRel), why)
	}
	if st, err := os.Stat(filepath.Join(p.Root, ServeWasmServerFolder)); err == nil && !st.IsDir() {
		return nil, fmt.Sprintf("%s is a file, not a folder — the new server needs a folder named %s of its own to live in, and I won't move what's yours.", p.LinkPath(ServeWasmServerFolder), ServeWasmServerFolder)
	}
	if _, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(serverRel))); err == nil {
		return nil, fmt.Sprintf("there's already a %s in this project, but nothing in it serves — no ListenAndServe anywhere — so it isn't a server I can add the app's route to, and I won't overwrite what's yours.", p.LinkPath(serverRel))
	}

	content := serveWasmServerSource(folder)
	ef := finishFile(serverRel, nil, []byte(content))
	regLine := staticRegLineFor("http", "http", folder)
	return &EditPlan{
		Summary: fmt.Sprintf("Create the project's webserver in %s — the tiny server that listens at http://localhost:8080 — carrying one extra line, %s, so the wasm app in %s is served at http://localhost:8080/static/. This project has no server yet; this is the one that serves the app.",
			p.LinkPath(serverRel), regLine, folder),
		VerifyDesc: fmt.Sprintf("%s is a webserver serving the %s folder at /static/", filepath.Base(serverRel), folder),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(serverRel), insertedLine(ef.After, "http.ListenAndServe")),
		Files:      []EditFile{ef},
		Notes: []string{
			fmt.Sprintf("The new server is the house starter: one friendly handler at /, a line printed for every visit, and the %s folder's files at /static/ — only the files inside %s are reachable that way.", folder, folder),
			"The wasm app itself is untouched — its own go.mod, its files, and its build stay exactly as they are.",
		},
		verify: func(after *Project) bool {
			w := after.wireScan()
			siteOK := false
			for _, s := range w.sites {
				if s.file == serverRel {
					siteOK = true
				}
			}
			if !siteOK {
				return false
			}
			for _, r := range w.regs {
				if r.file == serverRel && r.route == StaticRoute &&
					strings.Contains(r.target, "FileServer") &&
					strings.Contains(r.target, "Dir("+strconv.Quote(folder)+")") {
					return true
				}
			}
			return false
		},
	}, ""
}
