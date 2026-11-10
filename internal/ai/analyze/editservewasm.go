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

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ServeWasmServerFolder is the folder the created server
// lives in when "serve the wasm app" makes one: a folder
// named "server" at the project root, holding the one
// main.go the run claim then runs by that folder's name.
const ServeWasmServerFolder = "server"

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

	w := p.wireScan()
	if _, _, refusal := p.staticRouteState(w, folder); refusal != "" {
		return nil, refusal
	}
	if len(w.sites) > 0 {
		// The chat routes this case to the serve-static edit;
		// this plan only assembles when there is no server to
		// add the route to.
		s := w.sites[0]
		return nil, fmt.Sprintf("this project already has a webserver — the one in %s — so no new server is needed: the %s folder's /static/ registration belongs on that server (the \"serve static files\" edit adds it there).", p.LinkPath(s.file), folder)
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
