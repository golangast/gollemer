package analyze

// The serve-static piece (chat/servestatic.go): wire ONE
// folder of files — pages, styles, scripts, images — into
// the project's existing webserver, at the conventional
// route /static/. It is the wiring machinery's shape
// (editwire.go) with a file server as the handler:
//
//	http.Handle("/static/", http.StripPrefix("/static/",
//	    http.FileServer(http.Dir("<folder>"))))
//
// inserted as one line immediately before the serve
// statement, on the default mux or the file's own
// *http.ServeMux. The folder path follows the serve-html
// convention (editservehtml.go): project-root-relative,
// because the chat runs programs from the project root, so
// http.Dir("jim/temp") reads the folder the user means.
//
// Content types are not assembled per-extension: FileServer
// picks each file's type from its extension (html, css, js,
// png, jpg, svg, ico, …), and the plan says so — the one
// mechanism covers images, js, and css alike.
//
// Conflicts are read project-wide, exactly like the wiring
// planner's: /static/ already serving THIS folder is a
// nothing-to-add answer; /static/ serving another folder or
// held by any other handler refuses with both named; no
// server is the two-step "add a webserver first" guidance; a
// custom server (one the wiring machinery cannot register
// on) is refused with the exact line handed over.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// StaticRoute is the one route the serve-static piece
// registers — the same route the webpage "assets" template
// links its files under.
const StaticRoute = "/static/"

// staticDirArgRe reads the folder out of a registration's
// rendered handler expression ("…http.Dir(\"static\")…") so
// a conflict can name the folder the route already serves.
var staticDirArgRe = regexp.MustCompile(`Dir\("([^"]*)"\)`)

// planServeStatic assembles the file-server registration.
func (p *Project) planServeStatic(req EditRequest) (*EditPlan, string) {
	folder := filepath.ToSlash(strings.TrimSpace(req.StaticFolder))
	if folder == "" {
		return nil, "that edit needs the folder whose files the server should serve."
	}
	if st, err := os.Stat(filepath.Join(p.Root, filepath.FromSlash(folder))); err != nil || !st.IsDir() {
		return nil, fmt.Sprintf("there's no folder %s in this project to serve files from — say \"create a folder %s\" first, put your pages, styles and images in it, then ask again", folder, folder)
	}

	w := p.wireScan()
	if len(w.sites) == 0 {
		target := "main.go"
		if eps := p.EntryPoints(); len(eps) > 0 {
			target = eps[0].File
		}
		return nil, fmt.Sprintf("there's no webserver in this project yet — nothing calls ListenAndServe, so there's no server to serve the files with. Add one first: say \"add a webserver to %s\", then ask me again.", target)
	}

	regLineFor := func(site *wireSite) string {
		return staticRegLine(site, folder)
	}

	// Conflicts at the route, read across the whole project:
	// a path answers for one thing, never two.
	for _, r := range w.regs {
		if r.route != StaticRoute {
			continue
		}
		if strings.Contains(r.target, "FileServer") {
			if m := staticDirArgRe.FindStringSubmatch(r.target); m != nil {
				if m[1] == folder {
					return nil, fmt.Sprintf("the server in %s already serves the %s folder at /static/ (%s:%d) — nothing to add.", p.LinkPath(r.file), folder, p.LinkPath(r.file), r.line)
				}
				return nil, fmt.Sprintf("/static/ already serves a different folder — %s:%d points it at the folder %q. I won't point one path at two folders; if the %s folder should be the served one instead, change that line by hand and ask again.", p.LinkPath(r.file), r.line, m[1], folder)
			}
		}
		return nil, fmt.Sprintf("/static/ is already taken — %s:%d registers it for %s. I won't point one path at two things, so the %s folder stays unserved until that path is free.", p.LinkPath(r.file), r.line, r.target, folder)
	}

	// The site: a composable server (the default mux or the
	// file's own mux — the wiring machinery's classification).
	var composable []*wireSite
	for _, s := range w.sites {
		if s.kind != siteCustom {
			composable = append(composable, s)
		}
	}
	if len(composable) == 0 {
		s := w.sites[0]
		return nil, fmt.Sprintf("%s already runs a server in its own way — %s — so a registration on the default mux would never be reached, and I won't add a second server. To serve the %s folder there, add this line where the server is set up, before it starts serving:\n\n%s",
			p.LinkPath(s.file), s.how, folder, regLineFor(s))
	}
	site := pickStaticSite(composable, folder)
	if site == nil {
		var b strings.Builder
		fmt.Fprintf(&b, "this project has more than one webserver that could serve the %s folder — ", folder)
		for i, s := range composable {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "the one in %s (%s)", p.LinkPath(s.file), s.funcName)
		}
		fmt.Fprintf(&b, " — and the sentence doesn't say which one, so I won't guess. The line is %s: add it by hand in the main you mean, just before its ListenAndServe.", regLineFor(composable[0]))
		return nil, b.String()
	}
	if why, blocked := p.fileBlocked(site.file); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(site.file), why)
	}
	if fileBytesBlocked(site.wf.src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(site.file))
	}

	// Assemble: one line, immediately before the statement
	// that serves, at that statement's indent — the wiring
	// planner's insertion rule, unchanged.
	regLine := regLineFor(site)
	line := lineOfPos(site.wf.fset, site.stmt.Pos())
	indent := lineIndent(site.wf.src, line)
	insertAt := lineStartOffset(site.wf.src, line)
	after := applyTextEdits(site.wf.src, []textEdit{{start: insertAt, end: insertAt, text: indent + regLine + "\n"}})
	ef := finishFile(site.file, site.wf.src, after)

	siteRel := site.file
	return &EditPlan{
		Summary: fmt.Sprintf("Serve the %s folder's files from the webserver in %s: one line — %s — added in %s just before it serves, so http://localhost:8080/static/style.css is answered with the %s folder's style.css. The same line covers the folder's scripts and images (.js, .png, .jpg, .svg, .ico): the file server picks each file's content type from its extension, so images, js, and css are all served by this one registration.",
			folder, p.LinkPath(site.file), regLine, site.funcName, folder),
		VerifyDesc: fmt.Sprintf("%s now serves the %s folder at /static/", filepath.Base(site.file), folder),
		Site:       fmt.Sprintf("%s:%d", p.LinkPath(site.file), insertedLine(ef.After, regLine)),
		Files:      []EditFile{ef},
		Notes: []string{
			fmt.Sprintf("Only the files inside the %s folder are reachable this way — nothing outside that folder is served.", folder),
			"Your existing server is untouched apart from the one registration above — no second ListenAndServe is added.",
		},
		verify: func(after *Project) bool {
			// The intended structure, read off the fresh
			// analysis: a /static/ registration in the
			// serving file, and it is a file server over
			// this folder.
			for _, r := range after.wireScan().regs {
				if r.file == siteRel && r.route == StaticRoute &&
					strings.Contains(r.target, "FileServer") &&
					strings.Contains(r.target, "Dir("+strconv.Quote(folder)+")") {
					return true
				}
			}
			return false
		},
	}, ""
}

// staticRegLine renders the registration for one site: the
// receiver is the site's mux variable on a mux site, else
// the file's own name for the net/http package; StripPrefix,
// FileServer, and Dir are always net/http's.
func staticRegLine(site *wireSite, folder string) string {
	httpQual := site.wf.httpName
	if httpQual == "" {
		httpQual = "http"
	}
	recv := httpQual
	if site.kind == siteMux {
		recv = site.muxVar
	}
	q := strconv.Quote(StaticRoute)
	return fmt.Sprintf("%s.Handle(%s, %s.StripPrefix(%s, %s.FileServer(%s.Dir(%s))))",
		recv, q, httpQual, q, httpQual, httpQual, strconv.Quote(folder))
}

// pickStaticSite chooses the server a folder is served
// from: the composable program whose own folder holds the
// static folder (the deepest such program wins — jim's
// server serves jim/temp), else the project-root program,
// else the only composable server. Nil when several
// unrelated programs could take it: a choice, not a guess.
func pickStaticSite(composable []*wireSite, folder string) *wireSite {
	var best *wireSite
	for _, s := range composable {
		if s.pkgDir != "" && (folder == s.pkgDir || strings.HasPrefix(folder, s.pkgDir+"/")) {
			if best == nil || len(s.pkgDir) > len(best.pkgDir) {
				best = s
			}
		}
	}
	if best != nil {
		return best
	}
	for _, s := range composable {
		if s.pkgDir == "" {
			return s
		}
	}
	if len(composable) == 1 {
		return composable[0]
	}
	return nil
}
