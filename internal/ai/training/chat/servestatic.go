package chat

// "Serve static files" — the builder side of the book's
// "static files" knowledge: wire a folder of pages, styles,
// scripts, and images into the project's existing webserver
// at /static/, through the wiring machinery's rules
// (analyze.EditServeStatic). Claimed deterministically in
// the unified loop beside serve-html and the page family,
// ahead of the phrasing layer.
//
// Grammar (exactly this):
//
//	serve [the] static files|assets|content [from <folder>]
//	add static file serving [from <folder>]
//	serve [the] <folder> folder|directory
//	let the server serve <asset words>
//	update|make [the] server [to] serve <asset words>
//
//	asset words: images, css, js, javascript, scripts,
//	styles, files, assets, pictures, photos, fonts, media,
//	pages, html — joined by "and" and commas; the tail must
//	be ONLY those words ("update the server to serve
//	images, js, css" is John's sentence).
//
// The boundary with the concept book is the tail: the book's
// static-files lines carry a "to <file>" target ("serve
// static files to main.go", "add static files to main.go",
// "add an html template to main.go") and stay the book's —
// every shape here ends at the asset words or at a "from
// <folder>" clause, never at a file target.
//
// The folder: named in the sentence, or the folder of the
// webpage this session just made (the natural pair: make a
// page in jim/temp, then "serve static files"), or — with
// neither — one clarify.go question over the project's
// folders that actually hold web files. A folder that
// doesn't exist is the create-folder two-step, said plainly.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// serveStaticRequest is one parsed serve-static ask.
type serveStaticRequest struct {
	folder string // the folder as typed ("" = unnamed: session default or a question)
}

// staticAssetWords are the words an asset tail may consist
// of (besides "and"): the things a beginner says a server
// should serve.
var staticAssetWords = map[string]bool{
	"images": true, "image": true, "pictures": true, "picture": true,
	"photos": true, "photo": true, "css": true, "js": true,
	"javascript": true, "scripts": true, "script": true,
	"styles": true, "style": true, "files": true, "assets": true,
	"fonts": true, "media": true, "pages": true, "html": true,
	"static": true,
}

// parseServeStatic reads one chat line as a serve-static
// ask. The second return is false for anything outside the
// grammar — including the concept book's targeted lines
// (see the file header).
func parseServeStatic(line string) (serveStaticRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return serveStaticRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(strings.TrimSuffix(f, ","))
	}

	switch low[0] {
	case "serve":
		i := 1
		if i < len(fields) && low[i] == "the" {
			i++
		}
		if i >= len(fields) {
			return serveStaticRequest{}, false
		}
		// serve [the] static files|assets|content [from <folder>]
		if low[i] == "static" {
			if i+1 >= len(fields) {
				return serveStaticRequest{}, false
			}
			switch low[i+1] {
			case "files", "assets", "content":
			default:
				return serveStaticRequest{}, false
			}
			return serveStaticFromTail(fields, low, i+2)
		}
		// serve [the] <folder> folder|directory
		last := low[len(fields)-1]
		if (last == "folder" || last == "directory") && len(fields)-1 > i {
			folder := strings.Join(fields[i:len(fields)-1], "/")
			if strings.HasPrefix(strings.ToLower(folder), "static") {
				return serveStaticRequest{}, false
			}
			return serveStaticRequest{folder: folder}, true
		}
		return serveStaticRequest{}, false
	case "add":
		// add static file serving [from <folder>] — the
		// book's "add static files to <file>" is a different
		// sentence and dies here, at the noun.
		if len(fields) >= 4 && low[1] == "static" && low[2] == "file" && low[3] == "serving" {
			return serveStaticFromTail(fields, low, 4)
		}
		return serveStaticRequest{}, false
	case "let":
		// let the server serve <asset words>
		if len(fields) >= 5 && low[1] == "the" && low[2] == "server" && low[3] == "serve" {
			if staticAssetTail(low[4:]) {
				return serveStaticRequest{}, true
			}
		}
		return serveStaticRequest{}, false
	case "update", "make":
		// update|make [the] server [to] serve <asset words>
		i := 1
		if i < len(fields) && low[i] == "the" {
			i++
		}
		if i >= len(fields) || low[i] != "server" {
			return serveStaticRequest{}, false
		}
		i++
		if i < len(fields) && low[i] == "to" {
			i++
		}
		if i >= len(fields) || low[i] != "serve" {
			return serveStaticRequest{}, false
		}
		i++
		if staticAssetTail(low[i:]) {
			return serveStaticRequest{}, true
		}
		return serveStaticRequest{}, false
	}
	return serveStaticRequest{}, false
}

// serveStaticFromTail reads the optional "from <folder>"
// clause that may end a serve-static shape. Anything else
// after the shape's own words — above all a "to <file>"
// target, which is the concept book's sentence — is not
// this claim's.
func serveStaticFromTail(fields, low []string, i int) (serveStaticRequest, bool) {
	if i == len(fields) {
		return serveStaticRequest{}, true
	}
	if low[i] != "from" {
		return serveStaticRequest{}, false
	}
	i++
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i >= len(fields) {
		return serveStaticRequest{}, false
	}
	rest := fields[i:]
	if n := len(rest); n > 0 {
		if l := strings.ToLower(rest[n-1]); l == "folder" || l == "directory" {
			rest = rest[:n-1]
		}
	}
	if len(rest) == 0 {
		return serveStaticRequest{}, false
	}
	return serveStaticRequest{folder: strings.Join(rest, "/")}, true
}

// staticAssetTail reports whether a word run is only asset
// words joined by "and" — the whole tail of a "let/update
// the server serve …" sentence, never a longer instruction.
func staticAssetTail(low []string) bool {
	if len(low) == 0 {
		return false
	}
	seen := false
	for _, w := range low {
		if w == "and" {
			continue
		}
		if !staticAssetWords[w] {
			return false
		}
		seen = true
	}
	return seen
}

// tryServeStatic is the chat-loop entry: it claims
// serve-static sentences and runs the flow on the chat's
// line reader.
func tryServeStatic(line string, r *LineReader) bool {
	return runServeStatic(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runServeStatic carries one line from sentence to resolved
// folder to the shared edit spine. It returns true when the
// line belonged to this claim at all.
func runServeStatic(line string, ask editAsk) bool {
	q, ok := parseServeStatic(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "serve-static")
	if !ok {
		return true
	}

	// The folder, in the brief's order: named in the
	// sentence; else the folder of the webpage this session
	// just made; else the project's own candidate folders —
	// one of them used plainly, several asked about, none an
	// honest stop.
	folder := q.folder
	if folder == "" {
		if d, yes := sessionWebpageFolder(rootRaw, rootAbs); yes {
			folder = d
		}
	}
	if folder == "" {
		cands := staticFolderCandidates(rootAbs)
		switch {
		case len(cands) == 1:
			folder = cands[0]
		case len(cands) > 1:
			picks := make([]clarifyCandidate, len(cands))
			for i, c := range cands {
				picks[i] = clarifyCandidate{Label: c, Words: []string{c}}
			}
			askPickQuestion(tag,
				"Which folder should the server serve at /static/? The line looks like http://localhost:8080/static/style.css for a style.css inside it.",
				"Answer with the number or the folder name",
				picks,
				func(idx int, ask2 editAsk) bool {
					return runServeStatic("serve the "+cands[idx]+" folder", ask2)
				})
			recordBrain("composer", false, "serve-static question: folder unnamed")
			return true
		default:
			fmt.Printf("%s Not serving static files yet: I couldn't find a folder in this project holding web files (pages, styles, scripts, or images) to serve. Put your files in a folder first — say \"create a folder static\" — or name the folder in the sentence, like \"serve the jim/temp folder\".\n", tag)
			recordBrain("composer", false, "serve-static refused: no candidate folder")
			return true
		}
	}

	// Settle the folder to a project-relative path and check
	// it is really a folder here. A named folder goes through
	// the containment fence; a session or candidate folder is
	// already a project-relative path from this session's own
	// records, and is checked on disk the same way.
	var rel string
	var abs string
	if q.folder != "" {
		a, r, refusal := fenceResolve(rootAbs, folder)
		if refusal != "" {
			fmt.Printf("%s Not serving static files: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "serve-static refused: folder outside project")
			return true
		}
		abs, rel = a, r
	} else {
		rel = folder
		abs = filepath.Join(rootAbs, filepath.FromSlash(folder))
	}
	if st, err := os.Stat(abs); err != nil {
		fmt.Printf("%s Not serving static files: there's no folder %s in this project — say \"create a folder %s\" first, put your pages, styles and images in it, then ask again.\n", tag, rel, rel)
		recordBrain("composer", false, "serve-static refused: folder missing")
		return true
	} else if !st.IsDir() {
		fmt.Printf("%s Not serving static files: %s is a file, not a folder — the file serving serves a whole folder of files at once. Name a folder instead, like \"serve the jim/temp folder\".\n", tag, rel)
		recordBrain("composer", false, "serve-static refused: folder is a file")
		return true
	}

	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "serve-static") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:         analyze.EditServeStatic,
		StaticFolder: rel,
	}, ask)
}

// sessionWebpageFolder is the serve-static folder default:
// the folder holding the webpage this session most recently
// landed (the webpage piece's file), when the session's
// context belongs to this project. "." when the page sits
// at the project root.
func sessionWebpageFolder(rootRaw, rootAbs string) (string, bool) {
	if workCtx.lastFile == "" || !strings.HasSuffix(strings.ToLower(workCtx.lastFile), ".html") {
		return "", false
	}
	if workCtx.root != "" && workCtx.root != rootRaw && workCtx.root != rootAbs {
		return "", false
	}
	if d := slashDir(workCtx.lastFile); d != "" {
		return d, true
	}
	return ".", true
}

// staticFolderCandidates finds the folders that could be
// served: project folders (the root itself excluded) that
// directly hold at least one web file — a page, style,
// script, or image. Shallow (two levels below the root),
// sorted, and blind to .git/vendor/hidden folders: enough
// to offer real choices without walking a whole tree.
func staticFolderCandidates(rootAbs string) []string {
	exts := map[string]bool{
		".html": true, ".css": true, ".js": true, ".png": true,
		".jpg": true, ".jpeg": true, ".gif": true, ".svg": true,
		".ico": true, ".webp": true,
	}
	var out []string
	var walk func(dirAbs, rel string, depth int)
	walk = func(dirAbs, rel string, depth int) {
		entries, err := os.ReadDir(dirAbs)
		if err != nil {
			return
		}
		holds := false
		for _, e := range entries {
			if !e.IsDir() && exts[strings.ToLower(filepath.Ext(e.Name()))] {
				holds = true
				break
			}
		}
		if holds && rel != "" {
			out = append(out, rel)
		}
		if depth >= 2 {
			return
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			name := e.Name()
			if strings.HasPrefix(name, ".") || name == "vendor" || name == "node_modules" {
				continue
			}
			child := name
			if rel != "" {
				child = rel + "/" + name
			}
			walk(filepath.Join(dirAbs, name), child, depth+1)
		}
	}
	walk(rootAbs, "", 0)
	sort.Strings(out)
	return out
}

// --- shared settle helpers (webpage.go uses these too) ----

// staticAssetsRoot settles the folder the two static-asset
// claims edit in: the chat's creation root, symlink-resolved
// and present. The failure lines are printed here, in the
// claim families' common words; the caller just stops.
func staticAssetsRoot(tag, what string) (rootRaw, rootAbs string, ok bool) {
	rootRaw = creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, what+" refused: no root")
		return "", "", false
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, what+" refused: root unreadable")
			return "", "", false
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, what+" refused: root missing")
		return "", "", false
	}
	return rootRaw, rootAbs, true
}

// settleStaticAssetsProject makes rootAbs the spine's
// current project, quietly — the serve-html flow's rule,
// including its one extension: a project with no Go packages
// yet gets an empty index rooted there, because these
// planners read files and wiring facts, and the spine
// re-analyzes for real after the write, before verifying.
func settleStaticAssetsProject(tag, rootRaw, rootAbs, what string) bool {
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		p, err := analyze.Analyze(rootAbs)
		switch {
		case err == nil:
			setCurrentProject(rootRaw, p)
		case strings.Contains(err.Error(), "no Go packages found"):
			setCurrentProject(rootRaw, &analyze.Project{Root: rootAbs})
		default:
			fmt.Printf("I couldn't analyze %s: %v\n", rootAbs, err)
			recordBrain("composer", false, what+" refused: project unreadable")
			return false
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, what+" refused: stale project unreadable")
		return false
	}
	return true
}
