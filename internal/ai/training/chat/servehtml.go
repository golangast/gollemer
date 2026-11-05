package chat

// "Serve an html file" — the serve-a-page recipe, claimed
// deterministically in the unified loop beside add-a-webserver,
// whose precedent it follows. It exists for the composite ask
// no single shape owned: "build and serve an html file" wants
// TWO artifacts — the page and the Go wiring that serves it —
// and before this claim the sentence fell through to shapes
// that each do half the job (the concept book's static-files
// row pastes a snippet into a function; add-a-webserver writes
// a server that serves a friendly line, not the user's page).
//
// Grammar (exactly this):
//
//	serve [up] [a|an|the] <page>.html [target]
//	serve [a|an|the] html file|page [target]
//	add|put|build|make|create|get [a|an|the] server|webserver|web server|http server
//	    [that] [to] <serve word> [up] <page>.html|html file|page [target]
//	serve word: serve|serves|serving|returns|gives|sends|shows|displays|hosts
//	target: to|in|into [the] [folder|file] <one token>
//
// The serve word's synonym family (returns/gives/…/hosts, and
// "serves up") was widened 2026-11-05: "build a server that
// returns a HTML file" is the way a beginner says this ask,
// and the capability sat unreachable behind the one verb the
// grammar happened to know. The widening is slot-shaped only:
// the line still needs the server word before it and a page
// after it, so the neighbours' sentences ("add a server that
// returns static files…") still die at the page spec, exactly
// as they died at the serve word before.
//
// The page defaults to index.html when the ask names no file
// ("serve an html file"). The target is optional: without one,
// the project's own main file is used — or created at the
// project root when the project has no program yet, the
// recipe's one deliberate extension of the add-server rules
// (a recipe that creates its page can create its program).
// A named file that does not exist is likewise created, when
// its folder exists. Everything else follows the neighbours:
// the containment fence, the folder rule (main.go, else the
// single package-main file, ambiguity refused), and the
// PlanEdit spine's gate, build, verify, and restore
// (analyze.EditServeHTML plans the two files as ONE plan).
//
// Non-claims are pinned as carefully as claims: the concept
// book's lines keep their claims — "add static files to
// main.go", "serve static files to main.go", "add an html
// template to main.go" — because this grammar fires only on
// serve-a-file intent: a serve word AND a page (an .html name,
// or the words "html file"/"html page").

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// serveHTMLRequest is one parsed recipe ask.
type serveHTMLRequest struct {
	html          string // the page as typed ("" = no name: the default index.html)
	target        string // the Go target as typed ("" = no target clause)
	targetMissing bool   // a target clause was started but named nothing
}

// parseServeHTML reads one chat line as a serve-a-page ask.
// The second return is false for anything outside the grammar
// — including the neighbours' sentences (see the file header).
func parseServeHTML(line string) (serveHTMLRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return serveHTMLRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	i := 0
	switch low[0] {
	case "serve":
		i = 1
		// "serve up <page>": the phrasal particle rides
		// behind the lead verb.
		if i < len(fields) && low[i] == "up" {
			i++
		}
	case "add", "put", "build", "make", "create", "get":
		i = 1
		if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
			i++
		}
		// The server word: the thing being built is a server.
		switch {
		case i < len(fields) && (low[i] == "server" || low[i] == "webserver"):
			i++
		case i+1 < len(fields) && low[i] == "web" && low[i+1] == "server":
			i += 2
		case i+1 < len(fields) && low[i] == "http" && low[i+1] == "server":
			i += 2
		default:
			return serveHTMLRequest{}, false
		}
		// The serve word: that serves / serving / to serve,
		// and its synonym family (see serveWordAt).
		if i < len(fields) && low[i] == "that" {
			i++
		}
		ni, ok := serveWordAt(low, i)
		if !ok {
			return serveHTMLRequest{}, false
		}
		i = ni
	default:
		return serveHTMLRequest{}, false
	}
	// The page: an .html name as typed, or the words "html
	// file"/"html page" (the default page). A bare "html" with
	// neither is not a page this grammar can point at — the
	// html-template row's sentences end exactly there.
	html, ni, ok := serveHTMLSpecAt(fields, low, i)
	if !ok {
		return serveHTMLRequest{}, false
	}
	i = ni
	req := serveHTMLRequest{html: html}
	if i == len(fields) {
		return req, true
	}
	// The optional target clause: one preposition, the
	// designator slots the neighbours read, then ONE token.
	if low[i] != "to" && low[i] != "in" && low[i] != "into" {
		return serveHTMLRequest{}, false
	}
	i++
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) {
		switch strings.TrimSuffix(low[i], "/") {
		case "folder", "directory", "file":
			i++
		}
	}
	if i >= len(fields) {
		// A target clause was started and named nothing
		// ("serve index.html to the file"): claimed, so the
		// flow can ask in serve terms — the add-server ruling.
		req.targetMissing = true
		return req, true
	}
	if len(fields[i:]) != 1 {
		// A longer tail is another ask glued behind this one;
		// multi-piece lines are not this claim's.
		return serveHTMLRequest{}, false
	}
	req.target = fields[i]
	return req, true
}

// serveWordAt reads the serve word at position i: one of the
// family's verbs (serve itself, or the ordinary words for
// what a server does with a page — returns, gives, sends,
// shows, displays, hosts, in third person after "that" and
// in base form behind "to": "a server to return the page"),
// optionally followed by the phrasal "up" ("serves up"). It
// returns where the next word stands. The family is verbs
// only: a word outside it ("the server runs…") is not this
// recipe's sentence.
func serveWordAt(low []string, i int) (int, bool) {
	if i+1 < len(low) && low[i] == "to" {
		i++
	}
	if i >= len(low) {
		return i, false
	}
	switch low[i] {
	case "serve", "serves", "serving",
		"returns", "return", "gives", "give", "sends", "send",
		"shows", "show", "displays", "display", "hosts", "host":
		i++
	default:
		return i, false
	}
	if i < len(low) && low[i] == "up" {
		i++
	}
	return i, true
}

// serveHTMLSpecAt reads the page spec at position i: an .html
// token (returned as typed), or "html file"/"html page"
// (returned as "", the default page). One leading determiner
// is skipped.
func serveHTMLSpecAt(fields, low []string, i int) (html string, next int, ok bool) {
	if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	if i >= len(fields) {
		return "", i, false
	}
	if strings.HasSuffix(low[i], ".html") {
		return fields[i], i + 1, true
	}
	if low[i] == "html" && i+1 < len(fields) && (low[i+1] == "file" || low[i+1] == "page") {
		return "", i + 2, true
	}
	return "", i, false
}

// serveTargets is the resolved recipe: the Go file (created by
// the plan when goExists is false) and the page, both
// project-relative slash paths.
type serveTargets struct {
	goRel    string
	goExists bool
	htmlRel  string
	htmlName string // the page's name as typed, or index.html
}

// resolveServeTargets settles the ask's two files inside
// rootAbs, or returns an honest refusal naming the working
// alternative. The Go side follows resolveServerFile's rules
// with the recipe's create move added; the page lands in the
// Go file's folder (or a folder inside it).
func resolveServeTargets(rootAbs string, q serveHTMLRequest) (serveTargets, string) {
	htmlName := q.html
	if htmlName == "" {
		htmlName = "index.html"
	}
	if strings.Contains(htmlName, "..") {
		return serveTargets{}, fmt.Sprintf("the page name %q steps outside the project — name a page inside the Go file's folder", q.html)
	}
	t := serveTargets{htmlName: htmlName}

	if q.target != "" {
		if strings.HasSuffix(strings.ToLower(q.target), ".html") {
			return serveTargets{}, fmt.Sprintf("%s is a page, not a Go file — the wiring goes in a .go file. Say \"serve %s to main.go\" and the page will be %s", q.target, htmlName, q.target)
		}
		abs, rel, refusal := serverTargetResolve(rootAbs, q.target)
		if refusal != "" {
			return serveTargets{}, refusal
		}
		goRel, goExists, refusal := resolveServeGoFile(rootAbs, abs, rel, q.target, htmlName)
		if refusal != "" {
			return serveTargets{}, refusal
		}
		t.goRel, t.goExists = goRel, goExists
	} else {
		goRel, goExists, refusal := resolveServeGoFile(rootAbs, rootAbs, "", "", htmlName)
		if refusal != "" {
			return serveTargets{}, refusal
		}
		t.goRel, t.goExists = goRel, goExists
	}

	// The page: in the Go file's folder, or a folder inside it
	// when the typed name carries a subpath.
	goDir := path.Dir(t.goRel)
	t.htmlRel = path.Join(goDir, htmlName)
	htmlAbs := filepath.Join(rootAbs, filepath.FromSlash(t.htmlRel))
	if dir := filepath.Dir(htmlAbs); dir != rootAbs {
		if st, err := os.Stat(dir); err != nil || !st.IsDir() {
			relDir := filepath.ToSlash(mustRel(rootAbs, dir))
			return serveTargets{}, fmt.Sprintf("there's no folder %s in this project for the page to live in — say \"create a folder %s\" first, then ask again", relDir, relDir)
		}
	}
	return t, ""
}

// resolveServeGoFile settles one Go target — a folder (disp ==
// "" means the project root itself) or a file, existing or to
// be created — to the project-relative .go file the wiring
// goes into. The folder rule is the add-server one (main.go,
// else the single package-main file, ambiguity refused), with
// the recipe's create move where the folder plainly has no
// program yet.
func resolveServeGoFile(rootAbs, abs, rel, disp, htmlName string) (goRel string, goExists bool, refusal string) {
	st, err := os.Stat(abs)
	if err == nil && st.IsDir() {
		folderDisp := disp
		if folderDisp == "" {
			folderDisp = "this project"
		}
		if fi, err := os.Stat(filepath.Join(abs, "main.go")); err == nil && !fi.IsDir() {
			return joinRel(rel, "main.go"), true, ""
		}
		cands := packageMainFiles(abs)
		switch len(cands) {
		case 1:
			return joinRel(rel, cands[0]), true, ""
		case 0:
			// No main program in the folder. The recipe
			// creates one — unless the folder already has Go
			// files of another package, where a package-main
			// file would clash with them at build time.
			if folderHasGoFiles(abs) {
				return "", false, fmt.Sprintf("%s has Go files but none of them is a main program — serving needs package main. Name the program file to create, like \"serve %s to %smain.go\"", folderDisp, htmlName, relPrefix(rel))
			}
			return joinRel(rel, "main.go"), false, ""
		default:
			return "", false, fmt.Sprintf("%s has more than one main-package file (%s) — say which one, like \"serve %s to %s%s\"", folderDisp, strings.Join(cands, ", "), htmlName, relPrefix(rel), cands[0])
		}
	}
	if err == nil {
		// An existing file that is not a folder.
		if !strings.HasSuffix(strings.ToLower(disp), ".go") {
			return "", false, fmt.Sprintf("%s isn't a Go file — the wiring goes into a .go file in package main", disp)
		}
		return rel, true, ""
	}
	// A missing target: the recipe creates the program file,
	// when its folder exists.
	if !strings.HasSuffix(strings.ToLower(disp), ".go") {
		return "", false, fmt.Sprintf("%s isn't a Go file — name a .go file for the wiring, like \"serve %s to main.go\"", disp, htmlName)
	}
	if _, err := os.Stat(filepath.Dir(abs)); err != nil {
		folder := filepath.ToSlash(mustRel(rootAbs, filepath.Dir(abs)))
		return "", false, fmt.Sprintf("there's no folder %s in this project — say \"create a folder %s\" first, then ask again", folder, folder)
	}
	return rel, false, ""
}

// folderHasGoFiles reports whether dir itself (not its
// subfolders) holds any non-test .go file — the folder rule's
// own read of "this folder has Go code", distinct from
// beginner.go's deeper project scan.
func folderHasGoFiles(dir string) bool {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false
	}
	for _, e := range entries {
		name := e.Name()
		if !e.IsDir() && strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go") {
			return true
		}
	}
	return false
}

// joinRel joins a project-relative folder ("" for the root)
// and a file name into a project-relative slash path.
func joinRel(rel, name string) string {
	if rel == "" || rel == "." {
		return name
	}
	return rel + "/" + name
}

// relPrefix renders a project-relative folder as a path
// prefix for suggestion sentences ("" for the root).
func relPrefix(rel string) string {
	if rel == "" || rel == "." {
		return ""
	}
	return rel + "/"
}

// tryServeHTML is the chat-loop entry: it claims serve-a-page
// sentences and runs the flow on the chat's line reader.
func tryServeHTML(line string, r *LineReader) bool {
	return runServeHTML(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runServeHTML carries one line from sentence to resolved
// targets to the shared edit spine. It returns true when the
// line belonged to this claim at all.
func runServeHTML(line string, ask editAsk) bool {
	q, ok := parseServeHTML(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.targetMissing {
		// The grammar matched and started a target clause but
		// named nothing ("serve index.html to the file"):
		// claimed, and answered in serve terms — this sentence
		// must never reach the intent layer.
		fmt.Printf("%s Not serving that yet: the sentence starts naming the Go file but doesn't finish — say it like \"serve index.html to main.go\" (a file) or \"serve index.html to jim\" (its folder), or leave the target off and I'll use the project's main file.\n", tag)
		recordBrain("composer", false, "serve-html refused: target unnamed")
		return true
	}
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "serve-html refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "serve-html refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "serve-html refused: root missing")
		return true
	}
	t, refusal := resolveServeTargets(rootAbs, q)
	if refusal != "" {
		fmt.Printf("%s Not serving that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "serve-html refused: target")
		return true
	}
	// The spine plans against the analyzed project; make the
	// target's root the current one, quietly (the add-server
	// flow's reasoning). One state is this recipe's own: a
	// project with no Go packages yet cannot be analyzed —
	// and does not need to be, when the plan CREATES the
	// program: an empty index rooted there is enough for the
	// planner (this recipe reads files, not the index), and
	// the spine re-analyzes for real after the write, before
	// verifying.
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		p, err := analyze.Analyze(rootAbs)
		switch {
		case err == nil:
			setCurrentProject(rootRaw, p)
		case !t.goExists && strings.Contains(err.Error(), "no Go packages found"):
			setCurrentProject(rootRaw, &analyze.Project{Root: rootAbs})
		default:
			fmt.Printf("I couldn't analyze %s: %v\n", rootAbs, err)
			recordBrain("composer", false, "serve-html refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "serve-html refused: stale project unreadable")
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:      analyze.EditServeHTML,
		File:      t.goRel,
		ServeHTML: t.htmlRel,
	}, ask)
}
