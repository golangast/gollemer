package chat

// "Create a homepage" — the page family, claimed
// deterministically in the unified loop beside
// serve-an-html-file and wire-a-handler, whose precedent it
// follows. It exists for the composite ask no single shape
// owned: a named page kind ("create a homepage", "create an
// about page", "create a contact page") wants THREE artifacts
// in one gated plan — the page itself, a handler that serves
// it (for form and contact pages, the book's form handler,
// whose fields match the page's fields), and — when the file
// already runs a composable server — the registration that
// wires the handler at the kind's route. Without a server the
// plan still lands page + handler and names the wiring
// sentence ("connect the About handler to the webserver") as
// the follow-up, the serve-html family's no-server answer in
// this family's words.
//
// Grammar (exactly this):
//
//	create|add|make|build [a|an|the] <kind phrase> [tail]
//	kind phrase:
//	  form:    handler with a form | handler named <Name> with a form | form page
//	  home:    homepage | home page
//	  about:   about page | about us page
//	  contact: contact page | contact us page
//	  list:    item list page | list page
//	  details: details page | item details page
//	tail: segments in any order, each at most once:
//	  named <Name>
//	  (to|in|into) [the] [folder|file|directory] <one token>
//
// A tail clause that starts but names nothing (a trailing
// "named", a trailing "in") still claims the line, so the
// flow can answer in page terms — the add-server ruling: a
// sentence about making a page must never reach the intent
// layer. Anything else trailing is not this claim's.
//
// Non-claims are pinned as carefully as claims: the kind words
// anchor the claim, so the book's form-handler sentences keep
// theirs — "add a form handler named Jake to jim/main.go"
// matches no kind phrase ("form page" needs page after form;
// the handler phrases start with "handler" and need "with a
// form" after it), "add a handler named Jake to jim/main.go"
// names no form, "add a contact form named X to main.go" is
// the book's name for its handler, not a kind phrase — and
// serve-html's lines ("serve home.html to …"), add-a-
// webserver's, wire-a-handler's, and the creations' lines all
// start with verbs or object words this grammar never reads.
//
// The plan itself is the analyzer's (analyze.EditPage): the
// chat layer owns only the sentence, the kind table (route,
// page, default handler name), the host file's settling, and
// — for the form kinds — the book's form-handler entry, which
// rides the request so the planner assembles the handler from
// the same piece the book would place alone. The gate, build,
// verify, and restore are the shared spine's, unchanged
// (executeEditPlan).

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// pageKindInfo is one row of the kind table the chat layer
// owns: the kind's route, its page file, the handler name an
// unnamed ask uses, the words its sentences and refusals are
// spelled with, and whether its handler is the book's form
// handler (its fields must match the page's).
type pageKindInfo struct {
	kind        string // the PageKind value: form|home|about|contact|list|details
	display     string // the kind's words: "form page", "homepage", …
	canonical   string // the kind's example sentence
	route       string // the address the handler registers at
	page        string // the page's file name, beside the handler
	defaultName string // the handler's name when the sentence names none
	needsBook   bool   // the handler is the book's form-handler entry
}

var pageKindTable = map[string]pageKindInfo{
	"form":    {kind: "form", display: "form page", canonical: "create a form page", route: "/form", page: "form.html", defaultName: "Form", needsBook: true},
	"home":    {kind: "home", display: "homepage", canonical: "create a homepage", route: "/", page: "index.html", defaultName: "Home"},
	"about":   {kind: "about", display: "about page", canonical: "create an about page", route: "/about", page: "about.html", defaultName: "About"},
	"contact": {kind: "contact", display: "contact page", canonical: "create a contact page", route: "/contact", page: "contact.html", defaultName: "Contact", needsBook: true},
	"list":    {kind: "list", display: "item list page", canonical: "create an item list page", route: "/items", page: "items.html", defaultName: "Items"},
	"details": {kind: "details", display: "details page", canonical: "create a details page", route: "/details", page: "details.html", defaultName: "Details"},
}

// pageRequest is one parsed page-family ask.
type pageRequest struct {
	kind          string // a pageKindTable key
	name          string // the handler's name as typed ("" = the kind's default)
	target        string // the Go target as typed ("" = no target clause)
	targetMissing bool   // a target clause was started but named nothing
	nameMissing   bool   // a "named" clause was started but named nothing
}

// info returns the request kind's table row.
func (q pageRequest) info() pageKindInfo { return pageKindTable[q.kind] }

// handlerName is the name the plan will use: the typed name,
// else the kind's default.
func (q pageRequest) handlerName() string {
	if q.name != "" {
		return q.name
	}
	return q.info().defaultName
}

// parsePageFamily reads one chat line as a page-family ask.
// The second return is false for anything outside the grammar
// — including every neighbour's sentence (see the file
// header). A line that matches the grammar's opening and then
// stops inside a tail clause (a trailing "named", a trailing
// "in") returns its missing flag set: claimed, so the flow
// can answer in page terms.
func parsePageFamily(line string) (pageRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return pageRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	i := 0
	switch low[0] {
	case "create", "add", "make", "build":
		i = 1
	default:
		return pageRequest{}, false
	}
	if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	// The kind phrase: exactly one, at this position.
	req, ni, ok := pageKindAt(fields, low, i)
	if !ok {
		return pageRequest{}, false
	}
	i = ni
	// The tail: a named segment and a target segment, in any
	// order, each at most once. Anything else trailing is
	// another ask glued behind this one — not this claim's.
	seenName := req.name != ""
	seenTarget := false
	for i < len(fields) {
		switch {
		case low[i] == "named":
			if seenName {
				return pageRequest{}, false
			}
			seenName = true
			i++
			if i >= len(fields) {
				req.nameMissing = true
				return req, true
			}
			// A target preposition where the name should be:
			// the named clause named nothing, and a target
			// clause follows it — still this claim's, answered
			// in page terms.
			if low[i] == "to" || low[i] == "in" || low[i] == "into" {
				req.nameMissing = true
				return req, true
			}
			if !isWireNameToken(fields[i], low[i]) {
				return pageRequest{}, false
			}
			req.name = fields[i]
			i++
		case low[i] == "to" || low[i] == "in" || low[i] == "into":
			if seenTarget {
				return pageRequest{}, false
			}
			seenTarget = true
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
				// A target clause was started and named
				// nothing ("create a homepage in"): claimed,
				// so the flow can ask in page terms.
				req.targetMissing = true
				return req, true
			}
			if low[i] == "named" {
				// A named clause where the target should be:
				// the target clause named nothing.
				req.targetMissing = true
				return req, true
			}
			req.target = fields[i]
			i++
		default:
			return pageRequest{}, false
		}
	}
	return req, true
}

// pageKindAt reads the kind phrase at position i. The handler
// phrases are form's alone and are read whole: "handler" by
// itself, or "handler named <Name>" without its "with a
// form", is the book's handler territory, never this claim's.
func pageKindAt(fields, low []string, i int) (pageRequest, int, bool) {
	at := func(words ...string) bool {
		if i+len(words) > len(fields) {
			return false
		}
		for j, w := range words {
			if low[i+j] != w {
				return false
			}
		}
		return true
	}
	switch {
	// form: "handler named <Name> with a form"
	case i+6 <= len(fields) && low[i] == "handler" && low[i+1] == "named" &&
		low[i+3] == "with" && low[i+4] == "a" && low[i+5] == "form":
		if !isWireNameToken(fields[i+2], low[i+2]) {
			return pageRequest{}, i, false
		}
		return pageRequest{kind: "form", name: fields[i+2]}, i + 6, true
	// form: "handler with a form"
	case at("handler", "with", "a", "form"):
		return pageRequest{kind: "form"}, i + 4, true
	// form: "form page"
	case at("form", "page"):
		return pageRequest{kind: "form"}, i + 2, true
	// home: "homepage" | "home page"
	case at("homepage"):
		return pageRequest{kind: "home"}, i + 1, true
	case at("home", "page"):
		return pageRequest{kind: "home"}, i + 2, true
	// about: "about us page" | "about page"
	case at("about", "us", "page"):
		return pageRequest{kind: "about"}, i + 3, true
	case at("about", "page"):
		return pageRequest{kind: "about"}, i + 2, true
	// contact: "contact us page" | "contact page"
	case at("contact", "us", "page"):
		return pageRequest{kind: "contact"}, i + 3, true
	case at("contact", "page"):
		return pageRequest{kind: "contact"}, i + 2, true
	// list: "item list page" | "list page"
	case at("item", "list", "page"):
		return pageRequest{kind: "list"}, i + 3, true
	case at("list", "page"):
		return pageRequest{kind: "list"}, i + 2, true
	// details: "item details page" | "details page"
	case at("item", "details", "page"):
		return pageRequest{kind: "details"}, i + 3, true
	case at("details", "page"):
		return pageRequest{kind: "details"}, i + 2, true
	}
	return pageRequest{}, i, false
}

// pageTargets is the resolved page ask: the handler's host Go
// file (created by the plan when goExists is false) and the
// page, both project-relative slash paths.
type pageTargets struct {
	goRel    string
	goExists bool
	htmlRel  string
}

// resolvePageTargets settles the ask's two files inside
// rootAbs, or returns an honest refusal naming the working
// alternative. The Go side is resolveServeGoFile's folder rule
// with this family's own words and one move of its own: a
// folder holding only library Go files takes the handler's
// own new file, never a package-main main.go that would clash
// with the library at build time. The page lands in the host
// file's folder.
func resolvePageTargets(rootAbs string, q pageRequest) (pageTargets, string) {
	info := q.info()
	var goRel string
	var goExists bool
	if q.target != "" {
		if strings.HasSuffix(strings.ToLower(q.target), ".html") {
			return pageTargets{}, fmt.Sprintf("%s is a page, not a Go file — the page's handler goes in a .go file and the page lands beside it. Say \"%s to main.go\"", q.target, info.canonical)
		}
		abs, rel, refusal := serverTargetResolve(rootAbs, q.target)
		if refusal != "" {
			return pageTargets{}, refusal
		}
		var r string
		goRel, goExists, r = resolvePageGoFile(rootAbs, abs, rel, q.target, q)
		if r != "" {
			return pageTargets{}, r
		}
	} else {
		var r string
		goRel, goExists, r = resolvePageGoFile(rootAbs, rootAbs, "", "", q)
		if r != "" {
			return pageTargets{}, r
		}
	}
	goDir := path.Dir(goRel)
	return pageTargets{
		goRel:    goRel,
		goExists: goExists,
		htmlRel:  path.Join(goDir, info.page),
	}, ""
}

// resolvePageGoFile settles one Go target — a folder (disp ==
// "" means the project root itself) or a file, existing or to
// be created — to the project-relative .go file the handler
// goes into. The folder rule is the serve-html one (main.go,
// else the single package-main file, ambiguity refused), with
// the library folder settled on the handler's own file.
func resolvePageGoFile(rootAbs, abs, rel, disp string, q pageRequest) (goRel string, goExists bool, refusal string) {
	info := q.info()
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
			// No main program in the folder. A folder with
			// no Go files yet takes a created main.go; a
			// folder whose Go files are a library package
			// takes the handler's own file instead — a
			// package-main file would clash with them.
			if folderHasGoFiles(abs) {
				name := strings.ToLower(q.handlerName()) + ".go"
				if fi, err := os.Stat(filepath.Join(abs, name)); err == nil && !fi.IsDir() {
					return joinRel(rel, name), true, ""
				}
				return joinRel(rel, name), false, ""
			}
			return joinRel(rel, "main.go"), false, ""
		default:
			return "", false, fmt.Sprintf("%s has more than one main-package file (%s) — say which one, like \"%s to %s%s\"", folderDisp, strings.Join(cands, ", "), info.canonical, relPrefix(rel), cands[0])
		}
	}
	if err == nil {
		// An existing file that is not a folder.
		if !strings.HasSuffix(strings.ToLower(disp), ".go") {
			return "", false, fmt.Sprintf("%s isn't a Go file — a page's handler is a Go function, so it goes in a .go file. Say \"%s to main.go\"", disp, info.canonical)
		}
		return rel, true, ""
	}
	// A missing target: the plan creates the host file, when
	// its folder exists.
	if !strings.HasSuffix(strings.ToLower(disp), ".go") {
		return "", false, fmt.Sprintf("%s isn't a Go file — name a .go file for the handler, like \"%s to main.go\"", disp, info.canonical)
	}
	if _, err := os.Stat(filepath.Dir(abs)); err != nil {
		folder := filepath.ToSlash(mustRel(rootAbs, filepath.Dir(abs)))
		return "", false, fmt.Sprintf("there's no folder %s in this project — say \"create a folder %s\" first, then ask again", folder, folder)
	}
	return rel, false, ""
}

// tryPageFamily is the chat-loop entry: it claims page-family
// sentences and runs the flow on the chat's line reader.
func tryPageFamily(line string, r *LineReader) bool {
	return runPageFamily(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runPageFamily carries one line from sentence to resolved
// targets to the shared edit spine. It returns true when the
// line belonged to this claim at all.
func runPageFamily(line string, ask editAsk) bool {
	q, ok := parsePageFamily(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	info := q.info()
	if q.targetMissing {
		// The grammar matched and started a target clause but
		// named nothing ("create a homepage in"): claimed, and
		// answered in page terms — this sentence must never
		// reach the intent layer.
		fmt.Printf("%s Not making that page yet: the sentence starts naming the file or folder but doesn't finish — say it like \"%s in jim\" (a folder) or \"%s to jim/main.go\" (a file), or leave the target off and I'll use the project's main file.\n", tag, info.canonical, info.canonical)
		recordBrain("composer", false, "page refused: target unnamed")
		return true
	}
	if q.nameMissing {
		// The grammar matched and started a "named" clause but
		// gave no name ("create a form page named"): claimed,
		// and answered in page terms.
		fmt.Printf("%s Not making that page yet: the sentence starts naming the handler with \"named\" but doesn't give the name — say it like \"%s named Signup\", or leave \"named\" off and I'll use the default name.\n", tag, info.canonical)
		recordBrain("composer", false, "page refused: name unnamed")
		return true
	}
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "page refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "page refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "page refused: root missing")
		return true
	}
	t, refusal := resolvePageTargets(rootAbs, q)
	if refusal != "" {
		fmt.Printf("%s Not making that page: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "page refused: target")
		return true
	}
	// The spine plans against the analyzed project; make the
	// target's root the current one, quietly (the serve-html
	// flow's reasoning). One state is this family's own, as it
	// is serve-html's: a project with no Go packages yet
	// cannot be analyzed — and does not need to be, when the
	// plan CREATES the host file: an empty index rooted there
	// is enough for the planner, and the spine re-analyzes
	// for real after the write, before verifying.
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		p, err := analyze.Analyze(rootAbs)
		switch {
		case err == nil:
			setCurrentProject(rootRaw, p)
		case !t.goExists && strings.Contains(err.Error(), "no Go packages found"):
			setCurrentProject(rootRaw, &analyze.Project{Root: rootAbs})
		default:
			fmt.Printf("I couldn't analyze %s: %v\n", rootAbs, err)
			recordBrain("composer", false, "page refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "page refused: stale project unreadable")
		return true
	}
	// The form kinds' handler is the book's form handler: the
	// entry rides the request so the planner assembles it from
	// the same piece the book would place alone, its fields
	// matched to the page's. No entry in this build, no plan —
	// said plainly, in this family's words.
	var entry *analyze.ConceptEntry
	if info.needsBook {
		for _, e := range conceptEntries() {
			if e.ID == "form-handler" {
				found := e
				entry = &found
				break
			}
		}
		if entry == nil {
			fmt.Printf("%s Not making that page: the concept book has no form handler piece in this build, so I can't assemble the %s's handler.\n", tag, info.display)
			recordBrain("composer", false, "page refused: no form handler piece")
			return true
		}
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:      analyze.EditPage,
		File:      t.goRel,
		PageKind:  q.kind,
		PageHTML:  t.htmlRel,
		PageName:  q.handlerName(),
		PageRoute: info.route,
		Concept:   entry,
	}, ask)
}
