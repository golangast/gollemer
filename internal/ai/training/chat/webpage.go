package chat

// "Add a webpage" — a plain webpage, from a template the
// user picks, written into one named .html file
// (analyze.EditWebpage). Claimed deterministically in the
// unified loop beside the page family, ahead of the phrasing
// layer. It exists for the live-session ask no piece owned
// (2026-11-09): the user had created jim/temp/index.html —
// empty — through the chat, and "add a webpage to
// jim/temp/index.html" could only be refused, because the
// page family builds page+handler pairs by kind and nothing
// assembled a webpage into a file.
//
// Grammar (exactly this):
//
//	add|create|make|build|put [a|an|the] [new]
//	    [basic|simple|responsive] webpage|web page
//	    [with assets]
//	    [to|in|into|at [the] [file|folder] <file.html>]
//
// The template slot: "basic" (also "simple"), "responsive",
// or the trailing "with assets" — the three templates in
// analyze.WebpageContent. No template named → ONE
// clarify.go question listing them; the answer completes
// the original ask. No file named (or a target clause
// started and left empty) → ONE clarify.go question asking
// which file; the answer completes the ask the same way.
//
// The boundary with the page family is the object word:
// this grammar fires only on "webpage"/"web page" with no
// kind word in front of it, and the page family sits earlier
// in the loop — "create a homepage", "create a form page",
// "create an about page" keep their family's claim, pinned
// in webpage_test.go both ways.
//
// File rules (planner-enforced, checked in the flow before
// any question is asked): a new file is created; an existing
// EMPTY .html is filled; an existing .html with anything in
// it is never overwritten — a plain refusal with the way
// back. A target that isn't .html refuses by name here.

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// webpageRequest is one parsed webpage ask.
type webpageRequest struct {
	template    string // "basic" | "responsive" | "assets"; "" = unnamed (the flow asks)
	file        string // the page file as typed; "" when fileMissing
	fileMissing bool   // no file named — claimed, the flow asks which file
}

// parseWebpage reads one chat line as a webpage ask. The
// second return is false for anything outside the grammar —
// including the page family's kind sentences (see the file
// header).
func parseWebpage(line string) (webpageRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return webpageRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return webpageRequest{}, false
	}
	i := 1
	if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	if i < len(fields) && low[i] == "new" {
		i++
	}
	// The template adjective rides in front of the object
	// ("a responsive webpage"); the assets template is the
	// trailing clause below.
	req := webpageRequest{}
	if i < len(fields) {
		switch low[i] {
		case "basic", "simple":
			req.template = "basic"
			i++
		case "responsive":
			req.template = "responsive"
			i++
		}
	}
	// The object: "webpage", or "web page". A kind word in
	// front of "page" (home, form, about…) is the page
	// family's sentence and never reaches this switch.
	switch {
	case i < len(fields) && low[i] == "webpage":
		i++
	case i+1 < len(fields) && low[i] == "web" && low[i+1] == "page":
		i += 2
	default:
		return webpageRequest{}, false
	}
	if i+1 < len(fields) && low[i] == "with" && low[i+1] == "assets" {
		if req.template != "" {
			// Two templates named in one sentence: not a
			// sentence this grammar can mean one thing by.
			return webpageRequest{}, false
		}
		req.template = "assets"
		i += 2
	}
	if i == len(fields) {
		// No target clause at all: claimed — the flow asks
		// which file, in webpage terms (the add-server
		// ruling: an ask about making a thing never falls
		// through to the intent layer to be misread).
		req.fileMissing = true
		return req, true
	}
	// The target clause: one preposition, the designator
	// slots the neighbours read, then ONE token.
	if low[i] != "to" && low[i] != "in" && low[i] != "into" && low[i] != "at" {
		return webpageRequest{}, false
	}
	i++
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) && (low[i] == "file" || low[i] == "folder" || low[i] == "directory") {
		i++
	}
	if i >= len(fields) {
		// A target clause was started and named nothing
		// ("add a webpage to the file"): claimed, asked.
		req.fileMissing = true
		return req, true
	}
	if len(fields[i:]) != 1 {
		return webpageRequest{}, false
	}
	req.file = fields[i]
	return req, true
}

// webpageCanonical renders the ask's canonical sentence for
// a template and file — the sentence a clarifying answer
// completes into. It must parse back through parseWebpage
// with exactly these slots (pinned in webpage_test.go).
func webpageCanonical(template, file string) string {
	switch template {
	case "assets":
		return "add a webpage with assets to " + file
	case "":
		return "add a webpage to " + file
	default:
		return "add a " + template + " webpage to " + file
	}
}

// webpageTemplateCandidates is the clarify.go choice for an
// unnamed template: the three templates, in the words the
// plan and the page itself use.
func webpageTemplateCandidates() []clarifyCandidate {
	return []clarifyCandidate{
		{Label: "basic", Desc: "a title, a heading, and a paragraph — the smallest complete page", Words: []string{"basic", "simple"}},
		{Label: "responsive", Desc: "fits phones and wide screens too — a viewport line and a little CSS", Words: []string{"responsive"}},
		{Label: "with assets", Desc: "also links a style sheet, a script, and a picture from the /static/ folder", Words: []string{"assets"}},
	}
}

// webpageTemplateIDs maps a clarify candidate index back to
// the template id, in candidate order.
var webpageTemplateIDs = []string{"basic", "responsive", "assets"}

// tryWebpage is the chat-loop entry: it claims webpage
// sentences and runs the flow on the chat's line reader.
func tryWebpage(line string, r *LineReader) bool {
	return runWebpage(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runWebpage carries one line from sentence to resolved
// file to the shared edit spine. It returns true when the
// line belonged to this claim at all.
func runWebpage(line string, ask editAsk) bool {
	q, ok := parseWebpage(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.fileMissing {
		// Claimed and asked in webpage terms (clarify.go):
		// the answer, one bare .html path, completes the ask
		// as its canonical sentence — with the template the
		// sentence named, when it named one.
		template := q.template
		printClarifyIntro(tag, "Not making that webpage yet: the sentence doesn't name the file — which file should the webpage go in?")
		printClarifyFooter("Answer with the .html file — \"index.html\" or \"jim/temp/index.html\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				return runWebpage(webpageCanonical(template, text), ask2)
			},
		})
		recordBrain("composer", false, "webpage question: file unnamed")
		return true
	}
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "webpage")
	if !ok {
		return true
	}
	abs, rel, refusal := fenceResolve(rootAbs, q.file)
	if refusal != "" {
		fmt.Printf("%s Not making that webpage: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "webpage refused: file outside project")
		return true
	}
	if !strings.HasSuffix(strings.ToLower(rel), ".html") {
		fmt.Printf("%s Not making that webpage: %s isn't an .html file — a webpage is an HTML page, so the file's name has to end in .html. Name an .html file and ask again.\n", tag, rel)
		recordBrain("composer", false, "webpage refused: not an html file")
		return true
	}
	if st, err := os.Stat(filepath.Dir(abs)); err != nil || !st.IsDir() {
		dirRel := path.Dir(rel)
		fmt.Printf("%s Not making that webpage: there's no folder %s in this project for the webpage to live in — say \"create a folder %s\" first, then ask again.\n", tag, dirRel, dirRel)
		recordBrain("composer", false, "webpage refused: folder missing")
		return true
	}
	// Facts before questions: a file that already has a page
	// in it refuses outright, and that is checkable now — the
	// template question is never asked for an ask that
	// cannot land.
	if b, err := os.ReadFile(abs); err == nil && len(bytes.TrimSpace(b)) > 0 {
		fmt.Printf("%s Not making that webpage: %s already has a page in it — I never overwrite what's yours. If you want a fresh template page there instead, empty the file yourself (or remove it) and ask again.\n", tag, rel)
		recordBrain("composer", false, "webpage refused: file not empty")
		return true
	}
	if q.template == "" {
		askPickQuestion(tag,
			fmt.Sprintf("I can put a webpage in %s — which template should it be?", rel),
			"Answer with the number or the name",
			webpageTemplateCandidates(),
			func(idx int, ask2 editAsk) bool {
				return runWebpage(webpageCanonical(webpageTemplateIDs[idx], rel), ask2)
			})
		recordBrain("composer", false, "webpage question: template unnamed")
		return true
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "webpage") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:            analyze.EditWebpage,
		File:            rel,
		WebpageTemplate: q.template,
	}, ask)
}
