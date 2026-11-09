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
//	    [basic|simple|responsive|two column(s)|
//	     three column(s)|grid|rows|material]
//	    webpage|web page
//	    [with assets]
//	    [to|in|into|at [the] [file|folder] <file.html>]
//
// The same ask in the word the templates themselves use.
// Later the same day John typed "add the template to
// jim/temp/index.html" and — this piece not yet shipped —
// the concept book's html/template piece took the line and
// refused it: an html template is a Go statement, and his
// file is not a Go file. He meant one of these page
// templates, so the template wording is this grammar too:
//
//	add|create|make|build|put [a|an|the] [new]
//	    [basic|simple|responsive|two column(s)|
//	     three column(s)|grid|rows|material|
//	     with assets] template
//	    [with assets]
//	    [to|in|into|at [the] [file|folder] <file.html>]
//	use [a|an|the] [basic|simple|responsive|two column(s)|
//	     three column(s)|grid|rows|material|
//	     with assets] template
//	    [for|to|in|into|at [the] [file|folder] <file.html>]
//
// The seating rule for the template wording: it claims
// only for an .html target — or for no target yet, when
// the flow asks which file, exactly as the webpage wording
// does. A .go target is never claimed here: "add the
// template to main.go" stays the concept book's
// html/template sentence, untouched.
//
// The grid takes exact counts (John, later still: "add
// template 2 columns 3 rows to jim/temp/index.html" — a
// grid of exactly that many boxes):
//
//	add a grid template with 2 columns [and 3 rows] to …
//	add template 2 columns 3 rows to …
//	add a 2 by 3 grid template to …
//	add the grid 2 columns 3 rows to …
//
// The last form names the grid as the object itself, the
// counts directly behind it — John's composition of the
// two counted shapes ("add the grid 2 columns 3 rows to
// jim/templates/home.html"). Aimed at a templated page's
// body file, the counts land as a section in that body
// instead (see the flow): the target decides which of the
// two counted pieces the sentence is.
//
// Numbers are digits or words, singular or plural. The
// counts clause trails the object, led by "with" or bare;
// the bare pair after a "template" object names the grid
// itself, so no grid word is needed there — but a lone
// column count names nothing ("add template 2 columns
// to …" keeps declining: the two-columns layout owns its
// wording, in front of the object). Columns only → the
// rows default to 2, and the plan states both counts.
// The range is 1 to 6 of each; outside it the ask claims
// and the flow refuses in this piece's own words.
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
//
// One refusal is an explanation instead (John, later
// still): "add the material template to
// jim/templates/home.html" aimed a whole template at a
// templated page's BODY file, three times in a row, and
// the generic refusal sent him in a circle — a template
// can never land in one part of a page, and the page's
// design already lives in its header. So a non-empty
// target is read before it is refused: a scaffold part
// (scaffoldPartKind — a body by its header and footer
// calls, a header or footer by its definition) gets the
// templated-page answer — what the file is, where the
// design lives, and the section sentence verified to land
// on the body — while a plain filled page keeps the
// refusal, byte-identical.

import (
	"bytes"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// webpageRequest is one parsed webpage ask.
type webpageRequest struct {
	template    string // "basic" | "responsive" | "assets"; "" = unnamed (the flow asks)
	file        string // the page file as typed; "" when fileMissing
	fileMissing bool   // no file named — claimed, the flow asks which file

	// The counted grid's counts, when the sentence named
	// them (template is "grid" then). gridRows is already
	// defaulted at parse: a columns-only clause means 2.
	gridCounted bool
	gridCols    int
	gridRows    int
}

// smallNumberWords are the count words the grid counts
// read. They run past the 1–6 range on purpose: an
// out-of-range ask in words ("seven columns") still parses
// as a count, so the flow can refuse it by name instead of
// the line falling through as some other sentence.
var smallNumberWords = map[string]int{
	"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
	"five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
	"ten": 10, "eleven": 11, "twelve": 12,
}

// parseCountWord reads one count token: a word from
// smallNumberWords, or a run of digits.
func parseCountWord(tok string) (int, bool) {
	if n, ok := smallNumberWords[tok]; ok {
		return n, true
	}
	if n, err := strconv.Atoi(tok); err == nil && n >= 0 {
		return n, true
	}
	return 0, false
}

// countAndNoun reads "<count> column(s)" or "<count>
// row(s)" at low[i] — the noun in either number, whatever
// the count. It returns the count and the words consumed,
// 0 when that pair does not stand there.
func countAndNoun(low []string, i int, noun string) (int, int) {
	if i+1 >= len(low) {
		return 0, 0
	}
	v, ok := parseCountWord(low[i])
	if !ok || (low[i+1] != noun && low[i+1] != noun+"s") {
		return 0, 0
	}
	return v, 2
}

// parseGridCounts reads a grid counts clause at low[i]:
// "[with] <n> column(s) [and] [<m> row(s)]". The columns
// are required; the rows may be absent (rowsSet false —
// the caller defaults them). It returns the counts and the
// words consumed, 0 when no clause stands there — a clause
// that starts and breaks (a count with no noun, a dangling
// "and" before the target) is no clause, and consumes
// nothing.
func parseGridCounts(low []string, i int) (cols, rows int, rowsSet bool, n int) {
	j := i
	if j < len(low) && low[j] == "with" {
		j++
	}
	c, w := countAndNoun(low, j, "column")
	if w == 0 {
		return 0, 0, false, 0
	}
	cols, j = c, j+w
	k := j
	if k < len(low) && low[k] == "and" {
		k++
	}
	if r, w2 := countAndNoun(low, k, "row"); w2 > 0 {
		rows, rowsSet, j = r, true, k+w2
	}
	return cols, rows, rowsSet, j - i
}

// matchWebpageTemplateName reads a template name at low[i]:
// the single words basic/simple/responsive/grid/rows, and
// the column names in their spaced and hyphenated, singular
// and plural forms ("two column", "two-columns", …). It
// returns the template id and the words consumed — 0 when
// no name stands there. The assets template is not matched
// here: its words ("with assets") ride in their own slots.
func matchWebpageTemplateName(low []string, i int) (string, int) {
	if i >= len(low) {
		return "", 0
	}
	switch low[i] {
	case "basic", "simple":
		return "basic", 1
	case "responsive":
		return "responsive", 1
	case "grid":
		return "grid", 1
	case "rows":
		return "rows", 1
	case "material":
		return "material", 1
	case "two-column", "two-columns":
		return "two-columns", 1
	case "three-column", "three-columns":
		return "three-columns", 1
	}
	if i+1 < len(low) && (low[i+1] == "column" || low[i+1] == "columns") {
		switch low[i] {
		case "two":
			return "two-columns", 2
		case "three":
			return "three-columns", 2
		}
	}
	return "", 0
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
	useVerb := low[0] == "use"
	switch low[0] {
	case "add", "create", "make", "build", "put", "use":
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
	// The counted grid's "2 by 3" rides in front of the
	// grid name ("add a 2 by 3 grid template to …"): the
	// pair, then the name it sizes. A pair in front of any
	// other name — or no name — is not this grammar's.
	req := webpageRequest{}
	byForm := false
	if i+2 < len(low) && low[i+1] == "by" {
		if c, ok1 := parseCountWord(low[i]); ok1 {
			if r, ok2 := parseCountWord(low[i+2]); ok2 {
				req.gridCounted, req.gridCols, req.gridRows = true, c, r
				byForm = true
				i += 3
			}
		}
	}
	// The template name rides in front of the object ("a
	// responsive webpage", "a two column template"); the
	// assets template's words are the clauses around it.
	if name, n := matchWebpageTemplateName(low, i); n > 0 {
		req.template = name
		i += n
	}
	if byForm && req.template != "grid" {
		return webpageRequest{}, false
	}
	// The grid as its own object ("add the grid 2 columns
	// 3 rows to …"): the name is followed directly by the
	// counts clause — no object word stands between them.
	// The "use" wording never takes this form (its rule
	// below stands: "use" belongs to the template wording).
	objectless := false
	if !useVerb && !byForm && req.template == "grid" {
		if cols, rows, rowsSet, cn := parseGridCounts(low, i); cn > 0 {
			req.gridCounted, req.gridCols, req.gridRows = true, cols, rows
			if !rowsSet {
				req.gridRows = 2
			}
			objectless = true
			i += cn
		}
	}
	// "add the with assets template…": the assets words can
	// stand in front of a "template" object, not only trail
	// the object as they do for "webpage".
	if req.template == "" && i+2 < len(fields) && low[i] == "with" && low[i+1] == "assets" && low[i+2] == "template" {
		req.template = "assets"
		i += 2
	}
	// The object: "webpage", "web page", or "template". A
	// kind word in front of "page" (home, form, about…) is
	// the page family's sentence and never reaches this
	// switch.
	templateWord := false
	switch {
	case objectless:
		// The grid named itself the object; no object
		// word follows its counts.
	case i < len(fields) && low[i] == "webpage":
		i++
	case i+1 < len(fields) && low[i] == "web" && low[i+1] == "page":
		i += 2
	case i < len(fields) && low[i] == "template":
		templateWord = true
		i++
	default:
		return webpageRequest{}, false
	}
	if useVerb && !templateWord {
		// "use" is this grammar's verb only for the
		// template wording ("use the grid template for
		// index.html") — "use a webpage" is not a
		// sentence this piece owns.
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
	// The grid counts clause trails the object: led by
	// "with" on a grid ask ("add a grid template with 2
	// columns and 3 rows to …"), or — John's live form —
	// the bare pair after a "template" object, where the
	// pair itself names the grid ("add template 2 columns
	// 3 rows to …"). A clause anywhere else means nothing
	// single here: counts on another template, a lone
	// column count with no grid named, a second clause
	// after the "by" form — all decline.
	if cols, rows, rowsSet, cn := parseGridCounts(low, i); cn > 0 {
		switch {
		case req.gridCounted:
			return webpageRequest{}, false
		case req.template == "grid":
			req.gridCounted, req.gridCols, req.gridRows = true, cols, rows
			if !rowsSet {
				req.gridRows = 2
			}
			i += cn
		case req.template == "" && templateWord && rowsSet:
			req.template = "grid"
			req.gridCounted, req.gridCols, req.gridRows = true, cols, rows
			i += cn
		default:
			return webpageRequest{}, false
		}
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
	// slots the neighbours read, then ONE token. The "use"
	// wording takes "for" as its preposition.
	if low[i] != "to" && low[i] != "in" && low[i] != "into" && low[i] != "at" && !(useVerb && low[i] == "for") {
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
	if templateWord && !strings.HasSuffix(strings.ToLower(req.file), ".html") {
		// The template wording claims for .html targets
		// only: with a .go target the line is the concept
		// book's html/template sentence — a Go statement
		// going into a function — and keeps that claim.
		return webpageRequest{}, false
	}
	if objectless && !strings.HasSuffix(strings.ToLower(req.file), ".html") {
		// The grid-object form seats like the template
		// wording it composes with: counted-grid sentences
		// claim for .html targets only — the bare pair's
		// main.go decline ("add template 2 columns to
		// main.go") is this form's too.
		return webpageRequest{}, false
	}
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

// webpageCanonicalCounted renders a counted grid ask's
// canonical sentence — the with-form, which parses back
// with exactly these counts (pinned in webpage_test.go).
func webpageCanonicalCounted(cols, rows int, file string) string {
	return fmt.Sprintf("add a grid template with %d columns and %d rows to %s", cols, rows, file)
}

// webpageTemplateCandidates is the clarify.go choice for an
// unnamed template: the eight templates, in the words the
// plan and the page itself use.
func webpageTemplateCandidates() []clarifyCandidate {
	return []clarifyCandidate{
		{Label: "basic", Desc: "a title, a heading, and a paragraph — the smallest complete page", Words: []string{"basic", "simple"}},
		{Label: "responsive", Desc: "fits phones and wide screens too — a viewport line and a little CSS", Words: []string{"responsive"}},
		{Label: "with assets", Desc: "also links a style sheet, a script, and a picture from the /static/ folder", Words: []string{"assets"}},
		{Label: "two columns", Desc: "a sidebar beside the main area, under a header", Words: []string{"two columns", "two column", "two-column", "two-columns"}},
		{Label: "three columns", Desc: "a sidebar on each side of the main area, under a header", Words: []string{"three columns", "three column", "three-column", "three-columns"}},
		{Label: "grid", Desc: "a heading and a grid of cards — three across on a wide screen, fewer on a small one", Words: []string{"grid"}},
		{Label: "rows", Desc: "full-width bands stacked down the page in different shades", Words: []string{"rows"}},
		{Label: "material", Desc: "Google's Material Design look, plain CSS, no framework", Words: []string{"material"}},
	}
}

// webpageTemplateIDs maps a clarify candidate index back to
// the template id, in candidate order.
var webpageTemplateIDs = []string{"basic", "responsive", "assets", "two-columns", "three-columns", "grid", "rows", "material"}

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
	if q.gridCounted && (q.gridCols < 1 || q.gridCols > 6 || q.gridRows < 1 || q.gridRows > 6) {
		// The sentence names an impossible grid: refuse in
		// this piece's words, naming what is out of range —
		// never a silent clamp, and before any question.
		var out []string
		if q.gridCols < 1 || q.gridCols > 6 {
			out = append(out, fmt.Sprintf("%d columns", q.gridCols))
		}
		if q.gridRows < 1 || q.gridRows > 6 {
			out = append(out, fmt.Sprintf("%d rows", q.gridRows))
		}
		fmt.Printf("%s Not making that webpage: a counted grid can have 1 to 6 columns and 1 to 6 rows — %s is outside that. Ask again with numbers in that range.\n", tag, strings.Join(out, " and "))
		recordBrain("composer", false, "webpage refused: grid counts out of range")
		return true
	}
	if q.fileMissing {
		// Claimed and asked in webpage terms (clarify.go):
		// the answer, one bare .html path, completes the ask
		// as its canonical sentence — with the template the
		// sentence named, when it named one, and a counted
		// grid's counts riding along.
		template := q.template
		gridCounted, gridCols, gridRows := q.gridCounted, q.gridCols, q.gridRows
		printClarifyIntro(tag, "Not making that webpage yet: the sentence doesn't name the file — which file should the webpage go in?")
		printClarifyFooter("Answer with the .html file — \"index.html\" or \"jim/temp/index.html\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				if gridCounted {
					return runWebpage(webpageCanonicalCounted(gridCols, gridRows, text), ask2)
				}
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
	// cannot land. One exception, by what the file IS: a
	// counted grid aimed at a templated page's body file
	// (its header and footer calls give it away) is a body
	// section ask in grid words — the counts land as a
	// section through the body section's own planner, and
	// the body is never replaced by a whole page.
	if b, err := os.ReadFile(abs); err == nil && len(bytes.TrimSpace(b)) > 0 {
		if q.gridCounted && isScaffoldBody(b) {
			return runScaffoldSectionPlan(tag, rootRaw, rootAbs, path.Dir(path.Dir(rel)), rel, q.gridCols, q.gridRows, ask)
		}
		if part := scaffoldPartKind(b); part != "" {
			// A whole template aimed at one part of a
			// templated page: the explanation, not the
			// refusal — the refusal's way back (empty the
			// file) would destroy the page to no purpose.
			printScaffoldPartAnswer(tag, rootAbs, part, rel)
			recordBrain("composer", false, "webpage guidance: template ask on scaffold "+part)
			return true
		}
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
		WebpageGridCols: q.gridCols,
		WebpageGridRows: q.gridRows,
	}, ask)
}

// printScaffoldPartAnswer answers a webpage-template ask
// whose target is one part of a templated page. A template
// is a whole page, so it can never land in a part — the
// answer says what the file is, where the page's design
// lives (the header, whose styles already cover the whole
// page), and what goes in the body instead. One shape for
// every template kind: the reading of the file, not the
// template named, decides the answer. Nothing is written.
func printScaffoldPartAnswer(tag, rootAbs, part, rel string) {
	dir := path.Dir(rel)
	folder := path.Dir(dir)
	where := "in " + folder
	if folder == "." {
		where = "at the top of this project"
	}
	header := path.Join(dir, "header.html")
	switch part {
	case "body":
		page := strings.TrimSuffix(path.Base(rel), ".html")
		fmt.Printf("%s Not making that webpage: %s is the body of the %s templated page %s — one of the page's three files, not a page on its own. The page's design, its Material styles, lives in the header file, %s, and it already covers the whole page, so there's no template to add to the body. What the body takes is sections: \"add the grid 2 columns 3 rows to %s\" adds a section of boxes to it.\n", tag, rel, page, where, header, rel)
	case "header":
		fmt.Printf("%s Not making that webpage: %s is the shared header of the templated pages %s — the file that holds the page's design, its Material styles, for every page that calls it. A template is a whole page, so there's nothing for one to add here. %s\n", tag, rel, where, scaffoldBodyGuidance(rootAbs, folder))
	case "footer":
		fmt.Printf("%s Not making that webpage: %s is the shared footer of the templated pages %s — the closing piece at the foot of every page that calls it. The page's design, its Material styles, lives in the header file, %s, not in the footer. %s\n", tag, rel, where, header, scaffoldBodyGuidance(rootAbs, folder))
	}
}

// scaffoldBodyGuidance names where a templated page's
// sections go: the folder's body file, with the counted
// section sentence spelled against it — the one sentence
// this answer names, walked through the real dispatch in
// webpage_test.go and asserted to land there (the
// capability seat's drift rule: an answer may never
// promise a line that doesn't work — the component
// sentence, which the component planner refuses on a body
// fragment for want of a </body>, is deliberately NOT
// named). With no body to point at, it names the place
// without promising a sentence.
func scaffoldBodyGuidance(rootAbs, folder string) string {
	bodies := scaffoldBodies(rootAbs, folder)
	if len(bodies) == 0 {
		return "Sections go in a page's own body file instead — the file in the same templates folder that calls the header at its top and the footer at its foot."
	}
	body := bodies[0]
	page := strings.TrimSuffix(path.Base(body), ".html")
	return fmt.Sprintf("Sections go in a page's body instead: \"add the grid 2 columns 3 rows to %s\" adds a section of boxes to the %s page's body.", body, page)
}
