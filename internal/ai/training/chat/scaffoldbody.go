package chat

// Columns and rows into a templated page's body (John
// 2026-11-09):
//
//	add <n> columns and <m> rows to the body in <folder>
//	add columns and rows to the body in <folder>
//	add the grid <n> columns and <m> rows to the body in <folder>
//
// The third shape is the counted grid's own word standing
// in front of the counts (or the bare pair) — the same
// section, named the way the grid taught it.
//
// The section — a counted grid of boxes, the counted grid's
// own discipline: digits or words, 1–6 of each, counts
// stated in the plan — lands in the scaffold's BODY file,
// below what the body already holds and above its footer
// call, with its styles joining the header's style block
// once (analyze.planScaffoldSection). "The body" resolves
// to the scaffold in the named folder: no scaffold there is
// the honest two-step (make the templated page first); a
// page can be named — "to the body of home in jim" — and
// when a folder holds several body files and the sentence
// names none, the clarify machinery asks which page, one
// question.
//
// Two readings are settled here and pinned: columns alone
// ("add 3 columns to the body in jim") means rows of 2, the
// counted grid's own default; and no counts at all ("add
// columns and rows to the body in jim") means 2 columns and
// 2 rows — the smallest useful section — with the counts
// stated in the plan like any other, so the gate shows
// exactly what will land.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// scaffoldSectionAsk is one parsed body-section sentence:
// the counts (always settled by the parser — the defaults
// above are applied there), whether the sentence named
// them, an optional page name, and the folder as typed.
type scaffoldSectionAsk struct {
	cols    int
	rows    int
	counted bool
	page    string
	folder  string
}

func parseScaffoldSection(line string) (scaffoldSectionAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 5 {
		return scaffoldSectionAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return scaffoldSectionAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++ // the article slot — optional, never required
	}
	if i < len(low) && low[i] == "grid" {
		i++ // the grid word can stand in front of the counts
	}
	var out scaffoldSectionAsk
	switch {
	case i+2 < len(low) && low[i] == "columns" && low[i+1] == "and" && low[i+2] == "rows":
		// No counts named: 2 columns and 2 rows (the file
		// comment above — the plan states them).
		out.cols, out.rows = 2, 2
		i += 3
	default:
		cols, rows, rowsSet, n := parseGridCounts(low, i)
		if n == 0 {
			return scaffoldSectionAsk{}, false
		}
		out.cols, out.rows, out.counted = cols, rows, true
		if !rowsSet {
			out.rows = 2 // columns alone: rows of 2
		}
		i += n
	}
	if i >= len(low) || low[i] != "to" {
		return scaffoldSectionAsk{}, false
	}
	i++
	if i < len(low) && low[i] == "the" {
		i++
	}
	if i >= len(low) || low[i] != "body" {
		return scaffoldSectionAsk{}, false
	}
	i++
	if i < len(low) && low[i] == "of" {
		i++
		if i < len(low) && low[i] == "the" {
			i++
		}
		if i >= len(low) {
			return scaffoldSectionAsk{}, false
		}
		out.page = fields[i]
		i++
		if i < len(low) && low[i] == "page" {
			i++
		}
	}
	if i < len(low) {
		if low[i] != "in" && low[i] != "into" && low[i] != "at" {
			return scaffoldSectionAsk{}, false
		}
		i++
		if i < len(low) && low[i] == "the" {
			i++
		}
		if i < len(low) && (low[i] == "folder" || low[i] == "directory") {
			i++
		}
		if i < len(low) {
			out.folder = fields[i]
			i++
		}
	}
	if i != len(low) {
		return scaffoldSectionAsk{}, false
	}
	return out, true
}

func (q scaffoldSectionAsk) canonical() string {
	s := fmt.Sprintf("add %d columns and %d rows to the body", q.cols, q.rows)
	if q.page != "" {
		s += " of " + q.page
	}
	if q.folder != "" {
		s += " in " + q.folder
	}
	return s
}

// tryScaffoldSection is the chat-loop entry: it claims
// body-section sentences and runs the flow on the chat's
// line reader.
func tryScaffoldSection(line string, r *LineReader) bool {
	return runScaffoldSection(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// isScaffoldBody reports whether file content is a
// templated page's body: it calls the shared header and
// the shared footer — the recognition every body routing
// in this package shares, so a file is routed by what it
// IS, never by its name.
func isScaffoldBody(b []byte) bool {
	s := string(b)
	return strings.Contains(s, "{{template \"header\"") && strings.Contains(s, "{{template \"footer\"")
}

// scaffoldPartKind reads file content as a templated
// page's part: "header" or "footer" when the file holds
// that shared definition, "body" when it calls both
// shared pieces (isScaffoldBody) — "" when it is none of
// them. Content is the recognition, as everywhere in this
// family: a webpage ask is answered by what its target
// file IS, never by the file's name alone.
func scaffoldPartKind(b []byte) string {
	s := string(b)
	switch {
	case strings.Contains(s, "{{define \"header\"}}"):
		return "header"
	case strings.Contains(s, "{{define \"footer\"}}"):
		return "footer"
	case isScaffoldBody(b):
		return "body"
	}
	return ""
}

// scaffoldBodies lists a folder's scaffold body files —
// the templates folder's pages other than the shared
// header and footer, each recognised by its own header and
// footer calls — sorted, as project-relative paths.
func scaffoldBodies(rootAbs, folderRel string) []string {
	dir := filepath.Join(rootAbs, filepath.FromSlash(folderRel), "templates")
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".html") || name == "header.html" || name == "footer.html" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			continue
		}
		if isScaffoldBody(b) {
			out = append(out, path.Join(folderRel, "templates", name))
		}
	}
	sort.Strings(out)
	return out
}

// runScaffoldSection carries one line from sentence to
// resolved body to the shared edit spine. It returns true
// when the line belonged to this claim at all.
func runScaffoldSection(line string, ask editAsk) bool {
	q, ok := parseScaffoldSection(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.cols < 1 || q.cols > 6 || q.rows < 1 || q.rows > 6 {
		var out []string
		if q.cols < 1 || q.cols > 6 {
			out = append(out, fmt.Sprintf("%d columns", q.cols))
		}
		if q.rows < 1 || q.rows > 6 {
			out = append(out, fmt.Sprintf("%d rows", q.rows))
		}
		fmt.Printf("%s Not adding that section: a columns-and-rows section can have 1 to 6 columns and 1 to 6 rows — %s is outside that. Ask again with numbers in that range.\n", tag, strings.Join(out, " and "))
		recordBrain("composer", false, "body section refused: counts out of range")
		return true
	}
	if q.folder == "" {
		printClarifyIntro(tag, "Not adding that section yet: the sentence doesn't name the folder — which folder's page should it go in?")
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				q2 := q
				q2.folder = text
				return runScaffoldSection(q2.canonical(), ask2)
			},
		})
		recordBrain("composer", false, "body section question: folder unnamed")
		return true
	}
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "body section")
	if !ok {
		return true
	}
	rel := "."
	if q.folder != "." {
		_, r, refusal := fenceResolve(rootAbs, q.folder)
		if refusal != "" {
			fmt.Printf("%s Not adding that section: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "body section refused: folder outside project")
			return true
		}
		rel = r
	}
	bodies := scaffoldBodies(rootAbs, rel)
	resolved := ""
	if q.page != "" {
		want := path.Join(rel, "templates", q.page+".html")
		for _, b := range bodies {
			if b == want {
				resolved = b
			}
		}
		if resolved == "" {
			fmt.Printf("%s Not adding that section: there's no templated page called %s in %s yet — say \"add a templated page called %s in %s\" first, then ask again.\n", tag, q.page, rel, q.page, rel)
			recordBrain("composer", false, "body section refused: named page missing")
			return true
		}
	} else {
		switch len(bodies) {
		case 0:
			fmt.Printf("%s Not adding that section: there's no templated page in %s yet — a body section belongs to one. Say \"add a templated page called <name> in %s\" first, then ask again.\n", tag, rel, rel)
			recordBrain("composer", false, "body section refused: no scaffold")
			return true
		case 1:
			resolved = bodies[0]
		default:
			// Several bodies and the sentence names none:
			// ask which page, one question — the pick lands
			// the section in that page's body only.
			cands := make([]clarifyCandidate, len(bodies))
			for i, b := range bodies {
				name := strings.TrimSuffix(path.Base(b), ".html")
				cands[i] = clarifyCandidate{Label: name, Desc: "the " + name + " page's body", Words: []string{name}}
			}
			cols, rows := q.cols, q.rows
			askPickQuestion(tag,
				fmt.Sprintf("The %s folder has more than one templated page — which page's body should the section of %s go in?", rel, analyze.ScaffoldSectionWords(cols, rows)),
				"Answer with the page's name or its number",
				cands,
				func(idx int, ask2 editAsk) bool {
					return runScaffoldSectionPlan(tag, rootRaw, rootAbs, rel, bodies[idx], cols, rows, ask2)
				})
			recordBrain("composer", false, "body section question: which page")
			return true
		}
	}
	return runScaffoldSectionPlan(tag, rootRaw, rootAbs, rel, resolved, q.cols, q.rows, ask)
}

// runScaffoldSectionPlan is the resolved tail: the project
// settles and the section runs through the edit spine.
func runScaffoldSectionPlan(tag, rootRaw, rootAbs, folderRel, bodyRel string, cols, rows int, ask editAsk) bool {
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "body section") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:       analyze.EditScaffoldSection,
		SiteFolder: folderRel,
		SiteBody:   bodyRel,
		SiteCols:   cols,
		SiteRows:   rows,
	}, ask)
}
