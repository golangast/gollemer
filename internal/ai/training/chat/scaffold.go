package chat

// The templated page scaffold (John 2026-11-09: the header,
// footer, and body as separate files, connected with Go
// templates):
//
//	add|create|make|build|put [a|an|the] [go]
//	    templated|template page called <name> in <folder>
//
// One ask creates four files in the folder: the shared
// header and footer under <folder>/templates/, the page's
// own body file beside them, and the handler that renders
// the body through them (analyze.planTemplatedPage — the
// split, the contents, and the no-overwrite rule are the
// planner's). No route is wired: the folder's server keeps
// the routes it has, and the handler waits for the wiring
// sentence.
//
// The name is required: a sentence that starts the shape
// but gives no name ("add a templated page in jim") is
// claimed and asked, one question, through the clarify
// machinery — and a sentence that names no folder is asked
// the same way. A folder that isn't there is the family's
// two-step, never a silent creation.

import (
	"fmt"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// templatedPageAsk is one parsed scaffold sentence: the
// page's name and folder as typed (either "" when the
// sentence starts the shape without it — the flow asks).
type templatedPageAsk struct {
	name   string
	folder string
}

func (q templatedPageAsk) canonical() string {
	s := "add a templated page"
	if q.name != "" {
		s += " called " + q.name
	}
	if q.folder != "" {
		s += " in " + q.folder
	}
	return s
}

func parseTemplatedPage(line string) (templatedPageAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return templatedPageAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return templatedPageAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++ // the article slot — optional, never required
	}
	if i < len(fields) && low[i] == "go" {
		i++ // "a go template page"
	}
	if i >= len(fields) || (low[i] != "templated" && low[i] != "template") {
		return templatedPageAsk{}, false
	}
	i++
	if i >= len(fields) || low[i] != "page" {
		return templatedPageAsk{}, false
	}
	i++
	var out templatedPageAsk
	if i < len(fields) && low[i] == "called" {
		i++
		if i < len(fields) && low[i] != "in" && low[i] != "into" && low[i] != "at" {
			out.name = fields[i]
			i++
		}
	}
	if i < len(fields) {
		if low[i] != "in" && low[i] != "into" && low[i] != "at" {
			return templatedPageAsk{}, false
		}
		i++
		if i < len(fields) && low[i] == "the" {
			i++
		}
		if i < len(fields) && (low[i] == "folder" || low[i] == "directory") {
			i++
		}
		if i < len(fields) {
			out.folder = fields[i]
			i++
		}
	}
	if i != len(fields) {
		return templatedPageAsk{}, false
	}
	return out, true
}

// tryTemplatedPage is the chat-loop entry: it claims
// templated-page sentences and runs the flow on the chat's
// line reader.
func tryTemplatedPage(line string, r *LineReader) bool {
	return runTemplatedPage(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runTemplatedPage carries one line from sentence to
// resolved name and folder to the shared edit spine. It
// returns true when the line belonged to this claim at all.
func runTemplatedPage(line string, ask editAsk) bool {
	q, ok := parseTemplatedPage(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.name == "" {
		// Claimed and asked in scaffold terms (clarify.go):
		// the answer, one name, completes the ask as its
		// canonical sentence — with the folder when the
		// sentence named one.
		folder := q.folder
		printClarifyIntro(tag, "Not making that templated page yet: the sentence doesn't give the page a name — what should I call it?")
		printClarifyFooter("Answer with one name — \"home\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyName,
			completeText: func(text string, ask2 editAsk) bool {
				return runTemplatedPage(templatedPageAsk{name: text, folder: folder}.canonical(), ask2)
			},
		})
		recordBrain("composer", false, "templated page question: name unnamed")
		return true
	}
	if q.folder == "" {
		name := q.name
		printClarifyIntro(tag, fmt.Sprintf("Not making the %s templated page yet: the sentence doesn't name the folder — which folder should the page live in?", name))
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				return runTemplatedPage(templatedPageAsk{name: name, folder: text}.canonical(), ask2)
			},
		})
		recordBrain("composer", false, "templated page question: folder unnamed")
		return true
	}
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "templated page")
	if !ok {
		return true
	}
	rel, abs := ".", rootAbs
	if q.folder != "." {
		a, r, refusal := fenceResolve(rootAbs, q.folder)
		if refusal != "" {
			fmt.Printf("%s Not making that templated page: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "templated page refused: folder outside project")
			return true
		}
		abs, rel = a, r
	}
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		if err == nil {
			fmt.Printf("%s Not making that templated page: %s isn't a folder — a templated page belongs to a folder, with its handler in the folder's package.\n", tag, rel)
		} else {
			fmt.Printf("%s Not making that templated page: there's no folder %s in this project for it to live in — say \"create a folder %s\" first, then ask again.\n", tag, rel, rel)
		}
		recordBrain("composer", false, "templated page refused: folder missing")
		return true
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "templated page") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:       analyze.EditTemplatedPage,
		SiteFolder: rel,
		SiteName:   q.name,
	}, ask)
}
