package chat

// The wasm parts piece (John 2026-11-10): a wasm app's
// header and footer as packages inside the app —
//
//	add|create|make|build|put [a|an|the]
//	    header and footer to the wasm app in <folder>
//	add|create|make|build|put [a|an|the]
//	    header|footer [to the wasm app] in <folder>
//	add|create|make|build|put [a|an|the]
//	    header|footer to <folder>
//
// One ask lands the part (or both parts) through the one
// edit spine (analyze.planWasmPart): <app>/header/header.go
// and <app>/footer/footer.go, each its own package with
// its own components markers, main.go importing them by
// the app's module path and mounting the header before
// the body components and the footer after. A body
// app bar or footer MOVES into its part rather than
// being duplicated — the move is the planner's.
//
// The claim is resolution-gated, the build claim's
// discipline: it fires only when the named (or the
// session's last-touched) folder already holds a wasm
// app, and declines untouched everywhere else — "add a
// header to jim" over a folder that holds no app keeps
// exactly the fate the line has always had. With no
// folder named and no app in the session, the sentence
// that names the app ("… to the wasm app") is asked the
// one folder question, like the serve claim; without
// either signal the line declines.

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// wasmPartAsk is one parsed parts sentence: which parts,
// and the folder as typed ("" when the sentence names
// none — the session, or the one question, supplies it).
// namedApp records that the sentence carried the "wasm
// app" words — the signal that makes a folderless ask
// this claim's to question about.
type wasmPartAsk struct {
	header   bool
	footer   bool
	folder   string
	namedApp bool
}

func (q wasmPartAsk) canonical() string {
	what := "a header and footer"
	if q.header && !q.footer {
		what = "a header"
	}
	if q.footer && !q.header {
		what = "a footer"
	}
	s := "add " + what + " to the wasm app"
	if q.folder != "" {
		s += " in " + q.folder
	}
	return s
}

func parseWasmPart(line string) (wasmPartAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return wasmPartAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return wasmPartAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	var out wasmPartAsk
	switch {
	case i+2 < len(fields) && low[i] == "header" && low[i+1] == "and" && low[i+2] == "footer":
		out.header, out.footer = true, true
		i += 3
	case i+2 < len(fields) && low[i] == "footer" && low[i+1] == "and" && low[i+2] == "header":
		out.header, out.footer = true, true
		i += 3
	case i < len(fields) && low[i] == "header":
		out.header = true
		i++
	case i < len(fields) && low[i] == "footer":
		out.footer = true
		i++
	default:
		return wasmPartAsk{}, false
	}
	if i == len(fields) {
		return out, true
	}
	if low[i] != "to" && low[i] != "in" && low[i] != "into" && low[i] != "at" {
		return wasmPartAsk{}, false
	}
	i++
	// The optional middle: "the wasm app", then its own
	// preposition and the folder.
	if i+1 < len(fields) && low[i] == "the" && low[i+1] == "wasm" {
		if i+2 >= len(fields) || low[i+2] != "app" {
			return wasmPartAsk{}, false
		}
		out.namedApp = true
		i += 3
		if i == len(fields) {
			return out, true
		}
		if low[i] != "to" && low[i] != "in" && low[i] != "into" && low[i] != "at" {
			return wasmPartAsk{}, false
		}
		i++
	}
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) && (low[i] == "folder" || low[i] == "directory") {
		i++
	}
	if i != len(fields)-1 {
		return wasmPartAsk{}, false
	}
	tail := low[len(low)-1]
	if strings.HasSuffix(tail, ".html") || strings.HasSuffix(tail, ".go") {
		// The slot names a folder; a file-shaped word
		// there is not this sentence.
		return wasmPartAsk{}, false
	}
	out.folder = fields[i]
	return out, true
}

// tryWasmPart is the chat-loop entry: it claims wasm
// parts sentences and runs the flow on the chat's line
// reader.
func tryWasmPart(line string, r *LineReader) bool {
	return runWasmPart(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runWasmPart carries one line for this claim. It returns
// false — the line untouched, nothing printed — for every
// line outside the claim's shapes, and for every shaped
// line whose folder (named or the session's) does not
// already hold a wasm app: resolution is part of the
// claim, so a folder with no app keeps the fate the
// sentence has always had further down the chain.
func runWasmPart(line string, ask editAsk) bool {
	q, ok := parseWasmPart(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootRaw, rootAbs, rok := staticAssetsRoot(tag, "wasm part")
	if !rok {
		return true
	}
	var folderRel string
	switch {
	case q.folder != "":
		_, rel, refusal := fenceResolve(rootAbs, q.folder)
		if refusal != "" {
			// Outside the project: not this claim's part.
			return false
		}
		if !wasmAppFolderOf(rootAbs, rel) {
			return false
		}
		folderRel = rel
	case workContextLive(rootAbs) && workCtx.wasmFolder != "" && wasmAppFolderOf(rootAbs, workCtx.wasmFolder):
		folderRel = workCtx.wasmFolder
		fmt.Printf("Reading the app as the wasm app in %s — the last wasm app this session touched.\n", folderRel)
	case q.namedApp:
		printClarifyIntro(tag, "Not adding a part yet: the sentence doesn't name the app's folder — which folder does the app live in?")
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				q2 := q
				q2.folder = text
				return runWasmPart(q2.canonical(), ask2)
			},
		})
		recordBrain("composer", false, "wasm part question: folder unnamed")
		return true
	default:
		return false
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "wasm part") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:           analyze.EditWasmPart,
		WasmFolder:     folderRel,
		WasmPartHeader: q.header,
		WasmPartFooter: q.footer,
	}, ask)
}
