package chat

// The wasm home piece (John 2026-11-10, "there's no home
// file when you do add wasm app"): a wasm app's home page
// as a file of its own —
//
//	add|create|make|build|put [a|an|the]
//	    home page to the wasm app in <folder>
//	add|create|make|build|put [a|an|the]
//	    home page to the wasm app
//
// Apps the skeleton makes carry the home page from birth
// (home.go, the Home builder — the welcome heading and
// the vector demo); this sentence gives one to an app
// made before they did, through the one edit spine
// (analyze.planWasmHome): the standard inline welcome
// block moves out of main.go into home.go, and main
// mounts Home where the block stood.
//
// The claim is resolution-gated, the parts claim's
// discipline: it fires only when the named (or the
// session's last-touched) folder already holds a wasm
// app, and declines untouched everywhere else. The shape
// itself does the guarding: it requires the words "home
// page to the wasm app", which the page family — seated
// far above this claim, and greedy for "home page" —
// cannot take, because its target tail reads one folder
// token and "the wasm app" is three. So "add a home page
// to jim" and "create a homepage" keep the page family's
// fate byte-identical, and this claim only ever sees the
// sentences that name the app. With no folder named and
// no app in the session, the sentence is asked the one
// folder question, like its siblings.

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// wasmHomeAsk is one parsed home-page sentence: the
// folder as typed ("" when the sentence names none — the
// session, or the one question, supplies it). Every
// parsed line names the app ("… to the wasm app"), so
// the folderless form is always this claim's to complete.
type wasmHomeAsk struct {
	folder string
}

func (q wasmHomeAsk) canonical() string {
	s := "add a home page to the wasm app"
	if q.folder != "" {
		s += " in " + q.folder
	}
	return s
}

func parseWasmHome(line string) (wasmHomeAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 7 {
		return wasmHomeAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return wasmHomeAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	if i+1 >= len(fields) || low[i] != "home" || low[i+1] != "page" {
		return wasmHomeAsk{}, false
	}
	i += 2
	// The middle is the claim's signature: a preposition,
	// then "the wasm app" — the words no neighbouring
	// grammar takes after "home page".
	if i >= len(fields) || (low[i] != "to" && low[i] != "in" && low[i] != "into" && low[i] != "at") {
		return wasmHomeAsk{}, false
	}
	i++
	if i+2 >= len(fields) || low[i] != "the" || (low[i+1] != "wasm" && low[i+1] != "webassembly") || low[i+2] != "app" {
		return wasmHomeAsk{}, false
	}
	i += 3
	var out wasmHomeAsk
	if i == len(fields) {
		return out, true
	}
	if low[i] != "in" && low[i] != "into" && low[i] != "at" && low[i] != "to" {
		return wasmHomeAsk{}, false
	}
	i++
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) && (low[i] == "folder" || low[i] == "directory") {
		i++
	}
	if i != len(fields)-1 {
		return wasmHomeAsk{}, false
	}
	tail := low[len(low)-1]
	if strings.HasSuffix(tail, ".html") || strings.HasSuffix(tail, ".go") {
		// The slot names a folder; a file-shaped word
		// there is not this sentence.
		return wasmHomeAsk{}, false
	}
	out.folder = fields[i]
	return out, true
}

// tryWasmHome is the chat-loop entry: it claims wasm
// home-page sentences and runs the flow on the chat's
// line reader.
func tryWasmHome(line string, r *LineReader) bool {
	return runWasmHome(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runWasmHome carries one line for this claim. It returns
// false — the line untouched, nothing printed — for every
// line outside the claim's shapes, and for every shaped
// line whose folder (named or the session's) does not
// already hold a wasm app: resolution is part of the
// claim, so a folder with no app keeps the fate the
// sentence has always had further down the chain.
func runWasmHome(line string, ask editAsk) bool {
	q, ok := parseWasmHome(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootRaw, rootAbs, rok := staticAssetsRoot(tag, "wasm home")
	if !rok {
		return true
	}
	var folderRel string
	switch {
	case q.folder != "":
		_, rel, refusal := fenceResolve(rootAbs, q.folder)
		if refusal != "" {
			// Outside the project: not this claim's app.
			return false
		}
		if !wasmAppFolderOf(rootAbs, rel) {
			return false
		}
		folderRel = rel
	case workContextLive(rootAbs) && workCtx.wasmFolder != "" && wasmAppFolderOf(rootAbs, workCtx.wasmFolder):
		folderRel = workCtx.wasmFolder
		fmt.Printf("Reading the app as the wasm app in %s — the last wasm app this session touched.\n", folderRel)
	default:
		printClarifyIntro(tag, "Not adding a home page yet: the sentence doesn't name the app's folder — which folder does the app live in?")
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				return runWasmHome(wasmHomeAsk{folder: text}.canonical(), ask2)
			},
		})
		recordBrain("composer", false, "wasm home question: folder unnamed")
		return true
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "wasm home") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:       analyze.EditWasmHome,
		WasmFolder: folderRel,
	}, ask)
}
