package chat

// The wasm app skeleton (John 2026-11-09, "have the
// components in gowasm" — and, when the simd question came
// back measured AMD64-only on Go 1.26, "why not just have
// it use 1.27"):
//
//	add|create|make|build|put [a|an|the]
//	    wasm|webassembly app in|into|at <folder>
//
// One ask creates the app in the folder (analyze.planWasmApp
// — the app's own module pinning Go 1.27, its main.go, a
// small simd vector unit, the host page, and the
// wasm_exec.js copied from this machine's Go 1.27 install
// when one is found). The app is a whole program of its
// own, apart from the project's build; the components that
// join it are the component piece's business, in its wasm
// mode (component.go).
//
// The folder is required: a sentence that starts the shape
// without one is claimed and asked, one question, through
// the clarify machinery — the templated page's discipline.
// A folder that isn't there is the family's two-step, never
// a silent creation.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// wasmAppAsk is one parsed wasm-app sentence: the folder
// as typed ("" when the sentence starts the shape without
// it — the flow asks).
type wasmAppAsk struct {
	folder string
}

func (q wasmAppAsk) canonical() string {
	s := "add a wasm app"
	if q.folder != "" {
		s += " in " + q.folder
	}
	return s
}

func parseWasmApp(line string) (wasmAppAsk, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return wasmAppAsk{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "create", "make", "build", "put":
	default:
		return wasmAppAsk{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++ // the article slot — optional, never required
	}
	if i >= len(fields) || (low[i] != "wasm" && low[i] != "webassembly") {
		return wasmAppAsk{}, false
	}
	i++
	if i >= len(fields) || low[i] != "app" {
		return wasmAppAsk{}, false
	}
	i++
	var out wasmAppAsk
	if i < len(fields) {
		if low[i] != "in" && low[i] != "into" && low[i] != "at" {
			return wasmAppAsk{}, false
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
		if strings.HasSuffix(low[len(low)-1], ".html") || strings.HasSuffix(low[len(low)-1], ".go") {
			// The slot names a folder; a file-shaped word
			// there is not this sentence.
			return wasmAppAsk{}, false
		}
	}
	if i != len(fields) {
		return wasmAppAsk{}, false
	}
	return out, true
}

// tryWasmApp is the chat-loop entry: it claims wasm-app
// sentences and runs the flow on the chat's line reader.
func tryWasmApp(line string, r *LineReader) bool {
	return runWasmApp(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runWasmApp carries one line from sentence to resolved
// folder to the shared edit spine. It returns true when
// the line belonged to this claim at all.
func runWasmApp(line string, ask editAsk) bool {
	q, ok := parseWasmApp(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if q.folder == "" {
		printClarifyIntro(tag, "Not making that wasm app yet: the sentence doesn't name the folder — which folder should the app live in?")
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				return runWasmApp(wasmAppAsk{folder: text}.canonical(), ask2)
			},
		})
		recordBrain("composer", false, "wasm app question: folder unnamed")
		return true
	}
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "wasm app")
	if !ok {
		return true
	}
	rel, abs := ".", rootAbs
	if q.folder != "." {
		a, r, refusal := fenceResolve(rootAbs, q.folder)
		if refusal != "" {
			fmt.Printf("%s Not making that wasm app: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "wasm app refused: folder outside project")
			return true
		}
		abs, rel = a, r
	}
	if rel == "." {
		fmt.Printf("%s Not making that wasm app: a wasm app needs a folder of its own — its go.mod makes the folder a module of its own, apart from the project's build, so the app can't be the project root. Name a folder: \"add a wasm app in jim\".\n", tag)
		recordBrain("composer", false, "wasm app refused: project root")
		return true
	}
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		if err == nil {
			fmt.Printf("%s Not making that wasm app: %s isn't a folder — a wasm app belongs to a folder, as its own module.\n", tag, rel)
		} else {
			fmt.Printf("%s Not making that wasm app: there's no folder %s in this project for it to live in — say \"create a folder %s\" first, then ask again.\n", tag, rel, rel)
		}
		recordBrain("composer", false, "wasm app refused: folder missing")
		return true
	}
	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "wasm app") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:       analyze.EditWasmApp,
		WasmFolder: rel,
	}, ask)
}

// isWasmMain reports whether file content is a wasm app's
// main: the js && wasm build tag and the syscall/js
// import — the chat-side twin of the analyzer's
// recognition, so a component ask is routed by what the
// file IS.
func isWasmMain(b []byte) bool {
	s := string(b)
	return strings.Contains(s, "js && wasm") && strings.Contains(s, "\"syscall/js\"")
}

// wasmAppFolderOf reads a folder's main.go and reports
// whether the folder holds a wasm app — the component
// piece's routing question for a folder (or main.go)
// target.
func wasmAppFolderOf(rootAbs, folderRel string) bool {
	b, err := os.ReadFile(filepath.Join(rootAbs, filepath.FromSlash(folderRel), "main.go"))
	if err != nil {
		return false
	}
	return isWasmMain(b)
}
