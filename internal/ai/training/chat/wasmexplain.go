package chat

// The wasm explanation (John 2026-11-10): "explain the
// wasm app in <folder>" walks the app's folder in plain
// words — and the same walk follows the FIRST successful
// build or serve of a given app automatically, once per
// app, so a beginner meets the tour exactly when the app
// first runs and is never lectured twice.
//
// The walk is built ONLY from the folder's actual
// contents: a file that isn't there gets no line, a file
// that is there gets the line its contents earn (the
// components and events are named from the app's own
// files, through the analyzer's one reading of them),
// and main.wasm is reported built or not built, as it
// stands. Nothing is remembered or guessed.
//
// The once-per-app memory is a sibling store in the
// receipts store's family (~/.gollemer_wasm_explained.json,
// the same 0600/atomic discipline), keyed by project +
// app folder. It is a courtesy, never a dependency: if
// the store can't be read or written the tour simply
// doesn't repeat-suppress — a build or serve is never
// blocked over it.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// --- the walk ---

// wasmEventCommentRe reads one event handler's sentence
// out of events.go: the comment the events piece writes
// above each handler IS the sentence that asked for it.
var wasmEventCommentRe = regexp.MustCompile(`(?m)^// When (.+)$`)

// wasmInstanceWords renders one mounted component the
// way a sentence says it: "a card", "a button called
// save".
func wasmInstanceWords(in analyze.WasmComponentInstance) string {
	s := "a " + analyze.MaterialComponentWords(in.Kind)
	if in.Name != "" {
		s += " called " + in.Name
	}
	return s
}

// wasmInstanceListWords renders a set of components as
// one list: "a card, a button called save".
func wasmInstanceListWords(insts []analyze.WasmComponentInstance) string {
	parts := make([]string, 0, len(insts))
	for _, in := range insts {
		parts = append(parts, wasmInstanceWords(in))
	}
	return strings.Join(parts, ", ")
}

// wasmAppExplanation builds the folder walk for the app
// in folderRel under rootAbs. Every line is earned by a
// file or folder actually present (or, for main.wasm, by
// its honest absence); the component and event names
// come from the analyzer's reading of the app's own
// files. The text's first line is the lead; callers add
// their own framing around it.
func wasmAppExplanation(rootAbs, folderRel string) string {
	appAbs := filepath.Join(rootAbs, filepath.FromSlash(folderRel))
	read := func(rel string) (string, bool) {
		b, err := os.ReadFile(filepath.Join(appAbs, filepath.FromSlash(rel)))
		if err != nil {
			return "", false
		}
		return string(b), true
	}
	var instances []analyze.WasmComponentInstance
	if proj, err := analyze.Analyze(rootAbs); err == nil && proj != nil {
		instances = proj.WasmAppComponents(folderRel)
	}
	byPart := map[string][]analyze.WasmComponentInstance{}
	for _, in := range instances {
		byPart[in.Part] = append(byPart[in.Part], in)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "The wasm app in %s, file by file:", folderRel)
	if goMod, ok := read("go.mod"); ok {
		pin := ""
		for _, ln := range strings.Split(goMod, "\n") {
			ln = strings.TrimSpace(ln)
			if strings.HasPrefix(ln, "go ") {
				pin = strings.TrimSpace(strings.TrimPrefix(ln, "go "))
			}
		}
		pinWords := "Go 1.27"
		if pin != "" {
			pinWords = "Go " + strings.TrimSuffix(pin, ".0")
		}
		fmt.Fprintf(&b, "\n  %s/go.mod — the app's own module: it makes the folder a module of its own, apart from the project's build, and pins %s — the portable simd package the vector unit uses exists for wasm only there.", folderRel, pinWords)
	}
	if main, ok := read("main.go"); ok {
		line := "\n  " + folderRel + "/main.go — the program: "
		var does []string
		if strings.Contains(main, "SetGCPercent") {
			does = append(does, "it sets the garbage collector's GOGC knob low in code (a browser page has no environment to pass the flag through)")
		}
		if strings.Contains(main, "AddVectors(") {
			does = append(does, "builds the welcome heading and the vector demo's sums into the page")
		}
		mounts := []string{}
		if strings.Contains(main, "header.Header(doc)") {
			mounts = append(mounts, "the header")
		}
		mounts = append(mounts, "the app's components")
		if strings.Contains(main, "footer.Footer(doc)") {
			mounts = append(mounts, "the footer")
		}
		does = append(does, "mounts "+strings.Join(mounts, ", then "))
		does = append(does, "then stays running — a wasm program whose main returns is finished, and the browser takes its page down with it")
		line += strings.Join(does, "; ") + "."
		b.WriteString(line)
	}
	if _, ok := read("vector.go"); ok {
		fmt.Fprintf(&b, "\n  %s/vector.go — the vector unit: it adds two rows of numbers a vector at a time with the portable simd package, and main prints the sums on the page as the proof it ran.", folderRel)
	}
	if _, ok := read("components.go"); ok {
		if body := byPart[""]; len(body) > 0 {
			fmt.Fprintf(&b, "\n  %s/components.go — the app's own components, as Go builders the program mounts in the order they were asked for: %s.", folderRel, wasmInstanceListWords(body))
		} else {
			fmt.Fprintf(&b, "\n  %s/components.go — the app's own components, as Go builders the program mounts in the order they were asked for.", folderRel)
		}
	}
	if events, ok := read("events.go"); ok {
		sentences := wasmEventCommentRe.FindAllStringSubmatch(events, -1)
		if len(sentences) > 0 {
			var said []string
			for _, m := range sentences {
				said = append(said, "when "+strings.ToLower(m[1][:1])+m[1][1:])
			}
			fmt.Fprintf(&b, "\n  %s/events.go — the events the app answers in the browser: %s. Every callback is kept in the file's list for the page's whole life — one Go lets go of would be garbage-collected and silently stop answering.", folderRel, strings.Join(said, "; "))
		} else {
			fmt.Fprintf(&b, "\n  %s/events.go — the events the app answers in the browser, one handler each, every callback kept in the file's list for the page's whole life.", folderRel)
		}
	}
	for _, part := range []string{"header", "footer"} {
		if _, ok := read(part + "/" + part + ".go"); !ok {
			continue
		}
		place := "at the top of the page"
		if part == "footer" {
			place = "at the foot of the page"
		}
		line := fmt.Sprintf("\n  %s/%s/ — the %s part, its own package (package %s): %s.go builds ", folderRel, part, part, part, part)
		rootWords := ""
		var rest []analyze.WasmComponentInstance
		for _, in := range byPart[part] {
			if in.Root {
				rootWords = "the " + analyze.MaterialComponentWords(in.Kind)
				if in.Name != "" {
					rootWords += " called " + in.Name
				}
			} else {
				rest = append(rest, in)
			}
		}
		if rootWords != "" {
			line += rootWords + " " + place
		} else {
			line += "the part " + place
		}
		if _, hasComp := read(part + "/components.go"); hasComp {
			if len(rest) > 0 {
				line += fmt.Sprintf(", and its components.go holds what the chat has added inside it: %s", wasmInstanceListWords(rest))
			} else {
				line += ", and its components.go holds what the chat has added inside it"
			}
		}
		line += ". Its own components land between its markers when you ask — \"add a chip to the " + part + " in " + folderRel + "\"."
		b.WriteString(line)
	}
	if _, ok := read("index.html"); ok {
		fmt.Fprintf(&b, "\n  %s/index.html — the host page: the Material styles, the #app element the program builds into, and the loader that starts the built app.", folderRel)
	}
	if _, ok := read("wasm_exec.js"); ok {
		fmt.Fprintf(&b, "\n  %s/wasm_exec.js — the Go runtime's glue, copied from this machine's Go 1.27 install: the host page loads it before the app, and the page can't start the app without it.", folderRel)
	}
	if st, err := os.Stat(filepath.Join(appAbs, "main.wasm")); err == nil {
		fmt.Fprintf(&b, "\n  %s/main.wasm — the built app, %s bytes: the file the browser fetches and runs. It is a build product, remade by every build.", folderRel, groupThousands(st.Size()))
	} else {
		fmt.Fprintf(&b, "\n  The app isn't built yet — there's no main.wasm in %s, so a served page would have no program to start. The build below makes it.", folderRel)
	}
	// The other folder of the serving arrangement: the
	// dedicated webserver in server/ at the project root,
	// named only when its file is really there AND really
	// serves this app (its /static/ registration names this
	// folder) — the walk never invents a server.
	servedByServerFolder := false
	serverPort := ""
	if srv, err := os.ReadFile(filepath.Join(rootAbs, analyze.ServeWasmServerFolder, "main.go")); err == nil {
		if strings.Contains(string(srv), `"`+analyze.StaticRoute+`"`) &&
			strings.Contains(string(srv), "Dir("+"\""+folderRel+"\")") {
			servedByServerFolder = true
			serverPort = serveStatementPort(rootAbs, analyze.ServeWasmServerFolder+"/main.go")
			fmt.Fprintf(&b, "\n  %s/main.go — the Go webserver, in its own folder at the project root: it hands the app's files to the browser at /static/ — the host page, the glue, and main.wasm. It is an ordinary Go program; the app itself runs in the browser, not in it.", analyze.ServeWasmServerFolder)
		}
	}
	fmt.Fprintf(&b, "\nTo build it: \"build the wasm app in %s\" runs the host page's own build command in %s (%s) — a first build fetches the Go 1.27 toolchain by itself.", folderRel, folderRel, analyze.WasmBuildCommand)
	if servedByServerFolder {
		address := "under /static/ on that server"
		if serverPort != "" {
			address = "at http://localhost:" + serverPort + "/static/"
		}
		fmt.Fprintf(&b, "\nTo serve it: two folders, two jobs — %s is the program that runs in the browser; the server folder is the Go program that hands it over. \"serve the wasm app in %s\" runs the server in server/, and the app answers %s.", folderRel, folderRel, address)
	} else {
		fmt.Fprintf(&b, "\nTo serve it: \"serve the wasm app in %s\" gives the app a dedicated Go webserver in a server folder at the project root (server/main.go) — its own folder, its one job — or adds the app's route to the one already there, and runs it: the app is handed to the browser under /static/, at http://localhost:8080/static/ with the house port, and the serve sentence reads your server's real port from its own serve statement.", folderRel)
	}
	b.WriteString("\nWhen the page opens: the browser loads wasm_exec.js, fetches main.wasm, and starts the Go program, which builds the page in the browser — and the events answer as you click and change things.")
	return b.String()
}

// --- the once-per-app store ---

// wasmExplainedData is the store's shape: the apps whose
// tour has already run, per project — project roots
// canonicalised the memory family's way, app folders
// project-relative.
type wasmExplainedData struct {
	Projects map[string][]string `json:"projects"`
}

var wasmExplainedStorePathOverride string

func wasmExplainedStorePath() string {
	if wasmExplainedStorePathOverride != "" {
		return wasmExplainedStorePathOverride
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ".gollemer_wasm_explained.json"
	}
	return filepath.Join(home, ".gollemer_wasm_explained.json")
}

type jsonWasmExplainedStore struct{ path string }

func (s jsonWasmExplainedStore) Load() (wasmExplainedData, error) {
	data, err := os.ReadFile(s.path)
	if errors.Is(err, fs.ErrNotExist) {
		return wasmExplainedData{Projects: map[string][]string{}}, nil
	}
	if err != nil {
		return wasmExplainedData{}, err
	}
	if len(strings.TrimSpace(string(data))) == 0 {
		return wasmExplainedData{Projects: map[string][]string{}}, nil
	}
	var out wasmExplainedData
	if err := json.Unmarshal(data, &out); err != nil {
		return wasmExplainedData{}, err
	}
	if out.Projects == nil {
		out.Projects = map[string][]string{}
	}
	return out, nil
}

func (s jsonWasmExplainedStore) Save(d wasmExplainedData) error {
	if d.Projects == nil {
		d.Projects = map[string][]string{}
	}
	data, err := json.MarshalIndent(d, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	dir := filepath.Dir(s.path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".gollemer-wasm-explained-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, s.path)
}

// wasmAppExplainedSeen reports whether the tour has
// already run for this app in this project. Any store
// trouble reads as "not seen" — the worst case is a
// repeated tour, never a blocked build.
func wasmAppExplainedSeen(rootAbs, folderRel string) bool {
	data, err := jsonWasmExplainedStore{path: wasmExplainedStorePath()}.Load()
	if err != nil {
		return false
	}
	for _, f := range data.Projects[canonicalMemoryRoot(rootAbs)] {
		if f == folderRel {
			return true
		}
	}
	return false
}

// markWasmAppExplained records that the tour ran for
// this app. Best-effort: every error is swallowed — the
// store is a courtesy (see the file comment).
func markWasmAppExplained(rootAbs, folderRel string) {
	store := jsonWasmExplainedStore{path: wasmExplainedStorePath()}
	data, err := store.Load()
	if err != nil {
		return
	}
	key := canonicalMemoryRoot(rootAbs)
	for _, f := range data.Projects[key] {
		if f == folderRel {
			return
		}
	}
	data.Projects[key] = append(data.Projects[key], folderRel)
	_ = store.Save(data)
}

// maybeExplainWasmAppFirstRun runs the tour after a
// build's or serve's success narration, the first time
// this app has built or served in this project — and
// only then. The store is consulted and updated
// best-effort; nothing here can fail the build or serve
// it follows.
func maybeExplainWasmAppFirstRun(tag, rootAbs, folderRel string) {
	if wasmAppExplainedSeen(rootAbs, folderRel) {
		return
	}
	fmt.Printf("%s Since this is the first time the app in %s has run here, here is what it is made of:\n%s\n", tag, folderRel, wasmAppExplanation(rootAbs, folderRel))
	markWasmAppExplained(rootAbs, folderRel)
}

// --- the on-demand claim ---

// parseExplainWasmApp reads the explanation sentence:
// "explain [the] wasm|webassembly app" with the family's
// optional folder tail. This parse claims by shape only.
func parseExplainWasmApp(line string) (folder string, hasFolder, ok bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return "", false, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	if low[0] != "explain" {
		return "", false, false
	}
	i := 1
	if low[i] == "the" || low[i] == "a" || low[i] == "an" {
		i++
	}
	if i >= len(fields) || (low[i] != "wasm" && low[i] != "webassembly") {
		return "", false, false
	}
	i++
	if i >= len(fields) || low[i] != "app" {
		return "", false, false
	}
	i++
	if i == len(fields) {
		return "", false, true
	}
	if low[i] != "in" && low[i] != "into" && low[i] != "at" {
		return "", false, false
	}
	i++
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) && (low[i] == "folder" || low[i] == "directory") {
		i++
	}
	if i != len(fields)-1 {
		return "", false, false
	}
	tail := low[len(low)-1]
	if strings.HasSuffix(tail, ".html") || strings.HasSuffix(tail, ".go") {
		return "", false, false
	}
	return fields[i], true, true
}

// tryExplainWasmApp is the chat-loop entry: it claims
// explain-the-wasm-app sentences and runs the flow on
// the chat's line reader.
func tryExplainWasmApp(line string, r *LineReader) bool {
	return runExplainWasmApp(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runExplainWasmApp carries one line for this claim: the
// folder named (fenced), the session's, or asked — the
// serve claim's three ways — and the folder must hold a
// wasm app, every other state answered for what it is.
// The explanation always answers on demand; the
// first-run store never suppresses it.
func runExplainWasmApp(line string, ask editAsk) bool {
	folder, hasFolder, ok := parseExplainWasmApp(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	_, rootAbs, rok := staticAssetsRoot(tag, "wasm explain")
	if !rok {
		return true
	}
	var folderRel string
	switch {
	case hasFolder:
		_, rel, refusal := fenceResolve(rootAbs, folder)
		if refusal != "" {
			fmt.Printf("%s Not explaining the wasm app: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("composer", false, "wasm explain refused: folder outside project")
			return true
		}
		folderRel = rel
	case workContextLive(rootAbs) && workCtx.wasmFolder != "":
		folderRel = workCtx.wasmFolder
		if wasmAppFolderOf(rootAbs, folderRel) {
			fmt.Printf("Reading the app as the wasm app in %s — the last wasm app this session touched.\n", folderRel)
		}
	default:
		printClarifyIntro(tag, "Not explaining a wasm app yet: the sentence doesn't name the app's folder — which folder does the app live in?")
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				return runExplainWasmApp("explain the wasm app in "+text, ask2)
			},
		})
		recordBrain("composer", false, "wasm explain question: folder unnamed")
		return true
	}
	abs := filepath.Join(rootAbs, filepath.FromSlash(folderRel))
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		if err == nil {
			fmt.Printf("%s Not explaining a wasm app: %s isn't a folder — a wasm app belongs to a folder, as its own module.\n", tag, folderRel)
		} else {
			fmt.Printf("%s Not explaining a wasm app: there's no folder %s in this project for an app to live in — an app starts with \"add a wasm app in %s\".\n", tag, folderRel, folderRel)
		}
		recordBrain("composer", false, "wasm explain refused: folder missing")
		return true
	}
	if !wasmAppFolderOf(rootAbs, folderRel) {
		fmt.Printf("%s Not explaining a wasm app: there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then ask me again.\n", tag, folderRel, folderRel)
		recordBrain("composer", false, "wasm explain refused: no app in folder")
		return true
	}
	fmt.Printf("%s %s\n", tag, wasmAppExplanation(rootAbs, folderRel))
	recordBrain("composer", true, "wasm app explained: "+folderRel)
	return true
}
