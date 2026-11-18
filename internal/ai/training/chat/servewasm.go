package chat

// Serve the wasm app (John 2026-11-10: "I want it to be
// served by the go server" — then, on the three-ask route
// that took: make it one sentence). The pieces all existed —
// a server, the /static/ registration, the run — but serving
// an app took three asks across three claims. This claim
// composes them into the one sentence the job is:
//
//	serve the wasm app [in <folder>]
//	serve the wasm [in <folder>]
//
// The line must name wasm: a bare "serve the app" is not
// this claim, and neither are its neighbours' sentences —
// "serve static files …" stays the serve-static claim's,
// "run the server …" the run claim's, "build the wasm app …"
// the build claim's, "add a wasm app …" the creation claim's
// (all pinned, both directions).
//
// The composition, in order, each step read from the project
// as it stands:
//
//  1. The folder named in the sentence — fenced to the
//     project — or, with none named, the session's
//     last-touched wasm app (workCtx.wasmFolder), with the
//     reading printed. With neither, one clarify.go question
//     asks for the folder, the wasm family's discipline for
//     a folderless shape (the app claim's own pattern).
//  2. The folder must hold a wasm app (the analyzer's
//     recognition: main.go carrying the js && wasm tag and
//     the syscall/js import). Anything else is the honest
//     answer for what the folder IS — missing, not a folder,
//     plain pages, no app yet — each naming its own route.
//  3. The app must be BUILT: no main.wasm, no serving — the
//     answer names the build sentence and stops. Serving an
//     unbuilt app would hand the user a page whose program
//     isn't there. (Auto-building is a v2 question; v1 says
//     the step out loud.)
//  4. The route: read through analyze.ServeWasmRouteFacts —
//     this sentence's own probe. Only the project's own
//     module counts: a registration or a server inside a
//     nested module (a folder with its own go.mod) is that
//     module's code, which this project can neither build
//     nor run, so it is invisible here — the first version
//     read the project-wide scan and ran a nested module's
//     server file straight into its module wall (John's
//     screenshot, 2026-11-10). And the sentence serves from
//     exactly two homes, the arrangement that stays easy to
//     explain: the dedicated webserver in server/ at the
//     project root, or a program at the root itself. A
//     module-visible /static/ held for a different folder is
//     the serve-static planner's conflict, in its own words,
//     and nothing is written. A route already serving THIS
//     folder from one of the two homes is already-there —
//     the edit is skipped and the flow goes straight to the
//     run. A route serving this folder from some OTHER
//     program deeper in the tree is named plainly before the
//     gate and left untouched: the sentence serves from its
//     own arrangement instead. Otherwise the edit lands
//     through the one spine (plan → gate → transaction →
//     build-check → partner confirmation): the serve-static
//     edit targeted at the server/ server (or the root
//     program) when one is there, else the serve-wasm plan —
//     the house server created in server/main.go, the route
//     already in it. After the spine, the facts are re-read:
//     the route serves this folder from a home, or the spine
//     already said why not and the flow stops — the run only
//     ever follows a registration that is really there.
//  5. The run: the serving file's program, through the run
//     claim's own machinery (runProgramClaim — its gate, its
//     build, its startup window, its stop shapes) — never a
//     second runner. The claim's final line is rendered from
//     the session program's observed state and gives the
//     app's address, http://localhost:<port>/static/, with
//     the port read from the serving file's own serve
//     statement (or the runner's confirmed URL when it has
//     one) — never an assumed port, and no address at all
//     when the run didn't stay up.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// parseServeWasmApp reads the serve sentence: "serve", an
// optional article, the wasm word ("wasm" or "webassembly",
// the family's vocabulary), then "app" — or, for the short
// form, no "app" at all — then an optional folder tail in
// the family's shapes (in / into / at, an optional "the", an
// optional "folder" or "directory" word, then one folder
// token). The build sentence's parser (wasmbuild.go) with
// the verb changed — the two sentences are one shape in two
// verbs, and their grammars should not drift apart. A
// file-shaped tail (a .go or .html name) is not this
// sentence: an app is named by its folder. Politeness
// wrappers come off first (stripProgramSofteners, the run
// claim's closed set).
func parseServeWasmApp(line string) (folder string, hasFolder, ok bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return "", false, false
	}
	fold := func(t string) string { return strings.ToLower(t) }
	fields = stripProgramSofteners(fields, fold)
	if len(fields) < 2 {
		return "", false, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = fold(f)
	}
	if low[0] != "serve" {
		return "", false, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	if i >= len(fields) || (low[i] != "wasm" && low[i] != "webassembly") {
		return "", false, false
	}
	i++
	if i < len(fields) && low[i] == "app" {
		i++
	}
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

// serveWasmAppCanonical renders the sentence with its folder
// named — the clarify completion's rebuild, the app claim's
// canonical() pattern.
func serveWasmAppCanonical(folder string) string {
	return "serve the wasm app in " + folder
}

// tryServeWasmApp is the chat-loop entry: it claims
// serve-the-wasm-app sentences and runs the flow on the
// chat's line reader.
func tryServeWasmApp(line string, r *LineReader, profile *UserProfile) bool {
	return runServeWasmApp(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	}, profile)
}

// runServeWasmApp carries one line for this claim. It returns
// false — the line untouched, nothing printed — for every
// line outside the claim's shapes; a shaped line is claimed
// (true) whether it ends served, refused, or asked about.
func runServeWasmApp(line string, ask editAsk, profile *UserProfile) bool {
	folder, hasFolder, ok := parseServeWasmApp(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [runner]>")
	rootRaw, rootAbs, rok := staticAssetsRoot(tag, "wasm serve")
	if !rok {
		return true
	}

	// The folder: named (fenced), the session's, or asked.
	var folderRel string
	switch {
	case hasFolder:
		_, rel, refusal := fenceResolve(rootAbs, folder)
		if refusal != "" {
			fmt.Printf("%s Not serving the wasm app: %s.\n", tag, strings.TrimSuffix(refusal, "."))
			recordBrain("runner", false, "wasm serve refused: folder outside project")
			return true
		}
		folderRel = rel
	case workContextLive(rootAbs) && workCtx.wasmFolder != "":
		folderRel = workCtx.wasmFolder
		if wasmAppFolderOf(rootAbs, folderRel) {
			fmt.Printf("Reading the app as the wasm app in %s — the last wasm app this session touched.\n", folderRel)
		}
	default:
		printClarifyIntro(tag, "Not serving the wasm app yet: the sentence doesn't name the app's folder — which folder does the app live in?")
		printClarifyFooter("Answer with the folder — \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				return runServeWasmApp(serveWasmAppCanonical(text), ask2, profile)
			},
		})
		recordBrain("runner", false, "wasm serve question: folder unnamed")
		return true
	}

	// The folder must hold a wasm app; every other state is
	// the honest answer for what it is.
	abs := filepath.Join(rootAbs, filepath.FromSlash(folderRel))
	if st, err := os.Stat(abs); err != nil || !st.IsDir() {
		if err == nil {
			fmt.Printf("%s Not serving the wasm app: %s isn't a folder — a wasm app belongs to a folder, as its own module.\n", tag, folderRel)
		} else {
			fmt.Printf("%s Not serving the wasm app: there's no folder %s in this project for the app to live in — say \"create a folder %s\" first, then \"add a wasm app in %s\", then ask again.\n", tag, folderRel, folderRel, folderRel)
		}
		recordBrain("runner", false, "wasm serve refused: folder missing")
		return true
	}
	if !wasmAppFolderOf(rootAbs, folderRel) {
		if folderHasHTML(rootAbs, folderRel) {
			fmt.Printf("%s Not serving a wasm app: the pages in %s are plain HTML, not a wasm app — pages are served by the file serving instead: \"serve static files from %s\" puts them on your server. An app starts with \"add a wasm app in %s\".\n", tag, folderRel, folderRel, folderRel)
		} else {
			fmt.Printf("%s Not serving the wasm app: there's no wasm app in %s yet — say \"add a wasm app in %s\" first, then ask again.\n", tag, folderRel, folderRel)
		}
		recordBrain("runner", false, "wasm serve refused: no app in folder")
		return true
	}

	// The app must be built: no main.wasm, no serving.
	if _, err := os.Stat(filepath.Join(abs, "main.wasm")); err != nil {
		fmt.Printf("%s Not serving the wasm app yet: the app in %s isn't built — there's no main.wasm in it, and a served page whose program isn't there can't start. Build it first — \"build the wasm app in %s\" — then ask me again.\n", tag, folderRel, folderRel)
		recordBrain("runner", false, "wasm serve refused: app not built")
		return true
	}

	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "wasm serve") {
		return true
	}

	// The route's state, before anything is planned — read
	// the serve sentence's own way (ServeWasmRouteFacts):
	// only this project's own module counts (a registration
	// inside a nested module's folder is another module's
	// code, invisible here), and the sentence serves from
	// exactly two homes: the dedicated server in server/,
	// or a program at the project root.
	facts := lastAnalyzeProject.ServeWasmRouteFacts(folderRel)
	if facts.Conflict != "" {
		fmt.Printf("%s Not serving the wasm app in %s: %s\n", tag, folderRel, facts.Conflict)
		recordBrain("runner", false, "wasm serve refused: /static/ taken")
		return true
	}
	serveFile := facts.ServedFile
	if serveFile == "" {
		// Find-or-make the arrangement, through the one
		// spine: extend the server in server/ (or the
		// root-level program) when one is there; create the
		// server in server/ when neither home has one.
		extendFile := facts.ServerFile
		if extendFile == "" {
			extendFile = facts.RootFile
		}
		if facts.OtherFile != "" {
			// Another program deeper in the tree also
			// registers /static/ for this app. It is named
			// plainly, before the gate, and left alone: this
			// sentence serves the app from its own
			// arrangement instead.
			from := "server/"
			if extendFile != "" {
				from = extendFile
			}
			fmt.Printf("%s %s also registers /static/ for %s in its own program; this sentence serves it from %s instead.\n", tag, facts.OtherFile, folderRel, from)
		}
		if extendFile != "" {
			executeEditPlan(analyze.EditRequest{
				Kind:           analyze.EditServeStatic,
				StaticFolder:   folderRel,
				StaticSiteFile: extendFile,
			}, ask)
		} else {
			executeEditPlan(analyze.EditRequest{
				Kind:       analyze.EditServeWasmApp,
				WasmFolder: folderRel,
			}, ask)
		}
		// Re-read the facts: the run only follows a
		// registration that is really there, in one of the
		// two homes. A refusal, a gate-out, or a rollback
		// has already been narrated by the spine; this flow
		// just stops.
		facts = lastAnalyzeProject.ServeWasmRouteFacts(folderRel)
		serveFile = facts.ServedFile
		if serveFile == "" {
			recordBrain("runner", false, "wasm serve stopped: route not in place")
			return true
		}
	}

	// The run — the run claim's own machinery, gate, build,
	// and startup narration included.
	runProgramClaim(serveFile, ask, profile)

	// The final line, from the session program's observed
	// state: the serving file's program, still running. A
	// declined gate, a program that ended inside its
	// startup window, or a different program owning the
	// session has already been narrated by the runner; this
	// claim adds nothing to those stories.
	p := sessionProgram
	if p == nil || p.rel != serveFile || p.hasExited() {
		return true
	}
	base := p.confirmedURL()
	if base == "" {
		if port := serveStatementPort(rootAbs, serveFile); port != "" {
			base = "http://localhost:" + port
		}
	}
	if base != "" {
		fmt.Printf("%s The wasm app in %s is served by your Go server — open %s/static/ in your browser to run it.\n", tag, folderRel, base)
	} else {
		fmt.Printf("%s The wasm app in %s is served at /static/ on the server in %s.\n", tag, folderRel, serveFile)
	}
	recordBrain("runner", true, "wasm app served: "+folderRel+" via "+serveFile)
	// The first successful serve of this app in this
	// project carries the folder tour (wasmexplain.go),
	// once per app — the build's first run and the
	// serve's share the one memory, so a built-then-
	// served app is toured exactly once, at the build.
	maybeExplainWasmAppFirstRun(tag, rootAbs, folderRel)
	return true
}

// serveStatementPort reads the port out of the serving
// file's own ListenAndServe address — ":8080" and
// "localhost:9000" both give their port. "" when the file
// names no literal address (or no literal port in it): the
// caller falls back rather than assuming one.
func serveStatementPort(rootAbs, rel string) string {
	src, err := os.ReadFile(filepath.Join(rootAbs, filepath.FromSlash(rel)))
	if err != nil {
		return ""
	}
	m := listenAddrRe.FindSubmatch(src)
	if m == nil {
		return ""
	}
	addr := string(m[1])
	i := strings.LastIndex(addr, ":")
	if i < 0 {
		return ""
	}
	port := addr[i+1:]
	if port == "" {
		return ""
	}
	for _, c := range port {
		if c < '0' || c > '9' {
			return ""
		}
	}
	return port
}
