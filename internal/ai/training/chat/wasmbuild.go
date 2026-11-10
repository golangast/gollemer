package chat

// Build the wasm app (John 2026-11-10, "how do I ask it to
// build the wasm"): the wasm piece creates an app and the
// events piece wires it, but until this claim the chat had
// no sentence that BUILDS the app — the user had to leave
// the chat and run the documented command by hand. This
// claim closes that loop:
//
//	build the wasm app [in <folder>]
//	build the wasm [in <folder>]
//
// The line must name wasm: a bare "build the app" is not
// this claim (the concept book owns the other "build a …"
// lines), and "build" must be the verb — add/create/make
// stay pure creation, below.
//
// One collision shapes the whole design: "build" is ALSO
// the create claim's verb ("build a wasm app in jim" is a
// shipped creation sentence, parseWasmApp), so this claim
// cannot own its sentence by shape alone. Resolution is
// the disambiguator — the run claim's server rule
// (runServerClaim), applied here: the claim fires only
// when the folder it resolves ALREADY holds a wasm app
// (the analyzer's recognition: main.go carrying the
// js && wasm tag and the syscall/js import). Everywhere
// else it declines, silently, and the line falls through
// to the create claim exactly as it does today — a missing
// folder meets the family's two-step there, an app-less
// folder gets its app created there, and the one behaviour
// that changes is the former dead end: an app in place and
// "build the wasm app in jim" used to meet the create
// plan's already-there refusal; now it compiles the app.
// Nothing runs on a miss, ever: the build starts only
// after the folder verifies as an app and the gate says
// yes.
//
// The folder resolves the events piece's way (wasmevent.go):
// named in the sentence — fenced to the project — or, with
// none named, the session's last-touched wasm app
// (workCtx.wasmFolder), with the reading printed before
// anything runs. With neither, the claim declines and the
// create claim's folder question owns the folderless
// sentence, as today.
//
// The build itself is the shared command and nothing
// else: analyze.WasmBuildCommand, the one string the host
// page's comment, the app plan, and the event notes all
// state — run as an argv ("go build -o main.wasm .") with
// the command's env vars (GOEXPERIMENT=simd, GOOS=js,
// GOARCH=wasm) in the app folder, no shell anywhere, the
// run claim's discipline (buildProgram) throughout: the
// [y/n] gate prints the exact command, the compiler's own
// output is the failure report, and success is narrated
// from the artifact's own stat — main.wasm exists, is new
// from this run, and how big it is. A repeat build over an
// existing main.wasm is normal: rebuilding after an edit
// is the point, so there is no already-there refusal. The
// app folder's own go.mod pins Go 1.27, so a first build
// fetches that toolchain by itself; when the machine can't
// (no network, no toolchain), the compiler's error is the
// answer, surfaced as-is. The gate and narration wear the
// runner's tag: this claim executes, it does not land an
// edit, so the partner confirmation does not apply.

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// wasmBuildTimeout bounds one build. The run claim's
// builds cap at two minutes; this one gets five, because a
// first build on a fresh machine also fetches the app's
// pinned Go 1.27 toolchain — part of this build's
// documented life (the app plan says so in the open).
const wasmBuildTimeout = 5 * time.Minute

// parseBuildWasmApp reads the build sentence: "build",
// an optional article, the wasm word ("wasm" or
// "webassembly", the create grammar's vocabulary), then
// "app" — or, for the short form, no "app" at all — then
// an optional folder tail in the family's shapes (in /
// into / at, an optional "the", an optional "folder" or
// "directory" word, then one folder token). A file-shaped
// tail (a .go or .html name) is not this sentence: an app
// is named by its folder. Politeness wrappers come off
// first (stripProgramSofteners, the run claim's closed
// set) — "can you build the wasm app in jim" is the same
// ask. This parse claims by shape only; whether the
// sentence FIRES is the resolver's call (see the file
// comment), never the parser's.
func parseBuildWasmApp(line string) (folder string, hasFolder, ok bool) {
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
	if low[0] != "build" {
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

// tryBuildWasmApp is the chat-loop entry: it claims wasm
// build sentences and runs the flow on the chat's line
// reader.
func tryBuildWasmApp(line string, r *LineReader, profile *UserProfile) bool {
	return runBuildWasmApp(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	}, profile)
}

// runBuildWasmApp carries one line for this claim. It
// returns false — the line untouched, nothing printed —
// for every line outside the claim's shapes AND for every
// shape whose folder does not already hold a wasm app:
// resolution is part of the claim (see the file comment),
// and on a miss the create claim below owns the sentence
// exactly as it did before this claim existed.
func runBuildWasmApp(line string, ask editAsk, profile *UserProfile) bool {
	folder, hasFolder, ok := parseBuildWasmApp(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [runner]>")
	_, rootAbs, rok := staticAssetsRoot(tag, "wasm build")
	if !rok {
		return true
	}
	var folderRel string
	if hasFolder {
		_, rel, refusal := fenceResolve(rootAbs, folder)
		if refusal != "" {
			// Outside the project: not this claim's build.
			// The create claim prints the family's fence
			// refusal for the same line, as today.
			return false
		}
		folderRel = rel
	} else {
		if !workContextLive(rootAbs) || workCtx.wasmFolder == "" || !wasmAppFolderOf(rootAbs, workCtx.wasmFolder) {
			// No app in the session to read the sentence
			// against: the create claim's folder question
			// owns the folderless sentence, as today.
			return false
		}
		folderRel = workCtx.wasmFolder
		fmt.Printf("Reading the app as the wasm app in %s — the last wasm app this session touched.\n", folderRel)
	}
	if !wasmAppFolderOf(rootAbs, folderRel) {
		return false
	}
	buildWasmAppClaim(tag, rootAbs, folderRel, ask, profile)
	return true
}

// buildWasmAppClaim gates and runs the one build for a
// folder already verified to hold a wasm app, and narrates
// the outcome from observed facts only: the gate's answer,
// the compiler's exit and output, and main.wasm's own stat
// afterwards.
func buildWasmAppClaim(tag, rootAbs, folderRel string, ask editAsk, profile *UserProfile) {
	appAbs := filepath.Join(rootAbs, filepath.FromSlash(folderRel))
	wasmAbs := filepath.Join(appAbs, "main.wasm")
	var before time.Time
	if st, err := os.Stat(wasmAbs); err == nil {
		before = st.ModTime()
	}
	prompt := "Build it? " + analyze.WasmBuildCommand + " (in " + folderRel + ")"
	if !runProgramGate(ask, profile, prompt) {
		fmt.Printf("%s OK — not built.\n", tag)
		recordBrain("runner", true, "wasm build declined: "+folderRel)
		return
	}
	out, err, timedOut := execWasmBuild(appAbs)
	if err != nil {
		fmt.Printf("%s %s\n", tag, wasmBuildFailText(folderRel, out, timedOut))
		recordBrain("runner", false, "wasm build failed: "+folderRel)
		return
	}
	if trimmed := strings.TrimSpace(out); trimmed != "" {
		// A successful build is usually silent; when it
		// speaks (a toolchain fetching itself, a note from
		// the compiler), its words print as they came.
		printProgramLines(strings.Split(strings.TrimRight(out, "\n"), "\n"))
	}
	st, serr := os.Stat(wasmAbs)
	if serr != nil {
		fmt.Printf("%s The build command finished without an error, but I can't find main.wasm in %s — so I can't say the app built.\n", tag, folderRel)
		recordBrain("runner", false, "wasm build unverified: "+folderRel)
		return
	}
	fresh := "is in place"
	if before.IsZero() || st.ModTime().After(before) {
		fresh = "is new from this build"
	}
	fmt.Printf("%s Built — the wasm app in %s compiled: %s/main.wasm %s, %s bytes. Serve the folder and open its page to run it — \"serve static files\" points this project's server at a folder.\n",
		tag, folderRel, folderRel, fresh, groupThousands(st.Size()))
	recordBrain("runner", true, "wasm app built: "+folderRel)
}

// execWasmBuild runs the shared build command in the app
// folder: the argv and env the documented command names,
// no shell, output combined — buildProgram's shape. The
// third return reports the timeout kill apart from an
// ordinary compiler failure, because the two are different
// stories and the narration says which one happened.
func execWasmBuild(appAbs string) (string, error, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), wasmBuildTimeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "build", "-o", "main.wasm", ".")
	cmd.Dir = appAbs
	cmd.Env = append(os.Environ(), "GOEXPERIMENT=simd", "GOOS=js", "GOARCH=wasm")
	out, err := cmd.CombinedOutput()
	return string(out), err, ctx.Err() == context.DeadlineExceeded
}

// wasmBuildFailText is the failure report: the build's own
// outcome first, then the compiler's own output, verbatim
// up to the run claim's readable cap (programBuildFailText
// is the model). The word "built" appears nowhere in it —
// nothing built.
func wasmBuildFailText(folderRel, out string, timedOut bool) string {
	var b strings.Builder
	if timedOut {
		fmt.Fprintf(&b, "It didn't build — the build in %s ran past %s and I stopped it. What it printed before then:\n", folderRel, wasmBuildTimeout)
	} else {
		fmt.Fprintf(&b, "It didn't build — the wasm app in %s doesn't compile yet. The compiler's own words:\n", folderRel)
	}
	lines := strings.Split(strings.TrimRight(out, "\n"), "\n")
	if strings.TrimSpace(out) == "" {
		lines = nil
	}
	const maxLines = 40
	shown := lines
	if len(shown) > maxLines {
		shown = shown[:maxLines]
	}
	for _, l := range shown {
		b.WriteString(l + "\n")
	}
	if len(lines) > maxLines {
		fmt.Fprintf(&b, "[… %d more lines not shown]\n", len(lines)-maxLines)
	}
	if len(lines) == 0 {
		b.WriteString("(it printed nothing)\n")
	}
	fmt.Fprintf(&b, "Fix those, then say \"build the wasm app in %s\" again.", folderRel)
	return strings.TrimRight(b.String(), "\n")
}

// groupThousands renders a byte count the way the house
// reports sizes: 2,734,961.
func groupThousands(n int64) string {
	s := fmt.Sprintf("%d", n)
	if len(s) <= 3 {
		return s
	}
	var parts []string
	for len(s) > 3 {
		parts = append([]string{s[len(s)-3:]}, parts...)
		s = s[:len(s)-3]
	}
	return strings.Join(append([]string{s}, parts...), ",")
}
