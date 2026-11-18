package chat

// "Add a webserver to…" — the sixth assembled edit shape, claimed
// deterministically in the unified loop beside the other
// write-side parsers. It exists because of a live dead end: with
// jim/main.go already on disk as a bare package clause, "add a
// webserver to the jim/main.go" fell through every write parser
// to the intent layer, whose analyze case answered "I couldn't
// read that codebase: jim/main.go is not a directory", and "add
// a webserver to jim/" got a full analyze report dumped on it.
// Both are edits, and a sentence that would write to the user's
// files is never left for a classifier to guess at.
//
// Grammar (exactly this):
//
//	add|put|make|create|build [a|an|the] webserver|web server|server
//	    to|in|into [the] [folder|file] <target>
//
// The verb family (make/create/build beside add/put) and the
// bare noun "server" were widened 2026-11-05: "make a
// webserver in <file>" and "add a server to <file>" are the
// beginner's spellings of this ask, and both parsed nowhere —
// the noun because this grammar insisted on the compound
// word, the verbs because the slot stopped at add|put. The
// widenings steal nothing: a line reaches the noun only
// behind one of the family's verbs, the preposition must
// follow the noun immediately (so serve-an-html-file's "build
// a server that serves…" still dies here, at the preposition),
// and the creations keep their own sentences — "create a file
// … with a webserver" names a file before any server word.
//
// The folder|file word is a designator — it names the KIND of
// target, never part of its path (creations read those words the
// same way). The second live failure of this feature was exactly
// that word: "add webserver to folder jim" kept "folder" in the
// target, resolved a path that doesn't exist, and refused while
// recommending a create command with "folder" baked into it —
// the server never got added. A designator with NO name after it
// still claims the line (parse returns ok with an empty target)
// so the flow can ask which file or folder, in webserver terms,
// instead of letting the sentence fall through to the analyzer.
//
// <target> is a .go file or a folder, resolved inside the same
// root the creations use (the open project, else the folder the
// chat started in — never outside it). A folder means its
// main.go, or its single package-main file when that is
// unambiguous. The plan itself is the composer's
// (analyze.EditAddServer): the shared webserver component
// spliced by file state, with honest refusals for a non-empty
// main, a library package, and a file that already serves. The
// gate, build, verify, and restore are the PlanEdit spine's,
// unchanged (executeEditPlan).

import (
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// parseAddServer reads one chat line as an add-a-webserver ask
// and returns the target as typed. The second return is false
// for anything outside the grammar — including the resources
// claim's "i want to build a webserver" (no add verb, no target)
// and the creations' "create a file …" (their verb, their claim).
// A line that matches the grammar but names nothing after the
// designator ("add a webserver to folder") returns ("", true):
// claimed, so the caller answers in webserver terms instead of
// the line falling through to the intent layer.
func parseAddServer(line string) (string, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 4 {
		return "", false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	switch low[0] {
	case "add", "put", "make", "create", "build":
	default:
		return "", false
	}
	i := 1
	// The determiner before the component's name: a, an, or
	// the. The slot's third live miss (2026-10-25, minutes
	// after the designator fix shipped) was "the": "add the
	// webserver to jim/main.go" fell through to the analyzer
	// while its "a" twin worked, because this slot read only
	// a/an. Widening it steals nothing — a line reaches the
	// webserver check only when the word after the determiner
	// IS the component's name, and no other claim owns an
	// "add the webserver…" sentence.
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	switch {
	case low[i] == "webserver" || low[i] == "server":
		i++
	case low[i] == "web" && i+1 < len(fields) && low[i+1] == "server":
		i += 2
	default:
		return "", false
	}
	if i >= len(fields) || (low[i] != "to" && low[i] != "in" && low[i] != "into") {
		return "", false
	}
	i++
	if i >= len(fields) {
		return "", false
	}
	// The target half: an optional article, an optional
	// designator (the kind word — creations' grammar slot, never
	// part of the path), then the target as typed.
	consumed := false
	if low[i] == "the" {
		consumed = true
		i++
	}
	if i < len(fields) {
		switch strings.TrimSuffix(low[i], "/") {
		case "folder", "directory", "file":
			consumed = true
			i++
		}
	}
	if i >= len(fields) {
		// Words followed the preposition but none of them named
		// a target ("…to folder", "…to the"): the shape is this
		// claim's, so it answers — it never falls through.
		return "", consumed
	}
	// The target is ONE path token, as in the concept grammar:
	// a longer tail is another ask glued behind this one ("add
	// a webserver to jim/main.go and add a for loop …"), and
	// joining it whole would let a compound be refused here as
	// a single server ask naming a file nobody typed.
	// Multi-piece lines are not this claim's.
	if len(fields[i:]) != 1 {
		return "", false
	}
	return fields[i], true
}

// serverTargetResolve turns a typed target into an absolute path
// guaranteed to sit inside rootAbs, or an honest refusal. It is
// the creations fence's containment logic (clean, prefix check,
// symlink-followed ancestors) with edit wording: the fence's own
// messages say "create", and an edit path shouldn't.
func serverTargetResolve(rootAbs, typed string) (abs, rel, refusal string) {
	if strings.ContainsAny(typed, "*?[") {
		return "", "", fmt.Sprintf("%q has wildcard characters in it — name one exact file", typed)
	}
	var relOS string
	if filepath.IsAbs(typed) {
		r, err := filepath.Rel(rootAbs, filepath.Clean(typed))
		if err != nil {
			return "", "", fmt.Sprintf("%q isn't inside %s — I only edit files inside the project folder", typed, rootAbs)
		}
		relOS = r
	} else {
		relOS = filepath.Clean(typed)
	}
	if relOS == "." || relOS == ".." || strings.HasPrefix(relOS, ".."+string(filepath.Separator)) {
		return "", "", fmt.Sprintf("%q isn't inside %s — I only edit files inside the project folder", typed, rootAbs)
	}
	abs = filepath.Join(rootAbs, relOS)
	anc := abs
	var tail []string
	for {
		if _, err := os.Lstat(anc); err == nil {
			break
		}
		parent := filepath.Dir(anc)
		if parent == anc {
			break
		}
		tail = append([]string{filepath.Base(anc)}, tail...)
		anc = parent
	}
	if resolved, err := filepath.EvalSymlinks(anc); err == nil {
		abs = resolved
		for _, seg := range tail {
			abs = filepath.Join(abs, seg)
		}
	}
	if abs == rootAbs || !strings.HasPrefix(abs, rootAbs+string(filepath.Separator)) {
		return "", "", fmt.Sprintf("%q ends up outside %s once links are followed — I only edit files inside the project folder", typed, rootAbs)
	}
	return abs, filepath.ToSlash(relOS), ""
}

// packageMainFiles lists the non-test .go files in dir whose
// package clause is main, in ReadDir (sorted) order.
func packageMainFiles(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		pf, err := parser.ParseFile(token.NewFileSet(), filepath.Join(dir, name), nil, parser.PackageClauseOnly)
		if err == nil && pf.Name != nil && pf.Name.Name == "main" {
			out = append(out, name)
		}
	}
	return out
}

// resolveServerFile settles a typed target to the project-relative
// .go file the component goes into, or an honest refusal naming
// the working alternative. A folder resolves to its main.go, else
// its one package-main file; no suitable file and several
// candidates are both refusals (v1 builds no picker).
func resolveServerFile(rootAbs, target string) (rel, refusal string) {
	disp := strings.TrimSuffix(target, "/")
	abs, rel, refusal := serverTargetResolve(rootAbs, target)
	if refusal != "" {
		return "", refusal
	}
	st, err := os.Stat(abs)
	if err != nil {
		sug := disp
		if !strings.HasSuffix(strings.ToLower(disp), ".go") {
			sug = disp + "/main.go"
		}
		return "", fmt.Sprintf("there's no %s in this project — if you meant a brand-new server file, say \"create a file %s with a webserver\" and I'll write the whole thing", disp, sug)
	}
	if st.IsDir() {
		if fi, err := os.Stat(filepath.Join(abs, "main.go")); err == nil && !fi.IsDir() {
			return rel + "/main.go", ""
		}
		cands := packageMainFiles(abs)
		switch len(cands) {
		case 1:
			return rel + "/" + cands[0], ""
		case 0:
			return "", fmt.Sprintf("%s/ has no main program to add a server to — the working move is \"create a file %s/main.go with a webserver\"", disp, disp)
		default:
			return "", fmt.Sprintf("%s/ has more than one main-package file (%s) — say which one, like \"add a webserver to %s/%s\"", disp, strings.Join(cands, ", "), disp, cands[0])
		}
	}
	if !strings.HasSuffix(strings.ToLower(target), ".go") {
		return "", fmt.Sprintf("%s isn't a Go file — a webserver goes into a .go file in package main", disp)
	}
	return rel, ""
}

// serverTargetMulti reports an add-webserver folder
// target as a several-host choice, under exactly the
// conditions resolveServerFile refuses as "multi": an
// existing folder with no main.go and two or more
// package-main files. Nil for every other target.
func serverTargetMulti(rootAbs, target string) *targetMulti {
	disp := strings.TrimSuffix(target, "/")
	abs, rel, refusal := serverTargetResolve(rootAbs, target)
	if refusal != "" {
		return nil
	}
	st, err := os.Stat(abs)
	if err != nil || !st.IsDir() {
		return nil
	}
	if fi, err := os.Stat(filepath.Join(abs, "main.go")); err == nil && !fi.IsDir() {
		return nil
	}
	cands := packageMainFiles(abs)
	if len(cands) < 2 {
		return nil
	}
	m := &targetMulti{disp: disp}
	for _, c := range cands {
		m.cands = append(m.cands, rel+"/"+c)
	}
	return m
}

// tryAddWebServer is the chat-loop entry: it claims
// add-a-webserver sentences and runs the flow on the chat's line
// reader.
func tryAddWebServer(line string, r *LineReader) bool {
	return runAddServer(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runAddServer carries one line from sentence to resolved target
// to the shared edit spine. It returns true when the line
// belonged to this claim at all.
func runAddServer(line string, ask editAsk) bool {
	target, ok := parseAddServer(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if target == "" {
		// The grammar matched but named nothing ("add a
		// webserver to folder"): claimed, and asked in
		// webserver terms (clarify.go) — the answer, one
		// bare file or folder, completes the ask as its
		// canonical sentence. This sentence must never reach
		// the intent layer, where a path-less line earns a
		// full analyze report.
		printClarifyIntro(tag, "Not adding that yet: the sentence doesn't name the file or folder — where should the webserver go?")
		printClarifyFooter("Answer with a file or folder — \"jim/main.go\", or \"jim\"")
		setPendingQuestion(&pendingQuestion{
			kind: clarifyTarget,
			completeText: func(text string, ask2 editAsk) bool {
				return runAddServer("add a webserver to "+text, ask2)
			},
		})
		recordBrain("composer", false, "add-server question: no target named")
		return true
	}
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "add-server refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "add-server refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "add-server refused: root missing")
		return true
	}
	// A folder with several main-package files is a choice,
	// not a fact: ask which one (clarify.go) instead of only
	// refusing. The resolver below keeps every other target
	// outcome, in its own words.
	if multi := serverTargetMulti(rootAbs, target); multi != nil {
		cands := make([]clarifyCandidate, len(multi.cands))
		for i, c := range multi.cands {
			cands[i] = clarifyCandidate{Label: c, Words: []string{filepath.Base(c)}}
		}
		askPickQuestion(tag,
			fmt.Sprintf("%s/ has more than one main-package file — which one should the webserver go in?", multi.disp),
			"Answer with the path or the number",
			cands,
			func(idx int, ask2 editAsk) bool {
				if spliced, ok := spliceTypedTarget(line, target, multi.cands[idx]); ok {
					return runAddServer(spliced, ask2)
				}
				_, refusal := resolveServerFile(rootAbs, target)
				fmt.Printf("%s Not adding that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
				return true
			})
		recordBrain("composer", false, "add-server question: folder has several main files")
		return true
	}
	rel, refusal := resolveServerFile(rootAbs, target)
	if refusal != "" {
		fmt.Printf("%s Not adding that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "add-server refused: target")
		return true
	}
	// The spine plans against the analyzed project; make the
	// target's root the current one, quietly. (Not ensureProject:
	// its demo fallback would aim the edit at a stand-in project.)
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "add-server refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "add-server refused: stale project unreadable")
		return true
	}
	return executeEditPlan(analyze.EditRequest{Kind: analyze.EditAddServer, File: rel}, ask)
}
