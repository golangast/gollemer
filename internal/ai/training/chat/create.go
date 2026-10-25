package chat

// Simple creations, claimed deterministically in the unified loop
// beside the assembled edits and renames: a sentence that would
// write to the user's disk is never left for the intent classifier
// or a neural model to guess at (the classifier read "create a
// folder jim" as a social greeting for weeks).
//
// The v1 grammar is deliberately small:
//
//	create|make [a] folder|directory [named|called] <name>
//	create|make [a] file [named|called] [<file>]
//	    [in folder <dir>] [with a <kind>]
//
// <kind> has exactly one member today: a webserver — a small
// working net/http program from the fixed template below. A .go
// file with no kind gets just its package clause (the sibling
// package's name when the folder is an existing Go package, else
// package main); any other file starts empty.
//
// Everything rides the one write gate every other write uses: the
// exact plan — every path, and a file's full content — prints
// first, then [y/n]. Nothing is written before a yes. The fence:
// targets resolve inside the creation root (the open project, else
// the folder the chat started in), symlinks are followed before
// the check so a linked folder cannot smuggle a target outside,
// existing targets are refused (never overwrite, never merge), and
// one operation happens per ask. After a yes, a .go file is
// compiled — the package it landed in, or the file itself when no
// module owns it — and a failed check removes exactly what this
// operation created (nothing pre-existed; the fence proved it).
// One exception, honestly made: a kindless file whose whole
// content is a bare "package main" clause is a starting point,
// not a program — linking it could only ever say main() is
// undeclared — so its check is a full parse, not a build.
//
// Creation is done with os calls only, and verification builds run
// as exec argument lists like the edit spine's — no shell, ever.
//
// The wider action family that is NOT creation — delete, move,
// disk rename, install, download — is claimed here too, but only
// to answer honestly: what Gollemer can't do, the exact terminal
// command, and the nearest real capability. It never performs
// them. Code-symbol renames ("rename Foo to Bar") stay with the
// rename feature: the boundary shapes below only fire on
// disk-shaped asks (a file/folder/directory word, or a path-like
// target).

import (
	"context"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/pkg/termui"
)

// --- parsing: creations ---

// createReq is one parsed creation ask. refusal is set when the
// line is unmistakably a creation ask but not one v1 can honestly
// build (no file name, an unknown starter kind, a webserver asked
// into a non-.go name): the claim stands and the refusal is said
// out loud instead of the line falling through to a brain.
type createReq struct {
	dir     bool   // folder ask (false = file ask)
	name    string // folder name as typed (folder asks)
	file    string // file name as typed (file asks)
	loc     string // folder the file goes in, as typed; "" = root
	kind    string // "" or "webserver"
	refusal string
}

// parseCreate reads one chat line as a creation ask. The second
// return is false for anything outside the grammar above —
// including the resource claim's goal statements ("i want to
// build a webserver": no create verb, no concrete target) and the
// composer's imperatives ("build api for /…").
func parseCreate(line string) (createReq, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return createReq{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	if low[0] != "create" && low[0] != "make" {
		return createReq{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" {
		i++
	}
	if i >= len(fields) {
		return createReq{}, false
	}
	switch low[i] {
	case "folder", "directory":
		i++
		if i < len(fields) && (low[i] == "named" || low[i] == "called") {
			i++
		}
		if i >= len(fields) {
			return createReq{}, false
		}
		return createReq{dir: true, name: strings.Join(fields[i:], " ")}, true
	case "file":
		i++
		if i < len(fields) && (low[i] == "named" || low[i] == "called") {
			i++
		}
		req := createReq{}
		// Optional file name: the next token, unless a clause
		// opens first ("create a file in folder jim with a
		// webserver" names no file — the kind supplies one).
		if i < len(fields) && low[i] != "in" && low[i] != "with" {
			req.file = fields[i]
			i++
		}
		seenLoc, seenKind := false, false
		for i < len(fields) {
			switch {
			case low[i] == "in" && !seenLoc:
				seenLoc = true
				i++
				if i < len(fields) && low[i] == "the" {
					i++
				}
				if i >= len(fields) || low[i] != "folder" {
					return createReq{}, false
				}
				i++
				start := i
				for i < len(fields) && low[i] != "with" {
					i++
				}
				if start == i {
					return createReq{}, false
				}
				req.loc = strings.Join(fields[start:i], "/")
			case low[i] == "with" && !seenKind:
				seenKind = true
				i++
				if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
					i++
				}
				start := i
				for i < len(fields) && low[i] != "in" {
					i++
				}
				switch kindPhrase := strings.Join(low[start:i], " "); kindPhrase {
				case "webserver", "web server", "http server":
					req.kind = "webserver"
				case "":
					return createReq{}, false
				default:
					req.refusal = fmt.Sprintf("the only starter I can put in a new file is a working webserver (a small Go program) — I can't write %q into a file for you", "a "+kindPhrase)
				}
			default:
				return createReq{}, false
			}
		}
		if req.refusal == "" {
			switch {
			case req.kind == "webserver" && req.file == "":
				req.file = "main.go" // the starter's own name
			case req.file == "":
				req.refusal = `that ask needs a file name — say it like "create a file notes.txt", adding "in folder <name>" when you want it somewhere`
			case req.kind == "webserver" && !strings.HasSuffix(strings.ToLower(req.file), ".go"):
				req.refusal = fmt.Sprintf("my webserver starter is a Go program, so it needs a .go file name — %q would never run as one", req.file)
			}
		}
		return req, true
	}
	return createReq{}, false
}

// --- parsing: the boundary family ---

// boundaryReq is a disk action Gollemer will not perform, parsed
// only far enough to answer honestly: the verb, the exact terminal
// command ("" when no single honest command exists), and the
// nearest real capability.
type boundaryReq struct {
	verb string
	cmd  string
	near string
}

var boundaryFillers = map[string]bool{
	"the": true, "a": true, "an": true,
	"file": true, "folder": true, "directory": true,
}

// parseBoundary claims imperative delete/remove, move, disk
// rename, install, and download lines — to answer them, never to
// run them. Disk-shaped only: without a file/folder/directory word
// or a path-like target the line is not claimed, so "rename Foo to
// Bar" keeps its owner and questions ("how do i delete a file")
// keep theirs — a question never starts with the bare verb.
func parseBoundary(line string) (boundaryReq, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 2 {
		return boundaryReq{}, false
	}
	low := make([]string, len(fields))
	for j, f := range fields {
		low[j] = strings.ToLower(f)
	}
	hasTypeWord := false
	isFolder := false
	for _, w := range low {
		if w == "file" || w == "folder" || w == "directory" {
			hasTypeWord = true
		}
		if w == "folder" || w == "directory" {
			isFolder = true
		}
	}
	// rest is the line past the verb with leading filler words off,
	// in the user's own case.
	rest := func(from int) string {
		j := from
		for j < len(fields) && boundaryFillers[low[j]] {
			j++
		}
		return strings.Join(fields[j:], " ")
	}
	pathLike := func(tok string) bool {
		return strings.ContainsAny(tok, "/.") || strings.Contains(tok, "\\")
	}
	nearCreate := "create a file or folder here behind a [y/n] gate, and explain any command in plain words before you run it"

	switch low[0] {
	case "delete", "remove":
		target := rest(1)
		if target == "" {
			return boundaryReq{}, false
		}
		if !hasTypeWord && !pathLike(strings.Fields(target)[0]) {
			return boundaryReq{}, false
		}
		cmd := "rm " + target
		if isFolder {
			cmd = "rm -rf " + target
		}
		return boundaryReq{verb: "delete", cmd: cmd, near: nearCreate}, true
	case "move":
		// Split at the " to " the destination hangs on.
		toAt := -1
		for j := 1; j < len(fields); j++ {
			if low[j] == "to" {
				toAt = j
				break
			}
		}
		target := rest(1)
		if toAt > 1 {
			// Re-cut the target at the "to": rest() joined past it.
			head := fields[1:toAt]
			k := 0
			for k < len(head) && boundaryFillers[strings.ToLower(head[k])] {
				k++
			}
			target = strings.Join(head[k:], " ")
		}
		if target == "" || (!hasTypeWord && !pathLike(strings.Fields(target)[0])) {
			return boundaryReq{}, false
		}
		dest := ""
		if toAt >= 0 {
			dest = rest(toAt + 1)
		}
		if dest == "" {
			dest = "<new home>"
		}
		return boundaryReq{verb: "move", cmd: "mv " + target + " " + dest, near: nearCreate}, true
	case "rename":
		toAt := -1
		for j := 1; j < len(fields); j++ {
			if low[j] == "to" {
				toAt = j
				break
			}
		}
		if toAt < 0 {
			return boundaryReq{}, false
		}
		head := fields[1:toAt]
		k := 0
		for k < len(head) && boundaryFillers[strings.ToLower(head[k])] {
			k++
		}
		from := strings.Join(head[k:], " ")
		to := rest(toAt + 1)
		if from == "" || to == "" {
			return boundaryReq{}, false
		}
		if !hasTypeWord && !pathLike(from) && !pathLike(to) {
			return boundaryReq{}, false // a code name — the rename feature's line
		}
		return boundaryReq{verb: "rename", cmd: "mv " + from + " " + to,
			near: `rename a Go function, type, or field everywhere it's used — say "rename Foo to Bar"`}, true
	case "install":
		target := rest(1)
		if target == "" {
			return boundaryReq{}, false
		}
		cmd := ""
		if seg := strings.SplitN(target, "/", 2); len(seg) == 2 && strings.Contains(seg[0], ".") {
			cmd = "go install " + target + "@latest" // a Go package path
		}
		return boundaryReq{verb: "install", cmd: cmd,
			near: "explain what a package or command does, and pull up learning resources for it"}, true
	case "download":
		target := rest(1)
		if target == "" {
			return boundaryReq{}, false
		}
		cmd := ""
		if strings.Contains(target, "://") {
			cmd = "curl -LO " + target
		} else if pathLike(target) {
			cmd = "curl -O " + target
		}
		near := nearCreate
		if strings.Contains(target, "github.com") || strings.HasSuffix(target, ".git") {
			near = `fetch a repo AND open it here — say "clone <github link>" — or explain any of its code`
		}
		return boundaryReq{verb: "download", cmd: cmd, near: near}, true
	}
	return boundaryReq{}, false
}

// answer renders the boundary reply: the limit first, then the
// exact command when there is one, then the nearest real help.
func (b boundaryReq) answer() string {
	var s strings.Builder
	switch b.verb {
	case "delete":
		s.WriteString("I can't delete files or folders — deleting is yours alone.")
	case "move":
		s.WriteString("I can't move files or folders around on disk.")
	case "rename":
		s.WriteString("I can't rename files on disk — the renaming I do is Go names inside your code.")
	case "install":
		s.WriteString("I can't install software — nothing gets added to your machine from here.")
	case "download":
		s.WriteString("I can't download things for you.")
	}
	if b.cmd != "" {
		s.WriteString(" In your terminal, the command is: " + b.cmd + ".")
	}
	s.WriteString(" What I can do: " + b.near + ".")
	return s.String()
}

// --- the fence ---

// creationRoot is where creations land: the open project's root
// when one is open, else the folder the chat started in (which is
// the working directory when no project root was set).
func creationRoot() string {
	if lastAnalyzeRoot != "" {
		return lastAnalyzeRoot
	}
	return startFolder()
}

// fenceResolve turns a typed name into an absolute path guaranteed
// to sit inside rootAbs (itself already symlink-resolved), or
// returns an honest refusal. The nearest existing ancestor is
// followed through symlinks before the containment check, so a
// linked folder inside the project cannot smuggle a target out.
func fenceResolve(rootAbs, typed string) (abs, rel, refusal string) {
	if strings.ContainsAny(typed, "*?[") {
		return "", "", fmt.Sprintf("%q has wildcard characters in it — I create one thing at a time, by its exact name", typed)
	}
	var relOS string
	if filepath.IsAbs(typed) {
		r, err := filepath.Rel(rootAbs, filepath.Clean(typed))
		if err != nil {
			return "", "", fmt.Sprintf("%q isn't inside %s — I only create inside the project folder", typed, rootAbs)
		}
		relOS = r
	} else {
		relOS = filepath.Clean(typed)
	}
	if relOS == "." || relOS == ".." || strings.HasPrefix(relOS, ".."+string(filepath.Separator)) {
		return "", "", fmt.Sprintf("%q climbs out of %s — I only create inside the project folder", typed, rootAbs)
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
		return "", "", fmt.Sprintf("%q ends up outside %s once links are followed — I only create inside the project folder", typed, rootAbs)
	}
	return abs, filepath.ToSlash(relOS), ""
}

// siblingPackage reads the package clause of the Go files already
// in dir — non-test files first, then test files with the _test
// suffix off. "" when the folder holds no readable Go package.
func siblingPackage(dir string) string {
	entries, err := os.ReadDir(dir) // sorted by name: deterministic
	if err != nil {
		return ""
	}
	var files []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".go") {
			files = append(files, e.Name())
		}
	}
	clause := func(wantTest bool) string {
		for _, f := range files {
			if strings.HasSuffix(f, "_test.go") != wantTest {
				continue
			}
			pf, err := parser.ParseFile(token.NewFileSet(), filepath.Join(dir, f), nil, parser.PackageClauseOnly)
			if err == nil && pf.Name != nil {
				return pf.Name.Name
			}
		}
		return ""
	}
	if p := clause(false); p != "" {
		return p
	}
	return strings.TrimSuffix(clause(true), "_test")
}

// webserverSource is the one starter v1 can write: a small,
// working net/http server. The text itself lives with the
// composer (analyze.WebServerSource) — the same component the
// "add a webserver to…" edit splices into an existing file — so
// the two write paths can never drift into two different servers.
const webserverSource = analyze.WebServerSource

// --- the plan ---

// createItem is one path a plan will create: an absolute path for
// the os calls, and the project-relative slash form for display.
type createItem struct {
	abs string
	rel string
}

// createPlan is everything a yes will do, settled before the gate:
// the folders to make (shallow first — only ones that don't exist
// yet) and, for a file ask, the file and its full content.
type createPlan struct {
	dirs    []createItem
	file    *createItem // nil for a folder-only plan
	content []byte
	isGo    bool
	// pkg is the package clause a .go file carries; bareClause
	// marks the kindless form whose whole content is that clause.
	pkg        string
	bareClause bool
}

// planCreate settles a parsed ask against the fence and the disk.
// A non-empty refusal is an honest line to say instead of a plan.
func planCreate(req createReq, rootAbs string) (*createPlan, string) {
	if req.dir {
		abs, rel, refusal := fenceResolve(rootAbs, req.name)
		if refusal != "" {
			return nil, refusal
		}
		if _, err := os.Lstat(abs); err == nil {
			return nil, fmt.Sprintf("%s is already there — I never overwrite or merge what's yours", rel)
		}
		return &createPlan{dirs: missingDirs(rootAbs, rel)}, ""
	}

	typed := req.file
	if req.loc != "" {
		typed = req.loc + "/" + req.file
	}
	abs, rel, refusal := fenceResolve(rootAbs, typed)
	if refusal != "" {
		return nil, refusal
	}
	if _, err := os.Lstat(abs); err == nil {
		return nil, fmt.Sprintf("%s is already there — I never overwrite what's yours", rel)
	}
	plan := &createPlan{file: &createItem{abs: abs, rel: rel}}
	dirAbs := filepath.Dir(abs)
	if dirAbs != rootAbs {
		// The location: an existing folder is fine, an existing
		// file is a refusal, a missing one joins the plan.
		if st, err := os.Stat(dirAbs); err == nil {
			if !st.IsDir() {
				locRel := filepath.ToSlash(mustRel(rootAbs, dirAbs))
				return nil, fmt.Sprintf("%s is a file, not a folder — the new file can't go inside it", locRel)
			}
		} else {
			plan.dirs = missingDirs(rootAbs, filepath.ToSlash(mustRel(rootAbs, dirAbs)))
		}
	}

	isGo := strings.HasSuffix(strings.ToLower(req.file), ".go")
	plan.isGo = isGo
	switch {
	case req.kind == "webserver":
		// A webserver is a main package; dropping one into a
		// library package would not even compile where it lands.
		if sib := siblingPackage(dirAbs); sib != "" && sib != "main" {
			return nil, fmt.Sprintf("that folder is package %s — a webserver needs a main package of its own. Ask for it somewhere new, like \"create a file server.go in folder web with a webserver\"", sib)
		}
		plan.pkg = "main"
		plan.content = []byte(webserverSource)
	case isGo:
		pkg := siblingPackage(dirAbs)
		if pkg == "" {
			pkg = "main"
		}
		plan.pkg = pkg
		plan.bareClause = true
		plan.content = []byte("package " + pkg + "\n")
	default:
		plan.content = []byte{}
	}
	return plan, ""
}

func mustRel(base, target string) string {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return target
	}
	return rel
}

// missingDirs lists the not-yet-existing folders named by the
// slash-relative path rel, shallow first.
func missingDirs(rootAbs, rel string) []createItem {
	var out []createItem
	parts := strings.Split(rel, "/")
	cur := rootAbs
	for i, part := range parts {
		cur = filepath.Join(cur, part)
		if _, err := os.Stat(cur); err != nil {
			out = append(out, createItem{abs: cur, rel: strings.Join(parts[:i+1], "/")})
		}
	}
	return out
}

// planText renders the gate's plan: every path, and the file's
// full content indented under its line — nothing is written yet,
// and nothing about the yes is a surprise. An empty file says so.
func (p *createPlan) planText() string {
	var b strings.Builder
	b.WriteString(termui.SectionHeader("CREATE") + "\n")
	for _, d := range p.dirs {
		b.WriteString("  folder  " + d.rel + "/\n")
	}
	if p.file != nil {
		if len(p.content) == 0 {
			b.WriteString("  file    " + p.file.rel + "  (empty)\n")
		} else {
			b.WriteString("  file    " + p.file.rel + "\n")
			for _, ln := range strings.Split(strings.TrimRight(string(p.content), "\n"), "\n") {
				if ln == "" {
					b.WriteString("\n")
				} else {
					b.WriteString("          " + ln + "\n")
				}
			}
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// items counts what a yes creates, for the gate prompt's grammar.
func (p *createPlan) items() int {
	n := len(p.dirs)
	if p.file != nil {
		n++
	}
	return n
}

// --- verification ---

// checkGoFile compiles what a creation just wrote: the package the
// file landed in when a module owns it (the edit spine's standard),
// or the file itself, ad hoc, when no go.mod claims it. Builds run
// as argument lists with the binary pointed at a scratch name in
// the same folder and removed after — nothing is left behind.
func checkGoFile(fileAbs string) (string, bool) {
	dir := filepath.Dir(fileAbs)
	inModule := false
	for d := dir; ; d = filepath.Dir(d) {
		if _, err := os.Stat(filepath.Join(d, "go.mod")); err == nil {
			inModule = true
			break
		}
		if filepath.Dir(d) == d {
			break
		}
	}
	scratch := filepath.Join(dir, ".gollemer-create-check")
	defer os.Remove(scratch)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	var cmd *exec.Cmd
	if inModule {
		cmd = exec.CommandContext(ctx, "go", "build", "-o", scratch, ".")
		cmd.Dir = dir
	} else {
		cmd = exec.CommandContext(ctx, "go", "build", "-o", scratch, fileAbs)
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// trimBuildOut keeps the compiler's actual complaint readable:
// the first few lines, capped.
func trimBuildOut(out string) string {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) > 6 {
		lines = lines[:6]
	}
	s := strings.Join(lines, "\n")
	if len(s) > 800 {
		s = s[:800]
	}
	return s
}

// --- the flow ---

// tryCreate is the chat-loop entry: it claims creation asks and
// the boundary family, and runs the gate flow on the chat's line
// reader.
func tryCreate(line string, r *LineReader) bool {
	return runCreate(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runCreate carries one line from sentence to created (and
// verified) thing, an honest refusal, or a clean "not created". It
// returns true when the line belonged to this claim at all.
func runCreate(line string, ask editAsk) bool {
	tag := cTag("gollemer [composer]>")
	if b, ok := parseBoundary(line); ok {
		fmt.Printf("%s %s\n", tag, b.answer())
		recordBrain("composer", true, "boundary answer: "+b.verb+" (not performed)")
		return true
	}
	req, ok := parseCreate(line)
	if !ok {
		return false
	}
	if req.refusal != "" {
		fmt.Printf("%s Not creating that: %s.\n", tag, strings.TrimSuffix(req.refusal, "."))
		recordBrain("composer", false, "create refused: unbuildable ask")
		return true
	}
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a folder to create in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "create refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		if abs, aerr := filepath.Abs(rootRaw); aerr == nil {
			rootAbs = abs
		} else {
			fmt.Printf("%s I couldn't find the folder I'd create in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "create refused: root unreadable")
			return true
		}
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd create in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "create refused: root missing")
		return true
	}
	plan, refusal := planCreate(req, rootAbs)
	if refusal != "" {
		fmt.Printf("%s Not creating that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "create refused at the fence")
		return true
	}
	fmt.Printf("%s Here's what I'll create — nothing is written yet:\n\n%s\n", tag, plan.planText())
	prompt := "Create it? [y/n] "
	if plan.items() > 1 {
		prompt = "Create these? [y/n] "
	}
	ans, aerr := ask(prompt)
	if aerr != nil || (strings.ToLower(strings.TrimSpace(ans)) != "y" && strings.ToLower(strings.TrimSpace(ans)) != "yes") {
		fmt.Printf("%s Not created — nothing was written.\n", tag)
		recordBrain("composer", true, "create declined at the gate")
		return true
	}

	// Apply: folders first (deepest via MkdirAll on the plan's
	// last), then the file. Track exactly what appeared so a
	// failure can remove precisely that — the fence proved none
	// of it pre-existed.
	if len(plan.dirs) > 0 {
		deepest := plan.dirs[len(plan.dirs)-1]
		if merr := os.MkdirAll(deepest.abs, 0o755); merr != nil {
			fmt.Printf("%s I couldn't create %s/ (%v) — nothing was created.\n", tag, deepest.rel, merr)
			recordBrain("composer", false, "create failed: mkdir")
			return true
		}
	}
	if plan.file != nil {
		if werr := os.WriteFile(plan.file.abs, plan.content, 0o644); werr != nil {
			removeCreated(plan)
			fmt.Printf("%s I couldn't write %s (%v) — I removed the folders I made; nothing half-created is left.\n", tag, plan.file.rel, werr)
			recordBrain("composer", false, "create failed: write; cleaned up")
			return true
		}
	}

	// Verify: a folder must simply be there; a .go file must
	// compile where it landed. A failed check removes exactly
	// what this operation created and reports the real complaint.
	if plan.file == nil {
		d := plan.dirs[len(plan.dirs)-1]
		if st, serr := os.Stat(d.abs); serr != nil || !st.IsDir() {
			removeCreated(plan)
			fmt.Printf("%s I made %s/ but it isn't there when I check — I cleaned up; nothing was created.\n", tag, d.rel)
			recordBrain("composer", false, "create failed: folder verify")
			return true
		}
		fmt.Printf("%s Done — created %s/. I checked — it's there.\n", tag, d.rel)
		recordBrain("composer", true, "create applied: folder "+d.rel)
		return true
	}
	if _, serr := os.Stat(plan.file.abs); serr != nil {
		removeCreated(plan)
		fmt.Printf("%s I wrote %s but it isn't there when I check — I cleaned up; nothing was created.\n", tag, plan.file.rel)
		recordBrain("composer", false, "create failed: file verify")
		return true
	}
	how := "I checked — it's there."
	if plan.isGo && plan.bareClause && plan.pkg == "main" {
		// A bare "package main" clause is a starting point the
		// user asked for, not a program: linking it can only ever
		// complain that main() is undeclared — the user's next
		// step, not a defect of the creation. The honest check
		// here is that the file is valid Go.
		if _, perr := parser.ParseFile(token.NewFileSet(), plan.file.abs, nil, parser.AllErrors); perr != nil {
			rel := plan.file.rel
			removeCreated(plan)
			fmt.Printf("%s I created %s but it isn't valid Go, so I removed it again — nothing of yours was touched.\n", tag, rel)
			fmt.Printf("The parser said: %v\n", perr)
			recordBrain("composer", false, "create rolled back: parse failed")
			return true
		}
		how = "I checked — it parses as valid Go, ready for your code."
	} else if plan.isGo {
		out, bok := checkGoFile(plan.file.abs)
		if !bok {
			rel := plan.file.rel
			removeCreated(plan)
			fmt.Printf("%s I created %s but it doesn't build, so I removed it again — nothing of yours was touched.\n", tag, rel)
			fmt.Printf("The build said: %s\nIn plain words: %s\n", trimBuildOut(out), explainBuildError(out))
			recordBrain("composer", false, "create rolled back: build failed")
			return true
		}
		how = "I compiled it to check — it builds."
	}
	created := make([]string, 0, len(plan.dirs)+1)
	for _, d := range plan.dirs {
		created = append(created, d.rel+"/")
	}
	created = append(created, plan.file.rel)
	fmt.Printf("%s Done — created %s. %s\n", tag, strings.Join(created, " and "), how)
	note := "create applied: file " + plan.file.rel
	if req.kind != "" {
		note += " (" + req.kind + ")"
	}
	recordBrain("composer", true, note)
	return true
}

// removeCreated deletes exactly what a plan created: the file,
// then the folders deepest-first (os.Remove only takes empty
// folders, so anything the user has since put inside stays).
func removeCreated(p *createPlan) {
	if p.file != nil {
		_ = os.Remove(p.file.abs)
	}
	for i := len(p.dirs) - 1; i >= 0; i-- {
		_ = os.Remove(p.dirs[i].abs)
	}
}
