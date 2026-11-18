package chat

// "Serve a file" — a handler that answers with one file the
// user already has, through http.ServeFile (which picks the
// content type from the file's extension, so the one piece
// serves a logo, a stylesheet, a data file, or a page as
// what it is). Claimed deterministically in the unified loop
// beside serve-static, whose discipline it mirrors — the
// wiring machinery's conflict rules, the clarify question
// when the ask is one fact short, honest answers for every
// state that can't land — and planned by the shared spine
// (analyze.EditServeFile). It exists for the live dead end
// of 2026-11-10: four wordings of this ask ("add a handler
// that serves a file in jim", "add a handler named jim that
// returns a file in jim/main.go", …) all fell to the
// boundary, because serve-static serves whole folders and
// serve-html is a page recipe that creates its page —
// neither owns a handler serving one existing file.
//
// Grammar (exactly this):
//
//	add|create|make [a|an|the] handler [named <Name>]
//	    that serves|returns <file spec> [target] [at <route>]
//	serve [the] [file] <file path> from [a|an|the] handler
//	    [named <Name>] [target] [at <route>]
//	file spec: a file | the file | <file name> |
//	    <file path> | html file|page (the default page,
//	    index.html — serve-html's own default)
//	target: in|into|to [the] [folder|file] <one token>
//
// The file is named in the ask or resolved in the target
// folder: exactly one servable file there is the file;
// several ask ONE clarify.go question; none is an honest
// answer. A file that isn't there is never invented — the
// piece creates handlers, not files. The handler's name is
// the sentence's, or derived from the file's name (logo.png
// → LogoHandler) — the page family's defaulting, one step
// earlier. The target follows the neighbours' rules: a .go
// file is the host; a folder resolves to its server file
// (main.go, else its one package-main file, several asked
// about, none landing the handler in a new file of the
// folder's own package); no target is the project's own
// program. A folder inside its own module (a wasm app) can
// never host the handler — a separate program this
// project's server can't reach into.

import (
	"fmt"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// serveFileRequest is one parsed serve-a-file ask.
type serveFileRequest struct {
	name     string // the handler's name as typed ("" = derived from the file)
	file     string // the served file as typed ("" = unnamed)
	htmlPage bool   // the "html file" spec: the default page, index.html
	target   string // the target as typed ("" = no target clause)
	route    string // the "at </path>" route as typed ("" = derived)
}

// parseServeFile reads one chat line as a serve-a-file ask.
// The second return is false for anything outside the
// grammar — including the neighbours' sentences: the book's
// handler rows carry a kind word before "handler" ("add a
// form handler named…"), serve-html's build-a-server shapes
// name a server where this grammar names a handler, and
// serve-static's tails end at asset words or a folder,
// never at "that serves a file".
func parseServeFile(line string) (serveFileRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 3 {
		return serveFileRequest{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(strings.Trim(f, ",;:\"'()"))
	}
	var req serveFileRequest
	i := 0
	switch low[0] {
	case "add", "create", "make":
		i = 1
		if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
			i++
		}
		if i >= len(fields) || low[i] != "handler" {
			return serveFileRequest{}, false
		}
		i++
		if ni, ok := serveFileNameAt(fields, low, i, &req); ok {
			i = ni
		} else if i < len(fields) && low[i] == "named" {
			return serveFileRequest{}, false
		}
		if i >= len(fields) || low[i] != "that" {
			return serveFileRequest{}, false
		}
		i++
		if i >= len(fields) {
			return serveFileRequest{}, false
		}
		switch low[i] {
		case "serves", "serve", "returns", "return":
			i++
		default:
			return serveFileRequest{}, false
		}
		ni, ok := serveFileSpecAt(fields, low, i, &req)
		if !ok {
			return serveFileRequest{}, false
		}
		i = ni
	case "serve":
		i = 1
		if i < len(fields) && low[i] == "the" {
			i++
		}
		if i < len(fields) && low[i] == "file" {
			i++
		}
		if i < len(fields) && isServeFileToken(fields[i]) {
			req.file = fields[i]
			i++
		}
		if i >= len(fields) || low[i] != "from" {
			return serveFileRequest{}, false
		}
		i++
		if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
			i++
		}
		if i >= len(fields) || low[i] != "handler" {
			return serveFileRequest{}, false
		}
		i++
		if ni, ok := serveFileNameAt(fields, low, i, &req); ok {
			i = ni
		} else if i < len(fields) && low[i] == "named" {
			return serveFileRequest{}, false
		}
	default:
		return serveFileRequest{}, false
	}
	if _, ok := serveFileTail(fields, low, i, &req); !ok {
		return serveFileRequest{}, false
	}
	return req, true
}

// serveFileNameAt reads an optional "named <Name>" at i.
// ok is false when no name slot stands there; a slot that
// starts ("named") but names nothing usable is the
// caller's to reject — the parse claims nothing for it.
func serveFileNameAt(fields, low []string, i int, req *serveFileRequest) (int, bool) {
	if i+1 < len(fields) && low[i] == "named" && isWireNameToken(fields[i+1], low[i+1]) {
		req.name = fields[i+1]
		return i + 2, true
	}
	return i, false
}

// serveFileSpecAt reads the file spec at i in an add-form
// sentence: [a|an|the] then "file" (unnamed, or one file
// name riding behind it), "html file|page" (the default
// page), or one file name on its own.
func serveFileSpecAt(fields, low []string, i int, req *serveFileRequest) (int, bool) {
	if i < len(fields) && (low[i] == "a" || low[i] == "an" || low[i] == "the") {
		i++
	}
	if i >= len(fields) {
		return i, false
	}
	if low[i] == "html" && i+1 < len(fields) && (low[i+1] == "file" || low[i+1] == "page") {
		req.htmlPage = true
		return i + 2, true
	}
	if low[i] == "file" {
		i++
		if i < len(fields) && isServeFileToken(fields[i]) {
			req.file = fields[i]
			return i + 1, true
		}
		return i, true
	}
	if isServeFileToken(fields[i]) {
		req.file = fields[i]
		return i + 1, true
	}
	return i, false
}

// isServeFileToken reports whether a token is shaped like
// a file name: its last path segment carries an extension
// ("logo.png", "jim/index.html"). A bare word with no
// extension is a folder or a thing, never this family's
// file — "serves the logo in jim" is not claimed, by the
// same tightness that keeps the neighbours' lines theirs.
func isServeFileToken(tok string) bool {
	t := strings.Trim(tok, ",;:\"'()")
	if t == "" {
		return false
	}
	seg := t
	if j := strings.LastIndex(t, "/"); j >= 0 {
		seg = t[j+1:]
	}
	dot := strings.LastIndex(seg, ".")
	if dot <= 0 || dot == len(seg)-1 {
		return false
	}
	for _, r := range seg[dot+1:] {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			return false
		}
	}
	return true
}

// serveFileTail reads what may end a serve-file shape: an
// optional target clause and an optional "at <route>", in
// that order, and then the end of the line. Anything else —
// above all a longer tail, another ask glued on — is not
// this claim's.
func serveFileTail(fields, low []string, i int, req *serveFileRequest) (int, bool) {
	if i == len(fields) {
		return i, true
	}
	if low[i] == "at" {
		if i+2 == len(fields) {
			req.route = fields[i+1]
			return i + 2, true
		}
		return i, false
	}
	if low[i] != "in" && low[i] != "into" && low[i] != "to" {
		return i, false
	}
	i++
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) {
		switch strings.TrimSuffix(low[i], "/") {
		case "folder", "directory", "file":
			i++
		}
	}
	if i >= len(fields) {
		return i, false
	}
	switch rest := len(fields) - i; {
	case rest == 1:
		req.target = fields[i]
		return len(fields), true
	case rest == 3 && low[i+1] == "at":
		req.target = fields[i]
		req.route = fields[i+2]
		return len(fields), true
	}
	return i, false
}

// tryServeFile is the chat-loop entry: it claims
// serve-a-file sentences and runs the flow on the chat's
// line reader.
func tryServeFile(line string, r *LineReader) bool {
	return runServeFile(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runServeFile carries one line from sentence to resolved
// host and file to the shared edit spine. It returns true
// when the line belonged to this claim at all.
func runServeFile(line string, ask editAsk) bool {
	q, ok := parseServeFile(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootRaw, rootAbs, ok := staticAssetsRoot(tag, "serve-file")
	if !ok {
		return true
	}
	refuse := func(format string, args ...any) bool {
		fmt.Printf("%s Not serving that file: %s.\n", tag, strings.TrimSuffix(fmt.Sprintf(format, args...), "."))
		recordBrain("composer", false, "serve-file refused: target")
		return true
	}

	// The target first: it settles both the host file and
	// the folder a bare file name resolves in.
	folderRel := ""
	hostRel := ""
	switch {
	case q.target == "":
		host, multi := resolveServeFileHost(rootAbs, "")
		if multi != nil {
			askServeFileHostPick(tag, q, multi, ask)
			return true
		}
		hostRel = host
	default:
		abs, rel, refusal := serverTargetResolve(rootAbs, q.target)
		if refusal != "" {
			return refuse("%s", refusal)
		}
		st, err := os.Stat(abs)
		switch {
		case err == nil && st.IsDir():
			folderRel = rel
			if refusal := serveFileModuleRefusal(rootAbs, folderRel); refusal != "" {
				return refuse("%s", refusal)
			}
			host, multi := resolveServeFileHost(rootAbs, folderRel)
			if multi != nil {
				askServeFileHostPick(tag, q, multi, ask)
				return true
			}
			hostRel = host
		case err == nil:
			// An existing file: a .go file is a host;
			// anything else isn't one.
			if !strings.HasSuffix(strings.ToLower(q.target), ".go") {
				return refuse("%s is a file, not a folder or a Go file — the handler goes into a .go file in the folder the served file lives in. Name the folder, like \"add a handler that serves %s in %s\"", q.target, serveFileSpecWords(q), serveFileFolderWords(slashDir(rel)))
			}
			folderRel = slashDir(rel)
			if refusal := serveFileModuleRefusal(rootAbs, folderRel); refusal != "" {
				return refuse("%s", refusal)
			}
			hostRel = rel
		case strings.HasSuffix(strings.ToLower(q.target), ".go"):
			// A missing .go file is created by the plan —
			// when its folder exists (the page family's rule).
			folderRel = slashDir(rel)
			if _, serr := os.Stat(filepath.Dir(abs)); serr != nil {
				return refuse("there's no folder %s in this project — say \"create a folder %s\" first, then ask again", folderRel, folderRel)
			}
			if refusal := serveFileModuleRefusal(rootAbs, folderRel); refusal != "" {
				return refuse("%s", refusal)
			}
			hostRel = rel
		default:
			return refuse("there's no folder %s in this project — say \"create a folder %s\" first, put the file in it, then ask again", q.target, q.target)
		}
	}
	folderAbs := rootAbs
	if folderRel != "" {
		folderAbs = filepath.Join(rootAbs, filepath.FromSlash(folderRel))
	}

	// The served file: named (a path as typed, or a bare
	// name in the target folder), the default page, or
	// resolved among the folder's servable files.
	serveRel := ""
	switch {
	case q.file != "" && strings.Contains(q.file, "/"):
		_, rel, refusal := fenceResolve(rootAbs, q.file)
		if refusal != "" {
			return refuse("%s", refusal)
		}
		st, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(rel)))
		switch {
		case err != nil:
			return refuse("there's no file %s in this project — a handler can only serve a file that's really there. Put the file there first, then ask again", rel)
		case st.IsDir():
			return refuse("%s is a folder — a handler serves one file. For the whole folder the sentence is \"serve static files from %s\"", rel, rel)
		case strings.HasSuffix(strings.ToLower(rel), ".go"):
			return refuse("%s is a Go source file — a handler serves a file the browser downloads: a page, an image, a stylesheet, data. Name one of those instead", rel)
		}
		serveRel = rel
	case q.file != "" || q.htmlPage:
		name := q.file
		if q.htmlPage {
			name = "index.html"
		}
		serveRel = joinRel(folderRel, name)
		st, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(serveRel)))
		switch {
		case err != nil && q.htmlPage:
			return refuse("there's no index.html in %s yet. If you want the page created and served in one ask, say \"serve index.html to %s\" — that makes the page and its wiring together. Or put the file there and ask me again", serveFileFolderWords(folderRel), serveFileFolderWords(folderRel))
		case err != nil:
			return refuse("there's no %s in %s — a handler can only serve a file that's really there. Put %s there first, then ask again", name, serveFileFolderWords(folderRel), name)
		case st.IsDir():
			return refuse("%s is a folder — a handler serves one file. For the whole folder the sentence is \"serve static files from %s\"", serveRel, serveRel)
		case strings.HasSuffix(strings.ToLower(name), ".go"):
			return refuse("%s is a Go source file — a handler serves a file the browser downloads: a page, an image, a stylesheet, data. Name one of those instead", name)
		}
	default:
		cands := servableFileCandidates(folderAbs)
		switch {
		case len(cands) == 1:
			serveRel = joinRel(folderRel, cands[0])
		case len(cands) > 1:
			picks := make([]clarifyCandidate, len(cands))
			for i, c := range cands {
				picks[i] = clarifyCandidate{Label: c, Words: []string{c}}
			}
			askPickQuestion(tag,
				fmt.Sprintf("More than one file in %s could be served — which one should the handler serve? Nothing's been written.", serveFileFolderWords(folderRel)),
				"Answer with the number or the file name",
				picks,
				func(idx int, ask2 editAsk) bool {
					return runServeFile(serveFileCanonical(q, cands[idx]), ask2)
				})
			recordBrain("composer", false, "serve-file question: which file")
			return true
		default:
			return refuse("there's no file in %s for a handler to serve yet — the folder holds no page, image, stylesheet, script, or data file. Put the file there first, or name it in the sentence, like \"add a handler that serves logo.png in %s\"", serveFileFolderWords(folderRel), serveFileFolderWords(folderRel))
		}
	}

	// The handler's name: the sentence's, or derived from
	// the file's name (logo.png → LogoHandler).
	name := q.name
	if name == "" {
		name = serveFileHandlerName(path.Base(serveRel))
		if name == "" {
			return refuse("I can't make a handler name out of %q — give it one in the sentence, like \"add a handler named Logo that serves %s in %s\"", path.Base(serveRel), path.Base(serveRel), serveFileFolderWords(folderRel))
		}
	}

	if !settleStaticAssetsProject(tag, rootRaw, rootAbs, "serve-file") {
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:           analyze.EditServeFile,
		File:           hostRel,
		ServeFile:      serveRel,
		ServeFileName:  name,
		ServeFileRoute: q.route,
	}, ask)
}

// serveFileFolderWords renders a folder for a sentence: the
// project root reads as "this project", any other folder as
// itself.
func serveFileFolderWords(rel string) string {
	if rel == "" || rel == "." {
		return "this project"
	}
	return rel
}

// serveFileSpecWords renders the ask's file spec for a
// refusal that re-spells the sentence: the file as typed,
// the default page, or the unnamed shape.
func serveFileSpecWords(q serveFileRequest) string {
	switch {
	case q.file != "":
		return q.file
	case q.htmlPage:
		return "index.html"
	default:
		return "a file"
	}
}

// serveFileCanonical rebuilds the ask as its full sentence
// for a clarify completion: the picked file fills the file
// slot, and the target rides as typed when there was one.
func serveFileCanonical(q serveFileRequest, fileName string) string {
	var b strings.Builder
	b.WriteString("add a handler")
	if q.name != "" {
		b.WriteString(" named " + q.name)
	}
	b.WriteString(" that serves " + fileName)
	if q.target != "" {
		b.WriteString(" in " + q.target)
	}
	if q.route != "" {
		b.WriteString(" at " + q.route)
	}
	return b.String()
}

// askServeFileHostPick asks which of a folder's several
// main-package files should hold the handler (clarify.go) —
// the spelled-out choice, asked instead of guessed. The
// answer completes the original ask with the file as its
// target, so every fact is re-read at completion time.
func askServeFileHostPick(tag string, q serveFileRequest, multi []string, ask editAsk) {
	cands := make([]clarifyCandidate, len(multi))
	for i, c := range multi {
		cands[i] = clarifyCandidate{Label: c, Words: []string{path.Base(c)}}
	}
	askPickQuestion(tag,
		fmt.Sprintf("More than one main-package file could hold the handler — %s. Which one should it go in? Nothing's been written.", strings.Join(multi, " or ")),
		"Answer with the path or the number",
		cands,
		func(idx int, ask2 editAsk) bool {
			qq := q
			qq.target = multi[idx]
			return runServeFile(serveFileCanonical(qq, serveFileSpecWords(qq)), ask2)
		})
	recordBrain("composer", false, "serve-file question: folder has several main files")
}

// resolveServeFileHost settles the host .go file inside one
// folder (rel, "" = the project root): its main.go, else
// its one package-main file, else — the folder holding Go
// files of another package — a new handlers.go in that
// package, else a new main.go. multi carries the candidate
// files when the folder holds several package-main files:
// a choice, never a guess.
func resolveServeFileHost(rootAbs, rel string) (host string, multi []string) {
	abs := rootAbs
	if rel != "" {
		abs = filepath.Join(rootAbs, filepath.FromSlash(rel))
	}
	if fi, err := os.Stat(filepath.Join(abs, "main.go")); err == nil && !fi.IsDir() {
		return joinRel(rel, "main.go"), nil
	}
	cands := packageMainFiles(abs)
	switch {
	case len(cands) == 1:
		return joinRel(rel, cands[0]), nil
	case len(cands) > 1:
		out := make([]string, len(cands))
		for i, c := range cands {
			out[i] = joinRel(rel, c)
		}
		return "", out
	}
	if folderHasGoFiles(abs) {
		return joinRel(rel, "handlers.go"), nil
	}
	return joinRel(rel, "main.go"), nil
}

// serveFileModuleRefusal answers for a target folder inside
// its own module: a separate program (a wasm app's, above
// all) that this project's server — and this project's
// analysis — cannot reach into. "" when the folder is an
// ordinary folder of this project.
func serveFileModuleRefusal(rootAbs, folderRel string) string {
	if folderRel == "" || folderRel == "." {
		return ""
	}
	parts := strings.Split(folderRel, "/")
	for i := 1; i <= len(parts); i++ {
		sub := strings.Join(parts[:i], "/")
		if _, err := os.Stat(filepath.Join(rootAbs, filepath.FromSlash(sub), "go.mod")); err != nil {
			continue
		}
		msg := fmt.Sprintf("%s is its own module — a separate program — so a handler for this project can't be added inside it. Name a folder in this project instead", sub)
		// A wasm app gets its own way in named: its files
		// are served by its server, not by a handler here.
		appDir := filepath.Join(rootAbs, filepath.FromSlash(sub))
		if _, err := os.Stat(filepath.Join(appDir, "index.html")); err == nil {
			if _, err := os.Stat(filepath.Join(appDir, "wasm_exec.js")); err == nil {
				msg = fmt.Sprintf("%s is the wasm app — its own module and its own program, so a Go handler can't be added inside it. Its files are served by its server instead: say \"serve the wasm app in %s\"", sub, sub)
			}
		}
		return msg
	}
	return ""
}

// servableFileCandidates lists one folder's files a handler
// could serve: regular files directly in it, not hidden,
// carrying an extension that isn't Go's own (.go, .mod,
// .sum are the project's machinery, not downloads). Sorted.
func servableFileCandidates(dirAbs string) []string {
	entries, err := os.ReadDir(dirAbs)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if strings.HasPrefix(name, ".") || !strings.Contains(name, ".") {
			continue
		}
		switch strings.ToLower(filepath.Ext(name)) {
		case ".go", ".mod", ".sum":
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// serveFileHandlerName derives the bare ask's handler name
// from the served file's name: the stem camel-cased, plus
// "Handler" (logo.png → LogoHandler, my-report.pdf →
// MyReportHandler, 404.html → Serve404Handler). "" when no
// usable name can be made.
func serveFileHandlerName(base string) string {
	stem := base
	if i := strings.LastIndex(stem, "."); i > 0 {
		stem = stem[:i]
	}
	var b strings.Builder
	up := true
	for _, r := range stem {
		switch {
		case unicode.IsLetter(r):
			if up {
				r = unicode.ToUpper(r)
				up = false
			}
			b.WriteRune(r)
		case unicode.IsDigit(r):
			b.WriteRune(r)
		default:
			up = true
		}
	}
	name := b.String()
	if name == "" {
		return ""
	}
	if r := []rune(name)[0]; unicode.IsDigit(r) {
		name = "Serve" + name
	}
	name += "Handler"
	if !token.IsIdentifier(name) {
		return ""
	}
	return name
}
