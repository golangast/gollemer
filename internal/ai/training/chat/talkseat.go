package chat

// The talk seat (LOCAL variant, 2026-11-03): a read-only
// conversation route seated in the unified loop's tail, after
// tryDeterministicAnswer and before the routed model's
// PredictWithTrace fallback. Lines nobody deterministic claimed
// get one composed, local answer here; lines the seat declines
// fall through to the existing neural fallback byte-identically.
//
// The seat NEVER writes, gates, or edits, and it makes ZERO
// network calls: no endpoint, no key, no dial — the responder
// is composed entirely from assets this project already owns
// (the analyzer's index, the session's working context, and
// the local social model file on disk). Edit-shaped lines
// never reach it: every deterministic claim, including the
// write boundary and the intent layer, sits ahead of this seat.
//
// The responder seam copies the speccopy proposer exactly
// (proposeddecl.go): a package var holding a TalkResponder,
// filled once by a lazy loader behind a *sync.Once. If the
// composed responder cannot initialize — the social model
// file absent or unusable — the var stays nil, the seat never
// fires, and the chat behaves byte-identically to before.
//
// A future locally-trained talk model (a .gob file present on
// disk, speccopy-style) registers through this same var: an
// adapter implementing TalkResponder is assigned to
// talkResponderVar by its loader, and no chat code changes —
// the seat, the context assembly, and the claim-checker below
// are all model-agnostic.

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// TalkDomain is the talk seat's tag word: seat replies print
// as "gollemer [talk]>", in the unified chat's tag idiom. It
// renders purple by default (termui.BrainColorName), like
// social — the text marker is the distinction that matters.
const TalkDomain = "talk"

// TalkResponse is one seat answer. Factual marks replies that
// are deterministic by construction (index facts, the curated
// Go knowledge base): they skip the claim-checker, because
// every name in them came from the verified source itself.
// Non-factual replies (social model generation) are
// claim-checked against the index before they print. Trace
// carries the social model's thought trace, when it produced
// the reply, so the loop's /thoughts display works as it does
// for the neural tail.
type TalkResponse struct {
	Text    string
	Factual bool
	Trace   *seq2seq.ThoughtTrace
}

// TalkResponder is the talk seat's generation seam: given the
// assembled context and the user's line, produce a reply, or
// ok=false to decline (the line then keeps today's routed
// fallback). The v1 implementation is localTalkResponder,
// composed from the index, the session context, and the
// social model; a future local .gob talk model implements
// this same one-method interface.
type TalkResponder interface {
	Respond(ctx TalkContext, line string) (TalkResponse, bool)
}

var (
	talkOnce = &sync.Once{} // a pointer so tests can swap in a fresh one
	// talkResponderVar is the loaded responder seam. Nil means
	// the responder is unavailable and the talk seat never
	// fires — byte-identical prior behavior.
	talkResponderVar TalkResponder
)

// talkResponder lazily loads the composed local responder
// once and returns it, or nil when it cannot initialize.
func talkResponder() TalkResponder {
	talkOnce.Do(loadTalkResponder)
	return talkResponderVar
}

// loadTalkResponder builds the v1 composed responder. Its one
// hard requirement is the local social model: without it the
// conversation arm cannot answer, so the whole seat stays off
// (the speccopy rule — every failure is silent, the var stays
// nil, and the chat behaves exactly as before). The model is
// loaded through the brain loader's own pattern
// (loadRealModel), from the same roots the proposer reads.
func loadTalkResponder() {
	for _, base := range []string{unifiedProjectRoot, startFolder()} {
		if base == "" {
			continue
		}
		if _, err := os.Stat(RealModelPath(base, SocialDomain)); err != nil {
			continue
		}
		model, err := loadRealModel(base, SocialDomain)
		if err != nil {
			continue
		}
		talkResponderVar = &localTalkResponder{social: model}
		return
	}
}

// TalkContext is everything the seat assembles for one line,
// per the design's context list, adapted honestly to a local
// composition. Every piece is used by the composition:
//
//   - Story: the project overview paragraph — the framing
//     fact the project-overview answer is built from.
//   - Digest / Names: the capped index digest — package
//     answers and near-match suggestions are read from it,
//     and Names is the existence list loose lookups use.
//   - FocusFile / FocusBody: the working file's contents,
//     size-capped (the file the line names, else the session's
//     last file) — file answers quote its declarations and
//     its real line count from the body itself.
//   - Work / WorkLive: the session working context
//     (sessionctx.go) — "it" resolves from it, piece first,
//     then last file, then working folder.
//   - Turns: the in-session transcript tail, capped — the
//     fallback "it" source (a name the conversation itself
//     mentioned) and, through Conv.SocialInput, the social
//     model's context conditioning.
//   - Project: the fresh index every fact answer and the
//     claim-checker verify against.
//   - SocialRoute: whether this line routed to the social
//     domain (or would fall back to the social model) — the
//     gate for the conversation arm.
type TalkContext struct {
	Line        string
	Domain      string
	SocialRoute bool
	Project     *analyze.Project
	Story       string
	Digest      string
	Names       []string
	FocusFile   string
	FocusBody   string
	FocusCapped bool
	Work        workContext
	WorkLive    bool
	Turns       []Turn
	Conv        *Conversation
	Root        string
	State       systemState
	StateText   string
	MemoryText  string
	SessionText string
}

// talkFocusCap caps the working-file body fed to the seat.
const talkFocusCap = 8000

// talkDigestCap caps the index digest's rendered length.
const talkDigestCap = 3500

// buildTalkContext assembles the TalkContext for one line:
// the fresh project index first (analyzing the chat's own
// project on demand, re-reading silently when the tree moved
// since the last read), then every derived piece.
func buildTalkContext(line, domain string, conv *Conversation, socialRoute bool) TalkContext {
	ctx := TalkContext{Line: line, Domain: domain, SocialRoute: socialRoute, Conv: conv}
	// Capture the read-baseline change list BEFORE
	// ensureTalkProject's silent re-read can move the
	// baseline: state answers report what changed since the
	// read the session had when this line arrived.
	entryRoot := unifiedProjectRoot
	if lastAnalyzeProject != nil && lastAnalyzeProject.Root != "" {
		entryRoot = lastAnalyzeProject.Root
	}
	preChanged, preChangeNote := changedGoFilesSinceRead(entryRoot)
	p := ensureTalkProject()
	ctx.Project = p
	if conv != nil {
		turns := conv.Turns()
		if len(turns) > 6 {
			turns = turns[len(turns)-6:]
		}
		ctx.Turns = append([]Turn{}, turns...)
	}
	// The session working context, gated exactly as the
	// existing resolvers gate it: live only for its own root.
	if rootAbs, ok := claimRootAbs(unifiedProjectRoot); ok && workContextLive(rootAbs) {
		ctx.Work = workCtx
		ctx.WorkLive = true
	}
	if p == nil {
		ctx.Root = canonicalMemoryRoot(entryRoot)
		ctx.MemoryText = memoryContextText(ctx.Root)
		ctx.SessionText = sessionSummaryText()
		ctx.State = buildSystemState(ctx.Root)
		ctx.State.Changed = preChanged
		ctx.State.ChangeNote = preChangeNote
		ctx.StateText = ctx.State.Text()
		return ctx
	}
	ctx.Story = p.Overview()
	// The focus file: the .go file the line names (when it is
	// really in the project), else the session's last file.
	focus := ""
	if m := talkGoFileRe.FindStringSubmatch(line); m != nil {
		if rel := cleanTalkPath(m[1]); talkFileExists(p, rel) {
			focus = rel
		}
	}
	if focus == "" && ctx.WorkLive && ctx.Work.lastFile != "" && talkFileExists(p, ctx.Work.lastFile) {
		focus = ctx.Work.lastFile
	}
	ctx.FocusFile = focus
	if focus != "" {
		if data, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(focus))); err == nil {
			body := string(data)
			if len(body) > talkFocusCap {
				body = body[:talkFocusCap]
				ctx.FocusCapped = true
			}
			ctx.FocusBody = body
		}
	}
	detailed := map[string]bool{}
	if ctx.WorkLive {
		detailed[ctx.Work.folder] = true
		if d := slashDir(ctx.Work.lastFile); d != "" {
			detailed[d] = true
		}
	}
	if focus != "" {
		detailed[slashDir(focus)] = true
	}
	ctx.Digest, ctx.Names = buildTalkDigest(p, detailed)
	ctx.Root = canonicalMemoryRoot(p.Root)
	ctx.MemoryText = memoryContextText(ctx.Root)
	ctx.SessionText = sessionSummaryText()
	ctx.State = buildSystemState(ctx.Root)
	ctx.State.Changed = preChanged
	ctx.State.ChangeNote = preChangeNote
	ctx.StateText = ctx.State.Text()
	return ctx
}

// ensureTalkProject returns the session's project index,
// fresh: the analyzed project when it is current, a silent
// re-read when the tree changed under it, or a fresh analysis
// of the folder the chat runs in when nothing was analyzed
// yet. Unlike ensureFreshProject it prints nothing — a talk
// answer must not be preceded by write-path warnings — and
// unlike ensureProject it never substitutes the bundled demo:
// talk is about the user's own project or it declines.
func ensureTalkProject() *analyze.Project {
	if lastAnalyzeProject != nil {
		if !currentProjectStale() {
			return lastAnalyzeProject
		}
		root := lastAnalyzeProject.Root
		fresh, err := analyze.Analyze(root)
		if err != nil {
			// The tree is mid-edit; the stale index is still
			// the best honest source available.
			return lastAnalyzeProject
		}
		setCurrentProject(lastAnalyzeRoot, fresh)
		return fresh
	}
	root := unifiedProjectRoot
	if root == "" {
		root = startFolder()
	}
	if root == "" {
		return nil
	}
	abs, err := filepath.EvalSymlinks(root)
	if err != nil {
		abs, err = filepath.Abs(root)
		if err != nil {
			return nil
		}
	}
	p, err := analyze.Analyze(abs)
	if err != nil {
		return nil
	}
	setCurrentProject(root, p)
	return p
}

// buildTalkDigest renders the capped index digest and the
// flat name list. Packages in a detailed directory (the
// working folder, the focus file's folder) get signatures;
// every other package is names only. The digest is a real
// input to the composition: package answers quote its lines
// and the overview answer lists packages from it.
func buildTalkDigest(p *analyze.Project, detailed map[string]bool) (string, []string) {
	var names []string
	pkgs := append([]*analyze.Package{}, p.Packages...)
	sort.Slice(pkgs, func(i, j int) bool { return pkgs[i].Dir < pkgs[j].Dir })
	var lines []string
	for _, pkg := range pkgs {
		dir := pkg.Dir
		if dir == "" {
			dir = "."
		}
		var b strings.Builder
		if detailed[pkg.Dir] {
			fmt.Fprintf(&b, "package %s (%s/):", pkg.Name, dir)
			var fns []string
			for _, fn := range pkg.Funcs {
				if fn.IsTest {
					continue
				}
				fns = append(fns, strings.TrimPrefix(fn.Sig, "func "))
				names = append(names, fn.Name)
			}
			sort.Strings(fns)
			if len(fns) > 0 {
				fmt.Fprintf(&b, " functions: %s;", strings.Join(fns, "; "))
			}
			var tys []string
			for _, t := range pkg.Types {
				tys = append(tys, t.Name+" ("+t.Kind+")")
				names = append(names, t.Name)
			}
			sort.Strings(tys)
			if len(tys) > 0 {
				fmt.Fprintf(&b, " types: %s;", strings.Join(tys, "; "))
			}
		} else {
			fmt.Fprintf(&b, "package %s (%s/):", pkg.Name, dir)
			var fns []string
			for _, fn := range pkg.Funcs {
				if fn.IsTest {
					continue
				}
				fns = append(fns, fn.Name)
				names = append(names, fn.Name)
			}
			sort.Strings(fns)
			if len(fns) > 0 {
				fmt.Fprintf(&b, " functions %s;", strings.Join(fns, ", "))
			}
			var tys []string
			for _, t := range pkg.Types {
				tys = append(tys, t.Name)
				names = append(names, t.Name)
			}
			sort.Strings(tys)
			if len(tys) > 0 {
				fmt.Fprintf(&b, " types %s;", strings.Join(tys, ", "))
			}
		}
		lines = append(lines, strings.TrimSuffix(b.String(), ";"))
	}
	digest := strings.Join(lines, "\n")
	if len(digest) > talkDigestCap {
		digest = digest[:talkDigestCap] + "\n…"
	}
	sort.Strings(names)
	return digest, names
}

// tryTalkSeat is the seat's chat-loop entry, seated in the
// tail after tryDeterministicAnswer: it prints the composed
// reply under the talk tag and records it in the conversation.
// It returns false — and prints nothing — when the responder
// is unavailable or declines the line, so the routed neural
// fallback proceeds byte-identically.
func tryTalkSeat(line, domain string, conv *Conversation, socialRoute bool, showThoughts bool) bool {
	resp := talkResponder()
	if resp == nil {
		return false
	}
	ctx := buildTalkContext(line, domain, conv, socialRoute)
	out, ok := resp.Respond(ctx, line)
	if !ok {
		return false
	}
	text := strings.TrimSpace(out.Text)
	if text == "" {
		return false
	}
	// Claim-checking: deterministic-fact replies are factual
	// by construction and skip the checker; generated replies
	// (the social arm) are checked against the index, and a
	// project name that does not exist never prints as bare
	// fact — its sentence is marked, or suppressed when
	// marking would mangle it. See talkseat_check.go.
	if !out.Factual {
		text = talkCheckReply(text, ctx.Project)
		if strings.TrimSpace(text) == "" {
			return false
		}
	}
	fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", TalkDomain)), text)
	conv.AddReply(text, TalkDomain, false)
	if showThoughts {
		printThoughtTrace(out.Trace)
	}
	return true
}

// tryTalkStateAnswer is the state arm's deterministic
// claim in the unified loop, seated after recall and before
// routing/intent: state questions are facts, and a model or
// classifier must never get the chance to steal one (the
// intent layer reads "branch" as "brains"). It shares the
// exact arm the talk seat's responder runs first.
func tryTalkStateAnswer(line string, conv *Conversation) bool {
	if !isTalkStateLine(line) {
		return false
	}
	ctx := buildTalkContext(line, "", conv, false)
	out, ok := talkStateAnswer(ctx, line)
	if !ok {
		return false
	}
	fmt.Printf("%s %s\n", cTag(fmt.Sprintf("gollemer [%s]>", TalkDomain)), out)
	if conv != nil {
		conv.AddUser(line)
		conv.AddReply(out, TalkDomain, false)
	}
	return true
}

// localTalkResponder is the v1 composed responder: project
// facts from the index, "it" from the session context,
// conversation from the local social model, general Go from
// the curated knowledge base — in that order, and nothing
// else. It holds the seat's own lazy-loaded social handle.
type localTalkResponder struct {
	social *seq2seq.Seq2Seq
}

// Respond runs the v1 composition in order. The first arm
// that can answer honestly wins; no arm invents.
func (r *localTalkResponder) Respond(ctx TalkContext, line string) (TalkResponse, bool) {
	// 0. System-state facts: the clock, the OS/environment,
	// the branch and read-baseline changes, and what this
	// session has made are read from the freshly assembled
	// state block — never routed through a model.
	if out, ok := talkStateAnswer(ctx, line); ok {
		return TalkResponse{Text: out, Factual: true}, true
	}
	// 0.5. Personal facts: what the user has told the chat
	// about themselves (their dog's name, where they live,
	// their favorite color) is answered from the long-term
	// fact store (factmemory.go) — the stored value exactly,
	// or the honest not-told-me — never from the social arm
	// below, which would invent a name. The unified loop
	// seats the same claim ahead of the intent layer
	// (tryFactRecallAnswer); this arm is the seat's own copy
	// for lines that reach it by other paths.
	if out, ok := factRecallAnswer(line); ok {
		return TalkResponse{Text: out, Factual: true}, true
	}
	// 1. Project-fact second pass: a broader, looser pass
	// over the index than the deterministic Q&A that already
	// declined this line — it auto-analyzed the project (in
	// buildTalkContext), it answers file and package
	// questions the Q&A shapes don't cover, it answers
	// parameter/return questions from real signatures, and
	// when the named thing is NOT in the index it says so
	// honestly instead of letting a model improvise.
	if out, ok := talkProjectAnswer(ctx, line); ok {
		return TalkResponse{Text: out, Factual: true}, true
	}
	// 2. Session-context "it" resolution: a subject-less
	// question reads "it" from the work in progress, and the
	// answer names what "it" was read as.
	if out, ok := talkItAnswer(ctx, line); ok {
		return TalkResponse{Text: out, Factual: true}, true
	}
	// 3. Social conversation: the line routed to the social
	// domain (or would fall back to the social model) is
	// answered by the social model through this seat's own
	// handle — same conditioning (SocialInput) and
	// post-processing (tidyDecode) as the neural tail. The
	// reply is claim-checked by the seat before it prints.
	if ctx.SocialRoute && r.social != nil {
		in := strings.ToLower(strings.TrimSpace(line))
		if ctx.Conv != nil {
			in = ctx.Conv.SocialInput(in)
		}
		if ans, trace, err := r.social.PredictWithTrace(in, 40); err == nil {
			if reply := tidyDecode(ans); reply != "" {
				return TalkResponse{Text: reply, Factual: false, Trace: trace}, true
			}
		}
	}
	// 4. General Go questions the curated knowledge base
	// covers. (The error translator and the beginner five sit
	// ahead of this seat in the chain and already claimed
	// everything they own; the knowledge base is the
	// deterministic explainer still reachable from the tail.
	// tryDeterministicAnswer consults it for go-routed lines and,
	// since the 2026-11-05 seat change, claims its strongest tier
	// ahead of the route for concept questions in any domain —
	// this pass remains the broader net for whatever reaches the
	// tail: non-question shapes and subjects the audited claim
	// declined.) Arm order is unchanged: the social arm above
	// still speaks first for social-routed lines that reach this
	// seat, which is exactly why the audited claim had to live in
	// tryDeterministicAnswer rather than here.
	if out, ok := LookupGoKnowledge(line); ok {
		return TalkResponse{Text: out, Factual: true}, true
	}
	// 5. No honest answer: decline, and today's routed-model
	// fallback proceeds byte-identically.
	return TalkResponse{}, false
}

var (
	talkGoFileRe   = regexp.MustCompile(`([A-Za-z0-9_./-]+\.go)\b`)
	talkOverviewRe = regexp.MustCompile(`(?i)^\s*(what is this (project|repo|repository|codebase|code)|what does this (project|repo|repository|codebase) do|what'?s this (project|repo|codebase)|tell me about this (project|repo|codebase)|describe this (project|repo|codebase))\s*\??\s*$`)
	talkExistRe    = regexp.MustCompile(`(?i)^\s*(?:does|do) (?:this project|the project|it) have (?:a |an |the )?(function|func|type|struct|interface) (?:called |named )?([A-Za-z_][A-Za-z0-9_.]*)\s*\??\s*$`)
	talkExistRe2   = regexp.MustCompile(`(?i)^\s*is there (?:a |an |the )?(function|func|type|struct|interface) (?:called |named )?([A-Za-z_][A-Za-z0-9_.]*)\b.*\??\s*$`)
	talkPkgFuncsRe = regexp.MustCompile(`(?i)^\s*what (functions|funcs|types) are in (?:the )?(.+?)(?: package)?\s*\??\s*$`)
	talkTakeRe     = regexp.MustCompile(`(?i)^\s*what does (?:the )?(.+?) (?:take|accept|need)\s*\??\s*$`)
	talkReturnRe   = regexp.MustCompile(`(?i)^\s*what does (?:the )?(.+?) return\s*\??\s*$`)
	talkTypeIsRe   = regexp.MustCompile(`(?i)^\s*what (?:type|kind) is (?:the )?(.+?)\s*\??\s*$`)
)

// cleanTalkPath normalizes a .go path as typed: surrounding
// punctuation and a leading ./ come off.
func cleanTalkPath(s string) string {
	s = strings.Trim(s, `"'(),;:!?`)
	s = strings.TrimPrefix(s, "./")
	return s
}

// talkFileExists reports whether rel is one of the project's
// analyzed files.
func talkFileExists(p *analyze.Project, rel string) bool {
	if p == nil {
		return false
	}
	for _, pkg := range p.Packages {
		for _, f := range pkg.Files {
			if f == rel {
				return true
			}
		}
	}
	return false
}

// talkProjectFiles lists the project's Go files, sorted.
func talkProjectFiles(p *analyze.Project) []string {
	var out []string
	for _, pkg := range p.Packages {
		out = append(out, pkg.Files...)
	}
	sort.Strings(out)
	return out
}

// talkPackageForFile finds the package holding rel, the
// longest-directory match (the analyzer's own rule, restated
// here because its helper is unexported).
func talkPackageForFile(p *analyze.Project, rel string) *analyze.Package {
	dir := rel
	if i := strings.LastIndex(dir, "/"); i >= 0 {
		dir = dir[:i]
	} else {
		dir = ""
	}
	var best *analyze.Package
	for _, pkg := range p.Packages {
		if pkg.Dir == dir || strings.HasPrefix(dir, pkg.Dir+"/") {
			if best == nil || len(pkg.Dir) > len(best.Dir) {
				best = pkg
			}
		}
	}
	return best
}

// talkFindType resolves a type by bare or dotted name.
func talkFindType(p *analyze.Project, name string) *analyze.Type {
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if strings.EqualFold(t.Name, name) || strings.EqualFold(t.Pkg+"."+t.Name, name) {
				return t
			}
		}
	}
	return nil
}

// talkFindPackage resolves a package by name or directory.
func talkFindPackage(p *analyze.Project, name string) *analyze.Package {
	lower := strings.ToLower(strings.TrimSpace(name))
	for _, pkg := range p.Packages {
		short := pkg.Dir
		if i := strings.LastIndex(short, "/"); i >= 0 {
			short = short[i+1:]
		}
		if strings.EqualFold(pkg.Name, name) || strings.EqualFold(pkg.Dir, name) || strings.EqualFold(short, lower) {
			return pkg
		}
	}
	return nil
}

// cleanTalkSymbol strips articles, quotes, and trailing role
// words ("the Blorp handler" -> "Blorp") from a name as typed.
// It never invents: an unresolvable phrase simply won't
// resolve downstream.
func cleanTalkSymbol(s string) string {
	s = strings.TrimSpace(s)
	s = strings.Trim(s, `"'().,?!`)
	s = strings.TrimPrefix(s, "the ")
	for {
		trimmed := s
		for _, w := range []string{" function", " func", " method", " type", " struct", " interface", " handler", " package"} {
			trimmed = strings.TrimSuffix(trimmed, w)
		}
		if trimmed == s {
			break
		}
		s = trimmed
	}
	return strings.TrimSpace(strings.Trim(s, `"'().,?!`))
}

// isTalkNameShape reports whether a candidate reads as a
// project name — a single capitalized or dotted identifier —
// rather than a general concept ("a goroutine"). Only
// name-shaped candidates earn the honest not-here answer;
// concept-shaped lines fall through to the knowledge base
// and the social arm untouched.
func isTalkNameShape(name string) bool {
	if name == "" || strings.ContainsAny(name, " \t") {
		return false
	}
	if strings.Contains(name, ".") {
		last := name[strings.LastIndex(name, ".")+1:]
		return last != "" && last[0] >= 'A' && last[0] <= 'Z'
	}
	return name[0] >= 'A' && name[0] <= 'Z'
}

// nearTalkNames lists real project names resembling a missing
// candidate (substring either way, case-insensitive) — the
// "what does exist nearby" of the honest answer. Only real
// names from the digest list are ever offered.
func nearTalkNames(ctx TalkContext, name string) []string {
	lower := strings.ToLower(name)
	if i := strings.LastIndex(lower, "."); i >= 0 {
		lower = lower[i+1:]
	}
	if lower == "" {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, n := range ctx.Names {
		ln := strings.ToLower(n)
		if seen[n] || ln == lower {
			continue
		}
		if strings.Contains(ln, lower) || strings.Contains(lower, ln) {
			seen[n] = true
			out = append(out, n)
			if len(out) >= 5 {
				break
			}
		}
	}
	return out
}

// talkHonestMissing renders the not-here answer for a named
// thing the index does not hold, with near real names when
// the lookup found any.
func talkHonestMissing(ctx TalkContext, kind, name string) string {
	var b strings.Builder
	if kind != "" {
		fmt.Fprintf(&b, "I don't see a %s called %s in this project.", kind, name)
	} else {
		fmt.Fprintf(&b, "I don't see anything called %s in this project.", name)
	}
	if near := nearTalkNames(ctx, name); len(near) > 0 {
		fmt.Fprintf(&b, " Names that are here: %s.", strings.Join(near, ", "))
	}
	return b.String()
}

// talkResolveLoose resolves a candidate the forgiving way the
// chat's own lookups do (findFuncByQuery for functions, then
// types, then packages) and renders its detail card.
func talkResolveLoose(ctx TalkContext, name string) (string, bool) {
	p := ctx.Project
	if fn := findFuncByQuery(p, name); fn != nil {
		return strings.TrimSpace(fn.Detail(p)), true
	}
	if t := talkFindType(p, name); t != nil {
		return strings.TrimSpace(t.Detail(p)), true
	}
	if pkg := talkFindPackage(p, name); pkg != nil {
		return strings.TrimSpace(p.PackageDetail(pkg)), true
	}
	return "", false
}

// talkProjectAnswer is composition step 1: the project-fact
// second pass. See Respond for the contract.
func talkProjectAnswer(ctx TalkContext, line string) (string, bool) {
	p := ctx.Project
	if p == nil {
		return "", false
	}
	// A named .go file first — existing or not, the line is
	// unmistakably about this project's files, and a missing
	// file earns the honest answer, never an invented tour.
	if m := talkGoFileRe.FindStringSubmatch(line); m != nil {
		rel := cleanTalkPath(m[1])
		if talkFileExists(p, rel) {
			return talkFileAnswer(ctx, rel), true
		}
		files := talkProjectFiles(p)
		shown := files
		if len(shown) > 8 {
			shown = shown[:8]
		}
		return fmt.Sprintf("I don't see a file called %s in this project. The Go files here are: %s.", rel, strings.Join(shown, ", ")), true
	}
	// "What is this project?" — the framing story plus the
	// digest's package lines.
	if talkOverviewRe.MatchString(line) {
		var b strings.Builder
		b.WriteString(ctx.Story)
		if ctx.Digest != "" {
			b.WriteString("\nPackages here:\n")
			b.WriteString(ctx.Digest)
		}
		return b.String(), true
	}
	// Existence questions, answered yes with facts or no
	// with the honest wording.
	for _, re := range []*regexp.Regexp{talkExistRe, talkExistRe2} {
		if m := re.FindStringSubmatch(line); m != nil {
			kind, name := m[1], cleanTalkSymbol(m[2])
			wantFunc := kind == "function" || kind == "func"
			if wantFunc {
				if fn := findFuncByQuery(p, name); fn != nil {
					return "Yes — " + strings.TrimSpace(fn.Detail(p)), true
				}
			} else if t := talkFindType(p, name); t != nil {
				return "Yes — " + strings.TrimSpace(t.Detail(p)), true
			}
			return talkHonestMissing(ctx, kind, name), true
		}
	}
	// "What functions are in the store package?" — read from
	// the package's real declarations.
	if m := talkPkgFuncsRe.FindStringSubmatch(line); m != nil {
		what := strings.ToLower(m[1])
		pkgName := cleanTalkSymbol(m[2])
		pkg := talkFindPackage(p, pkgName)
		if pkg == nil {
			var pkgs []string
			for _, pk := range p.Packages {
				pkgs = append(pkgs, pk.Name)
			}
			sort.Strings(pkgs)
			return fmt.Sprintf("I don't see a package called %s in this project. The packages here are: %s.", pkgName, strings.Join(pkgs, ", ")), true
		}
		dir := pkg.Dir
		if dir == "" {
			dir = "."
		}
		var b strings.Builder
		if what == "types" {
			fmt.Fprintf(&b, "The %s package (%s/) has %d type(s):", pkg.Name, dir, len(pkg.Types))
			for _, t := range pkg.Types {
				fmt.Fprintf(&b, "\n  type %s — %s (%s:%d)", t.Name, t.Kind, p.LinkPath(t.File), t.Line)
			}
		} else {
			var fns []*analyze.Func
			for _, fn := range pkg.Funcs {
				if !fn.IsTest {
					fns = append(fns, fn)
				}
			}
			fmt.Fprintf(&b, "The %s package (%s/) has %d function(s):", pkg.Name, dir, len(fns))
			for _, fn := range fns {
				fmt.Fprintf(&b, "\n  %s (%s:%d)", strings.TrimPrefix(fn.Sig, "func "), p.LinkPath(fn.File), fn.Line)
			}
		}
		return b.String(), true
	}
	// Parameter and return questions, from the real signature.
	if m := talkTakeRe.FindStringSubmatch(line); m != nil {
		name := cleanTalkSymbol(m[1])
		if fn := findFuncByQuery(p, name); fn != nil {
			params := talkSigParams(fn.Sig)
			if params == "" {
				return fmt.Sprintf("%s takes no parameters. Its full signature is %s (%s:%d).", fn.Name, fn.Sig, p.LinkPath(fn.File), fn.Line), true
			}
			return fmt.Sprintf("%s takes: %s. Its full signature is %s (%s:%d).", fn.Name, params, fn.Sig, p.LinkPath(fn.File), fn.Line), true
		}
		if isTalkNameShape(name) {
			return talkHonestMissing(ctx, "", name), true
		}
		return "", false
	}
	if m := talkReturnRe.FindStringSubmatch(line); m != nil {
		name := cleanTalkSymbol(m[1])
		if fn := findFuncByQuery(p, name); fn != nil {
			res := talkSigResults(fn.Sig)
			if res == "" {
				return fmt.Sprintf("%s returns nothing. Its full signature is %s (%s:%d).", fn.Name, fn.Sig, p.LinkPath(fn.File), fn.Line), true
			}
			return fmt.Sprintf("%s returns: %s. Its full signature is %s (%s:%d).", fn.Name, res, fn.Sig, p.LinkPath(fn.File), fn.Line), true
		}
		if isTalkNameShape(name) {
			return talkHonestMissing(ctx, "", name), true
		}
		return "", false
	}
	if m := talkTypeIsRe.FindStringSubmatch(line); m != nil {
		name := cleanTalkSymbol(m[1])
		if t := talkFindType(p, name); t != nil {
			return strings.TrimSpace(t.Detail(p)), true
		}
		if isTalkNameShape(name) {
			return talkHonestMissing(ctx, "type", name), true
		}
		return "", false
	}
	// The deterministic Q&A's own shapes, retried now that a
	// project is guaranteed analyzed — the first pass may
	// have declined only because nothing was analyzed yet.
	if out, cands, ok := p.AnswerCands(line); ok {
		setFuncPick(cands)
		return out, true
	}
	// A question in a Q&A shape naming a thing the index does
	// not hold: the honest not-here answer, when the name is
	// name-shaped. Loose resolution gets first chance — a
	// forgiving match is still a fact.
	if name, ok := talkQuestionCandidate(line); ok && isTalkNameShape(name) {
		if out, found := talkResolveLoose(ctx, name); found {
			return out, true
		}
		return talkHonestMissing(ctx, "", name), true
	}
	return "", false
}

// talkQuestionCandidate extracts the named subject from the
// Q&A question shapes the seat mirrors for its missing-name
// pass (the same shapes analyze/qa.go matches, restated here
// because the seat needs the candidate, not the answer).
func talkQuestionCandidate(line string) (string, bool) {
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`(?i)^\s*(?:so\s+)?what does (?:the )?(.+?) (?:do|call)\s*\??\s*$`),
		regexp.MustCompile(`(?i)^\s*how does (?:the )?(.+?) work\s*\??\s*$`),
		regexp.MustCompile(`(?i)^\s*tell me about (?:the )?(.+?)\s*\.?\s*$`),
		regexp.MustCompile(`(?i)^\s*what(?:'s| is) (?:the )?(.+?)\s*\??\s*$`),
		regexp.MustCompile(`(?i)^\s*where is (?:the )?(.+?) (?:defined|declared|located)\s*\??\s*$`),
		regexp.MustCompile(`(?i)^\s*(?:what|who) (?:calls|uses) (?:the )?(.+?)\s*\??\s*$`),
		regexp.MustCompile(`(?i)^\s*explain (?:the )?(.+?)\s*\.?\s*$`),
		regexp.MustCompile(`(?i)^\s*show me (?:the )?(.+?)\s*\.?\s*$`),
	}
	for _, re := range patterns {
		if m := re.FindStringSubmatch(line); m != nil {
			name := cleanTalkSymbol(m[1])
			lower := strings.ToLower(name)
			if strings.HasPrefix(lower, "a ") || strings.HasPrefix(lower, "an ") {
				return "", false
			}
			return name, true
		}
	}
	return "", false
}

// talkSigParams extracts the parameter list text from a
// rendered signature ("func Save(path string, data []byte)
// error" -> "path string, data []byte").
func talkSigParams(sig string) string {
	open := strings.Index(sig, "(")
	if open < 0 {
		return ""
	}
	depth := 0
	for i := open; i < len(sig); i++ {
		switch sig[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return strings.TrimSpace(sig[open+1 : i])
			}
		}
	}
	return ""
}

// talkSigResults extracts the result list text from a
// rendered signature ("" when the function returns nothing).
func talkSigResults(sig string) string {
	open := strings.Index(sig, "(")
	if open < 0 {
		return ""
	}
	depth := 0
	for i := open; i < len(sig); i++ {
		switch sig[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return strings.TrimSpace(sig[i+1:])
			}
		}
	}
	return ""
}

// talkFileAnswer renders the "what's in <file>" answer from
// the index: the file's own one-liner (DescribeFile), its
// real declarations with locations, and — when the file is
// the context's focus file — its true line count, counted
// from the capped body the context assembled.
func talkFileAnswer(ctx TalkContext, rel string) string {
	p := ctx.Project
	var b strings.Builder
	pkg := talkPackageForFile(p, rel)
	pkgName := ""
	if pkg != nil {
		pkgName = pkg.Name
	}
	fmt.Fprintf(&b, "%s — package %s.", rel, pkgName)
	if desc := p.DescribeFile(rel); desc != "" {
		fmt.Fprintf(&b, " %s.", strings.TrimSuffix(desc, "."))
	}
	type decl struct {
		line int
		text string
	}
	var decls []decl
	for _, pk := range p.Packages {
		for _, fn := range pk.Funcs {
			if fn.File == rel && !fn.IsTest {
				decls = append(decls, decl{fn.Line, fmt.Sprintf("%s (%s:%d)", strings.TrimPrefix(fn.Sig, "func "), p.LinkPath(fn.File), fn.Line)})
			}
		}
		for _, t := range pk.Types {
			if t.File == rel {
				decls = append(decls, decl{t.Line, fmt.Sprintf("type %s — %s (%s:%d)", t.Name, t.Kind, p.LinkPath(t.File), t.Line)})
			}
		}
	}
	sort.Slice(decls, func(i, j int) bool { return decls[i].line < decls[j].line })
	if len(decls) > 0 {
		b.WriteString("\nIt declares:")
		for _, d := range decls {
			fmt.Fprintf(&b, "\n  %s", d.text)
		}
	} else {
		b.WriteString("\nIt declares no functions or types.")
	}
	if ctx.FocusFile == rel && ctx.FocusBody != "" && !ctx.FocusCapped {
		fmt.Fprintf(&b, "\nThe file is %d lines long.", strings.Count(ctx.FocusBody, "\n")+1)
	}
	return b.String()
}

// --- Session-context "it" resolution (composition step 2) ---

var (
	talkItInRe      = regexp.MustCompile(`(?i)^\s*what(?:'s| is) in it\s*\??\s*$`)
	talkItPkgQRe    = regexp.MustCompile(`(?i)^\s*is it in (?:the )?(.+?) package\s*\??\s*$`)
	talkItInQRe     = regexp.MustCompile(`(?i)^\s*is it in (?:the )?([A-Za-z_][A-Za-z0-9_]*)\s*\??\s*$`)
	talkItWhatPkgRe = regexp.MustCompile(`(?i)^\s*what package is it in\s*\??\s*$`)
	talkItFolderQRe = regexp.MustCompile(`(?i)^\s*(?:which|what) folder is it in\s*\??\s*$`)
	talkItGenericRe = regexp.MustCompile(`(?i)^\s*(what does it do|how does it work|what is it|where is it|tell me about it|explain it|who calls it|what does it take|what does it return)\s*\??\s*$`)
)

// talkSubject is one resolved reading of "it": exactly one of
// fn / typ / file is set, plus the label that names the
// reading in beginner-true words.
type talkSubject struct {
	fn    *analyze.Func
	typ   *analyze.Type
	file  string
	label string
	pkg   string
}

// talkResolveSubject settles "it" from the session context:
// the last piece landed first (by name, in the last file when
// it is still there), then the last file, then the working
// folder's package, then a name the conversation itself
// mentioned (the transcript tail the context carries).
func talkResolveSubject(ctx TalkContext) (talkSubject, bool) {
	p := ctx.Project
	if p == nil {
		return talkSubject{}, false
	}
	if ctx.WorkLive {
		w := ctx.Work
		if w.pieceName != "" {
			// Prefer the piece in the file it landed in.
			if w.lastFile != "" {
				for _, pkg := range p.Packages {
					for _, fn := range pkg.Funcs {
						if fn.Name == w.pieceName && fn.File == w.lastFile {
							return talkSubject{fn: fn, pkg: fn.Pkg, label: fmt.Sprintf("%s — the piece you just landed in %s", fn.Display(), w.lastFile)}, true
						}
					}
					for _, t := range pkg.Types {
						if t.Name == w.pieceName && t.File == w.lastFile {
							return talkSubject{typ: t, pkg: t.Pkg, label: fmt.Sprintf("%s — the piece you just landed in %s", t.Name, w.lastFile)}, true
						}
					}
				}
			}
			if out, found := talkResolveSubjectByName(ctx, w.pieceName); found {
				out.label = fmt.Sprintf("%s — the piece you just landed in %s", subjectDisplay(out), w.lastFile)
				return out, true
			}
		}
		if w.lastFile != "" && talkFileExists(p, w.lastFile) {
			how := w.lastHow
			if how == "" {
				how = "touched"
			}
			pkgName := ""
			if pkg := talkPackageForFile(p, w.lastFile); pkg != nil {
				pkgName = pkg.Name
			}
			return talkSubject{file: w.lastFile, pkg: pkgName, label: fmt.Sprintf("%s — the file you just %s", w.lastFile, how)}, true
		}
		if w.folder != "" {
			if pkg := talkFindPackage(p, w.folder); pkg != nil {
				return talkSubject{file: w.folder, pkg: pkg.Name, label: fmt.Sprintf("the %s package — the folder you're working in", pkg.Name)}, true
			}
		}
	}
	// Transcript fallback: the last project name the
	// conversation itself mentioned.
	names := map[string]bool{}
	for _, n := range ctx.Names {
		names[n] = true
	}
	for i := len(ctx.Turns) - 1; i >= 0; i-- {
		for _, field := range strings.FieldsFunc(ctx.Turns[i].Text, func(r rune) bool {
			return !(r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '_' || r == '.')
		}) {
			name := strings.Trim(field, ".")
			if names[name] {
				if out, found := talkResolveSubjectByName(ctx, name); found {
					out.label = fmt.Sprintf("%s — the one the conversation just mentioned", subjectDisplay(out))
					return out, true
				}
			}
		}
	}
	return talkSubject{}, false
}

func subjectDisplay(s talkSubject) string {
	if s.fn != nil {
		return s.fn.Display()
	}
	if s.typ != nil {
		return s.typ.Name
	}
	return s.file
}

// talkResolveSubjectByName resolves a bare name to a subject.
func talkResolveSubjectByName(ctx TalkContext, name string) (talkSubject, bool) {
	p := ctx.Project
	if fn := findFuncByQuery(p, name); fn != nil {
		return talkSubject{fn: fn, pkg: fn.Pkg}, true
	}
	if t := talkFindType(p, name); t != nil {
		return talkSubject{typ: t, pkg: t.Pkg}, true
	}
	return talkSubject{}, false
}

// talkItAnswer is composition step 2. Every answer names its
// reading first ("Validate — the function you just added to
// api/validate.go — ..."), per the session-context rule that
// a resolution is always visible.
func talkItAnswer(ctx TalkContext, line string) (string, bool) {
	p := ctx.Project
	if p == nil {
		return "", false
	}
	// "What's in it?" is a container question: the file the
	// session is working on, not the piece inside it.
	if talkItInRe.MatchString(line) {
		if ctx.WorkLive && ctx.Work.lastFile != "" && talkFileExists(p, ctx.Work.lastFile) {
			label := fmt.Sprintf("Reading \"it\" as %s — %s.", ctx.Work.lastFile, ctx.Work.lastFileReason())
			return label + "\n" + talkFileAnswer(ctx, ctx.Work.lastFile), true
		}
		if subj, ok := talkResolveSubject(ctx); ok && subj.file != "" {
			return subj.label + ":\n" + talkFileAnswer(ctx, subj.file), true
		}
		return "", false
	}
	if m := talkItPkgQRe.FindStringSubmatch(line); m != nil {
		asked := cleanTalkSymbol(m[1])
		subj, ok := talkResolveSubject(ctx)
		if !ok {
			return "", false
		}
		askedPkg := talkFindPackage(p, asked)
		match := subj.pkg != "" && (strings.EqualFold(subj.pkg, asked) || (askedPkg != nil && strings.EqualFold(subj.pkg, askedPkg.Name)))
		if match {
			return fmt.Sprintf("Yes — %s is in the %s package.", subj.label, subj.pkg), true
		}
		if subj.pkg != "" {
			return fmt.Sprintf("No — %s is in the %s package, not the %s package.", subj.label, subj.pkg, asked), true
		}
		return "", false
	}
	// The package question without the word "package" —
	// the short follow-up form ("is it in greet?").
	if m := talkItInQRe.FindStringSubmatch(line); m != nil {
		asked := m[1]
		subj, ok := talkResolveSubject(ctx)
		if !ok || subj.pkg == "" {
			return "", false
		}
		if strings.EqualFold(subj.pkg, asked) {
			return fmt.Sprintf("Yes — %s is in %s.", subj.label, asked), true
		}
		return fmt.Sprintf("No — %s is in the %s package, not in %s.", subj.label, subj.pkg, asked), true
	}
	if talkItWhatPkgRe.MatchString(line) {
		subj, ok := talkResolveSubject(ctx)
		if !ok || subj.pkg == "" {
			return "", false
		}
		return fmt.Sprintf("%s is in the %s package.", subj.label, subj.pkg), true
	}
	if talkItFolderQRe.MatchString(line) {
		subj, ok := talkResolveSubject(ctx)
		if !ok {
			return "", false
		}
		file := subj.file
		if subj.fn != nil {
			file = subj.fn.File
		}
		if subj.typ != nil {
			file = subj.typ.File
		}
		if file == "" {
			return "", false
		}
		folder := slashDir(file)
		if folder == "" {
			return fmt.Sprintf("%s is in the project root folder (%s).", subj.label, file), true
		}
		return fmt.Sprintf("%s is in the %s folder (%s).", subj.label, folder, file), true
	}
	if m := talkItGenericRe.FindStringSubmatch(line); m != nil {
		subj, ok := talkResolveSubject(ctx)
		if !ok {
			return "", false
		}
		verb := strings.ToLower(m[1])
		switch {
		case strings.HasPrefix(verb, "where"):
			if subj.fn != nil {
				return fmt.Sprintf("%s is defined at %s:%d.", subj.label, p.LinkPath(subj.fn.File), subj.fn.Line), true
			}
			if subj.typ != nil {
				return fmt.Sprintf("%s is defined at %s:%d.", subj.label, p.LinkPath(subj.typ.File), subj.typ.Line), true
			}
			return fmt.Sprintf("%s — that's where it lives.", subj.label), true
		case strings.HasPrefix(verb, "who calls"):
			if subj.fn != nil {
				if out, _, ok := p.AnswerCands("what calls " + subj.fn.Name); ok {
					return subj.label + ":\n" + out, true
				}
			}
			return "", false
		case strings.HasPrefix(verb, "what does it take"):
			if subj.fn != nil {
				if params := talkSigParams(subj.fn.Sig); params != "" {
					return fmt.Sprintf("%s takes: %s.", subj.label, params), true
				}
				return fmt.Sprintf("%s takes no parameters.", subj.label), true
			}
			return "", false
		case strings.HasPrefix(verb, "what does it return"):
			if subj.fn != nil {
				if res := talkSigResults(subj.fn.Sig); res != "" {
					return fmt.Sprintf("%s returns: %s.", subj.label, res), true
				}
				return fmt.Sprintf("%s returns nothing.", subj.label), true
			}
			return "", false
		default: // what does it do / how does it work / what is it / tell me about it / explain it
			if subj.fn != nil {
				return subj.label + ":\n" + strings.TrimSpace(subj.fn.Detail(p)), true
			}
			if subj.typ != nil {
				return subj.label + ":\n" + strings.TrimSpace(subj.typ.Detail(p)), true
			}
			if subj.file != "" && talkFileExists(p, subj.file) {
				return subj.label + ":\n" + talkFileAnswer(ctx, subj.file), true
			}
			return "", false
		}
	}
	return "", false
}
