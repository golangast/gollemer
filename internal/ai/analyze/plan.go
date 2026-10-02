package analyze

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// A featurePlan is the multi-file answer for an "add" issue whose
// mechanism match isn't what the issue is about. Pointing at one anchor
// misleads when the real change spans config, flags, a flow, helpers,
// and a new file — so the plan names each with its role, grounded in
// the actual code.
type featurePlan struct {
	configType  *Type            // type Config struct (fallback when no YAML match)
	configPkg   *Package
	flagsFunc   *Func            // the flag-parsing function
	newFlags    []string         // --flags from the issue not covered by Config fields
	flow        *Func            // the orchestrating function to extend
	flowCall    *Func            // the called seed the feature hooks into
	helpers     []*Func          // similar existing functions to imitate
	helperNotes map[string]string // func ID -> why, e.g. "the issue names `--log-json`"
	newFile     string           // suggested new file, relative to root ("" = none)
	noun        string           // the feature's artifact noun ("report")
}

var planStopwords = map[string]bool{
	"feature": true, "request": true, "bug": true, "issue": true,
	"add": true, "new": true, "support": true, "proposal": true,
	"please": true, "would": true, "like": true, "want": true,
	"need": true, "make": true, "use": true, "using": true,
	"with": true, "for": true, "from": true, "that": true,
	"this": true, "have": true, "should": true, "could": true,
	"there": true, "when": true, "what": true, "which": true,
	"also": true, "just": true, "into": true, "about": true,
	"the": true, "and": true, "are": true, "can": true,
}

// stem reduces a word to a crude root so "deletion", "delete" and
// "deleted" match each other.
func stem(w string) string {
	w = strings.ToLower(w)
	for _, suf := range []string{"ing", "ion", "ed", "es", "s"} {
		if strings.HasSuffix(w, suf) && len(w) > len(suf)+2 {
			if suf == "s" && strings.HasSuffix(w, "ss") {
				continue
			}
			return w[:len(w)-len(suf)]
		}
	}
	return w
}

// titleNouns returns the significant stemmed words of the issue title,
// plus the unstemmed words for display.
func titleNouns(title string) (stems, words []string) {
	seen := map[string]bool{}
	for _, w := range strings.FieldsFunc(strings.ToLower(title),
		func(r rune) bool { return r < 'a' || r > 'z' }) {
		if len(w) < 3 || planStopwords[w] {
			continue
		}
		s := stem(w)
		if !seen[s] {
			seen[s] = true
			stems = append(stems, s)
			words = append(words, w)
		}
	}
	return stems, words
}

// tokenizeName splits a camelCase identifier into stemmed words:
// WriteJSONLog -> [write json log], LogDeletionToFileAsJson ->
// [log delet to file as json]. Consecutive capitals stay together
// unless followed by a lowercase letter (JSONLog -> JSON + Log).
func tokenizeName(name string) []string {
	isUpper := func(r rune) bool { return r >= 'A' && r <= 'Z' }
	isLower := func(r rune) bool { return r >= 'a' && r <= 'z' }
	isDigit := func(r rune) bool { return r >= '0' && r <= '9' }
	runes := []rune(name)
	var toks []string
	start := 0
	flush := func(end int) {
		if w := stem(string(runes[start:end])); w != "" {
			toks = append(toks, w)
		}
		start = end
	}
	for i := 1; i < len(runes); i++ {
		r, prev := runes[i], runes[i-1]
		switch {
		case r == '_' || r == '-':
			flush(i)
			start = i + 1
		case isUpper(r) && (isLower(prev) || isDigit(prev)):
			flush(i) // write|JSON
		case isUpper(r) && isUpper(prev) && i+1 < len(runes) && isLower(runes[i+1]):
			flush(i) // JSON|Log
		case (isDigit(r) && !isDigit(prev)) || (!isDigit(r) && isDigit(prev)):
			flush(i)
		}
	}
	flush(len(runes))
	return toks
}

// wordTokens splits free text into stemmed words of length 2+.
func wordTokens(s string) []string {
	var out []string
	for _, w := range strings.FieldsFunc(strings.ToLower(s),
		func(r rune) bool { return r < 'a' || r > 'z' }) {
		if len(w) >= 2 {
			out = append(out, stem(w))
		}
	}
	return out
}

type funcSeed struct {
	fn    *Func
	score int
}

// funcSeeds ranks non-test functions by overlap with keywords: name hits
// count most, then doc, then package.
func (p *Project) funcSeeds(keywords []string, n int) []funcSeed {
	var seeds []funcSeed
	for _, pkg := range p.Packages {
		pkgHay := strings.ToLower(pkg.Name + " " + pkg.Dir)
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			nameToks := tokenizeName(fn.Name)
			docHay := strings.ToLower(fn.Doc)
			score := 0
			for _, kw := range keywords {
				for _, t := range nameToks {
					if t == kw {
						score += 3
						break
					}
				}
				if strings.Contains(docHay, kw) {
					score += 2
				}
				if strings.Contains(pkgHay, kw) {
					score++
				}
			}
			if score > 0 {
				seeds = append(seeds, funcSeed{fn, score})
			}
		}
	}
	sort.Slice(seeds, func(i, j int) bool {
		if seeds[i].score != seeds[j].score {
			return seeds[i].score > seeds[j].score
		}
		return seeds[i].fn.Name < seeds[j].fn.Name
	})
	if len(seeds) > n {
		seeds = seeds[:n]
	}
	return seeds
}

var flagWordRe = regexp.MustCompile(`^--([a-zA-Z][a-zA-Z0-9-]*)`)

// codeRefs finds functions the issue names through `backticked` words:
// every word of the code word must appear in the function name.
// Single-word code words are too generic to be references.
func (p *Project) codeRefs(c IssueConcepts) map[string][]*Func {
	out := map[string][]*Func{}
	for _, w := range c.CodeWords {
		toks := wordTokens(w)
		if len(toks) < 2 {
			continue
		}
		for _, pkg := range p.Packages {
			for _, fn := range pkg.Funcs {
				if fn.IsTest {
					continue
				}
				nameSet := map[string]bool{}
				for _, t := range tokenizeName(fn.Name) {
					nameSet[t] = true
				}
				all := true
				for _, t := range toks {
					if !nameSet[t] {
						all = false
						break
					}
				}
				if all {
					out[w] = append(out[w], fn)
				}
			}
		}
	}
	return out
}

// findFlow picks the orchestrating function behind candidate funcs: walk
// each candidate's callers up toward main and take the non-main function
// with the most callees. Constructors never orchestrate a feature.
func (p *Project) findFlow(cands []*Func) *Func {
	seen := map[string]*Func{}
	var walk func(fn *Func)
	walk = func(fn *Func) {
		if fn == nil || seen[fn.ID] != nil {
			return
		}
		seen[fn.ID] = fn
		for _, id := range fn.Callers {
			walk(p.byID[id])
		}
	}
	for _, c := range cands {
		walk(c)
	}
	var best *Func
	for _, fn := range seen {
		if fn.IsMain || fn.IsTest || fn.IsInit || isConstructor(fn.Name) {
			continue
		}
		if best == nil || len(fn.Calls) > len(best.Calls) {
			best = fn
		}
	}
	return best
}

func isConstructor(name string) bool {
	return strings.HasPrefix(name, "New")
}

// findConfigType locates "type Config struct", preferring config packages.
func (p *Project) findConfigType() (*Type, *Package) {
	var best *Type
	var bestPkg *Package
	bestScore := -1
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind != "struct" || t.Name != "Config" {
				continue
			}
			score := 0
			if strings.Contains(strings.ToLower(pkg.Dir), "config") {
				score += 2
			}
			if strings.Contains(strings.ToLower(pkg.Name), "config") {
				score++
			}
			if score > bestScore {
				bestScore, best, bestPkg = score, t, pkg
			}
		}
	}
	return best, bestPkg
}

// findFlagsFunc finds the flag-parsing function in the config package:
// a name with "flag" in it, else a file with "flag" in its name.
func findFlagsFunc(pkg *Package) *Func {
	for _, fn := range pkg.Funcs {
		if !fn.IsTest && strings.Contains(strings.ToLower(fn.Name), "flag") {
			return fn
		}
	}
	for _, fn := range pkg.Funcs {
		if !fn.IsTest && strings.Contains(strings.ToLower(fn.File), "flag") {
			return fn
		}
	}
	return nil
}

// siteMatchesTitle reports whether the mechanism site is what the issue
// title is about: the title names a mechanism word and that mechanism
// lives in the site's package. Otherwise a single anchor misleads —
// "Add Deletion Report Export" anchored on the file manager because the
// issue says "files" — and the multi-file plan takes over.
func (p *Project) siteMatchesTitle(c IssueConcepts, site *MechanismSite) bool {
	if site == nil {
		return false
	}
	words := map[string]bool{}
	for _, w := range strings.FieldsFunc(strings.ToLower(c.Title),
		func(r rune) bool { return r < 'a' || r > 'z' }) {
		words[w] = true
		words[singular(w)] = true
	}
	for _, m := range mechanismTable {
		if words[m] {
			if s := p.FindMechanism(m); s != nil && s.Pkg == site.Pkg {
				return true
			}
		}
	}
	return false
}

// flagsSignal reports whether the issue talks about user-facing options.
func flagsSignal(c IssueConcepts) bool {
	for _, w := range c.CodeWords {
		if strings.HasPrefix(w, "--") {
			return true
		}
	}
	lower := strings.ToLower(c.Title + "\n" + c.Body)
	for _, w := range []string{"flag", "option", "cli"} {
		if strings.Contains(lower, w) {
			return true
		}
	}
	return false
}

// newFlags returns the issue's --flags not already covered by the Config
// struct's fields: those are the options being added.
func newFlags(c IssueConcepts, t *Type) []string {
	fieldToks := map[string]bool{}
	for _, f := range t.Fields {
		for _, tok := range tokenizeName(f) {
			fieldToks[tok] = true
		}
	}
	var out []string
	seen := map[string]bool{}
	for _, w := range c.CodeWords {
		m := flagWordRe.FindStringSubmatch(w)
		if m == nil || seen[m[0]] {
			continue
		}
		seen[m[0]] = true
		covered := true
		for _, tok := range wordTokens(m[1]) {
			if !fieldToks[tok] {
				covered = false
				break
			}
		}
		if !covered {
			out = append(out, "--"+m[1])
		}
	}
	return out
}

// planNoun names the feature's artifact from the issue's own --flags:
// --report and --report-format agree on "report".
func planNoun(c IssueConcepts) string {
	counts := map[string]int{}
	best, bestN := "", 0
	for _, w := range c.CodeWords {
		m := flagWordRe.FindStringSubmatch(w)
		if m == nil {
			continue
		}
		noun := strings.Split(m[1], "-")[0]
		if len(noun) < 3 {
			continue
		}
		counts[noun]++
		if counts[noun] > bestN {
			bestN, best = counts[noun], noun
		}
	}
	return best
}

// planFileExists reports whether the package dir already has a file
// starting with noun.
func planFileExists(p *Project, dir, noun string) bool {
	for _, pkg := range p.Packages {
		if pkg.Dir != dir {
			continue
		}
		for _, f := range pkg.Files {
			base := f[strings.LastIndex(f, "/")+1:]
			if strings.HasPrefix(strings.ToLower(base), strings.ToLower(noun)) {
				return true
			}
		}
	}
	return false
}

// buildFeaturePlan assembles the multi-file change plan, or nil when the
// pieces don't come together (then the single-anchor answer stands).
func (p *Project) buildFeaturePlan(c IssueConcepts) *featurePlan {
	stems, _ := titleNouns(c.Title)
	seen := map[string]bool{}
	var keywords []string
	for _, k := range append(append([]string{}, stems...), c.Mechanisms...) {
		k = stem(k)
		if !seen[k] {
			seen[k] = true
			keywords = append(keywords, k)
		}
	}

	seeds := p.funcSeeds(keywords, 8)
	refs := p.codeRefs(c)

	cands := []*Func{}
	candSeen := map[string]bool{}
	addCand := func(fn *Func) {
		if fn != nil && !candSeen[fn.ID] {
			candSeen[fn.ID] = true
			cands = append(cands, fn)
		}
	}
	for _, s := range seeds {
		addCand(s.fn)
	}
	for _, fns := range refs {
		if len(fns) > 3 {
			continue // too generic to be a reference
		}
		for _, fn := range fns {
			addCand(fn)
		}
	}
	if len(cands) == 0 {
		return nil
	}
	flow := p.findFlow(cands)
	if flow == nil {
		return nil
	}

	fp := &featurePlan{flow: flow, helperNotes: map[string]string{}}

	// Keyword relevance per function.
	seedScore := map[string]int{}
	for _, s := range seeds {
		seedScore[s.fn.ID] = s.score
	}

	// The anchor call: among the flow's callees, prefer one made inside a
	// loop (the per-item hook), then keyword score, then the title's
	// action noun, then brevity.
	action := ""
	if stems, _ := titleNouns(c.Title); len(stems) > 0 {
		action = stems[0]
	}
	betterCall := func(a, b *Func) bool {
		al, bl := flow.LoopCalls[a.ID], flow.LoopCalls[b.ID]
		if al != bl {
			return al
		}
		sa, sb := seedScore[a.ID], seedScore[b.ID]
		if sa != sb {
			return sa > sb
		}
		am := action != "" && strings.Contains(strings.ToLower(a.Name), action)
		bm := action != "" && strings.Contains(strings.ToLower(b.Name), action)
		if am != bm {
			return am
		}
		return len(a.Name) < len(b.Name)
	}
	seenC := map[string]bool{}
	for _, id := range flow.Calls {
		if seenC[id] {
			continue
		}
		seenC[id] = true
		fn := p.byID[id]
		if fn == nil || fn.IsTest || fn.ID == flow.ID {
			continue
		}
		if fp.flowCall == nil || betterCall(fn, fp.flowCall) {
			fp.flowCall = fn
		}
	}

	// Helpers: code the issue explicitly names comes first, then top
	// seeds. The flow itself, constructors, and funcs the flow already
	// calls are excluded — those belong to BEHAVIOR.
	addHelper := func(fn *Func, note string) {
		if fn.ID == flow.ID || isConstructor(fn.Name) ||
			(fp.flowCall != nil && fn.ID == fp.flowCall.ID) {
			return
		}
		for _, h := range fp.helpers {
			if h.ID == fn.ID {
				return
			}
		}
		if len(fp.helpers) >= 2 {
			return
		}
		fp.helpers = append(fp.helpers, fn)
		if note != "" {
			fp.helperNotes[fn.ID] = note
		}
	}
	var words []string
	for w := range refs {
		words = append(words, w)
	}
	sort.Strings(words)
	for _, w := range words {
		fns := refs[w]
		if len(fns) > 3 {
			continue // too generic to be a reference
		}
		sort.Slice(fns, func(i, j int) bool {
			si, sj := seedScore[fns[i].ID], seedScore[fns[j].ID]
			if si != sj {
				return si > sj
			}
			return fns[i].Name < fns[j].Name
		})
		for _, fn := range fns {
			addHelper(fn, fmt.Sprintf("the issue names `%s`", w))
		}
	}

	// Config + flags, when the issue talks about user-facing options.
	if flagsSignal(c) {
		if t, pkg := p.findConfigType(); t != nil {
			fp.configType, fp.configPkg = t, pkg
			fp.flagsFunc = findFlagsFunc(pkg)
			fp.newFlags = newFlags(c, t)
		}
	}

	// New-file hint, named by the issue's own --flags.
	if len(fp.helpers) > 0 {
		if noun := planNoun(c); noun != "" {
			dir := fp.helpers[0].PkgDir
			if !planFileExists(p, dir, noun) {
				fp.newFile = dir + "/" + noun + ".go"
				fp.noun = noun
			}
		}
	}
	return fp
}

// writeFeaturePlan renders the plan. yamlShown tells whether GuideIssue
// already printed the YAML config section, so the fallback is skipped.
func (p *Project) writeFeaturePlan(c IssueConcepts, fp *featurePlan, yamlShown bool) string {
	var b strings.Builder

	if !yamlShown && fp.configType != nil {
		fmt.Fprintf(&b, "CONFIG — put the new options here:\n")
		fmt.Fprintf(&b, "  %s:%d\n", p.LinkPath(fp.configType.File), fp.configType.Line)
		fmt.Fprintf(&b, "  type %s struct\n", fp.configType.Name)
		if len(fp.newFlags) > 0 {
			fmt.Fprintf(&b, "  Add a field for each new flag: %s.\n", strings.Join(quoteAll(fp.newFlags), ", "))
		}
		fmt.Fprintf(&b, "  Why here: every run flows through this struct, so one new field reaches them all.\n\n")
	}

	if fp.flagsFunc != nil {
		fmt.Fprintf(&b, "FLAGS — parse the new flags here:\n")
		fmt.Fprintf(&b, "  %s:%d\n", p.LinkPath(fp.flagsFunc.File), fp.flagsFunc.Line)
		fmt.Fprintf(&b, "  %s\n", strings.TrimPrefix(fp.flagsFunc.Sig, "func "))
		fmt.Fprintf(&b, "  Why here: this is where CLI flags become %s.\n\n", fp.configType.Name)
	}

	fmt.Fprintf(&b, "BEHAVIOR — build it here:\n")
	fmt.Fprintf(&b, "  %s:%d\n", p.LinkPath(fp.flow.File), fp.flow.Line)
	fmt.Fprintf(&b, "  %s\n", strings.TrimPrefix(fp.flow.Sig, "func "))
	if fp.flowCall != nil {
		fmt.Fprintf(&b, "  Why here: `%s` already calls `%s` — it visits every file the feature must cover, so hook the new logic into this flow.\n",
			fp.flow.Name, fp.flowCall.Name)
	} else {
		fmt.Fprintf(&b, "  Why here: this is the flow the feature belongs in.\n")
	}
	b.WriteString("\n")

	for _, h := range fp.helpers {
		fmt.Fprintf(&b, "HELPERS — put the new code near this:\n")
		fmt.Fprintf(&b, "  %s:%d\n", p.LinkPath(h.File), h.Line)
		fmt.Fprintf(&b, "  %s\n", strings.TrimPrefix(h.Sig, "func "))
		if d := oneLineDoc(h.Doc); d != "" {
			fmt.Fprintf(&b, "  %s\n", d)
		}
		if note := fp.helperNotes[h.ID]; note != "" {
			fmt.Fprintf(&b, "  %s — the new code belongs next to it.\n", note)
		}
		b.WriteString("\n")
	}

	if fp.newFile != "" {
		fmt.Fprintf(&b, "NEW CODE — this is a new capability:\n")
		fmt.Fprintf(&b, "  Consider a new file %s for the new %s code.\n\n", p.LinkPath(fp.newFile), fp.noun)
	}
	return b.String()
}
