package analyze

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// IssueConcepts is what the NLP layer pulls out of a feature request or
// bug report. Everything here is deterministic: no model call, just the
// issue's own words shaped into something the engine can match against.
type IssueConcepts struct {
	// Title is the short human name: "retry+backoff mechanism".
	Title string
	// YamlKeys are config keys from YAML examples in the issue:
	// kind, spec, transformers, retry, limit...
	YamlKeys []string
	// KeyIndents maps each YAML key to its indent level, so the engine
	// can tell struct-level keys (same indent as kind:) from nested
	// ones (file: under spec:) and wrappers (sources: above).
	KeyIndents map[string]int
	// CodeWords are `backticked` words: file names, symbols, commands.
	CodeWords []string
	// Mechanisms are how-words: url, http, fetch, file...
	Mechanisms []string
	// Action is "add" (feature) or "fix" (bug).
	Action string
	// Body is the raw issue text, kept for quoting.
	Body string
}

var (
	yamlKeyRe  = regexp.MustCompile(`(?m)^([ \t]*)([a-zA-Z_][a-zA-Z0-9_-]*)[ \t]*:`)
	codeWordRe = regexp.MustCompile("`([^`\\s][^`]*?)`")
	wordRe     = regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+]*`)
	fenceRe    = regexp.MustCompile("(?s)```.*?```")
)

// mechanismTable maps how-words to the package-name fragments they imply.
var mechanismTable = []string{
	"url", "http", "https", "fetch", "download", "request", "api",
	"file", "read", "network", "tcp", "tls", "socket",
	"retry", "backoff", "timeout", "ratelimit",
	"cache", "database", "sql", "git", "docker", "yaml", "json",
}

var addWords = []string{"feature", "request", "add", "support", "would like", "proposal", "new"}
var fixWords = []string{"bug", "fix", "broken", "fails", "failed", "error", "panic", "wrong", "incorrect"}

// ExtractIssueConcepts reads a GitHub issue (title + body) and pulls out
// the concepts the engine needs: what config it talks about, what
// mechanism it touches, and whether it asks to add or fix.
func ExtractIssueConcepts(text string) IssueConcepts {
	c := IssueConcepts{Body: text}
	lower := strings.ToLower(text)

	// Title: "Feature Request: X", "Bug: X", "# X", or the first line.
	c.Title = issueTitle(text)

	// YAML keys: from fenced code blocks first — that's where config
	// examples live. Prose lines like "Example:" or "Pipeline ID:"
	// would pollute the keys, so whole-text scan is only the fallback.
	// An issue can fence several blocks (error output + YAML example):
	// the block with the most keys is the config example.
	keySource := text
	bestFence := []string{}
	bestIndents := map[string]int{}
	for _, fence := range fenceRe.FindAllString(text, -1) {
		var ks []string
		seen := map[string]bool{}
		indents := map[string]int{}
		for _, m := range yamlKeyRe.FindAllStringSubmatch(fence, -1) {
			k := strings.ToLower(m[2])
			if !seen[k] {
				seen[k] = true
				ks = append(ks, k)
				indents[k] = len(m[1])
			}
		}
		if len(ks) > len(bestFence) {
			bestFence = ks
			bestIndents = indents
		}
	}
	if len(bestFence) > 0 {
		c.YamlKeys = bestFence
		c.KeyIndents = bestIndents
	} else {
		seen := map[string]bool{}
		c.KeyIndents = map[string]int{}
		for _, m := range yamlKeyRe.FindAllStringSubmatch(keySource, -1) {
			k := strings.ToLower(m[2])
			if !seen[k] {
				seen[k] = true
				c.YamlKeys = append(c.YamlKeys, k)
				c.KeyIndents[k] = len(m[1])
			}
		}
	}

	// `backticked` words.
	cseen := map[string]bool{}
	for _, m := range codeWordRe.FindAllStringSubmatch(text, -1) {
		w := strings.TrimSpace(m[1])
		if w != "" && !cseen[w] {
			cseen[w] = true
			c.CodeWords = append(c.CodeWords, w)
		}
	}

	// Mechanism how-words present in the text (singular-normalized:
	// "URLs" counts as "url").
	words := map[string]bool{}
	for _, w := range wordRe.FindAllString(lower, -1) {
		words[w] = true
		words[singular(w)] = true
	}
	for _, m := range mechanismTable {
		if words[m] {
			c.Mechanisms = append(c.Mechanisms, m)
		}
	}

	// Add or fix? The raw first line decides first: "Feature Request:"
	// is an add even when the body describes failures at length.
	c.Action = "add"
	rawFirst := ""
	for _, line := range strings.Split(text, "\n") {
		if strings.TrimSpace(line) != "" {
			rawFirst = strings.ToLower(strings.TrimSpace(line))
			break
		}
	}
	switch {
	case strings.Contains(rawFirst, "feature request") || strings.Contains(rawFirst, "proposal"):
		c.Action = "add"
	case strings.HasPrefix(rawFirst, "bug:") || strings.HasPrefix(rawFirst, "# bug"):
		c.Action = "fix"
	default:
		adds, fixes := 0, 0
		for _, w := range addWords {
			if strings.Contains(lower, w) {
				adds++
			}
		}
		for _, w := range fixWords {
			if strings.Contains(lower, w) {
				fixes++
			}
		}
		if fixes > adds {
			c.Action = "fix"
		}
	}
	return c
}

// singular trims a simple English plural: "urls" -> "url".
func singular(w string) string {
	if len(w) > 3 && strings.HasSuffix(w, "s") && !strings.HasSuffix(w, "ss") {
		return w[:len(w)-1]
	}
	return w
}

func issueTitle(text string) string {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		line = strings.TrimPrefix(line, "#")
		line = strings.TrimSpace(line)
		for _, p := range []string{"Feature Request:", "Feature request:", "Bug:", "Issue:"} {
			if strings.HasPrefix(line, p) {
				line = strings.TrimSpace(strings.TrimPrefix(line, p))
			}
		}
		if len(line) > 80 {
			line = line[:77] + "..."
		}
		return line
	}
	return "this change"
}

// ConfigMatch is a struct whose fields overlap the issue's YAML keys.
// Go config is structs: the issue's YAML example names the struct's
// fields, so matching them finds where new options belong.
type ConfigMatch struct {
	Type  *Type
	Pkg   *Package
	Hits  []string // issue YAML keys matched by field names
	New   []string // issue YAML keys NOT in the struct: the proposed additions
	Score int
}

// structFields returns a struct's own fields plus one level of embedded
// struct fields (resource.ResourceConfig inlines into source.Config).
func (p *Project) structFields(t *Type) []string {
	out := append([]string{}, t.Fields...)
	byName := map[string]*Type{}
	for _, pkg := range p.Packages {
		for _, ot := range pkg.Types {
			if ot.Kind == "struct" {
				byName[ot.Name] = ot
			}
		}
	}
	for _, f := range t.Fields {
		if emb, ok := byName[f]; ok && emb != t {
			out = append(out, emb.Fields...)
		}
	}
	return out
}

// FindConfigStruct finds the struct the issue's YAML example describes.
// The issue shows kind:/spec:/transformers: — the struct with fields
// Kind/Spec/Transformers is where a new retry: option would live.
func (p *Project) FindConfigStruct(keys []string) []ConfigMatch {
	if len(keys) == 0 {
		return nil
	}
	var out []ConfigMatch
	for _, pkg := range p.Packages {
		for _, t := range pkg.Types {
			if t.Kind != "struct" {
				continue
			}
			fields := map[string]bool{}
			for _, f := range p.structFields(t) {
				fields[strings.ToLower(f)] = true
			}
			var hits, newKeys []string
			for _, k := range keys {
				if fields[k] {
					hits = append(hits, k)
				} else {
					newKeys = append(newKeys, k)
				}
			}
			if len(hits) >= 2 {
				out = append(out, ConfigMatch{
					Type:  t,
					Pkg:   pkg,
					Hits:  hits,
					New:   newKeys,
					Score: len(hits),
				})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		if out[i].Type.Exported != out[j].Type.Exported {
			return out[i].Type.Exported
		}
		return len(out[i].Type.Fields) < len(out[j].Type.Fields)
	})
	return out
}

// MechanismSite is the package where the issue's how-word lives:
// "URLs" → the shared HTTP client every source uses.
type MechanismSite struct {
	Pkg     *Package
	Keyword string
	KeyFunc *Func // most-called function in the package
	Score   int
	Why     string
}

// FindMechanism locates the package behind a how-word, preferring the
// shared one: the package many others import is the choke point where
// one change (retry around the fetch) covers every caller.
func (p *Project) FindMechanism(keyword string) *MechanismSite {
	kw := strings.ToLower(keyword)
	// importerCount: how many project packages import each package.
	// The most-imported match is the shared choke point: one change
	// there reaches every caller (retry around the one HTTP client).
	importerCount := map[*Package]int{}
	for _, pkg := range p.Packages {
		for _, imp := range pkg.Internal {
			if target := p.importTarget(imp); target != nil && target != pkg {
				importerCount[target]++
			}
		}
	}
	bestPkg, bestScore := (*Package)(nil), -1
	for _, pkg := range p.Packages {
		base := 0
		name := strings.ToLower(pkg.Name)
		if strings.Contains(name, kw) || strings.Contains(name, singular(kw)) {
			base += 3
		}
		if strings.Contains(strings.ToLower(pkg.Doc), kw) {
			base++
		}
		for _, f := range pkg.Files {
			if strings.Contains(strings.ToLower(f), kw) {
				base++
				break
			}
		}
		if base == 0 {
			continue
		}
		// Name/doc/file matches decide; shared-infra only breaks ties.
		// (Otherwise a package imported by 60 others wins on plumbing.)
		score := base*100 + importerCount[pkg]
		if score > bestScore {
			bestScore, bestPkg = score, pkg
		}
	}
	if bestPkg == nil {
		return nil
	}
	var key *Func
	most := -1
	for _, fn := range bestPkg.Funcs {
		if len(fn.Callers) > most {
			most, key = len(fn.Callers), fn
		}
	}
	return &MechanismSite{Pkg: bestPkg, Keyword: kw, KeyFunc: key, Score: bestScore}
}

// GuideIssue renders the full where-to-edit answer for an issue:
// the config struct to extend, the mechanism to wrap, and the call
// chain the change sits in. One clear path, not a hit list.
func (p *Project) GuideIssue(c IssueConcepts) string {
	var b strings.Builder
	verb := "TO ADD"
	if c.Action == "fix" {
		verb = "TO FIX"
	}
	fmt.Fprintf(&b, "%s %q:\n\n", verb, c.Title)

	structs := p.FindConfigStruct(c.YamlKeys)
	// Try every how-word; keep the best-scoring site. "http" beating
	// "fetch" is how the shared client wins over a local helper.
	var site *MechanismSite
	for _, m := range c.Mechanisms {
		if s := p.FindMechanism(m); s != nil && (site == nil || s.Score > site.Score) {
			site = s
		}
	}

	if len(structs) > 0 {
		m := structs[0]
		fmt.Fprintf(&b, "CONFIG — put the new option here:\n")
		fmt.Fprintf(&b, "  %s:%d\n", m.Type.File, m.Type.Line)
		fmt.Fprintf(&b, "  type %s struct — fields %s match the issue's example\n",
			m.Type.Name, strings.Join(quoteAll(m.Hits), ", "))
		if d := oneLineDoc(m.Type.Doc); d != "" {
			fmt.Fprintf(&b, "  %s\n", d)
		}
		if len(m.New) > 0 {
			if add, nested := newKeysAtLevel(c, m); len(add) > 0 {
				kw := "key"
				if len(add) > 1 {
					kw = "keys"
				}
				fmt.Fprintf(&b, "  Add the issue's new %s here: %s\n", kw, strings.Join(quoteAll(add), ", "))
				for _, a := range add {
					if kids := nested[a]; len(kids) > 0 {
						fmt.Fprintf(&b, "    with %s underneath\n", strings.Join(quoteAll(kids), ", "))
					}
				}
			}
		}
		fmt.Fprintf(&b, "  Why here: every %s flows through this struct, so one new field reaches them all.\n\n",
			m.Pkg.Name)
	}

	if site != nil {
		action := "wrap"
		if c.Action == "fix" {
			action = "fix"
		}
		fmt.Fprintf(&b, "BEHAVIOR — %s the work here:\n", action)
		fmt.Fprintf(&b, "  package %s (%s)\n", site.Pkg.Name, site.Pkg.Dir)
		if site.KeyFunc != nil {
			fmt.Fprintf(&b, "  %s:%d — %s\n", site.KeyFunc.File, site.KeyFunc.Line,
				strings.TrimPrefix(site.KeyFunc.Sig, "func "))
			if d := oneLineDoc(site.KeyFunc.Doc); d != "" {
				fmt.Fprintf(&b, "  %s\n", d)
			}
		}
		fmt.Fprintf(&b, "  Why here: the issue is %q; this is the shared %s path — %s it once, every caller gets it.\n\n",
			c.Title, site.Keyword, action)
	}

	// You-are-here: walk from the mechanism (or config) toward main.
	var anchor *Func
	if site != nil && site.KeyFunc != nil {
		anchor = site.KeyFunc
	} else if len(structs) > 0 {
		anchor = keyFuncOf(p, structs[0].Pkg)
	}
	if anchor != nil {
		chain := callerChain(p, anchor, 5)
		fmt.Fprintf(&b, "YOU ARE HERE:\n  ")
		names := make([]string, len(chain))
		for i, fn := range chain {
			names[i] = fn.Name
		}
		fmt.Fprintf(&b, "%s → yours here\n", strings.Join(names, " → "))
	}

	if len(structs) == 0 && site == nil {
		return p.GuideChange(c.Title)
	}
	return b.String()
}

// keyFuncOf returns the most-called function in a package.
func keyFuncOf(p *Project, pkg *Package) *Func {
	var key *Func
	most := -1
	for _, fn := range pkg.Funcs {
		if len(fn.Callers) > most {
			most, key = len(fn.Callers), fn
		}
	}
	return key
}

func quoteAll(ss []string) []string {
	out := make([]string, len(ss))
	for i, s := range ss {
		out[i] = "`" + s + "`"
	}
	return out
}

// newKeysAtLevel splits the issue's unmatched keys into struct-level
// additions and the sub-keys nested under each addition. The struct's
// level is the indent of the matched keys (kind:/spec:); keys above it
// (sources:) are wrappers, keys below it belong to their parent — only
// children of a NEW key (limit: under retry:) are shown, children of an
// existing field (file: under spec:) belong to another struct.
func newKeysAtLevel(c IssueConcepts, m ConfigMatch) (add []string, nested map[string][]string) {
	if len(c.KeyIndents) == 0 {
		return topNew(m.New, 5), nil
	}
	counts := map[int]int{}
	for _, h := range m.Hits {
		counts[c.KeyIndents[h]]++
	}
	level, best := 0, -1
	for ind, n := range counts {
		if n > best {
			level, best = ind, n
		}
	}
	pos := map[string]int{}
	for i, k := range c.YamlKeys {
		pos[k] = i
	}
	atLevel := map[string]bool{}
	for _, k := range m.New {
		if ind, ok := c.KeyIndents[k]; ok && ind == level {
			atLevel[k] = true
		}
	}
	parentOf := func(k string) string {
		ind := c.KeyIndents[k]
		// Nearest preceding key with a smaller indent: the YAML parent.
		for i := pos[k] - 1; i >= 0; i-- {
			if o := c.YamlKeys[i]; c.KeyIndents[o] < ind {
				return o
			}
		}
		return ""
	}
	nested = map[string][]string{}
	for _, k := range m.New {
		ind, ok := c.KeyIndents[k]
		if !ok {
			continue
		}
		switch {
		case ind == level:
			add = append(add, k)
		case ind > level:
			if p := parentOf(k); atLevel[p] {
				nested[p] = append(nested[p], k)
			}
		}
	}
	return add, nested
}

// topNew picks the proposed-new keys worth naming: short config-ish
// words first (retry, limit), not values or prose.
func topNew(keys []string, n int) []string {
	var good, rest []string
	for _, k := range keys {
		if len(k) <= 12 && !strings.ContainsAny(k, "_-") {
			good = append(good, k)
		} else {
			rest = append(rest, k)
		}
	}
	out := append(good, rest...)
	if len(out) > n {
		out = out[:n]
	}
	return out
}

func oneLineDoc(doc string) string {
	doc = strings.TrimSpace(doc)
	if i := strings.Index(doc, "\n"); i >= 0 {
		doc = doc[:i]
	}
	doc = strings.TrimSpace(doc)
	if len(doc) > 120 {
		doc = doc[:117] + "..."
	}
	return doc
}
