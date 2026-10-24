package chat

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// resources.go — build-goal resources: when the user states a
// learning intent to build something ("i want to build a
// webserver", "how do i build a REST API", "resources for chat"),
// answer with a resource pack instead of routing the line to a
// brain that has to guess. The pack comes from John's own Advocates
// doc — his compilation of Go learning links — snapshotted into
// data/advocates.json by scripts/import_advocates (the doc stays
// the source of truth; the chat reads the snapshot, offline).
//
// Claimed deterministically with the other sentence parsers, before
// the intent classifier — but ONLY the learning shapes. Imperative
// build commands for THIS project ("build api for /search",
// "build me a webserver") are not shapes here at all: they belong
// to the composer and the edits family, and this parser must never
// steal them. A goal the doc knows nothing about is claimed only
// when the user explicitly asked for resources, or the goal's words
// at least touch the doc's vocabulary — a statement like "i want
// to build my own PC" keeps falling through to the router.

// --- Advocates snapshot loading (mirrors defaultIntentModel) ---

type advocatesEntry struct {
	Label string `json:"label"`
	URL   string `json:"url"`
}

type advocatesTopic struct {
	Name    string           `json:"name"`
	Entries []advocatesEntry `json:"entries"`
}

type advocatesDoc struct {
	Source string           `json:"source"`
	Topics []advocatesTopic `json:"topics"`
}

var (
	advocatesOnce  = &sync.Once{} // a pointer so tests can swap in a fresh one
	advocatesCache *advocatesDoc
)

// advocates lazily loads data/advocates.json from the chat's
// project root (or the working directory). Absent file = nil, and
// the feature degrades to its built-in seed list.
func advocates() *advocatesDoc {
	advocatesOnce.Do(func() {
		for _, base := range []string{unifiedProjectRoot, startFolder()} {
			if base == "" {
				continue
			}
			blob, err := os.ReadFile(filepath.Join(base, "data", "advocates.json"))
			if err != nil {
				continue
			}
			var d advocatesDoc
			if json.Unmarshal(blob, &d) == nil && len(d.Topics) > 0 {
				advocatesCache = &d
				return
			}
		}
	})
	return advocatesCache
}

// --- goal parsing ---

var buildGoalRes = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^\s*(?:i want to|i wanna|i'?d like to|i would like to) build (.+)$`),
	regexp.MustCompile(`(?i)^\s*how (?:do|can) i build (.+)$`),
	regexp.MustCompile(`(?i)^\s*(?:i'?m|i am) building (.+)$`),
	regexp.MustCompile(`(?i)^\s*help me build (.+)$`),
	regexp.MustCompile(`(?i)^\s*building (.+)$`),
}

var resourcesRes = []*regexp.Regexp{
	regexp.MustCompile(`(?i)^\s*resources for (.+)$`),
	regexp.MustCompile(`(?i)^\s*resources (.+)$`),
}

// parseBuildGoal extracts the thing the user wants to build.
// explicit reports the "resources …" shapes, where the user asked
// for resources outright (claimed even on a vocabulary miss).
func parseBuildGoal(line string) (goal string, explicit bool, ok bool) {
	// A line naming a repo or a path is never a build goal — repo
	// questions keep their deterministic route.
	if strings.Contains(line, "://") || strings.Contains(line, "github.com/") {
		return "", false, false
	}
	match := func(res []*regexp.Regexp) (string, bool) {
		for _, re := range res {
			if m := re.FindStringSubmatch(line); m != nil {
				return m[1], true
			}
		}
		return "", false
	}
	raw, explicit := match(resourcesRes)
	if !explicit {
		var found bool
		raw, found = match(buildGoalRes)
		if !found {
			return "", false, false
		}
	}
	goal = cleanGoal(raw)
	if goal == "" {
		return "", false, false
	}
	// A single dotted token is a symbol reference, not a goal —
	// symbolpick owns those. A slash means a path slipped in.
	if !strings.Contains(goal, " ") && strings.Contains(goal, ".") {
		return "", false, false
	}
	if strings.Contains(goal, "/") {
		return "", false, false
	}
	if len(strings.Fields(goal)) > 6 {
		return "", false, false // a sentence, not a goal
	}
	// "Build a function that …" is a code-writing ask — the
	// gocode brain's line, not a learning goal.
	if termSet(goal)["function"] {
		return "", false, false
	}
	return goal, explicit, true
}

// cleanGoal normalizes the captured phrase: trailing punctuation
// and filler off, the leading article off.
func cleanGoal(raw string) string {
	g := strings.TrimSpace(strings.Trim(raw, ".?!"))
	for _, suf := range []string{" in go", " with go", " using go", " in golang", " with golang"} {
		g = strings.TrimSuffix(g, suf)
	}
	for _, art := range []string{"a ", "an ", "the "} {
		if strings.HasPrefix(strings.ToLower(g), art) {
			g = g[len(art):]
			break
		}
	}
	return strings.TrimSpace(g)
}

// --- vocabulary ---

var goalStopwords = map[string]bool{
	"a": true, "an": true, "the": true, "and": true, "or": true,
	"of": true, "for": true, "to": true, "in": true, "on": true,
	"with": true, "from": true, "my": true, "own": true, "i": true, "me": true,
	"it": true, "is": true, "go": true, "golang": true, "code": true,
	"how": true, "do": true, "what": true, "your": true, "you": true,
	"this": true, "that": true, "using": true, "use": true,
}

// goalSynonyms expands a goal term into the doc's own vocabulary:
// his topics say "http package" and "scaffolding restful api", not
// "webserver".
var goalSynonyms = map[string][]string{
	"webserver":  {"web", "server", "http", "rest", "scaffolding"},
	"server":     {"http", "rest", "scaffolding"},
	"web":        {"http", "frontend"},
	"http":       {"server", "rest"},
	"rest":       {"api", "scaffolding"},
	"api":        {"rest", "scaffolding", "database", "graphql"},
	"cli":        {"cobra", "command", "flag"},
	"command":    {"cobra", "cli", "flag"},
	"terminal":   {"cli", "cobra"},
	"chat":       {"websocket", "frontend", "svelte"},
	"websocket":  {"chat", "frontend"},
	"game":       {"wasm", "frontend", "svelte"},
	"website":    {"frontend", "http", "template", "html"},
	"frontend":   {"svelte", "vue", "flutter", "html"},
	"database":   {"sql", "post", "get", "graphql"},
	"sql":        {"database"},
	"scraper":    {"http", "get", "parsing", "html"},
	"scrape":     {"http", "get", "parsing"},
	"test":       {"testing", "mocking"},
	"testing":    {"test", "mocking"},
	"mock":       {"mocking", "testing"},
	"docker":     {"container"},
	"container":  {"docker"},
	"file":       {"reading", "writing"},
	"jwt":        {"security", "crypto"},
	"auth":       {"jwt", "security", "crypto"},
	"security":   {"jwt", "crypto"},
	"template":   {"html"},
	"wasm":       {"web"},
	"blog":       {"frontend", "template"},
	"worker":     {"goroutine", "channel", "concurrency"},
	"queue":      {"worker", "channel"},
	"concurrent": {"goroutine", "channel", "concurrency"},
}

// normTerm light-stems one token (plural -s, the one -ful that
// matters) so goal words and doc words meet in the middle.
func normTerm(tok string) string {
	tok = strings.ToLower(tok)
	if tok == "restful" {
		return "rest"
	}
	if len(tok) > 3 && strings.HasSuffix(tok, "s") && !strings.HasSuffix(tok, "ss") {
		tok = tok[:len(tok)-1]
	}
	return tok
}

// termSet tokenizes text (a goal, a topic name, a label, a URL)
// into the normalized vocabulary.
func termSet(s string) map[string]bool {
	out := map[string]bool{}
	for _, f := range strings.FieldsFunc(s, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9')
	}) {
		t := normTerm(f)
		if len(t) < 2 || goalStopwords[t] {
			continue
		}
		out[t] = true
	}
	return out
}

// expandGoalTerms adds ONE level of synonyms to the goal's
// terms. The original terms are snapshotted first: ranging over
// the map while inserting would visit some newly-added terms and
// expand them too, making the set (and every score downstream)
// depend on Go's map iteration order.
func expandGoalTerms(goal string) map[string]bool {
	out := termSet(goal)
	orig := make([]string, 0, len(out))
	for t := range out {
		orig = append(orig, t)
	}
	for _, t := range orig {
		for _, syn := range goalSynonyms[t] {
			out[syn] = true
		}
	}
	return out
}

// urlFurniture is stripped from URL terms: words that appear in
// addresses without meaning anything about the content — most
// importantly "web", which every web.archive.org URL carries and
// which otherwise matches any goal whose terms include it.
var urlFurniture = map[string]bool{
	"web": true, "www": true, "com": true, "org": true,
	"net": true, "html": true, "htm": true, "php": true,
	"http": true, "https": true,
}

func urlTerms(raw string) map[string]bool {
	u, err := url.Parse(raw)
	if err != nil {
		return termSet(raw)
	}
	out := termSet(u.Host + " " + u.Path)
	for f := range urlFurniture {
		delete(out, f)
	}
	return out
}

func intersect(a, b map[string]bool) int {
	n := 0
	for t := range a {
		if b[t] {
			n++
		}
	}
	return n
}

// --- matching ---

type topicMatch struct {
	topic   advocatesTopic
	docIdx  int
	score   int
	nameHit bool
	entries []advocatesEntry // matching entries, doc order
}

// matchTopics scores every topic against the goal's expanded
// terms: a topic-name hit weighs most (his topic names are the
// index), entry labels and URLs corroborate.
func matchTopics(doc *advocatesDoc, goal string) []topicMatch {
	e := expandGoalTerms(goal)
	var out []topicMatch
	for i, t := range doc.Topics {
		tm := topicMatch{topic: t, docIdx: i}
		nameHits := intersect(e, termSet(t.Name))
		tm.nameHit = nameHits > 0
		tm.score = 3 * nameHits
		for _, en := range t.Entries {
			terms := termSet(en.Label)
			for k := range urlTerms(en.URL) {
				terms[k] = true
			}
			if intersect(e, terms) > 0 {
				tm.entries = append(tm.entries, en)
				// Entry corroboration is capped: without a cap
				// the biggest topic (General, 80+ links) outranks
				// the topic the goal actually named on volume
				// alone.
				if len(tm.entries) <= 3 {
					tm.score++
				}
			}
		}
		// A name-matched topic with no entry-level hit still has
		// something to show: its first entries are the topic.
		if tm.nameHit && len(tm.entries) == 0 {
			n := 2
			if len(t.Entries) < n {
				n = len(t.Entries)
			}
			tm.entries = append(tm.entries, t.Entries[:n]...)
		}
		if tm.score > 0 {
			out = append(out, tm)
		}
	}
	return out
}

func foundationTopic(name string) bool {
	n := strings.ToLower(name)
	return strings.Contains(n, "beginner") || n == "official resources" || n == "syntax"
}

// rankMatches orders the pack: topics the goal named go first
// (foundations among them first — learn before you build), then
// label-only matches; score, then his doc order, break ties.
func rankMatches(ms []topicMatch) {
	sort.SliceStable(ms, func(i, j int) bool {
		a, b := ms[i], ms[j]
		if a.nameHit != b.nameHit {
			return a.nameHit
		}
		if a.nameHit && foundationTopic(a.topic.Name) != foundationTopic(b.topic.Name) {
			return foundationTopic(a.topic.Name)
		}
		if a.score != b.score {
			return a.score > b.score
		}
		return a.docIdx < b.docIdx
	})
}

const packEntryBudget = 6
const packTopicBudget = 3

// --- seed fallback (no advocates.json in this checkout) ---

// resourceSeed is the short built-in list: concepts the code
// already knows, with their gobyexample pages — the same map the
// learn-path leans on, extended with the canonical build topics.
var resourceSeed = []struct {
	terms []string
	label string
	url   string
}{
	{[]string{"http", "server", "webserver", "web"}, "HTTP Servers (Go by Example)", "https://gobyexample.com/http-servers"},
	{[]string{"http", "client"}, "HTTP Clients (Go by Example)", "https://gobyexample.com/http-clients"},
	{[]string{"api", "rest", "json"}, "JSON (Go by Example)", "https://gobyexample.com/json"},
	{[]string{"cli", "command", "flag"}, "Command-Line Flags (Go by Example)", "https://gobyexample.com/command-line-flags"},
	{[]string{"cli", "command", "subcommand"}, "Command-Line Subcommands (Go by Example)", "https://gobyexample.com/command-line-subcommands"},
	{[]string{"template", "html"}, "Text Templates (Go by Example)", "https://gobyexample.com/text-templates"},
	{[]string{"test", "testing"}, "Testing and Benchmarking (Go by Example)", "https://gobyexample.com/testing-and-benchmarking"},
	{[]string{"goroutine", "concurrent", "concurrency", "worker"}, "Goroutines (Go by Example)", "https://gobyexample.com/goroutines"},
	{[]string{"channel", "concurrent", "concurrency"}, "Channels (Go by Example)", "https://gobyexample.com/channels"},
	{[]string{"worker", "pool", "concurrent"}, "Worker Pools (Go by Example)", "https://gobyexample.com/worker-pools"},
	{[]string{"file", "read"}, "Reading Files (Go by Example)", "https://gobyexample.com/reading-files"},
	{[]string{"file", "write"}, "Writing Files (Go by Example)", "https://gobyexample.com/writing-files"},
	{[]string{"interface"}, "Interfaces (Go by Example)", "https://gobyexample.com/interfaces"},
	{[]string{"error"}, "Errors (Go by Example)", "https://gobyexample.com/errors"},
	{[]string{"mutex", "lock", "shared"}, "Mutexes (Go by Example)", "https://gobyexample.com/mutexes"},
	{[]string{"context", "timeout", "cancel"}, "Context (Go by Example)", "https://gobyexample.com/context"},
}

func matchSeed(goal string) []advocatesEntry {
	e := expandGoalTerms(goal)
	type scored struct {
		en    advocatesEntry
		score int
		idx   int
	}
	var hits []scored
	for i, s := range resourceSeed {
		score := 0
		for _, t := range s.terms {
			if e[normTerm(t)] {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, scored{advocatesEntry{Label: s.label, URL: s.url}, score, i})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].idx < hits[j].idx
	})
	var out []advocatesEntry
	for _, h := range hits {
		out = append(out, h.en)
	}
	return out
}

// --- project example ---

var httpServerCalls = map[string]bool{
	"http.ListenAndServe":    true,
	"http.ListenAndServeTLS": true,
	"http.HandleFunc":        true,
	"http.Handle":            true,
	"http.NewServeMux":       true,
	"http.Server":            true,
}

// findProjectExample looks for the thing itself in the analyzed
// project: for a server goal, a function that actually starts or
// registers an HTTP server; generically, a function whose name or
// file carries the goal's words. Never invents: no hit, no line.
func findProjectExample(p *analyze.Project, goal string) *analyze.Func {
	if p == nil {
		return nil
	}
	e := expandGoalTerms(goal)
	serverGoal := e["http"] || e["server"] || e["websocket"]
	if serverGoal {
		var best *analyze.Func
		for _, fn := range p.AllFuncs() {
			if fn.IsTest {
				continue
			}
			for _, c := range fn.ExtCalls {
				if httpServerCalls[c] {
					if best == nil || fn.Score > best.Score {
						best = fn
					}
					break
				}
			}
		}
		if best != nil {
			return best
		}
	}
	var best *analyze.Func
	bestScore := 0
	for _, fn := range p.AllFuncs() {
		if fn.IsTest {
			continue
		}
		score := 0
		if intersect(e, termSet(fn.Name)) > 0 {
			score += 3
		}
		if intersect(e, termSet(fn.File)) > 0 {
			score += 2
		}
		if score > bestScore || (score == bestScore && score > 0 && best != nil && fn.Score > best.Score) {
			best, bestScore = fn, score
		}
	}
	if bestScore == 0 {
		return nil
	}
	return best
}

// --- the answer ---

// tryBuildResources claims a build-goal statement and prints its
// resource pack. See the file header for the ownership rules.
func tryBuildResources(line string, profile *UserProfile) bool {
	goal, explicit, ok := parseBuildGoal(line)
	if !ok {
		return false
	}
	doc := advocates()
	var matches []topicMatch
	var seed []advocatesEntry
	qualified := false
	bestScore := 0
	if doc != nil {
		matches = matchTopics(doc, goal)
		for _, m := range matches {
			if m.score >= 3 {
				qualified = true
			}
			if m.score > bestScore {
				bestScore = m.score
			}
		}
	} else {
		seed = matchSeed(goal)
		qualified = len(seed) > 0
	}
	if !qualified && !explicit {
		// Not our line unless the goal at least brushes the
		// vocabulary — or the project on screen already contains
		// the thing (checked without forcing an analysis).
		if bestScore < 2 && findProjectExample(lastAnalyzeProject, goal) == nil {
			return false
		}
	}

	// The example search may analyze the folder the chat is in —
	// the same on-ramp the other Q&A paths use — but only once a
	// pack (or a near miss) is actually being answered.
	if lastAnalyzeProject == nil {
		ensureProject()
	}
	example := findProjectExample(lastAnalyzeProject, goal)

	var b strings.Builder
	switch {
	case doc != nil && qualified:
		fmt.Fprintf(&b, "Resources for building %s — from your Advocates doc:\n", goal)
	case doc != nil:
		fmt.Fprintf(&b, "I don't have a topic for %q in your Advocates doc.\n", goal)
	case qualified:
		fmt.Fprintf(&b, "Resources for building %s:\n", goal)
	default:
		b.WriteString("I don't have resources for that in my built-in list — the Advocates snapshot (data/advocates.json) isn't in this checkout.\n")
	}
	if example != nil && lastAnalyzeProject != nil {
		fmt.Fprintf(&b, "\nThis project already has one: %s in %s:%d — say \"walk me through %s\" to see how it works.\n",
			example.Display(), lastAnalyzeProject.LinkPath(example.File), example.Line, example.Name)
	}

	switch {
	case doc != nil && qualified:
		rankMatches(matches)
		var shown []topicMatch
		for _, m := range matches {
			if m.score >= 3 {
				shown = append(shown, m)
			}
		}
		b.WriteString("\n")
		budget := packEntryBudget
		remCount := 0
		var remTopics []string
		for ti, m := range shown {
			if ti >= packTopicBudget {
				if len(m.entries) > 0 {
					remCount += len(m.entries)
					remTopics = append(remTopics, fmt.Sprintf("%q", m.topic.Name))
				}
				continue
			}
			take := len(m.entries)
			if take > budget {
				take = budget
			}
			for _, en := range m.entries[:take] {
				fmt.Fprintf(&b, "• %s — %s\n", en.Label, en.URL)
			}
			budget -= take
			if left := len(m.entries) - take; left > 0 {
				remCount += left
				remTopics = append(remTopics, fmt.Sprintf("%q", m.topic.Name))
			}
		}
		if remCount > 0 {
			// Name at most two topics; a tail that lists nine
			// topics is a second wall of text, not a pointer.
			named := remTopics
			extra := 0
			if len(named) > 2 {
				extra = len(named) - 2
				named = named[:2]
			}
			tail := strings.Join(named, ", ")
			if extra == 1 {
				tail += " and 1 other topic"
			} else if extra > 1 {
				tail += fmt.Sprintf(" and %d other topics", extra)
			}
			fmt.Fprintf(&b, "\n+%d more in %s.\n", remCount, tail)
		}
	case doc != nil:
		// Near miss: name the closest topic as a closest, never as
		// a match, and point at the concept walkers.
		var best *topicMatch
		for i := range matches {
			if best == nil || matches[i].score > best.score {
				best = &matches[i]
			}
		}
		if best != nil && bestScore >= 2 {
			fmt.Fprintf(&b, "Closest on the shelf: %q — say \"resources for %s\" and I'll pull it up.\n",
				best.topic.Name, strings.ToLower(best.topic.Name))
		}
		b.WriteString("For the concepts themselves: /learn-path walks a project's ideas in order, each linked to gobyexample, and /glossary defines the Go terms.\n")
	case qualified:
		b.WriteString("\n")
		for i, en := range seed {
			if i >= packEntryBudget {
				fmt.Fprintf(&b, "\n+%d more in the built-in list.\n", len(seed)-packEntryBudget)
				break
			}
			fmt.Fprintf(&b, "• %s — %s\n", en.Label, en.URL)
		}
	default:
		b.WriteString("For the concepts themselves: /learn-path walks a project's ideas in order, each linked to gobyexample, and /glossary defines the Go terms.\n")
	}

	// Composer tie-in: only where a real composer capability
	// exists — the API composer. No vapor offers for other goals.
	rawTerms := termSet(goal)
	if rawTerms["api"] || rawTerms["rest"] {
		b.WriteString("\nOr skip the reading — say \"build api for /users\" and the composer assembles a working API from pieces, here in this project.\n")
	}
	printPaced(b.String())
	recordBrain("analyst", true, "build resources: "+goal)
	return true
}
