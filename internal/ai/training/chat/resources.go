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

// --- curated reference sources (built-in) ---

// referenceEntry is one curated pointer into a canonical Go
// text — Effective Go, section by section, and the Go Wiki's
// Switch page — keyed to the topics a builder asks about.
// Unlike the Advocates snapshot these are built in (they are
// the language's own documentation, not John's curation), and
// unlike the seed list they are SECTION links. Every anchor is
// the section's real id, taken from the page's own source or a
// verbatim link to it — never guessed: the resources build's
// zero-invented-URLs rule applies to anchors too, so a topic
// whose section id could not be verified that way links the
// page itself and its label names the section to read.
type referenceEntry struct {
	source string   // the header the entry prints under
	terms  []string // topic phrases; the first is the primary
	label  string
	url    string
}

const effectiveGoURL = "https://go.dev/doc/effective_go"

// howToCodeURL is the DigitalOcean tutorial series "How To
// Code in Go" — the book John handed over as a PDF (Bates,
// LaNou & Raymond; the text of these tutorials). Entries
// below link the series page and name the chapter in the
// label: the series page is the verified address (checked at
// build time), and per-chapter URLs were not individually
// verified, so — by the same rule that keeps unverified
// Effective Go anchors page-level — no chapter URL is
// invented here.
const howToCodeURL = "https://www.digitalocean.com/community/tutorial-series/how-to-code-in-go"

// astaxie32URL is §3.2 of Astaxie's "Build Web Application
// with Golang" — the address John gave, read at build time.
const astaxie32URL = "https://astaxie.gitbooks.io/build-web-application-with-golang/content/en/03.2.html"

// practicalGo1URL is chapter 1 of Practical Go Lessons —
// the address John gave, read at build time.
const practicalGo1URL = "https://www.practical-go-lessons.com/chap-1-programming-a-computer"

// yourbasicURL is the YourBasic Go go-to guide's index —
// the one verified address for the site (read at build
// time for the basics sweep, 2026-11-01). Per-topic
// addresses were not individually verified, so — by the
// same rule that keeps unverified anchors page-level —
// every YourBasic entry links the index and its label
// names the article to read.
const yourbasicURL = "https://yourbasic.org/golang/"

// callicoderTypesURL is Callicoder's "Basic Types,
// Operators, and Type Conversion" article — verified at
// build time for the basics sweep (2026-11-01), the same
// way the functions and constants articles were read.
const callicoderTypesURL = "https://www.callicoder.com/golang-basic-types-operators-type-conversion/"

// The security sweep's sources (2026-11-01), each address
// read and verified at build time: the OWASP Cheat Sheet
// Series for the JWT / IDOR / password-storage / HTTP
// headers guidance, MDN for CORS, and pkg.go.dev's net/http
// page for the two stdlib anchors (its index lists both
// symbols — CrossOriginProtection and
// Server.ListenAndServeTLS — so the anchors name real
// sections of a verified page).
const (
	owaspJWTURL       = "https://cheatsheetseries.owasp.org/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html"
	owaspIDORURL      = "https://cheatsheetseries.owasp.org/cheatsheets/Insecure_Direct_Object_Reference_Prevention_Cheat_Sheet.html"
	owaspPasswordsURL = "https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html"
	owaspHeadersURL   = "https://cheatsheetseries.owasp.org/cheatsheets/HTTP_Headers_Cheat_Sheet.html"
	mdnCORSURL        = "https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS"
	pkgHTTPURL        = "https://pkg.go.dev/net/http"
	crossOriginURL    = pkgHTTPURL + "#CrossOriginProtection"
	listenServeTLSURL = pkgHTTPURL + "#Server.ListenAndServeTLS"
)

// sqlInterfaceURL is the Go Wiki's SQLInterface page — the
// database/sql usage-patterns page, read in full at build
// time (the database build, 2026-11-01). Per-section
// anchors were not individually verified, so — by the same
// rule that keeps unverified anchors page-level — every
// SQLInterface entry links the page and its label names the
// section to read.
const sqlInterfaceURL = "https://go.dev/wiki/SQLInterface"

// The cheatsheet sweep's sources (2026-11-01), each address
// read and verified at build time: Devhints' one-page Go
// syntax sheet, GoInterview.dev's interactive cheatsheet,
// and programming-idioms.org's Go cheatsheet — a few
// hundred little tasks, each with a community Go answer.
// The three sheets themselves seat here as reading; what
// they teach that the book can assemble became rows, and
// the full per-topic mapping (rows / duplicates /
// exclusions with their walls) is in the sweep's commit
// message and its test file.
const (
	devhintsURL          = "https://devhints.io/go"
	gointerviewURL       = "https://app.gointerview.dev/cheatsheet"
	programmingIdiomsURL = "https://programming-idioms.org/cheatsheet/Go"
)

var referenceSeed = []referenceEntry{
	{"Effective Go", []string{"formatting", "gofmt", "format"}, "Formatting", effectiveGoURL + "#formatting"},
	{"Effective Go", []string{"comments", "commentary", "doc comments"}, "Commentary", effectiveGoURL + "#commentary"},
	{"Effective Go", []string{"names", "naming"}, "Names", effectiveGoURL + "#names"},
	{"Effective Go", []string{"package names", "package naming"}, "Package names", effectiveGoURL + "#package-names"},
	{"Effective Go", []string{"getters", "setters", "getter", "setter"}, "Getters", effectiveGoURL + "#Getters"},
	{"Effective Go", []string{"interface names", "naming interfaces"}, "Interface names", effectiveGoURL + "#interface-names"},
	{"Effective Go", []string{"mixedcaps", "camel case", "camelcase"}, "MixedCaps", effectiveGoURL + "#mixed-caps"},
	{"Effective Go", []string{"semicolons", "semicolon"}, "Semicolons", effectiveGoURL + "#semicolons"},
	{"Effective Go", []string{"if statements", "conditionals"}, "If", effectiveGoURL + "#if"},
	{"Effective Go", []string{"for loops", "loops", "looping"}, "For", effectiveGoURL + "#for"},
	{"Effective Go", []string{"switch statements", "switch"}, "Switch", effectiveGoURL + "#switch"},
	{"Effective Go", []string{"type switch", "type switches"}, "Type switch", effectiveGoURL + "#type_switch"},
	{"Effective Go", []string{"multiple return values", "multiple returns", "returning two values"}, "Multiple return values", effectiveGoURL + "#multiple-returns"},
	{"Effective Go", []string{"named result parameters", "named return values", "named results"}, "Named result parameters", effectiveGoURL + "#named-results"},
	{"Effective Go", []string{"defer", "deferred calls"}, "Defer", effectiveGoURL + "#defer"},
	{"Effective Go", []string{"allocation", "allocating", "new"}, "Allocation with new", effectiveGoURL + "#allocation_new"},
	{"Effective Go", []string{"blank identifier", "underscore"}, "The blank identifier", effectiveGoURL + "#blank"},
	{"Effective Go", []string{"concurrency", "goroutines", "channels", "worker", "concurrent"}, "Concurrency", effectiveGoURL + "#concurrency"},
	{"Effective Go", []string{"goroutines", "goroutine"}, "Goroutines", effectiveGoURL + "#goroutines"},
	{"Effective Go", []string{"channels", "channel"}, "Channels", effectiveGoURL + "#channels"},
	{"Effective Go", []string{"web server", "http server", "webserver"}, "A web server", effectiveGoURL},
	{"Effective Go", []string{"arrays", "array"}, "Arrays", effectiveGoURL},
	{"Effective Go", []string{"slices", "slice"}, "Slices", effectiveGoURL},
	{"Effective Go", []string{"maps", "map"}, "Maps", effectiveGoURL},
	{"Effective Go", []string{"printing", "printf", "print"}, "Printing", effectiveGoURL},
	{"Effective Go", []string{"append", "appending"}, "append", effectiveGoURL},
	{"Effective Go", []string{"initialization", "constants", "variables", "init function"}, "Initialization: constants, variables, the init function", effectiveGoURL},
	{"Effective Go", []string{"methods", "method", "receivers", "pointer receivers", "value receivers"}, "Methods: pointers vs. values", effectiveGoURL},
	{"Effective Go", []string{"interfaces", "interface", "type assertions"}, "Interfaces and other types", effectiveGoURL},
	{"Effective Go", []string{"embedding", "embed", "embedded types"}, "Embedding", effectiveGoURL},
	{"Effective Go", []string{"errors", "error handling"}, "Errors", effectiveGoURL},
	{"Effective Go", []string{"panic", "recover"}, "panic and recover", effectiveGoURL},
	{"the Go Wiki", []string{"switch", "switch statements"}, "Switch", "https://go.dev/wiki/Switch"},
	// The three-sources build (2026-10-31): the DigitalOcean
	// book's chapters the concept book doesn't already serve
	// as rows — pointers to reading, seated beside Effective
	// Go. Every entry links the verified series page; the
	// label names the chapter.
	{"How To Code in Go", []string{"struct tags", "struct tag", "tags"}, "How To Use Struct Tags in Go (Chapter 39)", howToCodeURL},
	{"How To Code in Go", []string{"build tags", "build constraints"}, "Customizing Go Binaries with Build Tags (Chapter 34)", howToCodeURL},
	{"How To Code in Go", []string{"ldflags", "version information", "version info"}, "Using ldflags to Set Version Information for Go Applications (Chapter 42)", howToCodeURL},
	{"How To Code in Go", []string{"cross compiling", "different operating systems", "building for other platforms", "goos"}, "Building Go Applications for Different Operating Systems and Architectures (Chapter 41)", howToCodeURL},
	{"How To Code in Go", []string{"writing packages", "write packages", "importing packages", "packages", "package visibility", "exported"}, "How To Write Packages in Go (Chapters 23–25)", howToCodeURL},
	{"How To Code in Go", []string{"flag package", "command line flags", "flags"}, "How To Use the Flag Package in Go (Chapter 43)", howToCodeURL},
	{"How To Code in Go", []string{"custom errors", "creating errors", "error types"}, "Creating Custom Errors in Go (Chapter 21)", howToCodeURL},
	{"How To Code in Go", []string{"variadic functions", "variadic", "ellipsis"}, "How To Use Variadic Functions in Go (Chapter 31)", howToCodeURL},
	{"How To Code in Go", []string{"pointers", "pointer"}, "Understanding Pointers in Go (Chapter 35)", howToCodeURL},
	{"How To Code in Go", []string{"type conversions", "converting data types", "casting"}, "How To Convert Data Types in Go (Chapter 15)", howToCodeURL},
	{"How To Code in Go", []string{"installing go", "getting started", "set up go"}, "How To Install Go and Set Up a Local Programming Environment (Chapters 4–6)", howToCodeURL},
	// Astaxie §3.2: the simple-web-server chapter whose form
	// parsing the form-values row teaches — the reading
	// behind the row.
	{"Astaxie's Build Web Application with Golang", []string{"form parsing", "parseform", "form values", "reading form values"}, "Build a Simple Web Server (§3.2)", astaxie32URL},
	// Practical Go Lessons, chapter 1: no Go code at all —
	// what a computer is, what a program is, compiled vs
	// interpreted. Definitional ground a beginner's first
	// "resources for …" asks stand on.
	{"Practical Go Lessons", []string{"how a computer works", "computer", "hardware", "cpu"}, "Programming a Computer (Chapter 1)", practicalGo1URL},
	{"Practical Go Lessons", []string{"compiled language", "interpreted", "compiler", "what is a program"}, "Programming a Computer (Chapter 1)", practicalGo1URL},
	// The basics sweep (2026-11-01): YourBasic's go-to
	// guide and Callicoder's tutorial series, learned the
	// five-sources way. What the concept book can assemble
	// became rows; what is reference-shaped — cheat sheets,
	// gotchas, comparisons, the race/deadlock explainers —
	// seats here as reading. YourBasic labels name the
	// article; the link is the verified index (see
	// yourbasicURL above).
	{"YourBasic", []string{"go vs java", "java", "coming from java"}, "Go vs. Java: 15 main differences / Java to Go in-depth tutorial", yourbasicURL},
	{"YourBasic", []string{"nil map", "assignment to entry in nil map", "nil maps"}, "Gotcha: Assignment to entry in nil map", yourbasicURL},
	{"YourBasic", []string{"nil pointer dereference", "invalid memory address", "nil pointer"}, "Gotcha: Invalid memory address or nil pointer dereference", yourbasicURL},
	{"YourBasic", []string{"multiple-value in single-value context", "single-value context"}, "Gotcha: Multiple-value in single-value context", yourbasicURL},
	{"YourBasic", []string{"array won't change", "arrays are values", "array copy"}, "Gotcha: Array won't change (arrays are values)", yourbasicURL},
	{"YourBasic", []string{"regexp cheat sheet", "regex syntax", "regular expression cheat sheet"}, "Regexp tutorial and cheat sheet", yourbasicURL},
	{"YourBasic", []string{"string cheat sheet", "string handling", "string overview"}, "Go string handling overview [cheat sheet]", yourbasicURL},
	{"YourBasic", []string{"printf formatting", "format verbs", "printf verbs"}, "fmt.Printf formatting tutorial and cheat sheet", yourbasicURL},
	{"YourBasic", []string{"bitwise operators", "bit operators", "bit shifting"}, "Bitwise operators [cheat sheet]", yourbasicURL},
	{"YourBasic", []string{"operators", "arithmetic operators", "comparison operators"}, "Operators: complete list", yourbasicURL},
	{"YourBasic", []string{"project layout", "standard layout", "project structure"}, "Start a new Go project [standard layout]", yourbasicURL},
	{"YourBasic", []string{"data races", "data race", "race conditions"}, "Data races explained", yourbasicURL},
	{"YourBasic", []string{"race detector", "detect data races", "find data races"}, "How to detect data races", yourbasicURL},
	{"YourBasic", []string{"deadlocks", "deadlock", "debug deadlocks"}, "How to debug deadlocks", yourbasicURL},
	{"YourBasic", []string{"parallel computation", "efficient parallel", "parallelism"}, "3 rules for efficient parallel computation", yourbasicURL},
	{"YourBasic", []string{"object-oriented programming", "oop", "inheritance", "classes"}, "Object-oriented programming without inheritance", yourbasicURL},
	{"YourBasic", []string{"zero values", "zero value", "default zero values"}, "Default zero values for all Go types", yourbasicURL},
	{"YourBasic", []string{"int vs int64", "int or int64", "which integer type"}, "Pick the right one: int vs. int64", yourbasicURL},
	{"YourBasic", []string{"redeclaring variables", "redeclare variables", "variable shadowing"}, "Redeclaring variables", yourbasicURL},
	{"YourBasic", []string{"initialization order", "execution order", "package initialization"}, "Package initialization and program execution order", yourbasicURL},
	{"Callicoder", []string{"basic types", "data types", "operators and types"}, "Basic Types, Operators, and Type Conversion", callicoderTypesURL},
	// The security sweep (2026-11-01): the reading behind
	// the security rows. What the book can assemble became
	// rows (CORS, the JWT pair, the ownership check, the
	// nonce and CSRF middleware, security headers, the
	// flagged cookie, server timeouts, the body limit, the
	// rate limiter); what stays judgment — password storage
	// above all, since bcrypt lives in x/crypto and the book
	// writes stdlib only — seats here, with the duplicates
	// (SQL injection is the parameterized database rows;
	// XSS is html/template doing its escaping) named in the
	// sweep's coverage report, not rebuilt.
	{"OWASP Cheat Sheets", []string{"jwt", "json web token", "json web tokens"}, "JSON Web Token Cheat Sheet for Java", owaspJWTURL},
	{"OWASP Cheat Sheets", []string{"idor", "insecure direct object reference", "object-level authorization"}, "Insecure Direct Object Reference Prevention Cheat Sheet", owaspIDORURL},
	{"OWASP Cheat Sheets", []string{"password storage", "storing passwords", "hash passwords", "password hashing", "bcrypt"}, "Password Storage Cheat Sheet", owaspPasswordsURL},
	{"OWASP Cheat Sheets", []string{"security headers", "http headers", "response headers"}, "HTTP Headers Cheat Sheet", owaspHeadersURL},
	{"MDN Web Docs", []string{"cors", "cross-origin resource sharing", "cross origin requests"}, "Cross-Origin Resource Sharing (CORS)", mdnCORSURL},
	{"Go standard library", []string{"tls", "https", "certificates", "listenandservetls"}, "net/http: Server.ListenAndServeTLS", listenServeTLSURL},
	{"Go standard library", []string{"cross origin protection", "crossoriginprotection", "built-in csrf"}, "net/http: CrossOriginProtection (Go 1.25+)", crossOriginURL},
	// The database build (2026-11-01): the Go Wiki's
	// SQLInterface page, section by section — the reading
	// behind the book's database family. What the page
	// teaches as assemble-able shapes the concept book
	// already owns as rows (the Context forms ride along
	// here as reading); what it teaches as judgment — that
	// sql.Open never connects, that a driver registers its
	// name through a blank import, how NULL rides in a
	// scan — seats here. Two sections yield no entry at
	// all: implementing a driver (driver.Driver/Conn/Stmt/
	// Tx is a whole multi-type package, not reading for a
	// row) and the page's generic reflection table helper
	// (generics — the book's named type-parameter wall);
	// both exclusions are named in the build's coverage
	// report, not rebuilt.
	{"the Go Wiki", []string{"database", "databases", "database/sql", "sql database"}, "SQLInterface: how database/sql works", sqlInterfaceURL},
	{"the Go Wiki", []string{"database driver", "drivers", "blank import", "driver registration"}, "SQLInterface: Database driver — a driver registers its name through a blank import", sqlInterfaceURL},
	{"the Go Wiki", []string{"connecting to a database", "sql.open", "pingcontext", "open does not connect"}, "SQLInterface: Connecting to a database — sql.Open does not connect; PingContext proves it", sqlInterfaceURL},
	{"the Go Wiki", []string{"executing queries", "querycontext", "execcontext", "prepared statements"}, "SQLInterface: Executing queries — the Context forms (ExecContext, QueryContext, QueryRowContext, PrepareContext)", sqlInterfaceURL},
	{"the Go Wiki", []string{"transactions", "transaction", "begintx", "commit or rollback"}, "SQLInterface: Transactions — BeginTx, then Commit or Rollback", sqlInterfaceURL},
	{"the Go Wiki", []string{"null values", "nullstring", "nullable columns", "sql null"}, "SQLInterface: Dealing with NULL — sql.NullString and the other Null types", sqlInterfaceURL},
	// The cheatsheet sweep (2026-11-01): the three sheets
	// themselves, as reading. Devhints is a syntax sheet —
	// nearly every section duplicates a book row (the two
	// that didn't, the while loop and type conversions,
	// became rows), and its methods sections stand behind
	// the receiver wall like every source's. GoInterview is
	// interview-shaped: its sections duplicate the rows and
	// the sync/context/io machinery a sentence can't carry
	// stays reading. Programming Idioms is a few hundred
	// tiny tasks: the statement-sized ones the book lacked
	// became rows (the map and slice verbs, reading input,
	// swapping, summing, the string toolbox's other half,
	// the file verbs), the rest are duplicates of rows or
	// exclusions with named walls — algorithms, whole
	// programs, data-structure designs, expression
	// fragments, and the signature machinery (parameters
	// and results the slots can't carry).
	{"Devhints", []string{"cheatsheet", "devhints", "cheat sheet", "quick reference", "syntax reference"}, "Go cheatsheet — the one-page syntax reference", devhintsURL},
	{"GoInterview", []string{"gointerview", "interview", "interview cheatsheet", "interview questions"}, "Go Cheatsheet 2026 — interactive reference with code examples", gointerviewURL},
	{"Programming Idioms", []string{"idioms", "programming idioms", "code idioms"}, "Programming Idioms — Go: small tasks, each answered in Go", programmingIdiomsURL},
}

// referenceMatchCap bounds the reference lines in one pack:
// pointers, not a second pack.
const referenceMatchCap = 3

// matchReferences scores the curated sources against the goal:
// an entry qualifies when two of its topic terms meet the
// goal's expanded terms, or when its primary phrase appears
// whole in the goal as typed. One stray shared word never
// drags a section in.
func matchReferences(goal string) []referenceEntry {
	raw := termSet(goal)
	expanded := expandGoalTerms(goal)
	type scored struct {
		e     referenceEntry
		score int
		idx   int
	}
	var hits []scored
	for i, e := range referenceSeed {
		terms := map[string]bool{}
		for _, t := range e.terms {
			for k := range termSet(t) {
				terms[k] = true
			}
		}
		score := intersect(expanded, terms)
		primaryHit := false
		if p := termSet(e.terms[0]); len(p) > 0 && intersect(raw, p) == len(p) {
			primaryHit = true
		}
		if score >= 2 || primaryHit {
			hits = append(hits, scored{e, score, i})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].score != hits[j].score {
			return hits[i].score > hits[j].score
		}
		return hits[i].idx < hits[j].idx
	})
	if len(hits) > referenceMatchCap {
		hits = hits[:referenceMatchCap]
	}
	// Group by source (referenceSeed order), best first within
	// a source, so a pack never interleaves two sources'
	// headers.
	sort.SliceStable(hits, func(i, j int) bool {
		return referenceSourceRank(hits[i].e.source) < referenceSourceRank(hits[j].e.source)
	})
	out := make([]referenceEntry, 0, len(hits))
	for _, h := range hits {
		out = append(out, h.e)
	}
	return out
}

func referenceSourceRank(source string) int {
	for i, e := range referenceSeed {
		if e.source == source {
			return i
		}
	}
	return len(referenceSeed)
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

	// Curated reference sections ride along where a pack (or an
	// explicit resources ask) earned them: the goal's own words
	// matched a section's topic, so the pointer is earned,
	// never sprayed. Reference lines are not pack entries (no
	// bullet), so the entry budget is untouched.
	if refs := matchReferences(goal); len(refs) > 0 && (qualified || explicit) {
		b.WriteString("\n")
		lastSource := ""
		for _, e := range refs {
			if e.source != lastSource {
				fmt.Fprintf(&b, "From %s:\n", e.source)
				lastSource = e.source
			}
			fmt.Fprintf(&b, "%s — %s\n", e.label, e.url)
		}
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
