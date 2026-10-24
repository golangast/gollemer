package chat

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// resourcesFixtureDoc is a miniature Advocates snapshot in the
// shipped file's shape: an http topic, the scaffolding topic (the
// doc's own words for REST APIs), a beginner topic, and a big
// general topic whose entries only corroborate.
func resourcesFixtureDoc() *advocatesDoc {
	return &advocatesDoc{
		Source: "fixture",
		Topics: []advocatesTopic{
			{Name: "Beginner", Entries: []advocatesEntry{
				{Label: "Example problems", URL: "https://tour.golang.org/welcome/1"},
				{Label: "HTTP servers for beginners", URL: "https://example.com/beginner-http"},
			}},
			{Name: "http package", Entries: []advocatesEntry{
				{Label: "Http package go wiki", URL: "https://golang.org/doc/articles/wiki/"},
				{Label: "Logging http", URL: "https://example.com/logging-http"},
				{Label: "security", URL: "https://example.com/security"},
			}},
			{Name: "Resources on scaffolding restful api from url", Entries: []advocatesEntry{
				{Label: "Restful resource", URL: "https://example.com/restful-resource"},
				{Label: "curl builder", URL: "https://example.com/curl"},
			}},
			{Name: "General", Entries: []advocatesEntry{
				{Label: "Server patterns", URL: "https://example.com/server-patterns"},
				{Label: "HTTP caching", URL: "https://example.com/http-caching"},
				{Label: "All about slices", URL: "https://example.com/slices"},
			}},
			{Name: "My Own", Entries: []advocatesEntry{
				{Label: "My project notes", URL: "https://example.com/own-notes"},
			}},
		},
	}
}

func withAdvocatesDoc(t *testing.T, doc *advocatesDoc) {
	t.Helper()
	oldCache, oldOnce, oldRoot := advocatesCache, advocatesOnce, unifiedProjectRoot
	advocatesCache = doc
	advocatesOnce = &sync.Once{}
	unifiedProjectRoot = t.TempDir() // no data/advocates.json under it
	t.Cleanup(func() {
		advocatesCache, advocatesOnce, unifiedProjectRoot = oldCache, oldOnce, oldRoot
	})
}

// serverProjectFixture is a tiny project that already runs an
// HTTP server: main → runServer → http.ListenAndServe.
func serverProjectFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/srvfix\n\ngo 1.21\n")
	write("main.go", `package main

import "net/http"

func main() { runServer() }

func runServer() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {})
	_ = http.ListenAndServe(":8080", nil)
}
`)
	return dir
}

func useProject(t *testing.T, dir string) {
	t.Helper()
	oldProj, oldRoot := lastAnalyzeProject, lastAnalyzeRoot
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	lastAnalyzeProject = p
	lastAnalyzeRoot = dir
	t.Cleanup(func() { lastAnalyzeProject, lastAnalyzeRoot = oldProj, oldRoot })
}

func TestParseBuildGoalShapes(t *testing.T) {
	cases := []struct {
		line     string
		goal     string
		explicit bool
		ok       bool
	}{
		{"i want to build a webserver", "webserver", false, true},
		{"I wanna build a REST API", "REST API", false, true},
		{"how do i build a web server in go", "web server", false, true},
		{"i'm building a chat app", "chat app", false, true},
		{"help me build a cli tool", "cli tool", false, true},
		{"building a game", "game", false, true},
		{"resources for websockets", "websockets", true, true},
		{"resources http package", "http package", true, true},
		// Not learning shapes — other owners keep these lines.
		{"build api for /search and /index", "", false, false}, // the composer
		{"build me a webserver", "", false, false},             // imperative: composer/edits family
		{"rename Count to Total", "", false, false},
		{"what does codec.decoderBincBytes.mapEnd do", "", false, false},
		{"what does https://github.com/ugorji/go do", "", false, false},
		{"add a case for \"rain\" to the weather switch", "", false, false},
		{"where does Items get its data", "", false, false},
		// A code-writing ask, not a build goal: the gocode brain's.
		{"i want to build a function that reverses a string", "", false, false},
		// A dotted symbol is symbolpick's, never a goal.
		{"resources for http.Server", "", false, false},
	}
	for _, c := range cases {
		goal, explicit, ok := parseBuildGoal(c.line)
		if ok != c.ok || goal != c.goal || explicit != c.explicit {
			t.Errorf("parseBuildGoal(%q) = (%q, %v, %v), want (%q, %v, %v)",
				c.line, goal, explicit, ok, c.goal, c.explicit, c.ok)
		}
	}
}

// The canonical case: a webserver goal over the fixture doc, in a
// project that already runs a server.
func TestBuildResourcesWebserverPack(t *testing.T) {
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("i want to build a webserver", nil) {
			t.Error("webserver goal not claimed")
		}
	})
	for _, want := range []string{
		"Resources for building webserver — from your Advocates doc:",
		"This project already has one: main.runServer in ",
		"walk me through runServer",
		"https://golang.org/doc/articles/wiki/",
		"https://example.com/restful-resource",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("pack missing %q\n---\n%s", want, out)
		}
	}
	// The composer tie-in is for API goals only — not offered here.
	if strings.Contains(out, "build api for") {
		t.Errorf("composer offered for a non-API goal:\n%s", out)
	}
	// Entry cap: at most 6 link lines, however big the doc gets.
	if n := strings.Count(out, "• "); n > packEntryBudget {
		t.Errorf("pack printed %d entries, budget is %d\n%s", n, packEntryBudget, out)
	}
	// Ranking: the scaffolding topic (score 7) precedes the http
	// package (score 5); both are name hits.
	if strings.Index(out, "restful-resource") > strings.Index(out, "articles/wiki") {
		t.Errorf("pack order wrong (scaffolding should lead):\n%s", out)
	}
	// Below-bar corroboration stays out: General scores 2 here.
	if strings.Contains(out, "server-patterns") {
		t.Errorf("below-bar topic leaked into the pack:\n%s", out)
	}
}

func TestBuildResourcesComposerTieIn(t *testing.T) {
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("i want to build a rest api", nil) {
			t.Error("rest api goal not claimed")
		}
	})
	if !strings.Contains(out, "https://example.com/restful-resource") {
		t.Errorf("scaffolding topic missing from API pack:\n%s", out)
	}
	if !strings.Contains(out, `say "build api for /users"`) {
		t.Errorf("composer tie-in missing for an API goal:\n%s", out)
	}
}

// A goal the doc barely brushes (two label hits, no topic named)
// is a near miss: claimed, answered plainly, the closest topic
// labeled as a closest — never dressed up as a match.
func TestBuildResourcesNearMiss(t *testing.T) {
	withAdvocatesDoc(t, &advocatesDoc{Topics: []advocatesTopic{
		{Name: "General", Entries: []advocatesEntry{
			{Label: "Server patterns", URL: "https://example.com/server-patterns"},
			{Label: "HTTP caching", URL: "https://example.com/http-caching"},
		}},
	}})
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("i want to build a server", nil) {
			t.Error("near-miss goal not claimed")
		}
	})
	if !strings.Contains(out, `I don't have a topic for "server"`) {
		t.Errorf("near miss not answered plainly:\n%s", out)
	}
	if !strings.Contains(out, `Closest on the shelf: "General"`) {
		t.Errorf("closest topic not named:\n%s", out)
	}
	// The near-miss path prints no pack entries.
	if strings.Contains(out, "• ") {
		t.Errorf("near miss printed pack entries:\n%s", out)
	}
	// But the project example still leads — the project HAS one.
	if !strings.Contains(out, "This project already has one:") {
		t.Errorf("project example missing from near-miss answer:\n%s", out)
	}
}

// An explicit resources ask for something nowhere in the doc:
// claimed (he asked), plain no-match, no invented links.
func TestBuildResourcesExplicitNoMatch(t *testing.T) {
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("resources for underwater basket weaving", nil) {
			t.Error("explicit resources ask not claimed")
		}
	})
	if !strings.Contains(out, "I don't have a topic for") {
		t.Errorf("no-match not plain:\n%s", out)
	}
	if strings.Contains(out, "example.com/") || strings.Contains(out, "golang.org/") {
		t.Errorf("no-match invented a link:\n%s", out)
	}
}

// Statements outside the feature's world keep falling through:
// the doc's own "My Own" topic must not turn "build my own PC"
// into a resource pack.
func TestBuildResourcesForeignStatementsFallThrough(t *testing.T) {
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))

	for _, line := range []string{
		"i want to build my own pc",
		"i'm building a deck this weekend",
		"building on what you said about the router yesterday",
	} {
		if tryBuildResources(line, nil) {
			t.Errorf("foreign statement claimed: %q", line)
		}
	}
}

// Without data/advocates.json the feature degrades to the
// built-in gobyexample seed instead of erroring.
func TestBuildResourcesSeedFallback(t *testing.T) {
	withAdvocatesDoc(t, nil)
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("i want to build a webserver", nil) {
			t.Error("seed-mode webserver goal not claimed")
		}
	})
	if !strings.Contains(out, "https://gobyexample.com/http-servers") {
		t.Errorf("seed pack missing the HTTP servers page:\n%s", out)
	}
	if strings.Contains(out, "Advocates") {
		t.Errorf("seed pack credited the absent doc:\n%s", out)
	}
}

// The other deterministic parsers' lines are untouched, checked at
// this parser's own gate (it runs before the intent layer in the
// loops, so its false is what keeps their routing intact).
func TestBuildResourcesDoesNotSteal(t *testing.T) {
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))

	for _, line := range []string{
		"build api for /search and /index",
		"build an api with /users and /health",
		"build me a webserver",
		"rename runServer to serve",
		"codec.decoderBincBytes.mapEnd",
		"what does https://github.com/ugorji/go do",
		"where does Items get its data",
	} {
		if tryBuildResources(line, nil) {
			t.Errorf("stole another parser's line: %q", line)
		}
	}
	// The composer's canonical phrasing still classifies as
	// build_api with the real model loaded, at full confidence —
	// the resources parser running first in the loops never sees
	// it as a goal (asserted above), so its route is unchanged.
	// (The "/search and /index" phrasing was already a classifier
	// miss → find before this feature existed; also unchanged.)
	m := withRealIntentModel(t)
	intent, _, ok := m.ClassifyIntent("build an api with /users and /health")
	if !ok || intent != "build_api" {
		t.Errorf("classifier read the composer line as %q (ok=%v), want build_api", intent, ok)
	}
}

// Synonym expansion is exactly one level and deterministic:
// "webserver" expands to the doc's server vocabulary, never to
// the synonyms of those synonyms (web → frontend was the leak
// that put CSS links at the top of a webserver pack).
func TestExpandGoalTermsOneLevel(t *testing.T) {
	e := expandGoalTerms("webserver")
	for _, want := range []string{"web", "server", "http", "rest", "scaffolding"} {
		if !e[want] {
			t.Errorf("expanded terms missing %q: %v", want, e)
		}
	}
	for _, bad := range []string{"frontend", "svelte", "database", "graphql"} {
		if e[bad] {
			t.Errorf("expansion leaked two levels: %q in %v", bad, e)
		}
	}
	// Scoring is stable across calls (no map-order dependence).
	doc := resourcesFixtureDoc()
	a, b := matchTopics(doc, "webserver"), matchTopics(doc, "webserver")
	if len(a) != len(b) {
		t.Fatalf("matchTopics unstable: %d vs %d topics", len(a), len(b))
	}
	for i := range a {
		if a[i].score != b[i].score || a[i].topic.Name != b[i].topic.Name {
			t.Errorf("matchTopics unstable at %d: %+v vs %+v", i, a[i], b[i])
		}
	}
}

// A pack that overflows its budget folds the remainder: at most
// two topics are named, the rest become a count.
func TestBuildResourcesRemainderFoldsTopics(t *testing.T) {
	mk := func(name string, labels ...string) advocatesTopic {
		t := advocatesTopic{Name: name}
		for _, l := range labels {
			t.Entries = append(t.Entries, advocatesEntry{Label: l, URL: "https://example.com/" + strings.ReplaceAll(strings.ToLower(l), " ", "-")})
		}
		return t
	}
	withAdvocatesDoc(t, &advocatesDoc{Topics: []advocatesTopic{
		mk("http package", "HTTP one", "HTTP two"),
		mk("server guide", "Server one", "Server two"),
		mk("General", "Server a", "Server b", "Server c", "Server d", "Server e"),
		mk("More", "Server f", "Server g", "Server h", "Server i"),
		mk("Even More", "Server j", "Server k", "Server l", "Server m"),
	}})
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("i want to build a server", nil) {
			t.Error("server goal not claimed")
		}
	})
	// Printed: 2 + 2 + 2 (of General's 5) = 6. Remainder:
	// 3 + 4 + 4 = 11 across three topics, two named, one folded.
	if !strings.Contains(out, `+11 more in "General", "More" and 1 other topic.`) {
		t.Errorf("remainder tail wrong:\n%s", out)
	}
	if strings.Contains(out, "Even More") {
		t.Errorf("folded topic was named anyway:\n%s", out)
	}
}

// The project-example search: the http signal wins for a server
// goal; nothing is claimed when the project has no instance.
func TestFindProjectExample(t *testing.T) {
	dir := serverProjectFixture(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	fn := findProjectExample(p, "webserver")
	if fn == nil || fn.Name != "runServer" {
		t.Fatalf("findProjectExample(webserver) = %v, want runServer", fn)
	}
	if got := findProjectExample(p, "quantum blockchain"); got != nil {
		t.Errorf("findProjectExample(quantum blockchain) = %v, want nil", got.Display())
	}
	if got := findProjectExample(nil, "webserver"); got != nil {
		t.Errorf("findProjectExample(nil project) = %v, want nil", got)
	}
}
