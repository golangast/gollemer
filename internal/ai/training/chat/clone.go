package chat

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	// clone <url> [in folder <name>] — "in folder" is John's phrasing;
	// a bare "in <name>" works too.
	cloneRe      = regexp.MustCompile(`(?i)^clone\s+(\S+?)(?:\s+in\s+(?:folder\s+)?([A-Za-z0-9_.-]+))?\s*$`)
	folderNameRe = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)
)

// parseCloneCommand pulls the URL and folder out of a clone request.
// It only claims the message when it starts with "clone" and names a
// GitHub URL, so "clone the repo" still reaches the normal brains.
func parseCloneCommand(line string) (rawURL, folder string, ok bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(strings.ToLower(trimmed), "clone ") {
		return "", "", false
	}
	m := cloneRe.FindStringSubmatch(trimmed)
	if m == nil || !githubURL.MatchString(m[1]) {
		return "", "", false
	}
	return m[1], m[2], true
}

// tryCloneProject clones a GitHub repo into ./<folder> — the folder you
// named, right where you are — and analyzes it on the spot, so
// follow-up questions ("what does X do", "where would I add Y") work
// immediately. Paths print relative to this directory, so file:line
// links in the terminal click straight through to the file.
func tryCloneProject(line string) (string, bool) {
	rawURL, folder, ok := parseCloneCommand(line)
	if !ok {
		return "", false
	}
	gm := githubURL.FindStringSubmatch(rawURL)
	repo := strings.TrimSuffix(gm[2], ".git")
	if folder == "" {
		folder = repo
	}
	if !folderNameRe.MatchString(folder) || folder == "." || folder == ".." {
		return "That folder name won't work — try a simple name like `example`.", true
	}
	cwd, err := os.Getwd()
	if err != nil {
		return fmt.Sprintf("I couldn't tell where I am: %v", err), true
	}
	dest := filepath.Join(cwd, folder)

	if fi, err := os.Stat(dest); err == nil {
		if !fi.IsDir() {
			return fmt.Sprintf("`./%s` exists but isn't a folder — pick another name.", folder), true
		}
		entries, _ := os.ReadDir(dest)
		if len(entries) > 0 {
			if _, err := os.Stat(filepath.Join(dest, ".git")); err != nil {
				return fmt.Sprintf("`./%s` already exists and isn't a git repo — pick another folder name, or `analyze ./%s` to look at what's there.", folder, folder), true
			}
			return analyzeCloned(dest, fmt.Sprintf("`./%s` is already cloned — here's what's in it:\n\n", folder)), true
		}
	}

	if _, err := exec.LookPath("git"); err != nil {
		return "git isn't installed, so I can't clone.", true
	}
	url := rawURL
	if !strings.HasPrefix(strings.ToLower(url), "http") {
		url = "https://" + url
	}
	fmt.Printf("   cloning into ./%s…\n", folder)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, "git", "clone", "--depth", "1", url, dest).CombinedOutput(); err != nil {
		return fmt.Sprintf("Clone failed: %v\n%s", err, strings.TrimSpace(string(out))), true
	}
	return analyzeCloned(dest, fmt.Sprintf("Cloned into `./%s`. Here's what I see:\n\n", folder)), true
}

// analyzeCloned analyzes a freshly cloned repo, remembers it for
// follow-up questions, and renders the same report as analyze.
func analyzeCloned(root, intro string) string {
	p, err := analyze.Analyze(root)
	if err != nil {
		return fmt.Sprintf("%sI couldn't analyze it: %v", intro, err)
	}
	setCurrentProject(root, p)
	return intro + renderAnalyzeReport(p)
}

// repoRootRe finds a candidate repo reference: an optional scheme,
// then github.com/<owner>/<repo>. Whether it names the repo ROOT
// (rather than a page inside it) is decided by what follows the
// match — see repoQuestionURL.
var repoRootRe = regexp.MustCompile(`(?i)(?:https?://)?\bgithub\.com/([\w.-]+)/([\w.-]+)`)

// repoRootURL returns the canonical URL when the line names a
// GitHub repo ROOT — and only the root: a URL that continues into
// the repo (/issues/…, /pull/…, /blob/…, any deeper path) names a
// page, not the project. Issue/PR URLs belong to tryIssueWhere, and
// other deep links fall through untouched rather than being
// force-cloned. Whether the line is a question, a getting-started
// ask, or neither is the callers' business — this is the shared
// root test underneath repoQuestionURL, repoBeginURL, and the
// dispatch floor.
func repoRootURL(line string) (string, bool) {
	loc := repoRootRe.FindStringSubmatchIndex(line)
	if loc == nil {
		return "", false
	}
	owner := line[loc[2]:loc[3]]
	repo := strings.TrimSuffix(line[loc[4]:loc[5]], ".git")
	repo = strings.TrimRight(repo, ".")
	if owner == "" || repo == "" {
		return "", false
	}
	// What follows the repo name decides whether this is the root:
	// end of line, whitespace, or closing punctuation is fine; a
	// further path segment ("/issues/12", "/blob/main/x.go") is a
	// page inside the repo, not the repo itself.
	if rest := line[loc[1]:]; rest != "" {
		if c := rest[0]; c == '/' {
			rest = rest[1:]
			if rest != "" && !strings.ContainsRune(" \t?.,;:!?)\"']", rune(rest[0])) {
				return "", false
			}
		}
	}
	return fmt.Sprintf("https://github.com/%s/%s", owner, repo), true
}

// repoQuestionURL detects a natural-language question ABOUT a GitHub
// repo: a repo-root URL plus analysis intent words ("analyze
// https://github.com/foo/bar", "take a look at this repo: <url>",
// "what does https://github.com/foo/bar do?"). It returns the
// canonical repo URL.
func repoQuestionURL(line string) (string, bool) {
	url, ok := repoRootURL(line)
	if !ok {
		return "", false
	}
	ll := strings.ToLower(line)
	// Analysis/question intent words. A bare URL mention ("my code
	// is at <url>") has none and falls through as before.
	intents := []string{
		"analyz", "look at", "check out", "examine", "what is", "what does",
		"how does", "explain", "tell me about", "show me", "take a look",
		"explore", "understand", "review", "inspect",
	}
	for _, in := range intents {
		if strings.Contains(ll, in) {
			return url, true
		}
	}
	return "", false
}

// repoBeginPhrases are the getting-started shapes: the user is not
// asking what the repo IS, they are asking where to begin READING
// it. The d1b80d4 intent-word family had none of these, so "where
// to begin for https://github.com/wailsapp/wails" fell through the
// whole dispatch chain to the social brain, which answered a
// greeting. Matched as substrings of the lowercased line; a claim
// additionally requires a repo-root URL (repoBeginURL), so these
// words on their own claim nothing here — "where do i start" about
// the current project keeps its existing owners (the /start guide,
// the learn-path machinery).
var repoBeginPhrases = []string{
	"where to begin", "where to start",
	"where do i begin", "where do i start",
	"where do we begin", "where do we start",
	"where should i begin", "where should i start",
	"where can i begin", "where can i start",
	"how do i start", "how do i begin",
	"how to start", "how to begin",
	"getting started", "get started",
	"begin with", "start with", "begin from", "start from",
	"starting with", "beginning with",
	"first steps", "start reading", "begin reading",
	"where would i start", "where would i begin",
}

// repoBeginURL detects a getting-started ask about a GitHub repo:
// a repo-root URL plus one of repoBeginPhrases.
func repoBeginURL(line string) (string, bool) {
	url, ok := repoRootURL(line)
	if !ok {
		return "", false
	}
	ll := strings.ToLower(line)
	for _, p := range repoBeginPhrases {
		if strings.Contains(ll, p) {
			return url, true
		}
	}
	return "", false
}

// tryRepoBegin answers a getting-started ask about a repo with the
// repo's reading guide — where a beginner's first hour goes —
// instead of the plain analyze report. Same clone path as
// tryNaturalAnalyze: the codebase-maps cache, the repo made the
// current project for follow-ups.
func tryRepoBegin(line string) (string, bool) {
	url, ok := repoBeginURL(line)
	if !ok {
		return "", false
	}
	return handleRepoGuide(url)
}

// tryNaturalAnalyze answers a natural-language repo analysis
// request (see repoQuestionURL) with the standard analyze flow: the
// repo is shallow-cloned into the codebase-maps cache (reused when
// already present), analyzed, and made the current project, so the
// report — summary and story — IS the answer to "what does it do",
// and follow-up questions are about that repo.
func tryNaturalAnalyze(line string) (string, bool) {
	url, ok := repoQuestionURL(line)
	if !ok {
		return "", false
	}
	return handleGoAnalyze("analyze " + url)
}

// tryRepoURLDispatch is the intent layer's deterministic claim for
// repo-URL questions, hooked into tryIntentDispatch before the
// classifier (the same place, and for the same reason, as the
// exact-symbol paths in symbolpick.go): the line names the exact
// project to read, so there is nothing to classify. Left to the
// classifier, "what does <url> do" was read as a summarize request
// and answered about the folder the chat happened to start in —
// tryNaturalAnalyze, which lives behind the classifier in
// tryDeterministicAnswer, never saw the line.
func tryRepoURLDispatch(line string) bool {
	// A getting-started ask gets the reading guide, not the plain
	// report — checked first, since a begin line may also carry an
	// analysis word ("explain where to begin for <url>").
	if out, ok := tryRepoBegin(line); ok {
		fmt.Println(out)
		recordBrain("analyst", true, "repo begin question")
		recordIntentOutcome(line, "learn_path", 1, IntentOutcomeDispatched)
		return true
	}
	out, ok := tryNaturalAnalyze(line)
	if !ok {
		return false
	}
	fmt.Println(out)
	recordBrain("analyst", true, "repo url question")
	recordIntentOutcome(line, "analyze", 1, IntentOutcomeDispatched)
	return true
}

// tryRepoURLFloor is the dispatch floor for repo-root URLs, called
// by tryIntentDispatch when every other layer has passed on the
// line: a line that names a whole GitHub repo must never reach the
// router and the brains. Left unclaimed, the router could send it
// to the social brain, which free-associates on a word in the repo
// name ("wails" became a greeting). The fallback is the standard
// analyze flow for that repo — or, when even that cannot run, an
// honest line about what a repo URL can be used for here.
//
// Lines another deterministic layer owns never reach the floor:
// the clone command (tryCloneProject runs later in the loops) is
// excluded here, issue/PR/deep links are not repo roots by
// definition, and plain tool commands that merely mention a repo
// path ("go get github.com/…") belong to their command brains.
func tryRepoURLFloor(line string) bool {
	url, ok := repoRootURL(line)
	if !ok {
		return false
	}
	if _, _, ok := parseCloneCommand(line); ok {
		return false
	}
	ll := strings.ToLower(strings.TrimSpace(line))
	if strings.HasPrefix(ll, "go ") || strings.HasPrefix(ll, "git ") || strings.HasPrefix(ll, "make ") {
		return false
	}
	if strings.Contains(line, ".go:") {
		return false // a pasted compiler error — the translator owns it
	}
	out, ok := handleGoAnalyze("analyze " + url)
	if !ok {
		fmt.Printf("That's a link to the %s repo. I can analyze it (\"analyze %s\"), show you where to begin reading it (\"where to begin for %s\"), or answer \"where for <issue-url>\" for one of its issues.\n", url, url, url)
		recordBrain("analyst", false, "repo url floor: no analyze path")
		return true
	}
	fmt.Println(out)
	recordBrain("analyst", true, "repo url floor")
	recordIntentOutcome(line, "analyze", 1, IntentOutcomeDispatched)
	return true
}
