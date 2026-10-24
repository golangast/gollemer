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
	lastAnalyzeRoot = root
	lastAnalyzeProject = p
	return intro + renderAnalyzeReport(p)
}

// repoRootRe finds a candidate repo reference: an optional scheme,
// then github.com/<owner>/<repo>. Whether it names the repo ROOT
// (rather than a page inside it) is decided by what follows the
// match — see repoQuestionURL.
var repoRootRe = regexp.MustCompile(`(?i)(?:https?://)?\bgithub\.com/([\w.-]+)/([\w.-]+)`)

// repoQuestionURL detects a natural-language question ABOUT a GitHub
// repo: a repo-root URL plus analysis intent words ("analyze
// https://github.com/foo/bar", "take a look at this repo: <url>",
// "what does https://github.com/foo/bar do?"). It returns the
// canonical repo URL.
//
// Only the repo root counts: a URL that continues into the repo
// (/issues/…, /pull/…, /blob/…, any deeper path) names a page, not
// the project — issue/PR URLs belong to tryIssueWhere, and other
// deep links fall through untouched rather than being force-cloned.
func repoQuestionURL(line string) (string, bool) {
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
			return fmt.Sprintf("https://github.com/%s/%s", owner, repo), true
		}
	}
	return "", false
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
	out, ok := tryNaturalAnalyze(line)
	if !ok {
		return false
	}
	fmt.Println(out)
	recordBrain("analyst", true, "repo url question")
	recordIntentOutcome(line, "analyze", 1, IntentOutcomeDispatched)
	return true
}
