package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	issueURLRe  = regexp.MustCompile(`github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/issues/(\d+)`)
	issueListRe = regexp.MustCompile(`github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/issues(?:$|/(?:$|[^0-9])|[^0-9/])`)
	whereForRe  = regexp.MustCompile(`(?i)^where for (.+)$`)
	bareNumRe   = regexp.MustCompile(`^\d+$`)
)

// listedRef remembers one entry from a listed set of issues or PRs so
// the user can reply with just its number.
type listedRef struct {
	kind        string // "issues" or "pull"
	owner, repo string
	num, title  string
}

var lastList []listedRef

type githubListItem struct {
	Number      int             `json:"number"`
	Title       string          `json:"title"`
	PullRequest json.RawMessage `json:"pull_request"`
}

type githubIssue struct {
	Title string `json:"title"`
	Body  string `json:"body"`
}

// tryIssueWhere answers "where do I make this change?" for a GitHub
// issue. Paste the issue URL (or say "where for <url>"): the issue text
// is fetched, the repo is cloned once and cached, and the engine
// renders the guided answer — the config struct to extend, the shared
// mechanism to wrap, and the call chain it sits in.
func tryIssueWhere(line string) (string, bool) {
	line = expandBareNumber(line)
	if m := issueListRe.FindStringSubmatch(line); m != nil {
		return listIssues(m[1], m[2]), true
	}
	owner, repo, num := "", "", ""
	if m := issueURLRe.FindStringSubmatch(line); m != nil {
		owner, repo, num = m[1], m[2], m[3]
	} else if m := whereForRe.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
		if im := issueURLRe.FindStringSubmatch(m[1]); im != nil {
			owner, repo, num = im[1], im[2], im[3]
		}
	}
	if owner == "" {
		return "", false
	}

	fmt.Printf("   fetching issue #%s…\n", num)
	issue, err := fetchIssue(owner, repo, num)
	if err != nil {
		return fmt.Sprintf("Couldn't fetch that issue: %s", err), true
	}
	// If the chat is already looking at this repo (e.g. you cloned it
	// with `clone <url> in folder example`), analyze your checkout —
	// so the file links point at the folder you cloned to.
	dir, err := checkoutFor(owner, repo)
	if err != nil {
		return err.Error(), true
	}
	fmt.Printf("   analyzing %s/%s…\n", owner, repo)
	p, err := analyze.Analyze(dir)
	if err != nil {
		return fmt.Sprintf("Couldn't analyze %s/%s: %s", owner, repo, err), true
	}
	lastAnalyzeRoot = dir
	lastAnalyzeProject = p
	concepts := analyze.ExtractIssueConcepts(issue.Title + "\n\n" + issue.Body)
	out := p.GuideIssue(concepts)
	var b strings.Builder
	fmt.Fprintf(&b, "Issue #%s: %s\n\n%s", num, issue.Title, out)
	return b.String(), true
}

// expandBareNumber rewrites a lone "320" into the full issue/PR URL
// when the user just picked from a listed set of issues or PRs.
// Issue and PR numbers share one sequence per repo, so the number is
// unambiguous.
func expandBareNumber(line string) string {
	n := strings.TrimSpace(line)
	if !bareNumRe.MatchString(n) || len(lastList) == 0 {
		return line
	}
	for _, r := range lastList {
		if r.num == n {
			return fmt.Sprintf("https://github.com/%s/%s/%s/%s", r.owner, r.repo, r.kind, n)
		}
	}
	return line
}

// listIssues fetches the repo's open issues (excluding pull requests)
// and remembers them so the user can reply with just a number.
func listIssues(owner, repo string) string {
	var items []githubListItem
	err := githubGet(fmt.Sprintf("https://api.github.com/repos/%s/%s/issues?state=open&per_page=10", owner, repo), &items)
	if err != nil {
		return fmt.Sprintf("Couldn't list %s/%s issues: %s", owner, repo, err)
	}
	var refs []listedRef
	for _, it := range items {
		if len(it.PullRequest) > 0 {
			continue // the issues endpoint also returns PRs
		}
		refs = append(refs, listedRef{kind: "issues", owner: owner, repo: repo,
			num: fmt.Sprint(it.Number), title: it.Title})
	}
	lastList = refs
	if len(refs) == 0 {
		return fmt.Sprintf("%s/%s has no open issues.", owner, repo)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Open issues in %s/%s:\n", owner, repo)
	for _, r := range refs {
		fmt.Fprintf(&b, "\n  #%s — %s", r.num, r.title)
	}
	b.WriteString("\n\nReply with the number, or paste the issue URL.")
	return b.String()
}

// cwd returns the working directory, or "" when it can't be told.
func cwd() string {
	if c, err := os.Getwd(); err == nil {
		return c
	}
	return ""
}

// gitOriginRepo returns "owner/repo" for the origin remote of the git
// checkout at dir, or "" when dir isn't a git repo or has no origin.
func gitOriginRepo(dir string) string {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "git", "-C", dir, "remote", "get-url", "origin").Output()
	if err != nil {
		return ""
	}
	m := githubURL.FindStringSubmatch(strings.TrimSpace(string(out)))
	if m == nil {
		return ""
	}
	return strings.ToLower(m[1] + "/" + strings.TrimSuffix(m[2], ".git"))
}

// fetchIssue reads a public issue's title and body via the GitHub API.
// No auth: public issues only, which is all a beginner needs here.
func fetchIssue(owner, repo, num string) (*githubIssue, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET",
		fmt.Sprintf("https://api.github.com/repos/%s/%s/issues/%s", owner, repo, num), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "gollemer")
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("GitHub API: %s", resp.Status)
	}
	var issue githubIssue
	if err := json.NewDecoder(resp.Body).Decode(&issue); err != nil {
		return nil, err
	}
	return &issue, nil
}
