package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	issueURLRe = regexp.MustCompile(`github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/issues/(\d+)`)
	whereForRe = regexp.MustCompile(`(?i)^where for (.+)$`)
)

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
	fmt.Printf("   cloning %s/%s (once, cached)…\n", owner, repo)
	dir, err := ensureGitHubRepo("github.com/" + owner + "/" + repo)
	if err != nil {
		return fmt.Sprintf("Couldn't clone %s/%s: %s", owner, repo, err), true
	}
	fmt.Printf("   analyzing %s/%s…\n", owner, repo)
	p, err := analyze.Analyze(dir)
	if err != nil {
		return fmt.Sprintf("Couldn't analyze %s/%s: %s", owner, repo, err), true
	}
	concepts := analyze.ExtractIssueConcepts(issue.Title + "\n\n" + issue.Body)
	out := p.GuideIssue(concepts)
	var b strings.Builder
	fmt.Fprintf(&b, "Issue #%s: %s\n\n%s", num, issue.Title, out)
	return b.String(), true
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
