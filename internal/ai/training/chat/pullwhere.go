package chat

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	pullURLRe  = regexp.MustCompile(`github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pull/(\d+)`)
	pullListRe = regexp.MustCompile(`github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/pulls(?:$|/(?:$|[^0-9])|[^0-9/])`)
	closesRe   = regexp.MustCompile(`(?i)(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)`)
)

// tryPullWhere answers "what did this change?" for a pasted GitHub pull
// request URL (or "where for <url>"). It fetches the PR's changed files
// via the GitHub API, analyzes the repo, and lists each file with a
// plain-words one-liner of what the file is for — beginner-friendly
// diff triage without reading the diff.
func tryPullWhere(line string) (string, bool) {
	line = expandBareNumber(line)
	if m := pullListRe.FindStringSubmatch(line); m != nil {
		return listPulls(m[1], m[2]), true
	}
	owner, repo, num := pullTarget(line)
	if owner == "" {
		return "", false
	}

	fmt.Printf("   fetching PR #%s…\n", num)
	pr, err := fetchPull(owner, repo, num)
	if err != nil {
		return fmt.Sprintf("Couldn't fetch that PR: %s", err), true
	}
	files, err := fetchPullFiles(owner, repo, num)
	if err != nil {
		return fmt.Sprintf("Couldn't fetch that PR's files: %s", err), true
	}
	if len(files) == 0 {
		return fmt.Sprintf("PR #%s (%s) has no changed files I can see.", num, pr.Title), true
	}

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
	return renderPull(p, num, pr, files), true
}

// listPulls fetches the repo's open pull requests and remembers them
// so the user can reply with just a number.
func listPulls(owner, repo string) string {
	var items []githubListItem
	err := githubGet(fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls?state=open&per_page=10", owner, repo), &items)
	if err != nil {
		return fmt.Sprintf("Couldn't list %s/%s pull requests: %s", owner, repo, err)
	}
	lastList = nil
	for _, it := range items {
		lastList = append(lastList, listedRef{kind: "pull", owner: owner, repo: repo,
			num: fmt.Sprint(it.Number), title: it.Title})
	}
	if len(lastList) == 0 {
		return fmt.Sprintf("%s/%s has no open pull requests.", owner, repo)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Open pull requests in %s/%s:\n", owner, repo)
	for _, r := range lastList {
		fmt.Fprintf(&b, "\n  #%s — %s", r.num, r.title)
	}
	b.WriteString("\n\nReply with the number, or paste the PR URL.")
	return b.String()
}

// pullTarget extracts owner/repo/number from a pull-request URL pasted
// directly or via "where for <url>".
func pullTarget(line string) (owner, repo, num string) {
	if m := pullURLRe.FindStringSubmatch(line); m != nil {
		return m[1], m[2], m[3]
	}
	if m := whereForRe.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
		if pm := pullURLRe.FindStringSubmatch(m[1]); pm != nil {
			return pm[1], pm[2], pm[3]
		}
	}
	return "", "", ""
}

// checkoutFor returns the local directory to analyze for owner/repo: the
// checkout the chat is already looking at when its git origin matches
// (so file links point at the folder you cloned to), else the cached
// shallow clone.
func checkoutFor(owner, repo string) (string, error) {
	if want := strings.ToLower(owner + "/" + repo); lastAnalyzeRoot != "" && gitOriginRepo(lastAnalyzeRoot) == want {
		if rel, err := filepath.Rel(cwd(), lastAnalyzeRoot); err == nil {
			fmt.Printf("   using your %s checkout…\n", rel)
		}
		return lastAnalyzeRoot, nil
	}
	fmt.Printf("   cloning %s/%s (once, cached)…\n", owner, repo)
	dir, err := ensureGitHubRepo("github.com/" + owner + "/" + repo)
	if err != nil {
		return "", fmt.Errorf("couldn't clone %s/%s: %w", owner, repo, err)
	}
	return dir, nil
}

type githubPull struct {
	Title string `json:"title"`
	Body  string `json:"body"`
}

type githubPullFile struct {
	Filename  string `json:"filename"`
	Status    string `json:"status"` // added, modified, removed, renamed
	Additions int    `json:"additions"`
	Deletions int    `json:"deletions"`
}

func githubGet(url string, out any) error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "gollemer")
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("GitHub API: %s", resp.Status)
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func fetchPull(owner, repo, num string) (*githubPull, error) {
	var pr githubPull
	if err := githubGet(fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls/%s", owner, repo, num), &pr); err != nil {
		return nil, err
	}
	return &pr, nil
}

func fetchPullFiles(owner, repo, num string) ([]githubPullFile, error) {
	var files []githubPullFile
	if err := githubGet(fmt.Sprintf("https://api.github.com/repos/%s/%s/pulls/%s/files?per_page=100", owner, repo, num), &files); err != nil {
		return nil, err
	}
	return files, nil
}

// renderPull lists the PR's changed files grouped by change kind, each
// with its diff size and a plain-words one-liner from the analyzed code.
func renderPull(p *analyze.Project, num string, pr *githubPull, files []githubPullFile) string {
	var b strings.Builder
	fmt.Fprintf(&b, "PR #%s: %s\n", num, pr.Title)
	if m := closesRe.FindStringSubmatch(pr.Body); m != nil {
		fmt.Fprintf(&b, "Implements issue #%s.\n", m[1])
	}
	adds, dels := 0, 0
	for _, f := range files {
		adds += f.Additions
		dels += f.Deletions
	}
	fmt.Fprintf(&b, "\nTHIS PR TOUCHED %d FILES (+%d −%d):\n", len(files), adds, dels)

	groups := []struct {
		title string
		want  []string
	}{
		{"NEW FILES", []string{"added"}},
		{"CHANGED", []string{"modified", "renamed", "changed"}},
		{"REMOVED", []string{"removed"}},
	}
	shown := 0
	for _, g := range groups {
		var inGroup []githubPullFile
		for _, f := range files {
			for _, w := range g.want {
				if f.Status == w {
					inGroup = append(inGroup, f)
				}
			}
		}
		// Anything with an unexpected status still gets shown.
		if g.title == "CHANGED" {
			for _, f := range files {
				known := false
				for _, gg := range groups {
					for _, w := range gg.want {
						if f.Status == w {
							known = true
						}
					}
				}
				if !known {
					inGroup = append(inGroup, f)
				}
			}
		}
		if len(inGroup) == 0 {
			continue
		}
		fmt.Fprintf(&b, "\n  %s:\n", g.title)
		for _, f := range inGroup {
			fmt.Fprintf(&b, "    %s  (+%d −%d)\n", p.LinkPath(f.Filename), f.Additions, f.Deletions)
			if d := p.DescribeFile(f.Filename); d != "" {
				fmt.Fprintf(&b, "      — %s\n", d)
			}
			shown++
		}
	}
	if shown == 0 {
		fmt.Fprintf(&b, "\n  (no file details)\n")
	}
	return b.String()
}
