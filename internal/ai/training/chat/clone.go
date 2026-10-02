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

// tryCloneProject clones a GitHub repo into ~/workspace/<folder> — a
// place you can see — and analyzes it on the spot, so follow-up
// questions ("what does X do", "where would I add Y") work immediately.
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
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Sprintf("I couldn't find your home directory: %v", err), true
	}
	dest := filepath.Join(home, "workspace", folder)

	if fi, err := os.Stat(dest); err == nil {
		if !fi.IsDir() {
			return fmt.Sprintf("`~/workspace/%s` exists but isn't a folder — pick another name.", folder), true
		}
		entries, _ := os.ReadDir(dest)
		if len(entries) > 0 {
			if _, err := os.Stat(filepath.Join(dest, ".git")); err != nil {
				return fmt.Sprintf("`~/workspace/%s` already exists and isn't a git repo — pick another folder name, or `analyze ~/workspace/%s` to look at what's there.", folder, folder), true
			}
			return analyzeCloned(dest, fmt.Sprintf("`~/workspace/%s` is already cloned — here's what's in it:\n\n", folder)), true
		}
	}

	if _, err := exec.LookPath("git"); err != nil {
		return "git isn't installed, so I can't clone.", true
	}
	url := rawURL
	if !strings.HasPrefix(strings.ToLower(url), "http") {
		url = "https://" + url
	}
	fmt.Printf("   cloning into ~/workspace/%s…\n", folder)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, "git", "clone", "--depth", "1", url, dest).CombinedOutput(); err != nil {
		return fmt.Sprintf("Clone failed: %v\n%s", err, strings.TrimSpace(string(out))), true
	}
	return analyzeCloned(dest, fmt.Sprintf("Cloned into `~/workspace/%s`. Here's what I see:\n\n", folder)), true
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
