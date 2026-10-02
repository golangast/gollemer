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

// unifiedProjectRoot is the repo directory the chat was started in
// (projectRoot in runUnifiedChat). Bare "analyze this project" requests
// read it.
var unifiedProjectRoot string

// lastAnalyzeRoot remembers the last successfully analyzed project so a
// follow-up "where would I add X" with no path reuses it.
var lastAnalyzeRoot string

// lastAnalyzeProject caches the analyzed project itself so follow-up
// questions ("what does routeDomain do") are answered from memory
// instead of re-parsing the tree.
var lastAnalyzeProject *analyze.Project

// tryCodebaseQuestion answers a natural-language question about the last
// analyzed project. It only fires when the question names a symbol that
// actually exists in the project, so it can never steal a question meant
// for another brain ("what does a goroutine do" stays with the Go brain,
// "what does make chat do" stays with the makefile brain).
func tryCodebaseQuestion(line string) (string, bool) {
	if lastAnalyzeProject == nil {
		return "", false
	}
	return lastAnalyzeProject.Answer(line)
}

var (
	quotedPath = regexp.MustCompile(`"([^"]+)"|'([^']+)'`)
	githubURL  = regexp.MustCompile(`(?i)(?:https?://)?github\.com/([\w.-]+)/([\w.-]+)`)
	// visualFollowup marks "show me a visual" style requests: a
	// show/draw/give/make/generate verb aimed at a visual noun. The
	// verb+noun shape keeps "show me routeDomain" (a source-excerpt
	// question, answered by the Q&A layer) and "how do I render html
	// templates" (a Go question) out.
	visualFollowup = regexp.MustCompile(`(?i)\b(show|draw|give|make|generate)\b.{0,24}?\b(visual|diagram|picture|graphs?)\b`)
	whereTask      = regexp.MustCompile(`(?i)\bwhere\b.{0,40}?\b(add|change|update|modify|fix|edit|put|implement|wire|hook)\b\s+(.{2,80})`)
)

// handleGoAnalyze answers a GoAnalyzeDomain message: it resolves the target
// project (path, github URL, or the current repo), runs the AST analysis,
// and renders the report. It returns ok=false only when the message
// doesn't actually name an analyzable target, letting the neural fallback
// try.
func handleGoAnalyze(line string) (out string, ok bool) {
	task, isWhere := parseWhereTask(line)
	raw := extractAnalyzePath(line)
	if raw == "" && !isWhere {
		// "analyze this project" with no path: read the repo the chat
		// runs in.
		raw = unifiedProjectRoot
	}
	if raw == "" && isWhere && lastAnalyzeRoot != "" {
		raw = lastAnalyzeRoot
	}
	if raw == "" {
		if isWhere && unifiedProjectRoot != "" {
			raw = unifiedProjectRoot
		} else {
			return "", false
		}
	}
	root, err := resolveAnalyzeRoot(raw)
	if err != nil {
		return fmt.Sprintf("I couldn't read that codebase: %v", err), true
	}
	fmt.Println("[reading the codebase — parsing Go files...]")
	p, err := analyze.Analyze(root)
	if err != nil {
		return fmt.Sprintf("I couldn't analyze %s: %v", root, err), true
	}
	lastAnalyzeRoot = root
	lastAnalyzeProject = p
	var b strings.Builder
	if isWhere {
		b.WriteString(p.WhereToChangeText(task))
		return b.String(), true
	}
	b.WriteString(renderAnalyzeReport(p))
	return b.String(), true
}

// renderAnalyzeReport is the standard codebase report: summary, story,
// reading guide, visuals. The clone command reuses it so a fresh clone
// gets the same treatment as analyze.
func renderAnalyzeReport(p *analyze.Project) string {
	var b strings.Builder
	b.WriteString(p.Summary())
	b.WriteString("\n")
	b.WriteString(p.Story())
	b.WriteString("\n")
	b.WriteString(p.ReadingGuide())
	b.WriteString("\n")
	b.WriteString(p.VisualReport())
	return b.String()
}

// parseWhereTask extracts the task description from "where would I add X".
// It strips a trailing path ("... in ~/foo" / "... in this project").
func parseWhereTask(line string) (task string, isWhere bool) {
	m := whereTask.FindStringSubmatch(line)
	if m == nil {
		return "", false
	}
	task = strings.TrimSpace(m[2])
	// Cut a trailing location: "a retry helper in ~/proj" -> "a retry helper".
	if i := strings.LastIndex(task, " in "); i > 0 {
		task = strings.TrimSpace(task[:i])
	}
	task = strings.Trim(task, `"'.,`)
	if task == "" {
		return "", false
	}
	return task, true
}

// extractAnalyzePath finds a filesystem path or GitHub URL in the message.
func extractAnalyzePath(line string) string {
	if m := quotedPath.FindStringSubmatch(line); m != nil {
		for _, g := range m[1:] {
			if g != "" {
				return g
			}
		}
	}
	if u := githubURL.FindString(line); u != "" {
		return u
	}
	lower := strings.ToLower(line)
	for _, phrase := range []string{"this project", "this repo", "this repository", "this codebase"} {
		if strings.Contains(lower, phrase) {
			return unifiedProjectRoot
		}
	}
	for _, tok := range strings.Fields(line) {
		clean := strings.TrimRight(tok, `.,;:!?`)
		// Strip one layer of surrounding quotes/brackets, but keep a
		// leading ./ or ~/ intact.
		clean = strings.TrimPrefix(clean, `"`)
		clean = strings.TrimPrefix(clean, `'`)
		clean = strings.TrimPrefix(clean, "(")
		clean = strings.TrimPrefix(clean, "[")
		clean = strings.TrimSuffix(clean, `"`)
		clean = strings.TrimSuffix(clean, `'`)
		clean = strings.TrimSuffix(clean, ")")
		clean = strings.TrimSuffix(clean, "]")
		if clean == "and/or" {
			continue
		}
		if strings.Contains(clean, "/") || strings.HasPrefix(clean, "~") {
			return clean
		}
	}
	return ""
}

// resolveAnalyzeRoot turns a user-supplied path or GitHub URL into a local
// directory: ~ is expanded, github.com URLs are shallow-cloned into
// ~/workspace/codebase-maps (cached between runs).
func resolveAnalyzeRoot(raw string) (string, error) {
	if githubURL.MatchString(raw) {
		return ensureGitHubRepo(raw)
	}
	path := raw
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("can't expand ~: %w", err)
		}
		path = filepath.Join(home, strings.TrimPrefix(path, "~"))
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	fi, err := os.Stat(abs)
	if err != nil {
		return "", fmt.Errorf("no such directory: %s", raw)
	}
	if !fi.IsDir() {
		return "", fmt.Errorf("%s is not a directory", raw)
	}
	return abs, nil
}

// ensureGitHubRepo shallow-clones owner/repo into ~/workspace/codebase-maps
// (reused if already present) and returns the local directory.
func ensureGitHubRepo(rawURL string) (string, error) {
	if _, err := exec.LookPath("git"); err != nil {
		return "", fmt.Errorf("git isn't installed, so I can't clone %s", rawURL)
	}
	m := githubURL.FindStringSubmatch(rawURL)
	owner, repo := m[1], strings.TrimSuffix(m[2], ".git")
	url := rawURL
	if !strings.HasPrefix(strings.ToLower(url), "http") {
		url = "https://" + url
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dest := filepath.Join(home, "workspace", "codebase-maps", owner+"-"+repo)
	if fi, err := os.Stat(dest); err == nil && fi.IsDir() {
		return dest, nil // cached clone
	}
	fmt.Printf("[cloning %s (shallow)...]\n", url)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "clone", "--depth", "1", url, dest)
	if out, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("clone failed: %v\n%s", err, strings.TrimSpace(string(out)))
	}
	return dest, nil
}
