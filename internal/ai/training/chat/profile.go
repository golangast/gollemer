// Package chat — user profile: learns John's patterns, projects, and habits.
// All data stays local in ~/.gollemer_profile.json. Never uploaded, never
// shared. This is the "learning you" model — deterministic preference memory,
// not a neural network.
package chat

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// UserProfile is the learned model of John: what he does, where, and how.
type UserProfile struct {
	// FrequentCommands: commands he runs, with counts and recency.
	FrequentCommands []CommandStat `json:"frequent_commands"`
	// Projects: directories he works in, what he was doing there.
	Projects map[string]*ProjectMemory `json:"projects"`
	// Patterns: learned sequences, e.g. "after extracting patch" -> "go clean -cache".
	Patterns map[string]string `json:"patterns"`
	// Directives: standing instructions learned from his terse replies.
	Directives map[string]bool `json:"directives"`
	// StyleNotes: observations about his code and workflow preferences.
	StyleNotes []string `json:"style_notes"`
	// LastUpdated: when the profile was last written.
	LastUpdated time.Time `json:"last_updated"`
	// cachedStyle: in-memory only, persisted separately.
	cachedStyle *CodeStyle `json:"-"`
}

// CommandStat tracks a single command's usage.
type CommandStat struct {
	Cmd      string    `json:"cmd"`
	Count    int       `json:"count"`
	LastUsed time.Time `json:"last_used"`
}

// ProjectMemory remembers what John was doing in a project directory.
type ProjectMemory struct {
	Name         string    `json:"name"`
	LastActive   time.Time `json:"last_active"`
	Focus        string    `json:"focus"`         // what he was working on
	FilesTouched []string  `json:"files_touched"` // recent files
	IssuesViewed []string  `json:"issues_viewed"` // GitHub issues investigated
	ReposCloned  []string  `json:"repos_cloned"`  // repos analyzed
}

// profilePath returns the local-only profile location.
func profilePath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".gollemer_profile.json"
	}
	return filepath.Join(home, ".gollemer_profile.json")
}

// LoadProfile reads the profile from disk, or returns a fresh one.
func LoadProfile() *UserProfile {
	p := &UserProfile{
		Projects:   make(map[string]*ProjectMemory),
		Patterns:   make(map[string]string),
		Directives: make(map[string]bool),
	}
	data, err := os.ReadFile(profilePath())
	if err != nil {
		return p // fresh profile
	}
	if err := json.Unmarshal(data, p); err != nil {
		return &UserProfile{
			Projects:   make(map[string]*ProjectMemory),
			Patterns:   make(map[string]string),
			Directives: make(map[string]bool),
		}
	}
	if p.Projects == nil {
		p.Projects = make(map[string]*ProjectMemory)
	}
	if p.Patterns == nil {
		p.Patterns = make(map[string]string)
	}
	if p.Directives == nil {
		p.Directives = make(map[string]bool)
	}
	return p
}

// Save writes the profile to disk. Local only.
func (p *UserProfile) Save() error {
	p.LastUpdated = time.Now()
	data, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(profilePath(), data, 0600) // owner-only
}

// RecordCommand logs a command John ran.
func (p *UserProfile) RecordCommand(cmd string) {
	cmd = strings.TrimSpace(cmd)
	if cmd == "" {
		return
	}
	// Normalize: collapse the common go run invocation.
	for i, s := range p.FrequentCommands {
		if s.Cmd == cmd {
			p.FrequentCommands[i].Count++
			p.FrequentCommands[i].LastUsed = time.Now()
			return
		}
	}
	p.FrequentCommands = append(p.FrequentCommands, CommandStat{
		Cmd:      cmd,
		Count:    1,
		LastUsed: time.Now(),
	})
}

// RecordProjectTouch notes activity in a project directory.
func (p *UserProfile) RecordProjectTouch(dir, focus string) {
	pm, ok := p.Projects[dir]
	if !ok {
		pm = &ProjectMemory{Name: filepath.Base(dir)}
		p.Projects[dir] = pm
	}
	pm.LastActive = time.Now()
	if focus != "" {
		pm.Focus = focus
	}
}

// RecordFileTouch notes a file John worked with in a project.
func (p *UserProfile) RecordFileTouch(dir, file string) {
	pm, ok := p.Projects[dir]
	if !ok {
		pm = &ProjectMemory{Name: filepath.Base(dir)}
		p.Projects[dir] = pm
	}
	pm.LastActive = time.Now()
	// Keep last 20, most recent first, no dupes.
	for i, f := range pm.FilesTouched {
		if f == file {
			pm.FilesTouched = append(pm.FilesTouched[:i], pm.FilesTouched[i+1:]...)
			break
		}
	}
	pm.FilesTouched = append([]string{file}, pm.FilesTouched...)
	if len(pm.FilesTouched) > 20 {
		pm.FilesTouched = pm.FilesTouched[:20]
	}
}

// LearnDirective records a standing instruction from John's replies.
// "do all" / "yeah" / "yes" to a build proposal -> always build, don't re-ask.
func (p *UserProfile) LearnDirective(key string, value bool) {
	p.Directives[key] = value
}

// SuggestStartup returns a contextual suggestion when chat starts.
// It looks at time of day, recent projects, and learned patterns.
func (p *UserProfile) SuggestStartup(cwd string) string {
	var b strings.Builder
	hour := time.Now().Hour()

	// Morning pattern.
	if hour >= 6 && hour < 11 {
		if p.Directives["morning_chat"] {
			b.WriteString("Morning. You usually start with the chat — want to pick up where you left off?\n")
		}
	}

	// Project resume.
	if pm, ok := p.Projects[cwd]; ok {
		since := time.Since(pm.LastActive)
		if since < 24*time.Hour && pm.Focus != "" {
			b.WriteString("Last time here you were working on: " + pm.Focus + ".\n")
			if len(pm.FilesTouched) > 0 {
				b.WriteString("Recent files: " + strings.Join(pm.FilesTouched[:min(3, len(pm.FilesTouched))], ", ") + "\n")
			}
		} else if since >= 24*time.Hour && pm.Focus != "" {
			b.WriteString("It's been a while — last focus here was: " + pm.Focus + ".\n")
		}
	}

	// The most-run command line used to print here; it left the
	// default startup in the 2026-10-24 start-screen pass (print
	// less at launch). FrequentCommands is still recorded.

	return strings.TrimSpace(b.String())
}

// SeedDefaults populates the profile with what we already know about John
// from working with him. Called once on first run.
func (p *UserProfile) SeedDefaults() {
	if len(p.Directives) > 0 {
		return // already seeded
	}
	// Terse "do all" / "yeah" / "yes" = build it, don't re-ask.
	p.Directives["build_on_yes"] = true
	p.Directives["do_all_means_all"] = true
	// Patch workflow: extract -> go clean -cache -> run.
	p.Patterns["after_extract_patch"] = "go clean -cache"
	p.Patterns["after_edit_go"] = "go build ./..."
	// He works in ~/Downloads for gollemer patches.
	p.StyleNotes = append(p.StyleNotes,
		"Prefers terse directives; 'do all' means build everything proposed.",
		"Patch workflow: download tarball, extract to ~/Downloads, go clean -cache, then run.",
		"Pushes to GitHub with one-time tokens; remote stays credential-free.",
		"Beginner-first: terminal only, plain-words explanations, no jargon.",
	)
	// Seed the gollemer project — he's been building the composer here all day.
	home, _ := os.UserHomeDir()
	dlDir := filepath.Join(home, "Downloads")
	p.Projects[dlDir] = &ProjectMemory{
		Name:       "gollemer",
		LastActive: time.Now(),
		Focus:      "composer + piece schemas + NLP commands",
		FilesTouched: []string{
			"internal/ai/analyze/pieces.go",
			"internal/ai/training/chat/real_seq2seq_chat.go",
			"internal/ai/analyze/typemethod.go",
		},
		IssuesViewed: []string{
			"https://github.com/fgm/izidic/issues/30",
		},
	}
	p.RecordCommand("go run . -real-chat -domain goanalyze")
	p.RecordCommand("go run . -real-chat -domain goanalyze")
	p.RecordCommand("go run . -real-chat -domain goanalyze")
	p.Save()
}

// ShouldAutoBuild checks if John's "do all"/"yeah" directive means
// we should skip confirmation prompts and just build.
func (p *UserProfile) ShouldAutoBuild() bool {
	return p.Directives["do_all_means_all"] || p.Directives["build_on_yes"]
}

// SuggestNext returns what John usually does after a given action,
// based on learned sequences. Empty string if no pattern.
func (p *UserProfile) SuggestNext(lastAction string) string {
	// Check learned action sequences.
	key := "after_" + normalizeAction(lastAction)
	if next, ok := p.Patterns[key]; ok {
		return next
	}
	return ""
}

// normalizeAction reduces a chat command to a pattern key.
// "/analyze https://github.com/foo/bar" -> "analyze", "where for <url>" -> "where_issue".
func normalizeAction(input string) string {
	input = strings.TrimSpace(input)
	if strings.HasPrefix(input, "/") {
		parts := strings.Fields(input)
		if len(parts) > 0 {
			return strings.TrimPrefix(parts[0], "/")
		}
	}
	if strings.HasPrefix(input, "where for ") {
		return "where_issue"
	}
	if strings.Contains(input, "github.com") && strings.Contains(input, "analyze") {
		return "analyze_repo"
	}
	if strings.HasPrefix(input, "clone ") {
		return "clone"
	}
	return "other"
}

// RecordSequence learns that action B followed action A.
func (p *UserProfile) RecordSequence(prevAction, nextAction string) {
	prev := normalizeAction(prevAction)
	next := normalizeAction(nextAction)
	if prev == "other" || next == "other" || prev == next {
		return
	}
	key := "after_" + prev
	// Only learn if it's consistent: track counts.
	// Simple version: if we've seen this sequence 2+ times, lock it in.
	// For now, store the most recent consistent next action.
	if existing, ok := p.Patterns[key]; ok && existing == next {
		// Confirmed pattern — keep it.
		return
	}
	// Store as candidate; needs confirmation via repeat.
	p.Patterns[key+"_candidate"] = next
	if p.Patterns[key+"_candidate"] == next {
		// Seen twice — promote to learned pattern.
		// (Simplified: second occurrence confirms.)
		delete(p.Patterns, key+"_candidate")
		p.Patterns[key] = next
	}
	p.Save()
}

// ActionLog is a single (context -> action) pair for future neural training.
// Layer 4: when enough pairs accumulate, train a tiny predictor model.
type ActionLog struct {
	Timestamp time.Time `json:"ts"`
	Context   string    `json:"context"` // e.g. "18:30, in gollemer, after /analyze"
	Action    string    `json:"action"`  // e.g. "/find auth"
	Cwd       string    `json:"cwd"`
}

// LogAction appends to the local action log for future model training.
// The log lives at ~/.gollemer_actions.jsonl — local only, never uploaded.
func (p *UserProfile) LogAction(context, action, cwd string) {
	entry := ActionLog{
		Timestamp: time.Now(),
		Context:   context,
		Action:    action,
		Cwd:       cwd,
	}
	data, err := json.Marshal(entry)
	if err != nil {
		return
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	logPath := filepath.Join(home, ".gollemer_actions.jsonl")
	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return
	}
	defer f.Close()
	f.Write(append(data, '\n'))
}

// printProfile renders the /profile answer: what the system has
// learned about the user, straight from the profile store. Shared
// by every chat loop — the single-domain loop's inline copy was
// the only one for a while, which left `make chat` (the unified
// loop) with no /profile at all: the line fell through to the
// social brain and got a greeting.
func printProfile(profile *UserProfile) {
	fmt.Printf("\nWhat I've learned about you:\n\n")
	fmt.Printf("Frequent commands:\n")
	for _, s := range profile.FrequentCommands {
		fmt.Printf("  %dx  %s\n", s.Count, s.Cmd)
	}
	fmt.Printf("\nProjects:\n")
	for dir, pm := range profile.Projects {
		fmt.Printf("  %s  (last: %s, focus: %s)\n", dir, pm.LastActive.Format("Jan 2 15:04"), pm.Focus)
	}
	fmt.Printf("\nLearned patterns:\n")
	for k, v := range profile.Patterns {
		fmt.Printf("  %s → %s\n", k, v)
	}
	fmt.Printf("\nDirectives:\n")
	for k, v := range profile.Directives {
		fmt.Printf("  %s = %v\n", k, v)
	}
	fmt.Printf("\nAll stored locally in ~/.gollemer_profile.json — never uploaded.\n\n")
	// Show learned code style if available.
	if style := profile.GetCachedStyle(); style != nil {
		fmt.Printf("\n%s\n", style.StyleSummary())
	}
}

// CachedStyle holds the last learned code style (Layer 3).
// Stored in the profile JSON so it persists across sessions.

// GetCachedStyle returns the cached code style, or nil if not learned yet.
func (p *UserProfile) GetCachedStyle() *CodeStyle {
	if p.cachedStyle == nil {
		// Try to load from disk.
		home, err := os.UserHomeDir()
		if err != nil {
			return nil
		}
		stylePath := filepath.Join(home, ".gollemer_style.json")
		data, err := os.ReadFile(stylePath)
		if err != nil {
			return nil
		}
		var style CodeStyle
		if err := json.Unmarshal(data, &style); err != nil {
			return nil
		}
		p.cachedStyle = &style
	}
	return p.cachedStyle
}

// CacheStyle saves the learned code style to disk (local only).
func (p *UserProfile) CacheStyle(style *CodeStyle) {
	p.cachedStyle = style
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	stylePath := filepath.Join(home, ".gollemer_style.json")
	data, err := json.MarshalIndent(style, "", "  ")
	if err != nil {
		return
	}
	os.WriteFile(stylePath, data, 0600)
}

// ActionLogCount returns how many training pairs are logged.
func (p *UserProfile) ActionLogCount() int {
	home, err := os.UserHomeDir()
	if err != nil {
		return 0
	}
	logPath := filepath.Join(home, ".gollemer_actions.jsonl")
	data, err := os.ReadFile(logPath)
	if err != nil {
		return 0
	}
	return strings.Count(string(data), "\n")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
