// Package chat — user profile: learns John's patterns, projects, and habits.
// All data stays local in ~/.gollemer_profile.json. Never uploaded, never
// shared. This is the "learning you" model — deterministic preference memory,
// not a neural network.
package chat

import (
	"encoding/json"
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
}

// CommandStat tracks a single command's usage.
type CommandStat struct {
	Cmd      string    `json:"cmd"`
	Count    int       `json:"count"`
	LastUsed time.Time `json:"last_used"`
}

// ProjectMemory remembers what John was doing in a project directory.
type ProjectMemory struct {
	Name         string   `json:"name"`
	LastActive   time.Time `json:"last_active"`
	Focus        string   `json:"focus"`         // what he was working on
	FilesTouched []string `json:"files_touched"` // recent files
	IssuesViewed []string `json:"issues_viewed"` // GitHub issues investigated
	ReposCloned  []string `json:"repos_cloned"`  // repos analyzed
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

	// Most frequent command suggestion.
	if len(p.FrequentCommands) > 0 {
		best := p.FrequentCommands[0]
		for _, s := range p.FrequentCommands[1:] {
			if s.Count > best.Count {
				best = s
			}
		}
		if best.Count >= 3 {
			b.WriteString("Your most-run command: " + best.Cmd + "\n")
		}
	}

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
	p.RecordCommand("go run main.go -real-chat -domain goanalyze")
	p.RecordCommand("go run main.go -real-chat -domain goanalyze")
	p.RecordCommand("go run main.go -real-chat -domain goanalyze")
	p.Save()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
