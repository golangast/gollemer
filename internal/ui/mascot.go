package ui

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Expression string

const (
	Neutral   Expression = "◔ϖ◔" // Idle / Default
	Thinking  Expression = "◕‿◕" // Processing / Scaffolding
	Happy     Expression = "ᵔᴥᵔ" // Success / Build Passed
	Shocked   Expression = "•̀ o •́" // Validation Error / Missing Input
	Fixing    Expression = "🛠️"   // Auto-repairing code
	Disturbed Expression = "°ϖ°" // Build Panic / Error
	Alert     Expression = "•̀ϖ•́"
	Think     Expression = "•ϖ•"
)

// Legacy Mood aliases for backward compatibility if needed in RunLLM's existing code
const (
	MoodIdle    = "◔ϖ◔"
	MoodHappy   = "◕ϖ◕"
	MoodThink   = "•ϖ•"
	MoodAlert   = "•̀ϖ•́"
	MoodWaiting = "°ϖ°"
)

type Activity struct {
	Path      string
	Timestamp time.Time
	Delta     int64
}

type MoodManager struct {
	History     []Activity
	LastSayTime time.Time
}

type Mascot struct {
	Name  string
	Color string
	Mood  *MoodManager
}

const (
	ColorCyan  = "\033[36m"
	ColorReset = "\033[0m"
)

func NewMascot() *Mascot {
	return &Mascot{
		Name:  "Gollemer", 
		Color: ColorCyan,
		Mood:  &MoodManager{History: make([]Activity, 0), LastSayTime: time.Now()},
	}
}

// Speak handles the typewriter effect
func (m *Mascot) Speak(exp any, message string) {
	face := m.toFace(exp)
	fmt.Printf("%s/ʕ%sʔ/ > %s", m.Color, face, ColorReset)
	for _, char := range message {
		fmt.Printf("%c", char)
		os.Stdout.Sync()
		time.Sleep(20 * time.Millisecond)
		if char == '.' || char == '!' || char == '?' {
			time.Sleep(150 * time.Millisecond)
		}
	}
	fmt.Println()
}

// ShowMascot immediately prints the mascot.
func (m *Mascot) ShowMascot(exp any) {
	face := m.toFace(exp)
	fmt.Printf("%s/ʕ%sʔ/ > %s", m.Color, face, ColorReset)
}

// Say is a helper that wraps Speak for convenient usage
func (m *Mascot) Say(exp Expression, message string) {
	m.Speak(exp, message)
}

// DrawHUD prints a visual header for the tutorial
func (m *Mascot) DrawHUD(step int, total int) {
	fmt.Printf("\n%s[ Tutorial Step %d/%d ] [ Mood: %s ]%s\n", m.Color, step, total, Neutral, ColorReset)
	fmt.Println("-------------------------------------------")
}

// toFace converts various input types to a face string
func (m *Mascot) toFace(exp any) string {
	switch v := exp.(type) {
	case Expression:
		return string(v)
	case string:
		return v
	default:
		return string(Neutral)
	}
}

// --- Background Awareness and Reactions ---

func (m *Mascot) ReactToFileChange(path string, status string) {
	filename := filepath.Base(path)
	// Don't react more than once per minute for background noise to avoid spamming
	if time.Since(m.Mood.LastSayTime) < 60*time.Second {
		return
	}

	m.Mood.LastSayTime = time.Now()

	switch {
	case strings.Contains(filename, "main.go"):
		m.Say(Thinking, "Refining the entry point? I'll be ready to 'run' when you are.")
	case strings.Contains(filename, "model"):
		m.Say(Shocked, "Adjusting the weights? Let's hope those gradients don't explode!")
	case strings.Contains(filename, "simd"):
		m.Say(Fixing, "Optimizing the primitives... I love a fast backend.")
	case strings.Contains(filename, "tag"):
		m.Say(Thinking, "Fixing the tagger logic? I always appreciate a more precise eye.")
	default:
		m.Say(Neutral, fmt.Sprintf("I noticed a pulse in %s.", filename))
	}
}

func (m *Mascot) RecordActivity(path string, delta int64) {
	m.Mood.History = append(m.Mood.History, Activity{
		Path:      path,
		Timestamp: time.Now(),
		Delta:     delta,
	})
	
	count := m.GetVelocity()
	if count == 20 {
		m.Say(Happy, "20 changes this hour! You're in the zone, Zachary.")
	} else if count == 50 {
		m.Say(Happy, "50 changes! You're moving at light speed! 🏔️⚡")
	}
}

func (m *Mascot) GetVelocity() int {
	count := 0
	hourAgo := time.Now().Add(-1 * time.Hour)
	for _, a := range m.Mood.History {
		if a.Timestamp.After(hourAgo) {
			count++
		}
	}
	return count
}

func (m *Mascot) SuggestCommit() string {
	if len(m.Mood.History) == 0 {
		return "Initial pulse."
	}

	counts := make(map[string]int)
	for _, a := range m.Mood.History {
		counts[filepath.Base(a.Path)]++
	}

	var bestFile string
	max := 0
	for f, c := range counts {
		if c > max {
			max = c
			bestFile = f
		}
	}

	switch {
	case bestFile == "main.go":
		return "Refactor entry point and orchestration"
	case bestFile == "simd.go" || bestFile == "primitives.go":
		return "Optimize SIMD vector operations for performance"
	case bestFile == "adamw.go" || bestFile == "optimizer.go":
		return "Tune optimizer and gradient scaling"
	case bestFile == "model.go" || bestFile == "moe.go":
		return "Update MoE gating logic and expert weights"
	case strings.HasSuffix(bestFile, ".html") || strings.HasSuffix(bestFile, ".go"):
		return fmt.Sprintf("Iterative updates to %s", bestFile)
	default:
		return "Update files and refine system logic"
	}
}
