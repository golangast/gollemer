package discovery

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type ProjectContext struct {
	IsGollemer bool
	HasData    bool
	HasModel   bool
	LastFile   string
}

type Quest struct {
	File string
	Line int
	Text string
}

type FolderState struct {
	LastSize int64
	LastMod  time.Time
}

func ScanProject() ProjectContext {
	ctx := ProjectContext{}

	// 1. Check for Gollemer-specific directories
	if _, err := os.Stat("internal/moe"); err == nil {
		ctx.IsGollemer = true
	}

	// 2. Check for training data or model weights
	if _, err := os.Stat("trainingdata/conversing.csv"); err == nil {
		ctx.HasData = true
	}
	if _, err := os.Stat("models/weights.bin"); err == nil {
		ctx.HasModel = true
	}

	return ctx
}

func GetExpertAdvice(ctx ProjectContext) string {
	switch {
	case ctx.IsGollemer && !ctx.HasModel:
		return "I see the MoE architecture, but no trained weights! Should we **start a training run**?"
	case ctx.IsGollemer && ctx.HasModel:
		return "Model weights detected. Want to **run an inference test** on a sample sentence?"
	case !ctx.IsGollemer:
		if _, err := os.Stat("go.mod"); err != nil {
			return "I don't see a `go.mod` here. Should we `go mod init` a new project?"
		}
		return "This doesn't look like a Gollemer project yet. Should I **initialize the NLP structure** for you?"
	default:
		return "Ready to code! What's the focus for this session?"
	}
}

func GetDirSize(path string) (int64, error) {
	var size int64
	err := filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size, err
}

func FormatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.2f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

// ScanQuests scans source files for TODO and FIXME patterns.
func ScanQuests() []Quest {
	var quests []Quest
	filepath.Walk(".", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if !info.IsDir() && filepath.Ext(path) == ".go" && !contains(path, "vendor") && !contains(path, ".git") {
			file, err := os.Open(path)
			if err != nil {
				return nil
			}
			defer file.Close()

			scanner := bufio.NewScanner(file)
			lineNum := 1
			for scanner.Scan() {
				line := scanner.Text()
				if strings.Contains(line, "// TODO:") || strings.Contains(line, "// FIXME:") {
					parts := strings.Split(line, "//")
					if len(parts) > 1 {
						cleanText := strings.TrimSpace(parts[1])
						quests = append(quests, Quest{File: path, Line: lineNum, Text: cleanText})
					}
				}
				lineNum++
			}
		}
		return nil
	})
	return quests
}

func contains(path, substr string) bool {
	return (len(path) >= len(substr) && (path[:len(substr)] == substr || path[len(path)-len(substr):] == substr || strings.Contains(path, substr)))
}

// LoadPersonalQuests reads tasks.txt from ~/.gollemer
func LoadPersonalQuests() []string {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}
	path := filepath.Join(home, ".gollemer", "tasks.txt")

	file, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer file.Close()

	var tasks []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		task := strings.TrimSpace(scanner.Text())
		if task != "" && !strings.HasPrefix(task, "#") {
			tasks = append(tasks, task)
		}
	}
	return tasks
}

// GetGitBranch returns the current branch name or "main" fallback.
func GetGitBranch() string {
	data, err := os.ReadFile(".git/HEAD")
	if err != nil {
		return "main"
	}
	content := string(data)
	if strings.Contains(content, "ref: refs/heads/") {
		return strings.TrimSpace(strings.Replace(content, "ref: refs/heads/", "", 1))
	}
	return "main"
}

func QuickCheck(path string, state *FolderState) bool {
	info, err := os.Stat(path)
	if err != nil {
		return false
	}

	// Check if size or modification time has changed
	if info.Size() != state.LastSize || info.ModTime().After(state.LastMod) {
		state.LastSize = info.Size()
		state.LastMod = info.ModTime()
		return true // Something changed!
	}
	return false
}
