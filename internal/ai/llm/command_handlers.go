package llm

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/platform/sqlite_db"
)

func (r *Runner) handleGoCommand(targetDirectory string) string {
	if targetDirectory == "" {
		return ""
	}
	if targetDirectory == "root" {
		targetDirectory = "/"
	}
	err := os.Chdir(targetDirectory)
	if err != nil {
		return fmt.Sprintf("I couldn't change the directory to %s: %v", targetDirectory, err)
	}
	predictedSentence := fmt.Sprintf("Changed directory to %s.", targetDirectory)
	currentAbsDir, err := os.Getwd()
	if err == nil {
		saveLastDirectory(currentAbsDir)
	}
	return predictedSentence
}

func (r *Runner) handleMoveCommand(fileName, targetDirectory string) string {
	if fileName == "" {
		return "Please specify a file to move."
	}
	if targetDirectory == "" {
		return "Please specify a destination directory."
	}
	if _, err := os.Stat(targetDirectory); os.IsNotExist(err) {
		return fmt.Sprintf("Destination directory '%s' does not exist.", targetDirectory)
	}
	destFile := filepath.Join(targetDirectory, filepath.Base(fileName))
	err := os.Rename(fileName, destFile)
	if err != nil {
		return fmt.Sprintf("I couldn't move the file '%s' to '%s': %v", fileName, targetDirectory, err)
	}
	predictedSentence := fmt.Sprintf("I have moved the file '%s' to '%s'.", fileName, targetDirectory)
	if tree, err := generateDirectoryTree(".", "", 0, 2, destFile); err == nil {
		predictedSentence += "\n\n" + tree
	}
	return predictedSentence
}

func (r *Runner) handleListCommand(objectType string, objectTypeParts []string, targetDirectory string) string {
	if strings.Contains(objectType, "handler") {
		return r.listHandlers(targetDirectory)
	}
	target := "."
	if targetDirectory != "" {
		target = targetDirectory
	}
	files, err := os.ReadDir(target)
	if err != nil {
		return fmt.Sprintf("I couldn't list the contents of %s: %v", target, err)
	}
	var items []string
	showFiles := contains(objectTypeParts, "file") || contains(objectTypeParts, "files")
	showFolders := contains(objectTypeParts, "folder") || contains(objectTypeParts, "folders")
	for _, file := range files {
		isDir := file.IsDir()
		if (!showFiles && !showFolders) || (showFiles && showFolders) || (showFiles && !isDir) || (showFolders && isDir) {
			items = append(items, file.Name())
		}
	}
	return "Here are the contents of the directory:\n" + strings.Join(items, "\n")
}

func (r *Runner) listHandlers(targetDirectory string) string {
	targetPath := "main.go"
	if targetDirectory != "" {
		if _, err := os.Stat(filepath.Join(targetDirectory, "main.go")); err == nil {
			targetPath = filepath.Join(targetDirectory, "main.go")
		} else {
			targetPath = filepath.Join(targetDirectory, "cmd", filepath.Base(targetDirectory), "main.go")
		}
	}
	content, err := os.ReadFile(targetPath)
	if err != nil {
		return fmt.Sprintf("I couldn't read %s to list handlers.", targetPath)
	}
	lines := strings.Split(string(content), "\n")
	var handlers []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if after, ok := strings.CutPrefix(trimmed, "http.HandleFunc("); ok {
			args := strings.TrimSuffix(after, ")")
			parts := strings.SplitN(args, ",", 2)
			if len(parts) == 2 {
				path := strings.Trim(strings.TrimSpace(parts[0]), "\"")
				funcName := strings.TrimSpace(parts[1])
				handlers = append(handlers, fmt.Sprintf("%s -> %s", path, funcName))
			}
		}
	}
	if len(handlers) > 0 {
		return fmt.Sprintf("Registered Handlers in %s:\n%s", targetPath, strings.Join(handlers, "\n"))
	}
	return fmt.Sprintf("No handlers found in %s.", targetPath)
}

func (r *Runner) handleGrepCommand(searchTerm, targetDirectory string) string {
	if searchTerm == "" {
		return "Please provide text to search for."
	}
	target := "."
	if targetDirectory != "" {
		target = targetDirectory
	}
	var results []string
	filepath.Walk(target, func(path string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() && !strings.Contains(path, "/.") {
			if content, err := os.ReadFile(path); err == nil && strings.Contains(string(content), searchTerm) {
				lines := strings.Split(string(content), "\n")
				for i, line := range lines {
					if strings.Contains(line, searchTerm) {
						trimmed := strings.TrimSpace(line)
						if len(trimmed) > 80 {
							trimmed = trimmed[:80] + "..."
						}
						results = append(results, fmt.Sprintf("%s:%d: %s", path, i+1, trimmed))
					}
				}
			}
		}
		return nil
	})
	if len(results) == 0 {
		return fmt.Sprintf("No matches found for '%s'.", searchTerm)
	}
	output := strings.Join(results, "\n")
	if len(results) > 20 {
		output = strings.Join(results[:20], "\n") + fmt.Sprintf("\n... and %d more.", len(results)-20)
	}
	return output
}

func (r *Runner) handleHelpCommand(intentData *IntentDataLayer) string {
	var sb strings.Builder
	sb.WriteString("--- ʕ◔ϖ◔ʔ Gollemer Help ---\n\n")
	sb.WriteString("Categorized Commands:\n")
	sb.WriteString("  [Menu]          menu\n")
	sb.WriteString("  [Tutorial]      tutorial\n")
	sb.WriteString("  [📁 Navigation]  go <dir>, list, tree, pwd\n")
	sb.WriteString("  [🛠️  Files]       create file <name>, create folder <name>, delete <name>, move <file> to <dir>, cat, grep\n")
	sb.WriteString("  [🌐 Web]         create webserver <name>, create handler <name>, create page <name>, run, stop, verify\n")
	sb.WriteString("  [🧠 Learning]    learn from <dir>, learn object <word>, show learning path\n")
	sb.WriteString("  [⚙️  System]      history, clear, exit\n\n")
	return sb.String()
}

func (r *Runner) handleTutorialLogic(command, objectType, predictedSentence string) string {
	if !r.TutorialState.Active {
		return predictedSentence
	}
	step := r.TutorialState.Step
	nextStep := step
	msg := ""

	switch step {
	case 1:
		if command == "create" && (strings.Contains(objectType, "folder") || strings.Contains(objectType, "directory")) {
			nextStep = 2
			msg = "\n\n[Tutorial] Great job! You created a folder. Now create a file inside it."
		}
	case 2:
		if command == "create" && strings.Contains(objectType, "file") {
			nextStep = 3
			msg = "\n\n[Tutorial] Excellent! Now create a webserver: 'create webserver myserver'."
		}
	case 3:
		if command == "create" && strings.Contains(objectType, "webserver") {
			nextStep = 4
			msg = "\n\n[Tutorial] Fantastic! Now run it: 'run webserver <name>'."
		}
	case 4:
		if (command == "run" || command == "start") && strings.Contains(objectType, "webserver") {
			if r.Client.WaitForPulse(":8080", 5*time.Second, r.Mascot) {
				r.TutorialState.Active = false
				sqlite_db.SyncStep(r.DB, 5, false)
				return predictedSentence + "\n\n[Tutorial] Awesome! Tutorial completed!"
			}
			return predictedSentence + "\n\n[Tutorial] I don't hear a heartbeat on :8080 yet."
		}
	}

	if nextStep != step {
		r.TutorialState.Step = nextStep
		sqlite_db.SyncStep(r.DB, nextStep, true)
		loc, _ := r.Mascot.CalculateProjectSize(r.ProjectRoot)
		r.Mascot.DrawHUD(nextStep, 4, loc)
		return predictedSentence + msg
	}
	return predictedSentence
}
