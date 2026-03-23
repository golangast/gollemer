package llm

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/tagger/tag"
)

var absoluteLastDirConfigPath string // Global variable for the absolute path to last_dir.txt

// contains is a helper function to check if a string is in a slice of strings.
func contains(s []string, e string) bool {
	return slices.Contains(s, e)
}

func FindProjectRoot() (string, error) {
	currentDir, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to get current working directory: %v", err)
	}

	for {
		goModPath := filepath.Join(currentDir, "go.mod")
		if _, err := os.Stat(goModPath); err == nil {
			return currentDir, nil // Found go.mod, this is the project root
		}

		parentDir := filepath.Dir(currentDir)
		if parentDir == currentDir {
			// Reached the filesystem root without finding go.mod
			return "", fmt.Errorf("go.mod not found in current directory or any parent directories")
		}
	        currentDir = parentDir
	}
}

func findGoModInfo() (modulePath string, projectRoot string, err error) {
	currentDir, err := os.Getwd()
	if err != nil {
		return "", "", fmt.Errorf("failed to get current working directory: %v", err)
	}

	dir := currentDir
	for {
		goModPath := filepath.Join(dir, "go.mod")
		if _, statErr := os.Stat(goModPath); statErr == nil {
			// Found go.mod
			content, readErr := os.ReadFile(goModPath)
			if readErr != nil {
				return "", "", fmt.Errorf("failed to read go.mod file: %v", readErr)
			}
			data := string(content)
			lines := strings.Split(data, "\n")
			for _, line := range lines {
				if after, ok := strings.CutPrefix(line, "module "); ok {
					return strings.TrimSpace(after), dir, nil
				}
			}
			return "", "", fmt.Errorf("module path not found in go.mod")
		}

		parentDir := filepath.Dir(dir)
		if parentDir == dir {
			return "", "", fmt.Errorf("go.mod not found in any parent directory")
		}
		dir = parentDir
	}
}

func buildWasm(wasmDir string) {
	if wasmDir == "" {
		wasmDir = "."
	}
	if _, err := os.Stat(wasmDir); os.IsNotExist(err) {
		return
	}

	// 1. Ensure wasm_exec.js exists
	goroot, err := exec.Command("go", "env", "GOROOT").Output()
	if err == nil {
		gorootPath := strings.TrimSpace(string(goroot))
		// Try multiple locations for wasm_exec.js
		srcs := []string{
			filepath.Join(gorootPath, "misc", "wasm", "wasm_exec.js"),
			filepath.Join(gorootPath, "lib", "wasm", "wasm_exec.js"),
		}

		var src string
		for _, s := range srcs {
			if _, err := os.Stat(s); err == nil {
				src = s
				break
			}
		}

		if src != "" {
			dst := filepath.Join(wasmDir, "wasm_exec.js")
			content, err := os.ReadFile(src)
			if err == nil {
				err = os.WriteFile(dst, content, 0644)
				if err == nil {
					fmt.Printf("✅ Copied wasm_exec.js to %s\n", wasmDir)
				} else {
					fmt.Printf("⚠️  Failed to write wasm_exec.js to %s: %v\n", wasmDir, err)
				}
			} else {
				fmt.Printf("⚠️  Failed to read wasm_exec.js from %s: %v\n", src, err)
			}
		} else {
			fmt.Printf("⚠️  Could not find wasm_exec.js in GOROOT (%s)\n", gorootPath)
		}
	}

	fmt.Printf("🏗️  Building WASM in %s...\n", wasmDir)
	// Check for wasm.go or main.go, or in a wasm/ subdirectory
	wasmFile := ""
	candidates := []string{
		"wasm.go",
		"main.go",
		filepath.Join("wasm", "main.go"),
		filepath.Join("wasm", "wasm.go"),
	}

	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(wasmDir, c)); err == nil {
			wasmFile = c
			break
		}
	}

	if wasmFile == "" {
		fmt.Printf("⚠️  No wasm file found in %s (checked %v), skipping.\n", wasmDir, candidates)
		return
	}

	// Determine if we should use -mod=mod
	args := []string{"build"}
	gowork, _ := exec.Command("go", "env", "GOWORK").Output()
	if strings.TrimSpace(string(gowork)) == "" || strings.TrimSpace(string(gowork)) == "off" {
		args = append(args, "-mod=mod")
	}
	args = append(args, "-o", "main.wasm", wasmFile)

	cmd := exec.Command("go", args...)
	cmd.Dir = wasmDir
	cmd.Env = append(os.Environ(), "GOOS=js", "GOARCH=wasm")
	output, err := cmd.CombinedOutput()
	if err != nil {
		fmt.Printf("❌ WASM build failed in %s: %v\n%s\n", wasmDir, err, string(output))
	} else {
		fmt.Printf("✅ WASM build successful: %s/main.wasm updated.\n", wasmDir)
	}
}

func findName(taggedData tag.Tag, kb *KnowledgeBase) string {
	// First, look for a FILENAME tag
	for i, tag := range taggedData.NerTag {
		if tag == "FILENAME" {
			return taggedData.Tokens[i]
		}
	}

	// Fallback for "named"
	for i, token := range taggedData.Tokens {
		if (token == "named" || token == "called") && i+1 < len(taggedData.Tokens) {
			return taggedData.Tokens[i+1]
		}
	}

	// Fallback for NAME tag
	for i, tag := range taggedData.NerTag {
		if tag == "NAME" {
			return taggedData.Tokens[i]
		}
	}

	// Final heuristic fallback: first non-keyword after a known Object Type token
	objectTypeKeywords := map[string]bool{
		"handler": true, "webserver": true, "page": true, "file": true,
		"folder": true, "database": true, "structure": true, "component": true,
	}
	for i, token := range taggedData.Tokens {
		lower := strings.ToLower(token)
		if (objectTypeKeywords[lower] || (kb != nil && kb.KnownObjects[lower])) && i+1 < len(taggedData.Tokens) {
			// Skip noise words to find the actual name
			j := i + 1
			for j < len(taggedData.Tokens) {
				candidate := taggedData.Tokens[j]
				lowerC := strings.ToLower(candidate)
				if lowerC == "named" || lowerC == "called" || lowerC == "the" || lowerC == "a" || lowerC == "an" || lowerC == "with" {
					j++
					continue
				}
				return candidate
			}
		}
	}

	return ""
}

func saveLastDirectory(dirPath string) {
	err := os.WriteFile(absoluteLastDirConfigPath, []byte(dirPath), 0644)
	if err != nil {
		log.Printf("Error saving last directory to %s: %v", absoluteLastDirConfigPath, err)
	}
}

func loadLastDirectory() (string, error) {
	content, err := os.ReadFile(absoluteLastDirConfigPath)
	if err != nil {
		return "", fmt.Errorf("error reading last directory from %s: %v", absoluteLastDirConfigPath, err)
	}
	return strings.TrimSpace(string(content)), nil
}

// findClosestObject uses Levenshtein distance to find the nearest known object.
func findClosestObject(target string, known map[string]bool) (string, int) {
	closest := ""
	minDist := 999

	for obj := range known {
		dist := levenshteinDistance(target, obj)
		if dist < minDist {
			minDist = dist
			closest = obj
		}
	}
	return closest, minDist
}

func levenshteinDistance(s1, s2 string) int {
	s1Raw := []rune(s1)
	s2Raw := []rune(s2)
	len1 := len(s1Raw)
	len2 := len(s2Raw)

	column := make([]int, len1+1)
	for y := 1; y <= len1; y++ {
		column[y] = y
	}

	for x := 1; x <= len2; x++ {
		column[0] = x
		lastkey := x - 1
		for y := 1; y <= len1; y++ {
			oldkey := column[y]
			var incr int
			if s1Raw[y-1] != s2Raw[x-1] {
				incr = 1
			}

			column[y] = min(column[y]+1, min(column[0]+1, lastkey+incr))
			lastkey = oldkey
		}
	}
	return column[len1]
}

// cleanTokenize splits text into tokens, separating punctuation.
func cleanTokenize(text string) []string {
	var tokens []string
	var currentToken strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if currentToken.Len() > 0 {
				tokens = append(tokens, strings.ToLower(currentToken.String()))
				currentToken.Reset()
			}
		} else if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			if r == '\'' && currentToken.Len() > 0 {
				currentToken.WriteRune(r)
			} else {
				if currentToken.Len() > 0 {
					tokens = append(tokens, strings.ToLower(currentToken.String()))
					currentToken.Reset()
				}
				tokens = append(tokens, string(r))
			}
		} else {
			currentToken.WriteRune(r)
		}
	}
	if currentToken.Len() > 0 {
		tokens = append(tokens, strings.ToLower(currentToken.String()))
	}
	return tokens
}

func detectWebserverName(projectRoot string) string {
	cwd, _ := os.Getwd()
	// 1. Check CWD
	if _, err := os.Stat("main.go"); err == nil {
		content, _ := os.ReadFile("main.go")
		if strings.Contains(string(content), "net/http") {
			return filepath.Base(cwd)
		}
	}
	// 2. Check cmd/
	cmdDir := filepath.Join(projectRoot, "cmd")
	entries, _ := os.ReadDir(cmdDir)
	var servers []string
	for _, e := range entries {
		if e.IsDir() {
			if _, err := os.Stat(filepath.Join(cmdDir, e.Name(), "main.go")); err == nil {
				servers = append(servers, e.Name())
			}
		}
	}
	if len(servers) == 1 {
		return servers[0]
	}
	// 3. Check project root if it has main.go
	if _, err := os.Stat(filepath.Join(projectRoot, "main.go")); err == nil {
		content, _ := os.ReadFile(filepath.Join(projectRoot, "main.go"))
		if strings.Contains(string(content), "net/http") {
			return filepath.Base(projectRoot)
		}
	}
	return ""
}

// intentIcons maps intents to their visual representation.
var intentIcons = map[string]string{
	"create_webserver": "🌐 [Webserver]",
	"create_handler":   "🔌 [Handler]",
	"create_database":  "🗄️  [Database]",
	"create_page":      "📄 [Page]",
	"create_file":      "📝 [File]",
	"create_folder":    "📁 [Folder]",
	"create_structure": "🏗️  [Structure]",
	"move_file":        "🚚 [Move]",
	"create_object":    "🔨 [Object]",
	"stop":             "🛑 [Stop]",
	"run_webserver":    "🚀 [Run]",
	"watch":            "👁️  [Watch]",
}

func isCreatingCommand(input string) bool {
	l := strings.ToLower(input)
	keywords := []string{"create", "add", "new", "make", "generate", "setup", "init"}
	for _, k := range keywords {
		if strings.Contains(l, k) {
			return true
		}
	}
	return false
}
