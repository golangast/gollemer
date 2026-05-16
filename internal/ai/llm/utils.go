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
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
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

	objectTypeKeywords := map[string]bool{
		"handler": true, "webserver": true, "page": true, "file": true,
		"folder": true, "directory": true, "database": true, "structure": true, "component": true,
	}

	// Context-aware fallback: If there is exactly one word and it's not a known command/object/stopword,
	// it's almost certainly the 'name' or target we were asking for.
	if len(taggedData.Tokens) == 1 {
		t := strings.ToLower(taggedData.Tokens[0])
		if !objectTypeKeywords[t] && (kb == nil || (!kb.KnownObjects[t] && !kb.KnownCommands[t] && !kb.StopWords[t])) {
			return taggedData.Tokens[0]
		}
	}

	// Final heuristic fallback: first non-keyword after a known Object Type token
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
			if (r == '\'' || r == '_') && currentToken.Len() > 0 {
				currentToken.WriteRune(r)
			} else if r == '_' {
				// Allow starting underscores for special tokens like __intent__
				currentToken.WriteRune(r)
			} else {
				if currentToken.Len() > 0 {
					tokens = append(tokens, strings.ToLower(currentToken.String()))
					currentToken.Reset()
				}
				// Skip non-ASCII symbols to keep the model context clean
				if r < 128 {
					tokens = append(tokens, string(r))
				}
			}
		} else {
			if r < 128 || unicode.IsLetter(r) { // Allow UTF-8 letters but keep symbols clean
				currentToken.WriteRune(r)
			}
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

// isGarbageOutput returns true when a neural decoder response looks like the
// low-confidence token soup an under-trained seq2seq model produces
// (e.g. ". to . type to . deglaze to the to . boolean").
// Rules:
//  1. First token is a punctuation-only string (`.`, `-`,`,` …)
//  2. More than 40% of tokens are stopwords or single-char punctuation
//  3. Average word length falls below 2.5 characters
func isGarbageOutput(response string) bool {
	if response == "" {
		return false
	}
	tokens := strings.Fields(response)
	if len(tokens) == 0 {
		return false
	}

	// Rule 1: leading punctuation - only reject if the WHOLE response is just punctuation
	isAllPunct := true
	for _, t := range tokens {
		if len(t) > 1 || unicode.IsLetter(rune(t[0])) || unicode.IsDigit(rune(t[0])) {
			isAllPunct = false
			break
		}
	}
	if isAllPunct {
		return true
	}

	garbageSet := map[string]bool{
		".": true, "-": true, ",": true, "to": true, "the": true,
		"a": true, "an": true, "of": true, "and": true, "is": true,
		"it": true, "i": true, "you": true, "that": true, "be": true,
		"in": true, "on": true, "at": true, "for": true, "with": true,
		"?": true, "!": true, ":": true, ";": true, "hi": true,
	}

	garbageCount := 0
	totalLen := 0
	for _, t := range tokens {
		totalLen += len(t)
		if garbageSet[strings.ToLower(t)] {
			garbageCount++
		}
	}

	// Rule 2: >60% garbage tokens (tighten from 70%)
	if float64(garbageCount)/float64(len(tokens)) > 0.60 {
		return true
	}

	// Rule 3: very short average word length
	avgLen := float64(totalLen) / float64(len(tokens))
	if avgLen < 1.8 { // Tighten from 1.5
		return true
	}

	return false
}

// isLowQualitySocialResponse catches high-entropy "bag of words" outputs that
// can pass isGarbageOutput but are still unusable in conversation.
func isLowQualitySocialResponse(response string) bool {
	if response == "" {
		return false
	}

	rawTokens := strings.Fields(strings.ToLower(response))
	if len(rawTokens) == 0 {
		return false
	}

	pronouns := map[string]bool{
		"i": true, "i'm": true, "im": true, "me": true, "my": true, "mine": true,
		"you": true, "your": true, "yours": true, "we": true, "our": true, "ours": true,
		"it": true, "its": true, "they": true, "them": true, "their": true,
	}
	functionWords := map[string]bool{
		"a": true, "an": true, "the": true, "and": true, "or": true, "but": true,
		"to": true, "of": true, "in": true, "on": true, "for": true, "with": true,
		"is": true, "are": true, "am": true, "was": true, "were": true, "be": true, "been": true, "being": true,
		"that": true, "this": true, "these": true, "those": true, "as": true, "at": true,
		"what": true, "who": true, "where": true, "when": true, "why": true, "how": true,
		"can": true, "could": true, "should": true, "would": true, "will": true, "shall": true,
		"do": true, "does": true, "did": true, "done": true, "doing": true,
	}

	trimmedTokens := make([]string, 0, len(rawTokens))
	unique := make(map[string]bool)
	pronounCount := 0
	functionCount := 0

	for _, t := range rawTokens {
		tt := strings.Trim(t, ".,!?;:\"'`()[]{}")
		if tt == "" {
			continue
		}
		trimmedTokens = append(trimmedTokens, tt)
		unique[tt] = true
		if pronouns[tt] {
			pronounCount++
		}
		if functionWords[tt] {
			functionCount++
		}
	}

	tokenCount := len(trimmedTokens)

	// Check for repetitive tokens (stuck decoder)
	for i := 0; i < tokenCount-1; i++ {
		if trimmedTokens[i] == trimmedTokens[i+1] {
			// Double repetition of common tokens or any triple repetition
			if tokenCount > 10 || i+2 < tokenCount && trimmedTokens[i+1] == trimmedTokens[i+2] {
				return true
			}
		}
	}
	uniqueRatio := float64(len(unique)) / float64(tokenCount)
	functionRatio := float64(functionCount) / float64(tokenCount)
	punctuationCount := strings.Count(response, ".") + strings.Count(response, "!") + strings.Count(response, "?")

	// High repetition ratio check
	if uniqueRatio < 0.6 && tokenCount > 8 {
		return true
	}

	// Long, mostly-unique outputs with little linguistic glue are typically token soup.
	if tokenCount >= 12 && uniqueRatio > 0.85 && functionRatio < 0.22 {
		return true
	}

	// Long social responses should usually contain at least one pronoun and some sentence punctuation.
	if tokenCount >= 15 && pronounCount == 0 && punctuationCount == 0 {
		return true
	}

	return false
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

// lookupVocab tries to find a token ID in the given vocabulary with fallbacks.
func lookupVocab(token string, vocab *mainvocab.Vocabulary) int {
	token = strings.ToLower(strings.TrimSpace(token))
	id := vocab.GetTokenID(token)
	
	// If it's a real token (not UNK and not PAD unless specifically requested)
	if id > 1 || (id == 0 && token == "<pad>") {
		return id
	}

	// Try stripping punctuation
	stripped := strings.Trim(token, ".,!?;:'\"")
	if stripped != "" && stripped != token {
		sid := vocab.GetTokenID(stripped)
		if sid > 1 {
			return sid
		}
	}

	// Last resort: return UNK
	if id == 1 && len(token) > 3 {
		// Log rare words that trigger UNK for debugging
		// log.Printf("🔍 [Vocab] Token '%s' mapped to UNK", token)
	}
	return id
}


func goImports(path string) {
	cmd := exec.Command("go", "run", "golang.org/x/tools/cmd/goimports@latest", "-w", path)
	if output, err := cmd.CombinedOutput(); err != nil {
		log.Printf("⚠️  goimports failed on %s: %v\n%s", path, err, string(output))
	}
}
