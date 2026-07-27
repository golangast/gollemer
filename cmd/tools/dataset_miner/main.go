// DatasetMiner scans target Go repositories and parses Git history into structured
// training triplets (instruction, before_code, target_patch). It uses go/parser to
// filter out any commits where the code before or after fails to parse as valid Go.
//
// Usage:
//
//	go run cmd/tools/dataset_miner/main.go \
//	  -repo="https://github.com/gin-gonic/gin" \
//	  -out="data/training/mined_patches.json" \
//	  -max=10000
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/parser"
	"go/token"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// TrainingTriplet is the core data structure for patch-based training.
type TrainingTriplet struct {
	Instruction string `json:"instruction"`
	BeforeCode  string `json:"before_code"`
	TargetPatch string `json:"target_patch"`
	Repo        string `json:"repo"`
	CommitHash  string `json:"commit_hash"`
	FilePath    string `json:"file_path"`
	ValidGo     bool   `json:"valid_go"`
}

// FIMExample is a Fill-In-The-Middle training example.
type FIMExample struct {
	Prefix string `json:"prefix"` // code before the edit point
	Suffix string `json:"suffix"` // code after the edit point
	Middle string `json:"middle"` // the inserted code
	Repo   string `json:"repo"`
}

func main() {
	repoURL := flag.String("repo", "https://github.com/gin-gonic/gin", "Git repository URL to mine")
	outputPath := flag.String("out", "data/training/mined_patches.json", "Output JSON file path")
	maxCommits := flag.Int("max", 10000, "Maximum number of commits to process")
	cloneDir := flag.String("clone-dir", "/tmp/gollemer_repos", "Directory to clone repos into")
	flag.Parse()

	if err := run(*repoURL, *outputPath, *maxCommits, *cloneDir); err != nil {
		log.Fatalf("Fatal: %v", err)
	}
}

func run(repoURL, outputPath string, maxCommits int, cloneDir string) error {
	// Extract repo name from URL
	repoName := extractRepoName(repoURL)
	repoPath := filepath.Join(cloneDir, repoName)

	// Clone or pull the repository
	if err := ensureRepo(repoURL, repoPath); err != nil {
		return fmt.Errorf("clone repo: %w", err)
	}

	log.Printf("Mining commits from %s (max: %d)...", repoName, maxCommits)

	// Get commit log with diffs
	triplets, fimExamples, err := mineCommits(repoPath, repoName, maxCommits)
	if err != nil {
		return fmt.Errorf("mine commits: %w", err)
	}

	// Save triplets
	if err := saveJSON(outputPath, triplets); err != nil {
		return fmt.Errorf("save triplets: %w", err)
	}
	log.Printf("Saved %d training triplets to %s", len(triplets), outputPath)

	// Save FIM examples
	fimPath := strings.Replace(outputPath, ".json", "_fim.json", 1)
	if err := saveJSON(fimPath, fimExamples); err != nil {
		return fmt.Errorf("save FIM examples: %w", err)
	}
	log.Printf("Saved %d FIM examples to %s", len(fimExamples), fimPath)

	// Print summary
	validCount := 0
	for _, t := range triplets {
		if t.ValidGo {
			validCount++
		}
	}
	log.Printf("Summary: %d total triplets, %d AST-valid, %d FIM examples",
		len(triplets), validCount, len(fimExamples))

	return nil
}

// ensureRepo clones the repo if not present, or pulls latest changes.
func ensureRepo(repoURL, repoPath string) error {
	if _, err := os.Stat(repoPath); os.IsNotExist(err) {
		log.Printf("Cloning %s into %s...", repoURL, repoPath)
		cmd := exec.Command("git", "clone", repoURL, repoPath)
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("git clone: %w", err)
		}
	} else {
		log.Printf("Pulling latest in %s...", repoPath)
		cmd := exec.Command("git", "-C", repoPath, "pull", "--ff-only")
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Printf("Warning: git pull failed (continuing with existing): %v", err)
		}
	}
	return nil
}

// mineCommits extracts structured training triplets from git history.
func mineCommits(repoPath, repoName string, maxCommits int) ([]TrainingTriplet, []FIMExample, error) {
	// Get commit log with patches for .go files
	cmd := exec.Command("git", "-C", repoPath, "log",
		"--diff-filter=M", // Only modifications
		"-p",              // With patch text
		"--no-merges",     // Skip merge commits
		"--max-count="+fmt.Sprintf("%d", maxCommits),
		"--", "*.go", // Only Go files
	)
	output, err := cmd.Output()
	if err != nil {
		return nil, nil, fmt.Errorf("git log: %w", err)
	}

	// Parse the output into structured triplets
	triplets, fimExamples := parseGitLog(string(output), repoName)

	return triplets, fimExamples, nil
}

// parseGitLog processes raw git log -p output into structured training data.
func parseGitLog(logOutput string, repoName string) ([]TrainingTriplet, []FIMExample) {
	var triplets []TrainingTriplet
	var fimExamples []FIMExample

	// Split by commit boundary
	commitRegex := regexp.MustCompile(`(?m)^commit ([a-f0-9]{40})\n`)
	commitMatches := commitRegex.FindAllStringSubmatchIndex(logOutput, -1)

	for i, match := range commitMatches {
		start := match[0]
		var end int
		if i+1 < len(commitMatches) {
			end = commitMatches[i+1][0]
		} else {
			end = len(logOutput)
		}

		commitBlock := logOutput[start:end]
		commitHash := logOutput[match[2]:match[3]]

		// Extract commit message (first line = instruction)
		instruction := extractInstruction(commitBlock)

		// Parse individual file diffs
		fileDiffs := parseFileDiffs(commitBlock)
		for _, fd := range fileDiffs {
			if fd.before == "" && fd.after == "" {
				continue
			}

			// Validate with go/parser
			validBefore := isValidGo(fd.before)
			validAfter := isValidGo(fd.after)
			valid := validBefore && validAfter

			// Build SEARCH/REPLACE patch format
			patch := buildSearchReplacePatch(fd.before, fd.after)

			triplet := TrainingTriplet{
				Instruction: instruction,
				BeforeCode:  fd.before,
				TargetPatch: patch,
				Repo:        repoName,
				CommitHash:  commitHash,
				FilePath:    fd.filePath,
				ValidGo:     valid,
			}
			triplets = append(triplets, triplet)

			// Generate FIM example if both sides are valid Go
			if valid && fd.before != "" && fd.after != "" {
				fim := buildFIMExample(fd.before, fd.after, repoName)
				if fim != nil {
					fimExamples = append(fimExamples, *fim)
				}
			}
		}
	}

	return triplets, fimExamples
}

// fileDiff holds a parsed file-level diff.
type fileDiff struct {
	filePath string
	before   string
	after    string
}

// parseFileDiffs extracts before/after code for each file in a commit.
func parseFileDiffs(commitBlock string) []fileDiff {
	var diffs []fileDiff

	// Split by diff --git
	diffRegex := regexp.MustCompile(`(?m)^diff --git a/(.+?) b/(.+?)$`)
	diffMatches := diffRegex.FindAllStringSubmatchIndex(commitBlock, -1)

	for i, dm := range diffMatches {
		start := dm[0]
		var end int
		if i+1 < len(diffMatches) {
			end = diffMatches[i+1][0]
		} else {
			end = len(commitBlock)
		}

		diffBlock := commitBlock[start:end]
		filePath := commitBlock[dm[2]:dm[3]]

		// Skip test files to focus on production code
		if strings.HasSuffix(filePath, "_test.go") {
			continue
		}

		before, after := extractBeforeAfter(diffBlock)
		if before != "" || after != "" {
			diffs = append(diffs, fileDiff{
				filePath: filePath,
				before:   before,
				after:    after,
			})
		}
	}

	return diffs
}

// extractBeforeAfter parses a unified diff to extract the before and after code.
func extractBeforeAfter(diffBlock string) (string, string) {
	var beforeLines, afterLines []string

	lines := strings.Split(diffBlock, "\n")
	inHunk := false

	for _, line := range lines {
		if strings.HasPrefix(line, "@@") {
			inHunk = true
			continue
		}
		if !inHunk {
			continue
		}
		if len(line) == 0 {
			continue
		}

		switch line[0] {
		case ' ':
			// Context line - present in both
			content := line[1:]
			beforeLines = append(beforeLines, content)
			afterLines = append(afterLines, content)
		case '-':
			// Removed line
			beforeLines = append(beforeLines, line[1:])
		case '+':
			// Added line
			afterLines = append(afterLines, line[1:])
		}
	}

	return strings.Join(beforeLines, "\n"), strings.Join(afterLines, "\n")
}

// extractInstruction gets the first line of the commit message.
func extractInstruction(commitBlock string) string {
	lines := strings.Split(commitBlock, "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "    ") {
			msg := strings.TrimSpace(line)
			if msg != "" {
				return msg
			}
		}
		// After the header, before the first file diff
		if strings.HasPrefix(line, "diff --git") {
			break
		}
		_ = i
	}
	return "refactor code"
}

// isValidGo checks if a string parses as valid Go source code.
func isValidGo(code string) bool {
	if strings.TrimSpace(code) == "" {
		return true // Empty is technically valid
	}
	fset := token.NewFileSet()
	_, err := parser.ParseFile(fset, "", code, parser.AllErrors)
	return err == nil
}

// buildSearchReplacePatch creates a SEARCH/REPLACE block from before/after code.
func buildSearchReplacePatch(before, after string) string {
	if before == "" {
		return fmt.Sprintf("<<<<<<< SEARCH\n\n=======\n%s\n>>>>>>> REPLACE", after)
	}
	if after == "" {
		return fmt.Sprintf("<<<<<<< SEARCH\n%s\n=======\n\n>>>>>>> REPLACE", before)
	}
	return fmt.Sprintf("<<<<<<< SEARCH\n%s\n=======\n%s\n>>>>>>> REPLACE", before, after)
}

// buildFIMExample creates a Fill-In-The-Middle example from a diff.
// It finds the changed region and splits the after-code into prefix/suffix/middle.
func buildFIMExample(before, after string, repoName string) *FIMExample {
	beforeLines := strings.Split(before, "\n")
	afterLines := strings.Split(after, "\n")

	// Find the first differing line
	firstDiff := -1
	minLen := len(beforeLines)
	if len(afterLines) < minLen {
		minLen = len(afterLines)
	}
	for i := 0; i < minLen; i++ {
		if beforeLines[i] != afterLines[i] {
			firstDiff = i
			break
		}
	}
	if firstDiff == -1 && len(beforeLines) != len(afterLines) {
		firstDiff = minLen
	}
	if firstDiff == -1 {
		return nil // No difference
	}

	// Find the last differing line
	lastDiff := -1
	for i := 0; i < minLen; i++ {
		bi := len(beforeLines) - 1 - i
		ai := len(afterLines) - 1 - i
		if bi < 0 || ai < 0 || beforeLines[bi] != afterLines[ai] {
			lastDiff = len(afterLines) - i
			break
		}
	}
	if lastDiff == -1 {
		lastDiff = len(afterLines)
	}

	// Prefix: lines before the change
	prefix := strings.Join(afterLines[:firstDiff], "\n")

	// Suffix: lines after the change
	suffix := strings.Join(afterLines[lastDiff:], "\n")

	// Middle: the changed/inserted lines
	middle := strings.Join(afterLines[firstDiff:lastDiff], "\n")

	// Only create FIM if there's actual content in all three parts
	if prefix == "" && suffix == "" {
		return nil
	}
	if middle == "" {
		return nil
	}

	return &FIMExample{
		Prefix: prefix,
		Suffix: suffix,
		Middle: middle,
		Repo:   repoName,
	}
}

// extractRepoName extracts a short name from a Git URL.
func extractRepoName(repoURL string) string {
	parts := strings.Split(strings.TrimSuffix(repoURL, ".git"), "/")
	if len(parts) >= 2 {
		return strings.Join(parts[len(parts)-2:], "_")
	}
	return strings.ReplaceAll(strings.TrimSuffix(repoURL, ".git"), "/", "_")
}

// saveJSON writes data as indented JSON to a file.
func saveJSON(path string, data interface{}) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create dir %s: %w", dir, err)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file %s: %w", path, err)
	}
	defer f.Close()

	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}
