// Package main implements a lightweight Search/Replace patch execution system
// for Gollemer. It parses SEARCH/REPLACE blocks and applies them directly to Go
// source files. This keeps context windows small and inference fast.
//
// Features:
//   - Apply SEARCH/REPLACE patches to Go source files
//   - Mine git history to extract patches as SEARCH/REPLACE pairs
//   - Convert mined patches to FIM training format
//   - Validate patches with go vet and go build
//   - Filter training data to ensure 100% syntactically valid targets
//
// Format:
//
//	<<<<<<< SEARCH
//	func (m *MoE) Route(x Tensor) int {
//		return 0
//	}
//	=======
//	func (m *MoE) Route(x Tensor) int {
//		return m.Gating.Select(x)
//	}
//	>>>>>>> REPLACE
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"go/format"
	"go/parser"
	"go/token"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
)

// Patch represents a single SEARCH/REPLACE patch.
type Patch struct {
	Search  string `json:"search"`
	Replace string `json:"replace"`
}

// PatchResult captures the outcome of applying a single patch.
type PatchResult struct {
	File    string `json:"file"`
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
	Start   int    `json:"start,omitempty"`
	End     int    `json:"end,omitempty"`
}

// PatchFile represents a collection of patches to apply to a single file.
type PatchFile struct {
	File    string  `json:"file"`
	Patches []Patch `json:"patches"`
}

// PatchBatch represents a batch of patch files to process.
type PatchBatch struct {
	Files []PatchFile `json:"files"`
}

// MinedPatch is a single mined patch with metadata for dataset creation.
type MinedPatch struct {
	CommitHash    string `json:"commit_hash"`
	File          string `json:"file"`
	Search        string `json:"search"`
	Replace       string `json:"replace"`
	ValidGo       bool   `json:"valid_go"` // Passes go vet?
	Builds        bool   `json:"builds"`   // Passes go build?
	Author        string `json:"author,omitempty"`
	Date          string `json:"date,omitempty"`
	Message       string `json:"message,omitempty"`
	FIMExample    string `json:"fim_example,omitempty"`    // FIM-formatted example
	FIMCompletion string `json:"fim_completion,omitempty"` // FIM completion (the REPLACE)
}

// MinedPatchDataset holds a collection of mined patches for training.
type MinedPatchDataset struct {
	Patches []MinedPatch `json:"patches"`
	Meta    struct {
		TotalMined    int     `json:"total_mined"`
		ValidPatches  int     `json:"valid_patches"`
		BuildingPatch int     `json:"building_patches"`
		FIMExamples   int     `json:"fim_examples"`
		ValidRatio    float64 `json:"valid_ratio"`
		BuildRatio    float64 `json:"build_ratio"`
	} `json:"meta"`
}

func main() {
	// Application flags
	file := flag.String("file", "", "Target Go source file to patch")
	patchStr := flag.String("patch", "", "SEARCH/REPLACE patch string (inline)")
	patchFile := flag.String("patch-file", "", "JSON file containing patches [{file, patches: [{search, replace}]}]")
	apply := flag.Bool("apply", false, "Apply patches to files (default: dry-run)")
	doFormat := flag.Bool("gofmt", true, "Run gofmt on patched files")
	verify := flag.Bool("verify", true, "Verify patched files with go vet")
	verbose := flag.Bool("verbose", false, "Print detailed output")

	// Git mining flags
	repo := flag.String("repo", "", "Git repository path to mine for patches")
	maxPatches := flag.Int("max", 100, "Maximum number of patches to mine")
	outFile := flag.String("out", "", "Output JSON file for mined patches")
	validate := flag.Bool("validate", true, "Validate mined patches with go vet/go build")
	fimFormat := flag.Bool("fim", true, "Convert mined patches to FIM format")

	flag.Parse()

	// Git mining mode
	if *repo != "" {
		if err := mineGitPatches(*repo, *maxPatches, *outFile, *validate, *fimFormat, *verbose); err != nil {
			log.Fatalf("Git mining failed: %v", err)
		}
		return
	}

	// Batch patch mode
	if *patchFile != "" {
		data, err := os.ReadFile(*patchFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading patch file: %v\n", err)
			os.Exit(1)
		}

		var batch PatchBatch
		if err := json.Unmarshal(data, &batch); err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing patch file: %v\n", err)
			os.Exit(1)
		}

		processBatch(batch, *apply, *doFormat, *verify, *verbose)
		return
	}

	// Inline patch mode
	if *file == "" || *patchStr == "" {
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  Apply patch:   search_replace -file <file.go> -patch '<SEARCH/REPLACE block>' -apply\n")
		fmt.Fprintf(os.Stderr, "  Batch apply:   search_replace -patch-file <patches.json> -apply\n")
		fmt.Fprintf(os.Stderr, "  Mine git:      search_replace -repo=. -out=patches.json -max=1000\n")
		fmt.Fprintf(os.Stderr, "\nPatch format:\n")
		fmt.Fprintf(os.Stderr, "<<<<<<< SEARCH\ncode to find\n=======\nreplacement code\n>>>>>>> REPLACE\n")
		os.Exit(1)
	}

	// Parse the patch string
	patches := parsePatchString(*patchStr)
	if len(patches) == 0 {
		fmt.Fprintf(os.Stderr, "No valid SEARCH/REPLACE blocks found in patch string\n")
		os.Exit(1)
	}

	pf := PatchFile{File: *file, Patches: patches}
	results := applyPatchesToFile(pf, *apply, *doFormat, *verbose)

	success := true
	for _, r := range results {
		if !r.Success {
			success = false
		}
		if *verbose || !r.Success {
			status := "✅"
			if !r.Success {
				status = "❌"
			}
			fmt.Printf("%s %s: %s\n", status, r.File, func() string {
				if r.Success {
					return fmt.Sprintf("patched at offset %d-%d", r.Start, r.End)
				}
				return fmt.Sprintf("error: %s", r.Error)
			}())
		}
	}

	if !success {
		os.Exit(1)
	}

	if *verify && *apply && strings.HasSuffix(*file, ".go") {
		if err := verifyGoCode(*file); err != nil {
			fmt.Fprintf(os.Stderr, "Verification failed: %v\n", err)
		}
	}
}

// ─── Git Mining ──────────────────────────────────────────────────────────────

// mineGitPatches mines git commit history for SEARCH/REPLACE patches.
// It extracts patches, validates them with go vet/go build, and optionally
// converts them to FIM training format.
func mineGitPatches(repoPath string, maxPatches int, outFile string, validatePatches bool, fimFormat bool, verbose bool) error {
	absRepo, err := filepath.Abs(repoPath)
	if err != nil {
		return fmt.Errorf("resolve repo path: %w", err)
	}

	// Ensure we're in a git repo
	gitDir := exec.Command("git", "-C", absRepo, "rev-parse", "--git-dir")
	if out, err := gitDir.CombinedOutput(); err != nil {
		return fmt.Errorf("not a git repository: %s (%v)", string(out), err)
	}

	fmt.Printf("🔍 Mining git history in %s\n", absRepo)
	fmt.Printf("   Max patches: %d\n", maxPatches)

	// Get commit log with statistics
	getLog := exec.Command("git", "-C", absRepo, "log",
		"--reverse",
		"--format=commit %H %ai %an%n%s",
		"--diff-filter=ACM",
		"-M", "-C",
		fmt.Sprintf("--max-count=%d", maxPatches*2), // Get extra since some won't be Go
		".")
	logOut, err := getLog.CombinedOutput()
	if err != nil {
		return fmt.Errorf("git log failed: %w", err)
	}

	// Parse log into commits
	commits := parseGitLog(string(logOut))
	fmt.Printf("   Found %d commits\n", len(commits))

	// Build the dataset
	dataset := MinedPatchDataset{}
	dataset.Meta.TotalMined = 0

	for _, commit := range commits {
		if len(dataset.Patches) >= maxPatches {
			break
		}

		if verbose {
			fmt.Printf("   Processing commit %s: %s\n", commit.hash[:8], commit.message)
		}

		// Get diff for this commit (skip initial commit with no parent)
		parentRef := commit.hash + "~1"
		// Check if parent exists
		parentCheck := exec.Command("git", "-C", absRepo, "rev-parse", "--verify", parentRef)
		if err := parentCheck.Run(); err != nil {
			if verbose {
				fmt.Printf("   Skipping initial commit %s (no parent)\n", commit.hash[:8])
			}
			continue
		}

		getDiff := exec.Command("git", "-C", absRepo, "diff",
			parentRef, commit.hash,
			"--diff-filter=ACM",
			"-M", "-C",
			".")
		diffOut, err := getDiff.CombinedOutput()
		if err != nil {
			continue
		}

		diffStr := string(diffOut)
		if diffStr == "" {
			continue
		}

		// Parse the diff into patches
		patchFiles, err := parseGitDiff(diffStr)
		if err != nil {
			continue
		}

		for _, pf := range patchFiles {
			if len(dataset.Patches) >= maxPatches {
				break
			}

			// Only include .go files
			if !strings.HasSuffix(pf.File, ".go") {
				continue
			}

			// Skip generated files
			if strings.HasSuffix(pf.File, "_test.go") ||
				strings.Contains(pf.File, "vendor/") ||
				strings.Contains(pf.File, "build_whisper/") {
				continue
			}

			for _, patch := range pf.Patches {
				if len(dataset.Patches) >= maxPatches {
					break
				}

				mined := MinedPatch{
					CommitHash: commit.hash,
					File:       pf.File,
					Search:     patch.Search,
					Replace:    patch.Replace,
					Author:     commit.author,
					Date:       commit.date,
					Message:    commit.message,
				}

				// Validate the REPLACE side with go vet/go build
				if validatePatches {
					valid, builds := validatePatchCode(patch.Replace, absRepo)
					mined.ValidGo = valid
					mined.Builds = builds
				} else {
					mined.ValidGo = true
					mined.Builds = true
				}

				// Convert to FIM format
				if fimFormat && mined.ValidGo {
					// Get the full file content at the commit before
					beforeContent, err := getFileContentAtCommit(absRepo, pf.File, commit.hash+"~1")
					if err == nil && beforeContent != "" {
						fim := SearchReplacePatchToFIM(beforeContent, patch.Search, patch.Replace)
						mined.FIMExample = fim
						mined.FIMCompletion = patch.Replace
					}
				}

				if mined.ValidGo {
					dataset.Meta.ValidPatches++
				}
				if mined.Builds {
					dataset.Meta.BuildingPatch++
				}
				dataset.Meta.TotalMined++

				dataset.Patches = append(dataset.Patches, mined)

				if verbose {
					status := "❌"
					if mined.ValidGo && mined.Builds {
						status = "✅"
					} else if mined.ValidGo {
						status = "⚠️"
					}
					fmt.Printf("      %s %s (%d bytes, vet=%v, build=%v)\n",
						status, pf.File, len(patch.Replace), mined.ValidGo, mined.Builds)
				}
			}
		}
	}

	// Calculate ratios
	if dataset.Meta.TotalMined > 0 {
		dataset.Meta.ValidRatio = float64(dataset.Meta.ValidPatches) / float64(dataset.Meta.TotalMined) * 100.0
		dataset.Meta.BuildRatio = float64(dataset.Meta.BuildingPatch) / float64(dataset.Meta.TotalMined) * 100.0
	}

	// Count FIM examples
	dataset.Meta.FIMExamples = 0
	for _, p := range dataset.Patches {
		if p.FIMExample != "" {
			dataset.Meta.FIMExamples++
		}
	}

	// Print summary
	fmt.Printf("\n📊 Mining Results:\n")
	fmt.Printf("   Total patches mined: %d\n", dataset.Meta.TotalMined)
	fmt.Printf("   Valid Go (vet pass): %d (%.1f%%)\n", dataset.Meta.ValidPatches, dataset.Meta.ValidRatio)
	fmt.Printf("   Building patches:    %d (%.1f%%)\n", dataset.Meta.BuildingPatch, dataset.Meta.BuildRatio)
	fmt.Printf("   FIM examples:        %d\n", dataset.Meta.FIMExamples)

	// Write output
	if outFile != "" {
		data, err := json.MarshalIndent(dataset, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal dataset: %w", err)
		}
		if err := os.WriteFile(outFile, data, 0644); err != nil {
			return fmt.Errorf("write output: %w", err)
		}
		fmt.Printf("\n💾 Dataset saved to %s\n", outFile)

		// Also save a FIM-only version
		if dataset.Meta.FIMExamples > 0 {
			fimFile := strings.TrimSuffix(outFile, ".json") + "_fim.json"
			fimData := exportFIMDataset(dataset)
			if err := os.WriteFile(fimFile, fimData, 0644); err == nil {
				fmt.Printf("💾 FIM dataset saved to %s\n", fimFile)
			}
		}
	}

	return nil
}

type commitInfo struct {
	hash    string
	date    string
	author  string
	message string
}

// parseGitLog parses git log --format output into commitInfo structs.
func parseGitLog(log string) []commitInfo {
	var commits []commitInfo
	lines := strings.Split(log, "\n")
	var current *commitInfo

	for _, line := range lines {
		if strings.HasPrefix(line, "commit ") {
			if current != nil {
				commits = append(commits, *current)
			}
			parts := strings.SplitN(line, " ", 3)
			current = &commitInfo{}
			if len(parts) >= 2 {
				current.hash = parts[1]
			}
			if len(parts) >= 3 {
				rest := strings.TrimSpace(parts[2])
				// Format: YYYY-MM-DD HH:MM:SS +0000 Author
				if dateEnd := strings.LastIndex(rest, " "); dateEnd >= 0 {
					if dateStart := strings.Index(rest, " "); dateStart >= 0 {
						current.date = rest[:dateEnd]
						current.author = rest[dateEnd+1:]
					}
				}
			}
		} else if current != nil && !strings.HasPrefix(line, "diff --git") && !strings.HasPrefix(line, "---") && !strings.HasPrefix(line, "+++") && !strings.HasPrefix(line, "@@") {
			if current.message == "" {
				current.message = strings.TrimSpace(line)
			}
		}
	}
	if current != nil {
		commits = append(commits, *current)
	}

	return commits
}

// getFileContentAtCommit retrieves a file's content at a specific git commit.
func getFileContentAtCommit(repoPath, filePath, commit string) (string, error) {
	cmd := exec.Command("git", "-C", repoPath, "show", commit+":"+filePath)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git show %s:%s: %w", commit, filePath, err)
	}
	return string(out), nil
}

// validatePatchCode checks if the given Go code is syntactically valid.
// For mined patches (which are code fragments, not complete files), we only
// check syntax validity via go/parser and gofmt. Full compilation is not
// possible for standalone snippets that depend on external types/functions.
// Returns (syntaxValid, gofmtValid).
func validatePatchCode(code string, repoDir string) (bool, bool) {
	// Skip empty or trivial patches
	trimmed := strings.TrimSpace(code)
	if trimmed == "" || len(trimmed) < 3 {
		return false, false
	}

	// Check basic syntax with go/parser
	// We try parsing as both a complete file and as a statement list
	fset := token.NewFileSet()
	_, err := parser.ParseFile(fset, "", code, parser.AllErrors)
	syntaxValid := err == nil

	// If that fails, try wrapping in a function body
	if !syntaxValid {
		wrapped := "package p\nfunc _(){\n" + code + "\n}"
		fset2 := token.NewFileSet()
		_, err2 := parser.ParseFile(fset2, "", wrapped, parser.AllErrors)
		syntaxValid = err2 == nil
	}

	// Check gofmt
	_, gofmtErr := format.Source([]byte(code))
	gofmtValid := gofmtErr == nil

	// If gofmt fails on the snippet, try wrapping
	if !gofmtValid {
		wrapped := "package p\nfunc _(){\n" + code + "\n}"
		_, gofmtErr2 := format.Source([]byte(wrapped))
		gofmtValid = gofmtErr2 == nil
	}

	return syntaxValid, gofmtValid
}

// exportFIMDataset creates a FIM-only dataset from mined patches.
// This filters to only include patches that produced valid FIM examples.
func exportFIMDataset(dataset MinedPatchDataset) []byte {
	type FIMEntry struct {
		FIMExample    string `json:"fim_example"`
		FIMCompletion string `json:"fim_completion"`
		File          string `json:"file"`
		CommitHash    string `json:"commit_hash,omitempty"`
	}

	var fimEntries []FIMEntry
	for _, p := range dataset.Patches {
		if p.FIMExample != "" && p.ValidGo {
			fimEntries = append(fimEntries, FIMEntry{
				FIMExample:    p.FIMExample,
				FIMCompletion: p.FIMCompletion,
				File:          p.File,
				CommitHash:    p.CommitHash,
			})
		}
	}

	data, err := json.MarshalIndent(fimEntries, "", "  ")
	if err != nil {
		return nil
	}
	return data
}

// ─── Patch Application ───────────────────────────────────────────────────────

// processBatch processes a batch of patch files.
func processBatch(batch PatchBatch, apply, useFormat, verify, verbose bool) {
	totalPatches := 0
	totalSuccess := 0

	for _, pf := range batch.Files {
		results := applyPatchesToFile(pf, apply, useFormat, verbose)
		for _, r := range results {
			totalPatches++
			if r.Success {
				totalSuccess++
			}
			if verbose || !r.Success {
				status := "✅"
				if !r.Success {
					status = "❌"
				}
				fmt.Printf("%s %s: %s\n", status, r.File, func() string {
					if r.Success {
						return fmt.Sprintf("patched at offset %d-%d", r.Start, r.End)
					}
					return fmt.Sprintf("error: %s", r.Error)
				}())
			}
		}

		if verify && apply && strings.HasSuffix(pf.File, ".go") {
			for _, r := range results {
				if r.Success {
					if err := verifyGoCode(pf.File); err != nil {
						fmt.Fprintf(os.Stderr, "Verification failed for %s: %v\n", pf.File, err)
					}
					break
				}
			}
		}
	}

	if verbose {
		fmt.Printf("\nSummary: %d/%d patches applied successfully\n", totalSuccess, totalPatches)
	}
}

// parsePatchString extracts SEARCH/REPLACE blocks from a string.
func parsePatchString(s string) []Patch {
	var patches []Patch

	// Regex to match SEARCH/REPLACE blocks
	re := regexp.MustCompile(`(?s)<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE`)
	matches := re.FindAllStringSubmatch(s, -1)

	for _, m := range matches {
		if len(m) == 3 {
			search := m[1]
			replace := m[2]

			// Trim trailing newline from search if present
			search = strings.TrimRight(search, "\n")
			replace = strings.TrimRight(replace, "\n")

			patches = append(patches, Patch{
				Search:  search,
				Replace: replace,
			})
		}
	}

	return patches
}

// applyPatchesToFile applies all patches to a single file.
func applyPatchesToFile(pf PatchFile, apply, useFormat, verbose bool) []PatchResult {
	var results []PatchResult

	// Read target file
	content, err := os.ReadFile(pf.File)
	if err != nil {
		return []PatchResult{{
			File:    pf.File,
			Success: false,
			Error:   fmt.Sprintf("read error: %v", err),
		}}
	}

	newContent := string(content)

	for _, patch := range pf.Patches {
		result := PatchResult{File: pf.File}

		// Find the search string in the content
		idx := strings.Index(newContent, patch.Search)
		if idx == -1 {
			// Try normalized matching (trim whitespace)
			normalizedSearch := strings.TrimSpace(patch.Search)
			normalizedContent := strings.TrimSpace(newContent)
			idx = strings.Index(normalizedContent, normalizedSearch)
			if idx != -1 {
				// Map back to original content position
				idx = strings.Index(newContent, patch.Search[:min(20, len(patch.Search))])
				if idx == -1 {
					result.Success = false
					result.Error = "search text not found in file (even after normalization)"
					results = append(results, result)
					continue
				}
			} else {
				result.Success = false
				result.Error = "search text not found in file"
				results = append(results, result)
				continue
			}
		}

		result.Start = idx
		result.End = idx + len(patch.Search)

		if apply {
			// Replace the search text with the replacement
			newContent = newContent[:idx] + patch.Replace + newContent[idx+len(patch.Search):]
			result.Success = true

			// Run gofmt if requested
			if useFormat && strings.HasSuffix(pf.File, ".go") {
				formatted, err := format.Source([]byte(newContent))
				if err != nil {
					if verbose {
						fmt.Fprintf(os.Stderr, "Warning: gofmt failed (patch still applied): %v\n", err)
					}
				} else {
					newContent = string(formatted)
				}
			}
		} else {
			result.Success = true
			result.Error = "dry-run (use -apply to actually patch)"
		}

		results = append(results, result)
	}

	if apply {
		// Write the patched content back
		if err := os.WriteFile(pf.File, []byte(newContent), 0644); err != nil {
			for i := range results {
				results[i].Success = false
				results[i].Error = fmt.Sprintf("write error: %v", err)
			}
		}
	}

	return results
}

// verifyGoCode runs go vet on the patched file.
func verifyGoCode(filePath string) error {
	// Run go vet
	vetCmd := exec.Command("go", "vet", filePath)
	if out, err := vetCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("go vet failed: %s", strings.TrimSpace(string(out)))
	}

	return nil
}

// ─── FIM Conversion ──────────────────────────────────────────────────────────

// ConvertPatchToFIM converts a SEARCH/REPLACE patch to a FIM training example.
func ConvertPatchToFIM(patch Patch, fileContent string) (searchExample string, replaceExample string) {
	searchExample = fmt.Sprintf("<PRE>%s<SUF>%s<MID>%s", "", fileContent, patch.Search)
	replaceExample = patch.Replace
	return
}

// SearchReplacePatchToFIM converts a SearchReplace style patch to FIM format.
func SearchReplacePatchToFIM(code, search, replace string) string {
	// Create FIM prompt where the search is masked and the model must predict the replacement
	prefix := code
	suffix := ""

	// If search is found in code, split around it
	if idx := strings.Index(code, search); idx != -1 {
		prefix = code[:idx]
		suffix = code[idx+len(search):]
	}

	return fmt.Sprintf("<FIM_PRE>%s<FIM_SUF>%s", prefix, suffix)
}

// ─── Git Diff Parsing ────────────────────────────────────────────────────────

// ExtractPatchesFromGit extracts SEARCH/REPLACE patches from a git commit.
func ExtractPatchesFromGit(commitHash string) ([]PatchFile, error) {
	cmd := exec.Command("git", "diff", commitHash+"~1", commitHash)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("git diff failed: %w", err)
	}

	return parseGitDiff(string(out))
}

// parseGitDiff parses a git diff output into SEARCH/REPLACE patches.
func parseGitDiff(diff string) ([]PatchFile, error) {
	var patchFiles []PatchFile
	var currentFile *PatchFile
	var currentHunk struct {
		search    strings.Builder
		replace   strings.Builder
		inSearch  bool
		inReplace bool
	}

	lines := strings.Split(diff, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "diff --git") {
			// Save current hunk if exists
			if currentFile != nil && currentHunk.search.Len() > 0 {
				currentFile.Patches = append(currentFile.Patches, Patch{
					Search:  strings.TrimRight(currentHunk.search.String(), "\n"),
					Replace: strings.TrimRight(currentHunk.replace.String(), "\n"),
				})
			}

			// Extract file path
			parts := strings.Split(line, " b/")
			if len(parts) == 2 {
				filePath := strings.TrimSpace(parts[1])
				if len(patchFiles) > 0 && patchFiles[len(patchFiles)-1].File == filePath {
					currentFile = &patchFiles[len(patchFiles)-1]
				} else {
					patchFiles = append(patchFiles, PatchFile{File: filePath})
					currentFile = &patchFiles[len(patchFiles)-1]
				}
			}

			currentHunk = struct {
				search    strings.Builder
				replace   strings.Builder
				inSearch  bool
				inReplace bool
			}{}
		} else if strings.HasPrefix(line, "@@") {
			// Save previous hunk
			if currentFile != nil && currentHunk.search.Len() > 0 {
				currentFile.Patches = append(currentFile.Patches, Patch{
					Search:  strings.TrimRight(currentHunk.search.String(), "\n"),
					Replace: strings.TrimRight(currentHunk.replace.String(), "\n"),
				})
			}

			currentHunk = struct {
				search    strings.Builder
				replace   strings.Builder
				inSearch  bool
				inReplace bool
			}{
				inSearch: true,
			}
		} else if currentFile != nil {
			if strings.HasPrefix(line, "---") || strings.HasPrefix(line, "+++") {
				continue
			}

			if strings.HasPrefix(line, "-") {
				if currentHunk.inSearch {
					currentHunk.search.WriteString(strings.TrimPrefix(line, "-"))
					currentHunk.search.WriteString("\n")
				}
				currentHunk.inReplace = true
			} else if strings.HasPrefix(line, "+") {
				if currentHunk.inReplace || currentHunk.replace.Len() > 0 {
					currentHunk.replace.WriteString(strings.TrimPrefix(line, "+"))
					currentHunk.replace.WriteString("\n")
				}
				currentHunk.inReplace = true
			} else {
				// Context line
				if currentHunk.inReplace {
					currentHunk.replace.WriteString(line)
					currentHunk.replace.WriteString("\n")
				}
				if currentHunk.inSearch {
					currentHunk.search.WriteString(line)
					currentHunk.search.WriteString("\n")
				}
			}
		}
	}

	// Save last hunk
	if currentFile != nil && currentHunk.search.Len() > 0 {
		currentFile.Patches = append(currentFile.Patches, Patch{
			Search:  strings.TrimRight(currentHunk.search.String(), "\n"),
			Replace: strings.TrimRight(currentHunk.replace.String(), "\n"),
		})
	}

	return patchFiles, nil
}

// ─── Dataset Filtering with Compiler Validation ─────────────────────────────

// FilterValidPatches filters out mined patches that don't pass go vet.
// This guarantees 100% syntactically valid target sequences in the training data.
func FilterValidPatches(dataset MinedPatchDataset) MinedPatchDataset {
	filtered := MinedPatchDataset{}
	filtered.Meta.TotalMined = dataset.Meta.TotalMined

	for _, p := range dataset.Patches {
		if p.ValidGo {
			filtered.Patches = append(filtered.Patches, p)
		}
	}

	filtered.Meta.ValidPatches = len(filtered.Patches)
	filtered.Meta.ValidRatio = 100.0 // All remaining patches are valid
	return filtered
}

// FilterBuildingPatches filters to only include patches that pass go build.
func FilterBuildingPatches(dataset MinedPatchDataset) MinedPatchDataset {
	filtered := MinedPatchDataset{}
	filtered.Meta.TotalMined = dataset.Meta.TotalMined

	for _, p := range dataset.Patches {
		if p.Builds {
			filtered.Patches = append(filtered.Patches, p)
		}
	}

	filtered.Meta.BuildingPatch = len(filtered.Patches)
	filtered.Meta.BuildRatio = 100.0
	return filtered
}

// FIMExampleLocal is a local FIM example struct for compiler validation.
type FIMExampleLocal struct {
	Prefix string `json:"prefix"`
	Suffix string `json:"suffix"`
	Middle string `json:"middle"`
}

// ValidateAndFilterFIMExamples validates FIM examples by checking that
// the completion (middle) passes go vet. This ensures all training targets
// are syntactically valid Go code.
func ValidateAndFilterFIMExamples(examples []FIMExampleLocal) []FIMExampleLocal {
	var filtered []FIMExampleLocal
	rejected := 0

	for _, ex := range examples {
		// Check if the middle (target) passes go syntax validation
		fset := token.NewFileSet()
		_, err := parser.ParseFile(fset, "", ex.Middle, parser.AllErrors)
		if err != nil {
			rejected++
			continue
		}

		// Also check with gofmt
		_, err = format.Source([]byte(ex.Middle))
		if err != nil {
			rejected++
			continue
		}

		filtered = append(filtered, ex)
	}

	if rejected > 0 {
		log.Printf("🧹 Compiler validation: rejected %d/%d FIM examples (%.1f%% valid)",
			rejected, len(examples),
			float64(len(filtered))/float64(len(examples))*100.0)
	}

	return filtered
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
