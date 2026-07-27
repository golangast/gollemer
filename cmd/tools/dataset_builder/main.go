// DatasetBuilder reads raw mined patches from dataset_miner and converts them into
// FIM (Fill-In-The-Middle) format and augmented SEARCH/REPLACE training data.
// It also splits data into train/val/test sets and augments with concept-guided examples.
//
// Usage:
//
//	go run cmd/tools/dataset_builder/main.go \
//	  -in="data/training/mined_patches.json" \
//	  -out="data/training/fim_dataset.json"
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// FIMTrainingExample is a complete training example for FIM-style learning.
type FIMTrainingExample struct {
	Instruction string `json:"instruction"`
	Prompt      string `json:"prompt"`     // <PRE> prefix <SUF> suffix <MID>
	Completion  string `json:"completion"` // The code to insert
	Repo        string `json:"repo,omitempty"`
}

// SearchReplaceExample is a training example for SEARCH/REPLACE patch generation.
type SearchReplaceExample struct {
	Instruction string `json:"instruction"`
	BeforeCode  string `json:"before_code"`
	TargetPatch string `json:"target_patch"`
	ValidGo     bool   `json:"valid_go"`
}

// AugmentedExample pairs a patch with concept guidance.
type AugmentedExample struct {
	Instruction   string   `json:"instruction"`
	BeforeCode    string   `json:"before_code"`
	TargetPatch   string   `json:"target_patch"`
	Concepts      []string `json:"concepts"`
	RequiredPrims []string `json:"required_primitives"`
}

// Dataset holds all training data splits.
type Dataset struct {
	Train []interface{} `json:"train"`
	Val   []interface{} `json:"val"`
	Test  []interface{} `json:"test"`
	Meta  DatasetMeta   `json:"meta"`
}

// DatasetMeta contains dataset statistics.
type DatasetMeta struct {
	TotalExamples   int            `json:"total_examples"`
	ValidGoExamples int            `json:"valid_go_examples"`
	FIMExamples     int            `json:"fim_examples"`
	SearchReplace   int            `json:"search_replace"`
	Repos           map[string]int `json:"repos"`
	ConceptsFound   []string       `json:"concepts_found"`
}

func main() {
	inputPath := flag.String("in", "data/training/mined_patches.json", "Input mined patches JSON")
	outputPath := flag.String("out", "data/training/fim_dataset.json", "Output dataset JSON")
	valSplit := flag.Float64("val-split", 0.1, "Fraction of data for validation")
	testSplit := flag.Float64("test-split", 0.05, "Fraction of data for testing")
	flag.Parse()

	if err := run(*inputPath, *outputPath, *valSplit, *testSplit); err != nil {
		log.Fatalf("Fatal: %v", err)
	}
}

func run(inputPath, outputPath string, valSplit, testSplit float64) error {
	log.Printf("Loading mined patches from %s...", inputPath)

	// Load mined patches
	triplets, err := loadTriplets(inputPath)
	if err != nil {
		return fmt.Errorf("load triplets: %w", err)
	}
	log.Printf("Loaded %d total triplets", len(triplets))

	// Also try to load FIM examples if available
	fimPath := strings.Replace(inputPath, ".json", "_fim.json", 1)
	fimExamples := loadFIMExamples(fimPath)
	log.Printf("Loaded %d FIM examples", len(fimExamples))

	// 1. Convert to FIM format
	var fimTrain []FIMTrainingExample
	var searchReplace []SearchReplaceExample
	var augmented []AugmentedExample

	for _, t := range triplets {
		// Skip triplets with empty before or after
		if t.BeforeCode == "" {
			continue
		}

		// Build FIM example
		fim := buildFIMFromTriplet(t)
		if fim != nil {
			fimTrain = append(fimTrain, *fim)
		}

		// Add as search/replace example
		sr := SearchReplaceExample{
			Instruction: t.Instruction,
			BeforeCode:  t.BeforeCode,
			TargetPatch: t.TargetPatch,
			ValidGo:     t.ValidGo,
		}
		searchReplace = append(searchReplace, sr)

		// Build augmented example with concept extraction
		aug := buildAugmentedExample(t)
		if aug != nil {
			augmented = append(augmented, *aug)
		}
	}

	// 2. Convert FIM examples from mined data
	var allFIM []FIMTrainingExample
	for _, f := range fimExamples {
		fimExample := FIMTrainingExample{
			Instruction: "Fill in the middle of this Go code",
			Prompt:      fmt.Sprintf("<PRE>%s<SUF>%s<MID>", f.Prefix, f.Suffix),
			Completion:  f.Middle,
			Repo:        f.Repo,
		}
		allFIM = append(allFIM, fimExample)
	}
	// Add the converted ones from triplets
	allFIM = append(allFIM, fimTrain...)

	log.Printf("Generated %d FIM examples, %d SEARCH/REPLACE, %d augmented",
		len(allFIM), len(searchReplace), len(augmented))

	// 3. Split into train/val/test
	rand.Shuffle(len(allFIM), func(i, j int) {
		allFIM[i], allFIM[j] = allFIM[j], allFIM[i]
	})

	n := len(allFIM)
	nVal := int(float64(n) * valSplit)
	nTest := int(float64(n) * testSplit)

	trainData := make([]interface{}, len(searchReplace))
	for i, sr := range searchReplace {
		trainData[i] = sr
	}
	// Also add FIM examples and augmented
	for _, f := range allFIM[:max(1, n-nVal-nTest)] {
		trainData = append(trainData, f)
	}
	for _, a := range augmented {
		trainData = append(trainData, a)
	}

	valData := make([]interface{}, 0)
	if nVal > 0 && len(allFIM) > nVal {
		for _, f := range allFIM[n-nVal-nTest : n-nTest] {
			valData = append(valData, f)
		}
	}

	testData := make([]interface{}, 0)
	if nTest > 0 && len(allFIM) > nTest {
		for _, f := range allFIM[n-nTest:] {
			testData = append(testData, f)
		}
	}

	// 4. Build metadata
	meta := DatasetMeta{
		TotalExamples:   len(triplets),
		ValidGoExamples: countValidGo(triplets),
		FIMExamples:     len(allFIM),
		SearchReplace:   len(searchReplace),
		Repos:           countRepos(triplets),
		ConceptsFound:   extractConceptList(searchReplace),
	}

	dataset := Dataset{
		Train: trainData,
		Val:   valData,
		Test:  testData,
		Meta:  meta,
	}

	// 5. Save
	if err := saveDataset(outputPath, dataset); err != nil {
		return fmt.Errorf("save dataset: %w", err)
	}

	log.Printf("Dataset saved to %s", outputPath)
	log.Printf("  Train: %d, Val: %d, Test: %d examples", len(trainData), len(valData), len(testData))
	log.Printf("  Valid Go: %d/%d (%.1f%%)", meta.ValidGoExamples, meta.TotalExamples,
		float64(meta.ValidGoExamples)/float64(max(1, meta.TotalExamples))*100)

	return nil
}

// buildFIMFromTriplet converts a TrainingTriplet into a FIMTrainingExample.
func buildFIMFromTriplet(t TrainingTriplet) *FIMTrainingExample {
	before := t.BeforeCode
	after := extractAfterFromPatch(t.TargetPatch)
	if after == "" {
		return nil
	}

	// Parse the before code to find insertion point
	prefix, middle, suffix := splitIntoFIM(before, after)
	if middle == "" {
		return nil
	}

	return &FIMTrainingExample{
		Instruction: t.Instruction,
		Prompt:      fmt.Sprintf("<PRE>%s<SUF>%s<MID>", prefix, suffix),
		Completion:  middle,
		Repo:        t.Repo,
	}
}

// extractAfterFromPatch extracts the REPLACE section from a SEARCH/REPLACE patch.
func extractAfterFromPatch(patch string) string {
	replaceRegex := regexp.MustCompile(`=======\n(.*?)\n>>>>>>> REPLACE`)
	matches := replaceRegex.FindStringSubmatch(patch)
	if len(matches) >= 2 {
		return matches[1]
	}
	return ""
}

// splitIntoFIM splits the after-code into prefix/suffix/middle based on differences from before.
func splitIntoFIM(before, after string) (string, string, string) {
	beforeLines := strings.Split(before, "\n")
	afterLines := strings.Split(after, "\n")

	// Find first diff
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
	if firstDiff == -1 {
		if len(beforeLines) != len(afterLines) {
			firstDiff = minLen
		} else {
			return "", "", ""
		}
	}

	// Find last diff
	lastDiff := len(afterLines)
	for i := 0; i < minLen; i++ {
		bi := len(beforeLines) - 1 - i
		ai := len(afterLines) - 1 - i
		if bi < 0 || ai < 0 || beforeLines[bi] != afterLines[ai] {
			lastDiff = len(afterLines) - i
			break
		}
	}

	prefix := strings.Join(afterLines[:firstDiff], "\n")
	middle := strings.Join(afterLines[firstDiff:lastDiff], "\n")
	suffix := strings.Join(afterLines[lastDiff:], "\n")

	return prefix, middle, suffix
}

// buildAugmentedExample creates a concept-augmented training example.
func buildAugmentedExample(t TrainingTriplet) *AugmentedExample {
	// Extract concept terms from the instruction
	concepts := extractConceptsFromText(t.Instruction)
	if len(concepts) == 0 {
		return nil
	}

	// Extract required Go primitives from the before/after code
	prims := extractPrimitivesFromCode(t.BeforeCode)
	prims = append(prims, extractPrimitivesFromCode(extractAfterFromPatch(t.TargetPatch))...)

	// Deduplicate
	seen := make(map[string]bool)
	var unique []string
	for _, p := range prims {
		if !seen[p] {
			seen[p] = true
			unique = append(unique, p)
		}
	}

	return &AugmentedExample{
		Instruction:   t.Instruction,
		BeforeCode:    t.BeforeCode,
		TargetPatch:   t.TargetPatch,
		Concepts:      concepts,
		RequiredPrims: unique,
	}
}

// extractConceptsFromText finds concept-like terms in text.
func extractConceptsFromText(text string) []string {
	knownConcepts := map[string]bool{
		"worker pool": true, "goroutine": true, "concurrency": true,
		"caching": true, "cache": true, "memoization": true,
		"error handling": true, "error": true, "logging": true,
		"context": true, "deadline": true, "cancelation": true,
		"sync": true, "mutex": true, "waitgroup": true,
		"channel": true, "pipeline": true, "stream": true,
		"graceful shutdown": true, "signal": true,
		"rate limit": true, "throttle": true, "circuit breaker": true,
		"retry": true, "backoff": true, "timeout": true,
		"singleton": true, "dependency injection": true, "di": true,
		"observer": true, "pubsub": true, "event": true,
	}

	var found []string
	textLower := strings.ToLower(text)

	for concept := range knownConcepts {
		if strings.Contains(textLower, concept) {
			found = append(found, concept)
		}
	}

	return found
}

// extractPrimitivesFromCode finds Go language primitives in code.
func extractPrimitivesFromCode(code string) []string {
	primitives := []string{
		"chan", "go ", "sync.WaitGroup", "sync.Mutex", "sync.RWMutex",
		"context.Context", "defer", "error", "interface{}",
		"time.Ticker", "time.Duration", "os.Signal",
		"json.Marshal", "json.Unmarshal", "errors.New",
		"http.Handler", "http.HandlerFunc",
	}

	var found []string
	for _, p := range primitives {
		if strings.Contains(code, p) {
			found = append(found, p)
		}
	}
	return found
}

// loadTriplets loads TrainingTriplet slices from JSON.
func loadTriplets(path string) ([]TrainingTriplet, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var triplets []TrainingTriplet
	if err := json.Unmarshal(data, &triplets); err != nil {
		return nil, err
	}
	return triplets, nil
}

// TrainingTriplet mirrors the miner's structure for loading.
type TrainingTriplet struct {
	Instruction string `json:"instruction"`
	BeforeCode  string `json:"before_code"`
	TargetPatch string `json:"target_patch"`
	Repo        string `json:"repo"`
	CommitHash  string `json:"commit_hash"`
	FilePath    string `json:"file_path"`
	ValidGo     bool   `json:"valid_go"`
}

// loadFIMExamples loads FIM examples from a JSON file.
func loadFIMExamples(path string) []FIMExample {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var examples []FIMExample
	if err := json.Unmarshal(data, &examples); err != nil {
		log.Printf("Warning: could not load FIM examples: %v", err)
		return nil
	}
	return examples
}

// FIMExample mirrors the miner's FIMExample structure.
type FIMExample struct {
	Prefix string `json:"prefix"`
	Suffix string `json:"suffix"`
	Middle string `json:"middle"`
	Repo   string `json:"repo"`
}

// countValidGo counts how many triplets have valid Go on both sides.
func countValidGo(triplets []TrainingTriplet) int {
	count := 0
	for _, t := range triplets {
		if t.ValidGo {
			count++
		}
	}
	return count
}

// countRepos builds a map of repo -> commit count.
func countRepos(triplets []TrainingTriplet) map[string]int {
	repos := make(map[string]int)
	for _, t := range triplets {
		repos[t.Repo]++
	}
	return repos
}

// extractConceptList finds unique concepts across all examples.
func extractConceptList(examples []SearchReplaceExample) []string {
	seen := make(map[string]bool)
	for _, ex := range examples {
		for _, c := range extractConceptsFromText(ex.Instruction) {
			seen[c] = true
		}
	}
	var list []string
	for c := range seen {
		list = append(list, c)
	}
	return list
}

// saveDataset writes the dataset to a JSON file.
func saveDataset(path string, dataset Dataset) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create dir: %w", err)
	}
	data, err := json.MarshalIndent(dataset, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	return os.WriteFile(path, data, 0644)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
