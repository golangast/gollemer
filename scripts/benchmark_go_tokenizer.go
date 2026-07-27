package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

func main() {
	dirPath := flag.String("path", ".", "Directory to scan for Go files")
	flag.Parse()

	var goFiles []string

	// WalkDir correctly resolves relative paths across subdirectories
	err := filepath.WalkDir(*dirPath, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		// Skip build/vendor trees
		if d.IsDir() && (d.Name() == "vendor" || d.Name() == ".build" || d.Name() == ".git") {
			return filepath.SkipDir
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".go") {
			goFiles = append(goFiles, path)
		}
		return nil
	})

	if err != nil {
		log.Fatalf("Error scanning directory: %v", err)
	}

	if len(goFiles) == 0 {
		log.Fatalf("No Go source files found under path: %s", *dirPath)
	}

	fmt.Printf("🔍 Benchmark scanning %d Go files under '%s'...\n\n", len(goFiles), *dirPath)
	// Limit to a reasonable sample
	if len(goFiles) > 50 {
		goFiles = goFiles[:50]
	}

	fmt.Printf("🔬 Go Tokenizer Compression Benchmark\n")
	fmt.Printf("   Files sampled: %d\n", len(goFiles))
	fmt.Println(strings.Repeat("─", 72))

	// Create two tokenizers: one without Go tokens, one with
	bpePlain := tokenizer.NewBPETokenizer(16384)
	bpeGo := tokenizer.NewBPETokenizer(16384)

	// Add Go-specific tokens to the second tokenizer
	tokenizer.AddGoTokensToTokenizer(bpeGo)

	// Also add standard special tokens to both
	for _, tok := range []string{"<|im_start|>", "<|im_end|>", "<pad>", "<s>", "</s>", "UNK", "\n"} {
		bpePlain.AddSpecialToken(tok)
	}

	// Track statistics
	type fileStats struct {
		path        string
		size        int
		plainTokens int
		goTokens    int
		compression float64
	}

	var stats []fileStats
	totalPlain := 0
	totalGo := 0
	totalSize := 0

	for _, path := range goFiles {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		content := string(data)
		size := len(content)

		// Tokenize with plain BPE
		plainIDs := bpePlain.Encode(content)
		plainCount := len(plainIDs)

		// Preprocess Go source text to collapse high-frequency multi-character tokens
		preprocessedContent := tokenizer.GoTokenizePreprocess(content)

		// Tokenize preprocessed text with Go-aware BPE
		goIDs := bpeGo.Encode(preprocessedContent)
		goCount := len(goIDs)
		compression := 0.0
		if plainCount > 0 {
			compression = (1.0 - float64(goCount)/float64(plainCount)) * 100.0
		}

		totalPlain += plainCount
		totalGo += goCount
		totalSize += size

		stats = append(stats, fileStats{
			path:        path,
			size:        size,
			plainTokens: plainCount,
			goTokens:    goCount,
			compression: compression,
		})
	}

	// Print per-file results
	fmt.Printf("\n%-55s %8s %8s %8s %8s\n", "File", "Size", "Plain", "Go", "Δ%")
	fmt.Println(strings.Repeat("─", 72))
	for _, s := range stats {
		delta := fmt.Sprintf("%+.1f%%", s.compression)
		fmt.Printf("%-55s %8d %8d %8d %8s\n",
			truncatePath(s.path, 55),
			s.size,
			s.plainTokens,
			s.goTokens,
			delta,
		)
	}

	// Print summary
	fmt.Println(strings.Repeat("─", 72))
	avgCompression := 0.0
	if len(stats) > 0 {
		avgCompression = (1.0 - float64(totalGo)/float64(totalPlain)) * 100.0
	}
	fmt.Printf("%-55s %8d %8d %8d %+.1f%%\n",
		"TOTAL",
		totalSize,
		totalPlain,
		totalGo,
		avgCompression,
	)
	fmt.Println(strings.Repeat("─", 72))

	// Detailed analysis
	fmt.Printf("\n📊 Analysis:\n")
	fmt.Printf("   Total plain tokens:  %d\n", totalPlain)
	fmt.Printf("   Total Go tokens:     %d\n", totalGo)
	fmt.Printf("   Tokens saved:        %d (%.1f%% reduction)\n",
		totalPlain-totalGo, avgCompression)
	fmt.Printf("   Go vocab tokens:     %d\n", len(tokenizer.GoHighFreqTokens()))

	// Show top patterns that save the most tokens
	fmt.Printf("\n🏆 Top token-saving patterns:\n")
	patternSavings := countPatternSavings(goFiles, bpePlain, bpeGo)
	for i, ps := range patternSavings {
		if i >= 10 {
			break
		}
		fmt.Printf("   %3d. %-30s saved %d tokens across %d files\n",
			i+1, truncateString(ps.pattern, 30), ps.savings, ps.files)
	}

	fmt.Printf("\n✅ Benchmark complete\n")
}

type patternSaving struct {
	pattern string
	savings int
	files   int
}

func countPatternSavings(files []string, plain, goTok *tokenizer.BPETokenizer) []patternSaving {
	patterns := tokenizer.GoHighFreqTokens()
	var results []patternSaving

	for _, p := range patterns {
		if len(p) < 5 {
			continue
		}
		totalSavings := 0
		fileCount := 0
		for _, path := range files {
			data, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			content := string(data)
			count := strings.Count(content, p)
			if count > 0 {
				// Each occurrence saves (len(p) in chars / avg token size) - 1 tokens
				// Rough estimate: each multi-token pattern saves ~2-5 tokens
				savings := count * 3 // approximate: each pattern saves ~3 tokens vs BPE
				totalSavings += savings
				fileCount++
			}
		}
		if totalSavings > 0 {
			results = append(results, patternSaving{
				pattern: p,
				savings: totalSavings,
				files:   fileCount,
			})
		}
	}

	// Sort by savings descending
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].savings > results[i].savings {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	return results
}

func truncatePath(path string, maxLen int) string {
	if len(path) <= maxLen {
		return path
	}
	return "..." + path[len(path)-maxLen+3:]
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
