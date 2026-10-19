package main

import (
	"fmt"
	"os"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// Builds the cross-project pattern library by analyzing multiple repos.
// Usage: buildpatterns <output.json> <repo1> <repo2> ...
func main() {
	if len(os.Args) < 3 {
		fmt.Println("Usage: buildpatterns <output.json> <repo1> [repo2...]")
		os.Exit(1)
	}
	output := os.Args[1]
	repos := os.Args[2:]

	lib := analyze.NewPatternLibrary()
	for _, repo := range repos {
		fmt.Printf("Analyzing %s...\n", repo)
		proj, err := analyze.Analyze(repo)
		if err != nil {
			fmt.Printf("  Failed: %v\n", err)
			continue
		}
		before := len(lib.Patterns)
		lib.LearnFromProject(proj)
		fmt.Printf("  Learned: %d patterns (total %d)\n", len(lib.Patterns)-before, len(lib.Patterns))
	}

	if err := lib.Save(output); err != nil {
		fmt.Printf("Save failed: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nSaved %d patterns to %s\n", len(lib.Patterns), output)
	for kind, pat := range lib.Patterns {
		fmt.Printf("  %s: %d examples\n", kind, pat.Count)
	}
}
