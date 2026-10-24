package chat

import (
	"fmt"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// pieces_cmds.go — the Composer's library commands: /build-api,
// /learn-pieces, /export-understanding. Their bodies began life
// inline in the single-domain loop; they live here so the slash
// commands and the intent layer ("build an api with /users",
// "learn the pieces", "export the understanding pairs") run the
// exact same code.

// runExportUnderstanding generates structure->language training
// pairs from the analyzed project and writes them out. This is
// "train on how the program knows code" — not character
// prediction. Caller guarantees a project is analyzed.
func runExportUnderstanding() {
	pairs := analyze.GenerateStructurePairs(lastAnalyzeProject)
	path := "/tmp/understanding_pairs.jsonl"
	if err := analyze.ExportStructurePairs(pairs, path); err != nil {
		fmt.Printf("Couldn't export: %v\n", err)
		return
	}
	// Count by kind.
	kinds := make(map[string]int)
	for _, p := range pairs {
		kinds[p.Kind]++
	}
	fmt.Printf("Exported %d structure→language pairs to %s\n", len(pairs), path)
	for k, v := range kinds {
		fmt.Printf("  %s: %d\n", k, v)
	}
	fmt.Println("\nThese teach a model HOW the analyzer understands code:")
	fmt.Println("  Input:  structural facts (function X on Y, locks for reading, ...)")
	fmt.Println("  Output: natural explanation (X returns... It locks...)")
	fmt.Println("\nThe model learns facts→words, not characters→characters.")
}

// runLearnPieces extracts composable piece schemas from doc
// comments in the analyzed project. Pieces are learned from
// natural language docs, not hardcoded. Caller guarantees a
// project is analyzed.
func runLearnPieces() {
	lib := analyze.NewPieceLibrary()
	n := lib.LearnFromDocs(lastAnalyzeProject)
	fmt.Printf("Learned %d pieces from doc comments.\n", n)
	// Show a few.
	shown := 0
	for name, p := range lib.Pieces {
		if shown >= 5 {
			break
		}
		fmt.Printf("  %s: %s\n", name, p.Purpose)
		shown++
	}
	if n > 5 {
		fmt.Printf("  ... and %d more.\n", n-5)
	}
}

// runBuildAPI composes a working HTTP API from pieces, one route
// per path. Pieces are learned schemas (not memorized code);
// placement is constraint-solved; each piece explains why it's
// where it is. The composed program is saved to ./compose/main.go.
func runBuildAPI(paths []string) {
	lib := analyze.NewPieceLibrary()
	lib.SeedHTTP()
	// Refine from the current project if one is analyzed.
	if lastAnalyzeProject != nil {
		lib.LearnHTTPFromProject(lastAnalyzeProject)
	}
	req := analyze.ComposeRequest{Kind: "api", Address: ":8080"}
	for _, p := range paths {
		if !strings.HasPrefix(p, "/") {
			p = "/" + p
		}
		// Describe the data from the path: "/users" -> "list of users".
		desc := "data for " + p
		trimmed := strings.Trim(p, "/")
		if trimmed != "" {
			desc = "list of " + trimmed
		}
		req.Routes = append(req.Routes, analyze.ComposeRoute{
			Path: p, Returns: "json", DataDesc: desc,
		})
	}
	code, explanations := lib.Compose(req)
	fmt.Printf("\nComposed API (%d routes):\n\n%s\n", len(req.Routes), code)
	fmt.Printf("Why each piece is where it is:\n")
	for _, e := range explanations {
		fmt.Printf("  • %s\n", e)
	}
	fmt.Printf("\nThis was composed from learned pieces, not copied.\n")
	fmt.Printf("The data is a placeholder — replace the TODO with your real data source.\n")
	// Save to ./compose/main.go
	composeDir := "compose"
	if err := os.MkdirAll(composeDir, 0755); err != nil {
		fmt.Printf("Couldn't create compose dir: %v\n", err)
	} else {
		outPath := composeDir + "/main.go"
		if err := os.WriteFile(outPath, []byte(code), 0644); err != nil {
			fmt.Printf("Couldn't write %s: %v\n", outPath, err)
		} else {
			fmt.Printf("\nSaved to %s — run it with: cd compose && go mod init api && go run main.go\n", outPath)
		}
	}
}
