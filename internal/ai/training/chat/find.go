package chat

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// find.go — natural-language code search: "where is the auth
// logic?" finds funcs/types with "auth" in the name or doc. The
// body began life inline behind /find in the single-domain loop;
// it lives here so the slash command and the intent layer
// ("search the code for retries") run the exact same search.
// Caller guarantees a project is analyzed.

// runFindSearch prints every func/type in the analyzed project
// whose name or doc mentions any word of the concept.
func runFindSearch(concept string) {
	words := strings.Fields(concept)
	type hit struct {
		kind, name, file string
		line             int
		desc             string
	}
	var hits []hit
	for _, pkg := range lastAnalyzeProject.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			nl := strings.ToLower(fn.Name + " " + fn.Doc)
			matched := false
			for _, w := range words {
				if strings.Contains(nl, w) {
					matched = true
					break
				}
			}
			if matched {
				desc := fn.Sig
				if fn.Doc != "" {
					if s := analyze.PieceFirstSentence(fn.Doc); s != "" {
						desc = s
					}
				}
				hits = append(hits, hit{"func", fn.Name, fn.File, fn.Line, desc})
			}
		}
		for _, t := range pkg.Types {
			nl := strings.ToLower(t.Name)
			matched := false
			for _, w := range words {
				if strings.Contains(nl, w) {
					matched = true
					break
				}
			}
			if matched {
				hits = append(hits, hit{"type", t.Name, t.File, t.Line, t.Kind})
			}
		}
	}
	if len(hits) == 0 {
		fmt.Printf("No matches for %q.\n", concept)
	} else {
		fmt.Printf("Found %d match(es) for %q:\n", len(hits), concept)
		for i, h := range hits {
			if i >= 10 {
				fmt.Printf("  ... and %d more.\n", len(hits)-10)
				break
			}
			fmt.Printf("  %s %s (%s:%d)\n    %s\n", h.kind, h.name, h.file, h.line, h.desc)
		}
	}
}
