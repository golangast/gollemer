// Package chat — style learner (Layer 3): learns John's code conventions
// from his actual Go files. Naming patterns, test styles, error handling.
// Used to shape generated code to match HIS style, not generic Go style.
package chat

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// CodeStyle is what we've learned about how John writes Go.
type CodeStyle struct {
	// HandlerNaming: "XHandler" vs "handleX" vs "HandleX".
	HandlerNaming string `json:"handler_naming"`
	// TestStyle: "table-driven" vs "simple" vs "none observed".
	TestStyle string `json:"test_style"`
	// ErrorStyle: "wrapped" (fmt.Errorf %w) vs "plain" vs "panic".
	ErrorStyle string `json:"error_style"`
	// ReceiverNaming: single-letter ("c", "s") vs full word ("container").
	ReceiverNaming string `json:"receiver_naming"`
	// CommentDensity: "high" (>50% exported have docs), "medium", "low".
	CommentDensity string `json:"comment_density"`
	// CommonPrefixes: frequent function name prefixes he uses.
	CommonPrefixes []string `json:"common_prefixes"`
	// FilesScanned: how many files this was learned from.
	FilesScanned int `json:"files_scanned"`
}

// LearnStyle scans Go files in a directory and extracts John's conventions.
func LearnStyle(dir string) *CodeStyle {
	style := &CodeStyle{}
	fset := token.NewFileSet()

	var handlerNames []string
	var receiverNames []string
	var testFuncs, tableTests int
	var wrappedErrs, plainErrs int
	var exported, documented int
	prefixCount := make(map[string]int)

	filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			// Count test files separately.
			if strings.HasSuffix(path, "_test.go") {
				style.FilesScanned++
				src, err := os.ReadFile(path)
				if err == nil && strings.Contains(string(src), "[]struct{") {
					tableTests++
				}
				if err == nil {
					testFuncs += strings.Count(string(src), "func Test")
				}
			}
			return nil
		}
		// Skip vendor and hidden.
		if strings.Contains(path, "/vendor/") || strings.Contains(path, "/.") {
			return nil
		}

		src, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			return nil
		}
		style.FilesScanned++

		ast.Inspect(src, func(n ast.Node) bool {
			switch node := n.(type) {
			case *ast.FuncDecl:
				name := node.Name.Name
				// Handler naming.
				if strings.Contains(strings.ToLower(name), "handler") {
					handlerNames = append(handlerNames, name)
				}
				// Receiver naming.
				if node.Recv != nil && len(node.Recv.List) > 0 {
					for _, field := range node.Recv.List {
						for _, rn := range field.Names {
							receiverNames = append(receiverNames, rn.Name)
						}
					}
				}
				// Prefix counting (first 3+ chars).
				if len(name) >= 4 {
					prefix := name[:4]
					// Only count capitalized prefixes (exported conventions).
					if prefix[0] >= 'A' && prefix[0] <= 'Z' {
						prefixCount[prefix]++
					}
				}
				// Exported + documented.
				if ast.IsExported(name) {
					exported++
					if node.Doc != nil && len(node.Doc.List) > 0 {
						documented++
					}
				}
			case *ast.CallExpr:
				// Error wrapping detection.
				if sel, ok := node.Fun.(*ast.SelectorExpr); ok {
					if x, ok := sel.X.(*ast.Ident); ok && x.Name == "fmt" && sel.Sel.Name == "Errorf" {
						// Check for %w in args.
						for _, arg := range node.Args {
							if lit, ok := arg.(*ast.BasicLit); ok {
								if strings.Contains(lit.Value, "%w") {
									wrappedErrs++
								} else {
									plainErrs++
								}
							}
						}
					}
				}
			}
			return true
		})
		return nil
	})

	// Derive conventions.
	style.HandlerNaming = deriveHandlerNaming(handlerNames)
	style.ReceiverNaming = deriveReceiverNaming(receiverNames)
	style.TestStyle = deriveTestStyle(testFuncs, tableTests)
	style.ErrorStyle = deriveErrorStyle(wrappedErrs, plainErrs)
	style.CommentDensity = deriveCommentDensity(exported, documented)
	style.CommonPrefixes = topPrefixes(prefixCount, 5)

	return style
}

func deriveHandlerNaming(names []string) string {
	if len(names) == 0 {
		return "not observed"
	}
	suffix, prefix, other := 0, 0, 0
	for _, n := range names {
		l := strings.ToLower(n)
		if strings.HasSuffix(l, "handler") {
			suffix++
		} else if strings.HasPrefix(l, "handle") {
			prefix++
		} else {
			other++
		}
	}
	if suffix >= prefix && suffix >= other {
		return "XHandler (suffix)"
	}
	if prefix >= suffix && prefix >= other {
		return "handleX (prefix)"
	}
	return "mixed"
}

func deriveReceiverNaming(names []string) string {
	if len(names) == 0 {
		return "not observed"
	}
	short, long := 0, 0
	for _, n := range names {
		if len(n) <= 2 {
			short++
		} else {
			long++
		}
	}
	if short > long*2 {
		return "single-letter (c, s, d)"
	}
	if long > short*2 {
		return "full-word (container, server)"
	}
	return "mixed"
}

func deriveTestStyle(testFuncs, tableTests int) string {
	if testFuncs == 0 {
		return "no tests observed"
	}
	if tableTests*2 >= testFuncs {
		return "table-driven"
	}
	return "simple"
}

func deriveErrorStyle(wrapped, plain int) string {
	if wrapped+plain == 0 {
		return "not observed"
	}
	if wrapped > plain*2 {
		return "wrapped (fmt.Errorf %w)"
	}
	if plain > wrapped*2 {
		return "plain (errors.New / fmt.Errorf)"
	}
	return "mixed"
}

func deriveCommentDensity(exported, documented int) string {
	if exported == 0 {
		return "not observed"
	}
	ratio := float64(documented) / float64(exported)
	if ratio > 0.5 {
		return "high"
	}
	if ratio > 0.2 {
		return "medium"
	}
	return "low"
}

func topPrefixes(counts map[string]int, n int) []string {
	type kv struct {
		k string
		v int
	}
	var sorted []kv
	for k, v := range counts {
		sorted = append(sorted, kv{k, v})
	}
	// Simple bubble sort for small n.
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].v > sorted[i].v {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	var result []string
	for i := 0; i < len(sorted) && i < n; i++ {
		if sorted[i].v >= 2 { // only if seen 2+ times
			result = append(result, sorted[i].k)
		}
	}
	return result
}

// StyleSummary returns a human-readable summary for /profile.
func (s *CodeStyle) StyleSummary() string {
	var b strings.Builder
	b.WriteString("Code style (learned from " + strconv.Itoa(s.FilesScanned) + " files):\n")
	b.WriteString("  Handlers: " + s.HandlerNaming + "\n")
	b.WriteString("  Receivers: " + s.ReceiverNaming + "\n")
	b.WriteString("  Tests: " + s.TestStyle + "\n")
	b.WriteString("  Errors: " + s.ErrorStyle + "\n")
	b.WriteString("  Doc comments: " + s.CommentDensity + "\n")
	if len(s.CommonPrefixes) > 0 {
		b.WriteString("  Common prefixes: " + strings.Join(s.CommonPrefixes, ", ") + "\n")
	}
	return b.String()
}
