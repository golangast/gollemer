package analyze

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
)

// Improvement is a suggested fix for a function.
type Improvement struct {
	Kind        string
	Description string
	Suggestion  string
	Line        int
}

// SuggestImprovements analyzes a function's source and returns concrete improvements.
func SuggestImprovements(src string) []Improvement {
	var out []Improvement
	fset := token.NewFileSet()
	wrapped := src
	if !strings.Contains(src, "package ") {
		wrapped = "package p\n" + src
	}
	f, err := parser.ParseFile(fset, "src.go", wrapped, 0)
	if err != nil {
		return out
	}

	conc := AnalyzeConcurrencySource(src)
	for _, risk := range conc.Risks {
		out = append(out, Improvement{
			Kind:        "concurrency",
			Description: risk,
			Suggestion:  concurrencyFix(risk),
		})
	}

	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.ExprStmt:
			if call, ok := x.X.(*ast.CallExpr); ok {
				name := callName(call)
				if isLikelyErrorReturn(name) {
					out = append(out, Improvement{
						Kind:        "bug",
						Description: fmt.Sprintf("ignored return from %q", name),
						Suggestion:  "Check the error: if err := ...; err != nil { return err }",
						Line:        fset.Position(x.Pos()).Line,
					})
				}
			}
		case *ast.FuncDecl:
			if x.Body != nil && len(x.Body.List) == 0 {
				out = append(out, Improvement{
					Kind:        "idiom",
					Description: fmt.Sprintf("empty function %q", x.Name.Name),
					Suggestion:  stubSuggestion(x.Name.Name),
					Line:        fset.Position(x.Pos()).Line,
				})
			}
		}
		return true
	})
	return out
}

func concurrencyFix(risk string) string {
	if strings.Contains(risk, "goroutine leak") {
		return "Use sync.WaitGroup for coordination"
	}
	if strings.Contains(risk, "Unlock") {
		return "Add defer mu.Unlock() after Lock"
	}
	return "Review concurrency"
}

func isLikelyErrorReturn(name string) bool {
	lower := strings.ToLower(name)
	for _, s := range []string{"close", "write", "flush", "sync", "remove"} {
		if strings.Contains(lower, s) {
			return true
		}
	}
	return false
}

func stubSuggestion(name string) string {
	lower := strings.ToLower(name)
	if strings.HasPrefix(lower, "get") {
		return "Implement as getter"
	}
	return "Implement the body"
}

// FormatImprovements returns a human-readable summary.
func FormatImprovements(imps []Improvement) string {
	if len(imps) == 0 {
		return "No improvements suggested."
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Found %d:\n", len(imps)))
	for i, imp := range imps {
		b.WriteString(fmt.Sprintf("%d. [%s] %s -> %s\n", i+1, imp.Kind, imp.Description, imp.Suggestion))
	}
	return b.String()
}
