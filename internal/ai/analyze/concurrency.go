package analyze

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
)

// ConcurrencyReport describes the concurrency behavior of a function.
type ConcurrencyReport struct {
	LaunchesGoroutines bool
	GoroutineCount     int
	UsesChannels       bool
	ChannelOps         []string
	UsesMutex          bool
	MutexOps           []string
	UsesWaitGroup      bool
	UsesSelect         bool
	SelectCases        int
	Risks              []string
}

// AnalyzeConcurrencySource parses Go source (a single function or snippet)
// and reports concurrency primitives used.
func AnalyzeConcurrencySource(src string) *ConcurrencyReport {
	r := &ConcurrencyReport{}
	fset := token.NewFileSet()
	// Wrap in a package if it's just a function
	wrapped := src
	if !strings.Contains(src, "package ") {
		wrapped = "package p\n" + src
	}
	f, err := parser.ParseFile(fset, "src.go", wrapped, 0)
	if err != nil {
		return r
	}

	ast.Inspect(f, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.GoStmt:
			r.LaunchesGoroutines = true
			r.GoroutineCount++
		case *ast.SelectStmt:
			r.UsesSelect = true
			r.SelectCases += len(x.Body.List)
		case *ast.SendStmt:
			r.UsesChannels = true
			r.ChannelOps = append(r.ChannelOps, fmt.Sprintf("send on %s", exprStr(x.Chan)))
		case *ast.UnaryExpr:
			if x.Op.String() == "<-" {
				r.UsesChannels = true
				r.ChannelOps = append(r.ChannelOps, fmt.Sprintf("recv from %s", exprStr(x.X)))
			}
		case *ast.CallExpr:
			fname := callName(x)
			switch {
			case fname == "close":
				r.UsesChannels = true
				r.ChannelOps = append(r.ChannelOps, "close channel")
			case strings.HasSuffix(fname, ".Lock") || fname == "Lock":
				r.UsesMutex = true
				r.MutexOps = append(r.MutexOps, "Lock()")
			case strings.HasSuffix(fname, ".Unlock") || fname == "Unlock":
				r.UsesMutex = true
				r.MutexOps = append(r.MutexOps, "Unlock()")
			case strings.HasSuffix(fname, ".RLock"):
				r.UsesMutex = true
				r.MutexOps = append(r.MutexOps, "RLock()")
			case strings.HasSuffix(fname, ".RUnlock"):
				r.UsesMutex = true
				r.MutexOps = append(r.MutexOps, "RUnlock()")
			case strings.HasSuffix(fname, ".Add") || strings.HasSuffix(fname, ".Done") || strings.HasSuffix(fname, ".Wait"):
				r.UsesWaitGroup = true
			}
		case *ast.ChanType:
			r.UsesChannels = true
		}
		return true
	})

	if r.LaunchesGoroutines && !r.UsesWaitGroup && !r.UsesChannels {
		r.Risks = append(r.Risks, "goroutine leak risk: go without WaitGroup/channel coordination")
	}
	locks, unlocks := 0, 0
	for _, op := range r.MutexOps {
		if op == "Lock()" || op == "RLock()" {
			locks++
		}
		if op == "Unlock()" || op == "RUnlock()" {
			unlocks++
		}
	}
	if locks > unlocks {
		r.Risks = append(r.Risks, fmt.Sprintf("possible missing Unlock: %d Lock vs %d Unlock", locks, unlocks))
	}
	return r
}

// Describe returns a plain-words summary.
func (r *ConcurrencyReport) Describe() string {
	if !r.LaunchesGoroutines && !r.UsesChannels && !r.UsesMutex && !r.UsesWaitGroup && !r.UsesSelect {
		return ""
	}
	parts := []string{}
	if r.LaunchesGoroutines {
		parts = append(parts, fmt.Sprintf("launches %d goroutine(s)", r.GoroutineCount))
	}
	if r.UsesChannels {
		parts = append(parts, fmt.Sprintf("uses channels (%d ops)", len(r.ChannelOps)))
	}
	if r.UsesMutex {
		parts = append(parts, "uses mutex")
	}
	if r.UsesWaitGroup {
		parts = append(parts, "uses WaitGroup")
	}
	if r.UsesSelect {
		parts = append(parts, fmt.Sprintf("select with %d cases", r.SelectCases))
	}
	s := "Concurrency: " + strings.Join(parts, ", ") + "."
	if len(r.Risks) > 0 {
		s += " Risks: " + strings.Join(r.Risks, "; ") + "."
	}
	return s
}

func exprStr(e ast.Expr) string {
	switch x := e.(type) {
	case *ast.Ident:
		return x.Name
	case *ast.SelectorExpr:
		return exprStr(x.X) + "." + x.Sel.Name
	default:
		return "?"
	}
}

func callName(c *ast.CallExpr) string {
	switch f := c.Fun.(type) {
	case *ast.Ident:
		return f.Name
	case *ast.SelectorExpr:
		return exprStr(f.X) + "." + f.Sel.Name
	default:
		return ""
	}
}
