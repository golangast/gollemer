package analyze

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// This file makes "where do I change?" easy for beginners. Instead of a
// ranked list of hits, GuideChange gives one guided answer:
//
//	TO ADD "a new chat command":
//
//	START HERE: chat/real_seq2seq_chat.go:341
//	  func chat.tryDeterministicAnswer — ...
//
//	PUT IT HERE: inside tryDeterministicAnswer there's a chain of
//	handler checks — add yours next to these:
//	  • tryBareMakeTarget(...)
//	  • ...
//
//	YOU ARE HERE:
//	  main → runRealChat → tryDeterministicAnswer → yours here
//
// The anchor is the top WhereToChange hit; the "put it here" section is
// found by scanning the anchor for dispatch shapes (switch statements
// and if-chains comparing against literals) — the places where a
// codebase expects new cases to be added.

// dispatchSite is one place inside a function where new cases get added.
type dispatchSite struct {
	kind     string   // "switch" or "chain"
	line     int      // line of the switch/if
	examples []string // existing case labels or conditions, truncated
	more     int      // additional cases not shown
}

// GuideChange renders the beginner-guided "where do I change" answer
// for task. It falls back to the ranked list when no function anchor
// is found.
func (p *Project) GuideChange(task string) string {
	hits := p.WhereToChange(task)
	if len(hits) == 0 {
		return p.WhereToChangeText(task)
	}
	var anchor *Func
	for _, h := range hits {
		if strings.HasPrefix(h.What, "func ") {
			if fn, _, _, _ := p.resolve(strings.TrimPrefix(h.What, "func ")); fn != nil {
				anchor = fn
				break
			}
		}
	}
	if anchor == nil {
		return p.WhereToChangeText(task)
	}

	var b strings.Builder
	fmt.Fprintf(&b, "TO ADD %q:\n\n", task)
	fmt.Fprintf(&b, "START HERE: %s:%d\n", p.LinkPath(anchor.File), anchor.Line)
	fmt.Fprintf(&b, "  %s", anchor.Sig)
	if d := firstSentence(anchor.Doc); d != "" {
		fmt.Fprintf(&b, " — %s", d)
	}
	b.WriteString("\n")
	if why := hits[0].Why; why != "" {
		fmt.Fprintf(&b, "  Why here: %s.\n", why)
	}

	if sites := findDispatchSites(p.Root, anchor); len(sites) > 0 {
		b.WriteString("\nPUT IT HERE:\n")
		for _, s := range sites {
			where := "switch"
			if s.kind == "chain" {
				where = "chain of checks"
			}
			fmt.Fprintf(&b, "  Inside %s there's a %s (line %d) — add yours next to these:\n", anchor.Name, where, s.line)
			for _, ex := range s.examples {
				fmt.Fprintf(&b, "    • %s\n", ex)
			}
			if s.more > 0 {
				fmt.Fprintf(&b, "    • … (+%d more)\n", s.more)
			}
		}
	}

	if chain := callerChain(p, anchor, 5); len(chain) > 1 {
		b.WriteString("\nYOU ARE HERE:\n  ")
		names := make([]string, 0, len(chain))
		for _, fn := range chain {
			names = append(names, fn.Name)
		}
		b.WriteString(strings.Join(names, " → "))
		b.WriteString(" → yours here\n")
	}
	return b.String()
}

// findDispatchSites parses the anchor's file and looks for extension
// points in its body: switch statements and if/else chains that compare
// against literals — the shapes where new cases get added.
func findDispatchSites(root string, fn *Func) []dispatchSite {
	src, err := os.ReadFile(filepath.Join(root, fn.File))
	if err != nil {
		return nil
	}
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, fn.File, src, 0)
	if err != nil {
		return nil
	}
	decl := findFuncDecl(file, fn)
	if decl == nil || decl.Body == nil {
		return nil
	}
	var sites []dispatchSite
	ast.Inspect(decl.Body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.SwitchStmt:
			if ex, more := switchExamples(fset, src, s); len(ex) >= 2 {
				sites = append(sites, dispatchSite{
					kind:     "switch",
					line:     fset.Position(s.Pos()).Line,
					examples: ex,
					more:     more,
				})
			}
		case *ast.IfStmt:
			if ex, more, ok := chainExamples(fset, src, s); ok && len(ex) >= 2 {
				sites = append(sites, dispatchSite{
					kind:     "chain",
					line:     fset.Position(s.Pos()).Line,
					examples: ex,
					more:     more,
				})
			}
		}
		return true
	})
	return sites
}

// findFuncDecl locates the declaration matching fn by name and receiver.
func findFuncDecl(file *ast.File, fn *Func) *ast.FuncDecl {
	var found *ast.FuncDecl
	ast.Inspect(file, func(n ast.Node) bool {
		d, ok := n.(*ast.FuncDecl)
		if !ok || d.Name.Name != fn.Name {
			return true
		}
		hasRecv := d.Recv != nil && len(d.Recv.List) > 0
		if (fn.Receiver != "") != hasRecv {
			return true
		}
		found = d
		return false
	})
	return found
}

// switchExamples collects the case labels of a switch, truncated.
func switchExamples(fset *token.FileSet, src []byte, s *ast.SwitchStmt) ([]string, int) {
	var labels []string
	for _, stmt := range s.Body.List {
		cc, ok := stmt.(*ast.CaseClause)
		if !ok {
			continue
		}
		for _, e := range cc.List {
			if lit, ok := e.(*ast.BasicLit); ok {
				labels = append(labels, "case "+lit.Value+":")
			}
		}
	}
	return capExamples(labels, 5)
}

// chainExamples follows an if/else-if chain and collects the conditions
// that compare against a literal, e.g. `if line == "/flow"`.
func chainExamples(fset *token.FileSet, src []byte, s *ast.IfStmt) ([]string, int, bool) {
	var conds []string
	cur := s
	for cur != nil {
		if comparesLiteral(cur.Cond) {
			conds = append(conds, "if "+truncateRunes(nodeText(fset, src, cur.Cond), 56))
		}
		els, ok := cur.Else.(*ast.IfStmt)
		if !ok {
			break
		}
		cur = els
	}
	ex, more := capExamples(conds, 5)
	return ex, more, true
}

// comparesLiteral reports whether e is `x == "lit"` or `"lit" == x`.
func comparesLiteral(e ast.Expr) bool {
	bin, ok := e.(*ast.BinaryExpr)
	if !ok {
		return false
	}
	if bin.Op != token.EQL && bin.Op != token.NEQ {
		return false
	}
	_, lLit := bin.X.(*ast.BasicLit)
	_, rLit := bin.Y.(*ast.BasicLit)
	return lLit || rLit
}

// capExamples keeps the first n examples and counts the rest.
func capExamples(in []string, n int) ([]string, int) {
	if len(in) <= n {
		return in, 0
	}
	return in[:n], len(in) - n
}

// nodeText slices the source text of a node.
func nodeText(fset *token.FileSet, src []byte, n ast.Node) string {
	start := fset.Position(n.Pos()).Offset
	end := fset.Position(n.End()).Offset
	if start < 0 || end > len(src) || start >= end {
		return ""
	}
	return strings.TrimSpace(string(src[start:end]))
}

// callerChain walks from fn up through its most important caller to a
// root, returning the chain root-first. It caps at max functions.
func callerChain(p *Project, fn *Func, max int) []*Func {
	chain := []*Func{fn}
	seen := map[string]bool{fn.ID: true}
	cur := fn
	for len(chain) < max {
		var best *Func
		for _, id := range cur.Callers {
			c := p.byID[id]
			if c == nil || seen[c.ID] {
				continue
			}
			// Test callers are noise in a you-are-here chain: a beginner
			// wants the production path, not TestXxx.
			if strings.HasSuffix(c.File, "_test.go") {
				continue
			}
			if best == nil || c.Score > best.Score {
				best = c
			}
		}
		if best == nil {
			break
		}
		seen[best.ID] = true
		chain = append([]*Func{best}, chain...)
		cur = best
	}
	// Prefer a chain that starts at main; otherwise keep what we have.
	for i, fn := range chain {
		if fn.IsMain {
			return chain[i:]
		}
	}
	return chain
}

// truncateRunes shortens s to at most n runes.
func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return strings.TrimSpace(string(r[:n-1])) + "…"
}
