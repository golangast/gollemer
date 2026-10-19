package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// This file generates documentation from analyzed Go code: "document this
// project" or "document routeDomain". It takes the code and writes the docs
// — signatures, summaries, call relationships — in markdown. Everything is
// derived from the AST; nothing is guessed.

// Document renders full markdown documentation for the project: package
// overview, then every function and type with its signature and summary.
func (p *Project) Document() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("# %s\n\n", p.Module))
	if p.Module == "" {
		b.WriteString("# Project\n\n")
	}
	b.WriteString(fmt.Sprintf("%d packages, %d files, %d lines\n\n", len(p.Packages), p.Files, p.Lines))

	for _, pkg := range p.Packages {
		b.WriteString(fmt.Sprintf("## Package %s\n\n", pkg.Name))
		if pkg.Doc != "" {
			b.WriteString(pkg.Doc + "\n\n")
		}
		b.WriteString(fmt.Sprintf("Directory: `%s` — %d functions, %d types\n\n",
			pkg.Dir, len(pkg.Funcs), len(pkg.Types)))

		if len(pkg.Types) > 0 {
			b.WriteString("### Types\n\n")
			for _, t := range pkg.Types {
				b.WriteString(t.Document(p))
				b.WriteString("\n")
			}
		}

		if len(pkg.Funcs) > 0 {
			b.WriteString("### Functions\n\n")
			fns := append([]*Func{}, pkg.Funcs...)
			sort.Slice(fns, func(i, j int) bool { return fns[i].Name < fns[j].Name })
			for _, fn := range fns {
				if fn.IsTest {
					continue
				}
				b.WriteString(fn.Document(p))
				b.WriteString("\n")
			}
		}
	}
	return b.String()
}

// Document renders markdown documentation for a single function: signature,
// location, summary, and what it calls.
func (fn *Func) Document(p *Project) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("#### `%s`\n\n", fn.Sig))
	b.WriteString(fmt.Sprintf("Defined at `%s:%d`\n\n", fn.File, fn.Line))
	if s := fn.Summary(); s != "" {
		b.WriteString(s + "\n\n")
	}
	if len(fn.Calls) > 0 {
		b.WriteString("Calls: ")
		var names []string
		seen := map[string]bool{}
		for _, id := range fn.Calls {
			if callee := p.byID[id]; callee != nil && !seen[callee.ID] {
				seen[callee.ID] = true
				names = append(names, "`"+callee.Display()+"`")
			}
			if len(names) >= 10 {
				break
			}
		}
		b.WriteString(strings.Join(names, ", ") + "\n\n")
	}
	return b.String()
}

// Document renders markdown documentation for a single type.
func (t *Type) Document(p *Project) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("#### `%s.%s`\n\n", t.Pkg, t.Name))
	b.WriteString(fmt.Sprintf("Defined at `%s:%d`\n\n", t.File, t.Line))
	if s := firstSentence(t.Doc); s != "" {
		b.WriteString(s + "\n\n")
	}
	if len(t.Fields) > 0 {
		b.WriteString("Fields: `" + strings.Join(t.Fields, "`, `") + "`\n\n")
	}
	methods := p.methodsOf(t)
	if len(methods) > 0 {
		b.WriteString("Methods: `" + strings.Join(methods, "`, `") + "`\n\n")
	}
	return b.String()
}
