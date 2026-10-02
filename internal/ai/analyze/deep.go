package analyze

// This file turns raw AST facts into plain-English descriptions a new
// programmer can read at a glance: rendered signatures, one-line
// summaries, full detail cards, source excerpts, and a project overview
// paragraph. Everything here is derived from the parsed code and its
// doc comments — nothing is guessed.

import (
	"fmt"
	"go/ast"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// renderSig renders e.g. "func (c *Chat) Add(a int) string".
func renderSig(d *ast.FuncDecl, recv string) string {
	var b strings.Builder
	b.WriteString("func ")
	if d.Recv != nil && len(d.Recv.List) > 0 {
		b.WriteString("(")
		b.WriteString(fieldListString(d.Recv.List))
		b.WriteString(") ")
	}
	b.WriteString(d.Name.Name)
	b.WriteString("(")
	if d.Type.Params != nil {
		b.WriteString(fieldListString(d.Type.Params.List))
	}
	b.WriteString(")")
	if d.Type.Results != nil && len(d.Type.Results.List) > 0 {
		res := d.Type.Results.List
		if len(res) == 1 && len(res[0].Names) == 0 {
			b.WriteString(" " + typeString(res[0].Type))
		} else {
			b.WriteString(" (" + fieldListString(res) + ")")
		}
	}
	return b.String()
}

// fieldListString renders "a int, b string" for a parameter/result list.
func fieldListString(fields []*ast.Field) string {
	var parts []string
	for _, f := range fields {
		t := typeString(f.Type)
		if len(f.Names) == 0 {
			parts = append(parts, t)
			continue
		}
		var names []string
		for _, n := range f.Names {
			names = append(names, n.Name)
		}
		parts = append(parts, strings.Join(names, ", ")+" "+t)
	}
	return strings.Join(parts, ", ")
}

// typeString renders a type expression in short Go syntax.
func typeString(e ast.Expr) string {
	switch t := e.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.StarExpr:
		return "*" + typeString(t.X)
	case *ast.SelectorExpr:
		return typeString(t.X) + "." + t.Sel.Name
	case *ast.ArrayType:
		if t.Len == nil {
			return "[]" + typeString(t.Elt)
		}
		return "[...]" + typeString(t.Elt)
	case *ast.Ellipsis:
		return "..." + typeString(t.Elt)
	case *ast.MapType:
		return "map[" + typeString(t.Key) + "]" + typeString(t.Value)
	case *ast.ChanType:
		return "chan " + typeString(t.Value)
	case *ast.FuncType:
		return "func(...)"
	case *ast.InterfaceType:
		if t.Methods == nil || len(t.Methods.List) == 0 {
			return "any"
		}
		return "interface{...}"
	case *ast.StructType:
		return "struct{...}"
	case *ast.ParenExpr:
		return typeString(t.X)
	}
	return ""
}

// firstSentence returns the first sentence of s, trimmed.
func firstSentence(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	for i, r := range s {
		if r == '.' || r == '!' || r == '?' {
			if i+1 >= len(s) || s[i+1] == ' ' {
				return strings.TrimSpace(s[:i+1])
			}
		}
	}
	if len(s) > 140 {
		return s[:137] + "..."
	}
	return s
}

// Summary is the one-line plain-English description of a function: its
// doc comment's first sentence when it has one, otherwise an honest
// synthesis of what it calls.
func (fn *Func) Summary() string {
	if s := firstSentence(fn.Doc); s != "" {
		return s
	}
	if fn.IsMain {
		return "Program entry point."
	}
	if fn.IsInit {
		return "Package initializer (runs automatically on import)."
	}
	if len(fn.Calls) > 0 {
		names := make([]string, 0, 3)
		for _, c := range fn.Calls {
			if len(names) >= 3 {
				break
			}
			names = append(names, shortID(c))
		}
		more := ""
		if len(fn.Calls) > 3 {
			more = fmt.Sprintf(", and %d more", len(fn.Calls)-3)
		}
		return fmt.Sprintf("Undocumented; calls %s%s.", strings.Join(names, ", "), more)
	}
	return "Undocumented; see the source."
}

// Detail renders the full readable card for a function.
func (fn *Func) Detail(p *Project) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s\n", fn.Sig)
	fmt.Fprintf(&b, "  %s:%d\n", p.LinkPath(fn.File), fn.Line)
	fmt.Fprintf(&b, "  %s\n", fn.Summary())
	if len(fn.Calls) > 0 {
		fmt.Fprintf(&b, "  Calls (%d): %s\n", len(fn.Calls), joinShort(fn.Calls, 6))
	}
	if len(fn.Callers) > 0 {
		fmt.Fprintf(&b, "  Called by (%d): %s\n", len(fn.Callers), joinShort(fn.Callers, 6))
	} else if !fn.IsMain && !fn.IsTest {
		b.WriteString("  Called by: nobody in this project (entry point or dead code)\n")
	}
	return b.String()
}

// Detail renders the full readable card for a type.
func (t *Type) Detail(p *Project) string {
	var b strings.Builder
	fmt.Fprintf(&b, "type %s.%s — %s\n", t.Pkg, t.Name, t.Kind)
	fmt.Fprintf(&b, "  %s:%d\n", p.LinkPath(t.File), t.Line)
	if s := firstSentence(t.Doc); s != "" {
		fmt.Fprintf(&b, "  %s\n", s)
	}
	if len(t.Fields) > 0 {
		fmt.Fprintf(&b, "  Fields (%d): %s\n", len(t.Fields), strings.Join(limitStr(t.Fields, 10), ", "))
	}
	methods := p.methodsOf(t)
	if len(methods) > 0 {
		fmt.Fprintf(&b, "  Methods (%d): %s\n", len(methods), strings.Join(limitStr(methods, 10), ", "))
	}
	if t.Kind == "interface" && t.ImplCount > 0 {
		fmt.Fprintf(&b, "  Implemented by ~%d types\n", t.ImplCount)
	}
	return b.String()
}

// PackageDetail renders the readable card for a package.
func (p *Project) PackageDetail(pkg *Package) string {
	var b strings.Builder
	dir := pkg.Dir
	if dir == "" {
		dir = "."
	}
	fmt.Fprintf(&b, "package %s — %s\n", pkg.Name, dir)
	if s := firstSentence(pkg.Doc); s != "" {
		fmt.Fprintf(&b, "  %s\n", s)
	}
	fmt.Fprintf(&b, "  %d functions, %d types, %d files\n", len(pkg.Funcs), len(pkg.Types), len(pkg.Files))
	if len(pkg.Internal) > 0 {
		fmt.Fprintf(&b, "  Uses: %s\n", strings.Join(pkg.Internal, ", "))
	}
	// Key functions: top-scoring non-test functions with summaries.
	var fns []*Func
	for _, fn := range pkg.Funcs {
		if !fn.IsTest {
			fns = append(fns, fn)
		}
	}
	sort.Slice(fns, func(i, j int) bool { return fns[i].Score > fns[j].Score })
	if len(fns) > 0 {
		b.WriteString("  Key functions:\n")
		for i, fn := range fns {
			if i >= 5 {
				break
			}
			fmt.Fprintf(&b, "    %s — %s\n", fn.Display(), fn.Summary())
		}
	}
	return b.String()
}

// SourceOf returns up to maxLines of the function's source code.
func (p *Project) SourceOf(fn *Func, maxLines int) string {
	data, err := os.ReadFile(filepath.Join(p.Root, fn.File))
	if err != nil {
		return ""
	}
	lines := strings.Split(string(data), "\n")
	start := fn.Line - 1
	if start < 0 || start >= len(lines) {
		return ""
	}
	end := fn.EndLine
	if end <= start {
		end = start + 1
	}
	if end-start > maxLines {
		end = start + maxLines
	}
	if end > len(lines) {
		end = len(lines)
	}
	out := strings.Join(lines[start:end], "\n")
	if fn.EndLine-start > maxLines {
		out += fmt.Sprintf("\n  ... (%d more lines)", fn.EndLine-end)
	}
	return out
}

// Overview returns a plain-English paragraph describing the project for
// someone who has never seen it: what it is, where it starts, what the
// engine room is, and the order to read it in.
func (p *Project) Overview() string {
	name := p.Module
	if name == "" {
		name = filepathBase(p.Root)
	}
	if i := strings.LastIndex(name, "/"); i >= 0 {
		name = name[i+1:]
	}
	var b strings.Builder
	mains := p.EntryPoints()
	// The root main.go is "the" entry point when there are several.
	start := ""
	if len(mains) > 0 {
		start = mains[0].File
		for _, m := range mains {
			if m.PkgDir == "" {
				start = m.File
				break
			}
		}
	}
	if len(mains) == 0 {
		fmt.Fprintf(&b, "%s is a Go library (no runnable main): %d packages, about %s lines of Go. ",
			name, len(p.Packages), commas(p.Lines))
	} else {
		fmt.Fprintf(&b, "%s is a runnable Go program: %d packages, about %s lines of Go. ",
			name, len(p.Packages), commas(p.Lines))
		fmt.Fprintf(&b, "It starts in %s. ", p.LinkPath(start))
	}
	// One-liner from the root/main package doc, if the author wrote one.
	if one := p.projectOneLiner(); one != "" {
		fmt.Fprintf(&b, "In the author's words: %s ", one)
	}
	if top := p.TopFuncs(1); len(top) > 0 && len(top[0].Callers) > 0 {
		fn := top[0]
		fmt.Fprintf(&b, "The hardest-working function is %s, called from %d places — that's the engine room, the best place to look when something breaks. ",
			fn.Display(), len(fn.Callers))
	}
	chain := p.readingChain(6)
	if len(chain) > 1 {
		fmt.Fprintf(&b, "Read the code in this order: %s, because each layer builds on the one before it.",
			strings.Join(chain, " → "))
	}
	return b.String()
}

// projectOneLiner finds a human-written one-line description: the root
// package doc first, then the first main package's doc.
func (p *Project) projectOneLiner() string {
	if pkg := p.byPkgDir[""]; pkg != nil {
		if s := firstSentence(pkg.Doc); s != "" {
			return s
		}
	}
	for _, pkg := range p.Packages {
		if pkg.HasMain {
			if s := firstSentence(pkg.Doc); s != "" {
				return s
			}
		}
	}
	return ""
}

// readingChain returns short package names in dependency-first order.
func (p *Project) readingChain(n int) []string {
	var out []string
	for _, dir := range p.readingOrderDirs() {
		if dir == "" {
			out = append(out, "main")
			continue
		}
		short := dir
		if i := strings.LastIndex(dir, "/"); i >= 0 {
			short = dir[i+1:]
		}
		out = append(out, short)
		if len(out) >= n {
			break
		}
	}
	return out
}

// shortID renders "internal.ai.training.chat.routeDomain" as "chat.routeDomain".
func shortID(id string) string {
	parts := strings.Split(id, ".")
	if len(parts) >= 2 {
		return parts[len(parts)-2] + "." + parts[len(parts)-1]
	}
	return id
}

// joinShort renders callee/caller ID lists, capped at n with a "+k more".
func joinShort(ids []string, n int) string {
	var names []string
	for _, id := range ids {
		names = append(names, shortID(id))
	}
	sort.Strings(names)
	if len(names) > n {
		return strings.Join(names[:n], ", ") + fmt.Sprintf(" (+%d more)", len(names)-n)
	}
	return strings.Join(names, ", ")
}

func limitStr(ss []string, n int) []string {
	if len(ss) > n {
		return ss[:n]
	}
	return ss
}

func commas(n int) string {
	s := fmt.Sprintf("%d", n)
	var out []byte
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			out = append(out, ',')
		}
		out = append(out, byte(c))
	}
	return string(out)
}
