package analyze

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// KeyDispatch is one switch statement that dispatches on keyboard input:
// `switch msg.String()` with key-like cases ("ctrl+c", "esc", "enter").
// In Go TUI programs (Bubble Tea and friends) this is where keyboard
// shortcuts live — new hotkeys are new cases next to the existing ones.
type KeyDispatch struct {
	File string   // relative to project root
	Line int      // line of the switch
	Pkg  string   // package dir, relative to root
	Func *Func    // enclosing function, when resolved
	Keys []string // existing key literals, in order of appearance
}

// keyLikeRe recognizes key literals: "ctrl+c", "esc", "enter", "q".
var keyLikeRe = regexp.MustCompile(`(?i)^((ctrl|alt|shift)\+[\w+]+|(esc|enter|space|tab|up|down|left|right|del|delete|backspace|home|end|pgup|pgdown|insert)|[a-z0-9])$`)

// FindKeyDispatchSites scans the project for keyboard dispatch switches:
// a switch with at least two key-like string cases. Test files are
// skipped — shortcuts belong to production code. Sites are ranked by
// key count: the richest dispatch is the main one (App.Update with
// ctrl+c/q/esc/enter outranks a y/n confirmation prompt).
func (p *Project) FindKeyDispatchSites() []KeyDispatch {
	var out []KeyDispatch
	for _, pkg := range p.Packages {
		byFunc := map[string]*Func{}
		for _, fn := range pkg.Funcs {
			byFunc[fn.File+"\x00"+fn.Name] = fn
		}
		for _, rel := range pkg.Files {
			if strings.HasSuffix(rel, "_test.go") {
				continue
			}
			full := filepath.Join(p.Root, rel)
			src, err := os.ReadFile(full)
			if err != nil {
				continue
			}
			fset := token.NewFileSet()
			f, err := parser.ParseFile(fset, full, src, 0)
			if err != nil {
				continue
			}
			for _, decl := range f.Decls {
				fd, ok := decl.(*ast.FuncDecl)
				if !ok {
					continue
				}
				name := fd.Name.Name
				ast.Inspect(fd.Body, func(n ast.Node) bool {
					sw, ok := n.(*ast.SwitchStmt)
					if !ok {
						return true
					}
					var keys []string
					seen := map[string]bool{}
					for _, stmt := range sw.Body.List {
						cc, ok := stmt.(*ast.CaseClause)
						if !ok {
							continue
						}
						for _, e := range cc.List {
							lit, ok := e.(*ast.BasicLit)
							if !ok || lit.Kind != token.STRING {
								continue
							}
							k := strings.Trim(lit.Value, `"`)
							k = strings.Trim(k, "`")
							if keyLikeRe.MatchString(k) && !seen[k] {
								seen[k] = true
								keys = append(keys, k)
							}
						}
					}
					if len(keys) >= 2 {
						out = append(out, KeyDispatch{
							File: rel,
							Line: fset.Position(sw.Pos()).Line,
							Pkg:  pkg.Dir,
							Func: byFunc[rel+"\x00"+name],
							Keys: keys,
						})
					}
					return true
				})
			}
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if len(out[i].Keys) != len(out[j].Keys) {
			return len(out[i].Keys) > len(out[j].Keys)
		}
		return out[i].File < out[j].File
	})
	return out
}

// wantsKeys reports whether the issue asks for keyboard shortcuts.
func wantsKeys(c IssueConcepts) bool {
	text := strings.ToLower(c.Title + "\n" + c.Body)
	for _, w := range []string{"keyboard", "shortcut", "hotkey", "hot-key"} {
		if strings.Contains(text, w) {
			return true
		}
	}
	return false
}

// keyLiterals normalizes the issue's backticked hotkeys (`Ctrl+A`)
// into the case-literal form the code uses ("ctrl+a").
func keyLiterals(c IssueConcepts) []string {
	var out []string
	seen := map[string]bool{}
	for _, w := range c.CodeWords {
		k := strings.ToLower(strings.TrimSpace(w))
		if k == "" || seen[k] {
			continue
		}
		seen[k] = true
		if keyLikeRe.MatchString(k) || strings.Contains(k, "+") {
			out = append(out, k)
		}
	}
	return out
}

// recvName renders "App.Update" for a method, "Update" for a function.
func recvName(fn *Func) string {
	if fn.Receiver != "" {
		return fn.Receiver + "." + fn.Name
	}
	return fn.Name
}

// writeKeyVisual draws how a keypress travels through the program — the
// pressed key, the message the toolkit makes of it, and the chain of
// functions down to the switch — followed by a plain-words explanation
// of the pattern. Every line is grounded in the analyzed code: the
// chain comes from the call graph, the switch from the dispatch scan.
func writeKeyVisual(b *strings.Builder, p *Project, c IssueConcepts, sites []KeyDispatch) {
	top := sites[0]
	lits := keyLiterals(c)
	pressed := "a key"
	if len(lits) > 0 {
		pressed = "`" + lits[0] + "`"
	}
	framework, msgType := "the UI toolkit", "a key message"
	if top.Func != nil && strings.Contains(top.Func.Sig, "tea.KeyMsg") {
		framework, msgType = "Bubble Tea", "tea.KeyMsg"
	}
	var chain []*Func
	if top.Func != nil {
		chain = callerChain(p, top.Func, 4)
	}

	b.WriteString("HOW A KEYPRESS FLOWS:\n\n")
	fmt.Fprintf(b, "  you press %s\n", pressed)
	b.WriteString("        ↓\n")
	fmt.Fprintf(b, "  %s — %s turns the press into a message\n", msgType, framework)
	for _, fn := range chain {
		b.WriteString("        ↓\n")
		if fn == top.Func {
			fmt.Fprintf(b, "  %s (%s:%d) ← the switch\n", recvName(fn), top.File, top.Line)
		} else {
			fmt.Fprintf(b, "  %s (%s)\n", recvName(fn), fn.File)
		}
	}
	b.WriteString("        ↓\n")
	if len(lits) > 0 {
		fmt.Fprintf(b, "  your new case %s → yours here\n\n", strings.Join(quoteAll(lits), ", "))
	} else {
		b.WriteString("  your new case → yours here\n\n")
	}

	// Plain-words explanation of the pattern, from the real code.
	b.WriteString("IN PLAIN WORDS:\n")
	if len(chain) > 0 && top.Func != nil {
		var sample []string
		if len(top.Keys) > 3 {
			sample = top.Keys[:3]
		} else {
			sample = top.Keys
		}
		fw := strings.ToUpper(framework[:1]) + framework[1:]
		fmt.Fprintf(b, "  Nothing here reads the keyboard directly. %s turns each\n", fw)
		if len(chain) > 1 {
			fmt.Fprintf(b, "  keypress into a message and hands it to %s,\n", recvName(chain[0]))
			fmt.Fprintf(b, "  which passes it to %s.\n", recvName(top.Func))
		} else {
			fmt.Fprintf(b, "  keypress into a message and hands it to %s.\n", recvName(top.Func))
		}
		fmt.Fprintf(b, "  That function is one big switch matching key names like %s.\n",
			strings.Join(quoteAll(sample), ", "))
		b.WriteString("  A keyboard shortcut is just a new case in that switch:\n")
		b.WriteString("  when the key matches, your code runs.\n\n")
	} else {
		b.WriteString("  Nothing here reads the keyboard directly. The toolkit turns\n")
		b.WriteString("  each keypress into a message and delivers it to a key-handling\n")
		b.WriteString("  function — one big switch matching key names. A keyboard\n")
		b.WriteString("  shortcut is just a new case in that switch: when the key\n")
		b.WriteString("  matches, your code runs.\n\n")
	}
}
