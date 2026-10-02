package analyze

import (
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
