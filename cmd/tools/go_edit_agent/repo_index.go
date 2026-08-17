package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// MethodLoc describes where a method is declared in the repo.
type MethodLoc struct {
	Path        string // file path
	PackageName string // package name
	Pointer     bool   // receiver pointer?
}

// buildRepoSymbolIndex walks the project root and collects top-level
// function/type names and the files they are defined in.
func buildRepoSymbolIndex(root string) map[string][]string {
	index := make(map[string][]string)
	fset := token.NewFileSet()
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			// skip vendor and .git
			if d.Name() == "vendor" || d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		_, perr := parser.ParseFile(fset, path, nil, 0)
		if perr != nil {
			return nil
		}
		// we rely on a simple text scan below; skip using node.Decls here
		// fallback: scan file text for "func <name>" patterns
		bs, rerr := os.ReadFile(path)
		if rerr != nil {
			return nil
		}
		text := string(bs)
		// naive scan
		for _, line := range strings.Split(text, "\n") {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "func ") {
				// extract token after func
				rest := strings.TrimPrefix(line, "func ")
				name := ""
				for i, ch := range rest {
					if ch == '(' || ch == ' ' || ch == '{' || ch == '\t' {
						name = rest[:i]
						break
					}
				}
				if name == "" {
					continue
				}
				// strip receiver like (r *Type)Name -> if name contains ')', take after
				if idx := strings.LastIndex(name, ")"); idx != -1 && idx+1 < len(name) {
					name = name[idx+1:]
				}
				if name != "" {
					index[name] = append(index[name], path)
				}
			}
			// types
			if strings.HasPrefix(line, "type ") {
				rest := strings.TrimPrefix(line, "type ")
				parts := strings.Fields(rest)
				if len(parts) > 0 {
					tname := parts[0]
					index[tname] = append(index[tname], path)
				}
			}
		}
		return nil
	})
	// normalize order: prefer files in same package/directory grouping by path depth
	for name, paths := range index {
		sort.Slice(paths, func(i, j int) bool { return len(paths[i]) < len(paths[j]) })
		index[name] = paths
	}
	return index
}

// buildRepoMethodsByType builds a map receiverType -> map[methodName][]filepaths
func buildRepoMethodsByType(root string) map[string]map[string][]MethodLoc {
	out := make(map[string]map[string][]MethodLoc)
	fset := token.NewFileSet()
	filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			if d.Name() == "vendor" || d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		node, perr := parser.ParseFile(fset, path, nil, 0)
		if perr != nil {
			return nil
		}
		for _, decl := range node.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok {
				if fn.Recv != nil && len(fn.Recv.List) > 0 {
					// get receiver type name
					var recvName string
					var ptr bool
					switch expr := fn.Recv.List[0].Type.(type) {
					case *ast.StarExpr:
						if id, ok := expr.X.(*ast.Ident); ok {
							recvName = id.Name
							ptr = true
						}
					case *ast.Ident:
						recvName = expr.Name
						ptr = false
					}
					if recvName != "" {
						if out[recvName] == nil {
							out[recvName] = make(map[string][]MethodLoc)
						}
						pkgName := ""
						if node != nil && node.Name != nil {
							pkgName = node.Name.Name
						}
						out[recvName][fn.Name.Name] = append(out[recvName][fn.Name.Name], MethodLoc{Path: path, PackageName: pkgName, Pointer: ptr})
					}
				}
			}
		}
		return nil
	})
	// Normalize method lists
	for typ, m := range out {
		// ensure consistent ordering of keys
		_ = typ
		for mn, locs := range m {
			sort.Slice(locs, func(i, j int) bool { return len(locs[i].Path) < len(locs[j].Path) })
			m[mn] = locs
		}
		out[typ] = m
	}
	return out
}

// bestRepoPath chooses the most appropriate file path among candidates for a
// symbol, preferring same directory as currentFile, then shorter paths.
func bestRepoPath(candidates []string, currentFile string) string {
	if len(candidates) == 0 {
		return ""
	}
	if currentFile == "" {
		return candidates[0]
	}
	curDir := filepath.Dir(currentFile)
	// prefer candidate in same directory
	for _, p := range candidates {
		if filepath.Dir(p) == curDir {
			return p
		}
	}
	// otherwise prefer the shortest path
	best := candidates[0]
	for _, p := range candidates[1:] {
		if len(p) < len(best) {
			best = p
		}
	}
	return best
}

// bestMethodLoc chooses the best MethodLoc among candidates using heuristics.
func bestMethodLoc(cands []MethodLoc, currentFile string) *MethodLoc {
	if len(cands) == 0 {
		return nil
	}
	curDir := filepath.Dir(currentFile)
	// prefer same directory
	for _, m := range cands {
		if filepath.Dir(m.Path) == curDir {
			return &m
		}
	}
	// prefer same package name
	if currentFile != "" {
		// attempt to find package name of currentFile
		fset := token.NewFileSet()
		if node, err := parser.ParseFile(fset, currentFile, nil, 0); err == nil && node != nil {
			pkg := node.Name.Name
			for _, m := range cands {
				if m.PackageName == pkg {
					return &m
				}
			}
		}
	}
	// fallback to shortest path
	best := cands[0]
	for _, m := range cands[1:] {
		if len(m.Path) < len(best.Path) {
			best = m
		}
	}
	return &best
}

// findProjectRoot searches upward from startDir for a directory containing go.mod
func findProjectRoot(startDir string) string {
	dir := startDir
	for {
		gm := filepath.Join(dir, "go.mod")
		if _, err := os.Stat(gm); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return ""
		}
		dir = parent
	}
}
