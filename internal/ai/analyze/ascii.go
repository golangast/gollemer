package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// RenderASCIIGraph draws the package dependency graph as plain text for
// the terminal: one block per package, listing the other project
// packages it imports. Entry-point packages (func main) are marked with
// ★. Read "A → B" as "A imports B".
func RenderASCIIGraph(p *Project) string {
	var b strings.Builder
	name := p.Module
	if name == "" {
		name = p.Root
	}
	fmt.Fprintf(&b, "Dependency graph: %s\n", name)
	b.WriteString("A → B means A imports B. ★ = entry point (func main).\n\n")

	mains := map[string]bool{}
	for _, m := range p.EntryPoints() {
		mains[m.PkgDir] = true
	}

	pkgs := append([]*Package(nil), p.Packages...)
	sort.Slice(pkgs, func(i, j int) bool { return pkgs[i].Dir < pkgs[j].Dir })

	for _, pkg := range pkgs {
		label := pkg.Dir
		if label == "" {
			label = "(root)"
		}
		star := " "
		if mains[pkg.Dir] {
			star = "★"
		}
		fmt.Fprintf(&b, "%s 📦 %s\n", star, label)
		deps := append([]string(nil), pkg.Internal...)
		sort.Strings(deps)
		for i, d := range deps {
			branch := "├──"
			if i == len(deps)-1 {
				branch = "└──"
			}
			fmt.Fprintf(&b, "  %s → %s\n", branch, d)
		}
		if len(deps) == 0 {
			b.WriteString("  └── (no project imports)\n")
		}
	}
	return b.String()
}
