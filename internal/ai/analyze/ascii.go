package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// RenderASCIIGraph draws the package dependency graph as plain text for
// the terminal. Simplified for beginners (2026-10-20): only packages
// that actually connect to something get a block — entry points and
// packages with internal imports, sorted by how much the rest of the
// project depends on them. Packages that import nothing internal are
// folded into a single summary line instead of one dead block each,
// and the old "(no project imports)" line is gone entirely: absence
// means absence. Read "A → B" as "A imports B".
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

	// Clean dep lists (an empty string slipped in as a blank arrow in
	// real output) and count how many packages import each package.
	cleanDeps := func(pkg *Package) []string {
		var deps []string
		for _, d := range pkg.Internal {
			if strings.TrimSpace(d) != "" {
				deps = append(deps, d)
			}
		}
		sort.Strings(deps)
		return deps
	}
	indegree := map[string]int{}
	for _, pkg := range p.Packages {
		for _, d := range cleanDeps(pkg) {
			indegree[d]++
		}
	}

	var blocks, leaves []*Package
	for _, pkg := range p.Packages {
		if mains[pkg.Dir] || len(cleanDeps(pkg)) > 0 {
			blocks = append(blocks, pkg)
		} else {
			leaves = append(leaves, pkg)
		}
	}
	// Entry points first, then most-imported (the hubs), then name.
	sort.Slice(blocks, func(i, j int) bool {
		if mains[blocks[i].Dir] != mains[blocks[j].Dir] {
			return mains[blocks[i].Dir]
		}
		if indegree[blocks[i].Dir] != indegree[blocks[j].Dir] {
			return indegree[blocks[i].Dir] > indegree[blocks[j].Dir]
		}
		return blocks[i].Dir < blocks[j].Dir
	})

	for _, pkg := range blocks {
		label := pkg.Dir
		if label == "" {
			label = "(root)"
		}
		star := " "
		if mains[pkg.Dir] {
			star = "★"
		}
		fmt.Fprintf(&b, "%s 📦 %s\n", star, label)
		deps := cleanDeps(pkg)
		for i, d := range deps {
			branch := "├──"
			if i == len(deps)-1 {
				branch = "└──"
			}
			fmt.Fprintf(&b, "  %s → %s\n", branch, d)
		}
	}

	// Fold the leaves into one line. Naming a few keeps the orientation
	// value; the long tail is counted, not listed.
	if len(leaves) > 0 {
		sort.Slice(leaves, func(i, j int) bool { return leaves[i].Dir < leaves[j].Dir })
		var names []string
		for _, pkg := range leaves {
			label := pkg.Dir
			if label == "" {
				label = "(root)"
			}
			names = append(names, label)
		}
		shown := names
		extra := 0
		if len(shown) > 8 {
			extra = len(shown) - 8
			shown = shown[:8]
		}
		fmt.Fprintf(&b, "… %d package(s) import nothing internal: %s", len(names), strings.Join(shown, ", "))
		if extra > 0 {
			fmt.Fprintf(&b, " (+%d more)", extra)
		}
		b.WriteString("\n")
	}
	return b.String()
}
