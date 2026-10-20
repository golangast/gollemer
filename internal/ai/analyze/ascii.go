package analyze

import (
	"fmt"
	"sort"
	"strings"
)

// RenderASCIIGraph draws the package dependency graph as plain text for
// the terminal, simplified for beginners in three layers (2026-10-20):
//
//  1. Only packages that connect to something get a block; packages
//     that import nothing internal fold into one summary line, and
//     tooling (scripts/, cmd/tools/) folds into its own line — build
//     helpers are not the product.
//  2. Deep utility clusters group into their folder: 3+ packages under
//     one 3-segment prefix (internal/ai/neural/…) render as a single
//     node with merged arrows.
//  3. Implied arrows are dropped (transitive reduction): if A → B and
//     B → C are shown, a direct A → C arrow adds no information.
//
// Entry points (func main) are marked ★ and always get a block.
// Read "A → B" as "A imports B". Nothing is hidden — folding and
// grouping only change how much is expanded.
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
	// --- Layer 1a: tooling classification ---
	isTooling := func(dir string) bool {
		return dir == "scripts" || strings.HasPrefix(dir, "scripts/") ||
			strings.HasPrefix(dir, "cmd/tools/")
	}

	// --- Layer 2: group clusters under a shared 3-segment prefix ---
	groupRoot := func(dir string) string {
		parts := strings.Split(dir, "/")
		if len(parts) >= 4 {
			return strings.Join(parts[:3], "/")
		}
		return ""
	}
	rootCount := map[string]int{}
	rootHasMain := map[string]bool{}
	for _, pkg := range p.Packages {
		if r := groupRoot(pkg.Dir); r != "" {
			rootCount[r]++
			if mains[pkg.Dir] {
				rootHasMain[r] = true
			}
		}
	}
	// nodeOf maps a package dir to its display node: its group root, or
	// itself when ungrouped.
	nodeOf := map[string]string{}
	groupMembers := map[string][]string{}
	for _, pkg := range p.Packages {
		node := pkg.Dir
		if r := groupRoot(pkg.Dir); r != "" && rootCount[r] >= 3 && !rootHasMain[r] {
			node = r
			groupMembers[r] = append(groupMembers[r], pkg.Dir)
		}
		nodeOf[pkg.Dir] = node
	}

	// Node edges: union of member deps, mapped through nodeOf, with
	// self-arrows removed.
	nodeEdges := map[string][]string{}
	nodeHasMain := map[string]bool{}
	for _, pkg := range p.Packages {
		node := nodeOf[pkg.Dir]
		if mains[pkg.Dir] {
			nodeHasMain[node] = true
		}
		seen := map[string]bool{}
		for _, e := range nodeEdges[node] {
			seen[e] = true
		}
		for _, d := range cleanDeps(pkg) {
			t := nodeOf[d]
			if t == "" {
				t = d // dep not in the package list: show as-is
			}
			if t != node && !seen[t] {
				seen[t] = true
				nodeEdges[node] = append(nodeEdges[node], t)
			}
		}
	}

	// Blocks: nodes that are entry points or have outgoing arrows.
	// Tooling nodes fold away, unless that would leave no blocks.
	var blocks, toolNodes []string
	seenNode := map[string]bool{}
	for _, pkg := range p.Packages {
		node := nodeOf[pkg.Dir]
		if seenNode[node] {
			continue
		}
		seenNode[node] = true
		if isTooling(pkg.Dir) && node == pkg.Dir {
			toolNodes = append(toolNodes, node)
			continue
		}
		if nodeHasMain[node] || len(nodeEdges[node]) > 0 {
			blocks = append(blocks, node)
		}
	}
	if len(blocks) == 0 && len(toolNodes) > 0 {
		blocks, toolNodes = toolNodes, nil
	}

	// --- Layer 3: transitive reduction on the displayed edges ---
	displayEdges := map[string][]string{}
	for _, node := range blocks {
		displayEdges[node] = append([]string(nil), nodeEdges[node]...)
	}
	reachableWithout := func(from, target, skipFrom, skipTo string) bool {
		visited := map[string]bool{from: true}
		queue := []string{from}
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			for _, nxt := range displayEdges[cur] {
				if cur == skipFrom && nxt == skipTo {
					continue
				}
				if nxt == target {
					return true
				}
				if !visited[nxt] {
					visited[nxt] = true
					queue = append(queue, nxt)
				}
			}
		}
		return false
	}
	for _, node := range blocks {
		var kept []string
		for _, t := range displayEdges[node] {
			if !reachableWithout(node, t, node, t) {
				kept = append(kept, t)
			}
		}
		sort.Strings(kept)
		displayEdges[node] = kept
	}

	// Sort blocks: entry points first, then most-imported, then name.
	indegree := map[string]int{}
	for _, node := range blocks {
		for _, t := range displayEdges[node] {
			indegree[t]++
		}
	}
	sort.Slice(blocks, func(i, j int) bool {
		if nodeHasMain[blocks[i]] != nodeHasMain[blocks[j]] {
			return nodeHasMain[blocks[i]]
		}
		if indegree[blocks[i]] != indegree[blocks[j]] {
			return indegree[blocks[i]] > indegree[blocks[j]]
		}
		return blocks[i] < blocks[j]
	})

	label := func(dir string) string {
		if dir == "" {
			return "(root)"
		}
		return dir
	}
	for _, node := range blocks {
		star := " "
		if nodeHasMain[node] {
			star = "★"
		}
		head := label(node)
		if members, ok := groupMembers[node]; ok {
			head = fmt.Sprintf("%s (%d packages)", node, len(members))
		}
		fmt.Fprintf(&b, "%s 📦 %s\n", star, head)
		deps := displayEdges[node]
		for i, d := range deps {
			branch := "├──"
			if i == len(deps)-1 {
				branch = "└──"
			}
			fmt.Fprintf(&b, "  %s → %s\n", branch, label(d))
		}
	}

	// --- Layer 1b: fold leaves and tooling into summary lines ---
	type foldEntry struct {
		text  string
		count int
	}
	var leafEntries []foldEntry
	inBlock := map[string]bool{}
	for _, node := range blocks {
		inBlock[node] = true
	}
	seenFold := map[string]bool{}
	for _, pkg := range p.Packages {
		node := nodeOf[pkg.Dir]
		if inBlock[node] || seenFold[node] {
			continue
		}
		if isTooling(pkg.Dir) && node == pkg.Dir {
			continue // tooling line below
		}
		seenFold[node] = true
		if members, ok := groupMembers[node]; ok {
			leafEntries = append(leafEntries, foldEntry{
				text:  fmt.Sprintf("%s (%d packages)", node, len(members)),
				count: len(members),
			})
		} else {
			leafEntries = append(leafEntries, foldEntry{text: label(pkg.Dir), count: 1})
		}
	}
	writeFold := func(entries []foldEntry, line func(total int, names string) string) {
		if len(entries) == 0 {
			return
		}
		sort.Slice(entries, func(i, j int) bool { return entries[i].text < entries[j].text })
		total := 0
		var names []string
		for _, e := range entries {
			total += e.count
			names = append(names, e.text)
		}
		shown, extra := names, 0
		if len(shown) > 8 {
			extra = len(shown) - 8
			shown = shown[:8]
		}
		b.WriteString(line(total, strings.Join(shown, ", ")))
		if extra > 0 {
			fmt.Fprintf(&b, " (+%d more)", extra)
		}
		b.WriteString("\n")
	}
	writeFold(leafEntries, func(total int, names string) string {
		return fmt.Sprintf("… %d package(s) import nothing internal: %s", total, names)
	})

	var toolEntries []foldEntry
	for _, node := range toolNodes {
		toolEntries = append(toolEntries, foldEntry{text: label(node), count: 1})
	}
	writeFold(toolEntries, func(_ int, names string) string {
		return "… plus tooling (build helpers, not the product): " + names
	})

	return b.String()
}
