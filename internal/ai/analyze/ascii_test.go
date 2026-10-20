package analyze

import (
	"strings"
	"testing"
)

func TestRenderASCIIGraph(t *testing.T) {
	p := &Project{Module: "example.com/demo"}
	chat := &Package{Name: "chat", Dir: "internal/chat", Internal: []string{"internal/analyze"}}
	an := &Package{Name: "analyze", Dir: "internal/analyze"}
	main := &Package{Name: "main", Dir: "cmd/app"}
	p.Packages = []*Package{chat, an, main}
	p.byID = map[string]*Func{
		"cmd/app.main": {Name: "main", PkgDir: "cmd/app", IsMain: true},
	}
	out := RenderASCIIGraph(p)
	for _, want := range []string{
		"Dependency graph: example.com/demo",
		"A → B means A imports B",
		"★ 📦 cmd/app",
		"📦 internal/chat",
		"└── → internal/analyze",
		// The import-less package is folded into the summary line,
		// not given a dead block of its own.
		"import nothing internal: internal/analyze",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("RenderASCIIGraph missing %q\n---\n%s", want, out)
		}
	}
	if strings.Contains(out, "(no project imports)") {
		t.Errorf("RenderASCIIGraph still prints the noise line\n---\n%s", out)
	}
}

// The three simplification cuts: tooling folds away, deep clusters
// group into one node, and implied arrows are dropped.
func TestRenderASCIIGraphSimplified(t *testing.T) {
	p := &Project{Module: "example.com/big"}
	mk := func(dir string, deps ...string) *Package {
		return &Package{Name: dir, Dir: dir, Internal: deps}
	}
	// Neural cluster: 3 packages under internal/ai/neural/. tensor
	// reaches outside the cluster (pkg/c), so the group gets a block.
	nn := mk("internal/ai/neural/nn", "internal/ai/neural/tensor")
	tensor := mk("internal/ai/neural/tensor", "pkg/c")
	vocab := mk("internal/ai/neural/nnu/vocab")
	// Chat imports nn and tensor; the tensor arrow is implied via nn.
	chat := mk("internal/ai/training/chat",
		"internal/ai/neural/nn", "internal/ai/neural/tensor")
	// Tooling: a scripts main package.
	scripts := mk("scripts", "internal/ai/neural/nn")
	// Transitive case: a → b, a → c, b → c. The direct a → c arrow is
	// implied and must be dropped; b → c stays.
	pa := mk("pkg/a", "pkg/b", "pkg/c")
	pb := mk("pkg/b", "pkg/c")
	pc := mk("pkg/c")
	p.Packages = []*Package{nn, tensor, vocab, chat, scripts, pa, pb, pc}
	p.byID = map[string]*Func{
		"scripts.main": {Name: "main", PkgDir: "scripts", IsMain: true},
	}
	out := RenderASCIIGraph(p)
	for _, want := range []string{
		"📦 internal/ai/neural (3 packages)", // grouped cluster block
		"📦 internal/ai/training/chat",       // chat block stays separate
		"→ internal/ai/neural",              // chat arrow points at the group
		"plus tooling",                      // scripts folded, not a block
		"scripts",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("simplified graph missing %q\n---\n%s", want, out)
		}
	}
	for _, banned := range []string{
		"📦 scripts",               // tooling must not get a block
		"📦 internal/ai/neural/nn", // members must not get own blocks
		"📦 internal/ai/neural/tensor",
	} {
		if strings.Contains(out, banned) {
			t.Errorf("simplified graph should not contain %q\n---\n%s", banned, out)
		}
	}
	// Exactly two "→ pkg/c" arrows survive: the group's (via tensor)
	// and b → c. The direct a → c arrow was implied and dropped.
	if n := strings.Count(out, "→ pkg/c"); n != 2 {
		t.Errorf("want exactly 2 arrows to pkg/c after reduction, got %d\n---\n%s", n, out)
	}
}
