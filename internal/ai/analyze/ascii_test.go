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
