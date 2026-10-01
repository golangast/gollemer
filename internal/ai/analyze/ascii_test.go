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
		"📦 internal/analyze",
		"(no project imports)",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("RenderASCIIGraph missing %q\n---\n%s", want, out)
		}
	}
}
