package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// effectsFixture: Run transitively mutates through Wipe (os.Remove);
// Check only reads; Guarded gates os.Remove behind a config flag;
// Load gates on an error check, not a flag.
func effectsFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module example.com/fx\n\ngo 1.21\n")
	write("main.go", "package main\n\nimport \"example.com/fx/work\"\n\nfunc main() { work.Run() }\n")
	write("work/work.go", `package work

import "os"

// Config holds the run options.
type Config struct {
	DryRun bool
}

// Run processes every item.
func Run() {
	for _, it := range list() {
		Wipe(it)
	}
}

func list() []string { return nil }

// Wipe deletes one item.
func Wipe(path string) {
	os.Remove(path)
}

// Check only stats a path.
func Check(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Guarded deletes unless the dry-run flag is set.
func Guarded(cfg *Config, path string) {
	if cfg.DryRun {
		return
	}
	os.Remove(path)
}

// Load reads a file, bailing on error (not a flag).
func Load(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}
`)
	return root
}

func fxFunc(t *testing.T, p *Project, name string) *Func {
	t.Helper()
	for _, fn := range p.byID {
		if fn.Name == name {
			return fn
		}
	}
	t.Fatalf("func %s not found", name)
	return nil
}

func TestEffectsDirect(t *testing.T) {
	p, err := Analyze(effectsFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	wipe := fxFunc(t, p, "Wipe")
	if !wipe.Mutates {
		t.Error("Wipe should directly mutate (os.Remove)")
	}
	if !wipe.MutatesAll {
		t.Error("Wipe should transitively mutate")
	}
	found := false
	for _, e := range wipe.ExtCalls {
		if e == "os.Remove" {
			found = true
		}
	}
	if !found {
		t.Errorf("Wipe.ExtCalls = %v, want os.Remove", wipe.ExtCalls)
	}

	check := fxFunc(t, p, "Check")
	if check.Mutates || check.MutatesAll {
		t.Error("Check only reads (os.Stat); should not mutate")
	}
}

func TestEffectsTransitive(t *testing.T) {
	p, err := Analyze(effectsFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	run := fxFunc(t, p, "Run")
	if run.Mutates {
		t.Error("Run does not directly mutate")
	}
	if !run.MutatesAll {
		t.Error("Run should transitively mutate through Wipe")
	}
}

func TestGuardsFlagVsError(t *testing.T) {
	p, err := Analyze(effectsFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	guarded := fxFunc(t, p, "Guarded")
	if len(guarded.Guards) != 1 {
		t.Fatalf("Guarded should have 1 guard, got %d", len(guarded.Guards))
	}
	g := guarded.Guards[0]
	if !g.FlagLike {
		t.Errorf("guard %q should be flag-like", g.Cond)
	}
	if g.Cond != "cfg.DryRun" {
		t.Errorf("guard cond = %q, want cfg.DryRun", g.Cond)
	}
	found := false
	for _, e := range g.ExtCalls {
		if e == "os.Remove" {
			found = true
		}
	}
	if !found {
		t.Errorf("guard ExtCalls = %v, want os.Remove", g.ExtCalls)
	}

	load := fxFunc(t, p, "Load")
	if len(load.Guards) != 1 {
		t.Fatalf("Load should have 1 guard, got %d", len(load.Guards))
	}
	if load.Guards[0].FlagLike {
		t.Errorf("err != nil guard should not be flag-like")
	}

	wipe := fxFunc(t, p, "Wipe")
	if len(wipe.Guards) != 0 {
		t.Errorf("Wipe has no ifs; got %d guards", len(wipe.Guards))
	}
}

// The dry-run plan names the mutations the new flag must guard.
func TestPlanNamesGuardedMutations(t *testing.T) {
	p, err := Analyze(dryRunFixture(t))
	if err != nil {
		t.Fatal(err)
	}
	out := p.GuideIssue(ExtractIssueConcepts(dryRunIssue))
	for _, want := range []string{
		"Guard these with the new flag",
		"`DoWork` → os.Remove",
		"Check the new flag where",
		"`cfg.Verbose`",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("plan missing %q\n%s", want, out)
		}
	}
}
