package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// understandFixture builds a fresh backup-tool program exercising the
// model's understanding: aliases (dry := cfg.DryRun), threading
// (main -> Run -> Collect), transitive effects (Run -> Copy ->
// os.WriteFile), and field writes in main.
func understandFixture(t *testing.T) *Project {
	t.Helper()
	dir := t.TempDir()
	write := func(rel, src string) {
		full := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module evalprog\n\ngo 1.21\n")
	write("config/config.go", `package config

type Config struct {
	DryRun  bool
	Verbose bool
	Dest    string
}
`)
	write("backup/backup.go", `package backup

import "os"
import "evalprog/config"

func Run(cfg *config.Config) error {
	dry := cfg.DryRun
	for _, f := range Collect(cfg) {
		if dry {
			continue
		}
		if err := Copy(f, cfg.Dest); err != nil {
			return err
		}
	}
	return nil
}

func Collect(cfg *config.Config) []string {
	if cfg.Verbose {
		println("scanning")
	}
	return nil
}

func Copy(src, dest string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dest, data, 0644)
}
`)
	write("main.go", `package main

import "evalprog/config"
import "evalprog/backup"

func main() {
	cfg := &config.Config{}
	cfg.Dest = "/backup"
	cfg.Verbose = true
	backup.Run(cfg)
}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestUnderstandEval is a blind evaluation of code understanding on a
// program the model has never seen. Each case needs inference, not
// lookup: aliases, threading, transitive effects, field writes.
func TestUnderstandEval(t *testing.T) {
	p := understandFixture(t)
	cases := []struct {
		question string
		want     []string // every one must appear
		notWant  []string // none may appear
	}{
		{
			// Transitive: Copy writes directly, Run writes through Copy.
			question: "where does it write files",
			want:     []string{"Copy", "Run"},
		},
		{
			// Alias: dry := cfg.DryRun — the read is Run's.
			question: "what reads DryRun",
			want:     []string{"Run"},
		},
		{
			// Threading: Run passes cfg to Collect, which reads Verbose.
			question: "what reads Verbose",
			want:     []string{"Collect", "Run"},
		},
		{
			// Writes happen in main, not in the backup package.
			question: "where is Dest set",
			want:     []string{"main"},
			notWant:  []string{"Copy"},
		},
		{
			question: "where is Verbose set",
			want:     []string{"main"},
		},
		{
			question: "what calls Copy",
			want:     []string{"Run"},
		},
		{
			question: "where is DryRun set",
			want:     []string{},
		},
	}
	for _, tc := range cases {
		out, ok := p.Answer(tc.question)
		if len(tc.want) == 0 {
			// Expect unhandled: nothing sets DryRun.
			if ok {
				t.Errorf("%q: expected unhandled, got:\n%s", tc.question, out)
			}
			continue
		}
		if !ok {
			t.Errorf("%q: not handled", tc.question)
			continue
		}
		for _, w := range tc.want {
			if !strings.Contains(out, w) {
				t.Errorf("%q: missing %q:\n%s", tc.question, w, out)
			}
		}
		for _, w := range tc.notWant {
			if strings.Contains(out, w) {
				t.Errorf("%q: should not contain %q:\n%s", tc.question, w, out)
			}
		}
	}
	// Negative: no network use anywhere.
	if _, ok := p.Answer("where does it use the network"); ok {
		t.Error("network question should be unhandled")
	}
}

// TestUnderstandEvalIssue runs a feature issue against the fresh program:
// the flag plan must name the config, the mutation, and the alias guard.
func TestUnderstandEvalIssue(t *testing.T) {
	p := understandFixture(t)
	c := ExtractIssueConcepts("Add --force flag to overwrite without asking\n\n" +
		"Add flag: `--force` to skip the dry-run skip and always write.")
	out := p.GuideIssue(c)
	for _, want := range []string{
		"config.go",    // CONFIG: the options struct
		"flags",        // FLAGS section (case-insensitive check below)
		"Copy",         // BEHAVIOR: reaches os.WriteFile
		"os.WriteFile", // the mutation the flag gates
		"dry",          // the alias guard: where flags are checked
	} {
		if !strings.Contains(strings.ToLower(out), strings.ToLower(want)) {
			t.Errorf("issue plan missing %q:\n%s", want, out)
		}
	}
}
