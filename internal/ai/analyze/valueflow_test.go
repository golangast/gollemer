package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// valueFlowFixture builds a project where a Config value threads
// main -> middle -> leaf, and leaf reads cfg.DryRun. The index must
// attribute the read to all three (direct + threaded), with no
// feature-specific logic anywhere.
func valueFlowFixture(t *testing.T) *Project {
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
	write("go.mod", "module example.com/vflow\n\ngo 1.21\n")
	write("config/config.go", `package config

type Config struct {
	DryRun  bool
	Verbose bool
}
`)
	write("main.go", `package main

import "example.com/vflow/config"
import "example.com/vflow/flow"

func main() {
	cfg := &config.Config{DryRun: true}
	flow.Middle(cfg)
}
`)
	write("flow/flow.go", `package flow

import "example.com/vflow/config"

func Middle(cfg *config.Config) {
	if cfg.Verbose {
		leaf(cfg)
	}
}

func leaf(cfg *config.Config) {
	if cfg.DryRun {
		wipe()
	}
	probe()
}

func wipe() { _ = wipe2() }
func wipe2() {}
func probe() {}
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestValueFlowThreading(t *testing.T) {
	p := valueFlowFixture(t)
	ids := p.FieldReaders("config.Config.DryRun")
	names := map[string]bool{}
	for _, id := range ids {
		if fn := p.byID[id]; fn != nil {
			names[fn.Name] = true
		}
	}
	// leaf reads directly; middle threads cfg into leaf; main threads cfg into middle.
	for _, want := range []string{"leaf", "Middle", "main"} {
		if !names[want] {
			t.Errorf("config.Config.DryRun readers = %v, missing %s", names, want)
		}
	}
	// wipe is not a reader — threading follows only the value's type.
	if names["wipe"] {
		t.Errorf("wipe should not read config.Config.DryRun: %v", names)
	}
}

func TestValueFlowNoReads(t *testing.T) {
	p := valueFlowFixture(t)
	if got := p.FieldReaders("config.Config.Nope"); len(got) != 0 {
		t.Errorf("unknown field readers = %v, want empty", got)
	}
}

func TestEffectCategories(t *testing.T) {
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
	write("go.mod", "module example.com/fx\n\ngo 1.21\n")
	write("a.go", `package fx

import "os"
import "net/http"

func reader() { os.ReadFile("x") }
func fetcher() { http.Get("http://x") }
func writer() { os.WriteFile("x", nil, 0o644) }
func caller() { writer() }
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	byName := map[string]*Func{}
	for _, fn := range p.byID {
		byName[fn.Name] = fn
	}
	if !hasEffect(byName["reader"].Effects, fxRead) {
		t.Errorf("reader.Effects = %v, want fs-read", byName["reader"].Effects)
	}
	if !hasEffect(byName["fetcher"].Effects, fxNet) {
		t.Errorf("fetcher.Effects = %v, want net", byName["fetcher"].Effects)
	}
	if !hasEffect(byName["writer"].Effects, fxWrite) {
		t.Errorf("writer.Effects = %v, want fs-write", byName["writer"].Effects)
	}
	// Transitive: caller inherits fs-write through writer.
	if !hasEffect(byName["caller"].EffectsAll, fxWrite) {
		t.Errorf("caller.EffectsAll = %v, want fs-write", byName["caller"].EffectsAll)
	}
	if byName["caller"].MutatesAll != true || byName["reader"].Mutates {
		t.Errorf("Mutates derivation wrong: caller.MutatesAll=%v reader.Mutates=%v",
			byName["caller"].MutatesAll, byName["reader"].Mutates)
	}
}

func TestAnswerWhatReads(t *testing.T) {
	p := valueFlowFixture(t)
	out, ok := p.Answer("what reads DryRun")
	if !ok {
		t.Fatal("Answer(\"what reads DryRun\") not handled")
	}
	for _, want := range []string{"leaf", "Middle", "main"} {
		if !strings.Contains(out, want) {
			t.Errorf("answer missing %s:\n%s", want, out)
		}
	}
}

func TestAnswerWhereUsed(t *testing.T) {
	p := valueFlowFixture(t)
	if _, ok := p.Answer("where is Verbose used"); !ok {
		t.Error("Answer(\"where is Verbose used\") not handled")
	}
}

func TestAnswerEffects(t *testing.T) {
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
	write("go.mod", "module example.com/fx2\n\ngo 1.21\n")
	write("a.go", `package fx2

import "os"

func writer() { os.WriteFile("x", nil, 0o644) }
`)
	p, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	out, ok := p.Answer("where does it write files")
	if !ok {
		t.Fatal("Answer(\"where does it write files\") not handled")
	}
	if !strings.Contains(out, "writer") {
		t.Errorf("answer missing writer:\n%s", out)
	}
	if _, ok := p.Answer("where does it use the network"); ok {
		t.Error("network question should not be handled when nothing uses the network")
	}
}
