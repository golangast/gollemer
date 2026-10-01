package chat

import (
	"os"
	"path/filepath"
	"testing"
)

// The allowlist is parsed from a Makefile: lowercase targets at line
// start, chat entry points excluded, everything else in.
func TestInitMakeAllowlist(t *testing.T) {
	dir := t.TempDir()
	mk := "start:\n\t@echo hi\n\nchat:\n\t@echo nested\n\ndebug-chat:\n\t@echo nested\n\ntrain-go:\n\t@echo train\n\n.PHONY: help\nMEM_LIMIT = 2500MiB\n"
	if err := os.WriteFile(filepath.Join(dir, "Makefile"), []byte(mk), 0644); err != nil {
		t.Fatal(err)
	}
	old := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{}
	defer func() { makeTargetsAllowlist = old }()
	initMakeAllowlist(dir)
	for _, want := range []string{"start", "train-go"} {
		if !makeTargetsAllowlist[want] {
			t.Errorf("target %q missing from allowlist", want)
		}
	}
	for _, banned := range []string{"chat", "debug-chat"} {
		if makeTargetsAllowlist[banned] {
			t.Errorf("target %q must be excluded from allowlist", banned)
		}
	}
}

// A missing Makefile fails closed: nothing is ever offered.
func TestInitMakeAllowlistMissing(t *testing.T) {
	old := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{}
	defer func() { makeTargetsAllowlist = old }()
	initMakeAllowlist(t.TempDir())
	if got := runnableMakeTarget("run make eval"); got != "" {
		t.Errorf("runnableMakeTarget with empty allowlist = %q, want \"\"", got)
	}
}

func TestRunnableMakeTarget(t *testing.T) {
	old := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{"eval": true, "train-go": true}
	defer func() { makeTargetsAllowlist = old }()

	ok := map[string]string{
		"run make eval":     "eval",
		"run make train-go": "train-go",
		"  run make eval  ": "eval",
		"RUN MAKE EVAL":     "eval",
		"run make EVAL":     "eval",
	}
	for in, want := range ok {
		if got := runnableMakeTarget(in); got != want {
			t.Errorf("runnableMakeTarget(%q) = %q, want %q", in, got, want)
		}
	}
	bad := []string{
		"", "run make", "make eval", "run make chat", // chat excluded
		"run make not-a-target",                       // not in allowlist
		"run make eval && rm -rf /",                   // chained
		"run make eval; echo hi",                      // metachars
		"please run make eval for me",                 // chatter
	}
	for _, in := range bad {
		if got := runnableMakeTarget(in); got != "" {
			t.Errorf("runnableMakeTarget(%q) = %q, want \"\"", in, got)
		}
	}
}

// The repo's own Makefile must parse to a non-empty allowlist so the
// chat can offer every real target.
func TestRepoMakefileParses(t *testing.T) {
	old := makeTargetsAllowlist
	makeTargetsAllowlist = map[string]bool{}
	defer func() { makeTargetsAllowlist = old }()
	root := filepath.Join("..", "..", "..", "..")
	initMakeAllowlist(root)
	for _, want := range []string{"eval", "smarter", "start", "help", "explain", "flow", "sel"} {
		if !makeTargetsAllowlist[want] {
			t.Errorf("repo target %q missing from allowlist", want)
		}
	}
	if makeTargetsAllowlist["chat"] || makeTargetsAllowlist["debug-chat"] {
		t.Error("chat entry points must stay out of the allowlist")
	}
}
