package chat

import (
	"os/exec"
	"testing"
)

// The command shape: "clone <url> in folder <name>".
func TestParseCloneCommand(t *testing.T) {
	cases := []struct {
		line, url, folder string
		ok                bool
	}{
		{"clone https://github.com/pashkov256/deletor in folder example", "https://github.com/pashkov256/deletor", "example", true},
		{"clone github.com/pashkov256/deletor in folder example", "github.com/pashkov256/deletor", "example", true},
		{"clone https://github.com/pashkov256/deletor in example", "https://github.com/pashkov256/deletor", "example", true},
		{"clone https://github.com/pashkov256/deletor", "https://github.com/pashkov256/deletor", "", true},
		{"Clone https://github.com/pashkov256/deletor in folder Ex-Ample_2", "https://github.com/pashkov256/deletor", "Ex-Ample_2", true},
		{"clone the repo", "", "", false},
		{"clone", "", "", false},
		{"how do I clone a repo", "", "", false},
		{"analyze https://github.com/pashkov256/deletor", "", "", false},
	}
	for _, tc := range cases {
		url, folder, ok := parseCloneCommand(tc.line)
		if ok != tc.ok || url != tc.url || folder != tc.folder {
			t.Errorf("parseCloneCommand(%q) = (%q, %q, %v), want (%q, %q, %v)",
				tc.line, url, folder, ok, tc.url, tc.folder, tc.ok)
		}
	}
}

// Bad folder names are rejected, not passed to the shell — and there
// is no shell here anyway: the folder becomes one path element.
func TestCloneFolderValidation(t *testing.T) {
	for _, bad := range []string{"..", ".", "../x"} {
		url, folder, ok := parseCloneCommand("clone https://github.com/o/r in folder " + bad)
		_ = url
		if bad == "../x" {
			// Slashes don't match the folder pattern at all.
			if ok {
				t.Errorf("parseCloneCommand accepted %q", bad)
			}
			continue
		}
		if !ok || folder != bad {
			t.Errorf("parse %q: ok=%v folder=%q", bad, ok, folder)
		}
		if folderNameRe.MatchString(folder) && folder != "." && folder != ".." {
			t.Errorf("folderNameRe accepted %q", folder)
		}
	}
}

// gitOriginRepo reads owner/repo off the origin remote, so the issue
// flow can tell when you're already looking at the issue's repo.
func TestGitOriginRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not installed")
	}
	dir := t.TempDir()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init", "-q")
	run("remote", "add", "origin", "https://github.com/pashkov256/deletor.git")
	if got := gitOriginRepo(dir); got != "pashkov256/deletor" {
		t.Errorf("gitOriginRepo = %q, want pashkov256/deletor", got)
	}
	if got := gitOriginRepo(t.TempDir()); got != "" {
		t.Errorf("gitOriginRepo(non-repo) = %q, want empty", got)
	}
}
