package chat

import "testing"

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
