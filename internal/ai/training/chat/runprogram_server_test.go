package chat

import (
	"net/http"
	"strings"
	"testing"
)

// The "run the server <name>" suite. John's second screenshot
// of 2026-11-09: in his gollemer checkout he typed "run the
// server jim" — jim/ being the folder holding his server —
// and the line fell to the intent layer, which ran the HOST
// repo's `make start` and narrated invented success. These
// tests pin the claim that takes the sentence first: the
// named folder's one runnable program is what runs, through
// the ordinary run gate and build.

const jimFolderServer = `package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, greeting())
	})
	fmt.Println("listening at http://localhost:8093 — jim's own server")
	if err := http.ListenAndServe(":8093", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`

// "run the server jim" in John's layout: a Makefile with a
// start target in the project root, the server as a two-file
// package inside jim/. Jim's server must run — named by the
// gate as its package — and the host's make start must not.
func TestRunServerFolderRunsJim(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "Makefile", "start:\n\t@echo HOST MAKE START RAN\n")
	writeFile(t, dir, "jim/main.go", jimFolderServer)
	writeFile(t, dir, "jim/help.go", "package main\n\nfunc greeting() string {\n\treturn \"hello from jim\"\n}\n")
	opened := withRunSession(t, dir)
	profile := &UserProfile{}

	var prompt string
	recYes := func(p string) (string, error) { prompt = p; return "y", nil }

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("run the server jim", recYes, profile) })
	if !handled {
		t.Fatalf("\"run the server jim\" was not claimed:\n%s", out)
	}
	if !strings.Contains(prompt, "go run ./jim") {
		t.Errorf("gate prompt = %q, want it to name jim's package (go run ./jim)", prompt)
	}
	for _, want := range []string{"is running in the background", "http://localhost:8093"} {
		if !strings.Contains(out, want) {
			t.Errorf("server start report missing %q:\n%s", want, out)
		}
	}
	if len(*opened) != 1 || (*opened)[0] != "http://localhost:8093" {
		t.Errorf("browser opened %v, want [http://localhost:8093]", *opened)
	}
	if !programRunning() {
		t.Fatal("jim's server should be running in the background")
	}
	for _, banned := range []string{"HOST MAKE START RAN", "make start", "It ran fine"} {
		if strings.Contains(out, banned) {
			t.Errorf("output contains %q — the make-start failure shape:\n%s", banned, out)
		}
	}

	// A real visit proves jim's program — not the host's
	// Makefile — is what answered.
	resp, err := http.Get("http://localhost:8093/")
	if err != nil {
		t.Fatalf("visit to jim's server failed: %v", err)
	}
	body := make([]byte, 128)
	n, _ := resp.Body.Read(body)
	resp.Body.Close()
	if !strings.Contains(string(body[:n]), "hello from jim") {
		t.Errorf("visit body = %q, want jim's own answer", string(body[:n]))
	}

	out = captureStdout(t, func() { handled = claimChain("stop the program", yes, profile) })
	if !handled || !strings.Contains(out, "Stopped jim/main.go") || programRunning() {
		t.Errorf("\"stop the program\" did not stop jim's server:\n%s", out)
	}
	if !portFree("8093") {
		t.Error("port 8093 still bound after stop")
	}
}

// The softened spelling claims the same way, on a program
// that finishes inside the startup window.
func TestRunServerCanYouFolderQuick(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "srv/main.go", quickProgram)
	withRunSession(t, dir)
	profile := &UserProfile{Directives: map[string]bool{"do_all_means_all": true}}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("can you run the server srv", yes, profile) })
	if !handled {
		t.Fatalf("\"can you run the server srv\" was not claimed:\n%s", out)
	}
	for _, want := range []string{"hello from the quick program", "It ran and finished (exit code 0)"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
}

// A .go target under the server shape is "run <file>":
// same claim, same report, no folder resolution involved.
func TestRunServerFileTarget(t *testing.T) {
	dir := runFixtureRoot(t)
	writeFile(t, dir, "jake/main.go", quickProgram)
	withRunSession(t, dir)
	profile := &UserProfile{Directives: map[string]bool{"do_all_means_all": true}}

	var handled bool
	out := captureStdout(t, func() { handled = runRunProgram("run the server jake/main.go", yes, profile) })
	if !handled {
		t.Fatalf("\"run the server jake/main.go\" was not claimed:\n%s", out)
	}
	if !strings.Contains(out, "hello from the quick program") {
		t.Errorf("the named file did not run:\n%s", out)
	}
}

// The declines: resolution finds no single program to run,
// so THIS claim lets the line go — nothing claimed, nothing
// printed, nothing started. Picking a program is never
// invented here.
func TestRunServerFolderDeclines(t *testing.T) {
	mainSrc := "package main\n\nfunc main() {}\n"
	cases := []struct {
		name  string
		files map[string]string
		line  string
	}{
		{"library folder", map[string]string{"lib/lib.go": "package lib\n\nfunc Add(a, b int) int { return a + b }\n"}, "run the server lib"},
		{"two mains", map[string]string{"duo/a.go": mainSrc, "duo/b.go": mainSrc}, "run the server duo"},
		{"only a test file", map[string]string{"tst/main_test.go": "package main\n\nimport \"testing\"\n\nfunc TestX(t *testing.T) {}\n"}, "run the server tst"},
		{"no go files", map[string]string{"web/index.html": "<html></html>\n"}, "run the server web"},
		{"missing folder", nil, "run the server nope"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := runFixtureRoot(t)
			for rel, src := range tc.files {
				writeFile(t, dir, rel, src)
			}
			withRunSession(t, dir)
			profile := &UserProfile{Directives: map[string]bool{"do_all_means_all": true}}
			var handled bool
			out := captureStdout(t, func() { handled = runRunProgram(tc.line, yes, profile) })
			if handled {
				t.Errorf("%q was claimed, want the claim to decline:\n%s", tc.line, out)
			}
			if strings.Contains(out, "gollemer [runner]") {
				t.Errorf("a decline must print nothing, got:\n%s", out)
			}
			if sessionProgram != nil {
				t.Error("a decline must not start a program")
			}
		})
	}
}
