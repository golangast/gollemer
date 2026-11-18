package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The serve sentence behind a module wall, and in the
// two-home arrangement (2026-11-10, from John's live
// screenshot and his serving-shape rule). A registration
// or a server inside a NESTED module — paces/project has
// its own go.mod — belongs to that module: this project
// can neither build nor run it, so the sentence must not
// read it as "already serving", must not run its file, and
// must make its own server in server/ instead. A
// registration in the project's own module but in a deeper
// program of its own is visible but not adopted either:
// it is named before the gate, left byte-unchanged, and
// the app is served from server/.

const walledServerForJim = `package main

import "net/http"

func main() {
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim"))))
	http.ListenAndServe(":9090", nil)
}
`

const deepServerForJimChat = `package main

import "net/http"

func main() {
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim"))))
	http.ListenAndServe(":9091", nil)
}
`

func serveModuleFixture(t *testing.T, extra map[string]string) string {
	t.Helper()
	files := map[string]string{
		"jim/go.mod":  "module jim\n\ngo 1.27.0\n",
		"jim/main.go": wasmMainSrc,
	}
	for rel, src := range extra {
		files[rel] = src
	}
	dir := staticAssetsFixture(t, staticAssetsFiles(files))
	writeFile(t, dir, "jim/main.wasm", "not a real binary — presence is what the flow checks")
	oldOpen := openBrowserURL
	openBrowserURL = func(string) error { return nil }
	t.Cleanup(func() { openBrowserURL = oldOpen })
	t.Cleanup(func() { stopProgramAtChatEnd(); sessionProgram = nil })
	return dir
}

// John's exact shape: the project root module, the built
// app in jim, and a /static/ registration for jim inside
// paces/project — a nested module the root cannot build.
// The sentence creates its own server in the root module,
// reaches the run with the address line, and the walled
// file is byte-unchanged.
func TestServeWasmAppNestedModule(t *testing.T) {
	if !portFree("8080") {
		t.Skip("port 8080 is in use; the nested-module serve proof needs it")
	}
	yes := func(string) (string, error) { return "y", nil }
	dir := serveModuleFixture(t, map[string]string{
		"paces/project/go.mod":      "module example.com/nested\n\ngo 1.21\n",
		"paces/project/api/main.go": walledServerForJim,
	})
	walledBefore, err := os.ReadFile(filepath.Join(dir, "paces", "project", "api", "main.go"))
	if err != nil {
		t.Fatal(err)
	}

	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in jim", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	srv, err := os.ReadFile(filepath.Join(dir, "server", "main.go"))
	if err != nil {
		t.Fatalf("no server was created in the root module: %v", err)
	}
	if !strings.Contains(string(srv), `http.FileServer(http.Dir("jim"))`) {
		t.Errorf("the created server does not carry the app's route:\n%s", srv)
	}
	if !strings.Contains(out, "The server in server/main.go now serves the wasm app in jim at /static/") {
		t.Errorf("the created-server confirmation is missing:\n%s", out)
	}
	if !strings.Contains(out, "The wasm app in jim is served by your Go server — open http://localhost:8080/static/ in your browser to run it.") {
		t.Errorf("the address line is missing — the sentence stopped at the module wall:\n%s", out)
	}
	if !programRunning() {
		t.Fatalf("the created server is not running:\n%s", out)
	}
	walledAfter, err := os.ReadFile(filepath.Join(dir, "paces", "project", "api", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if string(walledBefore) != string(walledAfter) {
		t.Errorf("the walled file changed")
	}
}

// The same registration in the project's OWN module, in a
// deeper program of its own (tools/deep): the sentence
// names it before the gate, leaves it byte-unchanged,
// creates server/, and serves from there.
func TestServeWasmAppDeepProgramNamed(t *testing.T) {
	if !portFree("8080") {
		t.Skip("port 8080 is in use; the deep-program serve proof needs it")
	}
	yes := func(string) (string, error) { return "y", nil }
	dir := serveModuleFixture(t, map[string]string{
		"tools/deep/main.go": deepServerForJimChat,
	})
	deepBefore, err := os.ReadFile(filepath.Join(dir, "tools", "deep", "main.go"))
	if err != nil {
		t.Fatal(err)
	}

	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in jim", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	notice := "tools/deep/main.go also registers /static/ for jim in its own program; this sentence serves it from server/ instead."
	ni := strings.Index(out, notice)
	if ni < 0 {
		t.Fatalf("the other registration was not named:\n%s", out)
	}
	if gi := strings.Index(out, "Here's the edit"); gi < 0 || ni > gi {
		t.Errorf("the naming did not come before the gate:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "server", "main.go")); err != nil {
		t.Errorf("server/ was not created: %v", err)
	}
	if !strings.Contains(out, "open http://localhost:8080/static/ in your browser to run it.") {
		t.Errorf("the address line is missing:\n%s", out)
	}
	if !programRunning() {
		t.Fatalf("the created server is not running:\n%s", out)
	}
	deepAfter, err := os.ReadFile(filepath.Join(dir, "tools", "deep", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if string(deepBefore) != string(deepAfter) {
		t.Errorf("the deep program's file changed")
	}
}

// The route already serving from server/ itself: no edit
// is planned, the flow goes straight to the run.
func TestServeWasmAppAlreadyServedFromServerFolder(t *testing.T) {
	if !portFree("9091") {
		t.Skip("port 9091 is in use; the already-served proof needs it")
	}
	yes := func(string) (string, error) { return "y", nil }
	serveModuleFixture(t, map[string]string{
		"server/main.go": deepServerForJimChat,
	})
	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in jim", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	if strings.Contains(out, "Here's the edit") {
		t.Errorf("an already-serving route was re-planned:\n%s", out)
	}
	if !strings.Contains(out, "The wasm app in jim is served by your Go server — open http://localhost:9091/static/ in your browser to run it.") {
		t.Errorf("the address line with the server's own port is missing:\n%s", out)
	}
	if !programRunning() {
		t.Fatalf("the server/ server is not running:\n%s", out)
	}
}

// A server already waiting in server/ without the route:
// the serve-static splice lands in THAT file (the
// composition's targeted edit), and it is the one that
// runs.
func TestServeWasmAppExtendsServerFolder(t *testing.T) {
	if !portFree("8080") {
		t.Skip("port 8080 is in use; the extend-server proof needs it")
	}
	yes := func(string) (string, error) { return "y", nil }
	dir := serveModuleFixture(t, map[string]string{
		"server/main.go": "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.HandleFunc(\"/\", func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(\"home\")) })\n\thttp.ListenAndServe(\":8080\", nil)\n}\n",
	})
	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in jim", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	srv, err := os.ReadFile(filepath.Join(dir, "server", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(srv), `http.FileServer(http.Dir("jim"))`) {
		t.Errorf("the splice did not land in server/'s server:\n%s", srv)
	}
	if !strings.Contains(out, "now serves the jim folder's files at /static/") {
		t.Errorf("the serve-static confirmation is missing:\n%s", out)
	}
	if !strings.Contains(out, "open http://localhost:8080/static/ in your browser to run it.") {
		t.Errorf("the address line is missing:\n%s", out)
	}
	if !programRunning() {
		t.Fatalf("the extended server is not running:\n%s", out)
	}
}
