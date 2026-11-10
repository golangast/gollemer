package chat

import (
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// Serve the wasm app: one sentence composing the route and
// the run. These tests drive the whole flow through
// claimChain — the folder verification, the unbuilt stop,
// the session reading, the created-server and existing-server
// landings, the conflict, and the full HTTP smoke.

// wasmMainSrc is a minimal app main.go: the recognition the
// flow checks is the js && wasm tag plus the syscall/js
// import, nothing more.
const wasmMainSrc = "//go:build js && wasm\n\npackage main\n\nimport \"syscall/js\"\n\nfunc main() {\n\t_ = js.Global()\n\tselect {}\n}\n"

func TestParseServeWasmApp(t *testing.T) {
	claims := []struct {
		line      string
		folder    string
		hasFolder bool
	}{
		{"serve the wasm app in jim", "jim", true},
		{"serve the wasm app", "", false},
		{"serve the wasm in jim", "jim", true},
		{"serve the wasm", "", false},
		{"serve a wasm app in jim", "jim", true},
		{"serve the webassembly app in jim/app", "jim/app", true},
		{"serve the wasm app into jim", "jim", true},
		{"serve the wasm app at the folder jim", "jim", true},
		{"please serve the wasm app in jim", "jim", true},
		{"Serve the Wasm App in Jim.", "Jim", true}, // the folder keeps its typed case, as in the build sentence's parser
	}
	for _, tc := range claims {
		folder, hasFolder, ok := parseServeWasmApp(tc.line)
		if !ok || folder != tc.folder || hasFolder != tc.hasFolder {
			t.Errorf("parseServeWasmApp(%q) = (%q, %v, %v), want (%q, %v, true)", tc.line, folder, hasFolder, ok, tc.folder, tc.hasFolder)
		}
	}
	declines := []string{
		"serve the app in jim", // no wasm word — not this claim
		"serve the app",
		"serve static files from jim", // the serve-static claim's
		"serve the static files in jim",
		"run the server jim",        // the run claim's
		"run the wasm app in jim",   // the run claim's too
		"build the wasm app in jim", // the build claim's
		"build the wasm app",
		"add a wasm app in jim",             // the app claim's
		"serve the wasm app in jim/main.go", // a file is not an app folder
		"serve the wasm app in jim/index.html",
		"serve the wasm app in",      // a tail with no folder
		"serve the component in jim", // a different piece entirely
	}
	for _, line := range declines {
		if _, _, ok := parseServeWasmApp(line); ok {
			t.Errorf("parseServeWasmApp(%q) claimed; want decline", line)
		}
		if runServeWasmApp(line, func(string) (string, error) { return "y", nil }, &UserProfile{}) {
			t.Errorf("runServeWasmApp(%q) claimed; want the line left for its owner", line)
		}
	}
}

// The folder states that are not "a built app": each gets its
// honest answer and its own route, and nothing is written or
// started.
func TestServeWasmAppFolderAnswers(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }

	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in jim", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	if !strings.Contains(out, "there's no folder jim in this project") ||
		!strings.Contains(out, `"create a folder jim"`) {
		t.Errorf("missing-folder answer wrong:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "server")); !os.IsNotExist(err) {
		t.Errorf("a refusal created the server folder")
	}

	staticAssetsFixture(t, staticAssetsFiles(map[string]string{"jim": "a file, not a folder\n"}))
	out = captureStdout(t, func() { claimChain("serve the wasm app in jim", yes, &UserProfile{}) })
	if !strings.Contains(out, "jim isn't a folder") {
		t.Errorf("not-a-folder answer wrong:\n%s", out)
	}

	staticAssetsFixture(t, staticAssetsFiles(map[string]string{"jim/notes.txt": "notes\n"}))
	out = captureStdout(t, func() { claimChain("serve the wasm app in jim", yes, &UserProfile{}) })
	if !strings.Contains(out, "there's no wasm app in jim yet") ||
		!strings.Contains(out, `"add a wasm app in jim"`) {
		t.Errorf("no-app answer wrong:\n%s", out)
	}

	staticAssetsFixture(t, staticAssetsFiles(map[string]string{"jim/index.html": "<!doctype html>\n<title>pages</title>\n"}))
	out = captureStdout(t, func() { claimChain("serve the wasm app in jim", yes, &UserProfile{}) })
	if !strings.Contains(out, "plain HTML, not a wasm app") ||
		!strings.Contains(out, `"serve static files from jim"`) {
		t.Errorf("plain-pages answer wrong:\n%s", out)
	}
}

// An app that exists but was never built stops the flow with
// the build sentence named — no server is made, nothing runs.
func TestServeWasmAppUnbuiltStops(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		claimChain("add a wasm app in jim", yes, &UserProfile{})
	})
	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in jim", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	if !strings.Contains(out, "isn't built") ||
		!strings.Contains(out, `"build the wasm app in jim"`) {
		t.Errorf("unbuilt answer wrong:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "server")); !os.IsNotExist(err) {
		t.Errorf("the unbuilt stop created the server folder")
	}
	if sessionProgram != nil {
		t.Errorf("the unbuilt stop left a program in the session")
	}
}

// The folderless sentence reads the session's last-touched
// app, prints the reading, and a declined plan gate lands
// nothing — the composed spine, end to end, minus the run.
func TestServeWasmAppSessionFolderGateDeclined(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }
	no := func(string) (string, error) { return "n", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		claimChain("add a wasm app in jim", yes, &UserProfile{})
	})
	writeFile(t, dir, "jim/main.wasm", "not a real binary — the serve flow checks presence, not content")
	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app", no, &UserProfile{}) {
			t.Error("the folderless serve sentence was not claimed")
		}
	})
	if !strings.Contains(out, "Reading the app as the wasm app in jim — the last wasm app this session touched.") {
		t.Errorf("the session reading was not printed:\n%s", out)
	}
	if !strings.Contains(out, "Not applied") {
		t.Errorf("the declined gate was not narrated:\n%s", out)
	}
	if _, err := os.Stat(filepath.Join(dir, "server", "main.go")); !os.IsNotExist(err) {
		t.Errorf("a declined gate still created the server")
	}
	if sessionProgram != nil {
		t.Errorf("a declined gate left a program in the session")
	}
}

// The folderless sentence with no session app asks its one
// question; the answer completes the original sentence.
func TestServeWasmAppAsksForFolder(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }
	staticAssetsFixture(t, staticAssetsFiles(nil))
	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app", yes, &UserProfile{}) {
			t.Error("the folderless serve sentence was not claimed")
		}
	})
	if !hasPendingQuestion() {
		t.Fatalf("no clarify question was set:\n%s", out)
	}
	if !strings.Contains(out, "which folder does the app live in?") {
		t.Errorf("the question was not asked:\n%s", out)
	}
	// The answer "jim" completes the sentence — and jim is
	// not a folder here, so the completion lands on the
	// family's two-step answer.
	out = captureStdout(t, func() {
		if !claimChain("jim", yes, &UserProfile{}) {
			t.Error("the clarify answer was not claimed")
		}
	})
	if !strings.Contains(out, "there's no folder jim in this project") {
		t.Errorf("the completed sentence did not run:\n%s", out)
	}
}

// An existing server takes the serve-static edit, then runs:
// the registration lands in the server's file, the partner
// confirmation speaks, and the final line gives the app's
// address from the serve statement's own port. The repeat
// sentence finds the route already serving and goes straight
// to the run.
func TestServeWasmAppExistingServer(t *testing.T) {
	if !portFree("8080") {
		t.Skip("port 8080 is in use; the existing-server serve proof needs it")
	}
	yes := func(string) (string, error) { return "y", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
		"app/go.mod":  "module example.com/app\n\ngo 1.21\n",
		"app/main.go": wasmMainSrc,
	}))
	writeFile(t, dir, "app/main.wasm", "not a real binary — presence is what the flow checks")
	oldOpen := openBrowserURL
	openBrowserURL = func(string) error { return nil }
	t.Cleanup(func() { openBrowserURL = oldOpen })
	t.Cleanup(func() { stopProgramAtChatEnd(); sessionProgram = nil })

	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in app", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	src, err := os.ReadFile(filepath.Join(dir, "jim", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(src), `http.FileServer(http.Dir("app"))`) {
		t.Errorf("the registration did not land in the server's file:\n%s", src)
	}
	if !strings.Contains(out, "now serves the app folder's files at /static/") {
		t.Errorf("the serve-static confirmation is missing:\n%s", out)
	}
	if !strings.Contains(out, "The wasm app in app is served by your Go server — open http://localhost:8080/static/ in your browser to run it.") {
		t.Errorf("the final address line is missing:\n%s", out)
	}
	if !programRunning() {
		t.Fatalf("the server is not running after the serve sentence:\n%s", out)
	}

	out = captureStdout(t, func() { claimChain("serve the wasm app in app", yes, &UserProfile{}) })
	if strings.Contains(out, "Here's the edit") {
		t.Errorf("the repeat sentence re-planned an already-serving route:\n%s", out)
	}
	if !strings.Contains(out, "The wasm app in app is served by your Go server") {
		t.Errorf("the repeat sentence lost the address line:\n%s", out)
	}
}

// A /static/ route held by a different folder is the
// serve-static conflict in its own words: nothing is
// written, nothing starts.
func TestServeWasmAppConflict(t *testing.T) {
	yes := func(string) (string, error) { return "y", nil }
	conflicted := strings.Replace(staticJimMain,
		"\thttp.ListenAndServe",
		"\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"other\"))))\n\thttp.ListenAndServe", 1)
	dir := staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": conflicted,
		"app/go.mod":  "module example.com/app\n\ngo 1.21\n",
		"app/main.go": wasmMainSrc,
	}))
	writeFile(t, dir, "app/main.wasm", "not a real binary — presence is what the flow checks")
	before, err := os.ReadFile(filepath.Join(dir, "jim", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in app", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	if !strings.Contains(out, "/static/ already serves a different folder") ||
		!strings.Contains(out, `"other"`) {
		t.Errorf("the conflict was not named:\n%s", out)
	}
	after, err := os.ReadFile(filepath.Join(dir, "jim", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Errorf("the conflict wrote to the server's file")
	}
	if sessionProgram != nil {
		t.Errorf("the conflict left a program in the session")
	}
}

// The full smoke, in the site package's HTTP-proof style: a
// project with nothing but an app gets its server created,
// the app is served over real HTTP — the host page at
// /static/, the program bytes at /static/main.wasm — and the
// stop sentence frees the port.
func TestServeWasmAppHTTPSmoke(t *testing.T) {
	if !portFree("8080") {
		t.Skip("port 8080 is in use; the serve smoke needs it")
	}
	yes := func(string) (string, error) { return "y", nil }
	dir := staticAssetsFixture(t, staticAssetsFiles(nil))
	captureStdout(t, func() {
		claimChain("create a folder jim", yes, &UserProfile{})
		claimChain("add a wasm app in jim", yes, &UserProfile{})
	})
	writeFile(t, dir, "jim/main.wasm", "fake wasm bytes for the smoke")
	oldOpen := openBrowserURL
	openBrowserURL = func(string) error { return nil }
	t.Cleanup(func() { openBrowserURL = oldOpen })
	t.Cleanup(func() { stopProgramAtChatEnd(); sessionProgram = nil })

	out := captureStdout(t, func() {
		if !claimChain("serve the wasm app in jim", yes, &UserProfile{}) {
			t.Error("the serve sentence was not claimed")
		}
	})
	if !strings.Contains(out, "The server in server/main.go now serves the wasm app in jim at /static/") {
		t.Errorf("the created-server confirmation is missing:\n%s", out)
	}
	if !strings.Contains(out, "open http://localhost:8080/static/ in your browser") {
		t.Errorf("the final address line is missing:\n%s", out)
	}
	if !programRunning() {
		t.Fatalf("the created server is not running:\n%s", out)
	}

	client := &http.Client{Timeout: 2 * time.Second}
	get := func(path string) (int, string) {
		deadline := time.Now().Add(10 * time.Second)
		for {
			resp, err := client.Get("http://localhost:8080" + path)
			if err == nil {
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				return resp.StatusCode, string(body)
			}
			if time.Now().After(deadline) {
				t.Fatalf("GET %s never answered: %v", path, err)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}
	code, body := get("/static/")
	if code != 200 {
		t.Errorf("GET /static/ = %d, want 200", code)
	}
	if !strings.Contains(body, "wasm_exec.js") || !strings.Contains(body, "main.wasm") {
		t.Errorf("GET /static/ did not return the app's host page:\n%s", body)
	}
	code, body = get("/static/main.wasm")
	if code != 200 || body != "fake wasm bytes for the smoke" {
		t.Errorf("GET /static/main.wasm = %d %q, want 200 and the placed bytes", code, body)
	}

	captureStdout(t, func() { claimChain("stop the program", yes, &UserProfile{}) })
	deadline := time.Now().Add(5 * time.Second)
	for !portFree("8080") && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
	if !portFree("8080") {
		t.Errorf("the port is still held after the stop sentence")
	}
}
