package analyze

import (
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// serveProject writes a small module and analyzes it.
func serveProject(t *testing.T, files map[string]string) (string, *Project) {
	t.Helper()
	dir := t.TempDir()
	for rel, src := range files {
		abs := filepath.Join(dir, rel)
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	proj, err := Analyze(dir)
	if err != nil || proj == nil {
		t.Fatalf("fixture project did not analyze: %v", err)
	}
	return dir, proj
}

// applyServePlan writes the plan's files into dir and verifies
// against a fresh analysis — the spine's own sequence, minus
// the gate.
func applyServePlan(t *testing.T, dir string, plan *EditPlan) {
	t.Helper()
	for _, f := range plan.Files {
		abs := filepath.Join(dir, f.Rel)
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, f.After, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	fresh, err := Analyze(dir)
	if err != nil || fresh == nil {
		t.Fatalf("edited project did not analyze: %v", err)
	}
	if !plan.Verify(fresh) {
		t.Errorf("plan does not verify against the edited project")
	}
}

func planServe(t *testing.T, proj *Project, goRel, htmlRel string) (*EditPlan, string) {
	t.Helper()
	return proj.PlanEdit(EditRequest{Kind: EditServeHTML, File: goRel, ServeHTML: htmlRel})
}

func count(s, sub string) int { return strings.Count(s, sub) }

// A missing Go file: the plan creates BOTH the page and the
// whole program, page first. (The fixture project carries a
// subpackage so it analyzes; the chat flow's empty-index
// fallback covers the no-packages-at-all project, and the
// session tests exercise it.)
func TestPlanServeHTMLMissingFileCreatesBoth(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":         "module example.com/serve\\n\\ngo 1.21\\n",
		"store/store.go": "package store\n\nfunc Load() int { return 1 }\n",
	})
	plan, refusal := planServe(t, proj, "main.go", "index.html")
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	if len(plan.Files) != 2 || plan.Files[0].Rel != "index.html" || plan.Files[1].Rel != "main.go" {
		t.Fatalf("plan files = %+v, want [index.html main.go]", plan.Files)
	}
	if plan.Files[0].Before != nil || !strings.Contains(string(plan.Files[0].After), "TODO: your page") {
		t.Errorf("page half is not the starter page:\n%s", plan.Files[0].After)
	}
	goSrc := string(plan.Files[1].After)
	// An index page registers at its folder's address "/",
	// never at its own name (see serveRouteOf).
	for _, want := range []string{`http.HandleFunc("/", func`, `http.ServeFile(w, r, "index.html")`, `http.ListenAndServe(":8080", nil)`} {
		if !strings.Contains(goSrc, want) {
			t.Errorf("program missing %q:\n%s", want, goSrc)
		}
	}
	applyServePlan(t, dir, plan)
}

// A bare package clause becomes the whole program; an existing
// page is not among the plan's files.
func TestPlanServeHTMLBareClauseAndExistingPage(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":     "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go":    "package main\n",
		"index.html": "<html><body>mine</body></html>\n",
	})
	plan, refusal := planServe(t, proj, "main.go", "index.html")
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "main.go" {
		t.Fatalf("plan files = %+v, want only main.go (the page is untouched)", plan.Files)
	}
	if !strings.Contains(string(plan.Files[0].After), `http.ServeFile(w, r, "index.html")`) {
		t.Errorf("program does not serve the page:\n%s", plan.Files[0].After)
	}
	joined := strings.Join(plan.Notes, "\n")
	if !strings.Contains(joined, "already exists — it is not changed") {
		t.Errorf("plan does not say the page is untouched:\n%s", joined)
	}
	if !strings.Contains(joined, "Run it from the project root") {
		t.Errorf("plan does not say where to run it from:\n%s", joined)
	}
}

// An empty main is filled; a main-less file grows one; imports
// arrive with either.
func TestPlanServeHTMLEmptyMainAndNoMain(t *testing.T) {
	for name, src := range map[string]string{
		"empty main": "package main\n\nfunc main() {}\n",
		"no main":    "package main\n\nfunc helper() int { return 1 }\n",
	} {
		_, proj := serveProject(t, map[string]string{
			"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
			"main.go": src,
		})
		plan, refusal := planServe(t, proj, "main.go", "page.html")
		if refusal != "" {
			t.Fatalf("%s: refused: %s", name, refusal)
		}
		goFile := plan.Files[len(plan.Files)-1]
		out := string(goFile.After)
		for _, want := range []string{`"net/http"`, `"fmt"`, `http.HandleFunc("/page.html"`, `http.ServeFile(w, r, "page.html")`, "http.ListenAndServe"} {
			if !strings.Contains(out, want) {
				t.Errorf("%s: result missing %q:\n%s", name, want, out)
			}
		}
		if name == "no main" && !strings.Contains(out, "func helper() int { return 1 }") {
			t.Errorf("existing code was lost:\n%s", out)
		}
	}
}

// A non-empty main with no server: the by-hand refusal, with
// the recipe's exact lines.
func TestPlanServeHTMLBusyMainRefuses(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"working\") }\n",
	})
	_, refusal := planServe(t, proj, "main.go", "index.html")
	if !strings.Contains(refusal, "already does its own work") ||
		!strings.Contains(refusal, `http.ServeFile(w, r, "index.html")`) ||
		!strings.Contains(refusal, "http.ListenAndServe") {
		t.Errorf("refusal is not the by-hand answer:\n%s", refusal)
	}
}

// A library package: serving is a program's job.
func TestPlanServeHTMLLibraryRefuses(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"lib.go":  "package lib\n\nfunc Thing() {}\n",
		"main.go": "package main\n\nfunc main() {}\n",
	})
	_, refusal := planServe(t, proj, "lib.go", "index.html")
	if !strings.Contains(refusal, "package lib") || !strings.Contains(refusal, "serve index.html to main.go") {
		t.Errorf("refusal does not name the program:\n%s", refusal)
	}
}

// The compose state: a file already serving on the default mux
// gains ONE registration before its ListenAndServe — never a
// second server. (The page here is home.html; an index page
// into this same server is the conflict case below, because
// the address an index page needs — "/" — is the webserver's
// own.)
func TestPlanServeHTMLComposesWithExistingServer(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": WebServerSource,
	})
	plan, refusal := planServe(t, proj, "main.go", "home.html")
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	goFile := plan.Files[len(plan.Files)-1]
	out := string(goFile.After)
	if n := count(out, "http.ListenAndServe"); n != 1 {
		t.Errorf("composed file has %d ListenAndServe calls, want exactly 1:\n%s", n, out)
	}
	regAt := strings.Index(out, `http.HandleFunc("/home.html"`)
	serveAt := strings.Index(out, "http.ListenAndServe")
	if regAt < 0 || serveAt < 0 || regAt > serveAt {
		t.Errorf("registration is not before the serve call:\n%s", out)
	}
	if !strings.Contains(out, `"hello — your webserver is running"`) {
		t.Errorf("the existing handler was disturbed:\n%s", out)
	}
	applyServePlan(t, dir, plan)
}

// An index page asked into a server whose "/" already
// answers with the friendly line: the folder address the
// page needs is taken by a different handler, so the recipe
// refuses rather than registering an address the page can
// never be served at (the old "/index.html" registration
// redirected INTO that handler — the page was unviewable
// either way).
func TestPlanServeHTMLIndexIntoTakenRootConflicts(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": WebServerSource,
	})
	_, refusal := planServe(t, proj, "main.go", "index.html")
	if !strings.Contains(refusal, "already answers http://localhost:8080/ with a different handler") {
		t.Errorf("index-into-taken-root refusal = %q", refusal)
	}
}

// Compose when the registration must land inside a nested
// block: the serve call in an if statement still gets its
// registration first, in the enclosing block.
func TestPlanServeHTMLComposesBeforeNestedServe(t *testing.T) {
	src := `package main

import (
	"fmt"
	"net/http"
)

func main() {
	fmt.Println("starting")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("stopped:", err)
	}
}
`
	_, proj := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": src,
	})
	plan, refusal := planServe(t, proj, "main.go", "index.html")
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	out := string(plan.Files[len(plan.Files)-1].After)
	if count(out, "http.ListenAndServe") != 1 {
		t.Errorf("second server emitted:\n%s", out)
	}
	if strings.Index(out, `http.HandleFunc("/", func`) > strings.Index(out, "http.ListenAndServe") {
		t.Errorf("registration does not precede the serve call:\n%s", out)
	}
}

// Already serving THIS page: nothing to add. Route taken by a
// different handler: a conflict. A custom mux: the by-hand
// answer naming the mux.
func TestPlanServeHTMLServingStates(t *testing.T) {
	// The already-wired shape is the CURRENT one: an index
	// page registered at its folder's address.
	servingThis := `package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "index.html")
	})
	fmt.Println("listening")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("stopped:", err)
	}
}
`
	_, proj := serveProject(t, map[string]string{
		"go.mod":     "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go":    servingThis,
		"index.html": "<html></html>\n",
	})
	_, refusal := planServe(t, proj, "main.go", "index.html")
	if !strings.Contains(refusal, "already serves") {
		t.Errorf("same-page state refusal = %q", refusal)
	}

	// Same wiring, page missing: the plan is the page alone.
	dir, proj2 := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": servingThis,
	})
	plan, refusal := planServe(t, proj2, "main.go", "index.html")
	if refusal != "" {
		t.Fatalf("page-only state refused: %s", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "index.html" {
		t.Fatalf("page-only plan files = %+v", plan.Files)
	}
	applyServePlan(t, dir, plan)

	conflict := strings.Replace(servingThis, `http.ServeFile(w, r, "index.html")`, `fmt.Fprintln(w, "other")`, 1)
	_, proj3 := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": conflict,
	})
	_, refusal = planServe(t, proj3, "main.go", "index.html")
	if !strings.Contains(refusal, "different handler") {
		t.Errorf("conflict refusal = %q", refusal)
	}

	mux := `package main

import (
	"fmt"
	"net/http"
)

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "hi")
	})
	if err := http.ListenAndServe(":8080", mux); err != nil {
		fmt.Println("stopped:", err)
	}
}
`
	_, proj4 := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": mux,
	})
	// A non-index page: the mux's own "/" registration is not
	// this page's route, so the state is the custom-server
	// one — the by-hand line written against the mux. (An
	// index page into this file is the route-conflict state:
	// the folder address it needs is answered, by the mux.)
	_, refusal = planServe(t, proj4, "main.go", "home.html")
	if !strings.Contains(refusal, "mux.HandleFunc") || !strings.Contains(refusal, "won't add a second ListenAndServe") {
		t.Errorf("custom-mux refusal does not hand over the line:\n%s", refusal)
	}
}

// A page in a subfolder of the Go file's folder: the route is
// the page's path within the program's folder, and the
// ServeFile argument is the page's project-root-relative
// path — the two are different strings on purpose.
func TestPlanServeHTMLSubpathPage(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":      "module example.com/serve\\n\\ngo 1.21\\n",
		"jim/main.go": "package main\n",
	})
	plan, refusal := planServe(t, proj, "jim/main.go", "jim/pages/home.html")
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	out := string(plan.Files[len(plan.Files)-1].After)
	if !strings.Contains(out, `http.ServeFile(w, r, "jim/pages/home.html")`) ||
		!strings.Contains(out, `http.HandleFunc("/pages/home.html"`) {
		t.Errorf("subpath page wired wrong:\n%s", out)
	}
	if !strings.Contains(strings.Join(plan.Notes, "\n"), "Run it from the project root") {
		t.Errorf("run-from note wrong for a folder program:\n%s", strings.Join(plan.Notes, "\n"))
	}
}

// runWiredProgram builds the wired project in dir and runs
// the program the way the chat's own runner runs it: the
// working directory is the PROJECT ROOT, never the program's
// folder. The returned fetch GETs one route from the running
// program. Presence-only assertions are how the bare-path
// wiring survived review — this helper exists so the serve
// tests assert what a browser would actually get.
func runWiredProgram(t *testing.T, dir, pkg string) func(route string) (int, string) {
	t.Helper()
	bin := filepath.Join(dir, "servefetch-bin")
	build := exec.Command("go", "build", "-o", bin, pkg)
	build.Dir = dir
	build.Env = append(os.Environ(), "GOEXPERIMENT=")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("the wired project does not build: %v\n%s", err, out)
	}
	cmd := exec.Command(bin)
	cmd.Dir = dir // the runner's working directory: the project root
	if err := cmd.Start(); err != nil {
		t.Fatalf("the wired program does not start: %v", err)
	}
	t.Cleanup(func() {
		cmd.Process.Kill()
		cmd.Wait()
	})
	client := &http.Client{Timeout: 2 * time.Second}
	return func(route string) (int, string) {
		deadline := time.Now().Add(30 * time.Second)
		for {
			resp, err := client.Get("http://localhost:8080" + route)
			if err == nil {
				body, _ := io.ReadAll(resp.Body)
				resp.Body.Close()
				return resp.StatusCode, string(body)
			}
			if time.Now().After(deadline) {
				t.Fatalf("the program never answered %s (last error: %v)", route, err)
			}
			time.Sleep(150 * time.Millisecond)
		}
	}
}

// A second page composed into a file already serving a first
// page: the plan verifies against its own result (the verify
// must find THIS page's ServeFile among the file's calls, not
// just the first), and asking for either again is the
// already-serves answer. The first page is the folder's
// index — the recipe creates the whole program for it, so
// its "/" is the page's own (composing an index page onto a
// server that already owns "/" is the conflict state, pinned
// above).
func TestPlanServeHTMLSecondPageComposes(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":         "module example.com/serve\n\ngo 1.21\n",
		"store/store.go": "package store\n\nfunc Load() int { return 1 }\n",
	})
	plan, refusal := planServe(t, proj, "api/main.go", "api/index.html")
	if refusal != "" {
		t.Fatalf("first page refused: %s", refusal)
	}
	applyServePlan(t, dir, plan)

	fresh, err := Analyze(dir)
	if err != nil || fresh == nil {
		t.Fatalf("project did not re-analyze: %v", err)
	}
	plan2, refusal := planServe(t, fresh, "api/main.go", "api/home.html")
	if refusal != "" {
		t.Fatalf("second page refused: %s", refusal)
	}
	applyServePlan(t, dir, plan2) // asserts plan2 verifies
	out, err := os.ReadFile(filepath.Join(dir, "api", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`http.ServeFile(w, r, "api/index.html")`, `http.ServeFile(w, r, "api/home.html")`} {
		if !strings.Contains(string(out), want) {
			t.Errorf("two-page file missing %q:\n%s", want, out)
		}
	}
	if n := count(string(out), "http.ListenAndServe"); n != 1 {
		t.Errorf("two-page file has %d ListenAndServe calls, want exactly 1", n)
	}

	fresh2, err := Analyze(dir)
	if err != nil || fresh2 == nil {
		t.Fatalf("project did not re-analyze: %v", err)
	}
	_, refusal = planServe(t, fresh2, "api/main.go", "api/home.html")
	if !strings.Contains(refusal, "already serves") {
		t.Errorf("re-ask of the second page = %q, want the already-serves answer", refusal)
	}
	_, refusal = planServe(t, fresh2, "api/main.go", "api/index.html")
	if !strings.Contains(refusal, "already serves") {
		t.Errorf("re-ask of the index page = %q, want the already-serves answer", refusal)
	}
}

// The fetched-page contract: a project wired by this recipe
// serves its page over HTTP when the program runs from the
// project root — for a created program in a subfolder, for a
// composed server in a subfolder with the page already there,
// and for a root program with an existing page.
func TestServeHTMLPageFetchesFromProjectRoot(t *testing.T) {
	// A program the recipe creates in a subfolder, the page
	// created with it.
	t.Run("created program in a subfolder", func(t *testing.T) {
		dir, proj := serveProject(t, map[string]string{
			"go.mod":         "module example.com/serve\n\ngo 1.21\n",
			"store/store.go": "package store\n\nfunc Load() int { return 1 }\n",
		})
		plan, refusal := planServe(t, proj, "api/main.go", "api/home.html")
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		applyServePlan(t, dir, plan)
		fetch := runWiredProgram(t, dir, "./api")
		code, body := fetch("/home.html")
		if code != 200 || !strings.Contains(body, "TODO: your page") {
			t.Errorf("GET /home.html from the project root = %d, body:\n%s", code, body)
		}
	})

	// The recipe composed onto a server already running in a
	// subfolder; the page already exists, is not overwritten,
	// and is what comes back over HTTP. The server's own root
	// handler answers too.
	t.Run("composed server in a subfolder, page already there", func(t *testing.T) {
		dir, proj := serveProject(t, map[string]string{
			"go.mod":        "module example.com/serve\n\ngo 1.21\n",
			"api/main.go":   WebServerSource,
			"api/home.html": "<html><body><h1>paces marker page</h1></body></html>\n",
		})
		plan, refusal := planServe(t, proj, "api/main.go", "api/home.html")
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		applyServePlan(t, dir, plan)
		fetch := runWiredProgram(t, dir, "./api")
		code, body := fetch("/home.html")
		if code != 200 || !strings.Contains(body, "paces marker page") {
			t.Errorf("GET /home.html from the project root = %d, body:\n%s", code, body)
		}
		if code, _ := fetch("/"); code != 200 {
			t.Errorf("GET / from the project root = %d, want the existing handler's 200", code)
		}
	})

	// An index page, created with its program in a subfolder:
	// it answers at its folder's address — and at its own
	// name too, because the ServeFile redirect lands on the
	// folder address, which now serves it. (At the page's
	// own registration — the pre-fix wiring — both addresses
	// 404'd: the redirect target had nothing behind it.)
	t.Run("created program in a subfolder, index page", func(t *testing.T) {
		dir, proj := serveProject(t, map[string]string{
			"go.mod":         "module example.com/serve\n\ngo 1.21\n",
			"store/store.go": "package store\n\nfunc Load() int { return 1 }\n",
		})
		plan, refusal := planServe(t, proj, "api/main.go", "api/index.html")
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		if !strings.Contains(strings.Join(plan.Notes, "\n"), "index page") {
			t.Errorf("index plan's notes don't say why the address is the folder's:\n%s", strings.Join(plan.Notes, "\n"))
		}
		applyServePlan(t, dir, plan)
		fetch := runWiredProgram(t, dir, "./api")
		for _, route := range []string{"/", "/index.html"} {
			code, body := fetch(route)
			if code != 200 || !strings.Contains(body, "TODO: your page") {
				t.Errorf("GET %s from the project root = %d, body:\n%s", route, code, body)
			}
		}
	})

	// A program at the project root, the page already there.
	// (The page is home.html, not index.html: net/http's
	// ServeFile redirects every /index.html request to ./ —
	// stdlib behavior for any wiring, not this recipe's path
	// contract — so an index page is fetched at its folder
	// address, as in the subtest above.)
	t.Run("root program, page already there", func(t *testing.T) {
		dir, proj := serveProject(t, map[string]string{
			"go.mod":         "module example.com/serve\n\ngo 1.21\n",
			"store/store.go": "package store\n\nfunc Load() int { return 1 }\n",
			"home.html":      "<html><body><h1>root marker page</h1></body></html>\n",
		})
		plan, refusal := planServe(t, proj, "main.go", "home.html")
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		applyServePlan(t, dir, plan)
		fetch := runWiredProgram(t, dir, ".")
		code, body := fetch("/home.html")
		if code != 200 || !strings.Contains(body, "root marker page") {
			t.Errorf("GET /home.html from the project root = %d, body:\n%s", code, body)
		}
	})
}

// A page outside the Go file's folder is refused by the
// planner itself (the chat layer never sends one, and the
// planner stays honest if asked directly).
func TestPlanServeHTMLPageOutsideFolderRefuses(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":      "module example.com/serve\\n\\ngo 1.21\\n",
		"jim/main.go": "package main\n",
	})
	_, refusal := planServe(t, proj, "jim/main.go", "index.html")
	if !strings.Contains(refusal, "same folder") {
		t.Errorf("outside-folder refusal = %q", refusal)
	}
}
