package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	for _, want := range []string{`http.HandleFunc("/index.html"`, `http.ServeFile(w, r, "index.html")`, `http.ListenAndServe(":8080", nil)`} {
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
// second server.
func TestPlanServeHTMLComposesWithExistingServer(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":  "module example.com/serve\\n\\ngo 1.21\\n",
		"main.go": WebServerSource,
	})
	plan, refusal := planServe(t, proj, "main.go", "index.html")
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	goFile := plan.Files[len(plan.Files)-1]
	out := string(goFile.After)
	if n := count(out, "http.ListenAndServe"); n != 1 {
		t.Errorf("composed file has %d ListenAndServe calls, want exactly 1:\n%s", n, out)
	}
	regAt := strings.Index(out, `http.HandleFunc("/index.html"`)
	serveAt := strings.Index(out, "http.ListenAndServe")
	if regAt < 0 || serveAt < 0 || regAt > serveAt {
		t.Errorf("registration is not before the serve call:\n%s", out)
	}
	if !strings.Contains(out, `"hello — your webserver is running"`) {
		t.Errorf("the existing handler was disturbed:\n%s", out)
	}
	applyServePlan(t, dir, plan)
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
	if strings.Index(out, `http.HandleFunc("/index.html"`) > strings.Index(out, "http.ListenAndServe") {
		t.Errorf("registration does not precede the serve call:\n%s", out)
	}
}

// Already serving THIS page: nothing to add. Route taken by a
// different handler: a conflict. A custom mux: the by-hand
// answer naming the mux.
func TestPlanServeHTMLServingStates(t *testing.T) {
	servingThis := `package main

import (
	"fmt"
	"net/http"
)

func main() {
	http.HandleFunc("/index.html", func(w http.ResponseWriter, r *http.Request) {
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
	_, refusal = planServe(t, proj4, "main.go", "index.html")
	if !strings.Contains(refusal, "mux.HandleFunc") || !strings.Contains(refusal, "won't add a second ListenAndServe") {
		t.Errorf("custom-mux refusal does not hand over the line:\n%s", refusal)
	}
}

// A page in a subfolder of the Go file's folder: the ServeFile
// argument is the path between them.
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
	if !strings.Contains(out, `http.ServeFile(w, r, "pages/home.html")`) ||
		!strings.Contains(out, `http.HandleFunc("/pages/home.html"`) {
		t.Errorf("subpath page wired wrong:\n%s", out)
	}
	if !strings.Contains(strings.Join(plan.Notes, "\n"), "Run it from the jim folder") {
		t.Errorf("run-from note wrong for a folder program:\n%s", strings.Join(plan.Notes, "\n"))
	}
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
