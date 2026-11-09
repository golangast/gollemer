package analyze

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// scaffoldFixtureFiles is the minimal project the scaffold
// tests build on: a module whose jim folder holds a
// package-main program.
func scaffoldFixtureFiles() map[string]string {
	return map[string]string{
		"go.mod":      "module example.com/scaffold\n\ngo 1.26\n",
		"jim/main.go": "package main\n\nimport \"net/http\"\n\nfunc main() { _ = http.ListenAndServe(\":8080\", nil) }\n",
	}
}

// The scaffold lands as four files: the shared header and
// footer, the page's body, and the handler — with the nav
// and footer markup being the component piece's own
// constants, the handler parsing the three files from
// project-root-relative literals, and no route wired.
func TestPlanTemplatedPageFiles(t *testing.T) {
	dir, p := serveProject(t, scaffoldFixtureFiles())
	plan, refusal := p.PlanEdit(EditRequest{Kind: EditTemplatedPage, SiteFolder: "jim", SiteName: "home"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan = %v, refusal = %q", plan, refusal)
	}
	if len(plan.Files) != 4 {
		t.Fatalf("plan writes %d files, want 4", len(plan.Files))
	}
	if !strings.Contains(plan.Summary, "Create the home templated page in jim") {
		t.Errorf("plan summary doesn't name the scaffold:\n%s", plan.Summary)
	}
	applyServePlan(t, dir, plan)
	read := func(rel string) string {
		b, err := os.ReadFile(filepath.Join(p.Root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		return string(b)
	}
	header := read("jim/templates/header.html")
	for _, want := range []string{`{{define "header"}}`, MaterialNavigationBarSnippet, "--primary:", ".site-nav {", ".site-footer {", "styles live in the header"} {
		if !strings.Contains(header, want) {
			t.Errorf("header missing %q:\n%s", want, header)
		}
	}
	if strings.Contains(header, "<script") {
		t.Error("header carries a script — the package is zero-JavaScript")
	}
	footer := read("jim/templates/footer.html")
	for _, want := range []string{`{{define "footer"}}`, MaterialFooterSnippet, "</body>"} {
		if !strings.Contains(footer, want) {
			t.Errorf("footer missing %q:\n%s", want, footer)
		}
	}
	body := read("jim/templates/home.html")
	for _, want := range []string{`{{template "header" .}}`, "Welcome to home", `{{template "footer" .}}`} {
		if !strings.Contains(body, want) {
			t.Errorf("body missing %q:\n%s", want, body)
		}
	}
	handler := read("jim/homepage.go")
	for _, want := range []string{
		"package main", "func HomePage(w http.ResponseWriter, r *http.Request)",
		`template.ParseFiles("jim/templates/home.html", "jim/templates/header.html", "jim/templates/footer.html")`,
		`page.ExecuteTemplate(w, "home.html", nil)`, "run the server from the project root", "http.Error",
	} {
		if !strings.Contains(handler, want) {
			t.Errorf("handler missing %q:\n%s", want, handler)
		}
	}
	for _, banned := range []string{"template.Must", "HandleFunc"} {
		if strings.Contains(handler, banned) {
			t.Errorf("handler carries %q — parse errors are answered honestly and no route is wired:\n%s", banned, handler)
		}
	}
}

// The pinned refusals: a missing folder is a two-step, an
// existing shared file is never overwritten, a bad name and
// the shared pieces' own names refuse in their own words,
// a mixed-package folder can't say which package the
// handler joins, and a taken handler name refuses.
func TestPlanTemplatedPageRefusals(t *testing.T) {
	cases := []struct {
		name   string
		files  map[string]string
		folder string
		page   string
		want   string
	}{
		{"missing folder", scaffoldFixtureFiles(), "nook", "home", "no folder nook"},
		{"header exists", withFile(scaffoldFixtureFiles(), "jim/templates/header.html", "mine\n"), "jim", "home", "already there"},
		{"body exists", withFile(scaffoldFixtureFiles(), "jim/templates/home.html", "mine\n"), "jim", "home", "already there"},
		{"handler file exists", withFile(scaffoldFixtureFiles(), "jim/homepage.go", "package main\n"), "jim", "home", "already there"},
		{"bad name", scaffoldFixtureFiles(), "jim", "2home", "isn't a plain page name"},
		{"shared name", scaffoldFixtureFiles(), "jim", "header", "shared pieces"},
		{"mixed packages", withFile(scaffoldFixtureFiles(), "jim/other.go", "package other\n"), "jim", "home", "mixes package"},
		{"handler taken", withFile(scaffoldFixtureFiles(), "jim/extra.go", "package main\n\nimport \"net/http\"\n\nfunc HomePage(w http.ResponseWriter, r *http.Request) {}\n"), "jim", "home", "already has a HomePage"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, p := serveProject(t, c.files)
			plan, refusal := p.PlanEdit(EditRequest{Kind: EditTemplatedPage, SiteFolder: c.folder, SiteName: c.page})
			if plan != nil || !strings.Contains(refusal, c.want) {
				t.Errorf("plan = %v, refusal = %q; want a refusal containing %q", plan != nil, refusal, c.want)
			}
		})
	}
}

func withFile(files map[string]string, rel, content string) map[string]string {
	out := map[string]string{}
	for k, v := range files {
		out[k] = v
	}
	out[rel] = content
	return out
}

// The real proof: the scaffold is created in a fixture
// project, the handler is wired by hand at /home, the
// project builds, the server runs, and the page that comes
// back carries the navigation bar AND the footer AND the
// body — the composition rendered, not just three files on
// disk. Editing only the footer file and restarting changes
// the served footer, proving the parts are separate files.
func TestTemplatedPageHTTPSmoke(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	main := fmt.Sprintf(`package main

import (
	"fmt"
	"net/http"
)

func main() {
	// ROUTES
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "root") })
	_ = http.ListenAndServe("127.0.0.1:%d", nil)
}
`, port)
	root, p := serveProject(t, map[string]string{
		"go.mod":      "module example.com/scaffoldsmoke\n\ngo 1.26\n",
		"jim/main.go": main,
	})
	plan, refusal := p.PlanEdit(EditRequest{Kind: EditTemplatedPage, SiteFolder: "jim", SiteName: "home"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan = %v, refusal = %q", plan, refusal)
	}
	applyServePlan(t, root, plan)
	// Wire the handler by hand, the way the wiring sentence
	// would: one registration line in the fixture's main.
	mainPath := filepath.Join(root, "jim", "main.go")
	src, err := os.ReadFile(mainPath)
	if err != nil {
		t.Fatal(err)
	}
	wired := strings.Replace(string(src), "// ROUTES", "http.HandleFunc(\"/home\", HomePage)", 1)
	if wired == string(src) {
		t.Fatal("the fixture's route marker is gone")
	}
	if err := os.WriteFile(mainPath, []byte(wired), 0o644); err != nil {
		t.Fatal(err)
	}
	bin := filepath.Join(t.TempDir(), "scaffoldsmoke")
	build := exec.Command("go", "build", "-o", bin, "./jim")
	build.Dir = root
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("fixture build: %v\n%s", err, out)
	}
	url := fmt.Sprintf("http://127.0.0.1:%d/home", port)
	start := func() (*exec.Cmd, string) {
		cmd := exec.Command(bin)
		cmd.Dir = root // the handler's paths are project-root-relative
		var log strings.Builder
		cmd.Stdout, cmd.Stderr = &log, &log
		if err := cmd.Start(); err != nil {
			t.Fatalf("start server: %v", err)
		}
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) {
			resp, err := http.Get(url)
			if err == nil {
				b, _ := io.ReadAll(resp.Body)
				_ = resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					return cmd, string(b)
				}
			}
			time.Sleep(100 * time.Millisecond)
		}
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		t.Fatalf("the page never answered at %s; server log:\n%s", url, log.String())
		return nil, ""
	}
	cmd, page := start()
	for _, want := range []string{`class="site-nav"`, `class="site-footer"`, "Welcome to home"} {
		if !strings.Contains(page, want) {
			t.Errorf("served page missing %q — the composition did not render:\n%s", want, page)
		}
	}
	_ = cmd.Process.Kill()
	_ = cmd.Wait()
	// Edit ONLY the footer file, restart, and the served
	// footer changes while the body stands.
	footerPath := filepath.Join(root, "jim", "templates", "footer.html")
	fsrc, err := os.ReadFile(footerPath)
	if err != nil {
		t.Fatal(err)
	}
	edited := strings.Replace(string(fsrc), "TODO: your footer's line", "The smoke page's own footer line", 1)
	if edited == string(fsrc) {
		t.Fatal("the footer's TODO line is gone")
	}
	if err := os.WriteFile(footerPath, []byte(edited), 0o644); err != nil {
		t.Fatal(err)
	}
	cmd, page = start()
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()
	if !strings.Contains(page, "The smoke page's own footer line") {
		t.Errorf("after editing only the footer file, the served page doesn't carry the new footer:\n%s", page)
	}
	if !strings.Contains(page, "Welcome to home") {
		t.Errorf("after the footer edit, the body is missing from the served page:\n%s", page)
	}
	t.Logf("SMOKE OK: first GET carried the nav, the footer, and the body; after editing only the footer file and restarting, the new footer line was served with the body intact")
}
