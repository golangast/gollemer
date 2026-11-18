package analyze

import (
	"strings"
	"testing"
)

// The serve-the-wasm-app sentence's own reading of the
// project (2026-11-10, John's screenshot + his serving
// shape): only the project's own module counts — a server
// or a /static/ registration inside a NESTED module (a
// folder carrying its own go.mod under the root) belongs
// to that module, which this project can neither build nor
// run, so it is invisible to the sentence — and the
// sentence serves from exactly two homes, the dedicated
// server in server/ or a program at the project root. A
// registration serving the app from another program deeper
// in the tree is named, never run and never extended; the
// app gets the server/ arrangement instead. The
// serve-static family's project-wide scan
// (StaticRouteFacts) is untouched — the last test here
// pins both readings of the same project side by side.

const nestedServerForJim = `package main

import "net/http"

func main() {
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim"))))
	http.ListenAndServe(":9090", nil)
}
`

const deepServerForJim = `package main

import "net/http"

func main() {
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim"))))
	http.ListenAndServe(":9091", nil)
}
`

func TestFileInProjectModule(t *testing.T) {
	_, proj := serveWasmProject(t, map[string]string{
		"paces/project/go.mod":      "module example.com/nested\n\ngo 1.21\n",
		"paces/project/api/main.go": nestedServerForJim,
		"tools/deep/main.go":        deepServerForJim,
		"jim/go.mod":                "module jim\n\ngo 1.27.0\n",
		"jim/main.go":               "//go:build js && wasm\n\npackage main\n",
	})
	for rel, want := range map[string]bool{
		"main.go":                   true, // the root itself
		"tools/deep/main.go":        true, // no go.mod on the way down
		"doc.go":                    true,
		"paces/project/api/main.go": false, // behind paces/project/go.mod
		"jim/main.go":               false, // the app is its own module
	} {
		if got := proj.fileInProjectModule(rel); got != want {
			t.Errorf("fileInProjectModule(%q) = %v, want %v", rel, got, want)
		}
	}
}

// John's shape exactly: the only /static/ registration for
// the app lives inside a nested module. The sentence's
// facts see no route, no conflict, no server — and the
// create-a-server plan assembles.
func TestServeWasmNestedModuleInvisible(t *testing.T) {
	dir, proj := serveWasmProject(t, map[string]string{
		"paces/project/go.mod":      "module example.com/nested\n\ngo 1.21\n",
		"paces/project/api/main.go": nestedServerForJim,
	})
	facts := proj.ServeWasmRouteFacts("jim")
	if facts.ServedFile != "" || facts.OtherFile != "" || facts.Conflict != "" ||
		facts.ServerFile != "" || facts.RootFile != "" {
		t.Errorf("the walled registration leaked into the serve facts: %+v", facts)
	}
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeWasmApp, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("the create plan refused behind the module wall: %q", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "server/main.go" {
		t.Fatalf("plan files = %+v, want the created server in the root module", plan.Files)
	}
	applyServePlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil || fresh == nil {
		t.Fatalf("edited project did not analyze: %v", err)
	}
	if f := fresh.ServeWasmRouteFacts("jim"); f.ServedFile != "server/main.go" {
		t.Errorf("facts after landing = %+v; want the created server serving from server/", f)
	}
}

// The same registration in the project's own module, in a
// folder hosting its own program: the facts call it
// OtherFile (named, never run), and the create plan
// proceeds to give the app its own server anyway.
func TestServeWasmDeepProgramNamed(t *testing.T) {
	_, proj := serveWasmProject(t, map[string]string{
		"tools/deep/main.go": deepServerForJim,
	})
	facts := proj.ServeWasmRouteFacts("jim")
	if facts.OtherFile != "tools/deep/main.go" || facts.OtherLine == 0 {
		t.Errorf("facts = %+v; want the deep program named as OtherFile with its line", facts)
	}
	if facts.ServedFile != "" || facts.Conflict != "" {
		t.Errorf("facts = %+v; want no ServedFile and no Conflict", facts)
	}
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeWasmApp, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("the create plan refused over a deep program's registration: %q", refusal)
	}
}

// The two homes, and the library registration that rides
// in the project's own programs.
func TestServeWasmRouteFactsHomes(t *testing.T) {
	// server/ serves: home.
	_, proj := serveWasmProject(t, map[string]string{
		"server/main.go": deepServerForJim,
	})
	if f := proj.ServeWasmRouteFacts("jim"); f.ServedFile != "server/main.go" {
		t.Errorf("server/ facts = %+v", f)
	}
	// A root-level program serves: home.
	_, proj = serveWasmProject(t, map[string]string{
		"main.go": deepServerForJim,
	})
	if f := proj.ServeWasmRouteFacts("jim"); f.ServedFile != "main.go" {
		t.Errorf("root facts = %+v", f)
	}
	// A server waiting in server/ without the route: the
	// extend candidate.
	_, proj = serveWasmProject(t, map[string]string{
		"server/main.go": "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.ListenAndServe(\":8080\", nil)\n}\n",
	})
	if f := proj.ServeWasmRouteFacts("jim"); f.ServerFile != "server/main.go" || f.ServedFile != "" {
		t.Errorf("extend facts = %+v", f)
	}
	// A library registration (no server in its folder):
	// the project's own route, the shipped answer.
	_, proj = serveWasmProject(t, map[string]string{
		"lib/routes.go": "package lib\n\nimport \"net/http\"\n\nfunc routes() {\n\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"jim\"))))\n}\n",
	})
	if f := proj.ServeWasmRouteFacts("jim"); f.ServedFile != "lib/routes.go" || f.OtherFile != "" {
		t.Errorf("library facts = %+v", f)
	}
	// A visible conflict for a different folder stands.
	_, proj = serveWasmProject(t, map[string]string{
		"tools/deep/main.go": strings.Replace(deepServerForJim, `Dir("jim")`, `Dir("other")`, 1),
	})
	if f := proj.ServeWasmRouteFacts("jim"); !strings.Contains(f.Conflict, "different folder") {
		t.Errorf("conflict facts = %+v", f)
	}
	// The same conflict behind a module wall: invisible.
	_, proj = serveWasmProject(t, map[string]string{
		"paces/project/go.mod":      "module example.com/nested\n\ngo 1.21\n",
		"paces/project/api/main.go": strings.Replace(nestedServerForJim, `Dir("jim")`, `Dir("other")`, 1),
	})
	if f := proj.ServeWasmRouteFacts("jim"); f.Conflict != "" {
		t.Errorf("walled conflict leaked: %+v", f)
	}
}

// The serve-static planner, untargeted, keeps every
// project-wide answer — and the composition's targeted
// request lands the line in the named server and nowhere
// else, even while another program serves the folder.
func TestPlanServeStaticTargetedSite(t *testing.T) {
	files := map[string]string{
		"server/main.go":     "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.ListenAndServe(\":8080\", nil)\n}\n",
		"tools/deep/main.go": deepServerForJim,
	}
	// Untargeted: the route already serves jim (from the
	// deep program) — the family's nothing-to-add, pinned.
	_, proj := serveWasmProject(t, files)
	if _, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim"}); !strings.Contains(refusal, "already serves the jim folder") {
		t.Errorf("untargeted refusal = %q; want the shipped nothing-to-add", refusal)
	}
	// Targeted at server/: the plan assembles, in that
	// file only.
	dir, proj := serveWasmProject(t, files)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim", StaticSiteFile: "server/main.go"})
	if refusal != "" || plan == nil {
		t.Fatalf("targeted plan refused: %q", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "server/main.go" {
		t.Fatalf("targeted plan files = %+v, want server/main.go only", plan.Files)
	}
	if !strings.Contains(string(plan.Files[0].After), serveWasmRegForJim) {
		t.Errorf("the registration did not land in the targeted server:\n%s", plan.Files[0].After)
	}
	applyServePlan(t, dir, plan)
	// Targeted at a file holding no server: a plain
	// refusal, nothing assembled.
	_, proj = serveWasmProject(t, files)
	if _, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim", StaticSiteFile: "doc.go"}); !strings.Contains(refusal, "no webserver in") {
		t.Errorf("targetless-file refusal = %q", refusal)
	}
	// A different folder's conflict refuses even a
	// targeted request: rule 4 stands.
	_, proj = serveWasmProject(t, map[string]string{
		"server/main.go":     "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.ListenAndServe(\":8080\", nil)\n}\n",
		"tools/deep/main.go": "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"other\"))))\n\thttp.ListenAndServe(\":9091\", nil)\n}\n",
	})
	if _, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim", StaticSiteFile: "server/main.go"}); !strings.Contains(refusal, "different folder") {
		t.Errorf("targeted conflict refusal = %q", refusal)
	}
}

// Both readings of John's project, side by side: the
// project-wide probe still sees the nested module's
// registration (the serve-static family's semantics are
// unchanged), while the serve sentence's probe does not.
func TestStaticRouteFactsProjectWideUnchanged(t *testing.T) {
	_, proj := serveWasmProject(t, map[string]string{
		"paces/project/go.mod":      "module example.com/nested\n\ngo 1.21\n",
		"paces/project/api/main.go": nestedServerForJim,
	})
	if f := proj.StaticRouteFacts("jim"); !f.Served || f.File != "paces/project/api/main.go" || !f.HasServer {
		t.Errorf("project-wide facts changed: %+v", f)
	}
	if f := proj.ServeWasmRouteFacts("jim"); f.ServedFile != "" || f.OtherFile != "" {
		t.Errorf("serve facts saw through the wall: %+v", f)
	}
}
