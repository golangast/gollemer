package analyze

import (
	"strings"
	"testing"
)

// The serve-wasm-app planner: the missing-server half of the
// chat's "serve the wasm app" composition — the house server
// created in server/main.go already carrying the /static/
// registration over the app folder. The conflicts and the
// route facts are the serve-static piece's, shared through
// staticRouteState / StaticRouteFacts.

const serveWasmRegForJim = `http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim"))))`

func serveWasmProject(t *testing.T, files map[string]string) (string, *Project) {
	t.Helper()
	base := map[string]string{
		"go.mod":         "module example.com/servewasm\n\ngo 1.21\n",
		"doc.go":         "package servewasm\n",
		"jim/index.html": "<!doctype html>\n",
	}
	for rel, src := range files {
		base[rel] = src
	}
	return serveProject(t, base)
}

func TestPlanServeWasmAppCreatesServer(t *testing.T) {
	dir, proj := serveWasmProject(t, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeWasmApp, WasmFolder: "jim"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan refused: %q", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "server/main.go" {
		t.Fatalf("plan files = %+v", plan.Files)
	}
	if len(plan.Files[0].Before) != 0 {
		t.Errorf("the server file is a creation; Before should be empty")
	}
	after := string(plan.Files[0].After)
	regAt := strings.Index(after, serveWasmRegForJim)
	serveAt := strings.Index(after, "http.ListenAndServe")
	if regAt < 0 {
		t.Fatalf("registration line missing:\\n%s", after)
	}
	if serveAt < 0 || regAt > serveAt {
		t.Errorf("registration is not before the serve statement:\\n%s", after)
	}
	if !strings.Contains(plan.Summary, "http://localhost:8080/static/") {
		t.Errorf("summary does not name the app's address: %q", plan.Summary)
	}
	applyServePlan(t, dir, plan)

	fresh, err := Analyze(dir)
	if err != nil || fresh == nil {
		t.Fatalf("edited project did not analyze: %v", err)
	}
	facts := fresh.StaticRouteFacts("jim")
	if !facts.HasServer || !facts.Served || facts.File != "server/main.go" || facts.Conflict != "" {
		t.Errorf("facts after landing = %+v; want the created server holding the route", facts)
	}
}

func TestPlanServeWasmAppRefusals(t *testing.T) {
	serverMain := "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.ListenAndServe(\":8080\", nil)\n}\n"
	regOther := "package lib\n\nimport \"net/http\"\n\nfunc routes() {\n\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"other\"))))\n}\n"
	regJim := "package lib\n\nimport \"net/http\"\n\nfunc routes() {\n\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"jim\"))))\n}\n"

	cases := []struct {
		name    string
		files   map[string]string
		folder  string
		wantSub string
	}{
		{"folder missing", nil, "nope", "there's no folder nope in this project"},
		{"server exists", map[string]string{"web/main.go": serverMain}, "jim", "already has a webserver"},
		{"route on another folder", map[string]string{"lib/routes.go": regOther}, "jim", "/static/ already serves a different folder"},
		{"route already serves the app", map[string]string{"lib/routes.go": regJim}, "jim", "already serves the jim folder at /static/"},
		{"dead server file is never overwritten", map[string]string{"server/main.go": "package main\n\nfunc main() {}\n"}, "jim", "won't overwrite"},
		{"server path taken by a file", map[string]string{"server": "not a folder\n"}, "jim", "is a file, not a folder"},
	}
	for _, tc := range cases {
		_, proj := serveWasmProject(t, tc.files)
		plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeWasmApp, WasmFolder: tc.folder})
		if plan != nil {
			t.Errorf("%s: got a plan, want a refusal", tc.name)
		}
		if !strings.Contains(refusal, tc.wantSub) {
			t.Errorf("%s: refusal %q does not contain %q", tc.name, refusal, tc.wantSub)
		}
	}
}

// The exported probe the chat flow reads: four states the
// composition branches on, all from the one scan.
func TestStaticRouteFactsStates(t *testing.T) {
	// No server, no route: free.
	_, proj := serveWasmProject(t, nil)
	if f := proj.StaticRouteFacts("jim"); f.HasServer || f.Served || f.Conflict != "" {
		t.Errorf("empty project facts = %+v", f)
	}

	// A server, no route: the serve-static edit's case.
	_, proj = serveWasmProject(t, map[string]string{
		"web/main.go": "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.ListenAndServe(\":8080\", nil)\n}\n",
	})
	if f := proj.StaticRouteFacts("jim"); !f.HasServer || f.Served || f.Conflict != "" {
		t.Errorf("server-only facts = %+v", f)
	}

	// A route held by another folder, no server anywhere:
	// still a conflict — the composite flow must not create a
	// server under a taken route.
	_, proj = serveWasmProject(t, map[string]string{
		"lib/routes.go": "package lib\n\nimport \"net/http\"\n\nfunc routes() {\n\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"other\"))))\n}\n",
	})
	f := proj.StaticRouteFacts("jim")
	if f.HasServer || f.Served || !strings.Contains(f.Conflict, "different folder") {
		t.Errorf("conflict facts = %+v", f)
	}

	// The route serving this folder names its file.
	_, proj = serveWasmProject(t, map[string]string{
		"web/main.go": "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.Handle(\"/static/\", http.StripPrefix(\"/static/\", http.FileServer(http.Dir(\"jim\"))))\n\thttp.ListenAndServe(\":8080\", nil)\n}\n",
	})
	if f := proj.StaticRouteFacts("jim"); !f.HasServer || !f.Served || f.File != "web/main.go" || f.Conflict != "" {
		t.Errorf("served facts = %+v", f)
	}
}
