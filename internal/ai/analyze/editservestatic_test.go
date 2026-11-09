package analyze

import (
	"strings"
	"testing"
)

// The serve-static planner: one file-server registration at
// /static/ on the project's existing server, under the
// wiring machinery's rules. The fixture server is the
// live-session shape: a jim program with two HandleFunc
// registrations and a bare ListenAndServe.

const staticJimMain = `package main

import "net/http"

func Jake(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte("jake"))
}

func main() {
	http.HandleFunc("/jake", Jake)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("root"))
	})
	http.ListenAndServe(":8080", nil)
}
`

const staticRegForJimTemp = `http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim/temp"))))`

func staticProject(t *testing.T, files map[string]string) (string, *Project) {
	t.Helper()
	base := map[string]string{
		"go.mod":             "module example.com/static\n\ngo 1.21\n",
		"doc.go":             "package static\n",
		"jim/main.go":        staticJimMain,
		"jim/temp/style.css": "body {}\n",
	}
	for rel, src := range files {
		base[rel] = src
	}
	return serveProject(t, base)
}

func TestPlanServeStaticRegisters(t *testing.T) {
	dir, proj := staticProject(t, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim/temp"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan refused: %q", refusal)
	}
	if len(plan.Files) != 1 || plan.Files[0].Rel != "jim/main.go" {
		t.Fatalf("plan files = %+v", plan.Files)
	}
	after := string(plan.Files[0].After)
	regAt := strings.Index(after, staticRegForJimTemp)
	serveAt := strings.Index(after, "http.ListenAndServe")
	if regAt < 0 {
		t.Fatalf("registration line missing:\n%s", after)
	}
	if serveAt < 0 || regAt > serveAt {
		t.Errorf("registration is not before the serve statement:\n%s", after)
	}
	if !strings.Contains(plan.Summary, "content type from its extension") {
		t.Errorf("summary does not state the content-type mechanism: %q", plan.Summary)
	}
	applyServePlan(t, dir, plan)
}

func TestPlanServeStaticRootProgram(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":        "module example.com/rootstatic\n\ngo 1.21\n",
		"main.go":       "package main\n\nimport \"net/http\"\n\nfunc main() {\n\thttp.ListenAndServe(\":8080\", nil)\n}\n",
		"assets/app.js": "console.log(1)\n",
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "assets"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan refused: %q", refusal)
	}
	if got := string(plan.Files[0].After); !strings.Contains(got, `http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("assets"))))`) {
		t.Errorf("root program registration missing:\n%s", got)
	}
}

func TestPlanServeStaticMuxSite(t *testing.T) {
	_, proj := staticProject(t, map[string]string{
		"jim/main.go": `package main

import "net/http"

func Jake(w http.ResponseWriter, r *http.Request) {}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/jake", Jake)
	http.ListenAndServe(":8080", mux)
}
`,
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim/temp"})
	if refusal != "" || plan == nil {
		t.Fatalf("plan refused: %q", refusal)
	}
	want := `mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim/temp"))))`
	if got := string(plan.Files[0].After); !strings.Contains(got, want) {
		t.Errorf("mux registration = %q missing:\n%s", want, got)
	}
}

func TestPlanServeStaticNoServer(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":         "module example.com/noserver\n\ngo 1.21\n",
		"main.go":        "package main\n\nfunc main() {}\n",
		"jim/temp/a.css": "body {}\n",
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim/temp"})
	if plan != nil || !strings.Contains(refusal, "no webserver in this project yet") || !strings.Contains(refusal, "add a webserver to main.go") {
		t.Errorf("plan=%v refusal=%q", plan != nil, refusal)
	}
}

func TestPlanServeStaticMissingFolder(t *testing.T) {
	_, proj := staticProject(t, nil)
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "nope"})
	if plan != nil || !strings.Contains(refusal, "create a folder nope") {
		t.Errorf("plan=%v refusal=%q", plan != nil, refusal)
	}
}

func TestPlanServeStaticRouteTakenByHandler(t *testing.T) {
	_, proj := staticProject(t, map[string]string{
		"jim/main.go": `package main

import "net/http"

func Other(w http.ResponseWriter, r *http.Request) {}

func main() {
	http.HandleFunc("/static/", Other)
	http.ListenAndServe(":8080", nil)
}
`,
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim/temp"})
	if plan != nil {
		t.Fatalf("a taken route was planned over: %+v", plan)
	}
	if !strings.Contains(refusal, "/static/ is already taken") || !strings.Contains(refusal, "Other") {
		t.Errorf("refusal does not name both sides: %q", refusal)
	}
}

func TestPlanServeStaticAlreadyServingSameFolder(t *testing.T) {
	_, proj := staticProject(t, map[string]string{
		"jim/main.go": `package main

import "net/http"

func main() {
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("jim/temp"))))
	http.ListenAndServe(":8080", nil)
}
`,
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim/temp"})
	if plan != nil || !strings.Contains(refusal, "already serves the jim/temp folder at /static/") || !strings.Contains(refusal, "nothing to add") {
		t.Errorf("plan=%v refusal=%q", plan != nil, refusal)
	}
}

func TestPlanServeStaticServingOtherFolder(t *testing.T) {
	_, proj := staticProject(t, map[string]string{
		"jim/main.go": `package main

import "net/http"

func main() {
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	http.ListenAndServe(":8080", nil)
}
`,
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim/temp"})
	if plan != nil || !strings.Contains(refusal, "already serves a different folder") || !strings.Contains(refusal, `"static"`) {
		t.Errorf("plan=%v refusal=%q", plan != nil, refusal)
	}
}

func TestPlanServeStaticCustomServer(t *testing.T) {
	_, proj := staticProject(t, map[string]string{
		"jim/main.go": `package main

import "net/http"

type app struct{}

func (a app) ServeHTTP(w http.ResponseWriter, r *http.Request) {}

func main() {
	http.ListenAndServe(":8080", app{})
}
`,
	})
	plan, refusal := proj.PlanEdit(EditRequest{Kind: EditServeStatic, StaticFolder: "jim/temp"})
	if plan != nil {
		t.Fatalf("a custom server was planned into: %+v", plan)
	}
	if !strings.Contains(refusal, "in its own way") || !strings.Contains(refusal, staticRegForJimTemp) {
		t.Errorf("custom refusal does not hand the line over: %q", refusal)
	}
}
