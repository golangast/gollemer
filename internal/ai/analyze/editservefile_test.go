package analyze

import (
	"strings"
	"testing"
)

// The serve-a-file planner's states, fixture by fixture: the
// composed landing onto a default-mux server, the uncomposed
// landing with its finishing sentence, the route conflict
// naming both, the already-serving answers (named and
// inline), the name collision, the missing file, a file
// outside the host folder, and an asked route. Fixture
// helpers are the serve/page tests' (serveProject,
// applyServePlan, planFileText).

const serveFileServerMain = `package main

import "net/http"

func main() {
	http.ListenAndServe(":8080", nil)
}
`

func serveFileReq(file, serve, name, route string) EditRequest {
	return EditRequest{Kind: EditServeFile, File: file, ServeFile: serve, ServeFileName: name, ServeFileRoute: route}
}

func TestPlanServeFileComposesOntoServer(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":       pageGoMod,
		"jim/main.go":  serveFileServerMain,
		"jim/logo.png": "PNG",
	})
	plan, refusal := proj.PlanEdit(serveFileReq("jim/main.go", "jim/logo.png", "LogoHandler", ""))
	if refusal != "" {
		t.Fatalf("composed ask refused: %s", refusal)
	}
	text := planFileText(t, plan, "jim/main.go")
	if !strings.Contains(text, `http.HandleFunc("/logo.png", LogoHandler)`) {
		t.Errorf("registration missing or misplaced:\n%s", text)
	}
	if !strings.Contains(text, `http.ServeFile(w, r, "jim/logo.png")`) {
		t.Errorf("handler body missing:\n%s", text)
	}
	if strings.Index(text, "HandleFunc") > strings.Index(text, "ListenAndServe") {
		t.Errorf("registration must land before the serve statement:\n%s", text)
	}
	if !strings.Contains(plan.Summary, "logo.png") || !strings.Contains(plan.Summary, "isn't changed") {
		t.Errorf("summary = %q", plan.Summary)
	}
	joined := strings.Join(plan.Notes, "\n")
	if !strings.Contains(joined, "content type") || !strings.Contains(joined, "untouched apart from the one registration") {
		t.Errorf("notes = %q", joined)
	}
	applyServePlan(t, dir, plan)

	// The same ask again is the already-serving answer.
	_, refusal = proj.PlanEdit(serveFileReq("jim/main.go", "jim/logo.png", "LogoHandler", ""))
	if !strings.Contains(refusal, "already serves") || !strings.Contains(refusal, "Nothing to add") {
		t.Errorf("second ask refusal = %q", refusal)
	}
}

func TestPlanServeFileUncomposedNamesFinishingSentence(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":       pageGoMod,
		"jim/util.go":  "package jim\n\nfunc Helper() {}\n",
		"jim/logo.png": "PNG",
	})
	plan, refusal := proj.PlanEdit(serveFileReq("jim/handlers.go", "jim/logo.png", "LogoHandler", ""))
	if refusal != "" {
		t.Fatalf("uncomposed ask refused: %s", refusal)
	}
	text := planFileText(t, plan, "jim/handlers.go")
	if !strings.Contains(text, "package jim") || !strings.Contains(text, "func LogoHandler(") {
		t.Errorf("created host wrong:\n%s", text)
	}
	joined := strings.Join(plan.Notes, "\n")
	if !strings.Contains(joined, `"connect the LogoHandler handler to the webserver at /logo.png"`) {
		t.Errorf("finishing sentence missing from notes: %q", joined)
	}
	if !strings.Contains(joined, "no webserver yet") {
		t.Errorf("no-server clause missing from notes: %q", joined)
	}
}

func TestPlanServeFileRouteConflictNamesBoth(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"jim/main.go": `package main

import "net/http"

func Other(w http.ResponseWriter, r *http.Request) {}

func main() {
	http.HandleFunc("/logo.png", Other)
	http.ListenAndServe(":8080", nil)
}
`,
		"jim/logo.png": "PNG",
	})
	_, refusal := proj.PlanEdit(serveFileReq("jim/main.go", "jim/logo.png", "LogoHandler", ""))
	if !strings.Contains(refusal, "already taken") || !strings.Contains(refusal, "Other") {
		t.Errorf("conflict refusal = %q", refusal)
	}
}

func TestPlanServeFileNameCollision(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"jim/main.go": `package main

import "net/http"

func LogoHandler(w http.ResponseWriter, r *http.Request) {}

func main() {
	http.ListenAndServe(":8080", nil)
}
`,
		"jim/logo.png": "PNG",
	})
	_, refusal := proj.PlanEdit(serveFileReq("jim/main.go", "jim/logo.png", "LogoHandler", ""))
	if !strings.Contains(refusal, "already has a declaration named LogoHandler") || !strings.Contains(refusal, "named <Name>") {
		t.Errorf("collision refusal = %q", refusal)
	}
}

func TestPlanServeFileMissingFileRefused(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":      pageGoMod,
		"jim/main.go": serveFileServerMain,
	})
	_, refusal := proj.PlanEdit(serveFileReq("jim/main.go", "jim/logo.png", "LogoHandler", ""))
	if !strings.Contains(refusal, "no file") || !strings.Contains(refusal, "really there") {
		t.Errorf("missing-file refusal = %q", refusal)
	}
}

func TestPlanServeFileInlineAlreadyServing(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"jim/main.go": `package main

import "net/http"

func main() {
	http.HandleFunc("/logo.png", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "jim/logo.png")
	})
	http.ListenAndServe(":8080", nil)
}
`,
		"jim/logo.png": "PNG",
	})
	_, refusal := proj.PlanEdit(serveFileReq("jim/main.go", "jim/logo.png", "LogoHandler", ""))
	if !strings.Contains(refusal, "already serves") || !strings.Contains(refusal, "nothing to add") {
		t.Errorf("inline-serving refusal = %q", refusal)
	}
}

func TestPlanServeFileCrossFolderAndAskedRoute(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":          pageGoMod,
		"jim/main.go":     serveFileServerMain,
		"assets/logo.png": "PNG",
	})
	plan, refusal := proj.PlanEdit(serveFileReq("jim/main.go", "assets/logo.png", "LogoHandler", ""))
	if refusal != "" {
		t.Fatalf("cross-folder ask refused: %s", refusal)
	}
	text := planFileText(t, plan, "jim/main.go")
	if !strings.Contains(text, `http.HandleFunc("/assets/logo.png", LogoHandler)`) {
		t.Errorf("cross-folder route wrong:\n%s", text)
	}
	if !strings.Contains(text, `http.ServeFile(w, r, "assets/logo.png")`) {
		t.Errorf("cross-folder serve path wrong:\n%s", text)
	}

	plan, refusal = proj.PlanEdit(serveFileReq("jim/main.go", "assets/logo.png", "LogoHandler", "/img/logo"))
	if refusal != "" {
		t.Fatalf("asked-route ask refused: %s", refusal)
	}
	text = planFileText(t, plan, "jim/main.go")
	if !strings.Contains(text, `http.HandleFunc("/img/logo", LogoHandler)`) {
		t.Errorf("asked route not used:\n%s", text)
	}
}
