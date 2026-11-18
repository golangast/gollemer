package analyze

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The full-stack planner (editfullstack.go): one named
// thing grown through every layer — the rqlite table and
// the /api/ handlers beside the project's server, and
// the form and list in the wasm app — in one plan. These
// tests plan against a fixture that carries a real wasm
// app, grown by the app's own planner so the stack lands
// on the exact files a session would have.

// stackFixture builds a project holding a wasm app in
// jim, created through EditWasmApp's own plan and applied
// to disk the way the spine applies it.
func stackFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module example.com/fixture\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "jim"), 0o755); err != nil {
		t.Fatal(err)
	}
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	app, refusal := proj.PlanEdit(EditRequest{Kind: EditWasmApp, WasmFolder: "jim"})
	if refusal != "" {
		t.Fatalf("the app plan refused: %s", refusal)
	}
	applyPlan(t, dir, app)
	return dir
}

func stackNotesRequest() EditRequest {
	return EditRequest{
		Kind:       EditFullStack,
		WasmFolder: "jim",
		StackThing: "notes",
		StackFields: []StackField{
			{Name: "title", Kind: StackFieldText},
			{Name: "pages", Kind: StackFieldNumber},
			{Name: "done", Kind: StackFieldYesNo},
		},
	}
}

func stackFileAfter(plan *EditPlan, rel string) (string, bool) {
	for _, f := range plan.Files {
		if f.Rel == rel {
			return string(f.After), true
		}
	}
	return "", false
}

// The create case: no server yet — the plan creates the
// house server in server/, the backend beside it, and the
// frontend in the app, and mounts the section in the
// app's components flow. The summary names the route, the
// table, and every file; the plan verifies against the
// applied project.
func TestPlanFullStackCreatesTheServer(t *testing.T) {
	dir := stackFixture(t)
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan, refusal := proj.PlanEdit(stackNotesRequest())
	if refusal != "" {
		t.Fatalf("the stack plan refused: %s", refusal)
	}
	for _, want := range []string{"/api/notes", "the notes table", "server/notes.go", "jim/notes.go", "server/main.go", "jim/main.go", "jim/index.html"} {
		if !strings.Contains(plan.Summary, want) {
			t.Errorf("summary is missing %q:\n%s", want, plan.Summary)
		}
	}
	notes := strings.Join(plan.Notes, "\n")
	for _, want := range []string{"rqlite", "rqlited -node-id=1 data/", "no accounts"} {
		if !strings.Contains(notes, want) {
			t.Errorf("notes are missing %q:\n%s", want, notes)
		}
	}
	backend, ok := stackFileAfter(plan, "server/notes.go")
	if !ok {
		t.Fatal("the plan has no backend file")
	}
	for _, want := range []string{
		"func openNotesStore()", "CREATE TABLE IF NOT EXISTS notes",
		"id INTEGER PRIMARY KEY", "title TEXT", "pages REAL", "done INTEGER",
		`http.HandleFunc("/api/notes", store.notesCollection)`,
		`http.HandleFunc("/api/notes/", store.notesRecord)`,
		"http.StatusCreated", "http.StatusNoContent",
	} {
		if !strings.Contains(backend, want) {
			t.Errorf("backend is missing %q", want)
		}
	}
	server, ok := stackFileAfter(plan, "server/main.go")
	if !ok {
		t.Fatal("the plan has no server file")
	}
	if strings.Count(server, "setupNotes()") != 1 {
		t.Errorf("the created server should carry exactly one setupNotes() call:\n%s", server)
	}
	if !strings.Contains(server, `http.Dir("jim")`) {
		t.Errorf("the created server should serve the app from jim:\n%s", server)
	}
	frontend, ok := stackFileAfter(plan, "jim/notes.go")
	if !ok {
		t.Fatal("the plan has no frontend file")
	}
	for _, want := range []string{"func notesSection(", "notesSave", "notesRefresh", `"/api/notes"`, "textContent"} {
		if !strings.Contains(frontend, want) {
			t.Errorf("frontend is missing %q", want)
		}
	}
	if strings.Contains(frontend, "innerHTML") {
		t.Error("the frontend must never build rows out of innerHTML strings")
	}
	appMain, ok := stackFileAfter(plan, "jim/main.go")
	if !ok || !strings.Contains(appMain, "app.Call(\"appendChild\", notesSection(doc))") {
		t.Errorf("the app main.go should mount the section:\n%s", appMain)
	}
	html, ok := stackFileAfter(plan, "jim/index.html")
	if !ok || !strings.Contains(html, "/* full-stack sections") {
		t.Errorf("the host page should carry the stack's styles once:\n%s", html)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("the plan does not verify against the applied project")
	}
}

// The extend case: the house server already serves the
// app — the plan edits it in place (one setup call added,
// no second serve line, no second static route) and
// creates only the thing's own files.
func TestPlanFullStackExtendsTheHouseServer(t *testing.T) {
	dir := stackFixture(t)
	if err := os.MkdirAll(filepath.Join(dir, "server"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "server", "main.go"), []byte(serveWasmServerSource("jim")), 0o644); err != nil {
		t.Fatal(err)
	}
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	req := stackNotesRequest()
	req.StackServerFile = "server/main.go"
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" {
		t.Fatalf("the stack plan refused: %s", refusal)
	}
	server, ok := stackFileAfter(plan, "server/main.go")
	if !ok {
		t.Fatal("the plan does not edit the house server")
	}
	if strings.Count(server, "http.ListenAndServe") != 1 {
		t.Errorf("the extended server grew a second serve line:\n%s", server)
	}
	if strings.Count(server, `"/static/"`) != 2 { // the one registration line, which names the route twice by construction
		t.Errorf("the extended server grew a second static route:\n%s", server)
	}
	if strings.Count(server, "setupNotes()") != 1 {
		t.Errorf("the extended server should carry exactly one setupNotes() call:\n%s", server)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Verify(fresh) {
		t.Error("the plan does not verify against the applied project")
	}
}

// Asked twice, the same thing is already there — the
// refusal says so and nothing is planned twice. A
// different thing accumulates beside the first.
func TestPlanFullStackAlreadyThereAndSecondThing(t *testing.T) {
	dir := stackFixture(t)
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	plan, refusal := proj.PlanEdit(stackNotesRequest())
	if refusal != "" {
		t.Fatalf("the stack plan refused: %s", refusal)
	}
	applyPlan(t, dir, plan)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	again, refusal := fresh.PlanEdit(stackNotesRequest())
	if refusal == "" {
		t.Fatalf("the repeat ask planned a second copy: %+v", again)
	}
	if !strings.Contains(refusal, "the notes stack is already there") {
		t.Errorf("repeat refusal wrong: %s", refusal)
	}
	books := EditRequest{
		Kind:            EditFullStack,
		WasmFolder:      "jim",
		StackServerFile: "server/main.go",
		StackThing:      "books",
		StackFields: []StackField{
			{Name: "name", Kind: StackFieldText},
		},
	}
	plan2, refusal := fresh.PlanEdit(books)
	if refusal != "" {
		t.Fatalf("the second thing refused: %s", refusal)
	}
	if _, ok := stackFileAfter(plan2, "server/books.go"); !ok {
		t.Error("the second thing has no backend file of its own")
	}
	if _, ok := stackFileAfter(plan2, "jim/books.go"); !ok {
		t.Error("the second thing has no frontend file of its own")
	}
	server, _ := stackFileAfter(plan2, "server/main.go")
	if !strings.Contains(server, "setupNotes()") || !strings.Contains(server, "setupBooks()") {
		t.Errorf("the server should carry both setups after the second thing:\n%s", server)
	}
	applyPlan(t, dir, plan2)
	fresh2, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !plan2.Verify(fresh2) {
		t.Error("the second thing's plan does not verify")
	}
}

// The folder without a wasm app meets the family's
// two-step, and a missing folder is named plainly.
func TestPlanFullStackNeedsTheAppFirst(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module example.com/fixture\n\ngo 1.21\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dir, "jim"), 0o755); err != nil {
		t.Fatal(err)
	}
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, refusal := proj.PlanEdit(stackNotesRequest())
	if refusal == "" || !strings.Contains(refusal, "add a wasm app in jim") {
		t.Fatalf("app-less folder refusal wrong: %s", refusal)
	}
	ghost := stackNotesRequest()
	ghost.WasmFolder = "ghost"
	_, refusal = proj.PlanEdit(ghost)
	if refusal == "" || !strings.Contains(refusal, "no folder ghost") {
		t.Fatalf("missing folder refusal wrong: %s", refusal)
	}
}

// The field rules refuse before anything is planned: a
// field named id, a repeat, and two fields that would
// share one Go name.
func TestPlanFullStackBadFields(t *testing.T) {
	dir := stackFixture(t)
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name   string
		fields []StackField
		want   string
	}{
		{"id field", []StackField{{Name: "id", Kind: StackFieldText}}, "grows its own id"},
		{"duplicate", []StackField{{Name: "title", Kind: StackFieldText}, {Name: "title", Kind: StackFieldNumber}}, "named twice"},
		{"go clash", []StackField{{Name: "title", Kind: StackFieldText}, {Name: "Title", Kind: StackFieldText}}, "both become the Go name"},
		{"no fields", nil, "at least one field"},
	} {
		req := stackNotesRequest()
		req.StackFields = tc.fields
		_, refusal := proj.PlanEdit(req)
		if refusal == "" || !strings.Contains(refusal, tc.want) {
			t.Errorf("%s: refusal = %v, want it to name %q", tc.name, refusal, tc.want)
		}
	}
}

// A server that already serves /static/ from another
// folder refuses in the serve-static family's own words:
// the route has one home, and the stack doesn't take it.
func TestPlanFullStackStaticConflict(t *testing.T) {
	dir := stackFixture(t)
	if err := os.MkdirAll(filepath.Join(dir, "server"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "server", "main.go"), []byte(serveWasmServerSource("other")), 0o644); err != nil {
		t.Fatal(err)
	}
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	req := stackNotesRequest()
	req.StackServerFile = "server/main.go"
	_, refusal := proj.PlanEdit(req)
	if refusal == "" || !strings.Contains(refusal, "already serves a different folder") || !strings.Contains(refusal, `"other"`) {
		t.Fatalf("static conflict refusal wrong: %s", refusal)
	}
}

// A component already carrying the thing's name refuses:
// the form's section and the component would share one
// id in the page, and the section builder would collide.
func TestPlanFullStackComponentNameTaken(t *testing.T) {
	dir := stackFixture(t)
	proj, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	comp, refusal := proj.PlanEdit(EditRequest{Kind: EditComponent, WasmFolder: "jim", Component: "card", ComponentName: "notes"})
	if refusal != "" {
		t.Fatalf("the component plan refused: %s", refusal)
	}
	applyPlan(t, dir, comp)
	fresh, err := Analyze(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, refusal = fresh.PlanEdit(stackNotesRequest())
	if refusal == "" || !strings.Contains(refusal, "component whose id is notes") {
		t.Fatalf("component-name refusal wrong: %s", refusal)
	}
}
