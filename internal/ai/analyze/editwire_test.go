package analyze

import (
	"strings"
	"testing"
)

// The wiring planner's states, fixture by fixture: the
// default-mux insert (position pinned), the mux variable,
// named and bare resolution, the route rules, and every
// refusal with its facts. applyServePlan is the serve tests'
// helper — it writes the plan's files and verifies against a
// fresh analysis, the spine's own sequence.

const wireGoMod = "module example.com/wire\n\ngo 1.21\n"

const wireJimMain = `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "thanks")
}

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "home")
	})
	fmt.Println("listening at http://localhost:8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`

func jimWireProject(t *testing.T) (string, *Project) {
	t.Helper()
	return serveProject(t, map[string]string{
		"go.mod":        wireGoMod,
		"jim/main.go":   wireJimMain,
		"other/keep.go": "package other\n\nfunc Keep() {}\n",
	})
}

func planWire(t *testing.T, proj *Project, name, route string) (*EditPlan, string) {
	t.Helper()
	return proj.PlanEdit(EditRequest{Kind: EditWireHandler, WireName: name, WireRoute: route})
}

func planFileText(t *testing.T, plan *EditPlan, rel string) string {
	t.Helper()
	for _, f := range plan.Files {
		if f.Rel == rel {
			return string(f.After)
		}
	}
	t.Fatalf("plan has no file %s", rel)
	return ""
}

// The headline: John's sentence (named form) against his
// fixture — the registration lands in main, BEFORE the
// ListenAndServe statement, and the plan verifies.
func TestPlanWireNamedDefaultMux(t *testing.T) {
	dir, proj := jimWireProject(t)
	plan, refusal := planWire(t, proj, "Jake", "")
	if refusal != "" {
		t.Fatalf("named wiring refused: %s", refusal)
	}
	after := planFileText(t, plan, "jim/main.go")
	reg := `http.HandleFunc("/jake", Jake)`
	ri, li := strings.Index(after, reg), strings.Index(after, "http.ListenAndServe")
	if ri < 0 {
		t.Fatalf("registration missing:\n%s", after)
	}
	if ri > li {
		t.Errorf("registration landed AFTER the serve call:\n%s", after)
	}
	if !strings.Contains(plan.Summary, reg) {
		t.Errorf("summary does not show the exact line: %q", plan.Summary)
	}
	applyServePlan(t, dir, plan)
}

// Case is forgiven and restored from the declaration: "jake"
// wires Jake, and the written line carries the declared name.
func TestPlanWireNamedLowercase(t *testing.T) {
	_, proj := jimWireProject(t)
	plan, refusal := planWire(t, proj, "jake", "")
	if refusal != "" {
		t.Fatalf("lowercase name refused: %s", refusal)
	}
	if after := planFileText(t, plan, "jim/main.go"); !strings.Contains(after, `http.HandleFunc("/jake", Jake)`) {
		t.Errorf("registration does not carry the declared name:\n%s", after)
	}
}

// The bare ask with exactly one unregistered handler resolves
// to that handler.
func TestPlanWireBareResolvesTheOneHandler(t *testing.T) {
	dir, proj := jimWireProject(t)
	plan, refusal := planWire(t, proj, "", "")
	if refusal != "" {
		t.Fatalf("bare wiring refused: %s", refusal)
	}
	if after := planFileText(t, plan, "jim/main.go"); !strings.Contains(after, `http.HandleFunc("/jake", Jake)`) {
		t.Errorf("bare ask wired the wrong thing:\n%s", after)
	}
	applyServePlan(t, dir, plan)
}

// The bare ask with two unregistered handlers names both and
// writes nothing.
func TestPlanWireBareSeveralHandlers(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }
func Pete(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "p") }

func main() {
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "", "")
	if plan != nil {
		t.Fatalf("bare ask with two handlers planned an edit")
	}
	for _, want := range []string{"More than one handler", `"connect the Jake handler to the webserver"`, `"connect the Pete handler to the webserver"`, "Nothing was written"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("guidance missing %q:\n%s", want, refusal)
		}
	}
}

// The mux variable: the registration joins the mux's own
// registrations, written against the variable.
func TestPlanWireMuxVariable(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func Home(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "home") }
func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", Home)
	fmt.Println("listening")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "")
	if refusal != "" {
		t.Fatalf("mux wiring refused: %s", refusal)
	}
	after := planFileText(t, plan, "main.go")
	reg := `mux.HandleFunc("/jake", Jake)`
	if !strings.Contains(after, reg) {
		t.Fatalf("mux registration missing:\n%s", after)
	}
	if strings.Index(after, reg) > strings.Index(after, "http.ListenAndServe") {
		t.Errorf("mux registration landed after the serve call:\n%s", after)
	}
	applyServePlan(t, dir, plan)
}

// http.Serve with a mux classifies the same way.
func TestPlanWireHTTPServeMux(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"main.go": `package main

import (
	"fmt"
	"net"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }

func main() {
	mux := http.NewServeMux()
	ln, err := net.Listen("tcp", ":8080")
	if err != nil {
		fmt.Println(err)
		return
	}
	_ = http.Serve(ln, mux)
}
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "")
	if refusal != "" {
		t.Fatalf("http.Serve wiring refused: %s", refusal)
	}
	if after := planFileText(t, plan, "main.go"); !strings.Contains(after, `mux.HandleFunc("/jake", Jake)`) {
		t.Errorf("http.Serve mux registration missing:\n%s", after)
	}
}

// The named route: "at /contact" is the path the handler
// answers at, not the derived one.
func TestPlanWireAtPath(t *testing.T) {
	_, proj := jimWireProject(t)
	plan, refusal := planWire(t, proj, "Jake", "/contact")
	if refusal != "" {
		t.Fatalf("at-path wiring refused: %s", refusal)
	}
	if after := planFileText(t, plan, "jim/main.go"); !strings.Contains(after, `http.HandleFunc("/contact", Jake)`) {
		t.Errorf("registration is not at the named path:\n%s", after)
	}
}

// A path without its slash is refused with the fix spelled.
func TestPlanWireRouteNeedsSlash(t *testing.T) {
	_, proj := jimWireProject(t)
	plan, refusal := planWire(t, proj, "Jake", "contact")
	if plan != nil || !strings.Contains(refusal, "needs to start with a slash") {
		t.Errorf("slash-less path: plan=%v refusal=%q", plan != nil, refusal)
	}
}

// The path is taken by another handler: the refusal names the
// existing registration — file, line, and what it points at.
func TestPlanWirePathTaken(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request)  { fmt.Fprintln(w, "j") }
func Login(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "l") }

func main() {
	http.HandleFunc("/jake", Login)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "")
	if plan != nil {
		t.Fatalf("taken path planned an edit")
	}
	for _, want := range []string{"/jake is already taken", "jim/main.go:12", "Login"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("path-taken refusal missing %q:\n%s", want, refusal)
		}
	}
}

// Already registered at the asked path: a plain answer, no
// plan.
func TestPlanWireAlreadyRegistered(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }

func main() {
	http.HandleFunc("/jake", Jake)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "")
	if plan != nil {
		t.Fatalf("already-registered handler planned an edit")
	}
	if !strings.Contains(refusal, "Jake is already registered at /jake") || !strings.Contains(refusal, "nothing to add") {
		t.Errorf("already-registered answer is not the plain one:\n%s", refusal)
	}
}

// Registered at a DIFFERENT path than the ask: refused with
// both paths named — never a silent second registration.
func TestPlanWireRegisteredElsewhere(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }

func main() {
	http.HandleFunc("/jake", Jake)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "/signup")
	if plan != nil {
		t.Fatalf("second-path registration planned an edit")
	}
	for _, want := range []string{"already registered at /jake", "/signup", "won't register it at /signup as well"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("different-path refusal missing %q:\n%s", want, refusal)
		}
	}
}

// No server anywhere: the refusal spells the working
// add-a-webserver sentence, aimed at the handler's own file.
func TestPlanWireNoServer(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }

func main() {
	fmt.Println("no server here")
}
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "")
	if plan != nil {
		t.Fatalf("server-less wiring planned an edit")
	}
	if !strings.Contains(refusal, `"add a webserver to jim/main.go"`) {
		t.Errorf("no-server refusal does not spell the working sentence:\n%s", refusal)
	}
}

// Cross-file, same package: the handler in a sibling file is
// registered by its bare name, and no import is added.
func TestPlanWireCrossFileSamePackage(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func main() {
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
		"jim/extra.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "")
	if refusal != "" {
		t.Fatalf("cross-file wiring refused: %s", refusal)
	}
	after := planFileText(t, plan, "jim/main.go")
	if !strings.Contains(after, `http.HandleFunc("/jake", Jake)`) {
		t.Errorf("cross-file registration missing:\n%s", after)
	}
	if strings.Count(after, "import") != 1 || strings.Contains(after, "example.com") {
		t.Errorf("same-package wiring added an import:\n%s", after)
	}
	applyServePlan(t, dir, plan)
}

// Cross-package, exported: the serving file gains the import
// and names the handler qualified.
func TestPlanWireCrossPackageExported(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func main() {
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
		"handlers/signup.go": `package handlers

import (
	"fmt"
	"net/http"
)

func Signup(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "s") }
`,
	})
	plan, refusal := planWire(t, proj, "Signup", "")
	if refusal != "" {
		t.Fatalf("cross-package wiring refused: %s", refusal)
	}
	after := planFileText(t, plan, "main.go")
	for _, want := range []string{`"example.com/wire/handlers"`, `http.HandleFunc("/signup", handlers.Signup)`} {
		if !strings.Contains(after, want) {
			t.Errorf("cross-package plan missing %q:\n%s", want, after)
		}
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "example.com/wire/handlers") {
		t.Errorf("plan notes do not mention the new import:\n%s", notes)
	}
	applyServePlan(t, dir, plan)
}

// Cross-package, unexported: refused with the reason and the
// fix (a capital letter), never a broken plan.
func TestPlanWireCrossPackageUnexported(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func main() {
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
		"handlers/signup.go": `package handlers

import (
	"fmt"
	"net/http"
)

func signup(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "s") }
`,
	})
	plan, refusal := planWire(t, proj, "signup", "")
	if plan != nil {
		t.Fatalf("unexported cross-package wiring planned an edit")
	}
	for _, want := range []string{"lowercase", "package handlers", "capital letter"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("unexported refusal missing %q:\n%s", want, refusal)
		}
	}
}

// A handler in another PROGRAM (package main in its own
// folder) can't be registered by this project's server.
func TestPlanWireOtherProgram(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func main() {
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
		"app/main.go": `package main

import (
	"fmt"
	"net/http"
)

func Login(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "l") }

func main() {
	fmt.Println("the app program")
}
`,
	})
	plan, refusal := planWire(t, proj, "Login", "")
	if plan != nil {
		t.Fatalf("other-program wiring planned an edit")
	}
	for _, want := range []string{"program of its own", "app/main.go"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("other-program refusal missing %q:\n%s", want, refusal)
		}
	}
}

// The named thing exists but isn't a handler: its real
// signature is quoted back.
func TestPlanWireNotAHandler(t *testing.T) {
	_, proj := jimWireProject(t)
	plan, refusal := planWire(t, proj, "main", "")
	if plan != nil {
		t.Fatalf("non-handler planned an edit")
	}
	for _, want := range []string{"isn't a handler", "func main()"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("not-a-handler refusal missing %q:\n%s", want, refusal)
		}
	}
}

// The named thing doesn't exist at all: the refusal spells
// the add-it-first sentence.
func TestPlanWireNoSuchHandler(t *testing.T) {
	_, proj := jimWireProject(t)
	plan, refusal := planWire(t, proj, "Frobnicate", "")
	if plan != nil {
		t.Fatalf("missing handler planned an edit")
	}
	for _, want := range []string{`handler named "Frobnicate"`, `"add a form handler named Frobnicate to jim/main.go"`} {
		if !strings.Contains(refusal, want) {
			t.Errorf("missing-handler refusal missing %q:\n%s", want, refusal)
		}
	}
}

// A custom server (its own handler value): the exact line is
// handed over, and no edit is planned.
func TestPlanWireCustomServer(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }

func main() {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "hi")
	})
	if err := http.ListenAndServe(":8080", handler); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "Jake", "")
	if plan != nil {
		t.Fatalf("custom-server wiring planned an edit")
	}
	for _, want := range []string{"its own handler (handler)", `http.HandleFunc("/jake", Jake)`} {
		if !strings.Contains(refusal, want) {
			t.Errorf("custom-server refusal missing %q:\n%s", want, refusal)
		}
	}
}

// The bare ask in a project with no handlers at all.
func TestPlanWireBareNoHandlers(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func main() {
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "", "")
	if plan != nil {
		t.Fatalf("handler-less bare ask planned an edit")
	}
	if !strings.Contains(refusal, "I don't see a handler in this project yet") {
		t.Errorf("bare/no-handlers refusal:\n%s", refusal)
	}
}

// The bare ask when the one handler is already registered:
// a plain nothing-to-add naming the answer it gives.
func TestPlanWireBareAllRegistered(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": wireGoMod,
		"jim/main.go": `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "j") }

func main() {
	http.HandleFunc("/jake", Jake)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planWire(t, proj, "", "")
	if plan != nil {
		t.Fatalf("all-registered bare ask planned an edit")
	}
	for _, want := range []string{"already registered", "Jake answers at /jake", "Nothing to add"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("all-registered answer missing %q:\n%s", want, refusal)
		}
	}
}
