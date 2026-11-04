package analyze

import (
	"regexp"
	"strings"
	"testing"
)

// The page family's states, fixture by fixture: every kind
// composed onto a default-mux server, the form kinds' field
// agreement between page and handler, the no-server and
// custom-server uncomposed landings, the homepage replacing
// the webserver component's starter, and every refusal.
// Fixture helpers (serveProject, applyServePlan, planFileText)
// are the serve/wire tests' — this file adds only its own.

// pageGoMod is the fixture module line.
const pageGoMod = "module example.com/page\n\ngo 1.21\n"

// pageServerMain is a default-mux server in main.go with no
// registrations of its own.
const pageServerMain = `package main

import (
	"fmt"
	"net/http"
)

func main() {
	fmt.Println("listening at http://localhost:8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`

// formHandlerEntry is the concept book's form-handler row,
// copied EXACTLY from the shipped data/concepts.json entry
// (email/subject/message). The chat-side tests exercise the
// real embedded entry; the analyze tests build from this
// literal so the template text is pinned here.
func formHandlerEntry() *ConceptEntry {
	return &ConceptEntry{
		ID:      "form-handler",
		Names:   []string{"form handler", "html form", "contact form", "form"},
		Kind:    ConceptDeclaration,
		Imports: []string{"net/http"},
		Template: "func {name}(w http.ResponseWriter, r *http.Request) {\n" +
			"\tif r.Method != http.MethodPost {\n" +
			"\t\t// TODO: show the form here — a browser asks for it with GET before it can send anything\n" +
			"\t\treturn\n" +
			"\t}\n" +
			"\tdetails := struct {\n" +
			"\t\tEmail string\n" +
			"\t\tSubject string\n" +
			"\t\tMessage string\n" +
			"\t}{\n" +
			"\t\tEmail: r.FormValue(\"email\"),\n" +
			"\t\tSubject: r.FormValue(\"subject\"),\n" +
			"\t\tMessage: r.FormValue(\"message\"),\n" +
			"\t}\n" +
			"\t// TODO: details holds the form's fields — rename them to your form's, and use them here\n" +
			"\t_ = details\n" +
			"}",
	}
}

func planPageReq(t *testing.T, proj *Project, req EditRequest) (*EditPlan, string) {
	t.Helper()
	req.Kind = EditPage
	return proj.PlanEdit(req)
}

// Routes are per-program: a registration in ANOTHER folder's
// package-main program neither blocks a page nor lends its
// starter to replace. (The handtest state that caught this:
// jim's webserver, landed by an earlier session, still
// answered "/" with the component starter when site's
// homepage was asked for — the ask refused, naming jim's
// registration, and the homepage never landed.)
// The no-server follow-up note names the exact sentence that
// registers the handler at the page's route WHEN FED BACK. For
// most kinds that is the wiring grammar's bare sentence (its
// default route is "/" + the lowercased handler name — /about
// for About), but a homepage answers at "/" (not "/home") and a
// named handler's default follows its name, so those sentences
// must carry the route with "at" or the page would land at the
// wrong path.
func TestPlanPageFollowupSentenceCarriesRouteWhenItDiffers(t *testing.T) {
	plainMain := map[string]string{
		"go.mod": pageGoMod,
		"main.go": `package main

import "fmt"

func main() {
	fmt.Println("just a program")
}
`,
	}
	noteFor := func(kind, page, name, route string) string {
		_, proj := serveProject(t, plainMain)
		plan, refusal := planPageReq(t, proj, EditRequest{
			File: "main.go", PageKind: kind, PageHTML: page, PageName: name, PageRoute: route,
		})
		if refusal != "" {
			t.Fatalf("kind %s refused: %s", kind, refusal)
		}
		return strings.Join(plan.Notes, "\n")
	}

	if notes := noteFor("home", "index.html", "Home", "/"); !strings.Contains(notes, `connect the Home handler to the webserver at /`) {
		t.Errorf("home note = %q, want the at-form sentence", notes)
	}
	if notes := noteFor("about", "about.html", "Solo", "/about"); !strings.Contains(notes, `connect the Solo handler to the webserver at /about`) {
		t.Errorf("solo note = %q, want the at-form sentence", notes)
	}
	if notes := noteFor("about", "about.html", "About", "/about"); !strings.Contains(notes, `connect the About handler to the webserver"`) {
		t.Errorf("about note = %q, want the bare sentence (its default route matches)", notes)
	}
}

func TestPlanPageOtherProgramsRoutesDoNotBlock(t *testing.T) {
	otherAbout := `package main

import (
	"fmt"
	"net/http"
)

func Other(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "other")
}

func main() {
	http.HandleFunc("/about", Other)
	fmt.Println("listening at http://localhost:8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println("the server stopped:", err)
	}
}
`
	// The homepage replaces its OWN program's starter, even
	// though another program's starter answers "/" too.
	_, proj := serveProject(t, map[string]string{
		"go.mod":        pageGoMod,
		"other/main.go": WebServerSource,
		"site/main.go":  WebServerSource,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{
		File: "site/main.go", PageKind: "home", PageHTML: "site/index.html", PageName: "Home", PageRoute: "/",
	})
	if refusal != "" {
		t.Fatalf("homepage refused beside another program's starter: %s", refusal)
	}
	if len(plan.Files) != 2 || plan.Files[1].Rel != "site/main.go" {
		t.Fatalf("plan files = %+v, want [site/index.html site/main.go]", plan.Files)
	}
	host := planFileText(t, plan, "site/main.go")
	if !strings.Contains(host, `http.HandleFunc("/", Home)`) {
		t.Errorf("own starter not replaced by the homepage registration:\n%s", host)
	}
	if strings.Contains(host, starterHelloMarker) {
		t.Errorf("own starter line survived the replacement:\n%s", host)
	}

	// An about page composes into its own program even
	// though another program already answers /about with a
	// handler of its own.
	_, proj2 := serveProject(t, map[string]string{
		"go.mod":        pageGoMod,
		"other/main.go": otherAbout,
		"site/main.go":  pageServerMain,
	})
	plan2, refusal2 := planPageReq(t, proj2, EditRequest{
		File: "site/main.go", PageKind: "about", PageHTML: "site/about.html", PageName: "About", PageRoute: "/about",
	})
	if refusal2 != "" {
		t.Fatalf("about page refused beside another program's /about: %s", refusal2)
	}
	host2 := planFileText(t, plan2, "site/main.go")
	if !strings.Contains(host2, `http.HandleFunc("/about", About)`) {
		t.Errorf("about registration missing:\n%s", host2)
	}
}

// htmlFieldNames extracts a starter page's form control names
// (name="..." attributes), in order.
var htmlNameRe = regexp.MustCompile(`name="([^"]+)"`)

func htmlFieldNames(page []byte) []string {
	var out []string
	for _, m := range htmlNameRe.FindAllStringSubmatch(string(page), -1) {
		out = append(out, m[1])
	}
	return out
}

func sameStringSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	set := map[string]int{}
	for _, s := range a {
		set[s]++
	}
	for _, s := range b {
		set[s]--
		if set[s] < 0 {
			return false
		}
	}
	return true
}

// Every kind, composed onto the default-mux server in the
// host file itself: the page carries the kind's markers, the
// handler serves it, and the registration lands at the kind's
// route BEFORE the serve call. Form kinds additionally pin
// the page/handler field agreement.
func TestPlanPageKindsCompose(t *testing.T) {
	cases := []struct {
		kind, name, file, page, route string
		concept                       bool
		markers                       []string
	}{
		{"form", "Form", "main.go", "form.html", "/form", true,
			[]string{"<title>form.html</title>", `<form method="post" action="/form">`, `<input type="email" name="email">`, `<textarea name="message" rows="5">`, `<button type="submit">Send</button>`}},
		{"home", "Home", "main.go", "index.html", "/", false,
			[]string{"<title>index.html</title>", "<h1>TODO: your homepage heading</h1>"}},
		{"about", "About", "main.go", "about.html", "/about", false,
			[]string{"<title>about.html</title>", "<h1>About us</h1>"}},
		{"contact", "Contact", "main.go", "contact.html", "/contact", true,
			[]string{"<title>contact.html</title>", "<h1>Contact us</h1>", `<form method="post" action="/contact">`}},
		{"list", "Items", "main.go", "items.html", "/items", false,
			[]string{"<title>items.html</title>", "<h1>TODO: your items</h1>", "<li>TODO: an item</li>", "loop over the data in the handler"}},
		{"details", "Details", "main.go", "details.html", "/details", false,
			[]string{"<title>details.html</title>", "<h1>TODO: the item's name</h1>", "<dl>", "<dt>Price</dt>"}},
	}
	for _, tc := range cases {
		t.Run(tc.kind, func(t *testing.T) {
			dir, proj := serveProject(t, map[string]string{
				"go.mod":  pageGoMod,
				"main.go": pageServerMain,
			})
			req := EditRequest{File: tc.file, PageKind: tc.kind, PageHTML: tc.page, PageName: tc.name, PageRoute: tc.route}
			if tc.concept {
				req.Concept = formHandlerEntry()
			}
			plan, refusal := planPageReq(t, proj, req)
			if refusal != "" {
				t.Fatalf("kind %s refused: %s", tc.kind, refusal)
			}
			if len(plan.Files) != 2 || plan.Files[0].Rel != tc.page || plan.Files[1].Rel != tc.file {
				t.Fatalf("plan files = %+v, want [%s %s]", plan.Files, tc.page, tc.file)
			}
			page := string(plan.Files[0].After)
			for _, want := range tc.markers {
				if !strings.Contains(page, want) {
					t.Errorf("starter page missing %q:\n%s", want, page)
				}
			}
			if n := strings.Count(page, "<li>TODO: an item</li>"); tc.kind == "list" && n != 3 {
				t.Errorf("list page has %d placeholder items, want 3:\n%s", n, page)
			}
			host := planFileText(t, plan, tc.file)
			if !strings.Contains(host, "func "+tc.name+"(w http.ResponseWriter, r *http.Request)") {
				t.Errorf("handler %s missing:\n%s", tc.name, host)
			}
			if !strings.Contains(host, `http.ServeFile(w, r, "`+tc.page+`")`) {
				t.Errorf("handler does not serve the page:\n%s", host)
			}
			reg := `http.HandleFunc("` + tc.route + `", ` + tc.name + `)`
			ri, li := strings.Index(host, reg), strings.Index(host, "http.ListenAndServe")
			if ri < 0 {
				t.Fatalf("registration %q missing:\n%s", reg, host)
			}
			if ri > li {
				t.Errorf("registration landed AFTER the serve call:\n%s", host)
			}
			if tc.concept {
				// The page's form fields and the handler's
				// FormValue reads are the same set, built
				// from the one list.
				pageFields := htmlFieldNames(plan.Files[0].After)
				handlerFields := formTemplateFields(host)
				if !sameStringSet(pageFields, handlerFields) {
					t.Errorf("page fields %v != handler fields %v", pageFields, handlerFields)
				}
				if !sameStringSet(pageFields, []string{"email", "subject", "message"}) {
					t.Errorf("form fields = %v, want email/subject/message", pageFields)
				}
				// The show-the-form comment is gone: the GET
				// branch serves the page now.
				if strings.Contains(host, formShowMarker) {
					t.Errorf("handler still carries the show-the-form TODO:\n%s", host)
				}
				notes := strings.Join(plan.Notes, "\n")
				if !strings.Contains(notes, "asks for exactly the fields "+tc.name+" reads") {
					t.Errorf("form agreement note missing:\n%s", notes)
				}
			}
			applyServePlan(t, dir, plan)
		})
	}
}

// No server anywhere: the plan lands page+handler with no
// registration, and the note spells the finishing sentence
// plus the no-webserver clause.
func TestPlanPageNoServerLandsUncomposed(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"main.go": `package main

import "fmt"

func main() {
	fmt.Println("just a program")
}
`,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "about", PageHTML: "about.html", PageName: "About", PageRoute: "/about"})
	if refusal != "" {
		t.Fatalf("no-server page refused: %s", refusal)
	}
	if len(plan.Files) != 2 {
		t.Fatalf("plan files = %+v, want [about.html main.go]", plan.Files)
	}
	host := planFileText(t, plan, "main.go")
	if !strings.Contains(host, "func About(w http.ResponseWriter, r *http.Request)") {
		t.Errorf("handler missing:\n%s", host)
	}
	if strings.Contains(host, "HandleFunc") {
		t.Errorf("uncomposed plan registered the handler anyway:\n%s", host)
	}
	notes := strings.Join(plan.Notes, "\n")
	for _, want := range []string{
		`Nothing serves /about yet. The finishing sentence is "connect the About handler to the webserver" — it registers About at /about.`,
		"This project has no webserver yet, so that sentence will first tell you the add-a-webserver step.",
	} {
		if !strings.Contains(notes, want) {
			t.Errorf("uncomposed note missing %q:\n%s", want, notes)
		}
	}
	applyServePlan(t, dir, plan)
}

// A server exists, but in another program: still uncomposed,
// and the note does NOT claim there is no webserver.
func TestPlanPageServerElsewhereLandsUncomposed(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":      pageGoMod,
		"main.go":     "package main\n\nimport \"fmt\"\n\nfunc main() { fmt.Println(\"root program\") }\n",
		"app/main.go": pageServerMain,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "about", PageHTML: "about.html", PageName: "About", PageRoute: "/about"})
	if refusal != "" {
		t.Fatalf("refused: %s", refusal)
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "Nothing serves /about yet.") {
		t.Errorf("uncomposed note missing:\n%s", notes)
	}
	if strings.Contains(notes, "no webserver yet") {
		t.Errorf("note claims no webserver, but app/main.go serves:\n%s", notes)
	}
	if len(plan.Files) != 2 {
		t.Fatalf("plan files = %+v, want page+host only (the other program is untouched)", plan.Files)
	}
}

// The homepage onto the webserver component: the starter
// registration at "/" is REPLACED by the homepage's — the
// friendly line is gone from the after bytes.
func TestPlanPageHomepageReplacesWebserverStarter(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":  pageGoMod,
		"main.go": WebServerSource,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "home", PageHTML: "index.html", PageName: "Home", PageRoute: "/"})
	if refusal != "" {
		t.Fatalf("homepage onto webserver refused: %s", refusal)
	}
	host := planFileText(t, plan, "main.go")
	if !strings.Contains(host, `http.HandleFunc("/", Home)`) {
		t.Errorf("homepage registration missing:\n%s", host)
	}
	if strings.Contains(host, "hello — your webserver is running") {
		t.Errorf("starter handler survived the replacement:\n%s", host)
	}
	if n := strings.Count(host, "http.ListenAndServe"); n != 1 {
		t.Errorf("after bytes have %d ListenAndServe calls, want 1:\n%s", n, host)
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, "starter handler at /") || !strings.Contains(notes, "is replaced by the registration above") {
		t.Errorf("starter-replaced note missing:\n%s", notes)
	}
	applyServePlan(t, dir, plan)
}

// About onto the same webserver fixture composes normally at
// /about — and the starter at "/" survives untouched.
func TestPlanPageAboutComposesBesideStarter(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod":  pageGoMod,
		"main.go": WebServerSource,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "about", PageHTML: "about.html", PageName: "About", PageRoute: "/about"})
	if refusal != "" {
		t.Fatalf("about onto webserver refused: %s", refusal)
	}
	host := planFileText(t, plan, "main.go")
	if !strings.Contains(host, `http.HandleFunc("/about", About)`) {
		t.Errorf("about registration missing:\n%s", host)
	}
	if !strings.Contains(host, "hello — your webserver is running") {
		t.Errorf("the starter handler was disturbed:\n%s", host)
	}
	applyServePlan(t, dir, plan)
}

// The route is held by the user's own handler: a refusal
// naming the incumbent, and nothing is planned.
func TestPlanPageRouteTakenRefuses(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func Login(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "login") }

func main() {
	http.HandleFunc("/about", Login)
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "about", PageHTML: "about.html", PageName: "AboutUs", PageRoute: "/about"})
	if plan != nil {
		t.Fatalf("taken route planned an edit")
	}
	for _, want := range []string{"/about is already taken", "registers it for Login", "AboutUs wasn't added"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("route-taken refusal missing %q:\n%s", want, refusal)
		}
	}
}

// The page already exists: never overwritten — a refusal in
// the family's words.
func TestPlanPagePageExistsRefuses(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod":     pageGoMod,
		"main.go":    pageServerMain,
		"about.html": "<html><body>mine</body></html>\n",
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "about", PageHTML: "about.html", PageName: "About", PageRoute: "/about"})
	if plan != nil {
		t.Fatalf("existing page planned an edit")
	}
	for _, want := range []string{"about.html is already there", "I never overwrite what's yours", "the about page wasn't made", "remove that file yourself first"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("page-exists refusal missing %q:\n%s", want, refusal)
		}
	}
}

// The handler's name is already declared in the host package:
// a refusal naming the declaring site and the way out.
func TestPlanPageHandlerNameCollisionRefuses(t *testing.T) {
	_, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func About(w http.ResponseWriter, r *http.Request) { fmt.Fprintln(w, "existing") }

func main() {
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "about", PageHTML: "about.html", PageName: "About", PageRoute: "/about"})
	if plan != nil {
		t.Fatalf("colliding name planned an edit")
	}
	for _, want := range []string{"package main already has a function named About", "main.go:", "about page handler can't take that name", `"named <Name>"`} {
		if !strings.Contains(refusal, want) {
			t.Errorf("collision refusal missing %q:\n%s", want, refusal)
		}
	}
}

// A mux-variable server composes as mux.HandleFunc.
func TestPlanPageMuxServerComposes(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

func main() {
	mux := http.NewServeMux()
	fmt.Println("listening")
	if err := http.ListenAndServe(":8080", mux); err != nil {
		fmt.Println(err)
	}
}
`,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "list", PageHTML: "items.html", PageName: "Items", PageRoute: "/items"})
	if refusal != "" {
		t.Fatalf("mux page refused: %s", refusal)
	}
	host := planFileText(t, plan, "main.go")
	reg := `mux.HandleFunc("/items", Items)`
	if !strings.Contains(host, reg) {
		t.Fatalf("mux registration missing:\n%s", host)
	}
	if strings.Index(host, reg) > strings.Index(host, "http.ListenAndServe") {
		t.Errorf("mux registration landed after the serve call:\n%s", host)
	}
	applyServePlan(t, dir, plan)
}

// A custom server (its own handler value) never composes:
// page+handler still land, with the follow-up note.
func TestPlanPageCustomServerLandsUncomposed(t *testing.T) {
	dir, proj := serveProject(t, map[string]string{
		"go.mod": pageGoMod,
		"main.go": `package main

import (
	"fmt"
	"net/http"
)

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
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "details", PageHTML: "details.html", PageName: "Details", PageRoute: "/details"})
	if refusal != "" {
		t.Fatalf("custom-server page refused: %s", refusal)
	}
	host := planFileText(t, plan, "main.go")
	if !strings.Contains(host, "func Details(w http.ResponseWriter, r *http.Request)") {
		t.Errorf("handler missing:\n%s", host)
	}
	if strings.Contains(host, `HandleFunc("/details"`) {
		t.Errorf("custom-server plan registered on a server it cannot reach:\n%s", host)
	}
	notes := strings.Join(plan.Notes, "\n")
	if !strings.Contains(notes, `connect the Details handler to the webserver`) {
		t.Errorf("follow-up note missing the connect sentence:\n%s", notes)
	}
	if strings.Contains(notes, "no webserver yet") {
		t.Errorf("note claims no webserver, but the custom server exists:\n%s", notes)
	}
	applyServePlan(t, dir, plan)
}

// A missing host file is created by the plan: the folder's
// own package when the folder has one, "main" for a fresh
// folder at a program-shaped name.
func TestPlanPageMissingHostFileCreatesIt(t *testing.T) {
	// A fresh folder: package main, handler inside, no server
	// anywhere to compose with.
	dir, proj := serveProject(t, map[string]string{
		"go.mod":         pageGoMod,
		"store/store.go": "package store\n\nfunc Load() int { return 1 }\n",
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "jim/detail.go", PageKind: "details", PageHTML: "jim/details.html", PageName: "Details", PageRoute: "/details"})
	if refusal != "" {
		t.Fatalf("missing-host page refused: %s", refusal)
	}
	if len(plan.Files) != 2 || plan.Files[0].Rel != "jim/details.html" || plan.Files[1].Rel != "jim/detail.go" {
		t.Fatalf("plan files = %+v, want [jim/details.html jim/detail.go]", plan.Files)
	}
	host := planFileText(t, plan, "jim/detail.go")
	if !strings.HasPrefix(host, "package main\n") || !strings.Contains(host, `"net/http"`) || !strings.Contains(host, "func Details(") {
		t.Errorf("created host file is wrong:\n%s", host)
	}
	applyServePlan(t, dir, plan)

	// An existing library package's folder: the new host file
	// takes that package's name.
	dir2, proj2 := serveProject(t, map[string]string{
		"go.mod":            pageGoMod,
		"handlers/keep.go":  "package handlers\n\nfunc Keep() {}\n",
		"web/main.go":       pageServerMain,
		"store/store.go":    "package store\n\nfunc Load() int { return 1 }\n",
		"handlers/dummy.go": "package handlers\n\nfunc Dummy() {}\n",
	})
	_ = dir2
	plan2, refusal := planPageReq(t, proj2, EditRequest{File: "handlers/contact.go", PageKind: "contact", PageHTML: "handlers/contact.html", PageName: "Contact", PageRoute: "/contact", Concept: formHandlerEntry()})
	if refusal != "" {
		t.Fatalf("library-folder page refused: %s", refusal)
	}
	host2 := planFileText(t, plan2, "handlers/contact.go")
	if !strings.HasPrefix(host2, "package handlers\n") {
		t.Errorf("created host file does not carry the folder's package:\n%s", host2)
	}
	applyServePlan(t, dir2, plan2)
}

// The form template has drifted — its show-the-form comment
// is gone: the shape-drift refusal, and nothing is planned.
func TestPlanPageFormTemplateDriftRefuses(t *testing.T) {
	entry := formHandlerEntry()
	entry.Template = strings.Replace(entry.Template,
		"\t\t// TODO: show the form here — a browser asks for it with GET before it can send anything\n", "", 1)
	_, proj := serveProject(t, map[string]string{
		"go.mod":  pageGoMod,
		"main.go": pageServerMain,
	})
	plan, refusal := planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "form", PageHTML: "form.html", PageName: "Form", PageRoute: "/form", Concept: entry})
	if plan != nil {
		t.Fatalf("drifted template planned an edit")
	}
	for _, want := range []string{"doesn't have the shape this recipe assembles from", "no place that shows the form", "Nothing was written"} {
		if !strings.Contains(refusal, want) {
			t.Errorf("drift refusal missing %q:\n%s", want, refusal)
		}
	}

	// A template naming no fields at all drifts the other way.
	entry2 := formHandlerEntry()
	entry2.Template = "func {name}(w http.ResponseWriter, r *http.Request) {\n\t// TODO: show the form here — a browser asks for it with GET before it can send anything\n}"
	plan, refusal = planPageReq(t, proj, EditRequest{File: "main.go", PageKind: "form", PageHTML: "form.html", PageName: "Form", PageRoute: "/form", Concept: entry2})
	if plan != nil {
		t.Fatalf("field-less template planned an edit")
	}
	if !strings.Contains(refusal, "it names no form fields") {
		t.Errorf("field-less refusal:\n%s", refusal)
	}
}

// The starter pages themselves: shell and per-kind bodies,
// pinned directly at the exported builder.
func TestPageStarterPageKinds(t *testing.T) {
	fields := []string{"email", "subject", "message"}
	home := string(PageStarterPage("home", "/", nil))
	for _, want := range []string{"<!DOCTYPE html>", "<title>index.html</title>", "<h1>TODO: your homepage heading</h1>"} {
		if !strings.Contains(home, want) {
			t.Errorf("home starter missing %q:\n%s", want, home)
		}
	}
	form := string(PageStarterPage("form", "/form", fields))
	for _, want := range []string{"<title>form.html</title>", `<form method="post" action="/form">`, `<input type="email" name="email">`, `<input type="text" name="subject">`, `<textarea name="message" rows="5"></textarea>`, `<button type="submit">Send</button>`} {
		if !strings.Contains(form, want) {
			t.Errorf("form starter missing %q:\n%s", want, form)
		}
	}
	details := string(PageStarterPage("details", "/details", nil))
	for _, want := range []string{"<title>details.html</title>", "<dt>Name</dt>", "<dt>Description</dt>", "<dt>Price</dt>"} {
		if !strings.Contains(details, want) {
			t.Errorf("details starter missing %q:\n%s", want, details)
		}
	}
}
