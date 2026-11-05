package chat

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The page family's grammar at the parser: every kind phrase,
// tails in both orders, and the pinned non-claims — the
// neighbours' sentences the kind words must never steal.

func TestParsePageFamilyClaims(t *testing.T) {
	cases := []struct {
		line, kind, name, target string
	}{
		// form: all three kind phrases.
		{"create a form page", "form", "", ""},
		{"add a handler with a form", "form", "", ""},
		{"make a handler named Signup with a form", "form", "Signup", ""},
		{"create a handler with a form named Signup", "form", "Signup", ""},
		{"build a form page named Signup to jim/main.go", "form", "Signup", "jim/main.go"},
		{"create a form page to jim/main.go named Signup", "form", "Signup", "jim/main.go"},
		// home.
		{"create a homepage", "home", "", ""},
		{"add a home page", "home", "", ""},
		// "put" joined the verbs 2026-11-05 — the family's
		// fifth spelling of the ask.
		{"put a homepage in jim", "home", "", "jim"},
		{"put a form page named Signup to jim/main.go", "form", "Signup", "jim/main.go"},
		{"create a homepage named Landing in jim", "home", "Landing", "jim"},
		{"make a home page in jim named Landing", "home", "Landing", "jim"},
		// about.
		{"create an about page", "about", "", ""},
		{"add an about us page to main.go", "about", "", "main.go"},
		{"create an about page named Story into the folder jim", "about", "Story", "jim"},
		// contact.
		{"create a contact page", "contact", "", ""},
		{"make a contact us page named Reach", "contact", "Reach", ""},
		{"add a contact page to jim/main.go named Reach", "contact", "Reach", "jim/main.go"},
		// list.
		{"create an item list page", "list", "", ""},
		{"add a list page to jim", "list", "", "jim"},
		// details.
		{"create a details page", "details", "", ""},
		{"build an item details page named Detail to app/main.go", "details", "Detail", "app/main.go"},
	}
	for _, c := range cases {
		req, ok := parsePageFamily(c.line)
		if !ok {
			t.Errorf("%q: not claimed", c.line)
			continue
		}
		if req.kind != c.kind || req.name != c.name || req.target != c.target {
			t.Errorf("%q: got {kind:%q name:%q target:%q}, want {kind:%q name:%q target:%q}",
				c.line, req.kind, req.name, req.target, c.kind, c.name, c.target)
		}
		if req.targetMissing || req.nameMissing {
			t.Errorf("%q: unexpected missing flag (targetMissing=%v nameMissing=%v)", c.line, req.targetMissing, req.nameMissing)
		}
	}
}

func TestParsePageFamilyNonClaims(t *testing.T) {
	for _, line := range []string{
		// The book's pieces — the reason the kind phrases are
		// shaped as they are.
		"add a form handler named Jake to jim/main.go",
		"add a handler named Jake to jim/main.go",
		"add a contact form named X to main.go",
		"add an html form named X to main.go",
		// The neighbours' sentences.
		"add a webserver to main.go",
		"serve an html file",
		"serve home.html to webtest/main.go",
		"connect the Jake handler to the webserver",
		"create a folder jim",
		"create a file jim/main.go with a webserver",
		// Trailing prose is another ask glued on — not claimed.
		"create a homepage please",
		"create a homepage and add a form page",
		// Not a page verb, or no kind phrase at all.
		"create a page",
		"make a homepageee",
		"delete a homepage",
	} {
		if req, ok := parsePageFamily(line); ok {
			t.Errorf("%q: claimed as %+v, want no claim", line, req)
		}
	}
}

// Started-but-unfinished tail clauses still claim: the flow
// answers in page terms instead of the line falling through.
func TestParsePageFamilyPartialClaims(t *testing.T) {
	for _, line := range []string{
		"create a homepage in",
		"create a homepage to",
		"create an about page into the folder",
		"create a form page to the file",
	} {
		req, ok := parsePageFamily(line)
		if !ok || !req.targetMissing || req.nameMissing {
			t.Errorf("%q: got %+v, %v — want claimed with targetMissing only", line, req, ok)
		}
	}
	for _, line := range []string{
		"create a form page named",
		"create a contact page named",
		"create a homepage named",
	} {
		req, ok := parsePageFamily(line)
		if !ok || !req.nameMissing || req.targetMissing {
			t.Errorf("%q: got %+v, %v — want claimed with nameMissing only", line, req, ok)
		}
	}
}

// The partial answers through the chain: claimed, spelled in
// page terms with the kind's own canonical sentence, nothing
// written.
func TestPageFamilyPartialsThroughChain(t *testing.T) {
	root := pageSiteRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	before := readBack(t, root, "main.go")

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("create a homepage in", yes, profile) })
	if !handled {
		t.Fatalf("target-missing page line fell through:\n%s", out)
	}
	if !strings.Contains(out, `"create a homepage in jim"`) || !strings.Contains(out, `"create a homepage to jim/main.go"`) {
		t.Errorf("target-missing answer does not spell both shapes:\n%s", out)
	}
	out = captureStdout(t, func() { handled = claimChain("create a form page named", yes, profile) })
	if !handled {
		t.Fatalf("name-missing page line fell through:\n%s", out)
	}
	if !strings.Contains(out, `"create a form page named Signup"`) {
		t.Errorf("name-missing answer does not spell the named shape:\n%s", out)
	}
	if got := readBack(t, root, "main.go"); got != before {
		t.Errorf("a partial answer changed main.go")
	}
	if fileExists(t, root, "index.html") || fileExists(t, root, "form.html") {
		t.Errorf("a partial answer created a page")
	}
}

// The target settling, at the resolver: the folder rule, the
// library folder's own-file move, and the refusals verbatim.
func TestResolvePageTargets(t *testing.T) {
	t.Run("root with no program creates main.go", func(t *testing.T) {
		dir := t.TempDir()
		got, refusal := resolvePageTargets(dir, pageRequest{kind: "home"})
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		if got.goRel != "main.go" || got.goExists || got.htmlRel != "index.html" {
			t.Errorf("got %+v, want {main.go false index.html}", got)
		}
	})
	t.Run("missing go file whose folder exists is created", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "jim/keep.txt", "x")
		got, refusal := resolvePageTargets(dir, pageRequest{kind: "about", target: "jim/new.go"})
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		if got.goRel != "jim/new.go" || got.goExists || got.htmlRel != "jim/about.html" {
			t.Errorf("got %+v, want {jim/new.go false jim/about.html}", got)
		}
	})
	t.Run("library folder takes the handler's own file", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "lib/lib.go", "package lib\n\nfunc Thing() {}\n")
		got, refusal := resolvePageTargets(dir, pageRequest{kind: "about", target: "lib"})
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		if got.goRel != "lib/about.go" || got.goExists || got.htmlRel != "lib/about.html" {
			t.Errorf("got %+v, want {lib/about.go false lib/about.html}", got)
		}
		// The handler's own file already there: it is the host.
		writeFile(t, dir, "lib/about.go", "package lib\n")
		got, refusal = resolvePageTargets(dir, pageRequest{kind: "about", target: "lib"})
		if refusal != "" {
			t.Fatalf("refused: %s", refusal)
		}
		if got.goRel != "lib/about.go" || !got.goExists {
			t.Errorf("got %+v, want {lib/about.go true}", got)
		}
		// A named handler names its own file, lowercased.
		got, _ = resolvePageTargets(dir, pageRequest{kind: "about", name: "Info", target: "lib"})
		if got.goRel != "lib/info.go" {
			t.Errorf("named library host = %q, want lib/info.go", got.goRel)
		}
	})
	t.Run("two main-package files refuse with both named", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "duo/a.go", "package main\n")
		writeFile(t, dir, "duo/b.go", "package main\n")
		_, refusal := resolvePageTargets(dir, pageRequest{kind: "home", target: "duo"})
		want := `duo has more than one main-package file (a.go, b.go) — say which one, like "create a homepage to duo/a.go"`
		if refusal != want {
			t.Errorf("refusal = %q, want %q", refusal, want)
		}
	})
	t.Run("html target refuses in page words", func(t *testing.T) {
		dir := t.TempDir()
		_, refusal := resolvePageTargets(dir, pageRequest{kind: "about", target: "about.html"})
		want := `about.html is a page, not a Go file — the page's handler goes in a .go file and the page lands beside it. Say "create an about page to main.go"`
		if refusal != want {
			t.Errorf("refusal = %q, want %q", refusal, want)
		}
	})
	t.Run("existing non-go file refuses in page words", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, dir, "notes.txt", "hi")
		_, refusal := resolvePageTargets(dir, pageRequest{kind: "home", target: "notes.txt"})
		want := `notes.txt isn't a Go file — a page's handler is a Go function, so it goes in a .go file. Say "create a homepage to main.go"`
		if refusal != want {
			t.Errorf("refusal = %q, want %q", refusal, want)
		}
	})
	t.Run("missing non-go target refuses in page words", func(t *testing.T) {
		dir := t.TempDir()
		_, refusal := resolvePageTargets(dir, pageRequest{kind: "home", target: "notes.txt"})
		want := `notes.txt isn't a Go file — name a .go file for the handler, like "create a homepage to main.go"`
		if refusal != want {
			t.Errorf("refusal = %q, want %q", refusal, want)
		}
	})
	t.Run("missing folder for a go target names the create line", func(t *testing.T) {
		dir := t.TempDir()
		_, refusal := resolvePageTargets(dir, pageRequest{kind: "home", target: "nope/main.go"})
		want := `there's no folder nope in this project — say "create a folder nope" first, then ask again`
		if refusal != want {
			t.Errorf("refusal = %q, want %q", refusal, want)
		}
	})
}

// pageSiteRoot is a project whose root program is the shared
// webserver component: package main, the starter "/"
// registration with the hello line, ListenAndServe.
func pageSiteRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/pagesite\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", analyze.WebServerSource)
	return dir
}

// The headline journey: a whole site built one page at a time
// into one running program — every kind lands its page, its
// handler, and its route, each behind the gate and the build.
func TestPageFamilySiteJourneyThroughChain(t *testing.T) {
	root := pageSiteRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	steps := []struct {
		line, page, handler, route string
	}{
		{"create a homepage", "index.html", "Home", "/"},
		{"create an about page", "about.html", "About", "/about"},
		{"create a contact page", "contact.html", "Contact", "/contact"},
		{"create an item list page", "items.html", "Items", "/items"},
		{"create a details page", "details.html", "Details", "/details"},
		{"create a form page", "form.html", "Form", "/form"},
	}
	for _, st := range steps {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(st.line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through the claim chain:\n%s", st.line, out)
		}
		requireNoAnalyzeOutput(t, st.line, out)
		if !fileExists(t, root, st.page) {
			t.Errorf("%q: %s was not created:\n%s", st.line, st.page, out)
		}
		goSrc := readBack(t, root, "main.go")
		if !strings.Contains(goSrc, "func "+st.handler+"(") {
			t.Errorf("%q: main.go declares no %s handler:\n%s", st.line, st.handler, goSrc)
		}
		reg := `http.HandleFunc("` + st.route + `", ` + st.handler + `)`
		ri, li := strings.Index(goSrc, reg), strings.Index(goSrc, "http.ListenAndServe")
		if ri < 0 {
			t.Errorf("%q: main.go has no registration %s:\n%s", st.line, reg, goSrc)
		} else if li >= 0 && ri > li {
			t.Errorf("%q: the registration landed after the serve call:\n%s", st.line, goSrc)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Errorf("%q: spine build/verify not reported:\n%s", st.line, out)
		}
		if st.handler == "Home" && strings.Contains(goSrc, "hello — your webserver is running") {
			t.Errorf("the homepage did not take over the starter registration:\n%s", goSrc)
		}
	}
	// The contact page and its handler agree on the fields.
	contact := readBack(t, root, "contact.html")
	for _, field := range []string{`name="email"`, `name="subject"`, `name="message"`} {
		if !strings.Contains(contact, field) {
			t.Errorf("contact.html is missing %s:\n%s", field, contact)
		}
	}
	goSrc := readBack(t, root, "main.go")
	for _, field := range []string{`FormValue("email")`, `FormValue("subject")`, `FormValue("message")`} {
		if !strings.Contains(goSrc, field) {
			t.Errorf("the handlers do not read %s:\n%s", field, goSrc)
		}
	}
}

// The no-server journey: the page and its handler land, the
// answer names the wiring sentence, and feeding that sentence
// back — after adding the server the empty main can take —
// completes the site.
func TestPageFamilyNoServerFeedbackJourney(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "go.mod", "module example.com/noserver\n\ngo 1.21\n")
	writeFile(t, root, "main.go", "package main\n\nfunc main() {}\n")
	withChatSession(t, root, true)
	profile := &UserProfile{}

	var handled bool
	out := captureStdout(t, func() { handled = claimChain("create an about page", yes, profile) })
	if !handled {
		t.Fatalf("the about ask fell through:\n%s", out)
	}
	if !fileExists(t, root, "about.html") {
		t.Fatalf("the page was not created:\n%s", out)
	}
	if got := readBack(t, root, "main.go"); !strings.Contains(got, "func About(") {
		t.Fatalf("the handler was not written:\n%s", got)
	}
	if !strings.Contains(out, "connect the About handler to the webserver") {
		t.Errorf("the answer does not name the wiring follow-up:\n%s", out)
	}

	out = captureStdout(t, func() { handled = claimChain("add a webserver to main.go", yes, profile) })
	if !handled {
		t.Fatalf("add-a-webserver fell through:\n%s", out)
	}
	if got := readBack(t, root, "main.go"); !strings.Contains(got, "http.ListenAndServe") {
		t.Fatalf("the webserver did not land:\n%s", got)
	}

	out = captureStdout(t, func() { handled = claimChain("connect the About handler to the webserver", yes, profile) })
	if !handled {
		t.Fatalf("the wiring sentence fell through:\n%s", out)
	}
	if got := readBack(t, root, "main.go"); !strings.Contains(got, `http.HandleFunc("/about", About)`) {
		t.Errorf("About was not registered at /about:\n%s", got)
	}
}

// An existing page is never overwritten: the ask writes
// nothing at all and says so, naming the page.
func TestPageFamilyExistingPageRefused(t *testing.T) {
	root := pageSiteRoot(t)
	writeFile(t, root, "about.html", "<html><body>mine</body></html>\n")
	withChatSession(t, root, true)
	profile := &UserProfile{}
	before, _ := os.ReadFile(filepath.Join(root, "main.go"))
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("create an about page", yes, profile) })
	if !handled {
		t.Fatalf("the about ask fell through:\n%s", out)
	}
	if !strings.Contains(out, "about.html") || !strings.Contains(out, "never overwrite") {
		t.Errorf("the answer does not name the page and the rule:\n%s", out)
	}
	after, _ := os.ReadFile(filepath.Join(root, "main.go"))
	if string(after) != string(before) {
		t.Errorf("the refused ask changed main.go:\n%s", out)
	}
	if got := readBack(t, root, "about.html"); got != "<html><body>mine</body></html>\n" {
		t.Errorf("the existing page was changed:\n%s", got)
	}
}

// A gate decline writes nothing — no page, no handler.
func TestPageFamilyGateDeclineWritesNothing(t *testing.T) {
	root := pageSiteRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	before := readBack(t, root, "main.go")
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("create an about page", no, profile) })
	if !handled {
		t.Fatalf("the about ask fell through:\n%s", out)
	}
	if !strings.Contains(out, "Not applied") {
		t.Errorf("the decline was not reported:\n%s", out)
	}
	if fileExists(t, root, "about.html") {
		t.Errorf("the decline created the page anyway")
	}
	if got := readBack(t, root, "main.go"); got != before {
		t.Errorf("the decline changed main.go")
	}
}

// Non-theft through the chain: the book's form-handler line,
// serve-html's line, and wire-a-handler's line each keep their
// own claim with the page family seated among them.
func TestPageFamilyDoesNotStealNeighbours(t *testing.T) {
	// The book's form handler lands alone: no page, no second
	// handler.
	root := t.TempDir()
	writeFile(t, root, "go.mod", "module example.com/bookform\n\ngo 1.21\n")
	writeFile(t, root, "main.go", "package main\n\nfunc main() {}\n")
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("add a form handler named Jake to main.go", yes, profile) })
	if !handled {
		t.Fatalf("the book's line fell through:\n%s", out)
	}
	if got := readBack(t, root, "main.go"); !strings.Contains(got, "func Jake(") {
		t.Errorf("the book's handler did not land:\n%s", got)
	}
	if fileExists(t, root, "form.html") {
		t.Errorf("the book's line created a page — the page family stole it")
	}
	if got := readBack(t, root, "main.go"); strings.Contains(got, "func Form(") {
		t.Errorf("the book's line grew a second, page-family handler:\n%s", got)
	}

	// Serve-html's line serves its page its own way: no named
	// page-family handler appears.
	root2 := t.TempDir()
	writeFile(t, root2, "go.mod", "module example.com/serveown\n\ngo 1.21\n")
	writeFile(t, root2, "main.go", "package main\n")
	withChatSession(t, root2, true)
	var handled2 bool
	out = captureStdout(t, func() { handled2 = claimChain("serve home.html to main.go", yes, profile) })
	if !handled2 {
		t.Fatalf("serve-html's line fell through:\n%s", out)
	}
	if !fileExists(t, root2, "home.html") {
		t.Errorf("serve-html did not create its page:\n%s", out)
	}
	if got := readBack(t, root2, "main.go"); !strings.Contains(got, "http.ServeFile") || strings.Contains(got, "func Home(") {
		t.Errorf("serve-html's wiring was not its own:\n%s", got)
	}

	// Wire-a-handler's line still wires.
	root3 := wireJimRoot(t)
	withChatSession(t, root3, true)
	var handled3 bool
	out = captureStdout(t, func() { handled3 = claimChain("connect the Jake handler to the webserver", yes, profile) })
	if !handled3 {
		t.Fatalf("the wiring line fell through:\n%s", out)
	}
	if got := readBack(t, root3, "jim/main.go"); !strings.Contains(got, `http.HandleFunc("/jake", Jake)`) {
		t.Errorf("the wiring did not land:\n%s", got)
	}
}

var (
	pageNameAttrRe  = regexp.MustCompile(`name="([^"]+)"`)
	pageFormValueRe = regexp.MustCompile(`FormValue\("([^"]+)"\)`)
)

func pageFieldSet(re *regexp.Regexp, text string) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range re.FindAllStringSubmatch(text, -1) {
		if !seen[m[1]] {
			seen[m[1]] = true
			out = append(out, m[1])
		}
	}
	sort.Strings(out)
	return out
}

// The field-match contract: the form kinds' pages and handlers
// are assembled from the same book entry the chat flowed
// through conceptEntries, so the fields a browser can submit
// are exactly the fields the handler reads.
func TestPageFamilyFormFieldsMatchHandler(t *testing.T) {
	for _, tc := range []struct{ line, page string }{
		{"create a contact page", "contact.html"},
		{"create a form page", "form.html"},
	} {
		root := pageSiteRoot(t)
		withChatSession(t, root, true)
		profile := &UserProfile{}
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(tc.line, yes, profile) })
		if !handled {
			t.Fatalf("%q fell through:\n%s", tc.line, out)
		}
		pageFields := pageFieldSet(pageNameAttrRe, readBack(t, root, tc.page))
		handlerFields := pageFieldSet(pageFormValueRe, readBack(t, root, "main.go"))
		if len(pageFields) == 0 {
			t.Fatalf("%s carries no named fields:\n%s", tc.page, readBack(t, root, tc.page))
		}
		if strings.Join(pageFields, ",") != strings.Join(handlerFields, ",") {
			t.Errorf("%q: page fields %v != handler FormValue fields %v", tc.line, pageFields, handlerFields)
		}
	}
}
