package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The three-sources build (2026-10-31): How To Code in Go
// (DigitalOcean), Astaxie §3.2, and Practical Go Lessons ch.1,
// learned as data in the five-sources pattern. These tests pin
// the build's three moving parts:
//
//   - Struct tags (the book's ch. 39) are NOT a concept row —
//     a tag belongs to a field — so they ride the assembled
//     add-field grammar as an optional tag phrase, pinned at
//     parse level, end to end, and at the planner's honest
//     refusal of a tag phrase it can't derive.
//   - Form values (Astaxie §3.2's form parsing) is a statement
//     row whose {request} slot binds the target handler's
//     real request parameter — pinned for its claims, its
//     landing in handlers (including one whose parameter is
//     named req), and its refusals outside handlers.
//   - The three sources' reading is seated on the resources
//     shelf (referenceSeed), pinned at the matcher and on the
//     printed page.

func TestParseAssembledEditStructTags(t *testing.T) {
	cases := []struct {
		line string
		want analyze.EditRequest
	}{
		// The tag phrase right after the type (form A).
		{"add a Name field of type string with a json tag name to Profile",
			analyze.EditRequest{Kind: analyze.EditAddField, FieldName: "Name", FieldType: "string", StructName: "Profile", FieldTag: `json:"name"`}},
		// The tag phrase at the end of the sentence.
		{"add a Name field of type string to Profile with a json tag name",
			analyze.EditRequest{Kind: analyze.EditAddField, FieldName: "Name", FieldType: "string", StructName: "Profile", FieldTag: `json:"name"`}},
		// Type-first (form B) with a tag.
		{"add a string field Name with an xml tag user_name to Profile",
			analyze.EditRequest{Kind: analyze.EditAddField, FieldName: "Name", FieldType: "string", StructName: "Profile", FieldTag: `xml:"user_name"`}},
		// A quoted value carrying a comma option.
		{"add a Name field of type string with a json tag \"name,omitempty\" to Profile",
			analyze.EditRequest{Kind: analyze.EditAddField, FieldName: "Name", FieldType: "string", StructName: "Profile", FieldTag: `json:"name,omitempty"`}},
		// A tag alongside the after-anchor.
		{"add a Name field of type string with a db tag created_at to Profile after Id",
			analyze.EditRequest{Kind: analyze.EditAddField, FieldName: "Name", FieldType: "string", StructName: "Profile", AfterField: "Id", FieldTag: `db:"created_at"`}},
		// The plain forms are untouched: no tag, no FieldTag.
		{"add a Name field of type string to Profile",
			analyze.EditRequest{Kind: analyze.EditAddField, FieldName: "Name", FieldType: "string", StructName: "Profile"}},
	}
	for _, c := range cases {
		got, ok := parseAssembledEdit(c.line)
		if !ok {
			t.Errorf("parseAssembledEdit(%q) not claimed, want %+v", c.line, c.want)
			continue
		}
		if got != c.want {
			t.Errorf("parseAssembledEdit(%q) =\n  %+v\nwant\n  %+v", c.line, got, c.want)
		}
	}
}

// tagProject seeds a scratch project holding a Profile struct.
func tagProject(t *testing.T) (string, *UserProfile) {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/tags\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	writeFile(t, dir, "profile.go", "package main\n\ntype Profile struct {\n\tId int\n}\n")
	withChatSession(t, dir, true)
	return dir, &UserProfile{}
}

func TestStructTagFieldEndToEnd(t *testing.T) {
	dir, profile := tagProject(t)
	var claimed bool
	out := captureStdout(t, func() {
		claimed = claimChain("add a Name field of type string with a json tag name to Profile", yes, profile)
	})
	if !claimed {
		t.Fatalf("tagged add-field fell through:\n%s", out)
	}
	if !strings.Contains(out, "carries the struct tag `json:\"name\"`") {
		t.Errorf("plan does not name the tag it will write:\n%s", out)
	}
	got := readBack(t, dir, "profile.go")
	if !strings.Contains(got, "Name string `json:\"name\"`") {
		t.Errorf("tagged field did not land:\n%s", got)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine build/verify not reported:\n%s", out)
	}
}

// A tag phrase the grammar cannot derive — no value, or a
// value with a space in it — is swallowed into the plain
// form's type span, and the planner must refuse it by name
// rather than write a field of a nonsense type.
func TestStructTagMalformedRefused(t *testing.T) {
	for _, line := range []string{
		"add a Name field of type string with a json tag to Profile",
		"add a Name field of type string with a json tag my name to Profile",
	} {
		dir, profile := tagProject(t)
		before := readBack(t, dir, "profile.go")
		var claimed bool
		out := captureStdout(t, func() { claimed = claimChain(line, yes, profile) })
		if !claimed {
			t.Errorf("%q: fell through entirely — the refusal must come from the edit chain", line)
			continue
		}
		if !strings.Contains(out, "a type can't hold a tag") {
			t.Errorf("%q: refusal does not name the problem:\n%s", line, out)
		}
		if got := readBack(t, dir, "profile.go"); got != before {
			t.Errorf("%q: refused edit changed the file:\n%s", line, got)
		}
	}
}

func TestParseConceptFormValues(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct {
		line, id, inFunc string
	}{
		{"add a form values in Login to web.go", "form-values", "Login"},
		{"add form values in Login to web.go", "form-values", "Login"},
		{"add a form data in Login to web.go", "form-values", "Login"},
		{"add a parse form in Login to web.go", "form-values", "Login"},
		{"add submitted form values in Login to web.go", "form-values", "Login"},
		// The form handler keeps its own asks — the new row
		// must not steal the declaration.
		{"add a form handler named Login to web.go", "form-handler", ""},
		{"add a form named Contact to web.go", "form-handler", ""},
	}
	for _, c := range cases {
		req, ok := parseConceptFrom(c.line, book)
		if !ok {
			t.Errorf("parseConcept(%q) not claimed, want row %q", c.line, c.id)
			continue
		}
		if req.entry.ID != c.id {
			t.Errorf("parseConcept(%q) = row %q, want %q", c.line, req.entry.ID, c.id)
		}
		if req.inFunc != c.inFunc {
			t.Errorf("parseConcept(%q) inFunc = %q, want %q", c.line, req.inFunc, c.inFunc)
		}
	}
}

// formProject seeds a scratch project whose web.go holds a
// handler with the given request parameter name and a plain
// helper function, plus a main in main.go.
func formProject(t *testing.T, param string) (string, *UserProfile) {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/forms\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	writeFile(t, dir, "web.go", "package main\n\nimport \"net/http\"\n\nfunc Login(w http.ResponseWriter, "+param+" *http.Request) {\n\t// TODO: show the login form here\n}\n\nfunc Helper() {\n\tprintln(\"helping\")\n}\n")
	withChatSession(t, dir, true)
	return dir, &UserProfile{}
}

func TestFormValuesEndToEnd(t *testing.T) {
	for _, param := range []string{"r", "req"} {
		dir, profile := formProject(t, param)
		var claimed bool
		out := captureStdout(t, func() {
			claimed = claimChain("add a form values in Login to web.go", yes, profile)
		})
		if !claimed {
			t.Fatalf("param %s: form values fell through:\n%s", param, out)
		}
		got := readBack(t, dir, "web.go")
		for _, want := range []string{
			param + ".ParseForm()",
			"for key, values := range " + param + ".Form {",
			param + ".FormValue(\"email\")",
		} {
			if !strings.Contains(got, want) {
				t.Errorf("param %s: landed piece missing %q:\n%s", param, want, got)
			}
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Errorf("param %s: spine build/verify not reported:\n%s", param, out)
		}
	}
}

// Outside a handler the row refuses by name: aimed at a
// plain function it says that function doesn't receive a
// request and points at the handler that does; aimed at a
// missing function it says so and lists what the file has.
func TestFormValuesRefusals(t *testing.T) {
	dir, profile := formProject(t, "r")
	before := readBack(t, dir, "web.go")

	var claimed bool
	out := captureStdout(t, func() {
		claimed = claimChain("add a form values in Helper to web.go", yes, profile)
	})
	if !claimed {
		t.Fatalf("form values into a plain function fell through:\n%s", out)
	}
	if !strings.Contains(out, "doesn't receive an http.Request") || !strings.Contains(out, "Login") {
		t.Errorf("refusal does not name the problem and the way out:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimed = claimChain("add a form values in Missing to web.go", yes, profile)
	})
	if !claimed || !strings.Contains(out, "no function named \"Missing\"") {
		t.Errorf("missing-handler refusal wrong (claimed=%v):\n%s", claimed, out)
	}
	if got := readBack(t, dir, "web.go"); got != before {
		t.Errorf("refused edits changed the file:\n%s", got)
	}
}

// The three sources on the shelf: each topic's goal earns its
// entry, under its own source's label, and nothing sprays.
func TestThreeSourcesShelf(t *testing.T) {
	want := []struct {
		goal  string
		label string
		url   string
	}{
		{"struct tags", "How To Use Struct Tags in Go (Chapter 39)", howToCodeURL},
		{"build tags", "Customizing Go Binaries with Build Tags (Chapter 34)", howToCodeURL},
		{"ldflags", "Using ldflags to Set Version Information for Go Applications (Chapter 42)", howToCodeURL},
		{"cross compiling", "Building Go Applications for Different Operating Systems and Architectures (Chapter 41)", howToCodeURL},
		{"writing packages", "How To Write Packages in Go (Chapters 23–25)", howToCodeURL},
		{"flag package", "How To Use the Flag Package in Go (Chapter 43)", howToCodeURL},
		{"custom errors", "Creating Custom Errors in Go (Chapter 21)", howToCodeURL},
		{"variadic functions", "How To Use Variadic Functions in Go (Chapter 31)", howToCodeURL},
		{"pointers", "Understanding Pointers in Go (Chapter 35)", howToCodeURL},
		{"type conversions", "How To Convert Data Types in Go (Chapter 15)", howToCodeURL},
		{"installing go", "How To Install Go and Set Up a Local Programming Environment (Chapters 4–6)", howToCodeURL},
		{"form parsing", "Build a Simple Web Server (§3.2)", astaxie32URL},
		{"form values", "Build a Simple Web Server (§3.2)", astaxie32URL},
		{"how a computer works", "Programming a Computer (Chapter 1)", practicalGo1URL},
		{"compiled language", "Programming a Computer (Chapter 1)", practicalGo1URL},
	}
	for _, w := range want {
		found := false
		for _, l := range referenceLabels(w.goal) {
			if l == w.label+"|"+w.url {
				found = true
			}
		}
		if !found {
			t.Errorf("goal %q: shelf entry %q not earned; got %v", w.goal, w.label, referenceLabels(w.goal))
		}
	}
	// Single-topic goals earn exactly their entry — no
	// pile-on from a neighbouring chapter.
	if got := referenceLabels("pointers"); len(got) != 1 {
		t.Errorf("goal pointers earned %d entries, want exactly 1: %v", len(got), got)
	}
	if got := referenceLabels("build tags"); len(got) != 1 {
		t.Errorf("goal build tags earned %d entries, want exactly 1: %v", len(got), got)
	}
}

// The shelf on the printed page: an explicit resources ask
// the Advocates doc doesn't cover rides the seed, grouped
// under the book's own name. (Called directly, like the
// five-sources reference tests — the full chain routes
// "build …" lines to the composer first.)
func TestBuildResourcesThreeSourcesRideAlong(t *testing.T) {
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("resources for struct tags", nil) {
			t.Error("explicit resources ask not claimed")
		}
	})
	for _, want := range []string{"From How To Code in Go:", "How To Use Struct Tags in Go (Chapter 39)", howToCodeURL} {
		if !strings.Contains(out, want) {
			t.Errorf("resources page missing %q:\n%s", want, out)
		}
	}
	out = captureStdout(t, func() {
		if !tryBuildResources("resources for form parsing", nil) {
			t.Error("explicit resources ask not claimed")
		}
	})
	for _, want := range []string{"From Astaxie's Build Web Application with Golang:", "Build a Simple Web Server (§3.2)", astaxie32URL} {
		if !strings.Contains(out, want) {
			t.Errorf("resources page missing %q:\n%s", want, out)
		}
	}
}
