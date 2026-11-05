package chat

import "testing"

// The phrasing layer's scoring view: the recipe's canonicals
// parse back as the serve-html family with the page (and the
// file, when named) as slots — and the concept rows' lines
// still score as their own families, never as the recipe.
func TestCanonicalSlotsServeHTML(t *testing.T) {
	ps, ok := CanonicalSlots("add a server that serves index.html to main.go")
	if !ok || ps.Family != "serve-html" || ps.Slots["html"] != "index.html" || ps.Slots["file"] != "main.go" {
		t.Errorf("canonical with file = %+v, %v", ps, ok)
	}
	ps, ok = CanonicalSlots("add a server that serves page.html")
	if !ok || ps.Family != "serve-html" || ps.Slots["html"] != "page.html" || len(ps.Slots) != 1 {
		t.Errorf("canonical without file = %+v, %v", ps, ok)
	}
	for line, family := range map[string]string{
		"add static files to main.go":                    "concept-static-files",
		"add an html template to main.go":                "concept-html-template",
		"add a webserver to main.go":                     "add-server",
		"add a server that serves index.html to main.go": "serve-html",
	} {
		ps, ok := CanonicalSlots(line)
		if !ok || ps.Family != family {
			t.Errorf("CanonicalSlots(%q) = %+v, %v — want family %s", line, ps, ok, family)
		}
	}
}

// The serve-a-page grammar, at the parse level: the brief's
// ask forms claim with the slots the flow needs, and the
// neighbours' sentences — the concept book's static-files and
// html-template lines above all — are never stolen.
func TestParseServeHTMLClaims(t *testing.T) {
	claims := []struct {
		line, html, target string
	}{
		{"serve an html file", "", ""},
		{"serve html file", "", ""},
		{"serve an html page", "", ""},
		{"serve index.html", "index.html", ""},
		{"serve the page.html", "page.html", ""},
		{"serve a home.html", "home.html", ""},
		{"serve index.html to main.go", "index.html", "main.go"},
		{"serve page.html in jim/main.go", "page.html", "jim/main.go"},
		{"serve an html file to jim", "", "jim"},
		{"serve index.html to the folder jim", "index.html", "jim"},
		{"build a server that serves index.html", "index.html", ""},
		{"build a server that serves index.html to main.go", "index.html", "main.go"},
		{"add a server that serves page.html to main.go", "page.html", "main.go"},
		{"add a webserver that serves index.html to jim/main.go", "index.html", "jim/main.go"},
		{"add the server serving home.html into app/main.go", "home.html", "app/main.go"},
		{"put an http server serving home.html into app/main.go", "home.html", "app/main.go"},
		{"make a server to serve index.html", "index.html", ""},
		{"get a web server that serves index.html to main.go", "index.html", "main.go"},
		{"create a server that serves index.html", "index.html", ""},
		// The serve word's synonym family (2026-11-05
		// widening): the beginner's verbs for what a server
		// does with a page.
		{"build a server that returns a HTML file", "", ""},
		{"build a server that returns index.html", "index.html", ""},
		{"build a server that returns index.html to jim/main.go", "index.html", "jim/main.go"},
		{"make a server that sends index.html", "index.html", ""},
		{"create a webserver that shows index.html", "index.html", ""},
		{"add a server that displays index.html", "index.html", ""},
		{"get a server that gives index.html", "index.html", ""},
		{"build a server that hosts index.html", "index.html", ""},
		{"build a server that serves up index.html", "index.html", ""},
		{"serve up index.html", "index.html", ""},
		{"serve up index.html to jim/main.go", "index.html", "jim/main.go"},
		{"make a server to return index.html", "index.html", ""},
	}
	for _, tc := range claims {
		got, ok := parseServeHTML(tc.line)
		if !ok {
			t.Errorf("parseServeHTML(%q) not claimed", tc.line)
			continue
		}
		if got.html != tc.html || got.target != tc.target || got.targetMissing {
			t.Errorf("parseServeHTML(%q) = %+v, want html=%q target=%q", tc.line, got, tc.html, tc.target)
		}
	}
}

// A started-but-unnamed target clause still claims: the flow
// asks in serve terms instead of the line falling through.
func TestParseServeHTMLTargetMissing(t *testing.T) {
	for _, line := range []string{
		"serve index.html to",
		"serve index.html to the",
		"serve index.html to the file",
		"serve index.html to folder",
		"add a server that serves index.html to the folder",
	} {
		got, ok := parseServeHTML(line)
		if !ok || !got.targetMissing {
			t.Errorf("parseServeHTML(%q) = %+v, %v — want claimed with targetMissing", line, got, ok)
		}
	}
}

// Not this claim's: the concept rows' lines, the add-server
// lines (no serves clause), the creations' lines, compounds,
// and prose. Each is owned elsewhere or by nobody.
func TestParseServeHTMLNonClaims(t *testing.T) {
	for _, line := range []string{
		// The concept book's rows keep their claims.
		"add static files to main.go",
		"serve static files to main.go",
		"add static file serving to main.go",
		"add an html template to main.go",
		"add html templates to main.go",
		"add a web template to main.go",
		"add routing to main.go",
		// Add-a-webserver's sentences: no page, no claim.
		"add a webserver to main.go",
		"add a web server to jim",
		"add a server to main.go",
		// The widened serve words still need a page after
		// them: the neighbours' shapes die at the page spec,
		// and a statement about a server is nobody's edit.
		"add a server that returns static files to main.go",
		"build a server that returns static files",
		"the server returns an html file",
		"my server sends the homepage",
		// The creations' sentences.
		"create a file jim/main.go with a webserver",
		"create a folder named jim",
		// A page with no serve intent.
		"add an html file to main.go",
		"open index.html",
		"what is an html file",
		// Incomplete or compound shapes.
		"serve",
		"serve html",
		"serve a server",
		"add a server that serves",
		"serve index.html and add a for loop to main.go",
		"serve index.html to main.go and then print",
		"serve index.html from main.go",
	} {
		if got, ok := parseServeHTML(line); ok {
			t.Errorf("parseServeHTML(%q) claimed %+v — not its sentence", line, got)
		}
	}
}
