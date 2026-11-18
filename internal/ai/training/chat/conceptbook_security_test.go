package chat

import (
	"strings"
	"testing"
)

// The security sweep (2026-11-01): the server / API / HTML
// security set learned the five-sources way. What the book's
// grammar can honestly assemble became rows — CORS, the JWT
// pair, the ownership check, the nonce and CSRF middleware,
// security headers, the flagged cookie, server timeouts, the
// body limit, the rate limiter. What stays judgment —
// password storage (bcrypt is x/crypto; the book writes
// stdlib only), TLS certificates — seats on the resources
// shelf, pinned below. SQL injection and XSS are duplicates
// (the parameterized database rows and html/template's
// escaping own them), not new rows.

// The new rows claim their own lines, in the family's
// natural forms.
func TestParseConceptClaimsSecurity(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct{ line, id string }{
		{"add a cors middleware named Allow to api.go", "cors"},
		{"add a cors named Allow to api.go", "cors"},
		{"add a cross origin middleware named Allow to api.go", "cors"},
		{"add a jwt middleware named Gate to api.go", "jwt-verify"},
		{"add a jwt auth named Gate to api.go", "jwt-verify"},
		{"add a jwt verification named Gate to api.go", "jwt-verify"},
		{"add a csp nonce middleware named Nonce to web.go", "csp-nonce"},
		{"add a nonce middleware named Nonce to web.go", "csp-nonce"},
		{"add a csrf protection named Guard to web.go", "csrf"},
		{"add a csrf middleware named Guard to web.go", "csrf"},
		{"add a security headers named Walls to web.go", "security-headers"},
		{"add a security headers middleware named Walls to web.go", "security-headers"},
		{"add a rate limit middleware named Throttle to api.go", "rate-limit"},
		{"add a rate limiter named Throttle to api.go", "rate-limit"},
		{"add a jwt in main to jim/main.go", "jwt-create"},
		{"add a json web token in main to jim/main.go", "jwt-create"},
		{"add a jwt token to jim/main.go", "jwt-create"},
		{"add an idor check in Login to web.go", "idor-check"},
		{"add an ownership check in Login to web.go", "idor-check"},
		{"add a secure cookie in Login to web.go", "secure-cookie"},
		{"add cookie flags in Login to web.go", "secure-cookie"},
		{"add a server timeouts to jim/main.go", "server-timeouts"},
		{"add server timeouts in main to jim/main.go", "server-timeouts"},
		{"add a body limit in Login to web.go", "body-limit"},
		{"add a request body limit in Upload to web.go", "body-limit"},
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
	}
}

// The new names must not widen the claim into a neighbour's
// lines, and the sweep's exclusions still fall through.
func TestParseConceptSecurityNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	neighbours := []struct{ line, id string }{
		{"add a middleware named Logger to jim/main.go", "middleware"},
		{"add a timing middleware named Timer to jim/main.go", "middleware-timing"},
		{"add a session handler named Session to jim/main.go", "session-handler"},
		{"add a timeout to jim/main.go", "timeout"},
		{"add a hash to jim/main.go", "sha256"},
		{"add form values in Login to web.go", "form-values"},
		{"add an html template in main to jim/main.go", "html-template"},
	}
	for _, c := range neighbours {
		req, ok := parseConceptFrom(c.line, book)
		if !ok {
			t.Errorf("parseConcept(%q) not claimed, want neighbour row %q", c.line, c.id)
			continue
		}
		if req.entry.ID != c.id {
			t.Errorf("parseConcept(%q) = row %q, want neighbour %q", c.line, req.entry.ID, c.id)
		}
	}
	for _, line := range []string{
		"add a tls certificate to jim/main.go",      // certificates are the user's; TLS seats on the shelf
		"add a password hashing to jim/main.go",     // bcrypt is x/crypto — shelf, never a row
		"add a bcrypt hash to jim/main.go",          // the algorithm's name does not make it stdlib
		"add an sql injection check to jim/main.go", // a duplicate: the parameterized rows own it
		"add a xss filter to jim/main.go",           // a duplicate: html/template escapes
	} {
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed row %q — that line is not the composer's", line, req.entry.ID)
		}
	}
}

// securityProject seeds web.go with a handler whose writer
// and request parameters are deliberately NOT w and r: the
// {response} and {request} slots must bind the real names.
func securityProject(t *testing.T) (string, *UserProfile) {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/security\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	writeFile(t, dir, "web.go", "package main\n\nimport \"net/http\"\n\nfunc Login(out http.ResponseWriter, req *http.Request) {\n\t// TODO: the login handler\n}\n\nfunc Helper() {\n\tprintln(\"helping\")\n}\n")
	withChatSession(t, dir, true)
	return dir, &UserProfile{}
}

// The three handler-shaped rows land in a real handler with
// BOTH slots bound from its signature — the landed code
// addresses the writer and the request by the names the
// user actually gave them.
func TestSecurityHandlerRowsEndToEnd(t *testing.T) {
	cases := []struct {
		line string
		want []string
	}{
		{"add a secure cookie in Login to web.go", []string{"http.SetCookie(out, cookie)", "Secure: true", "HttpOnly: true", "http.SameSiteLaxMode"}},
		{"add an idor check in Login to web.go", []string{"http.Error(out,", "http.StatusForbidden", "callerID", "recordOwnerID"}},
		{"add a body limit in Login to web.go", []string{"req.Body = http.MaxBytesReader(out, req.Body, limit)"}},
	}
	for _, c := range cases {
		dir, profile := securityProject(t)
		var claimed bool
		out := captureStdout(t, func() {
			claimed = claimChain(c.line, yes, profile)
		})
		if !claimed {
			t.Fatalf("%q fell through the chain:\n%s", c.line, out)
		}
		if !strings.Contains(out, "built the project and checked") {
			t.Errorf("%q: spine build/verify not reported:\n%s", c.line, out)
		}
		got := readBack(t, dir, "web.go")
		for _, w := range c.want {
			if !strings.Contains(got, w) {
				t.Errorf("%q: landed piece missing %q:\n%s", c.line, w, got)
			}
		}
		if strings.Contains(got, "{response}") || strings.Contains(got, "{request}") {
			t.Errorf("%q: an unbound slot survived into the file:\n%s", c.line, got)
		}
	}
}

// Outside a handler the rows refuse by name: aimed at a
// plain function, the refusal says that function receives
// no response writer and points at the handler that does.
func TestSecurityHandlerRowRefusals(t *testing.T) {
	dir, profile := securityProject(t)
	before := readBack(t, dir, "web.go")

	var claimed bool
	out := captureStdout(t, func() {
		claimed = claimChain("add a secure cookie in Helper to web.go", yes, profile)
	})
	if !claimed {
		t.Fatalf("secure cookie into a plain function fell through:\n%s", out)
	}
	if !strings.Contains(out, "doesn't receive an http.ResponseWriter") || !strings.Contains(out, "Login") {
		t.Errorf("refusal does not name the problem and the way out:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimed = claimChain("add a body limit in Missing to web.go", yes, profile)
	})
	if !claimed || !strings.Contains(out, "no function named \"Missing\"") {
		t.Errorf("missing-handler refusal wrong (claimed=%v):\n%s", claimed, out)
	}
	if got := readBack(t, dir, "web.go"); got != before {
		t.Errorf("refused edits changed the file:\n%s", got)
	}
}

// The JWT pair in one session: mint in main, guard as a
// middleware — and the pair shares one secret variable name
// across both templates (JWT_SECRET), never a literal.
func TestSecurityJWTPairEndToEnd(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/securityjwt\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", "package main\n\nfunc main() {\n\tprintln(\"root\")\n}\n")
	writeFile(t, dir, "jim/main.go", "package main\n\nfunc main() {\n\tprintln(\"ok\")\n}\n")
	withChatSession(t, dir, true)
	profile := &UserProfile{}

	var claimed bool
	out := captureStdout(t, func() {
		claimed = claimChain("add a jwt in main to jim/main.go", yes, profile)
	})
	if !claimed || !strings.Contains(out, "built the project and checked") {
		t.Fatalf("jwt create fell through or unverified:\n%s", out)
	}
	out = captureStdout(t, func() {
		claimed = claimChain("add a jwt middleware named Gate to jim/main.go", yes, profile)
	})
	if !claimed || !strings.Contains(out, "built the project and checked") {
		t.Fatalf("jwt middleware fell through or unverified:\n%s", out)
	}
	got := readBack(t, dir, "jim/main.go")
	for _, want := range []string{
		`os.Getenv("JWT_SECRET")`,
		"hmac.New(sha256.New",
		"base64.RawURLEncoding",
		"func Gate(next http.HandlerFunc) http.HandlerFunc",
		"http.StatusUnauthorized",
		"claims[\"exp\"]",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("jim/main.go missing %q after the pair landed:\n%s", want, got)
		}
	}
	if n := strings.Count(got, `os.Getenv("JWT_SECRET")`); n != 2 {
		t.Errorf("JWT_SECRET read %d times, want 2 (one per side of the pair):\n%s", n, got)
	}
}

// The phrasing layer hears the new families in held-out
// dress — and the naming discipline holds: a bare "token"
// (a word no security family trains) claims nothing, and a
// neighbour's own line still routes to the neighbour.
func TestPhrasingSecurityFamiliesEndToEnd(t *testing.T) {
	withPhrasingModel(t)
	routeCases := []struct {
		line string
		want []string
	}{
		{"could you stick a jwt in main to main.go",
			[]string{"add a jwt in main to main.go"}},
		{"kindly place an ownership check in Login to web.go",
			[]string{"add an idor check in Login to web.go"}},
		{"place a secure cookie in Login to web.go",
			[]string{"add a secure cookie in Login to web.go"}},
		{"insert a body limit in Upload to web.go",
			[]string{"add a body limit in Upload to web.go"}},
		{"place server timeouts in main to main.go",
			[]string{"add a server timeouts in main to main.go"}},
		{"could you stick a cors middleware named AllowSites in api.go",
			[]string{"add a cors middleware named AllowSites to api.go"}},
		{"get me a rate limiter named Throttle in api.go",
			[]string{"add a rate limit middleware named Throttle to api.go"}},
		{"place a nonce middleware named Nonce in web.go",
			[]string{"add a csp nonce middleware named Nonce to web.go"}},
		{"create a security headers middleware with the name Walls in web.go",
			[]string{"add a security headers named Walls to web.go"}},
		{"place a jwt auth named Gate in api.go",
			[]string{"add a jwt middleware named Gate to api.go"}},
		{"could you add a csrf protection named Guard to web.go",
			[]string{"add a csrf protection named Guard to web.go"}},
	}
	for _, c := range routeCases {
		got, ok := PhrasingRoute(c.line)
		if !ok {
			t.Errorf("PhrasingRoute(%q) not claimed, want %q", c.line, c.want)
			continue
		}
		if strings.Join(got, " ## ") != strings.Join(c.want, " ## ") {
			t.Errorf("PhrasingRoute(%q) = %q, want %q", c.line, got, c.want)
		}
	}
	// The bare word "token" belongs to no family: an ask
	// that names nothing more specific is not the phrasing
	// layer's to claim.
	if got, ok := PhrasingRoute("add a token in main to main.go"); ok {
		t.Errorf("PhrasingRoute(bare token) = %q, want no claim", got)
	}
}

// The security shelf: each topic's goal earns its entry,
// under its own source's label, at the verified address.
func TestSecurityShelf(t *testing.T) {
	want := []struct {
		goal  string
		label string
		url   string
	}{
		{"jwt", "JSON Web Token Cheat Sheet for Java", owaspJWTURL},
		{"json web token", "JSON Web Token Cheat Sheet for Java", owaspJWTURL},
		{"idor", "Insecure Direct Object Reference Prevention Cheat Sheet", owaspIDORURL},
		{"password hashing", "Password Storage Cheat Sheet", owaspPasswordsURL},
		{"storing passwords", "Password Storage Cheat Sheet", owaspPasswordsURL},
		{"security headers", "HTTP Headers Cheat Sheet", owaspHeadersURL},
		{"cors", "Cross-Origin Resource Sharing (CORS)", mdnCORSURL},
		{"tls", "net/http: Server.ListenAndServeTLS", listenServeTLSURL},
		{"cross origin protection", "net/http: CrossOriginProtection (Go 1.25+)", crossOriginURL},
	}
	for _, w := range want {
		found := false
		for _, l := range referenceLabels(w.goal) {
			if l == w.label+"|"+w.url {
				found = true
			}
		}
		if !found {
			t.Errorf("matchReferences(%q) lacks %q: %v", w.goal, w.label, referenceLabels(w.goal))
		}
	}
}
