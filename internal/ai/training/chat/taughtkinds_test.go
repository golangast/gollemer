package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The teach seat: grammar tables first, then the flows end to
// end through the claim chain — teach from an example in the
// fixture project, use the kind in another package, manage
// the store — with the store pointed at a throwaway file
// OUTSIDE the fixture project (it is the user's store, never
// a project's).

const taughtJimMain = `package main

import (
	"fmt"
	"net/http"
)

func Jake(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	fmt.Fprintf(w, "thanks, %s", r.FormValue("email"))
}

func Fact(n int) int {
	if n <= 1 {
		return 1
	}
	return n * Fact(n-1)
}

func Helper() string { return "jim" }

func main() {
	fmt.Println(Fact(3))
}
`

const taughtStoreFile = `package store

type Node struct {
	Val  int
	Next *Node
}

func (n *Node) Size() int {
	if n == nil {
		return 0
	}
	if n.Next == nil {
		return 1
	}
	return 1 + n.Next.Size()
}

type Store struct {
	Store string
	Next  *Store
}

const Answer = 42

func Helper() string { return "store" }
`

// taughtFixture builds the fixture project, starts an
// analyzed session in it, and points the taught store at a
// fresh file in a DIFFERENT temp dir (the store is the
// user's, outside every project tree).
func taughtFixture(t *testing.T) (dir, storePath string) {
	t.Helper()
	dir = t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/taught\n\ngo 1.21\n")
	writeFile(t, dir, "jim/main.go", taughtJimMain)
	writeFile(t, dir, "store/store.go", taughtStoreFile)
	writeFile(t, dir, "site/site.go", "package site\n")
	withChatSession(t, dir, true)
	storePath = filepath.Join(t.TempDir(), ".gollemer_taught.json")
	old := taughtStorePathOverride
	taughtStorePathOverride = storePath
	t.Cleanup(func() { taughtStorePathOverride = old })
	return dir, storePath
}

func taughtRun(t *testing.T, line string) (bool, string) {
	t.Helper()
	var handled bool
	out := captureStdout(t, func() { handled = claimChain(line, yes, &UserProfile{}) })
	return handled, out
}

func taughtStoreKinds(t *testing.T, storePath string) []taughtKind {
	t.Helper()
	d, err := loadTaughtKinds()
	if err != nil {
		t.Fatalf("store load: %v", err)
	}
	return d.Kinds
}

// --- grammar -----------------------------------------------------------------

func TestParseTeachAsk(t *testing.T) {
	cases := []struct {
		line, kind, name, file string
	}{
		{"learn this as a signup form from Jake in jim/main.go", "signup form", "Jake", "jim/main.go"},
		{"learn a sidebar page from Home in site/main.go", "sidebar page", "Home", "site/main.go"},
		{"learn an api client from Client", "api client", "Client", ""},
		{"teach you a repository from Store in store/store.go", "repository", "Store", "store/store.go"},
		{"teach a repository from Store", "repository", "Store", ""},
		{"learn this as a cart from Cart in the file store/cart.go", "cart", "Cart", "store/cart.go"},
	}
	for _, c := range cases {
		ta, ok := parseTeachAsk(c.line)
		if !ok {
			t.Errorf("%q: not claimed", c.line)
			continue
		}
		if normalizeKindPhrase(ta.kind) != c.kind || ta.name != c.name || ta.file != c.file {
			t.Errorf("%q: got {kind:%q name:%q file:%q}, want {%q %q %q}", c.line, ta.kind, ta.name, ta.file, c.kind, c.name, c.file)
		}
	}
	for _, line := range []string{
		"learn mode",
		"learn",
		"learn https://github.com/example/repo",
		"teach me",
		"learn a form",
		"learn from Jake in jim/main.go",
		"add a signup form named Signup to site/site.go",
		"what have I taught you",
		"remember: I prefer tabs",
	} {
		if ta, ok := parseTeachAsk(line); ok {
			t.Errorf("%q: claimed as %+v, want no claim", line, ta)
		}
	}
}

func TestParseTaughtUse(t *testing.T) {
	u, ok := parseTaughtUse("add a signup form named Signup to site/site.go")
	if !ok || u.kind != "signup form" || u.name != "Signup" || u.target != "site/site.go" || !u.named {
		t.Errorf("named form: got %+v ok=%v", u, ok)
	}
	u, ok = parseTaughtUse("create a cart named Cart in store")
	if !ok || u.kind != "cart" || u.name != "Cart" || u.target != "store" {
		t.Errorf("folder target: got %+v ok=%v", u, ok)
	}
	u, ok = parseTaughtUse("add a signup form to site/site.go")
	if !ok || u.kind != "signup form" || u.named || u.target != "site/site.go" {
		t.Errorf("nameless form: got %+v ok=%v", u, ok)
	}
	// The 2026-11-05 widening: put|build verbs, and "called"
	// as the name slot's second spelling — the kind must not
	// swallow the name.
	for _, line := range []string{
		"put a signup form named Signup to site/site.go",
		"build a signup form named Signup to site/site.go",
		"add a signup form called Signup to site/site.go",
	} {
		u, ok = parseTaughtUse(line)
		if !ok || u.kind != "signup form" || u.name != "Signup" || u.target != "site/site.go" || !u.named {
			t.Errorf("%q: got %+v ok=%v, want kind/name/target read", line, u, ok)
		}
	}
	for _, line := range []string{
		"add",
		"hello there",
		"connect the Jake handler to the webserver",
		"what have I taught you",
	} {
		if u, ok := parseTaughtUse(line); ok {
			t.Errorf("%q: claimed as %+v, want no claim", line, u)
		}
	}
}

// --- teach -------------------------------------------------------------------

func TestTeachSummaryAndStore(t *testing.T) {
	_, storePath := taughtFixture(t)
	handled, out := taughtRun(t, "learn this as a signup form from Jake in jim/main.go")
	if !handled {
		t.Fatalf("teach line fell through:\n%s", out)
	}
	for _, want := range []string{
		"Here's what I learned from Jake in jim/main.go:",
		`The kind: "signup form"`,
		"The example: the function Jake, at jim/main.go:",
		"What varies: the name — Jake becomes the new name",
		"What stays fixed: everything else — the code stays exactly as in Jake",
		"(fmt, net/http) travel with it",
		"Saved as yours:",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("teach output missing %q:\n%s", want, out)
		}
	}
	if _, err := os.Stat(storePath); err != nil {
		t.Fatalf("store not written at %s: %v", storePath, err)
	}
	kinds := taughtStoreKinds(t, storePath)
	if len(kinds) != 1 {
		t.Fatalf("store kinds = %d, want 1", len(kinds))
	}
	k := kinds[0]
	if k.Kind != "signup form" || k.Example != "Jake" || k.DeclKind != "function" || k.File != "jim/main.go" {
		t.Errorf("stored kind = %+v", k)
	}
	if !strings.Contains(k.Template, "func {{name}}(w http.ResponseWriter, r *http.Request)") {
		t.Errorf("template missing name slot:\n%s", k.Template)
	}
	if strings.Contains(k.Template, "Jake") {
		t.Errorf("template still names the example:\n%s", k.Template)
	}
	if len(k.Imports) != 2 {
		t.Errorf("stored imports = %v, want fmt and net/http", k.Imports)
	}
}

func TestTeachRecursiveAndTypeSlots(t *testing.T) {
	taughtFixture(t)
	if handled, out := taughtRun(t, "learn a factorial from Fact in jim/main.go"); !handled {
		t.Fatalf("teach Fact fell through:\n%s", out)
	}
	if handled, out := taughtRun(t, "learn a linked node from Node in store/store.go"); !handled {
		t.Fatalf("teach Node fell through:\n%s", out)
	}
	kinds := taughtStoreKinds(t, "")
	byKind := map[string]taughtKind{}
	for _, k := range kinds {
		byKind[k.Kind] = k
	}
	fact := byKind["factorial"].Template
	if !strings.Contains(fact, "return n * {{name}}(n-1)") {
		t.Errorf("recursive call not slotted:\n%s", fact)
	}
	node := byKind["linked node"].Template
	if !strings.Contains(node, "Next *{{name}}") || !strings.Contains(node, "type {{name}} struct") {
		t.Errorf("type self-reference not slotted:\n%s", node)
	}
}

// A struct field that shares the type's name is a field, not
// a self-reference: its name stays, its sibling field's type
// reference renames.
func TestTeachFieldSharingTypeName(t *testing.T) {
	taughtFixture(t)
	if handled, out := taughtRun(t, "learn a store box from Store in store/store.go"); !handled {
		t.Fatalf("teach Store fell through:\n%s", out)
	}
	kinds := taughtStoreKinds(t, "")
	if len(kinds) != 1 {
		t.Fatalf("kinds = %d", len(kinds))
	}
	tmpl := kinds[0].Template
	if !strings.Contains(tmpl, "Store string") {
		t.Errorf("field name was renamed:\n%s", tmpl)
	}
	if !strings.Contains(tmpl, "Next") || !strings.Contains(tmpl, "*{{name}}") {
		t.Errorf("self-reference not slotted:\n%s", tmpl)
	}
}

func TestTeachMethodAndValueRefused(t *testing.T) {
	taughtFixture(t)
	handled, out := taughtRun(t, "learn a sizer from Size in store/store.go")
	if !handled || !strings.Contains(out, "is a method") || !strings.Contains(out, "Node") {
		t.Errorf("method teach: handled=%v\n%s", handled, out)
	}
	handled, out = taughtRun(t, "learn a magic number from Answer in store/store.go")
	if !handled || !strings.Contains(out, "package-level value") {
		t.Errorf("value teach: handled=%v\n%s", handled, out)
	}
	if kinds := taughtStoreKinds(t, ""); len(kinds) != 0 {
		t.Errorf("refusals saved kinds: %v", kinds)
	}
}

func TestTeachNotFoundAndSeveral(t *testing.T) {
	taughtFixture(t)
	handled, out := taughtRun(t, "learn a widget from Frobnicate in jim/main.go")
	if !handled || !strings.Contains(out, "couldn't find a declaration named \"Frobnicate\" in jim/main.go") {
		t.Errorf("not-found in file: handled=%v\n%s", handled, out)
	}
	handled, out = taughtRun(t, "learn a helper from Helper")
	if !handled || !strings.Contains(out, "more than one declaration is named Helper") ||
		!strings.Contains(out, "jim/main.go") || !strings.Contains(out, "store/store.go") {
		t.Errorf("several matches: handled=%v\n%s", handled, out)
	}
	handled, out = taughtRun(t, "learn a widget from Frobnicate")
	if !handled || !strings.Contains(out, "couldn't find a declaration named \"Frobnicate\" in this project") {
		t.Errorf("not-found in project: handled=%v\n%s", handled, out)
	}
}

func TestTeachCollisionsRefused(t *testing.T) {
	taughtFixture(t)
	// A book name: taught kinds sit beside the book, never over it.
	handled, out := taughtRun(t, "learn a form handler from Jake in jim/main.go")
	if !handled || !strings.Contains(out, "already one of the concept book's pieces") {
		t.Errorf("book collision: handled=%v\n%s", handled, out)
	}
	// A page-family name.
	handled, out = taughtRun(t, "learn a homepage from Jake in jim/main.go")
	if !handled || !strings.Contains(out, "already one of the page family's pieces") {
		t.Errorf("page collision: handled=%v\n%s", handled, out)
	}
	// A proposer kind word.
	handled, out = taughtRun(t, "learn a struct from Node in store/store.go")
	if !handled || !strings.Contains(out, "Not learning that") {
		t.Errorf("kind-word collision: handled=%v\n%s", handled, out)
	}
	// A duplicate: teach once, then again under the same name.
	if handled, out := taughtRun(t, "learn a signup form from Jake in jim/main.go"); !handled {
		t.Fatalf("first teach fell through:\n%s", out)
	}
	handled, out = taughtRun(t, "learn a signup form from Fact in jim/main.go")
	if !handled || !strings.Contains(out, "you've already taught me a \"signup form\"") ||
		!strings.Contains(out, "forget the signup form") {
		t.Errorf("duplicate kind: handled=%v\n%s", handled, out)
	}
	if kinds := taughtStoreKinds(t, ""); len(kinds) != 1 || kinds[0].Example != "Jake" {
		t.Errorf("store after collisions = %+v", kinds)
	}
}

// --- use ---------------------------------------------------------------------

func TestUseLandsRenamedWithProvenance(t *testing.T) {
	dir, _ := taughtFixture(t)
	if handled, out := taughtRun(t, "learn this as a signup form from Jake in jim/main.go"); !handled {
		t.Fatalf("teach fell through:\n%s", out)
	}
	handled, out := taughtRun(t, "add a signup form named Signup to site/site.go")
	if !handled {
		t.Fatalf("use line fell through:\n%s", out)
	}
	if !strings.Contains(out, "From your example Jake in jim/main.go, taught") {
		t.Errorf("plan lacks the provenance mark:\n%s", out)
	}
	if !strings.Contains(out, "built the project and checked") {
		t.Errorf("spine did not build/verify:\n%s", out)
	}
	got := readBack(t, dir, "site/site.go")
	for _, want := range []string{
		"package site",
		"func Signup(w http.ResponseWriter, r *http.Request)",
		`"net/http"`, `"fmt"`,
		`thanks, %s`, // the example's own body — not the book's form handler
	} {
		if !strings.Contains(got, want) {
			t.Errorf("site/site.go missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "Jake") {
		t.Errorf("landed piece still names the example:\n%s", got)
	}

	// The create-a-file path: a new file in an existing folder.
	handled, out = taughtRun(t, "add a signup form named SignupTwo to site/forms.go")
	if !handled || !strings.Contains(out, "Done —") {
		t.Fatalf("create-file use: handled=%v\n%s", handled, out)
	}
	got = readBack(t, dir, "site/forms.go")
	if !strings.HasPrefix(got, "package site\n") || !strings.Contains(got, "func SignupTwo(") {
		t.Errorf("created file wrong:\n%s", got)
	}

	// The widened spellings (2026-11-05) land the TAUGHT
	// piece — provenance on the plan, the example's own body
	// in the file — and never the book's form handler, which
	// is what the phrasing layer used to answer these lines
	// with when the taught grammar declined them.
	for _, tc := range []struct{ line, file, fn string }{
		{"put a signup form named SignupThree to site/forms3.go", "site/forms3.go", "SignupThree"},
		{"add a signup form called SignupFour to site/forms4.go", "site/forms4.go", "SignupFour"},
	} {
		handled, out = taughtRun(t, tc.line)
		if !handled || !strings.Contains(out, "From your example Jake in jim/main.go, taught") {
			t.Errorf("%q: handled=%v, plan lacks taught provenance:\n%s", tc.line, handled, out)
			continue
		}
		got = readBack(t, dir, tc.file)
		if !strings.Contains(got, "func "+tc.fn+"(") || !strings.Contains(got, "thanks, %s") {
			t.Errorf("%q: landed file is not the taught template:\n%s", tc.line, got)
		}
	}
}

func TestUseRecursiveRenamesCalls(t *testing.T) {
	dir, _ := taughtFixture(t)
	if handled, out := taughtRun(t, "learn a factorial from Fact in jim/main.go"); !handled {
		t.Fatalf("teach fell through:\n%s", out)
	}
	handled, out := taughtRun(t, "add a factorial named Fib to site/site.go")
	if !handled || !strings.Contains(out, "Done —") {
		t.Fatalf("use: handled=%v\n%s", handled, out)
	}
	got := readBack(t, dir, "site/site.go")
	if !strings.Contains(got, "return n * Fib(n-1)") {
		t.Errorf("recursive call not renamed:\n%s", got)
	}
}

func TestUseGuidanceAndUnknownKind(t *testing.T) {
	taughtFixture(t)
	if handled, out := taughtRun(t, "learn a signup form from Jake in jim/main.go"); !handled {
		t.Fatalf("teach fell through:\n%s", out)
	}
	handled, out := taughtRun(t, "add a signup form to site/site.go")
	if !handled || !strings.Contains(out, "a signup form needs a name") {
		t.Errorf("unnamed use: handled=%v\n%s", handled, out)
	}
	handled, out = taughtRun(t, "add a signup form named Signup")
	if !handled || !strings.Contains(out, "doesn't name the file or folder") {
		t.Errorf("targetless use: handled=%v\n%s", handled, out)
	}
	// An unknown multiword kind gets the honest answer that
	// names the way in.
	handled, out = taughtRun(t, "add a flux capacitor named Flux to jim/main.go")
	if !handled || !strings.Contains(out, "I don't know a \"flux capacitor\"") ||
		!strings.Contains(out, "learn a flux capacitor from <Name> in <file>") {
		t.Errorf("unknown kind: handled=%v\n%s", handled, out)
	}
	// Shapes this claim must never answer: the proposer's kind
	// words and the book's names are not "unknown kinds" here.
	for _, line := range []string{
		"add a helper function named Help to jim/main.go",
		"add a struct named Thing to jim/main.go",
		"add a form handler named Pete to jim/main.go",
	} {
		var h bool
		captureStdout(t, func() { h = runTaughtKinds(line, yes) })
		if h {
			t.Errorf("%q: the teach seat claimed a neighbour's line", line)
		}
	}
}

// --- manage --------------------------------------------------------------------

func TestManageListForgetAndAftermath(t *testing.T) {
	taughtFixture(t)
	handled, out := taughtRun(t, "what have I taught you")
	if !handled || !strings.Contains(out, "You haven't taught me any pieces yet") {
		t.Errorf("empty list: handled=%v\n%s", handled, out)
	}
	if handled, out := taughtRun(t, "learn a signup form from Jake in jim/main.go"); !handled {
		t.Fatalf("teach fell through:\n%s", out)
	}
	if handled, out := taughtRun(t, "learn a linked node from Node in store/store.go"); !handled {
		t.Fatalf("teach fell through:\n%s", out)
	}
	handled, out = taughtRun(t, "what have I taught you")
	if !handled || !strings.Contains(out, `"signup form" — from your example Jake (function) in jim/main.go`) ||
		!strings.Contains(out, `"linked node" — from your example Node (type) in store/store.go`) {
		t.Errorf("list: handled=%v\n%s", handled, out)
	}
	// The list question's other two spellings (2026-11-05).
	for _, line := range []string{"what did you learn from me", "show me what i taught you"} {
		handled, out = taughtRun(t, line)
		if !handled || !strings.Contains(out, `"signup form"`) {
			t.Errorf("%q: handled=%v\n%s", line, handled, out)
		}
	}
	// Forget a kind that was never taught: the answer lists
	// what there is and forgets nothing.
	handled, out = taughtRun(t, "forget the flux capacitor")
	if !handled || !strings.Contains(out, "You haven't taught me a \"flux capacitor\"") ||
		!strings.Contains(out, "Nothing was forgotten") {
		t.Errorf("forget unknown: handled=%v\n%s", handled, out)
	}
	if kinds := taughtStoreKinds(t, ""); len(kinds) != 2 {
		t.Fatalf("kinds after no-op forget = %d", len(kinds))
	}
	handled, out = taughtRun(t, "forget the signup form")
	if !handled || !strings.Contains(out, "Forgotten: your \"signup form\" — from your example Jake in jim/main.go") {
		t.Errorf("forget one: handled=%v\n%s", handled, out)
	}
	// After forgetting, the kind is unknown again — honestly.
	handled, out = taughtRun(t, "add a signup form named Signup to site/site.go")
	if !handled || !strings.Contains(out, "I don't know a \"signup form\"") {
		t.Errorf("use after forget: handled=%v\n%s", handled, out)
	}
	// Forget the rest at once.
	handled, out = taughtRun(t, "forget what I taught you")
	if !handled || !strings.Contains(out, "Forgotten: everything you taught me — 1 piece:") {
		t.Errorf("forget all: handled=%v\n%s", handled, out)
	}
	if kinds := taughtStoreKinds(t, ""); len(kinds) != 0 {
		t.Errorf("kinds after forget-all = %d", len(kinds))
	}
}

// --- non-theft and no implicit capture ----------------------------------------

func TestOrdinaryLinesTeachNothing(t *testing.T) {
	_, storePath := taughtFixture(t)
	for _, line := range []string{
		"hello",
		"what does Jake do?",
		"add a print line to jim/main.go",
		"learn mode",
		"Jake is a great handler",
	} {
		var h bool
		captureStdout(t, func() { h = runTaughtKinds(line, yes) })
		if h {
			t.Errorf("%q: the teach seat claimed an ordinary line", line)
		}
	}
	if _, err := os.Stat(storePath); !os.IsNotExist(err) {
		t.Errorf("ordinary lines wrote the store (stat err = %v)", err)
	}
}

// The book keeps its own sentences with a taught kind one
// word away: "add a form handler named Pete" lands the BOOK
// piece (its body, its provenance), and the store stays out
// of it.
func TestBookFormHandlerNotStolen(t *testing.T) {
	dir, _ := taughtFixture(t)
	if handled, out := taughtRun(t, "learn a signup form from Jake in jim/main.go"); !handled {
		t.Fatalf("teach fell through:\n%s", out)
	}
	handled, out := taughtRun(t, "add a form handler named Pete to jim/main.go")
	if !handled {
		t.Fatalf("book line fell through:\n%s", out)
	}
	if !strings.Contains(out, "from the concept book") {
		t.Errorf("book line was not the book's:\n%s", out)
	}
	got := readBack(t, dir, "jim/main.go")
	if !strings.Contains(got, "func Pete(") {
		t.Errorf("book handler did not land:\n%s", got)
	}
	if strings.Contains(got, "thanks, %s") && strings.Count(got, "thanks, %s") > 1 {
		t.Errorf("the taught template leaked into the book's landing:\n%s", got)
	}
}

// The store's home: the default path is the user's home file,
// never a path inside a project.
func TestTaughtStoreLocation(t *testing.T) {
	old := taughtStorePathOverride
	taughtStorePathOverride = ""
	t.Cleanup(func() { taughtStorePathOverride = old })
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		t.Skip("no home dir to check against")
	}
	want := filepath.Join(home, ".gollemer_taught.json")
	if got := taughtStorePath(); got != want {
		t.Errorf("default store path = %q, want %q", got, want)
	}
}
