package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Journeys for the personal vocabulary (vocab.go): a name
// for a file or folder is taught once, explicitly, and from
// then on the deterministic sentences accept the name
// anywhere they accept the path — with the reading printed,
// never silently. The store is the user's own data in an
// isolated file; the neighbouring stores (taught, memory)
// are isolated alongside, because these journeys cross
// their claims.

func vocabFixture(t *testing.T, files map[string]string) (root, storePath string) {
	t.Helper()
	root = t.TempDir()
	for rel, content := range files {
		writeFile(t, root, rel, content)
	}
	clarifyPhrasing(t)
	withChatSession(t, root, true)
	t.Chdir(root)
	storePath = filepath.Join(t.TempDir(), ".gollemer_vocab.json")
	oldV := vocabStorePathOverride
	vocabStorePathOverride = storePath
	t.Cleanup(func() { vocabStorePathOverride = oldV })
	oldT := taughtStorePathOverride
	taughtStorePathOverride = filepath.Join(t.TempDir(), ".gollemer_taught.json")
	t.Cleanup(func() { taughtStorePathOverride = oldT })
	oldM := memoryStorePathOverride
	memoryStorePathOverride = filepath.Join(t.TempDir(), ".gollemer_memory.json")
	t.Cleanup(func() { memoryStorePathOverride = oldM })
	return root, storePath
}

func vocabRun(t *testing.T, lines ...string) string {
	t.Helper()
	profile := &UserProfile{}
	return captureStdout(t, func() {
		for _, line := range lines {
			claimChain(line, yes, profile)
		}
	})
}

const vocabMainGo = `package main

func main() {}
`

const vocabHandlersGo = `package main

import "net/http"

func Jake(w http.ResponseWriter, r *http.Request) {}
`

func vocabProject(t *testing.T) (string, string) {
	t.Helper()
	return vocabFixture(t, map[string]string{
		"go.mod":          "module example.com/vocab\n\ngo 1.21\n",
		"jim/main.go":     vocabMainGo,
		"jim/handlers.go": vocabHandlersGo,
		"api/main.go":     vocabMainGo,
	})
}

// --- grammar -----------------------------------------------------------------

func TestParseVocabTeach(t *testing.T) {
	cases := []struct {
		line       string
		this       bool
		designator string
		target     string
		phrase     string
	}{
		{"call jim/main.go the api file", false, "", "jim/main.go", "api file"},
		{"call this the api file", true, "", "", "api file"},
		{"call this file the api file", true, "file", "", "api file"},
		{"call this folder the web folder", true, "folder", "", "web folder"},
		{"call the file jim/db.go the db file", false, "file", "jim/db.go", "db file"},
		{"call the folder jim the web folder", false, "folder", "jim", "web folder"},
		{"call jim/ the web folder", false, "", "jim/", "web folder"},
		{"call store/store.go the store file.", false, "", "store/store.go", "store file"},
	}
	for _, c := range cases {
		ta, ok := parseVocabTeach(c.line)
		if !ok {
			t.Errorf("%q: not claimed", c.line)
			continue
		}
		if ta.this != c.this || ta.designator != c.designator || ta.target != c.target || normalizeKindPhrase(ta.phrase) != c.phrase {
			t.Errorf("%q: got %+v, want this=%v designator=%q target=%q phrase=%q", c.line, ta, c.this, c.designator, c.target, c.phrase)
		}
	}
	for _, line := range []string{
		"call me the api file",        // the memory store's shape
		"call the handler",            // no teach tail at all
		"call jim/main.go api file",   // no "the" before the name
		"call this",                   // nothing taught
		"call the api file the x",     // "the" not followed by a designator isn't a target
		"learn this as a signup form", // the teach seat's line, not this one's
		"what do you call things",     // the manage line, not a teach
	} {
		if ta, ok := parseVocabTeach(line); ok {
			t.Errorf("%q: claimed as %+v, want no claim", line, ta)
		}
	}
}

func TestParseFollowUpShapes(t *testing.T) {
	cases := []struct {
		line      string
		name      string
		target    string
		nameSet   bool
		targetSet bool
	}{
		{"another one", "", "", false, false},
		{"another one named Extra", "Extra", "", true, false},
		{"another one in jim/main.go", "", "jim/main.go", false, true},
		{"another one named Extra in jim/main.go", "Extra", "jim/main.go", true, true},
		{"another one called Extra to the api file", "Extra", "the api file", true, true},
		{"the same but for jim/other.go", "", "jim/other.go", false, true},
		{"the same thing in the api file", "", "the api file", false, true},
		{"same thing", "", "", false, false},
		{"the same", "", "", false, false},
	}
	for _, c := range cases {
		fu, ok := parseFollowUp(c.line)
		if !ok {
			t.Errorf("%q: not claimed", c.line)
			continue
		}
		if fu.name != c.name || fu.target != c.target || fu.nameSet != c.nameSet || fu.targetSet != c.targetSet {
			t.Errorf("%q: got %+v, want name=%q target=%q nameSet=%v targetSet=%v", c.line, fu, c.name, c.target, c.nameSet, c.targetSet)
		}
	}
	for _, line := range []string{
		"another one please",         // prose tail
		"another",                    // not the shape
		"one more time",              // not the shape
		"another one named",          // dangling name slot
		"another one in",             // dangling target slot
		"the same file",              // a reference, not a follow-up
		"add another one to main.go", // a sentence, not a follow-up
	} {
		if fu, ok := parseFollowUp(line); ok {
			t.Errorf("%q: claimed as %+v, want no claim", line, fu)
		}
	}
}

// --- the store ---------------------------------------------------------------

func TestVocabStoreRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), ".gollemer_vocab.json")
	old := vocabStorePathOverride
	vocabStorePathOverride = path
	t.Cleanup(func() { vocabStorePathOverride = old })

	d, err := loadVocab()
	if err != nil || len(d.Aliases) != 0 {
		t.Fatalf("empty store: %v %+v", err, d)
	}
	d.Aliases = append(d.Aliases, vocabAlias{Phrase: "api file", Display: "api file", Target: "jim/main.go", Root: "/proj"})
	if err := saveVocab(d); err != nil {
		t.Fatalf("save: %v", err)
	}
	st, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	if st.Mode().Perm() != 0o600 {
		t.Errorf("store mode = %v, want 0600", st.Mode().Perm())
	}
	back, err := loadVocab()
	if err != nil || len(back.Aliases) != 1 || back.Aliases[0].Target != "jim/main.go" {
		t.Fatalf("round trip: %v %+v", err, back)
	}
	if back.findAlias("/other", "api file") != nil {
		t.Error("an alias leaked into another project's lookup")
	}
	if back.findAlias("/proj", "api file") == nil {
		t.Error("the alias is missing from its own project's lookup")
	}
}

// --- teach + use journeys ------------------------------------------------------

func TestVocabTeachAndUseJourney(t *testing.T) {
	root, storePath := vocabProject(t)
	out := vocabRun(t, "call jim/main.go the api file")
	for _, want := range []string{
		"Learned your name for it:",
		`"the api file" — jim/main.go in this project.`,
		"separate from your memory and your taught pieces",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("teach echo missing %q:\n%s", want, out)
		}
	}
	raw, err := os.ReadFile(storePath)
	if err != nil || !strings.Contains(string(raw), `"api file"`) || !strings.Contains(string(raw), `"jim/main.go"`) {
		t.Errorf("store does not hold the alias: %v\n%s", err, raw)
	}
	// The name now works in a concept sentence, with the
	// reading printed before the plan.
	out = vocabRun(t, "add a print line in main to the api file")
	if !strings.Contains(out, `Reading "the api file" as jim/main.go — your name for it.`) {
		t.Errorf("use printed no reading:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "fmt.Println") {
		t.Errorf("the print line did not land in jim/main.go:\n%s", got)
	}
	// And in another family's sentence (add-a-webserver).
	out = vocabRun(t, "call api/main.go the server file", "add a webserver to the server file")
	if !strings.Contains(out, `Reading "the server file" as api/main.go — your name for it.`) {
		t.Errorf("webserver use printed no reading:\n%s", out)
	}
	if got := readBack(t, root, "api/main.go"); !strings.Contains(got, "ListenAndServe") {
		t.Errorf("the webserver did not land in api/main.go:\n%s", got)
	}
}

func TestVocabTeachThisAndFolder(t *testing.T) {
	root, _ := vocabProject(t)
	// Land a file, then name "this" — the file that landed.
	vocabRun(t, "create a file jim/db.go")
	out := vocabRun(t, "call this the db file")
	if !strings.Contains(out, `"the db file" — jim/db.go in this project.`) {
		t.Errorf("'this' did not read as the landed file:\n%s", out)
	}
	// Land a folder, then name "this" — the folder.
	vocabRun(t, "create a folder web")
	out = vocabRun(t, "call this the web folder")
	if !strings.Contains(out, `"the web folder" — web/ in this project.`) {
		t.Errorf("'this' did not read as the landed folder:\n%s", out)
	}
	// The folder name works as a folder target: a declaration
	// named into it lands as its own file there.
	out = vocabRun(t, "add a struct named WebProbe to the web folder")
	if !strings.Contains(out, `Reading "the web folder" as web — your name for it.`) {
		t.Errorf("folder use printed no reading:\n%s", out)
	}
	if got := readBack(t, root, "web/webprobe.go"); !strings.Contains(got, "type WebProbe struct") {
		t.Errorf("the struct did not land in the named folder:\n%s", got)
	}
	// A cold "this" refuses in words (fresh session state).
	withChatSession(t, root, true)
	out = vocabRun(t, "call this the api file")
	if !strings.Contains(out, `I don't have a "this" in this session yet`) {
		t.Errorf("cold 'this' was not refused in words:\n%s", out)
	}
}

func TestVocabTeachRefusals(t *testing.T) {
	_, _ = vocabProject(t)
	vocabRun(t, "call jim/main.go the api file")
	cases := []struct{ line, want string }{
		// Redefinition: refused with the current target named.
		{"call api/main.go the api file", `"api file" is already your name for jim/main.go in this project`},
		// A reserved reference phrase.
		{"call jim/main.go the other file", "Not naming it that"},
		// A book name (the families keep their words).
		{"call jim/main.go the print line", "Not naming it that"},
		// A path, not a name.
		{"call jim/main.go the jim/db.go", "is a path, not a name"},
		// An ambiguous bare target: neither file nor folder
		// can be told, and the refusal spells both forms.
		{"call newthing the new file", `say "call the file newthing the <name>" or "call the folder newthing the <name>"`},
	}
	for _, c := range cases {
		out := vocabRun(t, c.line)
		if !strings.Contains(out, c.want) {
			t.Errorf("%q: refusal missing %q:\n%s", c.line, c.want, out)
		}
	}
}

func TestVocabOnDiskNameRefused(t *testing.T) {
	root, _ := vocabFixture(t, map[string]string{
		"go.mod":      "module example.com/vocab2\n\ngo 1.21\n",
		"jim/main.go": vocabMainGo,
		"api file":    "a plain file whose name is a phrase\n",
	})
	out := vocabRun(t, "call jim/main.go the api file")
	if !strings.Contains(out, `there's already a file named "api file" in this project`) {
		t.Errorf("on-disk name was not refused:\n%s", out)
	}
	_ = root
}

func TestVocabLiteralWinsAtUse(t *testing.T) {
	root, _ := vocabProject(t)
	vocabRun(t, "call jim/main.go the api file")
	// A real file with the phrase's own name appears later:
	// the literal reading wins and the alias stops firing —
	// no reading line, and nothing lands in jim/main.go.
	writeFile(t, root, "api file", "a plain file whose name is a phrase\n")
	out := vocabRun(t, "add a print line in main to the api file")
	if strings.Contains(out, "your name for it") {
		t.Errorf("the alias fired over a real file of the same name:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); strings.Contains(got, "fmt.Println") {
		t.Errorf("the print line landed through the shadowed alias:\n%s", got)
	}
}

// --- manage --------------------------------------------------------------------

func TestVocabManageJourney(t *testing.T) {
	_, storePath := vocabProject(t)
	vocabRun(t,
		"call jim/main.go the api file",
		"call jim/handlers.go the handlers file",
	)
	out := vocabRun(t, "what do you call things")
	for _, want := range []string{
		"Your names for things in this project:",
		`"the api file" — jim/main.go (a file)`,
		`"the handlers file" — jim/handlers.go (a file)`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("list missing %q:\n%s", want, out)
		}
	}
	out = vocabRun(t, "forget the api file")
	if !strings.Contains(out, `Forgotten: "the api file" — your name for jim/main.go in this project.`) {
		t.Errorf("forget echo wrong:\n%s", out)
	}
	d, err := loadVocab()
	if err != nil || len(d.Aliases) != 1 || d.Aliases[0].Phrase != "handlers file" {
		t.Errorf("store after forget: %v %+v", err, d)
	}
	out = vocabRun(t, "forget what i call things")
	if !strings.Contains(out, "Forgotten: your 1 name for things in this project") {
		t.Errorf("forget-all echo wrong:\n%s", out)
	}
	d, _ = loadVocab()
	if len(d.Aliases) != 0 {
		t.Errorf("store after forget-all: %+v", d)
	}
	_ = storePath
	// After the forget, the name no longer resolves.
	out = vocabRun(t, "add a print line in main to the handlers file")
	if strings.Contains(out, "your name for it") {
		t.Errorf("a forgotten name still resolved:\n%s", out)
	}
}

func TestVocabForgetFallsThroughToTaught(t *testing.T) {
	_, _ = vocabProject(t)
	// A taught kind exists; "forget the <kind>" is the teach
	// seat's line, and the vocabulary (seated earlier) must
	// NOT claim it — it only forgets names that are its own.
	vocabRun(t,
		"learn this as a signup form from Jake in jim/handlers.go",
		"call jim/main.go the api file",
	)
	out := vocabRun(t, "forget the signup form")
	if !strings.Contains(out, `Forgotten: your "signup form"`) {
		t.Errorf("the taught forget was not reached:\n%s", out)
	}
	if d, _ := loadTaughtKinds(); len(d.Kinds) != 0 {
		t.Errorf("the taught kind survived its forget: %+v", d)
	}
	// The alias is untouched by the kind's forget.
	if d, _ := loadVocab(); len(d.Aliases) != 1 {
		t.Errorf("the alias was touched by the kind's forget: %+v", d)
	}
}

func TestVocabCrossProjectIsolation(t *testing.T) {
	rootA, storePath := vocabProject(t)
	vocabRun(t, "call jim/main.go the api file")
	// A second project, same store: the name is inert there.
	rootB := t.TempDir()
	writeFile(t, rootB, "go.mod", "module example.com/vocabB\n\ngo 1.21\n")
	writeFile(t, rootB, "jim/main.go", vocabMainGo)
	oldV := vocabStorePathOverride
	vocabStorePathOverride = storePath
	t.Cleanup(func() { vocabStorePathOverride = oldV })
	withChatSession(t, rootB, true)
	t.Chdir(rootB)
	out := vocabRun(t, "add a print line in main to the api file")
	if strings.Contains(out, "your name for it") {
		t.Errorf("project A's name fired in project B:\n%s", out)
	}
	if got := readBack(t, rootB, "jim/main.go"); strings.Contains(got, "fmt.Println") {
		t.Errorf("project B's file was edited through project A's name:\n%s", got)
	}
	out = vocabRun(t, "what do you call things")
	if !strings.Contains(out, "the names I know live in your other projects") {
		t.Errorf("project B's list did not own its emptiness:\n%s", out)
	}
	_ = rootA
}

// --- byte identity when nothing is taught --------------------------------------

func TestVocabNoNamesNoChange(t *testing.T) {
	root, _ := vocabProject(t)
	// No names taught: an ordinary target line is untouched
	// (no reading printed) and lands exactly as today.
	out := vocabRun(t, "add a print line in main to jim/main.go")
	if strings.Contains(out, "your name for it") || strings.Contains(out, "Reading \"") {
		t.Errorf("a reading printed with no names taught:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); !strings.Contains(got, "fmt.Println") {
		t.Errorf("the ordinary line did not land:\n%s", got)
	}
	// The rewrite pass itself is a no-op on ordinary lines,
	// and on follow-up-shaped lines even when a name exists.
	if got := rewriteSpokenTargets("add a print line in main to jim/main.go"); got != "add a print line in main to jim/main.go" {
		t.Errorf("rewrite changed an ordinary line: %q", got)
	}
}
