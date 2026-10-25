package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The routing fix (on top of the d1b80d4 repo-URL claim):
//
//  1. "where to begin for https://github.com/wailsapp/wails" got
//     "salutations! what a nice way to say hello!" — the repo-URL
//     claim's intent words had no begin/start shapes, so the line
//     fell through every layer to the social brain. Begin/start
//     phrasings now claim the repo too, and the answer is the
//     repo's READING GUIDE, not the plain report.
//  2. A floor under the whole intent dispatch: a line naming a
//     repo-root GitHub URL never reaches the router/brains at all.
//  3. /profile existed only in the single-domain loop; in
//     `make chat` (unified) it fell through to social. Both loops
//     now call the shared printProfile (profile.go), claimed in
//     the unified loop's slash block.

// wailsCache plants a tiny wails-shaped repo in the codebase-maps
// cache under a fake HOME — the screenshot's repo — so the begin
// claim resolves with no network. Returns the cache root.
func wailsCache(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	root := filepath.Join(home, "workspace", "codebase-maps", "wailsapp-wails")
	write := func(rel, src string) {
		full := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("go.mod", "module github.com/wailsapp/wails\n\ngo 1.21\n")
	write("app.go", `package wails

// RunZebraQuux starts the desktop app the zebra way.
func RunZebraQuux() int { return BootZebra() }

// BootZebra prepares the runtime.
func BootZebra() int { return 7 }
`)
	return root
}

// Every begin/start shape about a repo URL must be claimed before
// the real classifier, answered with THAT repo's reading guide
// (the where-to-start path), and the repo becomes the project.
func TestRepoBeginShapesGetReadingGuide(t *testing.T) {
	withRealIntentModel(t)
	cacheRoot := seedRepoCache(t)
	lines := []string{
		"where to begin for https://github.com/ugorji/go",
		"where do i start with https://github.com/ugorji/go",
		"how do i start with https://github.com/ugorji/go",
		"getting started with https://github.com/ugorji/go",
		"where should i start in https://github.com/ugorji/go",
		"start with https://github.com/ugorji/go",
		"where to start for https://github.com/ugorji/go",
	}
	for _, line := range lines {
		withPickState(t)
		withEditsProject(t, decoyProject(t))
		var handled bool
		out := captureStdout(t, func() { handled = tryIntentDispatch(line, nil) })
		if !handled {
			t.Errorf("%q not handled, output:\n%s", line, out)
			continue
		}
		// The reading guide, for the URL's repo — analyzer output,
		// recognizable by its header and the repo's own symbols.
		if !strings.Contains(out, "WHERE TO START") || !strings.Contains(out, "FrobnicateZanzibar") {
			t.Errorf("%q not answered with the repo's reading guide, output:\n%s", line, out)
		}
		// The guide answer is not the full analyze report.
		if strings.Contains(out, "VISUALS:") {
			t.Errorf("%q got the full report, not the reading guide, output:\n%s", line, out)
		}
		if strings.Contains(out, "salutations") || strings.Contains(out, "DecoyMarker") {
			t.Errorf("%q answered by the wrong brain/project, output:\n%s", line, out)
		}
		if lastAnalyzeRoot != cacheRoot {
			t.Errorf("%q: current project root = %q, want the cached repo %q", line, lastAnalyzeRoot, cacheRoot)
		}
	}
}

// The screenshot line itself, verbatim, against the screenshot's
// repo: a begin claim answered with wails' own reading guide.
func TestRepoBeginScreenshotLine(t *testing.T) {
	withRealIntentModel(t)
	cacheRoot := wailsCache(t)
	withPickState(t)
	withEditsProject(t, decoyProject(t))
	var handled bool
	out := captureStdout(t, func() {
		handled = tryIntentDispatch("where to begin for https://github.com/wailsapp/wails", nil)
	})
	if !handled {
		t.Fatalf("screenshot line not handled, output:\n%s", out)
	}
	if !strings.Contains(out, "WHERE TO START") || !strings.Contains(out, "RunZebraQuux") {
		t.Errorf("not wails' reading guide, output:\n%s", out)
	}
	if strings.Contains(out, "salutations") {
		t.Errorf("the social greeting is back, output:\n%s", out)
	}
	if lastAnalyzeRoot != cacheRoot {
		t.Errorf("current project root = %q, want the wails cache %q", lastAnalyzeRoot, cacheRoot)
	}
}

// The begin detector itself: root URL + a begin phrase, nothing
// else. Question shapes stay with repoQuestionURL; begin words
// without a URL claim nothing here.
func TestRepoBeginDetection(t *testing.T) {
	accept := []string{
		"where to begin for https://github.com/ugorji/go",
		"where do i begin with https://github.com/ugorji/go?",
		"how do i start with github.com/ugorji/go",
		"getting started with https://github.com/ugorji/go",
		"start with https://github.com/ugorji/go",
		"where should i start: https://github.com/ugorji/go",
	}
	for _, line := range accept {
		url, ok := repoBeginURL(line)
		if !ok || url != "https://github.com/ugorji/go" {
			t.Errorf("repoBeginURL(%q) = (%q, %v), want the ugorji repo URL", line, url, ok)
		}
	}
	reject := []string{
		"where to begin",                            // no URL — the current project's owners keep it
		"where do i start",                          // same
		"getting started with go",                   // no URL
		"what does https://github.com/ugorji/go do", // a question, not a begin ask
		"where to begin for https://github.com/ugorji/go/issues/12", // deep link: the where flow's
		"start with the codec package",                              // no URL at all
	}
	for _, line := range reject {
		if url, ok := repoBeginURL(line); ok {
			t.Errorf("repoBeginURL(%q) = %q, want no claim", line, url)
		}
	}
}

// The floor: a line naming a repo-root URL that NO deterministic
// layer claims (no intent word, no begin phrase, pure nonsense
// around the URL) still never reaches the brains — the intent
// dispatch answers it with that repo's analysis.
func TestRepoURLFloorNeverFallsThrough(t *testing.T) {
	withRealIntentModel(t)
	seedRepoCache(t)
	lines := []string{
		"https://github.com/ugorji/go",
		"blorp zanzibar https://github.com/ugorji/go quux",
		"my code lives at https://github.com/ugorji/go btw",
	}
	for _, line := range lines {
		withPickState(t)
		withEditsProject(t, decoyProject(t))
		var handled bool
		out := captureStdout(t, func() { handled = tryIntentDispatch(line, nil) })
		if !handled {
			t.Errorf("%q fell through the floor, output:\n%s", line, out)
			continue
		}
		if !strings.Contains(out, "ugorji-go") {
			t.Errorf("%q not answered about the URL's repo, output:\n%s", line, out)
		}
		if strings.Contains(out, "salutations") || strings.Contains(out, "DecoyMarker") {
			t.Errorf("%q answered by the wrong brain/project, output:\n%s", line, out)
		}
	}
}

// The floor's own boundaries: lines another layer owns, and lines
// with no repo root, are not the floor's to claim.
func TestRepoURLFloorBoundaries(t *testing.T) {
	seedRepoCache(t)
	notFloor := []string{
		"clone https://github.com/ugorji/go in folder ugorji", // tryCloneProject owns it, later in the loops
		"go get github.com/google/uuid",                       // a command mentioning a repo path
		"where for https://github.com/ugorji/go/issues/12",    // issue URL: not a repo root
		"https://github.com/ugorji/go/pull/3",                 // PR URL: not a repo root
		"what does codec.decoderBincBytes.mapEnd do",          // no URL at all
		"hello there", // plain chat
	}
	for _, line := range notFloor {
		if tryRepoURLFloor(line) {
			t.Errorf("tryRepoURLFloor claimed %q, which belongs elsewhere", line)
		}
	}
}

// /profile in the unified loop: the loop claims the exact line and
// calls the shared renderer. The renderer must answer with the
// profile store's content — the same content the single-domain
// loop has always shown — never a brain greeting.
func TestProfileCommandRendersProfile(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // no cached style file in play
	profile := &UserProfile{
		FrequentCommands: []CommandStat{{Cmd: "make chat", Count: 7}},
		Projects: map[string]*ProjectMemory{
			"/home/john/gollemer": {Focus: "the routing fix", LastActive: time.Now()},
		},
		Patterns:   map[string]string{"after editing": "run the tests"},
		Directives: map[string]bool{"terse": true},
	}
	out := captureStdout(t, func() { printProfile(profile) })
	for _, want := range []string{
		"What I've learned about you",
		"7x  make chat",
		"the routing fix",
		"after editing → run the tests",
		"terse = true",
		"never uploaded",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("profile output missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "salutations") {
		t.Errorf("profile answered by the social brain:\n%s", out)
	}
	// The unified loop's claim is the exact line, handled before
	// routing; the intent layer itself must not also claim it (it
	// falls through there, which is exactly why the loop owns it).
	withPickState(t)
	if tryIntentDispatchInner("/profile", nil) {
		t.Error("intent layer claimed /profile — the loop's exact claim should be the only owner")
	}
}
