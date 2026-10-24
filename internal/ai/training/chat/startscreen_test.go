package chat

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/golangast/gollemer/pkg/termui"
)

// startscreen_test.go — the welcome-back start screen: personal
// starters and the single continuity line, derived read-only from
// the profile, the Friend notebook, and the Watcher log. Every
// test runs against fixture stores under a fake HOME.

// startFixtureHome gives the test an empty fake HOME and pristine
// Friend state, with styling off (plain-mode rendering).
func startFixtureHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	friendResetForTest()
	t.Cleanup(friendResetForTest)
	oldT := termui.Enabled
	termui.Enabled = false
	t.Cleanup(func() { termui.Enabled = oldT })
	return home
}

func startFixtureProfile(t *testing.T, p *UserProfile) {
	t.Helper()
	if p.Projects == nil {
		p.Projects = map[string]*ProjectMemory{}
	}
	if p.Patterns == nil {
		p.Patterns = map[string]string{}
	}
	if p.Directives == nil {
		p.Directives = map[string]bool{}
	}
	if err := p.Save(); err != nil {
		t.Fatal(err)
	}
}

func startFixtureFriend(t *testing.T, mutate func(m *friendMemory)) {
	t.Helper()
	m := friendLoad()
	mutate(m)
	friendSave()
}

func startFixtureWatch(t *testing.T, home string, lines ...string) {
	t.Helper()
	body := strings.Join(lines, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(home, ".gollemer_watch.jsonl"),
		[]byte(body), 0600); err != nil {
		t.Fatal(err)
	}
}

// starterLines extracts the panel's ▸ lines from a rendered screen.
func starterLines(out string) []string {
	var ls []string
	for _, l := range strings.Split(out, "\n") {
		if strings.Contains(l, "▸") {
			ls = append(ls, l)
		}
	}
	return ls
}

func TestStartScreenMostRunCommandStarter(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	startFixtureProfile(t, &UserProfile{FrequentCommands: []CommandStat{
		{Cmd: "go vet ./...", Count: 2, LastUsed: time.Now()},
		{Cmd: "make test", Count: 5, LastUsed: time.Now().Add(-time.Hour)},
		{Cmd: "make once", Count: 1, LastUsed: time.Now()},
	}})
	out := startScreenText(root)
	if !strings.Contains(out, "run make test — your most-run command") {
		t.Errorf("most-run starter missing:\n%s", out)
	}
	// The count-1 command is not a habit and must not appear.
	if strings.Contains(out, "make once") {
		t.Errorf("one-off command leaked into the starters:\n%s", out)
	}
	if n := len(starterLines(out)); n != 3 {
		t.Errorf("want exactly 3 starter lines, got %d:\n%s", n, out)
	}
	// Generic fill follows in today's order behind the personal
	// starter: start, then ask (help yields — the cap is 3 total).
	ls := starterLines(out)
	if !strings.Contains(ls[1], "start — a six-stop") ||
		!strings.Contains(ls[2], "ask in plain words") {
		t.Errorf("generic fill order wrong:\n%s", out)
	}
}

func TestStartScreenMostRunCommandStarterNonMake(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	startFixtureProfile(t, &UserProfile{FrequentCommands: []CommandStat{
		{Cmd: "go test ./...", Count: 3, LastUsed: time.Now()},
	}})
	out := startScreenText(root)
	if !strings.Contains(out, "your most-run command — go test ./...") {
		t.Errorf("non-make command starter wrong:\n%s", out)
	}
}

func TestStartScreenGuidedResumeStarter(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	startFixtureFriend(t, func(m *friendMemory) {
		m.Guided = map[string]int{root: 2}
	})
	out := startScreenText(root)
	if !strings.Contains(out, "start — continue the tour · you left off at stop 3 of 6") {
		t.Errorf("guided resume starter missing:\n%s", out)
	}
	// The resume line replaces the generic tour line, not adds to it.
	if strings.Contains(out, "a six-stop guided tour of this project") {
		t.Errorf("generic start line should yield to the resume line:\n%s", out)
	}
	if n := len(starterLines(out)); n != 3 {
		t.Errorf("want exactly 3 starter lines, got %d:\n%s", n, out)
	}
}

func TestStartScreenTeachResumeSuppressesOffer(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	startFixtureFriend(t, func(m *friendMemory) { m.Taught = 2 })
	if !teachOfferDue() {
		t.Fatal("fixture should leave the teach offer due, or the test proves nothing")
	}
	out := startScreenText(root)
	if !strings.Contains(out, "teach me — pick up the training tour · lesson 3 of 6") {
		t.Errorf("teach resume starter missing:\n%s", out)
	}
	// The ✦ offer line must not repeat what the starter already says.
	if strings.Contains(out, "new here?") {
		t.Errorf("teach offer duplicated the resume starter:\n%s", out)
	}
}

func TestStartScreenContinuityWatchCount(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	startFixtureFriend(t, func(m *friendMemory) {
		m.WatchRead = "2026-10-20T09:00:00Z"
	})
	startFixtureWatch(t, home,
		`{"t":"2026-10-19T09:00:00Z","kind":"sig-changed","func":"Old.One","root":"`+root+`"}`, // before cursor
		`{"t":"2026-10-22T10:00:00Z","kind":"sig-changed","func":"Counts.Add","root":"`+root+`"}`,
		`{"t":"2026-10-22T11:00:00Z","kind":"behavior-changed","func":"Counts.Add","root":"`+root+`"}`,
		`{"t":"2026-10-23T09:00:00Z","kind":"new-func","func":"Timer.Start","root":"`+root+`"}`,
		`{"t":"2026-10-23T10:00:00Z","kind":"sig-changed","func":"Other.Pkg","root":"/somewhere/else"}`,
	)
	out := startScreenText(root)
	// Two distinct functions (Counts.Add twice counts once), the
	// pre-cursor event and the other project's event excluded.
	if !strings.Contains(out, "2 functions changed here since Oct 20.") {
		t.Errorf("watch continuity line wrong:\n%s", out)
	}
	// Rendering must not advance the Friend's cursor.
	if got := friendLoad().WatchRead; got != "2026-10-20T09:00:00Z" {
		t.Errorf("render consumed the watch cursor: %q", got)
	}
}

func TestStartScreenContinuityFocus(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	startFixtureProfile(t, &UserProfile{Projects: map[string]*ProjectMemory{
		root: {Name: "proj", Focus: "the composer and piece schemas"},
	}})
	out := startScreenText(root)
	if !strings.Contains(out, "last time: the composer and piece schemas") {
		t.Errorf("focus continuity line missing:\n%s", out)
	}
}

func TestStartScreenContinuityFriendReturn(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	yesterday := time.Now().AddDate(0, 0, -1).Format("2006-01-02")
	startFixtureFriend(t, func(m *friendMemory) {
		m.LastDay = yesterday
		m.LastTopic = "Freeze"
	})
	out := startScreenText(root)
	if !strings.Contains(out, "welcome back — last time you were deep in Freeze.") {
		t.Errorf("friend return line missing:\n%s", out)
	}
}

func TestStartScreenContinuityFriendReturnNotDue(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	// Same day: the Friend's return note is not due.
	startFixtureFriend(t, func(m *friendMemory) {
		m.LastDay = friendToday()
		m.LastTopic = "Freeze"
	})
	if out := startScreenText(root); strings.Contains(out, "welcome back") {
		t.Errorf("return line fired on the same day:\n%s", out)
	}
	// A suggestion already queued: the one-pending rule stands the
	// screen down — the Friend will speak its own note itself.
	friendResetForTest()
	startFixtureFriend(t, func(m *friendMemory) {
		m.LastDay = time.Now().AddDate(0, 0, -1).Format("2006-01-02")
		m.LastTopic = "Freeze"
	})
	friendPending = "already queued"
	if out := startScreenText(root); strings.Contains(out, "welcome back") {
		t.Errorf("return line fired while a suggestion was pending:\n%s", out)
	}
	if friendPending != "already queued" {
		t.Error("render consumed or replaced the pending suggestion")
	}
}

func TestStartScreenContinuityPriority(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	yesterday := time.Now().AddDate(0, 0, -1).Format("2006-01-02")
	startFixtureProfile(t, &UserProfile{Projects: map[string]*ProjectMemory{
		root: {Name: "proj", Focus: "the composer"},
	}})
	startFixtureFriend(t, func(m *friendMemory) {
		m.WatchRead = "2026-10-20T09:00:00Z"
		m.LastDay = yesterday
		m.LastTopic = "Freeze"
	})
	startFixtureWatch(t, home,
		`{"t":"2026-10-22T10:00:00Z","kind":"sig-changed","func":"Counts.Add","root":"`+root+`"}`,
	)
	out := startScreenText(root)
	// All three sources populated: exactly one line, the Watcher's.
	if !strings.Contains(out, "1 function changed here since Oct 20.") {
		t.Errorf("watch line should win:\n%s", out)
	}
	if strings.Contains(out, "last time:") || strings.Contains(out, "welcome back") {
		t.Errorf("lower-priority continuity lines leaked:\n%s", out)
	}

	// No watch events: the focus beats the Friend's return note.
	home2 := startFixtureHome(t)
	root2 := filepath.Join(home2, "proj")
	startFixtureProfile(t, &UserProfile{Projects: map[string]*ProjectMemory{
		root2: {Name: "proj", Focus: "the composer"},
	}})
	startFixtureFriend(t, func(m *friendMemory) {
		m.LastDay = yesterday
		m.LastTopic = "Freeze"
	})
	out2 := startScreenText(root2)
	if !strings.Contains(out2, "last time: the composer") {
		t.Errorf("focus line should win over the return note:\n%s", out2)
	}
	if strings.Contains(out2, "welcome back") {
		t.Errorf("return note leaked under the focus line:\n%s", out2)
	}
}

func TestStartScreenEmptyStoresGeneric(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	out := startScreenText(root)
	// The three generic starters, in order.
	i1 := strings.Index(out, "start — a six-stop guided tour of this project")
	i2 := strings.Index(out, "ask in plain words")
	i3 := strings.Index(out, "help — everything you can say")
	if !(0 <= i1 && i1 < i2 && i2 < i3) {
		t.Errorf("generic starters missing or out of order:\n%s", out)
	}
	// Nothing personal may be claimed without records.
	for _, banned := range []string{"welcome back", "last time:", "most-run", "left off", "changed here"} {
		if strings.Contains(out, banned) {
			t.Errorf("empty stores produced personal claim %q:\n%s", banned, out)
		}
	}
	// The capability lines are present, in order, beneath the panel.
	prev := strings.Index(out, "╯")
	if prev < 0 {
		t.Fatalf("panel border missing:\n%s", out)
	}
	for _, line := range capabilityLines {
		i := strings.Index(out, line)
		if i < 0 {
			t.Fatalf("capability line missing %q:\n%s", line, out)
		}
		if i < prev {
			t.Errorf("capability line out of order (before panel end or previous line): %q", line)
		}
		prev = i
	}
}

func TestStartScreenCorruptStoresGeneric(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	for name, body := range map[string]string{
		".gollemer_profile.json": "not json {{{",
		".gollemer_friend.json":  "}{ garbage",
		".gollemer_watch.jsonl":  "not a json line\n",
	} {
		if err := os.WriteFile(filepath.Join(home, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	friendResetForTest() // drop any cached (valid) state from fixtures
	out := startScreenText(root)
	for _, banned := range []string{"welcome back", "last time:", "most-run", "left off", "changed here"} {
		if strings.Contains(out, banned) {
			t.Errorf("corrupt stores produced personal claim %q:\n%s", banned, out)
		}
	}
	if !strings.Contains(out, "start — a six-stop guided tour of this project") {
		t.Errorf("corrupt stores broke the generic screen:\n%s", out)
	}
}

func TestStartScreenRenderIsReadOnly(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	startFixtureProfile(t, &UserProfile{
		FrequentCommands: []CommandStat{{Cmd: "make test", Count: 4, LastUsed: time.Now()}},
		Projects: map[string]*ProjectMemory{
			root: {Name: "proj", Focus: "the composer"},
		},
	})
	startFixtureFriend(t, func(m *friendMemory) {
		m.Guided = map[string]int{root: 1}
		m.Taught = 1
		m.WatchRead = "2026-10-20T09:00:00Z"
	})
	startFixtureWatch(t, home,
		`{"t":"2026-10-22T10:00:00Z","kind":"sig-changed","func":"Counts.Add","root":"`+root+`"}`,
	)
	hash := func(name string) string {
		data, err := os.ReadFile(filepath.Join(home, name))
		if err != nil {
			return "absent"
		}
		return hex.EncodeToString(sha256Sum(data))
	}
	names := []string{".gollemer_profile.json", ".gollemer_friend.json", ".gollemer_watch.jsonl"}
	before := map[string]string{}
	for _, n := range names {
		before[n] = hash(n)
	}
	out := startScreenText(root)
	_ = startScreenText(root)
	// Sanity: the fixtures actually engaged personalization.
	if !strings.Contains(out, "most-run") || !strings.Contains(out, "changed here") {
		t.Fatalf("fixtures did not personalize the screen:\n%s", out)
	}
	for _, n := range names {
		if after := hash(n); after != before[n] {
			t.Errorf("rendering mutated %s (before %s, after %s)", n, before[n], after)
		}
	}
}

func sha256Sum(b []byte) []byte {
	h := sha256.Sum256(b)
	return h[:]
}

// (stdout capture reuses captureStdout from makeexplain_test.go.)

func TestStartScreenPrintOrderAndPacing(t *testing.T) {
	home := startFixtureHome(t)
	root := filepath.Join(home, "proj")
	oldC, oldOn, oldInter, oldSleep := colorEnabled, paceOn, paceInteractive, paceSleep
	defer func() { colorEnabled, paceOn, paceInteractive, paceSleep = oldC, oldOn, oldInter, oldSleep }()
	colorEnabled = false
	want := startScreenText(root) + "\n"

	// Piped (non-interactive): everything at once, complete, in
	// order — identical bytes to the composed screen text.
	paceOn, paceInteractive = true, false
	if got := captureStdout(t, func() { printStartScreen(root) }); got != want {
		t.Errorf("plain print differs from composed screen:\ngot:\n%s\nwant:\n%s", got, want)
	}

	// Terminal: the capability block paces (pauses counted, not
	// slept), and the bytes are still complete and in order.
	pauses := 0
	paceSleep = func(time.Duration) { pauses++ }
	paceInteractive = true
	if got := captureStdout(t, func() { printStartScreen(root) }); got != want {
		t.Errorf("paced print differs from composed screen:\ngot:\n%s\nwant:\n%s", got, want)
	}
	if pauses == 0 {
		t.Error("capability block did not pace on a terminal")
	}
	// Pacing off: no pauses, same bytes.
	paceOn = false
	pauses = 0
	if got := captureStdout(t, func() { printStartScreen(root) }); got != want {
		t.Errorf("pace-off print differs:\ngot:\n%s\nwant:\n%s", got, want)
	}
	if pauses != 0 {
		t.Error("pacing off still paused")
	}
}

func TestSameProjectRoot(t *testing.T) {
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if !sameProjectRoot("/a/b", "/a/b/") || sameProjectRoot("/a/b", "/a/c") {
		t.Error("absolute path comparison wrong")
	}
	if !sameProjectRoot(".", wd) {
		t.Error("relative spelling of the cwd should match its absolute form")
	}
	if sameProjectRoot("", "/a") {
		t.Error("empty root must match nothing")
	}
}
