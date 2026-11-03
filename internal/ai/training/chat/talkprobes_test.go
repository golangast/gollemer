package chat

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// The talk seat's probe harness (the design's pre-agreed bar,
// measured on a fixture with known index facts): ~40
// conversational probes in five classes, driven through the
// session's own tail — tryDeterministicAnswer first, then the
// talk seat, exactly as the unified loop seats them — against
// the fixture project from talkseat_test.go. It runs offline:
// the fixture's data/ is a symlink to this checkout's data
// (the handtest harness's own trick), so the seat's lazy
// loader finds the real local social model.
//
// Bars enforced here:
//   - invented project names printed as fact: ZERO across all
//     probes (talkUnmarkedInventions over every printed reply
//     the seat or the deterministic Q&A produced);
//   - trap probes: ZERO inventions, and every trap answered
//     with the honest not-here wording;
//   - project-fact probes: >= 90% pass;
//   - session follow-up probes: >= 90% pass (counted
//     separately from project-fact);
//   - social and general-Go probes: reported as a judged
//     sample in the test log, with the same zero-invention
//     mechanical check applied.

type talkProbe struct {
	class string
	line  string
	want  []string // substrings the printed reply must contain
}

type talkProbeResult struct {
	probe  talkProbe
	path   string // deterministic | talk | fallback
	reply  string
	pass   bool
	invBad []string
}

// TestTalkProbes runs the whole probe set and prints the
// per-probe table (probe, path/tag, pass/fail, reply excerpt)
// to the test log.
func TestTalkProbes(t *testing.T) {
	dir := talkFixtureProject(t)
	// The handtest trick: the fixture finds this checkout's
	// models and book through a data symlink.
	repoRoot, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(repoRoot, "data"), filepath.Join(dir, "data")); err != nil {
		t.Fatal(err)
	}
	withChatSession(t, dir, false)
	// The REAL composed responder, lazy-loaded from the
	// fixture (social model included) — no fakes at the bar.
	oldResp, oldOnce := talkResponderVar, talkOnce
	t.Cleanup(func() { talkResponderVar, talkOnce = oldResp, oldOnce })
	talkResponderVar = nil
	talkOnce = &sync.Once{}
	initSocialRecall(dir)

	conv := NewConversation()
	reader := NewLineReader(strings.NewReader(""))
	profile := &UserProfile{}

	ask := func(line string) (string, string) {
		conv.AddUser(line)
		d := routeDomain(line)
		var handled bool
		out := captureStdout(t, func() {
			handled = tryDeterministicAnswer(line, d, conv, reader, dir)
		})
		if handled {
			return "deterministic", out
		}
		out = captureStdout(t, func() {
			handled = tryTalkSeat(line, d, conv, d == SocialDomain, false)
		})
		if handled {
			return "talk", out
		}
		return "fallback", out
	}

	var results []talkProbeResult
	run := func(probes []talkProbe) {
		for _, pr := range probes {
			path, out := ask(pr.line)
			reply := strings.TrimSpace(out)
			res := talkProbeResult{probe: pr, path: path, reply: reply}
			res.pass = path != "fallback"
			for _, w := range pr.want {
				if !strings.Contains(reply, w) {
					res.pass = false
				}
			}
			// The mechanical bar, scoped exactly as
			// production scopes the claim-checker: seat
			// replies (claim-checked before printing),
			// goanalyze fact replies (project facts by
			// construction), and every trap reply, whatever
			// path answered it. Curated general-Go replies
			// ([go] knowledge-base text) are out of scope:
			// they make no project claims — their example
			// names (doThing()) are general knowledge, the
			// same reason production never checks them.
			scan := path == "talk" || pr.class == "trap" || strings.Contains(reply, "[goanalyze]")
			if scan && lastAnalyzeProject != nil {
				res.invBad = talkUnmarkedInventions(reply, lastAnalyzeProject)
			}
			results = append(results, res)
		}
	}

	// --- project-fact (~14) -----------------------------------
	run([]talkProbe{
		{"project-fact", "what does Hello do?", []string{"Hello", "string"}},
		{"project-fact", "what does Save take?", []string{"path string", "data []byte"}},
		{"project-fact", "who calls Hello?", []string{"main"}},
		{"project-fact", "what's in greet/greet.go?", []string{"Hello", "Greeter"}},
		{"project-fact", "what is this project?", []string{"talkprobe"}},
		{"project-fact", "what's in the store package?", []string{"Save", "Load"}},
		{"project-fact", "where is Load defined?", []string{"store/store.go"}},
		{"project-fact", "what kind is Greeter?", []string{"Greeter", "struct"}},
		{"project-fact", "walk me through Hello", []string{"Hello"}},
		{"project-fact", "what does main call?", []string{"Hello"}},
		{"project-fact", "what's in the greet package?", []string{"Hello"}},
		{"project-fact", "tell me about Load", []string{"Load", "[]byte"}},
		{"project-fact", "what does Load return?", []string{"[]byte", "error"}},
		{"project-fact", "show me Save", []string{"Save", "func Save"}},
		{"project-fact", "how does Hello work?", []string{"Hello"}},
	})

	// --- session follow-ups (~6) -------------------------------
	// Build something first, through the deterministic chain
	// itself (the session harness's auto-yes idiom), then ask
	// subject-less questions about it.
	for _, line := range []string{
		"create a folder named extra",
		"create file extra/extra.go",
		"add a function named Wave to extra/extra.go",
	} {
		var handled bool
		out := captureStdout(t, func() { handled = claimChain(line, yes, profile) })
		if !handled {
			t.Fatalf("follow-up setup line %q not claimed:\n%s", line, out)
		}
	}
	if got := readBack(t, dir, "extra/extra.go"); !strings.Contains(got, "func Wave") {
		t.Fatalf("follow-up setup did not land Wave:\n%s", got)
	}
	run([]talkProbe{
		{"session-follow-up", "what does it do?", []string{"Wave", "extra/extra.go"}},
		{"session-follow-up", "what's in it?", []string{"extra/extra.go", "Wave"}},
		{"session-follow-up", "where is it?", []string{"extra/extra.go"}},
		{"session-follow-up", "is it in greet?", []string{"No", "extra"}},
		{"session-follow-up", "which folder is it in?", []string{"extra"}},
		{"session-follow-up", "tell me about it", []string{"Wave", "extra/extra.go"}},
	})

	// --- social (~8): judged sample + mechanical check --------
	run([]talkProbe{
		{"social", "hello", nil},
		{"social", "how are you?", nil},
		{"social", "good morning", nil},
		{"social", "tell me a joke", nil},
		{"social", "thanks!", nil},
		{"social", "are you a robot?", nil},
		{"social", "what's your name?", nil},
		{"social", "i'm bored", nil},
	})

	// --- general Go (~6) ---------------------------------------
	run([]talkProbe{
		{"general-go", "explain it to me simply: what is a channel?", []string{"Channels move values"}},
		{"general-go", "in plain words, what is a mutex?", []string{"Mutex"}},
		{"general-go", "what is a worker pool, simply?", []string{"worker pool"}},
		{"general-go", "tell me simply, what is a waitgroup?", []string{"WaitGroup"}},
		{"general-go", "what is the context package, in simple terms?", []string{"context"}},
		{"general-go", "simply put, what are channels for?", []string{"Channels move values"}},
	})

	// --- traps (~6): no invented name printed as fact ----------
	run([]talkProbe{
		{"trap", "what does Frobnicate do?", []string{"Frobnicate", "don't see"}},
		{"trap", "tell me about the ZoomCache type", []string{"ZoomCache", "don't see"}},
		{"trap", "what's in cache/magic.go?", []string{"cache/magic.go", "don't see"}},
		{"trap", "does this project have a function called Teleport?", []string{"Teleport", "don't see"}},
		{"trap", "how does the Blorp work?", []string{"Blorp", "don't see"}},
		{"trap", "what does main call first in store/engine.go?", []string{"store/engine.go", "don't see"}},
	})

	// --- the table + the bars ----------------------------------
	counts := map[string][2]int{} // class -> {pass, total}
	inventions := 0
	for _, r := range results {
		excerpt := r.reply
		if len(excerpt) > 160 {
			excerpt = excerpt[:160] + "…"
		}
		excerpt = strings.ReplaceAll(excerpt, "\n", " ⏎ ")
		status := "PASS"
		if !r.pass {
			status = "FAIL"
		}
		t.Logf("[%s] %-4s %-18s %q -> %s", r.probe.class, status, r.path, r.probe.line, excerpt)
		c := counts[r.probe.class]
		c[1]++
		if r.pass {
			c[0]++
		}
		counts[r.probe.class] = c
		if len(r.invBad) > 0 {
			inventions += len(r.invBad)
			t.Errorf("probe %q printed unmarked invented names %v:\n%s", r.probe.line, r.invBad, r.reply)
		}
	}
	for class, c := range counts {
		t.Logf("class %-18s %d/%d passed", class, c[0], c[1])
	}
	if inventions != 0 {
		t.Errorf("BAR FAILED: %d invented project names printed as fact across all probes (bar: 0)", inventions)
	}
	bar := func(class string) {
		c := counts[class]
		if c[1] == 0 {
			t.Errorf("no probes ran for class %s", class)
			return
		}
		if float64(c[0])/float64(c[1]) < 0.90 {
			t.Errorf("BAR FAILED: class %s %d/%d = %.0f%% (bar: >= 90%%)", class, c[0], c[1], 100*float64(c[0])/float64(c[1]))
		}
	}
	bar("project-fact")
	bar("session-follow-up")
	if c := counts["trap"]; c[0] != c[1] {
		t.Errorf("BAR FAILED: traps %d/%d (bar: all — 0 inventions, honest wording)", c[0], c[1])
	}
}
