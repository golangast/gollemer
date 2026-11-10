package chat

import (
	"strings"
	"testing"
)

// The 2026-11-10 dispatch fix, pinned: an imperative add
// line no composer claim owns ends at the composer's honest
// boundary — never at an analysis brain. Two layers:
// (2a) the write boundary's destination arm (writeboundary.go)
// claims add|put|insert lines naming a thing and one bare
// destination word; (2b) the phrasing verdict
// (phrasingverdict.go), seated after every deterministic
// claim, answers a line the recognizer judges a command with
// the same boundary answer. Analysis questions, social
// lines, and statements decline both layers and proceed
// exactly as before — the false set below is the bar.

func TestWriteBoundaryDestinationArm(t *testing.T) {
	claims := map[string]string{ // line → the object named back
		"add templates in jim":                   "templates",
		"add header and footer templates to jim": "header and footer templates",
		"add a database to the project":          "database",
		"put the templates into jim":             "templates",
	}
	for line, obj := range claims {
		req, ok := parseWriteBoundary(line)
		if !ok {
			t.Errorf("parseWriteBoundary(%q) declined — the destination arm should claim it", line)
			continue
		}
		if req.object != obj {
			t.Errorf("parseWriteBoundary(%q).object = %q, want %q", line, req.object, obj)
		}
	}
	declines := []string{
		// The header's own deliberate examples: a pronoun
		// object, and a multi-word destination.
		"add it to the list of things to do",
		"add it to jim",
		"add milk to the shopping list",
		// create|write keep the file-target-only rule: they
		// lead generation asks the brains own.
		"create templates in jim",
		"write templates in jim",
		// The kinds the intent layer owns (the object set's
		// own exclusions): their head words pass the arm by.
		"add a comment to Describe",
		"add a test to Describe",
		"add a getter to Describe",
		// A behaviour clause keeps the line a generation ask.
		"add a function that reverses a string",
		// Questions and other brains' lines.
		"what does Add do",
		"compare Add and Get",
		"how is Add used",
	}
	for _, line := range declines {
		if req, ok := parseWriteBoundary(line); ok {
			t.Errorf("parseWriteBoundary(%q) claimed %+v — it must stay unclaimed", line, req)
		}
	}
}

// The verdict battery, run against the SHIPPED phrasing
// model (clarifyPhrasing swaps the real classifier in — the
// same seam the phrasing tests use). True set: imperative
// lines the recognizer judges commands — several of which
// the boundary's own shape test declines (the verdict-only
// lines the second layer exists for). False set: the
// regression bar — analysis questions, social lines, and
// statements must never earn a boundary answer.
func TestPhrasingCommandVerdictBattery(t *testing.T) {
	clarifyPhrasing(t)
	commands := []string{
		"add templates in jim",
		"could you stick a for loop in main for me?",
		"plese add a print line in main",
		"create a backup system in jim",
		"add some middleware",
		"add authentication",
		"put some validation in the signup form",
		"add a search box to the home page",
	}
	for _, line := range commands {
		if !phrasingCommandVerdict(line) {
			t.Errorf("phrasingCommandVerdict(%q) = false, want a command verdict", line)
		}
	}
	notCommands := []string{
		// Analysis questions — must still reach analysis.
		"what does Add do",
		"compare Add and Get",
		"how is Add used",
		"show me Add",
		"walk me through Add",
		"where is Add called",
		"what is a channel",
		"how do I serve static files",
		"which package do I use for json",
		"find code that handles retries",
		// Statements and social lines.
		"the server returns an html file",
		"my dog is bella",
		"hello",
		"how are you",
		"what's my dog's name",
		// Other layers' lines: the runner's, the intent
		// layer's build_api.
		"run make build",
		"build an api with /users and /health",
		// The recognizer abstains on these two — the first
		// falls to analysis as today, the second is closed
		// by the boundary's destination arm instead (the
		// layering is the design, pinned here so a future
		// model change shows up as a deliberate diff).
		"add caching to the user service",
		"add header and footer templates to jim",
	}
	for _, line := range notCommands {
		if phrasingCommandVerdict(line) {
			t.Errorf("phrasingCommandVerdict(%q) = true — a false boundary hit", line)
		}
	}
}

// End to end through the composer chain (claimChain mirrors
// the unified loop's claim order): the two stolen lines and
// one verdict-only line end at the boundary's answer, and
// the analysis questions are not claimed by the chain at
// all — they fall through to analysis byte-identical.
func TestDispatchUnservedAddsEndAtBoundary(t *testing.T) {
	staticAssetsFixture(t, staticAssetsFiles(map[string]string{
		"jim/main.go": staticJimMain,
	}))
	profile := &UserProfile{}
	boundaryLines := map[string]string{ // line → object named in the answer
		"add templates in jim":                   "templates",
		"add header and footer templates to jim": "header and footer templates",
		"add authentication":                     "authentication",
	}
	for line, obj := range boundaryLines {
		claimed := false
		out := captureStdout(t, func() { claimed = claimChain(line, yes, profile) })
		if !claimed {
			t.Errorf("claimChain(%q) declined — the line would reach an analysis brain:\n%s", line, out)
			continue
		}
		if !strings.Contains(out, "I can't add "+obj+" yet") {
			t.Errorf("claimChain(%q) did not answer at the boundary:\n%s", line, out)
		}
		if strings.Contains(out, "really used") || strings.Contains(out, "Name two functions") {
			t.Errorf("claimChain(%q) was answered by an analysis brain:\n%s", line, out)
		}
	}
	for _, line := range []string{"what does Add do", "compare Add and Get"} {
		claimed := false
		out := captureStdout(t, func() { claimed = claimChain(line, yes, profile) })
		if claimed {
			t.Errorf("claimChain(%q) claimed an analysis question — it must fall through untouched:\n%s", line, out)
		}
	}
	// "how is Add used" is read by the chain's own uses claim
	// (a standing seat, not one of this fix's layers) — the
	// pin is that neither new layer's answer touches it.
	out := captureStdout(t, func() { claimChain("how is Add used", yes, profile) })
	if strings.Contains(out, "I can't add") || strings.Contains(out, "pieces I can assemble") {
		t.Errorf("the dispatch fix answered an analysis question:\n%s", out)
	}
}
