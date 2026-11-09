package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// The 2026-11-09 live theft, pinned from both ends. In a fresh
// session in a repo whose code tracks a local variable named
// `need`, John typed "i need resources for a database" and the
// intent layer — which had labeled the line dataflow_q —
// answered with a varflow report about `need`. Two fixes meet
// here: the dataflow_q dispatch is gated on the line's own
// question shape (dataflowShape), and the resources ask itself
// is claimed deterministically by the book-resources claim
// (bookresources.go) ahead of the intent layer.

// needFixture writes a tiny project whose functions declare a
// local variable named `need` — the analyzed-repo condition
// that made John's statement's first word a tracked name.
func needFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	writeFile(t, dir, "go.mod", "module example.com/needdemo\n\ngo 1.21\n")
	writeFile(t, dir, "main.go", `package main

func ready() bool {
	need := true
	return need
}

func willing() bool {
	need := false
	return need
}

func main() {
	_ = ready()
	_ = willing()
}
`)
	return dir
}

// withAnalyzedProject points the chat's analyzed-project
// globals at p for the length of the test.
func withAnalyzedProject(t *testing.T, p *analyze.Project, root string) {
	t.Helper()
	oldProj, oldRoot := lastAnalyzeProject, lastAnalyzeRoot
	t.Cleanup(func() { lastAnalyzeProject, lastAnalyzeRoot = oldProj, oldRoot })
	lastAnalyzeProject = p
	lastAnalyzeRoot = root
}

// The gate's two directions at unit level: real dataflow
// shapes pass, plain statements mentioning tracked words do
// not — including codebase questions that are not dataflow
// questions.
func TestDataflowShapeGate(t *testing.T) {
	shaped := []string{
		"where does Items get its data",
		"where does Items get its data from",
		"where does Items come from",
		"where does Items change",
		"what changes Items",
		"what sets Items",
		"what writes to Items",
		"where is Items set",
		"where is Items changed",
		"who sets Items",
		"who changes Items",
		"who writes to Items",
	}
	for _, q := range shaped {
		if !dataflowShape(q) {
			t.Errorf("dataflowShape(%q) = false, want true", q)
		}
	}
	unshaped := []string{
		"i need resources for a database",
		"i need a database in jim",
		"Items is defined in the store package",
		"where does the team meet",
		"what is Items for",
		"why does Add call Save",
		"tell me about Items",
	}
	for _, q := range unshaped {
		if dataflowShape(q) {
			t.Errorf("dataflowShape(%q) = true, want false", q)
		}
	}
}

// The theft itself, at the dispatch the classifier feeds: the
// slot scan CAN find `need` in the line (that is what makes
// the gate necessary, and the sanity pins prove the fixture
// reproduces John's condition) — but the gated dispatch
// declines the line and prints no varflow report.
func TestDataflowDispatchGateBlocksStatement(t *testing.T) {
	dir := needFixture(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	withAnalyzedProject(t, p, dir)

	line := "i need resources for a database"
	if !p.HasData("need") {
		t.Fatal("fixture does not track a local named `need` — the test proves nothing")
	}
	if got := slotVarName(line); got != "need" {
		t.Fatalf("slotVarName(%q) = %q, want %q — the theft's loaded gun is not reproduced", line, got, "need")
	}
	report, ok := p.DataflowText("need")
	if !ok {
		t.Fatal("DataflowText(need) declined — the fixture cannot produce the stolen report")
	}

	var handled bool
	out := captureStdout(t, func() { handled = dispatchIntent("dataflow_q", line, nil) })
	if handled {
		t.Errorf("dispatchIntent(dataflow_q, %q) claimed the line; want the shape gate to decline it", line)
	}
	if strings.Contains(out, "local variable") || strings.Contains(out, report) {
		t.Errorf("the gated dispatch printed the varflow report:\n%s", out)
	}
}

// The gate's other direction: every real dataflow shape still
// dispatches, and its answer is byte-identical to what the
// pre-gate code printed — DataflowText over the extracted
// slot, verbatim.
func TestDataflowDispatchKeepsRealQuestions(t *testing.T) {
	dir := dataflowChatFixture(t)
	p, err := analyze.Analyze(dir)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	withAnalyzedProject(t, p, dir)

	for _, q := range []string{
		"where does Items get its data",
		"where does Items come from",
		"where does Items change",
		"what changes Items",
		"where is Items set",
		"who sets Items",
	} {
		name := slotVarName(q)
		want, ok := p.DataflowText(name)
		if !ok {
			t.Fatalf("DataflowText(%q from %q) declined", name, q)
		}
		var handled bool
		out := captureStdout(t, func() { handled = dispatchIntent("dataflow_q", q, nil) })
		if !handled {
			t.Errorf("dispatchIntent(dataflow_q, %q) declined a real dataflow question", q)
			continue
		}
		if strings.TrimSpace(out) != strings.TrimSpace(want) {
			t.Errorf("dispatchIntent(dataflow_q, %q) answer changed:\n--- got ---\n%s\n--- want ---\n%s", q, out, want)
		}
	}
}

// John's exact line, answered by the book: the database entry
// itself, and the one build pointer the catalog earns for the
// topic — the database connection piece, in the catalog's own
// words. No varflow furniture, no web addresses.
func TestBookResourcesAnswersDatabase(t *testing.T) {
	var handled bool
	out := captureStdout(t, func() { handled = tryBookResources("i need resources for a database") })
	if !handled {
		t.Fatal("tryBookResources declined John's line")
	}
	body, ok := LookupGoKnowledge("database")
	if !ok {
		t.Fatal("the book no longer answers \"database\" — the test's premise is gone")
	}
	if !strings.Contains(out, body) {
		t.Errorf("answer does not carry the book's database entry:\n%s", out)
	}
	if !strings.Contains(out, `"add a database connection function named Connect to db.go"`) {
		t.Errorf("answer is missing the database build pointer:\n%s", out)
	}
	if !strings.Contains(out, "rqlite") {
		t.Errorf("build pointer lost the catalog's rqlite wording:\n%s", out)
	}
	if strings.Contains(out, "local variable") {
		t.Errorf("answer contains varflow furniture:\n%s", out)
	}
	if strings.Contains(out, "http://") || strings.Contains(out, "https://") {
		t.Errorf("answer invented web resources:\n%s", out)
	}
}

// The claim's shape family: each lead-in variant is claimed
// and answers with its topic's book entry.
func TestBookResourcesShapes(t *testing.T) {
	for line, topic := range map[string]string{
		"i need resources for a database":      "database",
		"i want resources for channels":        "channels",
		"any resources for connection pooling": "connection pooling",
		"i'm looking for resources on testing": "testing",
		"please, i need resources for sqlite":  "sqlite",
	} {
		body, ok := LookupGoKnowledge(topic)
		if !ok {
			t.Fatalf("the book no longer answers %q — the test's premise is gone", topic)
		}
		var handled bool
		out := captureStdout(t, func() { handled = tryBookResources(line) })
		if !handled {
			t.Errorf("tryBookResources(%q) declined a resources shape", line)
			continue
		}
		if !strings.Contains(out, body) {
			t.Errorf("tryBookResources(%q) answer missing the %q entry:\n%s", line, topic, out)
		}
	}
}

// The seating boundary, pinned: the line-initial "resources
// for X" shape stays with the Advocates claim (it sits ahead
// of this one in every loop and claims it first), and asks
// that merely need something — a build ask above all — are
// never this claim's.
func TestBookResourcesDeclines(t *testing.T) {
	for _, line := range []string{
		"resources for databases",                        // tryBuildResources owns the line-initial shape
		"i need a database in jim",                       // a build ask: keeps its current routing
		"i need resources for a database in jim/main.go", // a path is not a topic
		"i need help with my database",                   // no resources ask in it
		"i want to build a webserver",                    // a build goal: tryBuildResources' shape
		"what is a database",                             // a concept question, not a resources ask
	} {
		var handled bool
		out := captureStdout(t, func() { handled = tryBookResources(line) })
		if handled {
			t.Errorf("tryBookResources(%q) claimed a line that is not its shape:\n%s", line, out)
		}
		if out != "" {
			t.Errorf("tryBookResources(%q) printed while declining:\n%s", line, out)
		}
	}
}

// A topic the book doesn't cover: an honest miss that names
// the topic, promises nothing, links nowhere — and, when the
// book holds near topics, names them as the closest it has.
func TestBookResourcesMissHonesty(t *testing.T) {
	if near := nearBookTopics("sourdough baking", 3); len(near) != 0 {
		t.Fatalf("premise gone: the book now has topics near sourdough baking: %v", near)
	}
	var handled bool
	out := captureStdout(t, func() { handled = tryBookResources("i need resources for sourdough baking") })
	if !handled {
		t.Fatal("a resources shape on an unknown topic must still be claimed — with an honest miss")
	}
	if !strings.Contains(out, `I don't have resources on "sourdough baking"`) {
		t.Errorf("miss does not name the topic honestly:\n%s", out)
	}
	if strings.Contains(out, "http://") || strings.Contains(out, "https://") {
		t.Errorf("miss invented web resources:\n%s", out)
	}
	if strings.Contains(out, "Closest topics") {
		t.Errorf("miss dressed up unrelated topics as near ones:\n%s", out)
	}

	if _, ok := LookupGoKnowledge("panic handling"); ok {
		t.Fatal("premise gone: the book now answers \"panic handling\" outright")
	}
	out = captureStdout(t, func() { handled = tryBookResources("i need resources for panic handling") })
	if !handled {
		t.Fatal("tryBookResources declined the panic-handling miss")
	}
	if !strings.Contains(out, `I don't have resources on "panic handling"`) {
		t.Errorf("miss does not name the topic honestly:\n%s", out)
	}
	near := nearBookTopics("panic handling", 3)
	if len(near) == 0 {
		t.Fatal("nearBookTopics(panic handling) is empty — the book's panic/error entries should be near")
	}
	if !strings.Contains(out, "Closest topics the book does cover") {
		t.Errorf("miss with near topics did not name them:\n%s", out)
	}
	for _, title := range near {
		if !strings.Contains(out, title) {
			t.Errorf("near topic %q missing from the miss:\n%s", title, out)
		}
	}
}

// John's session end to end through the unified loop's
// deterministic chain, in the session state he was in: the
// project analyzed, its code tracking a local named `need`.
// The line is claimed (by the book-resources seat beside the
// build-goal resources), the answer is the book's database
// entry, and no varflow report appears.
func TestBookResourcesInChainWithNeedProject(t *testing.T) {
	dir := needFixture(t)
	withChatSession(t, dir, true)
	if lastAnalyzeProject == nil || !lastAnalyzeProject.HasData("need") {
		t.Fatal("session fixture does not track a local named `need` — the test proves nothing")
	}
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("i need resources for a database", yes, profile) })
	if !handled {
		t.Fatalf("the chain declined John's line:\n%s", out)
	}
	body, _ := LookupGoKnowledge("database")
	if !strings.Contains(out, body) {
		t.Errorf("the chain's answer is not the book's database entry:\n%s", out)
	}
	if strings.Contains(out, "local variable") {
		t.Errorf("the chain printed the varflow report:\n%s", out)
	}
	// The build ask keeps its routing: the resources claim
	// never sees it, whatever else in the chain answers it.
	if parseBookResourceTopic("i need a database in jim") != "" {
		t.Error("the build ask parsed as a resources topic")
	}
}
