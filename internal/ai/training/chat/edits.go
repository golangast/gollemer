package chat

// Edits by assembly: the chat face of the Composer's write side.
// A plain change request — "add a backoff case to Parse", "add a
// Timeout field to Config", "register thanks in Greetings", "add a
// function like Greet named GreetLoudly" — is parsed HERE,
// deterministically, in the same spirit as the beginner five: a
// sentence that would write to the user's files is never left for
// the intent classifier or a neural model to guess at.
//
// The analyzer plans the edit from a real sibling pattern
// (analyze.PlanEdit); this layer shows the exact unified diff and
// asks [y/n] — the one and only write gate. On yes: apply, go
// build, re-analyze, verify the intended structure exists. On any
// failure after applying, the original bytes go back and the reply
// says exactly that. On no: nothing was ever written.
//
// Accepted forms (trailing "." or "?" optional):
//
//	add a <value> case to <Func> [function]
//	add a case for <value> to <Func>
//	add <value> case to the switch on <subject> in <Func>
//	in <Func> also handle <value>
//	add a <Field> field of type <T> to [the] <Struct> [struct] [after <Field>]
//	add a <T> field <Field> to [the] <Struct>
//	register <key> in [the] <Table> [table|map]
//	add <key> to [the] <Table> table|map|list
//	add a [new] function like <Func> named <NewName>
//	add a <NewName> function like <Func>
//	copy|clone <Func> as <NewName>

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	reEditFuncA  = regexp.MustCompile(`(?i)^add a (?:new )?function like ([A-Za-z_][\w.]*) named ([A-Za-z_]\w*)$`)
	reEditFuncC  = regexp.MustCompile(`(?i)^(?:copy|clone) ([A-Za-z_][\w.]*) as ([A-Za-z_]\w*)$`)
	reEditFuncB  = regexp.MustCompile(`(?i)^add (?:a |an )?([A-Za-z_]\w*) function like ([A-Za-z_][\w.]*)$`)
	reEditFieldA = regexp.MustCompile(`(?i)^add (?:a |an )?([A-Za-z_]\w*) field(?: of type (.+?))? to (?:the )?([A-Za-z_][\w.]*)(?: struct)?(?: after ([A-Za-z_]\w*))?$`)
	reEditFieldB = regexp.MustCompile(`(?i)^add (?:a |an )?(.+?) field ([A-Za-z_]\w*) to (?:the )?([A-Za-z_][\w.]*)(?: struct)?(?: after ([A-Za-z_]\w*))?$`)
	reEditHandle = regexp.MustCompile(`(?i)^in (.+?)(?:,)? also handle (.+)$`)
	reEditCaseA  = regexp.MustCompile(`(?i)^add (?:a |an )?case (?:for )?(.+?) (?:to|in) (.+)$`)
	reEditCaseB  = regexp.MustCompile(`(?i)^add (?:a |an )?(.+?) case (?:to|in) (.+)$`)
	reEditRegA   = regexp.MustCompile(`(?i)^register (.+?) in (?:the )?([A-Za-z_]\w*)(?: table| map)?$`)
	reEditRegB   = regexp.MustCompile(`(?i)^add (.+?) to (?:the )?([A-Za-z_]\w*) (?:table|map|list)$`)
)

// parseAssembledEdit turns one chat line into an EditRequest. The
// second return is false for anything that is not an edit request —
// including add-sentences the composer cannot assemble ("add a
// getter…", "add a retry helper…"), which belong to other brains.
func parseAssembledEdit(line string) (analyze.EditRequest, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)

	// Functions first: "add a function like…" must never fall into
	// the case/field forms below.
	if m := reEditFuncA.FindStringSubmatch(s); m != nil {
		return analyze.EditRequest{Kind: analyze.EditAddFunc, SiblingName: m[1], NewName: m[2]}, true
	}
	if m := reEditFuncC.FindStringSubmatch(s); m != nil {
		return analyze.EditRequest{Kind: analyze.EditAddFunc, SiblingName: m[1], NewName: m[2]}, true
	}
	if m := reEditFuncB.FindStringSubmatch(s); m != nil {
		return analyze.EditRequest{Kind: analyze.EditAddFunc, SiblingName: m[2], NewName: m[1]}, true
	}
	// Fields before cases: "field" never appears in a case request.
	// The type-first form is tried first — in "add an int field
	// Timeout…", the name-first pattern would otherwise read "int"
	// as the name and "Timeout" as the type. The name-first form's
	// own type must be spelled "of type …", which the type-first
	// pattern cannot match, so the two never collide.
	if m := reEditFieldB.FindStringSubmatch(s); m != nil {
		req := analyze.EditRequest{Kind: analyze.EditAddField, FieldType: strings.TrimSpace(m[1]), FieldName: m[2], StructName: m[3]}
		if m[4] != "" {
			req.AfterField = m[4]
		}
		return req, true
	}
	if m := reEditFieldA.FindStringSubmatch(s); m != nil {
		req := analyze.EditRequest{Kind: analyze.EditAddField, FieldName: m[1], FieldType: strings.TrimSpace(m[2]), StructName: m[3]}
		if m[4] != "" {
			req.AfterField = m[4]
		}
		return req, true
	}
	// "in <Func> also handle <value>".
	if m := reEditHandle.FindStringSubmatch(s); m != nil {
		fn, switchOn := parseCaseTarget(m[1])
		return analyze.EditRequest{Kind: analyze.EditAddCase, FuncName: fn, SwitchOn: switchOn, CaseExpr: strings.TrimSpace(m[2])}, true
	}
	if m := reEditCaseA.FindStringSubmatch(s); m != nil {
		fn, switchOn := parseCaseTarget(m[2])
		return analyze.EditRequest{Kind: analyze.EditAddCase, FuncName: fn, SwitchOn: switchOn, CaseExpr: strings.TrimSpace(m[1])}, true
	}
	if m := reEditCaseB.FindStringSubmatch(s); m != nil {
		fn, switchOn := parseCaseTarget(m[2])
		return analyze.EditRequest{Kind: analyze.EditAddCase, FuncName: fn, SwitchOn: switchOn, CaseExpr: strings.TrimSpace(m[1])}, true
	}
	if m := reEditRegA.FindStringSubmatch(s); m != nil {
		return analyze.EditRequest{Kind: analyze.EditRegister, TableName: m[2], Key: unquoteKey(m[1])}, true
	}
	if m := reEditRegB.FindStringSubmatch(s); m != nil {
		return analyze.EditRequest{Kind: analyze.EditRegister, TableName: m[2], Key: unquoteKey(m[1])}, true
	}
	return analyze.EditRequest{}, false
}

var reSwitchTarget = regexp.MustCompile(`(?i)^(?:the )?switch on (.+?) in (.+)$`)

// parseCaseTarget reads the target half of a case request:
// "<Func>", "<Func> function", or "the switch on <subject> in
// <Func>". Returns the function name and, when given, the switch
// subject to select among several switches.
func parseCaseTarget(s string) (fn, switchOn string) {
	s = strings.TrimSpace(s)
	if m := reSwitchTarget.FindStringSubmatch(s); m != nil {
		return cleanFuncName(m[2]), strings.TrimSpace(m[1])
	}
	return cleanFuncName(s), ""
}

func cleanFuncName(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimSuffix(s, " function")
	s = strings.TrimPrefix(s, "the ")
	return strings.TrimSpace(s)
}

// unquoteKey strips one pair of surrounding quotes from a typed key.
func unquoteKey(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

// tryAssembledEdit is the chat-loop entry: it claims edit-request
// sentences and runs the whole gate flow on the chat's line reader.
func tryAssembledEdit(line string, r *LineReader) bool {
	return runAssembledEdit(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// editAsk is the [y/n] prompt source — a function so tests can
// script the answer without a terminal.
type editAsk func(prompt string) (string, error)

// runAssembledEdit carries one edit request from sentence to
// verified change (or an honest refusal / a clean "not applied").
// It returns true when the line was an edit request, claimed or not.
func runAssembledEdit(line string, ask editAsk) bool {
	req, ok := parseAssembledEdit(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if !ensureProject() {
		fmt.Printf("%s I need a project to edit — analyze one first (or start me inside one).\n", tag)
		return true
	}
	return executeEditPlan(req, ask)
}

// executeEditPlan runs the one write spine for an assembled edit
// request whose project is already the current one (the caller
// acquired it — runAssembledEdit via ensureProject, the webserver
// flow via its own root rule): plan from the project's own code,
// show the diff, ask [y/n]. On yes: apply, go build, re-analyze,
// verify the intended structure exists. On any failure after
// applying, the original bytes go back and the reply says exactly
// that. On no: nothing was ever written.
func executeEditPlan(req analyze.EditRequest, ask editAsk) bool {
	tag := cTag("gollemer [composer]>")
	proj := lastAnalyzeProject
	plan, refusal := proj.PlanEdit(req)
	if refusal != "" || plan == nil {
		if refusal == "" {
			refusal = "I couldn't assemble that edit from this project's own code."
		}
		fmt.Printf("%s Not doing that edit: %s\n", tag, refusal)
		recordBrain("composer", false, "edit refused: "+req.Kind)
		return true
	}
	fmt.Printf("%s Here's the edit I assembled from this project's own code:\n\n%s\n", tag, plan.Summary)
	for _, n := range plan.Notes {
		fmt.Printf("%s\n", n)
	}
	fmt.Printf("\n%s\n", plan.Diff())
	ans, err := ask("Apply this edit? [y/n] ")
	if err != nil || (strings.ToLower(strings.TrimSpace(ans)) != "y" && strings.ToLower(strings.TrimSpace(ans)) != "yes") {
		fmt.Printf("%s Not applied — every file is exactly as it was.\n", tag)
		recordBrain("composer", true, "edit declined at the gate: "+req.Kind)
		return true
	}

	root := proj.Root
	if root == "" {
		root = lastAnalyzeRoot
	}
	// Apply. Any write failure restores every file already written.
	written, werr, wrel := applyPlanFiles(root, plan)
	if werr != nil {
		restoreEditFiles(root, plan, written)
		fmt.Printf("%s I couldn't write %s (%v) — so I restored everything; your files are exactly as they were.\n", tag, proj.LinkPath(wrel), werr)
		recordBrain("composer", false, "edit write failed; restored")
		return true
	}
	// Build, when the project has a go.mod at its root to build
	// against. No go.mod (a loose folder): the structural verify
	// below is the check, and the report says so.
	built := false
	if _, serr := os.Stat(filepath.Join(root, "go.mod")); serr == nil {
		out, bok := buildProject(root)
		if !bok {
			restoreEditFiles(root, plan, len(plan.Files))
			fmt.Printf("%s That edit doesn't build, so I restored your files — everything is exactly as it was.\n", tag)
			fmt.Printf("The build said: %s\nIn plain words: %s\n", strings.TrimSpace(out), explainBuildError(out))
			recordBrain("composer", false, "edit rolled back: build failed")
			return true
		}
		built = true
	}
	// Re-analyze and verify the intended structure now exists.
	fresh, aerr := analyze.Analyze(root)
	if aerr != nil || fresh == nil || !plan.Verify(fresh) {
		restoreEditFiles(root, plan, len(plan.Files))
		fmt.Printf("%s I applied the edit but couldn't verify the change is really there, so I restored your files — everything is exactly as it was.\n", tag)
		recordBrain("composer", false, "edit rolled back: verify failed")
		return true
	}
	lastAnalyzeProject = fresh
	how := "built the project and checked"
	if !built {
		how = "re-read the project and checked (no go.mod at its root, so no build to run)"
	}
	fmt.Printf("%s Done — %s, at %s. I %s that it's really there.\n", tag, plan.VerifyDesc, plan.Site, how)
	recordBrain("composer", true, "edit applied: "+req.Kind)
	return true
}

// buildProject runs `go build ./...` at root — the spine's one
// build check, shared by the assembled edits, renames, and error
// fixes. It returns the combined output and whether it passed.
func buildProject(root string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	cmd := exec.CommandContext(ctx, "go", "build", "./...")
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	return string(out), err == nil
}

// applyPlanFiles writes a plan's files in order. On failure it
// reports how many files were written and which one failed, so the
// caller can restore exactly the written prefix.
func applyPlanFiles(root string, plan *analyze.EditPlan) (written int, err error, failedRel string) {
	for _, f := range plan.Files {
		if werr := os.WriteFile(filepath.Join(root, f.Rel), f.After, 0644); werr != nil {
			return written, werr, f.Rel
		}
		written++
	}
	return written, nil, ""
}

// restoreEditFiles writes back the original bytes of the first n
// files of a plan. Best effort by design: it runs on paths the plan
// itself produced from files that were just read.
func restoreEditFiles(root string, plan *analyze.EditPlan, n int) {
	for i := 0; i < n && i < len(plan.Files); i++ {
		_ = os.WriteFile(filepath.Join(root, plan.Files[i].Rel), plan.Files[i].Before, 0644)
	}
}
