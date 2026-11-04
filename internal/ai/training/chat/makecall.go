package chat

// Make-and-call: the chat face of the one-ask, two-file
// recipe. "Make a function named Validate in api/validate.go
// and call it in api/main.go" declares a NEW function in one
// file and writes the call into another — one plan, one
// gate, one transaction, the whole project build-verified,
// both files restored together on any failure. The two-step
// forms stay exactly as they were (a concept declaration on
// its own; a uses line on its own); this claim exists only
// for the joined sentence, and it is claimed deterministically
// beside the uses, before the concept book — the book's
// grammar would read the first half, balk at the "and call
// it" tail, and the compound machinery would split the ask
// into two gated halves.
//
// The analyzer plans both halves (editmakecall.go): the
// callee from the concept book's function row through the
// declaration machinery — so the file may be created under
// the declaration rules — and the caller assembled from the
// PLANNED declaration's signature. The sentence's three
// shapes:
//
//	make|add|create|put a function [named] <Name> in <A>.go
//	  and call it in <B>.go
//	  and call it from <Func> in <B>.go
//	  and add|create|make a function [named] <Caller> in
//	    <B>.go that calls it
//
// A same-file ask (A and B the same path) is NOT claimed:
// the ordinary one-file shapes keep it.

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

var (
	reMakeCallNamed = regexp.MustCompile(`(?i)^(?:make|add|create|put) a function (?:named )?([A-Za-z_]\w*) in (.+?\.go) and (?:add|create|make) a function (?:named )?([A-Za-z_]\w*) in (.+?\.go) that calls it$`)
	reMakeCallFrom  = regexp.MustCompile(`(?i)^(?:make|add|create|put) a function (?:named )?([A-Za-z_]\w*) in (.+?\.go) and call it from ([A-Za-z_]\w*) in (.+?\.go)$`)
	reMakeCallBare  = regexp.MustCompile(`(?i)^(?:make|add|create|put) a function (?:named )?([A-Za-z_]\w*) in (.+?\.go) and call it in (.+?\.go)$`)
)

// parseMakeCall reads one chat line as a make-and-call
// request: the callee's name, its file as typed, the caller's
// file as typed, and the caller form — a new function's name
// (caller), an existing host function (host), or neither
// (bare: the planner hosts the call in the target package's
// main and refuses with both working forms when there is
// none). The second return is false for anything outside the
// grammar, including a same-file ask.
func parseMakeCall(line string) (name, calleeFile, targetFile, caller, host string, ok bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	if m := reMakeCallNamed.FindStringSubmatch(s); m != nil {
		name, calleeFile, caller, targetFile = m[1], cleanTypedFile(m[2]), m[3], cleanTypedFile(m[4])
	} else if m := reMakeCallFrom.FindStringSubmatch(s); m != nil {
		name, calleeFile, host, targetFile = m[1], cleanTypedFile(m[2]), m[3], cleanTypedFile(m[4])
	} else if m := reMakeCallBare.FindStringSubmatch(s); m != nil {
		name, calleeFile, targetFile = m[1], cleanTypedFile(m[2]), cleanTypedFile(m[3])
	} else {
		return "", "", "", "", "", false
	}
	if path.Clean(filepath.ToSlash(calleeFile)) == path.Clean(filepath.ToSlash(targetFile)) {
		// One file, not two: the recipe doesn't fire, and the
		// ordinary shapes keep the line.
		return "", "", "", "", "", false
	}
	return name, calleeFile, targetFile, caller, host, true
}

// makeCallFunctionEntry finds the book's function row — the
// one declaration this recipe declares.
func makeCallFunctionEntry() (analyze.ConceptEntry, bool) {
	for _, e := range conceptEntries() {
		if e.ID == "function" && e.Kind == analyze.ConceptDeclaration {
			return e, true
		}
	}
	return analyze.ConceptEntry{}, false
}

// tryMakeCall is the chat-loop entry: it claims make-and-call
// sentences and runs the whole flow on the chat's line
// reader.
func tryMakeCall(line string, r *LineReader) bool {
	return runMakeCall(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runMakeCall carries one make-and-call request from
// sentence to verified change (or an honest refusal / a
// clean "not applied") — the concept claim's project
// acquisition and target rules, then the shared edit spine,
// whose one gate covers both files.
func runMakeCall(line string, ask editAsk) bool {
	name, typedCallee, typedTarget, caller, host, ok := parseMakeCall(line)
	if !ok {
		return false
	}
	entry, found := makeCallFunctionEntry()
	if !found {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootRaw := creationRoot()
	if rootRaw == "" {
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "make-call refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "make-call refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "make-call refused: root missing")
		return true
	}
	// The callee's target resolves under the DECLARATION
	// rules (resolveConceptDeclTarget): its file may be new
	// when its folder exists; a missing folder refuses with
	// the create family's first step.
	calleeRel, refusal := resolveConceptDeclTarget(rootAbs, typedCallee, name, entry.Canonical())
	if refusal != "" {
		fmt.Printf("%s Not doing that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "make-call refused: callee target")
		return true
	}
	// The caller's file must already exist — the calling side
	// goes into a file that is there, the uses route's rule.
	abs, targetRel, refusal := serverTargetResolve(rootAbs, typedTarget)
	if refusal != "" {
		fmt.Printf("%s Not doing that: %s.\n", tag, strings.TrimSuffix(refusal, "."))
		recordBrain("composer", false, "make-call refused: target")
		return true
	}
	if st, serr := os.Stat(abs); serr != nil || st.IsDir() {
		fmt.Printf("%s Not doing that: there's no %s in this project — the calling side goes into a file that already exists. (For a brand-new file, say \"create a file %s\" first.)\n", tag, typedTarget, typedTarget)
		recordBrain("composer", false, "make-call refused: target missing")
		return true
	}
	if calleeRel == targetRel {
		// The two typed paths spelled the same file two
		// ways: not this recipe after all.
		return false
	}
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "make-call refused: project unreadable")
			return true
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "make-call refused: stale project unreadable")
		return true
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:           analyze.EditMakeCall,
		File:           calleeRel,
		Concept:        &entry,
		ConceptName:    name,
		MakeCallTarget: targetRel,
		MakeCallCaller: caller,
		MakeCallHost:   host,
	}, ask)
}
