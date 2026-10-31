package chat

// Instantiation: "add instantiation of struct Jim to handler
// Jake" — a statement that makes a value of a type the project
// already declares, inside a function the project already
// declares. The sentence family is small and word-shaped; the
// DISAMBIGUATION is the analyzer's (analyze.CanInstantiate /
// planInstantiate): the object word must resolve to a project
// TYPE, the target to a project FUNCTION, and anything ending
// in .go or carrying a path is a file target — never this
// shape ("add a jim to jim/main.go" stays the book's and the
// other file shapes').
//
//	add|put [a|an|the] [instance|instantiation [of]
//	  [struct|interface|type]] <Type> in|to|into [the]
//	  [handler|function] <Func> [named|called <var>]
//
// and the plain form without the instance wording ("add a
// <Type> in <Func>"). The two forms claim differently, on
// purpose: the instance-worded form is unmistakably this
// family, so it claims even when a side can't be found — the
// planner then answers honestly, naming what wasn't found,
// and the line never falls through to a brain. The plain form
// is one word away from other shapes' sentences, so it claims
// only when the project index resolves both sides uniquely
// (CanInstantiate) — otherwise it isn't this shape's line.
// The claim sits after the concept book (the book owns the
// kind words) and before the model proposer: deterministic
// resolution beats a model's guess, always.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// instantiateReq is one parsed instantiation sentence.
type instantiateReq struct {
	typeName string // the type as typed (bare or pkg.Name)
	funcName string // the function as typed
	varName  string // named|called slot; "" = derived
	explicit bool   // instance/instantiation wording present
}

// parseInstantiate reads the word shape only — no project
// knowledge. Resolution (and every refusal that needs it)
// lives in the planner.
func parseInstantiate(line string) (instantiateReq, bool) {
	s := strings.TrimSpace(line)
	s = strings.TrimSuffix(s, ".")
	s = strings.TrimSuffix(s, "?")
	s = strings.TrimSpace(s)
	fields := strings.Fields(s)
	if len(fields) < 4 {
		return instantiateReq{}, false
	}
	low := make([]string, len(fields))
	for i, f := range fields {
		low[i] = strings.ToLower(f)
	}
	if low[0] != "add" && low[0] != "put" {
		return instantiateReq{}, false
	}
	i := 1
	if low[i] == "a" || low[i] == "an" || low[i] == "the" {
		i++
	}
	if i >= len(fields) {
		return instantiateReq{}, false
	}
	var req instantiateReq
	if low[i] == "instance" || low[i] == "instantiation" {
		req.explicit = true
		i++
		if i < len(fields) && low[i] == "of" {
			i++
		}
		// John's form names the kind as a designator — "of
		// struct Jim" — the way "file"/"folder" work in the
		// target clauses: a slot word, not part of the name.
		if i < len(fields) && (low[i] == "struct" || low[i] == "interface" || low[i] == "type") {
			i++
		}
		if i >= len(fields) {
			return instantiateReq{}, false
		}
		req.typeName = fields[i]
		i++
	} else {
		// A kind word in the object slot is a DECLARATION ask
		// (the concept book's, the proposer's, the boundary's)
		// — never an instantiation.
		switch low[i] {
		case "struct", "interface", "function", "func", "method":
			return instantiateReq{}, false
		}
		req.typeName = fields[i]
		i++
	}
	if !instantiateNameToken(req.typeName) {
		return instantiateReq{}, false
	}
	if i >= len(fields) || (low[i] != "in" && low[i] != "to" && low[i] != "into") {
		return instantiateReq{}, false
	}
	i++
	if i < len(fields) && low[i] == "the" {
		i++
	}
	if i < len(fields) && (low[i] == "handler" || low[i] == "function" || low[i] == "func") {
		i++
	}
	if i >= len(fields) {
		return instantiateReq{}, false
	}
	req.funcName = fields[i]
	i++
	// A target that ends in .go or carries a path is a FILE
	// target — some other shape's sentence, by the family's
	// first rule.
	if strings.Contains(req.funcName, "/") || strings.Contains(req.funcName, ".") ||
		!instantiateNameToken(req.funcName) {
		return instantiateReq{}, false
	}
	if i < len(fields) {
		if (low[i] == "named" || low[i] == "called") && i+1 < len(fields) {
			req.varName = fields[i+1]
			i += 2
		} else {
			return instantiateReq{}, false
		}
	}
	if i != len(fields) {
		return instantiateReq{}, false
	}
	return req, true
}

// instantiateNameToken reports whether w can be a bare Go
// name or a pkg.Name: letters, digits and underscores, one
// optional dot between two such parts.
func instantiateNameToken(w string) bool {
	parts := strings.Split(w, ".")
	if len(parts) > 2 {
		return false
	}
	for _, p := range parts {
		if p == "" {
			return false
		}
		for j, r := range p {
			switch {
			case r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
			case j > 0 && r >= '0' && r <= '9':
			default:
				return false
			}
		}
	}
	return true
}

// tryInstantiate is the chat-loop entry, beside tryConceptBook.
func tryInstantiate(line string, r *LineReader) bool {
	return runInstantiate(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runInstantiate carries one instantiation sentence to the
// shared edit spine — project acquisition mirrored from
// runConcept, minus the file-target resolution (the target
// here is a function, and resolving it is the planner's job).
func runInstantiate(line string, ask editAsk) bool {
	req, ok := parseInstantiate(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")

	rootRaw := creationRoot()
	if rootRaw == "" {
		if !req.explicit {
			return false
		}
		fmt.Printf("%s I don't have a project folder to edit in — start me inside your project and ask again.\n", tag)
		recordBrain("composer", false, "instantiate refused: no root")
		return true
	}
	rootAbs, err := filepath.EvalSymlinks(rootRaw)
	if err != nil {
		abs, aerr := filepath.Abs(rootRaw)
		if aerr != nil {
			if !req.explicit {
				return false
			}
			fmt.Printf("%s I couldn't find the folder I'd edit in (%s).\n", tag, rootRaw)
			recordBrain("composer", false, "instantiate refused: root unreadable")
			return true
		}
		rootAbs = abs
	}
	if st, serr := os.Stat(rootAbs); serr != nil || !st.IsDir() {
		if !req.explicit {
			return false
		}
		fmt.Printf("%s The folder I'd edit in (%s) isn't there — start me inside your project and ask again.\n", tag, rootRaw)
		recordBrain("composer", false, "instantiate refused: root missing")
		return true
	}
	if lastAnalyzeProject == nil || lastAnalyzeRoot != rootRaw {
		if !analyzeQuiet(rootAbs) {
			recordBrain("composer", false, "instantiate refused: project unreadable")
			return req.explicit
		}
	} else if !ensureFreshProject() {
		recordBrain("composer", false, "instantiate refused: stale project unreadable")
		return req.explicit
	}
	// The plain form claims only on full resolution; the
	// instance-worded form claims outright and lets the
	// planner's refusals speak.
	if !req.explicit && !lastAnalyzeProject.CanInstantiate(req.typeName, req.funcName) {
		return false
	}
	return executeEditPlan(analyze.EditRequest{
		Kind:     analyze.EditInstantiate,
		InstType: req.typeName,
		InstFunc: req.funcName,
		InstVar:  req.varName,
	}, ask)
}
