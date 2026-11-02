package chat

// The gorqlite dependency gate (database build, 2026-11-01 —
// John's word: use the gorqlite client for the rqlite family,
// and "just have it ask you once installed to let it know").
//
// The book's gorqlite rows import github.com/rqlite/gorqlite —
// the one external module in the book. Assembled code naming
// that import cannot compile in a project that does not
// require the module, and the chat never fetches dependencies
// (the same rule the rqlite guide states: the chat installs
// nothing). So before a gorqlite row plans, the project's
// go.mod is checked:
//
//   - module present → the row plans like any other row;
//   - module absent → NO plan is offered. The claim answers
//     with the exact `go get` line for the user's own
//     terminal, says why, and holds the ask in the session.
//     A later tight follow-up ("installed", "done",
//     "gorqlite is installed" — claimed ONLY while an ask is
//     held) re-reads go.mod: present now, the ORIGINAL ask is
//     re-run so its plan comes back behind the usual [y/n]
//     gate (the word "installed" alone never writes anything);
//     still absent, the answer says so plainly and the ask
//     stays held.
//
// The gate is data-driven like the rows it guards: any book
// row whose imports name the client module is gated, with no
// concept-specific code. The rqlite HTTP rows import only the
// standard library, so they are never gated — every line they
// own today keeps its answer.

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// gorqliteModule is the external module the gorqlite book
// rows import.
const gorqliteModule = "github.com/rqlite/gorqlite"

// gorqliteGated reports whether a book row belongs to the
// gorqlite family: one of its imports is the client module
// (or a package under it). Read off the row's data, so a
// future row importing the module is gated automatically.
func gorqliteGated(e analyze.ConceptEntry) bool {
	for _, imp := range e.Imports {
		if imp == gorqliteModule || strings.HasPrefix(imp, gorqliteModule+"/") {
			return true
		}
	}
	return false
}

// goModRequires reads <root>/go.mod and reports whether the
// module path is among its requirements — a single-line
// require or an entry in a require block. Commented-out
// lines do not count. A missing or unreadable go.mod counts
// as absent.
func goModRequires(rootAbs, module string) bool {
	blob, err := os.ReadFile(filepath.Join(rootAbs, "go.mod"))
	if err != nil {
		return false
	}
	inBlock := false
	for _, ln := range strings.Split(string(blob), "\n") {
		ln = strings.TrimSpace(ln)
		if strings.HasPrefix(ln, "//") {
			continue
		}
		if inBlock {
			if ln == ")" {
				inBlock = false
				continue
			}
			if fields := strings.Fields(ln); len(fields) > 0 && fields[0] == module {
				return true
			}
			continue
		}
		if strings.HasPrefix(ln, "require (") {
			inBlock = true
			continue
		}
		if strings.HasPrefix(ln, "require ") {
			if fields := strings.Fields(strings.TrimPrefix(ln, "require ")); len(fields) > 0 && fields[0] == module {
				return true
			}
		}
	}
	return false
}

// pendingDepAsk is the session's one held gorqlite ask: the
// original sentence (re-run verbatim when the follow-up
// lands, so the plan offered is the plan that was asked
// for), a plain-words description of the piece for the
// gate's answers, and the project root whose go.mod the
// follow-up re-reads. Session state only — it lives in
// memory like the rest of the chat's session, never on disk.
type pendingDepAsk struct {
	line  string
	piece string
	root  string
}

var pendingGorqlite *pendingDepAsk

// describeDepAsk renders the held piece in plain words from
// the parsed ask: the row's canonical name, the declaration
// name when there is one, and where it was headed — the
// folder address for a no-file folder declaration, else the
// function and file it named.
func describeDepAsk(req conceptReq) string {
	s := "a " + req.entry.Canonical()
	if req.name != "" {
		s += " named " + req.name
	}
	if req.target == "" && req.entry.Folder != "" && req.name != "" {
		s += " (it would be created as " + req.entry.Folder + "/" + strings.ToLower(req.name) + ".go)"
		return s
	}
	if req.inFunc != "" {
		s += " in " + req.inFunc
	}
	if req.target != "" {
		s += " for " + req.target
	}
	return s
}

// gorqliteDepMissing answers a gorqlite-family ask whose
// project does not require the client yet: no plan, the
// exact command for the user's own terminal, the reason,
// and the ask held in the session for the follow-up.
func gorqliteDepMissing(req conceptReq, rootAbs, line string) {
	piece := describeDepAsk(req)
	pendingGorqlite = &pendingDepAsk{line: line, piece: piece, root: rootAbs}
	tag := cTag("gollemer [composer]>")
	fmt.Printf("%s That %s needs the gorqlite client, and this project doesn't have it yet — the code would import %s, and nothing compiles against a dependency the project doesn't list. I don't fetch dependencies for you, so run this in the project first:\n", tag, piece, gorqliteModule)
	fmt.Printf("%s   go get %s\n", tag, gorqliteModule)
	if note := gorqliteNewFileNote(req, rootAbs); note != "" {
		fmt.Printf("%s %s\n", tag, note)
	}
	fmt.Printf("%s Once you've run it, tell me \"installed\" and I'll bring that %s back as a plan for your okay.\n", tag, piece)
	recordBrain("composer", false, "concept refused: gorqlite dependency missing (go get "+gorqliteModule+")")
}

// gorqliteNewFileNote makes the gate's promise explicit for
// the one target shape whose plan creates something: a gated
// DECLARATION aimed at a .go file that doesn't exist yet, in
// a folder that does — when the re-offer lands, the plan
// creates the file (the declaration-create path), so the
// first answer says so. "" for every other shape: an
// existing file, a folder or folder-default target (the
// piece description already names the created path there),
// a statement row (statements never create files), or a
// folder that is itself missing — there the honest first
// step is the folder, and the re-offer will name it.
func gorqliteNewFileNote(req conceptReq, rootAbs string) string {
	if req.entry.Kind != analyze.ConceptDeclaration || req.target == "" {
		return ""
	}
	abs, rel, refusal := serverTargetResolve(rootAbs, req.target)
	if refusal != "" || !strings.HasSuffix(strings.ToLower(rel), ".go") {
		return ""
	}
	if _, err := os.Stat(abs); !os.IsNotExist(err) {
		return ""
	}
	if st, err := os.Stat(filepath.Dir(abs)); err != nil || !st.IsDir() {
		return ""
	}
	return fmt.Sprintf("%s doesn't exist yet — it will be created as part of the plan when it lands.", rel)
}

// gorqliteInstalledShape is the tight follow-up vocabulary:
// the line, whole and alone, says the install happened.
// These shapes are only ever consulted while an ask is held
// (runGorqliteInstalled), so they claim nothing in any other
// session state.
func gorqliteInstalledShape(line string) bool {
	s := strings.Join(strings.Fields(strings.ToLower(strings.TrimSpace(line))), " ")
	s = strings.TrimSuffix(s, ".")
	switch s {
	case "installed", "it's installed", "it is installed",
		"installed it", "i installed it", "i installed gorqlite",
		"installed gorqlite", "gorqlite is installed",
		"gorqlite installed", "done":
		return true
	}
	return false
}

// tryGorqliteInstalled is the chat-loop entry for the
// follow-up claim, seated beside the concept book.
func tryGorqliteInstalled(line string, r *LineReader) bool {
	return runGorqliteInstalled(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runGorqliteInstalled carries the follow-up: with no held
// ask, or a line outside the tight shapes, it does not claim
// at all. With a held ask it re-reads the project's go.mod —
// module present, the held ask is cleared and re-run so its
// original plan is offered behind the usual gate; module
// still absent, the answer says so plainly and the ask
// stays held.
func runGorqliteInstalled(line string, ask editAsk) bool {
	p := pendingGorqlite
	if p == nil || !gorqliteInstalledShape(line) {
		return false
	}
	tag := cTag("gollemer [composer]>")
	if !goModRequires(p.root, gorqliteModule) {
		fmt.Printf("%s I checked %s — %s isn't in its require list yet, so that %s still can't be planned. Run go get %s in the project, then tell me \"installed\" again; the ask stays held until then.\n", tag, filepath.Join(p.root, "go.mod"), gorqliteModule, p.piece, gorqliteModule)
		recordBrain("composer", false, "gorqlite follow-up: dependency still missing")
		return true
	}
	pendingGorqlite = nil
	fmt.Printf("%s %s is in go.mod now — here's that %s again:\n", tag, gorqliteModule, p.piece)
	runConcept(p.line, ask)
	return true
}
