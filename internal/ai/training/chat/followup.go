package chat

// Richer follow-ups (conversational-feel package, item 2,
// 2026-11-05): the session's own shorthand for "do that
// again, a little differently".
//
// The session context (sessionctx.go) has always let a
// sentence point at the work in progress — "to it", a bare
// file name — but repeating the work itself meant retyping
// the whole ask. Now the last landed piece can be repeated
// from three follow-up shapes:
//
//	another one [named <Name>] [in|to <target>]
//	the same [thing] [but] [for|in|to <target>]
//	same thing [but] [for|in|to <target>]
//
// and the target slot takes the spoken targets too — a
// taught name ("another one in the api file", vocab.go) or
// "the other file" (the session's second file, when exactly
// two are in play).
//
// The mechanics, pinned:
//
//   - A follow-up resolves ONLY against the session context:
//     the last landed piece's rebuildable snapshot
//     (landedPiece, sessionctx.go), its file, and the
//     session's other file. Nothing older is ever consulted,
//     and nothing is guessed — a piece whose family carries
//     no snapshot (the assembled edits, delegates, uses,
//     renames, make-and-call's compound, instantiations,
//     model-proposed declarations) gets an honest answer
//     naming the piece and asking for the full sentence.
//   - The repeat is a RECONSTRUCTION, never a replay of
//     state: the follow-up rebuilds the piece's own
//     canonical sentence — slots swapped per the follow-up's
//     own words — and runs it through the piece's own claim,
//     so every fact is re-read and re-checked, and the plan,
//     [y/n] gate, transaction and build verification are the
//     original's. A repeat that meets a fact (the name is
//     taken, the page exists) refuses exactly as the full
//     sentence would have.
//   - The name rule: a named piece repeated with a new
//     TARGET keeps its name (the swap is the ask — "the
//     same but for X"); a named piece repeated with NEITHER
//     a name nor a target asks for the name through the
//     clarifying machinery (clarify.go — the same pending
//     question, not a second system); a name given to a
//     piece that takes none (a statement, a webserver) is
//     answered honestly, never dropped in silence.
//   - A wiring repeat never carries the original's route:
//     the route was the first handler's address, not the
//     piece's shape, so the new handler wires at its own
//     default. A concept repeat never carries the original's
//     line-phrase placement: the new piece lands where a
//     fresh ask would put it.
//   - With no session context — no live project, or nothing
//     landed — the claim returns false without printing a
//     byte: follow-up lines keep today's behavior exactly.
//   - A created file repeats as a creation: its "name" is a
//     file name, asked with a target question, and a
//     webserver starter repeats as a webserver starter —
//     never silently as an empty file.

import (
	"fmt"
	"go/token"
	"strconv"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// followUpAsk is one parsed follow-up line: an optional new
// name and an optional new target, each exactly as typed.
type followUpAsk struct {
	name      string
	nameSet   bool
	target    string
	targetSet bool
}

// followUpNameToken reports whether a token can ride a
// follow-up's name slot: a declaration identifier, or a .go
// file name (a created file's new instance is named by its
// file name). One token, never a phrase.
func followUpNameToken(s string) bool {
	if s == "" || strings.ContainsAny(s, " \t") {
		return false
	}
	if token.IsIdentifier(s) {
		return true
	}
	return strings.HasSuffix(strings.ToLower(s), ".go") && !strings.Contains(s, "/")
}

// parseFollowUp reads the follow-up shapes. The name slot
// comes before the target slot (the target is always the
// tail, as in every grammar here); anything else trailing
// declines the line.
func parseFollowUp(line string) (followUpAsk, bool) {
	fields, low := splitSentence(line)
	if len(fields) == 0 {
		return followUpAsk{}, false
	}
	i := 0
	switch {
	case low[0] == "another" && len(fields) > 1 && low[1] == "one":
		i = 2
	case low[0] == "the" && len(fields) > 1 && low[1] == "same":
		i = 2
		if i < len(fields) && low[i] == "thing" {
			i++
		}
	case low[0] == "same":
		i = 1
		if i < len(fields) && low[i] == "thing" {
			i++
		}
	default:
		return followUpAsk{}, false
	}
	if i < len(fields) && low[i] == "but" {
		i++
	}
	var fu followUpAsk
	if i < len(fields) && (low[i] == "named" || low[i] == "called") {
		if i+1 >= len(fields) || !followUpNameToken(fields[i+1]) {
			return followUpAsk{}, false
		}
		fu.name, fu.nameSet = fields[i+1], true
		i += 2
	}
	if i < len(fields) {
		if !spokenTargetPrep(low[i]) && low[i] != "for" {
			return followUpAsk{}, false
		}
		i++
		if i >= len(fields) {
			return followUpAsk{}, false
		}
		fu.target, fu.targetSet = strings.Join(fields[i:], " "), true
	}
	return fu, true
}

// isFollowUpShape reports whether a line parses as a
// follow-up — the spoken-target pass (vocab.go) asks, so it
// can leave follow-up lines (and their target slots) to the
// follow-up claim itself.
func isFollowUpShape(line string) bool {
	_, ok := parseFollowUp(line)
	return ok
}

// followUpRunner is one family's own claim, in the run*
// form the claim chain uses.
type followUpRunner func(sentence string, ask editAsk) bool

// tryFollowUp is the chat-loop entry.
func tryFollowUp(line string, r *LineReader) bool {
	return runFollowUp(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

func runFollowUp(line string, ask editAsk) bool {
	fu, ok := parseFollowUp(line)
	if !ok {
		return false
	}
	tag := cTag("gollemer [composer]>")
	rootAbs, ok := vocabRootAbs()
	if !ok || !workContextLive(rootAbs) {
		return false
	}
	lp := workCtx.landed
	if lp == nil {
		// Context exists but the last piece has no sentence
		// this claim can rebuild: say so, naming the piece —
		// an honest answer, never a fabricated repeat. (No
		// piece at all means no context spoke: decline
		// silently, byte-identical.)
		if workCtx.piece == "" || workCtx.lastFile == "" {
			return false
		}
		desc := workCtx.piece
		if workCtx.pieceName != "" {
			desc += " " + workCtx.pieceName
		}
		fmt.Printf("%s The last piece that landed was %s, in %s — that one I can't repeat from a follow-up: it wasn't assembled from a sentence I can rebuild. Say the full ask and I'll do it again.\n", tag, desc, workCtx.lastFile)
		recordBrain("composer", false, "followup refused: piece not rebuildable")
		return true
	}

	// The target slot, resolved in the open: a taught name
	// or "the other file" reads as its path (the reading
	// prints), and the result must be ONE file or folder —
	// a follow-up's target is a single place.
	target := lp.file
	if fu.targetSet {
		if resolved, note := resolveSpokenTarget(fu.target, rootAbs); resolved != "" {
			fmt.Printf("%s %s\n", tag, note)
			fu.target = resolved
		}
		if len(strings.Fields(fu.target)) != 1 {
			fmt.Printf("%s Not repeating that: %q names more than one word — a follow-up's target is one file or folder, or a name you've given one (\"the api file\").\n", tag, fu.target)
			recordBrain("composer", false, "followup refused: target not one place")
			return true
		}
		target = fu.target
	}

	switch lp.family {
	case "concept":
		return followUpConcept(tag, lp, fu, target, ask)
	case "taught":
		return followUpTaught(tag, lp, fu, target, ask)
	case "server":
		if fu.nameSet {
			fmt.Printf("%s Not repeating that: a webserver takes no name — say \"another one\" and it lands as another webserver, or give the full sentence for something else.\n", tag)
			recordBrain("composer", false, "followup refused: webserver named")
			return true
		}
		return runFollowUpSentence(tag, "add a webserver to "+target, runAddServer, ask)
	case "serve":
		if fu.nameSet {
			fmt.Printf("%s Not repeating that: serving a page takes no name — say \"another one\" and %s is served from the new target too.\n", tag, lp.html)
			recordBrain("composer", false, "followup refused: serve named")
			return true
		}
		return runFollowUpSentence(tag, "serve "+lp.html+" to "+target, runServeHTML, ask)
	case "wire":
		return followUpWire(tag, lp, fu, ask)
	case "page":
		return followUpPage(tag, lp, fu, target, ask)
	case "create":
		return followUpCreate(tag, lp, fu, ask)
	}
	return false
}

// runFollowUpSentence announces the rebuilt sentence and
// runs it through its own claim: the interpretation is
// what's on show, and the claim's plan and gate do the
// rest. A sentence no claim takes (a rebuild the family
// can't read after all) is reported, never dropped.
func runFollowUpSentence(tag, sentence string, runner followUpRunner, ask editAsk) bool {
	fmt.Printf("%s I read that as: \"%s\".\n", tag, sentence)
	if !runner(sentence, ask) {
		fmt.Printf("%s That follow-up didn't run — the sentence I rebuilt for it is one no claim here takes. Say the ask in full and I'll do it.\n", tag)
		recordBrain("composer", false, "followup refused: rebuilt sentence unclaimed")
		return true
	}
	recordBrain("composer", true, "followup: "+sentence)
	return true
}

// askFollowUpName holds the clarifying name question for a
// follow-up whose piece needs a name the line didn't give:
// the answer — one identifier — completes the repeat
// through the family's own claim, exactly as if the
// follow-up had carried it.
func askFollowUpName(tag, intro string, complete func(name string) (string, followUpRunner)) {
	printClarifyIntro(tag, intro)
	printClarifyFooter("Answer with one name")
	setPendingQuestion(&pendingQuestion{
		kind: clarifyName,
		completeText: func(text string, ask2 editAsk) bool {
			sentence, runner := complete(text)
			return runFollowUpSentence(tag, sentence, runner, ask2)
		},
	})
}

// followUpConcept repeats a concept-book piece: a
// declaration (its name from the line, from the original
// when only the target swaps, or from the clarifying
// question when neither came) or a statement (its saying
// and host-function slots carried, its name refused — a
// statement takes none, the book's own doctrine).
func followUpConcept(tag string, lp *landedPiece, fu followUpAsk, target string, ask editAsk) bool {
	var entry *analyze.ConceptEntry
	entries := conceptEntries()
	for i := range entries {
		if entries[i].ID == lp.conceptID {
			entry = &entries[i]
			break
		}
	}
	if entry == nil {
		fmt.Printf("%s Not repeating that: the book doesn't carry the piece that landed last anymore, so I have no sentence to rebuild it from. Say the full ask and I'll do it.\n", tag)
		recordBrain("composer", false, "followup refused: concept gone")
		return true
	}
	canon := entry.Canonical()
	build := func(name string) string {
		s := fmt.Sprintf("add %s %s named %s", conceptArticle(canon), canon, name)
		if lp.like != "" {
			s += " like " + lp.like
		}
		return s + " to " + target
	}
	switch entry.Kind {
	case analyze.ConceptDeclaration:
		switch {
		case fu.nameSet:
			return runFollowUpSentence(tag, build(fu.name), runConcept, ask)
		case fu.targetSet:
			return runFollowUpSentence(tag, build(lp.name), runConcept, ask)
		default:
			askFollowUpName(tag,
				fmt.Sprintf("Another %s — what should I call it?", canon),
				func(name string) (string, followUpRunner) { return build(name), runConcept })
			recordBrain("composer", false, "followup question: declaration name")
			return true
		}
	case analyze.ConceptStatement:
		if fu.nameSet {
			fmt.Printf("%s Not repeating that: %s %s is a statement — a piece of code that lives inside a function — so it doesn't take a name of its own, and %q would be lost. Say \"another one\" and it lands unnamed, in the same place.\n",
				tag, conceptArticle(canon), canon, fu.name)
			recordBrain("composer", false, "followup refused: statement named")
			return true
		}
		s := fmt.Sprintf("add %s %s", conceptArticle(canon), canon)
		if lp.inFunc != "" {
			s += " in " + lp.inFunc
		}
		if lp.text != "" {
			s += " saying " + strconv.Quote(lp.text)
		}
		return runFollowUpSentence(tag, s+" to "+target, runConcept, ask)
	}
	fmt.Printf("%s Not repeating that: that piece's kind isn't one a follow-up can rebuild. Say the full ask and I'll do it.\n", tag)
	recordBrain("composer", false, "followup refused: concept kind")
	return true
}

// followUpTaught repeats a taught piece under the concept
// declarations' own name rule.
func followUpTaught(tag string, lp *landedPiece, fu followUpAsk, target string, ask editAsk) bool {
	build := func(name string) string {
		return fmt.Sprintf("add %s %s named %s to %s", conceptArticle(lp.kindPhrase), lp.kindPhrase, name, target)
	}
	switch {
	case fu.nameSet:
		return runFollowUpSentence(tag, build(fu.name), runTaughtKinds, ask)
	case fu.targetSet:
		return runFollowUpSentence(tag, build(lp.name), runTaughtKinds, ask)
	default:
		askFollowUpName(tag,
			fmt.Sprintf("Another %s — what should I call it?", lp.kindPhrase),
			func(name string) (string, followUpRunner) { return build(name), runTaughtKinds })
		recordBrain("composer", false, "followup question: taught name")
		return true
	}
}

// followUpWire repeats a wiring: a NEW handler registered
// with the project's webserver. Wiring names no file (it
// finds the server), so a target slot is answered honestly;
// the route is never carried — the new handler wires at its
// own default address.
func followUpWire(tag string, lp *landedPiece, fu followUpAsk, ask editAsk) bool {
	if fu.targetSet {
		fmt.Printf("%s Not repeating that: wiring doesn't name a file — it finds the webserver in the project and registers the handler with it. Give me the handler's name instead: \"another one named <Name>\".\n", tag)
		recordBrain("composer", false, "followup refused: wire target")
		return true
	}
	build := func(name string) string {
		return "connect the " + name + " handler to the webserver"
	}
	if fu.nameSet {
		return runFollowUpSentence(tag, build(fu.name), runWireHandler, ask)
	}
	askFollowUpName(tag,
		"Another handler to wire in — what's its name?",
		func(name string) (string, followUpRunner) { return build(name), runWireHandler })
	recordBrain("composer", false, "followup question: wire name")
	return true
}

// followUpPage repeats a page kind: the handler's name from
// the line, from the original when only the target swaps,
// or from the clarifying question — pages have default
// names, but a repeat at the same target with the default
// name would meet the original's route and page as facts,
// so the question is the honest shape here too.
func followUpPage(tag string, lp *landedPiece, fu followUpAsk, target string, ask editAsk) bool {
	info, ok := pageKindTable[lp.pageKind]
	if !ok {
		fmt.Printf("%s Not repeating that: that page kind isn't one I carry anymore. Say the full ask and I'll do it.\n", tag)
		recordBrain("composer", false, "followup refused: page kind gone")
		return true
	}
	build := func(name string) string {
		return fmt.Sprintf("%s named %s %s %s", info.canonical, name, pageTargetPrep(target), target)
	}
	switch {
	case fu.nameSet:
		return runFollowUpSentence(tag, build(fu.name), runPageFamily, ask)
	case fu.targetSet:
		return runFollowUpSentence(tag, build(lp.name), runPageFamily, ask)
	default:
		askFollowUpName(tag,
			fmt.Sprintf("Another %s — what should its handler be called?", info.display),
			func(name string) (string, followUpRunner) { return build(name), runPageFamily })
		recordBrain("composer", false, "followup question: page name")
		return true
	}
}

// followUpCreate repeats a file creation. The new instance
// is named by its file name: the line's name slot when it
// carries one, else a clarifying TARGET question (a file
// name is a path-shaped answer, not an identifier). The
// target slot, when present, names the folder the new file
// lands in; the creation's starter (a webserver) rides
// along — a repeat never silently downgrades the piece.
func followUpCreate(tag string, lp *landedPiece, fu followUpAsk, ask editAsk) bool {
	folder := lp.folder
	if fu.targetSet {
		folder = fu.target
	}
	build := func(file string) string {
		s := "create a file " + file
		if !strings.Contains(file, "/") && folder != "" {
			s += " in folder " + folder
		}
		if lp.createKind != "" {
			s += " with a " + lp.createKind
		}
		return s
	}
	if fu.nameSet {
		return runFollowUpSentence(tag, build(fu.name), runCreate, ask)
	}
	printClarifyIntro(tag, "Another file — what should it be called?")
	printClarifyFooter("Answer with a file name — \"db.go\", or \"api/db.go\" for a folder of its own")
	setPendingQuestion(&pendingQuestion{
		kind: clarifyTarget,
		completeText: func(text string, ask2 editAsk) bool {
			return runFollowUpSentence(tag, build(text), runCreate, ask2)
		},
	})
	recordBrain("composer", false, "followup question: file name")
	return true
}
