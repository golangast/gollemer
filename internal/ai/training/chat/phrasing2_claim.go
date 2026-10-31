package chat

// The phrasing layer's seat in the claim chain (attempt 2,
// 2026-10-31). It sits after every deterministic claim and
// the speccopy proposer, and before the write boundary: a
// line reaches it only when no grammar could read it, and
// what it does with the line is recognition, never
// composition —
//
//   - the seated arm maps the line to one or two canonical
//     sentences, each already validated through the real
//     parsers, or it does not claim the line at all;
//   - the user is told, in plain words, that the phrasing
//     layer read the line, and shown the canonical
//     sentence(s) it mapped to — the interpretation is what's
//     being approved at each gate;
//   - each piece then runs through the deterministic write
//     claims exactly as if the user had typed the canonical
//     sentence: its own plan, its own [y/n] gate, its own
//     transaction and build verification;
//   - a compound is one breath: a piece declined at its gate
//     (or refused by its own flow) stops the rest, and the
//     stop is said plainly. Validation was all-or-nothing
//     before anything ran, so a half-mapped compound never
//     starts.
//
// The seat was earned on the fixed judge (run 1's held-out
// file, unmodified, slot-exact scoring) by arm B — classify +
// extract (phrasing2.go, phrasing2_extract.go): single-unseen
// 98.2% against the pre-registered 90% bar, compound-seenpair
// 86.4% against 85%, compound-unseenpair 92.0% (frontier, no
// bar), zero none-class false claims. Its artifact
// (data/phrasing_model.json) is a small JSON classifier;
// absent, the claim never fires and the boundary answers
// exactly as before.

import (
	"fmt"
	"strings"
)

// phrasingProvider is the seated arm's route: validated
// canonical pieces for a line, or ok=false. A variable so
// tests can count consultations (the deterministic-hit
// protection proof: grammar-served lines must never reach
// the provider) and so the arm choice is one assignment.
var phrasingProvider = func(line string) ([]string, bool) {
	return PhrasingRoute(line)
}

// tryPhrasingLayer is the chat-loop entry: it claims lines
// the phrasing layer can map onto canonical sentences, and
// runs each piece through the deterministic write claims on
// the chat's line reader.
func tryPhrasingLayer(line string, r *LineReader) bool {
	return runPhrasingLayer(line, func(prompt string) (string, error) {
		return r.ReadLine(prompt)
	})
}

// runPhrasingLayer carries one line from the provider's
// validated pieces to their gated executions. It returns
// false — the line falls through to the boundary, answered
// exactly as today — whenever the provider cannot map the
// line. Once it claims, it always answers: a piece its own
// claim cannot take is reported, never silently dropped.
func runPhrasingLayer(line string, ask editAsk) bool {
	pieces, ok := phrasingProvider(line)
	if !ok || len(pieces) == 0 {
		return false
	}
	tag := cTag("gollemer [composer]>")
	quoted := make([]string, len(pieces))
	for i, p := range pieces {
		quoted[i] = "\"" + p + "\""
	}
	if len(pieces) == 1 {
		fmt.Printf("%s I read that through the phrasing layer as: %s.\n", tag, quoted[0])
	} else {
		fmt.Printf("%s I read that through the phrasing layer as two asks: %s and %s.\n", tag, quoted[0], quoted[1])
	}
	for i, piece := range pieces {
		prevNote := lastNote
		if !runWriteFamilyPiece(piece, ask) {
			fmt.Printf("%s That piece — %s — is one I couldn't carry out here, so I'm stopping there.\n", tag, quoted[i])
			recordBrain("composer", false, "phrasing piece unclaimed: "+piece)
			return true
		}
		stopped := lastNote != prevNote &&
			(strings.Contains(lastNote, "declined at the gate") || strings.Contains(lastNote, "refused"))
		if stopped && i+1 < len(pieces) {
			fmt.Printf("%s Stopping there — the rest of that ask won't run.\n", tag)
			return true
		}
	}
	return true
}

// runWriteFamilyPiece runs one canonical piece through the
// deterministic write claims the phrasing families belong to,
// in the chat loop's own order. The phrasing layer itself is
// deliberately not among them (no recursion), and neither are
// the rename/create/resources claims (not phrasing families)
// or the proposer and the boundary (the piece is canonical;
// if no writer takes it, that is the layer's to report).
func runWriteFamilyPiece(piece string, ask editAsk) bool {
	switch {
	case runAssembledEdit(piece, ask):
		return true
	case runAddServer(piece, ask):
		return true
	case runDelegate(piece, ask):
		return true
	case runConcept(piece, ask):
		return true
	case runInstantiate(piece, ask):
		return true
	}
	return false
}
