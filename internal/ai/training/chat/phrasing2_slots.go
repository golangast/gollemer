package chat

// Slot-exact scoring for the phrasing-layer probe, attempt 2
// (2026-10-31). Run 1 scored the mapper by raw string equality
// against one gold canonical sentence per ask — but several
// families have more than one canonical form the shipped
// parsers accept (add-field's two forms, register's two,
// clone's two), and a prediction landing in the OTHER valid
// form scored as a miss even though the layer above would have
// executed exactly the right edit. The declared metric for
// attempt 2 is therefore SLOT-EXACT: a canonical sentence is
// run through the REAL deterministic parsers (the same ones the
// claim chain uses), the resulting (family, slot map) is
// compared against the gold sentence's (family, slot map), and
// an unparseable prediction scores wrong. String-exact is
// still reported alongside, so run 1's 59.0% stays comparable.
//
// canonicalSlots is deliberately just a dispatch over the
// shipped parsers, in an order where no family's canonical can
// be stolen by another family's grammar (the bespoke shapes
// first, the concept book last — the claim chain's own order).

import (
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

// PhrasingPieceSlots is one canonical piece read by the real
// parsers: its phrasing family and the slot values the parsers
// bound, keyed by role.
type PhrasingPieceSlots struct {
	Family string
	Slots  map[string]string
}

// CanonicalSlots parses one canonical sentence (a single piece,
// no " ## " separator) through the deterministic parsers and
// returns its phrasing family and slot map. ok is false when no
// parser accepts the sentence — for scoring, that prediction is
// simply wrong; for the layer, it is the validation failure
// that keeps the line unclaimed.
func CanonicalSlots(sentence string) (PhrasingPieceSlots, bool) {
	s := strings.TrimSpace(sentence)
	if s == "none" {
		return PhrasingPieceSlots{Family: "none", Slots: map[string]string{}}, true
	}
	if req, ok := parseInstantiate(s); ok {
		slots := map[string]string{"type": req.typeName, "func": req.funcName}
		if req.varName != "" {
			slots["var"] = req.varName
		}
		return PhrasingPieceSlots{Family: "instantiate", Slots: slots}, true
	}
	if file, source, wrapper, ok := parseDelegate(s); ok {
		slots := map[string]string{"file": file, "source": source}
		if wrapper != "" {
			slots["wrapper"] = wrapper
		}
		return PhrasingPieceSlots{Family: "delegate", Slots: slots}, true
	}
	if req, ok := parseAssembledEdit(s); ok {
		switch req.Kind {
		case analyze.EditAddField:
			slots := map[string]string{
				"field": req.FieldName, "type": req.FieldType, "struct": req.StructName,
			}
			if req.AfterField != "" {
				slots["after"] = req.AfterField
			}
			return PhrasingPieceSlots{Family: "add-field", Slots: slots}, true
		case analyze.EditAddCase:
			slots := map[string]string{"func": req.FuncName, "case": req.CaseExpr}
			if req.SwitchOn != "" {
				slots["switchOn"] = req.SwitchOn
			}
			return PhrasingPieceSlots{Family: "add-case", Slots: slots}, true
		case analyze.EditRegister:
			return PhrasingPieceSlots{Family: "register", Slots: map[string]string{
				"table": req.TableName, "key": req.Key,
			}}, true
		case analyze.EditAddFunc:
			return PhrasingPieceSlots{Family: "clone", Slots: map[string]string{
				"sibling": req.SiblingName, "new": req.NewName,
			}}, true
		}
		// Other assembled kinds (signature changes) are not
		// phrasing families; fall through so the sentence is
		// judged by the remaining parsers or not at all.
	}
	if file, ok := parseAddServer(s); ok && file != "" {
		return PhrasingPieceSlots{Family: "add-server", Slots: map[string]string{"file": file}}, true
	}
	if req, ok := parseConcept(s); ok {
		family := "concept-" + req.entry.ID
		if req.entry.ID == "print-line" {
			family = "print"
		}
		slots := map[string]string{}
		if req.name != "" {
			slots["name"] = req.name
		}
		if req.textSet {
			slots["text"] = req.text
		}
		if req.like != "" {
			slots["like"] = req.like
		}
		if req.inFunc != "" {
			slots["func"] = req.inFunc
		}
		if req.target != "" {
			slots["target"] = req.target
		}
		return PhrasingPieceSlots{Family: family, Slots: slots}, true
	}
	return PhrasingPieceSlots{}, false
}

// CanonicalPiecesSlots splits a (possibly compound) canonical
// output on the " ## " separator and reads every piece.
// ok is false when any piece fails to parse.
func CanonicalPiecesSlots(output string) ([]PhrasingPieceSlots, bool) {
	if strings.TrimSpace(output) == "none" {
		return nil, true
	}
	parts := strings.Split(output, " ## ")
	out := make([]PhrasingPieceSlots, 0, len(parts))
	for _, p := range parts {
		ps, ok := CanonicalSlots(p)
		if !ok {
			return nil, false
		}
		out = append(out, ps)
	}
	return out, true
}

// SlotsEqual reports whether two slot maps bind the same roles
// to the same values.
func SlotsEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if bv, ok := b[k]; !ok || bv != v {
			return false
		}
	}
	return true
}

// PiecesSlotExact is the attempt-2 verdict metric for one item:
// the predicted pieces and the gold pieces must agree in count,
// and each aligned pair must agree on family and every slot.
func PiecesSlotExact(got, gold []PhrasingPieceSlots) bool {
	if len(got) != len(gold) {
		return false
	}
	for i := range gold {
		if got[i].Family != gold[i].Family || !SlotsEqual(got[i].Slots, gold[i].Slots) {
			return false
		}
	}
	return true
}
