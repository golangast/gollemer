package chat

import (
	"strings"
	"testing"
)

// Line placement in the phrasing extractor (2026-11-01): the
// number is a slot, carried digit-for-digit — never
// classified, never normalized away. These tests are
// model-independent: they pin the deterministic extraction
// and validation the trained classifier's answers pass
// through.

func TestExtractPlacedStatement(t *testing.T) {
	// The phrase ahead of the function slot.
	f := newFrag("stick a for loop on line 7 in Graze to trough.go")
	canon, slots, ok := extractForFamily(f, "concept-for-loop")
	if !ok {
		t.Fatalf("placed statement not extracted")
	}
	if slots["place"] != "after line 7" || slots["func"] != "Graze" || slots["target"] != "trough.go" {
		t.Errorf("slots = %v", slots)
	}
	if canon != "add a for loop in Graze to trough.go on line 7" {
		t.Errorf("canon = %q", canon)
	}
	if !validatePiece(canon, "concept-for-loop", slots) {
		t.Errorf("placed canonical does not validate: %q %v", canon, slots)
	}
	// The phrase between the function and the target clause:
	// the function binding must see past it.
	f = newFrag("stick a for loop in Graze on line 7 to trough.go")
	canon, slots, ok = extractForFamily(f, "concept-for-loop")
	if !ok || slots["func"] != "Graze" || slots["place"] != "after line 7" ||
		canon != "add a for loop in Graze to trough.go on line 7" {
		t.Errorf("mid-phrase extraction: ok=%v canon=%q slots=%v", ok, canon, slots)
	}
}

func TestExtractPlacedPrint(t *testing.T) {
	f := newFrag(`drop a print line saying "the lineup" in the Graze handler before line 3 of barn.go`)
	canon, slots, ok := extractForFamily(f, "print")
	if !ok {
		t.Fatalf("placed print not extracted")
	}
	if slots["place"] != "before line 3" || slots["text"] != "the lineup" {
		t.Errorf("slots = %v", slots)
	}
	if !strings.HasSuffix(canon, "before line 3") || !validatePiece(canon, "print", slots) {
		t.Errorf("canon = %q validates=%v", canon, validatePiece(canon, "print", slots))
	}
	// A placement inside the saying is text, never a slot.
	f = newFrag(`add a print line saying "meet on line 5" in main to jim/main.go`)
	_, slots, ok = extractForFamily(f, "print")
	if !ok || slots["text"] != "meet on line 5" {
		t.Fatalf("quoted placement misread: ok=%v slots=%v", ok, slots)
	}
	if _, has := slots["place"]; has {
		t.Errorf("quoted words became a placement slot: %v", slots)
	}
}

func TestExtractPlacedInstantiate(t *testing.T) {
	f := newFrag("add an instance of Lonny in the Graze handler before line 4")
	canon, slots, ok := extractForFamily(f, "instantiate")
	if !ok {
		t.Fatalf("placed instantiate not extracted")
	}
	if canon != "add instantiation of struct Lonny to handler Graze before line 4" {
		t.Errorf("canon = %q", canon)
	}
	if slots["place"] != "before line 4" || !validatePiece(canon, "instantiate", slots) {
		t.Errorf("slots = %v validates=%v", slots, validatePiece(canon, "instantiate", slots))
	}
	// The bare form with a tail phrase: the marker geometry
	// must see past the phrase.
	f = newFrag("add a Lonny in Graze on line 9")
	canon, slots, ok = extractForFamily(f, "instantiate")
	if !ok || slots["place"] != "after line 9" || slots["func"] != "Graze" {
		t.Errorf("bare placed instantiate: ok=%v canon=%q slots=%v", ok, canon, slots)
	}
}

func TestExtractPlaceVetoes(t *testing.T) {
	// A declaration family cannot spend a placement: its
	// canonical has no slot for it, so extraction fails
	// rather than dropping the qualifier.
	f := newFrag("add a struct named Config on line 4 to store.go")
	if _, _, ok := extractForFamily(f, "concept-struct"); ok {
		t.Errorf("declaration family claimed a placed fragment")
	}
	// A family with no placement at all, same ruling.
	f = newFrag("register thanks in Greetings on line 4")
	if _, _, ok := extractForFamily(f, "register"); ok {
		t.Errorf("register claimed a placed fragment")
	}
	// A broken number fails extraction for a family that
	// COULD spend a good one — claiming without the
	// placement would drop the qualifier.
	f = newFrag("add a for loop in main to jim/main.go on line twelve")
	if _, _, ok := extractForFamily(f, "concept-for-loop"); ok {
		t.Errorf("broken placement extracted")
	}
	f = newFrag("add a for loop in main to jim/main.go on line")
	if _, _, ok := extractForFamily(f, "concept-for-loop"); ok {
		t.Errorf("numberless placement extracted")
	}
}

// Placement digits ride through the shared-target completion:
// when the second fragment supplies the file, the completed
// first piece keeps its placement slot. (The full route needs
// the trained classifier to break family ties, so the pin
// sits at the completion seam the route itself uses.)
func TestRoutePlacedSharedPiece(t *testing.T) {
	f := newFrag("add a for loop in main on line 8")
	completed := withSharedTarget(f, "to", "jim/main.go")
	canon, slots, ok := extractForFamily(completed, "concept-for-loop")
	if !ok {
		t.Fatalf("completed placed piece not extracted: %q", completed)
	}
	if slots["place"] != "after line 8" || slots["func"] != "main" || slots["target"] != "jim/main.go" {
		t.Errorf("slots = %v", slots)
	}
	if !strings.HasSuffix(canon, "on line 8") || !validatePiece(canon, "concept-for-loop", slots) {
		t.Errorf("canon = %q validates=%v", canon, validatePiece(canon, "concept-for-loop", slots))
	}
}
