package chat

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Line placement in the deterministic grammar (2026-11-01):
// "on line N" / "at line N" / "after line N" / "before
// line N" rides the slot zone in either position or the tail
// behind the target — and the concept named "print line"
// never has its own "line" misread as the phrase.

func TestParseConceptLinePlacement(t *testing.T) {
	book := defaultBookEntries(t)
	cases := []struct {
		line   string
		id     string
		place  int
		before bool
	}{
		// The pinned trap: concept name contains "line".
		{"add a print line on line 12 in main to jim/main.go", "print-line", 12, false},
		{"add a print line in main to jim/main.go on line 12", "print-line", 12, false},
		{"add a print line before line 3 in main to jim/main.go", "print-line", 3, true},
		{"add a log line on line 5 to jim/main.go", "log-line", 5, false},
		// The phrase ahead of the function slot.
		{"add a for loop on line 8 in main to jim/main.go", "for-loop", 8, false},
		{"add a for loop at line 8 in main to jim/main.go", "for-loop", 8, false},
		{"add a for loop after line 8 in main to jim/main.go", "for-loop", 8, false},
		// The phrase behind the function slot, before the target.
		{"add a for loop in main before line 8 to jim/main.go", "for-loop", 8, true},
		// No function slot at all.
		{"add a for loop on line 8 to jim/main.go", "for-loop", 8, false},
		{"add a for loop to jim/main.go before line 8", "for-loop", 8, true},
		// Declarations take the phrase at file scope.
		{"add a struct named Config on line 4 to store.go", "struct", 4, false},
		{"add a struct named Config to store.go after line 4", "struct", 4, false},
		// A saying and a placement together.
		{`add a print line saying "hi" on line 2 in main to jim/main.go`, "print-line", 2, false},
	}
	for _, c := range cases {
		req, ok := parseConceptFrom(c.line, book)
		if !ok {
			t.Errorf("parseConcept(%q) not claimed", c.line)
			continue
		}
		if req.entry.ID != c.id || req.placeLine != c.place || req.placeBefore != c.before || req.placeBad {
			t.Errorf("parseConcept(%q) = {id:%q place:%d before:%v bad:%v}, want id %q place %d before %v",
				c.line, req.entry.ID, req.placeLine, req.placeBefore, req.placeBad, c.id, c.place, c.before)
		}
	}
}

// A started phrase with a broken number still claims — the
// refusal is runConcept's, in placement terms.
func TestParseConceptLinePlacementBroken(t *testing.T) {
	book := defaultBookEntries(t)
	for _, line := range []string{
		"add a print line on line 0 in main to jim/main.go",
		"add a print line on line -3 in main to jim/main.go",
		"add a print line on line twelve in main to jim/main.go",
		"add a print line on line in main to jim/main.go",
		"add a print line in main to jim/main.go on line",
		"add a print line in main to jim/main.go on line 0",
		"add a print line in main to jim/main.go before line nine",
	} {
		req, ok := parseConceptFrom(line, book)
		if !ok {
			t.Errorf("parseConcept(%q) not claimed — a started phrase must claim", line)
			continue
		}
		if !req.placeBad {
			t.Errorf("parseConcept(%q) claimed without placeBad: %+v", line, req)
		}
	}
}

func TestParseConceptLinePlacementNonClaims(t *testing.T) {
	book := defaultBookEntries(t)
	for _, line := range []string{
		// A second placement phrase is not this grammar's.
		"add a print line on line 5 on line 6 to jim/main.go",
		"add a print line on line 5 to jim/main.go on line 6",
		// A preposition without "line" right behind it.
		"add a print line on the line 5 to jim/main.go",
		"add a print line after 5 in main to jim/main.go",
		// Trailing words past the number.
		"add a print line on line 5 please to jim/main.go",
	} {
		if req, ok := parseConceptFrom(line, book); ok {
			t.Errorf("parseConcept(%q) claimed %+v — not the composer's line", line, req)
		}
	}
}

// End to end through the chain: the plan quotes the anchor,
// the piece lands where the line said, and an impossible line
// refuses in placement terms writing nothing.
func TestConceptLinePlacementLiveSession(t *testing.T) {
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	createJimThroughTheChain(t, profile)
	src := "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tfmt.Println(\"one\")\n\tfmt.Println(\"two\")\n}\n"
	if err := os.WriteFile(filepath.Join(root, "jim", "main.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	var handled bool
	out := captureStdout(t, func() {
		handled = claimChain("add a print line after line 6 in main to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("placed line not claimed:\n%s", out)
	}
	if !strings.Contains(out, "after line 6: `fmt.Println(\"one\")`") {
		t.Errorf("plan does not quote the anchor line:\n%s", out)
	}
	got := readBack(t, root, "jim/main.go")
	one := strings.Index(got, `fmt.Println("one")`)
	placed := strings.Index(got, `fmt.Println("")`)
	two := strings.Index(got, `fmt.Println("two")`)
	if !(one >= 0 && placed > one && two > placed) {
		t.Errorf("piece did not land between lines 6 and 7:\n%s", got)
	}

	out = captureStdout(t, func() {
		handled = claimChain("add a print line on line 99 in main to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("impossible line not claimed:\n%s", out)
	}
	if !strings.Contains(out, "past the end") || !strings.Contains(out, "9 lines") {
		t.Errorf("past-end refusal does not name the file's length:\n%s", out)
	}
	if got := readBack(t, root, "jim/main.go"); strings.Count(got, "Println") != 3 {
		t.Errorf("refused placement wrote anyway:\n%s", got)
	}

	out = captureStdout(t, func() {
		handled = claimChain("add a print line on line twelve in main to jim/main.go", yes, profile)
	})
	if !handled {
		t.Fatalf("broken line phrase not claimed:\n%s", out)
	}
	if !strings.Contains(out, "isn't a number I can aim at") {
		t.Errorf("broken-number refusal is not in placement terms:\n%s", out)
	}
}

// Instantiate takes the phrase at its tail.
func TestParseInstantiateLinePlacement(t *testing.T) {
	cases := []struct {
		line   string
		place  int
		before bool
	}{
		{"add a Jim in Jake on line 5", 5, false},
		{"add an instance of Jim in Jake before line 3", 3, true},
		{"add an instantiation of struct Jim in the handler Jake named jim at line 7", 7, false},
	}
	for _, c := range cases {
		req, ok := parseInstantiate(c.line)
		if !ok {
			t.Errorf("parseInstantiate(%q) not claimed", c.line)
			continue
		}
		if req.placeLine != c.place || req.placeBefore != c.before || req.placeBad {
			t.Errorf("parseInstantiate(%q) = place %d before %v bad %v, want %d/%v",
				c.line, req.placeLine, req.placeBefore, req.placeBad, c.place, c.before)
		}
	}
	// CanonicalSlots carries the placement as its own slot.
	if ps, ok := CanonicalSlots("add instantiation of struct Jim to handler Jake on line 5"); !ok ||
		ps.Family != "instantiate" || ps.Slots["place"] != "after line 5" {
		t.Errorf("CanonicalSlots instantiate place = %+v, %v", ps, ok)
	}
	if ps, ok := CanonicalSlots("add a print line on line 12 in main to jim/main.go"); !ok ||
		ps.Family != "print" || ps.Slots["place"] != "after line 12" {
		t.Errorf("CanonicalSlots print place = %+v, %v", ps, ok)
	}
	if _, ok := CanonicalSlots("add a print line on line 0 in main to jim/main.go"); ok {
		t.Errorf("CanonicalSlots claimed a broken placement")
	}
}
