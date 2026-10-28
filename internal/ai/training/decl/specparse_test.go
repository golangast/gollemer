package decl

import (
	"strings"
	"testing"
)

// presenceTags reduces every field tag to its presence: ParseSpecText
// cannot recover tag content (SpecText writes only "tagged"), so
// round-trip comparisons run on presence.
func presenceTags(s Spec) Spec {
	out := s
	out.Fields = append([]Field(nil), s.Fields...)
	for i := range out.Fields {
		if out.Fields[i].Tag != "" {
			out.Fields[i].Tag = "tagged"
		}
	}
	return out
}

func TestParseSpecTextRoundTrip(t *testing.T) {
	cases := []string{
		"spec struct user field id int tagged field tags []string embed record",
		"spec struct cache field label map[string]request field items list [ string ]",
		"spec interface store embed io.reader method read param id int result string result error method close",
		"spec func newaddress param name http.handler result address result error",
		"spec func render receiver w task pointer param ctx context.context param items pair [ string , int ] result error",
		"spec func start param handler func ( int ) error param ch chan string result list [ int64 ]",
		"spec func fetch param http.handler param limit int result string",
		"spec struct job field store http.responsewriter tagged field data [ ] * user",
		"spec interface queue method copy param uint64 param size int",
	}
	for _, in := range cases {
		spec, err := ParseSpecText(in)
		if err != nil {
			t.Errorf("ParseSpecText(%q): %v", in, err)
			continue
		}
		// Re-render and re-parse: the recovered spec must be a fixed
		// point of SpecText (modulo tag content, which SpecText drops
		// by design).
		again, err := ParseSpecText(SpecText(spec))
		if err != nil {
			t.Errorf("ParseSpecText(SpecText(%q)): %v", in, err)
			continue
		}
		if cat, det := CompareSpec(presenceTags(again), presenceTags(spec), true); cat != CatExact {
			t.Errorf("round trip of %q: %s (%s)\n first: %+v\n again: %+v", in, cat, det, spec, again)
		}
	}
}

func TestParseSpecTextRejectsNonSpec(t *testing.T) {
	for _, in := range []string{
		"",
		"make a struct named user with a field named id of type int",
		"spec widget user",
		"spec struct",
		"spec struct user bogus id int",
		"spec struct user field",
		"spec func f receiver s store sideways",
		"spec struct user field x map[string]",
	} {
		if _, err := ParseSpecText(in); err == nil {
			t.Errorf("ParseSpecText(%q) = nil error, want rejection", in)
		}
	}
}

// TestParseSpecTextOverGeneratedSpecs is the property test: for every
// spec the generator can produce, parsing its SpecText must recover
// the spec (structurally exact, tag presence only — content is not
// in the spec language).
func TestParseSpecTextOverGeneratedSpecs(t *testing.T) {
	ds, err := Generate(DefaultGenConfig())
	if err != nil {
		t.Fatal(err)
	}
	all := append(append([]Spec{}, ds.TrainSpecs...), ds.HeldoutSpecs...)
	if len(all) == 0 {
		t.Fatal("generator produced no specs")
	}
	for _, want := range all {
		text := SpecText(want)
		got, err := ParseSpecText(text)
		if err != nil {
			t.Fatalf("ParseSpecText(%q): %v", text, err)
		}
		if cat, det := CompareSpec(presenceTags(got), presenceTags(LowerSpec(want)), true); cat != CatExact {
			t.Fatalf("ParseSpecText(%q): %s (%s)\n got: %+v\nwant: %+v", text, cat, det, got, LowerSpec(want))
		}
	}
	if !strings.Contains(SpecText(all[0]), "spec ") {
		t.Fatal("SpecText sanity failed")
	}
}
