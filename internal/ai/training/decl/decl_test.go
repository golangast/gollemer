package decl

import (
	"math/rand"
	"strings"
	"testing"
)

func TestPoolInjectivity(t *testing.T) {
	if err := AssertPoolInjectivity(); err != nil {
		t.Fatal(err)
	}
}

func TestCanonicalExpr(t *testing.T) {
	cases := map[string]string{
		"[]*User":          "[]*User",
		"map[string] int":  "map[string]int",
		"time.Time":        "time.Time",
		"func(int)error":   "func(int) error",
		"Pair[string,int]": "Pair[string, int]",
	}
	for in, want := range cases {
		ref, err := MakeTypeRef(in)
		if err != nil {
			t.Fatalf("MakeTypeRef(%q): %v", in, err)
		}
		if ref.Expr != want {
			t.Errorf("MakeTypeRef(%q).Expr = %q, want %q", in, ref.Expr, want)
		}
	}
	classes := map[string]string{
		"int": ClassBuiltin, "User": ClassLocal, "time.Time": ClassImported,
		"[]User": ClassComposite, "*Config": ClassComposite,
	}
	for in, want := range classes {
		ref, _ := MakeTypeRef(in)
		if ref.Class != want {
			t.Errorf("MakeTypeRef(%q).Class = %q, want %q", in, ref.Class, want)
		}
	}
}

func sampleSpecs() []Spec {
	return []Spec{
		{Kind: KindStruct, Name: "User", Fields: []Field{
			{Name: "ID", Type: MustTypeRef("int"), Tag: `json:"id"`},
			{Name: "Name", Type: MustTypeRef("string")},
			{Name: "Tags", Type: MustTypeRef("[]string")},
			{Name: "Owner", Type: MustTypeRef("*Address")},
			{Embedded: true, Type: MustTypeRef("Record")},
		}},
		{Kind: KindStruct, Name: "Cache"},
		{Kind: KindInterface, Name: "Store", Embeds: []TypeRef{MustTypeRef("io.Reader")}, Methods: []Method{
			{Name: "Read", Params: []Param{{Name: "p", Type: MustTypeRef("[]byte")}}, Results: []TypeRef{MustTypeRef("int"), MustTypeRef("error")}},
			{Name: "Close", Results: []TypeRef{MustTypeRef("error")}},
		}},
		{Kind: KindInterface, Name: "Catalog"},
		{Kind: KindFunc, Name: "Fetch", Params: []Param{{Name: "id", Type: MustTypeRef("int")}}, Results: []TypeRef{MustTypeRef("string"), MustTypeRef("error")}},
		{Kind: KindFunc, Name: "Save", Receiver: &Receiver{Name: "s", Type: "Store", Pointer: true},
			Params: []Param{{Name: "item", Type: MustTypeRef("Record")}}, Results: []TypeRef{MustTypeRef("error")}},
		{Kind: KindFunc, Name: "NewClient", Results: []TypeRef{MustTypeRef("*Client")}},
		{Kind: KindFunc, Name: "Reset", Receiver: &Receiver{Name: "c", Type: "Cache"}},
	}
}

func TestRoundTripSampleSpecs(t *testing.T) {
	stats := NewFilterStats()
	for _, spec := range sampleSpecs() {
		if !Validate(spec, stats) {
			t.Errorf("Validate(%+v) = false, stats: %+v", spec, stats.Snapshot())
		}
	}
}

func TestFilterRejections(t *testing.T) {
	cases := []struct {
		name   string
		spec   Spec
		reason string
	}{
		{"keyword name", Spec{Kind: KindStruct, Name: "struct"}, ReasonInvalidSpecName},
		{"bad kind", Spec{Kind: "weird", Name: "User"}, ReasonUnknownKind},
		{"dup fields", Spec{Kind: KindStruct, Name: "User", Fields: []Field{
			{Name: "ID", Type: MustTypeRef("int")}, {Name: "ID", Type: MustTypeRef("string")}}}, ReasonDuplicateMember},
		{"dup params", Spec{Kind: KindFunc, Name: "Fetch", Params: []Param{
			{Name: "id", Type: MustTypeRef("int")}, {Name: "id", Type: MustTypeRef("int")}}}, ReasonDuplicateMember},
		{"dup methods", Spec{Kind: KindInterface, Name: "Store", Methods: []Method{{Name: "Read"}, {Name: "Read"}}}, ReasonDuplicateMember},
		{"non-type expr", Spec{Kind: KindStruct, Name: "User", Fields: []Field{
			{Name: "X", Type: TypeRef{Expr: "a + b"}}}}, ReasonNonTypeExpr},
	}
	for _, tc := range cases {
		stats := NewFilterStats()
		if Validate(tc.spec, stats) {
			t.Errorf("%s: Validate = true, want rejection %s", tc.name, tc.reason)
			continue
		}
		snap := stats.Snapshot()
		if snap.Rejections[tc.reason] != 1 {
			t.Errorf("%s: rejections = %v, want exactly %s", tc.name, snap.Rejections, tc.reason)
		}
		if snap.Samples[tc.reason] == "" {
			t.Errorf("%s: no sample recorded for %s", tc.name, tc.reason)
		}
	}
}

func TestDecodeFeasible(t *testing.T) {
	// Three same-typed params produce the token bigram ("int", ",")
	// twice — the greedy decoder can never emit it.
	bad := Spec{Kind: KindFunc, Name: "Sum", Params: []Param{
		{Name: "a", Type: MustTypeRef("int")},
		{Name: "b", Type: MustTypeRef("int")},
		{Name: "c", Type: MustTypeRef("int")},
	}}
	if DecodeFeasible(strings.ToLower(RenderDecl(bad))) {
		t.Error("DecodeFeasible(three int params) = true, want false")
	}
	good := Spec{Kind: KindFunc, Name: "Sum", Params: []Param{
		{Name: "a", Type: MustTypeRef("int")},
		{Name: "b", Type: MustTypeRef("string")},
		{Name: "c", Type: MustTypeRef("int")},
	}}
	if !DecodeFeasible(strings.ToLower(RenderDecl(good))) {
		t.Error("DecodeFeasible(mixed params) = false, want true")
	}
}

func TestSpecTextAndFrames(t *testing.T) {
	spec := sampleSpecs()[0]
	st := SpecText(spec)
	for _, want := range []string{"spec struct user", "field id int tagged", "field tags []string", "embed record"} {
		if !strings.Contains(st, want) {
			t.Errorf("SpecText = %q, missing %q", st, want)
		}
	}
	// Every frame of every shape fills without leftover placeholders.
	for _, s := range sampleSpecs() {
		for _, reserved := range []bool{false, true} {
			for _, f := range FramesFor(s, reserved) {
				out := Fill(f, s)
				if strings.Contains(out, "{") || strings.Contains(out, "}") {
					t.Errorf("frame %s on %+v left placeholders: %q", f.ID, s.Kind, out)
				}
				if n := len(SplitTokens(out)); n > MaxInputTokens {
					t.Errorf("frame %s input too long: %d tokens", f.ID, n)
				}
			}
		}
	}
}

func TestComboSplitDisjoint(t *testing.T) {
	// The same combination can never straddle the split: hash is a
	// pure function of the combination key.
	for _, s := range sampleSpecs() {
		canon, err := Canonicalize(s)
		if err != nil {
			t.Fatal(err)
		}
		if heldoutCombo(canon) != heldoutCombo(canon) {
			t.Fatal("heldoutCombo is not deterministic")
		}
	}
	a := Spec{Kind: KindStruct, Name: "User", Fields: []Field{{Name: "ID", Type: MustTypeRef("int")}}}
	b := Spec{Kind: KindStruct, Name: "User", Fields: []Field{{Name: "Name", Type: MustTypeRef("int")}}}
	// Same name + same member type multiset = same combination, even
	// with different field names.
	if comboKey(a) != comboKey(b) {
		t.Errorf("comboKey differs for same combination:\n%s\n%s", comboKey(a), comboKey(b))
	}
}

func TestCompareSpec(t *testing.T) {
	want := LowerSpec(sampleSpecs()[0])
	raw, err := ExtractSpec(RenderDecl(sampleSpecs()[0]))
	if err != nil {
		t.Fatal(err)
	}
	got := LowerSpec(raw)
	if cat, det := CompareSpec(got, want, true); cat != CatExact {
		t.Errorf("self-compare = %s (%s), want exact", cat, det)
	}
	// Drop a field from the rendering.
	broken := "type user struct { id int; name string }"
	got2, err := ExtractSpec(broken)
	if err != nil {
		t.Fatal(err)
	}
	if cat, _ := CompareSpec(got2, want, true); cat != CatMemberCount {
		t.Errorf("dropped fields = %s, want %s", cat, CatMemberCount)
	}
}

// Regression: parameter lists must be uniformly named or uniformly
// unnamed. A func declaration with a mix does not parse ("missing
// parameter name"), and in an interface method a leading unnamed
// parameter is silently absorbed as a NAME for the next parameter's
// type — Copy(uint64, size Token) declares uint64 and size, both
// Token. The filter must reject both shapes, and the sampler must
// never produce them.
func TestMixedParamsRejected(t *testing.T) {
	mixedFunc := Spec{Kind: KindFunc, Name: "Copy", Params: []Param{
		{Type: MustTypeRef("uint64")},
		{Name: "size", Type: MustTypeRef("Token")},
	}}
	if Validate(mixedFunc, NewFilterStats()) {
		t.Error("mixed-param func passed the filter")
	}
	mixedMethod := Spec{Kind: KindInterface, Name: "Cache", Methods: []Method{
		{Name: "Copy", Params: []Param{
			{Type: MustTypeRef("uint64")},
			{Name: "size", Type: MustTypeRef("Token")},
		}},
	}}
	if Validate(mixedMethod, NewFilterStats()) {
		t.Error("mixed-param interface method passed the filter")
	}
	s := &sampler{rng: rand.New(rand.NewSource(3))}
	for i := 0; i < 500; i++ {
		for _, kind := range []Kind{KindFunc, KindInterface} {
			spec := s.spec(kind)
			var lists [][]Param
			if kind == KindFunc {
				lists = append(lists, spec.Params)
			}
			for _, m := range spec.Methods {
				lists = append(lists, m.Params)
			}
			for _, ps := range lists {
				named, unnamed := 0, 0
				for _, p := range ps {
					if p.Name == "" {
						unnamed++
					} else {
						named++
					}
				}
				if named > 0 && unnamed > 0 {
					t.Fatalf("sampler produced mixed params: %+v", ps)
				}
			}
		}
	}
}

func TestCheckCompiles(t *testing.T) {
	for _, spec := range sampleSpecs() {
		ok, note := checkCompiles(LowerSpec(spec))
		if !ok {
			t.Errorf("checkCompiles(%s %s) = false: %s", spec.Kind, spec.Name, note)
		}
	}
}
