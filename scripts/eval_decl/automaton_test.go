package main

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/training/decl"
)

// Gold-stream fixtures: (input, gold declaration) pairs taken from
// the held-out set. The constraint is built from the input alone;
// the lowercased declaration's token stream must be accepted end to
// end. If the automaton rejected a gold stream, constrained decoding
// could never reproduce that target.
var goldFixtures = []struct{ input, decl string }{
	{
		"can you make me a go struct called job that has it embeds store, a http.responsewriter field called kind with a json tag and a slice of io.writer field called id",
		"type Job struct { Store; Kind http.ResponseWriter `json:\"kind\"`; ID []io.Writer }",
	},
	{
		"spec func start param ctx rune param path time.time result list[int64]",
		"func Start ( ctx rune, path time.Time ) List[int64] { panic ( \"not implemented\" ) }",
	},
	{
		"spec interface settings embed io.writer method build param [] * settings method merge param item pair[string, http.responsewriter] result error",
		"type Settings interface { io.Writer; Build ( [] * Settings ); Merge ( item Pair[string, http.ResponseWriter] ) error }",
	},
	{
		"spec func render receiver w task pointer param ctx rune param index io.writer",
		"func ( w * Task ) Render ( ctx rune, index io.Writer ) {}",
	},
	{
		"spec struct cache field label map[string]request",
		"type Cache struct { Label map[string]Request }",
	},
	{
		"spec struct item",
		"type Item struct{}",
	},
	{
		"spec func newaddress param name http.handler result address result error",
		"func NewAddress ( name http.Handler ) ( Address, error ) { panic ( \"not implemented\" ) }",
	},
}

func TestAutomatonAcceptsGoldStreams(t *testing.T) {
	for _, fx := range goldFixtures {
		con, err := buildConstraint(fx.input)
		if err != nil {
			t.Errorf("buildConstraint(%q): %v", fx.input, err)
			continue
		}
		toks := decl.SplitTokens(strings.ToLower(fx.decl))
		if !con.acceptsTokens(toks, nil) {
			t.Errorf("automaton rejects gold stream\n input: %q\n  decl: %q\n  toks: %q", fx.input, fx.decl, toks)
		}
	}
}

func TestAutomatonRejectsForeignAndWrongKind(t *testing.T) {
	con, err := buildConstraint("spec struct cache field label map[string]request")
	if err != nil {
		t.Fatal(err)
	}
	// A field name that never appears in the input is not admissible.
	foreign := decl.SplitTokens("type cache struct { email map[string]request }")
	if con.acceptsTokens(foreign, nil) {
		t.Error("automaton accepted a field name foreign to the input")
	}
	// The other kind's keyword is not admissible.
	wrongKind := decl.SplitTokens("type cache interface { label map[string]request }")
	if con.acceptsTokens(wrongKind, nil) {
		t.Error("automaton accepted an interface body for a struct constraint")
	}
	// A type atom foreign to the input is not admissible.
	foreignType := decl.SplitTokens("type cache struct { label map[string]session }")
	if con.acceptsTokens(foreignType, nil) {
		t.Error("automaton accepted a type atom foreign to the input")
	}
	// Truncated streams are not complete.
	truncated := decl.SplitTokens("type cache struct { label map[string]request")
	if con.acceptsTokens(truncated, nil) {
		t.Error("automaton accepted a truncated stream as complete")
	}
}

func TestSymbolizeWord(t *testing.T) {
	cases := map[string][]sym{
		"string;":        {{kind: sAtom, text: "string"}, {kind: sSemi}},
		"io.writer;":     {{kind: sAtom, text: "io.writer"}, {kind: sSemi}},
		");":             {{kind: sRParen}, {kind: sSemi}},
		"`json":          {{kind: sTagOpen}},
		"`;":             {{kind: sBacktick}, {kind: sSemi}},
		"`":              {{kind: sBacktick}},
		"http.handler":   {{kind: sAtom, text: "http.handler"}},
		"list":           {{kind: sAtom, text: "list"}},
		"->":             {{kind: sAtom, text: "->"}},
		"</s>":           {{kind: sEOS}},
		"UNK":            {{kind: sInvalid}},
		"map[string]int": nil, // not a vocabulary word; just exercise below
	}
	for word, want := range cases {
		if want == nil {
			continue
		}
		got := symbolizeWord(word)
		if len(got) != len(want) {
			t.Errorf("symbolizeWord(%q) = %v, want %v", word, got, want)
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("symbolizeWord(%q)[%d] = %v, want %v", word, i, got[i], want[i])
			}
		}
	}
}

func TestDeriveKindCensus(t *testing.T) {
	cases := map[string]decl.Kind{
		"spec struct user field id int":                                   decl.KindStruct,
		"spec interface store method read":                                decl.KindInterface,
		"spec func fetch result error":                                    decl.KindFunc,
		"make a struct named user with no fields":                         decl.KindStruct,
		"define an interface named store with no methods":                 decl.KindInterface,
		"could you create a function newuser which accepts a string name": decl.KindFunc,
		"method save on ( s store ): -> error":                            decl.KindFunc,
		"save ( s store ) ( )":                                            "",            // no derivation rule fires
		"fetch ( id int ) -> error":                                       decl.KindFunc, // F-T2 arrow form
		"func fetch ( id int ) error":                                     decl.KindFunc, // F-T1 signature form
	}
	for input, want := range cases {
		got, ok := deriveKind(input)
		if want == "" {
			if ok {
				t.Errorf("deriveKind(%q) = %s, want underivable", input, got)
			}
			continue
		}
		if !ok || got != want {
			t.Errorf("deriveKind(%q) = %s, %v; want %s", input, got, ok, want)
		}
	}
}
