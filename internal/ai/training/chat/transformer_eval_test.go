package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/transformer"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// evalTestVocab builds a vocabulary from the given texts with the same
// helper the trainer and eval harness share.
func evalTestVocab(texts []string) *vocab.Vocabulary {
	pairs := make([]ChatPair, len(texts))
	for i, s := range texts {
		pairs[i] = ChatPair{Input: s, Output: s}
	}
	return buildTransformerVocab(pairs)
}

func TestCanonicalOutputForm(t *testing.T) {
	cases := []struct{ in, want string }{
		{"command.go:1386", "command . go : 1 3 8 6"},
		{"examples/foo-bar/main.go", "examples / foo - bar / main . go"},
		{"ContainsGroup", "contains group"},
		{"HTTPServer", "http server"},
		{"", ""},
	}
	for _, c := range cases {
		if got := canonicalOutputForm(c.in); got != c.want {
			t.Errorf("canonicalOutputForm(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestOutputExpressible(t *testing.T) {
	v := evalTestVocab([]string{"command.go:1386", "the file lives in examples/main.go"})
	if !outputExpressible("command.go:1386", v) {
		t.Error("output built from training pieces should be expressible")
	}
	if !outputExpressible("main.go:13", v) {
		t.Error("recombination of seen pieces (main.go:13) should be expressible")
	}
	if outputExpressible("nonexistent.go:1", v) {
		t.Error("output with an unseen piece (nonexistent) must be inexpressible")
	}
	if !outputExpressible("", v) {
		t.Error("empty output is vacuously expressible (matches the trainer's coverage check)")
	}
}

func TestExactMatchDetokenizationRoundTrip(t *testing.T) {
	texts := []string{"command.go:1386", "examples/tui-daemon-combo/main.go:23"}
	v := evalTestVocab(texts)

	// Round-trip: encode an expressible output to IDs (the sequence a
	// perfect model would emit) and confirm the scorer calls it a match.
	for _, want := range texts {
		ids := encodeSubword(want, v)
		if !exactMatchOutput(ids, want, v) {
			t.Errorf("round-trip for %q failed: decoded %q, canonical %q",
				want, v.Decode(ids), canonicalOutputForm(want))
		}
		if exactMatchOutput(ids, "other.go:9", v) {
			t.Errorf("IDs for %q must not match a different expected output", want)
		}
	}

	// An inexpressible output encodes its unseen piece to UNK, so the
	// decoded form can never equal the canonical form: it can never be
	// an exact match, which is exactly why the verdict is read on the
	// expressible subset.
	ids := encodeSubword("unseen-file.go:1", v)
	if exactMatchOutput(ids, "unseen-file.go:1", v) {
		t.Error("inexpressible output must not exact-match (its unseen piece decodes to UNK)")
	}
	if !strings.Contains(v.Decode(ids), "UNK") {
		t.Errorf("expected UNK in decoded inexpressible output, got %q", v.Decode(ids))
	}
}

func TestArgmaxFirst(t *testing.T) {
	if got := argmaxFirst([]float32{0.1, 0.9, 0.4}); got != 1 {
		t.Errorf("argmaxFirst = %d, want 1", got)
	}
	if got := argmaxFirst([]float32{0.5, 0.5, 0.1}); got != 0 {
		t.Errorf("argmaxFirst tie = %d, want first index 0", got)
	}
	if got := argmaxFirst([]float32{-3, -1, -2}); got != 1 {
		t.Errorf("argmaxFirst negatives = %d, want 1", got)
	}
}

// TestGreedyGenerateRunsUntrained is a plumbing test for the decode loop:
// a tiny randomly-initialized transformer must generate without error and
// respect the max-length cap (its output content is meaningless).
func TestGreedyGenerateRunsUntrained(t *testing.T) {
	v := evalTestVocab([]string{"which file is f in", "f.go:1"})
	cfg := transformer.TransformerConfig{
		DModel:    32,
		NumHeads:  4,
		NumLayers: 1,
		DFF:       64,
		MaxLen:    32,
		VocabSize: v.Size(),
	}
	model, err := transformer.NewTransformer(cfg)
	if err != nil {
		t.Fatalf("NewTransformer: %v", err)
	}
	srcIDs := encodeSubword("which file is f in", v)
	out, err := greedyGenerateTransformer(model, v, srcIDs, 5)
	if err != nil {
		t.Fatalf("greedyGenerateTransformer: %v", err)
	}
	if len(out) > 5 {
		t.Errorf("generated %d tokens, cap was 5", len(out))
	}
	for _, id := range out {
		if id < 0 || id >= v.Size() {
			t.Errorf("generated id %d outside vocab size %d", id, v.Size())
		}
	}
}
