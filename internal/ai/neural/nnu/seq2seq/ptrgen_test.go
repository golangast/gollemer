package seq2seq

import (
	"math"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// The mixture distribution must be a distribution: generation,
// copy, and the final mix each sum to 1.
func TestPtrGenMixtureSumsToOne(t *testing.T) {
	const H = 8
	pg := NewPointerGen(H)
	dec := []float32{0.3, -0.2, 0.5, 0.1, -0.4, 0.2, 0.0, 0.7}
	enc := [][]float32{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8},
		{-0.5, 0.4, -0.3, 0.2, -0.1, 0.0, 0.1, -0.2},
		{0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2},
		{0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.1},
	}
	inputIDs := []int{4, 5, 4, 6}
	logits := []float32{0.2, -1.0, 0.5, 2.0, -0.5, 0.0, 1.0}
	final, cache := pg.ptrGenStep(dec, enc, inputIDs, len(inputIDs), logits)
	sum := func(xs []float32) float64 {
		var s float64
		for _, x := range xs {
			s += float64(x)
		}
		return s
	}
	if got := sum(final); math.Abs(got-1) > 1e-5 {
		t.Fatalf("mixture distribution sums to %v, want 1", got)
	}
	if got := sum(cache.pVocab); math.Abs(got-1) > 1e-5 {
		t.Fatalf("generation distribution sums to %v, want 1", got)
	}
	if got := sum(cache.copyDist); math.Abs(got-1) > 1e-5 {
		t.Fatalf("copy distribution sums to %v, want 1", got)
	}
	if cache.pGen < 0.5 || cache.pGen > 0.99 {
		t.Fatalf("initial p_gen = %v, want the gate to start mostly-generative (~0.88)", cache.pGen)
	}
}

// With the gate forced shut, the output must be carried by the copy
// path: a token that is in the input wins over the generation
// distribution's own top choice, which is absent from the input.
func TestPtrGenCopyPathReproducesInputToken(t *testing.T) {
	const H = 4
	pg := NewPointerGen(H)
	pg.Bg.Data[0] = -20 // p_gen ~= 0: copy carries the step
	dec := []float32{0.1, 0.2, 0.3, 0.4}
	enc := [][]float32{
		{0.1, 0.0, 0.0, 0.0},
		{0.0, 0.1, 0.0, 0.0},
		{0.0, 0.0, 0.1, 0.0},
		{0.0, 0.0, 0.0, 0.1},
		{0.2, 0.2, 0.2, 0.2},
	}
	inputIDs := []int{7, 7, 7, 3, 4}
	// Generation overwhelmingly favors token 2, which is NOT in the input.
	logits := []float32{0, 0, 10, 0, 0, 0, 0, 0, 0}
	final, cache := pg.ptrGenStep(dec, enc, inputIDs, len(inputIDs), logits)
	if cache.pGen > 0.01 {
		t.Fatalf("p_gen = %v with bias -20, want < 0.01", cache.pGen)
	}
	best := 0
	for i := 1; i < len(final); i++ {
		if final[i] > final[best] {
			best = i
		}
	}
	if best != 7 {
		t.Fatalf("mixture argmax = token %d, want the copied input token 7 (final=%v)", best, final)
	}
	if cache.copyDist[7] < 0.5 {
		t.Fatalf("copyDist[7] = %v, want the repeated input token to hold most copy mass", cache.copyDist[7])
	}
}

// The toy that decides whether the mechanism is real: a name that
// appears in training INPUTS but never as a training TARGET must be
// reproducible at the copy slot after training. A pure generator has
// no trained reason to emit it there; the copy path does.
func TestPtrGenToyLearnsToCopyHeldOutName(t *testing.T) {
	v := vocab.NewVocabulary()
	for _, w := range []string{"copy", "mention", "today", "please", "spec", "noted", "amber", "birch", "cobra", "zephyr"} {
		v.AddToken(w)
	}
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		t.Fatal(err)
	}
	model, err := NewSeq2Seq(v.Size(), v.Size(), 32, 64, tok, v)
	if err != nil {
		t.Fatal(err)
	}
	model.Decoder.MoE = nil
	model.Decoder.PtrGen = NewPointerGen(64)

	type pair struct{ in, out string }
	var pairs []pair
	for _, n := range []string{"amber", "birch", "cobra"} {
		pairs = append(pairs,
			pair{"copy " + n, "spec " + n},
			pair{"copy " + n + " please", "spec " + n},
			pair{"please copy " + n, "spec " + n},
		)
	}
	// zephyr is trained as an input token only — never a target.
	pairs = append(pairs,
		pair{"mention zephyr today", "spec noted"},
		pair{"today mention zephyr", "spec noted"},
		pair{"mention zephyr please", "spec noted"},
	)

	encode := func(s string, target bool) []int {
		ids, err := tok.Encode(s)
		if err != nil {
			t.Fatal(err)
		}
		if !target {
			return ids
		}
		out := []int{v.BosID}
		out = append(out, ids...)
		return append(out, v.EosID)
	}
	var inputs, targets [][]int
	maxIn, maxTg := 0, 0
	for _, p := range pairs {
		in := encode(NormalizeQuery(p.in), false)
		tg := encode(strings.ToLower(p.out), true)
		inputs = append(inputs, in)
		targets = append(targets, tg)
		if len(in) > maxIn {
			maxIn = len(in)
		}
		if len(tg) > maxTg {
			maxTg = len(tg)
		}
	}
	pad := func(rows [][]int, n int) [][]int {
		out := make([][]int, len(rows))
		for i, r := range rows {
			row := make([]int, n)
			copy(row, r)
			for k := len(r); k < n; k++ {
				row[k] = v.PaddingTokenID
			}
			out[i] = row
		}
		return out
	}
	inputs, targets = pad(inputs, maxIn), pad(targets, maxTg)

	opt := nn.NewOptimizer(model.Parameters(), 0.01, 1.0)
	adam, ok := opt.(*nn.Adam)
	if !ok {
		t.Fatal("optimizer is not *nn.Adam")
	}
	var firstLoss, lastLoss float32
	const steps = 600
	for s := 0; s < steps; s++ {
		adam.ZeroGrad()
		loss, err := TrainBatchPtrGen(model, inputs, targets, v.PaddingTokenID)
		if err != nil {
			t.Fatalf("step %d: %v", s, err)
		}
		if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
			t.Fatalf("step %d: loss is not finite: %v", s, loss)
		}
		if s == 0 {
			firstLoss = loss
		}
		lastLoss = loss
		adam.ClipGradients()
		adam.Step()
	}
	if lastLoss > firstLoss*0.5 {
		t.Fatalf("toy training barely learned: first loss %.4f, last %.4f", firstLoss, lastLoss)
	}
	if got, err := model.PredictPtrGen("copy amber", 8); err != nil || got != "spec amber" {
		t.Fatalf("trained name: PredictPtrGen(copy amber) = %q, %v; want \"spec amber\"", got, err)
	}
	if got, err := model.PredictPtrGen("copy zephyr", 8); err != nil || got != "spec zephyr" {
		t.Fatalf("held-out name: PredictPtrGen(copy zephyr) = %q, %v; want \"spec zephyr\" — the copy path did not learn to carry an input-only name", got, err)
	}
}
