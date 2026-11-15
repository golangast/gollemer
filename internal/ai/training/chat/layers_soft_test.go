package chat

// Concept-v2 pins (CONCEPT_V2_DESIGN §3):
//   - "frame-soft" is a config of its own: it requires the top
//     code, reports as "frame-soft", and round-trips through
//     the gob with the config intact
//   - the blend signature in training: ONE batch row puts
//     prefix gradient into MANY top-code rows (the hard frame
//     puts it into exactly one — the gold code's), and the top
//     head trains from the downstream losses, not only its CE
//   - at inference the prediction carries the effective
//     distribution; -force-argmax collapses it to one-hot and
//     -shuffle-top-dist rotates it, mechanically
//   - the repair loop still fires on a soft model, on the
//     nominal frame, exactly as in v1

import (
	"math"
	"path/filepath"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// smallSoftModel builds a copy + frame-soft layers model over
// the same 64-pair fixture as smallLayersModel.
func smallSoftModel(t *testing.T) (*seq2seq.LayersModel, *domainTrainData, []ChatPair) {
	t.Helper()
	root := layersTestRoot(t)
	dd, err := prepareDomainTrainData(root, SocialDomain)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	pairs := append([]ChatPair{}, dd.trainPairs[:64]...)
	cfg := seq2seq.DefaultLayersConfig()
	for _, name := range []string{"copy", "frame-soft"} {
		if err := cfg.Set(name, true); err != nil {
			t.Fatalf("set %s: %v", name, err)
		}
	}
	d, err := deriveLayers(pairs, dd.v, dd.tok, cfg)
	if err != nil {
		t.Fatalf("derive: %v", err)
	}
	base, err := seq2seq.NewSeq2Seq(dd.v.Size(), dd.v.Size(), cfg.EmbedDim, cfg.HiddenDim, dd.tok, dd.v)
	if err != nil {
		t.Fatalf("new base: %v", err)
	}
	m, err := seq2seq.NewLayersModel(base, cfg, d.modelDerivations())
	if err != nil {
		t.Fatalf("new layers model: %v", err)
	}
	return m, dd, pairs
}

func TestSoftFrameConfigGating(t *testing.T) {
	cfg := seq2seq.DefaultLayersConfig()
	if err := cfg.Set("frame-soft", true); err != nil {
		t.Fatalf("set frame-soft: %v", err)
	}
	if !cfg.Frame || !cfg.FrameSoft || !cfg.HasTop() || !cfg.HasSlots() {
		t.Fatalf("frame-soft should be the full frame, softened: %+v", cfg)
	}
	// The soft frame cannot combine with the slots-only variant
	// (there is no top code to soften): construction refuses.
	root := layersTestRoot(t)
	dd, err := prepareDomainTrainData(root, SocialDomain)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	bad := seq2seq.DefaultLayersConfig()
	if err := bad.Set("frame-soft", true); err != nil {
		t.Fatal(err)
	}
	if err := bad.Set("frame-slots", true); err != nil {
		t.Fatal(err)
	}
	base, err := seq2seq.NewSeq2Seq(dd.v.Size(), dd.v.Size(), bad.EmbedDim, bad.HiddenDim, dd.tok, dd.v)
	if err != nil {
		t.Fatalf("new base: %v", err)
	}
	if _, err := seq2seq.NewLayersModel(base, bad, seq2seq.LayersDerivations{}); err == nil {
		t.Fatal("frame-soft + frame-slots should be refused at construction")
	}
	// Naming: a soft model reports "frame-soft", never "frame".
	m, _, _ := smallSoftModel(t)
	names := strings.Join(m.TrainedLayers(), ",")
	if !strings.Contains(names, "frame-soft") || strings.Contains(names, "\"frame\"") {
		t.Fatalf("trained layers should name frame-soft: %v", m.TrainedLayers())
	}
	for _, n := range m.TrainedLayers() {
		if n == "frame" {
			t.Fatalf("soft model must not report the hard frame: %v", m.TrainedLayers())
		}
	}
	hard, _, _ := smallLayersModel(t)
	found := false
	for _, n := range hard.TrainedLayers() {
		if n == "frame" {
			found = true
		}
		if n == "frame-soft" {
			t.Fatalf("hard model must not report frame-soft: %v", hard.TrainedLayers())
		}
	}
	if !found {
		t.Fatalf("hard all-layers model should report frame: %v", hard.TrainedLayers())
	}
}

// nonzeroEmbRows counts embedding rows whose gradient vector is
// not identically zero.
func nonzeroEmbRows(ts *tensor.Tensor) int {
	dim := ts.Shape[1]
	rows := 0
	for r := 0; r < ts.Shape[0]; r++ {
		nonzero := false
		for k := 0; k < dim; k++ {
			if ts.Grad.Data[r*dim+k] != 0 {
				nonzero = true
				break
			}
		}
		if nonzero {
			rows++
		}
	}
	return rows
}

// oneRowBatch trains a single fixture pair through the model
// and returns the loss, with gradients left in place.
func oneRowBatch(t *testing.T, m *seq2seq.LayersModel, dd *domainTrainData, pairs []ChatPair) float32 {
	t.Helper()
	cfg := m.Config
	d, err := deriveLayers(pairs, dd.v, dd.tok, cfg)
	if err != nil {
		t.Fatalf("derive: %v", err)
	}
	var enc []layersEncodedPair
	for _, p := range pairs {
		if ep, ok := encodeLayersPair(p, dd.v, dd.tok, cfg, d); ok {
			enc = append(enc, ep)
		}
	}
	if len(enc) == 0 {
		t.Fatal("no encodable pairs")
	}
	batches := makeLayersBatches(enc[:1], 1, dd.v.PaddingTokenID)
	opt := nn.NewOptimizer(m.Parameters(), 1e-3, 1.0)
	opt.(*nn.Adam).ZeroGrad()
	loss, err := m.TrainBatchLayers(batches[0])
	if err != nil {
		t.Fatalf("TrainBatchLayers: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss not finite: %v", loss)
	}
	return loss
}

func TestSoftGradflowBlendSignature(t *testing.T) {
	// Soft: one row's prefix gradient reaches every code row
	// the distribution touches (all of them, at init), and the
	// seed-side blend does the same for the slot conditioning.
	m, dd, pairs := smallSoftModel(t)
	oneRowBatch(t, m, dd, pairs)
	if rows := nonzeroEmbRows(m.TopEmb.Weight); rows < 2 {
		t.Fatalf("soft prefix: gradient in %d top rows, want the blend's many", rows)
	}
	if rows := nonzeroEmbRows(m.TopSeedEmb.Weight); rows < 2 {
		t.Fatalf("soft slot conditioning: gradient in %d top-seed rows, want the blend's many", rows)
	}
	// The top head itself trains (CE + the blend paths join here).
	total := 0.0
	for _, ts := range []*tensor.Tensor{m.TopHead.Weights, m.TopHead.Biases} {
		if ts.Grad == nil {
			t.Fatal("top head: nil grad")
		}
		for _, g := range ts.Grad.Data {
			total += math.Abs(float64(g))
		}
	}
	if total == 0 {
		t.Fatal("top head: zero gradient after one soft batch")
	}

	// Hard contrast: the same single row puts prefix gradient
	// into exactly ONE top row (the gold code's) and seed-side
	// gradient into exactly one row — the commitment, visible
	// in the gradient's shape.
	hard, dd2, pairs2 := smallLayersModel(t)
	oneRowBatch(t, hard, dd2, pairs2)
	if rows := nonzeroEmbRows(hard.TopEmb.Weight); rows != 1 {
		t.Fatalf("hard prefix: gradient in %d top rows, want exactly 1 (the gold code)", rows)
	}
	if rows := nonzeroEmbRows(hard.TopSeedEmb.Weight); rows != 1 {
		t.Fatalf("hard slot conditioning: gradient in %d top-seed rows, want exactly 1", rows)
	}
}

func TestSoftGobRoundtripAndDiagnostics(t *testing.T) {
	m, _, _ := smallSoftModel(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "layers_soft.gob")
	if err := m.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := seq2seq.LoadLayers(path, m.Base.Tokenizer)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !loaded.Config.FrameSoft {
		t.Fatalf("FrameSoft did not round-trip through the gob: %+v", loaded.Config)
	}
	K := m.Config.NumTopCodes
	// NoRepair throughout: with repair on, a top re-pick commits
	// the distribution to one-hot BY DESIGN (the checker's
	// commitment), so the head's spread distribution is only
	// observable with the loop off.
	for _, in := range []string{"hello", "how are you", "my dog is bella"} {
		sess := &LayersSession{}
		want, err := PredictLayersText(m, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true})
		if err != nil {
			t.Fatalf("predict orig %q: %v", in, err)
		}
		got, err := PredictLayersText(loaded, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true})
		if err != nil {
			t.Fatalf("predict loaded %q: %v", in, err)
		}
		if want.Text != got.Text {
			t.Fatalf("roundtrip mismatch on %q: %q vs %q", in, want.Text, got.Text)
		}
		// The prediction carries the effective distribution: a
		// real distribution over the codes, not a commitment.
		if len(got.TopDist) != K {
			t.Fatalf("TopDist len = %d, want %d", len(got.TopDist), K)
		}
		var sum float32
		spread := 0
		for _, p := range got.TopDist {
			sum += p
			if p > 0.05 {
				spread++
			}
		}
		if math.Abs(float64(sum-1)) > 1e-4 {
			t.Fatalf("TopDist sums to %v, want 1", sum)
		}
		if spread < 2 {
			t.Fatalf("TopDist should be spread over codes, got %v", got.TopDist)
		}
		// -force-argmax collapses it to one-hot at the argmax.
		fa, err := PredictLayersText(loaded, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true, ForceArgmax: true})
		if err != nil {
			t.Fatalf("force-argmax %q: %v", in, err)
		}
		ones := 0
		for k, p := range fa.TopDist {
			if p == 1 {
				ones++
				if k != fa.Top {
					t.Fatalf("one-hot at %d but Top = %d", k, fa.Top)
				}
			} else if p != 0 {
				t.Fatalf("force-argmax TopDist not one-hot: %v", fa.TopDist)
			}
		}
		if ones != 1 {
			t.Fatalf("force-argmax TopDist should have exactly one 1: %v", fa.TopDist)
		}
		// -shuffle-top-dist rotates it by exactly one code.
		sd, err := PredictLayersText(loaded, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true, ShuffleTopDist: true})
		if err != nil {
			t.Fatalf("shuffle-top-dist %q: %v", in, err)
		}
		for k := 0; k < K; k++ {
			if sd.TopDist[k] != got.TopDist[(k-1+K)%K] {
				t.Fatalf("shuffle-top-dist is not the one-code rotation: at %d got %v want %v", k, sd.TopDist[k], got.TopDist[(k-1+K)%K])
			}
		}
	}
	// A hard model carries no distribution.
	hard, _, _ := smallLayersModel(t)
	hp, err := PredictLayersText(hard, "hello", &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20})
	if err != nil {
		t.Fatalf("hard predict: %v", err)
	}
	if hp.TopDist != nil {
		t.Fatalf("hard model should carry no TopDist, got %v", hp.TopDist)
	}
}

// The repair loop on a soft model: a forced frame whose tone
// contradicts the lexicon read is re-picked on the nominal
// frame, exactly as in v1 — and the reply still renders under
// a distribution.
func TestSoftFrameRepairStillFires(t *testing.T) {
	m, _, _ := smallSoftModel(t)
	in := "i am feeling down"
	words := tokenizeLower(seq2seq.NormalizeQuery(in))
	if lexTone := m.ToneLex.ToneOf(words); lexTone != ToneLow {
		t.Fatalf("premise broken: lexicon read = %d, want low (%d)", lexTone, ToneLow)
	}
	nat, err := PredictLayersText(m, in, &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true})
	if err != nil {
		t.Fatalf("natural predict: %v", err)
	}
	act := nat.Act
	if !m.ActFamily[nat.Top][act] {
		for a := 0; a < m.Config.NumAct; a++ {
			if m.ActFamily[nat.Top][a] {
				act = a
				break
			}
		}
	}
	ff := &seq2seq.ForcedFrame{
		Top: nat.Top, Act: act, Tone: ToneBright,
		TopProbs:  peaked(nat.Top, m.Config.NumTopCodes, 0.9),
		ActProbs:  peaked(act, m.Config.NumAct, 0.9),
		ToneProbs: peaked(ToneBright, m.Config.NumTone, 0.9),
	}
	got, err := PredictLayersText(m, in, &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20, ForceFrame: ff})
	if err != nil {
		t.Fatalf("repair predict: %v", err)
	}
	if got.Tone != ToneLow {
		t.Fatalf("tone not repaired to the lexicon read: got %d, repairs=%v", got.Tone, got.Repairs)
	}
	if len(got.Repairs) == 0 || !strings.Contains(got.Repairs[0], "tone check") {
		t.Fatalf("first repair should be the tone check, got %v", got.Repairs)
	}
	if len(got.TopDist) != m.Config.NumTopCodes {
		t.Fatalf("soft prediction should still carry its distribution, got %v", got.TopDist)
	}
}
