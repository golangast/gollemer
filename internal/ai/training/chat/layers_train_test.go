package chat

// Layers pilot pins (DESIGN §5):
//   - derivations are deterministic and do not touch the pairs
//   - the shipped production gob still loads and predicts through
//     the production loader (existing gobs keep working)
//   - a layers gob round-trips: save -> load -> identical replies,
//     and every inference switch executes
//   - one all-layers training step puts gradient into every layer's
//     parameters (the NCM build's silent-zero-grad failure mode)

import (
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

func layersTestRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", "..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(root, "data", "training", "chat_pairs.jsonl")); err != nil {
		t.Skipf("real corpus not present at %s", root)
	}
	return root
}

func allLayersCfg() seq2seq.LayersConfig {
	cfg := seq2seq.DefaultLayersConfig()
	for _, name := range []string{"char", "class", "frame", "copy", "turn", "memory"} {
		if err := cfg.Set(name, true); err != nil {
			panic(err)
		}
	}
	return cfg
}

func TestLayersDerivationsDeterministicAndPure(t *testing.T) {
	root := layersTestRoot(t)
	dd, err := prepareDomainTrainData(root, SocialDomain)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if len(dd.trainPairs) < 300 {
		t.Fatalf("only %d train pairs", len(dd.trainPairs))
	}
	pairs := append([]ChatPair{}, dd.trainPairs[:300]...)
	before := append([]ChatPair{}, pairs...)
	cfg := allLayersCfg()
	d1, err := deriveLayers(pairs, dd.v, dd.tok, cfg)
	if err != nil {
		t.Fatalf("derive 1: %v", err)
	}
	d2, err := deriveLayers(pairs, dd.v, dd.tok, cfg)
	if err != nil {
		t.Fatalf("derive 2: %v", err)
	}
	if !reflect.DeepEqual(pairs, before) {
		t.Fatal("derivations mutated the training pairs")
	}
	if !reflect.DeepEqual(d1.ClassOf, d2.ClassOf) {
		t.Fatal("word classes differ across identical derivations")
	}
	if !reflect.DeepEqual(d1.TopCentroids, d2.TopCentroids) {
		t.Fatal("top centroids differ across identical derivations")
	}
	if !reflect.DeepEqual(d1.ActFamily, d2.ActFamily) {
		t.Fatal("act family tables differ across identical derivations")
	}
	if d1.ActDist != d2.ActDist || d1.ToneDist != d2.ToneDist || d1.FactPairs != d2.FactPairs {
		t.Fatal("label/fact derivations differ across runs")
	}
	if len(d1.TopCensus) != cfg.NumTopCodes {
		t.Fatalf("top census has %d entries, want %d", len(d1.TopCensus), cfg.NumTopCodes)
	}
	total := 0
	for _, n := range d1.TopCensus {
		total += n
	}
	if total != len(pairs) {
		t.Fatalf("top census covers %d pairs, want %d (every pair gets a teacher frame)", total, len(pairs))
	}
	if len(d1.ClassOf) != dd.v.Size() {
		t.Fatalf("class map has %d entries, vocab %d", len(d1.ClassOf), dd.v.Size())
	}
	used := map[int]bool{}
	for _, c := range d1.ClassOf {
		used[c] = true
	}
	if len(used) < 10 {
		t.Fatalf("only %d distinct classes used", len(used))
	}
}

// smallLayersModel builds a layers model over a 64-pair subsample
// of the real corpus for the gradflow and roundtrip tests.
func smallLayersModel(t *testing.T) (*seq2seq.LayersModel, *domainTrainData, []ChatPair) {
	t.Helper()
	root := layersTestRoot(t)
	dd, err := prepareDomainTrainData(root, SocialDomain)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	pairs := append([]ChatPair{}, dd.trainPairs[:64]...)
	cfg := allLayersCfg()
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

func TestLayersGradflowAllLayers(t *testing.T) {
	m, dd, pairs := smallLayersModel(t)
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
	batches := makeLayersBatches(enc, 8, dd.v.PaddingTokenID)
	opt := nn.NewOptimizer(m.Parameters(), 1e-3, 1.0)
	adam := opt.(*nn.Adam)
	adam.ZeroGrad()
	loss, err := m.TrainBatchLayers(batches[0])
	if err != nil {
		t.Fatalf("TrainBatchLayers: %v", err)
	}
	if math.IsNaN(float64(loss)) || math.IsInf(float64(loss), 0) {
		t.Fatalf("loss not finite: %v", loss)
	}
	nonzero := func(name string, ts ...*tensor.Tensor) {
		t.Helper()
		total := 0.0
		for _, p := range ts {
			if p == nil || p.Grad == nil {
				t.Fatalf("%s: nil param or grad", name)
			}
			for _, g := range p.Grad.Data {
				total += math.Abs(float64(g))
			}
		}
		if total == 0 {
			t.Fatalf("%s: zero gradient after one all-layers batch", name)
		}
	}
	nonzero("char table", m.CharTable.Weight)
	nonzero("class table", m.ClassTable.Weight)
	nonzero("top emb (prefix)", m.TopEmb.Weight)
	nonzero("top emb (seed)", m.TopSeedEmb.Weight)
	nonzero("top head", m.TopHead.Weights, m.TopHead.Biases)
	nonzero("act emb", m.ActEmb.Weight)
	nonzero("act head", m.ActHead.Weights, m.ActHead.Biases)
	nonzero("tone emb", m.ToneEmb.Weight)
	nonzero("tone head", m.ToneHead.Weights, m.ToneHead.Biases)
	// TurnEmb rows only receive gradient when a previous turn
	// exists; the single-turn corpus never has one (DESIGN §3.7),
	// so the pin is that its grad tensor exists and is wired in —
	// the slot that actually trains here is NoneTurn.
	if m.TurnEmb.Weight.Grad == nil {
		t.Fatal("turn emb: nil grad tensor")
	}
	nonzero("none-turn", m.NoneTurn)
	nonzero("encoder embedding", m.Base.Encoder.Embedding.Weight)
	nonzero("decoder embedding", m.Base.Decoder.Embedding.Weight)
	nonzero("ptrgen", m.Base.Decoder.PtrGen.Parameters()...)
	adam.ClipGradients()
	adam.Step()
}

func TestLayersGobRoundtripAndSwitches(t *testing.T) {
	m, _, _ := smallLayersModel(t)
	dir := t.TempDir()
	path := filepath.Join(dir, "layers.gob")
	if err := m.Save(path); err != nil {
		t.Fatalf("save: %v", err)
	}
	loaded, err := seq2seq.LoadLayers(path, m.Base.Tokenizer)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	inputs := []string{"hello", "how are you", "my dog is bella"}
	for _, in := range inputs {
		sess := &LayersSession{}
		want, err := PredictLayersText(m, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20})
		if err != nil {
			t.Fatalf("predict orig %q: %v", in, err)
		}
		got, err := PredictLayersText(loaded, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20})
		if err != nil {
			t.Fatalf("predict loaded %q: %v", in, err)
		}
		if want.Text != got.Text {
			t.Fatalf("roundtrip mismatch on %q: %q vs %q", in, want.Text, got.Text)
		}
		// Every inference switch executes without error.
		for _, dis := range []map[string]bool{
			{"char": true}, {"class": true}, {"frame": true},
			{"copy": true}, {"turn": true}, {"memory": true},
			{"char": true, "class": true, "frame": true, "copy": true, "turn": true, "memory": true},
		} {
			if _, err := PredictLayersText(loaded, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20, Disable: dis}); err != nil {
				t.Fatalf("predict with disable %v: %v", dis, err)
			}
		}
		for _, o := range []seq2seq.LayersPredictOptions{
			{MaxLen: 20, ShuffleTop: true},
			{MaxLen: 20, NoRepair: true},
			{MaxLen: 20, FrameTopOff: true},
			{MaxLen: 20, FrameSlotsOff: true},
			{MaxLen: 20, ShuffleTop: true, NoRepair: true},
		} {
			if _, err := PredictLayersText(loaded, in, sess, o); err != nil {
				t.Fatalf("predict with options %+v: %v", o, err)
			}
		}
	}
}

// ---- the re-correction loop (DESIGN §12) ----

// peaked returns a probability vector with p on idx and the
// remaining mass spread evenly.
func peaked(idx, n int, p float32) []float32 {
	out := make([]float32, n)
	rest := (1 - p) / float32(n-1)
	for i := range out {
		out[i] = rest
	}
	out[idx] = p
	return out
}

// A tone slot that contradicts the lexicon re-read of the input
// is re-picked to the lexicon read (check (b)); with the loop off
// the corrupted slot stands.
func TestFrameRepairFixesCorruptedTone(t *testing.T) {
	m, _, _ := smallLayersModel(t)
	in := "i am feeling down"
	words := tokenizeLower(seq2seq.NormalizeQuery(in))
	lexTone := m.ToneLex.ToneOf(words)
	if lexTone != ToneLow {
		t.Fatalf("premise broken: lexicon read of %q = %d, want low (%d)", in, lexTone, ToneLow)
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
		Top: nat.Top, Act: act, Tone: ToneBright, // corrupted: bright against a low read
		TopProbs:  peaked(nat.Top, m.Config.NumTopCodes, 0.9),
		ActProbs:  peaked(act, m.Config.NumAct, 0.9),
		ToneProbs: peaked(ToneBright, m.Config.NumTone, 0.9),
	}
	nr, err := PredictLayersText(m, in, &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true, ForceFrame: ff})
	if err != nil {
		t.Fatalf("no-repair predict: %v", err)
	}
	if nr.Tone != ToneBright || len(nr.Repairs) != 0 {
		t.Fatalf("no-repair run should keep the corrupted tone: tone=%d repairs=%v", nr.Tone, nr.Repairs)
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
}

// An act slot outside the top code's family is re-picked to the
// head's best in-family act (check (a)); the repair log's first
// entry is that repair.
func TestFrameRepairFixesCorruptedAct(t *testing.T) {
	m, _, _ := smallLayersModel(t)
	in := "hello there"
	words := tokenizeLower(seq2seq.NormalizeQuery(in))
	lexTone := m.ToneLex.ToneOf(words)
	nat, err := PredictLayersText(m, in, &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true})
	if err != nil {
		t.Fatalf("natural predict: %v", err)
	}
	// Find a top code with a non-trivial family and corrupt the act.
	top, badAct, goodAct := -1, -1, -1
	for cand := 0; cand < m.Config.NumTopCodes; cand++ {
		bad, good := -1, -1
		for a := 0; a < m.Config.NumAct; a++ {
			if m.ActFamily[cand][a] {
				if good < 0 {
					good = a
				}
			} else if bad < 0 {
				bad = a
			}
		}
		if bad >= 0 && good >= 0 {
			top, badAct, goodAct = cand, bad, good
			break
		}
	}
	if top < 0 {
		t.Fatal("no top code has both an in-family and an out-of-family act; cannot corrupt")
	}
	actProbs := make([]float32, m.Config.NumAct)
	for i := range actProbs {
		actProbs[i] = 0.05
	}
	actProbs[badAct] = 0.34  // corrupted act: highest prob, but
	actProbs[goodAct] = 0.33 // lower confidence than the top
	ff := &seq2seq.ForcedFrame{
		Top: top, Act: badAct, Tone: lexTone, // tone = the lexicon read, so check (b) passes
		TopProbs:  peaked(top, m.Config.NumTopCodes, 0.9),
		ActProbs:  actProbs,
		ToneProbs: peaked(lexTone, m.Config.NumTone, 0.9),
	}
	nr, err := PredictLayersText(m, in, &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true, ForceFrame: ff})
	if err != nil {
		t.Fatalf("no-repair predict: %v", err)
	}
	if nr.Act != badAct {
		t.Fatalf("no-repair run should keep the corrupted act: got %d want %d", nr.Act, badAct)
	}
	got, err := PredictLayersText(m, in, &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20, ForceFrame: ff})
	if err != nil {
		t.Fatalf("repair predict: %v", err)
	}
	if len(got.Repairs) == 0 || !strings.Contains(got.Repairs[0], "act-family check") {
		t.Fatalf("first repair should be the act-family repair, got %v", got.Repairs)
	}
	if !strings.Contains(got.Repairs[0], "best in family") {
		t.Fatalf("act repair should land in the family, got %q", got.Repairs[0])
	}
	_ = nat
}

// A frame that already passes every check (found by search over
// forced frames) must produce byte-identical output with the
// loop on: repair fires nothing.
func TestFrameRepairLeavesCoherentFrameAlone(t *testing.T) {
	m, dd, _ := smallLayersModel(t)
	inputs := []string{"hello", "how are you", "good morning", "thank you", "what is your name"}
	found := false
	for _, in := range inputs {
		words := tokenizeLower(seq2seq.NormalizeQuery(in))
		lexTone := m.ToneLex.ToneOf(words)
		for top := 0; top < m.Config.NumTopCodes && !found; top++ {
			act := -1
			for a := 0; a < m.Config.NumAct; a++ {
				if m.ActFamily[top][a] {
					act = a
					break
				}
			}
			if act < 0 {
				continue
			}
			ff := &seq2seq.ForcedFrame{Top: top, Act: act, Tone: lexTone}
			opts := seq2seq.LayersPredictOptions{MaxLen: 20, NoRepair: true, ForceFrame: ff}
			base, err := PredictLayersText(m, in, &LayersSession{}, opts)
			if err != nil {
				t.Fatalf("search predict: %v", err)
			}
			// Coherence: the round-trip must agree — rendered top
			// matches (or is underivable) and the rendered tone
			// does not contradict.
			if rt, ok := m.DeriveReplyTop(base.TokenIDs); ok && rt != top {
				continue
			}
			repWords := make([]string, len(base.TokenIDs))
			for i, id := range base.TokenIDs {
				repWords[i] = dd.v.GetWord(id)
			}
			if lt := m.ToneLex.ToneOf(repWords); lt > 0 && lt != lexTone {
				continue
			}
			got, err := PredictLayersText(m, in, &LayersSession{}, seq2seq.LayersPredictOptions{MaxLen: 20, ForceFrame: ff})
			if err != nil {
				t.Fatalf("repair predict: %v", err)
			}
			if len(got.Repairs) != 0 {
				t.Fatalf("coherent frame (top=%d act=%d tone=%d on %q) drew repairs: %v", top, act, lexTone, in, got.Repairs)
			}
			if got.Residual {
				t.Fatalf("coherent frame marked residual")
			}
			if got.Text != base.Text {
				t.Fatalf("repair-on changed a coherent frame's reply: %q vs %q", got.Text, base.Text)
			}
			found = true
		}
	}
	if !found {
		t.Fatal("no coherent forced frame found across the probe inputs — the loop can never be quiet")
	}
}

func TestProductionGobStillLoads(t *testing.T) {
	root := layersTestRoot(t)
	path := filepath.Join(root, "data", "models", "gob_models", "real_tiny_seq2seq_social.gob")
	if _, err := os.Stat(path); err != nil {
		t.Skip("production social gob not present")
	}
	dd, err := prepareDomainTrainData(root, SocialDomain)
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	model, err := seq2seq.Load(path, dd.tok)
	if err != nil {
		t.Fatalf("production gob load: %v", err)
	}
	out, err := model.Predict("hello", 20)
	if err != nil {
		t.Fatalf("production predict: %v", err)
	}
	if out == "" {
		t.Fatal("production predict returned empty")
	}
}
