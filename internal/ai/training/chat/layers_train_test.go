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
	for _, name := range []string{"char", "class", "intent", "tone", "concept", "copy", "turn", "memory"} {
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
	if !reflect.DeepEqual(d1.Centroids, d2.Centroids) {
		t.Fatal("concept centroids differ across identical derivations")
	}
	if d1.IntentDist != d2.IntentDist || d1.ToneDist != d2.ToneDist || d1.FactPairs != d2.FactPairs {
		t.Fatal("label/fact derivations differ across runs")
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
	m, err := seq2seq.NewLayersModel(base, cfg, d.ClassOf, d.Centroids)
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
	nonzero("intent emb", m.IntentEmb.Weight)
	nonzero("intent head", m.IntentHead.Weights, m.IntentHead.Biases)
	nonzero("tone emb", m.ToneEmb.Weight)
	nonzero("tone head", m.ToneHead.Weights, m.ToneHead.Biases)
	nonzero("concept emb", m.ConceptEmb.Weight)
	nonzero("concept head", m.ConceptHead.Weights, m.ConceptHead.Biases)
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
			{"char": true}, {"class": true}, {"intent": true}, {"tone": true},
			{"concept": true}, {"copy": true}, {"turn": true}, {"memory": true},
			{"char": true, "class": true, "intent": true, "tone": true, "concept": true, "copy": true, "turn": true, "memory": true},
		} {
			if _, err := PredictLayersText(loaded, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20, Disable: dis}); err != nil {
				t.Fatalf("predict with disable %v: %v", dis, err)
			}
		}
		if _, err := PredictLayersText(loaded, in, sess, seq2seq.LayersPredictOptions{MaxLen: 20, ShuffleConcept: true}); err != nil {
			t.Fatalf("predict shuffled: %v", err)
		}
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
