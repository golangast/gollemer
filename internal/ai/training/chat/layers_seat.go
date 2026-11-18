package chat

// Layers adoption (2026-11-15, DESIGN §7's separate integration
// build): the chat's social seat can be served by the adopted
// layers model — the copy configuration that cleared every
// registered bar (Section S 58/160 with the paraphrase signature
// at 16/40, weak-spot 9/10, held-out 22/41, combination holdout
// 72.2%).
//
// Presence is the switch. When the layers gob exists at its
// canonical path (LayersModelPath) it becomes the social brain
// for every surface that generates social replies: the unified
// loop's routed fallback, the single-domain loops (the social
// loop's own turns and the gocode loop's chatter branch), and
// the talk seat's social arm. When it is absent, nothing in
// this file is constructed and the chat behaves byte-identically
// to before — the production Seq2Seq serves, beam search and
// all, and a checkout that never adopts never sees a difference.
//
// Generation starts from the judging protocol's own entry —
// PredictLayersText with the scorer's options (greedy,
// LayersPredictOptions{MaxLen: 60}) — and, since 2026-11-18,
// adds the checked-sampling rescue the decode probe measured
// on this model (layers_decode.go): when the greedy reply
// drops the user's topic, the seat draws up to 8 sampled
// replies at temperature 0.8 and serves the first that carries
// the topic; when none does, or the model carries a frame the
// sampled render does not support, the greedy reply stands.
// The probe's always-sample selection scored 240/426 against
// greedy's 176/426 on the pack-2 combination holdout and
// 67/160 against 58/160 on the frozen set; keeping a
// topic-carrying greedy reply instead of re-rolling it tracks
// that measurement with far fewer draws. Decoding itself is
// not re-implemented here — the sampled twin
// (PredictLayersTextSampled) shares the greedy entry's input
// composition exactly.
//
// The seats that wrap the social path are untouched and keep
// wrapping: the typo normalizer still reshapes the line inside
// SocialInput/SocialInputSituated (the loops arm it from the
// serving brain's vocabulary — see the call sites), the fact
// seat still captures and recalls ahead of generation, the
// situation line still composes the input, and every
// deterministic claim still sits ahead of the model. Callers
// compose the input and apply tidyDecode exactly as they do
// for production replies.
//
// One honest difference: a layers prediction carries no
// per-token ThoughtTrace, so the /thoughts display prints
// nothing for layers-served replies (printThoughtTrace(nil)
// is a no-op). The prediction's own diagnostics (frame, topic
// words, repairs) exist on LayersPrediction but the chat does
// not render them.

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// layersSocialBrain is the chat's handle on the adopted layers
// model: the loaded model plus the one session the predict API
// takes.
//
// The session is held, never observed into. The adopted
// configuration is copy-only, and PredictLayersText consults
// the session only under the memory and turn layers — so the
// session is inert by construction, and every call is identical
// to the scorer's fresh-session protocol. The chat's multi-turn
// context rides the composed input string (SocialInputSituated's
// "before you said ..." format), exactly as it does for the
// production model; no second state channel is introduced.
type layersSocialBrain struct {
	model *seq2seq.LayersModel
	sess  *LayersSession
	// rng feeds the checked-sampling rescue (layers_decode.go).
	// The chat is one conversation at a time, so the stream is
	// consumed serially; tests may reseed it for determinism.
	rng *rand.Rand
}

// loadLayersSocialBrain loads the layers model under a project
// root, or returns nil when the gob is absent. A gob that is
// present but unreadable is logged and also returns nil: the
// production model keeps the chat working, and the failure is
// on the record rather than silent.
func loadLayersSocialBrain(projectRoot string) *layersSocialBrain {
	path := LayersModelPath(projectRoot)
	if _, err := os.Stat(path); err != nil {
		return nil
	}
	m, err := loadLayersModel(path)
	if err != nil {
		log.Printf("[LAYERS] %v; the production social model serves instead", err)
		return nil
	}
	log.Printf("[LAYERS] social seat served by the layers model %s (trained layers: %v)", path, m.TrainedLayers())
	return &layersSocialBrain{model: m, sess: &LayersSession{}, rng: rand.New(rand.NewSource(time.Now().UnixNano()))}
}

// loadLayersModel mirrors scripts/score_layers' loadModel
// exactly — placeholder tokenizer in, the real tokenizer
// rebuilt from the model's own output vocabulary — so the
// object the chat predicts with is in the same state as the
// object the scorer judged. The model's configuration and
// class map travel inside the gob (seq2seq.LoadLayers reads
// only the gob); the .layers.json / .classes.json sidecars
// are derivation audit artifacts and nothing at inference
// reads them.
func loadLayersModel(path string) (*seq2seq.LayersModel, error) {
	placeholder, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		return nil, err
	}
	m, err := seq2seq.LoadLayers(path, placeholder)
	if err != nil {
		return nil, err
	}
	if m.Base.OutputVocab == nil || m.Base.OutputVocab.Size() == 0 {
		return nil, fmt.Errorf("layers model %s has no vocabulary", path)
	}
	tok, err := tokenizer.NewTokenizer(m.Base.OutputVocab)
	if err != nil {
		return nil, err
	}
	m.Base.Tokenizer = tok
	return m, nil
}

// predict generates one social reply for an already-composed
// input line and returns the model's raw text. rawLine is the
// user's own line before composition — the topic check
// (layers_decode.go) reads it, never the composed input, whose
// situation line and quoted history would lend the check words
// the user did not just say. Callers apply the same tidyDecode
// post-processing as production replies.
func (b *layersSocialBrain) predict(input, rawLine string) (string, error) {
	opts := seq2seq.LayersPredictOptions{MaxLen: 60}
	pred, err := PredictLayersText(b.model, input, b.sess, opts)
	if err != nil {
		return "", err
	}
	carries := func(reply string) bool { return layersReplyCarries(rawLine, reply) }
	draw := func() (string, error) {
		s, serr := PredictLayersTextSampled(b.model, input, b.sess, opts, layersSampleTemp, b.rng)
		if serr != nil {
			return "", serr
		}
		return s.Text, nil
	}
	return layersPickReply(pred.Text, carries, draw), nil
}
