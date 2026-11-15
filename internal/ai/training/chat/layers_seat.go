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
// Generation goes through the judging protocol's own entry —
// PredictLayersText with the scorer's options (greedy,
// LayersPredictOptions{MaxLen: 60}) — so the shipped behavior
// is the judged behavior: scripts/score_layers calls the same
// function with the same options and a fresh session. Nothing
// about decoding is re-implemented here.
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
	"os"

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
	return &layersSocialBrain{model: m, sess: &LayersSession{}}
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
// input line and returns the model's raw text. Callers apply
// the same tidyDecode post-processing as production replies.
func (b *layersSocialBrain) predict(input string) (string, error) {
	pred, err := PredictLayersText(b.model, input, b.sess, seq2seq.LayersPredictOptions{MaxLen: 60})
	if err != nil {
		return "", err
	}
	return pred.Text, nil
}
