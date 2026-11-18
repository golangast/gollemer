package chat

// The temperature-sampled twin of PredictLayersText, seated
// with the checked-sampling decode (layers_decode.go) on
// 2026-11-18. The input composition is PredictLayersText's
// own — NormalizeQuery, tokenizeLower, vocab ids, the memory
// layer's fact prefix, the turn layer's previous-reply ids —
// unchanged; only the final call differs, landing in
// PredictLayersSampled instead of PredictLayers. Seeding
// discipline belongs to the caller.

import (
	"math/rand"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// PredictLayersTextSampled generates one sampled reply for a raw
// user line under the session's state. The session is NOT
// advanced, exactly as with PredictLayersText.
func PredictLayersTextSampled(m *seq2seq.LayersModel, line string, sess *LayersSession, opts seq2seq.LayersPredictOptions, temp float64, rng *rand.Rand) (seq2seq.LayersPrediction, error) {
	v := m.Base.OutputVocab
	words := tokenizeLower(seq2seq.NormalizeQuery(line))
	ids := make([]int, len(words))
	for i, w := range words {
		ids[i] = v.GetTokenID(w)
	}
	surfs := append([]string{}, words...)
	if m.Config.Memory && sess != nil && len(sess.Facts) > 0 {
		prefixToks := LayersFactPrefixTokens(sess.Facts)
		prefixIDs := make([]int, len(prefixToks))
		for i, t := range prefixToks {
			prefixIDs[i] = v.GetTokenID(t)
		}
		ids = append(prefixIDs, ids...)
		surfs = append(append([]string{}, prefixToks...), surfs...)
	}
	if m.Config.Turn && sess != nil {
		opts.PrevReplyIDs = sess.PrevReply
	}
	return m.PredictLayersSampled(ids, surfs, opts, temp, rng)
}
