package chat

// Layers pilot: the session-aware text entry point for layers
// inference, shared by the trainer's probes and the standalone
// scorer. It composes the model's input exactly as training did:
// the memory layer's fact prefix (facts accumulated from the
// session's earlier lines by the fact seat's own extractor) leads,
// then the current line's tokens with their surfaces; the turn
// layer receives the previous reply's token ids. No situation
// line, no typo normalizer — the model-only judging protocol
// (DESIGN §6) measures the bare model.

import (
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// LayersSession carries the cross-turn state the memory and turn
// layers consume: the session's fact statements in order, and the
// previous reply's token ids.
type LayersSession struct {
	Facts     []string // rendered fact statements, oldest first
	PrevReply []int    // previous reply's token ids; nil before turn 2
}

// Observe records one completed turn: facts are extracted from the
// user's line (the fact seat's shapes) and the model's reply ids
// become the previous reply for the next turn.
func (s *LayersSession) Observe(userLine string, replyIDs []int) {
	s.Facts = append(s.Facts, LayersFactStatements(userLine)...)
	s.PrevReply = replyIDs
}

// PredictLayersText generates one reply for a raw user line under
// the session's state. The session is NOT advanced — call Observe
// with the result to thread it (the scorer's modes control exactly
// when state moves).
func PredictLayersText(m *seq2seq.LayersModel, line string, sess *LayersSession, opts seq2seq.LayersPredictOptions) (seq2seq.LayersPrediction, error) {
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
	return m.PredictLayers(ids, surfs, opts)
}
