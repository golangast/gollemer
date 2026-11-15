package seq2seq

// Layers pilot: greedy inference for LayersModel, mirroring the
// production PredictWithTrace walk (same anti-degeneration bans)
// with the layers' inference halves. The FRAME (DESIGN §12) is
// predicted first — top code from the encoder state, act/tone
// slots from the state plus the top code's seed row — then
// CHECKED by the deterministic re-correction loop and repaired
// (bounded re-picks of the lowest-confidence offending slot)
// before the decoder renders under it: top code as the one-step
// prefix, slots in the seed, copy mixing per step. On a model
// trained with the soft frame (concept v2), the prefix and the
// slot conditioning consume the head's DISTRIBUTION as a blend
// of code rows; the nominal frame (argmax + slots) is still
// what the checks judge, and a repair re-pick of the top
// collapses the distribution to the re-picked code. The scorer's
// switches ride on LayersPredictOptions: Disable forces trained
// layers off at inference; NoRepair disables the loop (the loop's
// own ablation); ShuffleTop permutes the predicted top code (the
// registered decoration diagnostic); FrameTopOff/FrameSlotsOff
// drop one frame half at inference. Switches are diagnostics on a
// trained model, never ablation verdicts — ablation arms are
// separate training runs.

import (
	"fmt"
	"math"

	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// MaxRepairPicks bounds the re-correction loop (DESIGN §12): at
// most this many slot re-picks across the pre-render checks and
// the round-trip check; a frame still inconsistent after the
// bound is used as-is and reported (Residual).
const MaxRepairPicks = 3

// LayersPredictOptions carries the per-call layer state.
type LayersPredictOptions struct {
	Disable       map[string]bool // layers forced off at inference
	NoRepair      bool            // disable the re-correction loop
	ShuffleTop    bool            // permute the predicted top code c -> (c+1) mod K; the shuffled top is held fixed under repair
	FrameTopOff   bool            // diagnostic: drop the top code at inference (slots behave as the slots-only arm)
	FrameSlotsOff bool            // diagnostic: drop the act/tone slots at inference (top-only behavior)
	ForceFrame    *ForcedFrame    // tests/diagnostics: use this frame instead of predicting one
	PrevReplyIDs  []int           // previous reply tokens (turn layer); nil = no previous turn
	MaxLen        int             // 0 = 60 (the house scorers' cap)

	// Soft-frame diagnostics (concept v2; meaningful only on a
	// model trained with frame-soft — on a hard model both are
	// no-ops, as there is no distribution to reshape).
	ShuffleTopDist bool // rotate the effective top distribution by one code (mass of k moves to k+1); held fixed under repair
	ForceArgmax    bool // collapse the effective distribution to one-hot at its argmax (the commitment ablation)
}

// ForcedFrame pins a frame for a prediction call. Probs may be
// supplied so the repair loop re-picks with realistic confidences;
// nil probs are treated as uniform.
type ForcedFrame struct {
	Top, Act, Tone                int
	TopProbs, ActProbs, ToneProbs []float32
}

// FrameState is a filled frame plus the head confidences the
// repair loop compares when choosing the offending slot.
type FrameState struct {
	Top, Act, Tone                int // -1 when the part is absent
	TopConf, ActConf, ToneConf    float32
	topProbs, actProbs, toneProbs []float32
	// topDist is the EFFECTIVE top distribution the decoder
	// consumes (soft-frame models only; nil on hard models,
	// where the decoder reads Top's row directly). It starts as
	// the head's distribution; a repair re-pick of the top
	// collapses it to one-hot at the re-picked code — the
	// deterministic checker may commit where the head may not
	// (CONCEPT_V2_DESIGN §3).
	topDist []float32
}

// LayersPrediction is one layers-model reply plus its diagnostics.
type LayersPrediction struct {
	Text       string
	TokenIDs   []int
	Top        int      // top code actually fed to the decoder (-1 when none)
	TopDist    []float32 // effective top distribution fed to the decoder (soft-frame models only; nil otherwise)
	Act        int      // act slot actually fed (-1 when none)
	Tone       int      // tone slot actually fed (-1 when none)
	ReplyTop   int      // top code derived from the rendered reply (-1 when none)
	TopicWords []string // the frame's topic slot: the input's content words, carried by construction
	Repairs    []string // re-correction log (empty when the loop fired nothing)
	Residual   bool     // frame still inconsistent after the repair bound
}

// layerOn reports whether a layer is both trained into the model
// and not disabled for this call.
func (m *LayersModel) layerOn(name string, opts LayersPredictOptions) bool {
	if opts.Disable != nil && opts.Disable[name] {
		return false
	}
	return m.Config.On(name)
}

// frameParts reports which frame halves are live for this call:
// the top code and the act/tone slots, honoring the model's
// trained variant and the inference diagnostics.
func (m *LayersModel) frameParts(opts LayersPredictOptions) (top, slots bool) {
	if !m.layerOn("frame", opts) {
		return false, false
	}
	top = m.Config.HasTop() && !opts.FrameTopOff
	slots = m.Config.HasSlots() && !opts.FrameSlotsOff
	return top, slots
}

// PredictLayers generates one reply for the given input ids and
// their surface forms (aligned; surfaces drive the char layer,
// the frame's topic slot, and the tone check's lexicon re-read).
func (m *LayersModel) PredictLayers(inputIDs []int, surfaces []string, opts LayersPredictOptions) (LayersPrediction, error) {
	cfg := m.Config
	base := m.Base
	hidden := cfg.HiddenDim
	embDim := cfg.EmbedDim
	maxLen := opts.MaxLen
	if maxLen <= 0 {
		maxLen = 60
	}
	out := LayersPrediction{Top: -1, Act: -1, Tone: -1, ReplyTop: -1}
	if len(inputIDs) == 0 {
		return out, nil
	}

	// ---- Encoder (3D values pass, as in training).
	B, T := 1, len(inputIDs)
	tokRows := make([]float32, T*embDim)
	for t, id := range inputIDs {
		off := id * embDim
		copy(tokRows[t*embDim:(t+1)*embDim], base.Encoder.Embedding.Weight.Data[off:off+embDim])
	}
	comb := m.combineRows(tokRows, [][]int{inputIDs}, [][]string{surfaces}, B, T)
	encEmb := tensor.NewTensor([]int{B, T, embDim}, comb, false)
	h0 := tensor.NewTensor([]int{B, hidden}, make([]float32, hidden), false)
	c0 := tensor.NewTensor([]int{B, hidden}, make([]float32, hidden), false)
	encOut, encCell, err := base.Encoder.LSTM.Forward(encEmb, h0, c0)
	if err != nil {
		return out, err
	}
	encFinal := make([]float32, hidden)
	copy(encFinal, encOut.Data[(T-1)*hidden:T*hidden])

	// ---- Frame: predict, check, repair (DESIGN §12).
	hasTop, hasSlots := m.frameParts(opts)
	frame := FrameState{Top: -1, Act: -1, Tone: -1}
	topFixed := false
	if hasTop || hasSlots {
		frame = m.predictFrame(encFinal, hasTop, hasSlots, opts)
		if opts.ShuffleTop && frame.Top >= 0 {
			frame.Top = (frame.Top + 1) % cfg.NumTopCodes
			frame.TopConf = frame.topProbs[frame.Top]
			topFixed = true // the sabotage is the measurement; repair works around it, never undoes it
			// On a soft model the decoder reads the distribution,
			// not the nominal code: the sabotage must reach it, so
			// the distribution collapses to the shuffled code.
			frame.commitDist()
			if hasSlots {
				// Slots were predicted under the true top; re-read
				// them under the shuffled one, as the decoder's
				// conditioning will see it.
				m.refillSlots(&frame, encFinal)
			}
		}
		if opts.ForceArgmax {
			// The commitment ablation (concept v2): the same
			// weights, the distribution collapsed to its argmax
			// before any repair — v1's conditioning, exactly.
			frame.commitDist()
		}
		if opts.ShuffleTopDist && frame.topDist != nil {
			// The decoration diagnostic for the soft code: rotate
			// the whole distribution by one code and hold it
			// fixed under repair, as ShuffleTop holds its code.
			frame.topDist = rotateTopDist(frame.topDist)
			frame.Top = argmaxF(frame.topDist)
			frame.TopConf = frame.topDist[frame.Top]
			topFixed = true
			if hasSlots {
				m.refillSlots(&frame, encFinal)
			}
		}
	}
	out.TopicWords = ContentWordsOf(surfaces)
	var settled *rendered
	if (hasTop || hasSlots) && !opts.NoRepair {
		repairs, residual, last, rerr := m.repairFrame(&frame, encFinal, surfaces, hasTop, hasSlots, topFixed,
			func(fr FrameState) (rendered, error) {
				ids, text, rerr := m.renderReply(encOut, encCell, encFinal, fr, inputIDs, opts, maxLen)
				if rerr != nil {
					return rendered{}, rerr
				}
				return rendered{ids: ids, text: text}, nil
			})
		if rerr != nil {
			return out, rerr
		}
		out.Repairs = repairs
		out.Residual = residual
		settled = last
	}

	// ---- Final render under the settled frame (reusing the
	// repair loop's own render when the frame settled under it:
	// decoding is deterministic, so the two are the same bytes).
	var ids []int
	var text string
	if settled != nil {
		ids, text = settled.ids, settled.text
	} else {
		var rerr error
		ids, text, rerr = m.renderReply(encOut, encCell, encFinal, frame, inputIDs, opts, maxLen)
		if rerr != nil {
			return out, rerr
		}
	}
	out.TokenIDs, out.Text = ids, text
	out.Top, out.Act, out.Tone = frame.Top, frame.Act, frame.Tone
	if frame.topDist != nil {
		out.TopDist = append([]float32{}, frame.topDist...)
	}
	if rt, ok := m.DeriveReplyTop(ids); ok {
		out.ReplyTop = rt
	}
	return out, nil
}

// rendered is one exploratory render inside the repair loop.
type rendered struct {
	ids  []int
	text string
}

// commitDist collapses the frame's effective distribution to
// one-hot at its current Top (no-op on hard models, whose
// topDist is nil). Used where a hard commitment is the point:
// a repair re-pick (the checker's decision), the ShuffleTop
// sabotage reaching the decoder, and the ForceArgmax ablation.
func (fr *FrameState) commitDist() {
	if fr.topDist == nil || fr.Top < 0 {
		return
	}
	d := make([]float32, len(fr.topDist))
	d[fr.Top] = 1
	fr.topDist = d
}

// rotateTopDist rotates a distribution by one code: the mass of
// code k moves to code (k+1) mod K — the distributional analogue
// of the ShuffleTop permutation c -> (c+1) mod K.
func rotateTopDist(p []float32) []float32 {
	out := make([]float32, len(p))
	for k, v := range p {
		out[(k+1)%len(p)] = v
	}
	return out
}

// predictFrame fills a frame from the heads: top code from the
// encoder state, slots from the state plus the top code's seed
// row (the same conditioning as training). ForceFrame overrides.
func (m *LayersModel) predictFrame(encFinal []float32, hasTop, hasSlots bool, opts LayersPredictOptions) FrameState {
	fr := FrameState{Top: -1, Act: -1, Tone: -1}
	if hasTop {
		fr.topProbs = softmaxF(linearRow(m.TopHead, encFinal))
		fr.Top = argmaxF(fr.topProbs)
		fr.TopConf = fr.topProbs[fr.Top]
		if m.Config.FrameSoft {
			// The decoder will consume the distribution itself;
			// the nominal Top above is for checks and reporting.
			fr.topDist = append([]float32{}, fr.topProbs...)
		}
	}
	if hasSlots {
		m.refillSlots(&fr, encFinal)
	}
	if opts.ForceFrame != nil {
		ff := opts.ForceFrame
		if hasTop && ff.Top >= 0 {
			fr.Top = ff.Top
			if ff.TopProbs != nil {
				fr.topProbs = ff.TopProbs
				fr.TopConf = ff.TopProbs[ff.Top]
			}
			if fr.topDist != nil {
				// A forced frame on a soft model: the forced
				// distribution when one was supplied, else the
				// forced code as a commitment.
				if ff.TopProbs != nil {
					fr.topDist = append([]float32{}, ff.TopProbs...)
				} else {
					fr.commitDist()
				}
			}
		}
		if hasSlots {
			if ff.Act >= 0 {
				fr.Act = ff.Act
				if ff.ActProbs != nil {
					fr.actProbs = ff.ActProbs
					fr.ActConf = ff.ActProbs[ff.Act]
				}
			}
			if ff.Tone >= 0 {
				fr.Tone = ff.Tone
				if ff.ToneProbs != nil {
					fr.toneProbs = ff.ToneProbs
					fr.ToneConf = ff.ToneProbs[ff.Tone]
				}
			}
		}
	}
	return fr
}

// refillSlots (re)predicts the act/tone slots under the frame's
// current top code. On a soft model the conditioning row is the
// blended seed row of the frame's current effective
// distribution — the same form the slot heads trained under.
func (m *LayersModel) refillSlots(fr *FrameState, encFinal []float32) {
	hidden := m.Config.HiddenDim
	input := encFinal
	if fr.topDist != nil && m.TopSeedEmb != nil {
		combined := make([]float32, hidden)
		copy(combined, encFinal)
		for k, p := range fr.topDist {
			if p != 0 {
				m.addEmbRowScaled(combined, m.TopSeedEmb, k, p)
			}
		}
		input = combined
	} else if fr.Top >= 0 && m.TopSeedEmb != nil {
		combined := make([]float32, hidden)
		copy(combined, encFinal)
		m.addEmbRow(combined, m.TopSeedEmb, fr.Top)
		input = combined
	}
	fr.actProbs = softmaxF(linearRow(m.ActHead, input))
	fr.Act = argmaxF(fr.actProbs)
	fr.ActConf = fr.actProbs[fr.Act]
	fr.toneProbs = softmaxF(linearRow(m.ToneHead, input))
	fr.Tone = argmaxF(fr.toneProbs)
	fr.ToneConf = fr.toneProbs[fr.Tone]
}

// frameContext carries the repair loop's fixed inputs.
type frameContext struct {
	surfaces  []string
	hasTop    bool
	hasSlots  bool
	topFixed  bool
	lexTone   int // tone re-read from the input surfaces (-1 = check not applicable)
	triedTop  map[int]bool
	triedAct  map[int]bool
	triedTone map[int]bool
}

// repairFrame runs the re-correction loop (DESIGN §12): the
// pre-render checks, then render + round-trip, re-picking the
// lowest-confidence offending slot (others held fixed) at most
// MaxRepairPicks times. Returns the repair log, whether the
// frame is still inconsistent at the bound, and the render made
// under the settled frame when one exists (nil when the bound
// was hit on a pre-render check — the caller then renders with
// the best frame it has). The frame is updated in place.
func (m *LayersModel) repairFrame(fr *FrameState, encFinal []float32, surfaces []string, hasTop, hasSlots, topFixed bool, render func(FrameState) (rendered, error)) ([]string, bool, *rendered, error) {
	ctx := &frameContext{
		surfaces:  surfaces,
		hasTop:    hasTop,
		hasSlots:  hasSlots,
		topFixed:  topFixed,
		lexTone:   -1,
		triedTop:  map[int]bool{},
		triedAct:  map[int]bool{},
		triedTone: map[int]bool{},
	}
	if hasSlots {
		ctx.lexTone = m.ToneLex.ToneOf(surfaces)
	}
	if fr.Top >= 0 {
		ctx.triedTop[fr.Top] = true
	}
	if fr.Act >= 0 {
		ctx.triedAct[fr.Act] = true
	}
	if fr.Tone >= 0 {
		ctx.triedTone[fr.Tone] = true
	}
	var log []string
	picks := 0
	var lastRender *rendered
	for {
		// Pre-render checks (a) act-family, (b) tone-vs-lexicon.
		if offender, why, failed := m.preRenderOffender(fr, ctx); failed {
			if picks >= MaxRepairPicks {
				return log, true, nil, nil
			}
			note, ok := m.repick(fr, encFinal, ctx, offender)
			if !ok {
				return log, true, nil, nil
			}
			picks++
			lastRender = nil
			log = append(log, fmt.Sprintf("%s: %s", why, note))
			continue
		}
		if lastRender == nil {
			r, err := render(*fr)
			if err != nil {
				return log, false, nil, err
			}
			lastRender = &r
		}
		// Round-trip check (d): the rendered reply's own frame.
		if offender, why, failed := m.roundTripOffender(fr, *lastRender); failed {
			if ctx.topFixed && offender == "top" {
				return log, true, lastRender, nil // the shuffled top is the measurement; its mismatch is the datum
			}
			if picks >= MaxRepairPicks {
				return log, true, lastRender, nil
			}
			note, ok := m.repick(fr, encFinal, ctx, offender)
			if !ok {
				return log, true, lastRender, nil
			}
			picks++
			lastRender = nil
			log = append(log, fmt.Sprintf("%s: %s", why, note))
			continue
		}
		return log, false, lastRender, nil
	}
}

// preRenderOffender evaluates checks (a) and (b) and names the
// offending slot ("act", "tone", or "top" — (a) can indict either
// side of the pair; the confidence comparison happens in repick).
func (m *LayersModel) preRenderOffender(fr *FrameState, ctx *frameContext) (offender, why string, failed bool) {
	// (b) tone vs the lexicon re-read: the lexicon vetoes only on
	// positive evidence (a non-neutral read); a neutral read
	// abstains and the head's tone stands.
	if ctx.hasSlots && ctx.lexTone > 0 && fr.Tone >= 0 && fr.Tone != ctx.lexTone {
		return "tone", fmt.Sprintf("tone check: slot=%d lexicon=%d", fr.Tone, ctx.lexTone), true
	}
	// (a) the act must belong to the top code's family.
	if ctx.hasTop && ctx.hasSlots && fr.Top >= 0 && fr.Act >= 0 && m.ActFamily != nil {
		if !m.ActFamily[fr.Top][fr.Act] {
			return "act|top", fmt.Sprintf("act-family check: act=%d not in family of top=%d", fr.Act, fr.Top), true
		}
	}
	return "", "", false
}

// roundTripOffender evaluates check (d): the frame re-derived
// from the rendered reply must agree with the intended frame —
// top code by the centroids, tone by the lexicon over the reply's
// words (same positive-evidence rule as (b)).
func (m *LayersModel) roundTripOffender(fr *FrameState, r rendered) (offender, why string, failed bool) {
	if fr.Top >= 0 {
		if rt, ok := m.DeriveReplyTop(r.ids); ok && rt != fr.Top {
			return "top", fmt.Sprintf("round-trip: rendered top=%d intended=%d", rt, fr.Top), true
		}
	}
	if fr.Tone >= 0 {
		words := make([]string, len(r.ids))
		for i, id := range r.ids {
			words[i] = m.Base.OutputVocab.GetWord(id)
		}
		if lt := m.ToneLex.ToneOf(words); lt > 0 && lt != fr.Tone {
			return "tone", fmt.Sprintf("round-trip: rendered tone=%d intended=%d", lt, fr.Tone), true
		}
	}
	return "", "", false
}

// repick replaces the offending slot with the head's best untried
// value that satisfies the checks it can be judged against, the
// other slots held fixed. For the act-family pair it re-picks the
// lower-confidence side. Returns a log note and success.
func (m *LayersModel) repick(fr *FrameState, encFinal []float32, ctx *frameContext, offender string) (string, bool) {
	switch offender {
	case "tone":
		// The only tone that passes check (b) is the lexicon read;
		// for a round-trip tone failure the constraint is the
		// head's next-best tone that still passes (b). The
		// lexicon value is NOT subject to the tried set: a top
		// re-pick refills the slots and can clobber a tone this
		// loop already repaired, and re-applying the read against
		// the new frame is a fresh pick (still bounded by
		// MaxRepairPicks), not a repeat of a failed one.
		if ctx.lexTone > 0 {
			if fr.Tone == ctx.lexTone {
				return "", false
			}
			old := fr.Tone
			fr.Tone = ctx.lexTone
			fr.ToneConf = probOf(fr.toneProbs, fr.Tone)
			ctx.triedTone[fr.Tone] = true
			return fmt.Sprintf("tone %d->%d (lexicon read)", old, fr.Tone), true
		}
		v, ok := pickBest(fr.toneProbs, ctx.triedTone, nil)
		if !ok {
			return "", false
		}
		old := fr.Tone
		fr.Tone = v
		fr.ToneConf = probOf(fr.toneProbs, v)
		ctx.triedTone[v] = true
		return fmt.Sprintf("tone %d->%d (next best)", old, v), true
	case "act|top":
		// Re-pick the lower-confidence side of the pair.
		if fr.ActConf <= fr.TopConf || ctx.topFixed {
			v, ok := pickBest(fr.actProbs, ctx.triedAct, func(a int) bool {
				return m.ActFamily[fr.Top][a]
			})
			if !ok {
				return "", false
			}
			old := fr.Act
			fr.Act = v
			fr.ActConf = probOf(fr.actProbs, v)
			ctx.triedAct[v] = true
			return fmt.Sprintf("act %d->%d (best in family of top=%d)", old, v, fr.Top), true
		}
		v, ok := pickBest(fr.topProbs, ctx.triedTop, func(t int) bool {
			return m.ActFamily[t][fr.Act]
		})
		if !ok {
			return "", false
		}
		old := fr.Top
		fr.Top = v
		fr.TopConf = probOf(fr.topProbs, v)
		ctx.triedTop[v] = true
		fr.commitDist() // the checker's re-pick is a commitment (soft models)
		m.refillSlots(fr, encFinal)
		ctx.triedAct[fr.Act] = true
		ctx.triedTone[fr.Tone] = true
		return fmt.Sprintf("top %d->%d (best whose family holds act)", old, v), true
	case "top":
		if ctx.topFixed {
			return "", false
		}
		v, ok := pickBest(fr.topProbs, ctx.triedTop, nil)
		if !ok {
			return "", false
		}
		old := fr.Top
		fr.Top = v
		fr.TopConf = probOf(fr.topProbs, v)
		ctx.triedTop[v] = true
		fr.commitDist() // the checker's re-pick is a commitment (soft models)
		if ctx.hasSlots {
			m.refillSlots(fr, encFinal)
			ctx.triedAct[fr.Act] = true
			ctx.triedTone[fr.Tone] = true
		}
		return fmt.Sprintf("top %d->%d (next best)", old, v), true
	case "act":
		v, ok := pickBest(fr.actProbs, ctx.triedAct, nil)
		if !ok {
			return "", false
		}
		old := fr.Act
		fr.Act = v
		fr.ActConf = probOf(fr.actProbs, v)
		ctx.triedAct[v] = true
		return fmt.Sprintf("act %d->%d (next best)", old, v), true
	}
	return "", false
}

// pickBest returns the highest-probability value not yet tried
// and (when allowed is non-nil) passing allowed.
func pickBest(probs []float32, tried map[int]bool, allowed func(int) bool) (int, bool) {
	best, bestP := -1, float32(-1)
	for v, p := range probs {
		if tried[v] || (allowed != nil && !allowed(v)) {
			continue
		}
		if p > bestP {
			best, bestP = v, p
		}
	}
	return best, best >= 0
}

func probOf(probs []float32, v int) float32 {
	if v >= 0 && v < len(probs) {
		return probs[v]
	}
	return 0
}

func softmaxF(logits []float32) []float32 {
	out := make([]float32, len(logits))
	mx := float32(-1e30)
	for _, v := range logits {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range logits {
		out[i] = float32(math.Exp(float64(v - mx)))
		sum += out[i]
	}
	if sum > 0 {
		for i := range out {
			out[i] /= sum
		}
	}
	return out
}

// renderReply runs the greedy word loop under a settled frame:
// seed = encoder final + slot rows + turn row; the top code's
// prefix step first; copy mixing per step when the layer is on.
func (m *LayersModel) renderReply(encOut, encCell *tensor.Tensor, encFinal []float32, frame FrameState, inputIDs []int, opts LayersPredictOptions, maxLen int) ([]int, string, error) {
	cfg := m.Config
	base := m.Base
	vocab := base.OutputVocab
	hidden := cfg.HiddenDim
	embDim := cfg.EmbedDim
	T := len(inputIDs)

	seed := make([]float32, hidden)
	copy(seed, encFinal)
	if frame.Act >= 0 && m.ActEmb != nil {
		m.addEmbRow(seed, m.ActEmb, frame.Act)
	}
	if frame.Tone >= 0 && m.ToneEmb != nil {
		m.addEmbRow(seed, m.ToneEmb, frame.Tone)
	}
	if m.layerOn("turn", opts) {
		if opts.PrevReplyIDs != nil {
			if prevTop, ok := m.DeriveReplyTop(opts.PrevReplyIDs); ok {
				m.addEmbRow(seed, m.TurnEmb, prevTop)
			} else {
				for k := 0; k < hidden; k++ {
					seed[k] += m.NoneTurn.Data[k]
				}
			}
		} else {
			for k := 0; k < hidden; k++ {
				seed[k] += m.NoneTurn.Data[k]
			}
		}
	}
	h := tensor.NewTensor([]int{1, hidden}, seed, false)
	c := encCell

	// Top-code prefix step (output discarded, as in training the
	// prefix position's label is ignored). On a soft model the
	// step consumes the blended prefix row of the frame's
	// effective distribution; on a hard model, the code's row.
	if frame.Top >= 0 && m.TopEmb != nil {
		var emb []float32
		if frame.topDist != nil {
			emb = make([]float32, embDim)
			for k, p := range frame.topDist {
				if p != 0 {
					m.addEmbRowScaled(emb, m.TopEmb, k, p)
				}
			}
		} else {
			off := frame.Top * embDim
			emb = append([]float32{}, m.TopEmb.Weight.Data[off:off+embDim]...)
		}
		_, _, nh, nc, err := m.decStep(emb, h, c)
		if err != nil {
			return nil, "", err
		}
		h, c = nh, nc
	}

	// Encoder states for the copy layer.
	var encStates [][]float32
	validLen := 0
	if m.layerOn("copy", opts) {
		encStates = make([][]float32, T)
		for t := 0; t < T; t++ {
			encStates[t] = encOut.Data[t*hidden : (t+1)*hidden]
			if inputIDs[t] != vocab.PaddingTokenID {
				validLen = t + 1
			}
		}
	}

	prevID := -1
	usedBigrams := map[[2]int]bool{}
	fedID := vocab.BosID
	fedSurface := ""
	var tokenIDs []int
	for step := 0; step < maxLen; step++ {
		emb := m.wordStepEmbedding(fedID, fedSurface)
		logits, moeRow, nh, nc, err := m.decStep(emb, h, c)
		if err != nil {
			return nil, "", err
		}
		h, c = nh, nc
		var nextID int
		if m.layerOn("copy", opts) {
			probs, _ := base.Decoder.PtrGen.ptrGenStep(moeRow, encStates, inputIDs, validLen, logits)
			nextID = argmaxBannedProbs(probs, prevID, usedBigrams, tokenIDs)
		} else {
			nextID = argmaxBannedLogits(logits, prevID, usedBigrams, tokenIDs, step)
		}
		if (nextID == vocab.EosID && len(tokenIDs) > 0) || nextID == vocab.PaddingTokenID {
			break
		}
		if nextID == vocab.UnkID {
			if step == 0 {
				// Production skips a first-step <unk> without appending.
				fedID = nextID
				fedSurface = ""
				continue
			}
			break
		}
		tokenIDs = append(tokenIDs, nextID)
		if prevID >= 0 {
			usedBigrams[[2]int{prevID, nextID}] = true
		}
		prevID = nextID
		fedID = nextID
		fedSurface = vocab.GetWord(nextID)
	}
	return tokenIDs, vocab.Decode(tokenIDs), nil
}

// decStep runs one decoder step on a prepared embedding vector:
// LSTM -> MoE -> Output, returning vocab logits and the MoE row
// (the copy layer's decoder state), plus the new LSTM state.
func (m *LayersModel) decStep(emb []float32, h, c *tensor.Tensor) (logits, moeRow []float32, nh, nc *tensor.Tensor, err error) {
	hidden := m.Config.HiddenDim
	embT := tensor.NewTensor([]int{1, len(emb)}, emb, false)
	outT, newC, err := m.Base.Decoder.LSTM.Forward(embT, h, c)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	out3, err := outT.Reshape([]int{1, 1, hidden})
	if err != nil {
		return nil, nil, nil, nil, err
	}
	moeOut, _, err := m.Base.Decoder.MoE.Forward(out3)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	lg, err := m.Base.Decoder.Output.Forward(moeOut)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	moeRow = append([]float32{}, moeOut.Data[:hidden]...)
	return append([]float32{}, lg.Data...), moeRow, outT, newC, nil
}

// wordStepEmbedding builds the decoder's input vector for one fed
// token: token row + class row + char mean (layers on only).
func (m *LayersModel) wordStepEmbedding(id int, surface string) []float32 {
	embDim := m.Config.EmbedDim
	emb := make([]float32, embDim)
	off := id * embDim
	copy(emb, m.Base.Decoder.Embedding.Weight.Data[off:off+embDim])
	if m.Config.Class && m.ClassTable != nil {
		coff := m.classOf(id) * embDim
		for k := 0; k < embDim; k++ {
			emb[k] += m.ClassTable.Weight.Data[coff+k]
		}
	}
	if m.Config.Char && m.CharTable != nil {
		tmp := make([]float32, embDim)
		m.charVectorInto(surface, m.isSpecialID(id), tmp)
		for k := 0; k < embDim; k++ {
			emb[k] += tmp[k]
		}
	}
	return emb
}

// argmaxBannedLogits mirrors PredictWithTrace's greedy bans:
// max-logit token that is not the previous token and does not
// repeat a bigram already produced.
func argmaxBannedLogits(logits []float32, prevID int, used map[[2]int]bool, generated []int, step int) int {
	best, bestV := 0, float32(-1e30)
	for i, v := range logits {
		if i == prevID {
			continue
		}
		if len(generated) > 0 && used[[2]int{generated[len(generated)-1], i}] {
			continue
		}
		if v > bestV {
			bestV, best = v, i
		}
	}
	return best
}

// argmaxBannedProbs applies the same bans in probability space.
func argmaxBannedProbs(probs []float32, prevID int, used map[[2]int]bool, generated []int) int {
	best, bestV := 0, float32(-1)
	for i, v := range probs {
		if i == prevID {
			continue
		}
		if len(generated) > 0 && used[[2]int{generated[len(generated)-1], i}] {
			continue
		}
		if v > bestV {
			bestV, best = v, i
		}
	}
	return best
}

func argmaxF(xs []float32) int {
	best, bestV := 0, float32(-1e30)
	for i, v := range xs {
		if v > bestV {
			bestV, best = v, i
		}
	}
	return best
}
