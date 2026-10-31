package chat

// Phrasing-layer training: messy sentence -> canonical
// sentence(s), with a pointer-generator model (seq2seq.PointerGen).
//
// The protocol deliberately mirrors the speccopy trainer —
// same batch size, LR schedule shape, gradient clipping,
// divergence rollback, best/latest checkpoint discipline with
// persisted Adam state, early stopping, and epoch cap — with
// the differences the task itself dictates:
//
//   - the data is the phrasing generator's pairs
//     (preparePhrasingTrainData): messy inputs over the whole
//     add-family, canonical targets (compounds joined by the
//     " ## " separator), and a "none" target for lines the
//     layer must never claim;
//   - inputs are encoded with seq2seq.VerbatimTokens, not the
//     chat tokenizer: file names ("jim/main.go") and type
//     expressions ("time.Duration") must survive as the single
//     tokens the vocabulary knows — NormalizeQuery would drop
//     their dots and the copy mechanism could never restore
//     them. Training and inference (PredictPtrGenVerbatim) use
//     the identical stream;
//   - SLOT MASKING: each pair's slot values (the generator
//     records them) are replaced by the UNK token in both the
//     input and the target for a fraction of pairs — one slot
//     for some, every slot for others. The layer's held-out
//     world is exactly that condition (its name pools are
//     disjoint from training), so the model practices emitting
//     UNK by copying the attended position, which the verbatim
//     decoder turns back into the user's own word. A light
//     wrapper-noise pass also UNKs occasional input-only
//     filler words, so unseen wrapper vocabulary in the
//     encoder is a practiced condition too.
//
// Output: data/models/gob_models/real_tiny_seq2seq_phrasing.gob,
// written only on completion, from the best weights. No other
// model file is read or written by this trainer.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// phrasingPair is one training row of phrasing_pairs.jsonl:
// the chat pair plus the generator's slot record.
type phrasingPair struct {
	Input  string   `json:"input"`
	Output string   `json:"output"`
	Domain string   `json:"domain"`
	Slots  []string `json:"slots"`
}

// phrasingTrainData is the phrasing counterpart of
// domainTrainData: pairs with slots, vocabulary, tokenizer.
type phrasingTrainData struct {
	pairs []phrasingPair
	v     *vocab.Vocabulary
	tok   *tokenizer.Tokenizer
}

// preparePhrasingTrainData loads the phrasing pairs and builds
// the training vocabulary from them (train-only by
// construction — the held-out file is a different file and is
// never opened here).
func preparePhrasingTrainData(projectRoot string) (*phrasingTrainData, error) {
	path := PhrasingDatasetPath(projectRoot)
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("phrasing data: %w (generate it first: go run ./scripts/gen_phrasing_data)", err)
	}
	defer f.Close()
	var pairs []phrasingPair
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var p phrasingPair
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			return nil, fmt.Errorf("phrasing data: %w", err)
		}
		if p.Domain == PhrasingDomain {
			pairs = append(pairs, p)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	if len(pairs) < 10 {
		return nil, fmt.Errorf("need at least 10 phrasing pairs, have %d", len(pairs))
	}
	v := vocab.NewVocabulary()
	for _, p := range pairs {
		for _, t := range tokenizeLower(p.Input) {
			v.AddToken(t)
		}
		for _, t := range tokenizeLower(p.Output) {
			v.AddToken(t)
		}
	}
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		return nil, err
	}
	log.Printf("[REAL-SEQ2SEQ] phrasing: %d pairs, vocab size=%d", len(pairs), v.Size())
	return &phrasingTrainData{pairs: pairs, v: v, tok: tok}, nil
}

// slotTokenSeqs returns each slot value's token sequence in
// the verbatim tokenization (lowercased) — the exact runs the
// masking pass looks for in the input and target streams.
func slotTokenSeqs(slots []string) [][]string {
	var out [][]string
	for _, s := range slots {
		toks := seq2seq.VerbatimTokens(strings.ToLower(s))
		if len(toks) > 0 {
			out = append(out, toks)
		}
	}
	return out
}

// maskSeq replaces every occurrence of the token sequence seq
// in toks with the UNK token, one UNK per replaced token (a
// two-word print text becomes two UNKs, matching the two
// positions the decoder must copy).
func maskSeq(toks []string, seq []string) []string {
	if len(seq) == 0 || len(toks) < len(seq) {
		return toks
	}
	out := make([]string, 0, len(toks))
	for i := 0; i < len(toks); {
		match := i+len(seq) <= len(toks)
		if match {
			for j, st := range seq {
				if toks[i+j] != st {
					match = false
					break
				}
			}
		}
		if match {
			for range seq {
				out = append(out, "unk")
			}
			i += len(seq)
		} else {
			out = append(out, toks[i])
			i++
		}
	}
	return out
}

// encodePhrasingPair encodes one pair with the masking
// discipline in the file header. rng decides this pair's mode
// (fixed for the whole run — masking is a data property, like
// the probes' splits): 40% clean, 35% one slot masked, 25%
// every slot masked. Wrapper noise applies in every mode.
func encodePhrasingPair(v *vocab.Vocabulary, p phrasingPair, rng *rand.Rand) encodedPair {
	inToks := seq2seq.VerbatimTokens(strings.ToLower(p.Input))
	outToks := seq2seq.VerbatimTokens(strings.ToLower(p.Output))
	seqs := slotTokenSeqs(p.Slots)
	if len(seqs) > 0 {
		switch mode := rng.Intn(100); {
		case mode < 40:
			// clean: slots stay as vocabulary words
		case mode < 75:
			seq := seqs[rng.Intn(len(seqs))]
			inToks = maskSeq(inToks, seq)
			outToks = maskSeq(outToks, seq)
		default:
			for _, seq := range seqs {
				inToks = maskSeq(inToks, seq)
				outToks = maskSeq(outToks, seq)
			}
		}
	}
	// Wrapper noise: an input token that appears nowhere in
	// the target and is not punctuation may be UNKed — the
	// encoder learns that an unknown filler word carries no
	// slot and no command.
	inTarget := map[string]bool{}
	for _, t := range outToks {
		inTarget[t] = true
	}
	for i, t := range inToks {
		if !inTarget[t] && t != "\"" && rng.Intn(100) < 5 {
			inToks[i] = "unk"
		}
	}
	in := make([]int, 0, len(inToks))
	for _, t := range inToks {
		in = append(in, v.GetTokenID(t))
	}
	tg := make([]int, 0, len(outToks)+2)
	tg = append(tg, v.BosID)
	for _, t := range outToks {
		tg = append(tg, v.GetTokenID(t))
	}
	tg = append(tg, v.EosID)
	return encodedPair{input: in, target: tg}
}

// RunPhrasingTraining trains the phrasing-layer pointer-generator.
func RunPhrasingTraining(projectRoot string, resume bool) error {
	dd, err := preparePhrasingTrainData(projectRoot)
	if err != nil {
		return err
	}
	v := dd.v

	rng := rand.New(rand.NewSource(20261030))
	encoded := make([]encodedPair, 0, len(dd.pairs))
	masked := 0
	for _, p := range dd.pairs {
		ep := encodePhrasingPair(v, p, rng)
		if len(ep.input) == 0 || len(ep.target) < 2 || len(ep.target) > 82 || len(ep.input) > 64 {
			continue
		}
		for _, id := range ep.target {
			if id == v.UnkID {
				masked++
				break
			}
		}
		encoded = append(encoded, ep)
	}
	if len(encoded) == 0 {
		return fmt.Errorf("no encodable phrasing pairs")
	}
	log.Printf("[REAL-SEQ2SEQ] phrasing: %d encoded pairs (%d with at least one masked slot)", len(encoded), masked)
	sort.Slice(encoded, func(i, j int) bool { return len(encoded[i].input) < len(encoded[j].input) })

	curFP := vocabFingerprint(v)
	ckptPath, metaPath := ckptPaths(projectRoot, PhrasingDomain)
	latestPath, latestMetaPath := latestCkptPaths(projectRoot, PhrasingDomain)
	ckptOptPath := ckptPath + ".opt"
	latestOptPath := latestPath + ".opt"
	modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", "real_tiny_seq2seq_"+PhrasingDomain+".gob")

	startEpoch := 1
	lr := float32(realBaseLR)
	best := float32(math.Inf(1))
	epochsNoImprove := 0
	lastCheckpointBest := float32(math.Inf(1))
	lastCheckpointEpoch := 0
	var model *seq2seq.Seq2Seq
	var pendingLatest *seq2seq.Seq2Seq
	var resumeOptPath string
	var resumeOptEpoch int

	if resume {
		bm, bmeta, berr := loadTrainingCheckpoint(dd.tok, curFP, ckptPath, metaPath)
		lm, lmeta, lerr := loadTrainingCheckpoint(dd.tok, curFP, latestPath, latestMetaPath)
		switch {
		case lerr == nil && (berr != nil || lmeta.Epoch > bmeta.Epoch):
			resumeOptPath, resumeOptEpoch = latestOptPath, lmeta.Epoch
			startEpoch = lmeta.Epoch + 1
			lr = lmeta.LR
			best = float32(lmeta.Best)
			epochsNoImprove = lmeta.EpochsNoImprove
			if berr == nil {
				lastCheckpointBest = float32(bmeta.LastCheckpointBest)
				lastCheckpointEpoch = bmeta.Epoch
			} else {
				lastCheckpointBest = float32(lmeta.LastCheckpointBest)
			}
			if berr == nil && !(lmeta.Best < lmeta.LastCheckpointBest) {
				model = bm
				pendingLatest = lm
			} else {
				model = lm
			}
			log.Printf("[REAL-SEQ2SEQ] RESUMED from latest checkpoint: completed epoch=%d best=%.6f lr=%.6f no-improve=%d (continuing at epoch %d)",
				lmeta.Epoch, lmeta.Best, lmeta.LR, lmeta.EpochsNoImprove, startEpoch)
		case berr == nil:
			resumeOptPath, resumeOptEpoch = ckptOptPath, bmeta.Epoch
			model = bm
			startEpoch = bmeta.Epoch + 1
			lr = bmeta.LR
			best = float32(bmeta.Best)
			lastCheckpointBest = float32(bmeta.LastCheckpointBest)
			lastCheckpointEpoch = bmeta.Epoch
			epochsNoImprove = bmeta.EpochsNoImprove
			log.Printf("[REAL-SEQ2SEQ] RESUMED from checkpoint: completed epoch=%d best=%.6f lr=%.6f (continuing at epoch %d)",
				bmeta.Epoch, bmeta.Best, bmeta.LR, startEpoch)
		default:
			log.Printf("[REAL-SEQ2SEQ] -resume given but checkpoint unusable (%v); starting fresh from epoch 1", berr)
		}
	}
	if model != nil && (model.Decoder == nil || model.Decoder.PtrGen == nil) {
		return fmt.Errorf("phrasing checkpoint at %s is not a pointer-generator model; refusing to resume it as one", ckptPath)
	}
	if model == nil {
		embedDim, hiddenDim := 128, 256 // the spec probe's capacity
		model, err = seq2seq.NewSeq2Seq(v.Size(), v.Size(), embedDim, hiddenDim, dd.tok, v)
		if err != nil {
			return err
		}
		model.Decoder.MoE = nil
		model.Decoder.PtrGen = seq2seq.NewPointerGen(hiddenDim)
	}
	opt := nn.NewOptimizer(model.Parameters(), realBaseLR, realClip)
	adam, ok := opt.(*nn.Adam)
	if !ok {
		return fmt.Errorf("optimizer is not *nn.Adam")
	}
	adam.SetLearningRate(lr)
	optRestored := false
	if resumeOptPath != "" {
		if err := loadSpecCopyOptState(resumeOptPath, resumeOptEpoch, curFP, adam); err != nil {
			log.Printf("[REAL-SEQ2SEQ] WARNING: optimizer state at %s unusable (%v); resuming with FRESH Adam — expect a post-resume loss spike", resumeOptPath, err)
		} else {
			optRestored = true
			log.Printf("[REAL-SEQ2SEQ] optimizer state restored from %s (adam step=%d)", resumeOptPath, adam.SnapshotState().T)
		}
	}
	bestParams := adam.SnapshotParameters()
	var bestOptState *nn.OptimizerState
	if optRestored {
		bestOptState = adam.SnapshotState()
	}
	if pendingLatest != nil {
		if err := copyWeightsInto(model, pendingLatest); err != nil {
			return err
		}
		pendingLatest = nil
	}
	rollbacks := 0

	padID := v.PaddingTokenID
	batches := makeBatches(encoded, realBatchSize, padID)
	log.Printf("[REAL-SEQ2SEQ] %d encoded pairs, %d batches/epoch, lr=%.6f", len(encoded), len(batches), lr)

	for epoch := startEpoch; epoch <= realMaxEpochs; epoch++ {
		if epoch > 1 && (epoch-1)%40 == 0 {
			lr *= 0.5
			adam.SetLearningRate(lr)
		}
		var epochLoss float64
		for _, b := range batches {
			adam.ZeroGrad()
			loss, err := seq2seq.TrainBatchPtrGen(model, b.inputs, b.targets, padID)
			if err != nil {
				return fmt.Errorf("epoch %d: %w", epoch, err)
			}
			epochLoss += float64(loss)
			adam.ClipGradients()
			adam.Step()
		}
		avg := float32(epochLoss / float64(len(batches)))

		if math.IsNaN(float64(avg)) || math.IsInf(float64(avg), 0) || (epoch > 5 && avg > best*3) {
			rollbacks++
			log.Printf("[REAL-SEQ2SEQ] DIVERGENCE at epoch=%d loss=%.6f (best=%.6f); rolling back (lr %.6f -> %.6f)",
				epoch, avg, best, lr, lr*0.5)
			adam.RestoreParameters(bestParams)
			if bestOptState != nil {
				if err := adam.RestoreState(bestOptState); err != nil {
					log.Printf("[REAL-SEQ2SEQ] WARNING: could not restore best optimizer state on rollback: %v", err)
				}
			}
			lr *= 0.5
			adam.SetLearningRate(lr)
			if rollbacks > 5 {
				log.Printf("[REAL-SEQ2SEQ] too many rollbacks; stopping")
				break
			}
			continue
		}

		if avg < best {
			best = avg
			bestParams = adam.SnapshotParameters()
			bestOptState = adam.SnapshotState()
			epochsNoImprove = 0
		} else {
			epochsNoImprove++
		}
		log.Printf("[REAL-SEQ2SEQ] epoch=%d avg_loss=%.6f best=%.6f lr=%.6f", epoch, avg, best, lr)

		if best < lastCheckpointBest && epoch-lastCheckpointEpoch >= 2 {
			adam.RestoreParameters(bestParams)
			if bestOptState != nil {
				if err := adam.RestoreState(bestOptState); err != nil {
					log.Printf("[REAL-SEQ2SEQ] WARNING: could not restore best optimizer state for checkpoint: %v", err)
				}
			}
			meta := ckptMeta{
				Epoch:              epoch,
				Best:               float64(best),
				LastCheckpointBest: float64(best),
				LR:                 lr,
				EpochsNoImprove:    epochsNoImprove,
				VocabFP:            curFP,
			}
			if err := saveTrainingCheckpoint(model, meta, ckptPath, metaPath); err != nil {
				return err
			}
			if err := saveSpecCopyOptState(ckptOptPath, epoch, curFP, adam); err != nil {
				return err
			}
			lastCheckpointBest = best
			lastCheckpointEpoch = epoch
			log.Printf("[REAL-SEQ2SEQ] checkpointed best model (loss=%.6f) to %s", best, ckptPath)
		}
		latestLCB := lastCheckpointBest
		if math.IsInf(float64(latestLCB), 1) {
			latestLCB = best
		}
		latestMeta := ckptMeta{
			Epoch:              epoch,
			Best:               float64(best),
			LastCheckpointBest: float64(latestLCB),
			LR:                 lr,
			EpochsNoImprove:    epochsNoImprove,
			VocabFP:            curFP,
		}
		if err := saveTrainingCheckpoint(model, latestMeta, latestPath, latestMetaPath); err != nil {
			return err
		}
		if err := saveSpecCopyOptState(latestOptPath, epoch, curFP, adam); err != nil {
			return err
		}
		if epochsNoImprove >= 30 {
			log.Printf("[REAL-SEQ2SEQ] early stopping: 30 epochs without improvement")
			break
		}
	}

	adam.RestoreParameters(bestParams)
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		return err
	}
	if err := model.Save(modelPath); err != nil {
		return err
	}
	log.Printf("[REAL-SEQ2SEQ] saved best model (loss=%.6f) to %s", best, modelPath)

	for _, p := range []string{ckptPath, metaPath, latestPath, latestMetaPath, ckptOptPath, latestOptPath} {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			log.Printf("[REAL-SEQ2SEQ] warning: could not remove %s: %v", p, err)
		}
	}
	return nil
}
