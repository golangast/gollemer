package chat

// Phrasing attempt 2, arm A: the pointer-generator mapper
// retrained with the generalization levers run 1's verdict
// pointed at. Same protocol lineage as phrasing_train.go
// (itself a speccopy clone) with four deliberate differences:
//
//   - DATA: the v2 pairs (scripts/gen_phrasing_data2) — ~5x
//     denser per family, factor-crossed with minimal pairs,
//     slot values drawn mostly from large random pools.
//   - INDEXED UNK: slot masking no longer collapses every
//     masked value into the one UNK token. Each distinct
//     masked value (and each wrapper-noise token) gets the
//     placeholder unk<k>, k ranked by first appearance in the
//     input — the same ranking PredictPtrGenVerbatimIndexed
//     assigns to out-of-vocabulary words — so two unseen names
//     in one sentence are two different tokens end to end, and
//     emitting unk<k> is an unambiguous "copy the k-th
//     unknown". See seq2seq/ptrgen_verbatim_indexed.go.
//   - VALIDATION SELECTION: a slice carved from the TRAINING
//     pairs (never the judge) is decoded every second epoch;
//     the checkpoint that ships is the best on VALIDATION
//     slot-exact, not the best on train loss. Run 1's register
//     family went 15% -> 0% while train loss fell to 0.0022 —
//     train-loss selection is the failure this removes. The
//     best-train-loss checkpoint is scored on the same slice
//     at the end and both numbers are logged.
//   - SCHEDULE: the data is ~6x run 1's, so the epoch cap and
//     LR steps compress in step count (cap 22 epochs; LR
//     halves at epochs 12 and 18).
//
// Outcome (2026-10-31): NOT EARNED. Training was stopped
// after epoch 7 when the live weights diverged (train loss
// 0.095 -> 0.254, validation 0.715 -> 0.3825 between the
// epoch-4 and epoch-6 checks); the validation champion
// (epoch 4) was already banked as the production-named gob,
// per the design. On the judge it scored single-unseen
// 44.7% slot-exact (bar 90%), compound-seenpair 9.7% (bar
// 85%), with 31 none-class false claims; the best-train-loss
// checkpoint (epoch 3, loss 0.0953) scored 45.3% / 8.4% —
// selection was not the differentiator. Both checkpoints sit
// below run 1's slot-exact rescore (62.6% / 33.1%): density,
// factor crossing, random-name pools, and indexed UNK did
// not teach the mapper to recombine, and the train-carved
// validation slice (0.715 at the champion) could not see the
// gap, because it shares template families with training
// while the judge's are held out. Arm B (classify + extract)
// cleared both bars and is seated instead; this arm stays as
// the measured record.
//
// Output: data/models/gob_models/real_tiny_seq2seq_phrasing2.gob
// (the best-validation snapshot). Run 1's gob, data, and
// checkpoints are never touched: every path here is keyed on
// Phrasing2Domain.

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

const (
	phrasing2MaxEpochs    = 22
	phrasing2ValEvery     = 2   // epochs between validation checks
	phrasing2ValPatience  = 5   // checks without improvement before stopping
	phrasing2ValCheckSize = 400 // decoded validation pairs per check
	phrasing2NoisePct     = 12  // wrapper-noise UNK rate (run 1: 5)
)

// preparePhrasing2TrainData loads the v2 pairs, carves the
// validation slice (7%, seeded — a training-data property,
// never the judge), and builds the vocabulary, placeholders
// included.
func preparePhrasing2TrainData(projectRoot string) (train, val []phrasingPair, v *vocab.Vocabulary, tok *tokenizer.Tokenizer, err error) {
	path := Phrasing2DatasetPath(projectRoot)
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("phrasing2 data: %w (generate it first: go run ./scripts/gen_phrasing_data2)", err)
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
			return nil, nil, nil, nil, fmt.Errorf("phrasing2 data: %w", err)
		}
		if p.Domain == PhrasingDomain {
			pairs = append(pairs, p)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, nil, nil, nil, err
	}
	if len(pairs) < 100 {
		return nil, nil, nil, nil, fmt.Errorf("need at least 100 phrasing2 pairs, have %d", len(pairs))
	}
	shuf := rand.New(rand.NewSource(20261031))
	shuf.Shuffle(len(pairs), func(i, j int) { pairs[i], pairs[j] = pairs[j], pairs[i] })
	nVal := len(pairs) * 7 / 100
	val, train = pairs[:nVal], pairs[nVal:]

	v = vocab.NewVocabulary()
	for _, p := range train {
		for _, t := range tokenizeLower(p.Input) {
			v.AddToken(t)
		}
		for _, t := range tokenizeLower(p.Output) {
			v.AddToken(t)
		}
	}
	for k := 1; k <= seq2seq.MaxIndexedUnk; k++ {
		v.AddToken(seq2seq.IndexedUnkToken(k))
	}
	tok, err = tokenizer.NewTokenizer(v)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	log.Printf("[REAL-SEQ2SEQ] phrasing2: %d train pairs, %d validation pairs, vocab size=%d", len(train), len(val), v.Size())
	return train, val, v, tok, nil
}

// seqKey identifies one slot value by its token sequence.
func seqKey(seq []string) string { return strings.Join(seq, "\x00") }

// encodePhrasing2Pair encodes one pair with the indexed
// masking discipline. Modes (rng, fixed per pair as in run 1):
// 40% clean, 35% one slot masked, 25% every slot masked.
// Masked slot values and wrapper-noise tokens are replaced by
// indexed placeholders ranked by first appearance in the
// input; a value's every token takes its one placeholder, in
// input and target alike — the exact convention
// PredictPtrGenVerbatimIndexed decodes against.
func encodePhrasing2Pair(v *vocab.Vocabulary, p phrasingPair, rng *rand.Rand) encodedPair {
	inToks := seq2seq.VerbatimTokens(strings.ToLower(p.Input))
	outToks := seq2seq.VerbatimTokens(strings.ToLower(p.Output))
	seqs := slotTokenSeqs(p.Slots)

	masked := map[string]bool{} // seqKey -> masked this pair
	if len(seqs) > 0 {
		switch mode := rng.Intn(100); {
		case mode < 40:
			// clean: slots stay vocabulary words
		case mode < 75:
			masked[seqKey(seqs[rng.Intn(len(seqs))])] = true
		default:
			for _, s := range seqs {
				masked[seqKey(s)] = true
			}
		}
	}

	// Spans of the input covered by ANY slot occurrence —
	// noise never lands inside a slot value, masked or not
	// (the target keeps unmasked values literal; an UNK inside
	// one would teach corruption).
	inSlotSpan := make([]bool, len(inToks))
	for _, s := range seqs {
		for i := 0; i+len(s) <= len(inToks); i++ {
			match := true
			for j := range s {
				if inToks[i+j] != s[j] {
					match = false
					break
				}
			}
			if match {
				for j := range s {
					inSlotSpan[i+j] = true
				}
			}
		}
	}
	inTarget := map[string]bool{}
	for _, t := range outToks {
		inTarget[t] = true
	}
	isNoise := make([]bool, len(inToks))
	for i, t := range inToks {
		if !inSlotSpan[i] && !inTarget[t] && t != "\"" && rng.Intn(100) < phrasing2NoisePct {
			isNoise[i] = true
		}
	}

	// Assign placeholder indices by first appearance in the
	// input, then mask.
	indexOf := map[string]int{}
	nextIdx := 0
	placeholder := func(key string) string {
		k, ok := indexOf[key]
		if !ok {
			nextIdx++
			k = nextIdx
			indexOf[key] = k
		}
		if k > seq2seq.MaxIndexedUnk {
			return "unk"
		}
		return seq2seq.IndexedUnkToken(k)
	}
	maskedAt := func(i int) (seq []string, key string, is bool) {
		var best []string
		for _, s := range seqs {
			if !masked[seqKey(s)] || i+len(s) > len(inToks) {
				continue
			}
			match := true
			for j := range s {
				if inToks[i+j] != s[j] {
					match = false
					break
				}
			}
			if match && len(s) > len(best) {
				best = s
			}
		}
		if best == nil {
			return nil, "", false
		}
		return best, "s:" + seqKey(best), true
	}
	maskedIn := make([]string, 0, len(inToks))
	for i := 0; i < len(inToks); {
		if s, key, is := maskedAt(i); is {
			ph := placeholder(key)
			for range s {
				maskedIn = append(maskedIn, ph)
			}
			i += len(s)
			continue
		}
		if isNoise[i] {
			maskedIn = append(maskedIn, placeholder("n:"+inToks[i]))
			i++
			continue
		}
		maskedIn = append(maskedIn, inToks[i])
		i++
	}
	// Mask the target: every masked value, at every occurrence,
	// with its input-assigned placeholder. A masked value that
	// never appeared in the input (not expected — the
	// generator carries every slot) takes the next index in
	// slot order.
	maskedOut := make([]string, 0, len(outToks))
	for i := 0; i < len(outToks); {
		var best []string
		for _, s := range seqs {
			if !masked[seqKey(s)] || i+len(s) > len(outToks) {
				continue
			}
			match := true
			for j := range s {
				if outToks[i+j] != s[j] {
					match = false
					break
				}
			}
			if match && len(s) > len(best) {
				best = s
			}
		}
		if best != nil {
			ph := placeholder("s:" + seqKey(best))
			for range best {
				maskedOut = append(maskedOut, ph)
			}
			i += len(best)
			continue
		}
		maskedOut = append(maskedOut, outToks[i])
		i++
	}

	in := make([]int, 0, len(maskedIn))
	for _, t := range maskedIn {
		in = append(in, v.GetTokenID(t))
	}
	tg := make([]int, 0, len(maskedOut)+2)
	tg = append(tg, v.BosID)
	for _, t := range maskedOut {
		tg = append(tg, v.GetTokenID(t))
	}
	tg = append(tg, v.EosID)
	return encodedPair{input: in, target: tg}
}

// phrasing2ValScore decodes up to limit validation pairs with
// the indexed decoder and returns the slot-exact fraction —
// the checkpoint-selection metric (never computed on the
// judge).
func phrasing2ValScore(model *seq2seq.Seq2Seq, val []phrasingPair, limit int) float64 {
	n, exact := 0, 0
	for _, p := range val {
		if n >= limit {
			break
		}
		n++
		goldSlots, gok := CanonicalPiecesSlots(p.Output)
		if !gok {
			continue
		}
		raw, err := model.PredictPtrGenVerbatimIndexed(p.Input, 96)
		if err != nil {
			continue
		}
		got := PhrasingPieces(raw)
		if len(goldSlots) == 0 {
			if len(got) == 0 {
				exact++
			}
			continue
		}
		if len(got) == 0 {
			continue
		}
		gotSlots, ok := CanonicalPiecesSlots(strings.Join(got, " ## "))
		if ok && PiecesSlotExact(gotSlots, goldSlots) {
			exact++
		}
	}
	if n == 0 {
		return 0
	}
	return float64(exact) / float64(n)
}

// phrasing2ValState is the validation-selection record kept
// beside the checkpoints so a resumed run keeps its champion.
type phrasing2ValState struct {
	BestVal         float64 `json:"best_val_slot_exact"`
	BestValEpoch    int     `json:"best_val_epoch"`
	ChecksNoImprove int     `json:"checks_no_improve"`
}

func phrasing2LR(epoch int) float32 {
	switch {
	case epoch >= 18:
		return realBaseLR * 0.25
	case epoch >= 12:
		return realBaseLR * 0.5
	default:
		return realBaseLR
	}
}

// RunPhrasing2Training trains arm A: the phrasing pointer-
// generator on the v2 pairs, with validation-selected
// checkpointing.
func RunPhrasing2Training(projectRoot string, resume bool) error {
	trainPairs, valPairs, v, tok, err := preparePhrasing2TrainData(projectRoot)
	if err != nil {
		return err
	}

	rng := rand.New(rand.NewSource(20261031))
	encoded := make([]encodedPair, 0, len(trainPairs))
	masked := 0
	for _, p := range trainPairs {
		ep := encodePhrasing2Pair(v, p, rng)
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
		return fmt.Errorf("no encodable phrasing2 pairs")
	}
	log.Printf("[REAL-SEQ2SEQ] phrasing2: %d encoded pairs (%d with at least one plain-UNK target token)", len(encoded), masked)
	sort.Slice(encoded, func(i, j int) bool { return len(encoded[i].input) < len(encoded[j].input) })

	curFP := vocabFingerprint(v)
	ckptPath, metaPath := ckptPaths(projectRoot, Phrasing2Domain)
	latestPath, latestMetaPath := latestCkptPaths(projectRoot, Phrasing2Domain)
	ckptOptPath := ckptPath + ".opt"
	latestOptPath := latestPath + ".opt"
	modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", "real_tiny_seq2seq_"+Phrasing2Domain+".gob")
	bestValPath := modelPath + ".bestval"
	valStatePath := modelPath + ".valstate.json"

	valState := phrasing2ValState{BestVal: -1}
	if resume {
		if raw, rerr := os.ReadFile(valStatePath); rerr == nil {
			if jerr := json.Unmarshal(raw, &valState); jerr != nil {
				valState = phrasing2ValState{BestVal: -1}
			}
		}
		if _, serr := os.Stat(bestValPath); serr != nil {
			valState = phrasing2ValState{BestVal: -1} // champion file gone: re-earn it
		}
	}

	startEpoch := 1
	lr := phrasing2LR(1)
	best := float32(math.Inf(1))
	epochsNoImprove := 0
	lastCheckpointBest := float32(math.Inf(1))
	lastCheckpointEpoch := 0
	var model *seq2seq.Seq2Seq
	var pendingLatest *seq2seq.Seq2Seq
	var resumeOptPath string
	var resumeOptEpoch int

	if resume {
		bm, bmeta, berr := loadTrainingCheckpoint(tok, curFP, ckptPath, metaPath)
		lm, lmeta, lerr := loadTrainingCheckpoint(tok, curFP, latestPath, latestMetaPath)
		switch {
		case lerr == nil && (berr != nil || lmeta.Epoch > bmeta.Epoch):
			resumeOptPath, resumeOptEpoch = latestOptPath, lmeta.Epoch
			startEpoch = lmeta.Epoch + 1
			lr = phrasing2LR(startEpoch)
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
			log.Printf("[REAL-SEQ2SEQ] RESUMED from latest checkpoint: completed epoch=%d best=%.6f (continuing at epoch %d)", lmeta.Epoch, lmeta.Best, startEpoch)
		case berr == nil:
			resumeOptPath, resumeOptEpoch = ckptOptPath, bmeta.Epoch
			model = bm
			startEpoch = bmeta.Epoch + 1
			lr = phrasing2LR(startEpoch)
			best = float32(bmeta.Best)
			lastCheckpointBest = float32(bmeta.LastCheckpointBest)
			lastCheckpointEpoch = bmeta.Epoch
			epochsNoImprove = bmeta.EpochsNoImprove
			log.Printf("[REAL-SEQ2SEQ] RESUMED from checkpoint: completed epoch=%d best=%.6f (continuing at epoch %d)", bmeta.Epoch, bmeta.Best, startEpoch)
		default:
			log.Printf("[REAL-SEQ2SEQ] -resume given but checkpoint unusable (%v); starting fresh from epoch 1", berr)
		}
	}
	if model != nil && (model.Decoder == nil || model.Decoder.PtrGen == nil) {
		return fmt.Errorf("phrasing2 checkpoint at %s is not a pointer-generator model; refusing to resume it as one", ckptPath)
	}
	if model == nil {
		embedDim, hiddenDim := 128, 256 // the spec probe's capacity, as run 1
		model, err = seq2seq.NewSeq2Seq(v.Size(), v.Size(), embedDim, hiddenDim, tok, v)
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
	log.Printf("[REAL-SEQ2SEQ] phrasing2: %d encoded pairs, %d batches/epoch", len(encoded), len(batches))

	stop := false
	for epoch := startEpoch; epoch <= phrasing2MaxEpochs && !stop; epoch++ {
		lr = phrasing2LR(epoch)
		adam.SetLearningRate(lr)
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
			log.Printf("[REAL-SEQ2SEQ] DIVERGENCE at epoch=%d loss=%.6f (best=%.6f); rolling back (lr -> %.6f)", epoch, avg, best, lr*0.5)
			adam.RestoreParameters(bestParams)
			if bestOptState != nil {
				if err := adam.RestoreState(bestOptState); err != nil {
					log.Printf("[REAL-SEQ2SEQ] WARNING: could not restore best optimizer state on rollback: %v", err)
				}
			}
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
		log.Printf("[REAL-SEQ2SEQ] phrasing2 epoch=%d avg_loss=%.6f best=%.6f lr=%.6f", epoch, avg, best, lr)

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

		// Validation selection: decode the carved slice and
		// keep the champion. This — not train loss — decides
		// what ships.
		if epoch%phrasing2ValEvery == 0 {
			score := phrasing2ValScore(model, valPairs, phrasing2ValCheckSize)
			log.Printf("[REAL-SEQ2SEQ] phrasing2 VALIDATION epoch=%d slot-exact=%.4f (best=%.4f @ epoch %d)", epoch, score, valState.BestVal, valState.BestValEpoch)
			if score > valState.BestVal {
				valState.BestVal = score
				valState.BestValEpoch = epoch
				valState.ChecksNoImprove = 0
				tmp := bestValPath + ".tmp"
				if err := model.Save(tmp); err != nil {
					return err
				}
				if err := os.Rename(tmp, bestValPath); err != nil {
					return err
				}
				raw, _ := json.MarshalIndent(valState, "", "  ")
				if err := os.WriteFile(valStatePath, raw, 0o644); err != nil {
					return err
				}
				log.Printf("[REAL-SEQ2SEQ] phrasing2 new validation champion: %.4f @ epoch %d -> %s", score, epoch, bestValPath)
			} else {
				valState.ChecksNoImprove++
				raw, _ := json.MarshalIndent(valState, "", "  ")
				_ = os.WriteFile(valStatePath, raw, 0o644)
				if valState.ChecksNoImprove >= phrasing2ValPatience {
					log.Printf("[REAL-SEQ2SEQ] phrasing2 early stopping: %d validation checks without improvement", valState.ChecksNoImprove)
					stop = true
				}
			}
		}
	}

	// Report the best-train-loss checkpoint on the same
	// validation slice before anything is cleaned up: the
	// run-1 selection rule, measured on this run's data.
	if _, err := os.Stat(ckptPath); err == nil {
		if bm, _, lerr := loadTrainingCheckpoint(tok, curFP, ckptPath, metaPath); lerr == nil {
			s := phrasing2ValScore(bm, valPairs, phrasing2ValCheckSize)
			log.Printf("[REAL-SEQ2SEQ] phrasing2 best-TRAIN-loss checkpoint validation slot-exact=%.4f (vs best-validation %.4f @ epoch %d)", s, valState.BestVal, valState.BestValEpoch)
			// Keep it beside the production gob for the report.
			if raw, rerr := os.ReadFile(ckptPath); rerr == nil {
				_ = os.WriteFile(modelPath+".trainbest", raw, 0o644)
			}
		} else {
			log.Printf("[REAL-SEQ2SEQ] phrasing2 could not score best-train checkpoint: %v", lerr)
		}
	}

	// Ship the validation champion as the production gob.
	if _, err := os.Stat(bestValPath); err != nil {
		return fmt.Errorf("phrasing2: no validation champion was ever saved (%v) — refusing to ship a train-loss pick", err)
	}
	if raw, err := os.ReadFile(bestValPath); err == nil {
		if err := os.WriteFile(modelPath+".tmp", raw, 0o644); err != nil {
			return err
		}
		if err := os.Rename(modelPath+".tmp", modelPath); err != nil {
			return err
		}
	} else {
		return err
	}
	log.Printf("[REAL-SEQ2SEQ] phrasing2 saved best-validation model (val slot-exact=%.4f @ epoch %d, best train loss=%.6f) to %s",
		valState.BestVal, valState.BestValEpoch, best, modelPath)

	for _, p := range []string{ckptPath, metaPath, latestPath, latestMetaPath, ckptOptPath, latestOptPath, bestValPath, valStatePath} {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			log.Printf("[REAL-SEQ2SEQ] warning: could not remove %s: %v", p, err)
		}
	}
	return nil
}
