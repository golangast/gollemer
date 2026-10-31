package chat

// Copy/pointer re-seat probe training: phrasing -> spec with a
// pointer-generator model (seq2seq.PointerGen).
//
// Everything except the model and its training step is DELIBERATELY
// the spec probe's own machinery, reused verbatim: same dataset file
// and vocabulary (prepareDomainTrainData on SpecDomain: train-all,
// no augmentation, no in-trainer held-out), same batch size, LR
// schedule, gradient clipping, divergence rollback, best/latest
// checkpoint discipline, early stopping, and epoch cap (the
// constants and helpers of real_seq2seq_train.go). The differences
// are exactly the ones under test:
//
//   - the decoder carries a PointerGen head and NO MoE, and
//   - batches train with seq2seq.TrainBatchPtrGen, whose loss is the
//     mixture distribution's NLL.
//
// Output: data/models/gob_models/real_tiny_seq2seq_speccopy.gob,
// written only on completion, from the best weights — the same rule
// as every other domain. Scoring is scripts/eval_speccopy, a clone
// of scripts/eval_spec reading the mixture distribution.
//
// Converged rerun (2026-10-30): v1 finished undertrained (best
// 0.055505 @ epoch 74, early stop at 104) after VM-restart resumes
// repeatedly reset Adam's moments against mid-flight weights
// (post-resume spikes ~3x best, one divergence rollback). This rerun
// changes exactly two training-protocol variables and nothing else:
// (1) the Adam state is persisted in a validated .opt sidecar next
// to each checkpoint and restored on resume — and a divergence
// rollback now restores the best-time optimizer state together with
// the best weights, keeping the pair coherent; (2) peak LR is
// halved to specCopyPeakLR. Architecture, data, vocabulary, split,
// batch size, epoch cap, early stopping, and scoring are v1's.

import (
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// specCopyPeakLR is the converged-rerun protocol's peak learning rate:
// half of v1's 0.001. One of exactly two protocol changes in the rerun
// (2026-10-30); the other is optimizer-state persistence below.
const specCopyPeakLR = 0.0005

// specCopyOptCkpt is the optimizer-state sidecar written next to a
// speccopy checkpoint: the Adam state (step counter + m/v moments)
// that belongs to the checkpoint's weights, tagged with the epoch and
// vocabulary fingerprint of the checkpoint sidecar it accompanies, so
// a stale or mismatched state is refused at load instead of silently
// corrupting the resumed run.
type specCopyOptCkpt struct {
	Epoch   int
	VocabFP uint64
	State   *nn.OptimizerState
}

func saveSpecCopyOptState(path string, epoch int, fp uint64, adam *nn.Adam) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	encErr := gob.NewEncoder(f).Encode(specCopyOptCkpt{Epoch: epoch, VocabFP: fp, State: adam.SnapshotState()})
	if cerr := f.Close(); encErr == nil {
		encErr = cerr
	}
	if encErr != nil {
		os.Remove(tmp)
		return encErr
	}
	return os.Rename(tmp, path)
}

// loadSpecCopyOptState restores the optimizer state saved with the
// checkpoint at (epoch, fp). Any problem — missing file, corrupt gob,
// epoch/fingerprint mismatch, parameter-shape mismatch — is an error;
// the caller logs it loudly and falls back to a fresh Adam.
func loadSpecCopyOptState(path string, epoch int, fp uint64, adam *nn.Adam) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var oc specCopyOptCkpt
	if err := gob.NewDecoder(f).Decode(&oc); err != nil {
		return fmt.Errorf("decoding optimizer sidecar: %w", err)
	}
	if oc.Epoch != epoch {
		return fmt.Errorf("optimizer sidecar epoch %d != checkpoint epoch %d", oc.Epoch, epoch)
	}
	if oc.VocabFP != fp {
		return fmt.Errorf("optimizer sidecar vocab fingerprint %d != current %d", oc.VocabFP, fp)
	}
	return adam.RestoreState(oc.State)
}

// RunSpecCopyTraining trains the pointer-generator spec model.
func RunSpecCopyTraining(projectRoot string, resume bool) error {
	dd, err := prepareDomainTrainData(projectRoot, SpecDomain)
	if err != nil {
		return err
	}
	trainPairs, v, tok := dd.trainPairs, dd.v, dd.tok

	encoded := make([]encodedPair, 0, len(trainPairs))
	for _, p := range trainPairs {
		in := encodeInput(tok, p.Input)
		tg := encodeTarget(tok, v, p.Output)
		if len(in) == 0 || len(tg) < 2 || len(tg) > 60 {
			continue
		}
		encoded = append(encoded, encodedPair{input: in, target: tg})
	}
	if len(encoded) == 0 {
		return fmt.Errorf("no encodable training pairs")
	}
	sort.Slice(encoded, func(i, j int) bool { return len(encoded[i].input) < len(encoded[j].input) })

	curFP := vocabFingerprint(v)
	ckptPath, metaPath := ckptPaths(projectRoot, SpecCopyDomain)
	latestPath, latestMetaPath := latestCkptPaths(projectRoot, SpecCopyDomain)
	ckptOptPath := ckptPath + ".opt"
	latestOptPath := latestPath + ".opt"
	modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", "real_tiny_seq2seq_"+SpecCopyDomain+".gob")

	startEpoch := 1
	lr := float32(specCopyPeakLR)
	best := float32(math.Inf(1))
	epochsNoImprove := 0
	lastCheckpointBest := float32(math.Inf(1))
	lastCheckpointEpoch := 0
	var model *seq2seq.Seq2Seq
	var pendingLatest *seq2seq.Seq2Seq
	// resumeOptPath/resumeOptEpoch name the optimizer sidecar paired
	// with whichever checkpoint the resume below selects.
	var resumeOptPath string
	var resumeOptEpoch int

	if resume {
		bm, bmeta, berr := loadTrainingCheckpoint(tok, curFP, ckptPath, metaPath)
		lm, lmeta, lerr := loadTrainingCheckpoint(tok, curFP, latestPath, latestMetaPath)
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
		return fmt.Errorf("speccopy checkpoint at %s is not a pointer-generator model; refusing to resume it as one", ckptPath)
	}
	if model == nil {
		embedDim, hiddenDim := dimsForDomain(SpecDomain) // 128, 256 — the spec probe's capacity
		model, err = seq2seq.NewSeq2Seq(v.Size(), v.Size(), embedDim, hiddenDim, tok, v)
		if err != nil {
			return err
		}
		// The mechanism under test replaces the MoE refinement: the
		// decoder state goes straight to the output projection, and
		// the pointer-generator head mixes generation with copying.
		model.Decoder.MoE = nil
		model.Decoder.PtrGen = seq2seq.NewPointerGen(hiddenDim)
	}
	opt := nn.NewOptimizer(model.Parameters(), specCopyPeakLR, realClip)
	adam, ok := opt.(*nn.Adam)
	if !ok {
		return fmt.Errorf("optimizer is not *nn.Adam")
	}
	adam.SetLearningRate(lr)
	// Rerun protocol change #1: restore the Adam state that was saved
	// with the resumed checkpoint, so a restart continues the
	// optimizer exactly instead of resetting its moments against
	// mid-flight weights (v1: post-resume spikes ~3x best and a
	// divergence rollback). A missing/corrupt/mismatched sidecar is
	// NOT silent: it logs a loud warning and the run continues with a
	// fresh Adam, marked as degraded in the log.
	optRestored := false
	if resumeOptPath != "" {
		if err := loadSpecCopyOptState(resumeOptPath, resumeOptEpoch, curFP, adam); err != nil {
			log.Printf("[REAL-SEQ2SEQ] WARNING: optimizer state at %s unusable (%v); resuming with FRESH Adam — expect a post-resume loss spike (v1 behavior)", resumeOptPath, err)
		} else {
			optRestored = true
			log.Printf("[REAL-SEQ2SEQ] optimizer state restored from %s (adam step=%d)", resumeOptPath, adam.SnapshotState().T)
		}
	}
	bestParams := adam.SnapshotParameters()
	// The optimizer state paired with bestParams. Updated on every
	// new best; a divergence rollback restores BOTH, so the weights
	// and the moments that produced them stay coherent (v1's
	// RestoreParameters zeroed the moments).
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
