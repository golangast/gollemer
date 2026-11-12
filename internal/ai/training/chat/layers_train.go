package chat

// Layers pilot trainer (DESIGN §5): a SEPARATE entry point from
// RunRealSeq2SeqTrainingOpts. The production trainer, its data
// prep, and the production model file are untouched; this trainer
// mirrors the production loop's discipline exactly — same prep
// (prepareDomainTrainData with zero options), same dims, batch
// size, LR schedule, clip, divergence rollback, best-on-train-loss
// selection, per-epoch .latest checkpoints, improvement-gated
// .ckpt checkpoints, vocab-fingerprint resume refusal, and the
// same ckptMeta sidecar shape — with the layers' derivations run
// before epoch 1 and the model saved under its own name
// (real_tiny_seq2seq_layers_social.gob), never the production one.

import (
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

// LayersTrainOptions configures one layers run.
type LayersTrainOptions struct {
	Layers   []string // layer names to switch on (DESIGN §3 shelf)
	Epochs   int      // 0 = the production 200
	MaxPairs int      // 0 = all prepared pairs; else a seeded subsample (smoke/screening)
	Resume   bool
}

// LayersModelPath is the layers model's file under a project root.
func LayersModelPath(projectRoot string) string {
	return filepath.Join(projectRoot, "data", "models", "gob_models", "real_tiny_seq2seq_layers_social.gob")
}

func layersCkptPaths(projectRoot string) (ckptPath, metaPath, latestPath, latestMetaPath string) {
	mp := LayersModelPath(projectRoot)
	return mp + ".ckpt", mp + ".ckpt.json", mp + ".latest", mp + ".latest.json"
}

// RunLayersTraining trains the social model with the requested
// layers on. Social domain only — the pilot's corpus and bars are
// the social ones.
func RunLayersTraining(projectRoot string, opts LayersTrainOptions) error {
	cfg := seq2seq.DefaultLayersConfig()
	for _, name := range opts.Layers {
		if err := cfg.Set(name, true); err != nil {
			return err
		}
	}
	if cfg.Turn && !cfg.Concept {
		return fmt.Errorf("layers: turn requires concept")
	}
	epochs := opts.Epochs
	if epochs <= 0 {
		epochs = realMaxEpochs
	}

	dd, err := prepareDomainTrainData(projectRoot, SocialDomain)
	if err != nil {
		return err
	}
	trainPairs, v, tok := dd.trainPairs, dd.v, dd.tok
	probes := dd.probes

	// Optional seeded subsample (smoke/screening only; full runs
	// use the whole prepared set). Deterministic under the
	// registered seed.
	if opts.MaxPairs > 0 && opts.MaxPairs < len(trainPairs) {
		rng := rand.New(rand.NewSource(layersSeed))
		idx := rng.Perm(len(trainPairs))
		sub := make([]ChatPair, 0, opts.MaxPairs)
		for i := 0; i < opts.MaxPairs; i++ {
			sub = append(sub, trainPairs[idx[i]])
		}
		trainPairs = sub
		log.Printf("[LAYERS] subsample: %d of the prepared pairs (seeded, smoke/screening)", len(trainPairs))
	}

	// Derivations from the training pairs themselves, before
	// epoch 1 (DESIGN §3/§5); artifacts land next to the model.
	deriv, err := deriveLayers(trainPairs, v, tok, cfg)
	if err != nil {
		return err
	}
	modelPath := LayersModelPath(projectRoot)
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		return err
	}
	if err := writeLayersArtifacts(modelPath, deriv, v, cfg); err != nil {
		return err
	}
	log.Printf("[LAYERS] derivations ready: fact-bearing pairs=%d intent=%v tone=%v",
		deriv.FactPairs, deriv.IntentDist, deriv.ToneDist)

	// Encode everything (with the memory prefix and gold labels).
	encoded := make([]layersEncodedPair, 0, len(trainPairs))
	for _, p := range trainPairs {
		ep, ok := encodeLayersPair(p, v, tok, cfg, deriv)
		if !ok {
			continue
		}
		encoded = append(encoded, ep)
	}
	if len(encoded) == 0 {
		return fmt.Errorf("layers: no encodable training pairs")
	}
	sort.Slice(encoded, func(i, j int) bool { return len(encoded[i].input) < len(encoded[j].input) })

	curFP := vocabFingerprint(v)
	ckptPath, metaPath, latestPath, latestMetaPath := layersCkptPaths(projectRoot)

	startEpoch := 1
	lr := float32(realBaseLR)
	best := float32(math.Inf(1))
	epochsNoImprove := 0
	lastCheckpointBest := float32(math.Inf(1))
	lastCheckpointEpoch := 0
	var model *seq2seq.LayersModel
	var pendingLatest *seq2seq.LayersModel

	checkCfg := func(m *seq2seq.LayersModel) error {
		if m.Config != cfg {
			return fmt.Errorf("layers: checkpoint config %+v does not match requested %+v; refusing to resume across configurations", m.Config, cfg)
		}
		return nil
	}

	if opts.Resume {
		bm, bmeta, berr := loadLayersCheckpoint(tok, curFP, ckptPath, metaPath)
		lm, lmeta, lerr := loadLayersCheckpoint(tok, curFP, latestPath, latestMetaPath)
		if berr == nil {
			berr = checkCfg(bm)
		}
		if lerr == nil {
			lerr = checkCfg(lm)
		}
		switch {
		case lerr == nil && (berr != nil || lmeta.Epoch > bmeta.Epoch):
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
			log.Printf("[LAYERS] RESUMED from latest checkpoint: completed epoch=%d best=%.6f lr=%.6f no-improve=%d (continuing at epoch %d)",
				lmeta.Epoch, lmeta.Best, lmeta.LR, lmeta.EpochsNoImprove, startEpoch)
		case berr == nil:
			model = bm
			startEpoch = bmeta.Epoch + 1
			lr = bmeta.LR
			best = float32(bmeta.Best)
			lastCheckpointBest = float32(bmeta.LastCheckpointBest)
			lastCheckpointEpoch = bmeta.Epoch
			epochsNoImprove = bmeta.EpochsNoImprove
			log.Printf("[LAYERS] RESUMED from checkpoint: completed epoch=%d best=%.6f lr=%.6f (continuing at epoch %d)",
				bmeta.Epoch, bmeta.Best, bmeta.LR, startEpoch)
		default:
			log.Printf("[LAYERS] -resume given but checkpoint unusable (%v); starting fresh from epoch 1", berr)
		}
	}
	if model == nil {
		embedDim, hiddenDim := dimsForDomain(SocialDomain)
		cfg.EmbedDim, cfg.HiddenDim = embedDim, hiddenDim
		base, err := seq2seq.NewSeq2Seq(v.Size(), v.Size(), embedDim, hiddenDim, tok, v)
		if err != nil {
			return err
		}
		model, err = seq2seq.NewLayersModel(base, cfg, deriv.ClassOf, deriv.Centroids)
		if err != nil {
			return err
		}
	} else {
		cfg = model.Config
	}
	opt := nn.NewOptimizer(model.Parameters(), realBaseLR, realClip)
	adam, ok := opt.(*nn.Adam)
	if !ok {
		return fmt.Errorf("optimizer is not *nn.Adam")
	}
	adam.SetLearningRate(lr)
	bestParams := adam.SnapshotParameters()
	if pendingLatest != nil {
		if err := copyLayersWeightsInto(model, pendingLatest); err != nil {
			return err
		}
		pendingLatest = nil
	}
	rollbacks := 0

	padID := v.PaddingTokenID
	batches := makeLayersBatches(encoded, realBatchSize, padID)
	log.Printf("[LAYERS] %d encoded pairs, %d batches/epoch, lr=%.6f, layers=%v, epochs<=%d",
		len(encoded), len(batches), lr, opts.Layers, epochs)

	for epoch := startEpoch; epoch <= epochs; epoch++ {
		if epoch > 1 && (epoch-1)%40 == 0 {
			lr *= 0.5
			adam.SetLearningRate(lr)
		}
		var epochLoss float64
		for _, b := range batches {
			adam.ZeroGrad()
			loss, err := model.TrainBatchLayers(b)
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
			log.Printf("[LAYERS] DIVERGENCE at epoch=%d loss=%.6f (best=%.6f); rolling back (lr %.6f -> %.6f)",
				epoch, avg, best, lr, lr*0.5)
			adam.RestoreParameters(bestParams)
			lr *= 0.5
			adam.SetLearningRate(lr)
			if rollbacks > 5 {
				log.Printf("[LAYERS] too many rollbacks; stopping")
				break
			}
			continue
		}

		if avg < best {
			best = avg
			bestParams = adam.SnapshotParameters()
			epochsNoImprove = 0
		} else {
			epochsNoImprove++
		}
		log.Printf("[LAYERS] epoch=%d avg_loss=%.6f best=%.6f lr=%.6f", epoch, avg, best, lr)

		if best < lastCheckpointBest && epoch-lastCheckpointEpoch >= 2 {
			adam.RestoreParameters(bestParams)
			meta := ckptMeta{
				Epoch:              epoch,
				Best:               float64(best),
				LastCheckpointBest: float64(best),
				LR:                 lr,
				EpochsNoImprove:    epochsNoImprove,
				VocabFP:            curFP,
			}
			if err := saveLayersCheckpoint(model, meta, ckptPath, metaPath); err != nil {
				return err
			}
			lastCheckpointBest = best
			lastCheckpointEpoch = epoch
			log.Printf("[LAYERS] checkpointed best model (loss=%.6f) to %s", best, ckptPath)
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
		if err := saveLayersCheckpoint(model, latestMeta, latestPath, latestMetaPath); err != nil {
			return err
		}
		if epochsNoImprove >= 30 {
			log.Printf("[LAYERS] early stopping: 30 epochs without improvement")
			break
		}
	}

	adam.RestoreParameters(bestParams)
	if err := model.Save(modelPath); err != nil {
		return err
	}
	log.Printf("[LAYERS] saved best model (loss=%.6f) to %s", best, modelPath)
	for _, p := range []string{ckptPath, metaPath, latestPath, latestMetaPath} {
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			log.Printf("[LAYERS] warning: could not remove %s: %v", p, err)
		}
	}

	// Held-out probes through the layers prediction path.
	log.Printf("[LAYERS] held-out probe: %d paraphrased questions the model never saw", len(probes))
	sess := &LayersSession{}
	for i, pr := range probes {
		pred, err := PredictLayersText(model, pr.in, sess, seq2seq.LayersPredictOptions{MaxLen: 40})
		if err != nil {
			log.Printf("[LAYERS] probe %d error: %v", i, err)
			continue
		}
		sess.Observe(pr.in, pred.TokenIDs)
		log.Printf("[LAYERS] Q: %s\n[LAYERS] A: %s", pr.in, tidyDecode(pred.Text))
	}
	return nil
}

// layersEncodedPair is one training pair with everything the
// layers batch needs precomputed.
type layersEncodedPair struct {
	input       []int
	inputSurfs  []string
	target      []int
	targetSurfs []string
	intent      int
	tone        int
	codes       [3]int
}

// encodeLayersPair encodes one pair for the layers trainer. The
// memory prefix (facts from the pair's OWN input) leads the input;
// labels come from the current line's words; concept codes from
// the reply's segments.
func encodeLayersPair(p ChatPair, v *vocab.Vocabulary, tok *tokenizer.Tokenizer, cfg seq2seq.LayersConfig, d *layersDerivations) (layersEncodedPair, bool) {
	words := tokenizeLower(seq2seq.NormalizeQuery(p.Input))
	ids := encodeInput(tok, p.Input)
	tg := encodeTarget(tok, v, p.Output)
	if len(ids) == 0 || len(tg) < 2 || len(tg) > 60 {
		return layersEncodedPair{}, false
	}
	ep := layersEncodedPair{target: tg}
	// Surfaces: prefer the tokenized words when they align with the
	// ids (the documented in-sync tokenization); else the vocab's
	// own words for the ids.
	if len(words) == len(ids) {
		ep.inputSurfs = append([]string{}, words...)
	} else {
		ep.inputSurfs = make([]string, len(ids))
		for i, id := range ids {
			ep.inputSurfs[i] = v.GetWord(id)
		}
	}
	ep.input = ids
	if cfg.Memory {
		facts := LayersFactStatements(p.Input)
		prefixToks := LayersFactPrefixTokens(facts)
		if len(prefixToks) > 0 {
			prefixIDs := make([]int, len(prefixToks))
			for i, t := range prefixToks {
				prefixIDs[i] = v.GetTokenID(t)
			}
			ep.input = append(prefixIDs, ep.input...)
			ep.inputSurfs = append(append([]string{}, prefixToks...), ep.inputSurfs...)
		}
	}
	ep.targetSurfs = make([]string, len(tg))
	for i, id := range tg {
		ep.targetSurfs[i] = v.GetWord(id)
	}
	if cfg.Intent {
		ep.intent = intentLabelFor(words)
	}
	if cfg.Tone {
		ep.tone = toneLabelFor(words, &d.ToneLex)
	}
	if cfg.Concept {
		reply := tg[1 : len(tg)-1]
		ep.codes = seq2seq.DeriveCodesWithCentroids(reply, d.Centroids, cfg.ConceptLen)
	}
	return ep, true
}

// makeLayersBatches pads a length-sorted slice of encoded pairs
// into LayersBatch groups of size, mirroring makeBatches.
func makeLayersBatches(enc []layersEncodedPair, size, padID int) []seq2seq.LayersBatch {
	var out []seq2seq.LayersBatch
	for i := 0; i < len(enc); i += size {
		end := i + size
		if end > len(enc) {
			end = len(enc)
		}
		grp := enc[i:end]
		maxIn, maxTg := 0, 0
		for _, p := range grp {
			if len(p.input) > maxIn {
				maxIn = len(p.input)
			}
			if len(p.target) > maxTg {
				maxTg = len(p.target)
			}
		}
		b := seq2seq.LayersBatch{PadID: padID}
		for _, p := range grp {
			in := make([]int, maxIn)
			copy(in, p.input)
			inS := make([]string, maxIn)
			copy(inS, p.inputSurfs)
			for k := len(p.input); k < maxIn; k++ {
				in[k] = padID
				inS[k] = ""
			}
			tg := make([]int, maxTg)
			copy(tg, p.target)
			tgS := make([]string, maxTg)
			copy(tgS, p.targetSurfs)
			for k := len(p.target); k < maxTg; k++ {
				tg[k] = padID
				tgS[k] = ""
			}
			b.InputIDs = append(b.InputIDs, in)
			b.InputSurfaces = append(b.InputSurfaces, inS)
			b.TargetIDs = append(b.TargetIDs, tg)
			b.TargetSurfs = append(b.TargetSurfs, tgS)
			b.IntentLabels = append(b.IntentLabels, p.intent)
			b.ToneLabels = append(b.ToneLabels, p.tone)
			b.ConceptCodes = append(b.ConceptCodes, p.codes)
			b.PrevCodes = append(b.PrevCodes, [3]int{})
			b.HasPrev = append(b.HasPrev, false) // single-turn corpus (DESIGN §3.7)
		}
		out = append(out, b)
	}
	return out
}

// saveLayersCheckpoint writes the layers model + sidecar with the
// production discipline (tmp file, then rename).
func saveLayersCheckpoint(model *seq2seq.LayersModel, meta ckptMeta, ckptPath, metaPath string) error {
	if err := os.MkdirAll(filepath.Dir(ckptPath), 0o755); err != nil {
		return err
	}
	tmp := ckptPath + ".tmp"
	if err := model.Save(tmp); err != nil {
		return err
	}
	if err := os.Rename(tmp, ckptPath); err != nil {
		return err
	}
	data, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	return os.WriteFile(metaPath, data, 0o644)
}

// loadLayersCheckpoint mirrors loadTrainingCheckpoint: the sidecar
// must parse and its vocab fingerprint must match, else the
// checkpoint is refused.
func loadLayersCheckpoint(tok *tokenizer.Tokenizer, curFP uint64, ckptPath, metaPath string) (*seq2seq.LayersModel, ckptMeta, error) {
	var meta ckptMeta
	data, err := os.ReadFile(metaPath)
	if err != nil {
		return nil, meta, fmt.Errorf("no checkpoint sidecar: %w", err)
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, meta, fmt.Errorf("bad checkpoint sidecar: %w", err)
	}
	if meta.VocabFP != curFP {
		return nil, meta, fmt.Errorf("checkpoint vocab fingerprint %d != current %d; refusing to resume across data changes", meta.VocabFP, curFP)
	}
	model, err := seq2seq.LoadLayers(ckptPath, tok)
	if err != nil {
		return nil, meta, err
	}
	return model, meta, nil
}

// copyLayersWeightsInto copies src's parameter values into dst's
// parameter tensors in place (dst keeps its own tensor pointers),
// mirroring copyWeightsInto for the base trainer.
func copyLayersWeightsInto(dst, src *seq2seq.LayersModel) error {
	dp := dst.Parameters()
	sp := src.Parameters()
	if len(dp) != len(sp) {
		return fmt.Errorf("layers: parameter count mismatch on resume (%d vs %d)", len(dp), len(sp))
	}
	for i := range dp {
		if len(dp[i].Data) != len(sp[i].Data) {
			return fmt.Errorf("layers: parameter %d size mismatch on resume", i)
		}
		copy(dp[i].Data, sp[i].Data)
	}
	return nil
}

var _ = strings.TrimSpace // anchor for future log trimming
