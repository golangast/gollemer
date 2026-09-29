package chat

// Real seq2seq training: genuine neural generation for social chat.
//
// The old -train-small-seq2seq path trained a bag-of-words classifier for the
// reported loss and then saved a RANDOMLY INITIALIZED seq2seq model, so the
// chat could never learn. This pipeline instead:
//
//  1. Loads the full chat dataset (data/training/chat_pairs.jsonl, seeded
//     from conversing.pb on first run) — retraining always sees ALL pairs,
//     old and new (rehearsal), so added data never erases learned data.
//  2. Splits base pairs into train / held-out (deterministic 80/20).
//  3. Augments train inputs with synonym swaps (teaches paraphrase, not
//     memorization).
//  4. Trains a real encoder-decoder LSTM with teacher forcing and
//     cross-entropy via seq2seq.TrainBatch, stabilized with LR decay,
//     gradient clipping, best-checkpointing on TRAIN loss, divergence
//     rollback, and early stopping.
//  5. Saves the best weights with NO ExactMap cheat sheet.
//  6. Probes the held-out set with paraphrased inputs the model never saw
//     and prints every generation verbatim — coherent answers here are the
//     operational definition of "understands" at this scale.

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

const (
	realBatchSize = 32
	realMaxEpochs = 200
	realBaseLR    = 1e-3
	realClip      = 1.0
)

// dimsForDomain returns (embeddingDim, hiddenDim) per domain. The gocode
// domain separates 24 near-identical code templates, so it gets a wider
// model; the MoE expert width scales as hiddenDim/2 automatically.
// social/makefile keep their original dims so existing .gob checkpoints
// (which store their dims) stay valid and retraining stays comparable.
func dimsForDomain(domain string) (embedDim, hiddenDim int) {
	if domain == GocodeDomain || domain == GoDomain || domain == GoCliDomain || domain == SocialDomain {
		return 128, 256
	}
	return 64, 128
}

// socialSynonyms drives augmentation: swapping a word for a synonym teaches
// the model that meaning survives rephrasing. Single words only; multi-word
// entries are split on spaces when substituted.
var socialSynonyms = map[string][]string{
	"hello": {"hi", "hey", "greetings"}, "hi": {"hello", "hey"}, "hey": {"hi", "hello"},
	"good": {"great", "nice", "fine"}, "great": {"good", "fantastic", "wonderful"},
	"fantastic": {"great", "wonderful"}, "nice": {"good", "lovely"}, "fine": {"good", "okay"},
	"bad": {"terrible", "awful"}, "terrible": {"bad", "awful"},
	"happy": {"glad", "pleased"}, "glad": {"happy", "pleased"}, "sad": {"unhappy", "down"},
	"thanks": {"thank you"}, "thank": {"thanks"},
	"please": {"kindly"}, "yes": {"yeah", "yep"}, "yeah": {"yes", "yep"},
	"no": {"nope", "nah"}, "love": {"adore"}, "like": {"enjoy"},
	"want": {"wish"}, "need": {"require"}, "help": {"assist", "aid"},
	"day": {"morning", "afternoon"}, "morning": {"day"}, "night": {"evening"},
	"today": {"this morning"}, "friend": {"buddy", "pal"},
	"awesome": {"amazing", "great"}, "cool": {"neat", "great"},
	"sorry": {"apologies"}, "welcome": {"greetings"},
	"bye": {"goodbye", "see you"}, "goodbye": {"bye"},
}

// tokenizeLower splits text exactly the way the tokenizer does
// (whitespace + {}[],:\"?!. as single-char tokens), lowercased so the
// model is case-insensitive. It must stay in sync with
// TokenizeAndConvertToIDs in internal/ai/neural/tokenizer.
func tokenizeLower(text string) []string {
	var toks []string
	runes := []rune(strings.ToLower(text))
	i := 0
	for i < len(runes) {
		r := runes[i]
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			i++
			continue
		}
		if strings.ContainsRune(`{}[],:\"?!.`, r) {
			toks = append(toks, string(r))
			i++
			continue
		}
		start := i
		for i < len(runes) && runes[i] != ' ' && runes[i] != '\t' && runes[i] != '\n' && runes[i] != '\r' && !strings.ContainsRune(`{}[],:\"`, runes[i]) {
			i++
		}
		if tok := string(runes[start:i]); tok != "" {
			toks = append(toks, tok)
		}
	}
	return toks
}

// gocodeSynonyms drives augmentation for the gocode domain. Unlike the social
// set, these are verb-level swaps that never change WHICH snippet is wanted:
// "write"/"make"/"create"/"build"/"code" and "give"/"show", "need"/"want" all
// keep the same target function. Social synonyms like hello->hi are BANNED
// here: "write a hello world program" vs "code a hello name function" must
// stay far apart, and swapping hello->hi blurs exactly that boundary.
var gocodeSynonyms = map[string][]string{
	"write":  {"create", "make", "code"},
	"make":   {"create", "build", "write"},
	"create": {"make", "build", "write"},
	"build":  {"make", "create"},
	"code":   {"write", "create"},
	"give":   {"show"},
	"show":   {"give"},
	"need":   {"want"},
	"want":   {"need"},
}

// synonymVariants returns up to n paraphrases of q, each swapping one word
// using the provided synonym map.
func synonymVariants(q string, n int, syns map[string][]string) []string {
	toks := tokenizeLower(q)
	var out []string
	for i, w := range toks {
		for _, syn := range syns[w] {
			nt := append([]string{}, toks...)
			nt[i] = syn
			out = append(out, strings.Join(nt, " "))
			if len(out) >= n {
				return out
			}
		}
	}
	return out
}

// normAugInput canonicalizes an input for collision detection: lowercase,
// punctuation dropped, whitespace collapsed. "I'm unhappy." and "i'm unhappy"
// are the same phrasing and must not carry different labels.
func normAugInput(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '\'' {
			b.WriteRune(r)
		} else if r == ' ' || r == '\t' {
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// augmentInputs expands trainBase with synonym variants, skipping any variant
// that collides (after normalization) with a different base pair's input
// carrying a different output.
func augmentInputs(base, trainBase []ChatPair, syns map[string][]string) []ChatPair {
	baseOutputByNorm := make(map[string]string, len(base))
	for _, p := range base {
		n := normAugInput(p.Input)
		if _, dup := baseOutputByNorm[n]; !dup {
			baseOutputByNorm[n] = p.Output
		}
	}
	trainPairs := make([]ChatPair, 0, len(trainBase)*10)
	emitted := make(map[string]string, len(trainPairs))
	for _, p := range trainBase {
		trainPairs = append(trainPairs, p)
		emitted[normAugInput(p.Input)] = p.Output
		for _, v := range synonymVariants(p.Input, 9, syns) {
			n := normAugInput(v)
			if out, collides := baseOutputByNorm[n]; collides && out != p.Output {
				continue // another pair owns this phrasing — don't contradict it
			}
			if out, dup := emitted[n]; dup && out != p.Output {
				continue // an earlier variant already claimed this phrasing
			}
			emitted[n] = p.Output
			trainPairs = append(trainPairs, ChatPair{Input: v, Output: p.Output})
		}
	}
	return trainPairs
}

// encodeInput tokenizes a question for the encoder: [ids...].
// Inputs are normalized the same way as at inference (see
// seq2seq.NormalizeQuery): punctuation variants of one phrasing share one
// encoding, so the model learns the intent once.
func encodeInput(tok *tokenizer.Tokenizer, q string) []int {
	ids, err := tok.Encode(seq2seq.NormalizeQuery(q))
	if err != nil {
		return nil
	}
	return ids
}

// encodeTarget tokenizes an answer for the decoder: [BOS, ids..., EOS].
func encodeTarget(tok *tokenizer.Tokenizer, v *vocab.Vocabulary, a string) []int {
	ids, err := tok.Encode(strings.ToLower(a))
	if err != nil {
		return nil
	}
	out := make([]int, 0, len(ids)+2)
	out = append(out, v.BosID)
	out = append(out, ids...)
	out = append(out, v.EosID)
	return out
}

type encodedPair struct {
	input  []int
	target []int
}

// RunRealSeq2SeqTraining trains the genuine social seq2seq model.
// Only pairs tagged with domain are used — stages stay separate by design.
func RunRealSeq2SeqTraining(projectRoot, domain string) error {
	seeded, err := SeedChatDataset(projectRoot)
	if err != nil {
		return err
	}
	log.Printf("[REAL-SEQ2SEQ] dataset: %d pairs (%s)", seeded, ChatDatasetPath(projectRoot))
	all, err := LoadChatDataset(ChatDatasetPath(projectRoot))
	if err != nil {
		return err
	}
	base := make([]ChatPair, 0, len(all))
	for _, p := range all {
		if p.Domain == domain {
			base = append(base, p)
		}
	}
	log.Printf("[REAL-SEQ2SEQ] domain=%q: %d pairs (of %d total)", domain, len(base), len(all))
	if len(base) < 10 {
		return fmt.Errorf("need at least 10 base pairs in domain %q, have %d", domain, len(base))
	}

	// Deterministic 80/20 split so results are reproducible.
	rng := rand.New(rand.NewSource(20260926))
	order := rng.Perm(len(base))
	nTrain := int(0.8 * float64(len(base)))
	trainBase := make([]ChatPair, 0, nTrain)
	heldBase := make([]ChatPair, 0, len(base)-nTrain)
	for i, idx := range order {
		if i < nTrain {
			trainBase = append(trainBase, base[idx])
		} else {
			heldBase = append(heldBase, base[idx])
		}
	}

	// Augment training inputs with synonym paraphrases (input side only).
	// Collision-aware: a variant is skipped when it normalizes to the same
	// input as a DIFFERENT base pair with a different output — otherwise
	// augmentation teaches near-identical inputs with conflicting labels
	// (e.g. "i'm unhappy" from "I'm sad" vs base "I'm unhappy."), which a
	// tiny model cannot satisfy and which blurs the class boundary.
	// The gocode domain uses code-safe synonyms only: social swaps like
	// hello->hi would blur "hello world program" vs "hello name function".
	syns := socialSynonyms
	if domain == GocodeDomain {
		syns = gocodeSynonyms
	}
	trainPairs := augmentInputs(base, trainBase, syns)

	// Held-out probes: paraphrases of held-out inputs the model never sees.
	type probe struct{ in, ref string }
	var probes []probe
	for _, p := range heldBase {
		probes = append(probes, probe{in: p.Input, ref: p.Output})
		for _, v := range synonymVariants(p.Input, 2, syns) {
			probes = append(probes, probe{in: v, ref: p.Output})
		}
	}
	log.Printf("[REAL-SEQ2SEQ] %d base pairs -> %d train (augmented %d), %d held-out base, %d probes",
		len(base), len(trainBase), len(trainPairs), len(heldBase), len(probes))

	// Vocabulary from TRAINING data only — held-out words stay unknown,
	// which is exactly what the probes test.
	v := vocab.NewVocabulary()
	for _, p := range trainPairs {
		for _, t := range tokenizeLower(p.Input) {
			v.AddToken(t)
		}
		for _, t := range tokenizeLower(p.Output) {
			v.AddToken(t)
		}
	}
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		return err
	}
	log.Printf("[REAL-SEQ2SEQ] vocab size=%d", v.Size())

	// Encode everything.
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

	// Length-bucketed batches: sort by input length so padding is minimal.
	sort.Slice(encoded, func(i, j int) bool { return len(encoded[i].input) < len(encoded[j].input) })

	embedDim, hiddenDim := dimsForDomain(domain)
	// Stage-3 bigger model: the accumulator-binding miss is a long-range
	// dependency limit, and the copy mechanism needs the extra capacity to
	// train stably. Other domains keep their dims so their checkpoints
	// stay valid.
	if domain == GocodeDomain {
		embedDim, hiddenDim = 256, 512
	}
	model, err := seq2seq.NewSeq2Seq(v.Size(), v.Size(), embedDim, hiddenDim, tok, v)
	if err != nil {
		return err
	}
	// The copy mechanism is gocode-only: a nil gate means exactly the old
	// behavior, so the other domains are untouched.
	if domain == GocodeDomain {
		model.Decoder.Copy = seq2seq.NewCopyGate(hiddenDim)
	}
	// NOTE: no SetExactMap call — the saved model must generate, not retrieve.
	opt := nn.NewOptimizer(model.Parameters(), realBaseLR, realClip)
	adam, ok := opt.(*nn.Adam)
	if !ok {
		return fmt.Errorf("optimizer is not *nn.Adam")
	}

	padID := v.PaddingTokenID
	lr := float32(realBaseLR)
	best := float32(math.Inf(1))
	bestParams := adam.SnapshotParameters()
	epochsNoImprove := 0
	lastCheckpointBest := float32(math.Inf(1))
	modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", "real_tiny_seq2seq_"+domain+".gob")
	rollbacks := 0

	batches := makeBatches(encoded, realBatchSize, padID)
	log.Printf("[REAL-SEQ2SEQ] %d encoded pairs, %d batches/epoch, lr=%.6f", len(encoded), len(batches), lr)

	for epoch := 1; epoch <= realMaxEpochs; epoch++ {
		// Step decay every 40 epochs.
		if epoch > 1 && (epoch-1)%40 == 0 {
			lr *= 0.5
			adam.SetLearningRate(lr)
		}
		var epochLoss float64
		for _, b := range batches {
			adam.ZeroGrad()
			loss, err := seq2seq.TrainBatch(model, b.inputs, b.targets, padID)
			if err != nil {
				return fmt.Errorf("epoch %d: %w", epoch, err)
			}
			epochLoss += float64(loss)
			adam.ClipGradients()
			adam.Step()
		}
		avg := float32(epochLoss / float64(len(batches)))

		// Divergence rollback: restore best weights, drop momentum, halve LR.
		if math.IsNaN(float64(avg)) || math.IsInf(float64(avg), 0) || (epoch > 5 && avg > best*3) {
			rollbacks++
			log.Printf("[REAL-SEQ2SEQ] DIVERGENCE at epoch=%d loss=%.6f (best=%.6f); rolling back (lr %.6f -> %.6f)",
				epoch, avg, best, lr, lr*0.5)
			adam.RestoreParameters(bestParams)
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
			epochsNoImprove = 0
		} else {
			epochsNoImprove++
		}
		// Best is chosen on TRAIN loss so the held-out probes stay untouched.
		log.Printf("[REAL-SEQ2SEQ] epoch=%d avg_loss=%.6f best=%.6f lr=%.6f", epoch, avg, best, lr)
		if epoch%40 == 0 {
			if model.Decoder.MoE != nil {
				log.Printf("[REAL-SEQ2SEQ] MoE expert usage: %.2f %.2f %.2f %.2f",
					model.Decoder.MoE.ExpertUsage()[0], model.Decoder.MoE.ExpertUsage()[1],
					model.Decoder.MoE.ExpertUsage()[2], model.Decoder.MoE.ExpertUsage()[3])
			}
			// Periodic best-checkpoint: a killed run must never lose everything.
			// Training continues from the restored best weights, which is safe.
			if best < lastCheckpointBest {
				adam.RestoreParameters(bestParams)
				if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
					return err
				}
				if err := model.Save(modelPath); err != nil {
					return err
				}
				lastCheckpointBest = best
				log.Printf("[REAL-SEQ2SEQ] checkpointed best model (loss=%.6f) to %s", best, modelPath)
			}
		}
		if epochsNoImprove >= 30 {
			log.Printf("[REAL-SEQ2SEQ] early stopping: 30 epochs without improvement")
			break
		}
	}

	// Restore the best weights seen, then save — no ExactMap, ever.
	adam.RestoreParameters(bestParams)
	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		return err
	}
	if err := model.Save(modelPath); err != nil {
		return err
	}
	log.Printf("[REAL-SEQ2SEQ] saved best model (loss=%.6f) to %s", best, modelPath)

	// Held-out probe: every generation printed verbatim, good or bad.
	log.Printf("[REAL-SEQ2SEQ] held-out probe: %d paraphrased questions the model never saw", len(probes))
	goCase := map[string]string{}
	if domain == GocodeDomain {
		goCase = goIdentCaseMap(projectRoot)
	}
	for i, pr := range probes {
		out, err := model.Predict(strings.ToLower(pr.in), 40)
		if err != nil {
			log.Printf("[REAL-SEQ2SEQ] probe %d error: %v", i, err)
			continue
		}
		shown := tidyDecode(out)
		if domain == GocodeDomain {
			shown = tidyGoCode(out, goCase)
		}
		log.Printf("[REAL-SEQ2SEQ] Q: %s\n[REAL-SEQ2SEQ] A: %s", pr.in, shown)
	}
	return nil
}

type batch struct {
	inputs  [][]int
	targets [][]int
}

// makeBatches pads each bucket to its own max lengths.
func makeBatches(enc []encodedPair, size, padID int) []batch {
	var out []batch
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
		b := batch{}
		for _, p := range grp {
			in := make([]int, maxIn)
			copy(in, p.input)
			for k := len(p.input); k < maxIn; k++ {
				in[k] = padID
			}
			tg := make([]int, maxTg)
			copy(tg, p.target)
			for k := len(p.target); k < maxTg; k++ {
				tg[k] = padID
			}
			b.inputs = append(b.inputs, in)
			b.targets = append(b.targets, tg)
		}
		out = append(out, b)
	}
	return out
}

// tidyDecode fixes the spacing the word-join decoder leaves around punctuation.
func tidyDecode(s string) string {
	for _, p := range []string{"?", "!", ".", ",", ":", ";", "'"} {
		s = strings.ReplaceAll(s, " "+p, p)
	}
	s = strings.ReplaceAll(s, "  ", " ")
	return strings.TrimSpace(s)
}

// RunReclassifyDomains re-tags the whole dataset with the current classifier.
func RunReclassifyDomains(projectRoot string) error {
	counts, err := ReclassifyDomains(projectRoot)
	if err != nil {
		return err
	}
	log.Printf("[DATA] reclassified: %v", counts)
	return nil
}

// RunImportChatPairs imports new training pairs through the quality gate.
func RunImportChatPairs(projectRoot, importPath string) error {
	admitted, quarantined, err := ImportChatPairs(projectRoot, importPath)
	if err != nil {
		return err
	}
	log.Printf("[DATA] admitted %d new pairs", admitted)
	for _, q := range quarantined {
		log.Printf("[DATA] quarantined %s", q)
	}
	if admitted == 0 && len(quarantined) == 0 {
		log.Printf("[DATA] nothing to import")
	}
	return nil
}
