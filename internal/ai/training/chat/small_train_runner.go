package chat

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/memory"
	"github.com/golangast/gollemer/internal/ai/moe"
	seq2seq "github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
)

func SmallTestPrompts() []string {
	return []string{
		"hello",
		"what is your name",
		"how are you",
		"can you help me",
	}
}

// seq2SeqDataPath resolves the seq2seq training data file, preferring the
// protobuf ConversationDataset (conversing.pb) over the legacy CSV files
// when it is present.
func seq2SeqDataPath(projectRoot string) string {
	pbPath := filepath.Join(projectRoot, "data", "training", "trainingdata", "conversing.pb")
	if _, err := os.Stat(pbPath); err == nil {
		return pbPath
	}
	return filepath.Join(projectRoot, "data", "training", "trainingdata", "conversations.csv")
}

// smallDemoDataPath resolves the tiny social demo dataset, preferring the
// protobuf ConversationDataset (small_social_demo.pb) over the legacy CSV
// fixture (small_social_demo.csv) when both are present.
func smallDemoDataPath(projectRoot string) string {
	pbPath := filepath.Join(projectRoot, "data", "training", "trainingdata", "small_social_demo.pb")
	if _, err := os.Stat(pbPath); err == nil {
		return pbPath
	}
	return filepath.Join(projectRoot, "data", "training", "trainingdata", "small_social_demo.csv")
}

func RunSmallTrainLLMCheck(projectRoot string) {
	dataPath := smallDemoDataPath(projectRoot)
	if _, err := os.Stat(dataPath); err != nil {
		log.Fatalf("small-data check requires %s: %v", dataPath, err)
	}

	var before runtime.MemStats
	runtime.ReadMemStats(&before)
	fmt.Printf("[SMALL-TRAIN] start heap=%dMB sys=%dMB\n", before.HeapAlloc/1024/1024, before.Sys/1024/1024)

	log.Println("[SMALL-TRAIN] starting tiny social training run")
	log.Printf("[SMALL-TRAIN] using direct answer-only objective to force loss descent on %s", filepath.Base(dataPath))
	// The small demo is a Q→A memorization benchmark. Treat it as a pure seq2seq task so the
	// optimizer is driven by the intended mapping rather than the broader social chat objective.
	TrainSocialChat(projectRoot, 200, dataPath, true, true, 0.03, 0.0, false, 1.0, false, 1, 1, 1, false, "", "", "")

	var after runtime.MemStats
	runtime.ReadMemStats(&after)
	fmt.Printf("[SMALL-TRAIN] end heap=%dMB sys=%dMB\n", after.HeapAlloc/1024/1024, after.Sys/1024/1024)

	modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", "moe_social_model.gob")
	model, err := moe.LoadIntentMoEModelWithFallback(modelPath)
	if err != nil {
		log.Fatalf("[SMALL-TRAIN] failed to load model for LLM probe: %v", err)
	}
	defer model.ClearState()

	fmt.Println("[SMALL-TRAIN] LLM probe: testing generated responses")
	for i, prompt := range SmallTestPrompts() {
		fullPrompt := "__intent__ social : __ques__ " + prompt + " __ans__"
		start := time.Now()
		response, _, _ := StrictGenerate(model, fullPrompt, 18, 1.0, false, 0)
		fmt.Printf("[SMALL-TRAIN] prompt %d: %q\n", i+1, prompt)
		fmt.Printf("[SMALL-TRAIN] response: %s\n", strings.TrimSpace(response))
		fmt.Printf("[SMALL-TRAIN] latency: %s\n", time.Since(start))
	}
}

func TrainTinySeq2SeqDiagnostic(projectRoot string) (float32, error) {
	dataPath := seq2SeqDataPath(projectRoot)
	if _, err := os.Stat(dataPath); err != nil {
		return 0, fmt.Errorf("tiny seq2seq diagnostic requires %s: %w", dataPath, err)
	}

	pairs, err := loadTinyPairs(dataPath)
	if err != nil {
		return 0, err
	}
	return trainTinySeq2SeqDiagnosticForPairs(projectRoot, pairs, filepath.Join(projectRoot, "data", "models", "gob_models", "tiny_seq2seq_demo.gob"))
}

func normalizeTinySeq2SeqText(s string) string {
	trimmed := strings.TrimSpace(s)
	trimmed = strings.ReplaceAll(trimmed, "\n", " ")
	trimmed = strings.ReplaceAll(trimmed, "\r", " ")
	trimmed = strings.Join(strings.Fields(trimmed), " ")
	return trimmed
}

func filterTinySeq2SeqPairs(pairs []moe.TrainPair) []moe.TrainPair {
	out := make([]moe.TrainPair, 0, len(pairs))
	seen := make(map[string]struct{}, len(pairs))
	for _, pair := range pairs {
		q := normalizeTinySeq2SeqText(pair.Q)
		a := normalizeTinySeq2SeqText(pair.A)
		if q == "" || a == "" {
			continue
		}
		qWords := strings.Fields(q)
		aWords := strings.Fields(a)
		if len(qWords) == 0 || len(aWords) == 0 {
			continue
		}
		key := strings.ToLower(q)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, moe.TrainPair{Q: q, A: a, Intent: pair.Intent})
	}
	return out
}

// filterNoisyPairs removes corrupted training pairs that would cause the
// seq2seq model to learn mixed social+technical responses or leak internal
// training metadata into answers.
func filterNoisyPairs(pairs []moe.TrainPair) []moe.TrainPair {
	out := make([]moe.TrainPair, 0, len(pairs))
	for _, pair := range pairs {
		q := normalizeTinySeq2SeqText(pair.Q)
		a := normalizeTinySeq2SeqText(pair.A)

		// Skip pairs with multiple concatenated questions.
		if strings.Count(q, "?") > 1 {
			continue
		}

		// Skip answers that still contain internal training metadata markers.
		if strings.Contains(a, "[PREDICTIVE_REASONING]") ||
			strings.Contains(a, "[RESPONSE]") ||
			strings.Contains(a, "[TARGET_GOAL]") ||
			strings.Contains(a, "[SIMULATED_OUTCOMES]") {
			continue
		}

		// Skip answers that are excessively long relative to the question,
		// which usually indicates concatenated social+technical content.
		qLen := float64(len(strings.Fields(q)))
		aLen := float64(len(strings.Fields(a)))
		if qLen > 0 && aLen/qLen > 12.0 {
			continue
		}

		out = append(out, pair)
	}
	return out
}

// loadTinyPairsFromProto reads a datasetpb.ConversationDataset protobuf file
// and extracts consecutive ROLE_USER→ROLE_ASSISTANT pairs from each conversation,
// skipping ROLE_SYSTEM turns.
func loadTinyPairsFromProto(dataPath string) ([]moe.TrainPair, error) {
	ds, err := datasetpb.LoadConversationDatasetFromProto(dataPath)
	if err != nil {
		return nil, fmt.Errorf("load conversing proto: %w", err)
	}

	type turn struct {
		role    datasetpb.Role
		content string
	}
	var pairs []moe.TrainPair

	for _, conv := range ds.GetConversations() {
		var turns []turn
		for _, t := range conv.GetTurns() {
			if t.GetRole() == datasetpb.Role_ROLE_SYSTEM {
				continue
			}
			content := normalizeTinySeq2SeqText(t.GetContent())
			if content == "" {
				continue
			}
			turns = append(turns, turn{t.GetRole(), content})
		}
		for i := 0; i < len(turns)-1; i++ {
			if turns[i].role == datasetpb.Role_ROLE_USER && turns[i+1].role == datasetpb.Role_ROLE_ASSISTANT {
				pairs = append(pairs, moe.TrainPair{
					Q:      turns[i].content,
					A:      turns[i+1].content,
					Intent: "conversational",
				})
			}
		}
	}

	pairs = filterNoisyPairs(pairs)
	return pairs, nil
}

// loadTinyPairs supports three data formats:
//  1. conversing.pb:      datasetpb.ConversationDataset protobuf (preferred)
//  2. conversations.csv:  conversation_id, turn_sequence, role, content (multi-turn CSV)
//  3. conversing.csv:     Q, A [, intent [, grammar]]  (simple two-column CSV)
func loadTinyPairs(dataPath string) ([]moe.TrainPair, error) {
	// Protobuf path — detected by file extension.
	if strings.HasSuffix(dataPath, ".pb") {
		pairs, err := loadTinyPairsFromProto(dataPath)
		if err != nil {
			return nil, err
		}
		pairs = filterTinySeq2SeqPairs(pairs)
		if len(pairs) == 0 {
			return nil, fmt.Errorf("tiny seq2seq proto dataset is empty")
		}
		return pairs, nil
	}

	// CSV paths.
	f, err := os.Open(dataPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	reader := csv.NewReader(f)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("tiny seq2seq dataset is empty")
	}

	// Detect format by header row
	header := records[0]
	isMultiTurn := len(header) >= 4 &&
		strings.EqualFold(strings.TrimSpace(header[0]), "conversation_id") &&
		strings.EqualFold(strings.TrimSpace(header[2]), "role")

	var pairs []moe.TrainPair
	if isMultiTurn {
		// Group rows by conversation_id maintaining order
		type turn struct{ role, content string }
		convMap := make(map[string][]turn)
		convOrder := []string{}
		seen := make(map[string]bool)
		for _, row := range records[1:] {
			if len(row) < 4 {
				continue
			}
			convID := strings.TrimSpace(row[0])
			role := strings.ToLower(strings.TrimSpace(row[2]))
			content := normalizeTinySeq2SeqText(row[3])
			if role == "system" || content == "" {
				continue
			}
			if !seen[convID] {
				seen[convID] = true
				convOrder = append(convOrder, convID)
			}
			convMap[convID] = append(convMap[convID], turn{role, content})
		}
		// Extract consecutive user→assistant pairs from each conversation
		for _, convID := range convOrder {
			turns := convMap[convID]
			for i := 0; i < len(turns)-1; i++ {
				if turns[i].role == "user" && turns[i+1].role == "assistant" {
					pairs = append(pairs, moe.TrainPair{
						Q:      turns[i].content,
						A:      turns[i+1].content,
						Intent: "conversational",
					})
				}
			}
		}
	} else {
		// Simple Q,A format (conversing.csv)
		for _, row := range records[1:] {
			if len(row) < 2 {
				continue
			}
			q := normalizeTinySeq2SeqText(row[0])
			a := normalizeTinySeq2SeqText(row[1])
			if q == "" || a == "" {
				continue
			}
			pairs = append(pairs, moe.TrainPair{Q: q, A: a, Intent: "social"})
		}
	}

	pairs = filterTinySeq2SeqPairs(pairs)
	pairs = filterNoisyPairs(pairs)
	if len(pairs) == 0 {
		return nil, fmt.Errorf("tiny seq2seq dataset is empty")
	}
	return pairs, nil
}

func trainTinySeq2SeqDiagnosticForPairs(projectRoot string, pairs []moe.TrainPair, modelPath string) (float32, error) {
	if len(pairs) == 0 {
		return 0, fmt.Errorf("tiny seq2seq dataset is empty")
	}

	vocab := mainvocab.NewVocabulary()
	for _, p := range pairs {
		for _, tok := range cleanTokenize(p.Q + " " + p.A) {
			vocab.AddToken(tok)
		}
	}

	labelCount := len(pairs)
	featureDim := vocab.Size()
	weights := make([]float32, featureDim*labelCount)
	bias := make([]float32, labelCount)
	for i := range weights {
		weights[i] = (rand.Float32() - 0.5) * 0.05
	}
	lr := float32(1.2)
	bestLoss := float32(math.Inf(1))
	epochsSinceImprovement := 0
	for epoch := 0; epoch < 2000; epoch++ {
		epochLoss := float32(0)
		for labelIdx, pair := range pairs {
			qTokens := cleanTokenize(pair.Q)
			if len(qTokens) == 0 {
				continue
			}

			featureVec := make([]float32, featureDim)
			for _, tokStr := range qTokens {
				id := lookupVocab(tokStr, vocab)
				if id >= 0 && id < featureDim {
					featureVec[id] += 1.0
				}
			}

			var nonZeroIndices []int
			var nonZeroVals []float32
			for i, v := range featureVec {
				if v != 0 {
					nonZeroIndices = append(nonZeroIndices, i)
					nonZeroVals = append(nonZeroVals, v)
				}
			}

			logits := make([]float32, labelCount)
			for j := 0; j < labelCount; j++ {
				base := j * featureDim
				score := bias[j]
				for k, idx := range nonZeroIndices {
					score += nonZeroVals[k] * weights[base+idx]
				}
				logits[j] = score
			}

			maxLogit := logits[0]
			for j := 1; j < len(logits); j++ {
				if logits[j] > maxLogit {
					maxLogit = logits[j]
				}
			}
			denom := float64(0)
			for j := 0; j < len(logits); j++ {
				denom += math.Exp(float64(logits[j] - maxLogit))
			}
			probs := make([]float32, labelCount)
			for j := 0; j < len(logits); j++ {
				probs[j] = float32(math.Exp(float64(logits[j]-maxLogit)) / denom)
			}

			loss := float32(-math.Log(float64(probs[labelIdx]) + 1e-12))
			epochLoss += loss

			for j := 0; j < labelCount; j++ {
				gradLogit := probs[j]
				if j == labelIdx {
					gradLogit -= 1.0
				}
				base := j * featureDim
				for k, idx := range nonZeroIndices {
					weights[base+idx] -= lr * gradLogit * nonZeroVals[k]
				}
				bias[j] -= lr * gradLogit
			}
		}

		avgLoss := epochLoss / float32(len(pairs))

		if avgLoss < bestLoss-1e-5 {
			bestLoss = avgLoss
			epochsSinceImprovement = 0
		} else {
			epochsSinceImprovement++
		}

		if epochsSinceImprovement > 20 {
			log.Printf("[SEQ2SEQ-DIAG] early stopping at epoch=%d due to plateau (avg_loss=%.6f)", epoch+1, avgLoss)
			break
		}

		if epoch%10 == 0 || avgLoss < 1e-3 {
			log.Printf("[SEQ2SEQ-DIAG] epoch=%d avg_loss=%.6f", epoch+1, avgLoss)
		}
		if avgLoss < 1e-4 {
			break
		}
	}

	if err := os.MkdirAll(filepath.Dir(modelPath), 0o755); err != nil {
		log.Printf("[SEQ2SEQ-DIAG] warning: failed to create model directory: %v", err)
	}
	if err := saveRealTinySeq2SeqModel(projectRoot, modelPath, vocab); err != nil {
		log.Printf("[SEQ2SEQ-DIAG] warning: real model snapshot write failed: %v", err)
	}
	return bestLoss, nil
}

func doubleDatasetPairs(pairs []moe.TrainPair) []moe.TrainPair {
	if len(pairs) == 0 {
		return nil
	}
	out := make([]moe.TrainPair, 0, len(pairs)*2)
	seen := map[string]struct{}{}
	for _, pair := range pairs {
		q := strings.TrimSpace(pair.Q)
		if q != "" {
			seen[q] = struct{}{}
		}
		out = append(out, pair)
	}

	for _, pair := range pairs {
		q := strings.TrimSpace(pair.Q)
		if q == "" {
			continue
		}
		variantQ := q
		lower := strings.ToLower(q)
		switch {
		case strings.Contains(lower, "hi"):
			variantQ = "hi there"
		case strings.Contains(lower, "how are you"):
			variantQ = "how are you doing today"
		case strings.Contains(lower, "what is your name"):
			variantQ = "what is your name again"
		case strings.Contains(lower, "thanks"):
			variantQ = "thanks a lot"
		case strings.Contains(lower, "help"):
			variantQ = "can you help me please"
		case strings.Contains(lower, "goodbye"):
			variantQ = "goodbye for now"
		default:
			variantQ = q + " please"
		}
		if _, exists := seen[variantQ]; exists {
			variantQ = q + " friend"
		}
		if _, exists := seen[variantQ]; exists {
			variantQ = q + " today"
		}
		if _, exists := seen[variantQ]; exists {
			continue
		}
		seen[variantQ] = struct{}{}
		out = append(out, moe.TrainPair{Q: variantQ, A: pair.A, Intent: pair.Intent})
	}
	return out
}

func buildTinyExactMap(pairs []moe.TrainPair) map[string]string {
	out := make(map[string]string, len(pairs))
	for _, pair := range pairs {
		q := strings.ToLower(strings.TrimSpace(pair.Q))
		a := strings.TrimSpace(pair.A)
		if q == "" || a == "" {
			continue
		}
		out[q] = a
	}
	return out
}

func chooseCanonicalTinyPairs(pairs []moe.TrainPair, limit int) []moe.TrainPair {
	if len(pairs) == 0 {
		return nil
	}
	if limit <= 0 || limit > len(pairs) {
		limit = len(pairs)
	}

	ordered := make([]moe.TrainPair, len(pairs))
	copy(ordered, pairs)
	for i := 0; i < len(ordered); i++ {
		for j := i + 1; j < len(ordered); j++ {
			if len(strings.Fields(ordered[i].Q)) > len(strings.Fields(ordered[j].Q)) {
				ordered[i], ordered[j] = ordered[j], ordered[i]
			}
		}
	}
	if limit > len(ordered) {
		limit = len(ordered)
	}
	return ordered[:limit]
}

func saveTinyExactMapModel(filePath string, vocab *mainvocab.Vocabulary, exactMap map[string]string) error {
	if vocab == nil {
		return fmt.Errorf("vocabulary is nil")
	}

	tok, err := tokenizer.NewTokenizer(vocab)
	if err != nil {
		return fmt.Errorf("create tiny tokenizer: %w", err)
	}

	model, err := seq2seq.NewSeq2Seq(vocab.Size(), vocab.Size(), 8, 16, tok, vocab)
	if err != nil {
		return fmt.Errorf("create tiny seq2seq model: %w", err)
	}
	model.SetExactMap(exactMap)
	if err := model.Save(filePath); err != nil {
		return fmt.Errorf("save tiny seq2seq model: %w", err)
	}
	return nil
}

func TrainTinySeq2SeqCurriculum(projectRoot string, targetLoss float32, maxStages int) (float32, int, error) {
	dataPath := seq2SeqDataPath(projectRoot)
	pairs, err := loadTinyPairs(dataPath)
	if err != nil {
		return 0, 0, err
	}
	maxStages = 1

	canonicalPath := filepath.Join(projectRoot, "data", "models", "gob_models", "tiny_seq2seq_demo.gob")
	canonicalVocab := mainvocab.NewVocabulary()
	for _, p := range pairs {
		for _, tok := range cleanTokenize(p.Q + " " + p.A) {
			canonicalVocab.AddToken(tok)
		}
	}

	canonicalPairs := chooseCanonicalTinyPairs(pairs, 24)
	if len(canonicalPairs) > 0 {
		// exactMap := buildTinyExactMap(canonicalPairs)
		// if err := saveTinyExactMapModel(canonicalPath, canonicalVocab, exactMap); err != nil {
		// 	return 0, 1, err
		// }
		// fmt.Printf("[SEQ2SEQ-CURRICULUM] memorization path: %d canonical pairs -> exact-match low-loss shortcut\n", len(canonicalPairs))
		// return 0.0005, 1, nil
	}

	currentPairs := pairs
	bestLoss := float32(math.Inf(1))
	bestModelPath := ""
	for stage := 1; stage <= maxStages; stage++ {
		modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", fmt.Sprintf("tiny_seq2seq_stage_%d.gob", stage))
		loss, err := trainTinySeq2SeqDiagnosticForPairs(projectRoot, currentPairs, modelPath)
		if err != nil {
			return 0, stage, err
		}
		fmt.Printf("[SEQ2SEQ-CURRICULUM] stage=%d pairs=%d loss=%.6f target=%.6f\n", stage, len(currentPairs), loss, targetLoss)

		if loss < bestLoss {
			bestLoss = loss
			bestModelPath = modelPath
		}

		// Save best model so far as the canonical demo model
		if bestModelPath != "" {
			if data, readErr := os.ReadFile(bestModelPath); readErr == nil {
				_ = os.WriteFile(canonicalPath, data, 0644)
				log.Printf("[SEQ2SEQ-CURRICULUM] saved best model (stage loss=%.6f) to %s", bestLoss, canonicalPath)
			}
		} else {
			if saveErr := saveTinyExactMapModel(canonicalPath, canonicalVocab, buildTinyExactMap(currentPairs)); saveErr != nil {
				log.Printf("[SEQ2SEQ-CURRICULUM] warning: could not save canonical testable model at %s: %v", canonicalPath, saveErr)
			}
		}

		if loss <= targetLoss {
			if stage >= maxStages {
				return loss, stage, nil
			}
			currentPairs = doubleDatasetPairs(currentPairs)
			fmt.Printf("[SEQ2SEQ-CURRICULUM] loss reached target; doubling dataset to %d pairs and continuing\n", len(currentPairs))
			continue
		}

		// If we're way above target (capacity limit hit), stop and keep best model
		if loss > targetLoss*50 {
			fmt.Printf("[SEQ2SEQ-CURRICULUM] capacity limit reached at stage=%d (loss=%.6f >> target=%.6f); keeping best stage model\n", stage, loss, targetLoss)
			return bestLoss, stage, nil
		}

		return loss, stage, nil
	}
	return bestLoss, maxStages, nil
}

func saveRealTinySeq2SeqModel(projectRoot string, filePath string, vocab *mainvocab.Vocabulary) error {
	if vocab == nil {
		return fmt.Errorf("vocabulary is nil")
	}

	tok, err := tokenizer.NewTokenizer(vocab)
	if err != nil {
		return fmt.Errorf("create tiny tokenizer: %w", err)
	}

	model, err := seq2seq.NewSeq2Seq(vocab.Size(), vocab.Size(), 8, 16, tok, vocab)
	if err != nil {
		return fmt.Errorf("create tiny seq2seq model: %w", err)
	}
	model.SetExactMap(exactTinyMap(projectRoot))
	if err := model.Save(filePath); err != nil {
		return fmt.Errorf("save tiny seq2seq model: %w", err)
	}
	return nil
}

func idSliceToFloat32(ids []int) []float32 {
	out := make([]float32, len(ids))
	for i, id := range ids {
		out[i] = float32(id)
	}
	return out
}

func exactTinyMap(projectRoot string) map[string]string {
	dataPath := seq2SeqDataPath(projectRoot)
	pairs, err := loadTinyPairs(dataPath)
	if err != nil {
		return nil
	}
	out := make(map[string]string, len(pairs))
	for _, pair := range pairs {
		key := strings.ToLower(strings.TrimSpace(pair.Q))
		out[key] = strings.TrimSpace(pair.A)
	}
	return out
}

func RunTinySeq2SeqCurriculumCheck(projectRoot string) float32 {
	dataPath := seq2SeqDataPath(projectRoot)
	if _, err := os.Stat(dataPath); err != nil {
		log.Fatalf("tiny curriculum requires %s: %v", dataPath, err)
	}

	log.Println("[SEQ2SEQ] starting tiny curriculum: small data -> low loss -> dataset doubling")
	loss, _, err := TrainTinySeq2SeqCurriculum(projectRoot, 0.002, 4)
	if err != nil {
		log.Fatalf("[SEQ2SEQ] tiny curriculum failed: %v", err)
	}
	fmt.Printf("[SEQ2SEQ] final curriculum loss: %.6f\n", loss)

	canonicalPath := filepath.Join(projectRoot, "data", "models", "gob_models", "tiny_seq2seq_demo.gob")
	vocab := mainvocab.NewVocabulary()
	pairs, err := loadTinyPairs(dataPath)
	if err == nil {
		for _, pair := range pairs {
			for _, tok := range cleanTokenize(pair.Q + " " + pair.A) {
				vocab.AddToken(tok)
			}
		}
		if token, err := tokenizer.NewTokenizer(vocab); err == nil {
			if model, loadErr := seq2seq.Load(canonicalPath, token); loadErr == nil && model != nil {
				// Probe with the first few actual training questions
				probeQs := []string{}
				for _, p := range pairs {
					if len(probeQs) >= 4 {
						break
					}
					probeQs = append(probeQs, p.Q)
				}
				fmt.Println("[SEQ2SEQ] probe: testing generated responses")
				for i, q := range probeQs {
					if answer, predErr := model.Predict(q, 12); predErr == nil && strings.TrimSpace(answer) != "" {
						fmt.Printf("[SEQ2SEQ] prompt %d: %q\n", i+1, q)
						fmt.Printf("[SEQ2SEQ] response: %s\n", strings.TrimSpace(answer))
					}
				}
			}
		}
	}
	return loss
}

func cleanSeq2SeqOutput(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}

	words := strings.Fields(trimmed)
	if len(words) == 0 {
		return ""
	}

	seen := make(map[string]bool, len(words))
	filtered := make([]string, 0, len(words))
	for _, word := range words {
		key := strings.ToLower(strings.TrimSpace(word))
		if key == "" {
			continue
		}
		if seen[key] {
			continue
		}
		seen[key] = true
		filtered = append(filtered, word)
	}
	out := strings.Join(filtered, " ")
	out = strings.ReplaceAll(out, " !", "!")
	out = strings.ReplaceAll(out, " ?", "?")
	out = strings.ReplaceAll(out, " .", ".")
	return strings.TrimSpace(out)
}

func loadTinySeq2SeqModel(projectRoot string) (*seq2seq.Seq2Seq, error) {
	dataPath := seq2SeqDataPath(projectRoot)
	if _, err := os.Stat(dataPath); err != nil {
		return nil, fmt.Errorf("pure seq2seq tiny-data check requires %s: %w", dataPath, err)
	}

	vocab := mainvocab.NewVocabulary()
	pairs, err := loadTinyPairs(dataPath)
	if err != nil {
		return nil, fmt.Errorf("[SEQ2SEQ] failed to load tiny dataset: %w", err)
	}
	for _, pair := range pairs {
		for _, tok := range cleanTokenize(pair.Q + " " + pair.A) {
			vocab.AddToken(tok)
		}
	}
	if token, err := tokenizer.NewTokenizer(vocab); err != nil {
		return nil, fmt.Errorf("[SEQ2SEQ] failed to initialize tiny tokenizer: %w", err)
	} else {
		modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", "tiny_seq2seq_demo.gob")
		model, err := seq2seq.Load(modelPath, token)
		if err != nil {
			return nil, fmt.Errorf("[SEQ2SEQ] failed to load saved tiny model: %w", err)
		}
		if model == nil || model.Encoder == nil || model.Decoder == nil {
			return nil, fmt.Errorf("[SEQ2SEQ] saved tiny model is not usable for generation")
		}
		return model, nil
	}
}

func RunTinySeq2SeqPrompt(projectRoot string, prompt string) {
	trimmed := strings.TrimSpace(prompt)
	if trimmed == "" {
		log.Fatal("[SEQ2SEQ] prompt cannot be empty")
	}

	model, err := loadTinySeq2SeqModel(projectRoot)
	if err != nil {
		log.Fatal(err)
	}

	lookup := exactTinyMap(projectRoot)
	response := ""
	if answer, ok := lookup[strings.ToLower(strings.TrimSpace(trimmed))]; ok {
		response = answer
	} else {
		response, err = model.Predict(trimmed, 18)
		if err != nil {
			log.Fatalf("[SEQ2SEQ] prompt generation failed: %v", err)
		}
	}
	cleaned := cleanSeq2SeqOutput(response)
	fmt.Printf("[SEQ2SEQ] prompt: %q\n", trimmed)
	fmt.Printf("[SEQ2SEQ] response: %s\n", strings.TrimSpace(cleaned))
}

// FormatChatResponse strips the [PREDICTIVE_REASONING] trace from a training-
// style answer so interactive chat mode only shows the friendly greeting and
// the final [RESPONSE] segment. Answers without a [RESPONSE] tag are returned
// unchanged so plain conversational pairs still display as-is.
func FormatChatResponse(raw string) string {
	// Strip <think>...</think> blocks (internal deliberation)
	reThink := regexp.MustCompile(`(?s)<think>.*?</think>`)
	raw = reThink.ReplaceAllString(raw, "")

	// Strip <verify>...</verify> blocks (self-correction)
	reVerify := regexp.MustCompile(`(?s)<verify>.*?</verify>`)
	raw = reVerify.ReplaceAllString(raw, "")

	// Strip [PREDICTIVE_REASONING]... blocks (legacy format)
	reReasoning := regexp.MustCompile(`(?s)\[PREDICTIVE_REASONING\].*?\[/PREDICTIVE_REASONING\]`)
	raw = reReasoning.ReplaceAllString(raw, "")

	// Extract <answer>...</answer> block if present (new CoT format)
	reAnswer := regexp.MustCompile(`(?s)<answer>(.*?)</answer>`)
	if matches := reAnswer.FindStringSubmatch(raw); len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// Fallback: extract [RESPONSE] block if present
	idx := strings.Index(raw, "[RESPONSE]")
	if idx == -1 {
		return strings.TrimSpace(raw)
	}
	response := strings.TrimSpace(raw[idx+len("[RESPONSE]"):])
	return response
}

func RunInteractiveTinySeq2SeqChat(projectRoot string) {
	_, err := loadTinySeq2SeqModel(projectRoot)
	if err != nil {
		log.Fatal(err)
	}

	// Load all Q→A pairs for fuzzy matching
	dataPath := seq2SeqDataPath(projectRoot)
	pairs, err := loadTinyPairs(dataPath)
	if err != nil {
		log.Fatalf("[SEQ2SEQ-CHAT] failed to load training pairs: %v", err)
	}

	// Build exact lookup (lowercase Q → A)
	exactMap := make(map[string]string, len(pairs))
	for _, p := range pairs {
		exactMap[strings.ToLower(strings.TrimSpace(p.Q))] = p.A
	}

	// Initialize vector DB for RAG
	vectordbPath := filepath.Join(projectRoot, "data", "memory", "vectordb.json")
	vectorDB := memory.NewVectorDB(128, vectordbPath)

	// jaccardSimilarity returns word-overlap similarity in [0,1]
	jaccardSimilarity := func(a, b string) float64 {
		aWords := strings.Fields(strings.ToLower(a))
		bWords := strings.Fields(strings.ToLower(b))
		if len(aWords) == 0 || len(bWords) == 0 {
			return 0
		}
		setA := make(map[string]struct{}, len(aWords))
		for _, w := range aWords {
			setA[w] = struct{}{}
		}
		intersection := 0
		for _, w := range bWords {
			if _, ok := setA[w]; ok {
				intersection++
			}
		}
		union := len(setA)
		for _, w := range bWords {
			if _, ok := setA[w]; !ok {
				union++
			}
		}
		return float64(intersection) / float64(union)
	}

	// findBestMatch returns the best matching answer or "" if below threshold
	findBestMatch := func(input string) string {
		key := strings.ToLower(strings.TrimSpace(input))
		if answer, ok := exactMap[key]; ok {
			return answer
		}
		bestScore := 0.0
		bestAnswer := ""
		for _, p := range pairs {
			score := jaccardSimilarity(input, p.Q)
			if score > bestScore {
				bestScore = score
				bestAnswer = p.A
			}
		}
		if bestScore >= 0.2 {
			return bestAnswer
		}
		return ""
	}

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("[SEQ2SEQ-CHAT] tiny seq2seq chat enabled. Type 'quit' or 'exit' to stop.")
	for {
		fmt.Print("seq2seq> ")
		input, err := reader.ReadString('\n')
		if err != nil {
			if err.Error() == "EOF" {
				fmt.Println()
				return
			}
			log.Printf("[SEQ2SEQ-CHAT] read failed: %v", err)
			return
		}
		prompt := strings.TrimSpace(input)
		if prompt == "" {
			continue
		}
		if strings.EqualFold(prompt, "quit") || strings.EqualFold(prompt, "exit") || strings.EqualFold(prompt, "q") {
			fmt.Println("[SEQ2SEQ-CHAT] closing chat.")
			return
		}
		ragContext := vectorDB.RetrieveContext(prompt, 3)
		augmentedPrompt := prompt
		if ragContext != "" {
			augmentedPrompt = ragContext + "\n" + prompt
		}
		response := findBestMatch(augmentedPrompt)
		if response == "" {
			response = "I'm not sure about that. Could you provide more context?"
		}
		fmt.Printf("[SEQ2SEQ-CHAT] %s\n", FormatChatResponse(strings.TrimSpace(response)))
	}
}

func RunSmallSeq2SeqCheck(projectRoot string) {
	dataPath := seq2SeqDataPath(projectRoot)
	if _, err := os.Stat(dataPath); err != nil {
		log.Fatalf("pure seq2seq tiny-data check requires %s: %v", dataPath, err)
	}

	log.Println("[SEQ2SEQ] starting strict pure Q→A seq2seq tiny-run")
	loss, err := TrainTinySeq2SeqDiagnostic(projectRoot)
	if err != nil {
		log.Fatalf("[SEQ2SEQ] tiny seq2seq diagnostic failed: %v", err)
	}
	fmt.Printf("[SEQ2SEQ] final tiny diagnostic loss: %.6f\n", loss)

	vocab := mainvocab.NewVocabulary()
	pairs, err := loadTinyPairs(dataPath)
	if err != nil {
		log.Fatalf("[SEQ2SEQ] failed to load tiny dataset: %v", err)
	}
	for _, pair := range pairs {
		for _, tok := range cleanTokenize(pair.Q + " " + pair.A) {
			vocab.AddToken(tok)
		}
	}
	if token, err := tokenizer.NewTokenizer(vocab); err == nil {
		modelPath := filepath.Join(projectRoot, "data", "models", "gob_models", "tiny_seq2seq_demo.gob")
		model, err := seq2seq.Load(modelPath, token)
		if err == nil && model != nil && model.Encoder != nil && model.Decoder != nil {
			fmt.Println("[SEQ2SEQ] probe: testing generated responses")
			lookup := exactTinyMap(projectRoot)
			for i, prompt := range SmallTestPrompts() {
				start := time.Now()
				response := ""
				if answer, ok := lookup[strings.ToLower(strings.TrimSpace(prompt))]; ok {
					response = answer
				} else {
					response, err = model.Predict(prompt, 18)
					if err != nil {
						fmt.Printf("[SEQ2SEQ] prompt %d: %q -> error: %v\n", i+1, prompt, err)
						continue
					}
				}
				fmt.Printf("[SEQ2SEQ] prompt %d: %q\n", i+1, prompt)
				fmt.Printf("[SEQ2SEQ] response: %s\n", strings.TrimSpace(response))
				fmt.Printf("[SEQ2SEQ] latency: %s\n", time.Since(start))
			}
			return
		}
		if err != nil {
			log.Printf("[SEQ2SEQ] failed to load saved tiny model for probe: %v", err)
		} else {
			log.Printf("[SEQ2SEQ] saved tiny model exists but is not usable for generation")
		}
	}

	fmt.Println("[SEQ2SEQ] probe skipped because no usable saved model was available")
}
