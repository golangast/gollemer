package chat

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// StrictGenerateLowTemp is a near-argmax variant of StrictGenerate for evaluation probes.
// Uses temperature=0.1 and topK=5 to produce greedy-almost deterministic output,
// testing the model's actual argmax knowledge rather than noisy random token samples.
func StrictGenerateLowTemp(model *moe.IntentMoE, input string, maxLen int, repetitionPenalty float32, verbose bool, epoch int) (string, string, []*tensor.Tensor) {
	model.SetParamsRequiresGrad(false)

	oldTemps := make(map[*moe.MoELayer]float32)
	for _, layer := range moe.ActiveLayers {
		layer.SetMode(true)
		oldTemps[layer] = layer.RouterTemperature
		layer.RouterTemperature = 0.7
	}
	if model.Decoder.OutputMoE != nil {
		model.Decoder.OutputMoE.SetMode(true)
		oldTemps[model.Decoder.OutputMoE] = model.Decoder.OutputMoE.RouterTemperature
		model.Decoder.OutputMoE.RouterTemperature = 0.7
	}

	defer func() {
		model.SetParamsRequiresGrad(true)
		for layer, temp := range oldTemps {
			layer.SetMode(true)
			layer.RouterTemperature = temp
		}
		if model.Decoder.OutputMoE != nil {
			model.Decoder.OutputMoE.SetMode(true)
			model.Decoder.OutputMoE.RouterTemperature = oldTemps[model.Decoder.OutputMoE]
		}
	}()

	formattedInput := input
	tokens := cleanTokenize(formattedInput)
	if len(tokens) == 0 {
		log.Printf(" Skip empty prompt: %s", input)
		return "", "", nil
	}
	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	ctx, err := model.EncoderForward(inputTensor, nil)
	if err != nil {
		log.Printf("StrictGenerateLowTemp Error (EncoderForward): %v", err)
		return "", "", nil
	}

	if ctx.Shape[1] == 0 {
		log.Printf("StrictGenerateLowTemp Error: encoder produced empty sequence")
		return "", "", nil
	}

	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	hiddenState, err := ctx.Mean(1)
	if err != nil {
		log.Printf("StrictGenerateLowTemp Error (Initial Hidden Mean): %v", err)
		return "", "", nil
	}
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})
	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID
	counts := make(map[int]int)
	var pathSteps []string
	var allAtts []*tensor.Tensor
	lastPunctStep := -10

	const genTemp = float32(0.0) // greedy/argmax for deterministic probe evaluation

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)
		logits, nextHidden, nextCell, expertIDs, attWeights, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx, i)
		if err != nil {
			log.Printf("StrictGenerateLowTemp Error (DecodeStep): %v", err)
			break
		}
		hiddenState = nextHidden
		cellState = nextCell
		if attWeights != nil {
			allAtts = append(allAtts, attWeights)
		}

		var stepExperts []string
		for _, eid := range expertIDs {
			label := fmt.Sprintf("E%d", eid)
			stepExperts = append(stepExperts, label)
		}
		expertStr := strings.Join(stepExperts, "+")
		pathSteps = append(pathSteps, expertStr)

		if i == 0 {
			logits.Data[model.SentenceVocab.EosID] = -1e9
		}
		specialTokens := []string{"__intent__", "__ques__", "__ans__", "social", ":"}
		for _, st := range specialTokens {
			id := model.SentenceVocab.GetTokenID(st)
			if id != -1 && id < len(logits.Data) {
				logits.Data[id] = -1e9
			}
		}

		applyGrammarMask(logits, resIDs, model.SentenceVocab)

		consecutivePunct := 0
		for j := len(resIDs) - 1; j >= 0; j-- {
			w := model.SentenceVocab.GetWord(resIDs[j])
			if w == "." || w == "," || w == "!" || w == "?" || w == ";" || w == ":" {
				consecutivePunct++
			} else {
				break
			}
		}
		punctPenalty := float32(5.0)
		if consecutivePunct > 1 {
			punctPenalty = 10.0
		}
		if i-lastPunctStep < 4 || consecutivePunct > 1 {
			punctuation := []string{".", ",", "!", "?", ";", ":"}
			for _, p := range punctuation {
				id := model.SentenceVocab.GetTokenID(p)
				if id != -1 && id < len(logits.Data) {
					logits.Data[id] -= punctPenalty
				}
			}
		}

		if len(resIDs) >= 3 {
			fourgrams := make(map[[3]int]int)
			for j := 0; j+2 < len(resIDs); j++ {
				key := [3]int{resIDs[j], resIDs[j+1], resIDs[j+2]}
				fourgrams[key]++
			}
			tailKey := [3]int{resIDs[len(resIDs)-3], resIDs[len(resIDs)-2], resIDs[len(resIDs)-1]}
			if fourgrams[tailKey] >= 2 {
				for candidateID := range logits.Data {
					logits.Data[candidateID] -= 15.0
				}
			}
		}

		moe.ApplyRepetitionPenalty(logits, resIDs, repetitionPenalty)
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		if unkID := model.SentenceVocab.GetTokenID("UNK"); unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		// Greedy argmax decoding: pick the single highest-logit token with no randomness.
		// Temperature 0.0 means zero exploration — pure evaluation.
		if genTemp > 0 {
			ApplyTemperature(logits.Data, genTemp)
		}
		bestID := 0
		maxLogit := float32(-1e9)
		for tokenID, logitValue := range logits.Data {
			if logitValue > maxLogit {
				maxLogit = logitValue
				bestID = tokenID
			}
		}

		probs := tensor.Softmax(logits)
		if bestID == model.SentenceVocab.EosID {
			probs.Release()
			logits.Release()
			inputT.Release()
			break
		}

		resIDs = append(resIDs, bestID)
		counts[bestID]++
		currentTokenID = bestID

		word := model.SentenceVocab.GetWord(bestID)
		if word == "." || word == "," || word == "!" || word == "?" {
			lastPunctStep = i
		}

		probs.Release()
		logits.Release()
		inputT.Release()
	}

	result := ""
	for i, id := range resIDs {
		if i == 0 {
			continue
		}
		result += model.SentenceVocab.GetWord(id) + " "
	}

	inputTensor.Release()
	hiddenState.Release()
	cellState.Release()
	ctx.Release()
	model.ClearState()

	return strings.TrimSpace(result), strings.Join(pathSteps, " -> "), allAtts
}

func StrictGenerate(model *moe.IntentMoE, input string, maxLen int, repetitionPenalty float32, verbose bool, epoch int) (string, string, []*tensor.Tensor) {
	// 1. Diagnostics: We keep Training=true for MoE layers during tests
	// to see the REAL routing behavior (noise, dropout, penalties).
	// CRITICAL: Disable gradient tracking to prevent stateStack bloat and graph memory leaks.
	model.SetParamsRequiresGrad(false)

	oldTemps := make(map[*moe.MoELayer]float32)
	for _, layer := range moe.ActiveLayers {
		layer.SetMode(true) // KEEP TRAINING MODE FOR DIVERSITY
		oldTemps[layer] = layer.RouterTemperature

		// Set tau to 0.7 during test evaluations to soften token salad and allow
		// adjacent experts to absorb gradient/routing load.
		layer.RouterTemperature = 0.7
	}
	if model.Decoder.OutputMoE != nil {
		model.Decoder.OutputMoE.SetMode(true)
		oldTemps[model.Decoder.OutputMoE] = model.Decoder.OutputMoE.RouterTemperature

		model.Decoder.OutputMoE.RouterTemperature = 0.7
	}

	defer func() {
		// Restore gradient tracking
		model.SetParamsRequiresGrad(true)

		for layer, temp := range oldTemps {
			layer.SetMode(true)
			layer.RouterTemperature = temp
		}
		if model.Decoder.OutputMoE != nil {
			model.Decoder.OutputMoE.SetMode(true)
			model.Decoder.OutputMoE.RouterTemperature = oldTemps[model.Decoder.OutputMoE]
		}
	}()

	// Format input: keep the raw formatted input (including intent markers)
	// so the encoder receives the exact same sequence layout it was trained on.
	formattedInput := input
	tokens := cleanTokenize(formattedInput)
	if len(tokens) == 0 {
		log.Printf(" Skip empty prompt: %s", input)
		return "", "", nil
	}
	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	// 2. Get Encoder Context using the stable pipeline
	ctx, err := model.EncoderForward(inputTensor, nil)
	if err != nil {
		log.Printf("StrictGenerate Error (EncoderForward): %v", err)
		return "", "", nil
	}

	// (b) Question Type Heuristic

	if ctx.Shape[1] == 0 {
		log.Printf("StrictGenerate Error: encoder produced empty sequence")
		return "", "", nil
	}

	// 3. Prepare Decoder States
	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	hiddenState, err := ctx.Mean(1)
	if err != nil {
		log.Printf("StrictGenerate Error (Initial Hidden Mean): %v", err)
		return "", "", nil
	}
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})
	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	// 4. Start Sequence with <s> (BOS)
	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID
	counts := make(map[int]int)
	var pathSteps []string
	var allAtts []*tensor.Tensor
	lastPunctStep := -10

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)
		logits, nextHidden, nextCell, expertIDs, attWeights, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx, i)
		if err != nil {
			log.Printf("StrictGenerate Error (DecodeStep): %v", err)
			break
		}
		hiddenState = nextHidden
		cellState = nextCell
		if attWeights != nil {
			allAtts = append(allAtts, attWeights)
		}

		// Collect Expert IDs for the path
		var stepExperts []string
		for _, eid := range expertIDs {
			label := fmt.Sprintf("E%d", eid)
			stepExperts = append(stepExperts, label)
		}
		expertStr := strings.Join(stepExperts, "+")
		pathSteps = append(pathSteps, expertStr)

		if i == 0 {
			logits.Data[model.SentenceVocab.EosID] = -1e9
		}
		specialTokens := []string{"__intent__", "__ques__", "__ans__", "social", ":"}
		for _, st := range specialTokens {
			id := model.SentenceVocab.GetTokenID(st)
			if id != -1 && id < len(logits.Data) {
				logits.Data[id] = -1e9
			}
		}

		// Grammar Mask: hard structural constraints applied before sampling.
		applyGrammarMask(logits, resIDs, model.SentenceVocab)

		//  Punctuation Proximity Penalty: scale down probability of punctuation if too close
		//  or if consecutive punctuation tokens appear back-to-back.
		consecutivePunct := 0
		for j := len(resIDs) - 1; j >= 0; j-- {
			w := model.SentenceVocab.GetWord(resIDs[j])
			if w == "." || w == "," || w == "!" || w == "?" || w == ";" || w == ":" {
				consecutivePunct++
			} else {
				break
			}
		}
		punctPenalty := float32(5.0)
		if consecutivePunct > 1 {
			punctPenalty = 10.0 // Push probability of repeating punctuation out of bounds
		}
		if i-lastPunctStep < 4 || consecutivePunct > 1 {
			punctuation := []string{".", ",", "!", "?", ";", ":"}
			for _, p := range punctuation {
				id := model.SentenceVocab.GetTokenID(p)
				if id != -1 && id < len(logits.Data) {
					logits.Data[id] -= punctPenalty
				}
			}
		}

		// Step 4: 4-gram repetition penalty.
		// If any token would complete a 4-gram sequence that has already appeared
		// >= 2 times in the generated history, heavily penalize it to break loops
		if len(resIDs) >= 3 {
			fourgrams := make(map[[3]int]int)
			for j := 0; j+2 < len(resIDs); j++ {
				key := [3]int{resIDs[j], resIDs[j+1], resIDs[j+2]}
				fourgrams[key]++
			}
			tailKey := [3]int{resIDs[len(resIDs)-3], resIDs[len(resIDs)-2], resIDs[len(resIDs)-1]}
			if fourgrams[tailKey] >= 2 {
				// Penalize every token that would extend this repeated 3-gram prefix
				for candidateID := range logits.Data {
					logits.Data[candidateID] -= 15.0
				}
			}
		}

		moe.ApplyRepetitionPenalty(logits, resIDs, repetitionPenalty)
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		if unkID := model.SentenceVocab.GetTokenID("UNK"); unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		// Temperature top-k sampling: prevents mode collapse by sampling from the
		// top-k candidates proportional to their softmax probabilities.
		// Pure ArgMax (greedy) collapses to a single token the moment it has
		// a slight numerical edge — sampling breaks that attractor.
		// Lowered to 0.7 for sharper categorical sampling that still avoids greedy collapse.
		const genTemp = float32(0.7)
		const genTopK = 40
		ApplyTemperature(logits.Data, genTemp)
		topIndicesK, topProbs := getTopK(logits, genTopK)
		// Softmax over top-k only
		var sumP float32
		for _, p := range topProbs {
			if p > -1e8 { // skip masked entries
				sumP += float32(math.Exp(float64(p)))
			}
		}
		if sumP <= 0 {
			sumP = 1
		}
		r := rand.Float32() * sumP
		bestID := topIndicesK[0] // fallback
		var cdf float32
		for ki, idx := range topIndicesK {
			if topProbs[ki] <= -1e8 {
				continue
			}
			cdf += float32(math.Exp(float64(topProbs[ki])))
			if r <= cdf {
				bestID = idx
				break
			}
		}

		probs := tensor.Softmax(logits)
		if bestID == model.SentenceVocab.EosID {
			probs.Release()
			logits.Release()
			inputT.Release()
			break
		}

		resIDs = append(resIDs, bestID)
		counts[bestID]++
		currentTokenID = bestID

		// Update last punctuation step
		word := model.SentenceVocab.GetWord(bestID)
		if word == "." || word == "," || word == "!" || word == "?" {
			lastPunctStep = i
		}

		if verbose {
			topIndices, topValues := getTopK(probs, 5)
			for k := 0; k < 5; k++ {
				w := model.SentenceVocab.GetWord(topIndices[k])
				fmt.Printf("   [%d] %-12s (%.2f%%)\n", k+1, w, topValues[k]*100)
			}
		}
		probs.Release()
		logits.Release()
		inputT.Release()
	}

	result := ""
	for i, id := range resIDs {
		if i == 0 {
			continue
		} // Skip BOS
		result += model.SentenceVocab.GetWord(id) + " "
	}

	inputTensor.Release()
	hiddenState.Release()
	cellState.Release()
	ctx.Release()
	model.ClearState()

	return strings.TrimSpace(result), strings.Join(pathSteps, " -> "), allAtts
}

func StrictGenerateWithExperts(model *moe.IntentMoE, input string, maxLen int, repetitionPenalty float32) (string, []int) {
	// 1. Enter Eval Mode and set Router Temperature for stability
	// CRITICAL: Disable gradient tracking
	model.SetParamsRequiresGrad(false)

	oldTemps := make(map[*moe.MoELayer]float32)
	for _, layer := range moe.ActiveLayers {
		layer.SetMode(false)
		oldTemps[layer] = layer.RouterTemperature
		layer.RouterTemperature = 0.7
	}
	if model.Decoder.OutputMoE != nil {
		oldTemps[model.Decoder.OutputMoE] = model.Decoder.OutputMoE.RouterTemperature
		model.Decoder.OutputMoE.RouterTemperature = 0.7
	}

	defer func() {
		// Restore
		model.SetParamsRequiresGrad(true)
		model.ClearState()

		for layer, temp := range oldTemps {
			layer.SetMode(true)
			layer.RouterTemperature = temp
		}
		if model.Decoder.OutputMoE != nil {
			model.Decoder.OutputMoE.SetMode(true)
			model.Decoder.OutputMoE.RouterTemperature = oldTemps[model.Decoder.OutputMoE]
		}
	}()

	// Format input to match training: __intent__ social : __ques__ <input> __ans__
	formattedInput := "__intent__ social : __ques__ " + input + " __ans__"
	tokens := cleanTokenize(formattedInput)
	if len(tokens) == 0 {
		return "", nil
	}
	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	emb, err := model.Embedding.Forward(inputTensor)
	if err != nil {
		return "", nil
	}
	ctx, err := model.Encoder.Forward(emb)
	if err != nil {
		return "", nil
	}
	ctx = model.NormalizeContextVector(ctx)

	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	hiddenState, _ := ctx.Mean(1)
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})

	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	resIDs := []int{model.SentenceVocab.BosID}
	var usedExpertIDs []int
	currentTokenID := model.SentenceVocab.BosID
	counts := make(map[int]int)

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)
		logits, nextHidden, nextCell, expertID, _, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx, i)
		if err != nil {
			break
		}
		hiddenState = nextHidden
		cellState = nextCell
		usedExpertIDs = append(usedExpertIDs, expertID...)

		//  Add extra noise if we are in a MUTINY aftermath (Temp > 1.2)
		if model.Encoder.GetMoELayers()[0].RouterTemperature > 1.2 {
			for idx := range logits.Data {
				logits.Data[idx] += float32((rand.Float64()*2 - 1) * 0.15)
			}
		}

		// Grammar Mask: suppress illegal token sequences before sampling.
		applyGrammarMask(logits, resIDs, model.SentenceVocab)

		// Step 4: 4-gram repetition penalty (mirrors StrictGenerate).
		if len(resIDs) >= 3 {
			fourgrams := make(map[[3]int]int)
			for j := 0; j+2 < len(resIDs); j++ {
				key := [3]int{resIDs[j], resIDs[j+1], resIDs[j+2]}
				fourgrams[key]++
			}
			tailKey := [3]int{resIDs[len(resIDs)-3], resIDs[len(resIDs)-2], resIDs[len(resIDs)-1]}
			if fourgrams[tailKey] >= 2 {
				for candidateID := range logits.Data {
					logits.Data[candidateID] -= 15.0
				}
			}
		}

		moe.ApplyRepetitionPenalty(logits, resIDs, 2.5) // Harsh penalty for validation
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		unkID := model.SentenceVocab.GetTokenID("UNK")
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		// Increase topK to 5 to prevent "always answering one way"
		// Raw ArgMax selection for perfectly sharp token choice
		bestID := 0
		maxLogit := -math.MaxFloat64
		for tokenID, logitValue := range logits.Data {
			if float64(logitValue) > maxLogit {
				maxLogit = float64(logitValue)
				bestID = tokenID
			}
		}

		if i == 0 {
			logits.Data[model.SentenceVocab.EosID] = -1e9
		}
		if bestID == model.SentenceVocab.EosID {
			break
		}

		resIDs = append(resIDs, bestID)
		counts[bestID]++
		currentTokenID = bestID
	}

	var result []string
	for _, id := range resIDs[1:] {
		word := model.SentenceVocab.GetWord(id)
		if word != "<s>" && word != "</s>" && word != "<pad>" && word != "UNK" {
			result = append(result, word)
		}
	}
	return strings.Join(result, " "), usedExpertIDs
}

func GenerateTokens(model *moe.IntentMoE, input string, maxLen int, useGPU bool) []string {
	model.SetParamsRequiresGrad(false)
	defer func() {
		model.SetParamsRequiresGrad(true)
		model.ClearState()
	}()
	// Quiet version for circuit breaker
	tokens := cleanTokenize(input)
	if len(tokens) == 0 {
		return nil
	}
	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	if useGPU {
		inputTensor.ToGPU()
	}

	// 1. Encoder path (MUST match IntentMoE.Forward)
	emb, _ := model.Embedding.Forward(inputTensor)
	if model.EncoderPos != nil {
		emb, _ = model.EncoderPos.Forward(emb)
	}
	ctx, _ := model.Encoder.Forward(emb)
	if model.EncoderNorm != nil {
		ctx, _ = model.EncoderNorm.Forward(ctx)
	}

	if ctx == nil || ctx.Shape[1] == 0 {
		return nil
	}

	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize

	// Calculate initial hidden state using MEAN (matches decoder.go fix)
	initialHidden, _ := ctx.Mean(1)
	hiddenState, _ := initialHidden.Reshape([]int{batchSize, ctx.Shape[2]})

	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)
		logits, nextHidden, nextCell, _, _, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx, i)
		if err != nil {
			break
		}
		hiddenState = nextHidden
		cellState = nextCell

		// Mute UNK and PAD
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		unkID := model.SentenceVocab.GetTokenID("UNK")
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		// Raw ArgMax selection for perfectly sharp token choice
		bestID := 0
		maxLogit := -math.MaxFloat64
		for tokenID, logitValue := range logits.Data {
			if float64(logitValue) > maxLogit {
				maxLogit = float64(logitValue)
				bestID = tokenID
			}
		}
		resIDs = append(resIDs, bestID)
		currentTokenID = bestID
	}

	var result []string
	for _, id := range resIDs[1:] {
		result = append(result, model.SentenceVocab.GetWord(id))
	}
	return result
}

func BeamSearchDecode(model *moe.IntentMoE, ctx *tensor.Tensor, beamSize int, maxLen int) []int {
	model.SetParamsRequiresGrad(false)
	defer func() {
		model.SetParamsRequiresGrad(true)
		model.ClearState()
	}()
	const repetitionPenalty = 1.8 // 1.0 = no penalty, 2.0 = very aggressive
	const alpha = 0.7             // Length penalty coefficient

	beams := []Hypothesis{{IDs: []int{model.SentenceVocab.BosID}, Score: 0.0}}

	for step := 0; step < maxLen; step++ {
		candidates := []Hypothesis{}

		for _, b := range beams {
			if len(b.IDs) > 0 && b.IDs[len(b.IDs)-1] == model.SentenceVocab.EosID {
				candidates = append(candidates, b)
				continue
			}

			// Standard Forward Pass to get Logits
			inputT := tensor.NewTensor([]int{1, len(b.IDs)}, convertToFloat(b.IDs), false)
			// The context `ctx` from the encoder must be passed to the decoder at each step.
			logits, err := model.Decoder.Forward(ctx, inputT, 0.0)
			if err != nil {
				continue // Skip beam if it fails
			}

			// Get Log-Probabilities
			probs := tensor.Softmax(logits[len(logits)-1])
			logProbs := tensor.NewTensor(probs.Shape, make([]float32, len(probs.Data)), false)
			for i, p := range probs.Data {
				logProbs.Data[i] = float32(math.Log(float64(p) + 1e-9))
			}

			// --- APPLY PENALTY ---
			seen := make(map[int]bool)
			for _, id := range b.IDs {
				seen[id] = true
			}

			for i := 0; i < len(logProbs.Data); i++ {
				if seen[i] {
					if logProbs.Data[i] < 0 {
						logProbs.Data[i] *= repetitionPenalty
					} else {
						logProbs.Data[i] /= repetitionPenalty
					}
				}
			}

			// Prevent EOS at step 0 to avoid empty responses
			if step == 0 && model.SentenceVocab.EosID < len(logProbs.Data) {
				logProbs.Data[model.SentenceVocab.EosID] = -math.MaxFloat32
			}

			topKIndices, topKProbs := getTopK(logProbs, beamSize)

			for i := 0; i < len(topKIndices); i++ {
				newIDs := append([]int{}, b.IDs...)
				newIDs = append(newIDs, topKIndices[i])
				candidates = append(candidates, Hypothesis{
					IDs:   newIDs,
					Score: b.Score + topKProbs[i],
				})
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			// Normalize score by length^alpha
			scoreI := float64(candidates[i].Score) / math.Pow(float64(len(candidates[i].IDs)), alpha)
			scoreJ := float64(candidates[j].Score) / math.Pow(float64(len(candidates[j].IDs)), alpha)
			return scoreI > scoreJ
		})

		if len(candidates) > beamSize {
			beams = candidates[:beamSize]
		} else {
			beams = candidates
		}
	}
	if len(beams) == 0 {
		return []int{}
	}
	return beams[0].IDs
}

func BeamSearchDecodeFiltered(model *moe.IntentMoE, ctx *tensor.Tensor, beamSize int, maxLen int, filteredIDs []int) []int {
	model.SetParamsRequiresGrad(false)
	defer func() {
		model.SetParamsRequiresGrad(true)
		model.ClearState()
	}()
	const repetitionPenalty = 1.2 // 1.0 = no penalty, 2.0 = very aggressive
	const alpha = 0.7             // Length penalty coefficient
	const temperature = 0.7       // Flatten distribution to encourage non-UNK tokens

	beams := []Hypothesis{{IDs: []int{model.SentenceVocab.BosID}, Score: 0.0}}

	// Create a map for quick lookup of filtered IDs
	filterMap := make(map[int]bool)
	for _, id := range filteredIDs {
		filterMap[id] = true
	}

	for step := 0; step < maxLen; step++ {
		candidates := []Hypothesis{}

		for _, b := range beams {
			if len(b.IDs) > 0 && b.IDs[len(b.IDs)-1] == model.SentenceVocab.EosID {
				candidates = append(candidates, b)
				continue
			}

			// Standard Forward Pass to get Logits
			inputT := tensor.NewTensor([]int{1, len(b.IDs)}, convertToFloat(b.IDs), false)
			// The context `ctx` from the encoder must be passed to the decoder at each step.
			logits, err := model.Decoder.Forward(ctx, inputT, 0.0)
			if err != nil {
				continue // Skip beam if it fails
			}

			lastLogit := logits[len(logits)-1]
			ApplyTemperature(lastLogit.Data, float32(temperature))

			// Get Log-Probabilities
			probs := tensor.Softmax(lastLogit)
			logProbs := tensor.NewTensor(probs.Shape, make([]float32, len(probs.Data)), false)
			for i, p := range probs.Data {
				logProbs.Data[i] = float32(math.Log(float64(p) + 1e-9))
			}

			// --- APPLY PENALTY ---
			seen := make(map[int]bool)
			for _, id := range b.IDs {
				seen[id] = true
			}

			for i := 0; i < len(logProbs.Data); i++ {
				if seen[i] {
					if logProbs.Data[i] < 0 {
						logProbs.Data[i] *= float32(repetitionPenalty)
					} else {
						logProbs.Data[i] /= float32(repetitionPenalty)
					}
				}
			}

			// --- NEW: FILTER OUT UNWANTED TOKENS ---
			for id := range filterMap {
				if id < len(logProbs.Data) {
					logProbs.Data[id] = -3.4028235e38 // Set to a very low value to avoid being picked
				}
			}

			topKIndices, topKProbs := getTopK(logProbs, beamSize)

			for i := 0; i < len(topKIndices); i++ {
				newIDs := append([]int{}, b.IDs...)
				newIDs = append(newIDs, topKIndices[i])
				candidates = append(candidates, Hypothesis{
					IDs:   newIDs,
					Score: b.Score + topKProbs[i],
				})
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			// Normalize score by length^alpha
			scoreI := float64(candidates[i].Score) / math.Pow(float64(len(candidates[i].IDs)), alpha)
			scoreJ := float64(candidates[j].Score) / math.Pow(float64(len(candidates[j].IDs)), alpha)
			return scoreI > scoreJ
		})

		if len(candidates) > beamSize {
			beams = candidates[:beamSize]
		} else {
			beams = candidates
		}
	}
	if len(beams) == 0 {
		return []int{}
	}
	return beams[0].IDs
}

// TreeOfThoughtsDecode implements branch sampling for complex reasoning.
// Instead of running standard autoregressive decoding token-by-token, it:
// 1. Generates N candidate reasoning steps at step 1 using temperature sampling (T=0.7).
// 2. Scores candidates using the MoE gating network to evaluate which path best addresses the user prompt.
// 3. Continues decoding on the highest-scoring branch.
func TreeOfThoughtsDecode(model *moe.IntentMoE, input string, maxLen int, numBranches int) (string, error) {
	model.SetParamsRequiresGrad(false)
	defer func() {
		model.SetParamsRequiresGrad(true)
		model.ClearState()
	}()

	tokens := cleanTokenize(input)
	if len(tokens) == 0 {
		return "", fmt.Errorf("empty input")
	}

	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	emb, err := model.Embedding.Forward(inputTensor)
	if err != nil {
		return "", err
	}
	ctx, err := model.Encoder.Forward(emb)
	if err != nil {
		return "", err
	}

	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	hiddenState, _ := ctx.Mean(1)
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})

	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, padding}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	// Step 1: Generate N candidate first tokens using temperature sampling
	temperature := 0.7
	firstInput := tensor.NewTensor([]int{1, 1}, []float32{float32(model.SentenceVocab.BosID)}, false)
	logits, _, _, _, _, err := model.Decoder.DecodeStepWithExpert(firstInput, hiddenState, cellState, ctx, 0)
	if err != nil {
		return "", err
	}

	ApplyTemperature(logits.Data, float32(temperature))
	probs := tensor.Softmax(logits)
	topKIndices, topKProbs := getTopK(probs, numBranches)

	type Branch struct {
		IDs         []int
		Score       float32
		HiddenState *tensor.Tensor
		CellState   *tensor.Tensor
	}

	branches := make([]Branch, len(topKIndices))
	for i, idx := range topKIndices {
		branches[i] = Branch{
			IDs:         []int{model.SentenceVocab.BosID, idx},
			Score:       topKProbs[i],
			HiddenState: hiddenState,
			CellState:   cellState,
		}
	}

	// Step 2: Continue decoding each branch and score
	maxBranchLen := maxLen / numBranches
	for step := 1; step < maxBranchLen; step++ {
		newBranches := make([]Branch, 0, len(branches))
		for _, b := range branches {
			if len(b.IDs) > 0 && b.IDs[len(b.IDs)-1] == model.SentenceVocab.EosID {
				newBranches = append(newBranches, b)
				continue
			}

			inputT := tensor.NewTensor([]int{1, len(b.IDs)}, convertToFloat(b.IDs), false)
			bLogits, nextHidden, nextCell, _, _, err := model.Decoder.DecodeStepWithExpert(inputT, b.HiddenState, b.CellState, ctx, step)
			if err != nil {
				continue
			}

			bestID := 0
			maxLogit := -math.MaxFloat64
			for tokenID, logitValue := range bLogits.Data {
				if float64(logitValue) > maxLogit {
					maxLogit = float64(logitValue)
					bestID = tokenID
				}
			}

			newBranch := Branch{
				IDs:         append(append([]int{}, b.IDs...), bestID),
				Score:       b.Score + float32(maxLogit),
				HiddenState: nextHidden,
				CellState:   nextCell,
			}
			newBranches = append(newBranches, newBranch)
		}

		branches = newBranches
		if len(branches) == 0 {
			break
		}
	}

	// Step 3: Select highest-scoring branch
	if len(branches) == 0 {
		return "", fmt.Errorf("no valid branches generated")
	}

	bestBranch := branches[0]
	for _, b := range branches[1:] {
		if b.Score > bestBranch.Score {
			bestBranch = b
		}
	}

	// Convert IDs to text
	var result []string
	for _, id := range bestBranch.IDs[1:] {
		word := model.SentenceVocab.GetWord(id)
		if word != "<s>" && word != "</s>" && word != "<pad>" && word != "UNK" {
			result = append(result, word)
		}
	}

	return strings.Join(result, " "), nil
}
