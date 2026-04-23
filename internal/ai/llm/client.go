package llm

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/tagger/nertagger"
	"github.com/golangast/gollemer/internal/ai/tagger/postagger"
	"github.com/golangast/gollemer/internal/ai/tagger/tag"
	"github.com/golangast/gollemer/internal/platform/ui"
	"github.com/golangast/gollemer/internal/platform/watcher"
)

// GollemerMoEClient implements the MoEClient interface using the existing NLP pipeline.
type GollemerMoEClient struct {
	KB                *KnowledgeBase
	Model             *moe.IntentMoE
	SocialModel       *moe.IntentMoE // Specialized model for social conversations
	W2V               *word2vec.SimpleWord2Vec
	ChatBank          []ChatPair
	History           []ChatPair
	lastMoEPrediction string
	CommandAnchors    map[string][]float64
}

func (c *GollemerMoEClient) PushHistory(q, a, intent string) {
	c.History = append(c.History, ChatPair{Q: q, A: a, Intent: intent})
	if len(c.History) > 10 { // Keep last 10 turns
		c.History = c.History[1:]
	}
}

func (c *GollemerMoEClient) LoadChatBank(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		log.Printf("⚠️  Could not load ChatBank from %s: %v", path, err)
		return
	}

	var pairs []ChatPair
	// Try JSON first if it doesn't look like a CSV header or if extension is .json
	if !strings.HasSuffix(strings.ToLower(path), ".csv") {
		if err := json.Unmarshal(data, &pairs); err == nil {
			c.ChatBank = pairs
			log.Printf("✅ Loaded %d prompts into ChatBank from JSON", len(c.ChatBank))
			return
		}
	}

	// Try CSV parsing
	reader := csv.NewReader(strings.NewReader(string(data)))
	records, err := reader.ReadAll()
	if err != nil {
		log.Printf("⚠️  Error parsing CSV from %s: %v", path, err)
		return
	}

	for i, record := range records {
		if i == 0 && strings.Contains(strings.ToLower(record[0]), "intent") {
			continue // Skip header
		}
		if len(record) >= 3 {
			// Pattern is record[1], Response is record[2]
			q := strings.Trim(record[1], "\" ")
			a := strings.Trim(record[2], "\" ")
			intent := strings.Trim(record[0], "\" ")
			if q != "" && a != "" {
				pairs = append(pairs, ChatPair{Q: q, A: a, Intent: intent})
			}
		}
	}

	c.ChatBank = pairs
}

func (c *GollemerMoEClient) RetrieveChatResponse(input string) (string, string, float64) {
	if len(c.ChatBank) == 0 || c.W2V == nil {
		return "", "", 0
	}

	bestScore := -1.0
	bestResponse := ""
	bestIntent := ""

	input = strings.TrimSpace(strings.ToLower(input))
	inputWords := cleanTokenize(input)

	// 1. Identify context
	var lastTurnIntent string
	var lastTurnQA string
	if len(c.History) > 0 {
		last := c.History[len(c.History)-1]
		lastTurnIntent = last.Intent
		lastTurnQA = last.Q + " " + last.A
	}

	// 2. Identify if continuation
	isCont := false
	contWords := []string{"then", "next", "now", "after", "else", "more", "further", "follow", "what now", "how about", "anything"}
	for _, cw := range contWords {
		if strings.Contains(input, cw) {
			isCont = true
			break
		}
	}
	if !isCont && (len(inputWords) < 3 && !isCreatingCommand(input)) {
		isCont = true
	}

	// 3. Embed current input
	inputEmbed := c.getSentenceEmbedding(input)
	if inputEmbed == nil {
		return "", "", 0
	}

	// 4. Embed history (last turn context)
	var historyEmbed []float64
	if lastTurnQA != "" {
		historyEmbed = c.getSentenceEmbedding(lastTurnQA)
	}

	for _, pair := range c.ChatBank {
		pairQ := strings.TrimSpace(strings.ToLower(pair.Q))
		pairEmbed := c.getSentenceEmbedding(pairQ)
		if pairEmbed == nil {
			continue
		}

		// --- [Fix: Exact Match Boost] ---
		// If the query is an exact match, it should ALWAYS win over similar embeddings.
		if input == pairQ {
			return pair.A, pair.Intent, 1.0
		}

		score := cosineSimilarity(inputEmbed, pairEmbed)

		// 5. Aggressive Context Reinforcement for continuations
		if isCont && historyEmbed != nil {
			histScore := cosineSimilarity(historyEmbed, pairEmbed)
			score = 0.2*score + 0.8*histScore
		}

		// 6. Topic Bias: Boost if it stays in the same intent family
		if lastTurnIntent != "" {
			lastParts := strings.Split(lastTurnIntent, "_")
			currParts := strings.Split(pair.Intent, "_")
			if len(lastParts) > 0 && len(currParts) > 0 && lastParts[0] == currParts[0] {
				score *= 1.05 // Subtle category boost
			}
		}

		// 7. Repetition Penalty
		for i := len(c.History) - 1; i >= 0; i-- {
			if c.History[i].A == pair.A {
				recencyWeight := float64(i+1) / float64(len(c.History))
				score *= (0.4 + (0.3 * (1.0 - recencyWeight)))
				break
			}
		}

		if score > bestScore {
			bestScore = score
			bestResponse = pair.A
			bestIntent = pair.Intent
		}
	}

	if bestScore > 1.0 {
		bestScore = 1.0
	}

	return bestResponse, bestIntent, bestScore
}

func (c *GollemerMoEClient) getSentenceEmbedding(text string) []float64 {
	words := cleanTokenize(text)
	if len(words) == 0 {
		return nil
	}
	embedding := make([]float64, c.W2V.VectorSize)
	count := 0
	for _, w := range words {
		if id, ok := c.W2V.Vocabulary[w]; ok {
			vec := c.W2V.WordVectors[id]
			for i, v := range vec {
				embedding[i] += v
			}
			count++
		}
	}
	if count > 0 {
		for i := range embedding {
			embedding[i] /= float64(count)
		}
		return embedding
	}
	return nil
}

// isSocialIntent detects if a query is social/conversational rather than technical
func isSocialIntent(input string) bool {
	lowerInput := strings.ToLower(input)

	// Social intent keywords
	socialKeywords := []string{
		"how are you", "how you doing", "how's it going", "what's up",
		"favorite", "like", "love", "hate", "enjoy", "think", "feel", "opinion",
		"holiday", "vacation", "weekend", "party", "friend", "family",
		"weather", "beautiful", "fun", "interesting", "amazing", "cool",
		"tell me about", "what do you think", "do you ever", "have you ever",
		"personal", "life", "work", "hobby", "passion", "dream",
		"meeting", "people", "connection", "relationship", "love", "dating",
		"hope", "wish", "amazing", "wonderful", "boring", "difficult",
	}

	// Technical keywords (should NOT be treated as social)
	technicalKeywords := []string{
		"create", "file", "handler", "project", "function", "class",
		"go", "code", "program", "run", "build", "deploy",
		"database", "sql", "api", "server", "client", "network",
		"import", "package", "module", "library", "framework",
	}

	// If it contains technical keywords, it's NOT social
	for _, tech := range technicalKeywords {
		if strings.Contains(lowerInput, tech) {
			return false
		}
	}

	// If it contains social keywords, it IS social
	for _, social := range socialKeywords {
		if strings.Contains(lowerInput, social) {
			return true
		}
	}

	// Default: if it's short and has no technical keywords, it might be social
	if len(strings.Fields(lowerInput)) <= 3 && !strings.Contains(lowerInput, "list") {
		return true
	}

	return false
}

func (c *GollemerMoEClient) PredictIntent(input string) (string, float64) {
	c.lastMoEPrediction = "" // Clear previous turn's chat prediction
	lowerInput := strings.ToLower(input)

	// --- SOCIAL ROUTING: If it's a social query and we have a social model, use it ---
	if isSocialIntent(input) && c.SocialModel != nil {
		// Generate response using social model
		response := c.GenerateSocialResponse(input)
		if response != "" {
			c.lastMoEPrediction = response
			log.Printf("🧠 Neural Social Match: Using weights from moe_social_model.gob")
			return "social_chat", 0.95
		}
		log.Printf("⚖️  Quality Gate: Social model output was too high-entropy (word salad); falling back to retrieval.")
	}

	// --- 0. Instant Heuristics for Dynamic Queries ---
	if (strings.Contains(lowerInput, "time") || strings.Contains(lowerInput, "clock")) &&
		(strings.Contains(lowerInput, "what") || strings.Contains(lowerInput, "know") || strings.Contains(lowerInput, "tell")) {
		return "time_query", 0.95
	}
	if strings.Contains(lowerInput, "weather") {
		return "weather_query", 0.90
	}
	if lowerInput == "pwd" || (strings.Contains(lowerInput, "directory") && (strings.Contains(lowerInput, "where") || strings.Contains(lowerInput, "what") || strings.Contains(lowerInput, "current"))) {
		return "pwd_query", 0.99
	}
	if strings.Contains(lowerInput, "who are you") || strings.Contains(lowerInput, "your name") || lowerInput == "identity" {
		return "identity_query", 0.99
	}
	if (strings.Contains(lowerInput, "webserver") || strings.Contains(lowerInput, "app")) &&
		(strings.Contains(lowerInput, "name") || strings.Contains(lowerInput, "identify") || strings.Contains(lowerInput, "what") || strings.Contains(lowerInput, "which")) {
		return "webserver_identity_query", 0.99
	}
	if lowerInput == "hi" || lowerInput == "hello" || lowerInput == "hey" || lowerInput == "greeting" {
		return "greeting_query", 0.99
	}

	// --- 0.2. Common Coding Queries ---
	if strings.Contains(lowerInput, "better at go") || strings.Contains(lowerInput, "learn go") || strings.Contains(lowerInput, "go tutorial") {
		c.lastMoEPrediction = "The best way to get better at Go is to build projects! Start with a simple webserver or try the 'tutorial' command here."
		return "gollemer_logic", 0.99
	}

	if lowerInput == "help" || strings.HasPrefix(lowerInput, "help ") || lowerInput == "help me" {
		return "help_command", 0.99
	}
	// --- 1. Primary Command Heuristics ---
	if intent, score := c.checkCommandHeuristics(lowerInput); score > 0.8 {
		return intent, score
	}

	// --- 1. Combined Retrieval & Neural Logic ---
	retrievedResp, retrievedIntent, retrievedScore := c.RetrieveChatResponse(input)
	log.Printf("🔍 Intent Retrieval Top Score: %.4f (%s)", retrievedScore, retrievedIntent)

	var neuralResponse string
	var neuralIntent string
	var neuralScore float64

	if c.Model != nil && c.W2V != nil {
		formattedInput := fmt.Sprintf("[Intent: social] [QUES] %s [ANS]", lowerInput)
		cleanWords := cleanTokenize(formattedInput)
		var tokenIDs []int
		for _, w := range cleanWords {
			if c.Model.SentenceVocab != nil {
				tokenIDs = append(tokenIDs, lookupVocab(w, c.Model.SentenceVocab))
			} else if id, ok := c.W2V.Vocabulary[w]; ok {
				tokenIDs = append(tokenIDs, id)
			}
		}

		if len(tokenIDs) > 0 {
			inputTensor := tensor.NewTensor([]int{1, len(tokenIDs)}, make([]float32, len(tokenIDs)), false)
			for i, id := range tokenIDs {
				inputTensor.Data[i] = float32(id)
			}

			contextVector, err := c.Model.EncoderForward(inputTensor, nil)
			if err == nil {
				contextVector = c.Model.NormalizeContextVector(contextVector)
				posTags := postagger.TagTokens(cleanWords)
				taggedData := nertagger.Nertagger(tag.Tag{Tokens: cleanWords, PosTag: posTags})

				if c.Model.SentenceVocab != nil {
					outputIDs, err := c.Model.GreedySearchDecodeWithTemp(
						contextVector, 20,
						c.Model.SentenceVocab.BosID, c.Model.SentenceVocab.EosID,
						0.4, 1.2, 0.3, 50, taggedData,
					)
					if err == nil && len(outputIDs) > 0 {
						var decodedWords []string
						for _, id := range outputIDs {
							w := c.Model.SentenceVocab.GetWord(id)
							if w != "</s>" && w != "<s>" && w != "<pad>" && w != "UNK" && w != "" {
								decodedWords = append(decodedWords, w)
							}
						}
						neuralResponse = strings.Join(decodedWords, " ")
						if neuralResponse != "" {
							log.Printf("🧠 Neural Model (moe_classification_model.gob) generated: %s", neuralResponse)
							if isGarbageOutput(neuralResponse) {
								log.Printf("⚖️  Quality Gate: Main model output rejected (structural incoherence).")
								neuralResponse = ""
							} else if strings.HasPrefix(neuralResponse, "create webserver") {
								neuralIntent = "create_webserver"
								neuralScore = 0.99
							} else if strings.HasPrefix(neuralResponse, "create handler") {
								neuralIntent = "create_handler"
								neuralScore = 0.99
							} else {
								neuralIntent = "chat_response"
								neuralScore = 0.91
							}
						}
					}
				}
			}
		}
	}

	// --- 2. Winner Selection ---
	if neuralIntent != "" && neuralIntent != "chat_response" {
		return neuralIntent, neuralScore
	}

	const retrievalThreshold = 0.96
	if retrievedScore >= retrievalThreshold {
		c.lastMoEPrediction = retrievedResp
		return retrievedIntent, retrievedScore
	}

	if neuralResponse != "" {
		c.lastMoEPrediction = neuralResponse
		return neuralIntent, neuralScore
	}

	if retrievedScore > 0.8 {
		c.lastMoEPrediction = retrievedResp
		return retrievedIntent, retrievedScore
	}

	// --- 3. Weighted Keyword "Fuzzy" Match ---
	if len(c.CommandAnchors) > 0 && c.W2V != nil {
		userVec := c.getSentenceEmbedding(lowerInput)
		if userVec != nil {
			bestMatch := ""
			maxSim := 0.75
			for intent, anchorVec := range c.CommandAnchors {
				if anchorVec == nil {
					continue
				}
				sim := cosineSimilarity(userVec, anchorVec)
				if sim > maxSim {
					maxSim = sim
					bestMatch = intent
				}
			}
			if bestMatch != "" {
				return bestMatch, maxSim
			}
		}
	}

	if lowerInput != "" {
		c.lastMoEPrediction = "I'm picking up some signal, but I'm not sure what you need. Want to see the menu?"
		return "chat_response", 0.05
	}

	return "", 0.0
}

func (c *GollemerMoEClient) GenerateSocialResponse(input string) string {
	if c.SocialModel == nil {
		return ""
	}

	// Use the same StrictGenerate-equivalent decoder path as training validation.
	// GreedySearchDecodeWithTemp is a DIFFERENT decoder and produces word salad
	// because it was never used during training. Match exactly what was trained.
	model := c.SocialModel
	if model.SentenceVocab == nil || model.Decoder == nil || model.Embedding == nil || model.Encoder == nil {
		return ""
	}

	// Format input to match training exactly: [QUES] <input> [ANS]
	formattedInput := "[QUES] " + input + " [ANS]"
	tokens := cleanTokenize(formattedInput)
	if len(tokens) == 0 {
		return ""
	}

	inputIDs := make([]float32, len(tokens))
	for i, t := range tokens {
		inputIDs[i] = float32(lookupVocab(t, model.SentenceVocab))
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputIDs)}, inputIDs, false)

	// Encode
	emb, err := model.Embedding.Forward(inputTensor)
	if err != nil {
		log.Printf("GenerateSocialResponse Error (Embedding): %v", err)
		return ""
	}
	ctx, err := model.Encoder.Forward(emb)
	if err != nil {
		log.Printf("GenerateSocialResponse Error (Encoder): %v", err)
		return ""
	}
	ctx = model.NormalizeContextVector(ctx)
	if ctx.Shape[1] == 0 {
		return ""
	}

	// Init decoder hidden/cell state from encoder context mean
	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	hiddenState, err := ctx.Mean(1)
	if err != nil {
		return ""
	}
	hiddenState, _ = hiddenState.Reshape([]int{batchSize, ctx.Shape[2]})
	if hiddenState.Shape[1] != hiddenSize {
		if hiddenState.Shape[1] > hiddenSize {
			hiddenState, _ = hiddenState.Slice(1, 0, hiddenSize)
		} else {
			pad := tensor.NewTensor([]int{batchSize, hiddenSize - hiddenState.Shape[1]}, make([]float32, batchSize*(hiddenSize-hiddenState.Shape[1])), false)
			hiddenState, _ = tensor.Concat([]*tensor.Tensor{hiddenState, pad}, 1)
		}
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	resIDs := []int{model.SentenceVocab.BosID}
	currentTokenID := model.SentenceVocab.BosID
	counts := make(map[int]int)
	unkID := model.SentenceVocab.GetTokenID("UNK")
	maxLen := 30

	for i := 0; i < maxLen; i++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentTokenID)}, false)
		logits, nextHidden, nextCell, _, err := model.Decoder.DecodeStepWithExpert(inputT, hiddenState, cellState, ctx)
		if err != nil {
			break
		}
		hiddenState = nextHidden
		cellState = nextCell

		// Suppress EOS for first 5 tokens — same as StrictGenerate training validation
		if i < 5 {
			logits.Data[model.SentenceVocab.EosID] = -1e9
		}

		// Repetition penalty
		moe.ApplyRepetitionPenalty(logits, resIDs, 1.2)

		// Frequency penalty
		const freqPenalty = 0.5
		for id, count := range counts {
			if id < len(logits.Data) {
				logits.Data[id] -= freqPenalty * float32(count)
			}
		}

		// Mute PAD and UNK
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}

		bestID, err := moe.SampleFromLogits(logits, 0.8, 1, 0.9)
		if err != nil {
			break
		}
		if bestID == model.SentenceVocab.EosID {
			break
		}

		resIDs = append(resIDs, bestID)
		counts[bestID]++
		currentTokenID = bestID
	}

	// Decode tokens to words
	var result []string
	for _, id := range resIDs[1:] {
		w := model.SentenceVocab.GetWord(id)
		if w != "</s>" && w != "<s>" && w != "<pad>" && w != "UNK" && w != "" {
			result = append(result, w)
		}
	}

	response := strings.Join(result, " ")
	if response == "" {
		return ""
	}
	log.Printf("🎭 Social Model generated: %s", response)
	if isGarbageOutput(response) || isLowQualitySocialResponse(response) {
		log.Printf("🗑️  Social output rejected (quality gate): %s", response)
		return ""
	}
	return response
}

func (c *GollemerMoEClient) checkCommandHeuristics(lowerInput string) (string, float64) {
	createVerbs := []string{"create", "make", "add", "generate", "initialize", "init", "new", "setup"}
	for _, v := range createVerbs {
		if lowerInput == v || strings.HasPrefix(lowerInput, v+" ") {
			c.lastMoEPrediction = "I am on it! I'll generate the new resources for you right away."
			if lowerInput == v {
				return "create_generic", 0.95
			}
			return c.handleCreateCommand(lowerInput)
		}
	}

	if strings.HasPrefix(lowerInput, "list ") || lowerInput == "list" || strings.HasPrefix(lowerInput, "ls ") || lowerInput == "ls" {
		c.lastMoEPrediction = "Checking the directory for you... here is what I found:"
		return "list_query", 0.99
	}
	if strings.HasPrefix(lowerInput, "go ") || lowerInput == "go" || strings.HasPrefix(lowerInput, "cd ") || lowerInput == "cd" || strings.HasPrefix(lowerInput, "goto ") {
		return "go_query", 0.99
	}
	if strings.HasPrefix(lowerInput, "cat ") || strings.HasPrefix(lowerInput, "read ") {
		return "cat_query", 0.99
	}
	if strings.HasPrefix(lowerInput, "run ") || lowerInput == "run" || strings.HasPrefix(lowerInput, "start ") || lowerInput == "start" {
		c.lastMoEPrediction = "Launching the application! 🚀 Hang tight."
		return "run_webserver", 0.99
	}
	if lowerInput == "profile" || strings.HasPrefix(lowerInput, "show profile") || strings.HasPrefix(lowerInput, "project status") {
		return "profile_query", 0.99
	}
	if strings.HasPrefix(lowerInput, "stop ") || lowerInput == "stop" || strings.HasPrefix(lowerInput, "kill ") || lowerInput == "kill" || strings.HasPrefix(lowerInput, "terminate ") {
		return "stop", 0.99
	}
	if lowerInput == "watch" || lowerInput == "monitor" || lowerInput == "guard" {
		return "watch", 0.99
	}
	if lowerInput == "pwd" || lowerInput == "history" || lowerInput == "clear" || lowerInput == "cls" {
		return lowerInput + "_query", 0.99
	}

	return "", 0.0
}

func (c *GollemerMoEClient) handleCreateCommand(lowerInput string) (string, float64) {
	targets := map[string]string{
		"webserver": "create_webserver", "site": "create_webserver", "project": "create_webserver",
		"page": "create_page", "view": "create_page", "handler": "create_handler", "route": "create_handler",
		"database": "create_database", "db": "create_database", "file": "create_file", "folder": "create_folder",
		"directory": "create_folder", "form": "create_form", "structure": "create_structure",
	}
	for key, intent := range targets {
		if strings.Contains(lowerInput, key) {
			return intent, 0.95
		}
	}
	return "create_generic", 0.81
}

func (c *GollemerMoEClient) ExtractEntities(input string, intent string) map[string]any {
	words := strings.Fields(input)
	posTags := postagger.TagTokens(words)
	taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

	entities := make(map[string]any)

	if (intent == "social_chat" || intent == "chat_response" || strings.HasPrefix(intent, "Social_") || strings.HasPrefix(intent, "System_") || strings.HasPrefix(intent, "gollemer_")) && c.lastMoEPrediction != "" {
		entities["response"] = c.lastMoEPrediction
		return entities
	}

	name := findName(taggedData, c.KB)

	if strings.Contains(strings.ToLower(input), "data structure") {
		words := strings.Fields(input)
		for i, w := range words {
			if strings.ToLower(w) == "structure" && i > 0 && strings.ToLower(words[i-1]) == "data" {
				if i+1 < len(words) {
					candidate := words[i+1]
					if strings.ToLower(candidate) != "to" && strings.ToLower(candidate) != "in" && strings.ToLower(candidate) != "with" {
						name = candidate
					}
				}
			}
		}
	}

	if intent == "create_page" {
		lowerInput := strings.ToLower(input)
		pageIdx := strings.Index(lowerInput, "page")
		if pageIdx != -1 {
			remaining := input[pageIdx+4:]
			endMarkers := []string{" in ", " to ", " for ", " with ", " wasm ", " webserver "}
			endIdx := len(remaining)
			for _, marker := range endMarkers {
				mIdx := strings.Index(strings.ToLower(remaining), marker)
				if mIdx != -1 && mIdx < endIdx {
					endIdx = mIdx
				}
			}
			candidate := strings.TrimSpace(remaining[:endIdx])
			if candidate != "" {
				name = candidate
			}
		}
	}

	if name != "" {
		entities["name"] = name
	}

	for i, token := range taggedData.Tokens {
		lower := strings.ToLower(token)
		if lower == "url" && i+1 < len(taggedData.Tokens) {
			val := taggedData.Tokens[i+1]
			if val == "is" && i+2 < len(taggedData.Tokens) {
				val = taggedData.Tokens[i+2]
			}
			if strings.HasPrefix(val, "/") {
				entities["url"] = val
			}
		} else if strings.HasPrefix(token, "/") && !strings.Contains(token, ".") && (intent == "create_handler" || intent == "create_page") {
			entities["url"] = token
		}
	}

	for i, token := range taggedData.Tokens {
		lowerToken := strings.ToLower(token)
		if (lowerToken == "in" || lowerToken == "into" || lowerToken == "to") && i+1 < len(taggedData.Tokens) {
			j := i + 1
			for ; j < len(taggedData.Tokens); j++ {
				t := strings.ToLower(taggedData.Tokens[j])
				if t == "the" || t == "a" || t == "an" || t == "folder" || t == "directory" {
					continue
				}
				break
			}
			if j < len(taggedData.Tokens) {
				entities["path"] = taggedData.Tokens[j]
			}
		}
	}

	if strings.Contains(input, "fields") {
		fields := make(map[string]string)
		parts := strings.Fields(input)
		startIdx := -1

		for i, p := range parts {
			if strings.ToLower(p) == "fields" {
				startIdx = i + 1
				break
			}
		}

		if startIdx != -1 {
			for i := startIdx; i < len(parts)-1; i += 2 {
				if strings.ToLower(parts[i]) == "and" {
					i--
					continue
				}
				fields[parts[i]] = parts[i+1]
			}
		}
		if len(fields) > 0 {
			entities["tables"] = fields
		}
	}

	if intent == "help_command" {
		for cmd := range c.KB.KnownCommands {
			if strings.Contains(strings.ToLower(input), cmd) {
				entities["command"] = cmd
				break
			}
		}
	}

	return entities
}

func (c *GollemerMoEClient) ResolveServerPath(name string) (string, error) {
	cwd, _ := os.Getwd()
	projectRoot, _ := FindProjectRoot()

	paths := []string{
		name,
		filepath.Join(cwd, name),
		filepath.Join("cmd", name),
		filepath.Join(projectRoot, name),
		filepath.Join(projectRoot, "cmd", name),
		filepath.Join(cwd, "..", name),
	}

	for _, p := range paths {
		target := filepath.Join(p, "main.go")
		if _, err := os.Stat(target); err == nil {
			abs, _ := filepath.Abs(p)
			return abs, nil
		}
	}
	return "", fmt.Errorf("webserver '%s' not found", name)
}

func (c *GollemerMoEClient) WaitForPulse(address string, timeout time.Duration, mascot *ui.Mascot) bool {
	client := http.Client{Timeout: 1 * time.Second}
	deadline := time.Now().Add(timeout)

	mascot.Say(ui.Think, "Checking for a pulse on "+address+"...")

	for time.Now().Before(deadline) {
		resp, err := client.Get("http://localhost" + address)
		if err == nil {
			resp.Body.Close()
			return true
		}
		time.Sleep(500 * time.Millisecond)
	}
	return false
}

func (c *GollemerMoEClient) RunDoctor(m *ui.Mascot) {
	m.Say(ui.Think, "Initiating full system diagnostic... 🩺")

	misplaced := c.ScanForRootServers()
	for _, s := range misplaced {
		m.ProposeMove(s, func() error {
			return c.MoveToCmd(s, m)
		})
	}

	mainFiles := []string{"main.go"}
	for _, f := range mainFiles {
		if _, err := os.Stat(f); err == nil {
			content, _ := os.ReadFile(f)
			if !strings.Contains(string(content), "// HANDLER_REGISTRATIONS_GO_HERE") {
				m.ConfirmRepair("Missing handler anchor in "+f, func() error {
					return InjectPlaceholder(f)
				})
			}
		}
	}

	m.Say(ui.Happy, "Diagnostics complete. Project is healthy!")
}

func (c *GollemerMoEClient) RunAudit(m *ui.Mascot) {
	scanRoot := "."
	m.Say(ui.Think, "Performing a deep architectural audit of the current directory... 🔍")
	m.AuditProjectSize(scanRoot)

	filepath.Walk(scanRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() || !strings.HasSuffix(path, ".go") || strings.Contains(path, "vendor") || strings.Contains(path, "wasm") {
			return nil
		}
		m.AnalyzeComplexity(path)
		return nil
	})

	m.HuntDeadCode(scanRoot)
	m.AuditGlobalState(scanRoot)
	m.SimulateRaceConditions(scanRoot)
	m.AuditMemoryLeaks(scanRoot)

	m.Say(ui.Happy, "Audit complete. I've highlighted potential issues in the current workspace.")
}

func (c *GollemerMoEClient) ScanForRootServers() []string {
	var misplaced []string
	entries, _ := os.ReadDir(".")
	ignore := map[string]bool{"cmd": true, "internal": true, "vendor": true, "data/training/trainingdata": true, "examples/learningfolder": true, "pkg": true}

	for _, entry := range entries {
		if entry.IsDir() && !ignore[entry.Name()] && !strings.HasPrefix(entry.Name(), ".") {
			mainPath := filepath.Join(entry.Name(), "main.go")
			if _, err := os.Stat(mainPath); err == nil {
				misplaced = append(misplaced, entry.Name())
			}
		}
	}
	return misplaced
}

func (c *GollemerMoEClient) MoveToCmd(name string, m *ui.Mascot) error {
	targetDir := filepath.Join("cmd", name)
	if err := os.MkdirAll("cmd", 0755); err != nil {
		return err
	}
	if err := os.Rename(name, targetDir); err != nil {
		return err
	}
	cmd := exec.Command("go", "mod", "tidy")
	cmd.Dir = targetDir
	return cmd.Run()
}

func (c *GollemerMoEClient) StartBackgroundWatcher(m *ui.Mascot, projectRoot string) {
	w := watcher.NewWorkspace()
	w.Scan(projectRoot)

	ticker := time.NewTicker(2500 * time.Millisecond)
	go func() {
		for range ticker.C {
			changes := w.Scan(projectRoot)
			for path, status := range changes {
				m.RecordActivity(path, 0)
				m.ReactToFileChange(path, status)
			}
		}
	}()
}

func (c *GollemerMoEClient) MascotCommit(m *ui.Mascot, reader *bufio.Reader) {
	suggestion := m.SuggestCommit()

	fmt.Printf("\n%s/ʕ◕‿◕ʔ/ > \"I've been watching your pulse. I suggest this commit message: '%s'\"%s\n", ui.ColorCyan, suggestion, ui.ColorReset)
	fmt.Print(">> Press Enter to use, or type a new message (or 'c' to cancel): ")

	input, _ := reader.ReadString('\n')
	input = strings.TrimSpace(input)

	if strings.ToLower(input) == "c" {
		m.Say(ui.Neutral, "Commit cancelled. Let's keep refining!")
		return
	}

	finalMessage := suggestion
	if input != "" {
		finalMessage = input
	}

	m.Say(ui.Thinking, "Executing git commit...")

	cmd := exec.Command("git", "commit", "-am", finalMessage)
	output, err := cmd.CombinedOutput()

	if err != nil {
		m.Say(ui.Disturbed, "Git hit a snag: "+strings.TrimSpace(string(output)))
		return
	}

	m.Say(ui.Happy, "Success! Changes pushed to the timeline. Velocity: "+strconv.Itoa(m.GetVelocity())+" pulses/hr.")
}
