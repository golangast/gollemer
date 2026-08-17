package llm

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/moe"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/word2vec"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
	"github.com/golangast/gollemer/internal/ai/orchestrator"
	"github.com/golangast/gollemer/internal/ai/tagger/nertagger"
	"github.com/golangast/gollemer/internal/ai/tagger/postagger"
	"github.com/golangast/gollemer/internal/ai/tagger/tag"
	"github.com/golangast/gollemer/internal/platform/ui"
	"github.com/golangast/gollemer/internal/platform/watcher"
)

// GollemerMoEClient implements the MoEClient interface using the existing NLP pipeline.
//
// ChatBank overrides have been removed. The neural MoE weights handle ALL
// responses — both conversational and code queries — directly using BPE
// tokenization and ChatML format (when BPE tokenizer is available).
type GollemerMoEClient struct {
	KB                *KnowledgeBase
	Model             *moe.IntentMoE
	SocialModel       *moe.IntentMoE // Specialized model for social conversations
	W2V               *word2vec.SimpleWord2Vec
	History           []ChatPair
	lastMoEPrediction string
	CommandAnchors    map[string][]float64
	SocialConfig      *orchestrator.SafeConfig
	BPETokenizer      *tokenizer.BPETokenizer // BPE tokenizer for ChatML-based inference
	ChatBank          []ChatPair              // Deprecated: kept for backward compat; neural MoE used instead

	// Multiconversational session layer.
	Sessions  *SessionManager
	SessionID string
}

// session returns the Conversation for the client's active SessionID.
// It lazily initialises Sessions if nil (backward-compat for tests that
// construct GollemerMoEClient directly without calling runner.Init).
func (c *GollemerMoEClient) session() *Conversation {
	if c.Sessions == nil {
		c.Sessions = NewSessionManager()
	}
	if c.SessionID == "" {
		c.SessionID = "default"
	}
	return c.Sessions.GetOrCreate(c.SessionID)
}

func (c *GollemerMoEClient) PushHistory(q, a, intent string) {
	c.History = append(c.History, ChatPair{Q: q, A: a, Intent: intent})
	if len(c.History) > 10 { // Keep last 10 turns
		c.History = c.History[1:]
	}

	// Mirror into the session's Message log so the token-aware context window
	// stays in sync with the flat ChatPair history.
	sess := c.session()
	if q != "" {
		sess.mu.RLock()
		lastIsSameUser := len(sess.Messages) > 0 && sess.Messages[len(sess.Messages)-1].Role == "user" &&
			strings.TrimSpace(sess.Messages[len(sess.Messages)-1].Content) == strings.TrimSpace(q)
		sess.mu.RUnlock()
		if !lastIsSameUser {
			sess.AddMessage("user", q)
		}
	}
	if a != "" {
		sess.AddMessage("assistant", a)
	}

	// Dynamic Learning is temporarily disabled to prevent the model from
	// corrupting its own training dataset with unverified "word salad" generations.
	// We rely on the Teacher (Ollama) AI Supervisor for dataset evolution instead.
}

// GetLastPrediction returns the most recently generated model response.
// It is the exported accessor for the unexported lastMoEPrediction field.
func (c *GollemerMoEClient) GetLastPrediction() string {
	return c.lastMoEPrediction
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
			c.ChatBank = append(c.ChatBank, pairs...)
			log.Printf("✅ Loaded %d prompts into ChatBank from JSON (Total: %d)", len(pairs), len(c.ChatBank))
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

	var isMultiTurn = false
	for i, record := range records {
		if i == 0 {
			headerStr := strings.ToLower(strings.Join(record, ","))
			if strings.Contains(headerStr, "conversation_id") {
				isMultiTurn = true
				continue // Skip header row
			} else if strings.Contains(headerStr, "query") ||
				strings.Contains(headerStr, "intent") ||
				strings.Contains(headerStr, "question") {
				continue // Skip header row
			}
		}

		if isMultiTurn {
			if len(record) >= 4 {
				// Format: conversation_id, turn_sequence, role, content
				role := strings.ToLower(strings.Trim(record[2], "\" "))
				content := strings.Trim(record[3], "\" ")

				if role == "user" {
					// temporarily store user content in Q (we will pair it with the next assistant turn)
					pairs = append(pairs, ChatPair{Q: content})
				} else if role == "assistant" && len(pairs) > 0 {
					// assign to the last user turn
					lastIdx := len(pairs) - 1
					if pairs[lastIdx].A == "" {
						pairs[lastIdx].A = content
						pairs[lastIdx].Intent = "social"
					}
				}
			}
		} else {
			if len(record) >= 2 {
				// conversing.csv columns: query, answer, intent, grammar
				q := strings.Trim(record[0], "\" ")

				// Strip evaluation harness prose so text-overlap matching works correctly
				qLower := strings.ToLower(q)
				prefixes := []string{"i welcome you with ", "i will now ask "}
				for _, p := range prefixes {
					if strings.HasPrefix(qLower, p) {
						q = q[len(p):]
						qLower = strings.ToLower(q)
					}
				}
				q = strings.TrimSpace(q)

				a := strings.Trim(record[1], "\" ")
				intent := ""
				if len(record) >= 3 {
					intent = strings.Trim(record[2], "\" ")
				}
				if q != "" && a != "" {
					pairs = append(pairs, ChatPair{Q: q, A: a, Intent: intent})
				}
			}
		}
	}

	// Filter out any unpaired multi-turn user queries
	var finalPairs []ChatPair
	for _, p := range pairs {
		if p.Q != "" && p.A != "" {
			finalPairs = append(finalPairs, p)
		}
	}

	c.ChatBank = append(c.ChatBank, finalPairs...)
	log.Printf("✅ Loaded %d prompts from %s (Total ChatBank: %d)", len(finalPairs), filepath.Base(path), len(c.ChatBank))
}

func (c *GollemerMoEClient) LoadDeviceIntentsBank(path string) {
	f, err := os.Open(path)
	if err != nil {
		log.Printf("⚠️  Failed to load device intents from %s: %v", filepath.Base(path), err)
		return
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.Comma = '\t'
	reader.FieldsPerRecord = -1
	reader.LazyQuotes = true
	records, err := reader.ReadAll()
	if err != nil {
		log.Printf("⚠️  Error parsing TSV from %s: %v", path, err)
		return
	}

	var pairs []ChatPair
	for i, record := range records {
		if i == 0 || len(record) < 5 {
			continue
		}

		input := record[0]
		intent := record[1]

		roles := make(map[string]string)
		if record[2] != "" {
			roles["action"] = record[2]
		}
		if record[3] != "" {
			roles["target"] = record[3]
		}
		if record[4] != "" {
			roles["device"] = record[4]
		}

		rolesJSON, _ := json.Marshal(roles)
		answerStr := fmt.Sprintf("[INTENT: %s] %s", intent, string(rolesJSON))
		pairs = append(pairs, ChatPair{Q: input, A: answerStr, Intent: intent})
	}

	c.ChatBank = append(c.ChatBank, pairs...)
	log.Printf("✅ Loaded %d device intents from %s (Total ChatBank: %d)", len(pairs), filepath.Base(path), len(c.ChatBank))
}

func (c *GollemerMoEClient) buildQueryContext(input string) string {
	return c.session().BuildContextStringWithUserInput(defaultMaxTokens, input)
}

func (c *GollemerMoEClient) RetrieveChatResponse(input string) (string, string, float64) {
	if len(c.ChatBank) == 0 {
		return "", "", 0
	}

	// If W2V is unavailable, fall back to text-based matching so ChatBank
	// still produces answers even without embeddings.
	// Treat the 6-word dummy W2V fallback the same as nil — it can't embed
	// conversational queries, so fall through to text-based matching instead.
	if c.W2V == nil || c.W2V.VocabSize < 50 {
		normalInput := strings.TrimSpace(strings.ToLower(input))
		inputWords := strings.Fields(normalInput)
		wordSet := make(map[string]bool, len(inputWords))
		for _, w := range inputWords {
			wordSet[w] = true
		}

		type candidate struct {
			answer string
			intent string
			score  float64
		}
		var candidates []candidate

		for _, pair := range c.ChatBank {
			pairQ := strings.TrimSpace(strings.ToLower(pair.Q))
			// Exact match — score 0.85 (capped, not 1.0)
			if pairQ == normalInput {
				candidates = append(candidates, candidate{pair.A, pair.Intent, 0.85})
				continue
			}
			// Word-overlap match
			pairWords := strings.Fields(pairQ)
			if len(pairWords) == 0 {
				continue
			}
			matches := 0
			for _, w := range pairWords {
				if wordSet[w] {
					matches++
				}
			}
			score := float64(matches) / float64(len(pairWords))
			if score >= 0.4 {
				// Cap overlap scores at 0.75 to signal we're estimating
				if score > 0.75 {
					score = 0.75
				}
				candidates = append(candidates, candidate{pair.A, pair.Intent, score})
			}
		}

		if len(candidates) == 0 {
			return "", "", 0
		}

		// Sort descending by score
		for i := 0; i < len(candidates)-1; i++ {
			for j := i + 1; j < len(candidates); j++ {
				if candidates[j].score > candidates[i].score {
					candidates[i], candidates[j] = candidates[j], candidates[i]
				}
			}
		}

		// Among top-scoring candidates (within 0.15 of best), pick one randomly
		// to avoid always copy-pasting the exact same answer.
		best := candidates[0]
		var topCands []candidate
		for _, c := range candidates {
			if best.score-c.score <= 0.15 && c.intent == best.intent {
				topCands = append(topCands, c)
			}
		}
		pick := topCands[rand.Intn(len(topCands))]
		return pick.answer, pick.intent, pick.score
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
	if c.W2V == nil {
		return nil
	}
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
	inputWords := strings.Fields(lowerInput)
	wordMap := make(map[string]bool)
	for _, w := range inputWords {
		// Clean punctuation from words for better matching
		w = strings.TrimFunc(w, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsNumber(r)
		})
		if w != "" {
			wordMap[w] = true
		}
	}

	// Social intent keywords (full words or common phrases)
	socialPhrases := []string{
		"how are you", "how you doing", "how's it going", "what's up",
		"tell me about", "what do you think", "do you ever", "have you ever",
	}

	for _, phrase := range socialPhrases {
		if strings.Contains(lowerInput, phrase) {
			return true
		}
	}

	socialKeywords := []string{
		"favorite", "like", "love", "hate", "enjoy", "think", "feel", "opinion",
		"holiday", "vacation", "weekend", "party", "friend", "family",
		"weather", "beautiful", "fun", "interesting", "amazing", "cool",
		"personal", "life", "work", "hobby", "passion", "dream",
		"meeting", "people", "connection", "relationship", "dating",
		"hope", "wish", "wonderful", "boring", "difficult",
		"hello", "hi", "hey", "goodbye", "bye", "thanks", "thank",
		"joke", "story", "real", "sleep", "tired", "happy", "sad", "name",
		"who", "what", "where", "how", "why", "can", "do", "you", "me",
		"gollemer", "ai", "assistant",
	}

	for _, kw := range socialKeywords {
		if wordMap[kw] {
			return true
		}
	}

	// Technical keywords (should NOT be treated as social)
	technicalKeywords := []string{
		"file", "handler", "project", "function", "class",
		"code", "program", "build", "deploy", "run", "webserver",
		"database", "sql", "api", "server", "client", "network",
		"import", "package", "module", "library", "framework",
		"struct", "interface", "channel", "routine", "pointer",
	}

	for _, tech := range technicalKeywords {
		if wordMap[tech] {
			return false
		}
	}

	// Default: if it's short and has no technical keywords, it might be social
	if len(inputWords) <= 3 && !strings.Contains(lowerInput, "list") {
		return true
	}

	return false
}

func (c *GollemerMoEClient) PredictIntent(input string) (string, float64) {
	c.lastMoEPrediction = "" // Clear previous turn's chat prediction
	lowerInput := strings.ToLower(input)

	// (Moved to after command heuristics)

	// --- 0. Instant Heuristics for Dynamic Queries ---
	if lowerInput == "what did i say" || lowerInput == "what did i say?" || lowerInput == "what was my last message" {
		if len(c.History) > 0 {
			lastQ := c.History[len(c.History)-1].Q
			c.lastMoEPrediction = fmt.Sprintf("You said: '%s'", lastQ)
			return "chat_response", 0.99
		}
		c.lastMoEPrediction = "I don't have any past conversation history to reference."
		return "chat_response", 0.99
	}
	if lowerInput == "what did you say" || lowerInput == "what did you say?" || lowerInput == "what was your last message" {
		if len(c.History) > 0 {
			lastA := c.History[len(c.History)-1].A
			c.lastMoEPrediction = fmt.Sprintf("I said: '%s'", lastA)
			return "chat_response", 0.99
		}
		c.lastMoEPrediction = "I haven't said anything yet."
		return "chat_response", 0.99
	}

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
	if strings.Contains(lowerInput, "what do you like to do") || strings.Contains(lowerInput, "what do you enjoy doing") ||
		strings.Contains(lowerInput, "what are your hobbies") || strings.Contains(lowerInput, "what are you into") {
		c.lastMoEPrediction = "I enjoy helping people build Go projects, debug code, and turn rough ideas into working software."
		return "chat_response", 0.99
	}
	// Fast-path status / small-talk queries — handle these before neural routing
	// so the model gives a sensible reply even while still training.
	if strings.Contains(lowerInput, "how are you") || strings.Contains(lowerInput, "how are you doing") ||
		strings.Contains(lowerInput, "how's it going") || strings.Contains(lowerInput, "how do you do") ||
		strings.Contains(lowerInput, "how have you been") || lowerInput == "how r u" || lowerInput == "hru" {
		c.lastMoEPrediction = "I am doing well, thank you for asking! How can I help you today?"
		return "chat_response", 0.99
	}
	if strings.Contains(lowerInput, "good morning") || strings.Contains(lowerInput, "good evening") ||
		strings.Contains(lowerInput, "good afternoon") || strings.Contains(lowerInput, "good night") {
		c.lastMoEPrediction = "Hello! Good to hear from you. What can I help you with?"
		return "chat_response", 0.99
	}
	if strings.Contains(lowerInput, "tell me a joke") || strings.Contains(lowerInput, "tell me joke") {
		c.lastMoEPrediction = "Why did the gopher cross the road? To get to the other cluster!"
		return "chat_response", 0.99
	}
	if lowerInput == "what can you do" || lowerInput == "what do you do" || strings.Contains(lowerInput, "what are you capable") ||
		strings.Contains(lowerInput, "how can you help") || strings.Contains(lowerInput, "how can you assist") ||
		strings.Contains(lowerInput, "what can you help") || strings.Contains(lowerInput, "how do you help") {
		c.lastMoEPrediction = "I can chat, create Go webservers, edit files, fix syntax errors, and answer questions about your project!"
		return "chat_response", 0.99
	}
	if strings.Contains(lowerInput, "what do you like to do") || strings.Contains(lowerInput, "what do you enjoy doing") ||
		strings.Contains(lowerInput, "what are your hobbies") || strings.Contains(lowerInput, "what are you into") {
		c.lastMoEPrediction = "I enjoy helping people build Go projects, debug code, and turn rough ideas into working software."
		return "chat_response", 0.99
	}
	if strings.Contains(lowerInput, "are you a human") || strings.Contains(lowerInput, "are you real") || strings.Contains(lowerInput, "are you an ai") {
		c.lastMoEPrediction = "No, I am Gollemer — an AI assistant built in Go. I am here to help you build and code!"
		return "chat_response", 0.99
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

	// --- 1.5 Conversational (Social) Routing ---
	// If it's not a command, we want to chat! Pass it to the conversational MoE.
	if c.SocialModel != nil {
		response := c.GenerateSocialResponse(input)
		if response != "" {
			c.lastMoEPrediction = response
			log.Printf("🧠 Neural Social Match: Using weights from moe_social_model.gob")
			return "social_chat", 0.95
		}
		log.Printf("⚖️  Quality Gate: Social model output was too high-entropy (word salad); trying retrieval fallback.")
		if len(c.ChatBank) > 0 {
			retrievedResp, retrievedIntent, retrievedScore := c.RetrieveChatResponse(input)
			if retrievedScore > 0.35 && retrievedResp != "" {
				log.Printf("✅ Retrieval Fallback: score=%.4f intent=%s", retrievedScore, retrievedIntent)
				c.lastMoEPrediction = paraphraseResponse(retrievedResp)
				return "social_chat", retrievedScore
			}
		}
	} else {
		// No conversational model loaded — try retrieval directly
		if len(c.ChatBank) > 0 {
			retrievedResp, retrievedIntent, retrievedScore := c.RetrieveChatResponse(input)
			// Higher threshold since we don't have neural confidence
			if retrievedScore > 0.7 && retrievedResp != "" {
				log.Printf("✅ Social Retrieval (no neural model): score=%.4f intent=%s", retrievedScore, retrievedIntent)
				c.lastMoEPrediction = retrievedResp
				if retrievedIntent != "" {
					return retrievedIntent, retrievedScore
				}
				return "social_chat", retrievedScore
			}
		}
	}

	// --- 1. Combined Retrieval & Neural Logic ---
	retrievedResp, retrievedIntent, retrievedScore := c.RetrieveChatResponse(input)
	log.Printf("🔍 Intent Retrieval Top Score: %.4f (%s)", retrievedScore, retrievedIntent)

	var neuralResponse string
	var neuralIntent string
	var neuralScore float64

	if c.Model != nil && c.W2V != nil {
		formattedInput := fmt.Sprintf("__intent__ social : __ques__ %s __ans__", lowerInput)
		cleanWords := cleanTokenize(formattedInput)
		var tokenIDs []int
		var activeVocab *mainvocab.Vocabulary
		if c.Model.SocialVocab != nil {
			activeVocab = c.Model.SocialVocab
		} else {
			activeVocab = c.Model.SentenceVocab
		}

		for _, w := range cleanWords {
			if activeVocab != nil {
				tokenIDs = append(tokenIDs, lookupVocab(w, activeVocab))
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
					suppressedIDs := make(map[int]bool)
					techKeywords := []string{
						"func", "struct", "type", "interface", "package", "import", "chan", "goroutine",
						"fmt", "log", "http", "json", "os", "io", "string", "int", "bool", "float64",
						"err", "nil", "return", "var", "const", "for", "if", "else", "switch", "case",
						"devops", "kubernetes", "docker", "server", "endpoint", "api", "database", "sql",
						"query", "pointer", "memory", "cpu", "ram", "network", "tcp", "udp", "socket",
						"session", "certificate", "tag", "ui", "status", "traffic", "receipt", "opinion",
						"ADV", "AUX", "PRON", "VERB", "NOUN", "PREP", "ADJ", "CONJ", "DET", "INTJ", "PROPN",
						"__ans__", "__ques__", "__intent__", "social:", "code:",
						"{", "}", "(", ")", "[", "]", ":=", "==", "!=", "<=", ">=", "&&", "||", "++", "--",
					}
					for _, kw := range techKeywords {
						id := activeVocab.GetTokenID(kw)
						if id > 0 {
							suppressedIDs[id] = true
						}
					}

					outputIDs, err := c.Model.GreedySearchDecodeWithTemp(
						contextVector, 20,
						c.Model.SentenceVocab.BosID, c.Model.SentenceVocab.EosID,
						0.4, 1.2, 0.3, 50, taggedData, suppressedIDs,
					)
					if err == nil && len(outputIDs) > 0 {
						var decodedWords []string
						for _, id := range outputIDs {
							w := activeVocab.TokenToWord[id]
							if w != "</s>" && w != "<s>" && w != "<pad>" && w != "UNK" && w != "" {
								decodedWords = append(decodedWords, w)
							}
						}
						neuralResponse = strings.Join(decodedWords, " ")
						if neuralResponse != "" {
							log.Printf("🧠 Neural Model generated: %s", neuralResponse)

							// Dynamic Intent Parsing!
							// If the model learned to guess the intent, it will output [INTENT: xxx]
							if strings.HasPrefix(neuralResponse, "[INTENT: ") {
								endIdx := strings.Index(neuralResponse, "]")
								if endIdx > 9 {
									neuralIntent = neuralResponse[9:endIdx]
									neuralResponse = strings.TrimSpace(neuralResponse[endIdx+1:])
									neuralScore = 0.99
								}
							}

							if neuralIntent == "" {
								if false {
									// logic for retrieving response if needed
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
	}

	// --- 2. Winner Selection ---
	if neuralIntent != "" && neuralIntent != "chat_response" {
		return neuralIntent, neuralScore
	}

	const retrievalThreshold = 0.96

	// Parse embedded intent from retrieved response if present
	if strings.HasPrefix(retrievedResp, "[INTENT: ") {
		endIdx := strings.Index(retrievedResp, "]")
		if endIdx > 9 {
			// Override retrieved intent with embedded one to ensure consistency
			retrievedIntent = retrievedResp[9:endIdx]
			retrievedResp = strings.TrimSpace(retrievedResp[endIdx+1:])
		}
	}

	if retrievedScore >= retrievalThreshold {
		c.lastMoEPrediction = retrievedResp
		return retrievedIntent, retrievedScore
	}

	if neuralResponse != "" {
		c.lastMoEPrediction = neuralResponse
		return neuralIntent, neuralScore
	}

	// Dynamic Command Intent Guessing
	// If the retrieved intent is a known command (not chat_response) and we have decent
	// confidence (> 0.45), trust the fuzzy intent guess! This is what ties NLP to commands.
	if retrievedScore > 0.45 && retrievedIntent != "chat_response" && retrievedIntent != "social_chat" && retrievedIntent != "social" && retrievedIntent != "" {
		c.lastMoEPrediction = retrievedResp
		return retrievedIntent, retrievedScore
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
		c.lastMoEPrediction = "what did you say?"
		return "chat_response", 0.05
	}

	return "", 0.0
}

// lookupChatBank does a direct string-based lookup against the loaded training pairs.
// It does NOT require W2V embeddings — it works on raw text similarity.
// Returns the best matching answer and a confidence score [0,1].
func (c *GollemerMoEClient) lookupChatBank(input, intent string) (string, float64) {
	if len(c.ChatBank) == 0 {
		return "", 0
	}
	normalInput := strings.TrimSpace(strings.ToLower(input))

	// Pass 1: exact match on question
	for _, pair := range c.ChatBank {
		if strings.TrimSpace(strings.ToLower(pair.Q)) == normalInput {
			return pair.A, 1.0
		}
	}

	// Pass 2: best word-overlap match within the same intent family
	inputWords := strings.Fields(normalInput)
	wordSet := make(map[string]bool, len(inputWords))
	for _, w := range inputWords {
		wordSet[w] = true
	}

	bestScore := 0.0
	bestAnswer := ""
	for _, pair := range c.ChatBank {
		// Prefer same-intent rows but don't exclude cross-intent
		intentBonus := 0.0
		if strings.EqualFold(pair.Intent, intent) {
			intentBonus = 0.25
		}
		pairWords := strings.Fields(strings.ToLower(pair.Q))
		matches := 0
		for _, w := range pairWords {
			if wordSet[w] {
				matches++
			}
		}
		if len(pairWords) == 0 {
			continue
		}
		score := float64(matches)/float64(len(pairWords)) + intentBonus
		if score > bestScore {
			bestScore = score
			bestAnswer = pair.A
		}
	}
	if bestScore >= 0.35 {
		return bestAnswer, bestScore
	}
	return "", 0
}

// supervisorCompleteSentenceGuided uses the grammar skeleton for the given intent to
// guide a greedy decode pass, strongly boosting words that match the expected
// POS type at each position. It also applies a 'guidance boost' for tokens
// present in a target answer (e.g. from ChatBank) to nudge the model toward
// proven linguistic patterns.
func (c *GollemerMoEClient) supervisorCompleteSentenceGuided(ctx *tensor.Tensor, intent string, model *moe.IntentMoE, guidanceIDs map[int]bool, boost float32, suppressedIDs map[int]bool) string {
	if model.SentenceVocab == nil || model.Decoder == nil {
		return ""
	}

	// Resolve grammar skeleton for this intent
	parent := "social"
	child := intent
	rule, hasRule := model.Rules.GetRuleByIntent(parent, child)

	// Prepare initial hidden state from encoder context
	batchSize := 1
	hiddenSize := model.Decoder.LSTM.HiddenSize
	initHidden, err := ctx.Mean(1)
	if err != nil {
		return ""
	}
	initHidden, _ = initHidden.Reshape([]int{batchSize, ctx.Shape[2]})
	if initHidden.Shape[1] > hiddenSize {
		initHidden, _ = initHidden.Slice(1, 0, hiddenSize)
	} else if initHidden.Shape[1] < hiddenSize {
		pad := tensor.NewTensor([]int{batchSize, hiddenSize - initHidden.Shape[1]}, make([]float32, batchSize*(hiddenSize-initHidden.Shape[1])), false)
		initHidden, _ = tensor.Concat([]*tensor.Tensor{initHidden, pad}, 1)
	}
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float32, batchSize*hiddenSize), false)

	hidden := initHidden
	cell := cellState
	currentID := model.SentenceVocab.BosID
	unkID := model.SentenceVocab.GetTokenID("UNK")

	var generated []string
	var decodedIDs []int

	for step := 0; step < 20; step++ {
		inputT := tensor.NewTensor([]int{1, 1}, []float32{float32(currentID)}, false)
		logits, nextH, nextC, _, _, decErr := model.Decoder.DecodeStepWithExpert(inputT, hidden, cell, ctx, step)
		if decErr != nil {
			break
		}
		hidden = nextH
		cell = nextC
		logits.ToCPU()

		// Suppress garbage tokens
		logits.Data[model.SentenceVocab.PaddingTokenID] = -1e9
		if unkID != -1 {
			logits.Data[unkID] = -1e9
		}
		logits.Data[model.SentenceVocab.BosID] = -1e9
		if step < 4 && model.SentenceVocab.EosID >= 0 && model.SentenceVocab.EosID < len(logits.Data) {
			logits.Data[model.SentenceVocab.EosID] = -1e9
		}

		// Suppress evaluation harness and special routing tokens
		specialTokens := []string{"__intent__", "__ques__", "__ans__", "social", ":"}
		for _, st := range specialTokens {
			id := model.SentenceVocab.GetTokenID(st)
			if id != -1 && id < len(logits.Data) {
				logits.Data[id] = -1e9
			}
		}

		// 🚫 Social-context technical vocabulary suppression
		// Prevent the decoder from ever emitting DevOps / code jargon in a social response.
		for id := range suppressedIDs {
			if id < len(logits.Data) {
				logits.Data[id] = -1e9
			}
		}

		// 🧭 Guidance boost: favour tokens from the ChatBank target answer
		if len(guidanceIDs) > 0 {
			for id := range guidanceIDs {
				if id < len(logits.Data) {
					logits.Data[id] += boost
				}
			}
		}

		// 🧬 Grammar skeleton enforcement — strong boost for matching POS type
		if hasRule && step < len(rule.GrammarSkeleton) {
			expectedType := rule.GrammarSkeleton[step]
			for idx := range logits.Data {
				if logits.Data[idx] < -1e8 {
					continue
				}
				word := model.SentenceVocab.GetWord(idx)
				actualType := moe.MapWordToGrammarType(word)
				if expectedType != "OTHER" && actualType == expectedType {
					logits.Data[idx] += 6.0 // Strong pull toward correct POS
				} else if expectedType != "OTHER" && actualType != "OTHER" {
					// Only penalize a mismatch when the skeleton specifies a concrete type.
					// If expectedType == "OTHER" (i.e. skeleton is flexible at this position),
					// we do not penalize anything — the model is free to pick any token.
					logits.Data[idx] -= 4.0 // Penalize wrong POS
				}
			}
			// Required keyword boost: if a required keyword exists in the vocab, give it a large bonus
			for _, kw := range rule.RequiredKeywords {
				kwID := model.SentenceVocab.GetTokenID(kw)
				if kwID > 1 && kwID < len(logits.Data) {
					logits.Data[kwID] += 4.0
				}
			}
		}

		// Repetition penalty
		moe.ApplyRepetitionPenalty(logits, decodedIDs, 2.5)
		if len(decodedIDs) >= 1 {
			lastID := decodedIDs[len(decodedIDs)-1]
			if lastID < len(logits.Data) {
				logits.Data[lastID] -= 3.0
			}
		}

		// Use sampling instead of greedy to allow model logic to break ties between boosted tokens
		temp := float32(0.8)
		topK := 5
		topP := float32(0.85)
		if c.SocialConfig != nil {
			sc := c.SocialConfig.Get()
			if sc.RouterTemperature > 0 {
				temp = sc.RouterTemperature
			}
			if sc.TopK > 0 {
				topK = sc.TopK
			}
			if sc.TopP > 0 {
				topP = sc.TopP
			}
		}

		bestID, err := moe.SampleFromLogits(logits, temp, topK, topP)
		if err != nil {
			break
		}
		if bestID == model.SentenceVocab.EosID {
			break
		}
		word := model.SentenceVocab.GetWord(bestID)
		if word == "" || word == "<pad>" || word == "UNK" {
			continue
		}
		generated = append(generated, word)
		decodedIDs = append(decodedIDs, bestID)
		currentID = bestID
	}

	if len(generated) == 0 {
		return ""
	}
	return strings.Join(generated, " ")
}

// resolveContextQuery handles memory and pronoun queries by scanning c.History for
// the specific entity (file, folder, object) the user is referring to and composing
// a precise answer. Returns "" if the query is not a context question.
func (c *GollemerMoEClient) resolveContextQuery(lowerInput string) string {
	if len(c.History) == 0 {
		return ""
	}

	// --- Classify what the user is asking about ---
	isFileQ := strings.Contains(lowerInput, "file") || strings.Contains(lowerInput, "files")
	isFolderQ := strings.Contains(lowerInput, "folder") || strings.Contains(lowerInput, "directory") || strings.Contains(lowerInput, "folders")
	isObjectQ := strings.Contains(lowerInput, "it") || strings.Contains(lowerInput, "that") || strings.Contains(lowerInput, "what was") || strings.Contains(lowerInput, "what is")
	isWhereQ := strings.HasPrefix(lowerInput, "where")
	isWhatQ := strings.HasPrefix(lowerInput, "what")
	isRememberQ := strings.Contains(lowerInput, "remember") || strings.Contains(lowerInput, "recall") ||
		strings.Contains(lowerInput, "do you know") || strings.Contains(lowerInput, "do you have")
	isHistoryQ := strings.Contains(lowerInput, "we talking") || strings.Contains(lowerInput, "we talk") ||
		strings.Contains(lowerInput, "were we") || strings.Contains(lowerInput, "did we") ||
		strings.Contains(lowerInput, "last time") || strings.Contains(lowerInput, "conversation")

	isContextQ := (isFileQ || isFolderQ || isObjectQ) && (isWhatQ || isWhereQ || isRememberQ || isHistoryQ)
	if !isContextQ {
		return ""
	}

	// --- Extract entities from history ---
	var files []string
	var folders []string
	seenFiles := map[string]bool{}
	seenFolders := map[string]bool{}

	for _, pair := range c.History {
		for _, text := range []string{pair.Q, pair.A} {
			words := strings.Fields(strings.ToLower(text))
			for wi, w := range words {
				w = strings.Trim(w, "',\".")
				// Files: ends with known extension
				if (strings.HasSuffix(w, ".go") || strings.HasSuffix(w, ".json") ||
					strings.HasSuffix(w, ".txt") || strings.HasSuffix(w, ".md") ||
					strings.HasSuffix(w, ".csv") || strings.HasSuffix(w, ".js") ||
					strings.HasSuffix(w, ".ts") || strings.HasSuffix(w, ".py") ||
					strings.HasSuffix(w, ".html") || strings.HasSuffix(w, ".css")) &&
					!seenFiles[w] {
					files = append(files, w)
					seenFiles[w] = true
				}
				// Folders: word after "folder" or "directory" or "into"
				if (w == "folder" || w == "directory" || w == "into" || w == "to") && wi+1 < len(words) {
					candidate := strings.Trim(words[wi+1], "',\".")
					if candidate != "" && candidate != "the" && candidate != "a" && !seenFolders[candidate] {
						folders = append(folders, candidate)
						seenFolders[candidate] = true
					}
				}
			}
		}
	}

	// --- Compose the answer ---
	if isFileQ && !isFolderQ {
		switch len(files) {
		case 0:
			return "I don't see any specific file mentioned in our recent conversation."
		case 1:
			if isWhereQ {
				return fmt.Sprintf("The file '%s' was mentioned in our conversation. You can check its location with 'list' or 'tree'.", files[0])
			}
			return fmt.Sprintf("The file we were talking about was '%s'.", files[0])
		default:
			return fmt.Sprintf("We've mentioned these files: %s.", strings.Join(files, ", "))
		}
	}

	if isFolderQ && !isFileQ {
		switch len(folders) {
		case 0:
			return "I don't see any specific folder mentioned in our recent conversation."
		case 1:
			return fmt.Sprintf("The folder we were working with was '%s'.", folders[0])
		default:
			return fmt.Sprintf("We've mentioned these folders: %s.", strings.Join(folders, ", "))
		}
	}

	// Both files and folders — give a combined answer
	if (isFileQ || isFolderQ) && (len(files) > 0 || len(folders) > 0) {
		parts := []string{}
		if len(files) > 0 {
			parts = append(parts, fmt.Sprintf("files: %s", strings.Join(files, ", ")))
		}
		if len(folders) > 0 {
			parts = append(parts, fmt.Sprintf("folders: %s", strings.Join(folders, ", ")))
		}
		return "In our recent conversation we talked about " + strings.Join(parts, " and ") + "."
	}

	return ""
}

func (c *GollemerMoEClient) FormatChatMLPrompt(userInput string) string {
	return "<|im_start|>system\nYou are Gollemer, an expert AI Go development assistant.<|im_end|>\n" +
		"<|im_start|>user\n" + userInput + "<|im_end|>\n" +
		"<|im_start|>assistant\n"
}

func (c *GollemerMoEClient) GenerateSocialResponse(input string) string {
	log.Printf("📡 GenerateSocialResponse called")
	if c.SocialModel == nil || c.SocialModel.SentenceVocab == nil {
		return ""
	}

	// Tokenize using the isolated social vocabulary
	vocab := c.SocialModel.SocialVocab
	if vocab == nil {
		vocab = c.SocialModel.SentenceVocab
	}
	words := strings.Fields(strings.ToLower(input))
	var tokenIDs []int
	for _, w := range words {
		id := vocab.GetTokenID(w)
		if id < 0 {
			id = vocab.UnkID
			if id < 0 {
				id = 1 // fallback UNK
			}
		}
		tokenIDs = append(tokenIDs, id)
	}
	if len(tokenIDs) == 0 {
		return ""
	}

	var generatedIDs []int
	currentSequence := append([]int(nil), tokenIDs...)

	eosID := vocab.EosID

	maxNewTokens := 128

	suppressedIDs := make(map[int]bool)
	techKeywords := []string{
		"func", "struct", "type", "interface", "package", "import", "chan", "goroutine",
		"fmt", "log", "http", "json", "os", "io", "string", "int", "bool", "float64",
		"err", "nil", "return", "var", "const", "for", "if", "else", "switch", "case",
		"devops", "kubernetes", "docker", "server", "endpoint", "api", "database", "sql",
		"query", "pointer", "memory", "cpu", "ram", "network", "tcp", "udp", "socket",
		"session", "certificate", "tag", "ui", "status", "traffic", "receipt", "opinion",
		"ADV", "AUX", "PRON", "VERB", "NOUN", "PREP", "ADJ", "CONJ", "DET", "INTJ", "PROPN",
		"__ans__", "__ques__", "__intent__", "social:", "code:",
		"{", "}", "(", ")", "[", "]", ":=", "==", "!=", "<=", ">=", "&&", "||", "++", "--",
	}
	for _, kw := range techKeywords {
		id := vocab.GetTokenID(kw)
		if id > 0 {
			suppressedIDs[id] = true
		}
	}

	for i := 0; i < maxNewTokens; i++ {
		nextTokenID := c.SocialModel.PredictNextToken(currentSequence, suppressedIDs)

		// Stop immediately if the model predicts EOS or 0
		if nextTokenID == eosID || nextTokenID == 0 {
			break
		}

		generatedIDs = append(generatedIDs, nextTokenID)
		currentSequence = append(currentSequence, nextTokenID)
	}

	// Decode using the social model's vocabulary
	var wordsOut []string
	for _, id := range generatedIDs {
		w := vocab.GetWord(id)
		if w == "" || w == "UNK" || w == "<pad>" || w == "<s>" || w == "</s>" {
			continue
		}
		wordsOut = append(wordsOut, w)
	}
	responseText := strings.Join(wordsOut, " ")
	log.Printf("🎭 Neural Social Model generated: '%s'", responseText)
	return responseText
}

func (c *GollemerMoEClient) checkCommandHeuristics(lowerInput string) (string, float64) {
	// Fix / edit commands must come BEFORE create verb matching so phrases
	// like "add } to jim.go" aren't caught by the "add" create verb handler.
	// Any "add X to file.go" / "add X to jim" pattern routes to fix_query; the
	// fix_query handler in commands.go then decides between go_edit_agent
	// (semantic edits: struct/field/func/type) and GoFixer (syntax patches)
	// via its fallbackEdit keyword routing.
	if strings.HasPrefix(lowerInput, "fix ") || strings.HasPrefix(lowerInput, "edit ") ||
		strings.HasPrefix(lowerInput, "update ") || strings.HasPrefix(lowerInput, "patch ") ||
		strings.HasPrefix(lowerInput, "add }") || strings.HasPrefix(lowerInput, "add missing") ||
		strings.HasPrefix(lowerInput, "add the") || strings.HasPrefix(lowerInput, "add fmt") ||
		strings.HasPrefix(lowerInput, "add func") || strings.HasPrefix(lowerInput, "add line") ||
		strings.HasPrefix(lowerInput, "remove the") || strings.HasPrefix(lowerInput, "delete the") ||
		strings.Contains(lowerInput, " add ") || strings.Contains(lowerInput, "missing ") ||
		(strings.HasPrefix(lowerInput, "add ") && (strings.Contains(lowerInput, ".go") || strings.Contains(lowerInput, "jim"))) {
		return "fix_query", 0.95
	}

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
	if strings.HasPrefix(lowerInput, "move ") || lowerInput == "move" || strings.HasPrefix(lowerInput, "can you move ") || strings.HasPrefix(lowerInput, "could you move ") || strings.HasPrefix(lowerInput, "take ") || strings.HasPrefix(lowerInput, "bring ") {
		return "move_query", 0.99
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
	if strings.Contains(lowerInput, "what do you like to do") || strings.Contains(lowerInput, "what do you enjoy doing") ||
		strings.Contains(lowerInput, "what are your hobbies") || strings.Contains(lowerInput, "what are you into") {
		c.lastMoEPrediction = "I enjoy helping people build Go projects, debug code, and turn rough ideas into working software."
		return "chat_response", 0.99
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

	// For device intents, c.lastMoEPrediction might hold the JSON roles (the [INTENT] prefix is stripped earlier).
	lastPred := strings.TrimSpace(c.lastMoEPrediction)
	if strings.HasPrefix(lastPred, "{") && strings.HasSuffix(lastPred, "}") {
		var roles map[string]string
		if err := json.Unmarshal([]byte(lastPred), &roles); err == nil {
			for k, v := range roles {
				entities[k] = v
			}
		}
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

func (c *GollemerMoEClient) ResetSocialRouter(m *ui.Mascot) {
	if c.SocialModel == nil {
		m.Say(ui.Neutral, "No social model loaded to reset.")
		return
	}
	m.Say(ui.Think, "Resetting social model router weights to break expert monopoly... ⚡")

	layers := c.SocialModel.Encoder.GetMoELayers()
	if c.SocialModel.Decoder.OutputMoE != nil {
		layers = append(layers, c.SocialModel.Decoder.OutputMoE)
	}

	for _, layer := range layers {
		if layer.GatingNetwork != nil && layer.GatingNetwork.Linear != nil {
			// Re-initialize weights with small random values
			data := layer.GatingNetwork.Linear.Weights.Data
			for i := range data {
				data[i] = (rand.Float32() - 0.5) * 0.1
			}
			if layer.GatingNetwork.Linear.Biases != nil {
				for i := range layer.GatingNetwork.Linear.Biases.Data {
					layer.GatingNetwork.Linear.Biases.Data[i] = 0
				}
			}
		}
	}
	m.Say(ui.Happy, "Router weights reset. The model will now explore other experts during the next training/chat session.")
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
