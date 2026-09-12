package chat

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/memory"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/training/makefile"
)

type DialogRole string

const (
	RoleUser      DialogRole = "user"
	RoleAssistant DialogRole = "assistant"
)

// DialogTurn is a single turn (one speaker's message) in a multi-turn conversation.
type DialogTurn struct {
	Role    DialogRole
	Content string
}

type ConversationSample struct {
	Dialogue []DialogTurn
}

type ChatSession struct {
	History       []ConversationTurn
	MaxHistory    int // Number of exchanges to remember
	ContextVector []float32
	mu            sync.Mutex
}

func NewChatSession(maxHistory int, vectorSize int) *ChatSession {
	return &ChatSession{
		History:       make([]ConversationTurn, 0),
		MaxHistory:    maxHistory,
		ContextVector: make([]float32, vectorSize),
	}
}

func (s *ChatSession) AddToHistory(turn ConversationTurn) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.History) >= s.MaxHistory {
		s.History = s.History[1:] // Slide the window
	}
	s.History = append(s.History, turn)

	s.updateContextVector()
}

func (s *ChatSession) updateContextVector() {
	if len(s.History) == 0 {
		return
	}

	// Simple weighted average: newer turns have more weight.
	for i := range s.ContextVector {
		s.ContextVector[i] = 0
	}

	var totalWeight float32 = 0.0
	for i, turn := range s.History {
		weight := float32(i + 1) // Simple linear weight
		for j, val := range turn.Input {
			if j < len(s.ContextVector) {
				s.ContextVector[j] += val * weight
			}
		}
		totalWeight += weight
	}

	if totalWeight > 0 {
		for i := range s.ContextVector {
			s.ContextVector[i] /= totalWeight
		}
	}
}

func (s *ChatSession) GetContextVector() []float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	ctxCopy := make([]float32, len(s.ContextVector))
	copy(ctxCopy, s.ContextVector)
	return ctxCopy
}

// GetLastBotTurn returns the bot's most recent response text (lower-cased) for
// context carry-over in intent classification. Returns "" if no history yet.
func (s *ChatSession) GetLastBotTurn() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.History) == 0 {
		return ""
	}
	return strings.ToLower(s.History[len(s.History)-1].Response)
}

func StartChat(model *moe.IntentMoE) {

	session := NewChatSession(3, model.Embedding.DimModel)
	// 1. Define the "Core Identity"
	// Keep it short so it doesn't eat up the RNN's memory (hidden state)
	const systemPrompt = "System: You are a friendly, helpful assistant. Tone: Kind."

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("\n---  MoE Chatbot (Stateful Memory Enabled) ---")

	for {
		fmt.Print("\nYou: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" {
			break
		}

		// NEW: Inject contextual clues for verbose output.
		injectContextualClues(session)

		// TODO: Implement full prompt chaining.
		// A full implementation would involve parsing the current input to identify
		// the intent and any missing entities. If entities are missing (e.g., user says
		// "create a file" without a name), and the input is a continuation ("for it", "do it"),
		// the system would look at `session.History` for the last relevant entity and
		// inject it into the current command's context before execution. This requires
		// a dialogue manager and an integrated NER component.

		// Sentiment Analysis & Emotional Steering
		sentiment := GetSentimentScore(input)
		isApologetic := false
		if sentiment < -0.5 {
			isApologetic = true
			// fmt.Println(" [System Note: Bot is in 'Apologetic Mode']")
			// for _, layer := range moe.ActiveLayers {
			// 	// Manually add a bias to the router's logits for Expert 7
			// 	// This makes it 5x more likely to be chosen for this specific turn
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] += 2.0
			// 	}
			// }
		}

		// 1. Tokenize and embed current input
		tokens := cleanTokenize(input)
		ids := make([]float32, len(tokens))
		avgInputEmbedding := make([]float32, model.Embedding.DimModel)
		tokenCount := 0
		for i, t := range tokens {
			id := lookupVocab(t, model.SentenceVocab)
			ids[i] = float32(id)
			// Get actual embedding from the model's weights for the history history
			if id >= 0 && id < model.Embedding.VocabSize {
				start := id * model.Embedding.DimModel
				vec := model.Embedding.Weight.Data[start : start+model.Embedding.DimModel]
				for d := 0; d < model.Embedding.DimModel; d++ {
					avgInputEmbedding[d] += vec[d]
				}
				tokenCount++
			}
		}
		if tokenCount > 0 {
			for d := range avgInputEmbedding {
				avgInputEmbedding[d] /= float32(tokenCount)
			}
		}

		// 2. Combine with context vector
		contextVector := session.GetContextVector()
		const lambda = 0.3 // Context decay factor
		if len(contextVector) == model.Embedding.DimModel {
			for i := 0; i < len(ids); i++ {
				// This is a conceptual change. The actual implementation
				// would modify the embedding tensor, not the IDs.
				// This logic is now handled in the Reply/StreamReply methods.
			}
		}

		// 3. Standard Inference
		inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

		// Inference (Eval Mode)
		for _, l := range moe.ActiveLayers {
			l.SetMode(false)
		}

		emb, _ := model.Embedding.Forward(inputT)
		ctx, _ := model.Encoder.Forward(emb)

		// 4. Beam Search Decoding
		// We use BeamSize 5, MaxLen 50, and Repetition Penalty 1.2
		outIDs := BeamSearchDecodeFiltered(model, ctx, 5, 50, []int{model.SentenceVocab.GetTokenID("UNK")})

		// 5. Convert IDs back to Words
		var response []string
		for _, id := range outIDs {
			word := model.SentenceVocab.GetWord(id)
			if word != "<s>" && word != "</s>" && word != "<pad>" {
				response = append(response, word)
			}
		}
		botResponse := strings.Join(response, " ")

		// 6. Print Routing Insight
		fmt.Printf("Bot [%s]: %s\n", getExpertPath(), FormatUserOutput(botResponse))

		// 7. Save this turn to memory
		newTurn := ConversationTurn{
			Input:    avgInputEmbedding,
			RawInput: input,                   // Save original input
			Intent:   "chat_response",         // Placeholder, would be resolved by classifier
			Entities: make(map[string]string), // Placeholder
			Response: botResponse,
		}
		session.AddToHistory(newTurn)

		// Reset Emotional Steering
		if isApologetic {
			// for _, layer := range moe.ActiveLayers {
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] -= 2.0
			// 	}
			// }
		}

		// Cleanup memory for the next turn
		model.Detach()
	}
}

type RetrievalPair struct {
	Q string
	A string
}

type MoEChatBot struct {
	model           *moe.IntentMoE
	session         *ChatSession
	systemPrompt    string
	vectorDB        *memory.VectorDB
	retrievalPairs  []RetrievalPair // fallback retrieval when neural output is incoherent
	makefileTargets []makefile.MakeTarget
}

func NewMoEChatBot(model *moe.IntentMoE) *MoEChatBot {
	bot := &MoEChatBot{
		model:        model,
		session:      NewChatSession(5, model.Embedding.DimModel),
		systemPrompt: "System: You are a friendly, helpful assistant. Tone: Kind.",
	}
	bot.loadRetrievalPairs()
	bot.loadMakefileTargets()
	return bot
}

// loadMakefileTargets loads makefile.yaml targets for command prediction.
func (b *MoEChatBot) loadMakefileTargets() {
	targets, err := makefile.ParseMakefile(makefile.ParseOptions{MakefilePath: filepath.Join(".", "Makefile")})
	if err != nil {
		log.Printf("[CHAT] Makefile target loader: %v", err)
		return
	}
	b.makefileTargets = targets
	log.Printf("[CHAT] Loaded %d makefile targets for command prediction", len(targets))
}

// loadRetrievalPairs reads social_replies.yaml and the .pb pairs for instant retrieval.
func (b *MoEChatBot) loadRetrievalPairs() {
	yamlPath := filepath.Join(".", "data", "training", "trainingdata", "social_replies.yaml")
	data, err := os.ReadFile(yamlPath)
	if err != nil {
		log.Printf("[CHAT] Retrieval fallback: could not load social_replies.yaml: %v", err)
		return
	}
	// Parse the YAML structure:
	// - role: "user" -> next line is content: "..."
	// - role: "assistant" -> next line is content: "..."
	lines := strings.Split(string(data), "\n")
	var currentQ, currentA string
	var nextIsUserContent, nextIsAssistantContent bool

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		if strings.HasPrefix(trimmed, `role: "user"`) {
			nextIsUserContent = true
		} else if strings.HasPrefix(trimmed, `role: "assistant"`) {
			nextIsAssistantContent = true
		} else if strings.HasPrefix(trimmed, "content: ") {
			contentStr := strings.TrimSpace(strings.TrimPrefix(trimmed, "content: "))
			contentStr = strings.Trim(contentStr, `"`)

			if nextIsUserContent {
				currentQ = contentStr
				nextIsUserContent = false
			} else if nextIsAssistantContent {
				currentA = contentStr
				nextIsAssistantContent = false

				if currentQ != "" && currentA != "" {
					b.retrievalPairs = append(b.retrievalPairs, RetrievalPair{Q: currentQ, A: currentA})
					currentQ, currentA = "", ""
				}
			}
		}
	}
	log.Printf("[CHAT] Retrieval fallback loaded %d pairs from social_replies.yaml", len(b.retrievalPairs))
}

// retrievalLookup finds the best matching answer using Jaccard word-overlap similarity.
func (b *MoEChatBot) retrievalLookup(query string) (string, float32) {
	if len(b.retrievalPairs) == 0 {
		return "", 0
	}
	qWords := make(map[string]bool)
	for _, w := range strings.Fields(strings.ToLower(query)) {
		if len(w) > 2 { // skip stop words
			qWords[w] = true
		}
	}
	var bestScore float32
	var bestAnswer string
	for _, pair := range b.retrievalPairs {
		pWords := make(map[string]bool)
		for _, w := range strings.Fields(strings.ToLower(pair.Q)) {
			if len(w) > 2 {
				pWords[w] = true
			}
		}
		// Jaccard similarity
		var intersection, union float32
		for w := range qWords {
			if pWords[w] {
				intersection++
			}
		}
		for w := range pWords {
			qWords[w] = true // temp merge
		}
		union = float32(len(qWords))
		// restore qWords
		for w := range pWords {
			if !qWords[w] {
				delete(qWords, w)
			}
		}
		if union > 0 {
			score := intersection / union
			if score > bestScore {
				bestScore = score
				bestAnswer = pair.A
			}
		}
	}
	return bestAnswer, bestScore
}

func (b *MoEChatBot) ensureVectorDB(projectRoot string) {
	if b.vectorDB == nil {
		if projectRoot == "" {
			projectRoot = "."
		}
		vectordbPath := filepath.Join(projectRoot, "data", "memory", "vectordb.json")
		b.vectorDB = memory.NewVectorDB(128, vectordbPath)
	}
}

func (b *MoEChatBot) Reply(input string) string {
	modelMutex.Lock()
	defer modelMutex.Unlock()

	// Sentiment Analysis & Emotional Steering
	sentiment := GetSentimentScore(input)
	isApologetic := false
	if sentiment < -0.5 {
		isApologetic = true
		// fmt.Println(" [System Note: Bot is in 'Apologetic Mode']")
		// for _, layer := range moe.ActiveLayers {
		// 	if len(layer.RouterBias) > 7 {
		// 		layer.RouterBias[7] += 2.0
		// 	}
		// }
	}

	// 0. Predictive Intent Classification — weighted scoring, not naive keyword match.
	// The old approach triggered on "go" in any sentence (e.g. "let's go"), tagging
	// everything as "tech". Now we use explicit phrase-level and bigram matching with
	// separate social/tech score accumulators. Social signals always compete.
	lowerInput := strings.ToLower(input)

	var socialScore, techScore float32

	// ── Social signals ──────────────────────────────────────────────────────────
	socialPhrases := []string{
		"how are you", "how's it going", "how you doing", "feeling", "doing well",
		"what's up", "sup", "good morning", "good evening", "good afternoon",
		"hello", "hey", "hi ", "howdy", "greetings",
		"who are you", "what are you", "your name", "about you", "tell me about",
		"thank", "please", "sorry", "nice to meet", "goodbye", "bye", "see you",
		"haha", "lol", "funny", "joke",
		"i'm bored", "entertain", "story", "chat", "talk",
	}
	for _, phrase := range socialPhrases {
		if strings.Contains(lowerInput, phrase) {
			if len(phrase) > 5 {
				socialScore += 2.5
			} else {
				socialScore += 1.0
			}
		}
	}

	// ── Tech signals ────────────────────────────────────────────────────────────
	// Use word-boundary style matching to avoid "go" in "I'm going to..."
	techPhrases := []string{
		"golang", "goroutine", "channel", "mutex", "interface{}", "struct{",
		"func ", "func(", " func", "error handling", "nil pointer",
		"compile", "runtime", "garbage collector", "package ", "import ",
		"context.context", "http handler", "middleware", "api", "endpoint",
		"database", "sql", "query", "schema", "orm",
		"what is a ", "what does ", "how do i ", "how do you ", "explain ",
		"define ", "difference between", "when to use", "how to ",
		"function", "variable", "pointer", "array", "slice", "map[",
		"architecture", "design pattern", "dependency", "module",
	}
	for _, phrase := range techPhrases {
		if strings.Contains(lowerInput, phrase) {
			if len(phrase) > 6 {
				techScore += 2.5
			} else {
				techScore += 1.0
			}
		}
	}

	// Exact "go" only scores if adjacent to a programming concept
	if strings.Contains(lowerInput, " go ") &&
		(techScore > 0 || strings.Contains(lowerInput, "program") || strings.Contains(lowerInput, "language")) {
		techScore += 1.5
	}

	// ── Makefile signals ────────────────────────────────────────────────────────
	var makefileScore float32
	makefilePhrases := []string{
		"make ", "makefile", "run make", "build project", "compile project",
		"train model", "start training", "clean models", "run tests",
		"generate protobuf", "convert yaml", "export labels",
		"run metrics", "install hooks", "fresh start", "resume training",
		"how do i run", "how do i build", "how do i train", "how do i clean",
		"how do i test", "how do i generate", "command to", "target to",
	}
	for _, phrase := range makefilePhrases {
		if strings.Contains(lowerInput, phrase) {
			if len(phrase) > 6 {
				makefileScore += 2.5
			} else {
				makefileScore += 1.5
			}
		}
	}

	// Sentiment can influence social leaning: strong emotion = social
	if sentiment > 0.5 || sentiment < -0.3 {
		socialScore += 0.8
	}

	// Previous-turn carry-over: if we already have a social context, lean social
	if strings.Contains(b.session.GetLastBotTurn(), "hello") ||
		strings.Contains(b.session.GetLastBotTurn(), "i'm doing") {
		socialScore += 1.0
	}

	predictedIntent := "social"
	if makefileScore > techScore && makefileScore > socialScore {
		predictedIntent = "makefile"
	} else if techScore > socialScore {
		predictedIntent = "tech"
	}

	var makefilePredictions string
	if predictedIntent == "makefile" && len(b.makefileTargets) > 0 {
		ranked := makefile.TopKMakefileCommands(b.makefileTargets, input, 3)
		if len(ranked) > 0 {
			var predParts []string
			predParts = append(predParts, "Top matches:")
			for _, r := range ranked {
				pct := int(r.Score * 100)
				predParts = append(predParts, fmt.Sprintf("- make %s (%d%%)", r.Name, pct))
			}
			makefilePredictions = strings.Join(predParts, "\n")
		}
	} else if len(b.makefileTargets) > 0 {
		if ranked := makefile.TopKMakefileCommands(b.makefileTargets, input, 3); len(ranked) > 0 && ranked[0].Score > 0 {
			var predParts []string
			predParts = append(predParts, "Top matches:")
			for _, r := range ranked {
				pct := int(r.Score * 100)
				predParts = append(predParts, fmt.Sprintf("- make %s (%d%%)", r.Name, pct))
			}
			makefilePredictions = strings.Join(predParts, "\n")
		}
	}

	// Extract topic for notepad logging
	predictedTopic := "general conversation"
	if predictedIntent == "tech" {
		predictedTopic = "programming / technical"
	} else {
		if strings.Contains(lowerInput, "who are you") || strings.Contains(lowerInput, "your name") {
			predictedTopic = "identity"
		} else if strings.Contains(lowerInput, "how are you") || strings.Contains(lowerInput, "feeling") {
			predictedTopic = "wellbeing check"
		} else if strings.Contains(lowerInput, "joke") || strings.Contains(lowerInput, "funny") {
			predictedTopic = "entertainment"
		}
	}

	// The model was trained with format: [BOS, __intent__, <intent>, :, __ques__, <words...>, EOS]
	// Inference MUST match this exactly — no __ans__ suffix, BOS+EOS wrappers required.
	formattedInput := "__intent__ " + predictedIntent + " : __ques__ " + input

	// 1. Tokenize and embed current input — with BOS and EOS wrappers to match training
	tokens := cleanTokenize(formattedInput)
	bosID := b.model.SentenceVocab.BosID
	eosID := b.model.SentenceVocab.EosID
	ids := make([]float32, len(tokens)+2)
	ids[0] = float32(bosID)
	for i, t := range tokens {
		ids[i+1] = float32(lookupVocab(t, b.model.SentenceVocab))
	}
	ids[len(ids)-1] = float32(eosID)
	avgInputEmbedding := make([]float32, b.model.Embedding.DimModel)
	tokenCount := 0
	for _, rawID := range ids {
		id := int(rawID)
		if id >= 0 && id < b.model.Embedding.VocabSize {
			start := id * b.model.Embedding.DimModel
			vec := b.model.Embedding.Weight.Data[start : start+b.model.Embedding.DimModel]
			for d := 0; d < b.model.Embedding.DimModel; d++ {
				avgInputEmbedding[d] += vec[d]
			}
			tokenCount++
		}
	}
	if tokenCount > 0 {
		for d := range avgInputEmbedding {
			avgInputEmbedding[d] /= float32(tokenCount)
		}
	}

	inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

	// Inference (Eval Mode)
	for _, l := range moe.ActiveLayers {
		l.SetMode(false)
	}

	emb, _ := b.model.Embedding.Forward(inputT)

	// Apply Positional Encoding to embeddings for word-order awareness
	if b.model.EncoderPos != nil {
		emb, _ = b.model.EncoderPos.Forward(emb)
	}

	// 2. Combine with context vector (session history blending)
	contextVector := b.session.GetContextVector()
	const lambda = 0.3 // Context decay factor
	if len(contextVector) == b.model.Embedding.DimModel {
		for i := 0; i < emb.Shape[1]; i++ {
			offset := i * b.model.Embedding.DimModel
			for j := 0; j < b.model.Embedding.DimModel; j++ {
				emb.Data[offset+j] += contextVector[j] * lambda
			}
		}
	}

	ctx, _ := b.model.Encoder.Forward(emb)

	// Normalize context vector
	if b.model.EncoderNorm != nil {
		ctx, _ = b.model.EncoderNorm.Forward(ctx)
	}

	// 3. Step-by-step decoding via GreedySearchDecodeWithTemp.
	// BeamSearchDecodeFiltered calls Decoder.Forward() which expects a full
	// target sequence (teacher-forced training mode) and fails for single-token
	// inputs at inference time. GreedySearchDecodeWithTemp uses Decoder.DecodeStep()
	// which is the correct autoregressive path.
	suppressedIDs := map[int]bool{
		b.model.SentenceVocab.GetTokenID("UNK"): true,
	}
	outIDs, decErr := b.model.GreedySearchDecodeWithTemp(
		ctx,
		50, // maxLen
		b.model.SentenceVocab.BosID,
		b.model.SentenceVocab.EosID,
		1.2, // temperature — more diversity
		4.0, // repetitionPenalty - strong penalty to break loops
		2.0, // frequencyPenalty - strong penalty to break loops
		40,  // topK
		suppressedIDs,
	)
	if decErr != nil {
		log.Printf("[CHAT] decoding error: %v", decErr)
		return ""
	}

	// 4. Convert IDs back to words.
	var response []string
	for _, id := range outIDs {
		word := b.model.SentenceVocab.GetWord(id)
		if word != "<s>" && word != "</s>" && word != "<pad>" {
			response = append(response, word)
		}
	}
	botResponse := strings.Join(response, " ")

	// ── Incoherence detection & retrieval fallback ──────────────────────────────
	// Detect if the neural output is word salad (model hasn't converged yet) and
	// substitute the best retrieval match instead.
	usedRetrieval := false
	retrievalScore := float32(0)
	isIncoherent := false
	if len(response) > 0 {
		// Check 1: repetition — if any single word appears more than 30% of tokens
		wordFreq := make(map[string]int)
		for _, w := range response {
			wordFreq[strings.ToLower(w)]++
		}
		maxFreq := 0
		for _, c := range wordFreq {
			if c > maxFreq {
				maxFreq = c
			}
		}
		if float32(maxFreq)/float32(len(response)) > 0.30 {
			isIncoherent = true
		}
		// Check 2: structural tokens leaked into output
		for _, w := range response {
			if w == "__intent__" || w == "__ques__" || w == "__ans__" {
				isIncoherent = true
				break
			}
		}
		// Check 3: too many punctuation/function tokens relative to words
		punctCount := 0
		for _, w := range response {
			if w == "." || w == "," || w == "?" || w == "!" || w == ")" || w == "(" {
				punctCount++
			}
		}
		if len(response) > 3 && float32(punctCount)/float32(len(response)) > 0.4 {
			isIncoherent = true
		}
	} else {
		isIncoherent = true // empty response
	}

	if isIncoherent {
		retrieved, score := b.retrievalLookup(input)
		if retrieved != "" && score > 0.1 {
			botResponse = retrieved
			usedRetrieval = true
			retrievalScore = score
		}
	}

	// Extract [REASONING] if present
	var reasoning string
	if rIdx := strings.Index(botResponse, "[REASONING]"); rIdx >= 0 {
		if respIdx := strings.Index(botResponse, "[RESPONSE]"); respIdx > rIdx {
			reasoning = strings.TrimSpace(botResponse[rIdx+len("[REASONING]") : respIdx])
			botResponse = strings.TrimSpace(botResponse[respIdx+len("[RESPONSE]"):])
		} else {
			reasoning = strings.TrimSpace(botResponse[rIdx+len("[REASONING]"):])
			botResponse = "" // The model only output reasoning
		}
	}

	// Attempt to predict the structural sub-intent and grammar skeleton
	var ruleName string
	var skeleton []string
	if b.model.Rules != nil {
		for name, rule := range b.model.Rules.Rules {
			if strings.HasPrefix(name, predictedIntent+":") {
				match := false
				for _, kw := range rule.RequiredKeywords {
					if strings.Contains(lowerInput, kw) {
						match = true
						break
					}
				}
				if match {
					ruleName = name
					skeleton = rule.GrammarSkeleton
					break
				}
			}
		}
	}

	// Always write the notepad plan/execution to a file in the current directory
	notepadPath := "gollemer_notepad.txt"
	f, err := os.OpenFile(notepadPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		f.WriteString("=== Turn Plan & Execution ===\n")
		f.WriteString(fmt.Sprintf("Input Analysis: Sentiment=%.2f | Social Score=%.1f | Tech Score=%.1f\n", sentiment, socialScore, techScore))
		f.WriteString(fmt.Sprintf("Predicted Intent: '%s' | Topic: %s\n", predictedIntent, predictedTopic))
		if usedRetrieval {
			f.WriteString(fmt.Sprintf("Source: RETRIEVAL FALLBACK (neural output was incoherent) | Match Score=%.2f\n", retrievalScore))
		} else {
			f.WriteString("Source: NEURAL (model output used directly)\n")
		}
		if ruleName != "" {
			f.WriteString(fmt.Sprintf("Structural Prediction: Matched rule '%s'. Expected structure: %v\n", ruleName, skeleton))
		} else {
			f.WriteString(fmt.Sprintf("Structural Prediction: No rule matched for '%s' intent. Using default generation.\n", predictedIntent))
		}
		f.WriteString(fmt.Sprintf("Execution: Injecting '__intent__ %s : __ques__' structural tokens before prompting model.\n", predictedIntent))
		if reasoning != "" {
			f.WriteString("Model Reasoning:\n" + reasoning + "\n")
		} else {
			f.WriteString("Model Reasoning: (Model is still training and hasn't output reasoning blocks yet)\n")
		}
		f.WriteString("Output: " + botResponse + "\n\n")
		f.Close()
		if usedRetrieval {
			fmt.Printf("📚 [Retrieval: match=%.2f]\n", retrievalScore)
		} else {
			fmt.Printf("🧠 [Neural output]\n")
		}
		fmt.Printf("📝 [Note: Plan & Execution logged to %s]\n", notepadPath)
	} else {
		log.Printf("⚠️ Failed to write to gollemer_notepad.txt: %v", err)
	}

	// 7. Save this turn to memory
	newTurn := ConversationTurn{
		Input:    avgInputEmbedding,
		RawInput: input,
		Intent:   "chat_response",         // Placeholder
		Entities: make(map[string]string), // Placeholder
		Response: botResponse,
	}
	b.session.AddToHistory(newTurn)

	// Reset Emotional Steering
	if isApologetic {
		// for _, layer := range moe.ActiveLayers {
		// 	if len(layer.RouterBias) > 7 {
		// 		layer.RouterBias[7] -= 2.0
		// 	}
		// }
	}

	// Cleanup memory for the next turn
	b.model.Detach()

	if makefilePredictions != "" {
		botResponse = botResponse + "\n\n" + makefilePredictions
	}

	return botResponse
}

func (b *MoEChatBot) StreamReply(userInput string) <-chan string {
	wordChan := make(chan string)

	go func() {
		defer close(wordChan)
		modelMutex.Lock()
		defer modelMutex.Unlock()

		// Sentiment Analysis & Emotional Steering
		sentiment := GetSentimentScore(userInput)
		isApologetic := false
		if sentiment < -0.5 {
			isApologetic = true
			// for _, layer := range moe.ActiveLayers {
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] += 2.0
			// 	}
			// }
		}

		// 1. Tokenize and embed current input
		tokens := cleanTokenize(userInput)
		ids := make([]float32, len(tokens))
		avgInputEmbedding := make([]float32, b.model.Embedding.DimModel)
		tokenCount := 0
		for i, t := range tokens {
			id := lookupVocab(t, b.model.SentenceVocab)
			ids[i] = float32(id)
			if id >= 0 && id < b.model.Embedding.VocabSize {
				start := id * b.model.Embedding.DimModel
				vec := b.model.Embedding.Weight.Data[start : start+b.model.Embedding.DimModel]
				for d := 0; d < b.model.Embedding.DimModel; d++ {
					avgInputEmbedding[d] += vec[d]
				}
				tokenCount++
			}
		}
		if tokenCount > 0 {
			for d := range avgInputEmbedding {
				avgInputEmbedding[d] /= float32(tokenCount)
			}
		}

		// 2. Combine with context vector
		contextVector := b.session.GetContextVector()
		const lambda = 0.3 // Context decay factor
		if len(contextVector) == b.model.Embedding.DimModel {
			// This logic will be applied to the embedding tensor below
		}
		inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

		// 3. Encode (Eval mode)
		for _, l := range moe.ActiveLayers {
			l.SetMode(false)
		}
		emb, _ := b.model.Embedding.Forward(inputT)

		// Apply context vector to embeddings
		if len(contextVector) == b.model.Embedding.DimModel {
			for i := 0; i < emb.Shape[1]; i++ {
				offset := i * b.model.Embedding.DimModel
				for j := 0; j < b.model.Embedding.DimModel; j++ {
					emb.Data[offset+j] += contextVector[j] * lambda
				}
			}
		}
		b.model.Encoder.Forward(emb)

		// 4. Decode Loop
		currIDs := []float32{float32(b.model.SentenceVocab.BosID)}
		var responseTokens []string

		for i := 0; i < 50; i++ {

			decInputT := tensor.NewTensor([]int{1, len(currIDs)}, currIDs, false)
			logits, _, _ := b.model.Forward(0.0, nil, decInputT)

			lastLogit := logits[len(logits)-1]
			nextID := b.sampleNextToken(lastLogit)

			if nextID == b.model.SentenceVocab.EosID {
				break
			}

			word := b.model.SentenceVocab.GetWord(nextID)
			if word != "<s>" && word != "</s>" && word != "<pad>" {
				wordChan <- word
				responseTokens = append(responseTokens, word)
			}
			currIDs = append(currIDs, float32(nextID))
		}

		// Save to history
		newTurn := ConversationTurn{
			Input:    avgInputEmbedding,
			RawInput: userInput,
			Intent:   "chat_response",         // Placeholder
			Entities: make(map[string]string), // Placeholder
			Response: strings.Join(responseTokens, " "),
		}
		b.session.AddToHistory(newTurn)

		// Reset Emotional Steering
		if isApologetic {
			// for _, layer := range moe.ActiveLayers {
			// 	if len(layer.RouterBias) > 7 {
			// 		layer.RouterBias[7] -= 2.0
			// 	}
			// }
		}

		// Cleanup
		b.model.Detach()
	}()

	return wordChan
}

func (b *MoEChatBot) sampleNextToken(logit *tensor.Tensor) int {
	probs := tensor.Softmax(logit)

	// Simple Greedy for now:
	var maxVal float32 = -1.0
	bestID := 0
	for i, v := range probs.Data {
		if v > maxVal {
			maxVal = v
			bestID = i
		}
	}
	return bestID
}

func StressTestBot(model *moe.IntentMoE) {
	const numUsers = 50
	const messagesPerUser = 5

	var wg sync.WaitGroup
	startTime := time.Now()

	fmt.Printf(" Starting Stress Test: %d Users, %d Messages each...\n", numUsers, messagesPerUser)

	for i := 0; i < numUsers; i++ {
		wg.Add(1)
		go func(userID int) {
			defer wg.Done()

			// Each user gets their own "Stateful Bot" instance
			// sharing the SAME underlying Model weights
			userBot := NewMoEChatBot(model)

			for m := 0; m < messagesPerUser; m++ {
				msg := fmt.Sprintf("User %d message %d: How are the experts doing?", userID, m)

				startMsg := time.Now()
				_ = userBot.Reply(msg)
				elapsed := time.Since(startMsg)

				if userID == 0 && m == 0 {
					fmt.Printf(" Sample Latency (User 0): %v\n", elapsed)
				}
			}
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(startTime)
	totalMsgs := numUsers * messagesPerUser
	fmt.Printf("\n---  Stress Test Results ---\n")
	fmt.Printf("Total Time:      %v\n", totalTime)
	fmt.Printf("Total Messages:  %d\n", totalMsgs)
	fmt.Printf("Throughput:      %.2f msgs/sec\n", float64(totalMsgs)/totalTime.Seconds())
}

// FormatUserOutput strips the internal scratchpad block from model output.
func FormatUserOutput(rawResponse string) string {
	if idx := strings.Index(rawResponse, "</think>"); idx != -1 {
		return strings.TrimSpace(rawResponse[idx+len("</think>"):])
	}
	return rawResponse
}
