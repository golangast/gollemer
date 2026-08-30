package chat

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/golangast/gollemer/internal/ai/memory"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
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
		fmt.Printf("Bot [%s]: %s\n", getExpertPath(), botResponse)

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

type MoEChatBot struct {
	model        *moe.IntentMoE
	session      *ChatSession
	systemPrompt string
	vectorDB     *memory.VectorDB
}

func NewMoEChatBot(model *moe.IntentMoE) *MoEChatBot {
	return &MoEChatBot{
		model:        model,
		session:      NewChatSession(5, model.Embedding.DimModel),
		systemPrompt: "System: You are a friendly, helpful assistant. Tone: Kind.",
	}
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

	// 1. Tokenize and embed current input
	tokens := cleanTokenize(input)
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

	inputT := tensor.NewTensor([]int{1, len(ids)}, ids, false)

	// Inference (Eval Mode)
	for _, l := range moe.ActiveLayers {
		l.SetMode(false)
	}

	emb, _ := b.model.Embedding.Forward(inputT)

	// 2. Combine with context vector
	contextVector := b.session.GetContextVector()
	const lambda = 0.3 // Context decay factor
	if len(contextVector) == b.model.Embedding.DimModel {
		for i := 0; i < emb.Shape[1]; i++ { // For each token in sequence
			offset := i * b.model.Embedding.DimModel
			for j := 0; j < b.model.Embedding.DimModel; j++ {
				emb.Data[offset+j] += contextVector[j] * lambda
			}
		}
	}

	ctx, _ := b.model.Encoder.Forward(emb)

	// 4. Beam Search Decoding
	outIDs := BeamSearchDecodeFiltered(b.model, ctx, 5, 50, []int{b.model.SentenceVocab.GetTokenID("UNK")})

	// 5. Convert IDs back to Words
	var response []string
	for _, id := range outIDs {
		word := b.model.SentenceVocab.GetWord(id)
		if word != "<s>" && word != "</s>" && word != "<pad>" {
			response = append(response, word)
		}
	}
	botResponse := strings.Join(response, " ")

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
