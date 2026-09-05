package chat

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/memory"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/context"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// loadSentenceVocabJSON loads the sentence_vocab.json file which stores a
// word→id map and converts it into a Vocabulary.
func loadSentenceVocabJSON(path string) (*mainvocab.Vocabulary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var wordToID map[string]int
	if err := json.Unmarshal(data, &wordToID); err != nil {
		return nil, err
	}
	if len(wordToID) == 0 {
		return nil, fmt.Errorf("empty vocab map in %s", path)
	}

	// Find the max ID to size the TokenToWord slice.
	maxID := -1
	for _, id := range wordToID {
		if id > maxID {
			maxID = id
		}
	}

	v := &mainvocab.Vocabulary{
		WordToToken: wordToID,
		TokenToWord: make([]string, maxID+1),
	}
	for word, id := range wordToID {
		v.TokenToWord[id] = word
	}

	// Restore special token IDs.
	if id, ok := wordToID["<pad>"]; ok {
		v.PaddingTokenID = id
	}
	if id, ok := wordToID["UNK"]; ok {
		v.UnkID = id
	}
	if id, ok := wordToID["<s>"]; ok {
		v.BosID = id
	}
	if id, ok := wordToID["</s>"]; ok {
		v.EosID = id
	}
	return v, nil
}

// ensureSentenceVocab guarantees intentModel.SentenceVocab is populated with a
// meaningful vocabulary. It tries (in order):
//  1. The model's own embedded vocab (if large enough).
//  2. seq2seq_output_vocab.gob (if it exists).
//  3. sentence_vocabulary.gob (if it exists).
//  4. sentence_vocab.json (the JSON word-map, always present after training).
//  5. A minimal hard-coded fallback.
func ensureSentenceVocab(intentModel *moe.IntentMoE, projectRoot string) {
	const minVocabSize = 100

	if intentModel.SentenceVocab != nil && intentModel.SentenceVocab.Size() >= minVocabSize {
		log.Printf("[CHAT] using embedded vocab (size=%d)", intentModel.SentenceVocab.Size())
		return
	}

	// 2. seq2seq_output_vocab.gob
	if p := filepath.Join(projectRoot, "data/models/gob_models/seq2seq_output_vocab.gob"); fileExists(p) {
		if v, err := mainvocab.LoadVocabulary(p); err == nil && v.Size() >= minVocabSize {
			intentModel.SentenceVocab = v
			log.Printf("[CHAT] loaded vocab from seq2seq_output_vocab.gob (size=%d)", v.Size())
			return
		}
	}

	// 3. sentence_vocabulary.gob
	if p := filepath.Join(projectRoot, "data/models/gob_models/sentence_vocabulary.gob"); fileExists(p) {
		if v, err := mainvocab.LoadVocabulary(p); err == nil && v.Size() >= minVocabSize {
			intentModel.SentenceVocab = v
			log.Printf("[CHAT] loaded vocab from sentence_vocabulary.gob (size=%d)", v.Size())
			return
		}
	}

	// 4. sentence_vocab.json  (word→id map produced by build_vocab_from_training / prune_vocab)
	if p := filepath.Join(projectRoot, "data/models/gob_models/sentence_vocab.json"); fileExists(p) {
		if v, err := loadSentenceVocabJSON(p); err == nil && v.Size() >= minVocabSize {
			intentModel.SentenceVocab = v
			log.Printf("[CHAT] loaded vocab from sentence_vocab.json (size=%d)", v.Size())
			return
		}
	}

	// 5. Hard-coded minimal fallback (chat will produce very short/empty replies).
	log.Printf("[CHAT] WARNING: no large vocab found; using minimal fallback — run 'make train' to fix")
	v := mainvocab.NewVocabulary()
	for _, tok := range []string{"<pad>", "<s>", "</s>", "UNK", "user", "assistant", "system", "\n"} {
		v.AddToken(tok)
	}
	v.PaddingTokenID = v.GetTokenID("<pad>")
	v.BosID = v.GetTokenID("<s>")
	v.EosID = v.GetTokenID("</s>")
	intentModel.SentenceVocab = v
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// RunMoEChat loads the full MoE social/chat model and starts an interactive
// multi-turn REPL with conversation history and co-reference resolution.
func RunMoEChat(projectRoot string) {
	modelPath := filepath.Join(projectRoot, "data/models/gob_models/moe_social_model.gob")
	if _, err := os.Stat(modelPath); err != nil {
		log.Fatalf("[CHAT] MoE social model not found at %s. Run 'make train' first.", modelPath)
	}

	intentModel, err := moe.LoadIntentMoEModelWithFallback(modelPath)
	if err != nil {
		log.Fatalf("[CHAT] failed to load model: %v", err)
	}
	intentModel.RepairArchitecture()

	moe.ActiveLayers = findMoELayers(intentModel)
	if len(moe.ActiveLayers) == 0 {
		log.Fatalf("[CHAT] model has no active MoE layers")
	}
	log.Printf("[CHAT] loaded model with %d active MoE layers", len(moe.ActiveLayers))

	ensureSentenceVocab(intentModel, projectRoot)
	log.Printf("[CHAT] vocab size = %d | BOS=%d EOS=%d PAD=%d",
		intentModel.SentenceVocab.Size(),
		intentModel.SentenceVocab.BosID,
		intentModel.SentenceVocab.EosID,
		intentModel.SentenceVocab.PaddingTokenID,
	)

	vectordbPath := filepath.Join(projectRoot, "data", "memory", "vectordb.json")
	vectorDB := memory.NewVectorDB(128, vectordbPath)

	convCtx := context.NewConversationContext(6)
	bot := NewMoEChatBot(intentModel)
	bot.vectorDB = vectorDB

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("\n--- Gollemer MoE Chat (multi-turn + CoT) ---")
	fmt.Println("Type 'quit' or 'exit' to stop.")

	for {
		fmt.Print("\nYou: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println()
			return
		}
		prompt := strings.TrimSpace(input)
		if prompt == "" {
			continue
		}
		if strings.EqualFold(prompt, "quit") || strings.EqualFold(prompt, "exit") {
			fmt.Println("[CHAT] closing.")
			return
		}

		resolved := convCtx.ResolveCoReference(prompt)
		history := convCtx.GetConversationHistory()
		augmented := resolved
		if history != "" {
			augmented = history + "Human: " + resolved + "\nAI: "
		}

		response := bot.Reply(augmented)
		if response == "" {
			response = "[no response generated]"
		}

		fmt.Printf("Bot: %s\n", FormatUserOutput(response))

		convCtx.AddTurn("chat", nil, resolved)
		convCtx.AddResponse(response)
	}
}
