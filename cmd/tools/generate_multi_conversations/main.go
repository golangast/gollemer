// Command generate_multi_conversations builds multi-turn conversation training
// data from the single-turn command_examples.csv corpus.
//
// It reads the social chit-chat and code_update command examples and weaves
// them into realistic multi-turn dialogues, then appends them to
// data/training/trainingdata/conversations.pb in protobuf format.
//
// Usage:
//
//	go run ./cmd/tools/generate_multi_conversations \
//	  -in=data/training/command_examples.csv \
//	  -out=data/training/trainingdata/conversations.pb \
//	  -conversations=50
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	trainingpb "github.com/golangast/gollemer/internal/ai/training/proto"
)

// CommandExample mirrors the simple.CommandExample struct but is parsed
// directly from the CSV using the standard encoding/csv package, which
// correctly handles double-quoted multi-line fields.
type CommandExample struct {
	Type      string // "social" or "code_update"
	Prompt    string // natural-language request
	Response  string // social response (only for social)
	CodeAfter string // transformed code (only for code_update)
}

// ConversationTurn is a single message in a multi-turn conversation.
type ConversationTurn struct {
	Role    string // "system", "user", or "assistant"
	Content string
}

// Conversation is a complete multi-turn dialogue.
type Conversation struct {
	ID    string
	Turns []ConversationTurn
}

// systemPrompt is the standard system message used across all conversations.
const systemPrompt = "You are Gollemer, an expert Go development assistant. Explain your code modifications clearly before providing code snippets."

// loadCommandExamples reads the command examples CSV using the standard
// encoding/csv package, which correctly handles double-quoted multi-line
// fields (e.g. Go code snippets spanning multiple lines).
func loadCommandExamples(path string) ([]CommandExample, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open csv: %w", err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = -1 // allow variable field counts
	reader.LazyQuotes = true

	var examples []CommandExample
	lineNum := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read csv line %d: %w", lineNum+1, err)
		}
		lineNum++

		// Skip header row.
		if lineNum == 1 {
			continue
		}

		if len(record) < 4 {
			log.Printf("⚠️  Skipping record %d: %d fields, want 4: %v", lineNum, len(record), record)
			continue
		}

		examples = append(examples, CommandExample{
			Type:      strings.TrimSpace(record[0]),
			Prompt:    strings.TrimSpace(record[1]),
			Response:  strings.TrimSpace(record[2]),
			CodeAfter: strings.TrimSpace(record[3]),
		})
	}

	return examples, nil
}

// buildConversation constructs a realistic multi-turn dialogue by interleaving
// social chit-chat with code-update requests. For code_update examples, the
// assistant response is a natural-language explanation of the code change,
// followed by the code snippet itself.
func buildConversation(id string, social []CommandExample, codeUpdates []CommandExample, rng *rand.Rand) Conversation {
	conv := Conversation{
		ID: id,
		Turns: []ConversationTurn{
			{Role: "system", Content: systemPrompt},
		},
	}

	// Pick a random number of turns (3-7 user/assistant pairs).
	numPairs := 3 + rng.Intn(5)

	for i := 0; i < numPairs; i++ {
		// Alternate between social and code_update, but bias toward code_update
		// since this is a Go development assistant.
		useSocial := rng.Intn(3) == 0 // ~33% social, ~67% code

		if useSocial && len(social) > 0 {
			ex := social[rng.Intn(len(social))]
			conv.Turns = append(conv.Turns,
				ConversationTurn{Role: "user", Content: ex.Prompt},
				ConversationTurn{Role: "assistant", Content: ex.Response},
			)
		} else if len(codeUpdates) > 0 {
			ex := codeUpdates[rng.Intn(len(codeUpdates))]
			// For code_update examples, the assistant response is a
			// natural-language explanation followed by the code snippet.
			assistantResponse := ex.Response
			if assistantResponse == "" {
				assistantResponse = fmt.Sprintf("I'll add the requested Go code snippet:\n%s", ex.CodeAfter)
			} else {
				assistantResponse = fmt.Sprintf("%s\n```go\n%s\n```", assistantResponse, ex.CodeAfter)
			}
			conv.Turns = append(conv.Turns,
				ConversationTurn{Role: "user", Content: ex.Prompt},
				ConversationTurn{Role: "assistant", Content: assistantResponse},
			)
		}
	}

	// Always end with a social closing if we have social examples.
	if len(social) > 0 && rng.Intn(2) == 0 {
		ex := social[rng.Intn(len(social))]
		conv.Turns = append(conv.Turns,
			ConversationTurn{Role: "user", Content: ex.Prompt},
			ConversationTurn{Role: "assistant", Content: ex.Response},
		)
	}

	return conv
}

// writeConversationsProto appends conversations to the output protobuf file.
func writeConversationsProto(path string, conversations []Conversation) error {
	// Ensure the output directory exists.
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Load existing conversations if the file exists.
	var allConversations []*trainingpb.Conversation
	if _, err := os.Stat(path); err == nil {
		if existing, err := trainingpb.LoadConversationsFromProto(path); err == nil {
			allConversations = append(allConversations, existing...)
		}
	}

	// Convert new conversations to protobuf.
	for _, conv := range conversations {
		pbConv := &trainingpb.Conversation{Id: conv.ID}
		for i, turn := range conv.Turns {
			pbConv.Turns = append(pbConv.Turns, &trainingpb.ConversationTurn{
				ConversationId: conv.ID,
				TurnSequence:   int32(i + 1),
				Role:           turn.Role,
				Content:        turn.Content,
			})
		}
		allConversations = append(allConversations, pbConv)
	}

	return trainingpb.SaveConversationsToProto(path, allConversations)
}

func main() {
	inputPath := flag.String("in", "data/training/command_examples.csv", "path to command examples CSV (type,prompt,response,code_after)")
	outputPath := flag.String("out", "data/training/trainingdata/conversations.pb", "output path for multi-turn conversations protobuf")
	numConversations := flag.Int("conversations", 50, "number of multi-turn conversations to generate")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed for reproducible generation")
	idPrefix := flag.String("prefix", "conv_gen_", "prefix for generated conversation IDs")
	flag.Parse()

	// Load the command examples.
	examples, err := loadCommandExamples(*inputPath)
	if err != nil {
		log.Fatalf("load command examples: %v", err)
	}

	// Split into social and code_update.
	var social []CommandExample
	var codeUpdates []CommandExample
	for _, ex := range examples {
		switch ex.Type {
		case "social":
			social = append(social, ex)
		case "code_update":
			codeUpdates = append(codeUpdates, ex)
		}
	}

	log.Printf("Loaded %d examples (%d social, %d code_update)", len(examples), len(social), len(codeUpdates))
	if len(social) == 0 || len(codeUpdates) == 0 {
		log.Fatalf("need both social and code_update examples to build conversations")
	}

	rng := rand.New(rand.NewSource(*seed))

	// Generate conversations.
	conversations := make([]Conversation, 0, *numConversations)
	for i := 0; i < *numConversations; i++ {
		id := fmt.Sprintf("%s%03d", *idPrefix, i+1)
		conv := buildConversation(id, social, codeUpdates, rng)
		conversations = append(conversations, conv)
	}

	// Write to protobuf.
	if err := writeConversationsProto(*outputPath, conversations); err != nil {
		log.Fatalf("write conversations: %v", err)
	}

	// Print summary.
	totalTurns := 0
	for _, conv := range conversations {
		totalTurns += len(conv.Turns)
	}
	fmt.Printf("✅ Generated %d multi-turn conversations (%d total turns) → %s\n",
		len(conversations), totalTurns, *outputPath)

	// Show a sample conversation.
	fmt.Println("\n📝 Sample conversation:")
	for _, conv := range conversations[:1] {
		for _, turn := range conv.Turns {
			prefix := "  "
			switch turn.Role {
			case "system":
				prefix = "  [system] "
			case "user":
				prefix = "  👤 "
			case "assistant":
				prefix = "  🤖 "
			}
			fmt.Printf("%s%s\n", prefix, turn.Content)
		}
	}
}
