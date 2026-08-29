package main

import (
	"fmt"
	"os"

	trainingpb "github.com/golangast/gollemer/internal/ai/training/proto"
)

func main() {
	// Inspect command examples
	examples, err := trainingpb.LoadCommandExamplesFromProto("data/training/command_examples.pb")
	if err != nil {
		fmt.Printf("Error loading command examples: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Command examples: %d\n", len(examples))
	for i, ex := range examples {
		if i < 5 {
			fmt.Printf("  [%d] type=%s system=%q user=%q assistant=%q code_before=%q code_after=%q\n",
				i, ex.Type, ex.SystemPrompt, ex.UserPrompt, ex.AssistantResponse, ex.CodeBefore, ex.CodeAfter)
		}
	}

	// Inspect conversations
	convs, err := trainingpb.LoadConversationsFromProto("data/training/trainingdata/conversations.pb")
	if err != nil {
		fmt.Printf("Error loading conversations: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("\nConversations: %d\n", len(convs))
	for i, conv := range convs {
		if i < 3 {
			fmt.Printf("  [%d] id=%s turns=%d\n", i, conv.Id, len(conv.Turns))
			for j, turn := range conv.Turns {
				if j < 8 {
					fmt.Printf("    turn[%d] seq=%d role=%s content=%q\n", j, turn.TurnSequence, turn.Role, turn.Content)
				}
			}
		}
	}
}
