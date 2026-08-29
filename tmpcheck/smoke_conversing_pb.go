//go:build ignore

package main

import (
	"fmt"
	"strings"

	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
)

func main() {
	ds, err := datasetpb.LoadConversationDatasetFromProto("data/training/trainingdata/conversing.pb")
	if err != nil {
		panic(err)
	}
	pairCount := 0
	for _, conv := range ds.GetConversations() {
		fmt.Printf("=== %s (%d turns) ===\n", conv.GetConversationId(), len(conv.GetTurns()))
		turns := conv.GetTurns()
		for i := 0; i < len(turns)-1; i++ {
			t, next := turns[i], turns[i+1]
			if t.GetRole() == datasetpb.Role_ROLE_USER && next.GetRole() == datasetpb.Role_ROLE_ASSISTANT {
				fmt.Printf("  Q: %s\n  A: %s\n\n", strings.TrimSpace(t.GetContent()), strings.TrimSpace(next.GetContent()))
				pairCount++
			}
		}
	}
	fmt.Printf("Total Q→A pairs: %d\n", pairCount)
}
