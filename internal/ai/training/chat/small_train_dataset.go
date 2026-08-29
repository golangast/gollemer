package chat

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
	datasetpb "github.com/golangast/gollemer/internal/ai/training/proto/dataset"
)

// inferSmallDemoIntent derives a coarse intent label for the tiny social demo
// dataset purely from the user's query text. The protobuf ConversationDataset
// schema (see internal/ai/training/proto/dataset/dataset.proto) intentionally
// has no dedicated intent field — it only carries conversation_id,
// turn_sequence, role, and content — so intents are re-derived at load time.
// The heuristics below are tuned to reproduce the exact labels used in the
// original small_social_demo.csv fixture.
func inferSmallDemoIntent(query string) string {
	q := strings.ToLower(strings.TrimSpace(query))
	switch {
	case strings.Contains(q, "how are you"):
		return "status_check"
	case strings.Contains(q, "your name") || strings.Contains(q, "who are you"):
		return "identity"
	case strings.Contains(q, "thank"):
		return "thanks"
	case strings.Contains(q, "help"):
		return "help"
	case strings.Contains(q, "bye"):
		return "farewell"
	case strings.Contains(q, "hello") || strings.Contains(q, "hi"):
		return "greeting"
	default:
		return "social"
	}
}

// LoadSmallSocialDatasetFromProto reads a dataset.ConversationDataset protobuf
// file and expands each conversation's user→assistant turn pair into a
// moe.TrainPair, mirroring the behavior of loadCustomSocialPairs for the
// legacy CSV format.
func LoadSmallSocialDatasetFromProto(path string) ([]moe.TrainPair, error) {
	ds, err := datasetpb.LoadConversationDatasetFromProto(path)
	if err != nil {
		return nil, fmt.Errorf("load small social dataset proto: %w", err)
	}
	if len(ds.GetConversations()) == 0 {
		return nil, fmt.Errorf("small social dataset proto %s is empty", path)
	}

	pairs := make([]moe.TrainPair, 0, len(ds.GetConversations()))
	for _, conv := range ds.GetConversations() {
		var q, a string
		for _, turn := range conv.GetTurns() {
			content := strings.TrimSpace(turn.GetContent())
			if content == "" {
				continue
			}
			switch turn.GetRole() {
			case datasetpb.Role_ROLE_USER:
				q = content
			case datasetpb.Role_ROLE_ASSISTANT:
				a = content
			}
		}
		if q == "" || a == "" {
			continue
		}
		pairs = append(pairs, moe.TrainPair{
			Q:      q,
			A:      a,
			Intent: inferSmallDemoIntent(q),
		})
	}
	if len(pairs) == 0 {
		return nil, fmt.Errorf("small social dataset proto %s produced zero pairs", path)
	}
	return pairs, nil
}

// loadCustomSocialPairsAny dispatches between the protobuf and CSV loaders for
// the custom social dataset based on the file extension, so callers can point
// at either a small_social_demo.pb protobuf file or a legacy .csv fixture.
func loadCustomSocialPairsAny(path string) ([]moe.TrainPair, error) {
	if strings.HasSuffix(strings.ToLower(path), ".pb") {
		return LoadSmallSocialDatasetFromProto(path)
	}
	return loadCustomSocialPairs(path)
}
