package chat

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/moe"
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

// loadCustomSocialPairsAny dispatches to the legacy CSV loader for the custom
// social dataset. Protobuf dataset support was removed (pure-Go,
// zero-dependency requirement); callers must point at a .csv fixture.
func loadCustomSocialPairsAny(path string) ([]moe.TrainPair, error) {
	if strings.HasSuffix(strings.ToLower(path), ".pb") {
		return nil, fmt.Errorf("protobuf datasets are no longer supported (pure-Go zero-dependency build); use the .csv fixture instead of %s", path)
	}
	return loadCustomSocialPairs(path)
}
