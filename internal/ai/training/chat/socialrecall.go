package chat

// Social recall: deterministic answers from training data.
//
// The tiny social model doesn't reliably memorize every training pair —
// ask "have you ever played soccer" and it may answer with something
// unrelated even though the pair is in its data. This layer fixes that:
// when a social-routed message exactly matches a single-turn training
// input (after normalization), the trained output is returned verbatim
// instead of generating. Anything without an exact match falls through
// to the neural model, so generalization is untouched.

import (
	"encoding/json"
	"os"
	"strings"
	"sync"
)

var (
	socialRecallOnce sync.Once
	socialRecallMap  map[string]string
)

// loadRecallPairs reads single-turn pairs for one domain from the
// training JSONL, keyed by normalized input. Multi-turn history-format
// inputs are skipped (they're model-input format, not raw messages).
// Last pair wins on duplicates: the dataset is append-ordered and
// newer pairs override older ones.
func loadRecallPairs(projectRoot, domain string) map[string]string {
	out := map[string]string{}
	raw, err := os.ReadFile(ChatDatasetPath(projectRoot))
	if err != nil {
		return out
	}
	var p struct {
		Input  string `json:"input"`
		Output string `json:"output"`
		Domain string `json:"domain"`
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		p.Input, p.Output, p.Domain = "", "", ""
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			continue
		}
		if p.Domain != domain || p.Input == "" || p.Output == "" {
			continue
		}
		if strings.HasPrefix(strings.ToLower(p.Input), "before you said ") {
			continue
		}
		key := normalizeRecallInput(p.Input)
		if key == "" {
			continue
		}
		out[key] = p.Output
	}
	return out
}

// initSocialRecall loads every single-turn social pair into a
// normalized-input -> output map.
func initSocialRecall(projectRoot string) {
	socialRecallOnce.Do(func() {
		socialRecallMap = loadRecallPairs(projectRoot, SocialDomain)
	})
}

// normalizeRecallInput lowercases, collapses whitespace, and strips
// trailing punctuation so "Have you ever played soccer?" matches the
// trained "have you ever played soccer".
func normalizeRecallInput(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Join(strings.Fields(s), " ")
	s = strings.TrimRight(s, "?!.,")
	return strings.TrimSpace(s)
}

// LookupSocialRecall returns the trained output for an exact
// (normalized) training-input match. Call initSocialRecall first.
func LookupSocialRecall(input string) (string, bool) {
	if socialRecallMap == nil {
		return "", false
	}
	out, ok := socialRecallMap[normalizeRecallInput(input)]
	return out, ok
}

// SocialRecallSize reports how many pairs the recall layer indexed.
func SocialRecallSize() int {
	return len(socialRecallMap)
}
