package chat

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestRouteDomainCoverageAll checks the active router against every
// quality-gated training input: the neural router when its model is
// trained, skipped otherwise. The regex fallback alone misses 45 inputs
// (mostly "why does X call Y", which it has no pattern for); the neural
// router was trained to beat that to ~0.
func TestRouteDomainCoverageAll(t *testing.T) {
	root, err := filepath.Abs("../../../../")
	if err != nil {
		t.Fatal(err)
	}
	// The router resolves its model relative to the chat's project root.
	unifiedProjectRoot = root
	defer func() { unifiedProjectRoot = "" }()

	if _, err := os.Stat(RealModelPath(root, RouterDomain)); err != nil {
		t.Skip("router model not trained; run -train-real-seq2seq -domain router")
	}

	raw, err := os.ReadFile(filepath.Join(root, ChatDatasetRelPath))
	if err != nil {
		t.Fatal(err)
	}
	type pair struct {
		Input  string `json:"input"`
		Domain string `json:"domain"`
	}
	var misses []string
	counts := map[string]int{}
	for _, line := range strings.Split(string(raw), "\n") {
		var p pair
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			continue
		}
		if p.Domain == "" {
			p.Domain = SocialDomain
		}
		counts[p.Domain]++
		// History-format inputs ("before you said ... . now you say ...")
		// never pass through the router in production: routing happens on
		// the raw user message, and the history prefix is added afterwards
		// for the model only. They are model-input format, not
		// router-input format, so coverage does not apply to them.
		if strings.HasPrefix(strings.ToLower(p.Input), "before you said ") {
			continue
		}
		if got := routeDomain(p.Input); got != p.Domain {
			misses = append(misses, p.Domain+" != "+got+" :: "+p.Input)
		}
	}
	t.Logf("counts: %v", counts)
	for _, m := range misses {
		t.Logf("MISS: %s", m)
	}
	// The bar is beating the regex fallback the neural router replaces:
	// the hand-written rules miss 45 inputs on this set (mostly "why does
	// X call Y", which they have no pattern for). The neural router must
	// stay strictly better than that; the remaining misses are genuinely
	// ambiguous inputs, not regressions.
	regexMisses := 0
	for _, line := range strings.Split(string(raw), "\n") {
		var p pair
		if err := json.Unmarshal([]byte(line), &p); err != nil {
			continue
		}
		if p.Domain == "" {
			p.Domain = SocialDomain
		}
		if strings.HasPrefix(strings.ToLower(p.Input), "before you said ") {
			continue
		}
		if got := routeDomainRegex(p.Input); got != p.Domain {
			regexMisses++
		}
	}
	t.Logf("neural misses: %d, regex misses: %d", len(misses), regexMisses)
	if len(misses) >= regexMisses {
		t.Fatalf("neural router (%d misses) not better than regex fallback (%d)", len(misses), regexMisses)
	}
}
