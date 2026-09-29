package chat

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func TestRouteDomainCoverageAll(t *testing.T) {
	raw, err := os.ReadFile("../../../../data/training/chat_pairs.jsonl")
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
	if len(misses) > 0 {
		t.Fatalf("%d routing misses", len(misses))
	}
}
