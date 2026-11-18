package chat

import (
	"errors"
	"strings"
	"testing"
)

// The topic resolution and carry check (layers_decode.go):
// the longest bank phrase the line names wins; a reply
// carries the line when it names that phrase; a line naming
// no bank topic falls back to its own content words.
func TestLayersTopicCheck(t *testing.T) {
	if phrase, _ := layersTopicFor("i love rock climbing and chess"); phrase != "rock climbing" {
		t.Errorf("layersTopicFor(multi-topic) = %q, want the longest phrase %q", phrase, "rock climbing")
	}
	if phrase, _ := layersTopicFor("any tips for metal detecting"); phrase != "metal detecting" {
		t.Errorf("layersTopicFor(metal detecting) = %q, want %q", phrase, "metal detecting")
	}
	if phrase, _ := layersTopicFor("whats the time"); phrase != "" {
		t.Errorf("layersTopicFor(no topic) = %q, want \"\"", phrase)
	}
	if !layersReplyCarries("why is chess so popular", "why chess? because it slows the day down.") {
		t.Error("a reply naming the topic should carry the line")
	}
	if layersReplyCarries("why is chess so popular", "why snowboarding? because it slows the day down.") {
		t.Error("a reply naming a different topic should not carry the line")
	}
	if layersReplyCarries("why is chess so popular", "") {
		t.Error("an empty reply never carries")
	}
	// Fallback: no bank topic in the line — the line's own
	// content words decide.
	if !layersReplyCarries("i feel worn out today", "being worn out is rough; rest helps.") {
		t.Error("a reply sharing a content word should carry an untopiced line")
	}
	if layersReplyCarries("i feel worn out today", "that is completely understandable.") {
		t.Error("a reply sharing no content word should not carry an untopiced line")
	}
}

// The selection policy: a carrying greedy reply is kept
// without a single draw; a dropping one is rescued by the
// first carrying draw; errors and all-miss draws leave the
// greedy reply standing.
func TestLayersPickReply(t *testing.T) {
	carries := func(s string) bool { return strings.Contains(s, "chess") }

	draws := 0
	keep := layersPickReply("chess is great", carries, func() (string, error) {
		draws++
		return "a sampled reply", nil
	})
	if keep != "chess is great" || draws != 0 {
		t.Errorf("carrying greedy: got %q after %d draws, want the greedy reply and 0 draws", keep, draws)
	}

	pool := []string{"nothing here", "chess again", "chess once more"}
	i := 0
	rescued := layersPickReply("a greeting", carries, func() (string, error) {
		s := pool[i]
		i++
		return s, nil
	})
	if rescued != "chess again" || i != 2 {
		t.Errorf("rescue: got %q after %d draws, want %q after 2", rescued, i, "chess again")
	}

	failed := layersPickReply("a greeting", carries, func() (string, error) {
		return "", errors.New("frameless only")
	})
	if failed != "a greeting" {
		t.Errorf("draw error: got %q, want the greedy reply", failed)
	}

	missed := layersPickReply("a greeting", carries, func() (string, error) {
		return "still nothing", nil
	})
	if missed != "a greeting" {
		t.Errorf("all-miss draws: got %q, want the greedy reply", missed)
	}
}
