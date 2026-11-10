package chat

// Pins for the situation line (situation.go): the builder is
// deterministic from the state it is handed, the composed
// input carries it exactly when SocialInput composes, and a
// bare turn never grows one.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func situationAt(hour int) time.Time {
	return time.Date(2026, 11, 9, hour, 30, 0, 0, time.Local)
}

func TestSituationLineDayparts(t *testing.T) {
	cases := []struct {
		hour int
		want string
	}{
		{0, "night"}, {4, "night"}, {5, "morning"}, {11, "morning"},
		{12, "afternoon"}, {16, "afternoon"}, {17, "evening"},
		{21, "evening"}, {22, "night"}, {23, "night"},
	}
	for _, c := range cases {
		got := SituationLine(situationAt(c.hour), 1, nil)
		want := "[ situation : " + c.want + " . just started chatting ]"
		if got != want {
			t.Errorf("hour %d: got %q, want %q", c.hour, got, want)
		}
	}
}

func TestSituationLineDepth(t *testing.T) {
	if got := SituationLine(situationAt(10), 0, nil); got != "" {
		t.Errorf("no exchanges: got %q, want no line", got)
	}
	cases := []struct {
		exchanges int
		want      string
	}{
		{1, "just started chatting"},
		{2, "a few exchanges in"},
		{4, "a few exchanges in"},
		{5, "deep in conversation"},
		{12, "deep in conversation"},
	}
	for _, c := range cases {
		got := SituationLine(situationAt(10), c.exchanges, nil)
		want := "[ situation : morning . " + c.want + " ]"
		if got != want {
			t.Errorf("exchanges %d: got %q, want %q", c.exchanges, got, want)
		}
	}
}

func TestSituationLineTopics(t *testing.T) {
	at := situationAt(19)
	if got := SituationLine(at, 2, []string{"your dog"}); got !=
		"[ situation : evening . a few exchanges in . you told me about your dog ]" {
		t.Errorf("one topic: got %q", got)
	}
	if got := SituationLine(at, 2, []string{"your dog", "your city"}); got !=
		"[ situation : evening . a few exchanges in . you told me about your dog and your city ]" {
		t.Errorf("two topics: got %q", got)
	}
	if got := SituationLine(at, 6, []string{"your dog", "your city", "your job"}); got !=
		"[ situation : evening . deep in conversation . you told me about your dog, your city and your job ]" {
		t.Errorf("three topics: got %q", got)
	}
}

func TestSituationTopicPhrase(t *testing.T) {
	cases := map[string]string{
		"pet:dog":        "your dog",
		"pet:cat":        "your cat",
		"family:wife":    "your wife",
		"family:mother":  "your mother",
		"favorite:color": "your favorite color",
		"favorite:food":  "your favorite food",
		"city":           "your city",
		"job":            "your job",
		"birthday":       "your birthday",
		// Kinds with no clean phrase are skipped, honestly:
		// the user's name is the fact seat's to answer for,
		// and "learning" names no subject to phrase.
		"name":          "",
		"learning":      "",
		"unknown":       "",
		"pet:":          "",
		"pet:two words": "",
	}
	for key, want := range cases {
		if got := situationTopicPhrase(key); got != want {
			t.Errorf("key %q: got %q, want %q", key, got, want)
		}
	}
}

func TestExchangeCountSkipsRecall(t *testing.T) {
	c := NewConversation()
	if c.exchangeCount() != 0 {
		t.Fatalf("empty conversation: got %d exchanges", c.exchangeCount())
	}
	c.AddUser("hello")
	c.AddReply("hi there", SocialDomain, false)
	c.AddUser("what did you just say")
	c.AddReply("i said: \"hi there\"", SocialDomain, true) // recall: quotes, not an answer
	if c.exchangeCount() != 1 {
		t.Fatalf("after one exchange + recall: got %d, want 1", c.exchangeCount())
	}
	c.AddUser("how are you")
	c.AddReply("doing well", SocialDomain, false)
	if c.exchangeCount() != 2 {
		t.Fatalf("after two exchanges: got %d, want 2", c.exchangeCount())
	}
}

func TestSocialInputSituatedComposes(t *testing.T) {
	c := NewConversation()
	// No history: a first turn stands bare, situation or not.
	if got := c.SocialInputSituated("hello there", situationAt(9), nil); got != "hello there" {
		t.Fatalf("first turn: got %q, want the bare line", got)
	}
	c.AddUser("i love rainy days")
	c.AddReply("Rainy days are the best!", SocialDomain, false)
	got := c.SocialInputSituated("why do you like them", situationAt(9), []string{"your dog"})
	want := "[ situation : morning . just started chatting . you told me about your dog ] " +
		"before you said i love rainy days . before i said rainy days are the best! . now you say why do you like them"
	if got != want {
		t.Fatalf("composed:\n got %q\nwant %q", got, want)
	}
}

func TestSocialInputSituatedGateMatchesSocialInput(t *testing.T) {
	// After a non-social exchange the input stays bare —
	// the gate is SocialInput's own, unchanged.
	c := NewConversation()
	c.AddUser("what is a goroutine")
	c.AddReply("a goroutine is a lightweight thread", GoDomain, false)
	if got := c.SocialInputSituated("tell me more", situationAt(9), nil); got != "tell me more" {
		t.Fatalf("after a go exchange: got %q, want the bare line", got)
	}
	// After a history-hook claimed exchange it composes,
	// like SocialInput does.
	c2 := NewConversation()
	c2.AddUser("add a server")
	c2.AddClaimedReply("added the server")
	got := c2.SocialInputSituated("what next", situationAt(20), nil)
	if !strings.HasPrefix(got, "[ situation : evening . just started chatting ] before you said ") {
		t.Fatalf("after a claimed exchange: got %q, want a situated composed input", got)
	}
}

// The recall index must not learn situated multi-turn inputs
// as single-turn pairs (socialrecall.go's skip): a situated
// pair in the dataset leaves the index exactly as the bare
// pairs alone would.
func TestRecallIndexSkipsSituatedPairs(t *testing.T) {
	dir := t.TempDir()
	rel := ChatDatasetRelPath
	if err := os.MkdirAll(filepath.Join(dir, filepath.Dir(rel)), 0o755); err != nil {
		t.Fatal(err)
	}
	data := strings.Join([]string{
		`{"input":"hello","output":"hi there","domain":"social"}`,
		`{"input":"[ situation : morning . just started chatting ] before you said i love rainy days . before i said rainy days are the best! . now you say why","output":"because they are cozy","domain":"social"}`,
		`{"input":"before you said hi . before i said hello . now you say why","output":"because","domain":"social"}`,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(dir, rel), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}
	m := loadRecallPairs(dir, SocialDomain)
	if len(m) != 1 {
		t.Fatalf("recall index holds %d pairs, want exactly the 1 bare pair: %v", len(m), m)
	}
	if _, ok := m["hello"]; !ok {
		t.Fatalf("recall index lost the bare pair: %v", m)
	}
}
