package chat

import (
	"strings"
	"testing"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// testVocab builds a small vocabulary for the normalizer
// tests: the words the test lines are allowed to know.
func testVocab(words ...string) *vocab.Vocabulary {
	v := vocab.NewVocabulary()
	for _, w := range words {
		v.AddToken(w)
	}
	// Punctuation tokens the real vocabulary carries; they
	// must never become correction targets.
	for _, p := range []string{".", ",", "?", "!"} {
		v.AddToken(p)
	}
	return v
}

func normWith(t *testing.T, v *vocab.Vocabulary, line string, protected map[string]bool) (string, []TypoFix) {
	t.Helper()
	out, fixes := normalizeLine(line, v, typoTargetWords(v), protected)
	return out, fixes
}

func TestDamerauLevenshtein(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"going", "going", 0},
		{"gonig", "going", 1},  // adjacent transposition
		{"abcd", "abdc", 1},    // transposition at the end
		{"hoe", "hope", 1},     // insertion
		{"prett", "pretty", 1}, // insertion at the end
		{"takled", "talked", 1},
		{"kitten", "sitting", 3},
		{"ca", "abc", 2}, // true DL: transpose + insert (OSA says 3)
		{"flaw", "lawn", 2},
		{"", "abc", 3},
		{"abc", "", 3},
		{"wrok", "work", 1},
		{"wrok", "woke", 2},
	}
	for _, c := range cases {
		if got := damerauLevenshtein(c.a, c.b); got != c.want {
			t.Errorf("damerauLevenshtein(%q, %q) = %d, want %d", c.a, c.b, got, c.want)
		}
		if got := damerauLevenshtein(c.b, c.a); got != c.want {
			t.Errorf("damerauLevenshtein(%q, %q) = %d, want %d (symmetry)", c.b, c.a, got, c.want)
		}
	}
}

func TestNormalizeLineCorrectsTheObvious(t *testing.T) {
	v := testVocab("hey", "there", "how", "is", "it", "going", "good", "morning",
		"i", "hope", "you", "slept", "okay", "what", "up")
	out, fixes := normWith(t, v, "hey there how is it gonig", nil)
	if out != "hey there how is it going" {
		t.Fatalf("got %q, want the gonig->going correction", out)
	}
	if len(fixes) != 1 || fixes[0].From != "gonig" || fixes[0].To != "going" {
		t.Fatalf("fixes = %v", fixes)
	}
	// "hoe" is the eval's own S02c typo — but at 3 letters
	// it sits below the seat's length gate and must pass
	// through, however obvious the intended word is.
	out, fixes = normWith(t, v, "good morning i hoe you slept okay", nil)
	if out != "good morning i hoe you slept okay" || len(fixes) != 0 {
		t.Fatalf("3-letter token corrected: %q %v", out, fixes)
	}
}

func TestNormalizeLineRespectsPunctuationFabric(t *testing.T) {
	v := testVocab("hey", "there", "how", "is", "it", "going")
	out, _ := normWith(t, v, "hey there, how is it gonig?", nil)
	if out != "hey there, how is it going?" {
		t.Fatalf("got %q — the comma and question mark must survive byte-exact", out)
	}
}

func TestNormalizeLineTieMeansNoCorrection(t *testing.T) {
	// "wore" is one edit from both "work" (substitute) and
	// "word" (substitute): a tie, so the seat takes neither.
	v := testVocab("i", "really", "need", "work", "word")
	out, fixes := normWith(t, v, "i really need wore", nil)
	if out != "i really need wore" || len(fixes) != 0 {
		t.Fatalf("tie corrected anyway: %q %v", out, fixes)
	}
}

func TestNormalizeLineTooFarMeansNoCorrection(t *testing.T) {
	v := testVocab("i", "love", "this", "song")
	out, _ := normWith(t, v, "i love this zucchini", nil)
	if out != "i love this zucchini" {
		t.Fatalf("distant OOV word corrected: %q", out)
	}
}

func TestNormalizeLineLengthGate(t *testing.T) {
	v := testVocab("i", "am", "sad", "and", "the", "cat", "sat")
	// "sda" is a transposition of "sad" but only 3 letters:
	// below the gate, untouched.
	out, _ := normWith(t, v, "i am sda", nil)
	if out != "i am sda" {
		t.Fatalf("short token corrected: %q", out)
	}
}

func TestNormalizeLineNonAlphaUntouched(t *testing.T) {
	v := testVocab("i", "have", "a", "meeting", "in", "an", "hour", "go", "version",
		"the", "release", "know", "me", "later")
	// Digits, apostrophes, hyphens: never candidates.
	for _, line := range []string{
		"i have a meeting in an hour",
		"the go1.27 release",
		"i don't know",
		"e-mail me later",
	} {
		out, fixes := normWith(t, v, line, nil)
		if out != line || len(fixes) != 0 {
			t.Fatalf("%q -> %q (%v), want untouched", line, out, fixes)
		}
	}
}

func TestNormalizeLineTwoFixCap(t *testing.T) {
	v := testVocab("hey", "there", "how", "is", "it", "going", "good", "morning", "you")
	// Two typos: both corrected.
	out, fixes := normWith(t, v, "hey there how is it gonig this mornign", nil)
	if out != "hey there how is it going this morning" || len(fixes) != 2 {
		t.Fatalf("two-typo line: %q %v", out, fixes)
	}
	// Three defensible typos: the whole line passes
	// through untouched.
	line := "gonig mornign takled"
	v2 := testVocab("going", "morning", "talked")
	out2, fixes2 := normWith(t, v2, line, nil)
	if out2 != line || len(fixes2) != 0 {
		t.Fatalf("three-typo line rewritten: %q %v", out2, fixes2)
	}
}

func TestNormalizeLineProtectedWordsSurvive(t *testing.T) {
	// "bella" sits one edit from the vocabulary word
	// "bells"; as a fact-store value it is protected and
	// must survive. Unprotected, the same line shape would
	// still not correct — the protection is what the test
	// pins via a word that would otherwise flip: "willa"
	// (protected) vs "will" at distance 1.
	v := testVocab("my", "dog", "is", "a", "good", "girl", "will", "you")
	protected := map[string]bool{"bella": true, "willa": true}
	out, fixes := normWith(t, v, "my dog is bella", protected)
	if out != "my dog is bella" || len(fixes) != 0 {
		t.Fatalf("fact name rewritten: %q %v", out, fixes)
	}
	out, fixes = normWith(t, v, "willa is a good girl", protected)
	if out != "willa is a good girl" || len(fixes) != 0 {
		t.Fatalf("protected name rewritten: %q %v", out, fixes)
	}
	// Control: without the protection, willa flips to will.
	out, _ = normWith(t, v, "willa is a good girl", nil)
	if out != "will is a good girl" {
		t.Fatalf("control did not flip willa: %q", out)
	}
}

func TestNormalizeSocialCurrentUnarmedIsIdentity(t *testing.T) {
	// No model armed in this test process (and if another
	// test armed one, the switch discipline is covered by
	// the core tests above): a line with an obvious typo
	// must come back unchanged from the SEAT when the seat
	// has no vocabulary. This pins fail-open.
	typoMu.Lock()
	saved := typoVocab
	typoVocab = nil
	typoMu.Unlock()
	defer func() {
		typoMu.Lock()
		typoVocab = saved
		typoMu.Unlock()
	}()
	if got := normalizeSocialCurrent("hey there how is it gonig"); got != "hey there how is it gonig" {
		t.Fatalf("unarmed seat changed the line: %q", got)
	}
}

func TestNormalizeSocialLineExportedHandle(t *testing.T) {
	v := testVocab("hey", "there", "how", "is", "it", "going")
	if got := NormalizeSocialLine("hey there how is it gonig", v); got != "hey there how is it going" {
		t.Fatalf("exported handle: got %q", got)
	}
	if got := NormalizeSocialLine("hey there how is it gonig", nil); got != "hey there how is it gonig" {
		t.Fatalf("nil vocab must be identity: %q", got)
	}
}

func TestSplitIdentParts(t *testing.T) {
	cases := map[string][]string{
		"PriceTag":   {"price", "tag"},
		"priceTag":   {"price", "tag"},
		"HTTPServer": {"http", "server"},
		"my_var":     {"my", "var"},
		"Serve404":   {"serve404"},
		"main":       {"main"},
	}
	for in, want := range cases {
		got := splitIdentParts(in)
		if strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("splitIdentParts(%q) = %v, want %v", in, got, want)
		}
	}
}
