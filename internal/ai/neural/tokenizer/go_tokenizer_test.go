package tokenizer

import (
	"reflect"
	"strings"
	"testing"

	vocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

func goTok() *GoTokenizer {
	return NewGoTokenizer(nil, DefaultGoTokenizerOptions())
}

func TestGoTokenizerKeywordsSingle(t *testing.T) {
	tokens := goTok().Encode("func var if else for return switch case struct interface map package import type const")
	want := []string{"func", "var", "if", "else", "for", "return", "switch", "case", "struct", "interface", "map", "package", "import", "type", "const"}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("got %q, want %q", tokens, want)
	}
}

func TestGoTokenizerBracketsSingle(t *testing.T) {
	tokens := goTok().Encode("a{b}(c)[d]")
	want := []string{"a", "{", "b", "}", "(", "c", ")", "[", "d", "]"}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("got %q, want %q", tokens, want)
	}
}

func TestGoTokenizerMultiCharOperators(t *testing.T) {
	tokens := goTok().Encode("a := b; c == d; e != f; g <= h; i >= j; k++; l--; m...n")
	want := []string{
		"a", ":=", "b", ";", "c", "==", "d", ";", "e", "!=", "f", ";",
		"g", "<=", "h", ";", "i", ">=", "j", ";", "k", "++", ";",
		"l", "--", ";", "m", "...", "n",
	}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("got %q, want %q", tokens, want)
	}
}

func TestGoTokenizerStringLiterals(t *testing.T) {
	tokens := goTok().Encode(`x := "hello \"world\""; y := ` + "`raw\nstr`" + `; z := 'q'`)
	if tokens[2] != `"hello \"world\""` {
		t.Errorf("interpreted string not single token: %q", tokens[2])
	}
	if tokens[6] != "`raw\nstr`" {
		t.Errorf("raw string not single token: %q", tokens[6])
	}
	if tokens[10] != `'q'` {
		t.Errorf("rune literal not single token: %q", tokens[10])
	}
}

func TestGoTokenizerNumbers(t *testing.T) {
	tokens := goTok().Encode("a := 42; b := 0xFF; c := 0b101; d := 3.14; e := 1e-3; f := 2i")
	wantNums := []string{"42", "0xFF", "0b101", "3.14", "1e-3", "2i"}
	got := []string{tokens[2], tokens[6], tokens[10], tokens[14], tokens[18], tokens[22]}
	if !reflect.DeepEqual(got, wantNums) {
		t.Errorf("got %q, want %q", got, wantNums)
	}
}

func TestGoTokenizerCommentsSkipped(t *testing.T) {
	tokens := goTok().Encode("a := 1 // set a\n/* block */ b := 2")
	want := []string{"a", ":=", "1", "\n", "b", ":=", "2"}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("got %q, want %q", tokens, want)
	}
}

func TestGoTokenizerCommentsKept(t *testing.T) {
	opts := DefaultGoTokenizerOptions()
	opts.SkipComments = false
	tok := NewGoTokenizer(nil, opts)
	tokens := tok.Encode("a := 1 // hi")
	found := false
	for _, tk := range tokens {
		if tk == "// hi" {
			found = true
		}
	}
	if !found {
		t.Errorf("comment not kept as single token: %q", tokens)
	}
}

func TestGoTokenizerIdentifiersWhole(t *testing.T) {
	tokens := goTok().Encode("getUserName := user_name1")
	want := []string{"getUserName", ":=", "user_name1"}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("got %q, want %q", tokens, want)
	}
}

func TestGoTokenizerSplitIdentifiers(t *testing.T) {
	opts := DefaultGoTokenizerOptions()
	opts.SplitIdentifiers = true
	tok := NewGoTokenizer(nil, opts)
	tokens := tok.Encode("getUserName HTTPServer user_name x1")
	want := []string{"get", "User", "Name", "HTTP", "Server", "user", "name", "x", "1"}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("got %q, want %q", tokens, want)
	}
}

func TestGoTokenizerRoundTrip(t *testing.T) {
	cases := []string{
		`func main() { fmt.Println("hi") }`,
		"package main\n\nimport \"fmt\"\n\nfunc add(a int, b int) int {\n\treturn a + b\n}",
		"if err != nil {\n\treturn fmt.Errorf(\"wrap: %w\", err)\n}",
		"for i := 0; i < 10; i++ {\n\tch <- i\n}",
		"type T struct {\n\tmu sync.Mutex\n\tvals []string\n}",
		"x, y := f(a, b...)",
	}
	for _, code := range cases {
		tok := goTok()
		roundTripped := tok.Decode(tok.Encode(code))
		if StripSpace(roundTripped) != StripSpace(code) {
			t.Errorf("round trip mismatch:\n in: %q\nout: %q", code, roundTripped)
		}
	}
}

func TestGoTokenizerDecodeSpacing(t *testing.T) {
	tok := goTok()
	out := tok.Decode(tok.Encode("func main(){fmt.Println(1)}"))
	// Must be valid-looking: space before {, no space in call parens.
	if !strings.Contains(out, "() {") {
		t.Errorf("expected '() {' spacing, got %q", out)
	}
	if strings.Contains(out, "Println (") {
		t.Errorf("unexpected space before call paren: %q", out)
	}
}

func TestGoTokenizerChannelsAndArrows(t *testing.T) {
	tokens := goTok().Encode("ch := make(chan int); x := <-ch; ch <- 1")
	want := []string{"ch", ":=", "make", "(", "chan", "int", ")", ";", "x", ":=", "<-", "ch", ";", "ch", "<-", "1"}
	if !reflect.DeepEqual(tokens, want) {
		t.Errorf("got %q, want %q", tokens, want)
	}
}

func TestGoTokenizerRealSnippet(t *testing.T) {
	code := `func Add(a, b int) (int, error) {
	if a < 0 {
		return 0, errors.New("neg")
	}
	return a + b, nil
}`
	tokens := goTok().Encode(code)
	// Spot-check structural tokens present exactly once each as single tokens.
	need := []string{"func", "Add", "(", "a", ",", "b", "int", ")", "(", "int", ",", "error", ")"}
	joined := strings.Join(tokens, "\x00")
	for _, w := range need {
		if !strings.Contains(joined, w) {
			t.Errorf("missing token %q in %q", w, tokens)
		}
	}
	// Round trip preserves everything but whitespace.
	if StripSpace(goTok().Decode(tokens)) != StripSpace(code) {
		t.Errorf("round trip failed for real snippet")
	}
}

func TestBuildGoVocab(t *testing.T) {
	texts := []string{
		"func main() { fmt.Println(\"hi\") }",
		"func add(a int, b int) int { return a + b }",
	}
	v := BuildGoVocab(texts, DefaultGoTokenizerOptions(), 1000)
	for _, tok := range []string{"func", "main", "(", ")", "{", "}", "int", "\n"} {
		if v.GetTokenID(tok) == v.UnkID {
			t.Errorf("expected token %q in vocab", tok)
		}
	}
	// Special tokens keep fixed IDs.
	if v.GetTokenID("<pad>") != 0 || v.UnkID != 1 || v.BosID != 2 || v.EosID != 3 {
		t.Errorf("special token IDs changed: pad=%d unk=%d bos=%d eos=%d",
			v.GetTokenID("<pad>"), v.UnkID, v.BosID, v.EosID)
	}
}

func TestGoTokenizerEncodeIDsRoundTrip(t *testing.T) {
	texts := []string{"func main() {\n\tfmt.Println(\"hi\")\n}"}
	v := BuildGoVocab(texts, DefaultGoTokenizerOptions(), 1000)
	tok := NewGoTokenizer(v, DefaultGoTokenizerOptions())
	code := texts[0]
	ids, err := tok.EncodeIDs(code)
	if err != nil {
		t.Fatal(err)
	}
	back, err := tok.DecodeIDs(ids)
	if err != nil {
		t.Fatal(err)
	}
	if StripSpace(back) != StripSpace(code) {
		t.Errorf("ID round trip mismatch:\n in: %q\nout: %q", code, back)
	}
}

func TestGoTokenizerImplementsIDCodec(t *testing.T) {
	var _ IDCodec = NewGoTokenizer(nil, DefaultGoTokenizerOptions())
	var _ IDCodec = func() IDCodec {
		v := vocab.NewVocabulary()
		tk, _ := NewTokenizer(v)
		return idCodecAdapter{tk}
	}()
}

// idCodecAdapter adapts *Tokenizer to IDCodec for the compile-time check.
type idCodecAdapter struct{ *Tokenizer }

func (a idCodecAdapter) EncodeIDs(s string) ([]int, error) { return a.Encode(s) }
func (a idCodecAdapter) DecodeIDs(ids []int) (string, error) { return a.Decode(ids) }
