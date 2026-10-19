// Package tokenizer — Go-aware lexical tokenizer.
//
// GoTokenizer is a lexer-style tokenizer for Go source code. Unlike the
// whitespace/JSON tokenizer in tokenizer.go or the BPE subword tokenizer in
// bpe.go, it understands Go lexical structure:
//
//   - Go keywords (func, var, if, for, return, ...) are single tokens
//   - Multi-character operators (:=, ==, !=, <=, >=, ++, --, ..., <<, >>,
//     &&, ||, <-, etc.) are single tokens via longest-match
//   - Brackets, braces and parens are single tokens
//   - String literals (interpreted "...", raw `...`, rune '...') are single
//     tokens, escapes handled
//   - Numeric literals (decimal, hex, octal, binary, float, imaginary) are
//     single tokens
//   - Comments are skipped by default (or emitted as single tokens)
//   - Newlines are emitted as "\n" tokens so decode can reproduce line
//     structure (Go's semicolon insertion makes newlines significant)
//   - Identifiers are kept whole by default; optional camelCase/snake_case
//     splitting via GoTokenizerOptions.SplitIdentifiers
//
// String-level API (per spec):
//
//	tok := tokenizer.NewGoTokenizer(nil, tokenizer.DefaultGoTokenizerOptions())
//	tokens := tok.Encode("func main() {}") // []string
//	code   := tok.Decode(tokens)           // string
//
// ID-level API (drop-in alongside *Tokenizer for training pipelines):
//
//	ids, err := tok.EncodeIDs("func main() {}") // []int via Vocabulary
//	code, err := tok.DecodeIDs(ids)
//
// Both *Tokenizer and *GoTokenizer satisfy:
//
//	type IDCodec interface {
//	    EncodeIDs(string) ([]int, error)
//	    DecodeIDs([]int) (string, error)
//	}
//
// Vocabulary building:
//
//	v := tokenizer.BuildGoVocab(codeSamples, 8000) // *vocab.Vocabulary
package tokenizer

import (
	"errors"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	vocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// IDCodec is the ID-level codec interface shared by *Tokenizer and
// *GoTokenizer. Call sites can program against this interface to select
// the tokenizer without changing the rest of the pipeline.
type IDCodec interface {
	EncodeIDs(string) ([]int, error)
	DecodeIDs([]int) (string, error)
}

// GoTokenizerOptions controls GoTokenizer behavior.
type GoTokenizerOptions struct {
	// SkipComments drops // and /* */ comments from the token stream.
	// When false, each comment is emitted as a single token.
	SkipComments bool
	// EmitNewlines emits "\n" tokens for line breaks. Disable for a pure
	// token stream with no layout information.
	EmitNewlines bool
	// SplitIdentifiers splits identifiers on camelCase and snake_case
	// boundaries (e.g. "getUserName" -> "get","User","Name").
	// Default false: identifiers stay whole so generated code reproduces
	// names exactly.
	SplitIdentifiers bool
}

// DefaultGoTokenizerOptions returns the recommended options:
// skip comments, keep newlines, keep identifiers whole.
func DefaultGoTokenizerOptions() GoTokenizerOptions {
	return GoTokenizerOptions{
		SkipComments:     true,
		EmitNewlines:     true,
		SplitIdentifiers: false,
	}
}

// GoTokenizer is a lexical tokenizer for Go source code.
type GoTokenizer struct {
	Options    GoTokenizerOptions
	Vocabulary *vocab.Vocabulary // optional; used by EncodeIDs/DecodeIDs
}

// NewGoTokenizer creates a GoTokenizer. vocab may be nil if only the
// string-level API (Encode/Decode) is used.
func NewGoTokenizer(v *vocab.Vocabulary, opts GoTokenizerOptions) *GoTokenizer {
	return &GoTokenizer{Options: opts, Vocabulary: v}
}

// goKeywords is the full set of Go language keywords (go/spec).
var goKeywords = map[string]bool{
	"break": true, "case": true, "chan": true, "const": true,
	"continue": true, "default": true, "defer": true, "else": true,
	"fallthrough": true, "for": true, "func": true, "go": true,
	"goto": true, "if": true, "import": true, "interface": true,
	"map": true, "package": true, "range": true, "return": true,
	"select": true, "struct": true, "switch": true, "type": true,
	"var": true,
}

// goOperators3/2 are multi-character operators, longest match first.
var goOperators3 = []string{"...", "<<=", ">>=", "&^="}

var goOperators2 = []string{
	":=", "==", "!=", "<=", ">=", "++", "--",
	"<<", ">>", "&^", "+=", "-=", "*=", "/=", "%=",
	"&=", "|=", "^=", "&&", "||", "<-",
}

// goPunct are single-character tokens.
const goPunct = "+-*/%&|^<>=!()[]{}.,;:"

// Tokenize lexes Go source code into a token slice.
func (t *GoTokenizer) Tokenize(text string) []string {
	return t.lex(text)
}

// Encode lexes Go source code into a token slice (string-level API).
func (t *GoTokenizer) Encode(text string) []string {
	return t.Tokenize(text)
}

// Decode rejoins tokens into Go source text with idiomatic spacing.
// Newline tokens ("\n") become real newlines. The result is whitespace-
// approximate to the input; use StripSpace for exact round-trip checks.
func (t *GoTokenizer) Decode(tokens []string) string {
	var sb strings.Builder
	var prev string
	for i, tok := range tokens {
		if tok == "\n" {
			sb.WriteString("\n")
			prev = "\n"
			continue
		}
		if i > 0 && goNeedsSpace(prev, tok) {
			sb.WriteString(" ")
		}
		sb.WriteString(tok)
		prev = tok
	}
	return sb.String()
}

// EncodeIDs tokenizes text and maps tokens to vocabulary IDs.
// Unknown tokens map to the vocabulary's UNK id. This mirrors
// (*Tokenizer).Encode so GoTokenizer is a drop-in replacement.
func (t *GoTokenizer) EncodeIDs(text string) ([]int, error) {
	if t.Vocabulary == nil {
		return nil, errors.New("go tokenizer vocabulary is not set")
	}
	tokens := t.Tokenize(text)
	ids := make([]int, 0, len(tokens))
	for _, tok := range tokens {
		ids = append(ids, t.Vocabulary.GetTokenID(tok))
	}
	return ids, nil
}

// DecodeIDs maps token IDs back to tokens and rejoins them.
// BOS/EOS/PAD ids are skipped; unknown ids render as [id].
func (t *GoTokenizer) DecodeIDs(ids []int) (string, error) {
	if t.Vocabulary == nil {
		return "", errors.New("go tokenizer vocabulary is not set")
	}
	tokens := make([]string, 0, len(ids))
	for _, id := range ids {
		if id == t.Vocabulary.BosID || id == t.Vocabulary.EosID || id == t.Vocabulary.PaddingTokenID {
			continue
		}
		w := t.Vocabulary.GetWord(id)
		if w == "" {
			w = "[" + itoa(id) + "]"
		}
		tokens = append(tokens, w)
	}
	return t.Decode(tokens), nil
}

// lex is the core scanner.
func (t *GoTokenizer) lex(text string) []string {
	var tokens []string
	i := 0
	n := len(text)
	for i < n {
		c := text[i]

		// Whitespace: track newlines, skip the rest.
		if c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f' {
			i++
			continue
		}
		if c == '\n' {
			if t.Options.EmitNewlines {
				// Collapse runs of newlines into one token.
				if len(tokens) == 0 || tokens[len(tokens)-1] != "\n" {
					tokens = append(tokens, "\n")
				}
			}
			i++
			continue
		}

		// Comments.
		if c == '/' && i+1 < n && (text[i+1] == '/' || text[i+1] == '*') {
			comment, next := lexComment(text, i)
			if !t.Options.SkipComments {
				tokens = append(tokens, comment)
			} else if t.Options.EmitNewlines && strings.HasPrefix(comment, "/*") && strings.Contains(comment, "\n") {
				// A skipped block comment spanning lines still breaks the line.
				if len(tokens) == 0 || tokens[len(tokens)-1] != "\n" {
					tokens = append(tokens, "\n")
				}
			}
			i = next
			continue
		}

		// String / rune literals.
		if c == '"' || c == '`' || c == '\'' {
			lit, next := lexString(text, i)
			tokens = append(tokens, lit)
			i = next
			continue
		}

		// Numbers (also handles leading-dot floats like .5).
		if isDigit(c) || (c == '.' && i+1 < n && isDigit(text[i+1])) {
			num, next := lexNumber(text, i)
			tokens = append(tokens, num)
			i = next
			continue
		}

		// Identifiers and keywords.
		if isLetter(c) {
			j := i + 1
			for j < n && (isLetter(text[j]) || isDigit(text[j])) {
				j++
			}
			word := text[i:j]
			if t.Options.SplitIdentifiers && !goKeywords[word] {
				tokens = append(tokens, splitGoIdentifier(word)...)
			} else {
				tokens = append(tokens, word)
			}
			i = j
			continue
		}

		// Operators: longest match (3-char, then 2-char, then 1-char).
		matched := false
		if i+3 <= n {
			for _, op := range goOperators3 {
				if text[i:i+3] == op {
					tokens = append(tokens, op)
					i += 3
					matched = true
					break
				}
			}
		}
		if !matched && i+2 <= n {
			for _, op := range goOperators2 {
				if text[i:i+2] == op {
					tokens = append(tokens, op)
					i += 2
					matched = true
					break
				}
			}
		}
		if !matched {
			if strings.ContainsRune(goPunct, rune(c)) {
				tokens = append(tokens, string(c))
				i++
			} else {
				// Unknown rune: emit as its own token rather than dropping it.
				r, size := utf8.DecodeRuneInString(text[i:])
				if r == utf8.RuneError && size <= 1 {
					i++
				} else {
					tokens = append(tokens, string(r))
					i += size
				}
			}
		}
	}
	return tokens
}

// lexComment scans a // or /* */ comment starting at i (text[i] == '/').
// Returns the comment text and the index just past it.
func lexComment(text string, i int) (string, int) {
	n := len(text)
	if text[i+1] == '/' {
		j := i + 2
		for j < n && text[j] != '\n' {
			j++
		}
		return text[i:j], j
	}
	// Block comment (Go block comments do not nest).
	j := i + 2
	for j+1 < n && !(text[j] == '*' && text[j+1] == '/') {
		j++
	}
	if j+1 < n {
		j += 2
	} else {
		j = n
	}
	return text[i:j], j
}

// lexString scans a string or rune literal starting at i.
// Handles " interpreted" (escapes), `raw` (may span lines), 'rune' (escapes).
func lexString(text string, i int) (string, int) {
	n := len(text)
	quote := text[i]
	j := i + 1
	for j < n {
		c := text[j]
		if quote == '`' {
			if c == '`' {
				return text[i : j+1], j + 1
			}
			j++
			continue
		}
		if c == '\\' && j+1 < n {
			j += 2
			continue
		}
		if c == quote {
			return text[i : j+1], j + 1
		}
		if c == '\n' {
			// Unterminated literal: stop at the newline.
			return text[i:j], j
		}
		j++
	}
	return text[i:j], j
}

// lexNumber scans a numeric literal starting at i.
func lexNumber(text string, i int) (string, int) {
	n := len(text)
	j := i

	// Hex / binary / octal prefixes.
	if text[j] == '0' && j+1 < n {
		switch text[j+1] {
		case 'x', 'X':
			j += 2
			for j < n && isHexDigit(text[j]) {
				j++
			}
			// Hex float: 0x1p-2
			if j < n && (text[j] == 'p' || text[j] == 'P') {
				j++
				if j < n && (text[j] == '+' || text[j] == '-') {
					j++
				}
				for j < n && isDigit(text[j]) {
					j++
				}
			}
			return finishImaginary(text, i, j)
		case 'b', 'B':
			j += 2
			for j < n && (text[j] == '0' || text[j] == '1') {
				j++
			}
			return finishImaginary(text, i, j)
		case 'o', 'O':
			j += 2
			for j < n && text[j] >= '0' && text[j] <= '7' {
				j++
			}
			return finishImaginary(text, i, j)
		}
	}

	// Decimal / float.
	for j < n && isDigit(text[j]) {
		j++
	}
	if j < n && text[j] == '.' {
		// Don't consume the '.' of "..." or a selector like "1.foo" (invalid
		// Go anyway, but keep the lexer predictable): only a float if a digit
		// follows or we already saw digits and next isn't another '.'.
		if j+1 < n && text[j+1] == '.' {
			return finishImaginary(text, i, j)
		}
		j++
		for j < n && isDigit(text[j]) {
			j++
		}
	}
	if j < n && (text[j] == 'e' || text[j] == 'E') {
		k := j + 1
		if k < n && (text[k] == '+' || text[k] == '-') {
			k++
		}
		if k < n && isDigit(text[k]) {
			j = k
			for j < n && isDigit(text[j]) {
				j++
			}
		}
	}
	return finishImaginary(text, i, j)
}

// finishImaginary consumes a trailing 'i' for imaginary literals.
func finishImaginary(text string, i, j int) (string, int) {
	if j < len(text) && text[j] == 'i' {
		j++
	}
	return text[i:j], j
}

// splitGoIdentifier splits on snake_case and camelCase boundaries:
// "user_name" -> ["user","name"], "getUserName" -> ["get","User","Name"],
// "HTTPServer" -> ["HTTP","Server"], "x1" -> ["x1"].
func splitGoIdentifier(word string) []string {
	// Snake case first.
	if strings.Contains(word, "_") {
		parts := strings.Split(word, "_")
		var out []string
		for _, p := range parts {
			if p == "" {
				continue
			}
			out = append(out, splitCamel(p)...)
		}
		if len(out) > 0 {
			return out
		}
	}
	return splitCamel(word)
}

func splitCamel(word string) []string {
	runes := []rune(word)
	var parts []string
	start := 0
	for i := 1; i < len(runes); i++ {
		prev, cur := runes[i-1], runes[i]
		boundary := false
		if unicode.IsLower(prev) && unicode.IsUpper(cur) {
			boundary = true // get|User
		} else if unicode.IsUpper(prev) && unicode.IsUpper(cur) {
			// Acronym run: split before last upper when followed by lower.
			if i+1 < len(runes) && unicode.IsLower(runes[i+1]) {
				boundary = true // HTTP|Server
			}
		} else if unicode.IsLetter(prev) && unicode.IsDigit(cur) {
			boundary = true // x|1
		} else if unicode.IsDigit(prev) && unicode.IsLetter(cur) {
			boundary = true // 1|x
		}
		if boundary {
			parts = append(parts, string(runes[start:i]))
			start = i
		}
	}
	parts = append(parts, string(runes[start:]))
	return parts
}

// goNeedsSpace reports whether a space is needed between two adjacent
// tokens when rejoining.
func goNeedsSpace(prev, cur string) bool {
	if prev == "" || prev == "\n" {
		return false
	}
	// Never space before these.
	switch cur {
	case ",", ";", ")", "]", "}", ":", ".", "...", "++", "--":
		return false
	}
	// Never space after these.
	switch prev {
	case "(", "[", ".", "++", "--":
		return false
	}
	// Space before '{' (idiomatic; "func main() {").
	if cur == "{" {
		return true
	}
	// No space between identifier and '(' (call) unless prev is a keyword
	// ("if (", "for (", "switch (" are idiomatic with a space).
	if cur == "(" {
		return goKeywords[prev]
	}
	// Unary-ish prefix operators attach to what follows.
	if prev == "!" || prev == "<-" {
		return false
	}
	return true
}

// GoTokenFrequencies tokenizes each text and returns token -> count.
func GoTokenFrequencies(texts []string, opts GoTokenizerOptions) map[string]int {
	tok := NewGoTokenizer(nil, opts)
	freq := make(map[string]int)
	for _, text := range texts {
		for _, t := range tok.Tokenize(text) {
			freq[t]++
		}
	}
	return freq
}

// BuildGoVocab builds a *vocab.Vocabulary from Go source texts using the
// GoTokenizer. Tokens are added in descending frequency order after the
// four built-in special tokens (<pad>, UNK, <s>, </s>); newlines ("\n")
// are always included even if rare. At most maxSize tokens total;
// maxSize <= 0 means no limit.
func BuildGoVocab(texts []string, opts GoTokenizerOptions, maxSize int) *vocab.Vocabulary {
	freq := GoTokenFrequencies(texts, opts)
	type kv struct {
		tok string
		n   int
	}
	var sorted []kv
	for tok, n := range freq {
		sorted = append(sorted, kv{tok, n})
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].n != sorted[j].n {
			return sorted[i].n > sorted[j].n
		}
		return sorted[i].tok < sorted[j].tok
	})

	v := vocab.NewVocabulary() // <pad>=0, UNK=1, <s>=2, </s>=3
	// Guarantee "\n" is present when newlines are emitted.
	if opts.EmitNewlines {
		if _, ok := freq["\n"]; !ok {
			v.AddToken("\n")
		}
	}
	for _, kv := range sorted {
		if maxSize > 0 && v.Size() >= maxSize {
			break
		}
		v.AddToken(kv.tok)
	}
	return v
}

// StripSpace removes all whitespace; used for whitespace-insensitive
// round-trip comparisons of tokenized Go code.
func StripSpace(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, s)
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func isHexDigit(c byte) bool {
	return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')
}

func isLetter(c byte) bool {
	return c == '_' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	neg := i < 0
	if neg {
		i = -i
	}
	var buf [20]byte
	pos := len(buf)
	for i > 0 {
		pos--
		buf[pos] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		pos--
		buf[pos] = '-'
	}
	return string(buf[pos:])
}
