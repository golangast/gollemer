package chat

import (
	"strings"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// subwordTokens splits text into subword pieces for the mid-transformer
// experiments. It follows the name-splitting philosophy used for
// understanding v2 (analyze.SplitCamelWords): an identifier the model has
// never seen whole still decomposes into pieces it has seen, so a held-out
// file or function name is representable instead of collapsing to UNK —
// the failure that zeroed the LSTM's copy-experiment echo.
//
// Rules:
//   - The text is scanned rune by rune. Whitespace separates tokens.
//   - Any rune that is not a letter or digit is its own token (path
//     separators, dots, brackets, operators...), so "examples/foo-bar/main.go"
//     becomes examples / foo - bar / main . go with the separators kept.
//   - A letter/digit run (an identifier, keyword, or word) is split further
//     by splitIdentPiece: camelCase and acronym boundaries, with every
//     digit emitted individually so numbers compose from seen digits.
//   - Pieces are lowercased, matching the seq2seq side's case-insensitive
//     convention (tokenizeLower).
func subwordTokens(text string) []string {
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, splitIdentPiece(cur.String())...)
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case unicode.IsLetter(r) || unicode.IsDigit(r):
			cur.WriteRune(r)
		default:
			flush()
			out = append(out, string(r))
		}
	}
	flush()
	return out
}

// splitIdentPiece splits one alphanumeric run into subword pieces:
// lower/digit->upper boundaries and acronym boundaries (HTTPServer ->
// http, server, mirroring analyze.SplitCamelWords), and one token per
// digit. Pieces are lowercased.
func splitIdentPiece(s string) []string {
	var pieces []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			pieces = append(pieces, strings.ToLower(string(cur)))
			cur = cur[:0]
		}
	}
	runes := []rune(s)
	for i, r := range runes {
		if unicode.IsDigit(r) {
			flush()
			pieces = append(pieces, string(r))
			continue
		}
		if unicode.IsUpper(r) && len(cur) > 0 {
			prev := runes[i-1]
			nextIsLower := i+1 < len(runes) && unicode.IsLower(runes[i+1])
			if unicode.IsLower(prev) || unicode.IsDigit(prev) || (unicode.IsUpper(prev) && nextIsLower) {
				flush()
			}
		}
		cur = append(cur, r)
	}
	flush()
	return pieces
}

// encodeSubword maps text to vocab IDs via subwordTokens. Pieces missing
// from the vocabulary map to its UNK id.
func encodeSubword(text string, v *vocab.Vocabulary) []int {
	toks := subwordTokens(text)
	ids := make([]int, len(toks))
	for i, t := range toks {
		ids[i] = v.GetTokenID(t)
	}
	return ids
}
