package decl

import (
	"strings"
)

// SplitTokens splits text exactly the way the seq2seq training side
// does (chat.tokenizeLower): whitespace separates, and the characters
// {}[],:" plus sentence punctuation are single-char tokens. It is
// duplicated here because the original is unexported; the decl tests
// pin this splitter against the trainer's tokenizer on real pairs, so
// drift would fail loudly rather than silently corrupt the data.
func SplitTokens(text string) []string {
	var toks []string
	runes := []rune(strings.ToLower(text))
	i := 0
	for i < len(runes) {
		r := runes[i]
		if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
			i++
			continue
		}
		if strings.ContainsRune(`{}[],:\"?!.`, r) {
			toks = append(toks, string(r))
			i++
			continue
		}
		start := i
		for i < len(runes) && runes[i] != ' ' && runes[i] != '\t' && runes[i] != '\n' && runes[i] != '\r' && !strings.ContainsRune(`{}[],:\"`, runes[i]) {
			i++
		}
		if tok := string(runes[start:i]); tok != "" {
			toks = append(toks, tok)
		}
	}
	return toks
}

// MaxTargetTokens is the training-side cap with margin: the trainer
// drops pairs whose encoded target exceeds 60 tokens (BOS+EOS
// included), so generated targets stay at or below 58.
const MaxTargetTokens = 58

// MaxInputTokens keeps spec/phrasing inputs in a sane batch range.
// The trainer imposes no input cap; this is a pipeline policy so one
// outlier cannot dominate padding.
const MaxInputTokens = 72

// DecodeFeasible reports whether a target string's token stream can
// survive the model's greedy decoder at all: the decoder forbids an
// immediately repeated token and any repeated bigram, so a target
// containing either is unlearnable-by-construction and must never
// enter the dataset.
func DecodeFeasible(target string) bool {
	toks := SplitTokens(target)
	seen := make(map[[2]string]bool, len(toks))
	for i := 1; i < len(toks); i++ {
		if toks[i] == toks[i-1] {
			return false
		}
		pair := [2]string{toks[i-1], toks[i]}
		if seen[pair] {
			return false
		}
		seen[pair] = true
	}
	return true
}
