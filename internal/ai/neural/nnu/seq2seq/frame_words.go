package seq2seq

// Layers pilot (DESIGN §12): the frame's TOPIC slot is the
// input's content words, carried by construction. Content is
// defined against the closed class: FunctionWords below is the
// registered function-word set (articles, pronouns, auxiliaries,
// prepositions, conjunctions, wh-words, and discourse particles —
// a closed class of English, stated as a linguistic given like
// the frozen mutator's smaller drop set {the,a,an,to,of,in,on,
// for}). A content word is any other token containing a letter.
// The single implementation lives here: the frame's topic record,
// the round-trip topic diagnostic, and the combination-holdout
// tagging (chat package) all read it the same way.

import "unicode"

// FunctionWords is the registered closed-class set. Keys are
// lowercase surface forms (inputs are lowercased before use).
var FunctionWords = map[string]bool{
	// articles / determiners
	"the": true, "a": true, "an": true, "some": true, "any": true,
	"no": true, "every": true, "each": true, "this": true, "that": true,
	"these": true, "those": true,
	// pronouns
	"i": true, "you": true, "he": true, "she": true, "it": true,
	"we": true, "they": true, "me": true, "him": true, "her": true,
	"us": true, "them": true, "my": true, "your": true, "his": true,
	"its": true, "our": true, "their": true, "mine": true, "yours": true,
	"myself": true, "yourself": true,
	// auxiliaries / modals / copula
	"am": true, "is": true, "are": true, "was": true, "were": true,
	"be": true, "been": true, "being": true, "do": true, "does": true,
	"did": true, "have": true, "has": true, "had": true, "can": true,
	"could": true, "will": true, "would": true, "shall": true,
	"should": true, "may": true, "might": true, "must": true,
	"im": true, "ive": true, "youre": true, "dont": true, "cant": true,
	"wont": true, "isnt": true, "arent": true, "didnt": true,
	// prepositions
	"to": true, "of": true, "in": true, "on": true, "for": true,
	"at": true, "by": true, "with": true, "about": true, "from": true,
	"as": true, "into": true, "like": true, "through": true,
	"after": true, "over": true, "between": true, "out": true,
	"against": true, "during": true, "without": true, "before": true,
	"under": true, "around": true, "among": true, "up": true,
	"down": true, "off": true,
	// conjunctions / wh / particles
	"and": true, "but": true, "or": true, "so": true, "yet": true,
	"nor": true, "if": true, "then": true, "than": true, "when": true,
	"where": true, "who": true, "whom": true, "whose": true,
	"which": true, "what": true, "why": true, "how": true,
	"not": true, "just": true, "really": true, "very": true,
	"too": true, "also": true, "there": true, "here": true,
	"yes": true, "ok": true, "okay": true, "well": true,
}

// IsContentWord reports whether a (lowercased) surface token is a
// content word: it contains at least one letter and is not in the
// closed-class set.
func IsContentWord(w string) bool {
	if FunctionWords[w] {
		return false
	}
	for _, r := range w {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// ContentWordsOf returns the content words of a word sequence,
// deduplicated, in first-appearance order.
func ContentWordsOf(words []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, w := range words {
		if IsContentWord(w) && !seen[w] {
			seen[w] = true
			out = append(out, w)
		}
	}
	return out
}
