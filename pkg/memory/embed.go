// Deterministic lexical text embeddings for the memory indexer.
//
// EmbedText maps text to a dim-dimensional unit vector using hashed
// word tokens (FNV-1a). It is lexical, not semantic: it measures token
// overlap, which is enough to bias QueryContext's seed ranking toward
// code that mentions the query's vocabulary. It is deterministic
// across runs, needs no model and no network, and is explicitly not a
// substitute for a learned embedding model — swap the vectors on
// CodeNode.Embedding with model embeddings whenever one is available.
package memory

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// EmbedText returns the L2-normalized hashed bag-of-words embedding of
// text. It returns nil for non-positive dim. The empty text maps to
// the zero vector.
func EmbedText(text string, dim int) []float32 {
	if dim <= 0 {
		return nil
	}
	v := make([]float32, dim)
	for _, tok := range wordTokens(text) {
		h := fnv.New32a()
		_, _ = h.Write([]byte(tok))
		v[h.Sum32()%uint32(dim)]++
	}
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm == 0 {
		return v
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] = float32(float64(v[i]) / norm)
	}
	return v
}

// wordTokens lowercases text and splits it into alphanumeric tokens,
// keeping digits (identifiers like "utf8" stay whole). Tokens of length
// 1 are dropped as noise.
func wordTokens(text string) []string {
	var toks []string
	var sb strings.Builder
	flush := func() {
		if sb.Len() > 1 {
			toks = append(toks, sb.String())
		}
		sb.Reset()
	}
	for _, r := range strings.ToLower(text) {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			sb.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return toks
}
