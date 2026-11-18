package chat

// The typo normalizer: a deterministic seat in front of the
// social model that repairs a misspelled word before the
// model reads the line — John approved it 2026-11-10 as
// lever 2 of the robustness work, after the frozen
// robustness set (pretrain pilot) measured the production
// model at 9/40 on typo'd inputs: a word the model has never
// seen encodes as UNK, and one UNK in the wrong place can
// derail the whole reply. The seat gives the encoder the
// word the user meant, when — and only when — the evidence
// is unambiguous.
//
// Placement: the seat sits inside the social input
// composition (SocialInput and SocialInputSituated in
// conversation.go / situation.go), on the user's CURRENT
// line only, before composition. Every deterministic seat
// has already declined the line by the time anything is fed
// to the model, so nothing the book, the grammars, or the
// fact seat would have claimed is ever rewritten — the seat
// runs structurally after them. The situation line is
// system text prepended to the composed input; it never
// passes through here. Quoted history inside a composed
// input is not touched either: each turn is normalized on
// its own turn, as it arrives.
//
// The rules are deliberately narrow — a correction the seat
// cannot defend does not happen:
//
//   - a token is a candidate only when the social model's
//     own tokenizer would see it as one token (the
//     tokenizer's split rules), it is purely letters, it is
//     at least 4 letters long, and it is out of the model's
//     vocabulary;
//   - a candidate is replaced only when exactly one
//     vocabulary word is its unique nearest neighbour under
//     Damerau–Levenshtein distance — at most 1 edit for a
//     4–6 letter token, at most 2 for 7 letters or more.
//     Any tie for nearest, or no neighbour inside the
//     limit, leaves the token alone;
//   - words the user has taught the fact store (a pet's
//     name — bella must survive) and names declared in the
//     project index are protected: they are never replaced,
//     however close a vocabulary word sits;
//   - a line needing more than two replacements is not a
//     typo'd sentence, it is a different sentence: it passes
//     through completely untouched.
//
// The seat fails open everywhere: no model loaded, the
// switch off, any store unreadable — the line goes to the
// model exactly as typed. Corrections are silent in the
// reply; they are recorded (LastTypoFixes) and, with the
// seat in trace mode (GOLLEMER_TYPO_NORMALIZER=trace), they
// reach the debug log the chat already keeps behind its
// debug flag. GOLLEMER_TYPO_NORMALIZER=off disables the
// seat — the measurement harness's control position.

import (
	"log"
	"os"
	"strings"
	"sync"
	"unicode"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// typoNormalizerSetting is the seat's switch, pace.go style:
// unset is on, "trace" is on with the debug log line, "off"
// is off.
var typoNormalizerSetting = os.Getenv("GOLLEMER_TYPO_NORMALIZER")

// TypoFix records one replacement the seat made.
type TypoFix struct {
	From string
	To   string
}

var (
	typoMu      sync.Mutex
	typoVocab   *vocab.Vocabulary
	typoTargets []string
	typoFixes   []TypoFix

	typoIndexMu     sync.Mutex
	typoIndexProj   *analyze.Project
	typoIndexSet    map[string]bool
	typoIndexCached bool
)

// armTypoNormalizer hands the seat the vocabulary of the
// social model a chat loop has loaded. Called from the
// loops' own load sites (RunRealChat, runUnifiedChat); a
// loop that never loads a social model never arms the seat,
// and the seat then passes every line through untouched.
func armTypoNormalizer(m *seq2seq.Seq2Seq) {
	if m == nil || m.Tokenizer == nil || m.Tokenizer.Vocabulary == nil {
		return
	}
	v := m.Tokenizer.Vocabulary
	targets := typoTargetWords(v)
	typoMu.Lock()
	typoVocab = v
	typoTargets = targets
	typoMu.Unlock()
}

// LastTypoFixes returns the replacements the seat has made
// in this process, in order — the debug surface for a seat
// that is silent in the reply itself.
func LastTypoFixes() []TypoFix {
	typoMu.Lock()
	defer typoMu.Unlock()
	return append([]TypoFix(nil), typoFixes...)
}

// normalizeSocialCurrent applies the seat to the user's
// current line on its way into the social input
// composition. Any doubt — switch off, no model armed —
// returns the line byte-identical.
func normalizeSocialCurrent(current string) string {
	if typoNormalizerSetting == "off" {
		return current
	}
	typoMu.Lock()
	v, targets := typoVocab, typoTargets
	typoMu.Unlock()
	if v == nil {
		return current
	}
	out, fixes := normalizeLine(current, v, targets, typoProtectedWords())
	if len(fixes) == 0 {
		return current
	}
	typoMu.Lock()
	typoFixes = append(typoFixes, fixes...)
	typoMu.Unlock()
	if typoNormalizerSetting == "trace" {
		for _, f := range fixes {
			log.Printf("[TYPO-NORMALIZER] %q heard as %q", f.From, f.To)
		}
	}
	return out
}

// NormalizeSocialLine applies the seat's core to one line
// against an explicit vocabulary, with no protected words —
// the frozen-eval scorer's handle on the seat. The chat's
// own seat adds the fact-store and project-index protections
// from session state, which frozen offline prompts do not
// carry; everything else (candidate rules, distance table,
// the two-per-line cap) is the same code.
func NormalizeSocialLine(line string, v *vocab.Vocabulary) string {
	if v == nil {
		return line
	}
	out, _ := normalizeLine(line, v, typoTargetWords(v), nil)
	return out
}

// typoTargetWords lists the vocabulary words a typo may be
// corrected TO: purely alphabetic words. Punctuation tokens,
// the special tokens, and anything with a digit or a symbol
// inside are never a correction target.
func typoTargetWords(v *vocab.Vocabulary) []string {
	var out []string
	for w := range v.WordToToken {
		if isAlphaToken(w) {
			out = append(out, w)
		}
	}
	return out
}

// isAlphaToken reports whether s is one or more ASCII
// letters and nothing else.
func isAlphaToken(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < 'a' || s[i] > 'z' {
			return false
		}
	}
	return true
}

// typoMaxDist is the distance table: how many edits a
// correction may cost, by the typo'd token's length. One
// edit covers the observed typo shapes (a dropped letter, a
// swapped pair, one wrong letter) for short words; longer
// words earn a second edit because they have more letters
// to go wrong in.
func typoMaxDist(n int) int {
	if n >= 7 {
		return 2
	}
	return 1
}

// tokenSpan is one word token of a line, as the model's
// tokenizer would cut it: [start, end) byte offsets.
type tokenSpan struct {
	start, end int
}

// wordSpans cuts a line into word tokens with the model's
// own split rules (tokenizer.Tokenize): whitespace and
// {}[],:"?!. separate, each punctuation character standing
// alone. Only the word runs come back — punctuation and
// whitespace stay part of the line's untouched fabric, so a
// replacement rewrites exactly the word's bytes.
func wordSpans(line string) []tokenSpan {
	runes := []rune(line)
	// Byte offset of each rune, so spans address the string.
	offs := make([]int, len(runes)+1)
	pos := 0
	for i, r := range runes {
		offs[i] = pos
		pos += len(string(r))
	}
	offs[len(runes)] = pos
	isSep := func(r rune) bool {
		return unicode.IsSpace(r) || strings.ContainsRune(`{}[],:"?!.`, r)
	}
	var spans []tokenSpan
	i := 0
	for i < len(runes) {
		if isSep(runes[i]) {
			i++
			continue
		}
		start := i
		for i < len(runes) && !isSep(runes[i]) {
			i++
		}
		spans = append(spans, tokenSpan{offs[start], offs[i]})
	}
	return spans
}

// normalizeLine is the seat's core: it returns the line with
// its defensible typos corrected, and the fixes it made. A
// line with more than two defensible fixes comes back
// untouched with no fixes — it was never a typo'd sentence.
func normalizeLine(line string, v *vocab.Vocabulary, targets []string, protected map[string]bool) (string, []TypoFix) {
	spans := wordSpans(line)
	type pending struct {
		span tokenSpan
		from string
		to   string
	}
	var fixes []pending
	for _, sp := range spans {
		tok := line[sp.start:sp.end]
		low := strings.ToLower(tok)
		if len(low) < 4 || !isAlphaToken(low) {
			continue
		}
		if _, ok := v.WordToToken[low]; ok {
			continue
		}
		if protected[low] {
			continue
		}
		to, ok := typoUniqueNearest(low, targets)
		if !ok {
			continue
		}
		fixes = append(fixes, pending{sp, tok, to})
		if len(fixes) > 2 {
			return line, nil
		}
	}
	if len(fixes) == 0 {
		return line, nil
	}
	var b strings.Builder
	last := 0
	out := make([]TypoFix, 0, len(fixes))
	for _, f := range fixes {
		b.WriteString(line[last:f.span.start])
		b.WriteString(f.to)
		last = f.span.end
		out = append(out, TypoFix{From: f.from, To: f.to})
	}
	b.WriteString(line[last:])
	return b.String(), out
}

// typoUniqueNearest finds tok's nearest neighbour among the
// target words under Damerau–Levenshtein distance. It
// answers only when one word is strictly nearest and inside
// the length's distance limit — a tie, however close, is
// two defensible readings, and the seat takes neither.
func typoUniqueNearest(tok string, targets []string) (string, bool) {
	maxD := typoMaxDist(len(tok))
	best := maxD + 1
	bestWord := ""
	tied := false
	for _, w := range targets {
		if d := len(w) - len(tok); d > maxD || -d > maxD {
			continue
		}
		dist := damerauLevenshtein(tok, w)
		if dist > maxD {
			continue
		}
		switch {
		case dist < best:
			best, bestWord, tied = dist, w, false
		case dist == best:
			tied = true
		}
	}
	if bestWord == "" || tied {
		return "", false
	}
	return bestWord, true
}

// damerauLevenshtein is the true Damerau–Levenshtein
// distance (Lowrance–Wagner): insertions, deletions,
// substitutions, and transpositions of adjacent letters —
// the swap being the classic typed typo ("gonig" for
// "going"), which plain Levenshtein prices at two edits and
// this metric at one. Inputs here are short lowercase ASCII
// words; the full matrix is the honest price.
func damerauLevenshtein(a, b string) int {
	la, lb := len(a), len(b)
	inf := la + lb
	// d carries a sentinel row and column (index 0) around
	// the usual (la+1)x(lb+1) grid.
	d := make([][]int, la+2)
	for i := range d {
		d[i] = make([]int, lb+2)
	}
	d[0][0] = inf
	for i := 0; i <= la; i++ {
		d[i+1][0] = inf
		d[i+1][1] = i
	}
	for j := 0; j <= lb; j++ {
		d[0][j+1] = inf
		d[1][j+1] = j
	}
	lastRow := map[byte]int{}
	for i := 1; i <= la; i++ {
		lastMatchCol := 0
		for j := 1; j <= lb; j++ {
			i1 := lastRow[b[j-1]]
			j1 := lastMatchCol
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
				lastMatchCol = j
			}
			d[i+1][j+1] = minInt4(
				d[i][j]+cost,   // substitution
				d[i+1][j]+1,    // insertion
				d[i][j+1]+1,    // deletion
				d[i1][j1]+(i-i1-1)+1+(j-j1-1), // transposition
			)
		}
		lastRow[a[i-1]] = i
	}
	return d[la+1][lb+1]
}

func minInt4(a, b, c, d int) int {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	if d < m {
		m = d
	}
	return m
}

// typoProtectedWords builds the never-replace set from the
// session's own state: the words of every fact the user has
// taught the long-term store (values first — a pet's name
// lives there — plus labels, answers, and the facts' key
// subjects), and the names the project index declares. It
// fails open: an unreadable store or no project simply
// protects less, never more.
func typoProtectedWords() map[string]bool {
	set := map[string]bool{}
	addWords := func(s string) {
		for _, w := range strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
			return !unicode.IsLetter(r)
		}) {
			set[w] = true
		}
	}
	d, _ := loadMemorySlice(memoryCurrentRoot())
	for key, f := range d.Facts {
		addWords(f.Value)
		addWords(f.Label)
		addWords(f.Answer)
		if _, subject, ok := strings.Cut(key, ":"); ok {
			addWords(subject)
		}
	}
	for w := range typoIndexWords() {
		set[w] = true
	}
	return set
}

// typoIndexWords returns the project index's declared names
// as protected words: package names, function and method
// names, type names, and package-level variables — each as
// written (lowercased) and split into its word parts, so
// both "pricetag" and "price"/"tag" readings of a declared
// PriceTag are protected. Cached against the analyzed
// project; a new analysis rebuilds it.
func typoIndexWords() map[string]bool {
	proj := lastAnalyzeProject
	typoIndexMu.Lock()
	defer typoIndexMu.Unlock()
	if typoIndexCached && typoIndexProj == proj {
		return typoIndexSet
	}
	set := map[string]bool{}
	if proj != nil {
		addName := func(name string) {
			low := strings.ToLower(name)
			set[low] = true
			for _, part := range splitIdentParts(name) {
				set[part] = true
			}
		}
		for _, pkg := range proj.Packages {
			addName(pkg.Name)
			for _, fn := range pkg.Funcs {
				addName(fn.Name)
				if fn.Receiver != "" {
					addName(fn.Receiver)
				}
			}
			for _, t := range pkg.Types {
				addName(t.Name)
			}
			for _, vr := range pkg.Vars {
				addName(vr.Name)
			}
		}
	}
	typoIndexProj = proj
	typoIndexSet = set
	typoIndexCached = true
	return set
}

// splitIdentParts splits an identifier into its lowercase
// word parts: camel humps and underscores are the seams,
// and runs of capitals stay together until the last one
// starts a new word (HTTPServer -> http, server).
func splitIdentParts(name string) []string {
	var parts []string
	var cur []rune
	flush := func() {
		if len(cur) > 0 {
			parts = append(parts, strings.ToLower(string(cur)))
			cur = cur[:0]
		}
	}
	runes := []rune(name)
	for i, r := range runes {
		switch {
		case r == '_':
			flush()
		case unicode.IsUpper(r) && len(cur) > 0:
			prev := runes[i-1]
			var next rune
			if i+1 < len(runes) {
				next = runes[i+1]
			}
			if unicode.IsLower(prev) || unicode.IsDigit(prev) ||
				(unicode.IsUpper(prev) && next != 0 && unicode.IsLower(next)) {
				flush()
			}
			cur = append(cur, r)
		default:
			cur = append(cur, r)
		}
	}
	flush()
	return parts
}
