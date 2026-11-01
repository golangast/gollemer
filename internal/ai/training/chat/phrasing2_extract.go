package chat

// Arm B: classify + extract. See phrasing2.go for the design.
//
// The route has three deterministic stages around one small
// learned one:
//
//  1. SEGMENT: strip the courtesy wrappers, then split the
//     line on the closed glue vocabulary (the generator's glue
//     set, train and held-out alike — a hand-written splitter
//     is allowed to know the glue words; what it must never do
//     is guess a boundary). One fragment = a single ask, two =
//     a compound, anything else = no claim. Quoted spans and
//     multiword concept names containing "and" are protected
//     spans, never split points.
//  2. CLASSIFY each fragment with the intent-lineage softmax
//     (phrasingClassifier, gated at the intent layer's own
//     confidence/margin bar). "none" anywhere = no claim.
//  3. EXTRACT the family's slots from the fragment's own
//     tokens — files by their .go tail, texts by their quotes,
//     names by their markers (named/called/name is, handler,
//     like, calls…), the rest by the family's marker geometry —
//     render the canonical sentence, and require it to parse
//     through the real parsers with the classified family
//     before it may leave. A slot that cannot be pointed at in
//     the fragment is not invented: extraction fails, the
//     route fails, the line falls through exactly as today.

import (
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// PhrasingFragmentText normalizes a fragment (or a training
// fragment example) into the classifier's input form: wrappers
// stripped, tokens verbatim, single-spaced. Training and
// serving share this function so the classifier sees one
// normal form on both sides.
func PhrasingFragmentText(line string) string {
	return newFrag(PhrasingStripWrapper(line)).text()
}

// frag is one fragment as token streams: surface spelling and
// lowercase, index-aligned.
type frag struct {
	toks []string
	low  []string
}

// phrasingGerunds maps a wrapper-gerunded leading verb back
// to its base form ("would you mind copying ..." strips to
// "copying ...", and every downstream reader — classifier
// features, marker geometry — speaks base forms). The set is
// the generator's own gerund vocabulary.
var phrasingGerunds = map[string]string{
	"adding": "add", "placing": "place", "inserting": "insert",
	"getting": "get", "creating": "create", "registering": "register",
	"copying": "copy", "cloning": "clone", "having": "have",
	"making": "make", "putting": "put", "sticking": "stick",
	"wrapping": "wrap", "serving": "serve", "building": "build",
}

func newFrag(text string) frag {
	toks := seq2seq.VerbatimTokens(text)
	low := make([]string, len(toks))
	for i, t := range toks {
		low[i] = strings.ToLower(t)
	}
	if len(toks) > 0 {
		if base, ok := phrasingGerunds[low[0]]; ok {
			toks[0] = base
			low[0] = base
		}
	}
	return frag{toks: toks, low: low}
}

func (f frag) text() string { return strings.Join(f.toks, " ") }

func isCapTok(s string) bool {
	if s == "" {
		return false
	}
	c := s[0]
	return c >= 'A' && c <= 'Z'
}

// fileSlot returns the first .go token — the file slot every
// file-bearing family shares.
func (f frag) fileSlot() (string, int, bool) {
	for i, t := range f.low {
		if strings.HasSuffix(t, ".go") {
			return f.toks[i], i, true
		}
	}
	return "", -1, false
}

// quotedText returns the tokens between the first pair of
// quote tokens, joined as the user typed them.
func (f frag) quotedText() (string, bool) {
	open := -1
	for i, t := range f.toks {
		if t != `"` {
			continue
		}
		if open < 0 {
			open = i
			continue
		}
		if i > open+1 {
			return strings.Join(f.toks[open+1:i], " "), true
		}
		return "", false
	}
	return "", false
}

// markerValue returns the token after the first occurrence of
// any marker word ("named", "called", ...).
func (f frag) markerValue(markers ...string) (string, int, bool) {
	for i, t := range f.low {
		for _, m := range markers {
			if t == m && i+1 < len(f.toks) {
				return f.toks[i+1], i + 1, true
			}
		}
	}
	return "", -1, false
}

// inFuncSlot reads the "in <Func> to <file>" qualifier that
// statement and print fragments share: the token between "in"
// and the preposition that introduces the file. It is usually
// capitalized — but the compound convention's function is
// literally "main", so that name is accepted too.
func (f frag) inFuncSlot(fileIdx int) (string, bool) {
	for i := 0; i+2 < len(f.toks); i++ {
		if f.low[i] == "in" && (isCapTok(f.toks[i+1]) || f.low[i+1] == "main") &&
			(f.low[i+2] == "to" || f.low[i+2] == "into") && i+3 == fileIdx {
			return f.toks[i+1], true
		}
	}
	return "", false
}

// ---------- segmentation ----------

// phrasingGlues is the closed glue vocabulary, longest first:
// token sequences that join two asks into one breath. Derived
// from the generator's glue set (both dialects — the splitter
// is deterministic code, and a boundary it cannot name is a
// boundary it does not invent).
var phrasingGlues = []struct {
	toks   []string
	haveIt bool
}{
	{[]string{"and", "have", "it"}, true},
	{[]string{"then", "have", "it"}, true},
	{[]string{"and", "after", "that"}, false},
	{[]string{"and", "then"}, false},
	{[]string{"and", "next"}, false},
	{[]string{"and", "also"}, false},
	{[]string{"plus"}, false},
	{[]string{"and"}, false},
}

// protectedSpans marks token ranges a glue may not split
// inside: quoted spans, and multiword concept names that
// contain "and" (the book's own phrases, e.g. "strings and
// runes").
func protectedSpans(f frag) []bool {
	prot := make([]bool, len(f.toks))
	inQuote := false
	for i, t := range f.toks {
		if t == `"` {
			prot[i] = true
			inQuote = !inQuote
			continue
		}
		if inQuote {
			prot[i] = true
		}
	}
	for _, e := range conceptEntries() {
		for _, name := range e.Names {
			nt := strings.Fields(strings.ToLower(name))
			if len(nt) < 2 {
				continue
			}
			hasAnd := false
			for _, w := range nt {
				if w == "and" {
					hasAnd = true
				}
			}
			if !hasAnd {
				continue
			}
			for i := 0; i+len(nt) <= len(f.toks); i++ {
				match := true
				for j := range nt {
					if f.low[i+j] != nt[j] {
						match = false
						break
					}
				}
				if match {
					for j := range nt {
						prot[i+j] = true
					}
				}
			}
		}
	}
	return prot
}

// splitSentences splits a stripped line at ". " boundaries
// outside quotes — the generator's two-sentences glue, which
// the token stream itself erases.
func splitSentences(s string) []string {
	var parts []string
	inQuote := false
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '"' {
			inQuote = !inQuote
			continue
		}
		if inQuote || s[i] != '.' {
			continue
		}
		if i+2 < len(s) && s[i+1] == ' ' && isLetterByte(s[i+2]) {
			parts = append(parts, s[start:i])
			start = i + 2
			i++
		}
	}
	return append(parts, s[start:])
}

func isLetterByte(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// segmentLine splits a raw line into 1 or 2 fragments.
// ok=false when the line carries more than one boundary (a
// three-piece breath is not a shape the layer serves) or no
// content at all.
func segmentLine(line string) (frags []frag, haveIt bool, ok bool) {
	s := PhrasingStripWrapper(line)
	if s == "" {
		return nil, false, false
	}
	var out []frag
	for _, part := range splitSentences(s) {
		f := newFrag(part)
		if len(f.toks) == 0 {
			continue
		}
		prot := protectedSpans(f)
		type split struct {
			at, n  int
			haveIt bool
		}
		var splits []split
		for i := 0; i < len(f.toks); {
			hit := -1
			var hitGlue struct {
				toks   []string
				haveIt bool
			}
			for _, g := range phrasingGlues {
				if i+len(g.toks) > len(f.toks) {
					continue
				}
				match := true
				for j := range g.toks {
					if f.low[i+j] != g.toks[j] || prot[i+j] {
						match = false
						break
					}
				}
				if match {
					hit = i
					hitGlue = g
					break
				}
			}
			if hit < 0 {
				i++
				continue
			}
			splits = append(splits, split{at: hit, n: len(hitGlue.toks), haveIt: hitGlue.haveIt})
			i = hit + len(hitGlue.toks)
		}
		if len(splits) == 0 {
			out = append(out, f)
			continue
		}
		prev := 0
		for _, sp := range splits {
			out = append(out, fragFromToks(f.toks[prev:sp.at]))
			if sp.haveIt {
				haveIt = true
			}
			prev = sp.at + sp.n
		}
		out = append(out, fragFromToks(f.toks[prev:]))
	}
	// Empty fragments (a leading/trailing glue) carry no ask.
	var kept []frag
	for _, fr := range out {
		if len(fr.toks) > 0 {
			kept = append(kept, fr)
		}
	}
	if len(kept) == 0 || len(kept) > 2 {
		return nil, false, false
	}
	return kept, haveIt, true
}

func fragFromToks(toks []string) frag {
	cp := append([]string{}, toks...)
	low := make([]string, len(cp))
	for i, t := range cp {
		low[i] = strings.ToLower(t)
	}
	return frag{toks: cp, low: low}
}

// ---------- canonical rendering ----------

// phrasingArticle mirrors the generator's article rule (the
// canonical the user sees should read the way the family's own
// sentences read).
func phrasingArticle(word string) string {
	if word != "" && strings.ContainsRune("aeiou", rune(word[0])) {
		return "an"
	}
	return "a"
}

func renderConceptStmt(e analyze.ConceptEntry, fn, file string) string {
	s := "add " + phrasingArticle(e.Canonical()) + " " + e.Canonical()
	if fn != "" {
		s += " in " + fn
	}
	return s + " to " + file
}

func renderConceptDecl(e analyze.ConceptEntry, name, like, file string) string {
	s := "add " + phrasingArticle(e.Canonical()) + " " + e.Canonical() + " named " + name
	if like != "" {
		s += " like " + like
	}
	return s + " to " + file
}

func renderPrint(text, fn, file string) string {
	s := "add a print line saying \"" + text + "\""
	if fn != "" {
		s += " in " + fn
	}
	return s + " to " + file
}

func renderInstantiate(typ, fn, varName string) string {
	s := "add instantiation of struct " + typ + " to handler " + fn
	if varName != "" {
		s += " named " + varName
	}
	return s
}

// ---------- per-family extraction ----------
//
// Every extractor returns the rendered canonical and the slot
// map it bound (for the compound inheritance rule). ok=false
// when a required slot cannot be pointed at in the fragment.

func extractConceptStmt(f frag, e analyze.ConceptEntry) (string, map[string]string, bool) {
	file, fi, ok := f.fileSlot()
	if !ok {
		return "", nil, false
	}
	slots := map[string]string{"target": file}
	if fn, ok := f.inFuncSlot(fi); ok {
		slots["func"] = fn
	}
	return renderConceptStmt(e, slots["func"], file), slots, true
}

func extractConceptDecl(f frag, e analyze.ConceptEntry) (string, map[string]string, bool) {
	file, _, ok := f.fileSlot()
	if !ok {
		return "", nil, false
	}
	slots := map[string]string{"target": file}
	if like, _, ok := f.markerValue("like"); ok {
		slots["like"] = like
	}
	name, _, named := f.markerValue("named", "called")
	if !named {
		// "name is X" / "with the name X": the token after
		// "name" (skipping "is").
		for i, t := range f.low {
			if t == "name" && i+1 < len(f.toks) {
				cand := f.toks[i+1]
				if cand == "is" && i+2 < len(f.toks) {
					cand = f.toks[i+2]
				}
				if isCapTok(cand) {
					name, named = cand, true
				}
				break
			}
		}
	}
	if !named {
		// Fallback: the one capitalized token that is neither
		// the file, nor the like value, nor part of the
		// concept's own name.
		var caps []string
		nameToks := map[string]bool{}
		for _, w := range strings.Fields(strings.ToLower(e.Canonical())) {
			nameToks[w] = true
		}
		for i, t := range f.toks {
			if !isCapTok(t) || t == file || t == slots["like"] || nameToks[f.low[i]] {
				continue
			}
			caps = append(caps, t)
		}
		if len(caps) == 1 {
			name, named = caps[0], true
		}
	}
	if !named {
		return "", nil, false
	}
	slots["name"] = name
	return renderConceptDecl(e, name, slots["like"], file), slots, true
}

func extractPrint(f frag) (string, map[string]string, bool) {
	text, ok := f.quotedText()
	if !ok {
		// The print family's other shape: a bare print line
		// with no saying — the book's print-line row ("add a
		// print line in main to jim/main.go"), whose canonical
		// parses back as THIS family with no text slot. It
		// counts only when the fragment actually NAMES the row
		// (the scan's naming-evidence rule, applied here too):
		// without that check, this family's extractor would
		// "validate" any placed statement fragment a second
		// time and fell compounds by ambiguity.
		if !conceptPhrasePresent(f, "concept-print-line") {
			return "", nil, false
		}
		for _, e := range conceptEntries() {
			if e.ID == "print-line" {
				return extractConceptStmt(f, e)
			}
		}
		return "", nil, false
	}
	file, fi, ok := f.fileSlot()
	if !ok {
		return "", nil, false
	}
	slots := map[string]string{"text": text, "target": file}
	if fn, ok := f.inFuncSlot(fi); ok {
		slots["func"] = fn
	}
	return renderPrint(text, slots["func"], file), slots, true
}

func extractInstantiate(f frag) (string, map[string]string, bool) {
	if _, _, hasFile := f.fileSlot(); hasFile {
		return "", nil, false // an instantiate target is a bare function, never a file
	}
	// Marker evidence: two bare capitalized tokens alone do
	// not make an instantiation ("copying Nomad as Tundra"
	// binds type+handler spuriously without it). Either the
	// instance wording is present, or the fragment has the
	// bare form's shape: an add-family verb up front and a
	// Cap (in|to|into|inside) Cap tail.
	marker := false
	for _, t := range f.low {
		switch t {
		case "instance", "instantiation", "struct", "handler":
			marker = true
		}
	}
	if !marker {
		verbStart := map[string]bool{
			"add": true, "put": true, "make": true, "place": true,
			"create": true, "instantiate": true, "insert": true, "get": true,
		}
		// The tail may carry the var marker ("... in Sift
		// named tendy"); look past it.
		end := len(f.toks)
		if end >= 3 && (f.low[end-2] == "named" || f.low[end-2] == "called") {
			end -= 2
		}
		marker = end >= 2 && verbStart[f.low[0]] &&
			isCapTok(f.toks[end-1]) &&
			(f.low[end-2] == "in" || f.low[end-2] == "to" || f.low[end-2] == "into" || f.low[end-2] == "inside")
	}
	if !marker {
		return "", nil, false
	}
	slots := map[string]string{}
	used := map[int]bool{}
	if v, vi, ok := f.markerValue("named", "called"); ok {
		slots["var"] = v
		used[vi] = true
	}
	// Marker geometry first: "handler X" names the function,
	// "struct X" / "of X" names the type, a trailing
	// in/to/into/inside X names the function.
	for i, t := range f.low {
		if t == "handler" && i+1 < len(f.toks) && isCapTok(f.toks[i+1]) && slots["func"] == "" {
			slots["func"] = f.toks[i+1]
			used[i+1] = true
		}
		if (t == "struct" || t == "of") && i+1 < len(f.toks) && isCapTok(f.toks[i+1]) && slots["type"] == "" {
			slots["type"] = f.toks[i+1]
			used[i+1] = true
		}
	}
	if n := len(f.toks); n >= 2 && slots["func"] == "" {
		switch f.low[n-2] {
		case "in", "to", "into", "inside":
			if isCapTok(f.toks[n-1]) && !used[n-1] {
				slots["func"] = f.toks[n-1]
				used[n-1] = true
			}
		}
	}
	// Whatever capitalized tokens remain fill the open roles
	// in order: type first, then function.
	for i, t := range f.toks {
		if !isCapTok(t) || used[i] {
			continue
		}
		used[i] = true
		if slots["type"] == "" {
			slots["type"] = t
		} else if slots["func"] == "" {
			slots["func"] = t
		}
	}
	if slots["type"] == "" || slots["func"] == "" {
		return "", nil, false
	}
	return renderInstantiate(slots["type"], slots["func"], slots["var"]), slots, true
}

func extractDelegate(f frag) (string, map[string]string, bool) {
	file, _, ok := f.fileSlot()
	if !ok {
		return "", nil, false
	}
	slots := map[string]string{"file": file}
	callee := ""
	for i, t := range f.low {
		if (t == "calls" || t == "calling") && i+1 < len(f.toks) {
			callee = f.toks[i+1]
			break
		}
	}
	if callee == "" && len(f.toks) > 1 && f.low[0] == "wrap" {
		callee = f.toks[1] // "wrap <Callee> in <file>"
	}
	if callee == "" {
		return "", nil, false
	}
	slots["source"] = callee
	// Wrapper: "function <Name> in ..." or a named/called
	// marker on a wrapper word.
	for i, t := range f.low {
		if t == "function" && i+1 < len(f.toks) && isCapTok(f.toks[i+1]) {
			slots["wrapper"] = f.toks[i+1]
			break
		}
	}
	if slots["wrapper"] == "" {
		if w, _, ok := f.markerValue("named", "called"); ok {
			slots["wrapper"] = w
		}
	}
	canon := "add a function in " + file + " that calls " + callee
	if w := slots["wrapper"]; w != "" {
		canon = "add a function " + w + " in " + file + " that calls " + callee
	}
	return canon, slots, true
}

func extractClone(f frag) (string, map[string]string, bool) {
	if _, _, hasFile := f.fileSlot(); hasFile {
		return "", nil, false // clone names functions, never files
	}
	// (source, new) by the family's own shapes.
	for i, t := range f.low {
		if (t == "copy" || t == "clone") && i+1 < len(f.toks) && isCapTok(f.toks[i+1]) {
			src := f.toks[i+1]
			for j := i + 2; j+1 < len(f.toks); j++ {
				if (f.low[j] == "as" || f.low[j] == "into") && isCapTok(f.toks[j+1]) {
					return "copy " + src + " as " + f.toks[j+1],
						map[string]string{"sibling": src, "new": f.toks[j+1]}, true
				}
				if f.low[j] == "called" && isCapTok(f.toks[j+1]) {
					return "copy " + src + " as " + f.toks[j+1],
						map[string]string{"sibling": src, "new": f.toks[j+1]}, true
				}
			}
		}
		if t == "copy" && i+2 < len(f.toks) && f.low[i+1] == "of" && isCapTok(f.toks[i+2]) {
			// "make a copy of <Fn> called <New>"
			if nw, _, ok := f.markerValue("called", "named"); ok {
				return "copy " + f.toks[i+2] + " as " + nw,
					map[string]string{"sibling": f.toks[i+2], "new": nw}, true
			}
		}
		if t == "duplicate" && i+1 < len(f.toks) && isCapTok(f.toks[i+1]) {
			src := f.toks[i+1]
			if nw, _, ok := f.markerValue("called"); ok {
				return "copy " + src + " as " + nw, map[string]string{"sibling": src, "new": nw}, true
			}
			for j := i + 2; j+2 < len(f.toks); j++ {
				if f.low[j] == "call" && f.low[j+1] == "it" && isCapTok(f.toks[j+2]) {
					return "copy " + src + " as " + f.toks[j+2],
						map[string]string{"sibling": src, "new": f.toks[j+2]}, true
				}
			}
		}
		if t == "like" && i+1 < len(f.toks) && isCapTok(f.toks[i+1]) {
			if nw, _, ok := f.markerValue("named", "called"); ok {
				return "copy " + f.toks[i+1] + " as " + nw,
					map[string]string{"sibling": f.toks[i+1], "new": nw}, true
			}
		}
	}
	return "", nil, false
}

func extractServer(f frag) (string, map[string]string, bool) {
	file, _, ok := f.fileSlot()
	if !ok {
		return "", nil, false
	}
	for _, t := range f.low {
		if t == "webserver" || t == "server" {
			return "add a webserver to " + file, map[string]string{"file": file}, true
		}
	}
	return "", nil, false
}

// extractServeHTML reads a serve-a-page fragment: the recipe's
// two slots are the page (an .html token, or the words "html
// file"/"html page" for the default page) and, when present,
// the .go file every file-bearing family reads the same way.
// The serve evidence is required — a serve/server word must be
// in the fragment — so the concept book's page-adjacent lines
// ("add an html template…", "serve static files…", neither of
// which carries a page this extractor can point at) are never
// read as the recipe.
func extractServeHTML(f frag) (string, map[string]string, bool) {
	serve := false
	for _, t := range f.low {
		switch t {
		case "serve", "serves", "serving", "server", "webserver":
			serve = true
		}
	}
	if !serve {
		return "", nil, false
	}
	html := ""
	for i, t := range f.low {
		if strings.HasSuffix(t, ".html") {
			html = f.toks[i]
			break
		}
	}
	if html == "" {
		for i := 0; i+1 < len(f.low); i++ {
			if f.low[i] == "html" && (f.low[i+1] == "file" || f.low[i+1] == "page") {
				html = "index.html"
				break
			}
		}
	}
	if html == "" {
		return "", nil, false
	}
	slots := map[string]string{"html": html}
	canon := "add a server that serves " + html
	if file, _, ok := f.fileSlot(); ok {
		slots["file"] = file
		canon += " to " + file
	}
	return canon, slots, true
}

// fieldTypeSpan finds a field type in the token stream: the
// closed type vocabulary the add-field family speaks, as
// token sequences (VerbatimTokens splits [] and map brackets
// into their own tokens).
func fieldTypeSpan(f frag) (canon string, lo, hi int, ok bool) {
	type pat struct {
		toks  []string
		canon string
	}
	pats := []pat{
		{[]string{"map", "[", "string", "]", "int"}, "map[string]int"},
		{[]string{"[", "]", "string"}, "[]string"},
		{[]string{"[", "]", "int"}, "[]int"},
		{[]string{"time.duration"}, "time.Duration"},
		{[]string{"float64"}, "float64"},
		{[]string{"string"}, "string"},
		{[]string{"bool"}, "bool"},
		{[]string{"int"}, "int"},
	}
	for _, p := range pats {
		for i := 0; i+len(p.toks) <= len(f.toks); i++ {
			match := true
			for j := range p.toks {
				if f.low[i+j] != p.toks[j] {
					match = false
					break
				}
			}
			if match {
				return p.canon, i, i + len(p.toks), true
			}
		}
	}
	return "", -1, -1, false
}

func extractField(f frag) (string, map[string]string, bool) {
	if _, _, hasFile := f.fileSlot(); hasFile {
		return "", nil, false // a field's target is a bare struct name
	}
	typ, tlo, thi, ok := fieldTypeSpan(f)
	if !ok {
		return "", nil, false
	}
	inTypeSpan := func(i int) bool { return i >= tlo && i < thi }
	// The struct: after "give", or the capitalized token a
	// trailing to/in introduces.
	strct := ""
	for i, t := range f.low {
		if t == "give" && i+1 < len(f.toks) && isCapTok(f.toks[i+1]) {
			strct = f.toks[i+1]
			break
		}
	}
	if strct == "" {
		for i := len(f.toks) - 2; i >= 0; i-- {
			if (f.low[i] == "to" || f.low[i] == "in") && isCapTok(f.toks[i+1]) && !inTypeSpan(i+1) {
				strct = f.toks[i+1]
				break
			}
		}
	}
	// The field: the capitalized token adjacent to "field"
	// (either side), else the remaining capitalized token.
	field := ""
	for i, t := range f.low {
		if t != "field" {
			continue
		}
		if i+1 < len(f.toks) && isCapTok(f.toks[i+1]) && f.toks[i+1] != strct && !inTypeSpan(i+1) {
			field = f.toks[i+1]
		} else if i-1 >= 0 && isCapTok(f.toks[i-1]) && f.toks[i-1] != strct && !inTypeSpan(i-1) {
			field = f.toks[i-1]
		}
		break
	}
	if field == "" {
		var caps []string
		for i, t := range f.toks {
			if isCapTok(t) && t != strct && !inTypeSpan(i) {
				caps = append(caps, t)
			}
		}
		if len(caps) == 1 {
			field = caps[0]
		}
	}
	if strct == "" || field == "" {
		return "", nil, false
	}
	// The struct tag, when the fragment carries one: "with a
	// <kind> tag <value>", riding either field form (after the
	// type, or at the end behind the struct). A "tag" token
	// whose neighbours don't assemble is never silently
	// dropped — a fragment that asked for a tag this extractor
	// can't read is not claimed here. (The rendered canonical
	// still has to survive validatePiece against
	// parseAssembledEdit, so a coincidental "tag" — a struct
	// named Tag — can't invent one either.)
	tagKind, tagValue, tagLit := "", "", ""
	for i, t := range f.low {
		if t != "tag" || i == 0 || i+1 >= len(f.toks) {
			continue
		}
		v := strings.Trim(f.toks[i+1], `"`)
		lit, ok := assembleStructTag(f.toks[i-1], v)
		if !ok {
			return "", nil, false
		}
		tagKind, tagValue, tagLit = f.toks[i-1], v, lit
		break
	}
	slots := map[string]string{"field": field, "type": typ, "struct": strct}
	if tagLit != "" {
		slots["tag"] = tagLit
		return "add a " + field + " field of type " + typ + " with a " + tagKind + " tag " + tagValue + " to " + strct, slots, true
	}
	return "add a " + field + " field of type " + typ + " to " + strct, slots, true
}

func extractCase(f frag) (string, map[string]string, bool) {
	if _, _, hasFile := f.fileSlot(); hasFile {
		return "", nil, false // a case's target is a bare function name
	}
	// "… <Func> … handle <value> …" — the function is the
	// capitalized token before "handle", the value follows it.
	for i, t := range f.low {
		if t != "handle" {
			continue
		}
		if i+1 >= len(f.toks) {
			break
		}
		value := f.toks[i+1]
		for j := i - 1; j >= 0; j-- {
			if isCapTok(f.toks[j]) {
				return "add a case for " + value + " to " + f.toks[j],
					map[string]string{"case": value, "func": f.toks[j]}, true
			}
		}
		return "", nil, false
	}
	// "add a case for <value> to <Func>" — tried FIRST, as a
	// full scan: the bare-case pattern below would otherwise
	// read the article in "a case for ..." as the value.
	for i, t := range f.low {
		if t == "for" && i+1 < len(f.toks) {
			if fn, _, ok := f.markerValue("to"); ok && isCapTok(fn) {
				return "add a case for " + f.toks[i+1] + " to " + fn,
					map[string]string{"case": f.toks[i+1], "func": fn}, true
			}
		}
	}
	// "add a <value> case to <Func>": the value sits before
	// "case" and "to" follows it (in "a case for ..." the word
	// after "case" is "for", which excludes the shape here).
	for i, t := range f.low {
		if t == "case" && i-1 >= 0 && i+1 < len(f.toks) && f.low[i+1] == "to" {
			if fn, _, ok := f.markerValue("to"); ok && isCapTok(fn) {
				return "add a case for " + f.toks[i-1] + " to " + fn,
					map[string]string{"case": f.toks[i-1], "func": fn}, true
			}
		}
	}
	return "", nil, false
}

func extractRegister(f frag) (string, map[string]string, bool) {
	if _, _, hasFile := f.fileSlot(); hasFile {
		return "", nil, false // a register target is a bare table name
	}
	// Verb evidence: every register core carries an ask verb
	// (register/add/put/place/get/record...). Without one, an
	// "X in Y" fragment is prose or a question ("a slice in
	// Go"), never this family.
	verb := false
	for _, t := range f.low {
		switch t {
		case "register", "add", "put", "place", "get", "stick", "insert", "record", "have", "make", "create", "set":
			verb = true
		}
	}
	if !verb {
		return "", nil, false
	}
	// The table: the capitalized token before "table", or the
	// one an in/into introduces at the tail.
	table, tableIdx := "", -1
	for i, t := range f.low {
		if t == "table" && i-1 >= 0 && isCapTok(f.toks[i-1]) {
			table, tableIdx = f.toks[i-1], i-1
			break
		}
	}
	if table == "" {
		for i := len(f.toks) - 2; i >= 0; i-- {
			if (f.low[i] == "in" || f.low[i] == "into") && isCapTok(f.toks[i+1]) {
				table, tableIdx = f.toks[i+1], i+1
				break
			}
		}
	}
	if table == "" {
		return "", nil, false
	}
	// The key: the token just before the preposition that
	// introduces the table ("register K in T", "add K to the
	// T table", "put K in T").
	key := ""
	for i := tableIdx - 1; i >= 0; i-- {
		if f.low[i] == "in" || f.low[i] == "into" || f.low[i] == "to" {
			if i-1 >= 0 {
				key = f.toks[i-1]
			}
			break
		}
		if f.low[i] == "the" {
			continue
		}
		break
	}
	if key == "" || isCapTok(key) {
		return "", nil, false
	}
	return "register " + key + " in " + table,
		map[string]string{"key": key, "table": table}, true
}

// ---------- created-target compounds ----------
//
// A created-target compound is two asks in one breath where
// the second piece lands inside the function the first piece
// creates: "make a handler named Login in web.go and add a
// switch in it". The second piece's placement can be expressed
// three ways — the created name repeated, a pronoun ("in it",
// "in there", "inside it"), or silence — and all three mean
// the same two canonical pieces. The classifier learns to FLAG
// the shape (the fragments train in the family's own label);
// the binding itself is deterministic code, here, never the
// classifier's guess:
//
//   - placement() reads a statement fragment's placement
//     geometry and surfaces an unbound function slot AS unbound
//     (a pronoun, or an absence) — extraction never invents a
//     name to fill it;
//   - createdFuncOf() names the function a routed first piece
//     created, from that piece's OWN extracted slots;
//   - routeCreatedPiece() binds the unbound slot to that name
//     (and an unspoken file to the first piece's file), renders
//     the canonical, and puts it through the same parser
//     validation every other piece passes.
//
// A lone pronoun line ("add a switch in it") has no first
// piece, so nothing binds: its extraction fails for want of a
// file exactly as before, and the line falls through. A first
// piece that creates a TYPE (struct, interface, enum) creates
// nothing a statement can land in, so those compounds do not
// bind either.

// funcPlacement is how one fragment expresses where a
// statement lands: its file, when it names one, and its
// function — named, pronominal, or not expressed at all.
type funcPlacement struct {
	file    string
	hasFile bool
	fn      string // the named function, when one is expressed
	pronoun bool   // "in it" / "in there" / "inside it"
}

// placement reads the fragment's placement geometry WITHOUT
// binding anything: a pronoun target is surfaced as a pronoun
// and an omitted one as an absence — never as an invented
// name. Quoted spans (a print line's text) are protected: a
// "in it" inside the saying is text, not placement.
func (f frag) placement() funcPlacement {
	var p funcPlacement
	if file, _, ok := f.fileSlot(); ok {
		p.file, p.hasFile = file, true
	}
	prot := protectedSpans(f)
	for i := 0; i+1 < len(f.toks); i++ {
		if prot[i] || prot[i+1] {
			continue
		}
		switch f.low[i] {
		case "in", "inside", "into":
			switch {
			case f.low[i+1] == "it" || f.low[i+1] == "there":
				p.pronoun = true
			case !strings.HasSuffix(f.low[i+1], ".go") &&
				(isCapTok(f.toks[i+1]) || f.low[i+1] == "main"):
				if p.fn == "" {
					p.fn = f.toks[i+1]
				}
			}
		}
	}
	return p
}

// createdFuncOf reports the function a routed first piece
// creates, when it creates one a statement can land inside:
// a concept declaration whose book template opens
// "func {name}(" (the function row, the handler and
// middleware family, recursion), or a delegate with a named
// wrapper. The name and file are the piece's own extracted
// slots — words the user typed, carried, never invented.
func createdFuncOf(family string, slots map[string]string) (name, file string, ok bool) {
	if family == "delegate" {
		if w := slots["wrapper"]; w != "" {
			return w, slots["file"], true
		}
		return "", "", false
	}
	if !strings.HasPrefix(family, "concept-") {
		return "", "", false
	}
	id := strings.TrimPrefix(family, "concept-")
	for _, e := range conceptEntries() {
		if e.ID != id {
			continue
		}
		if e.Kind == analyze.ConceptDeclaration &&
			strings.HasPrefix(e.Template, "func {name}(") && slots["name"] != "" {
			return slots["name"], slots["target"], true
		}
		return "", "", false
	}
	return "", "", false
}

// routeCreatedPiece is a created-target compound's second
// half: classify the fragment (the classifier flags the
// shape), read its placement, bind what the placement leaves
// unbound to the function the first piece created, render the
// canonical, and validate it through the real parsers — the
// same gate every routed piece passes. Two placements are NOT
// this path's business and decline: a fully placed fragment
// (its own function AND file — the ordinary route owns it,
// and if that route failed, it failed for another reason) and
// a file-only fragment ("add a switch to main.go" — naming a
// file with no function is the planner's main rule, a reading
// to keep, not an omission to fill with the created name).
func routeCreatedPiece(f frag, createdName, createdFile string) (string, bool) {
	family, gated := phrasingClassifier(f.text())
	p := f.placement()
	if p.hasFile && p.fn != "" && !p.pronoun {
		return "", false // fully placed: not an unbound shape
	}
	if p.hasFile && !p.pronoun && p.fn == "" {
		return "", false // file-only: the planner's main rule
	}
	fn := p.fn
	if p.pronoun || fn == "" {
		fn = createdName
	}
	file := p.file
	if !p.hasFile {
		file = createdFile
	}
	if fn == "" || file == "" {
		return "", false
	}
	if gated && family == "print" {
		text, ok := f.quotedText()
		if !ok {
			return "", false
		}
		slots := map[string]string{"text": text, "func": fn, "target": file}
		canon := renderPrint(text, fn, file)
		if validatePiece(canon, "print", slots) {
			return canon, true
		}
		return "", false
	}
	// A confident non-concept family (add-case above all)
	// is not this path's business: its bare-name target is
	// its own, and the ordinary route owns the fragment.
	if gated && family != "" && family != "none" && !strings.HasPrefix(family, "concept-") {
		return "", false
	}
	// Concept pieces under the binding: the binding
	// supplies a placement, never a concept — the piece's
	// family must be NAMED in the fragment (the
	// naming-evidence rule, see routeFragment). The gated
	// family is the candidate when it is named. Every other
	// classifier outcome — the gate unsure, a gated "none",
	// or a gated concept the fragment never names (the
	// insert-theft shape, "… plus insert a xml inside it"
	// gated to rqlite-insert on the shared verb) — falls to
	// the named-concept scan, and a unique validated
	// rendering under the binding wins: the fallback's
	// discipline, applied on the one path where an unbound
	// placement leaves the binding as the only possible
	// reader.
	var candidates []string
	if gated && strings.HasPrefix(family, "concept-") && conceptPhrasePresent(f, family) {
		candidates = []string{family}
	} else {
		for _, fam := range allFamilies() {
			if strings.HasPrefix(fam, "concept-") && conceptPhrasePresent(f, fam) {
				candidates = append(candidates, fam)
			}
		}
	}
	bound, boundCount := "", 0
	for _, fam := range candidates {
		id := strings.TrimPrefix(fam, "concept-")
		for _, e := range conceptEntries() {
			if e.ID != id || e.Kind != analyze.ConceptStatement {
				continue
			}
			slots := map[string]string{"func": fn, "target": file}
			if canon := renderConceptStmt(e, fn, file); validatePiece(canon, fam, slots) {
				bound, boundCount = canon, boundCount+1
			}
		}
	}
	if boundCount == 1 {
		return bound, true
	}
	return "", false
}

// ---------- shared-target compounds ----------
//
// A shared-target compound states its file once, at the very
// end, for both pieces: "add a for loop in main and a print
// line in main to jim/main.go". Segmentation leaves the file
// inside the second fragment; the first fragment's extraction
// then fails for want of a target and — before this rule —
// the whole compound fell through. The distribution is
// deterministic route work, the created-target binding's
// sibling: when exactly ONE fragment of the two carries a
// file, that file is the compound's shared target, and a
// piece is routed with the target appended (" to <file>") —
// the completed fragment goes through the same classify /
// extract / validate pipeline as any fully placed piece, and
// it is tried BEFORE the bare fragment: a file-less
// fragment's geometry can be stolen by the bare-name families
// (register reads "place a sql transaction in Plumb" as
// registering the key "transaction" in a table "Plumb"), so
// the bare reading is only ever the fallback. When both
// fragments carry files (the same or conflicting ones)
// nothing distributes: each piece keeps the target it
// stated, exactly as before. And only the FILE distributes,
// never a function — see routeSharedPiece for the restraint
// that keeps a bare, home-less statement piece from being
// read as "in main".

// withSharedTarget returns the fragment with the shared file
// appended as its target clause, under the given preposition.
func withSharedTarget(f frag, prep, file string) frag {
	toks := append(append([]string{}, f.toks...), prep, file)
	return fragFromToks(toks)
}

// fileBearingFamily reports whether the family's canonical
// takes a .go file target — the concept families, print,
// delegate, add-server. The bare-name families (instantiate,
// clone, add-case, register) name people and tables, never
// files: appending the compound's file to one of their
// fragments corrupts it ("place a Pumice in Notifier" plus
// " to main.go" reads as a statement ask about a concept
// named Pumice), so the distribution never completes them.
func fileBearingFamily(family string) bool {
	switch {
	case family == "print" || family == "delegate" || family == "add-server" || family == "serve-html":
		return true
	case strings.HasPrefix(family, "concept-"):
		return true
	}
	return false
}

// routeSharedPiece retries one fragment with the compound's
// shared file appended, and applies the distribution's one
// restraint: what distributes is the FILE, never a function.
// A piece whose own placement is complete but for the file —
// it states its function ("in main", "in Jake"), or it is a
// declaration or server piece, whose home is the file itself —
// may take the shared file. A bare statement piece stating NO
// home at all may not: its silence is the created-target
// shape's to read (a creator binds it) or nobody's — reading
// it as "in main" would be inventing a placement the user
// never stated, and behind a TYPE declaration ("add an
// interface named Box to main.go and add a switch") it would
// silently land a statement in a main the user never
// mentioned.
func routeSharedPiece(f frag, file string) (string, string, map[string]string, bool) {
	// The completion is offered on the classifier's own
	// reading of the BARE fragment — that reading is the
	// evidence of what kind of piece this is:
	//   - a file-bearing family (a placed ask missing its
	//     file): complete it, and any validated reading of
	//     the completed fragment counts, subject to the
	//     func restraint below;
	//   - a bare-name family (instantiate, clone, add-case,
	//     register, add-field): its target is a name, and the
	//     completed fragment would ordinarily be a corruption
	//     of it — so the completion is heard only under the
	//     DECLARATION-shaped acceptance below. That is not a
	//     weakening: a bare-name fragment cannot validate as
	//     a concept declaration on its completed form unless
	//     it genuinely names the concept and carries the
	//     name (the naming evidence every concept win needs),
	//     which is precisely a declaration the bare gate
	//     misread. The gate reads the fragment segmentation
	//     stripped of its file — the weakest evidence on the
	//     table — and the 2026-11-01 serve-html retrain
	//     showed it flipping on exactly such a fragment
	//     ("create a session with the name Vespa" gated to
	//     add-field-tag on the "with the name" shape) while
	//     the completed fragment classified and validated
	//     cleanly as its session-handler declaration. A flat
	//     veto let the weakest reading silence the strongest;
	//     the declaration-only acceptance is the restraint
	//     that actually separates the two.
	//   - "none", or the gate itself unsure: complete it,
	//     but accept only a DECLARATION-shaped win (a concept
	//     declaration, an add-server) — the stripped form of
	//     a declaration is a complete ask except its target,
	//     while a statement-shaped win here is the corruption
	//     channel (a bare-name fragment misread through the
	//     appended file as a concept statement).
	bareFam, bareGated := phrasingClassifier(f.text())
	declOnly := !bareGated || bareFam == "none" || !fileBearingFamily(bareFam)
	accept := func(family string, slots map[string]string) bool {
		if declOnly && family != "add-server" && family != "serve-html" {
			isDecl := false
			if strings.HasPrefix(family, "concept-") {
				id := strings.TrimPrefix(family, "concept-")
				for _, e := range conceptEntries() {
					if e.ID == id && e.Kind == analyze.ConceptDeclaration {
						isDecl = true
					}
				}
			}
			if !isDecl {
				return false
			}
		}
		if strings.HasPrefix(family, "concept-") {
			id := strings.TrimPrefix(family, "concept-")
			for _, e := range conceptEntries() {
				if e.ID == id && e.Kind == analyze.ConceptStatement && slots["func"] == "" {
					return false
				}
			}
		}
		return true
	}
	// Two completions are tried: " to <file>" (the canonical
	// target clause) and " in <file>" (the form a stripped
	// declaration core was actually cut from — "create a
	// function with the name Plumb" completes naturally with
	// "in", and the hybrid "…name Plumb to kilo.go" is a
	// surface no dialect speaks, which the classifier reads
	// as none). The acceptance rules above police both: an
	// "in" completion that would read the file as a statement's
	// function dies on the func restraint.
	for _, prep := range []string{"to", "in"} {
		canon, family, slots, ok := routeFragment(withSharedTarget(f, prep, file))
		if ok && accept(family, slots) {
			return canon, family, slots, true
		}
	}
	return "", "", nil, false
}

// sharedFileOf returns the compound's one file when exactly
// one fragment carries it: the file and which fragment lacks
// it. ok is false when both carry one or neither does.
func sharedFileOf(a, b frag) (file string, lacking int, ok bool) {
	_, _, aHas := a.fileSlot()
	bFile, _, bHas := b.fileSlot()
	switch {
	case aHas && !bHas:
		f, _, _ := a.fileSlot()
		return f, 1, true
	case bHas && !aHas:
		return bFile, 0, true
	}
	return "", -1, false
}

// ---------- the route ----------

// extractForFamily runs the family's extractor on a fragment.
func extractForFamily(f frag, family string) (string, map[string]string, bool) {
	if strings.HasPrefix(family, "concept-") {
		id := strings.TrimPrefix(family, "concept-")
		for _, e := range conceptEntries() {
			if e.ID != id {
				continue
			}
			if e.Kind == "statement" {
				return extractConceptStmt(f, e)
			}
			return extractConceptDecl(f, e)
		}
		return "", nil, false
	}
	switch family {
	case "print":
		return extractPrint(f)
	case "instantiate":
		return extractInstantiate(f)
	case "delegate":
		return extractDelegate(f)
	case "clone":
		return extractClone(f)
	case "add-server":
		return extractServer(f)
	case "serve-html":
		return extractServeHTML(f)
	case "add-field", "add-field-tag":
		// One extractor for both: it renders the tagged
		// canonical exactly when the line carries a tag
		// phrase, and validatePiece's family check keeps
		// each family from claiming the other's lines.
		return extractField(f)
	case "add-case":
		return extractCase(f)
	case "register":
		return extractRegister(f)
	}
	return "", nil, false
}

// validatePiece renders-check: the canonical must parse
// through the real parsers, as the classified family, with
// the extracted slots.
func validatePiece(canon, family string, slots map[string]string) bool {
	ps, ok := CanonicalSlots(canon)
	if !ok || ps.Family != family {
		return false
	}
	return SlotsEqual(ps.Slots, slots)
}

// conceptPhrasePresent reports whether the fragment's token
// stream contains one of the concept's book names as a
// consecutive run — the naming evidence the fallback requires
// before a concept extraction counts.
func conceptPhrasePresent(f frag, family string) bool {
	id := strings.TrimPrefix(family, "concept-")
	for _, e := range conceptEntries() {
		if e.ID != id {
			continue
		}
		for _, name := range e.Names {
			nt := strings.Fields(strings.ToLower(name))
			if len(nt) == 0 || len(nt) > len(f.toks) {
				continue
			}
			for i := 0; i+len(nt) <= len(f.toks); i++ {
				match := true
				for j := range nt {
					if f.low[i+j] != nt[j] {
						match = false
						break
					}
				}
				if match {
					return true
				}
			}
		}
		return false
	}
	return false
}

// allFamilies lists every family the extractor knows — the
// fallback scan's search space. The print-line row is NOT
// listed as a concept family: CanonicalSlots reads its
// canonical as the "print" family (its parsers' own ruling),
// so a "concept-print-line" candidate could never validate —
// the row is served under "print", whose extractor carries
// its text-less shape (extractPrint).
func allFamilies() []string {
	out := []string{
		"print", "instantiate", "delegate", "clone", "add-server",
		"serve-html", "add-field", "add-field-tag", "add-case", "register",
	}
	for _, e := range conceptEntries() {
		if e.ID == "print-line" {
			continue
		}
		out = append(out, "concept-"+e.ID)
	}
	return out
}

// phrasingTopFamily is the classifier's ungated top-1 — the
// fallback's none-veto reads it (a line the classifier leans
// "none" on, at any confidence, is never geometry-claimed).
var phrasingTopFamily = func(line string) string {
	m := defaultPhrasingModel()
	if m == nil {
		return ""
	}
	fam, _, _ := m.Predict(line)
	return fam
}

// routeFragment classifies and extracts one fragment into its
// canonical sentence. The classifier proposes; slot geometry
// disposes, in a strict order:
//
//  1. A quoted span can only be a print line's text — no other
//     routed family has a saying slot — so when a fragment
//     carries one and print extraction validates, print wins
//     outright (the bare verb "log" otherwise pulls these to
//     the log-line concept, whose grammar has no text).
//  2. The classifier's gated family, when its extraction
//     validates. A gated "none" is final: no fallback. A
//     gated CONCEPT family additionally needs naming
//     evidence (its book phrase present in the fragment) —
//     see the guard at the step itself.
//  3. A fallback scan of every family's extractor — reached
//     when the gated family's extraction failed, or the gate
//     itself failed (held-out phrasings can leave the
//     classifier unsure of a fragment whose slots are
//     perfectly bindable). If exactly ONE family validates,
//     its geometry is the fragment's (e.g. "place salad into
//     Catalog" leans toward instantiate's "place a X in Y",
//     but only register's slots — a lowercase key and one
//     capitalized table — actually bind). The scan is vetoed
//     when the classifier's ungated top-1 is "none", and
//     concept families additionally require naming evidence
//     (conceptPhrasePresent). More than one validator is
//     ambiguity: no claim — unless the classifier's top-1 is
//     itself among the validators, in which case its reading
//     breaks the tie (the gate failed on confidence, not on
//     family, and geometry alone cannot choose).
func routeFragment(f frag) (canon, family string, slots map[string]string, ok bool) {
	// A pronominal placement ("in it", "inside it", "in
	// there") is an UNBOUND placement: no ordinary extractor
	// may claim the fragment by quietly dropping the pronoun
	// and reading the rest (a concept statement's file-only
	// form above all — "insert a variable inside it to
	// vault.go" is not "add a variable to vault.go"). The one
	// legitimate reader of a pronoun placement is the
	// created-target binding (routeCreatedPiece), which runs
	// before this function on a compound's second piece; any
	// other fragment that reaches here with a pronoun still in
	// its placement has nothing to bind to and does not claim.
	if f.placement().pronoun {
		return "", "", nil, false
	}
	if _, qok := f.quotedText(); qok {
		if c, s, eok := extractPrint(f); eok && validatePiece(c, "print", s) {
			return c, "print", s, true
		}
	}
	family, gated := phrasingClassifier(f.text())
	if gated && family == "none" {
		return "", "", nil, false
	}
	if gated && family != "" {
		// A gated CONCEPT claim must still name its concept —
		// the naming evidence the fallback scan below has
		// always required, applied to the gated step too. A
		// concept's extraction binds only file/func/name
		// slots, so a concept family whose name never appears
		// in the fragment can "validate" a line that never
		// asked for it, on the strength of one shared token:
		// measured 2026-10-31, the rqlite-insert family's
		// name word taught the classifier the verb "insert"
		// so strongly that the held-out dialect's generic
		// insert-led asks ("insert an http server into
		// api/main.go", "insert a xml in Nomad …") routed to
		// rqlite wholesale — 16 original-judge items, 3
		// created-judge items, 1 sources item, all thefts of
		// exactly this shape. Voiding the gated claim here
		// lets the fallback decide on geometry plus naming
		// evidence, the same terms it judges every other
		// family by. (The five-sources build met the same
		// mechanism in the training data and withheld the
		// insert-bearing names; a concept whose EVERY name
		// carries the generic verb — "rqlite insert" is the
		// row's canonical name — cannot be fixed that way,
		// so the rule lives in the route, where it binds
		// every concept family alike.)
		if !strings.HasPrefix(family, "concept-") || conceptPhrasePresent(f, family) {
			if canon, slots, eok := extractForFamily(f, family); eok && validatePiece(canon, family, slots) {
				return canon, family, slots, true
			}
		}
	}
	top := phrasingTopFamily(f.text())
	if top == "none" || top == "" {
		return "", "", nil, false
	}
	type candidate struct {
		canon string
		slots map[string]string
	}
	found := map[string]candidate{}
	var order []string
	for _, fam := range allFamilies() {
		if fam == family {
			continue
		}
		// Concept extraction binds only file/func/name slots,
		// so in the fallback — where the classifier is not the
		// evidence — a concept counts only when the fragment
		// actually NAMES it (one of the book's phrases appears
		// as a token run). Without this, every file-bearing
		// fragment would "validate" as every concept at once.
		if strings.HasPrefix(fam, "concept-") && !conceptPhrasePresent(f, fam) {
			continue
		}
		if c, s, eok := extractForFamily(f, fam); eok && validatePiece(c, fam, s) {
			found[fam] = candidate{c, s}
			order = append(order, fam)
		}
	}
	if len(order) == 1 {
		c := found[order[0]]
		return c.canon, order[0], c.slots, true
	}
	// More than one validator is ambiguity — with one
	// exception: when the classifier's own ungated top-1 is
	// among the validators, its reading breaks the tie. The
	// gate rejected this fragment on confidence, not on
	// family; geometry alone cannot choose between a delegate
	// and a function-with-a-stray-capital ("place a function
	// in vault.go calling Whisk" validates as both, the
	// callee doing duty as a declaration name), but the
	// classifier's strongest belief is exactly the evidence
	// the rest of the route runs on.
	if len(order) > 1 {
		if c, isTop := found[top]; isTop {
			return c.canon, top, c.slots, true
		}
	}
	return "", "", nil, false
}

// namedClauseGuard reports whether one fragment carries a
// name slot — "named <Name>" or "name is <Name>", the concept
// grammar's two spellings, read only BEFORE the first "that"
// exactly as the write boundary reads them — with a "that"
// clause riding behind it. Such a fragment is a specified
// declaration with an unspendable clause (the 2026-10-31 live
// miss: "add function named jake that returns a string to
// jim/main.go" was routed to the string-functions concept on
// the strength of the word "string"): a servable shape's
// canonical carries no behaviour clause, so a named line with
// one is never a messy version of a shape this layer serves —
// it is the write boundary's line, and the route declines it
// before classification, whatever the classifier thinks.
//
// Two reads keep the guard inside that rationale:
//   - It is per-FRAGMENT, not per-line: a compound's glue
//     ("… and after that …") contributes a "that" that is no
//     clause at all, and a declaration piece's own name slot
//     ("place a struct named Roaster in vault.go") precedes
//     it without specifying anything unspendable in ITS
//     fragment.
//   - The delegate's clause is spendable: "that calls
//     <Callee>" is the delegate family's own shape (its
//     canonical carries the clause), so a name slot before
//     THAT clause ("add a wrapper named Rowan in kilo.go
//     that calls Whisk") is a messy version of a servable
//     shape and routes on.
func namedClauseGuard(f frag) bool {
	thatAt := -1
	for i := 0; i < len(f.low); i++ {
		if f.low[i] == "that" {
			thatAt = i
			break
		}
	}
	if thatAt < 0 {
		return false
	}
	if thatAt+1 < len(f.low) && f.low[thatAt+1] == "calls" {
		return false
	}
	for i := 0; i < thatAt; i++ {
		switch {
		case f.low[i] == "named" && i+1 < thatAt:
			return true
		case f.low[i] == "name" && i+2 < thatAt && f.low[i+1] == "is":
			return true
		}
	}
	return false
}

// phrasingRouteB is arm B's route: segment, classify, extract,
// validate. See the file header.
func phrasingRouteB(line string) (pieces []string, ok bool) {
	frags, haveIt, ok := segmentLine(line)
	if !ok {
		return nil, false
	}
	// A fragment that opens with a question word is a
	// question, not an ask — no family core starts with one
	// (wrapper forms like "would you mind" are stripped
	// before fragments exist).
	for _, f := range frags {
		if len(f.low) > 0 {
			switch f.low[0] {
			case "how", "what", "why", "when", "where", "who", "is", "are", "was", "were", "do", "does", "did":
				return nil, false
			}
		}
	}
	// The named-clause guard (see namedClauseGuard): a named
	// declaration carrying an unspendable clause is the write
	// boundary's line, declined here before any classification
	// — a compound falls whole, exactly as a failed validation
	// would fell it.
	for _, f := range frags {
		if namedClauseGuard(f) {
			return nil, false
		}
	}
	if len(frags) == 1 {
		canon, _, _, ok := routeFragment(frags[0])
		if !ok {
			return nil, false
		}
		return []string{canon}, true
	}
	// Shared-target distribution (see the section above):
	// when the compound's only file rides in the second
	// fragment, the first piece is routed COMPLETED first —
	// the bare fragment is the fallback reading, never the
	// first.
	var canonA, famA string
	var slotsA map[string]string
	ok = false
	if file, lacking, shared := sharedFileOf(frags[0], frags[1]); shared && lacking == 0 {
		canonA, famA, slotsA, ok = routeSharedPiece(frags[0], file)
	}
	if !ok {
		canonA, famA, slotsA, ok = routeFragment(frags[0])
		if !ok {
			return nil, false
		}
	}
	// The have-it print tail carries only its text: its
	// function and file are inherited — the first piece's
	// handler when it has one, else the compound convention's
	// main in main.go (the generator's buildPrintAskShared
	// rule).
	if haveIt {
		// The generated dialect quotes the print text ("have
		// it print \"the results\""); a person typing the
		// sentence leaves the quotes off (John's own compound
		// does) — then the words after the print verb ARE the
		// text.
		text, qok := frags[1].quotedText()
		if !qok && len(frags[1].toks) >= 2 && frags[1].low[0] == "print" {
			text, qok = strings.Join(frags[1].toks[1:], " "), true
		}
		if qok {
			if _, _, hasFile := frags[1].fileSlot(); !hasFile {
				fn := "main"
				if famA == "instantiate" && slotsA["func"] != "" {
					fn = slotsA["func"]
				} else if slotsA["func"] != "" {
					fn = slotsA["func"]
				}
				canonB := renderPrint(text, fn, "main.go")
				if validatePiece(canonB, "print", map[string]string{
					"text": text, "func": fn, "target": "main.go",
				}) {
					return []string{canonA, canonB}, true
				}
				return nil, false
			}
		}
	}
	// Created-target binding gets first claim on an unbound
	// second piece — before the ordinary route, whose bare-name
	// families (register above all) can otherwise "validate" a
	// file-less fragment by geometry accident ("put a for loop
	// in Bake" reads as registering the key "loop" in a table
	// "Bake"). When the first piece created a function and the
	// second piece's placement is unbound — a pronoun, or no
	// file named — the binding is tried first; if it declines
	// (a family with its own bare-name target, like add-case),
	// the ordinary route still owns the fragment.
	if cName, cFile, creates := createdFuncOf(famA, slotsA); creates {
		if p := frags[1].placement(); p.pronoun || !p.hasFile {
			if bound, bok := routeCreatedPiece(frags[1], cName, cFile); bok {
				return []string{canonA, bound}, true
			}
		}
	}
	// Shared-target distribution, the other direction: the
	// compound's only file rode in the FIRST fragment and the
	// second piece is the one missing its target — completed
	// first, bare as the fallback, exactly as for piece one.
	var canonB string
	if file, lacking, shared := sharedFileOf(frags[0], frags[1]); shared && lacking == 1 {
		canonB, _, _, ok = routeSharedPiece(frags[1], file)
		if ok {
			return []string{canonA, canonB}, true
		}
	}
	canonB, _, _, ok = routeFragment(frags[1])
	if !ok {
		return nil, false
	}
	return []string{canonA, canonB}, true
}
