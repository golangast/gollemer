package chat

// Layers pilot derivations (DESIGN §3): everything the layers
// need is derived from the prepared training pairs themselves,
// deterministically (seed 20261112), before epoch 1, and saved as
// versioned artifacts next to the model:
//
//	word classes — distributional clustering (context vectors over
//	  the top-300 context tokens, seeded cosine k-means, k=47
//	  learned classes + class 0 for specials)
//	concept centroids — seeded cosine k-means over reply-segment
//	  bag-of-words vectors (32 gist clusters)
//	intent labels — ordered weak rules over the raw input
//	tone labels + lexicon — feelings vocabulary seeded from the
//	  house synonym table's affect words + stated seed lists,
//	  filtered to the training vocabulary
//	memory facts — the fact seat's own capture shapes
//	  (parseFactCapture) mined from each pair's own input
//
// Nothing here touches the corpus: derivations read the prepared
// pairs and write artifacts; the pairs themselves are unchanged.

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
)

// layersSeed is the registered derivation seed (DESIGN §3).
const layersSeed = 20261112

// layersContextTopK is the context-vector width (DESIGN §3.2).
const layersContextTopK = 300

// layersKMeansIters is the registered iteration count.
const layersKMeansIters = 25

// layersDerivations bundles everything derived from the training
// pairs for one layers run.
type layersDerivations struct {
	ClassOf    []int       // per token id (len = vocab size)
	Centroids  [][]float32 // concept centroids (NumConcepts x vocab size)
	ToneLex    toneLexicon
	IntentDist [7]int
	ToneDist   [4]int
	FactPairs  int // train pairs whose own input states a fact
	ClusterSizes []int
	ClassSizes   []int
}

// toneLexicon is the derived tone vocabulary (DESIGN §3.4).
type toneLexicon struct {
	Bright  []string `json:"bright"`
	Low     []string `json:"low"`
	Worried []string `json:"worried"`
	bright  map[string]bool
	low     map[string]bool
	worried map[string]bool
}

func (l *toneLexicon) index() {
	l.bright = wordSet(l.Bright)
	l.low = wordSet(l.Low)
	l.worried = wordSet(l.Worried)
}

func wordSet(ws []string) map[string]bool {
	m := make(map[string]bool, len(ws))
	for _, w := range ws {
		m[w] = true
	}
	return m
}

// deriveLayers runs every derivation over the prepared training
// pairs (post-augmentation, exactly the pairs the trainer trains
// on) and the vocabulary built from them.
func deriveLayers(pairs []ChatPair, v *vocab.Vocabulary, tok interface {
	Encode(string) ([]int, error)
}, cfg seq2seq.LayersConfig) (*layersDerivations, error) {
	d := &layersDerivations{}

	// Tokenize every pair once for the derivations that need ids.
	encs := make([]encPair, len(pairs))
	distinctReplies := map[string][]int{}
	for i, p := range pairs {
		inIDs := encodeInputStrings(tok, p.Input)
		outFull := encodeTargetStrings(tok, v, p.Output)
		encs[i] = encPair{inIDs: inIDs, outIDs: outFull}
		if len(outFull) >= 2 {
			reply := outFull[1 : len(outFull)-1]
			key := intsKey(reply)
			if _, ok := distinctReplies[key]; !ok {
				distinctReplies[key] = reply
			}
		}
	}

	if cfg.Class {
		classOf, sizes, err := deriveWordClasses(encs, v, cfg)
		if err != nil {
			return nil, err
		}
		d.ClassOf = classOf
		d.ClassSizes = sizes
	}
	if cfg.Concept {
		var replies [][]int
		for _, r := range distinctReplies {
			replies = append(replies, r)
		}
		sort.Slice(replies, func(a, b int) bool { return intsKey(replies[a]) < intsKey(replies[b]) })
		centroids, sizes, err := deriveConceptCentroids(replies, v.Size(), cfg)
		if err != nil {
			return nil, err
		}
		d.Centroids = centroids
		d.ClusterSizes = sizes
	}
	d.ToneLex = deriveToneLexicon(v)
	d.ToneLex.index()
	if cfg.Intent || cfg.Tone {
		for _, p := range pairs {
			words := tokenizeLower(seq2seq.NormalizeQuery(p.Input))
			if cfg.Intent {
				d.IntentDist[intentLabelFor(words)]++
			}
			if cfg.Tone {
				d.ToneDist[toneLabelFor(words, &d.ToneLex)]++
			}
		}
	}
	if cfg.Memory {
		for _, p := range pairs {
			if facts := LayersFactStatements(p.Input); len(facts) > 0 {
				d.FactPairs++
			}
		}
	}
	return d, nil
}

// encPair is the minimal encoding shape the derivations consume.
type encPair struct {
	inIDs  []int
	outIDs []int // full target [BOS..EOS]
}

func intsKey(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		fmt.Fprintf(&b, "%d,", id)
	}
	return b.String()
}

func encodeInputStrings(tok interface{ Encode(string) ([]int, error) }, q string) []int {
	ids, err := tok.Encode(seq2seq.NormalizeQuery(q))
	if err != nil {
		return nil
	}
	return ids
}

func encodeTargetStrings(tok interface{ Encode(string) ([]int, error) }, v *vocab.Vocabulary, a string) []int {
	ids, err := tok.Encode(strings.ToLower(a))
	if err != nil {
		return nil
	}
	out := make([]int, 0, len(ids)+2)
	out = append(out, v.BosID)
	out = append(out, ids...)
	out = append(out, v.EosID)
	return out
}

// ---- word classes (DESIGN §3.2) ----

// deriveWordClasses clusters the vocabulary's words by their
// distributional context vectors. Returns the class per token id
// (0 for specials) and the learned-cluster sizes.
func deriveWordClasses(pairs []encPair, v *vocab.Vocabulary, cfg seq2seq.LayersConfig) ([]int, []int, error) {
	vocabSize := v.Size()
	// Context vocabulary: the top-K most frequent non-special ids.
	freq := make([]int, vocabSize)
	for _, p := range pairs {
		for _, id := range p.inIDs {
			freq[id]++
		}
		for _, id := range p.outIDs {
			freq[id]++
		}
	}
	type idFreq struct{ id, n int }
	var ranked []idFreq
	for id := 0; id < vocabSize; id++ {
		if isSpecial(v, id) || freq[id] == 0 {
			continue
		}
		ranked = append(ranked, idFreq{id, freq[id]})
	}
	sort.Slice(ranked, func(a, b int) bool {
		if ranked[a].n != ranked[b].n {
			return ranked[a].n > ranked[b].n
		}
		return ranked[a].id < ranked[b].id
	})
	ctxIndex := map[int]int{}
	ctxIDs := []int{}
	for i, r := range ranked {
		if i >= layersContextTopK {
			break
		}
		ctxIndex[r.id] = len(ctxIDs)
		ctxIDs = append(ctxIDs, r.id)
	}
	K := len(ctxIDs)
	// Context counts with a ±2 window over inputs and outputs.
	counts := make([][]float64, vocabSize)
	for id := range counts {
		counts[id] = make([]float64, K)
	}
	addWindow := func(seq []int) {
		for i, id := range seq {
			if isSpecial(v, id) {
				continue
			}
			for j := i - 2; j <= i+2; j++ {
				if j == i || j < 0 || j >= len(seq) {
					continue
				}
				if ci, ok := ctxIndex[seq[j]]; ok {
					counts[id][ci]++
				}
			}
		}
	}
	for _, p := range pairs {
		addWindow(p.inIDs)
		if len(p.outIDs) >= 2 {
			addWindow(p.outIDs[1 : len(p.outIDs)-1])
		}
	}
	// Vectors for every non-special word with any context mass;
	// words without mass join the smallest cluster at the end.
	var words []int
	vecs := map[int][]float64{}
	for id := 0; id < vocabSize; id++ {
		if isSpecial(v, id) {
			continue
		}
		var mass float64
		for _, x := range counts[id] {
			mass += x
		}
		if mass == 0 {
			continue
		}
		vec := make([]float64, K)
		for k, x := range counts[id] {
			vec[k] = math.Log(1 + x)
		}
		normalize64(vec)
		words = append(words, id)
		vecs[id] = vec
	}
	k := cfg.NumClasses - 1 // class 0 is reserved for specials
	if len(words) < k {
		k = len(words)
	}
	assign, sizes := seededKMeans(words, vecs, k, K)
	classOf := make([]int, vocabSize)
	smallest := 0
	for c := 0; c < k; c++ {
		if sizes[c] < sizes[smallest] {
			smallest = c
		}
	}
	for id := 0; id < vocabSize; id++ {
		if isSpecial(v, id) {
			classOf[id] = 0
			continue
		}
		if c, ok := assign[id]; ok {
			classOf[id] = c + 1
		} else {
			classOf[id] = smallest + 1
		}
	}
	fullSizes := make([]int, cfg.NumClasses)
	fullSizes[0] = 4 // pad, unk, bos, eos
	for id := 0; id < vocabSize; id++ {
		if !isSpecial(v, id) {
			fullSizes[classOf[id]]++
		}
	}
	_ = sizes
	return classOf, fullSizes, nil
}

func isSpecial(v *vocab.Vocabulary, id int) bool {
	return id == v.PaddingTokenID || id == v.UnkID || id == v.BosID || id == v.EosID
}

// ---- concept centroids (DESIGN §3.5) ----

// deriveConceptCentroids clusters reply-segment bag-of-words
// vectors into cfg.NumConcepts gist centroids. Segment vectors
// are sparse (a segment holds a handful of distinct tokens), so
// the clustering runs on sparse vectors against dense centroids.
func deriveConceptCentroids(replies [][]int, vocabSize int, cfg seq2seq.LayersConfig) ([][]float32, []int, error) {
	var vecs []sparseVec
	for _, r := range replies {
		for _, seg := range seq2seq.SplitSegments(r, cfg.ConceptLen) {
			if len(seg) == 0 {
				continue
			}
			counts := map[int]float64{}
			for _, id := range seg {
				if id >= 0 && id < vocabSize {
					counts[id]++
				}
			}
			sv := sparseVec{}
			var norm float64
			for _, c := range counts {
				norm += c * c
			}
			norm = math.Sqrt(norm)
			if norm == 0 {
				continue
			}
			ids := make([]int, 0, len(counts))
			for id := range counts {
				ids = append(ids, id)
			}
			sort.Ints(ids)
			for _, id := range ids {
				sv.idx = append(sv.idx, id)
				sv.val = append(sv.val, counts[id]/norm)
			}
			vecs = append(vecs, sv)
		}
	}
	if len(vecs) == 0 {
		return nil, nil, fmt.Errorf("layers: no reply segments to cluster")
	}
	k := cfg.NumConcepts
	if len(vecs) < k {
		k = len(vecs)
	}
	centroids, sizes := kmeansSparse(vecs, k, vocabSize)
	out := make([][]float32, cfg.NumConcepts)
	for c := 0; c < cfg.NumConcepts; c++ {
		row := make([]float32, vocabSize)
		if c < len(centroids) {
			for i, x := range centroids[c] {
				row[i] = float32(x)
			}
		}
		out[c] = row
	}
	fullSizes := make([]int, cfg.NumConcepts)
	copy(fullSizes, sizes)
	return out, fullSizes, nil
}

// sparseVec is a sparse float64 vector (sorted indices).
type sparseVec struct {
	idx []int
	val []float64
}

func (s sparseVec) dotDense(dense []float64) float64 {
	var sum float64
	for i, id := range s.idx {
		sum += s.val[i] * dense[id]
	}
	return sum
}

func (s sparseVec) toDense(dim int) []float64 {
	out := make([]float64, dim)
	for i, id := range s.idx {
		out[id] = s.val[i]
	}
	return out
}

// kmeansSparse is seededKMeansVecs over sparse item vectors with
// dense centroids (same seed, init, and iteration count).
func kmeansSparse(vecs []sparseVec, k, dim int) ([][]float64, []int) {
	rng := rand.New(rand.NewSource(layersSeed))
	n := len(vecs)
	centroids := make([][]float64, k)
	first := rng.Intn(n)
	centroids[0] = vecs[first].toDense(dim)
	for c := 1; c < k; c++ {
		weights := make([]float64, n)
		var total float64
		for i, v := range vecs {
			best := math.Inf(-1)
			for j := 0; j < c; j++ {
				if d := v.dotDense(centroids[j]); d > best {
					best = d
				}
			}
			w := 1 - best
			if w < 0 {
				w = 0
			}
			weights[i] = w * w
			total += weights[i]
		}
		pick := 0
		if total > 0 {
			r := rng.Float64() * total
			acc := 0.0
			for i, w := range weights {
				acc += w
				if acc >= r {
					pick = i
					break
				}
			}
		} else {
			pick = rng.Intn(n)
		}
		centroids[c] = vecs[pick].toDense(dim)
	}
	assign := make([]int, n)
	for iter := 0; iter < layersKMeansIters; iter++ {
		for i, v := range vecs {
			best, bestSim := 0, math.Inf(-1)
			for c := 0; c < k; c++ {
				if d := v.dotDense(centroids[c]); d > bestSim {
					bestSim, best = d, c
				}
			}
			assign[i] = best
		}
		sums := make([][]float64, k)
		counts := make([]int, k)
		for c := 0; c < k; c++ {
			sums[c] = make([]float64, dim)
		}
		for i, v := range vecs {
			c := assign[i]
			counts[c]++
			for j, id := range v.idx {
				sums[c][id] += v.val[j]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				continue
			}
			for d := 0; d < dim; d++ {
				centroids[c][d] = sums[c][d] / float64(counts[c])
			}
			normalize64(centroids[c])
		}
	}
	sizes := make([]int, k)
	for _, c := range assign {
		sizes[c]++
	}
	return centroids, sizes
}

// ---- seeded cosine k-means ----

// seededKMeans clusters the given item vectors (keyed by item id)
// into k clusters with k-means++ init under the registered seed.
// Returns item->cluster and cluster sizes.
func seededKMeans(items []int, vecs map[int][]float64, k, dim int) (map[int]int, []int) {
	ordered := append([]int{}, items...)
	sort.Ints(ordered)
	vs := make([][]float64, len(ordered))
	for i, id := range ordered {
		vs[i] = vecs[id]
	}
	centroids, sizes, assignIdx := kmeansRun(vs, k, dim)
	assign := map[int]int{}
	for i, id := range ordered {
		assign[id] = assignIdx[i]
	}
	_ = centroids
	return assign, sizes
}

func kmeansRun(vecs [][]float64, k, dim int) ([][]float64, []int, []int) {
	rng := rand.New(rand.NewSource(layersSeed))
	n := len(vecs)
	// k-means++ init (cosine distance = 1 - dot on unit vectors).
	centroids := make([][]float64, k)
	first := rng.Intn(n)
	centroids[0] = append([]float64{}, vecs[first]...)
	for c := 1; c < k; c++ {
		weights := make([]float64, n)
		var total float64
		for i, v := range vecs {
			best := math.Inf(-1)
			for j := 0; j < c; j++ {
				if d := dot64(v, centroids[j]); d > best {
					best = d
				}
			}
			w := 1 - best
			if w < 0 {
				w = 0
			}
			weights[i] = w * w
			total += weights[i]
		}
		pick := 0
		if total > 0 {
			r := rng.Float64() * total
			acc := 0.0
			for i, w := range weights {
				acc += w
				if acc >= r {
					pick = i
					break
				}
			}
		} else {
			pick = rng.Intn(n)
		}
		centroids[c] = append([]float64{}, vecs[pick]...)
	}
	assign := make([]int, n)
	for iter := 0; iter < layersKMeansIters; iter++ {
		for i, v := range vecs {
			best, bestSim := 0, math.Inf(-1)
			for c := 0; c < k; c++ {
				if d := dot64(v, centroids[c]); d > bestSim {
					bestSim, best = d, c
				}
			}
			assign[i] = best
		}
		sums := make([][]float64, k)
		counts := make([]int, k)
		for c := 0; c < k; c++ {
			sums[c] = make([]float64, dim)
		}
		for i, v := range vecs {
			c := assign[i]
			counts[c]++
			for d := 0; d < dim; d++ {
				sums[c][d] += v[d]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				continue // empty cluster keeps its centroid
			}
			for d := 0; d < dim; d++ {
				centroids[c][d] = sums[c][d] / float64(counts[c])
			}
			normalize64(centroids[c])
		}
	}
	sizes := make([]int, k)
	for _, c := range assign {
		sizes[c]++
	}
	return centroids, sizes, assign
}

func dot64(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func normalize64(v []float64) {
	var n float64
	for _, x := range v {
		n += x * x
	}
	n = math.Sqrt(n)
	if n == 0 {
		return
	}
	for i := range v {
		v[i] /= n
	}
}

// ---- intent labels (DESIGN §3.3) ----

// Intent class ids (registered order).
const (
	IntentOther    = 0
	IntentGreeting = 1
	IntentQuestion = 2
	IntentRequest  = 3
	IntentFeeling  = 4
	IntentThanks   = 5
	IntentFarewell = 6
)

var intentFarewellWords = wordSet([]string{"bye", "goodbye"})
var intentThanksWords = wordSet([]string{"thanks", "thank"})
var intentGreetingHeads = wordSet([]string{"hello", "hi", "hey", "greetings"})
var intentFeelingWords = wordSet([]string{
	"feel", "feeling", "happy", "sad", "tired", "angry", "excited", "bored",
	"lonely", "scared", "worried", "anxious", "nervous", "hurt", "sick",
	"down", "unhappy", "glad", "upset", "stressed", "afraid",
})
var intentQuestionHeads = wordSet([]string{
	"what", "why", "how", "when", "where", "who", "which", "whose", "whom",
	"do", "does", "did", "is", "are", "was", "were", "can", "could", "would",
	"will", "should", "have", "has",
})

// intentLabelFor applies the registered ordered rules to the
// input's words (first match wins): farewell, thanks, greeting,
// request, question, feeling, other.
func intentLabelFor(words []string) int {
	if len(words) == 0 {
		return IntentOther
	}
	for _, w := range words {
		if intentFarewellWords[w] {
			return IntentFarewell
		}
	}
	for _, w := range words {
		if intentThanksWords[w] {
			return IntentThanks
		}
	}
	if intentGreetingHeads[words[0]] {
		return IntentGreeting
	}
	if len(words) >= 2 && words[0] == "good" &&
		(words[1] == "morning" || words[1] == "afternoon" || words[1] == "evening") {
		return IntentGreeting
	}
	joined := " " + strings.Join(words, " ") + " "
	if strings.Contains(joined, " please ") || strings.HasPrefix(joined, " please") ||
		strings.Contains(joined, " tell me ") || strings.Contains(joined, " show me ") ||
		strings.Contains(joined, " give me ") || strings.Contains(joined, " can you ") ||
		strings.Contains(joined, " could you ") || strings.Contains(joined, " would you ") {
		return IntentRequest
	}
	last := words[len(words)-1]
	if last == "?" || intentQuestionHeads[words[0]] {
		return IntentQuestion
	}
	for _, w := range words {
		if intentFeelingWords[w] {
			return IntentFeeling
		}
	}
	return IntentOther
}

// ---- tone labels (DESIGN §3.4) ----

// Tone class ids (registered order).
const (
	ToneNeutral = 0
	ToneBright  = 1
	ToneLow     = 2
	ToneWorried = 3
)

// deriveToneLexicon seeds the tone vocabulary from the house
// synonym table's affect words plus the registered seed lists,
// filtered to words the training vocabulary actually contains.
// (The shipped socialSynonyms table is a flat word->synonyms map
// with no category field; its affect entries seed the bright/low
// lists. DESIGN §11 records this reading.)
func deriveToneLexicon(v *vocab.Vocabulary) toneLexicon {
	brightSeeds := []string{"happy", "glad", "pleased", "great", "fantastic", "wonderful", "awesome", "amazing", "good", "nice", "lovely", "cool", "love", "excited", "fun", "joy"}
	lowSeeds := []string{"sad", "unhappy", "down", "tired", "sick", "hurt", "bored", "lonely", "cry", "bad", "terrible", "awful"}
	worriedSeeds := []string{"worried", "anxious", "nervous", "scared", "afraid", "stress", "stressed", "butterflies"}
	// Pull every synonym of the affect seeds that the house table
	// knows, so the lexicon is the table's feelings vocabulary,
	// not just the seeds.
	expand := func(seeds []string) []string {
		set := map[string]bool{}
		for _, s := range seeds {
			set[s] = true
			for _, syn := range socialSynonyms[s] {
				for _, w := range strings.Fields(syn) {
					set[w] = true
				}
			}
		}
		var out []string
		for w := range set {
			if _, ok := v.WordToToken[w]; ok {
				out = append(out, w)
			}
		}
		sort.Strings(out)
		return out
	}
	return toneLexicon{
		Bright:  expand(brightSeeds),
		Low:     expand(lowSeeds),
		Worried: expand(worriedSeeds),
	}
}

// toneLabelFor applies the registered set precedence over the
// whole line: any worried hit labels worried, else any bright hit
// labels bright, else any low hit labels low; no hit is neutral.
func toneLabelFor(words []string, lex *toneLexicon) int {
	bright, low := false, false
	for _, w := range words {
		switch {
		case lex.worried[w]:
			return ToneWorried
		case lex.bright[w]:
			bright = true
		case lex.low[w]:
			low = true
		}
	}
	if bright {
		return ToneBright
	}
	if low {
		return ToneLow
	}
	return ToneNeutral
}

// ---- memory facts (DESIGN §3.8) ----

// layersFactTokenCap is the registered memory-prefix cap.
const layersFactTokenCap = 16

// LayersFactStatements renders the facts a line states (by the
// fact seat's own capture shapes) as statement token strings —
// the seat's canonical answer sentence, lowercased, trailing
// period dropped. Exported: the trainer and the scorer share it.
func LayersFactStatements(line string) []string {
	_, fact, ok := parseFactCapture(line)
	if !ok {
		return nil
	}
	stmt := strings.ToLower(strings.TrimSuffix(strings.TrimSpace(fact.Answer), "."))
	return []string{stmt}
}

// LayersFactPrefixTokens flattens fact statements into prefix
// tokens under the cap: whole statements only, and when the cap
// would be exceeded the OLDEST statements are dropped (selection
// walks newest-first; output is in original statement order).
func LayersFactPrefixTokens(statements []string) []string {
	selected := make([]bool, len(statements))
	total := 0
	for i := len(statements) - 1; i >= 0; i-- {
		n := len(tokenizeLower(statements[i]))
		if total+n <= layersFactTokenCap {
			selected[i] = true
			total += n
		}
	}
	var out []string
	for i, s := range statements {
		if selected[i] {
			out = append(out, tokenizeLower(s)...)
		}
	}
	return out
}

// ---- artifacts (DESIGN §5) ----

type layersClassesArtifact struct {
	Version    int            `json:"version"`
	Seed       int            `json:"seed"`
	NumClasses int            `json:"num_classes"`
	Classes    map[string]int `json:"classes"`
}

type layersMetaArtifact struct {
	Version      int         `json:"version"`
	Seed         int         `json:"seed"`
	Layers       []string    `json:"layers"`
	ToneLexicon  toneLexicon `json:"tone_lexicon"`
	IntentDist   [7]int      `json:"intent_distribution"`
	ToneDist     [4]int      `json:"tone_distribution"`
	ConceptSizes []int       `json:"concept_cluster_sizes"`
	ClassSizes   []int       `json:"class_sizes"`
	FactPairs    int         `json:"fact_bearing_train_pairs"`
}

// writeLayersArtifacts writes <model>.classes.json and
// <model>.layers.json next to the model file.
func writeLayersArtifacts(modelPath string, d *layersDerivations, v *vocab.Vocabulary, cfg seq2seq.LayersConfig) error {
	dir := filepath.Dir(modelPath)
	base := filepath.Base(modelPath)
	if cfg.Class {
		classes := map[string]int{}
		for id := 0; id < v.Size(); id++ {
			classes[v.GetWord(id)] = d.ClassOf[id]
		}
		art := layersClassesArtifact{Version: 1, Seed: layersSeed, NumClasses: cfg.NumClasses, Classes: classes}
		data, err := json.MarshalIndent(art, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, base+".classes.json"), data, 0o644); err != nil {
			return err
		}
	}
	var layers []string
	for _, name := range []string{"char", "class", "intent", "tone", "concept", "copy", "turn", "memory"} {
		if cfg.On(name) {
			layers = append(layers, name)
		}
	}
	meta := layersMetaArtifact{
		Version:      1,
		Seed:         layersSeed,
		Layers:       layers,
		ToneLexicon:  d.ToneLex,
		IntentDist:   d.IntentDist,
		ToneDist:     d.ToneDist,
		ConceptSizes: d.ClusterSizes,
		ClassSizes:   d.ClassSizes,
		FactPairs:    d.FactPairs,
	}
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, base+".layers.json"), data, 0o644)
}
