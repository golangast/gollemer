package chat

// Layers pilot: the COMBINATION HOLDOUT builder (DESIGN §12).
// Tags every social pair in the production corpus by
// (top code x topic family x tone) using the same derivations a
// layers run trains with — top code of the gold reply (full-
// corpus frame derivations), topic family from clustering the
// input's content-word bag-of-words, tone from the lexicon rule —
// then removes a handful of whole CELLS from a copy of the
// corpus. A model trained on the reduced corpus has seen every
// component of a held-out cell (the top code with other topics,
// the topic with other top codes, the tone elsewhere) but never
// the combination: the decisive test of whether the hybrid frame
// composes or merely recalls.
//
// Everything is deterministic (the registered derivation seed):
// the same corpus produces the same census, the same cells, and
// byte-identical artifacts. The builder writes, into the output
// directory:
//
//	holdout_test.jsonl      the held-out cells' pairs (+ want lists)
//	holdout_seen.jsonl      a seeded sample of kept-cell pairs
//	chat_pairs_reduced.jsonl the corpus minus the held-out pairs
//	CENSUS.json             the full census, selection, and hashes
//	CENSUS.md               the human-readable census
//	HOLDOUT.sha256          hashes of the other four files

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// Registered holdout constants (DESIGN §12).
const (
	holdoutVersion    = 1
	holdoutTopicFams  = 12
	holdoutMinCell    = 40  // a cell needs at least this many pairs to be a candidate
	holdoutMinTotal   = 350 // selection stops once the test set reaches this (and the spans hold)
	holdoutMaxTotal   = 500 // ... and never exceeds this
	holdoutMaxCells   = 8
	holdoutTopicCap   = 2  // at most this many chosen cells per topic family
	holdoutRetainEach = 50 // each picked cell's top/topic/tone keeps >= this many pairs outside the picked cells
	holdoutSeenSample = 300
)

// holdoutCell is one (top, topic, tone) combination.
type holdoutCell struct {
	Top   int `json:"top"`
	Topic int `json:"topic"` // -1 = the input has no in-vocab content words
	Tone  int `json:"tone"`
}

type holdoutTestItem struct {
	Input  string   `json:"input"`
	Output string   `json:"output"`
	Top    int      `json:"top"`
	Topic  int      `json:"topic"`
	Tone   int      `json:"tone"`
	Want   []string `json:"want"`
}

type holdoutCensus struct {
	Version       int               `json:"version"`
	Seed          int               `json:"seed"`
	NumTopCodes   int               `json:"num_top_codes"`
	TopicFamilies int               `json:"topic_families"`
	CorpusPairs   int               `json:"corpus_lines"`
	SocialPairs   int               `json:"social_pairs"`
	TopCensus     []int             `json:"top_census_pairs"`
	TopicCensus   []int             `json:"topic_census_pairs"` // index 0..F-1, plus a trailing entry for topic -1
	ToneCensus    [4]int            `json:"tone_census_pairs"`
	CellsTotal    int               `json:"cells_total"`
	Candidates    int               `json:"candidate_cells_ge_min"`
	Chosen        []holdoutCell     `json:"chosen_cells"`
	ChosenPairs   int               `json:"chosen_pairs"`
	SpanTops      int               `json:"span_top_codes"`
	SpanTopics    int               `json:"span_topic_families"`
	SpanTones     int               `json:"span_tones"`
	Unjudgeable   int               `json:"excluded_no_want_pairs"`
	SeenSample    int               `json:"seen_sample_pairs"`
	Files         map[string]string `json:"file_sha256"`
}

// BuildCombinationHoldout runs the census and writes the holdout
// artifacts into outDir (created if needed). It never modifies
// the corpus in place.
func BuildCombinationHoldout(projectRoot, outDir string) error {
	dd, err := prepareDomainTrainData(projectRoot, SocialDomain)
	if err != nil {
		return err
	}
	cfg := seq2seq.DefaultLayersConfig()
	if err := cfg.Set("frame", true); err != nil {
		return err
	}
	deriv, err := deriveLayers(dd.trainPairs, dd.v, dd.tok, cfg)
	if err != nil {
		return err
	}

	// Read the corpus as raw lines: removal is by line, and all
	// non-held-out lines are preserved byte-for-byte.
	raw, err := os.ReadFile(ChatDatasetPath(projectRoot))
	if err != nil {
		return err
	}
	lines := strings.Split(strings.TrimRight(string(raw), "\n"), "\n")

	type tagged struct {
		line int
		pair ChatPair
		cell holdoutCell
		want []string
		ok   bool // top code derived; untaggable pairs belong to no cell
	}
	var social []tagged
	for li, line := range lines {
		var p ChatPair
		if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &p); err != nil {
			continue // not a pair line; preserved, never tagged
		}
		if p.Domain != "" && p.Domain != SocialDomain {
			continue
		}
		social = append(social, tagged{line: li, pair: p})
	}

	// Topic families: seeded k-means over input content-word BoW
	// (pairs without in-vocab content words are family -1 and do
	// not enter the clustering).
	contentIDs := make([][]int, len(social))
	var vecs []sparseVec
	vecPair := []int{} // index into social for each clustered vector
	for i, tg := range social {
		words := tokenizeLower(seq2seq.NormalizeQuery(tg.pair.Input))
		for _, w := range seq2seq.ContentWordsOf(words) {
			if id, ok := dd.v.WordToToken[w]; ok {
				contentIDs[i] = append(contentIDs[i], id)
			}
		}
		if sv, ok := bowSparse(contentIDs[i], dd.v.Size()); ok {
			vecs = append(vecs, sv)
			vecPair = append(vecPair, i)
		}
	}
	topicCentroids, _ := kmeansSparse(vecs, holdoutTopicFams, dd.v.Size())
	topicOf := make([]int, len(social))
	for i := range topicOf {
		topicOf[i] = -1
	}
	for vi, si := range vecPair {
		best, bestSim := 0, -1e18
		for c, cen := range topicCentroids {
			if d := vecs[vi].dotDense(cen); d > bestSim {
				bestSim, best = d, c
			}
		}
		topicOf[si] = best
	}

	// Tag every social pair.
	census := holdoutCensus{
		Version:       holdoutVersion,
		Seed:          layersSeed,
		NumTopCodes:   cfg.NumTopCodes,
		TopicFamilies: holdoutTopicFams,
		CorpusPairs:   len(lines),
		SocialPairs:   len(social),
		TopCensus:     make([]int, cfg.NumTopCodes),
		TopicCensus:   make([]int, holdoutTopicFams+1),
		Files:         map[string]string{},
	}
	cellPairs := map[holdoutCell][]int{} // cell -> indices into social
	for i, tg := range social {
		outIDs := encodeTargetStrings(dd.tok, dd.v, tg.pair.Output)
		var reply []int
		if len(outIDs) >= 2 {
			reply = outIDs[1 : len(outIDs)-1]
		}
		top, ok := seq2seq.DeriveTopWithCentroids(reply, deriv.TopCentroids)
		if !ok {
			continue // no derivable top (empty reply): untaggable, stays in the corpus
		}
		tone := deriv.ToneLex.ToneOf(tokenizeLower(seq2seq.NormalizeQuery(tg.pair.Input)))
		cell := holdoutCell{Top: top, Topic: topicOf[i], Tone: tone}
		social[i].cell = cell
		social[i].ok = true
		social[i].want = seq2seq.ContentWordsOf(tokenizeLower(tg.pair.Output))
		cellPairs[cell] = append(cellPairs[cell], i)
		census.TopCensus[top]++
		if cell.Topic >= 0 {
			census.TopicCensus[cell.Topic]++
		} else {
			census.TopicCensus[holdoutTopicFams]++
		}
		census.ToneCensus[tone]++
	}
	census.CellsTotal = len(cellPairs)

	// Component totals, for the retention rule: a picked cell's
	// top code, topic family, and tone must each keep at least
	// holdoutRetainEach pairs outside ALL picked cells — the
	// combination is held out, never a component.
	topTotal := append([]int{}, census.TopCensus...)
	topicTotal := append([]int{}, census.TopicCensus...)
	toneTotal := census.ToneCensus

	// Candidate cells, in the registered order: count desc, then
	// (top, topic, tone) asc.
	type cand struct {
		cell holdoutCell
		n    int
	}
	var cands []cand
	for cell, idxs := range cellPairs {
		if len(idxs) >= holdoutMinCell {
			cands = append(cands, cand{cell, len(idxs)})
		}
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].n != cands[b].n {
			return cands[a].n > cands[b].n
		}
		ca, cb := cands[a].cell, cands[b].cell
		if ca.Top != cb.Top {
			return ca.Top < cb.Top
		}
		if ca.Topic != cb.Topic {
			return ca.Topic < cb.Topic
		}
		return ca.Tone < cb.Tone
	})
	census.Candidates = len(cands)

	retains := func(cell holdoutCell, n int, pickedTop, pickedTopic, pickedTone map[int]int) bool {
		if topTotal[cell.Top]-pickedTop[cell.Top]-n < holdoutRetainEach {
			return false
		}
		ti := cell.Topic
		if ti < 0 {
			ti = holdoutTopicFams
		}
		if topicTotal[ti]-pickedTopic[ti]-n < holdoutRetainEach {
			return false
		}
		if toneTotal[cell.Tone]-pickedTone[cell.Tone]-n < holdoutRetainEach {
			return false
		}
		return true
	}
	picked := map[holdoutCell]bool{}
	pickedTop, pickedTopic, pickedTone := map[int]int{}, map[int]int{}, map[int]int{}
	topicCells := map[int]int{}
	total := 0
	spans := func() (int, int, int) {
		tops, topics, tones := map[int]bool{}, map[int]bool{}, map[int]bool{}
		for cell := range picked {
			tops[cell.Top] = true
			topics[cell.Topic] = true
			tones[cell.Tone] = true
		}
		return len(tops), len(topics), len(tones)
	}
	// Single registered pass over the candidates, in order. A
	// cell is picked when the retention rule and the per-topic
	// quota allow it and the total cap holds. The pass stops
	// once the test set reaches the minimum mass AND spans
	// >= 3 top codes, >= 3 topic families, and >= 2 tones —
	// the diversity is the point: a holdout drawn from one
	// topic and one tone would test only one axis of
	// recombination. If the scan ends first, the census records
	// the spans the corpus actually supports (e.g. tones low
	// and worried have no candidate cells at any top/topic —
	// their pairs are spread too thin; the census says so).
	for _, c := range cands {
		if nt, np, nn := spans(); total >= holdoutMinTotal && nt >= 3 && np >= 3 && nn >= 2 {
			break
		}
		if len(picked) >= holdoutMaxCells || total+c.n > holdoutMaxTotal {
			continue
		}
		if topicCells[c.cell.Topic] >= holdoutTopicCap {
			continue
		}
		if !retains(c.cell, c.n, pickedTop, pickedTopic, pickedTone) {
			continue
		}
		picked[c.cell] = true
		total += c.n
		pickedTop[c.cell.Top] += c.n
		ti := c.cell.Topic
		if ti < 0 {
			ti = holdoutTopicFams
		}
		pickedTopic[ti] += c.n
		pickedTone[c.cell.Tone] += c.n
		topicCells[c.cell.Topic]++
	}
	census.SpanTops, census.SpanTopics, census.SpanTones = spans()
	for cell := range picked {
		census.Chosen = append(census.Chosen, cell)
	}
	sort.Slice(census.Chosen, func(a, b int) bool {
		ca, cb := census.Chosen[a], census.Chosen[b]
		if ca.Top != cb.Top {
			return ca.Top < cb.Top
		}
		if ca.Topic != cb.Topic {
			return ca.Topic < cb.Topic
		}
		return ca.Tone < cb.Tone
	})
	census.ChosenPairs = total

	// Test items: every pair of every chosen cell, in corpus
	// order. Pairs whose reference reply has no content words
	// cannot be judged by the registered rule and are excluded
	// (counted, and they stay OUT of the reduced corpus too —
	// they are part of the held-out cells).
	heldLines := map[int]bool{}
	var testItems []holdoutTestItem
	var kept []tagged
	for _, tg := range social {
		if tg.ok && picked[tg.cell] {
			heldLines[tg.line] = true
			if len(tg.want) == 0 {
				census.Unjudgeable++
				continue
			}
			testItems = append(testItems, holdoutTestItem{
				Input: tg.pair.Input, Output: tg.pair.Output,
				Top: tg.cell.Top, Topic: tg.cell.Topic, Tone: tg.cell.Tone,
				Want: tg.want,
			})
		} else if tg.ok {
			kept = append(kept, tg)
		}
	}
	// Seen sample: a seeded sample of kept-cell pairs.
	rng := rand.New(rand.NewSource(layersSeed))
	perm := rng.Perm(len(kept))
	var seenItems []holdoutTestItem
	for _, pi := range perm {
		if len(seenItems) >= holdoutSeenSample {
			break
		}
		tg := kept[pi]
		if len(tg.want) == 0 {
			continue
		}
		seenItems = append(seenItems, holdoutTestItem{
			Input: tg.pair.Input, Output: tg.pair.Output,
			Top: tg.cell.Top, Topic: tg.cell.Topic, Tone: tg.cell.Tone,
			Want: tg.want,
		})
	}
	census.SeenSample = len(seenItems)

	// Reduced corpus: every line except the held-out cells' lines.
	var reduced []string
	for li, line := range lines {
		if !heldLines[li] {
			reduced = append(reduced, line)
		}
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	writeItems := func(name string, items []holdoutTestItem) error {
		var b strings.Builder
		enc := json.NewEncoder(&b)
		for _, it := range items {
			if err := enc.Encode(it); err != nil {
				return err
			}
		}
		return os.WriteFile(filepath.Join(outDir, name), []byte(b.String()), 0o644)
	}
	if err := writeItems("holdout_test.jsonl", testItems); err != nil {
		return err
	}
	if err := writeItems("holdout_seen.jsonl", seenItems); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "chat_pairs_reduced.jsonl"), []byte(strings.Join(reduced, "\n")+"\n"), 0o644); err != nil {
		return err
	}
	for _, name := range []string{"holdout_test.jsonl", "holdout_seen.jsonl", "chat_pairs_reduced.jsonl"} {
		data, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(data)
		census.Files[name] = hex.EncodeToString(sum[:])
	}
	censusData, err := json.MarshalIndent(census, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "CENSUS.json"), append(censusData, '\n'), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "CENSUS.md"), []byte(census.Markdown(cellPairs)), 0o644); err != nil {
		return err
	}
	var sums strings.Builder
	for _, name := range []string{"holdout_test.jsonl", "holdout_seen.jsonl", "chat_pairs_reduced.jsonl", "CENSUS.json", "CENSUS.md"} {
		data, err := os.ReadFile(filepath.Join(outDir, name))
		if err != nil {
			return err
		}
		sum := sha256.Sum256(data)
		fmt.Fprintf(&sums, "%s  %s\n", hex.EncodeToString(sum[:]), name)
	}
	return os.WriteFile(filepath.Join(outDir, "HOLDOUT.sha256"), []byte(sums.String()), 0o644)
}

// Markdown renders the human-readable census.
func (c *holdoutCensus) Markdown(cellPairs map[holdoutCell][]int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Combination holdout v%d — census\n\n", c.Version)
	fmt.Fprintf(&b, "Seed %d; top codes K=%d (whole-reply BoW k-means over the prepared train pairs);\n", c.Seed, c.NumTopCodes)
	fmt.Fprintf(&b, "topic families F=%d (input content-word BoW k-means; -1 = no in-vocab content words);\n", c.TopicFamilies)
	fmt.Fprintf(&b, "tone by the lexicon rule. Derivations are the frame derivations of DESIGN §12.\n\n")
	fmt.Fprintf(&b, "Corpus: %d lines, %d social pairs tagged. Cells total: %d; candidate cells (>= %d pairs): %d.\n\n",
		c.CorpusPairs, c.SocialPairs, c.CellsTotal, holdoutMinCell, c.Candidates)
	fmt.Fprintf(&b, "Pairs per top code: %v\n\n", c.TopCensus)
	fmt.Fprintf(&b, "Pairs per topic family (last entry = topic -1): %v\n\n", c.TopicCensus)
	fmt.Fprintf(&b, "Pairs per tone [neutral bright low worried]: %v\n\n", c.ToneCensus)
	fmt.Fprintf(&b, "## Chosen cells (%d pairs in %d cells)\n\n", c.ChosenPairs, len(c.Chosen))
	for _, cell := range c.Chosen {
		fmt.Fprintf(&b, "- top=%d topic=%d tone=%d: %d pairs\n", cell.Top, cell.Topic, cell.Tone, len(cellPairs[cell]))
	}
	fmt.Fprintf(&b, "\nChosen spans: %d top codes, %d topic families, %d tones.\n", c.SpanTops, c.SpanTopics, c.SpanTones)
	fmt.Fprintf(&b, "Excluded as unjudgeable (reference reply has no content words): %d pairs.\n", c.Unjudgeable)
	fmt.Fprintf(&b, "Seen-cell sample: %d pairs.\n\n", c.SeenSample)
	fmt.Fprintf(&b, "Selection rules (registered, DESIGN §12): candidates are cells with >= %d pairs,\n", holdoutMinCell)
	fmt.Fprintf(&b, "ordered by count desc then (top, topic, tone) asc; one pass picking cells the\n")
	fmt.Fprintf(&b, "retention rule allows, at most %d cells per topic family, never above %d pairs\n", holdoutTopicCap, holdoutMaxTotal)
	fmt.Fprintf(&b, "or %d cells in total; the pass stops once the set reaches >= %d pairs AND spans\n", holdoutMaxCells, holdoutMinTotal)
	fmt.Fprintf(&b, ">= 3 top codes, >= 3 topic families, and >= 2 tones. A cell is skipped when\n")
	fmt.Fprintf(&b, "removing it would leave its top code, topic family, or tone with fewer than\n")
	fmt.Fprintf(&b, "%d pairs outside the chosen cells. Census fact: tones low and worried have no\n", holdoutRetainEach)
	fmt.Fprintf(&b, "candidate cells (their pairs are spread across tops/topics below the minimum\n")
	fmt.Fprintf(&b, "cell mass), so tone recombination is tested across neutral x bright only.\n\n")
	b.WriteString("File hashes (sha256):\n\n")
	for _, name := range []string{"chat_pairs_reduced.jsonl", "holdout_seen.jsonl", "holdout_test.jsonl"} {
		fmt.Fprintf(&b, "- %s: %s\n", name, c.Files[name])
	}
	return b.String()
}
