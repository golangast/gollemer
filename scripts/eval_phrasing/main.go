// eval_phrasing scores the phrasing-layer model (messy
// sentence -> canonical sentence(s)) over the generator's
// held-out matrix (data/training/phrasing_heldout.jsonl). It
// answers the one question the layer's seat depends on: can a
// small pointer-generator RECOGNIZE an unseen phrasing of a
// seen shape — right command, every slot exact — when the slot
// values themselves are words it never trained on?
//
// Scoring is the production path, not a proxy: predictions come
// from seq2seq.PredictPtrGenVerbatim (the chat layer's own
// decoder), normalized exactly as the layer normalizes them
// (quote spacing collapsed, pieces split on " ## "), and an
// item is exact only when every piece string equals the gold
// canonical, in order. The "none" gold scores exact only when
// the prediction is "none".
//
// The pre-registered bar (set before training, 2026-10-30):
//
//	single-action, unseen phrasings of seen shapes: >= 90% exact
//	compounds, seen pairings + unseen phrasings:    >= 85% fully exact
//
// Cells reported: single-unseen (the bar cell, with a
// write-shapes-only subtotal that excludes the none class),
// single-seen (control: trained phrasing families over
// held-out slot pools), compound-seenpair (bar cell),
// compound-unseenpair (frontier, no bar). Plus per-family
// exactness and a confusion summary: misses by gold family
// with samples, none-class false claims, and claims lost to
// "none".
//
// Run: go run ./scripts/eval_phrasing [-model path] [-heldout path] [-out path] [-limit N]
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

type heldItem struct {
	Input    string `json:"input"`
	Output   string `json:"output"`
	Family   string `json:"family"`
	Cell     string `json:"cell"`
	Template string `json:"template"`
}

type cellStats struct {
	N          int     `json:"n"`
	Exact      int     `json:"exact"`
	ExactPct   float64 `json:"exact_pct"`
	PieceExact int     `json:"piece_exact"` // compound diagnostic: pieces right, of pieces total
	PieceTotal int     `json:"piece_total"`
}

type itemResult struct {
	Input  string `json:"input"`
	Gold   string `json:"gold"`
	Raw    string `json:"raw_output"`
	Family string `json:"family"`
	Cell   string `json:"cell"`
	Exact  bool   `json:"exact"`
}

type evalReport struct {
	Model             string                `json:"model"`
	Cells             map[string]*cellStats `json:"cells"`
	Families          map[string]*cellStats `json:"families"`
	SingleUnseenWrite *cellStats            `json:"single_unseen_write_only"`
	BarSingle         float64               `json:"bar_single_pct"`
	BarCompound       float64               `json:"bar_compound_pct"`
	SinglePass        bool                  `json:"single_bar_pass"`
	CompoundPass      bool                  `json:"compound_bar_pass"`
	NoneFalseClaims   int                   `json:"none_false_claims"`
	ClaimsLostToNone  int                   `json:"claims_lost_to_none"`
	MissSamplesByGold map[string][]string   `json:"miss_samples_by_gold_family"`
	Results           []itemResult          `json:"results"`
}

// phrasingPieces normalizes a raw model output into the pieces
// the chat layer would validate: quote spacing collapsed (the
// decoder space-joins its tokens, so a quoted print text comes
// back as `" the results "`), then split on the separator.
// "none" and "" yield no pieces. This mirrors
// chat.PhrasingPieces — the two must not drift.
//
// The collapse is token-based, not regex: a `"` field toggles
// in-quote state, and fields are rejoined with no space after
// an opening quote and no space before a closing quote. (A
// regex alternation eats the space before an opening quote
// first and leaves the inner space standing — measured bug,
// print family scored 0/72 before this fix.)
func phrasingPieces(raw string) []string {
	s := collapseQuoteSpacing(raw)
	if s == "" || s == "none" {
		return nil
	}
	var out []string
	for _, p := range strings.Split(s, " ## ") {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// collapseQuoteSpacing rejoins the decoder's space-joined
// tokens so a quoted span reads as `"text"` with no inner
// padding, matching the canonical targets byte for byte.
func collapseQuoteSpacing(raw string) string {
	fields := strings.Fields(raw)
	var b strings.Builder
	inQuote := false
	prevOpen := false // previous field was an opening quote
	for i, f := range fields {
		closing := f == `"` && inQuote
		if i > 0 && !prevOpen && !closing {
			b.WriteByte(' ')
		}
		b.WriteString(f)
		prevOpen = false
		if f == `"` {
			if inQuote {
				inQuote = false
			} else {
				inQuote = true
				prevOpen = true
			}
		}
	}
	return b.String()
}

func goldPieces(output string) []string {
	if output == "none" {
		return nil
	}
	return strings.Split(output, " ## ")
}

func main() {
	modelPath := flag.String("model", "data/models/gob_models/real_tiny_seq2seq_phrasing.gob", "trained phrasing model")
	heldoutPath := flag.String("heldout", "data/training/phrasing_heldout.jsonl", "held-out eval items")
	outPath := flag.String("out", "data/training/phrasing_eval_report.json", "report output (empty = stdout only)")
	limit := flag.Int("limit", 0, "score only the first N items (0 = all)")
	flag.Parse()

	items, err := loadItems(*heldoutPath)
	if err != nil {
		fail(err)
	}
	if *limit > 0 && *limit < len(items) {
		items = items[:*limit]
	}
	fmt.Printf("scoring %d held-out items against %s\n", len(items), *modelPath)

	placeholder, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		fail(err)
	}
	model, err := seq2seq.Load(*modelPath, placeholder)
	if err != nil {
		fail(fmt.Errorf("load model: %w (train it first: go run . -train-real-seq2seq -domain phrasing)", err))
	}
	if model.OutputVocab == nil || model.OutputVocab.Size() == 0 {
		fail(fmt.Errorf("model has no vocabulary"))
	}
	if model.Decoder == nil || model.Decoder.PtrGen == nil {
		fail(fmt.Errorf("model was trained without the pointer-generator head"))
	}

	rep := &evalReport{
		Model:             *modelPath,
		Cells:             map[string]*cellStats{},
		Families:          map[string]*cellStats{},
		SingleUnseenWrite: &cellStats{},
		BarSingle:         90,
		BarCompound:       85,
		MissSamplesByGold: map[string][]string{},
	}
	cell := func(m map[string]*cellStats, k string) *cellStats {
		c := m[k]
		if c == nil {
			c = &cellStats{}
			m[k] = c
		}
		return c
	}

	for _, item := range items {
		raw, perr := model.PredictPtrGenVerbatim(item.Input, 96)
		if perr != nil {
			raw = ""
		}
		got := phrasingPieces(raw)
		gold := goldPieces(item.Output)
		exact := equalStrings(got, gold)

		cs := cell(rep.Cells, item.Cell)
		fs := cell(rep.Families, item.Family)
		for _, c := range []*cellStats{cs, fs} {
			c.N++
			if exact {
				c.Exact++
			}
			// Piece-level diagnostic (compounds mainly): how
			// many gold pieces were produced exactly, aligned
			// by position.
			for i, gp := range gold {
				c.PieceTotal++
				if i < len(got) && got[i] == gp {
					c.PieceExact++
				}
			}
			if len(gold) == 0 {
				c.PieceTotal++
				if len(got) == 0 {
					c.PieceExact++
				}
			}
		}
		if item.Cell == "single-unseen" && item.Family != "none" {
			rep.SingleUnseenWrite.N++
			if exact {
				rep.SingleUnseenWrite.Exact++
			}
		}
		if item.Family == "none" && len(got) > 0 {
			rep.NoneFalseClaims++
		}
		if item.Family != "none" && len(got) == 0 && raw != "" {
			rep.ClaimsLostToNone++
		}
		if !exact && len(rep.MissSamplesByGold[item.Family]) < 3 {
			rep.MissSamplesByGold[item.Family] = append(rep.MissSamplesByGold[item.Family],
				fmt.Sprintf("in=%q gold=%q raw=%q", item.Input, item.Output, raw))
		}
		rep.Results = append(rep.Results, itemResult{
			Input: item.Input, Gold: item.Output, Raw: raw,
			Family: item.Family, Cell: item.Cell, Exact: exact,
		})
	}

	for _, m := range []map[string]*cellStats{rep.Cells, rep.Families} {
		for _, c := range m {
			c.ExactPct = pct(c.Exact, c.N)
		}
	}
	rep.SingleUnseenWrite.ExactPct = pct(rep.SingleUnseenWrite.Exact, rep.SingleUnseenWrite.N)
	if c := rep.Cells["single-unseen"]; c != nil {
		rep.SinglePass = c.N > 0 && pct(c.Exact, c.N) >= rep.BarSingle
	}
	if c := rep.Cells["compound-seenpair"]; c != nil {
		rep.CompoundPass = c.N > 0 && pct(c.Exact, c.N) >= rep.BarCompound
	}

	for _, k := range sortedKeys(rep.Cells) {
		c := rep.Cells[k]
		fmt.Printf("%-24s N=%-5d exact=%5.1f%% pieces=%5.1f%%\n", k, c.N, c.ExactPct, pct(c.PieceExact, c.PieceTotal))
	}
	fmt.Printf("%-24s N=%-5d exact=%5.1f%% (write shapes only, none class excluded)\n",
		"single-unseen-write", rep.SingleUnseenWrite.N, rep.SingleUnseenWrite.ExactPct)
	fmt.Printf("bar: singles >= %.0f%% -> %v; compounds >= %.0f%% -> %v\n",
		rep.BarSingle, rep.SinglePass, rep.BarCompound, rep.CompoundPass)
	fmt.Printf("none-class false claims: %d; claims lost to none: %d\n", rep.NoneFalseClaims, rep.ClaimsLostToNone)
	fmt.Println("families under 90%:")
	for _, k := range sortedKeys(rep.Families) {
		c := rep.Families[k]
		if c.ExactPct < 90 && c.N >= 5 {
			fmt.Printf("  %-28s N=%-4d exact=%5.1f%%\n", k, c.N, c.ExactPct)
		}
	}

	if *outPath != "" {
		raw, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			fail(err)
		}
		if err := os.WriteFile(*outPath, raw, 0o644); err != nil {
			fail(err)
		}
		fmt.Println("report written to", *outPath)
	}
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sortedKeys(m map[string]*cellStats) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func loadItems(path string) ([]heldItem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []heldItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var item heldItem
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, sc.Err()
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "eval_phrasing:", err)
	os.Exit(1)
}
