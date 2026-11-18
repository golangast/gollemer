// eval_phrasing2 scores phrasing-layer attempt 2 (2026-10-31)
// against the SAME judge run 1 faced — data/training/
// phrasing_heldout.jsonl, unmodified — under the metric
// declared before any attempt-2 training: SLOT-EXACT. A
// prediction's canonical sentence(s) are parsed by the real
// deterministic parsers (chat.CanonicalSlots) into (family,
// slot map) and compared with the gold's; an unparseable
// prediction scores wrong. Raw string-exact (run 1's metric)
// is printed alongside so the 59.0% comparison stays visible.
//
// Modes:
//
//	-mode goldcheck  parse every gold output; report parse rate
//	                 and single-item family agreement (judge sanity)
//	-mode baseline   run 1's gob + PredictPtrGenVerbatim
//	-mode a          arm A's gob + PredictPtrGenVerbatimIndexed
//	-mode b          arm B: chat.PhrasingRoute (classify+extract)
//
// The pre-registered bar is unchanged: single-unseen >= 90%
// slot-exact to integrate singles; compound-seenpair >= 85%
// fully slot-exact to integrate compounds. Compound-unseenpair
// is a frontier number with no bar.
//
// Run: go run ./scripts/eval_phrasing2 -mode baseline [-out path]
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
	"github.com/golangast/gollemer/internal/ai/training/chat"
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
	SlotExact  int     `json:"slot_exact"`
	SlotPct    float64 `json:"slot_exact_pct"`
	StrExact   int     `json:"string_exact"`
	StrPct     float64 `json:"string_exact_pct"`
	PieceExact int     `json:"piece_slot_exact"`
	PieceTotal int     `json:"piece_total"`
}

type itemResult struct {
	Input     string `json:"input"`
	Gold      string `json:"gold"`
	Raw       string `json:"raw_output"`
	Family    string `json:"family"`
	Cell      string `json:"cell"`
	SlotExact bool   `json:"slot_exact"`
	StrExact  bool   `json:"string_exact"`
}

type evalReport struct {
	Mode              string                `json:"mode"`
	Model             string                `json:"model,omitempty"`
	Cells             map[string]*cellStats `json:"cells"`
	Families          map[string]*cellStats `json:"families"`
	SingleUnseenWrite *cellStats            `json:"single_unseen_write_only"`
	BarSingle         float64               `json:"bar_single_pct"`
	BarCompound       float64               `json:"bar_compound_pct"`
	SinglePass        bool                  `json:"single_bar_pass_slotexact"`
	CompoundPass      bool                  `json:"compound_bar_pass_slotexact"`
	NoneFalseClaims   int                   `json:"none_false_claims"`
	ClaimsLostToNone  int                   `json:"claims_lost_to_none"`
	Results           []itemResult          `json:"results"`
}

// phrasingPieces normalizes a raw model output into pieces:
// quote spacing collapsed (token-based; mirrors eval_phrasing),
// split on the separator. "none"/"" yield no pieces.
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

func collapseQuoteSpacing(raw string) string {
	fields := strings.Fields(raw)
	var b strings.Builder
	inQuote := false
	prevOpen := false
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
	mode := flag.String("mode", "baseline", "goldcheck|baseline|a|b")
	modelPath := flag.String("model", "", "model gob (baseline/a modes)")
	heldoutPath := flag.String("heldout", "data/training/phrasing_heldout.jsonl", "held-out judge items")
	outPath := flag.String("out", "", "report output (empty = stdout only)")
	limit := flag.Int("limit", 0, "score only the first N items (0 = all)")
	flag.Parse()

	items, err := loadItems(*heldoutPath)
	if err != nil {
		fail(err)
	}
	if *limit > 0 && *limit < len(items) {
		items = items[:*limit]
	}

	if *mode == "goldcheck" {
		goldCheck(items)
		return
	}

	var predict func(input string) (raw string, pieces []string)
	modelDesc := ""
	switch *mode {
	case "baseline", "a":
		mp := *modelPath
		if mp == "" {
			if *mode == "baseline" {
				mp = "data/models/gob_models/real_tiny_seq2seq_phrasing.gob"
			} else {
				mp = "data/models/gob_models/real_tiny_seq2seq_phrasing2.gob"
			}
		}
		modelDesc = mp
		model := loadModel(mp)
		if *mode == "baseline" {
			predict = func(input string) (string, []string) {
				raw, err := model.PredictPtrGenVerbatim(input, 96)
				if err != nil {
					raw = ""
				}
				return raw, phrasingPieces(raw)
			}
		} else {
			predict = func(input string) (string, []string) {
				raw, err := model.PredictPtrGenVerbatimIndexed(input, 96)
				if err != nil {
					raw = ""
				}
				return raw, phrasingPieces(raw)
			}
		}
	case "b":
		modelDesc = "arm-b classify+extract (chat.PhrasingRoute)"
		predict = func(input string) (string, []string) {
			pieces, ok := chat.PhrasingRoute(input)
			if !ok {
				return "", nil
			}
			return strings.Join(pieces, " ## "), pieces
		}
	default:
		fail(fmt.Errorf("unknown mode %q", *mode))
	}

	fmt.Printf("scoring %d held-out items, mode=%s model=%s\n", len(items), *mode, modelDesc)
	rep := &evalReport{
		Mode:              *mode,
		Model:             modelDesc,
		Cells:             map[string]*cellStats{},
		Families:          map[string]*cellStats{},
		SingleUnseenWrite: &cellStats{},
		BarSingle:         90,
		BarCompound:       85,
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
		raw, got := predict(item.Input)
		gold := goldPieces(item.Output)
		strExact := equalStrings(got, gold)

		goldSlots, gok := chat.CanonicalPiecesSlots(item.Output)
		var slotExact bool
		var pieceHits, pieceTotal int
		if gok {
			if len(goldSlots) == 0 && len(got) == 0 {
				slotExact = true // none == none
				pieceHits, pieceTotal = 1, 1
			} else if len(got) > 0 {
				gotJoined := strings.Join(got, " ## ")
				gotSlots, ok := chat.CanonicalPiecesSlots(gotJoined)
				if ok {
					slotExact = chat.PiecesSlotExact(gotSlots, goldSlots)
					for i := range goldSlots {
						pieceTotal++
						if i < len(gotSlots) && gotSlots[i].Family == goldSlots[i].Family &&
							chat.SlotsEqual(gotSlots[i].Slots, goldSlots[i].Slots) {
							pieceHits++
						}
					}
				} else {
					pieceTotal = len(goldSlots)
				}
			} else {
				pieceTotal = len(goldSlots)
			}
		}

		for _, c := range []*cellStats{cell(rep.Cells, item.Cell), cell(rep.Families, item.Family)} {
			c.N++
			if slotExact {
				c.SlotExact++
			}
			if strExact {
				c.StrExact++
			}
			c.PieceExact += pieceHits
			c.PieceTotal += pieceTotal
		}
		if item.Cell == "single-unseen" && item.Family != "none" {
			rep.SingleUnseenWrite.N++
			if slotExact {
				rep.SingleUnseenWrite.SlotExact++
			}
			if strExact {
				rep.SingleUnseenWrite.StrExact++
			}
		}
		if item.Family == "none" && len(got) > 0 {
			rep.NoneFalseClaims++
		}
		if item.Family != "none" && len(got) == 0 && raw != "" {
			rep.ClaimsLostToNone++
		}
		rep.Results = append(rep.Results, itemResult{
			Input: item.Input, Gold: item.Output, Raw: raw,
			Family: item.Family, Cell: item.Cell, SlotExact: slotExact, StrExact: strExact,
		})
	}

	for _, m := range []map[string]*cellStats{rep.Cells, rep.Families} {
		for _, c := range m {
			c.SlotPct = pct(c.SlotExact, c.N)
			c.StrPct = pct(c.StrExact, c.N)
		}
	}
	rep.SingleUnseenWrite.SlotPct = pct(rep.SingleUnseenWrite.SlotExact, rep.SingleUnseenWrite.N)
	rep.SingleUnseenWrite.StrPct = pct(rep.SingleUnseenWrite.StrExact, rep.SingleUnseenWrite.N)
	if c := rep.Cells["single-unseen"]; c != nil {
		rep.SinglePass = c.N > 0 && c.SlotPct >= rep.BarSingle
	}
	if c := rep.Cells["compound-seenpair"]; c != nil {
		rep.CompoundPass = c.N > 0 && c.SlotPct >= rep.BarCompound
	}

	for _, k := range sortedKeys(rep.Cells) {
		c := rep.Cells[k]
		fmt.Printf("%-24s N=%-5d slot-exact=%5.1f%% string-exact=%5.1f%% pieces=%5.1f%%\n",
			k, c.N, c.SlotPct, c.StrPct, pct(c.PieceExact, c.PieceTotal))
	}
	fmt.Printf("%-24s N=%-5d slot-exact=%5.1f%% string-exact=%5.1f%% (write shapes only)\n",
		"single-unseen-write", rep.SingleUnseenWrite.N, rep.SingleUnseenWrite.SlotPct, rep.SingleUnseenWrite.StrPct)
	fmt.Printf("bar (slot-exact): single-unseen (full cell) >= %.0f%% -> %v; compound-seenpair >= %.0f%% -> %v\n",
		rep.BarSingle, rep.SinglePass, rep.BarCompound, rep.CompoundPass)
	fmt.Printf("none-class false claims: %d; claims lost to none: %d\n", rep.NoneFalseClaims, rep.ClaimsLostToNone)
	fmt.Println("families under 90% slot-exact (N>=5):")
	for _, k := range sortedKeys(rep.Families) {
		c := rep.Families[k]
		if c.SlotPct < 90 && c.N >= 5 {
			fmt.Printf("  %-28s N=%-4d slot=%5.1f%% string=%5.1f%%\n", k, c.N, c.SlotPct, c.StrPct)
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

// goldCheck verifies the judge itself: every gold output must
// parse through the real parsers, and for single items the
// parsed family must equal the item's declared family.
func goldCheck(items []heldItem) {
	parseFail, famMismatch, noneBad := 0, 0, 0
	for _, item := range items {
		slots, ok := chat.CanonicalPiecesSlots(item.Output)
		if !ok {
			parseFail++
			if parseFail <= 10 {
				fmt.Printf("PARSE FAIL family=%s gold=%q\n", item.Family, item.Output)
			}
			continue
		}
		if item.Family == "none" {
			if len(slots) != 0 {
				noneBad++
			}
			continue
		}
		if !strings.HasPrefix(item.Family, "compound:") {
			if len(slots) != 1 || slots[0].Family != item.Family {
				famMismatch++
				if famMismatch <= 10 {
					got := ""
					if len(slots) == 1 {
						got = slots[0].Family
					}
					fmt.Printf("FAMILY MISMATCH want=%s got=%s gold=%q\n", item.Family, got, item.Output)
				}
			}
		} else if len(slots) != 2 {
			famMismatch++
			fmt.Printf("COMPOUND PIECE COUNT family=%s pieces=%d gold=%q\n", item.Family, len(slots), item.Output)
		}
	}
	fmt.Printf("goldcheck: %d items, parse failures=%d, family mismatches=%d, none bad=%d\n",
		len(items), parseFail, famMismatch, noneBad)
}

func loadModel(path string) *seq2seq.Seq2Seq {
	placeholder, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		fail(err)
	}
	model, err := seq2seq.Load(path, placeholder)
	if err != nil {
		fail(fmt.Errorf("load model: %w", err))
	}
	if model.OutputVocab == nil || model.OutputVocab.Size() == 0 {
		fail(fmt.Errorf("model has no vocabulary"))
	}
	if model.Decoder == nil || model.Decoder.PtrGen == nil {
		fail(fmt.Errorf("model was trained without the pointer-generator head"))
	}
	return model
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
	fmt.Fprintln(os.Stderr, "eval_phrasing2:", err)
	os.Exit(1)
}
