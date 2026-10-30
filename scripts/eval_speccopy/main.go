// eval_speccopy scores the copy/pointer probe model (phrasing ->
// structured spec) — a clone of eval_spec with the model and its
// decoding swapped for the pointer-generator mixture
// (seq2seq.PredictPtrGen, greedy, no constrained decoding, no
// round-trip retry). Every scoring rule is eval_spec's own, so the
// numbers compare head-to-head with the spec probe.
// over the PHRASING items of the decl held-out matrix
// (data/training/decl_heldout.jsonl). It answers the re-seat question:
// if the model's only job is producing the spec and rendering stays
// deterministic, how exact is the spec on combinations it never saw?
//
// Per cell (combo x frame) it reports:
//
//   - whole-spec exact % (the headline): the prediction parsed by
//     decl.ParseSpecText and compared to the gold spec with the decl
//     eval's conventions (decl.CompareSpec in the LowerSpec world,
//     tags by presence — spec text carries no tag content — and
//     receiver names scored only where the input frame gave one,
//     frame M-T1, mirroring decl eval's rule for phrasing items).
//   - parse % of predictions, and the emittable ceiling: gold spec
//     texts the greedy decoder can produce at all (no repeated
//     bigram, within the token cap). Exact % is also given against
//     that ceiling.
//   - slot accuracies: kind, declaration name, member count, member
//     names (sequence-exact), member types (per aligned member),
//     embeds, receiver (presence/type/pointer; name on the strict
//     subset). Unparsed predictions score zero on every slot.
//   - the pipeline cross-check: each parsed spec is rendered with
//     the deterministic renderer (decl.RestoreSpec + decl.RenderDecl)
//     and the result AST-compared to the gold declaration exactly as
//     the decl eval scores code — the end-to-end number the re-seat
//     would deliver.
//
// Run: go run ./scripts/eval_speccopy [-model path] [-heldout path] [-out path] [-limit N]
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
	"github.com/golangast/gollemer/internal/ai/training/decl"
)

type slotStats struct {
	KindOK           int `json:"kind_ok"`
	NameOK           int `json:"name_ok"`
	MemberCountOK    int `json:"member_count_ok"`
	MemberNamesOK    int `json:"member_names_ok"`
	MemberTypesOK    int `json:"member_types_ok"`
	MemberTypesTotal int `json:"member_types_total"`
	EmbedsN          int `json:"embeds_n"`
	EmbedsOK         int `json:"embeds_ok"`
	ReceiverN        int `json:"receiver_n"`
	ReceiverOK       int `json:"receiver_ok"`
	ReceiverNameN    int `json:"receiver_name_n"`
	ReceiverNameOK   int `json:"receiver_name_ok"`
}

type cellStats struct {
	N              int       `json:"n"`
	Exact          int       `json:"exact"`
	Parsed         int       `json:"parsed"`
	PipelineExact  int       `json:"pipeline_exact"`
	PipeRestoreErr int       `json:"pipeline_restore_failures"`
	PipeExtractErr int       `json:"pipeline_extract_failures"`
	Emittable      int       `json:"gold_emittable"`
	Slots          slotStats `json:"slots"`
}

type itemResult struct {
	ID       string `json:"id"`
	Cell     string `json:"cell"`
	Category string `json:"category"`
	Parsed   bool   `json:"parsed"`
	Exact    bool   `json:"exact"`
	Pipeline bool   `json:"pipeline_exact"`
	Raw      string `json:"raw_output"`
}

type evalReport struct {
	Model   string                `json:"model"`
	Cells   map[string]*cellStats `json:"cells"`
	Overall *cellStats            `json:"overall"`
	Results []itemResult          `json:"results"`
}

func main() {
	modelPath := flag.String("model", "data/models/gob_models/real_tiny_seq2seq_speccopy.gob", "trained copy/pointer probe model")
	heldoutPath := flag.String("heldout", "data/training/decl_heldout.jsonl", "held-out eval items (decl matrix)")
	outPath := flag.String("out", "data/training/speccopy_eval_report.json", "report output (empty = stdout only)")
	limit := flag.Int("limit", 0, "score only the first N phrasing items (0 = all)")
	flag.Parse()

	items, err := loadItems(*heldoutPath)
	if err != nil {
		fail(err)
	}
	var phrasing []decl.EvalItem
	for _, item := range items {
		if item.InputForm == "phrasing" {
			phrasing = append(phrasing, item)
		}
	}
	if *limit > 0 && *limit < len(phrasing) {
		phrasing = phrasing[:*limit]
	}
	fmt.Printf("scoring %d phrasing items against %s\n", len(phrasing), *modelPath)

	placeholder, err := tokenizer.NewTokenizer(vocab.NewVocabulary())
	if err != nil {
		fail(err)
	}
	model, err := seq2seq.Load(*modelPath, placeholder)
	if err != nil {
		fail(fmt.Errorf("load model: %w (train it first: go run . -train-real-seq2seq -domain speccopy)", err))
	}
	if model.OutputVocab == nil || model.OutputVocab.Size() == 0 {
		fail(fmt.Errorf("model has no vocabulary"))
	}
	tok, err := tokenizer.NewTokenizer(model.OutputVocab)
	if err != nil {
		fail(err)
	}
	model.Tokenizer = tok
	if model.Decoder == nil || model.Decoder.PtrGen == nil {
		fail(fmt.Errorf("model was trained without the pointer-generator head"))
	}

	rep := &evalReport{Model: *modelPath, Cells: map[string]*cellStats{}, Overall: &cellStats{}}
	for _, item := range phrasing {
		cellKey := "phrasing|" + item.Combo + "|" + item.Frame
		cell := rep.Cells[cellKey]
		if cell == nil {
			cell = &cellStats{}
			rep.Cells[cellKey] = cell
		}
		res := evaluate(model, item, cell, rep.Overall)
		res.Cell = cellKey
		rep.Results = append(rep.Results, res)
	}

	keys := make([]string, 0, len(rep.Cells))
	for k := range rep.Cells {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		printCell(k, rep.Cells[k])
	}
	printCell("OVERALL", rep.Overall)

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

func printCell(label string, c *cellStats) {
	pct := func(a, b int) float64 {
		if b == 0 {
			return 0
		}
		return 100 * float64(a) / float64(b)
	}
	fmt.Printf("%-26s N=%-5d exact=%5.1f%% (of emittable %5.1f%%) parse=%5.1f%% pipeline=%5.1f%%\n",
		label, c.N, pct(c.Exact, c.N), pct(c.Exact, c.Emittable), pct(c.Parsed, c.N), pct(c.PipelineExact, c.N))
	fmt.Printf("%-26s slots: kind=%5.1f%% name=%5.1f%% count=%5.1f%% names=%5.1f%% types=%5.1f%% (%d/%d) embeds=%5.1f%% (n=%d) receiver=%5.1f%% (n=%d) recvname=%5.1f%% (n=%d)\n",
		"", pct(c.Slots.KindOK, c.N), pct(c.Slots.NameOK, c.N), pct(c.Slots.MemberCountOK, c.N),
		pct(c.Slots.MemberNamesOK, c.N), pct(c.Slots.MemberTypesOK, c.Slots.MemberTypesTotal),
		c.Slots.MemberTypesOK, c.Slots.MemberTypesTotal,
		pct(c.Slots.EmbedsOK, c.Slots.EmbedsN), c.Slots.EmbedsN,
		pct(c.Slots.ReceiverOK, c.Slots.ReceiverN), c.Slots.ReceiverN,
		pct(c.Slots.ReceiverNameOK, c.Slots.ReceiverNameN), c.Slots.ReceiverNameN)
}

// evaluate predicts one item, accumulates its outcome into the cell
// and overall counters, and returns the per-item record.
func evaluate(model *seq2seq.Seq2Seq, item decl.EvalItem, cells ...*cellStats) itemResult {
	res := itemResult{ID: item.ID, Category: decl.CatUnparsable}
	want := decl.LowerSpec(item.Spec)
	// Receiver names are scored only where the input gave one
	// (signature frame), mirroring decl eval's phrasing rule.
	strict := item.FrameID == "M-T1"

	raw, err := model.PredictPtrGen(item.Input, 64)
	if err == nil {
		res.Raw = raw
	}
	var got decl.Spec
	parsed := false
	if err == nil {
		if p, perr := decl.ParseSpecText(raw); perr == nil {
			got, parsed = p, true
		}
	}

	goldText := decl.SpecText(item.Spec)
	emittable := decl.DecodeFeasible(goldText) && len(decl.SplitTokens(goldText)) <= decl.MaxTargetTokens

	pipeExact := false
	restoreFail, extractFail := false, false
	if parsed {
		res.Category, _ = decl.CompareSpec(tagPresence(got), tagPresence(want), strict)
		res.Parsed = true
		res.Exact = res.Category == decl.CatExact
		if restored, ok := decl.RestoreSpec(got); !ok {
			restoreFail = true
		} else if extracted, xerr := decl.ExtractSpec(decl.RenderDecl(restored)); xerr != nil {
			extractFail = true
		} else {
			cat, _ := decl.CompareSpec(tagPresence(decl.LowerSpec(extracted)), tagPresence(want), strict)
			pipeExact = cat == decl.CatExact
		}
	}
	res.Pipeline = pipeExact

	for _, c := range cells {
		c.N++
		if emittable {
			c.Emittable++
		}
		if parsed {
			c.Parsed++
		}
		if res.Exact {
			c.Exact++
		}
		if pipeExact {
			c.PipelineExact++
		}
		if restoreFail {
			c.PipeRestoreErr++
		}
		if extractFail {
			c.PipeExtractErr++
		}
		scoreSlots(&c.Slots, got, want, parsed, strict)
	}
	return res
}

// scoreSlots folds one item's slot outcomes into the counters. All
// comparisons are in the LowerSpec world; an unparsed prediction
// scores zero everywhere but still fills the denominators that do
// not depend on it (embeds/receiver denominators key off the GOLD
// kind, so parse failures count as misses there too).
func scoreSlots(s *slotStats, got, want decl.Spec, parsed, strict bool) {
	if parsed {
		if got.Kind == want.Kind {
			s.KindOK++
		}
		if got.Name == want.Name {
			s.NameOK++
		}
		if memberCount(got) == memberCount(want) {
			s.MemberCountOK++
		}
		if equalStrings(memberKeys(got), memberKeys(want)) {
			s.MemberNamesOK++
		}
	}
	// Member types, per aligned member unit. Kinds that differ give
	// the gold units as denominator with no possible credit.
	gotUnits, wantUnits := memberTypeUnits(got), memberTypeUnits(want)
	if parsed && got.Kind == want.Kind {
		n := len(gotUnits)
		if len(wantUnits) < n {
			n = len(wantUnits)
		}
		for i := 0; i < n; i++ {
			if gotUnits[i] == wantUnits[i] {
				s.MemberTypesOK++
			}
		}
		s.MemberTypesTotal += len(wantUnits)
	} else {
		s.MemberTypesTotal += len(wantUnits)
	}
	// Embeds: gold structs and interfaces only.
	if want.Kind == decl.KindStruct || want.Kind == decl.KindInterface {
		s.EmbedsN++
		if parsed && equalStrings(embedExprs(got), embedExprs(want)) {
			s.EmbedsOK++
		}
	}
	// Receiver: gold methods only.
	if want.Kind == decl.KindFunc && want.Receiver != nil {
		s.ReceiverN++
		if parsed && got.Receiver != nil && got.Receiver.Type == want.Receiver.Type && got.Receiver.Pointer == want.Receiver.Pointer {
			s.ReceiverOK++
		}
		if strict {
			s.ReceiverNameN++
			if parsed && got.Receiver != nil && got.Receiver.Name == want.Receiver.Name &&
				got.Receiver.Type == want.Receiver.Type && got.Receiver.Pointer == want.Receiver.Pointer {
				s.ReceiverNameOK++
			}
		}
	}
}

// memberCount counts a spec's member slots: struct fields (embeds
// included), interface embeds + methods, func params + results.
func memberCount(s decl.Spec) int {
	switch s.Kind {
	case decl.KindStruct:
		return len(s.Fields)
	case decl.KindInterface:
		return len(s.Embeds) + len(s.Methods)
	case decl.KindFunc:
		return len(s.Params) + len(s.Results)
	}
	return 0
}

// memberKeys is the member-name sequence: struct fields by name
// (embeds keyed by type), interface methods by name, func params by
// name.
func memberKeys(s decl.Spec) []string {
	var out []string
	switch s.Kind {
	case decl.KindStruct:
		for _, f := range s.Fields {
			if f.Embedded {
				out = append(out, "embed:"+f.Type.Expr)
			} else {
				out = append(out, "field:"+f.Name)
			}
		}
	case decl.KindInterface:
		for _, m := range s.Methods {
			out = append(out, m.Name)
		}
	case decl.KindFunc:
		for _, p := range s.Params {
			out = append(out, p.Name)
		}
	}
	return out
}

// memberTypeUnits flattens a spec's member types into aligned units:
// one per struct field, one per interface method (its whole
// signature's type sequence), and one per func param and result.
func memberTypeUnits(s decl.Spec) []string {
	var out []string
	switch s.Kind {
	case decl.KindStruct:
		for _, f := range s.Fields {
			out = append(out, f.Type.Expr)
		}
	case decl.KindInterface:
		for _, m := range s.Methods {
			var parts []string
			for _, p := range m.Params {
				parts = append(parts, p.Type.Expr)
			}
			parts = append(parts, "->")
			for _, r := range m.Results {
				parts = append(parts, r.Expr)
			}
			out = append(out, strings.Join(parts, " "))
		}
	case decl.KindFunc:
		for _, p := range s.Params {
			out = append(out, p.Type.Expr)
		}
		for _, r := range s.Results {
			out = append(out, r.Expr)
		}
	}
	return out
}

// embedExprs is the ordered embed type sequence: embedded struct
// fields, or an interface's embeds.
func embedExprs(s decl.Spec) []string {
	var out []string
	switch s.Kind {
	case decl.KindStruct:
		for _, f := range s.Fields {
			if f.Embedded {
				out = append(out, f.Type.Expr)
			}
		}
	case decl.KindInterface:
		for _, e := range s.Embeds {
			out = append(out, e.Expr)
		}
	}
	return out
}

// tagPresence reduces every field tag to a presence marker — the
// decl eval's convention for spec text (see decl.ParseSpecText).
func tagPresence(s decl.Spec) decl.Spec {
	out := s
	out.Fields = append([]decl.Field(nil), s.Fields...)
	for i := range out.Fields {
		if out.Fields[i].Tag != "" {
			out.Fields[i].Tag = "tagged"
		}
	}
	return out
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

func loadItems(path string) ([]decl.EvalItem, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []decl.EvalItem
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var item decl.EvalItem
		if err := json.Unmarshal([]byte(line), &item); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, sc.Err()
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "eval_speccopy:", err)
	os.Exit(1)
}
