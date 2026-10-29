// gen_spec_data generates the spec-probe training set: phrasing ->
// structured-spec pairs derived from the SAME train specs that
// produced data/training/decl_pairs.jsonl (the decl pipeline's
// DefaultGenConfig, seed 20261026). The probe re-seats the decl
// experiment's model: rendering a spec to code is deterministic
// (decl.RenderDecl), so the only learned step worth measuring is
// phrasing -> spec. Training pairs are the decl run's own phrasing
// inputs (same specs, same seen frames) with the target replaced by
// the canonical spec text (decl.SpecText) — the form
// decl.ParseSpecText accepts.
//
// Checks run before anything is written, and the run fails rather
// than emitting a compromised dataset:
//
//   - Regeneration fidelity: decl.Generate(DefaultGenConfig()) must
//     reproduce the shipped decl_pairs.jsonl pair for pair, and the
//     shipped decl_heldout.jsonl unseen-combo specs must be exactly
//     the generator's held-out specs.
//   - Leak check: no held-out spec (full-spec key) appears among the
//     probe's training specs.
//   - Spec-text round-trip property: for every train spec,
//     SpecText -> ParseSpecText -> DeepEqual against the canonical
//     gold in the model's lowercase world. Tag CONTENT is the one
//     documented loss (SpecText writes only "tagged"; ParseSpecText
//     restores the base form), so the property is scored two ways:
//     strict DeepEqual, and DeepEqual with tags reduced to presence
//     (the decl eval's own convention for spec text). The presence
//     form must be 100% and every strict failure must be
//     tag-content-only.
//   - Emittability: targets the greedy decoder can never produce
//     (repeated bigram / immediate repeat, decl.DecodeFeasible) or
//     that exceed the trainer's target cap are excluded and counted
//     — the decl pipeline's standing rule for its own targets.
//
// Outputs (regenerable, intentionally NOT committed):
//
//	data/training/spec_pairs.jsonl   training pairs for -domain spec
//	data/training/spec_report.json   sizes and check results
//
// Run: go run ./scripts/gen_spec_data
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/training/decl"
)

type report struct {
	Seed                  int64          `json:"seed"`
	TrainSpecs            int            `json:"train_specs"`
	HeldoutSpecs          int            `json:"heldout_specs"`
	DeclPhrasingPairs     int            `json:"decl_phrasing_pairs"`
	SpecPairsWritten      int            `json:"spec_pairs_written"`
	SkippedInfeasible     int            `json:"skipped_decode_infeasible_targets"`
	SkippedTooLong        int            `json:"skipped_overlong_targets"`
	DuplicateInputTargets int            `json:"duplicate_inputs_with_conflicting_targets"`
	RegenPairsMatch       bool           `json:"regen_pairs_match_shipped"`
	ShippedHeldoutMatch   bool           `json:"shipped_heldout_specs_match_regen"`
	LeakedHeldoutSpecs    int            `json:"leaked_heldout_specs"`
	RoundTripStrict       int            `json:"roundtrip_strict_deepequal"`
	RoundTripTagOnly      int            `json:"roundtrip_strict_failures_tag_content_only"`
	RoundTripPresence     int            `json:"roundtrip_presence_deepequal"`
	RoundTripTotal        int            `json:"roundtrip_total_specs"`
	EvalGoldEmittable     map[string]int `json:"eval_phrasing_gold_emittable"`
	EvalGoldTotal         map[string]int `json:"eval_phrasing_gold_total"`
	MissingInputTokens    []string       `json:"missing_eval_input_tokens"`
	MissingGoldTokens     []string       `json:"missing_eval_gold_tokens"`
}

func main() {
	outDir := flag.String("out", "data/training", "output directory")
	dataDir := flag.String("data", "data/training", "directory holding the shipped decl datasets")
	flag.Parse()

	cfg := decl.DefaultGenConfig()
	ds, err := decl.Generate(cfg)
	if err != nil {
		fail(fmt.Errorf("generate: %w", err))
	}
	rep := &report{
		Seed:              cfg.Seed,
		TrainSpecs:        len(ds.TrainSpecs),
		HeldoutSpecs:      len(ds.HeldoutSpecs),
		EvalGoldEmittable: map[string]int{},
		EvalGoldTotal:     map[string]int{},
	}

	// --- Regeneration fidelity vs the shipped decl files. ---
	shippedPairs, err := loadJSONL[decl.Pair](filepath.Join(*dataDir, "decl_pairs.jsonl"))
	if err != nil {
		fail(err)
	}
	rep.RegenPairsMatch = reflect.DeepEqual(shippedPairs, ds.TrainPairs)
	if !rep.RegenPairsMatch {
		fail(fmt.Errorf("regenerated pairs do not match shipped decl_pairs.jsonl (%d shipped vs %d regenerated)",
			len(shippedPairs), len(ds.TrainPairs)))
	}
	shippedItems, err := loadJSONL[decl.EvalItem](filepath.Join(*dataDir, "decl_heldout.jsonl"))
	if err != nil {
		fail(err)
	}
	fileHeldout := map[string]bool{}
	for _, item := range shippedItems {
		if item.Combo == "unseen" {
			canon, err := decl.Canonicalize(item.Spec)
			if err != nil {
				fail(fmt.Errorf("canonicalize shipped heldout spec: %w", err))
			}
			fileHeldout[specKey(canon)] = true
		}
	}
	regenHeldout := map[string]bool{}
	for _, s := range ds.HeldoutSpecs {
		regenHeldout[specKey(s)] = true
	}
	rep.ShippedHeldoutMatch = reflect.DeepEqual(fileHeldout, regenHeldout)
	if !rep.ShippedHeldoutMatch {
		fail(fmt.Errorf("shipped held-out specs (%d) != regenerated held-out specs (%d)",
			len(fileHeldout), len(regenHeldout)))
	}

	// --- Leak check: no held-out spec among the training specs. ---
	trainKeys := map[string]bool{}
	for _, s := range ds.TrainSpecs {
		trainKeys[specKey(s)] = true
	}
	for k := range fileHeldout {
		if trainKeys[k] {
			rep.LeakedHeldoutSpecs++
		}
	}
	if rep.LeakedHeldoutSpecs > 0 {
		fail(fmt.Errorf("leak check failed: %d held-out specs appear in training specs", rep.LeakedHeldoutSpecs))
	}

	// --- Spec-text round-trip property over every train spec. ---
	rep.RoundTripTotal = len(ds.TrainSpecs)
	for _, s := range ds.TrainSpecs {
		text := decl.SpecText(s)
		got, err := decl.ParseSpecText(text)
		if err != nil {
			fail(fmt.Errorf("ParseSpecText(%q): %w", text, err))
		}
		want := decl.LowerSpec(s)
		if reflect.DeepEqual(got, want) {
			rep.RoundTripStrict++
			rep.RoundTripPresence++
			continue
		}
		if reflect.DeepEqual(presence(got), presence(want)) {
			rep.RoundTripPresence++
			if tagsOnlyDiff(got, want) {
				rep.RoundTripTagOnly++
				continue
			}
		}
		fail(fmt.Errorf("spec-text round trip mismatch beyond tag content:\n text: %s\n got: %+v\nwant: %+v", text, got, want))
	}
	if rep.RoundTripPresence != rep.RoundTripTotal {
		fail(fmt.Errorf("round-trip property: presence form %d/%d, want 100%%",
			rep.RoundTripPresence, rep.RoundTripTotal))
	}

	// --- Build phrasing -> spec pairs from the decl run's own
	// phrasing pairs. The rendered-decl output identifies the source
	// spec uniquely (verified: no two train specs render alike). ---
	byDecl := map[string]decl.Spec{}
	collisions := 0
	for _, s := range ds.TrainSpecs {
		k := strings.ToLower(decl.RenderDecl(s))
		if prev, ok := byDecl[k]; ok && specKey(prev) != specKey(s) {
			collisions++
			continue
		}
		byDecl[k] = s
	}
	if collisions > 0 {
		fail(fmt.Errorf("%d rendered-decl collisions among train specs; pair->spec mapping is ambiguous", collisions))
	}
	seenInput := map[string]string{}
	var pairs []decl.Pair
	for _, p := range ds.TrainPairs {
		if strings.HasPrefix(p.Input, "spec ") {
			continue // probe trains phrasing inputs only
		}
		rep.DeclPhrasingPairs++
		spec, ok := byDecl[p.Output]
		if !ok {
			fail(fmt.Errorf("phrasing pair output maps to no train spec: %q", p.Output))
		}
		target := decl.SpecText(spec)
		if len(decl.SplitTokens(target)) > decl.MaxTargetTokens {
			rep.SkippedTooLong++
			continue
		}
		if !decl.DecodeFeasible(target) {
			rep.SkippedInfeasible++
			continue
		}
		if prev, dup := seenInput[p.Input]; dup {
			if prev != target {
				rep.DuplicateInputTargets++
			}
			continue
		}
		seenInput[p.Input] = target
		pairs = append(pairs, decl.Pair{Input: p.Input, Output: target, Domain: "spec"})
	}
	rep.SpecPairsWritten = len(pairs)

	// --- Eval-side ceilings: gold spec text emittability per
	// phrasing cell, and token coverage of the probe's train pairs. ---
	trainTokens := map[string]bool{}
	for _, p := range pairs {
		for _, t := range decl.SplitTokens(p.Input) {
			trainTokens[t] = true
		}
		for _, t := range decl.SplitTokens(p.Output) {
			trainTokens[t] = true
		}
	}
	missingIn := map[string]bool{}
	missingGold := map[string]bool{}
	for _, item := range ds.EvalItems {
		if item.InputForm != "phrasing" {
			continue
		}
		cell := item.Combo + "|" + item.Frame
		rep.EvalGoldTotal[cell]++
		goldText := decl.SpecText(item.Spec)
		if decl.DecodeFeasible(goldText) && len(decl.SplitTokens(goldText)) <= decl.MaxTargetTokens {
			rep.EvalGoldEmittable[cell]++
		}
		for _, t := range decl.SplitTokens(item.Input) {
			if !trainTokens[t] {
				missingIn[t] = true
			}
		}
		for _, t := range decl.SplitTokens(goldText) {
			if !trainTokens[t] {
				missingGold[t] = true
			}
		}
	}
	rep.MissingInputTokens = sortedKeys(missingIn)
	rep.MissingGoldTokens = sortedKeys(missingGold)

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fail(err)
	}
	if err := writeJSONL(filepath.Join(*outDir, "spec_pairs.jsonl"), pairs); err != nil {
		fail(err)
	}
	raw, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(filepath.Join(*outDir, "spec_report.json"), raw, 0o644); err != nil {
		fail(err)
	}

	fmt.Printf("seed=%d train_specs=%d heldout_specs=%d\n", rep.Seed, rep.TrainSpecs, rep.HeldoutSpecs)
	fmt.Printf("regen fidelity: pairs match=%v heldout specs match=%v leaked=%d\n",
		rep.RegenPairsMatch, rep.ShippedHeldoutMatch, rep.LeakedHeldoutSpecs)
	fmt.Printf("round trip: strict=%d tag-content-only=%d presence=%d of %d\n",
		rep.RoundTripStrict, rep.RoundTripTagOnly, rep.RoundTripPresence, rep.RoundTripTotal)
	fmt.Printf("pairs: %d phrasing -> %d written (skipped infeasible=%d overlong=%d conflicting-dup-inputs=%d)\n",
		rep.DeclPhrasingPairs, rep.SpecPairsWritten, rep.SkippedInfeasible, rep.SkippedTooLong, rep.DuplicateInputTargets)
	for _, cell := range sortedKeys2(rep.EvalGoldTotal) {
		fmt.Printf("  eval cell phrasing|%s: gold emittable %d/%d\n", cell, rep.EvalGoldEmittable[cell], rep.EvalGoldTotal[cell])
	}
	fmt.Printf("missing tokens: input=%v gold=%v\n", rep.MissingInputTokens, rep.MissingGoldTokens)
}

// specKey is the full-spec identity, as in the decl generator.
func specKey(s decl.Spec) string {
	raw, _ := json.Marshal(s)
	return string(raw)
}

// presence reduces every field tag to a presence marker — the decl
// eval's convention for spec text, whose language carries no tag
// content (see decl.ParseSpecText).
func presence(s decl.Spec) decl.Spec {
	out := s
	out.Fields = append([]decl.Field(nil), s.Fields...)
	for i := range out.Fields {
		if out.Fields[i].Tag != "" {
			out.Fields[i].Tag = "tagged"
		}
	}
	return out
}

// tagsOnlyDiff reports whether two specs differ ONLY in field tag
// content.
func tagsOnlyDiff(a, b decl.Spec) bool {
	a2, b2 := a, b
	a2.Fields = append([]decl.Field(nil), a.Fields...)
	b2.Fields = append([]decl.Field(nil), b.Fields...)
	if len(a2.Fields) != len(b2.Fields) {
		return false
	}
	for i := range a2.Fields {
		a2.Fields[i].Tag, b2.Fields[i].Tag = "", ""
	}
	return reflect.DeepEqual(a2, b2)
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func sortedKeys2(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func loadJSONL[T any](path string) ([]T, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var rows []T
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var row T
		if err := json.Unmarshal([]byte(line), &row); err != nil {
			return nil, fmt.Errorf("%s: %w", path, err)
		}
		rows = append(rows, row)
	}
	return rows, nil
}

func writeJSONL[T any](path string, rows []T) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			return err
		}
	}
	return nil
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "gen_spec_data:", err)
	os.Exit(1)
}
