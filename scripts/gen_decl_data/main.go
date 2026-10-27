// gen_decl_data generates the decl experiment's datasets: synthetic
// (phrasing, structured spec, declaration) triples from the
// combinatorial pipeline in internal/ai/training/decl — schema
// registry, combinatorial spec generation, frame-based phrasing
// expansion (no external LLM), deterministic targets, the kind-aware
// filter, and the two-axis split (unseen combinations x unseen
// phrasing frames).
//
// Outputs (regenerable, intentionally NOT committed):
//
//	data/training/decl_pairs.jsonl    training pairs for -domain decl
//	data/training/decl_heldout.jsonl  held-out eval items (the matrix)
//	data/training/decl_report.json    sizes, filter telemetry, cuts
//
// Run: go run ./scripts/gen_decl_data [-seed N]
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/golangast/gollemer/internal/ai/training/decl"
)

type report struct {
	Seed         int64             `json:"seed"`
	TrainSpecs   int               `json:"train_specs"`
	HeldoutSpecs int               `json:"heldout_specs"`
	TrainPairs   int               `json:"train_pairs"`
	EvalItems    int               `json:"eval_items"`
	EvalCells    map[string]int    `json:"eval_cells"`
	Filter       *decl.FilterStats `json:"filter"`
	DesignCuts   []string          `json:"design_cuts"`
}

func main() {
	seed := flag.Int64("seed", 0, "generation seed (0 = the experiment default)")
	outDir := flag.String("out", "data/training", "output directory")
	flag.Parse()

	cfg := decl.DefaultGenConfig()
	if *seed != 0 {
		cfg.Seed = *seed
	}
	ds, err := decl.Generate(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "gen_decl_data:", err)
		os.Exit(1)
	}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fail(err)
	}
	writeJSONL(filepath.Join(*outDir, "decl_pairs.jsonl"), ds.TrainPairs)
	writeJSONL(filepath.Join(*outDir, "decl_heldout.jsonl"), ds.EvalItems)

	cells := map[string]int{}
	for _, item := range ds.EvalItems {
		cells[item.InputForm+"|"+item.Combo+"|"+item.Frame]++
	}
	rep := report{
		Seed:         cfg.Seed,
		TrainSpecs:   len(ds.TrainSpecs),
		HeldoutSpecs: len(ds.HeldoutSpecs),
		TrainPairs:   len(ds.TrainPairs),
		EvalItems:    len(ds.EvalItems),
		EvalCells:    cells,
		Filter:       ds.Stats,
		DesignCuts: []string{
			"named results are out of v1: results are unnamed type lists (the error-tuple relation is derived from the last result)",
			"method values / embedded methods promoted from embedded types are not modeled; embedding is represented for struct fields and interface embeds",
			"struct tags: json tags only, name = lowercase field name, optional ,omitempty",
			"generic types appear only as the two pool locals (List, Pair) instantiated; no generic declarations are generated",
		},
	}
	raw, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(filepath.Join(*outDir, "decl_report.json"), raw, 0o644); err != nil {
		fail(err)
	}

	keys := make([]string, 0, len(cells))
	for k := range cells {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Printf("seed=%d train_specs=%d heldout_specs=%d train_pairs=%d eval_items=%d\n",
		cfg.Seed, rep.TrainSpecs, rep.HeldoutSpecs, rep.TrainPairs, rep.EvalItems)
	for _, k := range keys {
		fmt.Printf("  cell %-28s %d\n", k, cells[k])
	}
	fmt.Printf("filter: total=%d valid=%d rejections=%v\n",
		ds.Stats.Total, ds.Stats.Valid, ds.Stats.Rejections)
}

func writeJSONL[T any](path string, rows []T) {
	f, err := os.Create(path)
	if err != nil {
		fail(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, r := range rows {
		if err := enc.Encode(r); err != nil {
			fail(err)
		}
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "gen_decl_data:", err)
	os.Exit(1)
}
