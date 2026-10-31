// train_phrasing_b trains arm B's family classifier for the
// phrasing layer (attempt 2): the intent layer's softmax over
// word n-gram + char-trigram features, with family labels
// instead of intents. Training examples are the v2 generator's
// fragments (data/training/phrasing_fragments2.jsonl): single
// asks as a glue split would leave them, plus each compound's
// two piece cores. Every example is normalized through
// chat.PhrasingFragmentText — the exact form the route
// classifies at serving time.
//
// Protocol: a stride carve (every 10th example) is held out
// from a first fit to REPORT the classifier's fragment
// accuracy honestly; the shipped artifact is then refit on
// all examples (the carve number is the honest one — it is
// printed and written into the report, never hidden by the
// refit). Output: data/phrasing_model.json (the intent model's
// JSON conventions) + data/training/phrasing_b_report.json.
//
// Run: go run ./scripts/train_phrasing_b
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/training/chat"
)

type fragRow struct {
	Input    string `json:"input"`
	Family   string `json:"family"`
	Template string `json:"template"`
}

type report struct {
	Examples      int            `json:"examples"`
	Families      int            `json:"families"`
	CarveSize     int            `json:"carve_size"`
	CarveAccuracy float64        `json:"carve_accuracy"`
	CarveGated    float64        `json:"carve_accuracy_gated"` // of the ones the gate accepts
	GateAccepted  float64        `json:"gate_accept_rate"`
	PerFamily     map[string]int `json:"per_family_examples"`
	Confusions    []string       `json:"top_confusions"`
}

func main() {
	fragPath := flag.String("fragments", "data/training/phrasing_fragments2.jsonl", "fragment examples")
	outPath := flag.String("out", "data/phrasing_model.json", "classifier artifact")
	repPath := flag.String("report", "data/training/phrasing_b_report.json", "training report")
	flag.Parse()

	rows, err := loadRows(*fragPath)
	if err != nil {
		fail(err)
	}
	var pairs []chat.IntentPair
	perFamily := map[string]int{}
	for _, r := range rows {
		pairs = append(pairs, chat.IntentPair{
			Input:  chat.PhrasingFragmentText(r.Input),
			Intent: r.Family,
		})
		perFamily[r.Family]++
	}
	if len(pairs) < 100 {
		fail(fmt.Errorf("need at least 100 fragment examples, have %d", len(pairs)))
	}

	// Stride carve for the honest number.
	var carve, rest []chat.IntentPair
	for i, p := range pairs {
		if i%10 == 0 {
			carve = append(carve, p)
		} else {
			rest = append(rest, p)
		}
	}
	probe := chat.TrainIntentModel(rest)
	correct, gatedCorrect, gated := 0, 0, 0
	confusions := map[string]int{}
	for _, p := range carve {
		got, _, _ := probe.Predict(p.Input)
		if got == p.Intent {
			correct++
		} else {
			confusions[p.Intent+" -> "+got]++
		}
		if fam, _, ok := probe.ClassifyIntent(p.Input); ok {
			gated++
			if fam == p.Intent {
				gatedCorrect++
			}
		}
	}
	rep := &report{
		Examples:      len(pairs),
		Families:      len(perFamily),
		CarveSize:     len(carve),
		CarveAccuracy: pct(correct, len(carve)),
		GateAccepted:  pct(gated, len(carve)),
		PerFamily:     perFamily,
	}
	if gated > 0 {
		rep.CarveGated = pct(gatedCorrect, gated)
	}
	type kv struct {
		k string
		v int
	}
	var cs []kv
	for k, v := range confusions {
		cs = append(cs, kv{k, v})
	}
	sort.Slice(cs, func(i, j int) bool { return cs[i].v > cs[j].v })
	for i := 0; i < len(cs) && i < 10; i++ {
		rep.Confusions = append(rep.Confusions, fmt.Sprintf("%s (%d)", cs[i].k, cs[i].v))
	}

	// Refit on everything and ship that artifact.
	model := chat.TrainIntentModel(pairs)
	if err := model.Save(*outPath); err != nil {
		fail(err)
	}
	raw, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(*repPath, raw, 0o644); err != nil {
		fail(err)
	}
	fmt.Printf("arm B classifier: %d examples, %d families\n", rep.Examples, rep.Families)
	fmt.Printf("carve (n=%d): raw accuracy %.2f%%, gate accepts %.1f%%, gated accuracy %.2f%%\n",
		rep.CarveSize, rep.CarveAccuracy, rep.GateAccepted, rep.CarveGated)
	fmt.Println("top confusions:")
	for _, c := range rep.Confusions {
		fmt.Println("  ", c)
	}
	fmt.Println("model written to", *outPath)
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

func loadRows(path string) ([]fragRow, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var rows []fragRow
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var r fragRow
		if err := json.Unmarshal([]byte(line), &r); err != nil {
			return nil, err
		}
		rows = append(rows, r)
	}
	return rows, sc.Err()
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "train_phrasing_b:", err)
	os.Exit(1)
}
