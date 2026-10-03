// gen_understanding_pairs distills the analyzer's code understanding
// into training pairs for the GoDomain (explainer) model: the model
// learns to answer "what does this do" from real code, the way the
// analyzer does. Run: go run ./scripts/gen_understanding_pairs
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

type pair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}

var repos = []string{
	"/home/hatch/workspace/gollemer/internal/ai/analyze",
	"/home/hatch/workspace/codebase-maps/pashkov256-deletor",
	"/home/hatch/workspace/codebase-maps/kavix-eko",
}

func main() {
	var out []pair
	seen := map[string]bool{}
	emit := func(in, op string) {
		if in == "" || op == "" || seen[in] {
			return
		}
		seen[in] = true
		out = append(out, pair{in, op})
	}
	for _, repo := range repos {
		p, err := analyze.Analyze(repo)
		if err != nil {
			fmt.Fprintf(os.Stderr, "skip %s: %v\n", repo, err)
			continue
		}
		emitRepo(p, emit)
	}
	// Deterministic order.
	sort.Slice(out, func(i, j int) bool { return out[i].Input < out[j].Input })
	f, err := os.Create("data/training/go_expansion_understanding.jsonl")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, pr := range out {
		if err := enc.Encode(pr); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	fmt.Printf("wrote %d pairs\n", len(out))
}

func emitRepo(p *analyze.Project, emit func(string, string)) {
	// Rank functions: documented, exported, or effectful first.
	type scored struct {
		fn    *analyze.Func
		score int
	}
	var ss []scored
	for _, fn := range p.AllFuncs() {
		if fn.IsTest || fn.Source() == "" {
			continue
		}
		intent := fn.Intent()
		if strings.HasPrefix(intent, "handles ") || intent == "" {
			continue // weak synthesis: skip
		}
		score := 0
		if fn.Exported {
			score += 2
		}
		if len(fn.Effects) > 0 {
			score += 2
		}
		if fn.Doc != "" {
			score += 1
		}
		ss = append(ss, scored{fn, score})
	}
	sort.Slice(ss, func(i, j int) bool {
		if ss[i].score != ss[j].score {
			return ss[i].score > ss[j].score
		}
		return ss[i].fn.Name < ss[j].fn.Name
	})
	n := 0
	for _, s := range ss {
		if n >= 80 {
			break
		}
		fn := s.fn
		intent := fn.Intent()
		// Short single-line pairs: the gate allows 40 tokens per side.
		emit(fmt.Sprintf("what does %s do", fn.Sig), intent)
		emit(fmt.Sprintf("what is %s for", fn.Sig), intent)
		n++
	}
	// Why-pairs: caller -> callee with a real answer.
	wn := 0
	for _, s := range ss {
		if wn >= 40 {
			break
		}
		fn := s.fn
		for _, id := range fn.Calls {
			callee := p.FuncByID(id)
			if callee == nil || callee.IsTest {
				continue
			}
			q := fmt.Sprintf("why does %s call %s", fn.Name, callee.Name)
			ans, ok := p.Answer(q)
			if !ok || strings.Contains(ans, "doesn't call") {
				continue
			}
			first := strings.SplitN(ans, "\n", 2)[0]
			emit(q, first)
			wn++
			break
		}
	}
}
