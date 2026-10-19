// Command gendistill generates distillation training data for the
// code-intent model: the analyzer (teacher) answers questions about real
// code, and each (code + question -> answer) triple becomes a training
// pair. The model learns to do what the analyzer does, from data.
//
// teacher-v2 (2026-10-12): the diagnostic over the teacher-fix training
// set showed 61.1% of answers contained project facts NOT derivable from
// the input snippet (fn.Detail's "Calls (N):"/"Called by (N):" graph
// lines, how-works step-by-step callee summaries, disambiguation lists),
// and 87.0% on the held-out set — teaching the model to recite memorized
// corpus facts instead of reading the [code] block. This version:
//
//   - forms questions with fn.Display() dotted names (pkg.Name or
//     pkg.Recv.Name), which resolve() matches exactly — no more
//     "which one did you mean" disambiguation pairs (verified 2026-10-07);
//   - restricts answers to facts derivable from the snippet: the
//     function's signature, its location, its one-line purpose, and the
//     names of callees as they appear in the snippet. NO whole-project
//     caller/callee lists. NO disambiguation lists.
//
// why-calls keeps the analyzer's answer: the callee's sig+doc travel in
// the input, and the diagnostic found those answers 99.8% derivable.
//
// Usage: gendistill -out data/training/codeintent_pairs.jsonl <repo...>
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
)

type distillPair struct {
	Input  string `json:"input"`
	Output string `json:"output"`
	Domain string `json:"domain"`
}

func main() {
	out := flag.String("out", "", "output JSONL path (required)")
	flag.Parse()
	if *out == "" || flag.NArg() == 0 {
		log.Fatal("usage: gendistill -out <pairs.jsonl> <repo...>")
	}

	var pairs []distillPair
	seen := map[string]bool{}
	emit := func(input, output string) {
		input = strings.TrimSpace(input)
		output = strings.TrimSpace(output)
		if input == "" || output == "" {
			return
		}
		// Safety net: dotted names should make disambiguation
		// impossible, but never teach a "which one did you mean" answer.
		if strings.Contains(output, "which one did you mean") {
			return
		}
		key := input + "\n" + output
		if seen[key] {
			return
		}
		seen[key] = true
		pairs = append(pairs, distillPair{Input: input, Output: output, Domain: "codeintent"})
	}

	for _, repo := range flag.Args() {
		p, err := analyze.Analyze(repo)
		if err != nil {
			log.Printf("skip %s: %v", repo, err)
			continue
		}
		n := 0
		for _, fn := range p.AllFuncs() {
			if fn.IsTest || len(fn.Src) < 80 || len(fn.Src) > 4000 {
				continue
			}
			if fn.Name == "" || fn.Display() == "" {
				continue
			}
			name := fn.Display() // dotted: exact resolution, no disambiguation
			ctx := fmt.Sprintf("[file %s:%d]\n[code]\n%s\n", fn.File, fn.Line, fn.Src)

			// what-does / what-for: the snippet-derivable card — signature,
			// location, purpose. No project-wide call-graph lines.
			card := teachCard(fn)
			emit(ctx+"[question] what does "+name+" do", card)
			emit(ctx+"[question] what is "+name+" for", card)
			n += 2
			// how-works: the card plus the callees named in the snippet.
			// No corpus summaries of those callees.
			emit(ctx+"[question] how does "+name+" work", card+snippetCalls(p, fn))
			n++
			// Grounding-forcing pairs (2026-10-15): questions whose answers
			// are ONLY in the input block, forcing the model to look at
			// the code instead of template-matching. The "which file"
			// answer is a verbatim copy target for the copy gate; the
			// signature must be read from the [code] block.
			emit(ctx+"[question] which file is "+name+" in", fmt.Sprintf("%s:%d", fn.File, fn.Line))
			n++
			emit(ctx+"[question] what is the signature of "+name, fn.Sig)
			n++
			// why-calls: the callee's signature and doc travel in the
			// input, so the analyzer's answer is snippet-derivable.
			for i, calleeID := range fn.Calls {
				if i >= 3 {
					break
				}
				callee := p.FuncByID(calleeID)
				if callee == nil || callee.Name == "" || callee.Display() == "" {
					continue
				}
				q := "why does " + name + " call " + callee.Display()
				if ans, ok := p.Answer(q); ok {
					calleeCtx := fmt.Sprintf("[callee %s]\n%s\n", callee.Sig, callee.Doc)
					emit(ctx+calleeCtx+"[question] "+q, ans)
					n++
				}
			}
		}
		log.Printf("%s: %d pairs", repo, n)
	}

	f, err := os.Create(*out)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, pr := range pairs {
		if err := enc.Encode(pr); err != nil {
			log.Fatal(err)
		}
	}
	log.Printf("wrote %d pairs to %s", len(pairs), *out)
}

// teachCard is the snippet-derivable answer card: signature, location,
// and the one-line purpose (doc first sentence when present, otherwise
// derived from the name). Deliberately no "Calls (N):"/"Called by (N):"
// project-graph lines — those facts are not in the [code] block.
func teachCard(fn *analyze.Func) string {
	return fmt.Sprintf("%s\n  %s:%d\n  %s\n", fn.Sig, fn.File, fn.Line, fn.Summary())
}

// snippetCalls lists the callees by the names they are called with in the
// snippet. A callee is listed only if its bare name appears in the
// function's source, so every named call is visible in the [code] block.
func snippetCalls(p *analyze.Project, fn *analyze.Func) string {
	var names []string
	seen := map[string]bool{}
	for _, id := range fn.Calls {
		callee := p.FuncByID(id)
		if callee == nil || callee.Name == "" || seen[callee.Name] {
			continue
		}
		seen[callee.Name] = true
		if !strings.Contains(fn.Src, callee.Name) {
			continue
		}
		names = append(names, callee.Name)
		if len(names) >= 8 {
			break
		}
	}
	if len(names) == 0 {
		return "Calls in this code: none.\n"
	}
	return "Calls in this code: " + strings.Join(names, ", ") + ".\n"
}
