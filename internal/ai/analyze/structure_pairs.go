// Package analyze — structure-to-language training pairs.
//
// This generates training data for a model that learns HOW THE PROGRAM
// KNOWS CODE — not character-by-character prediction like a transformer.
//
// The pipeline:
//   Go code → [Analyzer] → Structural facts → [Templates] → "Correct but rigid" explanation
//                                                        ↓
//                                              Training pair: (facts → explanation)
//                                                        ↓
//                                              [Small model] learns facts → fluent words
//
// The model never sees raw code as characters to predict. It sees the
// analyzer's structural understanding and learns to express it naturally.
// Structure is given; the model only learns the mapping to language.
package analyze

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// StructurePair is a single training example: structural facts → explanation.
type StructurePair struct {
	// Input: the analyzer's structural understanding, as text.
	// Not raw code — the FACTS the analyzer extracted.
	Input string `json:"input"`
	// Output: natural-language explanation of those facts.
	Output string `json:"output"`
	// Kind: what type of understanding this teaches.
	Kind string `json:"kind"` // "func_explain", "type_explain", "where_plan", "dataflow"
}

// GenerateStructurePairs walks a project and creates training pairs from
// the analyzer's own understanding. The "ground truth" explanations come
// from the current template system — correct but rigid. The model learns
// to produce the same understanding more fluently.
func GenerateStructurePairs(p *Project) []StructurePair {
	var pairs []StructurePair

	for _, pkg := range p.Packages {
		for _, fn := range pkg.Funcs {
			if fn.IsTest {
				continue
			}
			// Pair type 1: function structure → explanation.
			if pair := funcStructurePair(fn); pair != nil {
				pairs = append(pairs, *pair)
			}
		}
		for _, typ := range pkg.Types {
			// Pair type 2: type structure → explanation.
			if pair := typeStructurePair(typ, pkg); pair != nil {
				pairs = append(pairs, *pair)
			}
		}
	}

	return pairs
}

// funcStructurePair creates: (structural facts about a func) → (what it does).
func funcStructurePair(fn *Func) *StructurePair {
	if fn.Name == "" || fn.Sig == "" {
		return nil
	}

	// Build the structural facts — what the analyzer KNOWS, not the code text.
	// Keep it SHORT: the quality gate allows max 40 tokens per side.
	var facts strings.Builder
	// Just: name + key behaviors. Skip the full signature.
	facts.WriteString(fn.Name)
	if fn.Receiver != "" {
		facts.WriteString(" on " + strings.Trim(fn.Receiver, "*"))
	}

	// Control flow facts — abbreviated.
	src := strings.ToLower(fn.Src)
	var behaviors []string
	if strings.Contains(src, "rlock()") {
		behaviors = append(behaviors, "read-locks")
	} else if strings.Contains(src, ".lock()") {
		behaviors = append(behaviors, "locks")
	}
	if strings.Contains(src, "range") {
		behaviors = append(behaviors, "iterates")
	}
	if strings.Contains(fn.Sig, "error") {
		behaviors = append(behaviors, "may-fail")
	}
	if len(behaviors) > 0 {
		facts.WriteString(": " + strings.Join(behaviors, ","))
	}

	// The output: natural explanation from the template system.
	output := explainFuncFromFacts(fn, behaviors)

	return &StructurePair{
		Input:  facts.String(),
		Output: output,
		Kind:   "func_explain",
	}
}

// typeStructurePair creates: (structural facts about a type) → (what it's for).
func typeStructurePair(typ *Type, pkg *Package) *StructurePair {
	if typ.Name == "" {
		return nil
	}

	// Keep it SHORT for the quality gate.
	var facts strings.Builder
	facts.WriteString(typ.Name)

	// Use the type's own method list if available, else scan.
	methods := typ.Methods
	if len(methods) == 0 {
		for _, fn := range pkg.Funcs {
			if strings.Contains(fn.Receiver, typ.Name) {
				methods = append(methods, fn.Name)
			}
		}
	}
	if len(methods) > 0 {
		// Limit to first 5 methods to stay under token limit.
		if len(methods) > 5 {
			methods = methods[:5]
		}
		facts.WriteString(": " + strings.Join(methods, ","))
	}

	output := explainTypeFromFacts(typ, methods)

	return &StructurePair{
		Input:  facts.String(),
		Output: output,
		Kind:   "type_explain",
	}
}

// explainFuncFromFacts generates the "correct but rigid" explanation.
// The model will learn to produce this understanding more naturally.
func explainFuncFromFacts(fn *Func, behaviors []string) string {
	var b strings.Builder
	// Verb from name.
	verb := simpleVerb(fn.Name)
	b.WriteString(fmt.Sprintf("%s %s", fn.Name, verb))
	if fn.Receiver != "" {
		b.WriteString(fmt.Sprintf(" on the %s", strings.Trim(fn.Receiver, "*")))
	}
	b.WriteString(".")
	if len(behaviors) > 0 {
		b.WriteString(fmt.Sprintf(" It %s.", strings.Join(behaviors, ", it ")))
	}
	return b.String()
}

// explainTypeFromFacts generates the "correct but rigid" type explanation.
func explainTypeFromFacts(typ *Type, methods []string) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("%s is a struct", typ.Name))
	if len(methods) > 0 {
		b.WriteString(fmt.Sprintf(" with %d methods (%s)", len(methods), strings.Join(methods, ", ")))
	}
	b.WriteString(".")
	return b.String()
}

// simpleVerb extracts a verb phrase from a function name.
func simpleVerb(name string) string {
	lower := strings.ToLower(name)
	if strings.HasPrefix(lower, "get") && len(name) > 3 {
		return "returns the " + strings.ToLower(name[3:])
	}
	if strings.HasPrefix(lower, "set") && len(name) > 3 {
		return "sets the " + strings.ToLower(name[3:])
	}
	if strings.HasSuffix(lower, "s") && !strings.HasSuffix(lower, "ss") {
		return "returns the " + lower
	}
	// Simple conjugation.
	if strings.HasSuffix(lower, "e") {
		return lower + "s"
	}
	return lower + "s"
}

// ExportStructurePairs writes pairs to a JSONL file for training.
func ExportStructurePairs(pairs []StructurePair, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, p := range pairs {
		// Format for seq2seq: {"input": ..., "output": ...}
		data, err := json.Marshal(map[string]string{
			"input":  p.Input,
			"output": p.Output,
		})
		if err != nil {
			continue
		}
		f.Write(append(data, '\n'))
	}
	return nil
}
