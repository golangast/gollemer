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
	var facts strings.Builder
	facts.WriteString(fmt.Sprintf("function %s", fn.Name))
	if fn.Receiver != "" {
		facts.WriteString(fmt.Sprintf(" on %s", strings.Trim(fn.Receiver, "*")))
	}
	facts.WriteString(fmt.Sprintf(", signature %s", fn.Sig))

	// Control flow facts.
	src := strings.ToLower(fn.Src)
	var behaviors []string
	if strings.Contains(src, ".lock()") || strings.Contains(src, ".rlock()") {
		if strings.Contains(src, ".rlock()") {
			behaviors = append(behaviors, "locks for reading")
		} else {
			behaviors = append(behaviors, "locks for writing")
		}
	}
	if strings.Contains(src, "for ") && strings.Contains(src, "range") {
		behaviors = append(behaviors, "iterates over data")
	}
	if strings.Contains(src, "if err") {
		behaviors = append(behaviors, "handles errors")
	}
	if strings.Contains(fn.Sig, "error") {
		behaviors = append(behaviors, "returns error")
	}
	if len(behaviors) > 0 {
		facts.WriteString(fmt.Sprintf(", behaviors: %s", strings.Join(behaviors, ", ")))
	}

	// Doc comment if present — that's the human's own explanation.
	if fn.Doc != "" {
		first := firstSentence(fn.Doc)
		facts.WriteString(fmt.Sprintf(", documented as: %s", first))
	}

	// The output: natural explanation from the template system.
	// This is "correct but rigid" — the model learns to be fluent.
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

	var facts strings.Builder
	facts.WriteString(fmt.Sprintf("type %s struct", typ.Name))
	if len(typ.Fields) > 0 {
		var fieldNames []string
		for _, fname := range typ.Fields {
			// Skip embedded mutexes for the facts.
			if strings.Contains(fname, "Mutex") {
				continue
			}
			fieldNames = append(fieldNames, fname)
		}
		if len(fieldNames) > 0 {
			facts.WriteString(fmt.Sprintf(" with fields %s", strings.Join(fieldNames, ", ")))
		}
	}

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
		facts.WriteString(fmt.Sprintf(", methods: %s", strings.Join(methods, ", ")))
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
