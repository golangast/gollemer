package main

import (
	"fmt"
	"go/token"

	"github.com/golangast/gollemer/internal/ai/training/decl"
)

// The no-cheating contract, in code: a constraint is built from the
// item's INPUT STRING ONLY. buildConstraint never sees the eval
// item, its gold spec, or its target. Everything the decoder may use
// is derived mechanically from that string:
//
//   - KIND: from the spec header ("spec struct ...") or from the
//     kind word every phrasing frame carries ("struct", "interface",
//     "function", "method"; bare signature frames start with "func";
//     the terse arrow frame F-T2 is recognized by its "->"). If no
//     rule fires, derivation fails and the item is decoded
//     unconstrained and counted — never peeked at.
//   - NAMES (declaration, field, method, parameter, receiver names):
//     identifier-like tokens appearing in the input text that are
//     legal Go identifiers (token.IsIdentifier, so sentence words
//     like "struct" or "func" are excluded).
//   - TYPE ATOMS: identifier-like tokens of the input (dotted forms
//     like "http.handler" included) PLUS Go's builtin type atoms.
//     Builtins are constants of the language — identical for every
//     item — and phrasing frames routinely describe them verbally
//     ("an integer" for int), so they cannot be required to appear.
//     Two further structural constants of the pipeline's tag grammar
//     are admitted by the automaton, not by this set: the tag key
//     `json:"..."` and the ",omitempty" suffix (see automaton.go).
//     One identifier position is likewise widened by the automaton:
//     a method's receiver name admits any single-letter atom,
//     because the phrasing frames name only the receiver TYPE (the
//     letter itself is not in the sentence and is not scored there).
type constraint struct {
	kind      decl.Kind
	names     map[string]bool
	typeAtoms map[string]bool
}

// builtinTypeAtoms are Go's predeclared type identifiers — the same
// closed set the decl registry uses (spec.go's builtinTypes).
var builtinTypeAtoms = []string{
	"string", "int", "int64", "int32", "float64", "float32", "bool",
	"byte", "rune", "uint", "uint64", "uintptr", "any", "error",
	"complex128",
}

// deriveKind extracts the declaration kind from input text alone.
func deriveKind(input string) (decl.Kind, bool) {
	toks := decl.SplitTokens(input)
	if len(toks) >= 2 && toks[0] == "spec" {
		switch toks[1] {
		case "struct":
			return decl.KindStruct, true
		case "interface":
			return decl.KindInterface, true
		case "func":
			return decl.KindFunc, true
		}
		return "", false
	}
	has := map[string]bool{}
	for _, t := range toks {
		has[t] = true
	}
	switch {
	case has["struct"]:
		return decl.KindStruct, true
	case has["interface"]:
		return decl.KindInterface, true
	case has["function"] || has["method"]:
		return decl.KindFunc, true
	case len(toks) > 0 && toks[0] == "func":
		return decl.KindFunc, true
	case has["->"]:
		return decl.KindFunc, true
	}
	return "", false
}

// isSpecForm reports whether the input is in the structured spec
// language (and therefore carries its reference spec in-band, for
// the round-trip check). Derived from the text, like the kind.
func isSpecForm(input string) bool {
	toks := decl.SplitTokens(input)
	return len(toks) > 0 && toks[0] == "spec"
}

// identLike reports whether a token is an identifier or a dotted
// identifier path ("http.handler"), in the model's lowercase world.
func identLike(t string) bool {
	for _, part := range splitDot(t) {
		if part == "" {
			return false
		}
		for j, r := range part {
			switch {
			case r == '_' || (r >= 'a' && r <= 'z'):
			case j > 0 && r >= '0' && r <= '9':
			default:
				return false
			}
		}
	}
	return len(t) > 0
}

func splitDot(t string) []string {
	var out []string
	start := 0
	for i := 0; i < len(t); i++ {
		if t[i] == '.' {
			out = append(out, t[start:i])
			start = i + 1
		}
	}
	return append(out, t[start:])
}

// buildConstraint derives the decoding constraint from the input
// string alone. An error means the kind was not derivable; the
// caller falls back to unconstrained decoding for that item and
// counts it.
func buildConstraint(input string) (*constraint, error) {
	kind, ok := deriveKind(input)
	if !ok {
		return nil, fmt.Errorf("kind not derivable from input text")
	}
	c := &constraint{
		kind:      kind,
		names:     map[string]bool{},
		typeAtoms: map[string]bool{},
	}
	for _, t := range decl.SplitTokens(input) {
		if !identLike(t) {
			continue
		}
		c.typeAtoms[t] = true
		if token.IsIdentifier(t) {
			c.names[t] = true
		}
	}
	for _, b := range builtinTypeAtoms {
		c.typeAtoms[b] = true
	}
	return c, nil
}
