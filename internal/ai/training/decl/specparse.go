package decl

import (
	"fmt"
	"strings"
)

// ParseSpecText is the inverse of SpecText: it parses the model's
// structured-input language ("spec struct user field id int tagged
// ...") back into a Spec. It exists so that consumers of the spec
// form — the round-trip verifier in the eval harness above all — can
// recover the reference spec from the INPUT STRING ALONE, never from
// a dataset's stored gold spec.
//
// The slot grammar is unambiguous by construction: slot keywords
// (field, embed, method, param, result, receiver) never appear
// inside a type expression, and no pool name collides with one. A
// type span runs until the next slot keyword; a param span is a name
// plus a type span, or a bare type span — the two readings are
// distinguished by whether the tail parses as a type (two adjacent
// atoms never form a type, and a leading map/func/chan/[ /* token
// cannot be a name).
//
// One slot is NOT recoverable: a field's tag content. SpecText
// writes only the marker "tagged", so the parsed spec carries the
// conventional base form json:"<fieldname>"; the ",omitempty"
// variant is not representable in the spec language. Callers that
// compare a parsed spec against model output must therefore compare
// tag PRESENCE, not content (see eval_decl's round-trip check).
func ParseSpecText(input string) (Spec, error) {
	toks := SplitTokens(input)
	if len(toks) < 3 || toks[0] != "spec" {
		return Spec{}, fmt.Errorf("spec text: want \"spec <kind> <name> ...\", got %q", input)
	}
	kind := Kind(toks[1])
	switch kind {
	case KindStruct, KindInterface, KindFunc:
	default:
		return Spec{}, fmt.Errorf("spec text: unknown kind %q", toks[1])
	}
	spec := Spec{Kind: kind, Name: toks[2]}
	i := 3
	for i < len(toks) {
		switch toks[i] {
		case "field":
			if kind != KindStruct {
				return Spec{}, fmt.Errorf("spec text: field slot in %s spec", kind)
			}
			f, next, err := parseFieldSlot(toks, i)
			if err != nil {
				return Spec{}, err
			}
			spec.Fields = append(spec.Fields, f)
			i = next
		case "embed":
			span, next := slotSpan(toks, i+1)
			ref, err := spanType(span)
			if err != nil {
				return Spec{}, fmt.Errorf("spec text: embed: %w", err)
			}
			switch kind {
			case KindStruct:
				spec.Fields = append(spec.Fields, Field{Embedded: true, Type: ref})
			case KindInterface:
				spec.Embeds = append(spec.Embeds, ref)
			default:
				return Spec{}, fmt.Errorf("spec text: embed slot in %s spec", kind)
			}
			i = next
		case "method":
			if kind != KindInterface {
				return Spec{}, fmt.Errorf("spec text: method slot in %s spec", kind)
			}
			m, next, err := parseMethodSlot(toks, i)
			if err != nil {
				return Spec{}, err
			}
			spec.Methods = append(spec.Methods, m)
			i = next
		case "receiver":
			if kind != KindFunc {
				return Spec{}, fmt.Errorf("spec text: receiver slot in %s spec", kind)
			}
			r, next, err := parseReceiverSlot(toks, i)
			if err != nil {
				return Spec{}, err
			}
			spec.Receiver = r
			i = next
		case "param":
			if kind != KindFunc {
				return Spec{}, fmt.Errorf("spec text: param slot in %s spec (methods nest theirs)", kind)
			}
			p, next, err := parseParamSlot(toks, i)
			if err != nil {
				return Spec{}, err
			}
			spec.Params = append(spec.Params, p)
			i = next
		case "result":
			if kind != KindFunc {
				return Spec{}, fmt.Errorf("spec text: result slot in %s spec (methods nest theirs)", kind)
			}
			span, next := slotSpan(toks, i+1)
			ref, err := spanType(span)
			if err != nil {
				return Spec{}, fmt.Errorf("spec text: result: %w", err)
			}
			spec.Results = append(spec.Results, ref)
			i = next
		default:
			return Spec{}, fmt.Errorf("spec text: unexpected token %q at %d", toks[i], i)
		}
	}
	return Canonicalize(spec)
}

// slotKeywords terminate a type span.
var slotKeywords = map[string]bool{
	"field": true, "embed": true, "method": true,
	"param": true, "result": true, "receiver": true,
}

// slotSpan returns the tokens of a type span starting at i (everything
// up to the next slot keyword or end of input) and the index just
// past the span.
func slotSpan(toks []string, i int) ([]string, int) {
	j := i
	for j < len(toks) && !slotKeywords[toks[j]] {
		j++
	}
	return toks[i:j], j
}

// spanType reassembles a type span into an expression and validates
// it through the registry's canonicalizer.
func spanType(span []string) (TypeRef, error) {
	if len(span) == 0 {
		return TypeRef{}, fmt.Errorf("empty type span")
	}
	return MakeTypeRef(strings.Join(span, " "))
}

// parseFieldSlot parses "field <name> <type...> [tagged]" at i.
func parseFieldSlot(toks []string, i int) (Field, int, error) {
	if i+1 >= len(toks) {
		return Field{}, 0, fmt.Errorf("spec text: field slot without a name")
	}
	f := Field{Name: toks[i+1]}
	span, next := slotSpan(toks, i+2)
	tagged := false
	if n := len(span); n > 0 && span[n-1] == "tagged" {
		tagged = true
		span = span[:n-1]
	}
	ref, err := spanType(span)
	if err != nil {
		return Field{}, 0, fmt.Errorf("spec text: field %s: %w", f.Name, err)
	}
	f.Type = ref
	if tagged {
		// Only presence survives SpecText; carry the conventional
		// base form (see ParseSpecText's comment).
		f.Tag = `json:"` + f.Name + `"`
	}
	return f, next, nil
}

// parseMethodSlot parses "method <name> (param...|result...)*" at i.
func parseMethodSlot(toks []string, i int) (Method, int, error) {
	if i+1 >= len(toks) {
		return Method{}, 0, fmt.Errorf("spec text: method slot without a name")
	}
	m := Method{Name: toks[i+1]}
	j := i + 2
	for j < len(toks) && (toks[j] == "param" || toks[j] == "result") {
		if toks[j] == "param" {
			p, next, err := parseParamSlot(toks, j)
			if err != nil {
				return Method{}, 0, err
			}
			m.Params = append(m.Params, p)
			j = next
		} else {
			span, next := slotSpan(toks, j+1)
			ref, err := spanType(span)
			if err != nil {
				return Method{}, 0, fmt.Errorf("spec text: method %s result: %w", m.Name, err)
			}
			m.Results = append(m.Results, ref)
			j = next
		}
	}
	return m, j, nil
}

// parseParamSlot parses "param [<name>] <type...>" at i.
func parseParamSlot(toks []string, i int) (Param, int, error) {
	span, next := slotSpan(toks, i+1)
	if len(span) == 0 {
		return Param{}, 0, fmt.Errorf("spec text: param slot without a type")
	}
	// A leading type keyword or bracket can only be a type, and a
	// one-token span can only be a bare type.
	if len(span) == 1 || isTypeLead(span[0]) {
		ref, err := spanType(span)
		if err != nil {
			return Param{}, 0, fmt.Errorf("spec text: param: %w", err)
		}
		return Param{Type: ref}, next, nil
	}
	// Named reading: first token is the name, the rest must parse as
	// a type. (A bare two-atom span is never a type, so when the
	// named reading parses it is the only possible one.)
	if ref, err := spanType(span[1:]); err == nil {
		return Param{Name: span[0], Type: ref}, next, nil
	}
	ref, err := spanType(span)
	if err != nil {
		return Param{}, 0, fmt.Errorf("spec text: param %q: %w", strings.Join(span, " "), err)
	}
	return Param{Type: ref}, next, nil
}

// isTypeLead reports whether a token can only begin a type
// expression, never a parameter name.
func isTypeLead(tok string) bool {
	switch tok {
	case "map", "func", "chan", "[", "*", "(", ")", ",":
		return true
	}
	return false
}

// parseReceiverSlot parses "receiver <name> <type> value|pointer" at i.
func parseReceiverSlot(toks []string, i int) (*Receiver, int, error) {
	if i+3 >= len(toks) {
		return nil, 0, fmt.Errorf("spec text: receiver slot truncated")
	}
	r := &Receiver{Name: toks[i+1], Type: toks[i+2]}
	switch toks[i+3] {
	case "value":
	case "pointer":
		r.Pointer = true
	default:
		return nil, 0, fmt.Errorf("spec text: receiver kind %q, want value|pointer", toks[i+3])
	}
	return r, i + 4, nil
}
