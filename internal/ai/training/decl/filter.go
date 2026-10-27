package decl

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"reflect"
	"sync"
)

// The filter is the pipeline's gatekeeper: every generated triple
// passes through it before it may enter a dataset. Three tiers:
//
//  1. Hygiene: per-kind identifier validity (token.IsIdentifier) and
//     duplicate detection (fields, methods, named params — a
//     duplicated named parameter is a real Go compile error).
//  2. Type validity: the expression must parse AND its AST nodes must
//     be type-expression nodes — bare parser.ParseExpr accepts
//     non-type expressions like "a + b" and must not pass them.
//  3. Round-trip: spec -> rendered code -> parse -> extract spec ->
//     structural equality against the canonicalized spec. A
//     parse-success alone cannot catch a template that silently drops
//     a field; equality can.
//
// Rejections are tallied by reason with one stored sample (the spec's
// JSON plus the underlying error) per reason — never a print per spec.
// FilterStats is safe for parallel generation.

// Rejection reasons (stable strings; they appear verbatim in dataset
// reports).
const (
	ReasonInvalidSpecName   = "invalid_spec_name"
	ReasonUnknownKind       = "unknown_kind"
	ReasonInvalidMemberName = "invalid_member_name"
	ReasonDuplicateMember   = "duplicate_member"
	ReasonInvalidTypeExpr   = "invalid_type_expression"
	ReasonNonTypeExpr       = "non_type_expression"
	ReasonRenderFailed      = "render_failed"
	ReasonRoundTripParse    = "roundtrip_parse_failed"
	ReasonRoundTripMismatch = "roundtrip_structural_mismatch"
	ReasonDecodeInfeasible  = "decode_infeasible"
	ReasonOversize          = "oversize"
)

// FilterStats aggregates validation outcomes.
type FilterStats struct {
	mu         sync.Mutex
	Total      int               `json:"total"`
	Valid      int               `json:"valid"`
	Rejections map[string]int    `json:"rejections"`
	Samples    map[string]string `json:"samples"` // reason -> "specJSON :: error"
}

// NewFilterStats returns an empty stats tracker.
func NewFilterStats() *FilterStats {
	return &FilterStats{
		Rejections: map[string]int{},
		Samples:    map[string]string{},
	}
}

func (s *FilterStats) record(reason string, spec Spec, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Rejections[reason]++
	if _, exists := s.Samples[reason]; !exists {
		raw, _ := json.Marshal(spec)
		msg := string(raw)
		if err != nil {
			msg += " :: " + err.Error()
		}
		s.Samples[reason] = msg
	}
}

func (s *FilterStats) recordValid() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.Valid++
}

// Snapshot returns a copy safe to read after generation.
func (s *FilterStats) Snapshot() *FilterStats {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := &FilterStats{
		Total:      s.Total,
		Valid:      s.Valid,
		Rejections: make(map[string]int, len(s.Rejections)),
		Samples:    make(map[string]string, len(s.Samples)),
	}
	for k, v := range s.Rejections {
		out.Rejections[k] = v
	}
	for k, v := range s.Samples {
		out.Samples[k] = v
	}
	return out
}

// Validate runs the full filter over one spec. It reports whether the
// spec (and its rendered declaration) may enter a dataset, recording
// the outcome in stats.
func Validate(spec Spec, stats *FilterStats) bool {
	stats.mu.Lock()
	stats.Total++
	stats.mu.Unlock()

	canon, err := Canonicalize(spec)
	if err != nil {
		stats.record(ReasonInvalidTypeExpr, spec, err)
		return false
	}

	// Tier 1: hygiene.
	if !token.IsIdentifier(canon.Name) {
		stats.record(ReasonInvalidSpecName, spec, nil)
		return false
	}
	switch canon.Kind {
	case KindStruct, KindInterface, KindFunc:
	default:
		stats.record(ReasonUnknownKind, spec, nil)
		return false
	}
	if reason, bad := checkMembers(canon); bad {
		stats.record(reason, spec, nil)
		return false
	}

	// Tier 2: every type expression must parse as a TYPE.
	for _, ref := range allTypeRefs(canon) {
		if reason, bad := checkTypeExpr(ref.Expr); bad {
			stats.record(reason, spec, nil)
			return false
		}
	}

	// Tier 3: round-trip through the renderer and the AST.
	code := RenderDecl(canon)
	if code == "" {
		stats.record(ReasonRenderFailed, spec, nil)
		return false
	}
	lowerCode := RenderDecl(LowerSpec(canon))
	if !DecodeFeasible(lowerCode) {
		stats.record(ReasonDecodeInfeasible, spec, nil)
		return false
	}
	if len(SplitTokens(code)) > MaxTargetTokens {
		stats.record(ReasonOversize, spec, nil)
		return false
	}
	back, err := ExtractSpec(code)
	if err != nil {
		stats.record(ReasonRoundTripParse, spec, err)
		return false
	}
	if !reflect.DeepEqual(canon, back) {
		stats.record(ReasonRoundTripMismatch, spec, nil)
		return false
	}
	stats.recordValid()
	return true
}

// checkMembers enforces per-kind identifier and duplicate rules. It
// returns the rejection reason and whether the spec is bad.
func checkMembers(s Spec) (string, bool) {
	ident := func(name string) bool { return token.IsIdentifier(name) }
	switch s.Kind {
	case KindStruct:
		seen := map[string]bool{}
		for _, f := range s.Fields {
			if f.Embedded {
				key := BaseName(f.Type)
				if seen[key] {
					return ReasonDuplicateMember, true
				}
				seen[key] = true
				continue
			}
			if !ident(f.Name) {
				return ReasonInvalidMemberName, true
			}
			if seen[f.Name] {
				return ReasonDuplicateMember, true
			}
			seen[f.Name] = true
		}
	case KindInterface:
		seen := map[string]bool{}
		for _, e := range s.Embeds {
			key := BaseName(e)
			if seen[key] {
				return ReasonDuplicateMember, true
			}
			seen[key] = true
		}
		for _, m := range s.Methods {
			if !ident(m.Name) {
				return ReasonInvalidMemberName, true
			}
			if seen[m.Name] {
				return ReasonDuplicateMember, true
			}
			seen[m.Name] = true
			if reason, bad := checkParams(m.Params); bad {
				return reason, bad
			}
		}
	case KindFunc:
		if s.Receiver != nil && s.Receiver.Name != "" && !ident(s.Receiver.Name) {
			return ReasonInvalidMemberName, true
		}
		if reason, bad := checkParams(s.Params); bad {
			return reason, bad
		}
	}
	return "", false
}

// checkParams rejects bad identifiers and duplicated NAMED params.
func checkParams(params []Param) (string, bool) {
	seen := map[string]bool{}
	for _, p := range params {
		if p.Name == "" {
			continue
		}
		if !token.IsIdentifier(p.Name) {
			return ReasonInvalidMemberName, true
		}
		if seen[p.Name] {
			return ReasonDuplicateMember, true
		}
		seen[p.Name] = true
	}
	return "", false
}

// allTypeRefs flattens every type reference in a spec.
func allTypeRefs(s Spec) []TypeRef {
	var out []TypeRef
	for _, f := range s.Fields {
		out = append(out, f.Type)
	}
	out = append(out, s.Embeds...)
	for _, m := range s.Methods {
		for _, p := range m.Params {
			out = append(out, p.Type)
		}
		out = append(out, m.Results...)
	}
	for _, p := range s.Params {
		out = append(out, p.Type)
	}
	out = append(out, s.Results...)
	return out
}

// checkTypeExpr verifies that expr parses and that every node in it
// is a type-expression node (the check a bare ParseExpr omits).
func checkTypeExpr(expr string) (string, bool) {
	parsed, err := parser.ParseExpr(expr)
	if err != nil {
		return ReasonInvalidTypeExpr, true
	}
	if !isTypeExpr(parsed) {
		return ReasonNonTypeExpr, true
	}
	return "", false
}

// isTypeExpr reports whether an AST expression is a type expression,
// node by node.
func isTypeExpr(e ast.Expr) bool {
	switch n := e.(type) {
	case *ast.Ident, *ast.SelectorExpr, *ast.BasicLit:
		return true // BasicLit: array length
	case *ast.StarExpr:
		return isTypeExpr(n.X)
	case *ast.ArrayType:
		if n.Len != nil && !isTypeExpr(n.Len) {
			return false
		}
		return isTypeExpr(n.Elt)
	case *ast.MapType:
		return isTypeExpr(n.Key) && isTypeExpr(n.Value)
	case *ast.ParenExpr:
		return isTypeExpr(n.X)
	case *ast.Ellipsis:
		return isTypeExpr(n.Elt)
	case *ast.ChanType:
		return isTypeExpr(n.Value)
	case *ast.IndexExpr:
		return isTypeExpr(n.X) && isTypeExpr(n.Index)
	case *ast.IndexListExpr:
		if !isTypeExpr(n.X) {
			return false
		}
		for _, ix := range n.Indices {
			if !isTypeExpr(ix) {
				return false
			}
		}
		return true
	case *ast.FuncType:
		for _, f := range n.Params.List {
			if !isTypeExpr(f.Type) {
				return false
			}
		}
		if n.Results != nil {
			for _, f := range n.Results.List {
				if !isTypeExpr(f.Type) {
					return false
				}
			}
		}
		return true
	case *ast.StructType:
		for _, f := range n.Fields.List {
			if !isTypeExpr(f.Type) {
				return false
			}
		}
		return true
	case *ast.InterfaceType:
		for _, f := range n.Methods.List {
			if !isTypeExpr(f.Type) {
				return false
			}
		}
		return true
	}
	return false
}
