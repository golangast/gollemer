// Package decl is the data pipeline for the "decl" seq2seq experiment:
// a model that learns small Go declarations (structs, interfaces,
// function signatures) from structured specs and from natural phrasings.
//
// The pipeline is deliberately deterministic everywhere except the model:
//
//	spec (schema registry) -> render (template) -> filter (gatekeeper)
//	-> datasets (train pairs + held-out matrix) -> trainer -> eval
//
// Specs are the intermediate representation — the same shape the learned
// intent layer produces when it understands a sentence — so the model
// learns structure -> syntax, not word association. Every spec slot is
// fillable at runtime from a user's sentence or from the analyzer index;
// there is no slot the runtime cannot fill.
//
// Canonical form: type expressions are stored parse-and-printed
// (go/format), identifiers in their natural Go case (User, time.Time).
// The seq2seq trainer lowercases everything, so datasets are written in
// the lowercase rendering of the canonical spec; a pool-derived restore
// map brings case back for compilation checks. Lowercase forms of all
// pool atoms are injective by construction (asserted in tests).
package decl

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"sort"
	"strings"
)

// Kind is a declaration kind in the registry.
type Kind string

const (
	KindStruct    Kind = "struct"
	KindInterface Kind = "interface"
	KindFunc      Kind = "func"
)

// Type classes for TypeRef.Class: the class of the reference's head.
// Composite expressions (slices, maps, pointers, func types) report
// ClassComposite; their element classes are recoverable by walking the
// parsed expression.
const (
	ClassBuiltin   = "builtin"
	ClassLocal     = "local"
	ClassImported  = "imported"
	ClassComposite = "composite"
)

// TypeRef is one structured type reference. Expr is the canonical
// parse-and-printed form ("[]*User", "time.Time", "map[string]int").
// Package is the import path when Class is ClassImported ("time",
// "encoding/json", ...); for composites it is empty — use ImportPaths
// to enumerate every package an expression references.
type TypeRef struct {
	Expr    string `json:"expr"`
	Class   string `json:"class"`
	Package string `json:"package,omitempty"`
}

// Field is one struct field. Name is empty iff Embedded is true.
type Field struct {
	Name     string  `json:"name"`
	Embedded bool    `json:"embedded,omitempty"`
	Type     TypeRef `json:"type"`
	Tag      string  `json:"tag,omitempty"` // tag content without backticks: json:"id"
}

// Param is one named or unnamed parameter. Name may be empty.
type Param struct {
	Name string  `json:"name"`
	Type TypeRef `json:"type"`
}

// Method is one interface method signature.
type Method struct {
	Name    string    `json:"name"`
	Params  []Param   `json:"params,omitempty"`
	Results []TypeRef `json:"results,omitempty"`
}

// Receiver is a method receiver: (s Store) or (s *Store).
type Receiver struct {
	Name    string `json:"name"`
	Type    string `json:"type"` // local type name, canonical case
	Pointer bool   `json:"pointer,omitempty"`
}

// Spec is one declaration specification — the registry's per-kind shape.
// Structs use Fields (Embedded entries included) only; interfaces use
// Embeds + Methods; funcs use Receiver/Params/Results.
type Spec struct {
	Kind     Kind      `json:"kind"`
	Name     string    `json:"name"`
	Fields   []Field   `json:"fields,omitempty"`
	Embeds   []TypeRef `json:"embeds,omitempty"`
	Methods  []Method  `json:"methods,omitempty"`
	Receiver *Receiver `json:"receiver,omitempty"`
	Params   []Param   `json:"params,omitempty"`
	Results  []TypeRef `json:"results,omitempty"`
}

// ErrorTuple reports whether a result list ends in error — the
// relational slot the layered spec carries for signatures. It is
// derived, never stored: the last result's type decides it.
func ErrorTuple(results []TypeRef) bool {
	return len(results) > 0 && results[len(results)-1].Expr == "error"
}

// builtinTypes is the closed set of predeclared type identifiers the
// generator draws from (error is handled separately: it is a valid
// field type but lives mostly in result lists).
var builtinTypes = map[string]bool{
	"string": true, "int": true, "int64": true, "int32": true,
	"float64": true, "float32": true, "bool": true, "byte": true,
	"rune": true, "uint": true, "uint64": true, "any": true,
	"error": true, "uintptr": true, "complex128": true,
}

// importPaths maps a package's local name to its import path for every
// imported type the pools may reference.
var importPaths = map[string]string{
	"time":    "time",
	"context": "context",
	"json":    "encoding/json",
	"http":    "net/http",
	"url":     "net/url",
	"io":      "io",
	"sql":     "database/sql",
}

// CanonicalExpr parses a type expression and returns its go/format
// printed form — the ONE string form of a type used by the generator,
// the filter, and the datasets.
func CanonicalExpr(expr string) (string, error) {
	parsed, err := parser.ParseExpr(expr)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	if err := format.Node(&b, token.NewFileSet(), parsed); err != nil {
		return "", err
	}
	return b.String(), nil
}

// MakeTypeRef builds a TypeRef from any spelling of a type expression:
// the expression is canonicalized and its head classified.
func MakeTypeRef(expr string) (TypeRef, error) {
	canon, err := CanonicalExpr(expr)
	if err != nil {
		return TypeRef{}, fmt.Errorf("type %q: %w", expr, err)
	}
	parsed, err := parser.ParseExpr(canon)
	if err != nil {
		return TypeRef{}, err
	}
	ref := TypeRef{Expr: canon, Class: ClassComposite}
	switch n := parsed.(type) {
	case *ast.Ident:
		if builtinTypes[n.Name] {
			ref.Class = ClassBuiltin
		} else {
			ref.Class = ClassLocal
		}
	case *ast.SelectorExpr:
		if x, ok := n.X.(*ast.Ident); ok {
			if path, known := importPaths[x.Name]; known {
				ref.Class = ClassImported
				ref.Package = path
			}
		}
	}
	return ref, nil
}

// MustTypeRef is MakeTypeRef for pool literals, which are valid by
// construction; it panics on a bad pool entry (a build-time bug).
func MustTypeRef(expr string) TypeRef {
	ref, err := MakeTypeRef(expr)
	if err != nil {
		panic(err)
	}
	return ref
}

// ImportPaths returns the sorted set of import paths a type expression
// references (via selector heads known to the registry).
func ImportPaths(expr string) []string {
	parsed, err := parser.ParseExpr(expr)
	if err != nil {
		return nil
	}
	set := map[string]bool{}
	ast.Inspect(parsed, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if x, ok := sel.X.(*ast.Ident); ok {
				if path, known := importPaths[x.Name]; known {
					set[path] = true
				}
			}
		}
		return true
	})
	out := make([]string, 0, len(set))
	for p := range set {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}

// LocalNames returns the sorted set of local (project) type names a
// type expression references: identifiers that are neither builtins
// nor package qualifiers.
func LocalNames(expr string) []string {
	parsed, err := parser.ParseExpr(expr)
	if err != nil {
		return nil
	}
	set := map[string]bool{}
	ast.Inspect(parsed, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.SelectorExpr:
			// The X of a selector is a package name, not a type.
			if _, isIdent := node.X.(*ast.Ident); isIdent {
				return false // don't descend into X; still visit Sel (an ident, harmless)
			}
		case *ast.Ident:
			if !builtinTypes[node.Name] {
				set[node.Name] = true
			}
		}
		return true
	})
	out := make([]string, 0, len(set))
	for name := range set {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// BaseName is the named type at the core of a reference: for an
// embedded field or a receiver, the type's own name ("User" for User,
// *User, and []User alike — the first local name, if any).
func BaseName(ref TypeRef) string {
	if names := LocalNames(ref.Expr); len(names) > 0 {
		return names[0]
	}
	return ref.Expr
}

// Canonicalize returns the spec with every type expression in canonical
// form and empty slices normalized to nil, so two specs can be
// compared with reflect.DeepEqual. Class/Package are recomputed so a
// spec built by hand and one extracted from an AST converge.
func Canonicalize(s Spec) (Spec, error) {
	out := s
	canonRef := func(r TypeRef) (TypeRef, error) {
		return MakeTypeRef(r.Expr)
	}
	var err error
	if out.Fields != nil {
		fields := make([]Field, len(out.Fields))
		for i, f := range out.Fields {
			fields[i] = f
			if fields[i].Type, err = canonRef(f.Type); err != nil {
				return Spec{}, err
			}
		}
		out.Fields = fields
		if len(out.Fields) == 0 {
			out.Fields = nil
		}
	}
	if out.Embeds != nil {
		embeds := make([]TypeRef, len(out.Embeds))
		for i, e := range out.Embeds {
			if embeds[i], err = canonRef(e); err != nil {
				return Spec{}, err
			}
		}
		out.Embeds = embeds
		if len(out.Embeds) == 0 {
			out.Embeds = nil
		}
	}
	if out.Methods != nil {
		methods := make([]Method, len(out.Methods))
		for i, m := range out.Methods {
			methods[i], err = canonSignature(m)
			if err != nil {
				return Spec{}, err
			}
		}
		out.Methods = methods
		if len(out.Methods) == 0 {
			out.Methods = nil
		}
	}
	if out.Params != nil {
		params := make([]Param, len(out.Params))
		for i, p := range out.Params {
			params[i] = p
			if params[i].Type, err = canonRef(p.Type); err != nil {
				return Spec{}, err
			}
		}
		out.Params = params
		if len(out.Params) == 0 {
			out.Params = nil
		}
	}
	if out.Results != nil {
		results := make([]TypeRef, len(out.Results))
		for i, r := range out.Results {
			if results[i], err = canonRef(r); err != nil {
				return Spec{}, err
			}
		}
		out.Results = results
		if len(out.Results) == 0 {
			out.Results = nil
		}
	}
	return out, nil
}

func canonSignature(m Method) (Method, error) {
	out := m
	var err error
	if out.Params != nil {
		params := make([]Param, len(out.Params))
		for i, p := range out.Params {
			params[i] = p
			if params[i].Type, err = MakeTypeRef(p.Type.Expr); err != nil {
				return Method{}, err
			}
		}
		out.Params = params
		if len(out.Params) == 0 {
			out.Params = nil
		}
	}
	if out.Results != nil {
		results := make([]TypeRef, len(out.Results))
		for i, r := range out.Results {
			if results[i], err = MakeTypeRef(r.Expr); err != nil {
				return Method{}, err
			}
		}
		out.Results = results
		if len(out.Results) == 0 {
			out.Results = nil
		}
	}
	return out, nil
}

// LowerSpec returns the spec in the model's lowercase world: every
// name and expression lowercased. Structural comparison between a
// model output (extracted from lowercase text) and a canonical spec
// happens between LowerSpec(want) and the extracted spec.
func LowerSpec(s Spec) Spec {
	lowerRef := func(r TypeRef) TypeRef {
		return TypeRef{Expr: strings.ToLower(r.Expr), Class: r.Class, Package: r.Package}
	}
	out := Spec{Kind: s.Kind, Name: strings.ToLower(s.Name)}
	for _, f := range s.Fields {
		out.Fields = append(out.Fields, Field{
			Name: strings.ToLower(f.Name), Embedded: f.Embedded,
			Type: lowerRef(f.Type), Tag: f.Tag,
		})
	}
	for _, e := range s.Embeds {
		out.Embeds = append(out.Embeds, lowerRef(e))
	}
	for _, m := range s.Methods {
		lm := Method{Name: strings.ToLower(m.Name)}
		for _, p := range m.Params {
			lm.Params = append(lm.Params, Param{Name: strings.ToLower(p.Name), Type: lowerRef(p.Type)})
		}
		for _, r := range m.Results {
			lm.Results = append(lm.Results, lowerRef(r))
		}
		out.Methods = append(out.Methods, lm)
	}
	if s.Receiver != nil {
		out.Receiver = &Receiver{
			Name:    strings.ToLower(s.Receiver.Name),
			Type:    strings.ToLower(s.Receiver.Type),
			Pointer: s.Receiver.Pointer,
		}
	}
	for _, p := range s.Params {
		out.Params = append(out.Params, Param{Name: strings.ToLower(p.Name), Type: lowerRef(p.Type)})
	}
	for _, r := range s.Results {
		out.Results = append(out.Results, lowerRef(r))
	}
	// Normalize empties to nil the way Canonicalize does.
	c, err := Canonicalize(out)
	if err != nil {
		return out
	}
	return c
}
