package decl

import (
	"math/rand"
	"strings"
)

// The combinatorial sampler. Specs are constructed slot by slot from
// the pools with weighted choices: member counts, type categories,
// tags, embeddings, receivers, error-tuple results. The sampler
// avoids the common decode traps (a member named like its own type,
// a receiver/param name collision); the filter remains the single
// source of truth and rejects whatever slips through.

type sampler struct {
	rng *rand.Rand
}

func (s *sampler) pick(pool []string) string {
	return pool[s.rng.Intn(len(pool))]
}

// pickDistinct returns n distinct elements of pool (fewer if the pool
// runs out).
func (s *sampler) pickDistinct(pool []string, n int) []string {
	if n > len(pool) {
		n = len(pool)
	}
	idx := s.rng.Perm(len(pool))[:n]
	out := make([]string, 0, n)
	for _, i := range idx {
		out = append(out, pool[i])
	}
	return out
}

// count draws a member count from a weight table indexed by count.
func (s *sampler) count(weights ...float64) int {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	r := s.rng.Float64() * total
	for i, w := range weights {
		r -= w
		if r <= 0 {
			return i
		}
	}
	return len(weights) - 1
}

// simpleRef draws a named (non-composite) type reference.
func (s *sampler) simpleRef() TypeRef {
	switch r := s.rng.Float64(); {
	case r < 0.52:
		return MustTypeRef(s.pick(BuiltinPool))
	case r < 0.78:
		return MustTypeRef(s.pick(LocalTypes))
	default:
		return MustTypeRef(s.pick(ImportedPool))
	}
}

// typeRef draws any type reference: named types dominate, composites
// (slices, maps, pointers, func types, generics, channels) make up
// the rest. allowError admits the error type (fields may carry it
// rarely; results steer it separately).
func (s *sampler) typeRef(allowError bool) TypeRef {
	r := s.rng.Float64()
	switch {
	case r < 0.55:
		ref := s.simpleRef()
		if ref.Expr == "error" && !allowError {
			return MustTypeRef("string")
		}
		return ref
	case r < 0.68: // slice
		return MustTypeRef("[]" + s.simpleRef().Expr)
	case r < 0.76: // pointer
		return MustTypeRef("*" + s.namedRef().Expr)
	case r < 0.83: // map
		key := "string"
		if s.rng.Float64() < 0.3 {
			key = "int"
		}
		return MustTypeRef("map[" + key + "]" + s.simpleRef().Expr)
	case r < 0.87:
		return MustTypeRef("[]byte")
	case r < 0.91: // func type
		variants := []string{"func(int) error", "func(string) string", "func() error", "func(int, int) int"}
		return MustTypeRef(s.pick(variants))
	case r < 0.95: // generic instantiation
		if s.rng.Float64() < 0.5 {
			return MustTypeRef("List[" + s.simpleRef().Expr + "]")
		}
		return MustTypeRef("Pair[string, " + s.simpleRef().Expr + "]")
	case r < 0.98: // pointer to slice element style depth-2: []*T
		return MustTypeRef("[]*" + s.namedRef().Expr)
	default:
		return MustTypeRef("chan " + s.simpleRef().Expr)
	}
}

// namedRef draws a named type (local or imported) for pointer bases.
func (s *sampler) namedRef() TypeRef {
	if s.rng.Float64() < 0.7 {
		return MustTypeRef(s.pick(LocalTypes))
	}
	return MustTypeRef(s.pick(ImportedPool))
}

// resultList draws a result list with Go-idiomatic error steering:
// two-result lists usually end in error; a single result is error
// about a quarter of the time.
func (s *sampler) resultList() []TypeRef {
	switch n := s.count(0.30, 0.45, 0.25); n {
	case 0:
		return nil
	case 1:
		if s.rng.Float64() < 0.25 {
			return []TypeRef{MustTypeRef("error")}
		}
		return []TypeRef{s.nonErrorResult()}
	default:
		second := s.nonErrorResult()
		if s.rng.Float64() < 0.70 {
			second = MustTypeRef("error")
		}
		return []TypeRef{s.nonErrorResult(), second}
	}
}

func (s *sampler) nonErrorResult() TypeRef {
	ref := s.typeRef(false)
	if ref.Expr == "error" {
		return MustTypeRef("bool")
	}
	return ref
}

// params draws a parameter list: mostly named, names steered to
// idiomatic types half the time, never named like their own type.
func (s *sampler) params(max int) []Param {
	return s.paramsMode(max, false)
}

// paramsMode draws a parameter list. With uniform=true the list is
// all-named (80%) or all-unnamed. Uniform naming is required for
// every signature kind: func declarations reject mixed lists
// outright ("missing parameter name"), and in interface methods a
// leading unnamed parameter is silently absorbed as a NAME for the
// next parameter's type (Copy(uint64, size Token) declares params
// uint64 and size, both Token) — so the generator never mixes.
func (s *sampler) paramsMode(max int, funcDecl bool) []Param {
	n := s.count(0.22, 0.34, 0.26, 0.13, 0.05)
	if n > max {
		n = max
	}
	if n == 0 {
		return nil
	}
	names := s.pickDistinct(ParamNames, n)
	allNamed := s.rng.Float64() < 0.80
	out := make([]Param, 0, n)
	for _, name := range names {
		var ref TypeRef
		if steered := paramTypeForName(name); steered != "" && s.rng.Float64() < 0.55 {
			ref = MustTypeRef(steered)
		} else {
			ref = s.typeRef(false)
		}
		if strings.EqualFold(name, BaseName(ref)) {
			ref = MustTypeRef("string")
		}
		p := Param{Type: ref}
		switch {
		case funcDecl && !allNamed:
			// uniformly unnamed
		case funcDecl && allNamed:
			p.Name = name
		case s.rng.Float64() < 0.72:
			p.Name = name
		}
		out = append(out, p)
	}
	return out
}

// funcParams draws a uniformly-named parameter list, used for func
// declarations AND interface methods (see paramsMode).
func (s *sampler) funcParams(max int) []Param {
	return s.paramsMode(max, true)
}

func (s *sampler) spec(kind Kind) Spec {
	switch kind {
	case KindStruct:
		return s.structSpec()
	case KindInterface:
		return s.interfaceSpec()
	default:
		return s.funcSpec()
	}
}

func (s *sampler) structSpec() Spec {
	name := s.pick(LocalTypes)
	spec := Spec{Kind: KindStruct, Name: name}
	n := s.count(0.08, 0.18, 0.26, 0.24, 0.14, 0.10) // 0..5 fields
	// Optional embedding (never of the struct itself).
	if n >= 0 && s.rng.Float64() < 0.12 {
		emb := s.pick(LocalTypes)
		if emb != name {
			expr := emb
			if s.rng.Float64() < 0.30 {
				expr = "*" + emb
			}
			spec.Fields = append(spec.Fields, Field{Embedded: true, Type: MustTypeRef(expr)})
		}
	}
	remaining := n
	names := s.pickDistinct(FieldNames, remaining)
	for _, fname := range names {
		f := Field{Name: fname, Type: s.typeRef(true)}
		if strings.EqualFold(fname, BaseName(f.Type)) {
			f.Type = MustTypeRef("int")
		}
		if s.rng.Float64() < 0.30 {
			tag := `json:"` + strings.ToLower(fname) + `"`
			if s.rng.Float64() < 0.25 {
				tag = `json:"` + strings.ToLower(fname) + `,omitempty"`
			}
			f.Tag = tag
		}
		spec.Fields = append(spec.Fields, f)
	}
	return spec
}

func (s *sampler) interfaceSpec() Spec {
	name := s.pick(LocalTypes)
	spec := Spec{Kind: KindInterface, Name: name}
	if s.rng.Float64() < 0.15 {
		if s.rng.Float64() < 0.45 {
			spec.Embeds = append(spec.Embeds, MustTypeRef(s.pick([]string{"io.Reader", "io.Writer"})))
		} else if emb := s.pick(LocalTypes); emb != name {
			spec.Embeds = append(spec.Embeds, MustTypeRef(emb))
		}
	}
	n := s.count(0.10, 0.25, 0.30, 0.22, 0.13) // 0..4 methods
	for _, mname := range s.pickDistinct(VerbNames, n) {
		m := Method{Name: mname, Params: s.funcParams(3), Results: s.resultList()}
		spec.Methods = append(spec.Methods, m)
	}
	return spec
}

func (s *sampler) funcSpec() Spec {
	// Constructors: New<Type> returning the type (often with error).
	if s.rng.Float64() < 0.12 {
		t := s.pick(LocalTypes)
		spec := Spec{Kind: KindFunc, Name: "New" + t, Params: s.funcParams(2)}
		star := "*"
		if s.rng.Float64() < 0.25 {
			star = ""
		}
		res := []TypeRef{MustTypeRef(star + t)}
		if s.rng.Float64() < 0.35 {
			res = append(res, MustTypeRef("error"))
		}
		spec.Results = res
		return spec
	}
	spec := Spec{Kind: KindFunc, Name: s.pick(VerbNames)}
	if s.rng.Float64() < 0.45 {
		rt := s.pick(LocalTypes)
		spec.Receiver = &Receiver{
			Name:    s.pick(ReceiverNames),
			Type:    rt,
			Pointer: s.rng.Float64() < 0.60,
		}
	}
	spec.Params = s.funcParams(4)
	// A parameter may not reuse the receiver's name (same scope):
	// rename the parameter (blanking it would break the uniform
	// named/unnamed discipline of a func declaration).
	if spec.Receiver != nil {
		used := map[string]bool{spec.Receiver.Name: true}
		for _, p := range spec.Params {
			used[p.Name] = true
		}
		for i, p := range spec.Params {
			if p.Name == "" || p.Name != spec.Receiver.Name {
				continue
			}
			for _, cand := range ParamNames {
				if !used[cand] && !strings.EqualFold(cand, BaseName(p.Type)) {
					spec.Params[i].Name = cand
					used[cand] = true
					break
				}
			}
		}
	}
	spec.Results = s.resultList()
	return spec
}
