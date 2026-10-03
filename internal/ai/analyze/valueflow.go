package analyze

import (
	"sort"
	"strings"
)

// Value flow: which functions can observe which values. For every
// "pkg.Type.Field" read through a parameter or receiver, the index
// records the readers — directly, plus functions that thread a value of
// that type into a reader through a call. Fully general: no feature
// keywords, no per-issue logic. It answers "what reads X", "what would
// break if X changed", and "where do options flow" for any value.
//
// Example: main passes cfg to RunCLI, RunCLI reads cfg.Verbose — then
// both main and RunCLI are readers of config.Config.Verbose.

// FieldReaders returns the IDs of functions that read a "pkg.Type.Field"
// value, directly or through threaded calls, sorted.
func (p *Project) FieldReaders(field string) []string {
	return p.fieldReaders[field]
}

// FieldWriters returns the IDs of functions that write a
// "pkg.Type.Field" value, sorted.
func (p *Project) FieldWriters(field string) []string {
	return p.fieldWriters[field]
}

// buildFieldWriters computes the write index: which functions set which
// values. Direct only — unlike reads, a write is meaningful exactly
// where it happens.
func (p *Project) buildFieldWriters() {
	p.fieldWriters = map[string][]string{}
	byField := map[string]map[string]bool{}
	for _, fn := range p.byID {
		for _, w := range fn.Writes {
			m := byField[w]
			if m == nil {
				m = map[string]bool{}
				byField[w] = m
			}
			m[fn.ID] = true
		}
	}
	for f, m := range byField {
		p.fieldWriters[f] = sortedKeys(m)
	}
}

// buildFieldReaders computes the value-flow index by fixpoint over the
// call graph. Called from BuildGraph after the per-function passes.
func (p *Project) buildFieldReaders() {
	direct := map[string]map[string]bool{}
	add := func(field, id string) bool {
		m := direct[field]
		if m == nil {
			m = map[string]bool{}
			direct[field] = m
		}
		if m[id] {
			return false
		}
		m[id] = true
		return true
	}
	for _, fn := range p.byID {
		for _, r := range fn.Reads {
			add(r, fn.ID)
		}
	}
	// Thread values through calls: F calls G passing its own T-typed
	// parameter, G receives a T-typed parameter — then F can observe
	// every T field G reads, including reads G itself only threaded.
	// The fixpoint reads from the accumulated index (direct), not from
	// g.Reads, so chains like main -> Middle -> leaf propagate fully.
	for changed := true; changed; {
		changed = false
		for _, fn := range p.byID {
			for _, ca := range fn.CallArgs {
				if ca.Target == "" {
					continue
				}
				g := p.byID[ca.Target]
				if g == nil {
					continue
				}
				for _, a := range ca.Args {
					t := valueTypeOf(fn, a)
					if t == "" || !receivesType(g, t) {
						continue
					}
					prefix := t + "."
					for field, readers := range direct {
						if !strings.HasPrefix(field, prefix) || !readers[g.ID] {
							continue
						}
						if add(field, fn.ID) {
							changed = true
						}
					}
				}
			}
		}
	}
	p.fieldReaders = map[string][]string{}
	for f, m := range direct {
		p.fieldReaders[f] = sortedKeys(m)
	}
}

// valueTypeOf resolves the "pkg.Type" of a parameter, receiver, or local
// variable name. Empty when the type isn't a named struct/pointer shape.
func valueTypeOf(fn *Func, name string) string {
	if t := fn.ParamTypes[name]; t != "" {
		return t
	}
	return fn.localTypes[name]
}

// receivesType reports whether g has a parameter of type t.
func receivesType(g *Func, t string) bool {
	for _, pt := range g.ParamTypes {
		if pt == t {
			return true
		}
	}
	return false
}

// effectFuncs returns the functions with an effect category in their
// transitive effects, sorted by func ID, capped for display.
func (p *Project) effectFuncs(cat string, cap int) []*Func {
	var out []*Func
	for _, fn := range p.byID {
		if fn.IsTest {
			continue
		}
		if hasEffect(fn.EffectsAll, cat) {
			out = append(out, fn)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	if len(out) > cap {
		out = out[:cap]
	}
	return out
}
