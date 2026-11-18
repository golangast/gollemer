package decl

import (
	"strings"
)

// Phrasing frames: the linguistic diversification layer. Specs are
// rendered into natural phrasings by TEMPLATE FRAMES in three
// registers (terse shorthand, plain, verbose conversational),
// combinatorially filled — no external LLM writes phrasings for this
// project (the user's settled rule: no rented models).
//
// A subset of frames is RESERVED: phrasings produced by them never
// appear in training data, so the held-out eval measures phrasing
// generalization on genuinely unseen sentence structures. Reserved
// frames are built from the same word stock as seen frames (the
// generator asserts full word coverage), so the axis tests structure,
// not vocabulary.

type Register string

const (
	RegisterTerse   Register = "terse"
	RegisterPlain   Register = "plain"
	RegisterVerbose Register = "verbose"
)

// Frame is one phrasing template. Placeholders: {name}, {members}
// (struct/interface), {signature}, {params_code}, {results_code},
// {receiver_code}, {params_clause}, {results_clause},
// {receiver_clause}. Method frames phrase receiver-bearing funcs.
type Frame struct {
	ID       string
	Kind     Kind
	Method   bool
	Register Register
	Reserved bool
	Template string
}

// Frames is the full frame registry.
var Frames = []Frame{
	// --- structs ---
	{ID: "S-T1", Kind: KindStruct, Register: RegisterTerse, Template: "{name} struct: {members}"},
	{ID: "S-T2", Kind: KindStruct, Register: RegisterTerse, Template: "struct {name} with {members}"},
	{ID: "S-T3", Kind: KindStruct, Register: RegisterTerse, Reserved: true, Template: "{name}: struct, {members}"},
	{ID: "S-P1", Kind: KindStruct, Register: RegisterPlain, Template: "make a struct named {name} with {members}"},
	{ID: "S-P2", Kind: KindStruct, Register: RegisterPlain, Template: "create a struct called {name} containing {members}"},
	{ID: "S-P3", Kind: KindStruct, Register: RegisterPlain, Reserved: true, Template: "define a struct {name} with {members}"},
	{ID: "S-V1", Kind: KindStruct, Register: RegisterVerbose, Template: "can you make me a go struct called {name} that has {members}"},
	{ID: "S-V2", Kind: KindStruct, Register: RegisterVerbose, Template: "i really need a struct named {name} with {members}"},
	{ID: "S-V3", Kind: KindStruct, Register: RegisterVerbose, Reserved: true, Template: "please spin up a struct named {name} containing {members}"},
	{ID: "S-V4", Kind: KindStruct, Register: RegisterVerbose, Reserved: true, Template: "how about a struct {name} with {members}"},

	// --- interfaces ---
	{ID: "I-T1", Kind: KindInterface, Register: RegisterTerse, Template: "{name} interface: {members}"},
	{ID: "I-T2", Kind: KindInterface, Register: RegisterTerse, Template: "interface {name} with {members}"},
	{ID: "I-T3", Kind: KindInterface, Register: RegisterTerse, Reserved: true, Template: "{name}: interface, {members}"},
	{ID: "I-P1", Kind: KindInterface, Register: RegisterPlain, Template: "make an interface named {name} with {members}"},
	{ID: "I-P2", Kind: KindInterface, Register: RegisterPlain, Template: "create an interface called {name} with {members}"},
	{ID: "I-P3", Kind: KindInterface, Register: RegisterPlain, Template: "define an interface named {name} with {members}"},
	{ID: "I-V1", Kind: KindInterface, Register: RegisterVerbose, Template: "could you make me a go interface called {name} that has {members}"},
	{ID: "I-V2", Kind: KindInterface, Register: RegisterVerbose, Template: "how about an interface {name} with {members}"},
	{ID: "I-V3", Kind: KindInterface, Register: RegisterVerbose, Reserved: true, Template: "i really need an interface called {name} with {members}"},
	{ID: "I-V4", Kind: KindInterface, Register: RegisterVerbose, Reserved: true, Template: "please create an interface named {name} with {members}"},

	// --- functions (no receiver) ---
	{ID: "F-T1", Kind: KindFunc, Register: RegisterTerse, Template: "{signature}"},
	{ID: "F-T2", Kind: KindFunc, Register: RegisterTerse, Template: "{name} ( {params_code} ) -> {results_code}"},
	{ID: "F-T3", Kind: KindFunc, Register: RegisterTerse, Reserved: true, Template: "function {name} ( {params_code} ) returns {results_code}"},
	{ID: "F-P1", Kind: KindFunc, Register: RegisterPlain, Template: "make a function named {name} {params_clause} {results_clause}"},
	{ID: "F-P2", Kind: KindFunc, Register: RegisterPlain, Template: "write a function called {name} {params_clause} {results_clause}"},
	{ID: "F-P3", Kind: KindFunc, Register: RegisterPlain, Template: "please write a function named {name} {params_clause} {results_clause}"},
	{ID: "F-V1", Kind: KindFunc, Register: RegisterVerbose, Template: "can you write a function called {name} {params_clause} {results_clause}"},
	{ID: "F-V2", Kind: KindFunc, Register: RegisterVerbose, Template: "spin up a function named {name} {params_clause} {results_clause}"},
	{ID: "F-V3", Kind: KindFunc, Register: RegisterVerbose, Reserved: true, Template: "i need a function named {name} {params_clause} {results_clause}"},
	{ID: "F-V4", Kind: KindFunc, Register: RegisterVerbose, Reserved: true, Template: "could you create a function {name} {params_clause} {results_clause}"},

	// --- methods (receiver-bearing funcs) ---
	{ID: "M-T1", Kind: KindFunc, Method: true, Register: RegisterTerse, Template: "{signature}"},
	{ID: "M-T2", Kind: KindFunc, Method: true, Register: RegisterTerse, Reserved: true, Template: "method {name} on {receiver_code}: {params_code} -> {results_code}"},
	{ID: "M-P1", Kind: KindFunc, Method: true, Register: RegisterPlain, Template: "add a method named {name} {receiver_clause} {params_clause} {results_clause}"},
	{ID: "M-P2", Kind: KindFunc, Method: true, Register: RegisterPlain, Template: "make a method called {name} {receiver_clause} {params_clause} {results_clause}"},
	{ID: "M-P3", Kind: KindFunc, Method: true, Register: RegisterPlain, Reserved: true, Template: "define a method {name} {receiver_clause} {params_clause} {results_clause}"},
	{ID: "M-V1", Kind: KindFunc, Method: true, Register: RegisterVerbose, Template: "can you add a method called {name} {receiver_clause} {params_clause} {results_clause}"},
	{ID: "M-V2", Kind: KindFunc, Method: true, Register: RegisterVerbose, Reserved: true, Template: "i need a method {name} {receiver_clause} {params_clause} {results_clause}"},
}

// FrameByID looks up a frame.
func FrameByID(id string) (Frame, bool) {
	for _, f := range Frames {
		if f.ID == id {
			return f, true
		}
	}
	return Frame{}, false
}

// FramesFor returns the frames matching a spec's shape.
func FramesFor(s Spec, reserved bool) []Frame {
	var out []Frame
	for _, f := range Frames {
		if f.Kind != s.Kind || f.Reserved != reserved {
			continue
		}
		if s.Kind == KindFunc && f.Method != (s.Receiver != nil) {
			continue
		}
		out = append(out, f)
	}
	return out
}

// Fill renders a spec through a frame into a lowercase phrasing (the
// model's input world is lowercase).
func Fill(f Frame, s Spec) string {
	ls := LowerSpec(s)
	out := f.Template
	out = strings.ReplaceAll(out, "{name}", ls.Name)
	out = strings.ReplaceAll(out, "{members}", membersText(ls, f.Register))
	out = strings.ReplaceAll(out, "{signature}", signatureText(ls))
	out = strings.ReplaceAll(out, "{params_code}", paramsCode(ls.Params))
	out = strings.ReplaceAll(out, "{results_code}", resultsCode(ls.Results))
	out = strings.ReplaceAll(out, "{receiver_code}", receiverCode(ls.Receiver))
	out = strings.ReplaceAll(out, "{receiver_clause}", receiverClause(ls.Receiver))
	out = strings.ReplaceAll(out, "{params_clause}", paramsClause(ls.Params, f.Register))
	out = strings.ReplaceAll(out, "{results_clause}", resultsClause(ls.Results, f.Register))
	return strings.Join(strings.Fields(out), " ")
}

// signatureText is the terse full-signature phrasing: the rendered
// declaration without its scaffold body.
func signatureText(ls Spec) string {
	d := RenderDecl(ls)
	if i := strings.LastIndex(d, " {"); i >= 0 {
		return d[:i]
	}
	return d
}

func paramsCode(params []Param) string {
	parts := make([]string, 0, len(params))
	for _, p := range params {
		if p.Name != "" {
			parts = append(parts, p.Name+" "+TokenForm(p.Type.Expr))
		} else {
			parts = append(parts, TokenForm(p.Type.Expr))
		}
	}
	return strings.Join(parts, ", ")
}

func resultsCode(results []TypeRef) string {
	if len(results) == 0 {
		return "nothing"
	}
	parts := make([]string, 0, len(results))
	for _, r := range results {
		parts = append(parts, TokenForm(r.Expr))
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return "( " + strings.Join(parts, ", ") + " )"
}

func receiverCode(r *Receiver) string {
	if r == nil {
		return ""
	}
	t := r.Type
	if r.Pointer {
		t = "*" + t
	}
	return "( " + r.Name + " " + TokenForm(t) + " )"
}

func receiverClause(r *Receiver) string {
	if r == nil {
		return ""
	}
	if r.Pointer {
		return "on a pointer to the " + r.Type + " type"
	}
	return "on the " + r.Type + " type"
}

// membersText renders struct fields or interface members in a
// register's dialect.
func membersText(ls Spec, reg Register) string {
	switch ls.Kind {
	case KindStruct:
		if len(ls.Fields) == 0 {
			if reg == RegisterVerbose {
				return "no fields at all"
			}
			return "no fields"
		}
		items := make([]string, 0, len(ls.Fields))
		for _, f := range ls.Fields {
			items = append(items, structFieldText(f, reg))
		}
		return joinList(items)
	case KindInterface:
		var items []string
		for _, e := range ls.Embeds {
			items = append(items, embedText(e, reg))
		}
		for _, m := range ls.Methods {
			items = append(items, ifaceMethodText(m, reg))
		}
		if len(items) == 0 {
			if reg == RegisterVerbose {
				return "no methods at all"
			}
			return "no methods"
		}
		return joinList(items)
	}
	return ""
}

func structFieldText(f Field, reg Register) string {
	if f.Embedded {
		switch reg {
		case RegisterTerse:
			return TokenForm(f.Type.Expr)
		case RegisterPlain:
			return "an embedded " + TokenForm(f.Type.Expr)
		default:
			return "it embeds " + verbalType(f.Type, reg)
		}
	}
	switch reg {
	case RegisterTerse:
		s := f.Name + " " + TokenForm(f.Type.Expr)
		if f.Tag != "" {
			s += " `" + f.Tag + "`"
		}
		return s
	case RegisterPlain:
		s := "a field named " + f.Name + " of type " + verbalType(f.Type, reg)
		if f.Tag != "" {
			s += " with a json tag"
		}
		return s
	default:
		v := verbalType(f.Type, reg)
		s := article(v) + " " + v + " field called " + f.Name
		if f.Tag != "" {
			s += " with a json tag"
		}
		return s
	}
}

func embedText(e TypeRef, reg Register) string {
	switch reg {
	case RegisterTerse:
		return TokenForm(e.Expr)
	case RegisterPlain:
		return "an embedded " + TokenForm(e.Expr)
	default:
		return "it embeds " + TokenForm(e.Expr)
	}
}

func ifaceMethodText(m Method, reg Register) string {
	if reg == RegisterTerse {
		return RenderSignature(m.Name, m.Params, m.Results)
	}
	return "a " + m.Name + " method " + paramsClause(m.Params, reg) + " " + resultsClause(m.Results, reg)
}

// paramsClause renders "that takes ..." (plain) / "which accepts ..."
// (verbose) including the empty form.
func paramsClause(params []Param, reg Register) string {
	connector := "that takes"
	if reg == RegisterVerbose {
		connector = "which accepts"
	}
	if len(params) == 0 {
		return connector + " no arguments"
	}
	items := make([]string, 0, len(params))
	for _, p := range params {
		v := verbalType(p.Type, reg)
		item := article(v) + " " + v
		if p.Name != "" {
			item += " " + p.Name
		}
		items = append(items, item)
	}
	return connector + " " + joinList(items)
}

// resultsClause renders "and returns ..." (plain) / "and gives back
// ..." (verbose) including the empty form.
func resultsClause(results []TypeRef, reg Register) string {
	connector := "and returns"
	if reg == RegisterVerbose {
		connector = "and gives back"
	}
	if len(results) == 0 {
		if reg == RegisterVerbose {
			return "and does not return anything"
		}
		return "and returns nothing"
	}
	items := make([]string, 0, len(results))
	for _, r := range results {
		v := verbalType(r, reg)
		items = append(items, article(v)+" "+v)
	}
	return connector + " " + joinList(items)
}

// verbalType renders a type reference as natural-language noun phrase
// (no article). Builtins get word forms in the verbose register;
// composites are described structurally; anything exotic falls back
// to its canonical lowercase spelling, which is also how developers
// often say it.
func verbalType(ref TypeRef, reg Register) string {
	expr := ref.Expr
	if reg == RegisterVerbose {
		switch expr {
		case "int", "int64", "int32", "uint", "uint64":
			return "integer"
		case "bool":
			return "boolean"
		case "float64", "float32":
			return "decimal"
		}
	}
	switch {
	case expr == "[]byte":
		return "byte slice"
	case strings.HasPrefix(expr, "[]"):
		return "slice of " + verbalType(MustTypeRef(expr[2:]), reg)
	case strings.HasPrefix(expr, "*"):
		return "pointer to " + verbalType(MustTypeRef(expr[1:]), reg)
	case strings.HasPrefix(expr, "map["):
		rest := expr[len("map["):]
		idx := strings.Index(rest, "]")
		if idx >= 0 {
			key, val := rest[:idx], rest[idx+1:]
			return "map from " + verbalType(MustTypeRef(key), reg) + " to " + verbalType(MustTypeRef(val), reg)
		}
		return expr
	case strings.HasPrefix(expr, "chan "):
		return "channel of " + verbalType(MustTypeRef(expr[len("chan "):]), reg)
	}
	return TokenForm(expr)
}

// article returns "a" or "an" for a noun phrase.
func article(phrase string) string {
	if phrase == "" {
		return "a"
	}
	switch phrase[0] {
	case 'a', 'e', 'i', 'o', 'u':
		return "an"
	}
	return "a"
}

// joinList joins noun phrases the way people list them: "x", "x and
// y", "x, y and z".
func joinList(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	case 2:
		return items[0] + " and " + items[1]
	default:
		return strings.Join(items[:len(items)-1], ", ") + " and " + items[len(items)-1]
	}
}
