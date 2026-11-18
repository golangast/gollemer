package decl

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"strings"
)

// TokenForm spells a canonical type expression in the model's token
// dialect: parentheses and stars are space-separated. The trainer's
// tokenizer does not split ( ) or *, so "fetch(id int)" would fuse
// into the single token "fetch(id" — an atom the model can only learn
// combination by combination. Spaced, every token is a pool atom and
// any combination composes. The AST of the spaced form is identical;
// extraction canonicalizes it back.
func TokenForm(expr string) string {
	r := strings.NewReplacer("(", " ( ", ")", " ) ", "*", " * ")
	return strings.Join(strings.Fields(r.Replace(expr)), " ")
}

// RenderDecl renders a spec to its Go declaration in canonical case.
// The rendering is single-line with ";" separators inside bodies: the
// seq2seq decoder joins tokens with spaces, so the text the model
// learns to emit must stay parseable after that join — newlines would
// be lost, semicolons survive. Type expressions are written in
// TokenForm. A human-readable form is one format.Source call away;
// the AST is identical either way.
func RenderDecl(s Spec) string {
	switch s.Kind {
	case KindStruct:
		var members []string
		for _, f := range s.Fields {
			if f.Embedded {
				members = append(members, TokenForm(f.Type.Expr))
				continue
			}
			m := f.Name + " " + TokenForm(f.Type.Expr)
			if f.Tag != "" {
				m += " `" + NormalizeTag(f.Tag) + "`"
			}
			members = append(members, m)
		}
		if len(members) == 0 {
			return "type " + s.Name + " struct{}"
		}
		return "type " + s.Name + " struct { " + strings.Join(members, "; ") + " }"
	case KindInterface:
		var members []string
		for _, e := range s.Embeds {
			members = append(members, TokenForm(e.Expr))
		}
		for _, m := range s.Methods {
			members = append(members, RenderSignature(m.Name, m.Params, m.Results))
		}
		if len(members) == 0 {
			return "type " + s.Name + " interface{}"
		}
		return "type " + s.Name + " interface { " + strings.Join(members, "; ") + " }"
	case KindFunc:
		sig := RenderSignature(s.Name, s.Params, s.Results)
		// The scaffold body: empty for result-less functions. A
		// function WITH results and an empty body does not compile
		// ("missing return"), so result-bearing scaffolds carry the
		// standard panic stub — the minimal body that compiles.
		body := " {}"
		if len(s.Results) > 0 {
			body = ` { panic ( "not implemented" ) }`
		}
		if s.Receiver != nil {
			rt := s.Receiver.Type
			if s.Receiver.Pointer {
				rt = "*" + rt
			}
			return "func ( " + s.Receiver.Name + " " + TokenForm(rt) + " ) " + sig + body
		}
		return "func " + sig + body
	}
	return ""
}

// RenderSignature renders the bare signature form "Name(p T, q U)
// (R, error)" used for interface methods and, with a "func " prefix
// added by RenderDecl, for functions. A single result is written
// without parentheses, as Go idiom has it.
func RenderSignature(name string, params []Param, results []TypeRef) string {
	var b strings.Builder
	b.WriteString(name)
	b.WriteString(" ( ")
	parts := make([]string, 0, len(params))
	for _, p := range params {
		if p.Name != "" {
			parts = append(parts, p.Name+" "+TokenForm(p.Type.Expr))
		} else {
			parts = append(parts, TokenForm(p.Type.Expr))
		}
	}
	b.WriteString(strings.Join(parts, ", "))
	b.WriteString(" )")
	switch {
	case len(results) == 0:
		// nothing
	case len(results) == 1:
		b.WriteString(" " + TokenForm(results[0].Expr))
	default:
		rs := make([]string, 0, len(results))
		for _, r := range results {
			rs = append(rs, TokenForm(r.Expr))
		}
		b.WriteString(" ( " + strings.Join(rs, ", ") + " )")
	}
	return b.String()
}

// NormalizeTag canonicalizes a struct tag's content: all whitespace
// removed. Model output passes through a token join that inserts
// spaces inside tag literals ("json : " id ""); stripping whitespace
// restores the conventional form on both sides of every comparison,
// and tags in this pipeline never carry meaningful spaces.
func NormalizeTag(tag string) string {
	return strings.Join(strings.Fields(tag), "")
}

// SpecText renders a spec in the model's lowercase structured-input
// language — the form the intent layer's structured output takes when
// it reaches the model. Word delimiters (field, method, param, result,
// embed, tagged) survive the trainer's input normalization; type
// expressions are written in their lowercase canonical spelling.
func SpecText(s Spec) string {
	ls := LowerSpec(s)
	var b strings.Builder
	switch ls.Kind {
	case KindStruct:
		b.WriteString("spec struct " + ls.Name)
		for _, f := range ls.Fields {
			if f.Embedded {
				b.WriteString(" embed " + TokenForm(f.Type.Expr))
				continue
			}
			b.WriteString(" field " + f.Name + " " + TokenForm(f.Type.Expr))
			if f.Tag != "" {
				b.WriteString(" tagged")
			}
		}
	case KindInterface:
		b.WriteString("spec interface " + ls.Name)
		for _, e := range ls.Embeds {
			b.WriteString(" embed " + TokenForm(e.Expr))
		}
		for _, m := range ls.Methods {
			b.WriteString(" method " + m.Name)
			writeSpecParams(&b, m.Params)
			writeSpecResults(&b, m.Results)
		}
	case KindFunc:
		b.WriteString("spec func " + ls.Name)
		if ls.Receiver != nil {
			kind := "value"
			if ls.Receiver.Pointer {
				kind = "pointer"
			}
			b.WriteString(" receiver " + ls.Receiver.Name + " " + ls.Receiver.Type + " " + kind)
		}
		writeSpecParams(&b, ls.Params)
		writeSpecResults(&b, ls.Results)
	}
	return b.String()
}

func writeSpecParams(b *strings.Builder, params []Param) {
	for _, p := range params {
		if p.Name != "" {
			b.WriteString(" param " + p.Name + " " + TokenForm(p.Type.Expr))
		} else {
			b.WriteString(" param " + TokenForm(p.Type.Expr))
		}
	}
}

func writeSpecResults(b *strings.Builder, results []TypeRef) {
	for _, r := range results {
		b.WriteString(" result " + TokenForm(r.Expr))
	}
}

// ExtractSpec parses a declaration (in a package wrapper) and pulls
// the spec back out of the AST. It is the return leg of the filter's
// round-trip and the scorer's reader of model output. The returned
// spec carries the case of the source text it read.
func ExtractSpec(declText string) (Spec, error) {
	src := "package declcheck\n\n" + declText + "\n"
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "decl.go", src, parser.AllErrors)
	if err != nil {
		return Spec{}, err
	}
	for _, d := range file.Decls {
		switch decl := d.(type) {
		case *ast.GenDecl:
			if decl.Tok != token.TYPE || len(decl.Specs) != 1 {
				continue
			}
			ts, ok := decl.Specs[0].(*ast.TypeSpec)
			if !ok {
				continue
			}
			return extractTypeSpec(ts)
		case *ast.FuncDecl:
			return extractFuncDecl(decl)
		}
	}
	return Spec{}, fmt.Errorf("no declaration found")
}

func extractTypeSpec(ts *ast.TypeSpec) (Spec, error) {
	switch t := ts.Type.(type) {
	case *ast.StructType:
		spec := Spec{Kind: KindStruct, Name: ts.Name.Name}
		for _, f := range t.Fields.List {
			expr, err := exprString(f.Type)
			if err != nil {
				return Spec{}, err
			}
			ref, err := MakeTypeRef(expr)
			if err != nil {
				return Spec{}, err
			}
			tag := ""
			if f.Tag != nil {
				tag = NormalizeTag(strings.Trim(f.Tag.Value, "`"))
			}
			if len(f.Names) == 0 {
				spec.Fields = append(spec.Fields, Field{Embedded: true, Type: ref, Tag: tag})
				continue
			}
			for _, n := range f.Names {
				spec.Fields = append(spec.Fields, Field{Name: n.Name, Type: ref, Tag: tag})
			}
		}
		return Canonicalize(spec)
	case *ast.InterfaceType:
		spec := Spec{Kind: KindInterface, Name: ts.Name.Name}
		for _, m := range t.Methods.List {
			if len(m.Names) == 0 {
				expr, err := exprString(m.Type)
				if err != nil {
					return Spec{}, err
				}
				ref, err := MakeTypeRef(expr)
				if err != nil {
					return Spec{}, err
				}
				spec.Embeds = append(spec.Embeds, ref)
				continue
			}
			ft, ok := m.Type.(*ast.FuncType)
			if !ok {
				return Spec{}, fmt.Errorf("interface member %s is not a method", m.Names[0].Name)
			}
			params, results, err := extractSignature(ft)
			if err != nil {
				return Spec{}, err
			}
			spec.Methods = append(spec.Methods, Method{Name: m.Names[0].Name, Params: params, Results: results})
		}
		return Canonicalize(spec)
	}
	return Spec{}, fmt.Errorf("unsupported type declaration %T", ts.Type)
}

func extractFuncDecl(fd *ast.FuncDecl) (Spec, error) {
	spec := Spec{Kind: KindFunc, Name: fd.Name.Name}
	if fd.Recv != nil && len(fd.Recv.List) > 0 {
		r := fd.Recv.List[0]
		expr, err := exprString(r.Type)
		if err != nil {
			return Spec{}, err
		}
		recv := &Receiver{Type: expr}
		if strings.HasPrefix(expr, "*") {
			recv.Pointer = true
			recv.Type = strings.TrimPrefix(expr, "*")
		}
		if len(r.Names) > 0 {
			recv.Name = r.Names[0].Name
		}
		spec.Receiver = recv
	}
	params, results, err := extractSignature(fd.Type)
	if err != nil {
		return Spec{}, err
	}
	spec.Params = params
	spec.Results = results
	return Canonicalize(spec)
}

// extractSignature pulls params and results out of a FuncType. Grouped
// parameters (a, b int) expand to one Param per name; unnamed results
// are the only result form in this pipeline.
func extractSignature(ft *ast.FuncType) ([]Param, []TypeRef, error) {
	var params []Param
	if ft.Params != nil {
		for _, f := range ft.Params.List {
			expr, err := exprString(f.Type)
			if err != nil {
				return nil, nil, err
			}
			ref, err := MakeTypeRef(expr)
			if err != nil {
				return nil, nil, err
			}
			if len(f.Names) == 0 {
				params = append(params, Param{Type: ref})
				continue
			}
			for _, n := range f.Names {
				params = append(params, Param{Name: n.Name, Type: ref})
			}
		}
	}
	var results []TypeRef
	if ft.Results != nil {
		for _, f := range ft.Results.List {
			expr, err := exprString(f.Type)
			if err != nil {
				return nil, nil, err
			}
			ref, err := MakeTypeRef(expr)
			if err != nil {
				return nil, nil, err
			}
			count := len(f.Names)
			if count == 0 {
				count = 1
			}
			for i := 0; i < count; i++ {
				results = append(results, ref)
			}
		}
	}
	return params, results, nil
}

// exprString prints an AST expression in canonical form.
func exprString(e ast.Expr) (string, error) {
	var b strings.Builder
	if err := format.Node(&b, token.NewFileSet(), e); err != nil {
		return "", err
	}
	return b.String(), nil
}
