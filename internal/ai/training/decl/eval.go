package decl

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"sort"
	"strings"
)

// Eval scores model generations against held-out specs STRUCTURALLY:
// the generation is parsed, its spec extracted from the AST, and the
// two specs compared slot by slot. String equality is never used.
// A separate go/types check (against a stub universe where every
// referenced local type exists and imports are the real stdlib)
// measures whether the restored, case-corrected declaration compiles
// standalone.

// Taxonomy categories, in the order they are checked (first match
// is the recorded reason).
const (
	CatExact            = "exact"
	CatUnparsable       = "unparsable"
	CatWrongKind        = "wrong_kind"
	CatWrongName        = "wrong_name"
	CatReceiverMismatch = "receiver_mismatch"
	CatMemberCount      = "member_count"
	CatMemberName       = "member_name"
	CatWrongType        = "wrong_type"
	CatWrongOrder       = "wrong_order"
	CatTagMismatch      = "tag_mismatch"
	CatEmbedMismatch    = "embed_mismatch"
)

// ItemResult is the scored outcome of one eval item.
type ItemResult struct {
	ID        string `json:"id"`
	InputForm string `json:"input_form"`
	Combo     string `json:"combo"`
	Frame     string `json:"frame"`
	FrameID   string `json:"frame_id,omitempty"`
	Category  string `json:"category"`
	Detail    string `json:"detail,omitempty"`
	Parsed    bool   `json:"parsed"`
	Compiles  bool   `json:"compiles"`
	RawOutput string `json:"raw_output"`
}

// CellStats aggregates one matrix cell.
type CellStats struct {
	N        int            `json:"n"`
	Exact    int            `json:"exact"`
	Parsed   int            `json:"parsed"`
	Compiles int            `json:"compiles"`
	Taxonomy map[string]int `json:"taxonomy"`
}

// EvalReport is the full scoring output.
type EvalReport struct {
	Cells    map[string]*CellStats `json:"cells"`
	Taxonomy map[string]int        `json:"taxonomy"`
	Results  []ItemResult          `json:"results"`
}

// PredictFunc generates text for an input (the trained model).
type PredictFunc func(input string) (string, error)

// RunEval scores every item with the given predictor.
func RunEval(items []EvalItem, predict PredictFunc) *EvalReport {
	rep := &EvalReport{
		Cells:    map[string]*CellStats{},
		Taxonomy: map[string]int{},
	}
	for _, item := range items {
		res := ItemResult{ID: item.ID, InputForm: item.InputForm, Combo: item.Combo, Frame: item.Frame, FrameID: item.FrameID}
		raw, err := predict(item.Input)
		if err != nil {
			res.Category = CatUnparsable
			res.Detail = "predict error: " + err.Error()
		} else {
			res.RawOutput = raw
			scoreOne(item, raw, &res)
		}
		key := item.InputForm + "|" + item.Combo + "|" + item.Frame
		cell := rep.Cells[key]
		if cell == nil {
			cell = &CellStats{Taxonomy: map[string]int{}}
			rep.Cells[key] = cell
		}
		cell.N++
		if res.Parsed {
			cell.Parsed++
		}
		if res.Compiles {
			cell.Compiles++
		}
		if res.Category == CatExact {
			cell.Exact++
		}
		cell.Taxonomy[res.Category]++
		rep.Taxonomy[res.Category]++
		rep.Results = append(rep.Results, res)
	}
	return rep
}

func scoreOne(item EvalItem, raw string, res *ItemResult) {
	got, err := ExtractSpec(raw)
	if err != nil {
		res.Category = CatUnparsable
		res.Detail = err.Error()
		return
	}
	res.Parsed = true
	res.Compiles, res.Detail = checkCompiles(got)
	if res.Detail != "" && !res.Compiles {
		res.Detail = "compile: " + res.Detail
	} else {
		res.Detail = ""
	}
	// The receiver's NAME reaches the model only through the spec
	// form and the signature frames; phrasing frames name only the
	// receiver type, so the name is scored only where it was given.
	strict := item.InputForm == "spec" || item.FrameID == "M-T1"
	cat, detail := CompareSpec(got, LowerSpec(item.Spec), strict)
	res.Category = cat
	if detail != "" {
		if res.Detail != "" {
			res.Detail += "; "
		}
		res.Detail += detail
	}
}

// CompareSpec structurally compares an extracted (lowercase) spec
// against the wanted (lowercase) spec, returning the first taxonomy
// category that fails, or CatExact.
func CompareSpec(got, want Spec, strictReceiverName bool) (string, string) {
	if got.Kind != want.Kind {
		return CatWrongKind, fmt.Sprintf("got %s want %s", got.Kind, want.Kind)
	}
	if got.Name != want.Name {
		return CatWrongName, fmt.Sprintf("got %q want %q", got.Name, want.Name)
	}
	switch want.Kind {
	case KindStruct:
		return compareFields(got, want)
	case KindInterface:
		return compareInterface(got, want)
	case KindFunc:
		if cat, det := compareReceiver(got, want, strictReceiverName); cat != CatExact {
			return cat, det
		}
		if cat, det := compareParams(got.Params, want.Params); cat != CatExact {
			return cat, det
		}
		return compareResults(got.Results, want.Results)
	}
	return CatExact, ""
}

func compareReceiver(got, want Spec, strictName bool) (string, string) {
	gr, wr := got.Receiver, want.Receiver
	if (gr == nil) != (wr == nil) {
		return CatReceiverMismatch, "receiver presence differs"
	}
	if wr == nil {
		return CatExact, ""
	}
	if gr.Type != wr.Type || gr.Pointer != wr.Pointer {
		return CatReceiverMismatch, fmt.Sprintf("got (%s,%v) want (%s,%v)", gr.Type, gr.Pointer, wr.Type, wr.Pointer)
	}
	if strictName && gr.Name != wr.Name {
		return CatReceiverMismatch, fmt.Sprintf("receiver name got %q want %q", gr.Name, wr.Name)
	}
	return CatExact, ""
}

// memberKey identifies a struct field in sequence comparisons.
func memberKey(f Field) string {
	if f.Embedded {
		return "embed:" + f.Type.Expr
	}
	return "field:" + f.Name
}

func compareFields(got, want Spec) (string, string) {
	if len(got.Fields) != len(want.Fields) {
		return CatMemberCount, fmt.Sprintf("fields got %d want %d", len(got.Fields), len(want.Fields))
	}
	gotKeys := make([]string, len(got.Fields))
	wantKeys := make([]string, len(want.Fields))
	for i := range got.Fields {
		gotKeys[i] = memberKey(got.Fields[i])
		wantKeys[i] = memberKey(want.Fields[i])
	}
	if !equalStrings(gotKeys, wantKeys) {
		if sameMultiset(gotKeys, wantKeys) {
			return CatWrongOrder, "fields out of order"
		}
		return CatMemberName, fmt.Sprintf("got %v want %v", gotKeys, wantKeys)
	}
	for i := range got.Fields {
		if got.Fields[i].Type.Expr != want.Fields[i].Type.Expr {
			return CatWrongType, fmt.Sprintf("field %s: got %s want %s", wantKeys[i], got.Fields[i].Type.Expr, want.Fields[i].Type.Expr)
		}
	}
	for i := range got.Fields {
		if NormalizeTag(got.Fields[i].Tag) != NormalizeTag(want.Fields[i].Tag) {
			return CatTagMismatch, fmt.Sprintf("field %s: got tag %q want %q", wantKeys[i], got.Fields[i].Tag, want.Fields[i].Tag)
		}
	}
	return CatExact, ""
}

func compareInterface(got, want Spec) (string, string) {
	gotEmbeds := make([]string, len(got.Embeds))
	wantEmbeds := make([]string, len(want.Embeds))
	for i := range got.Embeds {
		gotEmbeds[i] = got.Embeds[i].Expr
	}
	for i := range want.Embeds {
		wantEmbeds[i] = want.Embeds[i].Expr
	}
	if !equalStrings(gotEmbeds, wantEmbeds) {
		return CatEmbedMismatch, fmt.Sprintf("embeds got %v want %v", gotEmbeds, wantEmbeds)
	}
	if len(got.Methods) != len(want.Methods) {
		return CatMemberCount, fmt.Sprintf("methods got %d want %d", len(got.Methods), len(want.Methods))
	}
	gotNames := make([]string, len(got.Methods))
	wantNames := make([]string, len(want.Methods))
	for i := range got.Methods {
		gotNames[i] = got.Methods[i].Name
		wantNames[i] = want.Methods[i].Name
	}
	if !equalStrings(gotNames, wantNames) {
		if sameMultiset(gotNames, wantNames) {
			return CatWrongOrder, "methods out of order"
		}
		return CatMemberName, fmt.Sprintf("methods got %v want %v", gotNames, wantNames)
	}
	for i := range got.Methods {
		if cat, det := compareParams(got.Methods[i].Params, want.Methods[i].Params); cat != CatExact {
			return cat, "method " + wantNames[i] + ": " + det
		}
		if cat, det := compareResults(got.Methods[i].Results, want.Methods[i].Results); cat != CatExact {
			return cat, "method " + wantNames[i] + ": " + det
		}
	}
	return CatExact, ""
}

func compareParams(got, want []Param) (string, string) {
	if len(got) != len(want) {
		return CatMemberCount, fmt.Sprintf("params got %d want %d", len(got), len(want))
	}
	gotNames := make([]string, len(got))
	wantNames := make([]string, len(want))
	for i := range got {
		gotNames[i] = got[i].Name
		wantNames[i] = want[i].Name
	}
	if !equalStrings(gotNames, wantNames) {
		if sameMultiset(gotNames, wantNames) {
			return CatWrongOrder, "params out of order"
		}
		return CatMemberName, fmt.Sprintf("param names got %v want %v", gotNames, wantNames)
	}
	for i := range got {
		if got[i].Type.Expr != want[i].Type.Expr {
			return CatWrongType, fmt.Sprintf("param %d: got %s want %s", i, got[i].Type.Expr, want[i].Type.Expr)
		}
	}
	return CatExact, ""
}

func compareResults(got, want []TypeRef) (string, string) {
	if len(got) != len(want) {
		return CatMemberCount, fmt.Sprintf("results got %d want %d", len(got), len(want))
	}
	gotExprs := make([]string, len(got))
	wantExprs := make([]string, len(want))
	for i := range got {
		gotExprs[i] = got[i].Expr
		wantExprs[i] = want[i].Expr
	}
	if !equalStrings(gotExprs, wantExprs) {
		if sameMultiset(gotExprs, wantExprs) {
			return CatWrongOrder, "results out of order"
		}
		return CatWrongType, fmt.Sprintf("results got %v want %v", gotExprs, wantExprs)
	}
	return CatExact, ""
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sameMultiset(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	as := append([]string{}, a...)
	bs := append([]string{}, b...)
	sort.Strings(as)
	sort.Strings(bs)
	return equalStrings(as, bs)
}

// checkCompiles type-checks the extracted (lowercase) spec's
// declaration: case is restored from the pools, the declaration is
// rendered, and it is checked in a stub universe — every referenced
// local type exists as an empty declaration (interfaces where the
// spec embeds them into an interface, generic declarations for the
// generic locals), imports are the real standard library.
func checkCompiles(gotLower Spec) (bool, string) {
	restored, ok := RestoreSpec(gotLower)
	if !ok {
		return false, "unrestorable atom"
	}
	declText := RenderDecl(restored)

	imports := map[string]bool{}
	stubKind := map[string]string{} // name -> "struct" | "interface" | generic decl
	note := func(expr string, asInterface bool) {
		for _, p := range ImportPaths(expr) {
			imports[p] = true
		}
		for _, ln := range LocalNames(expr) {
			if ln == restored.Name {
				continue
			}
			if _, isGeneric := GenericLocals[ln]; isGeneric {
				stubKind[ln] = "generic"
				continue
			}
			if asInterface {
				stubKind[ln] = "interface"
			} else if stubKind[ln] == "" {
				stubKind[ln] = "struct"
			}
		}
	}
	for _, ref := range allTypeRefs(restored) {
		note(ref.Expr, false)
	}
	for _, e := range restored.Embeds {
		note(e.Expr, restored.Kind == KindInterface)
	}
	if restored.Receiver != nil {
		stubKind[restored.Receiver.Type] = "struct"
	}

	var b strings.Builder
	b.WriteString("package declcheck\n")
	if len(imports) > 0 {
		b.WriteString("import (\n")
		paths := make([]string, 0, len(imports))
		for p := range imports {
			paths = append(paths, p)
		}
		sort.Strings(paths)
		for _, p := range paths {
			fmt.Fprintf(&b, "\t%q\n", p)
		}
		b.WriteString(")\n")
	}
	names := make([]string, 0, len(stubKind))
	for n := range stubKind {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		switch stubKind[n] {
		case "generic":
			b.WriteString(GenericLocals[n] + "\n")
		case "interface":
			b.WriteString("type " + n + " interface{}\n")
		default:
			b.WriteString("type " + n + " struct{}\n")
		}
	}
	b.WriteString(declText + "\n")
	// Referenced imports must be used; an extracted decl always uses
	// what it references, but a stub-only import would error. Imports
	// were collected from the decl's own expressions, so each is used
	// by construction.
	src := b.String()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "decl.go", src, parser.AllErrors)
	if err != nil {
		return false, "parse: " + err.Error()
	}
	conf := types.Config{Importer: importer.Default(), FakeImportC: true}
	if _, err := conf.Check("declcheck", fset, []*ast.File{file}, nil); err != nil {
		return false, err.Error()
	}
	return true, ""
}
