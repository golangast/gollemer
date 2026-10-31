package analyze

// Model-proposed declarations: the one place a trained model
// gets to suggest a write. The chat's speccopy proposer (a
// small pointer-generator trained to turn a sentence into a
// declaration spec) fires only where the concept book's grammar
// missed a declaration-shaped ask; the chat parses the model's
// spec text, restores its case through the decl registry's
// pool maps, and renders it deterministically — and THAT text
// is what arrives here, as data, exactly like a concept-book
// row's rendered template. This planner assembles it the way
// planConceptDeclaration assembles a book row: appended at
// file scope, imports the types lean on added from the
// sentence's own facts, the name checked against the package,
// the result self-checked by re-parsing, and the spine's gate
// + build + verify still ahead of it. The model proposes; it
// never writes, and a proposal whose name already exists (or
// whose text fails the assembler's own checks) is a refusal
// like any other planner's.

import (
	"fmt"
	"go/ast"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
)

// proposedNote is the plan's honesty line, printed before the
// diff and the gate: a beginner must be able to tell a model
// guess from an assembled piece at a glance.
const proposedNote = "This declaration was proposed by my small trained model from your sentence — a guess, not one of the concept book's assembled pieces, and the model is often wrong. Read the names and types in the diff before you approve; nothing is written unless you say yes, and the project still has to build."

// planProposedDecl assembles one model-proposed declaration
// into req.File.
func (p *Project) planProposedDecl(req EditRequest) (*EditPlan, string) {
	text := strings.TrimSpace(req.ProposedText)
	if text == "" {
		return nil, "that edit needs a declaration to add, and none was proposed."
	}
	rel := strings.TrimSpace(req.File)
	if rel == "" {
		return nil, "that edit needs a file to add the declaration to."
	}
	name := strings.TrimSpace(req.ProposedName)
	if name == "" {
		return nil, "the proposed declaration has no name to add."
	}
	if why, blocked := p.fileBlocked(rel); blocked {
		return nil, fmt.Sprintf("I won't edit %s — %s.", p.LinkPath(rel), why)
	}
	src, fset, fileAST, refusal := p.readProjectFile(rel)
	if refusal != "" {
		return nil, refusal
	}
	if fileBytesBlocked(src) {
		return nil, fmt.Sprintf("I won't edit %s — it is a generated file (marked DO NOT EDIT).", p.LinkPath(rel))
	}
	if !token.IsIdentifier(name) {
		return nil, fmt.Sprintf("%q isn't a name Go allows for a declaration.", name)
	}
	receiver := strings.TrimSpace(req.ProposedReceiver)
	// A method's name lives on its receiver, so only types and
	// plain functions collide with the package's own names.
	if receiver == "" {
		if pkg := p.pkgOfFile(rel); pkg != nil && p.pkgDeclares(pkg, name) {
			return nil, fmt.Sprintf("%s already exists in package %s (%s) — pick a different name for the declaration.", name, pkg.Name, p.LinkPath(rel))
		}
	}

	// Imports the declaration's types lean on, added the
	// concept family's way: only the ones the file lacks.
	var specs []string
	for _, path := range strings.Fields(req.ProposedImports) {
		specs = append(specs, strconv.Quote(path))
	}
	specs = missingImportSpecs(fileAST, specs)

	// Imports first, the declaration appended after — the same
	// coincident-offset care as the concept declaration pass.
	base := applyTextEdits(src, conceptImportEdits(fileAST, fset, src, specs))
	appendix := "\n" + text + "\n"
	if len(base) == 0 || base[len(base)-1] != '\n' {
		appendix = "\n" + appendix
	}
	after := append(append([]byte{}, base...), appendix...)
	ef := finishFile(rel, src, after)

	// AST self-check: the assembled file parses and now
	// declares the proposed name, in the proposed form.
	if bad := selfCheckProposed(ef.After, name, receiver); bad != "" {
		return nil, fmt.Sprintf("internal error in my assembler: %s — I haven't shown you a plan, because the code I generated failed my own checks. Nothing was written.", bad)
	}

	kindWord := proposedKindWord(req.ProposedKind, receiver)
	firstLine := strings.SplitN(text, "\n", 2)[0]
	plan := &EditPlan{
		Summary: fmt.Sprintf("Add the %s %s to %s, as proposed by the model.", kindWord, name, p.LinkPath(rel)),
		Site:    fmt.Sprintf("%s:%d", p.LinkPath(rel), insertedLine(ef.After, firstLine)),
		Files:   []EditFile{ef},
		Notes:   []string{proposedNote},
		verify: func(after *Project) bool {
			for _, pk := range after.Packages {
				for _, t := range pk.Types {
					if receiver == "" && t.Name == name && t.File == rel {
						return true
					}
				}
				for _, fn := range pk.Funcs {
					if fn.Name == name && fn.File == rel &&
						strings.TrimPrefix(fn.Receiver, "*") == receiver {
						return true
					}
				}
			}
			return false
		},
	}
	if receiver != "" {
		plan.VerifyDesc = fmt.Sprintf("%s now has the method %s on %s", filepath.Base(rel), name, receiver)
	} else {
		plan.VerifyDesc = fmt.Sprintf("%s now declares the %s %s", filepath.Base(rel), kindWord, name)
	}
	return plan, ""
}

// proposedKindWord names the declaration's kind the way a
// beginner would say it: a receiver-bearing func is a method.
func proposedKindWord(kind, receiver string) string {
	if receiver != "" {
		return "method"
	}
	if kind == "func" {
		return "function"
	}
	return kind
}

// selfCheckProposed re-parses the assembled file and confirms
// it declares the proposed name — a top-level type or plain
// function, or a method on the proposed receiver.
func selfCheckProposed(after []byte, name, receiver string) string {
	_, fileAST, err := parseGoSrc("proposed.go", after)
	if err != nil || fileAST == nil {
		return "the assembled file doesn't parse"
	}
	for _, d := range fileAST.Decls {
		switch d := d.(type) {
		case *ast.GenDecl:
			if receiver != "" || d.Tok != token.TYPE {
				continue
			}
			for _, spec := range d.Specs {
				if ts, ok := spec.(*ast.TypeSpec); ok && ts.Name.Name == name {
					return ""
				}
			}
		case *ast.FuncDecl:
			if d.Name.Name != name {
				continue
			}
			if receiver == "" && d.Recv == nil {
				return ""
			}
			if receiver != "" && d.Recv != nil && len(d.Recv.List) > 0 &&
				strings.TrimPrefix(typeString(d.Recv.List[0].Type), "*") == receiver {
				return ""
			}
		}
	}
	return fmt.Sprintf("the assembled file doesn't declare %s", name)
}
