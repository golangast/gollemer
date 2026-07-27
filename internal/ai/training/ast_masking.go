// Package training provides AST-aware masking for Fill-In-The-Middle (FIM) training.
// Instead of masking random contiguous spans of text, this masks entire AST nodes
// (e.g., an entire if block, function parameters list, or return type signature),
// forcing the model to think in logical syntax trees rather than random string chunks.
package training

import (
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"math/rand"
	"strings"
)

// ASTNodeKind represents the type of AST node that can be masked.
type ASTNodeKind string

const (
	MaskIfBlock       ASTNodeKind = "if_block"
	MaskForBlock      ASTNodeKind = "for_block"
	MaskSwitchBlock   ASTNodeKind = "switch_block"
	MaskSelectBlock   ASTNodeKind = "select_block"
	MaskDeferStmt     ASTNodeKind = "defer_stmt"
	MaskGoStmt        ASTNodeKind = "go_stmt"
	MaskReturnStmt    ASTNodeKind = "return_stmt"
	MaskAssignStmt    ASTNodeKind = "assign_stmt"
	MaskExprStmt      ASTNodeKind = "expr_stmt"
	MaskFuncParams    ASTNodeKind = "func_params"
	MaskFuncResults   ASTNodeKind = "func_results"
	MaskFuncBody      ASTNodeKind = "func_body"
	MaskStructType    ASTNodeKind = "struct_type"
	MaskInterfaceType ASTNodeKind = "interface_type"
	MaskCallExpr      ASTNodeKind = "call_expr"
	MaskBinaryExpr    ASTNodeKind = "binary_expr"
	MaskUnaryExpr     ASTNodeKind = "unary_expr"
	MaskCompositeLit  ASTNodeKind = "composite_lit"
	MaskFuncLit       ASTNodeKind = "func_lit"
	MaskTypeAssert    ASTNodeKind = "type_assert"
	MaskSendStmt      ASTNodeKind = "send_stmt"
	MaskIncDecStmt    ASTNodeKind = "inc_dec_stmt"
	MaskLabeledStmt   ASTNodeKind = "labeled_stmt"
	MaskBranchStmt    ASTNodeKind = "branch_stmt" // break, continue, goto, fallthrough
	MaskRangeClause   ASTNodeKind = "range_clause"
	MaskTypeSwitch    ASTNodeKind = "type_switch"
	MaskCommClause    ASTNodeKind = "comm_clause" // select communication clause
)

// MaskedNode represents a single masked AST node in the code.
type MaskedNode struct {
	Kind     ASTNodeKind `json:"kind"`
	Original string      `json:"original"` // The original code that was masked
	StartPos int         `json:"start_pos"`
	EndPos   int         `json:"end_pos"`
}

// ASTMaskingConfig configures the AST masking behavior.
type ASTMaskingConfig struct {
	// MaskProbability is the probability of masking a given eligible node (0.0-1.0)
	MaskProbability float64
	// MaxMaskedNodes is the maximum number of nodes to mask in a single example
	MaxMaskedNodes int
	// MinNodeSize is the minimum character length of a node to be eligible for masking
	MinNodeSize int
	// MaxNodeSize is the maximum character length of a node to be eligible for masking
	MaxNodeSize int
	// NodeWeights maps node kinds to their relative masking probability weights
	NodeWeights map[ASTNodeKind]float64
	// PreserveTopLevel if true, prevents masking top-level declarations
	PreserveTopLevel bool
}

// DefaultASTMaskingConfig returns sensible defaults for AST masking.
func DefaultASTMaskingConfig() ASTMaskingConfig {
	return ASTMaskingConfig{
		MaskProbability:  0.3,
		MaxMaskedNodes:   3,
		MinNodeSize:      5,
		MaxNodeSize:      500,
		PreserveTopLevel: true,
		NodeWeights: map[ASTNodeKind]float64{
			MaskIfBlock:      1.0,
			MaskForBlock:     1.0,
			MaskReturnStmt:   0.8,
			MaskAssignStmt:   0.6,
			MaskExprStmt:     0.4,
			MaskFuncParams:   0.7,
			MaskFuncResults:  0.7,
			MaskCallExpr:     0.5,
			MaskBinaryExpr:   0.3,
			MaskDeferStmt:    0.5,
			MaskGoStmt:       0.5,
			MaskCompositeLit: 0.4,
			MaskFuncLit:      0.6,
			MaskRangeClause:  0.6,
			MaskTypeAssert:   0.4,
			MaskSendStmt:     0.5,
			MaskBranchStmt:   0.3,
			MaskIncDecStmt:   0.2,
			MaskLabeledStmt:  0.3,
			MaskTypeSwitch:   0.7,
			MaskCommClause:   0.6,
			MaskSelectBlock:  0.8,
			MaskSwitchBlock:  0.7,
		},
	}
}

// ASTMasker performs AST-aware masking on Go source code.
type ASTMasker struct {
	config ASTMaskingConfig
	rng    *rand.Rand
}

// NewASTMasker creates a new AST masker with the given configuration.
func NewASTMasker(config ASTMaskingConfig) *ASTMasker {
	return &ASTMasker{
		config: config,
		rng:    rand.New(rand.NewSource(42)), // Fixed seed for reproducibility
	}
}

// MaskResult contains the masked code and information about what was masked.
type MaskResult struct {
	MaskedCode   string       `json:"masked_code"`
	OriginalCode string       `json:"original_code"`
	MaskedNodes  []MaskedNode `json:"masked_nodes"`
	Prefix       string       `json:"prefix"`
	Middle       string       `json:"middle"`
	Suffix       string       `json:"suffix"`
}

// MaskASTNodes masks eligible AST nodes in the given Go source code.
// It returns the masked code and a list of what was masked.
func (m *ASTMasker) MaskASTNodes(code string) (*MaskResult, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// Collect all eligible nodes
	var candidates []struct {
		kind     ASTNodeKind
		node     ast.Node
		original string
		start    int
		end      int
	}

	ast.Inspect(f, func(n ast.Node) bool {
		if n == nil {
			return true
		}

		start := fset.Position(n.Pos()).Offset
		end := fset.Position(n.End()).Offset
		size := end - start

		// Skip nodes that are too small or too large
		if size < m.config.MinNodeSize || size > m.config.MaxNodeSize {
			return true
		}

		// Skip top-level declarations if configured
		if m.config.PreserveTopLevel {
			switch n.(type) {
			case *ast.File, *ast.GenDecl, *ast.FuncDecl:
				return true
			}
		}

		var kind ASTNodeKind
		switch n.(type) {
		case *ast.IfStmt:
			kind = MaskIfBlock
		case *ast.ForStmt, *ast.RangeStmt:
			kind = MaskForBlock
		case *ast.SwitchStmt:
			kind = MaskSwitchBlock
		case *ast.TypeSwitchStmt:
			kind = MaskTypeSwitch
		case *ast.SelectStmt:
			kind = MaskSelectBlock
		case *ast.DeferStmt:
			kind = MaskDeferStmt
		case *ast.GoStmt:
			kind = MaskGoStmt
		case *ast.ReturnStmt:
			kind = MaskReturnStmt
		case *ast.AssignStmt:
			kind = MaskAssignStmt
		case *ast.ExprStmt:
			kind = MaskExprStmt
		case *ast.CallExpr:
			kind = MaskCallExpr
		case *ast.BinaryExpr:
			kind = MaskBinaryExpr
		case *ast.CompositeLit:
			kind = MaskCompositeLit
		case *ast.FuncLit:
			kind = MaskFuncLit
		case *ast.SendStmt:
			kind = MaskSendStmt
		case *ast.IncDecStmt:
			kind = MaskIncDecStmt
		case *ast.BranchStmt:
			kind = MaskBranchStmt
		case *ast.CommClause:
			kind = MaskCommClause
		default:
			return true
		}

		// Get the original source text
		original := code[start:end]

		// Check weight-based eligibility
		weight, hasWeight := m.config.NodeWeights[kind]
		if !hasWeight {
			weight = 0.5 // Default weight
		}

		candidates = append(candidates, struct {
			kind     ASTNodeKind
			node     ast.Node
			original string
			start    int
			end      int
		}{
			kind:     kind,
			node:     n,
			original: original,
			start:    start,
			end:      end,
		})

		// Don't descend into masked nodes to avoid double-masking
		if m.rng.Float64() < weight*m.config.MaskProbability {
			return false
		}
		return true
	})

	// Select nodes to mask based on weights
	m.maskNodes(candidates, &code)

	// Build the result
	result := &MaskResult{
		OriginalCode: code,
		MaskedCode:   code,
	}

	return result, nil
}

// maskNodes selects and masks nodes from the candidates list.
func (m *ASTMasker) maskNodes(candidates []struct {
	kind     ASTNodeKind
	node     ast.Node
	original string
	start    int
	end      int
}, code *string) {
	// Sort candidates by position (ascending)
	// Process in reverse order to preserve positions
	for i := len(candidates) - 1; i >= 0; i-- {
		c := candidates[i]

		// Weighted random selection
		weight := m.config.NodeWeights[c.kind]
		if weight == 0 {
			weight = 0.5
		}

		if m.rng.Float64() < weight*m.config.MaskProbability {
			// Replace the node with a mask token
			maskToken := fmt.Sprintf("<MASK:%s>", string(c.kind))
			*code = (*code)[:c.start] + maskToken + (*code)[c.end:]
		}
	}
}

// CreateFIMMask creates a FIM example by masking an AST node and splitting
// the code into prefix/middle/suffix around the masked region.
func (m *ASTMasker) CreateFIMMask(code string) (*FIMExample, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// Collect eligible nodes for masking
	type nodeInfo struct {
		kind     ASTNodeKind
		node     ast.Node
		original string
		start    int
		end      int
	}

	var candidates []nodeInfo

	ast.Inspect(f, func(n ast.Node) bool {
		if n == nil {
			return true
		}

		start := fset.Position(n.Pos()).Offset
		end := fset.Position(n.End()).Offset
		size := end - start

		if size < m.config.MinNodeSize || size > m.config.MaxNodeSize {
			return true
		}

		if m.config.PreserveTopLevel {
			switch n.(type) {
			case *ast.File, *ast.GenDecl, *ast.FuncDecl:
				return true
			}
		}

		var kind ASTNodeKind
		switch n.(type) {
		case *ast.IfStmt:
			kind = MaskIfBlock
		case *ast.ForStmt, *ast.RangeStmt:
			kind = MaskForBlock
		case *ast.SwitchStmt:
			kind = MaskSwitchBlock
		case *ast.TypeSwitchStmt:
			kind = MaskTypeSwitch
		case *ast.SelectStmt:
			kind = MaskSelectBlock
		case *ast.DeferStmt:
			kind = MaskDeferStmt
		case *ast.GoStmt:
			kind = MaskGoStmt
		case *ast.ReturnStmt:
			kind = MaskReturnStmt
		case *ast.AssignStmt:
			kind = MaskAssignStmt
		case *ast.ExprStmt:
			kind = MaskExprStmt
		case *ast.CallExpr:
			kind = MaskCallExpr
		case *ast.BinaryExpr:
			kind = MaskBinaryExpr
		case *ast.CompositeLit:
			kind = MaskCompositeLit
		case *ast.FuncLit:
			kind = MaskFuncLit
		case *ast.SendStmt:
			kind = MaskSendStmt
		case *ast.IncDecStmt:
			kind = MaskIncDecStmt
		case *ast.BranchStmt:
			kind = MaskBranchStmt
		case *ast.CommClause:
			kind = MaskCommClause
		default:
			return true
		}

		original := code[start:end]
		candidates = append(candidates, nodeInfo{
			kind:     kind,
			node:     n,
			original: original,
			start:    start,
			end:      end,
		})

		return false // Don't descend into candidates
	})

	if len(candidates) == 0 {
		return nil, fmt.Errorf("no eligible AST nodes found for masking")
	}

	// Select one node to mask (the "middle" of the FIM example)
	selected := candidates[m.rng.Intn(len(candidates))]

	// Split code into prefix, middle (masked), suffix
	prefix := code[:selected.start]
	middle := selected.original
	suffix := code[selected.end:]

	return &FIMExample{
		Prefix: prefix,
		Middle: middle,
		Suffix: suffix,
	}, nil
}

// CreateMaskFuncParams creates a FIM example that masks a function's parameter list.
func CreateMaskFuncParams(code string) (*FIMExample, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// Find a function with parameters
	var targetFunc *ast.FuncDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Type.Params != nil && len(fn.Type.Params.List) > 0 {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		return nil, fmt.Errorf("no function with parameters found")
	}

	paramsStart := fset.Position(targetFunc.Type.Params.Opening).Offset
	paramsEnd := fset.Position(targetFunc.Type.Params.Closing).Offset + 1

	prefix := code[:paramsStart]
	middle := code[paramsStart:paramsEnd]
	suffix := code[paramsEnd:]

	return &FIMExample{
		Prefix: prefix,
		Middle: middle,
		Suffix: suffix,
	}, nil
}

// CreateMaskFuncResults creates a FIM example that masks a function's return type(s).
func CreateMaskFuncResults(code string) (*FIMExample, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// Find a function with return types
	var targetFunc *ast.FuncDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Type.Results != nil && len(fn.Type.Results.List) > 0 {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		return nil, fmt.Errorf("no function with return types found")
	}

	resultsStart := fset.Position(targetFunc.Type.Results.Opening).Offset
	resultsEnd := fset.Position(targetFunc.Type.Results.Closing).Offset + 1

	prefix := code[:resultsStart]
	middle := code[resultsStart:resultsEnd]
	suffix := code[resultsEnd:]

	return &FIMExample{
		Prefix: prefix,
		Middle: middle,
		Suffix: suffix,
	}, nil
}

// CreateMaskIfBlock creates a FIM example that masks an entire if/else block.
func CreateMaskIfBlock(code string) (*FIMExample, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// Find an if statement
	var targetIf *ast.IfStmt
	ast.Inspect(f, func(n ast.Node) bool {
		if ifStmt, ok := n.(*ast.IfStmt); ok {
			targetIf = ifStmt
			return false
		}
		return true
	})

	if targetIf == nil {
		return nil, fmt.Errorf("no if statement found")
	}

	ifStart := fset.Position(targetIf.Pos()).Offset
	ifEnd := fset.Position(targetIf.End()).Offset

	prefix := code[:ifStart]
	middle := code[ifStart:ifEnd]
	suffix := code[ifEnd:]

	return &FIMExample{
		Prefix: prefix,
		Middle: middle,
		Suffix: suffix,
	}, nil
}

// CreateMaskReturnType creates a FIM example that masks just the return type expression.
func CreateMaskReturnType(code string) (*FIMExample, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", code, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	// Find a function with a single return type
	var targetFunc *ast.FuncDecl
	ast.Inspect(f, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Type.Results != nil && len(fn.Type.Results.List) == 1 {
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		return nil, fmt.Errorf("no function with single return type found")
	}

	// Get the return type expression
	retType := targetFunc.Type.Results.List[0].Type
	retStart := fset.Position(retType.Pos()).Offset
	retEnd := fset.Position(retType.End()).Offset

	prefix := code[:retStart]
	middle := code[retStart:retEnd]
	suffix := code[retEnd:]

	return &FIMExample{
		Prefix: prefix,
		Middle: middle,
		Suffix: suffix,
	}, nil
}

// GenerateASTMaskedFIMExamples generates multiple FIM examples from a single
// Go source file by masking different AST nodes.
func GenerateASTMaskedFIMExamples(code string, maxExamples int) ([]FIMExample, error) {
	masker := NewASTMasker(DefaultASTMaskingConfig())
	var examples []FIMExample

	// Try different masking strategies
	strategies := []func(string) (*FIMExample, error){
		masker.CreateFIMMask,
		CreateMaskFuncParams,
		CreateMaskFuncResults,
		CreateMaskIfBlock,
		CreateMaskReturnType,
	}

	for _, strategy := range strategies {
		if len(examples) >= maxExamples {
			break
		}
		ex, err := strategy(code)
		if err != nil {
			continue
		}
		if ex != nil && ex.Middle != "" {
			examples = append(examples, *ex)
		}
	}

	return examples, nil
}

// nodeToSource converts an AST node back to source code string.
func nodeToSource(node ast.Node) string {
	if node == nil {
		return ""
	}
	fset := token.NewFileSet()
	var buf strings.Builder
	if err := format.Node(&buf, fset, node); err != nil {
		return ""
	}
	return buf.String()
}

// exprString is a helper to convert an expression to a string.
func exprString(expr ast.Expr) string {
	if expr == nil {
		return ""
	}
	return types.ExprString(expr)
}
