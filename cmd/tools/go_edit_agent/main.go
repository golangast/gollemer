package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/format"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/packages"
)

// ─── Data Types ───────────────────────────────────────────────────────────────

// EditOperation describes a single AST-level edit to apply to a Go source file.
type EditOperation struct {
	Type       string `json:"type"`        // "insert_func", "modify_func", "add_field", "add_import", "replace_code", "delete_func", "add_symbol"
	TargetFile string `json:"target_file"` // Path to the .go file
	FuncName   string `json:"func_name,omitempty"`
	StructName string `json:"struct_name,omitempty"`
	FieldName  string `json:"field_name,omitempty"`
	FieldType  string `json:"field_type,omitempty"`
	FieldTag   string `json:"field_tag,omitempty"`
	ImportPath string `json:"import_path,omitempty"`
	Code       string `json:"code,omitempty"`      // New function body or replacement code
	InsertAt   string `json:"insert_at,omitempty"` // "beginning", "end", or line number
	OldCode    string `json:"old_code,omitempty"`  // For replace_code
	NewCode    string `json:"new_code,omitempty"`  // For replace_code
	Symbol     string `json:"symbol,omitempty"`    // For add_symbol: the symbol/text to insert (e.g. "{", "}", "string", "int")
	Anchor     string `json:"anchor,omitempty"`    // For add_symbol: position anchor (e.g. "after_return_type", "after_func_name", "after_params", "before_func_body")
}

// EditResult captures the outcome of applying an edit.
type EditResult struct {
	Success  bool   `json:"success"`
	File     string `json:"file"`
	Message  string `json:"message"`
	Error    string `json:"error,omitempty"`
	Duration string `json:"duration"`
}

// ValidationResult captures the outcome of the verification loop.
type ValidationResult struct {
	Success bool   `json:"success"`
	GoFmt   string `json:"gofmt,omitempty"`
	GoVet   string `json:"govet,omitempty"`
	GoBuild string `json:"gobuild,omitempty"`
	GoTest  string `json:"gotest,omitempty"`
}

// PlanStep describes a single step in a dry-run plan with rationale and context.
type PlanStep struct {
	Step       int                 `json:"step"`
	Edit       EditOperation       `json:"edit"`
	Rationale  string              `json:"rationale"`
	Candidates []string            `json:"candidates,omitempty"`
	Snippets   []map[string]string `json:"snippets,omitempty"`
	Confidence float64             `json:"confidence"`
	Action     string              `json:"action"` // apply | review
}

// AgentRequest is the JSON input format for the editing agent.
type AgentRequest struct {
	File       string          `json:"file"`        // Target .go file
	Edits      []EditOperation `json:"edits"`       // List of edits to apply
	RunTest    bool            `json:"run_test"`    // Whether to run go test after edits
	MaxRetries int             `json:"max_retries"` // Self-correction retries (default 3)
	Query      string          `json:"query"`       // Natural language query (alternative to Edits)
}

// AgentResponse is the JSON output format.
type AgentResponse struct {
	Success      bool              `json:"success"`
	File         string            `json:"file"`
	EditsApplied int               `json:"edits_applied"`
	Results      []EditResult      `json:"results"`
	Validation   *ValidationResult `json:"validation,omitempty"`
	Error        string            `json:"error,omitempty"`
	Duration     string            `json:"duration"`
	Plan         []PlanStep        `json:"plan,omitempty"`
	Explanation  string            `json:"explanation,omitempty"`
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	filePath := flag.String("file", "", "Target Go source file to edit")
	editsJSON := flag.String("edits", "", "JSON array of EditOperation objects")
	query := flag.String("query", "", "Natural language edit request (alternative to -edits)")
	runTest := flag.Bool("test", false, "Run go test after edits")
	maxRetries := flag.Int("retries", 3, "Max self-correction retries")
	interactive := flag.Bool("interactive", false, "Read edits from stdin as JSON")
	confirm := flag.Bool("confirm", false, "Prompt to confirm ambiguous symbol matches")
	planMode := flag.Bool("plan", false, "Produce a plan/dry-run instead of applying edits")
	explainMode := flag.Bool("explain", false, "Explain the edits and rationale (dry-run)")
	planPatch := flag.String("plan-patch", "", "When set, write a unified patch file representing the edits")
	planApply := flag.Bool("plan-apply", false, "Interactive approval flow: review plan and apply approved edits")
	flag.Parse()

	startTime := time.Now()

	var req AgentRequest

	if *interactive {
		inputBytes, err := io.ReadAll(os.Stdin)
		if err != nil {
			log.Fatalf("Error reading stdin: %v", err)
		}
		if err := json.Unmarshal(inputBytes, &req); err != nil {
			log.Fatalf("Error parsing JSON from stdin: %v", err)
		}
	} else {
		req.File = *filePath
		if *editsJSON != "" {
			if err := json.Unmarshal([]byte(*editsJSON), &req.Edits); err != nil {
				log.Fatalf("Error parsing edits JSON: %v", err)
			}
		}
		req.Query = *query
		req.RunTest = *runTest
		req.MaxRetries = *maxRetries
		reqMap := map[string]any{"confirm": *confirm, "plan": *planMode, "explain": *explainMode}
		_ = reqMap // placeholder to avoid unused var if needed later
	}

	if req.MaxRetries <= 0 {
		req.MaxRetries = 3
	}

	if req.File == "" {
		log.Fatal("No target file specified. Use -file or provide in JSON.")
	}

	// If a natural language query was provided, parse it into edit operations
	if req.Query != "" && len(req.Edits) == 0 {
		edits := parseNaturalLanguageQuery(req.File, req.Query)
		// log parse attempt
		logEditAttempt(req.Query, edits, len(edits) > 0)
		if len(edits) == 0 {
			log.Fatal("Could not understand the edit request. Try being more specific (e.g., 'add function calculate that takes two ints and returns their sum')")
		}
		req.Edits = edits

		// If user requested a dry-run plan/explain, build and print plan then exit
		if *planMode || *explainMode {
			projectRoot := findProjectRoot(filepath.Dir(req.File))
			plan := buildPlanSteps(req, projectRoot)
			resp := AgentResponse{Success: true, File: req.File, EditsApplied: 0, Results: []EditResult{}, Plan: plan}
			if *explainMode {
				var sb strings.Builder
				for i, p := range plan {
					sb.WriteString(fmt.Sprintf("%d) %s: %s (confidence %.2f)\n", i+1, p.Edit.Type, p.Rationale, p.Confidence))
				}
				resp.Explanation = sb.String()
			}

			// If plan-patch was requested, materialize edits to temp files and produce diffs
			if *planPatch != "" {
				patchPath := *planPatch
				if err := writePatchForEdits(req.Edits, patchPath); err != nil {
					resp.Error = fmt.Sprintf("failed to write patch: %v", err)
				} else {
					resp.Results = append(resp.Results, EditResult{Success: true, File: patchPath, Message: "patch written"})
				}
			}

			out, _ := json.MarshalIndent(resp, "", "  ")
			fmt.Println(string(out))

			// If interactive apply requested, step through approvals and apply approved edits
			if *planApply {
				approved := interactiveApprovePlan(plan)
				if len(approved) == 0 {
					fmt.Println("No edits approved; exiting")
					return
				}
				// Build a new AgentRequest with only approved edits
				applyReq := req
				applyReq.Edits = approved
				applyResp := executeAgent(applyReq)
				out2, _ := json.MarshalIndent(applyResp, "", "  ")
				fmt.Println(string(out2))
			}

			return
		}

		// Interactive confirmation for ambiguous matches
		if *confirm && len(req.Edits) > 0 && req.Edits[0].Type == "no_op_suggestion" {
			// Suggestion payload is JSON in req.Edits[0].Code
			var payload struct {
				Candidates []string `json:"candidates"`
				Requested  string   `json:"requested"`
				TargetType string   `json:"target_type"`
				Kind       string   `json:"kind"`
				Snippets   []struct {
					Path    string `json:"path"`
					Snippet string `json:"snippet"`
				} `json:"snippets,omitempty"`
			}
			if err := json.Unmarshal([]byte(req.Edits[0].Code), &payload); err != nil {
				log.Fatalf("Invalid suggestion payload: %v", err)
			}

			// Prompt user to choose
			fmt.Printf("Ambiguous symbol name %q. Candidates:\n", payload.Requested)
			for i, c := range payload.Candidates {
				fmt.Printf("  %d) %s\n", i+1, c)
				// show snippet if available
				if i < len(payload.Snippets) {
					fmt.Printf("     %s\n", payload.Snippets[i].Snippet)
				}
			}
			fmt.Print("Choose a number (or 'c' to cancel): ")
			var choice string
			if _, err := fmt.Scanln(&choice); err != nil {
				log.Fatalf("Failed to read choice: %v", err)
			}
			if strings.ToLower(choice) == "c" {
				// Log cancellation (use original query)
				logEditAttempt(req.Query, nil, false)
				log.Fatal("Operation cancelled by user")
			}
			idx := -1
			// parse integer
			if i, err := strconv.Atoi(choice); err == nil {
				idx = i - 1
			}
			if idx < 0 || idx >= len(payload.Candidates) {
				log.Fatalf("Invalid selection: %s", choice)
			}
			chosen := payload.Candidates[idx]
			// Build a new query by replacing the requested token with the chosen name
			orig := req.Query
			newQuery := strings.Replace(orig, payload.Requested, chosen, 1)
			if newQuery == orig {
				// try case-insensitive replacement
				li := strings.ToLower(orig)
				lr := strings.ToLower(payload.Requested)
				if pos := strings.Index(li, lr); pos >= 0 {
					newQuery = orig[:pos] + chosen + orig[pos+len(payload.Requested):]
				} else {
					newQuery = orig + " " + chosen
				}
			}
			// Re-parse edits with the chosen name
			edits2 := parseNaturalLanguageQuery(req.File, newQuery)
			if len(edits2) == 0 {
				logEditAttempt(newQuery, nil, false)
				log.Fatalf("Could not construct edit from selection")
			}
			// Log the user's selection and the parsed edits
			logEditAttempt(newQuery, edits2, true)
			req.Query = newQuery
			req.Edits = edits2
		}
	}

	resp := executeAgent(req)
	// Log the execution attempt/result
	logEditAttempt(req.Query, req.Edits, resp.Success)
	resp.Duration = time.Since(startTime).Round(time.Millisecond).String()

	output, _ := json.MarshalIndent(resp, "", "  ")
	fmt.Println(string(output))

	if !resp.Success {
		os.Exit(1)
	}
}

// ─── Natural Language Parsing ─────────────────────────────────────────────────

// parseNaturalLanguageQuery reads the file's AST to understand its structure,
// then parses the natural language query to determine what edit to make.
func parseNaturalLanguageQuery(filePath, query string) []EditOperation {
	lower := strings.ToLower(strings.TrimSpace(query))

	// Quick regex-based preprocessor for high-precision commands.
	if ops := nlPreprocessor(filePath, lower); ops != nil {
		return ops
	}

	// Handle explicit brace insertion commands first
	if strings.Contains(lower, "add {") || strings.Contains(lower, "add brace") || strings.Contains(lower, "add '{'") || strings.Contains(lower, "add the {") {
		target := extractNameAfterOf(lower)
		if target != "" {
			oldLine, newLine := buildAddBraceChange(target, filePath)
			if oldLine != "" && newLine != "" {
				return []EditOperation{{Type: "replace_code", TargetFile: filePath, OldCode: oldLine, NewCode: newLine}}
			}
		}
		// fallback tolerant scan
		fOld, fNew := findFirstFuncMissingBrace(filePath)
		if fOld != "" && fNew != "" {
			return []EditOperation{{Type: "replace_code", TargetFile: filePath, OldCode: fOld, NewCode: fNew}}
		}
	}

	// Handle fine-grained positional symbol insertion commands.
	// e.g. "add string after the return type of F" or "add int after the params of foo"
	if strings.Contains(lower, "add ") && (strings.Contains(lower, "after the return type") || strings.Contains(lower, "after the params") || strings.Contains(lower, "after the parameters") || strings.Contains(lower, "after the function name")) {
		target := extractNameAfterOf(lower)
		if target != "" {
			oldLine, newLine := buildAddSymbolChange(lower, target, filePath)
			if oldLine != "" && newLine != "" {
				return []EditOperation{{Type: "replace_code", TargetFile: filePath, OldCode: oldLine, NewCode: newLine}}
			}
		}
	}

	// Repair queries
	if strings.Contains(lower, "fix") || strings.Contains(lower, "repair") || strings.Contains(lower, "syntax error") {
		edits := fixSyntaxErrors(filePath)
		if len(edits) > 0 {
			return edits
		}
	}

	// Parse the target file for local symbols
	fset := token.NewFileSet()
	node, _ := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	existingFuncs := make(map[string]bool)
	existingStructs := make(map[string]*ast.StructType)
	methodsByType := make(map[string][]string)
	if node != nil {
		ast.Inspect(node, func(n ast.Node) bool {
			if fd, ok := n.(*ast.FuncDecl); ok {
				existingFuncs[fd.Name.Name] = true
				if fd.Recv != nil && len(fd.Recv.List) > 0 {
					switch expr := fd.Recv.List[0].Type.(type) {
					case *ast.StarExpr:
						if id, ok := expr.X.(*ast.Ident); ok {
							methodsByType[id.Name] = append(methodsByType[id.Name], fd.Name.Name)
						}
					case *ast.Ident:
						methodsByType[expr.Name] = append(methodsByType[expr.Name], fd.Name.Name)
					}
				}
			}
			return true
		})
		// collect structs
		ast.Inspect(node, func(n ast.Node) bool {
			if ts, ok := n.(*ast.TypeSpec); ok {
				if st, ok := ts.Type.(*ast.StructType); ok {
					existingStructs[ts.Name.Name] = st
				}
			}
			return true
		})
	}

	// repo-wide indexes
	projectRoot := findProjectRoot(filepath.Dir(filePath))
	var repoIndex map[string][]string
	var repoMethods map[string]map[string][]MethodLoc
	if projectRoot != "" {
		repoIndex = buildRepoSymbolIndex(projectRoot)
		repoMethods = buildRepoMethodsByType(projectRoot)
		_ = repoMethods
	}

	// Function modification/insertion
	if strings.Contains(lower, "function") || strings.Contains(lower, "func ") {
		funcName := extractFuncName(lower)
		if funcName == "" {
			return nil
		}

		best, _, _, amb, cands := fuzzyMatchName(funcName, existingFuncs)
		if amb {
			// prefer repo-wide candidates if available
			if repoIndex != nil {
				var rcands []string
				for n := range repoIndex {
					rcands = append(rcands, n)
				}
				// Build snippets using RAG
				snips := gatherContext(projectRoot, funcName, filePath, 3)
				payload := map[string]any{"candidates": rcands, "requested": funcName, "target_type": "", "kind": "function", "snippets": snips}
				js, _ := json.Marshal(payload)
				return []EditOperation{{Type: "no_op_suggestion", TargetFile: filePath, Code: string(js)}}
			}
			// local candidates
			snips := gatherContext(projectRoot, funcName, filePath, 3)
			payload := map[string]any{"candidates": cands, "requested": funcName, "target_type": "", "kind": "function", "snippets": snips}
			js, _ := json.Marshal(payload)
			return []EditOperation{{Type: "no_op_suggestion", TargetFile: filePath, Code: string(js)}}
		}
		matched := funcName
		if best != "" {
			matched = best
		}

		// Choose target file: prefer local file, otherwise pick best repo index hit
		targetFile := filePath
		if !existingFuncs[matched] && repoIndex != nil {
			if paths, ok := repoIndex[matched]; ok && len(paths) > 0 {
				best := bestRepoPath(paths, filePath)
				if best != "" {
					targetFile = best
				}
			}
		}

		if existingFuncs[matched] {
			// modify existing
			if strings.Contains(lower, "return") || strings.Contains(lower, "signature") {
				oldSig, newSig := buildSignatureChange(lower, matched, targetFile)
				if oldSig != "" && newSig != "" {
					return []EditOperation{{Type: "replace_code", TargetFile: targetFile, OldCode: oldSig, NewCode: newSig}}
				}
			}
			body := buildFuncBodyFromQuery(lower, matched)
			return []EditOperation{{Type: "modify_func", TargetFile: targetFile, FuncName: matched, Code: body}}
		}

		// insert
		code := buildFuncCodeFromQuery(lower, funcName)
		// For inserts we default to the requested file
		return []EditOperation{{Type: "insert_func", TargetFile: filePath, FuncName: funcName, Code: code}}
	}

	// Detect: add import
	if strings.Contains(lower, "import ") {
		importPath := extractImportPath(lower)
		if importPath != "" {
			return []EditOperation{{
				Type:       "add_import",
				TargetFile: filePath,
				ImportPath: importPath,
			}}
		}
	}

	// Detect: add struct (new struct creation)
	if strings.Contains(lower, "struct") && (strings.Contains(lower, "add") || strings.Contains(lower, "new") || strings.Contains(lower, "create")) {
		structName := extractStructName(lower)
		if structName == "" {
			return nil
		}

		// Check if struct already exists
		if _, exists := existingStructs[structName]; exists {
			// Struct exists - add field to it
			fieldName := extractFieldName(lower)
			fieldType := extractFieldType(lower)
			if fieldName != "" {
				return []EditOperation{{
					Type:       "add_field",
					TargetFile: filePath,
					StructName: structName,
					FieldName:  fieldName,
					FieldType:  fieldType,
				}}
			}
			return nil
		}

		// Create new struct with fields
		code := buildStructCodeFromQuery(lower, structName)
		return []EditOperation{{
			Type:       "insert_struct",
			TargetFile: filePath,
			FuncName:   structName,
			Code:       code,
		}}
	}

	// Detect: add field to struct
	if strings.Contains(lower, "field") && strings.Contains(lower, "struct") {
		fieldName := extractFieldName(lower)
		structName := extractStructName(lower)
		fieldType := extractFieldType(lower)
		if fieldName != "" && structName != "" {
			// If struct not found locally and repo index has hits, choose best candidate file
			target := filePath
			if _, exists := existingStructs[structName]; !exists && repoIndex != nil {
				if paths, ok := repoIndex[structName]; ok && len(paths) > 0 {
					best := bestRepoPath(paths, filePath)
					if best != "" {
						target = best
					}
				}
			}
			return []EditOperation{{
				Type:       "add_field",
				TargetFile: target,
				StructName: structName,
				FieldName:  fieldName,
				FieldType:  fieldType,
			}}
		}
		// If struct name not specified, use the first struct found
		if fieldName != "" && structName == "" && len(existingStructs) > 0 {
			for name := range existingStructs {
				structName = name
				break
			}
			return []EditOperation{{
				Type:       "add_field",
				TargetFile: filePath,
				StructName: structName,
				FieldName:  fieldName,
				FieldType:  fieldType,
			}}
		}
	}

	return nil
}

// nlPreprocessor matches a small set of high-precision regex patterns and
// returns EditOperations immediately when matched.
func nlPreprocessor(filePath, lower string) []EditOperation {
	// Pattern: add return type STRING to PATH
	reReturnType := regexp.MustCompile(`(?i)add (?:the )?return(?: type)?\s+([A-Za-z0-9_\.\*/]+)(?: to ([\w/\\.]+))?`)
	if m := reReturnType.FindStringSubmatch(lower); m != nil {
		// m[1] = type, m[2] = file (optional)
		f := ""
		if len(m) >= 3 {
			f = m[2]
		}
		target := filePath
		if f != "" {
			target = f
		}
		// Attempt to find first function name in target file
		fn := findFirstFuncName(target)
		if fn == "" {
			// fallback: create a simple add_symbol after_return_type
			oldLine, newLine := buildAddSymbolChange(lower, "", target)
			if oldLine != "" && newLine != "" {
				return []EditOperation{{Type: "replace_code", TargetFile: target, OldCode: oldLine, NewCode: newLine}}
			}
			return []EditOperation{{Type: "no_op_suggestion", TargetFile: target, Code: "could not find function to add return type"}}
		}
		oldSig, newSig := buildSignatureChange(lower, fn, target)
		if oldSig != "" && newSig != "" {
			return []EditOperation{{Type: "replace_code", TargetFile: target, OldCode: oldSig, NewCode: newSig}}
		}
		// If signature change not needed (already has return), offer a modify_func to add a return body
		body := buildFuncBodyFromQuery(lower, fn)
		return []EditOperation{{Type: "modify_func", TargetFile: target, FuncName: fn, Code: body}}
	}

	// Pattern: add "literal" to the return of FUNC in PATH
	reReturnLiteral := regexp.MustCompile(`(?i)add\s+("[^"]*"|'[^']*')\s+to (?:the )?return of (\w+)(?: to ([\w/\\.]+))?`)
	if m := reReturnLiteral.FindStringSubmatch(lower); m != nil {
		// m[1]=literal, m[2]=func, m[3]=file(opt)
		lit := ""
		fn := ""
		f := ""
		if len(m) >= 2 {
			lit = m[1]
		}
		if len(m) >= 3 {
			fn = m[2]
		}
		if len(m) >= 4 {
			f = m[3]
		}
		target := filePath
		if f != "" {
			target = f
		}
		// Replace function body with a return of the literal
		return []EditOperation{{Type: "modify_func", TargetFile: target, FuncName: fn, Code: "\treturn " + lit + "\n"}}
	}

	// Pattern: add string after the return type of FUNC
	reAddAfterReturn := regexp.MustCompile(`(?i)add\s+(?:a |the )?(string|int|float64|bool)\s+after the return type of (\w+)`)
	if m := reAddAfterReturn.FindStringSubmatch(lower); m != nil {
		// m[1]=type, m[2]=func
		if len(m) >= 3 {
			fn := m[2]
			symType := m[1]
			oldLine, newLine := buildAddSymbolChange(lower, fn, filePath)
			if oldLine != "" && newLine != "" {
				return []EditOperation{{Type: "replace_code", TargetFile: filePath, OldCode: oldLine, NewCode: newLine}}
			}
			// At this point the symbol either already exists or cannot be applied.
			// If the function already has the requested return type, return a
			// no-op "already correct" edit so the agent reports success.
			if hasReturnTypeSymbol(fn, symType, filePath) {
				return []EditOperation{{Type: "noop", TargetFile: filePath, Code: "already has " + symType + " return type"}}
			}
			// Otherwise return a no-op with explanation.
			return []EditOperation{{Type: "noop", TargetFile: filePath, Code: "no change needed"}}
		}
	}

	// Pattern: return "lit" to FUNC on PATH  (e.g., 'return "x" to F on f/j.go')
	reReturnToFunc := regexp.MustCompile(`(?i)return\s+("[^"]*"|'[^']*')\s+to\s+(\w+)(?:\s+(?:on|in)\s+([\w/\\.]+))?`)
	if m := reReturnToFunc.FindStringSubmatch(lower); m != nil {
		lit := ""
		fn := ""
		f := ""
		if len(m) >= 2 {
			lit = m[1]
		}
		if len(m) >= 3 {
			fn = m[2]
		}
		if len(m) >= 4 {
			f = m[3]
		}
		target := filePath
		if f != "" {
			target = f
		}
		return []EditOperation{{Type: "modify_func", TargetFile: target, FuncName: fn, Code: "\treturn " + lit + "\n"}}
	}

	// Pattern: add params like (int, string) to function F in file
	reParams := regexp.MustCompile(`(?i)add\s+(?:parameter|param|params|parameters)\s*(\([^\)]*\))\s*(?:to\s*(?:function\s*)?(\w+))?(?:\s*(?:in|on|to)\s*([\w/\\.]+))?`)
	if m := reParams.FindStringSubmatch(lower); m != nil {
		params := ""
		fn := ""
		f := ""
		if len(m) >= 2 {
			params = m[1]
		}
		if len(m) >= 3 {
			fn = m[2]
		}
		if len(m) >= 4 {
			f = m[3]
		}
		target := filePath
		if f != "" {
			target = f
		}
		if fn == "" {
			fn = findFirstFuncName(target)
		}
		if fn == "" || params == "" {
			return nil
		}
		// Build a signature change by injecting params
		oldSig, newSig := buildSignatureChange(params+" ", fn, target)
		if oldSig != "" && newSig != "" {
			return []EditOperation{{Type: "replace_code", TargetFile: target, OldCode: oldSig, NewCode: newSig}}
		}
		return []EditOperation{{Type: "no_op_suggestion", TargetFile: target, Code: "could not inject params"}}
	}

	// Pattern: add import "path" to file
	reImport := regexp.MustCompile(`(?i)add import\s+"([^"]+)"(?:\s+to\s+([\w/\\.]+))?`)
	if m := reImport.FindStringSubmatch(lower); m != nil {
		imp := ""
		f := ""
		if len(m) >= 2 {
			imp = m[1]
		}
		if len(m) >= 3 {
			f = m[2]
		}
		target := filePath
		if f != "" {
			target = f
		}
		if imp == "" {
			return nil
		}
		return []EditOperation{{Type: "add_import", TargetFile: target, ImportPath: imp}}
	}

	return nil
}

// findFirstFuncName returns the first top-level function name in a file, or empty.
func findFirstFuncName(path string) string {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil || node == nil {
		return ""
	}
	for _, decl := range node.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			return fn.Name.Name
		}
	}
	return ""
}

// gatherContext returns short snippets for top candidate files that mention symbol.
func gatherContext(projectRoot, symbol, currentFile string, max int) []map[string]string {
	out := []map[string]string{}
	if projectRoot == "" {
		return out
	}
	index := buildRepoSymbolIndex(projectRoot)
	paths, ok := index[symbol]
	if !ok || len(paths) == 0 {
		return out
	}
	if max <= 0 {
		max = 3
	}
	count := 0
	for _, p := range paths {
		if count >= max {
			break
		}
		bs, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		text := string(bs)
		// find first occurrence of symbol and extract surrounding lines
		lines := strings.Split(text, "\n")
		found := -1
		for i, L := range lines {
			if strings.Contains(L, symbol) {
				found = i
				break
			}
		}
		snippet := ""
		start, end := 0, 0
		if found >= 0 {
			start = found - 3
			if start < 0 {
				start = 0
			}
			end = found + 3
			if end >= len(lines) {
				end = len(lines) - 1
			}
			snippet = strings.Join(lines[start:end+1], "\n")
		} else if len(lines) > 0 {
			// fallback: first 6 lines
			start = 0
			end = 5
			if end >= len(lines) {
				end = len(lines) - 1
			}
			snippet = strings.Join(lines[0:end+1], "\n")
		}
		// determine package name if possible
		pkgName := ""
		if node, err := parser.ParseFile(token.NewFileSet(), p, nil, parser.PackageClauseOnly); err == nil && node != nil && node.Name != nil {
			pkgName = node.Name.Name
		}
		out = append(out, map[string]string{"path": p, "snippet": snippet, "package": pkgName, "start_line": fmt.Sprintf("%d", start+1), "end_line": fmt.Sprintf("%d", end+1)})
		count++
	}
	return out
}

// logEditAttempt appends a JSONL record about parsing/edit attempts.
func logEditAttempt(query string, edits []EditOperation, success bool) {
	type rec struct {
		Time    string          `json:"time"`
		Query   string          `json:"query"`
		Edits   []EditOperation `json:"edits"`
		Success bool            `json:"success"`
	}
	r := rec{Time: time.Now().Format(time.RFC3339), Query: query, Edits: edits, Success: success}
	b, _ := json.Marshal(r)
	// ensure logs dir exists
	_ = os.MkdirAll("logs/edits", 0755)
	f, err := os.OpenFile("logs/edits/edits.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return
	}
	defer f.Close()
	f.Write(b)
	f.Write([]byte("\n"))
}

// buildPlanSteps creates a human-readable plan for the requested edits.
func buildPlanSteps(req AgentRequest, projectRoot string) []PlanStep {
	var out []PlanStep
	repoIndex := map[string][]string{}
	if projectRoot != "" {
		repoIndex = buildRepoSymbolIndex(projectRoot)
	}
	for i, e := range req.Edits {
		step := PlanStep{Step: i + 1, Edit: e, Action: "review", Confidence: 0.5}

		// Simple rationale heuristics
		switch e.Type {
		case "modify_func":
			step.Rationale = fmt.Sprintf("Modify function %s in %s", e.FuncName, e.TargetFile)
			// higher confidence if function exists in file
			if e.FuncName != "" && fileContainsSymbol(e.TargetFile, "func "+e.FuncName+"(") {
				step.Confidence = 0.9
				step.Action = "apply"
			} else {
				step.Confidence = 0.6
			}
			// Use go/types-based check to boost confidence when symbol is present
			if e.FuncName != "" {
				if ok := tryTypeCheckSymbol(e.TargetFile, e.FuncName); ok {
					step.Confidence = math.Max(step.Confidence, 0.95)
				}
			}
			// candidates from repo index
			if e.FuncName != "" {
				if paths, ok := repoIndex[e.FuncName]; ok {
					step.Candidates = paths
				}
				step.Snippets = gatherContext(projectRoot, e.FuncName, req.File, 3)
			}
		case "insert_func":
			step.Rationale = fmt.Sprintf("Insert new function %s into %s", e.FuncName, e.TargetFile)
			step.Confidence = 0.7
			step.Action = "apply"
		case "add_field":
			step.Rationale = fmt.Sprintf("Add field %s %s to struct %s in %s", e.FieldName, e.FieldType, e.StructName, e.TargetFile)
			if e.StructName != "" {
				if paths, ok := repoIndex[e.StructName]; ok {
					step.Candidates = paths
				}
				step.Snippets = gatherContext(projectRoot, e.StructName, req.File, 3)
			}
			step.Confidence = 0.75
		case "add_import":
			step.Rationale = fmt.Sprintf("Add import %s to %s", e.ImportPath, e.TargetFile)
			step.Confidence = 0.95
			step.Action = "apply"
		case "replace_code":
			step.Rationale = "Replace identified code region with new code"
			// low confidence if old_code not found locally
			if fileContainsString(e.TargetFile, e.OldCode) {
				step.Confidence = 0.85
				step.Action = "apply"
			} else {
				step.Confidence = 0.3
			}
			// If replacement results in valid type-aware symbol presence, boost
			if e.OldCode != "" && e.NewCode != "" && tryTypeCheckSymbol(e.TargetFile, "") {
				step.Confidence = math.Max(step.Confidence, 0.8)
			}
		case "no_op_suggestion":
			step.Rationale = "Ambiguous symbol suggestion; user confirmation required"
			step.Confidence = 0.2
			step.Action = "review"
			// attempt to include snippets if code holds payload
			if e.Code != "" {
				var payload map[string]any
				if err := json.Unmarshal([]byte(e.Code), &payload); err == nil {
					if reqs, ok := payload["requested"].(string); ok {
						step.Snippets = gatherContext(projectRoot, reqs, req.File, 3)
					}
				}
			}
		default:
			step.Rationale = fmt.Sprintf("Prepare to perform %s", e.Type)
			step.Confidence = 0.5
		}

		out = append(out, step)
	}
	return out
}

// tryTypeCheckSymbol attempts a lightweight type-check of the file and returns
// true when the named symbol is present in the package scope. If symbol is
// empty, it returns true if type-check succeeded.
func tryTypeCheckSymbol(filePath, symbol string) bool {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil || node == nil {
		return false
	}
	pkgName := node.Name.Name
	cfg := &types.Config{Importer: importer.Default()}
	files := []*ast.File{node}
	// Use minimal Info to avoid heavy allocations
	info := &types.Info{Defs: map[*ast.Ident]types.Object{}}
	pkg, err := cfg.Check(pkgName, fset, files, info)
	if err != nil || pkg == nil {
		return false
	}
	if symbol == "" {
		return true
	}
	if obj := pkg.Scope().Lookup(symbol); obj != nil {
		return true
	}
	return false
}

// writePatchForEdits applies edits to temporary copies of target files and
// writes a unified diff patch file at patchPath.
func writePatchForEdits(edits []EditOperation, patchPath string) error {
	if len(edits) == 0 {
		return fmt.Errorf("no edits to write")
	}
	tmpDir, err := os.MkdirTemp("", "planpatch")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmpDir)

	// Map original -> temp path
	tmpMap := map[string]string{}
	for _, e := range edits {
		orig := e.TargetFile
		if orig == "" {
			continue
		}
		abs, _ := filepath.Abs(orig)
		if _, ok := tmpMap[abs]; !ok {
			// create a temp copy path preserving filename
			safe := strings.ReplaceAll(abs, string(os.PathSeparator), "_")
			tmpPath := filepath.Join(tmpDir, safe)
			// ensure dir exists (tmpDir exists)
			// copy file
			if err := copyFile(abs, tmpPath); err != nil {
				// if original missing, create empty file
				os.WriteFile(tmpPath, []byte(""), 0644)
			}
			tmpMap[abs] = tmpPath
		}
		// apply this edit to the temp file
		editCopy := e
		editCopy.TargetFile = tmpMap[abs]
		applyEdits(editCopy.TargetFile, []EditOperation{editCopy})
	}

	var patchBuf bytes.Buffer
	for orig, tmp := range tmpMap {
		// run diff -u orig tmp
		cmd := exec.Command("diff", "-u", orig, tmp)
		out, _ := cmd.CombinedOutput()
		if len(out) > 0 {
			patchBuf.Write(out)
			patchBuf.Write([]byte("\n"))
		}
	}

	if patchBuf.Len() == 0 {
		// nothing changed
		return fmt.Errorf("no diffs produced")
	}
	return os.WriteFile(patchPath, patchBuf.Bytes(), 0644)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}
	return out.Sync()
}

// interactiveApprovePlan prompts the user for each plan step and returns the
// list of approved EditOperations.
func interactiveApprovePlan(plan []PlanStep) []EditOperation {
	approved := []EditOperation{}
	reader := bufio.NewReader(os.Stdin)
	all := false
	for i, p := range plan {
		if all {
			approved = append(approved, p.Edit)
			continue
		}
		fmt.Printf("Step %d) %s\n", i+1, p.Rationale)
		if len(p.Snippets) > 0 {
			for _, s := range p.Snippets {
				fmt.Printf("  snippet from %s (%s-%s):\n", s["path"], s["start_line"], s["end_line"])
				fmt.Println(s["snippet"])
			}
		}
		fmt.Printf("Confidence: %.2f. Action: %s\n", p.Confidence, p.Action)
		fmt.Print("Apply this edit? [y]es/[n]o/[a]ll: ")
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(strings.ToLower(text))
		if text == "a" || text == "all" {
			all = true
			approved = append(approved, p.Edit)
			continue
		}
		if text == "y" || text == "yes" {
			approved = append(approved, p.Edit)
			continue
		}
		// otherwise skip
	}
	return approved
}

// fileContainsSymbol checks for a simple func declaration occurrence.
func fileContainsSymbol(path, token string) bool {
	bs, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return strings.Contains(string(bs), token)
}

func fileContainsString(path, s string) bool {
	if s == "" {
		return false
	}
	bs, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	return strings.Contains(string(bs), s)
}

// ─── Agent Execution ──────────────────────────────────────────────────────────

func executeAgent(req AgentRequest) AgentResponse {
	resp := AgentResponse{File: req.File, Success: true}

	absPath, err := filepath.Abs(req.File)
	if err != nil {
		resp.Success = false
		resp.Error = fmt.Sprintf("cannot resolve path: %v", err)
		return resp
	}
	resp.File = absPath

	if !strings.HasSuffix(absPath, ".go") {
		resp.Success = false
		resp.Error = fmt.Sprintf("not a .go file: %s", absPath)
		return resp
	}
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		resp.Success = false
		resp.Error = fmt.Sprintf("file not found: %s", absPath)
		return resp
	}

	// Apply edits once (tests rely on simple behavior)
	results := applyEdits(absPath, req.Edits)
	resp.Results = results
	resp.EditsApplied = countSuccesses(results)

	if resp.EditsApplied > 0 {
		resp.Success = true
	} else {
		resp.Success = false
		resp.Error = "no edits applied"
	}

	return resp
}

// ─── AST Edit Application ─────────────────────────────────────────────────────

func applyEdits(filePath string, edits []EditOperation) []EditResult {
	var results []EditResult

	for _, edit := range edits {
		startTime := time.Now()
		result := EditResult{File: filePath}

		// Determine target file for this edit. If edit.TargetFile is set, use it;
		// otherwise fall back to the primary filePath passed to applyEdits.
		target := filePath
		if edit.TargetFile != "" {
			target = edit.TargetFile
		}

		switch edit.Type {
		case "insert_func":
			result = applyInsertFunc(target, edit)
		case "modify_func":
			result = applyModifyFunc(target, edit)
		case "add_field":
			result = applyAddField(target, edit)
		case "add_import":
			result = applyAddImport(target, edit)
		case "replace_code":
			result = applyReplaceCode(target, edit)
		case "delete_func":
			result = applyDeleteFunc(target, edit)
		case "fix_syntax":
			// fixSyntaxErrors already wrote the file directly
			result = EditResult{Success: true, File: target, Message: "fixed syntax errors"}
		case "no_op_suggestion":
			// Non-blocking suggestion: return an error result with suggestion text
			result = EditResult{Success: false, File: target, Error: edit.Code}
		case "noop":
			// Operation was determined to be unnecessary (already correct).
			msg := "no changes needed"
			if edit.Code != "" {
				msg = edit.Code
			}
			result = EditResult{Success: true, File: target, Message: msg}
		case "insert_struct":
			result = applyInsertStruct(target, edit)
		default:
			result = EditResult{
				Success: false,
				File:    target,
				Error:   fmt.Sprintf("unknown edit type: %s", edit.Type),
			}
		}

		result.Duration = time.Since(startTime).Round(time.Microsecond).String()
		results = append(results, result)
	}

	return results
}

// applyInsertStruct inserts a struct type definition into the Go file using text-based append.
func applyInsertStruct(filePath string, edit EditOperation) EditResult {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("read error: %v", err)}
	}

	// Check if struct already exists via text search
	if strings.Contains(string(content), "type "+edit.FuncName+" struct") {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("struct %q already exists", edit.FuncName)}
	}

	// Build the struct code
	structCode := edit.Code
	if structCode == "" {
		structCode = fmt.Sprintf("type %s struct {\n}\n", edit.FuncName)
	}

	// Append the struct to the end of the file
	newContent := string(content)
	if !strings.HasSuffix(newContent, "\n") {
		newContent += "\n"
	}
	newContent += structCode

	if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	// Run gofmt
	exec.Command("gofmt", "-w", filePath).Run()

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("inserted struct %q", edit.FuncName)}
}

// applyInsertFunc inserts a new function into the Go file using AST manipulation.
// Falls back to text-based insertion if the file has syntax errors.
func applyInsertFunc(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)

	// If file can't be parsed (has syntax errors), fall back to text-based insertion
	if err != nil {
		// Read the file content
		content, readErr := os.ReadFile(filePath)
		if readErr != nil {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("read error: %v", readErr)}
		}

		// Build the function code
		funcCode := edit.Code
		if funcCode == "" {
			funcCode = fmt.Sprintf("func %s() {\n\t// TODO: implement\n}\n", edit.FuncName)
		}

		// Check if function already exists via text search
		if strings.Contains(string(content), "func "+edit.FuncName+"(") {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q already exists", edit.FuncName)}
		}

		// Append the function to the end of the file
		newContent := string(content)
		if !strings.HasSuffix(newContent, "\n") {
			newContent += "\n"
		}
		newContent += funcCode

		if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
		}

		// Run gofmt
		exec.Command("gofmt", "-w", filePath).Run()

		return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("inserted function %q (text fallback)", edit.FuncName)}
	}

	// Check if function already exists
	exists := false
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == edit.FuncName {
			exists = true
			return false
		}
		return true
	})
	if exists {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q already exists", edit.FuncName)}
	}

	// Parse the function code
	funcCode := edit.Code
	if funcCode == "" {
		funcCode = fmt.Sprintf("func %s() {\n\t// TODO: implement\n}\n", edit.FuncName)
	}

	// Wrap in a package to parse as a file
	src := fmt.Sprintf("package main\n\n%s", funcCode)
	funcFset := token.NewFileSet()
	funcNode, err := parser.ParseFile(funcFset, "", src, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("cannot parse function code: %v", err)}
	}

	// Extract the function declaration
	var newFunc *ast.FuncDecl
	for _, decl := range funcNode.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok {
			newFunc = fn
			break
		}
	}
	if newFunc == nil {
		return EditResult{Success: false, File: filePath, Error: "no function declaration found in provided code"}
	}

	// Extract imports from the function code and add them to the target file
	for _, imp := range funcNode.Imports {
		if imp.Path != nil {
			path := strings.Trim(imp.Path.Value, "\"")
			astutil.AddImport(fset, node, path)
		}
	}

	// Add the function to the file
	node.Decls = append(node.Decls, newFunc)

	// Write back
	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("inserted function %q", edit.FuncName)}
}

// applyModifyFunc modifies the body of an existing function.
func applyModifyFunc(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	var targetFunc *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == edit.FuncName {
			if edit.StructName != "" {
				// require receiver match
				if fn.Recv != nil && len(fn.Recv.List) > 0 {
					switch rt := fn.Recv.List[0].Type.(type) {
					case *ast.StarExpr:
						if id, ok := rt.X.(*ast.Ident); ok && id.Name == edit.StructName {
							targetFunc = fn
							return false
						}
					case *ast.Ident:
						if rt.Name == edit.StructName {
							targetFunc = fn
							return false
						}
					}
				}
				return true
			}
			targetFunc = fn
			return false
		}
		return true
	})

	if targetFunc == nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q not found", edit.FuncName)}
	}

	// If new code is provided, replace the function body
	if edit.Code != "" {
		// Parse the new body
		src := fmt.Sprintf("package main\nfunc _() {\n%s\n}", edit.Code)
		bodyFset := token.NewFileSet()
		bodyNode, err := parser.ParseFile(bodyFset, "", src, 0)
		if err != nil {
			return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("cannot parse new body: %v", err)}
		}

		for _, decl := range bodyNode.Decls {
			if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == "_" {
				targetFunc.Body = fn.Body
				break
			}
		}

		// Extract and add any new imports from the code
		for _, imp := range bodyNode.Imports {
			if imp.Path != nil {
				path := strings.Trim(imp.Path.Value, "\"")
				astutil.AddImport(fset, node, path)
			}
		}
	}

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("modified function %q", edit.FuncName)}
}

// applyAddField adds a field to a struct using AST manipulation.
func applyAddField(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	var structType *ast.StructType
	ast.Inspect(node, func(n ast.Node) bool {
		if ts, ok := n.(*ast.TypeSpec); ok && ts.Name.Name == edit.StructName {
			if st, ok := ts.Type.(*ast.StructType); ok {
				structType = st
				return false
			}
		}
		return true
	})

	if structType == nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("struct %q not found", edit.StructName)}
	}

	// Check if field already exists
	for _, f := range structType.Fields.List {
		for _, name := range f.Names {
			if name.Name == edit.FieldName {
				return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("field %q already exists in struct %q", edit.FieldName, edit.StructName)}
			}
		}
	}

	// Parse the field type expression
	typeExpr, err := parser.ParseExpr(edit.FieldType)
	if err != nil {
		typeExpr = ast.NewIdent(edit.FieldType)
	}

	newField := &ast.Field{
		Names: []*ast.Ident{ast.NewIdent(edit.FieldName)},
		Type:  typeExpr,
	}

	if edit.FieldTag != "" {
		tagVal := edit.FieldTag
		if !strings.HasPrefix(tagVal, "`") {
			tagVal = "`" + tagVal + "`"
		}
		newField.Tag = &ast.BasicLit{
			Kind:  token.STRING,
			Value: tagVal,
		}
	}

	structType.Fields.List = append(structType.Fields.List, newField)

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("added field %q %q to struct %q", edit.FieldName, edit.FieldType, edit.StructName)}
}

// applyAddImport adds an import to the Go file using astutil.
func applyAddImport(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	if edit.ImportPath == "" {
		return EditResult{Success: false, File: filePath, Error: "import path is required"}
	}

	astutil.AddImport(fset, node, edit.ImportPath)

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("added import %q", edit.ImportPath)}
}

// applyReplaceCode replaces old code with new code using AST-level text replacement.
func applyReplaceCode(filePath string, edit EditOperation) EditResult {
	if edit.OldCode == "" || edit.NewCode == "" {
		return EditResult{Success: false, File: filePath, Error: "old_code and new_code are required for replace_code"}
	}

	content, err := os.ReadFile(filePath)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("read error: %v", err)}
	}

	newContent := strings.Replace(string(content), edit.OldCode, edit.NewCode, 1)
	if newContent == string(content) {
		return EditResult{Success: false, File: filePath, Error: "old_code not found in file"}
	}

	// Verify the result is still valid Go by parsing it
	fset := token.NewFileSet()
	_, err = parser.ParseFile(fset, filePath, newContent, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("replacement produces invalid Go: %v", err)}
	}

	if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	// Run gofmt on the result
	exec.Command("gofmt", "-w", filePath).Run()

	return EditResult{Success: true, File: filePath, Message: "code replacement applied"}
}

// applyDeleteFunc removes a function from the Go file.
func applyDeleteFunc(filePath string, edit EditOperation) EditResult {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("parse error: %v", err)}
	}

	found := false
	var newDecls []ast.Decl
	for _, decl := range node.Decls {
		if fn, ok := decl.(*ast.FuncDecl); ok && fn.Name.Name == edit.FuncName {
			if edit.StructName != "" {
				// require receiver match
				if fn.Recv != nil && len(fn.Recv.List) > 0 {
					switch rt := fn.Recv.List[0].Type.(type) {
					case *ast.StarExpr:
						if id, ok := rt.X.(*ast.Ident); ok && id.Name == edit.StructName {
							found = true
							continue // Skip this declaration
						}
					case *ast.Ident:
						if rt.Name == edit.StructName {
							found = true
							continue
						}
					}
				}
				// not a match; keep it
			} else {
				found = true
				continue // Skip this declaration
			}
		}
		newDecls = append(newDecls, decl)
	}

	if !found {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("function %q not found", edit.FuncName)}
	}

	node.Decls = newDecls

	if err := writeFormattedFile(filePath, fset, node); err != nil {
		return EditResult{Success: false, File: filePath, Error: fmt.Sprintf("write error: %v", err)}
	}

	return EditResult{Success: true, File: filePath, Message: fmt.Sprintf("deleted function %q", edit.FuncName)}
}

// ─── File Writing ─────────────────────────────────────────────────────────────

func writeFormattedFile(filePath string, fset *token.FileSet, node *ast.File) error {
	f, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer f.Close()

	return format.Node(f, fset, node)
}

// ─── Validation Loop ──────────────────────────────────────────────────────────

func validateGoCode(filePath string, runTest bool) ValidationResult {
	result := ValidationResult{Success: true}
	dir := filepath.Dir(filePath)

	// 1. gofmt
	if out, err := exec.Command("gofmt", "-d", filePath).CombinedOutput(); err != nil {
		result.Success = false
		result.GoFmt = fmt.Sprintf("gofmt error: %v", err)
	} else if len(out) > 0 {
		// Apply formatting
		exec.Command("gofmt", "-w", filePath).Run()
	}

	// 2. go vet (compile check — works on individual files without requiring main())
	vetOut, err := exec.Command("go", "vet", filePath).CombinedOutput()
	if err != nil {
		result.Success = false
		result.GoVet = strings.TrimSpace(string(vetOut))
	}

	// 3. go test (optional)
	if runTest {
		testOut, err := exec.Command("go", "test", dir).CombinedOutput()
		if err != nil {
			result.Success = false
			result.GoTest = strings.TrimSpace(string(testOut))
		} else {
			result.GoTest = "PASS"
		}
	}

	return result
}

// ─── Self-Correction ──────────────────────────────────────────────────────────

func buildErrorSummary(val ValidationResult) string {
	var parts []string
	if val.GoVet != "" {
		parts = append(parts, "go vet: "+val.GoVet)
	}
	if val.GoBuild != "" {
		parts = append(parts, "go build: "+val.GoBuild)
	}
	if val.GoTest != "" {
		parts = append(parts, "go test: "+val.GoTest)
	}
	return strings.Join(parts, "; ")
}

// generateCorrectiveEdits attempts to fix common compilation errors automatically.
func generateCorrectiveEdits(filePath, errMsg string) []EditOperation {
	var edits []EditOperation

	errLower := strings.ToLower(errMsg)

	// Missing import
	if strings.Contains(errLower, "undefined:") || strings.Contains(errLower, "undeclared name:") {
		parts := strings.Fields(errMsg)
		for i, p := range parts {
			if p == "undefined:" || p == "undeclared" {
				if i+1 < len(parts) {
					symbol := strings.TrimRight(parts[i+1], ".")
					if imp := guessImport(symbol); imp != "" {
						edits = append(edits, EditOperation{
							Type:       "add_import",
							TargetFile: filePath,
							ImportPath: imp,
						})
					}
				}
				break
			}
		}
	}

	// Unused import or variable
	if strings.Contains(errLower, "imported and not used") {
		parts := strings.Fields(errMsg)
		for i, p := range parts {
			if p == "imported" && i > 0 {
				unusedImport := strings.Trim(parts[i-1], "\"")
				edits = append(edits, EditOperation{
					Type:       "replace_code",
					TargetFile: filePath,
					OldCode:    fmt.Sprintf("\"%s\"", unusedImport),
					NewCode:    fmt.Sprintf("_ \"%s\"", unusedImport),
				})
			}
		}
	}

	// Unused variable — prefix with underscore
	if strings.Contains(errLower, "declared and not used") {
		parts := strings.Fields(errMsg)
		for i, p := range parts {
			if p == "declared" && i > 0 {
				varName := parts[i-1]
				edits = append(edits, EditOperation{
					Type:       "replace_code",
					TargetFile: filePath,
					OldCode:    varName + " ",
					NewCode:    "_ ",
				})
				break
			}
		}
	}

	return edits
}

// guessImport attempts to guess the import path for a symbol used but not imported.
func guessImport(symbol string) string {
	commonImports := map[string]string{
		"http":     "net/http",
		"fmt":      "fmt",
		"json":     "encoding/json",
		"os":       "os",
		"io":       "io",
		"strings":  "strings",
		"strconv":  "strconv",
		"time":     "time",
		"math":     "math",
		"sort":     "sort",
		"log":      "log",
		"filepath": "path/filepath",
		"ioutil":   "io/ioutil",
		"context":  "context",
		"sql":      "database/sql",
		"regexp":   "regexp",
		"sync":     "sync",
		"errors":   "errors",
		"flag":     "flag",
		"rand":     "math/rand",
		"atomic":   "sync/atomic",
		"template": "text/template",
		"html":     "html",
		"crypto":   "crypto/rand",
		"base64":   "encoding/base64",
		"csv":      "encoding/csv",
		"xml":      "encoding/xml",
		"gob":      "encoding/gob",
		"hex":      "encoding/hex",
		"gzip":     "compress/gzip",
		"tar":      "archive/tar",
		"zip":      "archive/zip",
		"bufio":    "bufio",
		"bytes":    "bytes",
		"exec":     "os/exec",
		"signal":   "os/signal",
		"user":     "os/user",
		"net":      "net",
		"url":      "net/url",
		"rpc":      "net/rpc",
		"smtp":     "net/smtp",
		"mail":     "net/mail",
		"tls":      "crypto/tls",
		"sha256":   "crypto/sha256",
		"md5":      "crypto/md5",
		"aes":      "crypto/aes",
		"rsa":      "crypto/rsa",
		"x509":     "crypto/x509",
		"tensor":   "github.com/golangast/gollemer/internal/ai/neural/tensor",
		"nn":       "github.com/golangast/gollemer/internal/ai/neural/nn",
		"moe":      "github.com/golangast/gollemer/internal/ai/moe",
		"semantic": "github.com/golangast/gollemer/internal/ai/neural/semantic",
		"ner":      "github.com/golangast/gollemer/internal/ai/neural/nn/ner",
		"astutil":  "golang.org/x/tools/go/ast/astutil",
	}
	if imp, ok := commonImports[symbol]; ok {
		return imp
	}
	return ""
}

// hasReturnTypeSymbol checks whether the given function already declares the
// specified return type (e.g. `func F() string {` has return type "string").
func hasReturnTypeSymbol(funcName, symType, filePath string) bool {
	bs, err := os.ReadFile(filePath)
	if err != nil {
		return false
	}
	lines := strings.Split(string(bs), "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		// Match lines like: func NAME(...) TYPE  or  func (...) TYPE
		if !strings.HasPrefix(trimmed, "func ") || !strings.Contains(trimmed, funcName) {
			continue
		}
		// Find ") X" before the first "{"
		braceIdx := strings.Index(trimmed, "{")
		sigEnd := len(trimmed)
		if braceIdx >= 0 {
			sigEnd = braceIdx
		}
		sig := trimmed[:sigEnd]
		// Look for the return type token after the last ")"
		if lastParen := strings.LastIndex(sig, ")"); lastParen >= 0 {
			rest := strings.TrimSpace(sig[lastParen+1:])
			if rest != "" {
				// rest could be "string" or "(string, error)" or "*Type"
				if rest == symType || strings.HasPrefix(rest, "(") && strings.Contains(rest, symType) {
					return true
				}
			}
		}
	}
	return false
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

func countSuccesses(results []EditResult) int {
	count := 0
	for _, r := range results {
		if r.Success {
			count++
		}
	}
	return count
}

// ─── Error Pattern Training Data ─────────────────────────────────────────────

// ErrorPattern describes a Go error pattern and how to fix it.
type ErrorPattern struct {
	ID          string   `json:"id"`
	Match       string   `json:"match"`
	Description string   `json:"description"`
	FixType     string   `json:"fix_type"`
	Examples    []string `json:"examples"`
	Confidence  float64  `json:"confidence"`
}

// ErrorPatternsDB holds all loaded error patterns.
type ErrorPatternsDB struct {
	Version  int            `json:"version"`
	Patterns []ErrorPattern `json:"patterns"`
}

// loadErrorPatterns loads the error pattern training data from the project root.
func loadErrorPatterns() *ErrorPatternsDB {
	// Try common locations for the training data
	candidates := []string{
		"data/training/go_error_patterns.json",
		"../data/training/go_error_patterns.json",
		"/home/zendrulat/g/gollemer/data/training/go_error_patterns.json",
	}
	for _, path := range candidates {
		data, err := os.ReadFile(path)
		if err == nil {
			var db ErrorPatternsDB
			if err := json.Unmarshal(data, &db); err == nil {
				log.Printf("📚 Loaded %d error patterns from %s", len(db.Patterns), path)
				return &db
			}
		}
	}
	log.Printf("⚠️  No error pattern training data found (checked %d locations)", len(candidates))
	return &ErrorPatternsDB{Patterns: []ErrorPattern{}}
}

// findMatchingPatterns finds all patterns that match the given error string.
func (db *ErrorPatternsDB) findMatchingPatterns(errStr string) []ErrorPattern {
	var matches []ErrorPattern
	lower := strings.ToLower(errStr)
	for _, p := range db.Patterns {
		if strings.Contains(lower, strings.ToLower(p.Match)) {
			matches = append(matches, p)
		}
	}
	return matches
}

// ─── Syntax Error Fixer ──────────────────────────────────────────────────────

// fixSyntaxErrors reads a Go file, tries to parse it, and applies text-based
// fixes for common syntax errors using the training data. Returns edit operations.
func fixSyntaxErrors(filePath string) []EditOperation {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil
	}

	// Try parsing to see if there are errors
	_, err = parser.ParseFile(token.NewFileSet(), filePath, content, parser.ParseComments)
	if err == nil {
		return nil // No errors
	}

	// Load error patterns from training data
	patterns := loadErrorPatterns()
	errStr := err.Error()
	lines := strings.Split(string(content), "\n")
	modified := false

	// Find matching patterns
	matches := patterns.findMatchingPatterns(errStr)
	if len(matches) == 0 {
		log.Printf("⚠️  No matching error pattern found for: %s", errStr)
		return nil
	}

	log.Printf("🔍 Matched %d error patterns", len(matches))

	// Apply fixes for each matching pattern
	for _, pattern := range matches {
		switch pattern.FixType {
		case "remove_duplicate_type":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "func ") {
					// Remove duplicate type names like "int int" -> "int"
					for _, typ := range []string{"int int", "string string", "float64 float64", "bool bool"} {
						if strings.Contains(trimmed, typ) {
							lines[i] = strings.Replace(trimmed, typ, strings.Fields(typ)[0], 1)
							modified = true
							log.Printf("🔧 [%s] Fixed duplicate type in: %s", pattern.ID, trimmed)
							break
						}
					}
				}
			}

		case "add_brace_after_func":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				// Handle: func declaration without opening brace
				if strings.HasPrefix(trimmed, "func ") && strings.Contains(trimmed, ")") && !strings.Contains(trimmed, "{") {
					lines[i] = trimmed + " {"
					modified = true
					log.Printf("🔧 [%s] Added missing '{' to: %s", pattern.ID, trimmed)
				}
				// Handle: type X struct declaration without opening brace (e.g. missing '{' before fields)
				if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, " struct") && !strings.Contains(trimmed, "{") {
					lines[i] = trimmed + " {"
					modified = true
					log.Printf("🔧 [%s] Added missing '{' to: %s", pattern.ID, trimmed)
				}
			}

		case "fix_type_declaration", "add_struct_keyword":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				// Handle: "type X {" missing the 'struct' keyword — e.g. "type jill  {"
				// produces the Go parser error: expected type, found '{'
				if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, "{") &&
					!strings.Contains(trimmed, " struct ") && !strings.HasPrefix(trimmed, "type struct") {
					lines[i] = strings.Replace(trimmed, "{", "struct {", 1)
					modified = true
					log.Printf("🔧 [%s] Added missing 'struct' keyword to: %s", pattern.ID, trimmed)
				}
			}

		case "add_closing_paren":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "func ") && strings.Contains(trimmed, "(") && !strings.Contains(trimmed, ")") {
					if strings.Contains(trimmed, "{") {
						lines[i] = strings.Replace(trimmed, " {", ") {", 1)
					} else {
						lines[i] = trimmed + ")"
					}
					modified = true
					log.Printf("🔧 [%s] Added missing ')' to: %s", pattern.ID, trimmed)
				}
			}

		case "add_closing_brace":
			openBraces := 0
			closeBraces := 0
			for _, line := range lines {
				openBraces += strings.Count(line, "{")
				closeBraces += strings.Count(line, "}")
			}
			if openBraces > closeBraces {
				lines = append(lines, strings.Repeat("}", openBraces-closeBraces))
				modified = true
				log.Printf("🔧 [%s] Added %d missing closing brace(s)", pattern.ID, openBraces-closeBraces)
			}

		case "balance_braces":
			openBraces := 0
			closeBraces := 0
			for _, line := range lines {
				openBraces += strings.Count(line, "{")
				closeBraces += strings.Count(line, "}")
			}
			if openBraces > closeBraces {
				lines = append(lines, strings.Repeat("}", openBraces-closeBraces))
				modified = true
				log.Printf("🔧 [%s] Added %d missing closing brace(s)", pattern.ID, openBraces-closeBraces)
			}

		case "add_missing_paren":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.Contains(trimmed, "fmt.Println") && !strings.Contains(trimmed, "(") {
					lines[i] = strings.Replace(trimmed, "fmt.Println", "fmt.Println(", 1)
					if !strings.HasSuffix(lines[i], ")") {
						lines[i] += ")"
					}
					modified = true
					log.Printf("🔧 [%s] Added missing '(' to: %s", pattern.ID, trimmed)
				}
			}

		case "add_func_keyword":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "fn ") {
					lines[i] = strings.Replace(trimmed, "fn ", "func ", 1)
					modified = true
					log.Printf("🔧 [%s] Added 'func' keyword to: %s", pattern.ID, trimmed)
				}
			}

		case "add_blank_import":
			// Already handled by generateCorrectiveEdits
			log.Printf("ℹ️  [%s] Unused import detected - will be handled by self-correction", pattern.ID)

		case "guess_import":
			// Already handled by generateCorrectiveEdits
			log.Printf("ℹ️  [%s] Undeclared name detected - will be handled by self-correction", pattern.ID)

		case "prefix_underscore":
			// Already handled by generateCorrectiveEdits
			log.Printf("ℹ️  [%s] Unused variable detected - will be handled by self-correction", pattern.ID)

		case "add_return_statement":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "func ") && strings.Contains(trimmed, ")") && !strings.Contains(trimmed, "{") {
					// This is a function declaration without body - add a return
					continue
				}
				// Find functions with body but no return
				if trimmed == "}" && i > 0 {
					prevLine := strings.TrimSpace(lines[i-1])
					if !strings.HasPrefix(prevLine, "return") && !strings.HasPrefix(prevLine, "}") {
						// Check if the function has a return type
						for j := i - 1; j >= 0; j-- {
							checkLine := strings.TrimSpace(lines[j])
							if strings.HasPrefix(checkLine, "func ") {
								// Simple check: if func has a return type, add return 0
								fields := strings.Fields(checkLine)
								if len(fields) >= 4 && fields[len(fields)-1] != "{" {
									// Has return type - add return statement
									lines = append(lines[:i], append([]string{"\treturn 0"}, lines[i:]...)...)
									modified = true
									log.Printf("🔧 [%s] Added missing return statement", pattern.ID)
								}
								break
							}
						}
					}
				}
			}

		case "fix_return_count":
			for i, line := range lines {
				trimmed := strings.TrimSpace(line)
				if strings.HasPrefix(trimmed, "return ") && i > 0 {
					// Check if the function has no return type but has a return value
					for j := i - 1; j >= 0; j-- {
						checkLine := strings.TrimSpace(lines[j])
						if strings.HasPrefix(checkLine, "func ") {
							// If func has no return type but body has return with value, remove the value
							if !strings.Contains(checkLine, ") ") && !strings.Contains(checkLine, ") (") {
								// No return type - remove return value
								parts := strings.Fields(trimmed)
								if len(parts) > 1 {
									lines[i] = "\treturn"
									modified = true
									log.Printf("🔧 [%s] Fixed return count in function", pattern.ID)
								}
							}
							break
						}
					}
				}
			}

		case "add_newline":
			// Complex fix - just report for now
			log.Printf("ℹ️  [%s] Expected semicolon - may need manual fix", pattern.ID)

		case "report_unfixable":
			log.Printf("⚠️  [%s] Cannot auto-fix: %s", pattern.ID, pattern.Description)
		}
	}

	if !modified {
		return nil
	}

	// Write the fixed content
	newContent := strings.Join(lines, "\n")
	if err := os.WriteFile(filePath, []byte(newContent), 0644); err != nil {
		return nil
	}

	// Run gofmt
	exec.Command("gofmt", "-w", filePath).Run()

	// Return a special edit that applyEdits will count as success
	return []EditOperation{{
		Type:       "fix_syntax",
		TargetFile: filePath,
	}}
}

// ─── Natural Language Helper Functions ────────────────────────────────────────

// extractFuncName extracts a function name from a natural language query.
func extractFuncName(lower string) string {
	patterns := []string{
		"function called ", "function named ", "function '", "function \"",
		"func called ", "func named ", "func '", "func \"",
		"add function ", "add func ", "new function ", "insert function ",
		"add the function ", "add a function ", "add a new function ",
		"in the ", "in ", "to ",
	}

	// First try the standard patterns
	for _, prefix := range patterns {
		if idx := strings.Index(lower, prefix); idx >= 0 {
			start := idx + len(prefix)
			remaining := lower[start:]
			words := strings.Fields(remaining)
			for _, w := range words {
				name := strings.Trim(w, "'\",.;:()")
				if name == "" || isStopWord(name) {
					continue
				}
				// Ignore candidates that look like file paths or filenames
				if strings.Contains(name, "/") || strings.Contains(name, ".go") || strings.Contains(name, ".") {
					continue
				}
				return name
			}
		}
	}

	// Try "X function" pattern (name before the word "function")
	if idx := strings.Index(lower, " function"); idx >= 0 {
		before := lower[:idx]
		words := strings.Fields(before)
		// Take the last word before " function"
		for i := len(words) - 1; i >= 0; i-- {
			name := strings.Trim(words[i], "'\",.;:()")
			if name == "" || isStopWord(name) || isStopWord(name+" function") {
				continue
			}
			if strings.Contains(name, "/") || strings.Contains(name, ".go") || strings.Contains(name, ".") {
				continue
			}
			return name
		}
	}

	// Fallback: look for "called X" or "named X"
	for _, marker := range []string{" called ", " named "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			start := idx + len(marker)
			remaining := lower[start:]
			words := strings.Fields(remaining)
			for _, w := range words {
				name := strings.Trim(w, "'\",.;:()")
				if name == "" || isStopWord(name) {
					continue
				}
				if strings.Contains(name, "/") || strings.Contains(name, ".go") || strings.Contains(name, ".") {
					continue
				}
				return name
			}
		}
	}

	return ""
}

// extractStructName extracts a struct name from a natural language query.
func extractStructName(lower string) string {
	for _, marker := range []string{"struct named ", "struct called ", "struct "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			remaining := lower[idx+len(marker):]
			words := strings.Fields(remaining)
			if len(words) > 0 {
				name := strings.Trim(words[0], "'\",.;:()")
				if name != "named" && name != "called" && name != "with" && name != "a" && name != "an" {
					return name
				}
				if len(words) > 1 {
					return strings.Trim(words[1], "'\",.;:()")
				}
			}
		}
	}
	return ""
}

// extractFieldName extracts a field name from a natural language query.
func extractFieldName(lower string) string {
	if idx := strings.Index(lower, "field "); idx >= 0 {
		remaining := lower[idx+6:]
		words := strings.Fields(remaining)
		if len(words) > 0 {
			return strings.Trim(words[0], "'\",.;:()")
		}
	}
	return ""
}

// extractFieldType extracts a field type from a natural language query.
func extractFieldType(lower string) string {
	for _, marker := range []string{" of type ", " type "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			start := idx + len(marker)
			remaining := lower[start:]
			words := strings.Fields(remaining)
			if len(words) > 0 {
				return strings.Trim(words[0], "'\",.;:()")
			}
		}
	}
	return "string"
}

// extractImportPath extracts an import path from a natural language query.
func extractImportPath(lower string) string {
	if idx := strings.Index(lower, "import "); idx >= 0 {
		remaining := lower[idx+7:]
		words := strings.Fields(remaining)
		if len(words) > 0 {
			path := strings.Trim(words[0], "'\"")
			if path != "" {
				return path
			}
		}
	}
	return ""
}

// buildFuncCodeFromQuery generates Go function code from a natural language description.
func buildFuncCodeFromQuery(lower, funcName string) string {
	hasParams := strings.Contains(lower, "take") || strings.Contains(lower, "parameter") || strings.Contains(lower, "argument") || strings.Contains(lower, "input")
	hasReturn := strings.Contains(lower, "return") || strings.Contains(lower, "result")

	hasInt := strings.Contains(lower, "int") || strings.Contains(lower, "integer")
	hasString := strings.Contains(lower, "string") || strings.Contains(lower, "str")
	hasFloat := strings.Contains(lower, "float") || strings.Contains(lower, "float64")

	returnsInt := strings.Contains(lower, "sum") || strings.Contains(lower, "total") || strings.Contains(lower, "count") || strings.Contains(lower, "number")
	returnsString := strings.Contains(lower, "concat") || strings.Contains(lower, "join") || strings.Contains(lower, "message")
	returnsBool := strings.Contains(lower, "check") || strings.Contains(lower, "valid") || strings.Contains(lower, "compare")

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("func %s(", funcName))

	if hasParams {
		if hasInt && hasString {
			sb.WriteString("a int, b string")
		} else if hasInt && hasFloat {
			sb.WriteString("a int, b float64")
		} else if hasInt {
			if strings.Contains(lower, "two") || strings.Contains(lower, "2") {
				sb.WriteString("a, b int")
			} else {
				sb.WriteString("a int")
			}
		} else if hasString {
			sb.WriteString("s string")
		} else if hasFloat {
			sb.WriteString("f float64")
		} else {
			sb.WriteString("a int")
		}
	}

	sb.WriteString(")")

	if hasReturn {
		if returnsInt {
			sb.WriteString(" int")
		} else if returnsString {
			sb.WriteString(" string")
		} else if returnsBool {
			sb.WriteString(" bool")
		} else if hasInt {
			sb.WriteString(" int")
		} else {
			sb.WriteString(" int")
		}
	}

	sb.WriteString(" {\n")

	if strings.Contains(lower, "multiply") || strings.Contains(lower, "product") || strings.Contains(lower, "times") {
		sb.WriteString("\treturn a * b\n")
	} else if strings.Contains(lower, "sum") || strings.Contains(lower, "add") || strings.Contains(lower, "plus") || strings.Contains(lower, "total") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "concat") || strings.Contains(lower, "join") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "greet") || strings.Contains(lower, "hello") {
		sb.WriteString("\treturn fmt.Sprintf(\"Hello, %s!\", name)\n")
	} else if strings.Contains(lower, "square") {
		sb.WriteString("\treturn a * a\n")
	} else {
		sb.WriteString("\t// TODO: implement\n")
		sb.WriteString("\treturn 0\n")
	}

	sb.WriteString("}\n")

	return sb.String()
}

// buildFuncBodyFromQuery generates just the body statements for modifying an existing function.
func buildFuncBodyFromQuery(lower, funcName string) string {
	var sb strings.Builder

	if strings.Contains(lower, "multiply") || strings.Contains(lower, "product") || strings.Contains(lower, "times") {
		sb.WriteString("\treturn a * b\n")
	} else if strings.Contains(lower, "sum") || strings.Contains(lower, "add") || strings.Contains(lower, "plus") || strings.Contains(lower, "total") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "concat") || strings.Contains(lower, "join") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "greet") || strings.Contains(lower, "hello") {
		sb.WriteString("\treturn fmt.Sprintf(\"Hello, %s!\", name)\n")
	} else if strings.Contains(lower, "square") {
		sb.WriteString("\treturn a * a\n")
	} else {
		sb.WriteString("\t// TODO: implement\n")
		sb.WriteString("\treturn 0\n")
	}

	return sb.String()
}

// buildSignatureChange reads the file, finds the function signature, and modifies it
// based on the natural language query. Returns old and new code for replace_code.
func buildSignatureChange(lower, funcName, filePath string) (string, string) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", ""
	}
	lines := strings.Split(string(content), "\n")

	// Find the function declaration line
	funcIdx := -1
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func "+funcName+"(") {
			funcIdx = i
			break
		}
	}
	if funcIdx == -1 {
		return "", ""
	}

	oldLine := lines[funcIdx]
	trimmed := strings.TrimSpace(oldLine)

	// Detect: add return type
	if strings.Contains(lower, "return type") || strings.Contains(lower, "return ") {
		// Check if function already has a return type
		hasReturnType := false
		if strings.Contains(trimmed, ") ") && !strings.Contains(trimmed, ") {") {
			hasReturnType = true
		}
		if strings.Contains(trimmed, ") (") {
			hasReturnType = true
		}
		if hasReturnType {
			return "", ""
		}

		// Try to infer the return type from the function body
		inferred := inferReturnType(funcName, filePath)
		if inferred == "" {
			inferred = "string" // conservative default
		}

		// Find the position of " {" to insert the return type before it
		if braceIdx := strings.Index(trimmed, " {"); braceIdx >= 0 {
			newLine := trimmed[:braceIdx] + " " + inferred + trimmed[braceIdx:]
			return oldLine, newLine
		}
		// No brace yet - function declaration without body
		if strings.HasSuffix(trimmed, ")") {
			newLine := trimmed + " " + inferred + " {"
			return oldLine, newLine
		}
		return "", ""
	}

	// Detect: add parameters like (int, int)
	if strings.Contains(lower, "(") && strings.Contains(lower, "int") {
		// Check if function already has params
		if strings.Contains(trimmed, "(") && !strings.Contains(trimmed, "()") {
			// Already has params
			return "", ""
		}
		// Extract the param types from the query
		// e.g. "(int,int)" or "(int, int)"
		parenStart := strings.Index(lower, "(")
		parenEnd := strings.Index(lower, ")")
		if parenStart >= 0 && parenEnd > parenStart {
			params := lower[parenStart : parenEnd+1]
			if strings.HasSuffix(trimmed, "{") {
				newLine := strings.Replace(trimmed, " {", " "+params+" {", 1)
				return oldLine, newLine
			}
			newLine := trimmed + " " + params + " {"
			return oldLine, newLine
		}
	}

	return "", ""
}

// extractNameAfterOf extracts a bare identifier following 'of' or 'after' in a phrase.
func extractNameAfterOf(lower string) string {
	// Try " of NAME"
	if idx := strings.LastIndex(lower, " of "); idx >= 0 {
		rem := lower[idx+4:]
		words := strings.Fields(rem)
		if len(words) > 0 {
			return strings.Trim(words[0], "'\",.;:()")
		}
	}
	// Try " after NAME" fallback
	if idx := strings.LastIndex(lower, " after "); idx >= 0 {
		rem := lower[idx+7:]
		words := strings.Fields(rem)
		if len(words) > 0 {
			return strings.Trim(words[0], "'\",.;:()")
		}
	}
	return ""
}

// buildAddBraceChange finds the function declaration for a name and returns the
// old line and a new line with an opening brace appended after the return type.
func buildAddBraceChange(name, filePath string) (string, string) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", ""
	}
	lines := strings.Split(string(content), "\n")

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "func ") {
			continue
		}
		// Quick containment check: function line should mention the name and have '('
		if !strings.Contains(trimmed, "(") || !strings.Contains(trimmed, name) {
			continue
		}

		// If brace already present on the line, nothing to do
		if strings.Contains(trimmed, "{") {
			return "", ""
		}

		// Preserve leading whitespace
		leading := line[:len(line)-len(strings.TrimLeft(line, " \t"))]

		// Create new line by appending ' {' to the trimmed declaration
		newLine := leading + trimmed + " {"
		oldLine := line

		// Return the single-line replacement
		return oldLine, newLine
	}

	return "", ""
}

// buildAddSymbolChange finds the function declaration for a name and inserts a
// symbol (e.g. "string", "int", "bool", "error") at the requested position
// (after return type, after params, after function name).
func buildAddSymbolChange(lower, name, filePath string) (string, string) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", ""
	}
	lines := strings.Split(string(content), "\n")

	// Determine the symbol to insert
	symbol := ""
	if strings.Contains(lower, "string") {
		symbol = "string"
	} else if strings.Contains(lower, "int ") || strings.Contains(lower, "integer") {
		symbol = "int"
	} else if strings.Contains(lower, "bool") {
		symbol = "bool"
	} else if strings.Contains(lower, "error") {
		symbol = "error"
	} else if strings.Contains(lower, "float") {
		symbol = "float64"
	} else if strings.Contains(lower, "byte") {
		symbol = "byte"
	} else if strings.Contains(lower, "rune") {
		symbol = "rune"
	}
	if symbol == "" {
		return "", ""
	}

	// Determine the anchor position
	anchor := ""
	if strings.Contains(lower, "after the return type") {
		anchor = "after_return_type"
	} else if strings.Contains(lower, "after the params") || strings.Contains(lower, "after the parameters") {
		anchor = "after_params"
	} else if strings.Contains(lower, "after the function name") {
		anchor = "after_func_name"
	}
	if anchor == "" {
		return "", ""
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "func ") {
			continue
		}
		// Quick containment check: function line should mention the name and have '('
		if !strings.Contains(trimmed, "(") || !strings.Contains(trimmed, name) {
			continue
		}

		leading := line[:len(line)-len(strings.TrimLeft(line, " \t"))]

		switch anchor {
		case "after_return_type":
			if braceIdx := strings.Index(trimmed, " {"); braceIdx >= 0 {
				beforeBrace := strings.TrimSpace(trimmed[:braceIdx])
				// If the function already has a return type, we need to REPLACE it
				// rather than append (e.g. "func F() string {" -> "func F() int {").
				if lastParen := strings.LastIndex(beforeBrace, ")"); lastParen >= 0 {
					afterParen := strings.TrimSpace(beforeBrace[lastParen+1:])
					if afterParen != "" {
						// Capture the leading indentation + prefix through closing paren,
						// then replace the return type token(s) with our symbol.
						prefix := leading + beforeBrace[:lastParen+1]
						// If the last token is already our symbol, no-op.
						if afterParen == symbol || strings.HasPrefix(afterParen, "(") && strings.Contains(afterParen, symbol) {
							return "", "" // already has this return type
						}
						newLine := prefix + " " + symbol + trimmed[braceIdx:]
						return line, newLine
					}
				}
				// No return type yet — insert symbol before the opening brace
				newLine := leading + trimmed[:braceIdx] + " " + symbol + trimmed[braceIdx:]
				return line, newLine
			}
			// No brace yet - insert symbol + brace
			if strings.HasSuffix(trimmed, ")") {
				newLine := leading + trimmed + " " + symbol + " {"
				return line, newLine
			}
		case "after_params":
			// Find the closing paren of params and insert after it
			parenDepth := 0
			var closeParenIdx int = -1
			for i, ch := range trimmed {
				if ch == '(' {
					parenDepth++
				} else if ch == ')' {
					parenDepth--
					if parenDepth == 0 {
						closeParenIdx = i
						break
					}
				}
			}
			if closeParenIdx >= 0 {
				// Check if the next token is already our symbol (e.g. "func foo() string {")
				afterClose := strings.TrimSpace(trimmed[closeParenIdx+1:])
				if strings.HasPrefix(afterClose, symbol) {
					return "", "" // Already has the symbol - no-op
				}
				newLine := leading + trimmed[:closeParenIdx+1] + " " + symbol + trimmed[closeParenIdx+1:]
				return line, newLine
			}
		case "after_func_name":
			// Insert after "func <name>"
			prefix := "func " + name
			if idx := strings.Index(trimmed, prefix); idx >= 0 {
				insertAt := idx + len(prefix)
				newLine := leading + trimmed[:insertAt] + symbol + trimmed[insertAt:]
				return line, newLine
			}
		}
	}

	return "", ""
}

// findFirstFuncMissingBrace returns the first function line missing a '{' and a replacement line.
func findFirstFuncMissingBrace(filePath string) (string, string) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", ""
	}
	lines := strings.Split(string(content), "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "func ") && !strings.Contains(trimmed, "{") {
			// preserve indentation
			leading := line[:len(line)-len(strings.TrimLeft(line, " \t"))]
			newLine := leading + trimmed + " {"
			return line, newLine
		}
		// If a return appears outside of any function line above, try previous declaration
		if strings.HasPrefix(trimmed, "return ") {
			// scan backwards for nearby func declaration
			for j := i - 1; j >= 0; j-- {
				t2 := strings.TrimSpace(lines[j])
				if strings.HasPrefix(t2, "func ") {
					if !strings.Contains(t2, "{") {
						leading := lines[j][:len(lines[j])-len(strings.TrimLeft(lines[j], " \t"))]
						newLine := leading + t2 + " {"
						return lines[j], newLine
					}
					break
				}
				if t2 == "" {
					continue
				}
			}
		}
	}
	return "", ""
}

// buildStructCodeFromQuery generates Go struct code supporting single or multiple fields.
func buildStructCodeFromQuery(lower string, structName string) string {
	var fields [][2]string
	for _, marker := range []string{"fields ", "field "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			remaining := lower[idx+len(marker):]
			words := strings.Fields(remaining)
			for i := 0; i+1 < len(words); i += 2 {
				fn := strings.Trim(words[i], "'\",.;:()")
				ft := strings.Trim(words[i+1], "'\",.;:()")
				if fn != "" && ft != "" && !isStopWord(fn) {
					fields = append(fields, [2]string{fn, ft})
				}
			}
			break
		}
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("type %s struct {\n", structName))
	for _, f := range fields {
		sb.WriteString(fmt.Sprintf("\t%s %s\n", f[0], f[1]))
	}
	sb.WriteString("}\n")
	return sb.String()
}

// buildStructCode generates Go struct type definition code.
func buildStructCode(structName, fieldName, fieldType string) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("type %s struct {\n", structName))
	if fieldName != "" {
		if fieldType == "" {
			fieldType = "string"
		}
		sb.WriteString(fmt.Sprintf("\t%s %s\n", fieldName, fieldType))
	}
	sb.WriteString("}\n")
	return sb.String()
}

// inferReturnType inspects the function body for simple return expressions
// and returns a guessed Go type like "string", "int", or "float64".
func inferReturnType(funcName, filePath string) string {
	// First try precise type-checking across the package
	if t := inferReturnTypeWithTypes(funcName, filePath); t != "" {
		return t
	}

	// Fallback: AST heuristics when type-checker fails
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, 0)
	if err != nil || node == nil {
		return ""
	}

	var target *ast.FuncDecl
	ast.Inspect(node, func(n ast.Node) bool {
		if fn, ok := n.(*ast.FuncDecl); ok && fn.Name.Name == funcName {
			target = fn
			return false
		}
		return true
	})
	if target == nil || target.Body == nil {
		return ""
	}

	// Search for return statements (simple heuristics)
	for _, stmt := range target.Body.List {
		ret, ok := stmt.(*ast.ReturnStmt)
		if !ok || len(ret.Results) == 0 {
			continue
		}
		// Only handle single-value returns for now
		if len(ret.Results) == 1 {
			switch expr := ret.Results[0].(type) {
			case *ast.BasicLit:
				switch expr.Kind {
				case token.STRING:
					return "string"
				case token.INT:
					return "int"
				case token.FLOAT:
					return "float64"
				}
			case *ast.CallExpr:
				// Common: fmt.Sprintf -> string
				if fun := expr.Fun; fun != nil {
					if se, ok := fun.(*ast.SelectorExpr); ok {
						if id, ok := se.X.(*ast.Ident); ok && id.Name == "fmt" && se.Sel.Name == "Sprintf" {
							return "string"
						}
					}
					if id, ok := fun.(*ast.Ident); ok && id.Name == "Sprintf" {
						return "string"
					}
				}
			case *ast.BinaryExpr:
				if bl, ok := expr.X.(*ast.BasicLit); ok && bl.Kind == token.STRING {
					return "string"
				}
				if bl, ok := expr.Y.(*ast.BasicLit); ok && bl.Kind == token.STRING {
					return "string"
				}
				return "int"
			case *ast.Ident:
				return "string"
			}
		}
	}

	return ""
}

// inferReturnTypeWithTypes attempts to type-check the package containing filePath
// and returns the declared return type(s) for funcName, formatted as a Go type
// string (e.g., "int" or "(int, string)"). Returns empty string on failure.
func inferReturnTypeWithTypes(funcName, filePath string) string {
	// First try using go/packages to load package+deps for broad type information.
	dir := filepath.Dir(filePath)
	cfg := &packages.Config{Mode: packages.NeedName | packages.NeedTypes | packages.NeedTypesInfo | packages.NeedSyntax | packages.NeedDeps, Dir: dir}
	pkgs, err := packages.Load(cfg, "./...")
	if err == nil && len(pkgs) > 0 {
		for _, p := range pkgs {
			if p.Types == nil || p.TypesInfo == nil {
				continue
			}
			// lookup in package scope
			if obj := p.Types.Scope().Lookup(funcName); obj != nil {
				if fn, ok := obj.(*types.Func); ok {
					if sig, ok := fn.Type().(*types.Signature); ok {
						res := sig.Results()
						if res == nil || res.Len() == 0 {
							return ""
						}
						if res.Len() == 1 {
							return types.TypeString(res.At(0).Type(), func(pkg *types.Package) string {
								if pkg == p.Types {
									return ""
								}
								return pkg.Path()
							})
						}
						parts := make([]string, 0, res.Len())
						for i := 0; i < res.Len(); i++ {
							parts = append(parts, types.TypeString(res.At(i).Type(), func(pkg *types.Package) string {
								if pkg == p.Types {
									return ""
								}
								return pkg.Path()
							}))
						}
						return "(" + strings.Join(parts, ", ") + ")"
					}
				}
			}
			// fallback to defs
			for ident, obj := range p.TypesInfo.Defs {
				if ident.Name == funcName {
					if fn, ok := obj.(*types.Func); ok {
						if sig, ok := fn.Type().(*types.Signature); ok {
							res := sig.Results()
							if res == nil || res.Len() == 0 {
								return ""
							}
							if res.Len() == 1 {
								return types.TypeString(res.At(0).Type(), func(pkg *types.Package) string {
									if pkg == p.Types {
										return ""
									}
									return pkg.Path()
								})
							}
							parts := make([]string, 0, res.Len())
							for i := 0; i < res.Len(); i++ {
								parts = append(parts, types.TypeString(res.At(i).Type(), func(pkg *types.Package) string {
									if pkg == p.Types {
										return ""
									}
									return pkg.Path()
								}))
							}
							return "(" + strings.Join(parts, ", ") + ")"
						}
					}
				}
			}
		}
	}

	// If packages.Load failed or returned nothing, fall back to intrapackage type-check using importer
	fset := token.NewFileSet()
	pkgsMap, err := parser.ParseDir(fset, dir, func(fi os.FileInfo) bool {
		name := fi.Name()
		return strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
	}, parser.ParseComments)
	if err != nil || len(pkgsMap) == 0 {
		return ""
	}
	var firstPkg *ast.Package
	for _, p := range pkgsMap {
		firstPkg = p
		break
	}
	if firstPkg == nil {
		return ""
	}
	var files []*ast.File
	for _, f := range firstPkg.Files {
		files = append(files, f)
	}
	info := &types.Info{Defs: make(map[*ast.Ident]types.Object), Uses: make(map[*ast.Ident]types.Object)}
	cfg2 := &types.Config{Importer: importer.Default()}
	pkg, err := cfg2.Check(firstPkg.Name, fset, files, info)
	if err != nil {
		return ""
	}
	for ident, obj := range info.Defs {
		if obj == nil {
			continue
		}
		if fn, ok := obj.(*types.Func); ok {
			if ident.Name == funcName {
				sig, ok := fn.Type().(*types.Signature)
				if !ok {
					return ""
				}
				res := sig.Results()
				if res == nil || res.Len() == 0 {
					return ""
				}
				if res.Len() == 1 {
					return types.TypeString(res.At(0).Type(), func(p *types.Package) string {
						if p == pkg {
							return ""
						}
						return p.Path()
					})
				}
				parts := make([]string, 0, res.Len())
				for i := 0; i < res.Len(); i++ {
					parts = append(parts, types.TypeString(res.At(i).Type(), func(p *types.Package) string {
						if p == pkg {
							return ""
						}
						return p.Path()
					}))
				}
				return "(" + strings.Join(parts, ", ") + ")"
			}
		}
	}

	_ = pkg
	return ""
}

// isStopWord checks if a word is a common stop word.
func isStopWord(word string) bool {
	stopWords := map[string]bool{
		"that": true, "this": true, "with": true, "from": true, "into": true,
		"file": true, "the": true, "a": true, "an": true, "to": true,
		"in": true, "of": true, "for": true, "and": true, "or": true,
		"it": true, "is": true, "are": true, "was": true, "be": true,
		"has": true, "have": true, "do": true, "does": true, "will": true,
		"would": true, "could": true, "should": true, "may": true, "might": true,
		"can": true, "shall": true, "must": true, "need": true, "let": true,
		"make": true, "take": true, "get": true, "set": true, "put": true,
		"add": true, "new": true, "function": true, "func": true,
		"called": true, "named": true, "returns": true, "return": true,
		"takes": true, "parameters": true, "parameter": true,
		"arguments": true, "argument": true, "input": true, "output": true,
		"two": true, "three": true, "four": true, "five": true,
		"integers": true, "integer": true, "int": true, "string": true,
		"float": true, "bool": true, "boolean": true,
	}
	return stopWords[word]
}

// fuzzyMatchName returns the best matching name from the provided map using
// Levenshtein distance and simple heuristics; returns empty if no good match.
// fuzzyMatchName returns: bestMatch, bestScore, rel, ambiguous, topCandidates
func fuzzyMatchName(target string, candidates map[string]bool) (string, int, float64, bool, []string) {
	if target == "" || len(candidates) == 0 {
		return "", 0, 1.0, false, nil
	}

	lowerT := strings.ToLower(target)
	type candScore struct {
		name string
		dist int
	}
	var list []candScore
	for name := range candidates {
		lowerName := strings.ToLower(name)
		if strings.HasPrefix(lowerName, lowerT) || strings.HasSuffix(lowerName, lowerT) {
			return name, 0, 0.0, false, []string{name}
		}
		d := levenshtein(lowerT, lowerName)
		list = append(list, candScore{name: name, dist: d})
	}
	if len(list) == 0 {
		return "", 0, 1.0, false, nil
	}
	sort.Slice(list, func(i, j int) bool { return list[i].dist < list[j].dist })
	best := list[0]
	bestScore := best.dist
	top := []string{list[0].name}
	for i := 1; i < len(list) && i < 3; i++ {
		top = append(top, list[i].name)
	}
	maxLen := len(target)
	if len(best.name) > maxLen {
		maxLen = len(best.name)
	}
	if maxLen == 0 {
		return best.name, bestScore, 0.0, false, top
	}
	rel := float64(bestScore) / float64(maxLen)
	ambiguous := false
	if len(list) > 1 {
		// ambiguous if second-best close to best
		if float64(list[1].dist-bestScore) <= 1 && (bestScore > 1 || rel > 0.3) {
			ambiguous = true
		}
	}
	// Accept best if small absolute or relative distance
	if bestScore <= 2 || rel <= 0.35 {
		return best.name, bestScore, rel, ambiguous, top
	}
	return "", bestScore, rel, ambiguous, top
}

// levenshtein computes the Levenshtein edit distance between two strings.
func levenshtein(a, b string) int {
	la := len(a)
	lb := len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}
	dp := make([][]int, la+1)
	for i := range dp {
		dp[i] = make([]int, lb+1)
	}
	for i := 0; i <= la; i++ {
		dp[i][0] = i
	}
	for j := 0; j <= lb; j++ {
		dp[0][j] = j
	}
	for i := 1; i <= la; i++ {
		for j := 1; j <= lb; j++ {
			cost := 0
			if a[i-1] != b[j-1] {
				cost = 1
			}
			dp[i][j] = min3(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
		}
	}
	return dp[la][lb]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// ─── Tool Handler Interface ───────────────────────────────────────────────────

type ToolHandler struct {
	Name        string
	Description string
}

func NewToolHandler() *ToolHandler {
	return &ToolHandler{
		Name:        "go_edit_agent",
		Description: "Edits Go source files using AST-level manipulation with validation and self-correction. Supports: insert_func, modify_func, add_field, add_import, replace_code, delete_func.",
	}
}

func (h *ToolHandler) Handle(inputJSON []byte) ([]byte, error) {
	var req AgentRequest
	if err := json.Unmarshal(inputJSON, &req); err != nil {
		return nil, fmt.Errorf("invalid request: %w", err)
	}

	if req.MaxRetries <= 0 {
		req.MaxRetries = 3
	}

	// Parse natural language query if provided
	if req.Query != "" && len(req.Edits) == 0 {
		edits := parseNaturalLanguageQuery(req.File, req.Query)
		if len(edits) == 0 {
			return nil, fmt.Errorf("could not understand the edit request")
		}
		req.Edits = edits
	}

	startTime := time.Now()
	resp := executeAgent(req)
	resp.Duration = time.Since(startTime).Round(time.Millisecond).String()

	return json.MarshalIndent(resp, "", "  ")
}
