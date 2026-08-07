// Package errors provides an MoE-based error intent classification and auto-fix system.
// It takes raw compiler/linter output, classifies the error intent through the
// trained MoE classification model, and triggers the appropriate AST-based fixer.
package errors

import (
	"fmt"
	"regexp"
	"strings"
)

// ErrorIntent represents the classified intent of a compiler/linter error.
type ErrorIntent int

const (
	// IntentUnknown is the fallback when no intent can be determined.
	IntentUnknown ErrorIntent = iota
	// IntentUndefinedSymbol indicates a reference to an undefined symbol (variable, function, type).
	IntentUndefinedSymbol
	// IntentMissingImport indicates a missing import for a referenced package.
	IntentMissingImport
	// IntentMissingHandlerDefinition indicates a handler function is referenced but not defined.
	IntentMissingHandlerDefinition
	// IntentMissingFunctionBody indicates a function declaration exists but has no body.
	IntentMissingFunctionBody
	// IntentTypeMismatch indicates a type assignment or argument type mismatch.
	IntentTypeMismatch
	// IntentUnusedVariable indicates a declared variable that is not used.
	IntentUnusedVariable
	// IntentUnusedImport indicates an imported package that is not used.
	IntentUnusedImport
	// IntentMissingReturn indicates a function is missing a return statement.
	IntentMissingReturn
	// IntentSyntaxError indicates a general syntax error in the source.
	IntentSyntaxError
	// IntentMissingMethod indicates a type is missing a required method implementation.
	IntentMissingMethod
	// IntentUndeclaredName indicates a name used before declaration.
	IntentUndeclaredName
	// IntentPackageNotImported indicates a package path that is not imported.
	IntentPackageNotImported
	// IntentInvalidReceiver indicates an invalid method receiver type.
	IntentInvalidReceiver
	// IntentCannotAssign indicates an assignment to an unassignable value.
	IntentCannotAssign
	// IntentNonBoolUsedInIf indicates a non-boolean expression used in an if condition.
	IntentNonBoolUsedInIf
	// IntentAssignmentMismatch indicates an assignment mismatch error.
	IntentAssignmentMismatch
	// IntentInvalidBinaryOp indicates mismatched types in a binary operation.
	IntentInvalidBinaryOp
	// IntentNoNewVariables indicates no new variables on left side of :=.
	IntentNoNewVariables
	// IntentTooManyArgs indicates too many arguments in a function call.
	IntentTooManyArgs
	// IntentNotEnoughArgs indicates not enough arguments in a function call.
	IntentNotEnoughArgs
	// IntentCallNonFunction indicates calling a non-function value.
	IntentCallNonFunction
	// IntentCannotRange indicates a value that cannot be ranged over.
	IntentCannotRange
	// IntentCannotTakeAddress indicates a value whose address cannot be taken.
	IntentCannotTakeAddress
	// IntentInvalidIndirect indicates an invalid indirect operation.
	IntentInvalidIndirect
	// IntentDuplicateField indicates a duplicate field in a struct literal.
	IntentDuplicateField
	// IntentDuplicateKey indicates a duplicate key in a map literal.
	IntentDuplicateKey
	// IntentInvalidTypeAssertion indicates an invalid type assertion.
	IntentInvalidTypeAssertion
	// IntentInvalidSend indicates an invalid send on a channel.
	IntentInvalidSend
	// IntentInvalidReceive indicates an invalid receive from a channel.
	IntentInvalidReceive
	// IntentInvalidClose indicates an invalid close of a non-channel.
	IntentInvalidClose
	// IntentInvalidGoStmt indicates an invalid go statement.
	IntentInvalidGoStmt
	// IntentInvalidDeferStmt indicates an invalid defer statement.
	IntentInvalidDeferStmt
	// IntentInvalidFallthrough indicates an invalid fallthrough statement.
	IntentInvalidFallthrough
	// IntentInvalidBreak indicates an invalid break statement.
	IntentInvalidBreak
	// IntentInvalidContinue indicates an invalid continue statement.
	IntentInvalidContinue
	// IntentInvalidGoto indicates an invalid goto statement.
	IntentInvalidGoto
	// IntentInvalidCompositeLit indicates an invalid composite literal.
	IntentInvalidCompositeLit
	// IntentInvalidFunctionLit indicates an invalid function literal.
	IntentInvalidFunctionLit
	// IntentInvalidTypeSwitch indicates an invalid type switch.
	IntentInvalidTypeSwitch
	// IntentInvalidSelectStmt indicates an invalid select statement.
	IntentInvalidSelectStmt
	// IntentInvalidForStmt indicates an invalid for statement.
	IntentInvalidForStmt
	// IntentInvalidIfStmt indicates an invalid if statement.
	IntentInvalidIfStmt
	// IntentInvalidSwitchStmt indicates an invalid switch statement.
	IntentInvalidSwitchStmt
	// IntentInvalidTypeDecl indicates an invalid type declaration.
	IntentInvalidTypeDecl
	// IntentInvalidFuncDecl indicates an invalid function declaration.
	IntentInvalidFuncDecl
	// IntentInvalidMethodDecl indicates an invalid method declaration.
	IntentInvalidMethodDecl
	// IntentInvalidInterfaceDecl indicates an invalid interface declaration.
	IntentInvalidInterfaceDecl
	// IntentInvalidStructDecl indicates an invalid struct declaration.
	IntentInvalidStructDecl
	// IntentInvalidConstDecl indicates an invalid const declaration.
	IntentInvalidConstDecl
	// IntentInvalidVarDecl indicates an invalid var declaration.
	IntentInvalidVarDecl
	// IntentInvalidImportDecl indicates an invalid import declaration.
	IntentInvalidImportDecl
	// IntentInvalidPackageDecl indicates an invalid package declaration.
	IntentInvalidPackageDecl
	// IntentInvalidUseOfDotDotDot indicates an invalid use of ...
	IntentInvalidUseOfDotDotDot
	// IntentInvalidUseOfBlank indicates an invalid use of _.
	IntentInvalidUseOfBlank
	// IntentInvalidUseOfNil indicates an invalid use of nil.
	IntentInvalidUseOfNil
	// IntentTooManyReturnValues indicates a function returns more values than its signature declares.
	IntentTooManyReturnValues
	// IntentTooFewReturnValues indicates a function returns fewer values than its signature declares.
	IntentTooFewReturnValues
	// IntentWrongReturnType indicates a function returns a value of the wrong type.
	IntentWrongReturnType
)

// String returns a human-readable name for the error intent.
func (e ErrorIntent) String() string {
	switch e {
	case IntentUndefinedSymbol:
		return "UNDEFINED_SYMBOL"
	case IntentMissingImport:
		return "MISSING_IMPORT"
	case IntentMissingHandlerDefinition:
		return "MISSING_HANDLER_DEFINITION"
	case IntentMissingFunctionBody:
		return "MISSING_FUNCTION_BODY"
	case IntentTypeMismatch:
		return "TYPE_MISMATCH"
	case IntentUnusedVariable:
		return "UNUSED_VARIABLE"
	case IntentUnusedImport:
		return "UNUSED_IMPORT"
	case IntentMissingReturn:
		return "MISSING_RETURN"
	case IntentSyntaxError:
		return "SYNTAX_ERROR"
	case IntentMissingMethod:
		return "MISSING_METHOD"
	case IntentUndeclaredName:
		return "UNDECLARED_NAME"
	case IntentPackageNotImported:
		return "PACKAGE_NOT_IMPORTED"
	case IntentInvalidReceiver:
		return "INVALID_RECEIVER"
	case IntentCannotAssign:
		return "CANNOT_ASSIGN"
	case IntentNonBoolUsedInIf:
		return "NON_BOOL_IN_IF"
	case IntentTooManyReturnValues:
		return "TOO_MANY_RETURN_VALUES"
	case IntentTooFewReturnValues:
		return "TOO_FEW_RETURN_VALUES"
	case IntentWrongReturnType:
		return "WRONG_RETURN_TYPE"
	default:
		return "UNKNOWN"
	}
}

// ErrorIntentFromString parses a string into an ErrorIntent.
func ErrorIntentFromString(s string) ErrorIntent {
	switch strings.ToUpper(s) {
	case "UNDEFINED_SYMBOL":
		return IntentUndefinedSymbol
	case "MISSING_IMPORT":
		return IntentMissingImport
	case "MISSING_HANDLER_DEFINITION":
		return IntentMissingHandlerDefinition
	case "MISSING_FUNCTION_BODY":
		return IntentMissingFunctionBody
	case "TYPE_MISMATCH":
		return IntentTypeMismatch
	case "UNUSED_VARIABLE":
		return IntentUnusedVariable
	case "UNUSED_IMPORT":
		return IntentUnusedImport
	case "MISSING_RETURN":
		return IntentMissingReturn
	case "SYNTAX_ERROR":
		return IntentSyntaxError
	case "MISSING_METHOD":
		return IntentMissingMethod
	case "UNDECLARED_NAME":
		return IntentUndeclaredName
	case "PACKAGE_NOT_IMPORTED":
		return IntentPackageNotImported
	case "INVALID_RECEIVER":
		return IntentInvalidReceiver
	case "CANNOT_ASSIGN":
		return IntentCannotAssign
	case "NON_BOOL_IN_IF":
		return IntentNonBoolUsedInIf
	case "TOO_MANY_RETURN_VALUES":
		return IntentTooManyReturnValues
	case "TOO_FEW_RETURN_VALUES":
		return IntentTooFewReturnValues
	case "WRONG_RETURN_TYPE":
		return IntentWrongReturnType
	default:
		return IntentUnknown
	}
}

// ParsedError represents a structured compiler error with location information.
type ParsedError struct {
	// Raw is the original error line from compiler output.
	Raw string
	// File is the source file path extracted from the error.
	File string
	// Line is the line number extracted from the error.
	Line int
	// Column is the column number extracted from the error (0 if not available).
	Column int
	// Message is the error message text without location prefix.
	Message string
	// Symbol is the symbol name referenced in the error (if applicable).
	Symbol string
	// Package is the package path referenced in the error (if applicable).
	Package string
	// Intent is the classified error intent.
	Intent ErrorIntent
	// Confidence is the classification confidence score (0.0-1.0).
	Confidence float64
}

// String returns a formatted representation of the parsed error.
func (pe *ParsedError) String() string {
	return fmt.Sprintf("%s:%d:%d: [%s] %s", pe.File, pe.Line, pe.Column, pe.Intent, pe.Message)
}

// CompilerErrorPatterns defines regex patterns for matching common Go compiler errors.
var CompilerErrorPatterns = []struct {
	Pattern *regexp.Regexp
	Intent  ErrorIntent
	Extract func(matches []string) (file string, line, col int, message, symbol, pkg string)
}{
	// ── Undefined symbol (plain, no dot) ──────────────────────────────────────
	// ./main.go:124:3: undefined: server
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+undefined:\s+(\w+)$`),
		Intent:  IntentUndefinedSymbol,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("undefined: %s", m[4]), m[4], ""
		},
	},
	// ./main.go:124: undefined: server (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+undefined:\s+(\w+)$`),
		Intent:  IntentUndefinedSymbol,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("undefined: %s", m[3]), m[3], ""
		},
	},
	// ── Undefined symbol with package selector (e.g. "undefined: os.NonExistentField") ──
	// ./main.go:124:3: undefined: os.NonExistentField
	// This captures the dotted symbol and splits it into package + field.
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+undefined:\s+(\w+)\.(\w+)$`),
		Intent:  IntentUndefinedSymbol,
		Extract: func(m []string) (string, int, int, string, string, string) {
			pkg := m[4]
			field := m[5]
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("undefined: %s.%s", pkg, field), field, pkg
		},
	},
	// ./main.go:124: undefined: os.NonExistentField (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+undefined:\s+(\w+)\.(\w+)$`),
		Intent:  IntentUndefinedSymbol,
		Extract: func(m []string) (string, int, int, string, string, string) {
			pkg := m[3]
			field := m[4]
			return m[1], atoi(m[2]), 0, fmt.Sprintf("undefined: %s.%s", pkg, field), field, pkg
		},
	},
	// ./main.go:42:2: undeclared name: server
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+undeclared\s+name:\s+(\S+)$`),
		Intent:  IntentUndeclaredName,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("undeclared name: %s", m[4]), m[4], ""
		},
	},
	// ./main.go:18:5: undefined: handler (legacy handler pattern)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+undefined:\s+(\w+)$`),
		Intent:  IntentMissingHandlerDefinition,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("undefined handler: %s", m[4]), m[4], ""
		},
	},
	// ./main.go:10:2: imported and not used: "net/http"
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+imported\s+and\s+not\s+used:\s+"([^"]+)"$`),
		Intent:  IntentUnusedImport,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("unused import: %s", m[4]), "", m[4]
		},
	},
	// ./main.go:10:2: imported and not used: "fmt"
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+imported\s+and\s+not\s+used:\s+"([^"]+)"$`),
		Intent:  IntentUnusedImport,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("unused import: %s", m[3]), "", m[3]
		},
	},
	// ./main.go:15:6: handler declared and not used
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+(\w+)\s+declared\s+(?:and\s+)?not\s+used$`),
		Intent:  IntentUnusedVariable,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("unused: %s", m[4]), m[4], ""
		},
	},
	// ./main.go:15:6: handler declared but not used
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+(\w+)\s+declared\s+(?:but\s+)?not\s+used$`),
		Intent:  IntentUnusedVariable,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("unused: %s", m[3]), m[3], ""
		},
	},
	// ── Type mismatch: "cannot use x (type A) as type B in assignment" ─────────
	// ./main.go:20:6: cannot use x (type string) as type int in assignment
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+use\s+(\S+)\s+\(type\s+(\S+)\)\s+as\s+type\s+(\S+)\s+in`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("type mismatch: cannot use %s as %s", m[4], m[6]), m[4], ""
		},
	},
	// ./main.go:20: cannot use x (type string) as type int in assignment (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+cannot\s+use\s+(\S+)\s+\(type\s+(\S+)\)\s+as\s+type\s+(\S+)\s+in`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("type mismatch: cannot use %s as %s", m[3], m[5]), m[3], ""
		},
	},
	// ── Type mismatch: "cannot use x (type A) as type B value" (no "in ...") ──
	// ./main.go:20:6: cannot use x (type string) as type int value
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+use\s+(\S+)\s+\(type\s+(\S+)\)\s+as\s+type\s+(\S+)\s+value$`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("type mismatch: cannot use %s as %s", m[4], m[6]), m[4], ""
		},
	},
	// ./main.go:20: cannot use x (type string) as type int value (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+cannot\s+use\s+(\S+)\s+\(type\s+(\S+)\)\s+as\s+type\s+(\S+)\s+value$`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("type mismatch: cannot use %s as %s", m[3], m[5]), m[3], ""
		},
	},
	// ── Type mismatch: "cannot use x as type B value in variable declaration" ──
	// (no parenthetical type, e.g. "cannot use "hello" as type int value in variable declaration")
	// ./main.go:20:6: cannot use "hello" as type int value in variable declaration
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+use\s+(\S+)\s+as\s+type\s+(\S+)\s+value\s+in\s+variable\s+declaration$`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("type mismatch: cannot use %s as %s", m[4], m[5]), m[4], ""
		},
	},
	// ./main.go:20: cannot use "hello" as type int value in variable declaration (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+cannot\s+use\s+(\S+)\s+as\s+type\s+(\S+)\s+value\s+in\s+variable\s+declaration$`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("type mismatch: cannot use %s as %s", m[3], m[4]), m[3], ""
		},
	},
	// ── Type mismatch: "cannot use x (type A) as B" (no "type" prefix, no "in") ──
	// ./main.go:20:6: cannot use x (type string) as int
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+use\s+(\S+)\s+\(type\s+(\S+)\)\s+as\s+(\S+)$`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("type mismatch: cannot use %s as %s", m[4], m[6]), m[4], ""
		},
	},
	// ./main.go:20: cannot use x (type string) as int (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+cannot\s+use\s+(\S+)\s+\(type\s+(\S+)\)\s+as\s+(\S+)$`),
		Intent:  IntentTypeMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("type mismatch: cannot use %s as %s", m[3], m[5]), m[3], ""
		},
	},
	// ./main.go:30:1: missing return
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+missing\s+return$`),
		Intent:  IntentMissingReturn,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "missing return", "", ""
		},
	},
	// ./main.go:30: missing return
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+missing\s+return$`),
		Intent:  IntentMissingReturn,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, "missing return", "", ""
		},
	},
	// ./main.go:25:6: foo not declared by package bar
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+(\S+)\s+not\s+declared\s+by\s+package\s+(\S+)$`),
		Intent:  IntentPackageNotImported,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("%s not declared by package %s", m[4], m[5]), m[4], m[5]
		},
	},
	// ./main.go:35:6: cannot assign to x
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+assign\s+to\s+(\S+)$`),
		Intent:  IntentCannotAssign,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("cannot assign to %s", m[4]), m[4], ""
		},
	},
	// ── Non-boolean condition in if statement ──────────────────────────────────
	// ./main.go:40:6: non-boolean condition in if statement
	// (Note: the actual Go compiler says "non-boolean condition in if statement"
	//  without a symbol name, unlike the older "non-bool x used in if" form.)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+non-boolean\s+condition\s+in\s+if\s+statement$`),
		Intent:  IntentNonBoolUsedInIf,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "non-boolean condition in if statement", "", ""
		},
	},
	// ./main.go:40: non-boolean condition in if statement (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+non-boolean\s+condition\s+in\s+if\s+statement$`),
		Intent:  IntentNonBoolUsedInIf,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, "non-boolean condition in if statement", "", ""
		},
	},
	// ./main.go:40:6: non-bool x used in if (older Go form with symbol name)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+non-bool\s+(\S+)\s+used\s+in\s+if$`),
		Intent:  IntentNonBoolUsedInIf,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("non-bool %s used in if", m[4]), m[4], ""
		},
	},
	// ./main.go:40: non-bool x used in if (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+non-bool\s+(\S+)\s+used\s+in\s+if$`),
		Intent:  IntentNonBoolUsedInIf,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("non-bool %s used in if", m[3]), m[3], ""
		},
	},
	// ./main.go:45:1: syntax error: unexpected EOF
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+syntax\s+error:\s+(.+)$`),
		Intent:  IntentSyntaxError,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("syntax error: %s", m[4]), "", ""
		},
	},
	// ./main.go:45: syntax error: unexpected EOF
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+syntax\s+error:\s+(.+)$`),
		Intent:  IntentSyntaxError,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("syntax error: %s", m[3]), "", ""
		},
	},
	// ./main.go:50:6: invalid receiver int (basic type)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+receiver\s+(\S+)\s+\(`),
		Intent:  IntentInvalidReceiver,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid receiver: %s", m[4]), m[4], ""
		},
	},
	// ./main.go:55:6: foo.bar undefined (type baz has no field or method bar)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+(\S+)\.(\S+)\s+undefined\s+\(type\s+(\S+)\s+has\s+no\s+field\s+or\s+method\s+(\S+)\)$`),
		Intent:  IntentMissingMethod,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("%s.%s undefined on %s", m[4], m[5], m[6]), m[5], ""
		},
	},
	// ── Assignment mismatch ────────────────────────────────────────────────────
	// ./main.go:6:10: assignment mismatch: 2 variables but single returns 1 value
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+assignment\s+mismatch:\s+(\d+)\s+variables\s+but\s+(\S+)\s+returns\s+(\d+)\s+value`),
		Intent:  IntentAssignmentMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("assignment mismatch: %s variables but %s returns %s value", m[4], m[5], m[6]), m[5], ""
		},
	},
	// ./main.go:6: assignment mismatch: 2 variables but single returns 1 value (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+assignment\s+mismatch:\s+(\d+)\s+variables\s+but\s+(\S+)\s+returns\s+(\d+)\s+value`),
		Intent:  IntentAssignmentMismatch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("assignment mismatch: %s variables but %s returns %s value", m[3], m[4], m[5]), m[4], ""
		},
	},
	// ── Invalid binary operation ──────────────────────────────────────────────
	// ./main.go:6:6: invalid operation: s + x (mismatched types string and int)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+operation:\s*(\S+)\s*\+\s*(\S+)\s*\(mismatched\s+types\s+(\S+)\s+and\s+(\S+)\)`),
		Intent:  IntentInvalidBinaryOp,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid operation: %s + %s", m[4], m[5]), m[6], m[7]
		},
	},
	// ./main.go:6: invalid operation: s + x (mismatched types string and int) (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+invalid\s+operation:\s*(\S+)\s*\+\s*(\S+)\s*\(mismatched\s+types\s+(\S+)\s+and\s+(\S+)\)`),
		Intent:  IntentInvalidBinaryOp,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("invalid operation: %s + %s", m[3], m[4]), m[5], m[6]
		},
	},
	// ── No new variables on left side of := ────────────────────────────────────
	// ./main.go:6:6: no new variables on left side of :=
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+no\s+new\s+variables\s+on\s+left\s+side\s+of\s+:=`),
		Intent:  IntentNoNewVariables,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "no new variables on left side of :=", "", ""
		},
	},
	// ./main.go:6: no new variables on left side of := (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+no\s+new\s+variables\s+on\s+left\s+side\s+of\s+:=`),
		Intent:  IntentNoNewVariables,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, "no new variables on left side of :=", "", ""
		},
	},
	// ── Too many arguments ─────────────────────────────────────────────────────
	// ./main.go:6:6: too many arguments in call to foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+too\s+many\s+arguments\s+in\s+call\s+to\s+(\S+)`),
		Intent:  IntentTooManyArgs,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("too many arguments in call to %s", m[4]), m[4], ""
		},
	},
	// ./main.go:6: too many arguments in call to foo (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+too\s+many\s+arguments\s+in\s+call\s+to\s+(\S+)`),
		Intent:  IntentTooManyArgs,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("too many arguments in call to %s", m[3]), m[3], ""
		},
	},
	// ── Not enough arguments ───────────────────────────────────────────────────
	// ./main.go:6:6: not enough arguments in call to foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+not\s+enough\s+arguments\s+in\s+call\s+to\s+(\S+)`),
		Intent:  IntentNotEnoughArgs,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("not enough arguments in call to %s", m[4]), m[4], ""
		},
	},
	// ./main.go:6: not enough arguments in call to foo (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+not\s+enough\s+arguments\s+in\s+call\s+to\s+(\S+)`),
		Intent:  IntentNotEnoughArgs,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("not enough arguments in call to %s", m[3]), m[3], ""
		},
	},
	// ── Call non-function ──────────────────────────────────────────────────────
	// ./main.go:6:6: call of non-function foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+call\s+of\s+non-function\s+(\S+)`),
		Intent:  IntentCallNonFunction,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("call of non-function %s", m[4]), m[4], ""
		},
	},
	// ./main.go:6: call of non-function foo (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+call\s+of\s+non-function\s+(\S+)`),
		Intent:  IntentCallNonFunction,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("call of non-function %s", m[3]), m[3], ""
		},
	},
	// ── Cannot range ───────────────────────────────────────────────────────────
	// ./main.go:6:6: cannot range over foo (type bar)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+range\s+over\s+(\S+)\s+\(type\s+(\S+)\)`),
		Intent:  IntentCannotRange,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("cannot range over %s", m[4]), m[4], m[5]
		},
	},
	// ./main.go:6: cannot range over foo (type bar) (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+cannot\s+range\s+over\s+(\S+)\s+\(type\s+(\S+)\)`),
		Intent:  IntentCannotRange,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("cannot range over %s", m[3]), m[3], m[4]
		},
	},
	// ── Cannot take address ────────────────────────────────────────────────────
	// ./main.go:6:6: cannot take address of foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+take\s+address\s+of\s+(\S+)`),
		Intent:  IntentCannotTakeAddress,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("cannot take address of %s", m[4]), m[4], ""
		},
	},
	// ./main.go:6: cannot take address of foo (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+cannot\s+take\s+address\s+of\s+(\S+)`),
		Intent:  IntentCannotTakeAddress,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("cannot take address of %s", m[3]), m[3], ""
		},
	},
	// ── Duplicate field ────────────────────────────────────────────────────────
	// ./main.go:6:6: duplicate field foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+duplicate\s+field\s+(\S+)`),
		Intent:  IntentDuplicateField,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("duplicate field %s", m[4]), m[4], ""
		},
	},
	// ./main.go:6: duplicate field foo (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+duplicate\s+field\s+(\S+)`),
		Intent:  IntentDuplicateField,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("duplicate field %s", m[3]), m[3], ""
		},
	},
	// ── Duplicate key ──────────────────────────────────────────────────────────
	// ./main.go:6:6: duplicate key foo in map literal
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+duplicate\s+key\s+(\S+)\s+in\s+map\s+literal`),
		Intent:  IntentDuplicateKey,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("duplicate key %s in map literal", m[4]), m[4], ""
		},
	},
	// ./main.go:6: duplicate key foo in map literal (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+duplicate\s+key\s+(\S+)\s+in\s+map\s+literal`),
		Intent:  IntentDuplicateKey,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("duplicate key %s in map literal", m[3]), m[3], ""
		},
	},
	// ── Invalid type assertion ─────────────────────────────────────────────────
	// ./main.go:6:6: impossible type assertion: foo does not implement bar
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+impossible\s+type\s+assertion:\s+(\S+)\s+does\s+not\s+implement\s+(\S+)`),
		Intent:  IntentInvalidTypeAssertion,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("impossible type assertion: %s does not implement %s", m[4], m[5]), m[4], m[5]
		},
	},
	// ./main.go:6: impossible type assertion: foo does not implement bar (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+impossible\s+type\s+assertion:\s+(\S+)\s+does\s+not\s+implement\s+(\S+)`),
		Intent:  IntentInvalidTypeAssertion,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("impossible type assertion: %s does not implement %s", m[3], m[4]), m[3], m[4]
		},
	},
	// ── Invalid send on channel ────────────────────────────────────────────────
	// ./main.go:6:6: invalid operation: foo <- bar (send to non-channel type baz)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+operation:\s+(\S+)\s+<-\s+(\S+)\s+\(send\s+to\s+non-channel\s+type\s+(\S+)\)`),
		Intent:  IntentInvalidSend,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid send: %s <- %s", m[4], m[5]), m[4], m[6]
		},
	},
	// ── Invalid receive from channel ───────────────────────────────────────────
	// ./main.go:6:6: invalid operation: <-foo (receive from non-channel type bar)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+operation:\s*<-\s*(\S+)\s+\(receive\s+from\s+non-channel\s+type\s+(\S+)\)`),
		Intent:  IntentInvalidReceive,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid receive from %s", m[4]), m[4], m[5]
		},
	},
	// ── Invalid close ──────────────────────────────────────────────────────────
	// ./main.go:6:6: invalid operation: close(foo) (non-channel type bar)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+operation:\s+close\((\S+)\)\s+\(non-channel\s+type\s+(\S+)\)`),
		Intent:  IntentInvalidClose,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid close of %s", m[4]), m[4], m[5]
		},
	},
	// ── Invalid go statement ───────────────────────────────────────────────────
	// ./main.go:6:6: go must be followed by function call
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+go\s+must\s+be\s+followed\s+by\s+function\s+call`),
		Intent:  IntentInvalidGoStmt,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "go must be followed by function call", "", ""
		},
	},
	// ── Invalid defer statement ────────────────────────────────────────────────
	// ./main.go:6:6: defer must be followed by function call
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+defer\s+must\s+be\s+followed\s+by\s+function\s+call`),
		Intent:  IntentInvalidDeferStmt,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "defer must be followed by function call", "", ""
		},
	},
	// ── Invalid fallthrough ────────────────────────────────────────────────────
	// ./main.go:6:6: fallthrough statement out of place
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+fallthrough\s+statement\s+out\s+of\s+place`),
		Intent:  IntentInvalidFallthrough,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "fallthrough statement out of place", "", ""
		},
	},
	// ── Invalid break ──────────────────────────────────────────────────────────
	// ./main.go:6:6: break is not in a loop, switch, or select
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+break\s+is\s+not\s+in\s+a\s+loop,\s+switch,\s+or\s+select`),
		Intent:  IntentInvalidBreak,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "break is not in a loop, switch, or select", "", ""
		},
	},
	// ── Invalid continue ───────────────────────────────────────────────────────
	// ./main.go:6:6: continue is not in a loop
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+continue\s+is\s+not\s+in\s+a\s+loop`),
		Intent:  IntentInvalidContinue,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "continue is not in a loop", "", ""
		},
	},
	// ── Invalid goto ───────────────────────────────────────────────────────────
	// ./main.go:6:6: goto foo jumps over variable declaration
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+goto\s+(\S+)\s+jumps\s+over\s+variable\s+declaration`),
		Intent:  IntentInvalidGoto,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("goto %s jumps over variable declaration", m[4]), m[4], ""
		},
	},
	// ── Invalid composite literal ──────────────────────────────────────────────
	// ./main.go:6:6: invalid composite literal type foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+composite\s+literal\s+type\s+(\S+)`),
		Intent:  IntentInvalidCompositeLit,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid composite literal type %s", m[4]), m[4], ""
		},
	},
	// ── Invalid function literal ───────────────────────────────────────────────
	// ./main.go:6:6: invalid function literal
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+function\s+literal`),
		Intent:  IntentInvalidFunctionLit,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "invalid function literal", "", ""
		},
	},
	// ── Invalid type switch ────────────────────────────────────────────────────
	// ./main.go:6:6: cannot type switch on non-interface value foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+type\s+switch\s+on\s+non-interface\s+value\s+(\S+)`),
		Intent:  IntentInvalidTypeSwitch,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("cannot type switch on non-interface value %s", m[4]), m[4], ""
		},
	},
	// ── Invalid select statement ───────────────────────────────────────────────
	// ./main.go:6:6: select must have no entries
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+select\s+must\s+have\s+no\s+entries`),
		Intent:  IntentInvalidSelectStmt,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "select must have no entries", "", ""
		},
	},
	// ── Invalid for statement ──────────────────────────────────────────────────
	// ./main.go:6:6: cannot use range clause in for statement with no variables
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+use\s+range\s+clause\s+in\s+for\s+statement\s+with\s+no\s+variables`),
		Intent:  IntentInvalidForStmt,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "cannot use range clause in for statement with no variables", "", ""
		},
	},
	// ── Invalid if statement ───────────────────────────────────────────────────
	// ./main.go:6:6: missing condition in if statement
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+missing\s+condition\s+in\s+if\s+statement`),
		Intent:  IntentInvalidIfStmt,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "missing condition in if statement", "", ""
		},
	},
	// ── Invalid switch statement ───────────────────────────────────────────────
	// ./main.go:6:6: missing expression in switch
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+missing\s+expression\s+in\s+switch`),
		Intent:  IntentInvalidSwitchStmt,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "missing expression in switch", "", ""
		},
	},
	// ── Invalid type declaration ───────────────────────────────────────────────
	// ./main.go:6:6: invalid type: foo is not a type
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+type:\s+(\S+)\s+is\s+not\s+a\s+type`),
		Intent:  IntentInvalidTypeDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid type: %s is not a type", m[4]), m[4], ""
		},
	},
	// ── Invalid function declaration ───────────────────────────────────────────
	// ./main.go:6:6: missing function body
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+missing\s+function\s+body`),
		Intent:  IntentMissingFunctionBody,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "missing function body", "", ""
		},
	},
	// ./main.go:6: missing function body (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+missing\s+function\s+body`),
		Intent:  IntentMissingFunctionBody,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, "missing function body", "", ""
		},
	},
	// ── Invalid method declaration ─────────────────────────────────────────────
	// ./main.go:6:6: invalid method receiver
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+method\s+receiver`),
		Intent:  IntentInvalidMethodDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "invalid method receiver", "", ""
		},
	},
	// ── Invalid interface declaration ──────────────────────────────────────────
	// ./main.go:6:6: interface contains embedded non-interface foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+interface\s+contains\s+embedded\s+non-interface\s+(\S+)`),
		Intent:  IntentInvalidInterfaceDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("interface contains embedded non-interface %s", m[4]), m[4], ""
		},
	},
	// ── Invalid struct declaration ─────────────────────────────────────────────
	// ./main.go:6:6: invalid recursive type foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+recursive\s+type\s+(\S+)`),
		Intent:  IntentInvalidStructDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid recursive type %s", m[4]), m[4], ""
		},
	},
	// ── Invalid const declaration ──────────────────────────────────────────────
	// ./main.go:6:6: const declaration cannot have type without expression
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+const\s+declaration\s+cannot\s+have\s+type\s+without\s+expression`),
		Intent:  IntentInvalidConstDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "const declaration cannot have type without expression", "", ""
		},
	},
	// ── Invalid var declaration ────────────────────────────────────────────────
	// ./main.go:6:6: var declaration cannot have type without expression
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+var\s+declaration\s+cannot\s+have\s+type\s+without\s+expression`),
		Intent:  IntentInvalidVarDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "var declaration cannot have type without expression", "", ""
		},
	},
	// ── Invalid import declaration ─────────────────────────────────────────────
	// ./main.go:6:6: import path is invalid
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+import\s+path\s+is\s+invalid`),
		Intent:  IntentInvalidImportDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "import path is invalid", "", ""
		},
	},
	// ── Invalid package declaration ────────────────────────────────────────────
	// ./main.go:6:6: package statement must be first
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+package\s+statement\s+must\s+be\s+first`),
		Intent:  IntentInvalidPackageDecl,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "package statement must be first", "", ""
		},
	},
	// ── Invalid use of ... ─────────────────────────────────────────────────────
	// ./main.go:6:6: invalid use of ... in call to foo
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+invalid\s+use\s+of\s+\.\.\.\s+in\s+call\s+to\s+(\S+)`),
		Intent:  IntentInvalidUseOfDotDotDot,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("invalid use of ... in call to %s", m[4]), m[4], ""
		},
	},
	// ── Invalid use of _ ───────────────────────────────────────────────────────
	// ./main.go:6:6: cannot use _ as value
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+use\s+_\s+as\s+value`),
		Intent:  IntentInvalidUseOfBlank,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "cannot use _ as value", "", ""
		},
	},
	// ── Invalid use of nil ─────────────────────────────────────────────────────
	// ./main.go:6:6: cannot use nil as type int in assignment
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+cannot\s+use\s+nil\s+as\s+type\s+(\S+)\s+in\s+assignment`),
		Intent:  IntentInvalidUseOfNil,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("cannot use nil as type %s in assignment", m[4]), m[4], ""
		},
	},
	// ./main.go:6: cannot use nil as type int in assignment (no column)
	{
		Pattern: regexp.MustCompile(`^([^:]+):(\d+):\s+cannot\s+use\s+nil\s+as\s+type\s+(\S+)\s+in\s+assignment`),
		Intent:  IntentInvalidUseOfNil,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, fmt.Sprintf("cannot use nil as type %s in assignment", m[3]), m[3], ""
		},
	},
	// ── go vet: too many return values ─────────────────────────────────────────
	// vet: ft/jim.go:20:9: too many return values
	// Matches both bare "file:line:col: too many return values" and vet-prefixed.
	{
		Pattern: regexp.MustCompile(`^(?:vet:\s+)?([^:]+):(\d+):(\d+):\s+too\s+many\s+return\s+values`),
		Intent:  IntentTooManyReturnValues,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "too many return values", "", ""
		},
	},
	// vet: ft/jim.go:20: too many return values (no column)
	{
		Pattern: regexp.MustCompile(`^(?:vet:\s+)?([^:]+):(\d+):\s+too\s+many\s+return\s+values`),
		Intent:  IntentTooManyReturnValues,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, "too many return values", "", ""
		},
	},
	// ── go vet: not enough return values ───────────────────────────────────────
	// vet: ft/jim.go:20:9: not enough return values
	{
		Pattern: regexp.MustCompile(`^(?:vet:\s+)?([^:]+):(\d+):(\d+):\s+not\s+enough\s+return\s+values`),
		Intent:  IntentTooFewReturnValues,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), "not enough return values", "", ""
		},
	},
	// vet: ft/jim.go:20: not enough return values (no column)
	{
		Pattern: regexp.MustCompile(`^(?:vet:\s+)?([^:]+):(\d+):\s+not\s+enough\s+return\s+values`),
		Intent:  IntentTooFewReturnValues,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), 0, "not enough return values", "", ""
		},
	},
	// ── go vet: wrong return type ──────────────────────────────────────────────
	// ft/jim.go:20:9: cannot use 0 (untyped int constant) as string value in return statement
	{
		Pattern: regexp.MustCompile(`^(?:vet:\s+)?([^:]+):(\d+):(\d+):\s+cannot\s+use\s+(\S+).*\s+as\s+(\S+)\s+value\s+in\s+return\s+statement`),
		Intent:  IntentWrongReturnType,
		Extract: func(m []string) (string, int, int, string, string, string) {
			return m[1], atoi(m[2]), atoi(m[3]), fmt.Sprintf("wrong return type: %s used as %s", m[4], m[5]), m[4], m[5]
		},
	},
}

// atoi converts a string to an int, returning 0 on error.
func atoi(s string) int {
	var n int
	for _, c := range s {
		if c < '0' || c > '9' {
			return n
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// ParseErrorLine attempts to parse a single compiler error line into a ParsedError.
// Returns nil if the line does not match any known pattern.
func ParseErrorLine(line string) *ParsedError {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return nil
	}

	for _, cp := range CompilerErrorPatterns {
		matches := cp.Pattern.FindStringSubmatch(trimmed)
		if len(matches) > 0 {
			file, lineNum, col, msg, symbol, pkg := cp.Extract(matches)
			return &ParsedError{
				Raw:     trimmed,
				File:    file,
				Line:    lineNum,
				Column:  col,
				Message: msg,
				Symbol:  symbol,
				Package: pkg,
				Intent:  cp.Intent,
			}
		}
	}

	// Fallback: try to extract file:line:col: message pattern
	genericPattern := regexp.MustCompile(`^([^:]+):(\d+):(\d+):\s+(.+)$`)
	if matches := genericPattern.FindStringSubmatch(trimmed); len(matches) > 0 {
		return &ParsedError{
			Raw:     trimmed,
			File:    matches[1],
			Line:    atoi(matches[2]),
			Column:  atoi(matches[3]),
			Message: matches[4],
			Intent:  IntentUnknown,
		}
	}

	// Try file:line: message pattern
	genericPattern2 := regexp.MustCompile(`^([^:]+):(\d+):\s+(.+)$`)
	if matches := genericPattern2.FindStringSubmatch(trimmed); len(matches) > 0 {
		return &ParsedError{
			Raw:     trimmed,
			File:    matches[1],
			Line:    atoi(matches[2]),
			Message: matches[3],
			Intent:  IntentUnknown,
		}
	}

	return nil
}

// ParseCompilerOutput parses the full output of a Go compiler/test run
// and returns a list of structured ParsedErrors.
// It handles both regular compiler output and `go vet` output which may
// include a "# package" header line and "vet: " prefixed lines.
func ParseCompilerOutput(output string) []*ParsedError {
	var errors []*ParsedError
	seen := make(map[string]bool) // deduplicate

	for _, line := range strings.Split(output, "\n") {
		// Skip go vet header lines like "# command-line-arguments" and "# [command-line-arguments]"
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		// Strip leading "vet: " prefix that go vet emits on some errors
		if strings.HasPrefix(trimmed, "vet: ") {
			line = strings.TrimPrefix(trimmed, "vet: ")
		}

		pe := ParseErrorLine(line)
		if pe == nil {
			continue
		}
		// Deduplicate by file:line:message
		key := fmt.Sprintf("%s:%d:%s", pe.File, pe.Line, pe.Message)
		if seen[key] {
			continue
		}
		seen[key] = true
		errors = append(errors, pe)
	}

	return errors
}
