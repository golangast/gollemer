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
func ParseCompilerOutput(output string) []*ParsedError {
	var errors []*ParsedError
	seen := make(map[string]bool) // deduplicate

	for _, line := range strings.Split(output, "\n") {
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
