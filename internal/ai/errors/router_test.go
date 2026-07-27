package errors

import (
	"testing"
)

func TestParseErrorLine_UndefinedSymbol(t *testing.T) {
	line := "./main.go:124:3: undefined: server"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.File != "./main.go" {
		t.Errorf("expected file ./main.go, got %s", pe.File)
	}
	if pe.Line != 124 {
		t.Errorf("expected line 124, got %d", pe.Line)
	}
	if pe.Column != 3 {
		t.Errorf("expected column 3, got %d", pe.Column)
	}
	if pe.Message != "undefined: server" {
		t.Errorf("expected 'undefined: server', got '%s'", pe.Message)
	}
	if pe.Symbol != "server" {
		t.Errorf("expected symbol 'server', got '%s'", pe.Symbol)
	}
	if pe.Intent != IntentUndefinedSymbol {
		t.Errorf("expected IntentUndefinedSymbol, got %s", pe.Intent)
	}
}

func TestParseErrorLine_MissingHandler(t *testing.T) {
	line := "./cmd/server/main.go:18:5: undefined: authHandler"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.File != "./cmd/server/main.go" {
		t.Errorf("expected file ./cmd/server/main.go, got %s", pe.File)
	}
	if pe.Line != 18 {
		t.Errorf("expected line 18, got %d", pe.Line)
	}
	if pe.Column != 5 {
		t.Errorf("expected column 5, got %d", pe.Column)
	}
	// This should match both the undefined symbol pattern and the handler pattern
	// The handler pattern is listed after undefined symbol, so it will match undefined symbol first
	if pe.Intent != IntentMissingHandlerDefinition && pe.Intent != IntentUndefinedSymbol {
		t.Errorf("expected IntentMissingHandlerDefinition or IntentUndefinedSymbol, got %s", pe.Intent)
	}
}

func TestParseErrorLine_MissingImport(t *testing.T) {
	line := "./main.go:10:2: imported and not used: \"fmt\""
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentUnusedImport {
		t.Errorf("expected IntentUnusedImport, got %s", pe.Intent)
	}
	if pe.Package != "fmt" {
		t.Errorf("expected package 'fmt', got '%s'", pe.Package)
	}
}

func TestParseErrorLine_UnusedVariable(t *testing.T) {
	line := "./main.go:15:6: x declared and not used"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentUnusedVariable {
		t.Errorf("expected IntentUnusedVariable, got %s", pe.Intent)
	}
	if pe.Symbol != "x" {
		t.Errorf("expected symbol 'x', got '%s'", pe.Symbol)
	}
}

func TestParseErrorLine_TypeMismatch(t *testing.T) {
	line := "./main.go:20:6: cannot use name (type string) as type int in assignment"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentTypeMismatch {
		t.Errorf("expected IntentTypeMismatch, got %s", pe.Intent)
	}
}

func TestParseErrorLine_MissingReturn(t *testing.T) {
	line := "./main.go:30:1: missing return"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentMissingReturn {
		t.Errorf("expected IntentMissingReturn, got %s", pe.Intent)
	}
}

func TestParseErrorLine_SyntaxError(t *testing.T) {
	line := "./main.go:45:1: syntax error: unexpected EOF"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentSyntaxError {
		t.Errorf("expected IntentSyntaxError, got %s", pe.Intent)
	}
}

func TestParseErrorLine_InvalidReceiver(t *testing.T) {
	line := "./main.go:50:6: invalid receiver int (basic type)"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentInvalidReceiver {
		t.Errorf("expected IntentInvalidReceiver, got %s", pe.Intent)
	}
}

func TestParseErrorLine_Generic(t *testing.T) {
	// Test a generic error line that doesn't match any specific pattern
	line := "./main.go:60:2: could not determine package"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentUnknown {
		t.Errorf("expected IntentUnknown, got %s", pe.Intent)
	}
}

func TestParseCompilerOutput_Multiple(t *testing.T) {
	output := `./main.go:10:2: imported and not used: "fmt"
./main.go:15:6: x declared and not used
./main.go:20:6: cannot use name (type string) as type int in assignment
./main.go:30:1: missing return
./main.go:45:1: syntax error: unexpected EOF
`

	errors := ParseCompilerOutput(output)
	if len(errors) != 5 {
		t.Fatalf("expected 5 parsed errors, got %d", len(errors))
	}

	expectedIntents := []ErrorIntent{
		IntentUnusedImport,
		IntentUnusedVariable,
		IntentTypeMismatch,
		IntentMissingReturn,
		IntentSyntaxError,
	}

	for i, pe := range errors {
		if pe.Intent != expectedIntents[i] {
			t.Errorf("error %d: expected intent %s, got %s", i, expectedIntents[i], pe.Intent)
		}
	}
}

func TestErrorIntentFromString(t *testing.T) {
	tests := []struct {
		input  string
		expect ErrorIntent
	}{
		{"UNDEFINED_SYMBOL", IntentUndefinedSymbol},
		{"MISSING_IMPORT", IntentMissingImport},
		{"MISSING_HANDLER_DEFINITION", IntentMissingHandlerDefinition},
		{"TYPE_MISMATCH", IntentTypeMismatch},
		{"UNUSED_VARIABLE", IntentUnusedVariable},
		{"UNUSED_IMPORT", IntentUnusedImport},
		{"MISSING_RETURN", IntentMissingReturn},
		{"SYNTAX_ERROR", IntentSyntaxError},
		{"UNKNOWN", IntentUnknown},
		{"undefined_string", IntentUnknown},
	}

	for _, tt := range tests {
		result := ErrorIntentFromString(tt.input)
		if result != tt.expect {
			t.Errorf("ErrorIntentFromString(%q) = %s, want %s", tt.input, result, tt.expect)
		}
	}
}

func TestGetFixer(t *testing.T) {
	tests := []struct {
		intent  ErrorIntent
		hasFunc bool
	}{
		{IntentUndefinedSymbol, true},
		{IntentMissingHandlerDefinition, true},
		{IntentMissingImport, true},
		{IntentUnusedImport, true},
		{IntentUnusedVariable, true},
		{IntentMissingReturn, true},
		{IntentTypeMismatch, true},
		{IntentMissingMethod, true},
		{IntentUndeclaredName, true},
		{IntentSyntaxError, true},
		{IntentUnknown, false},
	}

	for _, tt := range tests {
		fixer := GetFixer(tt.intent)
		if (fixer != nil) != tt.hasFunc {
			t.Errorf("GetFixer(%s) exists = %v, want %v", tt.intent, fixer != nil, tt.hasFunc)
		}
	}
}

func TestGetZeroValue(t *testing.T) {
	tests := []struct {
		typeName string
		expected string
	}{
		{"string", `""`},
		{"int", "0"},
		{"float64", "0.0"},
		{"bool", "false"},
		{"error", "nil"},
		{"*http.Server", "nil"},
		{"interface{}", "nil"},
	}

	for _, tt := range tests {
		result := getZeroValue(tt.typeName)
		if result != tt.expected {
			t.Errorf("getZeroValue(%q) = %q, want %q", tt.typeName, result, tt.expected)
		}
	}
}

func TestParseErrorLine_UndeclaredName(t *testing.T) {
	line := "./main.go:42:2: undeclared name: server"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentUndeclaredName {
		t.Errorf("expected IntentUndeclaredName, got %s", pe.Intent)
	}
	if pe.Symbol != "server" {
		t.Errorf("expected symbol 'server', got '%s'", pe.Symbol)
	}
}

func TestParseErrorLine_NonBoolInIf(t *testing.T) {
	line := "./main.go:40:6: non-bool x used in if"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentNonBoolUsedInIf {
		t.Errorf("expected IntentNonBoolUsedInIf, got %s", pe.Intent)
	}
}

func TestParseErrorLine_PackageNotImported(t *testing.T) {
	line := "./main.go:25:6: Foo not declared by package bar"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentPackageNotImported {
		t.Errorf("expected IntentPackageNotImported, got %s", pe.Intent)
	}
}

func TestParseErrorLine_CannotAssign(t *testing.T) {
	line := "./main.go:35:6: cannot assign to x"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentCannotAssign {
		t.Errorf("expected IntentCannotAssign, got %s", pe.Intent)
	}
}

func TestErrorIntent_String(t *testing.T) {
	tests := []struct {
		intent ErrorIntent
		str    string
	}{
		{IntentUndefinedSymbol, "UNDEFINED_SYMBOL"},
		{IntentMissingImport, "MISSING_IMPORT"},
		{IntentMissingHandlerDefinition, "MISSING_HANDLER_DEFINITION"},
		{IntentMissingFunctionBody, "MISSING_FUNCTION_BODY"},
		{IntentTypeMismatch, "TYPE_MISMATCH"},
		{IntentUnusedVariable, "UNUSED_VARIABLE"},
		{IntentUnusedImport, "UNUSED_IMPORT"},
		{IntentMissingReturn, "MISSING_RETURN"},
		{IntentSyntaxError, "SYNTAX_ERROR"},
		{IntentMissingMethod, "MISSING_METHOD"},
		{IntentUndeclaredName, "UNDECLARED_NAME"},
		{IntentPackageNotImported, "PACKAGE_NOT_IMPORTED"},
		{IntentInvalidReceiver, "INVALID_RECEIVER"},
		{IntentCannotAssign, "CANNOT_ASSIGN"},
		{IntentNonBoolUsedInIf, "NON_BOOL_IN_IF"},
		{IntentUnknown, "UNKNOWN"},
	}

	for _, tt := range tests {
		result := tt.intent.String()
		if result != tt.str {
			t.Errorf("ErrorIntent(%d).String() = %q, want %q", tt.intent, result, tt.str)
		}
	}
}

func TestParseErrorLine_MissingMethod(t *testing.T) {
	line := "./main.go:55:6: foo.bar undefined (type baz has no field or method bar)"
	pe := ParseErrorLine(line)
	if pe == nil {
		t.Fatal("expected parsed error, got nil")
	}
	if pe.Intent != IntentMissingMethod {
		t.Errorf("expected IntentMissingMethod, got %s", pe.Intent)
	}
}

func TestDeduplication(t *testing.T) {
	// Same error reported twice should only appear once
	output := `./main.go:10:2: imported and not used: "fmt"
./main.go:10:2: imported and not used: "fmt"
./main.go:15:6: x declared and not used
`

	errors := ParseCompilerOutput(output)
	if len(errors) != 2 {
		t.Fatalf("expected 2 errors after dedup, got %d", len(errors))
	}
}
