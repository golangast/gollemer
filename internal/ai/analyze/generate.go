package analyze

import (
	"fmt"
	"strings"
)

// GenerateGetter creates a getter method for a struct field.
// Follows Go idiom: no "Get" prefix, just the field name.
// Example: GenerateGetter("Config", "c", "Timeout", "time.Duration") ->
//   func (c *Config) Timeout() time.Duration {
//       return c.Timeout
//   }
func GenerateGetter(structName, receiver, fieldName, fieldType string) string {
	// Go idiom: method name is the field name (exported)
	methodName := fieldName
	// If field is unexported, capitalize for the method
	if len(methodName) > 0 && methodName[0] >= 'a' && methodName[0] <= 'z' {
		methodName = strings.ToUpper(methodName[:1]) + methodName[1:]
	}
	return fmt.Sprintf(`func (%s *%s) %s() %s {
	return %s.%s
}`, receiver, structName, methodName, fieldType, receiver, fieldName)
}

// GenerateSetter creates a setter method for a struct field.
// Example: GenerateSetter("Config", "c", "Timeout", "time.Duration") ->
//   func (c *Config) SetTimeout(v time.Duration) {
//       c.Timeout = v
//   }
func GenerateSetter(structName, receiver, fieldName, fieldType string) string {
	methodName := "Set" + fieldName
	if len(fieldName) > 0 && fieldName[0] >= 'a' && fieldName[0] <= 'z' {
		methodName = "Set" + strings.ToUpper(fieldName[:1]) + fieldName[1:]
	}
	return fmt.Sprintf(`func (%s *%s) %s(v %s) {
	%s.%s = v
}`, receiver, structName, methodName, fieldType, receiver, fieldName)
}

// GenerateConstructor creates a constructor function for a struct.
// Example: GenerateConstructor("Config", []Field{{"Timeout", "time.Duration"}, {"Name", "string"}}) ->
//   func NewConfig(timeout time.Duration, name string) *Config {
//       return &Config{
//           Timeout: timeout,
//           Name: name,
//       }
//   }
type Field struct {
	Name string
	Type string
}

func GenerateConstructor(structName string, fields []Field) string {
	params := make([]string, len(fields))
	inits := make([]string, len(fields))
	for i, f := range fields {
		paramName := strings.ToLower(f.Name[:1]) + f.Name[1:]
		params[i] = fmt.Sprintf("%s %s", paramName, f.Type)
		inits[i] = fmt.Sprintf("\t\t%s: %s,", f.Name, paramName)
	}
	return fmt.Sprintf(`func New%s(%s) *%s {
	return &%s{
%s
	}
}`, structName, strings.Join(params, ", "), structName, structName, strings.Join(inits, "\n"))
}

// GenerateStringer creates a String() method.
// Follows the pattern: return a formatted string of the key fields.
func GenerateStringer(structName, receiver string, fields []Field) string {
	// Simple: fmt.Sprintf with field names
	formatParts := make([]string, len(fields))
	args := make([]string, len(fields))
	for i, f := range fields {
		formatParts[i] = fmt.Sprintf("%s=%%v", f.Name)
		args[i] = fmt.Sprintf("%s.%s", receiver, f.Name)
	}
	return fmt.Sprintf(`func (%s *%s) String() string {
	return fmt.Sprintf("%s{%s}", %s)
}`, receiver, structName, structName, strings.Join(formatParts, " "), strings.Join(args, ", "))
}
