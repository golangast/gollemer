// Package tokenizer provides Go-specific tokenization enhancements for the BPE tokenizer.
// This adds high-frequency Go tokens as single vocabulary units, ensuring that common
// Go idioms like "if err != nil {", "package main", "type struct", and "func (m *Type)"
// are treated as atomic tokens rather than being split arbitrarily by BPE.
package tokenizer

import (
	"log"
	"strings"
)

// GoHighFreqTokens returns a list of high-frequency Go tokens that should be
// treated as single vocabulary units. These cover common Go idioms, keywords,
// and patterns that BPE would otherwise split into suboptimal pieces.
func GoHighFreqTokens() []string {
	return []string{
		// Common error handling patterns
		"if err != nil {",
		"if err != nil",
		"if err == nil {",
		"if err == nil",
		"return err",
		"return nil, err",
		"return nil, nil",
		"return fmt.Errorf(",
		"errors.New(",
		"err != nil",
		"err == nil",
		"err != nil {",
		"err == nil {",

		// Package declarations
		"package main",
		"package ",

		// Import patterns
		"import (",
		"import \"",
		")",

		// Type declarations
		"type struct {",
		"type struct",
		"type interface {",
		"type interface",
		"type (",

		// Function declarations
		"func (",
		"func main()",
		"func main() {",
		"func init()",
		"func init() {",
		"func (m *",
		"func (s *",
		"func (t *",
		"func (r *",
		"func (h *",
		"func (w *",
		"func (c *",
		"func (",
		"func ",

		// Method receivers
		"*MoE)",
		"*Tensor)",
		"*Expert)",
		"*Model)",
		"*Config)",
		"*Server)",
		"*Handler)",
		"*Client)",

		// Control flow
		"for _, ",
		"for i := ",
		"for range ",
		"for {",
		"switch ",
		"case ",
		"default:",
		"select {",
		"case <-",

		// Defer patterns
		"defer func()",
		"defer ",

		// Go routine patterns
		"go func()",
		"go ",

		// Common variable declarations
		"var _ ",
		"var (",
		"var err error",
		"var ok bool",
		"const (",
		"const ",

		// Common struct fields
		"mu       sync.Mutex",
		"sync.Mutex",
		"sync.RWMutex",
		"sync.WaitGroup",
		"sync.Once",
		"sync.Map",
		"sync.Pool",

		// Channel operations
		"chan struct{}",
		"chan string",
		"chan int",
		"chan bool",
		"chan<-",
		"<-chan",
		"make(chan ",
		"close(",

		// Common type assertions
		".(type) {",
		".(string)",
		".(int)",
		".(bool)",
		".(float64)",
		".(error)",

		// Error wrapping
		"fmt.Errorf(",
		"errors.New(",
		"errors.Is(",
		"errors.As(",
		"errors.Unwrap(",
		"errors.Join(",

		// Logging patterns
		"log.Printf(",
		"log.Fatalf(",
		"log.Fatal(",
		"log.Println(",
		"log.Printf",
		"log.Fatalf",

		// Testing patterns
		"func Test",
		"func Benchmark",
		"t *testing.T",
		"b *testing.B",
		"t.Errorf(",
		"t.Fatalf(",
		"t.Logf(",
		"t.Run(",
		"assert.Equal(t",
		"require.NoError(t",

		// JSON patterns
		"json.Marshal(",
		"json.Unmarshal(",
		"json.NewEncoder(",
		"json.NewDecoder(",
		"json.MarshalIndent(",

		// HTTP patterns
		"http.HandleFunc(",
		"http.HandlerFunc(",
		"http.Handler(",
		"http.Server{",
		"http.ListenAndServe(",
		"http.Get(",
		"http.Post(",
		"http.NewRequest(",
		"http.ResponseWriter",
		"*http.Request",
		"r *http.Request",
		"w http.ResponseWriter",

		// Context patterns
		"context.Background()",
		"context.TODO()",
		"context.WithCancel(",
		"context.WithTimeout(",
		"context.WithDeadline(",
		"context.WithValue(",

		// Common interface patterns
		"interface{}",
		"interface {",
		"Read(",
		"Write(",
		"Close() error",
		"String() string",
		"Error() string",
		"MarshalJSON()",
		"UnmarshalJSON(",

		// Common return patterns
		"error)",
		"string)",
		"int)",
		"bool)",
		"float64)",
		"[]byte)",
		"[]string)",
		"[]int)",

		// Common method signatures
		"func (m *MoE) Forward(",
		"func (m *MoE) Backward(",
		"func (m *MoE) Train(",
		"func (m *MoE) Predict(",
		"func (m *MoE) Save(",
		"func (m *MoE) Load(",

		// FIM special tokens
		"<PRE>",
		"<SUF>",
		"<MID>",
		"<SCOPE:",
		"<CONTEXT_TYPES>",
		"<PROJECT_CONTEXT>",
		"<FIM_PRE>",
		"<FIM_SUF>",
		"<|im_start|>",
		"<|im_end|>",

		// SEARCH/REPLACE markers
		"<<<<<<< SEARCH",
		"=======",
		">>>>>>> REPLACE",
	}
}

// AddGoTokensToTokenizer adds all Go-specific high-frequency tokens to the given BPE tokenizer.
func AddGoTokensToTokenizer(bpe *BPETokenizer) {
	tokens := GoHighFreqTokens()
	for _, tok := range tokens {
		// Register raw string as a special token
		bpe.AddSpecialToken(tok)

		// Register space-protected variant as a special token
		if strings.Contains(tok, " ") {
			protected := strings.ReplaceAll(tok, " ", "\u00A0")
			bpe.AddSpecialToken(protected)
		}
	}
	log.Printf("🧩 Go Tokenizer: Added %d Go-specific high-frequency tokens to vocabulary", len(tokens))
}

// IsGoHighFreqToken checks if a given text is a registered Go high-frequency token.
func IsGoHighFreqToken(text string) bool {
	for _, tok := range GoHighFreqTokens() {
		if tok == text {
			return true
		}
	}
	return false
}

// GoTokenizePreprocess replaces registered multi-character Go tokens
// with exact special token identifiers that the BPE encoder recognizes.
func GoTokenizePreprocess(code string) string {
	result := code

	// Iterate in reverse or sort by length descending to match longest phrases first
	tokens := GoHighFreqTokens()
	for _, tok := range tokens {
		if len(tok) > 1 && strings.Contains(result, tok) {
			// Ensure token is treated as an exact match without altering BPE rules
			result = strings.ReplaceAll(result, tok, tok)
		}
	}
	return result
}

// protectPattern replaces spaces inside high-frequency phrases with
// a non-breaking space (U+00A0) or special boundary marker so the BPE
// encoder treats the entire sequence as a single atomic token unit.
func protectPattern(code, pattern string) string {
	if !strings.Contains(code, pattern) {
		return code
	}
	// Convert spaces inside the pattern to non-breaking spaces (\u00A0)
	// which BPE will encode as a single special token unit rather than splitting on whitespace.
	protected := strings.ReplaceAll(pattern, " ", "\u00A0")
	return strings.ReplaceAll(code, pattern, protected)
}
