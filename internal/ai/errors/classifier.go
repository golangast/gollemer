package errors

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/golangast/gollemer/internal/ai/llm"
	"github.com/golangast/gollemer/internal/ai/moe"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// ErrorClassifier wraps the trained MoE classification model to classify
// compiler/linter error intents from raw error output.
type ErrorClassifier struct {
	Model              *moe.IntentMoE
	QueryVocabulary    *vocab.Vocabulary
	ParentVocabulary   *vocab.Vocabulary
	ChildVocabulary    *vocab.Vocabulary
	SentenceVocabulary *vocab.Vocabulary
	Tokenizer          *tokenizer.Tokenizer
	MaxSeqLen          int
	ProjectRoot        string
	Verbose            bool
}

// NewErrorClassifier loads the trained MoE classification model and its
// associated vocabularies from the project's data directory.
func NewErrorClassifier(projectRoot string, verbose bool) (*ErrorClassifier, error) {
	modelPath := filepath.Join(projectRoot, "data/models/gob_models/moe_classification_model.gob")
	queryVocabPath := filepath.Join(projectRoot, "data/models/gob_models/query_vocabulary.gob")
	parentVocabPath := filepath.Join(projectRoot, "data/models/gob_models/parent_intent_vocabulary.gob")
	childVocabPath := filepath.Join(projectRoot, "data/models/gob_models/child_intent_vocabulary.gob")
	sentenceVocabPath := filepath.Join(projectRoot, "data/models/gob_models/sentence_vocabulary.gob")

	// Check if model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("MoE classification model not found at %s (run training first)", modelPath)
	}

	// Load vocabularies
	qv, err := vocab.LoadVocabulary(queryVocabPath)
	if err != nil {
		return nil, fmt.Errorf("load query vocabulary: %w", err)
	}
	pv, err := vocab.LoadVocabulary(parentVocabPath)
	if err != nil {
		return nil, fmt.Errorf("load parent intent vocabulary: %w", err)
	}
	cv, err := vocab.LoadVocabulary(childVocabPath)
	if err != nil {
		return nil, fmt.Errorf("load child intent vocabulary: %w", err)
	}
	sv, err := vocab.LoadVocabulary(sentenceVocabPath)
	if err != nil {
		return nil, fmt.Errorf("load sentence vocabulary: %w", err)
	}

	// Load model
	model, err := moe.LoadIntentMoEModelFromGOB(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load classification model: %w", err)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(qv)
	if err != nil {
		return nil, fmt.Errorf("create tokenizer: %w", err)
	}

	maxSeqLen := 64
	if model.Encoder != nil {
		// If the encoder holds sequence length configuration
		maxSeqLen = 64 // or fetch dynamically if available on your Encoder interface
	}

	if verbose {
		log.Printf("✅ ErrorClassifier loaded:")
		log.Printf("   Query vocab size: %d", qv.Size())
		log.Printf("   Parent intent vocab size: %d", pv.Size())
		log.Printf("   Child intent vocab size: %d", cv.Size())
		log.Printf("   Max sequence length: %d", maxSeqLen)
	}

	return &ErrorClassifier{
		Model:              model,
		QueryVocabulary:    qv,
		ParentVocabulary:   pv,
		ChildVocabulary:    cv,
		SentenceVocabulary: sv,
		Tokenizer:          tok,
		MaxSeqLen:          maxSeqLen,
		ProjectRoot:        projectRoot,
		Verbose:            verbose,
	}, nil
}

// NewErrorClassifierDefault loads the classifier using automatic project root detection.
func NewErrorClassifierDefault(verbose bool) (*ErrorClassifier, error) {
	projectRoot, err := llm.FindProjectRoot()
	if err != nil {
		return nil, fmt.Errorf("find project root: %w", err)
	}
	return NewErrorClassifier(projectRoot, verbose)
}

// ClassifyIntent runs the MoE model to classify a compiler error line into
// a parent and child intent. Returns the best-matching ErrorIntent.
func (ec *ErrorClassifier) ClassifyIntent(errorLine string) (ErrorIntent, float64) {
	// First try regex-based classification (fast path)
	for _, cp := range CompilerErrorPatterns {
		matches := cp.Pattern.FindStringSubmatch(strings.TrimSpace(errorLine))
		if len(matches) > 0 {
			return cp.Intent, 0.95
		}
	}

	// If regex doesn't match, use the MoE model for neural classification
	if ec.Model == nil {
		return IntentUnknown, 0.0
	}

	return ec.classifyWithMoE(errorLine)
}

// classifyWithMoE uses the trained MoE classification model to determine
// the error intent from the error text.
func (ec *ErrorClassifier) classifyWithMoE(errorText string) (ErrorIntent, float64) {
	// Tokenize the error text
	tokenIDs, err := ec.Tokenizer.Encode(errorText)
	if err != nil || len(tokenIDs) == 0 {
		return IntentUnknown, 0.0
	}

	batchSize := 1
	seqLen := len(tokenIDs)
	if seqLen > ec.MaxSeqLen {
		seqLen = ec.MaxSeqLen
		tokenIDs = tokenIDs[:ec.MaxSeqLen]
	}

	// Pad to max sequence length
	paddedData := make([]float32, batchSize*ec.MaxSeqLen)
	for i, id := range tokenIDs {
		paddedData[i] = float32(id)
	}
	// Fill remaining with padding token
	paddingID := float32(ec.QueryVocabulary.PaddingTokenID)
	for i := len(tokenIDs); i < ec.MaxSeqLen; i++ {
		paddedData[i] = paddingID
	}

	inputTensor := tensor.NewTensor([]int{batchSize, ec.MaxSeqLen}, paddedData, false)
	// Dummy target sentence tensor (not used during inference)
	targetSentence := tensor.NewTensor([]int{batchSize, ec.MaxSeqLen}, make([]float32, batchSize*ec.MaxSeqLen), false)

	// Run forward pass (expecting 3 return values: outputs slice, auxiliary/loss tensor, error)
	outputs, _, err := ec.Model.Forward(0.0, inputTensor, targetSentence)
	if err != nil || len(outputs) < 2 {
		if ec.Verbose {
			log.Printf("MoE forward pass failed: %v", err)
		}
		return IntentUnknown, 0.0
	}

	// Extract parent and child logits from the output tensor slice
	parentLogits := outputs[0]
	childLogits := outputs[1]

	// Get parent intent (intent category)
	parentIdx := argmax(parentLogits.Data)
	parentStr := ec.ParentVocabulary.GetWord(parentIdx)

	// Get child intent (specific error type)
	childIdx := argmax(childLogits.Data)
	childStr := ec.ChildVocabulary.GetWord(childIdx)

	// Compute confidence from softmax probabilities
	confidence := softmaxValue(parentLogits.Data, parentIdx)

	if ec.Verbose {
		log.Printf("MoE classified: parent=%q (idx=%d, conf=%.4f), child=%q (idx=%d)",
			parentStr, parentIdx, confidence, childStr, childIdx)
	}

	// Map MoE output to ErrorIntent
	// The parent vocabulary maps to categories like "compile_error", "test_failure", etc.
	// The child vocabulary maps to specific intents like "undefined_symbol", "missing_import"
	intent := mapChildToIntent(childStr)
	return intent, float64(confidence)
}

// ClassifyBatch classifies multiple error lines in batch through the MoE model.
func (ec *ErrorClassifier) ClassifyBatch(errorLines []string) []ClassifiedError {
	results := make([]ClassifiedError, 0, len(errorLines))

	for _, line := range errorLines {
		intent, confidence := ec.ClassifyIntent(line)
		pe := ParseErrorLine(line)
		if pe == nil {
			pe = &ParsedError{
				Raw:     line,
				Message: line,
				Intent:  intent,
			}
		}
		pe.Intent = intent
		pe.Confidence = confidence
		results = append(results, ClassifiedError{
			ParsedError: pe,
		})
	}

	return results
}

// ClassifiedError wraps a ParsedError with classification metadata.
type ClassifiedError struct {
	*ParsedError
}

// ClassifyAndSort classifies errors and returns them sorted by confidence
// (highest first), then by file line number.
func (ec *ErrorClassifier) ClassifyAndSort(output string) []ClassifiedError {
	parsed := ParseCompilerOutput(output)
	results := make([]ClassifiedError, 0, len(parsed))

	for _, pe := range parsed {
		intent, confidence := ec.ClassifyIntent(pe.Raw)
		pe.Intent = intent
		pe.Confidence = confidence
		results = append(results, ClassifiedError{ParsedError: pe})
	}

	// Sort: highest confidence first, then by file line
	sort.Slice(results, func(i, j int) bool {
		if results[i].Confidence != results[j].Confidence {
			return results[i].Confidence > results[j].Confidence
		}
		if results[i].File != results[j].File {
			return results[i].File < results[j].File
		}
		return results[i].Line < results[j].Line
	})

	return results
}

// argmax returns the index of the maximum value in a slice.
func argmax(data []float32) int {
	idx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			idx = i
		}
	}
	return idx
}

// softmaxValue computes the softmax probability at a given index.
func softmaxValue(data []float32, idx int) float32 {
	maxVal := data[0]
	for _, v := range data {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	for _, v := range data {
		sum += float64(exp(v - maxVal))
	}
	return float32(exp(data[idx]-maxVal)) / float32(sum)
}

// exp computes e^x for float32.
func exp(x float32) float64 {
	// Simple approximation using Go's math.Exp
	result := 1.0
	term := 1.0
	for i := 1; i <= 20; i++ {
		term *= float64(x) / float64(i)
		result += term
	}
	return result
}

// mapChildToIntent maps a child intent string from the MoE model to an ErrorIntent.
func mapChildToIntent(childStr string) ErrorIntent {
	normalized := strings.ToUpper(strings.TrimSpace(childStr))

	// Direct mapping
	if intent := ErrorIntentFromString(normalized); intent != IntentUnknown {
		return intent
	}

	// Fuzzy matching
	switch {
	case strings.Contains(normalized, "UNDEFINED"):
		return IntentUndefinedSymbol
	case strings.Contains(normalized, "MISSING_IMPORT"), strings.Contains(normalized, "NOT_IMPORTED"):
		return IntentMissingImport
	case strings.Contains(normalized, "HANDLER"):
		return IntentMissingHandlerDefinition
	case strings.Contains(normalized, "TYPE_MISMATCH"), strings.Contains(normalized, "CANNOT_USE"):
		return IntentTypeMismatch
	case strings.Contains(normalized, "UNUSED_VAR"), strings.Contains(normalized, "DECLARED_NOT_USED"):
		return IntentUnusedVariable
	case strings.Contains(normalized, "UNUSED_IMPORT"), strings.Contains(normalized, "IMPORTED_NOT_USED"):
		return IntentUnusedImport
	case strings.Contains(normalized, "MISSING_RETURN"):
		return IntentMissingReturn
	case strings.Contains(normalized, "SYNTAX"):
		return IntentSyntaxError
	case strings.Contains(normalized, "MISSING_METHOD"), strings.Contains(normalized, "NO_FIELD_OR_METHOD"):
		return IntentMissingMethod
	case strings.Contains(normalized, "UNDECLARED"):
		return IntentUndeclaredName
	default:
		return IntentUnknown
	}
}

// Ensure LoadMoEClassificationModelFromGOB is imported correctly
var _ = fmt.Sprintf // avoid unused import
