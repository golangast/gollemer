package moe

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nn"
	"github.com/golangast/gollemer/internal/ai/neural/tensor"
)

// IntentTagger represents a model that predicts intent and tags for a sequence.
type IntentTagger struct {
	Encoder    *MoELayer
	Embedding  *nn.Embedding
	IntentHead *nn.Linear
	TagHead    *nn.Linear
}

// EntityTriple represents a Subject-Action-Object triple extracted from text.
type EntityTriple struct {
	Subject string
	Action  string
	Object  string
}

// NewIntentTagger creates a new IntentTagger model.
func NewIntentTagger(vocabSize, embeddingDim, numExperts, intentVocabSize, tagVocabSize int) (*IntentTagger, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

	// Define the expert builder function
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewFeedForwardExpert(embeddingDim, embeddingDim, embeddingDim)
	}

	// Initialize the MoE encoder
	encoder, err := NewMoELayer(embeddingDim, embeddingDim, numExperts, 1, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE encoder: %w", err)
	}

	// Initialize the output heads
	intentHead, err := nn.NewLinear(embeddingDim, intentVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create intent head: %w", err)
	}
	tagHead, err := nn.NewLinear(embeddingDim, tagVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create tag head: %w", err)
	}

	return &IntentTagger{
		Encoder:    encoder,
		Embedding:  embedding,
		IntentHead: intentHead,
		TagHead:    tagHead,
	}, nil
}

// Forward performs the forward pass of the IntentTagger model.
func (m *IntentTagger) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if len(inputs) != 1 {
		return nil, nil, fmt.Errorf("IntentTagger.Forward expects 1 input (query token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Encoder forward pass
	encodedSequence, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	// For intent prediction, we can take the mean of the encoded sequence
	contextVector, err := encodedSequence.Mean(1)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get mean of encoded sequence: %w", err)
	}

	// Intent head
	intentLogits, err := m.IntentHead.Forward(contextVector)
	if err != nil {
		return nil, nil, fmt.Errorf("intent head forward failed: %w", err)
	}

	// Tag head - vectorized apply to all tokens in the sequence
	tagLogits, err := m.TagHead.Forward(encodedSequence)
	if err != nil {
		return nil, nil, fmt.Errorf("tag head forward failed: %w", err)
	}

	return intentLogits, tagLogits, nil
}

// Parameters returns all learnable parameters of the IntentTagger model.
func (m *IntentTagger) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	params = append(params, m.Embedding.Parameters()...)
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.IntentHead.Parameters()...)
	params = append(params, m.TagHead.Parameters()...)
	return params
}

// Backward performs the backward pass for the IntentTagger model.
func (m *IntentTagger) Backward(intentGrad, tagGrads *tensor.Tensor) error {
	// Backward pass for the heads
	if err := m.IntentHead.Backward(intentGrad); err != nil {
		return fmt.Errorf("intent head backward failed: %w", err)
	}
	if err := m.TagHead.Backward(tagGrads); err != nil {
		return fmt.Errorf("tag head backward failed: %w", err)
	}

	// Combine gradients for the encoder
	tagEncoderGrad := m.TagHead.Inputs()[0].Grad

	// Backward pass for the encoder - it now returns the input gradient
	err := m.Encoder.Backward(tagEncoderGrad)
	if err != nil {
		return fmt.Errorf("MoE encoder backward failed: %w", err)
	}

	// Backward pass for the embedding layer using the returned gradient
	if len(m.Encoder.Inputs()) > 0 {
		embeddingGrad := m.Encoder.Inputs()[0].Grad
		if err := m.Embedding.Backward(embeddingGrad); err != nil {
			return fmt.Errorf("embedding layer backward failed: %w", err)
		}
	}

	return nil
}

// ExtractEntities performs lightweight entity extraction from text using the
// tagger's embeddings and a lexicon-based heuristic. It returns Subject-Action-
// Object triples suitable for routing to specialized expert cartridges.
func (m *IntentTagger) ExtractEntities(text string) []EntityTriple {
	lower := strings.ToLower(text)
	var triples []EntityTriple

	subjectKeywords := map[string]string{
		"channel": "Channel", "goroutine": "Goroutine", "mutex": "Mutex",
		"interface": "Interface", "error": "Error", "context": "Context",
		"slice": "Slice", "map": "Map", "defer": "Defer", "init": "InitFunction",
		"package": "Package", "module": "Module", "struct": "Struct",
		"function": "Function", "vendor": "VendorDirectory", "test": "Test",
		"log": "Logger", "database": "Database", "http": "HTTPServer",
		"middleware": "Middleware", "panic": "Panic", "race": "RaceCondition",
		"build": "Build", "config": "Config", "dependency": "Dependency",
		"garbage collector": "GarbageCollector",
	}
	actionKeywords := map[string]string{
		"send": "Send", "receive": "Receive", "close": "Close",
		"lock": "Lock", "unlock": "Unlock", "protect": "Protect",
		"wrap": "Wrap", "propagate": "Propagate", "cancel": "Cancel",
		"schedule": "Schedule", "execute": "Execute", "run": "Run",
		"communicate": "Communicate", "synchronize": "Synchronize",
		"store": "Store", "embed": "Embed", "build": "Build",
		"import": "Import", "resolve": "Resolve", "manage": "Manage",
		"handle": "Handle", "implement": "Implement", "define": "Define",
	}
	objectKeywords := map[string]string{
		"channel": "Channel", "goroutine": "Goroutine", "mutex": "Mutex",
		"interface": "Interface", "error": "Error", "context": "Context",
		"slice": "Slice", "map": "Map", "defer": "Defer", "init": "InitFunction",
		"package": "Package", "module": "Module", "struct": "Struct",
		"function": "Function", "vendor": "VendorDirectory", "test": "Test",
		"log": "Logger", "database": "Database", "http": "HTTPServer",
		"middleware": "Middleware", "race": "RaceCondition",
		"build": "Build", "config": "Config", "dependency": "Dependency",
		"garbage collector": "GarbageCollector", "zero value": "ZeroValue",
		"panic": "Panic", "deadlock": "Deadlock",
	}

	var subjects, actions, objects []string
	for kw, label := range subjectKeywords {
		if strings.Contains(lower, kw) {
			subjects = append(subjects, label)
		}
	}
	for kw, label := range actionKeywords {
		if strings.Contains(lower, kw) {
			actions = append(actions, label)
		}
	}
	for kw, label := range objectKeywords {
		if strings.Contains(lower, kw) {
			objects = append(objects, label)
		}
	}

	if len(subjects) == 0 {
		subjects = append(subjects, "Code")
	}
	if len(actions) == 0 {
		actions = append(actions, "Use")
	}
	if len(objects) == 0 {
		objects = append(objects, "Implementation")
	}

	maxLen := len(subjects)
	if len(actions) > maxLen {
		maxLen = len(actions)
	}
	if len(objects) > maxLen {
		maxLen = len(objects)
	}
	for i := 0; i < maxLen; i++ {
		subj := subjects[i%len(subjects)]
		act := actions[i%len(actions)]
		obj := objects[i%len(objects)]
		if subj != obj {
			triples = append(triples, EntityTriple{
				Subject: subj,
				Action:  act,
				Object:  obj,
			})
		}
	}

	return triples
}
