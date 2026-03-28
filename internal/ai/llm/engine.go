package llm

import (
	"fmt"
	"github.com/golangast/gollemer/internal/ai/moe"
)

// GollemerEngine acts as the central orchestrator for processing user inputs.
// It uses an NLP model to detect intents and a ConversationBuffer to resolve context.
type GollemerEngine struct {
	NLP      *moe.IntentMoE      // The Mixture-of-Experts Classifier
	History  *ConversationBuffer // The Memory Stack with local persistence
}

// NewGollemerEngine initializes the engine and loads the persistent conversation state.
func NewGollemerEngine(model *moe.IntentMoE) (*GollemerEngine, error) {
	buffer, err := LoadBuffer()
	if err != nil {
		return nil, err
	}
	return &GollemerEngine{
		NLP:     model,
		History: buffer,
	}, nil
}

// Process runs the "Director" loop on a user's natural language command.
// It classifies intents, resolves pronouns ("it"), and manages confirmation flows.
func (e *GollemerEngine) Process(userInput string) string {
	// 1. Model Inference (Classify and Confidence)
	// (Assuming NLP model returns logits; we apply Softmax then ArgMax)
	// tokens := Tokenize(userInput)
	// logits := e.NLP.Forward(tokens...)
	// probabilities := tensor.Softmax(logits)
	// idx, confidence := tensor.ArgMax(probabilities)
	
	// Mock inference for logic demonstration (integration with real Forward pass below)
	idx, confidence := 0, 0.90 // file_delete
	
	// Use Safety Floor before mapping to frame
	if confidence < 0.85 {
		meta, _ := IntentRegistry[idx]
		return fmt.Sprintf("ʕ◔ϖ◔ʔ I'm not quite sure. Did you mean to %s?", meta.Label)
	}

	// 2. Map high-level output to a Life-Cycle Frame
	frame := MapToFrame(idx, confidence, userInput)

	// 3. Pronoun Resolution ("it", "them", "that")
	// If the user says "Delete it", we resolve "it" to the most recent target.
	if val, ok := frame.Entities["target"]; (ok && val == "it") || (!ok && frame.Label == "file_delete") {
		resolved := e.History.ResolveEntity("target")
		if resolved != "" {
			frame.Entities["target"] = resolved
		}
	}

	// 4. Update Memory Stack with the current turn
	e.History.Push(*frame)
	e.History.Save()

	// 5. Director Pattern: Act based on the Intent Lifecycle Status
	switch frame.Status {
	case AwaitingParams:
		return fmt.Sprintf("ʕ◔ϖ◔ʔ I understand you want to %s, but which target should I use?", frame.Label)
		
	case AwaitingConfirm:
		// Resolve a name/path for better confirmation messaging
		target := frame.Entities["target"]
		if target == "" { target = "this action" }
		return fmt.Sprintf("⚠️ [SAFETY] I'm ready to %s '%s'. Proceed? (y/n)", frame.Label, target)
		
	case AwaitingClarify:
		return "ʕ◔ϖ◔ʔ I'm not clear on that. Could you rephrase your command?"
		
	case Fulfilled:
		// Execute using the Action Dispatcher built previously
		err := Dispatch(frame.Label, frame.Entities)
		if err != nil {
			return fmt.Sprintf("❌ Error executing %s: %v", frame.Label, err)
		}
		return fmt.Sprintf("✅ I have successfully %s.", frame.Label)
		
	default:
		return "ʕ◕ϖ◕ʔ System Error: Unhandled frame status."
	}
}
