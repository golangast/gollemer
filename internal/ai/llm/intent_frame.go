package llm

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// IntentStatus represents the lifecycle stage of a detected intent.
type IntentStatus int

const (
	AwaitingParams  IntentStatus = iota // Missing required data like path or target
	AwaitingConfirm                     // Need explicit user "yes/no" before taking action
	AwaitingClarify                     // Model confidence too low, need to ask what user meant
	Fulfilled                           // Action successfully completed
	Failed                               // Error encountered during execution
)

// IntentFrame tracks the complete lifecycle and context of an intent turn.
type IntentFrame struct {
	ID        string            `json:"id"`       // Unique ID for the turn
	Label     string            `json:"label"`    // e.g., "file_manage"
	Entities  map[string]string `json:"entities"` // Captured data (e.g., path: "/var/log", resource: "cpu")
	Status    IntentStatus      `json:"status"`   // The current state of the intent
	Context   string            `json:"context"`  // Parent context (e.g., "deployment_flow")
	Timestamp time.Time         `json:"ts"`
}

// ConversationBuffer serves as the "Memory Stack" for resolving pronouns and tracking history.
type ConversationBuffer struct {
	Frames []IntentFrame `json:"frames"`
	Limit  int           `json:"limit"`
}

const statePath = "/tmp/gollemer_state.json"

// Save persists the conversation buffer to a local file for cross-session memory.
func (cb *ConversationBuffer) Save() error {
	data, err := json.MarshalIndent(cb, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(statePath, data, 0644)
}

// LoadBuffer restores the conversation buffer from the state file.
func LoadBuffer() (*ConversationBuffer, error) {
	data, err := os.ReadFile(statePath)
	if err != nil {
		// Return a fresh buffer if no state exists
		return &ConversationBuffer{Limit: 5, Frames: []IntentFrame{}}, nil
	}
	var cb ConversationBuffer
	if err := json.Unmarshal(data, &cb); err != nil {
		return &ConversationBuffer{Limit: 5, Frames: []IntentFrame{}}, err
	}
	return &cb, nil
}

// ResolveEntity walks backward through history to find the most recent entity of a specific type.
// Useful for resolving pronouns like "it" when the current intent lacks data.
func (cb *ConversationBuffer) ResolveEntity(entityType string) string {
	for i := len(cb.Frames) - 1; i >= 0; i-- {
		if val, ok := cb.Frames[i].Entities[entityType]; ok {
			return val
		}
	}
	return ""
}

// ResolveReference handles singular ("it", "that") vs plural ("them", "those") resolution.
func (cb *ConversationBuffer) ResolveReference(ref string) ([]string, error) {
	for i := len(cb.Frames) - 1; i >= 0; i-- {
		// In our system, "target" is the generic name for files/folders/resources
		var targets []string
		if val, ok := cb.Frames[i].Entities["target"]; ok {
			targets = append(targets, val)
		}
		if val, ok := cb.Frames[i].Entities["name"]; ok {
			targets = append(targets, val)
		}
		
		if len(targets) == 0 {
			continue
		}

		switch ref {
		case "it", "that", "this":
			if len(targets) == 1 {
				return targets, nil
			}
		case "them", "those", "all":
			if len(targets) >= 1 {
				return targets, nil
			}
		}
	}
	return nil, fmt.Errorf("could not resolve reference: %s", ref)
}

// Push adds a new frame to the stack and maintains the size limit.
func (cb *ConversationBuffer) Push(frame IntentFrame) {
	cb.Frames = append(cb.Frames, frame)
	if len(cb.Frames) > cb.Limit {
		cb.Frames = cb.Frames[len(cb.Frames)-cb.Limit:]
	}
}
