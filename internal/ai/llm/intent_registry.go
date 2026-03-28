package llm

import (
	"time"
)

// IntentMetadata stores the static policy of each intent (what it needs vs what it knows).
type IntentMetadata struct {
	Label          string
	RequiresEntity bool
	NeedsConfirm    bool
	DefaultEntity   string // e.g. "target" or "resource"
}

// IntentRegistry maps model output indices from softmax to their respective architectural policies.
// This is used for "Safety Floor" and "Mapping Logic" instead of hardcoding into the training loop.
var IntentRegistry = map[int]IntentMetadata{
	0: {Label: "file_delete", RequiresEntity: true, NeedsConfirm: true, DefaultEntity: "target"},
	1: {Label: "sys_monitor", RequiresEntity: false, NeedsConfirm: false, DefaultEntity: "resource"},
	2: {Label: "REF_ACTION", RequiresEntity: true, NeedsConfirm: true, DefaultEntity: "reference"},
}

// MapToFrame takes high-level model output and initializes the dynamic IntentFrame.
func MapToFrame(classIdx int, confidence float64, input string) *IntentFrame {
	meta, ok := IntentRegistry[classIdx]
	
	// Safety Floor: If confidence is too low, we trigger Clarify mode.
	if !ok || confidence < 0.80 {
		return &IntentFrame{
			Label:  "unknown",
			Status: AwaitingClarify,
			Timestamp: time.Now(),
		}
	}

	// Heuristic extraction based on the metadata (calling the existing extractor)
	// We'll convert the []Entity to map[string]string for the Frame.
	entitiesList := ExtractEntities(meta.Label, input)
	entitiesMap := make(map[string]string)
	for _, e := range entitiesList {
		entitiesMap[e.Type] = e.Value
	}

	frame := &IntentFrame{
		Label:    meta.Label,
		Entities: entitiesMap,
		Status:   Fulfilled, // Default to fulfilled unless metadata says otherwise
		Timestamp: time.Now(),
	}

	// Dynamic status adjustment based on intent requirements
	if meta.RequiresEntity && len(entitiesMap) == 0 {
		frame.Status = AwaitingParams
	} else if meta.NeedsConfirm {
		frame.Status = AwaitingConfirm
	}

	return frame
}
