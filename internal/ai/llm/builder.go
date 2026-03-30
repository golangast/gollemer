package llm

import "strings"

// BuildIntentObject takes raw tokens and their predicted IOB tags (from the NER/Tagging layer)
// and maps them into an IntentObject. This implements "Slot Filling" where "AI magic"
// becomes predictable, structured Go code.
func BuildIntentObject(tokens []string, tags []string, intent string, confidence float64) IntentObject {
	obj := IntentObject{
		Action:     intent,
		Confidence: confidence,
		Entities:   make(map[string]string),
	}

	for i := 0; i < len(tags); i++ {
		tag := tags[i]
		if strings.HasPrefix(tag, "B-") {
			key := strings.TrimPrefix(tag, "B-")
			value := tokens[i]
			
			// Handle multi-token entities (Inside tags)
			for j := i + 1; j < len(tags); j++ {
				if strings.HasPrefix(tags[j], "I-"+key) {
					value += " " + tokens[j]
					i = j // Advance main loop index
				} else {
					break
				}
			}
			obj.Entities[key] = value
		}
	}
	return obj
}
