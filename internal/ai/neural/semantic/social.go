package semantic

// SocialInteraction represents the complex structure of a social exchange.
type SocialInteraction struct {
	Intent            string             `json:"intent"`             // e.g., "greeting", "wellbeing_check", "farewell"
	SubIntent         string             `json:"sub_intent"`         // e.g., "personal_how_are_you"
	SpeechAct         string             `json:"speech_act"`         // e.g., "inquiry", "assertion", "directive"
	SocialDynamics    SocialDynamics     `json:"social_dynamics"`
	InternalState     InternalState      `json:"internal_state"`
	ResponseStrategy  ResponseStrategy   `json:"response_strategy"`
	TargetEntities    []SocialEntity     `json:"target_entities"`
}

// SocialDynamics defines the tone and relationship context.
type SocialDynamics struct {
	Tone               string  `json:"tone"`                // e.g., "friendly", "formal", "ironic"
	Formality          string  `json:"formality"`           // e.g., "casual", "professional"
	SentimentAlignment string  `json:"sentiment_alignment"` // e.g., "positive", "neutral", "negative"
	ReciprocityLevel   float64 `json:"reciprocity_level"`   // How much the bot should give back
}

// InternalState represents the bot's "mood" or system health.
type InternalState struct {
	AgentMood    string         `json:"agent_mood"`    // e.g., "helpful", "curious", "busy"
	EnergyLevel  string         `json:"energy_level"`  // e.g., "high", "low" (computational availability)
	SystemHealth map[string]any `json:"system_health"` // e.g., {"cpu": "nominal", "memory": "ok"}
}

// ResponseStrategy guides the generation of the final output.
type ResponseStrategy struct {
	PrimaryAction   string   `json:"primary_action"`   // e.g., "acknowledge_and_respond"
	SecondaryAction string   `json:"secondary_action"` // e.g., "ask_back"
	TemplateID      string   `json:"template_id"`      // e.g., "polite_wellbeing_reply"
	MultimodalHints []string `json:"multimodal_hints"` // e.g., "smile_icon", "soft_tone"
}

// SocialEntity represents participants in the interaction.
type SocialEntity struct {
	Name     string `json:"name"`
	Type     string `json:"type"` // e.g., "USER", "AGENT", "MENTIONED_PERSON"
	Relation string `json:"relation"`
}
