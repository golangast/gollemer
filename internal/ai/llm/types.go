package llm

// ChatPair represents a question-answer pair for conversational history.
type ChatPair struct {
	Q      string `json:"prompt"`
	A      string `json:"response"`
	Intent string `json:"intent"`
}

// ConversationState holds the current state of an ongoing dialogue.
type ConversationState struct {
	ActiveIntent      string
	Parameters        map[string]string
	Missing           []string
	IsActive          bool
	SuggestedObject   string
	WaitingForConfirm bool
	PendingAction     string
	PendingData       string
	JustConfirmed     bool
}

// TutorialState tracks the user's progress through the guided tutorial.
type TutorialState struct {
	Active bool
	Step   int
}
