package llm

import "fmt"

// IntentAction defines the interface for executable actions mapped to predicted intents.
type IntentAction interface {
	Execute(params map[string]string) error
}

// IntentHandler is a function-based alternative for dispatching intents.
type IntentHandler func(entities map[string]string) error

// ActionRegistry maps intent labels to their respective handler functions.
var ActionRegistry = map[string]IntentHandler{
	"sys_monitor": handleSysMonitor,
	"file_manage":  handleFileManage,
}

// Dispatch routes the predicted intent and extracted entities to the correct action logic.
func Dispatch(intent string, entities map[string]string) error {
	// 1. Try Function Registry first
	if handler, ok := ActionRegistry[intent]; ok {
		return handler(entities)
	}

	// 2. Fallback to Switch Statement (The basic pattern)
	switch intent {
	case "file_create":
		return CreateFileAction{}.Execute(entities)
	case "system_status":
		return CheckStatusAction{}.Execute(entities)
	default:
		return fmt.Errorf("unsupported intent: %s", intent)
	}
}

// --- Example Handler Implementations ---

func handleSysMonitor(ent map[string]string) error {
	// Logic for running 'top' or 'df -h' based on entities
	fmt.Printf("📊 Routing to System Monitor... resource: %s\n", ent["resource"])
	return nil
}

func handleFileManage(ent map[string]string) error {
	fmt.Printf("📁 Routing to File Manager... target: %s\n", ent["target"])
	return nil
}

// --- Basic Action Structs for the Switch fallback ---

type CreateFileAction struct{}
func (a CreateFileAction) Execute(params map[string]string) error {
	fmt.Println("Creating file with params:", params)
	return nil
}

type CheckStatusAction struct{}
func (a CheckStatusAction) Execute(params map[string]string) error {
	fmt.Println("Checking system status...")
	return nil
}
