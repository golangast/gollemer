package llm

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/platform/ui"
	"github.com/golangast/gollemer/internal/util/colors"
)

func (r *Runner) handleInteractiveQuery(query string) {
	if query == "" {
		return
	}

	// 1. Session Logic
	if r.SessionState.IsActive {
		r.handleSessionInput(query)
		return
	}

	// 2. Intent Animation
	stopAnim := make(chan bool)
	go r.Mascot.Spin([]string{"◡ϖ◡", "⊙ϖ⊙", "◠ϖ◠", "⊙ϖ⊙"}, "Thinking", stopAnim)

	intentData := r.Resolver.Resolve(query, nil)

	stopAnim <- true
	time.Sleep(50 * time.Millisecond)

	// 3. Early Intent Handling (Confirmation prompts etc.)
	if r.handleEarlyIntent(intentData, query) {
		return
	}

	// 4. Detailed Parsing
	intent := parse(query, r.KB)
	// 5. Response / Chat handling
	resp := r.extractResponse(intentData)
	hasCommand := intent.Command != "" || isCreatingCommand(query)

	if resp != "" {
		r.Mascot.Speak(ui.MoodIdle, resp)
		r.Client.PushHistory(query, resp, intentData.Intent)
		if !hasCommand {
			return
		}
	}

	// 6. Command Execution
	r.executeCommand(query, &intent, intentData)
}

func (r *Runner) handleSessionInput(query string) {
	if len(r.SessionState.Missing) > 0 {
		field := r.SessionState.Missing[0]
		r.SessionState.Parameters[field] = query
		r.SessionState.Missing = r.SessionState.Missing[1:]
	}

	if len(r.SessionState.Missing) > 0 {
		fmt.Printf("You need to provide a %s.\n", r.SessionState.Missing[0])
		return
	}

	r.SessionState.IsActive = false
	// Re-process with filled parameters
	r.handleInteractiveQuery(query)
}

func (r *Runner) handleEarlyIntent(intentData *IntentDataLayer, query string) bool {
	if intentData.Intent == "create_file" && !r.SessionState.WaitingForConfirm && !r.SessionState.IsActive {
		filename := ""
		if fn, ok := intentData.Parameters["name"].(string); ok {
			filename = fn
		}
		if filename != "" {
			r.SessionState.WaitingForConfirm = true
			r.SessionState.PendingAction = "create file"
			r.SessionState.PendingData = filename
			r.Mascot.Speak(ui.MoodWaiting, fmt.Sprintf("I noticed you mentioned %s. Should I scaffold that file for you?", filename))
			return true
		}
	}
	return false
}

func (r *Runner) extractResponse(intentData *IntentDataLayer) string {
	if intentData.Parameters != nil {
		if res, ok := intentData.Parameters["response"].(string); ok {
			return res
		}
	}
	return r.Client.lastMoEPrediction
}

func (r *Runner) executeCommand(query string, intent *Intent, intentData *IntentDataLayer) {
	command := intent.Command
	objectType := intent.ObjectType
	fileName := intent.Params["name"]
	if fileName == "" && intentData.Parameters != nil {
		if n, ok := intentData.Parameters["name"].(string); ok {
			fileName = n
		}
	}
	targetDirectory := intent.Params["target"]
	if targetDirectory == "" && intentData.Parameters != nil {
		if t, ok := intentData.Parameters["path"].(string); ok {
			targetDirectory = t
		}
	}

	// Direct parse for create_handler: extract name and url straight from the raw query
	handlerURL := intent.Params["url"]
	if intentData.Intent == "create_handler" || (command == "create" && objectType == "handler") {
		parts := strings.Fields(query)
		for i, p := range parts {
			lp := strings.ToLower(p)
			// Name: word after "handler" (skip "named","called")
			if (lp == "handler") && i+1 < len(parts) {
				j := i + 1
				for j < len(parts) {
					next := strings.ToLower(parts[j])
					if next == "named" || next == "called" || next == "with" {
						j++
						continue
					}
					if !strings.HasPrefix(next, "/") && fileName == "" {
						fileName = parts[j]
					}
					break
				}
			}
			// URL: word after "url"
			if (lp == "url" || lp == "at") && i+1 < len(parts) {
				handlerURL = parts[i+1]
			}
		}
	}
	if targetDirectory == "" && (command == "go" || command == "list" || command == "tree") {
		parts := strings.Fields(query)
		for i, p := range parts {
			lp := strings.ToLower(p)
			if lp == "go" || lp == "cd" || lp == "list" || lp == "ls" || lp == "tree" {
				if i+1 < len(parts) {
					targetDirectory = parts[i+1]
					break
				}
			}
		}
	}

	var predictedSentence string
	handled := true

	switch command {
	case "go":
		predictedSentence = r.handleGoCommand(targetDirectory)
	case "move":
		predictedSentence = r.handleMoveCommand(fileName, targetDirectory)
	case "list":
		predictedSentence = r.handleListCommand(objectType, intent.ObjectTypeParts, targetDirectory)
	case "tree":
		predictedSentence = r.handleTreeCommand(targetDirectory)
	case "grep":
		predictedSentence = r.handleGrepCommand(intent.Params["target"], targetDirectory)
	case "history":
		predictedSentence = r.handleHistoryCommand()
	case "help":
		predictedSentence = r.handleHelpCommand(intentData)
	case "create":
		predictedSentence = r.handleTemplateCreate(objectType, fileName, targetDirectory, handlerURL)
	case "identity":
		predictedSentence = "I am Gollemer, your AI coding assistant and project orchestrator."
	case "greeting":
		predictedSentence = "Hello! I'm ready to help you build something awesome. Try 'create webserver myapp'."
	case "status":
		predictedSentence = "I am functioning within normal parameters. Ready for your next command!"
	case "pwd":
		cwd, _ := os.Getwd()
		predictedSentence = fmt.Sprintf("The current directory is: %s", cwd)
	default:
		handled = false
	}

	if !handled {
		predictedSentence = "|ʕ>ϖ<ʔ| I'm sorry, I couldn't understand your request."
	}

	predictedSentence = r.handleTutorialLogic(command, objectType, predictedSentence)
	colors.AnimatedOutput("green", "black", predictedSentence, 1*time.Second)
	r.Client.PushHistory(query, predictedSentence, intentData.Intent)
}

func (r *Runner) handleTreeCommand(targetDirectory string) string {
	target := "."
	if targetDirectory != "" {
		target = targetDirectory
	}
	treeView, err := generateDirectoryTree(target, "", 0, -1, "")
	if err != nil {
		return fmt.Sprintf("I couldn't generate a tree for '%s': %v", target, err)
	}
	return fmt.Sprintf("Directory tree for '%s':\n%s", target, treeView)
}

func (r *Runner) handleHistoryCommand() string {
	limit := 10
	if limit > len(r.CommandHistory) {
		limit = len(r.CommandHistory)
	}
	var historyLines []string
	for i := len(r.CommandHistory) - limit; i < len(r.CommandHistory); i++ {
		historyLines = append(historyLines, fmt.Sprintf("%d  %s", i+1, r.CommandHistory[i]))
	}
	return strings.Join(historyLines, "\n")
}

func (r *Runner) handleTemplateCreate(objectType, fileName, targetDirectory, handlerURL string) string {
	msg, _ := handleGenericCreate(objectType, fileName, targetDirectory, handlerURL, r.KB)
	return msg
}
