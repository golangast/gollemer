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

	// 0. Handle prompts/confirmations that are outside the NLP intent flow
	if r.SessionState.WaitingForConfirm {
		lower := strings.ToLower(query)
		if lower == "y" || lower == "yes" {
			r.SessionState.WaitingForConfirm = false
			// Execute the pending action
			switch r.SessionState.PendingAction {
			case "create file":
				fileName := r.SessionState.PendingData
				resp := r.handleTemplateCreate("file", fileName, "", "")
				r.Mascot.Speak(ui.MoodHappy, resp)
			case "create folder":
				folderName := r.SessionState.PendingData
				resp := r.handleTemplateCreate("folder", folderName, "", "")
				r.Mascot.Speak(ui.MoodHappy, resp)
			}
			r.SessionState.PendingAction = ""
			r.SessionState.PendingData = ""
			return
		} else if lower == "n" || lower == "no" {
			r.SessionState.WaitingForConfirm = false
			r.SessionState.PendingAction = ""
			r.SessionState.PendingData = ""
			r.Mascot.Speak(ui.MoodHappy, "Alright, I'll ignore that for now.")
			return
		}
	}

	// 1. Session Logic: If we have an active intent waiting for data, 
	// we process the new input as part of that intent's context.
	var prevIntent *IntentDataLayer
	if r.SessionState.IsActive {
		prevIntent = r.SessionState.CurrentIntent
	}

	// 2. Intent Animation
	stopAnim := make(chan bool)
	go r.Mascot.Spin([]string{"◡ϖ◡", "⊙ϖ⊙", "◠ϖ◠", "⊙ϖ⊙"}, "Thinking", stopAnim)

	// RESOLVE: Build intent object using NLP/MoE, passing prevIntent for context memory.
	intentData := r.Resolver.Resolve(query, prevIntent)

	stopAnim <- true
	time.Sleep(50 * time.Millisecond)

	// 3. Early Intent Handling (Confirmation prompts etc.)
	if r.handleEarlyIntent(intentData, query) {
		return
	}

	// 4. Memory Persistence: If incomplete, store it and ask for more.
	if !intentData.IsComplete && intentData.Intent != "" && intentData.Intent != "chat_response" && len(intentData.Missing) > 0 {
		r.SessionState.IsActive = true
		r.SessionState.CurrentIntent = intentData
		r.Mascot.Speak(ui.MoodWaiting, fmt.Sprintf("I've identified your intent as '%s', but I'm missing some details: %s. Could you provide the %s?", 
			intentData.Intent, strings.Join(intentData.Missing, ", "), intentData.Missing[0]))
		return
	}

	// 5. Detailed Parsing Fallback (Keyword-based)
	intent := parse(query, r.KB)
	
	// 6. Response / Chat handling
	resp := r.extractResponse(intentData)
	// hasCommand is true only if there's a structural command to execute.
	// Simple chat intents should not trigger executeCommand.
	hasCommand := intent.Command != "" || isCreatingCommand(query)

	if resp != "" {
		r.Mascot.Speak(ui.MoodIdle, resp)
		r.Client.PushHistory(query, resp, intentData.Intent)
		if !hasCommand {
			return
		}
	}

	// 7. Success: Execution and State Cleanup
	r.executeCommand(query, &intent, intentData, resp)
	r.SessionState.IsActive = false
	r.SessionState.CurrentIntent = nil
}

func (r *Runner) handleSessionInput(query string) {
	// Logic moved into handleInteractiveQuery for unified processing.
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

func (r *Runner) executeCommand(query string, intent *Intent, intentData *IntentDataLayer, chatResponse string) {
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
	handlerURL := intent.Params["url"]

	// Shared helper: extract a directory from the raw query, skipping filler prepositions.
	extractDir := func() string {
		if targetDirectory != "" {
			return targetDirectory
		}
		skipWords := map[string]bool{"into": true, "to": true, "in": true, "up": true, "inside": true}
		parts := strings.Fields(query)
		for i, p := range parts {
			lp := strings.ToLower(p)
			if lp == "go" || lp == "cd" || lp == "goto" || lp == "list" || lp == "ls" || lp == "tree" {
				j := i + 1
				for j < len(parts) && skipWords[strings.ToLower(parts[j])] {
					j++
				}
				if j < len(parts) {
					return parts[j]
				}
			}
		}
		return ""
	}

	// If the NLP gave us a create_handler intent, do a richer raw-query parse for name/url.
	if intentData.Intent == "create_handler" || (command == "create" && objectType == "handler") {
		parts := strings.Fields(query)
		for i, p := range parts {
			lp := strings.ToLower(p)
			if lp == "handler" && i+1 < len(parts) {
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
			if (lp == "url" || lp == "at") && i+1 < len(parts) {
				handlerURL = parts[i+1]
			}
		}
	}

	var predictedSentence string
	handled := true

	// ── NLP-first dispatch ────────────────────────────────────────────────────
	// intentData.Intent is populated by HybridIntentResolver / the MoE model.
	// We map those semantic intents to handlers here so command execution is
	// driven by the trained model rather than hardcoded keyword matching.
	switch intentData.Intent {

	// ── Create family ─────────────────────────────────────────────────────────
	case "create_folder":
		predictedSentence = r.handleTemplateCreate("folder", fileName, targetDirectory, "")
	case "create_file":
		predictedSentence = r.handleTemplateCreate("file", fileName, targetDirectory, "")
	case "create_webserver":
		predictedSentence = r.handleTemplateCreate("webserver", fileName, targetDirectory, "")
	case "create_handler":
		predictedSentence = r.handleTemplateCreate("handler", fileName, targetDirectory, handlerURL)
	case "create_page":
		predictedSentence = r.handleTemplateCreate("page", fileName, targetDirectory, "")
	case "create_form":
		predictedSentence = r.handleTemplateCreate("form", fileName, targetDirectory, "")
	case "create_database":
		predictedSentence = r.handleTemplateCreate("database", fileName, targetDirectory, "")
	case "create_structure":
		predictedSentence = r.handleTemplateCreate("structure", fileName, targetDirectory, "")
	case "create_generic":
		predictedSentence = r.handleTemplateCreate(objectType, fileName, targetDirectory, handlerURL)

	// ── Navigation ────────────────────────────────────────────────────────────
	case "go_query":
		predictedSentence = r.handleGoCommand(extractDir())
	case "list_query":
		predictedSentence = r.handleListCommand(objectType, intent.ObjectTypeParts, extractDir())
	case "tree_query":
		predictedSentence = r.handleTreeCommand(extractDir())

	// ── Webserver lifecycle ───────────────────────────────────────────────────
	case "run_webserver", "run_query":
		command = "run"
		objectType = "webserver"
		predictedSentence = r.handleTemplateCreate("webserver", fileName, targetDirectory, "")
	case "stop":
		command = "stop"
		predictedSentence = r.handleTemplateCreate("stop", fileName, targetDirectory, "")

	// ── Utility ───────────────────────────────────────────────────────────────
	case "cat_query":
		predictedSentence = r.handleGrepCommand(fileName, targetDirectory)
	case "help_command":
		predictedSentence = r.handleHelpCommand(intentData)
	case "history_query":
		predictedSentence = r.handleHistoryCommand()
	case "pwd_query":
		cwd, _ := os.Getwd()
		predictedSentence = fmt.Sprintf("The current directory is: %s", cwd)

	// ── Social / personality ──────────────────────────────────────────────────
	case "identity_query":
		predictedSentence = "I am Gollemer, your AI coding assistant and project orchestrator."
	case "greeting", "greeting_query":
		predictedSentence = "Hello! I'm ready to help you build something awesome. Try 'create webserver myapp'."
	case "status_query":
		predictedSentence = "I am functioning within normal parameters. Ready for your next command!"

	default:
		// ── Legacy keyword-parsed fallback ────────────────────────────────────
		// Used when the NLP model returns an intent we don't have a handler for
		// yet, or when training data is sparse for a pattern.
		switch command {
		case "go":
			predictedSentence = r.handleGoCommand(extractDir())
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
	}

	if !handled && chatResponse == "" {
		predictedSentence = "|ʕ>ϖ<ʔ| I'm sorry, I couldn't understand your request."
	}

	// Derive canonical command/objectType from the NLP intent so tutorial logic
	// and history look correct regardless of which dispatch path ran.
	if strings.Contains(intentData.Intent, "_") {
		parts := strings.SplitN(intentData.Intent, "_", 2)
		if command == "" {
			command = parts[0]
		}
		if objectType == "" && len(parts) > 1 {
			objectType = parts[1]
		}
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
