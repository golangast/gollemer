package llm

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/golangast/gollemer/internal/hardware"
	"github.com/golangast/gollemer/internal/pipeline"
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

	// 0.5. Pronoun Resolution: if the user says "move it" or "move the file"
	// resolve the pronoun to the last concrete filename mentioned in history.
	{
		lq := strings.ToLower(strings.TrimSpace(query))
		isPronounCmd := strings.Contains(lq, "move it") || strings.Contains(lq, "put it") ||
			strings.Contains(lq, "move the file into") || strings.Contains(lq, "move the file to") ||
			strings.HasSuffix(lq, " the file")

		if isPronounCmd && len(r.Client.History) > 0 {
			resolvedFile := ""
			for i := len(r.Client.History) - 1; i >= 0 && resolvedFile == ""; i-- {
				for _, text := range []string{r.Client.History[i].Q, r.Client.History[i].A} {
					words := strings.Fields(strings.ToLower(text))
					for _, w := range words {
						w = strings.Trim(w, "',\".")
						if strings.Contains(w, ".") && !strings.HasPrefix(w, ".") {
							// Looks like a filename (has extension)
							resolvedFile = w
							break
						}
					}
					if resolvedFile != "" {
						break
					}
				}
			}
			if resolvedFile != "" {
				resolved := strings.ReplaceAll(query, " it ", " "+resolvedFile+" ")
				if strings.Contains(lq, "move the file into") || strings.Contains(lq, "move the file to") {
					resolved = strings.ReplaceAll(resolved, " the file ", " "+resolvedFile+" ")
				} else if strings.HasSuffix(lq, " the file") {
					resolved = strings.TrimSuffix(resolved, " the file") + " " + resolvedFile
				}
				if resolved != query {
					r.Mascot.Speak(ui.MoodIdle, fmt.Sprintf("(Resolved to: '%s')", resolved))
					query = resolved
				}
			}
		}

		// 0.6. Memory queries ("what file", "what folder", "do you remember"...) answered
		// directly from live history before hitting the neural pipeline.
		isMemQ := strings.Contains(lq, "what file") || strings.Contains(lq, "what folder") ||
			strings.Contains(lq, "were we talking") || strings.Contains(lq, "do you remember") ||
			strings.Contains(lq, "do you recall") || strings.Contains(lq, "what did i say") ||
			strings.Contains(lq, "what did we talk") || strings.Contains(lq, "what was i talking") ||
			strings.Contains(lq, "where is the file")

		if isMemQ {
			if len(r.Client.History) == 0 {
				resp := "I don't have any conversation history from this session yet."
				r.Mascot.Speak(ui.MoodIdle, resp)
				r.Client.PushHistory(query, resp, "history_recall")
				return
			}
			resp := r.Client.resolveContextQuery(lq)
			if resp == "" {
				// Generic summary fallback
				var lines []string
				for i, pair := range r.Client.History {
					line := fmt.Sprintf("Turn %d — You: \"%s\"", i+1, pair.Q)
					if pair.A != "" {
						short := pair.A
						if len(short) > 80 {
							short = short[:80] + "…"
						}
						line += fmt.Sprintf(" | Me: \"%s\"", short)
					}
					lines = append(lines, line)
				}
				resp = "Here's what we've discussed:\n" + strings.Join(lines, "\n")
			}
			r.Mascot.Speak(ui.MoodIdle, resp)
			r.Client.PushHistory(query, resp, "history_recall")
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

	// 5.5. Hardware intents bypass the chat response path entirely.
	if isHardwareIntent(intentData.Intent) {
		r.executeCommand(query, &intent, intentData, "")
		r.SessionState.IsActive = false
		r.SessionState.CurrentIntent = nil
		return
	}

	// 6. Response / Chat handling
	resp := r.extractResponse(intentData)
	// hasCommand is true only if there's a structural command to execute.
	// Simple chat intents should not trigger executeCommand.
	isChat := intentData.Intent == "social" || intentData.Intent == "greeting" || intentData.Intent == "goodbye"
	hasCommand := (intent.Command != "" || isCreatingCommand(query)) && !isChat

	if resp != "" {
		if strings.HasPrefix(resp, "<INTENT_") {
			interceptor := pipeline.NewTokenInterceptor()
			var finalProse strings.Builder
			interceptor.OnProse = func(token string) {
				finalProse.WriteString(token + " ")
			}
			stream := pipeline.TokenStreamFromString(resp)
			interceptor.Process(stream)
			cleanResp := strings.TrimSpace(finalProse.String())
			if cleanResp != "" {
				r.Mascot.Speak(ui.MoodIdle, cleanResp)
			}
			r.Client.PushHistory(query, cleanResp, intentData.Intent)
		} else {
			r.Mascot.Speak(ui.MoodIdle, resp)
			r.Client.PushHistory(query, resp, intentData.Intent)
		}
		
		if !hasCommand {
			return
		}
	}

	// 7. Success: Execution and State Cleanup
	r.executeCommand(query, &intent, intentData, resp)
	r.SessionState.IsActive = false
	r.SessionState.CurrentIntent = nil
}

var hardwareIntents = map[string]bool{
	"CAMERA_CAPTURE": true, "CAMERA_CAPTURE_ANALYZE": true,
	"AUDIO_VOLUME_UP": true, "AUDIO_VOLUME_DOWN": true,
	"MICROPHONE_MUTE": true, "MICROPHONE_UNMUTE": true,
}

func isHardwareIntent(intent string) bool {
	return hardwareIntents[intent]
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
		skipWords := map[string]bool{"into": true, "to": true, "in": true, "up": true, "inside": true, "directory": true, "folder": true}
		parts := strings.Fields(query)

		if strings.HasPrefix(strings.ToLower(query), "move ") {
			for i := len(parts) - 2; i >= 0; i-- {
				if skipWords[strings.ToLower(parts[i])] {
					return parts[i+1]
				}
			}
			if len(parts) > 2 {
				return parts[len(parts)-1]
			}
		}

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

	// ── Hardware Intents ──────────────────────────────────────────────────────
	case "CAMERA_CAPTURE", "CAMERA_CAPTURE_ANALYZE", "AUDIO_VOLUME_UP", "AUDIO_VOLUME_DOWN", "MICROPHONE_MUTE", "MICROPHONE_UNMUTE":
		payload := hardware.IntentPayload{
			Intent: intentData.Intent,
			Roles:  make(map[string]string),
		}
		if intentData.Parameters != nil {
			for k, v := range intentData.Parameters {
				if strV, ok := v.(string); ok {
					payload.Roles[k] = strV
				}
			}
		}
		err := hardware.HandleIntent(payload)
		if err != nil {
			predictedSentence = fmt.Sprintf("Hardware error: %v", err)
		} else {
			predictedSentence = fmt.Sprintf("Executed hardware command for intent: %s", intentData.Intent)
		}

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
		case "move_query", "move":
			if fileName == "" {
				parts := strings.Fields(query)
				for i, p := range parts {
					if strings.ToLower(p) == "file" || strings.ToLower(p) == "named" || strings.ToLower(p) == "folder" {
						if i+1 < len(parts) && parts[i+1] != "into" && parts[i+1] != "to" {
							fileName = parts[i+1]
							break
						}
					}
				}
				if fileName == "" && len(parts) >= 3 {
					// e.g. "move jake jimmy" -> parts[0]="move", parts[1]="jake", parts[2]="jimmy"
					fileName = parts[1]
				}
			}
			predictedSentence = r.handleMoveCommand(fileName, extractDir())
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
	
	// Prevent context poisoning: don't store massive terminal outputs or dir trees in history
	historySentence := predictedSentence
	if len(historySentence) > 200 || strings.Contains(historySentence, "├──") || strings.Contains(historySentence, "└──") {
		lines := strings.Split(historySentence, "\n")
		historySentence = lines[0]
	}
	r.Client.PushHistory(query, historySentence, intentData.Intent)
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
