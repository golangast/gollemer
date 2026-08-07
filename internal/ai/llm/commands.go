package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
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

	// ── Fix / edit ───────────────────────────────────────────────────────────
	case "fix_query":
		fileName := ""
		if fn, ok := intentData.Parameters["name"].(string); ok {
			fileName = fn
		}
		if fileName == "" {
			// Try to extract filename from query
			parts := strings.Fields(query)
			for _, p := range parts {
				if strings.Contains(p, ".go") || strings.Contains(p, ".js") || strings.Contains(p, ".ts") {
					fileName = p
					break
				}
			}
		}
		if fileName == "" {
			// Check if "jim" or "jim.go" or any existing file appears in query
			lowerQ := strings.ToLower(query)
			for _, p := range strings.Fields(lowerQ) {
				p = strings.Trim(p, "',\".")
				if p == "" {
					continue
				}
				if strings.Contains(p, "jim") || strings.HasSuffix(p, ".go") {
					fileName = p
					if !strings.HasSuffix(fileName, ".go") {
						fileName += ".go"
					}
					break
				}
				cand := p + ".go"
				if _, err := os.Stat(cand); err == nil {
					fileName = cand
					break
				}
			}
		}
		// If file not found at the given path, search subdirectories
		if fileName != "" {
			if _, err := os.Stat(fileName); os.IsNotExist(err) {
				// Search for the file in subdirectories
				var foundPath string
				filepath.Walk(".", func(path string, info os.FileInfo, walkErr error) error {
					if walkErr == nil && !info.IsDir() && info.Name() == fileName {
						foundPath = path
						return filepath.SkipDir
					}
					return nil
				})
				if foundPath != "" {
					fileName = foundPath
				}
			}
		}
		if fileName != "" {
			// ── Corpus-Driven Semantic Routing ──────────────────────────────────
			// Route to edit_agent or auto_fix by comparing the user's query
			// embedding against the pre-trained intent corpus embeddings.
			queryEmb := r.Client.getSentenceEmbedding(query)
			bestIntent := ""
			bestScore := -1.0

			if queryEmb != nil && r.SemanticRouter != nil {
				for _, entry := range r.SemanticRouter.Embeddings {
					score := cosineSimilarity(queryEmb, entry.Embedding)
					if score > bestScore {
						bestScore = score
						bestIntent = entry.Intent
					}
				}
			}

			// Fallback: keyword triggers that are unambiguous
			lq := strings.ToLower(query)
			fallbackEdit := strings.Contains(lq, "function") || strings.Contains(lq, "func") ||
				strings.Contains(lq, "import ") || strings.Contains(lq, "change ") ||
				strings.Contains(lq, "struct") || strings.Contains(lq, "add ") ||
				strings.Contains(lq, "create ") || strings.Contains(lq, "insert") ||
				strings.Contains(lq, "field") || strings.Contains(lq, "type ") ||
				strings.Contains(lq, "remove") || strings.Contains(lq, "delete")

			useEditAgent := (bestIntent == "edit_agent" && bestScore > 0.60) || fallbackEdit
			if useEditAgent {
				predictedSentence = r.handleLLMEditCommand(fileName, query)
			} else {
				predictedSentence = r.handleFixCommand(fileName)
			}
		} else {
			predictedSentence = "I couldn't determine which file to fix."
		}

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

// EditOperation describes a single AST-level edit to apply to a Go source file.
// Mirrors the type in cmd/tools/go_edit_agent/main.go for JSON serialization.
type EditOperation struct {
	Type       string `json:"type"`
	TargetFile string `json:"target_file"`
	FuncName   string `json:"func_name,omitempty"`
	StructName string `json:"struct_name,omitempty"`
	FieldName  string `json:"field_name,omitempty"`
	FieldType  string `json:"field_type,omitempty"`
	FieldTag   string `json:"field_tag,omitempty"`
	ImportPath string `json:"import_path,omitempty"`
	Code       string `json:"code,omitempty"`
	InsertAt   string `json:"insert_at,omitempty"`
	OldCode    string `json:"old_code,omitempty"`
	NewCode    string `json:"new_code,omitempty"`
}

// AgentResponse mirrors the response type from cmd/tools/go_edit_agent/main.go.
type AgentResponse struct {
	Success      bool   `json:"success"`
	File         string `json:"file"`
	EditsApplied int    `json:"edits_applied"`
	Error        string `json:"error,omitempty"`
	Duration     string `json:"duration"`
}

func (r *Runner) handleFixCommand(fileName string) string {
	// Read the file to understand its current state
	content, readErr := os.ReadFile(fileName)
	if readErr != nil {
		return fmt.Sprintf("⚠️ Could not read %s: %v", fileName, readErr)
	}
	lines := strings.Split(string(content), "\n")
	lineCount := len(lines)

	// Describe what we found
	var desc strings.Builder
	desc.WriteString(fmt.Sprintf("📄 I read %s (%d lines). ", fileName, lineCount))

	// Check for common issues
	hasMissingBrace := false
	hasMissingStructKeyword := false
	openBraces := 0
	closeBraces := 0
	funcsWithoutBrace := 0
	structsWithoutBrace := 0
	structsWithoutKeyword := 0
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		openBraces += strings.Count(line, "{")
		closeBraces += strings.Count(line, "}")
		// Detect: func declaration with params but no opening brace
		if strings.HasPrefix(trimmed, "func ") && strings.Contains(trimmed, ")") && !strings.Contains(trimmed, "{") {
			funcsWithoutBrace++
		}
		// Detect: type X struct declaration without opening brace
		if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, " struct") && !strings.Contains(trimmed, "{") {
			structsWithoutBrace++
		}
		// Detect: type X { declaration missing the 'struct' keyword (e.g. "type jill  {")
		if strings.HasPrefix(trimmed, "type ") && strings.Contains(trimmed, "{") &&
			!strings.Contains(trimmed, " struct ") && !strings.HasPrefix(trimmed, "type struct") {
			structsWithoutKeyword++
		}
	}
	if funcsWithoutBrace > 0 {
		hasMissingBrace = true
		desc.WriteString(fmt.Sprintf("🔍 Found %d function declaration(s) with params but missing '{'. ", funcsWithoutBrace))
	}
	if structsWithoutBrace > 0 {
		hasMissingBrace = true
		desc.WriteString(fmt.Sprintf("🔍 Found %d struct declaration(s) missing '{'. ", structsWithoutBrace))
	}
	if structsWithoutKeyword > 0 {
		hasMissingStructKeyword = true
		desc.WriteString(fmt.Sprintf("🔍 Found %d type declaration(s) with '{' but missing the 'struct' keyword (e.g. 'type jill  {'). ", structsWithoutKeyword))
	}
	if hasMissingStructKeyword {
		hasMissingBrace = true
	}
	if openBraces > closeBraces {
		hasMissingBrace = true
		desc.WriteString(fmt.Sprintf("🔍 Found %d opening braces but only %d closing braces — missing %d '}'. ", openBraces, closeBraces, openBraces-closeBraces))
	} else if closeBraces > openBraces {
		hasMissingBrace = true
		desc.WriteString(fmt.Sprintf("🔍 Found %d closing braces but only %d opening braces — missing %d '{'. ", closeBraces, openBraces, closeBraces-openBraces))
	}

	// Try parsing to detect syntax errors
	_, parseErr := parser.ParseFile(token.NewFileSet(), fileName, content, parser.ParseComments)
	if parseErr != nil {
		desc.WriteString(fmt.Sprintf("🔍 Parse error detected: %s. ", parseErr.Error()))
	}

	// Run go vet to detect semantic issues (e.g., package main without main())
	vetOut, _ := exec.Command("go", "vet", fileName).CombinedOutput()
	vetStr := strings.TrimSpace(string(vetOut))
	if vetStr != "" {
		desc.WriteString(fmt.Sprintf("🔍 Go vet found issues: %s. ", vetStr))
	}

	// Apply the fix
	fixer := NewGoFixer(fileName)
	err := fixer.Fix()
	if err == nil {
		// Read the file again to see what changed
		newContent, _ := os.ReadFile(fileName)
		newLines := strings.Split(string(newContent), "\n")
		if len(newLines) != lineCount || string(newContent) != string(content) {
			desc.WriteString("✅ Applied fix: ")
			if structsWithoutKeyword > 0 {
				desc.WriteString("added the missing 'struct' keyword to type declaration(s) (e.g. 'type jill  {' → 'type jill struct {').")
			} else if funcsWithoutBrace > 0 && structsWithoutBrace > 0 {
				desc.WriteString("added missing '{' after function and struct declaration(s).")
			} else if funcsWithoutBrace > 0 {
				desc.WriteString("added missing '{' after function declaration(s).")
			} else if structsWithoutBrace > 0 {
				desc.WriteString("added missing '{' after struct declaration(s).")
			} else if hasMissingBrace {
				desc.WriteString("added missing braces to balance the file.")
			} else {
				desc.WriteString("corrected syntax errors.")
			}
		} else if vetStr != "" {
			desc.WriteString("⚠️ File is syntactically valid but has semantic issues (go vet found problems). ")

			// Use the Hybrid Engine via the fix CLI with the correct package directory.
			absFile, absErr := filepath.Abs(fileName)
			if absErr != nil {
				absFile = fileName
			}
			fileDir := filepath.Dir(absFile)
			relDir, relErr := filepath.Rel(r.ProjectRoot, fileDir)
			if relErr != nil || relDir == "" {
				relDir = "."
			}
			cmd := exec.Command("go", "run", "./cmd/tools/fix/main.go", "-auto-apply", "-verbose", "./"+relDir)
			cmd.Dir = r.ProjectRoot
			output, fixErr := cmd.CombinedOutput()
			fixOutput := strings.TrimSpace(string(output))
			if fixErr == nil && strings.Contains(fixOutput, "✅") {
				desc.WriteString("🔄 Hybrid engine applied fixes:\n")
				for _, line := range strings.Split(fixOutput, "\n") {
					if strings.Contains(line, "✅") || strings.Contains(line, "❌") {
						desc.WriteString("  " + line + "\n")
					}
				}
			} else {
				desc.WriteString("🔄 Hybrid engine could not fix the semantic errors automatically.")
				if fixOutput != "" {
					desc.WriteString("\n  Details: " + fixOutput)
				}
			}
		} else {
			desc.WriteString("✅ File was already valid — no changes needed.")
		}
		return desc.String()
	}

	// GoFixer failed - try the go_edit_agent with a repair query
	desc.WriteString(fmt.Sprintf("⚠️ GoFixer could not auto-fix: %v. ", err))

	// Try go_edit_agent with a repair query
	editResp := r.handleLLMEditCommand(fileName, "fix syntax errors in "+fileName)
	if strings.Contains(editResp, "Successfully edited") {
		desc.WriteString("🔄 go_edit_agent attempted repairs. ")
		// Check if it's fixed now
		newContent, _ := os.ReadFile(fileName)
		_, newParseErr := parser.ParseFile(token.NewFileSet(), fileName, newContent, parser.ParseComments)
		if newParseErr == nil {
			desc.WriteString("✅ File now parses correctly!")
		} else {
			desc.WriteString(fmt.Sprintf("⚠️ Still has issues: %s", newParseErr.Error()))
		}
	} else {
		desc.WriteString("🔄 go_edit_agent also could not fix it automatically.")
	}

	return desc.String()
}

// handleLLMEditCommand uses the go_edit_agent binary to apply an intelligent edit
// to a Go file based on a natural language description.
// The agent reads the file's AST for context and parses the query itself.
func (r *Runner) handleLLMEditCommand(fileName, query string) string {
	// Locate the go_edit_agent binary. It is built at the project root, but the
	// runner may be invoked from a subdirectory, so `./go_edit_agent` would fail.
	agentBin := filepath.Join(r.ProjectRoot, "go_edit_agent")
	if _, err := os.Stat(agentBin); err != nil {
		return fmt.Sprintf("⚠️ go_edit_agent binary not found at %s (build it with: go build -o go_edit_agent ./cmd/tools/go_edit_agent)", agentBin)
	}

	// Call the go_edit_agent binary with the raw natural language query
	// The agent handles all parsing internally using its AST-aware parser.
	// NOTE: we use Output() (stdout only) because the agent's JSON response goes
	// to stdout while its log lines go to stderr. Combining them corrupts the JSON.
	var stderrBuf bytes.Buffer
	cmd := exec.Command(agentBin,
		"-file", fileName,
		"-query", query,
		"-retries", "2",
	)
	cmd.Stderr = &stderrBuf
	output, err := cmd.Output()
	if err != nil {
		return fmt.Sprintf("⚠️ Edit agent failed for %s: %v\n%s", fileName, err, strings.TrimSpace(stderrBuf.String()))
	}

	// Parse the response
	var resp AgentResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		return fmt.Sprintf("⚠️ Could not parse edit agent response: %v (raw output: %s)", err, string(output))
	}

	if resp.Success {
		return fmt.Sprintf("✅ Successfully edited %s (%d edits applied)", fileName, resp.EditsApplied)
	}

	return fmt.Sprintf("⚠️ Edit failed for %s: %s", fileName, resp.Error)
}

// parseEditFromQuery converts a natural language edit request into an EditOperation.
// It uses simple heuristics to detect the type of edit requested.
func parseEditFromQuery(query, fileName string) *EditOperation {
	lower := strings.ToLower(strings.TrimSpace(query))

	// Strip the filename reference from the query to avoid confusion
	// e.g. "add function jim to ft/jim.go" -> "add function jim"
	baseName := strings.TrimSuffix(filepath.Base(fileName), ".go")
	lower = strings.ReplaceAll(lower, " to "+baseName+".go", "")
	lower = strings.ReplaceAll(lower, " to file "+baseName+".go", "")
	lower = strings.ReplaceAll(lower, " to "+baseName, "")
	lower = strings.ReplaceAll(lower, " in "+baseName+".go", "")
	lower = strings.ReplaceAll(lower, " in file "+baseName+".go", "")
	lower = strings.ReplaceAll(lower, " in "+baseName, "")
	lower = strings.ReplaceAll(lower, " to file ", "")
	lower = strings.ReplaceAll(lower, " in file ", "")
	lower = strings.TrimSpace(lower)

	// Detect: "add a function called X" or "add function X" or "add a new function X"
	if strings.Contains(lower, "add") && strings.Contains(lower, "function") ||
		strings.Contains(lower, "add") && strings.Contains(lower, "func") ||
		strings.Contains(lower, "new function") ||
		strings.Contains(lower, "insert function") {

		// Extract function name
		funcName := extractFuncName(lower)
		if funcName == "" {
			return nil
		}

		// Build function code from the description
		code := buildFuncCodeFromQuery(lower, funcName)

		return &EditOperation{
			Type:       "insert_func",
			TargetFile: fileName,
			FuncName:   funcName,
			Code:       code,
		}
	}

	// Detect: "modify function X" or "update function X"
	if (strings.Contains(lower, "modify") || strings.Contains(lower, "update") || strings.Contains(lower, "change")) &&
		strings.Contains(lower, "function") {
		funcName := extractFuncName(lower)
		if funcName == "" {
			return nil
		}
		return &EditOperation{
			Type:       "modify_func",
			TargetFile: fileName,
			FuncName:   funcName,
			Code:       buildFuncCodeFromQuery(lower, funcName),
		}
	}

	// Detect: "delete function X" or "remove function X"
	if (strings.Contains(lower, "delete") || strings.Contains(lower, "remove")) &&
		strings.Contains(lower, "function") {
		funcName := extractFuncName(lower)
		if funcName == "" {
			return nil
		}
		return &EditOperation{
			Type:       "delete_func",
			TargetFile: fileName,
			FuncName:   funcName,
		}
	}

	// Detect: "add import X"
	if strings.Contains(lower, "add import") || strings.Contains(lower, "import ") {
		importPath := extractImportPath(lower)
		if importPath != "" {
			return &EditOperation{
				Type:       "add_import",
				TargetFile: fileName,
				ImportPath: importPath,
			}
		}
	}

	// Detect: "add field X to struct Y"
	if strings.Contains(lower, "add field") && strings.Contains(lower, "struct") {
		fieldName := extractFieldName(lower)
		structName := extractStructName(lower)
		fieldType := extractFieldType(lower)
		if fieldName != "" && structName != "" {
			return &EditOperation{
				Type:       "add_field",
				TargetFile: fileName,
				StructName: structName,
				FieldName:  fieldName,
				FieldType:  fieldType,
			}
		}
	}

	return nil
}

// extractFuncName extracts a function name from a natural language query.
func extractFuncName(lower string) string {
	// Pattern: "function called X" or "function named X" or "function X"
	patterns := []struct {
		prefix string
		offset int
	}{
		{"function called ", 0},
		{"function named ", 0},
		{"function '", 0},
		{"function \"", 0},
		{"func called ", 0},
		{"func named ", 0},
		{"func '", 0},
		{"func \"", 0},
		{"add function ", 0},
		{"add func ", 0},
		{"new function ", 0},
		{"insert function ", 0},
	}

	for _, p := range patterns {
		if idx := strings.Index(lower, p.prefix); idx >= 0 {
			start := idx + len(p.prefix)
			remaining := lower[start:]
			// Iterate through words to skip stop words and find the actual name
			words := strings.Fields(remaining)
			for _, w := range words {
				name := strings.Trim(w, "'\",.;:()")
				if name != "" && !isStopWord(name) {
					return name
				}
			}
		}
	}

	// Fallback: look for "called X" or "named X"
	for _, marker := range []string{" called ", " named "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			start := idx + len(marker)
			remaining := lower[start:]
			// Iterate through words to skip stop words and find the actual name
			words := strings.Fields(remaining)
			for _, w := range words {
				name := strings.Trim(w, "'\",.;:()")
				if name != "" && !isStopWord(name) {
					return name
				}
			}
		}
	}

	return ""
}

// extractStructName extracts a struct name from a natural language query.
func extractStructName(lower string) string {
	if idx := strings.Index(lower, "struct "); idx >= 0 {
		remaining := lower[idx+7:]
		words := strings.Fields(remaining)
		if len(words) > 0 {
			return strings.Trim(words[0], "'\",.;:()")
		}
	}
	return ""
}

// extractFieldName extracts a field name from a natural language query.
func extractFieldName(lower string) string {
	if idx := strings.Index(lower, "field "); idx >= 0 {
		remaining := lower[idx+6:]
		words := strings.Fields(remaining)
		if len(words) > 0 {
			return strings.Trim(words[0], "'\",.;:()")
		}
	}
	return ""
}

// extractFieldType extracts a field type from a natural language query.
func extractFieldType(lower string) string {
	// Look for "of type X" or "type X"
	for _, marker := range []string{" of type ", " type "} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			start := idx + len(marker)
			remaining := lower[start:]
			words := strings.Fields(remaining)
			if len(words) > 0 {
				return strings.Trim(words[0], "'\",.;:()")
			}
		}
	}
	return "string"
}

// extractImportPath extracts an import path from a natural language query.
func extractImportPath(lower string) string {
	// Pattern: "import X" or "import \"X\""
	if idx := strings.Index(lower, "import "); idx >= 0 {
		remaining := lower[idx+7:]
		words := strings.Fields(remaining)
		if len(words) > 0 {
			path := strings.Trim(words[0], "'\"")
			if path != "" {
				return path
			}
		}
	}
	return ""
}

// buildFuncCodeFromQuery generates Go function code from a natural language description.
func buildFuncCodeFromQuery(lower, funcName string) string {
	// Detect function signature patterns
	hasParams := strings.Contains(lower, "take") || strings.Contains(lower, "parameter") || strings.Contains(lower, "argument") || strings.Contains(lower, "input")
	hasReturn := strings.Contains(lower, "return") || strings.Contains(lower, "result")

	// Detect parameter types
	hasInt := strings.Contains(lower, "int") || strings.Contains(lower, "integer")
	hasString := strings.Contains(lower, "string") || strings.Contains(lower, "str")
	hasFloat := strings.Contains(lower, "float") || strings.Contains(lower, "float64")

	// Detect return type
	returnsInt := strings.Contains(lower, "sum") || strings.Contains(lower, "total") || strings.Contains(lower, "count") || strings.Contains(lower, "number")
	returnsString := strings.Contains(lower, "concat") || strings.Contains(lower, "join") || strings.Contains(lower, "message")
	returnsBool := strings.Contains(lower, "check") || strings.Contains(lower, "valid") || strings.Contains(lower, "compare")

	// Build the function signature
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("func %s(", funcName))

	if hasParams {
		if hasInt && hasString {
			sb.WriteString("a int, b string")
		} else if hasInt && hasFloat {
			sb.WriteString("a int, b float64")
		} else if hasInt {
			// Check if there are two integers
			if strings.Contains(lower, "two") || strings.Contains(lower, "2") {
				sb.WriteString("a, b int")
			} else {
				sb.WriteString("a int")
			}
		} else if hasString {
			sb.WriteString("s string")
		} else if hasFloat {
			sb.WriteString("f float64")
		} else {
			sb.WriteString("a int")
		}
	}

	sb.WriteString(")")

	if hasReturn {
		if returnsInt {
			sb.WriteString(" int")
		} else if returnsString {
			sb.WriteString(" string")
		} else if returnsBool {
			sb.WriteString(" bool")
		} else if hasInt {
			sb.WriteString(" int")
		} else {
			sb.WriteString(" int")
		}
	}

	sb.WriteString(" {\n")

	// Build the function body
	if strings.Contains(lower, "sum") || strings.Contains(lower, "add") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "concat") || strings.Contains(lower, "join") {
		sb.WriteString("\treturn a + b\n")
	} else if strings.Contains(lower, "multiply") || strings.Contains(lower, "product") {
		sb.WriteString("\treturn a * b\n")
	} else if strings.Contains(lower, "greet") || strings.Contains(lower, "hello") {
		sb.WriteString("\treturn fmt.Sprintf(\"Hello, %s!\", name)\n")
	} else if strings.Contains(lower, "square") {
		sb.WriteString("\treturn a * a\n")
	} else {
		sb.WriteString("\t// TODO: implement\n")
		sb.WriteString("\treturn 0\n")
	}

	sb.WriteString("}\n")

	return sb.String()
}

// isStopWord checks if a word is a common stop word that shouldn't be treated as a name.
func isStopWord(word string) bool {
	stopWords := map[string]bool{
		"that": true, "this": true, "with": true, "from": true, "into": true,
		"file": true, "the": true, "a": true, "an": true, "to": true,
		"in": true, "of": true, "for": true, "and": true, "or": true,
		"it": true, "is": true, "are": true, "was": true, "be": true,
		"has": true, "have": true, "do": true, "does": true, "will": true,
		"would": true, "could": true, "should": true, "may": true, "might": true,
		"can": true, "shall": true, "must": true, "need": true, "let": true,
		"make": true, "take": true, "get": true, "set": true, "put": true,
		"add": true, "new": true, "function": true, "func": true,
		"called": true, "named": true, "returns": true, "return": true,
		"takes": true, "parameters": true, "parameter": true,
		"arguments": true, "argument": true, "input": true, "output": true,
		"two": true, "three": true, "four": true, "five": true,
		"integers": true, "integer": true, "int": true, "string": true,
		"float": true, "bool": true, "boolean": true,
	}
	return stopWords[word]
}
func (r *Runner) handleTemplateCreate(objectType, fileName, targetDirectory, handlerURL string) string {
	msg, _ := handleGenericCreate(objectType, fileName, targetDirectory, handlerURL, r.KB)
	return msg
}
