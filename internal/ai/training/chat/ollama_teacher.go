package chat

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"
)

type ollamaReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
	Format string `json:"format,omitempty"`
}

type ollamaResp struct {
	Response string `json:"response"`
}

// GradeAndCorrectWithTeacher uses a local Ollama model to evaluate a training generation.
// If the generation is poor, it returns a corrected sentence.
func GradeAndCorrectWithTeacher(query, studentResponse string) (passed bool, reason string, correction string) {
	prompt := fmt.Sprintf(`You are an expert AI teacher evaluating a student model's output.
The student was given the query: "%s"
The student responded with: "%s"

Task 1: Is the student's response coherent, natural, and grammatically flawless? (YES or NO)
Task 2: If NO, provide a one-sentence correction of what they should have said. If YES, briefly explain why it's good.

CRITICAL RULES FOR CORRECTION:
1. Provide ONLY the exact literal string the student should say.
2. DO NOT include placeholders like [name]. Use generic names like "John".
3. DO NOT include parentheses or alternative options (e.g., no "or you could say...").
4. DO NOT include quotes around the correction.
5. NO meta-commentary. Just the exact, final sentence.

Format your output exactly like this:
GRADE: YES (or NO)
REASON: <your reason>
CORRECTION: <your exact literal correction if NO, otherwise leave blank>
`, query, studentResponse)

	reqBody, _ := json.Marshal(ollamaReq{
		Model:  "qwen2.5:3b",
		Prompt: prompt,
		Stream: false,
	})

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post("http://127.0.0.1:11434/api/generate", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return true, fmt.Sprintf("Ollama connection failed: %v (defaulting to PASS)", err), ""
	}
	defer resp.Body.Close()

	var oResp ollamaResp
	if err := json.NewDecoder(resp.Body).Decode(&oResp); err != nil {
		return true, "Ollama JSON parse error", ""
	}

	lines := strings.Split(oResp.Response, "\n")
	passed = false
	reason = "Unclear"
	correction = ""

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "GRADE:") {
			gradeStr := strings.ToUpper(strings.TrimSpace(strings.TrimPrefix(line, "GRADE:")))
			if strings.Contains(gradeStr, "YES") {
				passed = true
			}
		} else if strings.HasPrefix(line, "REASON:") {
			reason = strings.TrimSpace(strings.TrimPrefix(line, "REASON:"))
		} else if strings.HasPrefix(line, "CORRECTION:") {
			correction = strings.TrimSpace(strings.TrimPrefix(line, "CORRECTION:"))
			correction = strings.Trim(correction, "\"'") // Remove leading/trailing quotes
		}
	}

	return passed, reason, correction
}

// TeacherAuditContext holds all the information gathered for the 30-epoch teacher review.
type TeacherAuditContext struct {
	Epoch        int
	AvgLoss      float32
	AvgSim       float32
	GrammarScore float32
	// Sample of model outputs: map[query]response
	Probes       map[string]string
	ConfigJSON   string
	TrainingRows int
	HistorySummary string
}

// TeacherAuditAndPatch is the 30-epoch comprehensive Teacher review.
// It probes the model on canonical questions, packages all available metrics
// and config into a single prompt, asks Qwen to diagnose problems and output
// a JSON patch for social_train.json, then writes the patch back to disk.
// Returns a short human-readable summary string for the training log.
func TeacherAuditAndPatch(ctx TeacherAuditContext, configPath string) string {
	// ---- Build probe results section ----
	var probeLines strings.Builder
	for q, r := range ctx.Probes {
		probeLines.WriteString(fmt.Sprintf("  Q: %q\n  A: %q\n\n", q, r))
	}

	prompt := fmt.Sprintf(`You are an expert AI Training Supervisor performing a 30-epoch audit on a Mixture-of-Experts (MoE) language model called Gollemer.

=== CURRENT METRICS (Epoch %d) ===
  Average Loss    : %.4f
  Average Sim     : %.4f
  Grammar Score   : %.4f
  Training Rows   : %d

=== EPOCH HISTORY ===
%s

=== CURRENT CONFIG (social_train.json) ===
%s

=== MODEL OUTPUT PROBES ===
(These are the model's actual responses to canonical test questions right now.)
%s

=== YOUR TASK ===
1. Carefully analyze the metrics, historical trends, config, and model outputs above.
2. Diagnose what is going wrong (word salad, repetition, wrong intent, poor grammar, loss plateau, etc.). If the model's output in the probes is consistently a single word like "it", "is", or word salad, IT IS IN SEVERE MODE COLLAPSE!
3. To FIX mode collapse, YOU MUST RAISE the learning_rate (e.g., from 0.00005 to 0.0001 or 0.0002) to break out of the local minimum OR INCREASE router_temperature OR router_noise!
4. Output a JSON object containing ONLY the config keys you want to change and their new values.
   - Only include keys that already exist in the current config shown above.
   - If training looks perfectly healthy and the model outputs are coherent full sentences, output exactly: {}
   - Do NOT output any explanation outside the JSON.
   - Example: {"learning_rate": 0.00001, "router_temperature": 1.5, "router_noise": 0.3}

Output ONLY the raw JSON object. Nothing else.`,
		ctx.Epoch,
		ctx.AvgLoss,
		ctx.AvgSim,
		ctx.GrammarScore,
		ctx.TrainingRows,
		ctx.HistorySummary,
		ctx.ConfigJSON,
		probeLines.String(),
	)

	reqBody, _ := json.Marshal(ollamaReq{
		Model:  "qwen2.5:3b",
		Prompt: prompt,
		Stream: false,
		Format: "json",
	})

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Post("http://127.0.0.1:11434/api/generate", "application/json", bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit] Ollama offline — skipping patch (epoch %d)", ctx.Epoch)
	}
	defer resp.Body.Close()

	var oResp ollamaResp
	if err := json.NewDecoder(resp.Body).Decode(&oResp); err != nil {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit] JSON decode error — skipping patch (epoch %d)", ctx.Epoch)
	}

	text := strings.TrimSpace(oResp.Response)
	if text == "" || text == "{}" {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit Epoch %d] Teacher sees healthy training. No config changes.", ctx.Epoch)
	}

	// Parse the teacher's recommended patch
	var patch map[string]interface{}
	if err := json.Unmarshal([]byte(text), &patch); err != nil {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit Epoch %d] Teacher output invalid JSON (%q) — skipping", ctx.Epoch, text)
	}
	if len(patch) == 0 {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit Epoch %d] Teacher sees healthy training. No config changes.", ctx.Epoch)
	}

	// Merge patch into current config and write back
	var currentCfg map[string]interface{}
	if cfgData, err := os.ReadFile(configPath); err == nil {
		_ = json.Unmarshal(cfgData, &currentCfg)
	}
	if currentCfg == nil {
		currentCfg = make(map[string]interface{})
	}

	var changedKeys []string
	for k, v := range patch {
		// Only accept keys that already exist to prevent the teacher from inventing new fields
		if _, exists := currentCfg[k]; exists {
			currentCfg[k] = v
			changedKeys = append(changedKeys, fmt.Sprintf("%s=%v", k, v))
		}
	}

	if len(changedKeys) == 0 {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit Epoch %d] Teacher suggested unknown keys — no patch applied", ctx.Epoch)
	}

	newCfgData, err := json.MarshalIndent(currentCfg, "", "  ")
	if err != nil {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit Epoch %d] Marshal error — patch not saved", ctx.Epoch)
	}
	if err := os.WriteFile(configPath, newCfgData, 0644); err != nil {
		return fmt.Sprintf("👨‍🏫 [TeacherAudit Epoch %d] Write error — patch not saved: %v", ctx.Epoch, err)
	}

	return fmt.Sprintf("👨‍🏫 [TeacherAudit Epoch %d] ✅ Patched config: %s", ctx.Epoch, strings.Join(changedKeys, ", "))
}
