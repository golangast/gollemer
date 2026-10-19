// Package chat — action predictor (Layer 4): a tiny neural model that predicts
// John's next action from context. Trained on ~/.gollemer_actions.jsonl pairs.
// This is (context -> next_action) translation, NOT code generation.
// The model never touches code — it only learns John's habits.
package chat

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// PredictorConfig holds the predictor's settings.
type PredictorConfig struct {
	// MinPairs: don't train until we have this many logged actions.
	MinPairs int
	// ModelPath: where the trained predictor lives.
	ModelPath string
}

// DefaultPredictorConfig returns sensible defaults.
func DefaultPredictorConfig() PredictorConfig {
	home, _ := os.UserHomeDir()
	return PredictorConfig{
		MinPairs:  50, // need 50+ actions before training is worthwhile
		ModelPath: filepath.Join(home, ".gollemer_predictor.gob"),
	}
}

// ExportTrainingPairs converts the action log to seq2seq training pairs.
// Format: "context: <ctx>" -> "<action>"
// Writes to the given path in the chat_pairs.jsonl format.
func ExportTrainingPairs(outputPath string) (int, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return 0, err
	}
	logPath := filepath.Join(home, ".gollemer_actions.jsonl")
	f, err := os.Open(logPath)
	if err != nil {
		return 0, fmt.Errorf("no action log yet (need to use chat first): %w", err)
	}
	defer f.Close()

	out, err := os.Create(outputPath)
	if err != nil {
		return 0, err
	}
	defer out.Close()

	count := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var entry ActionLog
		if err := json.Unmarshal(scanner.Bytes(), &entry); err != nil {
			continue
		}
		// Input: context. Output: action.
		// Keep it simple: time-of-day + cwd basename + context -> action.
		hour := entry.Timestamp.Hour()
		timeSlot := "morning"
		if hour >= 12 && hour < 17 {
			timeSlot = "afternoon"
		} else if hour >= 17 && hour < 22 {
			timeSlot = "evening"
		} else if hour >= 22 || hour < 6 {
			timeSlot = "night"
		}
		cwdBase := filepath.Base(entry.Cwd)
		input := fmt.Sprintf("%s in %s after %s", timeSlot, cwdBase, entry.Context)
		// Write as JSONL pair.
		pair := map[string]string{"input": input, "output": entry.Action}
		data, _ := json.Marshal(pair)
		out.Write(append(data, '\n'))
		count++
	}
	return count, scanner.Err()
}

// PredictorStatus reports whether the predictor can be trained/used.
func PredictorStatus() string {
	var b strings.Builder
	cfg := DefaultPredictorConfig()

	// Count logged actions.
	home, _ := os.UserHomeDir()
	logPath := filepath.Join(home, ".gollemer_actions.jsonl")
	count := 0
	if data, err := os.ReadFile(logPath); err == nil {
		count = strings.Count(string(data), "\n")
	}

	b.WriteString(fmt.Sprintf("Logged actions: %d (need %d to train)\n", count, cfg.MinPairs))

	// Check if model exists.
	if _, err := os.Stat(cfg.ModelPath); err == nil {
		b.WriteString("Predictor model: trained ✓\n")
	} else {
		b.WriteString("Predictor model: not trained yet\n")
		if count >= cfg.MinPairs {
			b.WriteString("Ready to train! Run: /train-predictor\n")
		} else {
			b.WriteString(fmt.Sprintf("Keep using chat — %d more actions needed.\n", cfg.MinPairs-count))
		}
	}
	return b.String()
}

// BuildContextString creates the predictor input from current state.
func BuildContextString(cwd, lastAction string) string {
	hour := time.Now().Hour()
	timeSlot := "morning"
	if hour >= 12 && hour < 17 {
		timeSlot = "afternoon"
	} else if hour >= 17 && hour < 22 {
		timeSlot = "evening"
	} else if hour >= 22 || hour < 6 {
		timeSlot = "night"
	}
	cwdBase := filepath.Base(cwd)
	if lastAction == "" {
		return fmt.Sprintf("%s in %s starting fresh", timeSlot, cwdBase)
	}
	return fmt.Sprintf("%s in %s after %s", timeSlot, cwdBase, normalizeAction(lastAction))
}
