package chat

// Pure-neural chat: the antidote to the old -seq2seq-chat.
//
// The old chat loop loaded the saved model but never actually called it — it
// answered from exact-string / fuzzy Jaccard matching with a canned fallback,
// so no amount of training could ever change its behavior. RunRealChat does
// the opposite: every answer comes from Seq2Seq.Predict, nothing else. If the
// model generates word salad, you see the word salad — which is the point:
// this loop is the honest meter for whether training worked.

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
	mainvocab "github.com/golangast/gollemer/internal/ai/neural/nnu/vocab"
	"github.com/golangast/gollemer/internal/ai/neural/tokenizer"
)

// RealModelPath returns the per-domain model file.
func RealModelPath(projectRoot, domain string) string {
	return filepath.Join(projectRoot, "data", "models", "gob_models", "real_tiny_seq2seq_"+domain+".gob")
}

// RunRealChat starts an interactive chat loop backed purely by the trained
// neural model. No lookup tables, no fuzzy matching, no canned fallback.
func RunRealChat(projectRoot, domain string) error {
	modelPath := RealModelPath(projectRoot, domain)

	// seq2seq.Load needs a tokenizer argument, but the real vocabulary lives
	// inside the saved file — so load with a placeholder, then rebuild the
	// tokenizer from the model's own vocabulary.
	placeholder, err := tokenizer.NewTokenizer(mainvocab.NewVocabulary())
	if err != nil {
		return err
	}
	model, err := seq2seq.Load(modelPath, placeholder)
	if err != nil {
		return fmt.Errorf("load real model %s: %w (train it first with -train-real-seq2seq)", modelPath, err)
	}
	if model.OutputVocab == nil || model.OutputVocab.Size() == 0 {
		return fmt.Errorf("saved model has no vocabulary")
	}
	tok, err := tokenizer.NewTokenizer(model.OutputVocab)
	if err != nil {
		return err
	}
	model.Tokenizer = tok

	// A model carrying an ExactMap is a lookup table wearing a neural net's
	// clothes. Refuse it loudly rather than pretend it generates.
	if len(model.ExactMap) > 0 {
		return fmt.Errorf("refusing to chat: model carries an ExactMap of %d canned answers; retrain with -train-real-seq2seq", len(model.ExactMap))
	}
	// Models trained before the MoE decoder have no router/experts; the
	// decoder would nil-panic. Retrain rather than crash.
	if model.Decoder == nil || model.Decoder.MoE == nil {
		return fmt.Errorf("refusing to chat: model was trained without the MoE decoder; retrain with -train-real-seq2seq")
	}

	log.Printf("[REAL-CHAT] loaded %s (vocab=%d, hidden=%d). Pure neural generation — type /quit to exit.",
		modelPath, model.OutputVocab.Size(), model.HiddenDim)

	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	showThoughts := true
	for {
		fmt.Print("you> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "/quit" {
			break
		}
		if line == "/thoughts" {
			showThoughts = !showThoughts
			fmt.Printf("[thought process display %s]\n", map[bool]string{true: "on", false: "off"}[showThoughts])
			continue
		}
		// Lowercased to match training; the model is case-insensitive by construction.
		answer, trace, err := model.PredictWithTrace(strings.ToLower(line), 40)
		if err != nil {
			fmt.Printf("gollemer> [error: %v]\n", err)
			continue
		}
		fmt.Printf("gollemer> %s\n", tidyDecode(answer))
		if showThoughts {
			printThoughtTrace(trace)
		}
	}
	return sc.Err()
}

// printThoughtTrace renders the model's per-token thought process: which MoE
// experts it consulted for each generated token, the runners-up it
// considered, and the overall expert mix for the reply.
func printThoughtTrace(trace *seq2seq.ThoughtTrace) {
	if trace == nil || len(trace.Steps) == 0 {
		return
	}
	fmt.Printf("  💭 thought process (%d tokens, top-%d MoE routing):\n", len(trace.Steps), len(trace.Steps[0].Experts))
	for _, s := range trace.Steps {
		parts := make([]string, 0, len(s.Experts))
		for i, e := range s.Experts {
			parts = append(parts, fmt.Sprintf("E%d:%.2f", e, s.Gates[i]))
		}
		alt := ""
		if len(s.Candidates) > 1 {
			alt = fmt.Sprintf("  (also considered: %s)", strings.Join(s.Candidates[1:], ", "))
		}
		fmt.Printf("    %-12q → [%s]%s\n", s.Token, strings.Join(parts, " "), alt)
	}
	if len(trace.ExpertUsage) > 0 {
		parts := make([]string, 0, len(trace.ExpertUsage))
		for e, u := range trace.ExpertUsage {
			bars := int(u*20 + 0.5)
			parts = append(parts, fmt.Sprintf("E%d %s %3.0f%%", e, strings.Repeat("█", bars), u*100))
		}
		fmt.Printf("    expert mix: %s\n", strings.Join(parts, "  "))
	}
}
