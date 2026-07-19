package moe

import (
	"testing"
)

func TestEvaluateWindow(t *testing.T) {
	rule := IntentRule{}

	// Test missing words in verb phrase
	penalty := rule.EvaluateWindow("AUX", "NOUN", "EOS")
	if penalty < 0.3 {
		t.Errorf("Expected penalty for missing verb in verb phrase, got %f", penalty)
	}

	// Test improper punctuation
	penalty = rule.EvaluateWindow("PRON", "PREP", "EOS")
	if penalty < 0.4 {
		t.Errorf("Expected penalty for ending on PREP, got %f", penalty)
	}

	// Test valid
	penalty = rule.EvaluateWindow("PRON", "VERB", "NOUN")
	if penalty != 0.0 {
		t.Errorf("Expected no penalty for valid sequence, got %f", penalty)
	}
}

func TestDynamicLexicon(t *testing.T) {
	tag := MapWordToGrammarType("hello")
	if tag != "GREET" {
		t.Errorf("Expected GREET for hello, got %s", tag)
	}

	tag = MapWordToGrammarType("running")
	if tag != "VERB" {
		t.Errorf("Expected VERB for running, got %s", tag)
	}
}
