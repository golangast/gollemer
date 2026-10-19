package chat

import (
	"log"
	"strings"

	"github.com/golangast/gollemer/internal/ai/analyze"
	"github.com/golangast/gollemer/internal/ai/neural/nnu/seq2seq"
)

// Neural explanation over analyzer facts — the fallback for code-intent
// questions after the copy experiments proved the model cannot learn
// precise verbatim copying from the code block.
//
// The division of labor is strict:
//   - the ANALYZER supplies every exact fact (signature, file:line,
//     one-line summary) — deterministically, always correct;
//   - the MODEL supplies only fluent prose — it never emits a fact.
//
// The model cannot hallucinate a file path because its Sig/file:line
// lines are discarded and replaced with the analyzer's versions.

// codeIntentExplainer is the v3 codeintent model, loaded lazily. Nil when
// the checkpoint is absent; the deterministic templates remain the answer.
var codeIntentExplainer *seq2seq.Seq2Seq
var codeIntentExplainerTried bool

func loadCodeIntentExplainer(projectRoot string) *seq2seq.Seq2Seq {
	if codeIntentExplainerTried {
		return codeIntentExplainer
	}
	codeIntentExplainerTried = true
	m, err := loadRealModel(projectRoot, CodeIntentDomain)
	if err != nil {
		log.Printf("[EXPLAIN] no codeintent checkpoint (%v); deterministic answers only", err)
		return nil
	}
	codeIntentExplainer = m
	log.Printf("[EXPLAIN] loaded codeintent explainer model")
	return m
}

// explainWithCodeIntent answers an explanation-seeking question by composing
// analyzer facts with neural prose. It returns ok=false when the symbol
// doesn't resolve, the model is unavailable, or the model produces nothing
// usable — callers fall back to the deterministic template.
func explainWithCodeIntent(p *analyze.Project, name, question, projectRoot string) (string, bool) {
	facts, ok := p.FactsForExplain(name, question)
	if !ok {
		return "", false
	}
	model := loadCodeIntentExplainer(projectRoot)
	if model == nil {
		return "", false
	}
	raw, err := model.Predict(facts.Input, 100)
	if err != nil || strings.TrimSpace(raw) == "" {
		return "", false
	}
	prose := extractProse(raw)
	if !prosePassesGate(prose, facts.Summary) {
		return "", false
	}
	var b strings.Builder
	b.WriteString(facts.Sig + "\n")
	b.WriteString("  " + facts.FileLine + "\n")
	b.WriteString("  " + facts.Summary + "\n")
	b.WriteString("\nIn plain words: " + prose + "\n")
	return b.String(), true
}

// prosePassesGate rejects model prose that is clearly garbled: too short,
// dominated by UNK tokens, unclosed parens, or missing terminal punctuation.
// The analyzer header already carries the facts, so the bar for the bonus
// prose is coherence, not perfection. v3 rarely clears this bar — when it
// doesn't, the deterministic template remains the answer.
func prosePassesGate(prose, summary string) bool {
	prose = strings.TrimSpace(prose)
	if len(prose) < 20 || len(prose) > 400 {
		return false
	}
	words := strings.Fields(prose)
	if len(words) == 0 {
		return false
	}
	for _, w := range words {
		if w == "UNK" {
			return false
		}
	}
	// No unclosed parens (a tell of code-fragment regurgitation).
	if strings.Count(prose, "(") != strings.Count(prose, ")") {
		return false
	}
	// Must end like a sentence.
	last := prose[len(prose)-1]
	if last != '.' && last != '!' && last != '?' {
		return false
	}
	// Don't parrot the summary back verbatim (case-insensitive).
	if strings.EqualFold(prose, strings.TrimSpace(summary)) {
		return false
	}
	return true
}

// extractProse pulls the description lines from a codeintent model output.
// The trained format is Sig / file:line / description, so the first two
// non-empty lines are dropped (the model cannot copy them reliably; the
// analyzer's versions are used instead) and the rest become the prose.
func extractProse(raw string) string {
	var lines []string
	for _, ln := range strings.Split(strings.TrimSpace(raw), "\n") {
		if ln = strings.TrimSpace(ln); ln != "" {
			lines = append(lines, ln)
		}
	}
	if len(lines) > 2 {
		lines = lines[2:]
	}
	prose := strings.Join(lines, " ")
	if len(prose) > 600 {
		prose = prose[:600]
	}
	return strings.TrimSpace(prose)
}
