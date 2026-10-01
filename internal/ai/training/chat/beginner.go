package chat

import (
	"regexp"
	"strings"

	"github.com/golangast/gollemer/pkg/beginner"
)

// beginnerExplainRe catches "explain <concept>" requests, with an
// optional beginner marker up front ("beginner: explain channels").
var beginnerExplainRe = regexp.MustCompile(`(?i)^\s*(?:beginner\s*[:,]?\s*)?explain\s+([a-z][a-z\s]*?)\s*$`)

// handleBeginner answers explicit beginner-brain requests
// deterministically: concept explanations with analogies, or small
// generated programs with plain-English rationales. It always answers
// (never falls through to a neural model) because the router only
// sends it explicit beginner-marked requests.
func handleBeginner(line string) (string, bool) {
	if m := beginnerExplainRe.FindStringSubmatch(line); m != nil {
		if c, err := beginner.ExplainGoConcept(m[1]); err == nil {
			var b strings.Builder
			b.WriteString("**" + c.Concept + "** — " + c.Analogy + "\n")
			b.WriteString("```go\n" + c.Snippet + "```\n")
			b.WriteString("Key rules:\n")
			for _, r := range c.KeyRules {
				b.WriteString("- " + r + "\n")
			}
			return b.String(), true
		}
		// Not a known concept ("explain how to build an http server"):
		// fall through to code generation.
	}
	code, expl, err := beginner.GenerateGoFromCommand(line)
	if err != nil {
		return "Beginner brain couldn't build that. " + err.Error(), true
	}
	return "```go\n" + code + "```\n**Why this code:** " + expl, true
}
