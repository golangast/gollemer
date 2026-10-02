package chat

import (
	"fmt"
	"strings"

	"github.com/golangast/gollemer/pkg/errhelp"
)

// tryErrorHelp translates a pasted Go compiler error, vet finding, or
// panic into plain words — no command needed, just paste the error.
// Anything the pattern table doesn't recognize falls through to the
// Go brain, so an unfamiliar error still gets an explanation.
func tryErrorHelp(line string) (string, bool) {
	if !errhelp.LooksLikeGoError(line) {
		return "", false
	}
	h, ok := errhelp.Translate(line)
	if !ok {
		return "", false
	}
	var b strings.Builder
	fmt.Fprintf(&b, "🔍 %s\n%s\n\n", strings.ToUpper(h.Title), h.Matched)
	b.WriteString("What it means: ")
	b.WriteString(h.Meaning)
	b.WriteString("\n\nHow to fix: ")
	b.WriteString(h.Fix)
	b.WriteString("\n")
	return b.String(), true
}
