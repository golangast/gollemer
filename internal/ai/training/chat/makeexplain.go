package chat

import (
	"fmt"
	"regexp"
	"strings"
)

// This file makes Makefile targets directly accessible from the chat:
//   run make <target>      -> validates the target, offers [y/n], runs it
//   explain make <target>  -> says what the target does, offers [y/n], runs it
// Both work no matter which brain the router picked, because the input
// shapes are unambiguous and the target always comes from the Makefile
// allowlist — never from model output.

// makeTargetDocs holds the one-line "## target: description" docs parsed
// from the Makefile, filled by initMakeAllowlist alongside the allowlist.
var makeTargetDocs = map[string]string{}

// makeTargetRecipes holds the first recipe lines per target, for the
// "explain" preview.
var makeTargetRecipes = map[string][]string{}

// makeDocLine matches "## eval: score every brain on its fixed eval suite".
var makeDocLine = regexp.MustCompile(`^## ([a-z][a-z0-9_-]*):\s*(.*?)\s*$`)

// explainMakeRe matches "explain make eval", "what does make eval do",
// "what is make eval", "what's make eval".
var explainMakeRe = regexp.MustCompile(`(?i)^\s*(?:explain|what's|what is|what does)\s+make\s+([a-z][a-z0-9_-]*)(?:\s+do)?\s*\??\s*$`)

// directRunMakeRe matches a bare "run make <target>" typed by the user.
var directRunMakeRe = regexp.MustCompile(`(?i)^\s*run make ([a-z][a-z0-9_-]*)\s*$`)

// parseMakeDocs fills makeTargetDocs and makeTargetRecipes from the raw
// Makefile text. "## target: description" lines give the docs (with "#"
// continuation lines joined in); indented lines after "target:" give the
// recipe preview.
func parseMakeDocs(data string) {
	lines := strings.Split(data, "\n")
	for i, line := range lines {
		if m := makeDocLine.FindStringSubmatch(line); m != nil {
			doc := m[2]
			// Join "#"-comment continuation lines ("#   more text").
			for j := i + 1; j < len(lines); j++ {
				c := strings.TrimSpace(lines[j])
				if strings.HasPrefix(c, "#") && !strings.HasPrefix(c, "##") {
					rest := strings.TrimSpace(strings.TrimPrefix(c, "#"))
					if rest == "" {
						break
					}
					doc += " " + rest
				} else {
					break
				}
			}
			makeTargetDocs[m[1]] = doc
		}
		if m := makeTargetLine.FindStringSubmatch(line); m != nil {
			var recipe []string
			for j := i + 1; j < len(lines) && len(recipe) < 6; j++ {
				rl := lines[j]
				if rl == "" {
					break
				}
				if rl[0] == '\t' || rl[0] == ' ' {
					recipe = append(recipe, strings.TrimSpace(rl))
				} else {
					break
				}
			}
			if len(recipe) > 0 {
				makeTargetRecipes[m[1]] = recipe
			}
		}
	}
}

// tryBareMakeTarget handles a bare target name typed on its own
// ("eval", "explain", "smarter"): the chat treats it as
// "run make <target>" — validated against the Makefile allowlist and
// offered immediately. Only a single token ever matches, so normal
// sentences ("help me", "start over") can't trip it.
func tryBareMakeTarget(line string, r *LineReader, projectRoot string, conv *Conversation) bool {
	fields := strings.Fields(line)
	if len(fields) != 1 {
		return false
	}
	target := strings.ToLower(strings.Trim(fields[0], "?!.,;:"))
	if t := runnableMakeTarget("run make " + target); t != "" {
		out := "run make " + t
		fmt.Printf("gollemer [%s]> %s\n", MakefileDomain, out)
		conv.AddReply(out, MakefileDomain, false)
		offerRunMakeCommand(r, projectRoot, t)
		return true
	}
	return false
}

// tryDirectRunMake handles a bare "run make <target>" typed by the user:
// the target is validated against the Makefile allowlist and, when valid,
// the reply is printed and the run is offered — no brain needed.
func tryDirectRunMake(line string, r *LineReader, projectRoot string, conv *Conversation) bool {
	m := directRunMakeRe.FindStringSubmatch(line)
	if m == nil {
		return false
	}
	target := strings.ToLower(m[1])
	if t := runnableMakeTarget("run make " + target); t != "" {
		out := "run make " + t
		fmt.Printf("gollemer [%s]> %s\n", MakefileDomain, out)
		conv.AddReply(out, MakefileDomain, false)
		offerRunMakeCommand(r, projectRoot, t)
		return true
	}
	// Known Makefile target but not runnable from chat (chat/debug-chat),
	// or not a target at all: say so plainly.
	if isMakeTarget(target) {
		out := fmt.Sprintf("make %s starts a nested chat session — I can't run that from inside the chat.", target)
		fmt.Printf("gollemer [%s]> %s\n", MakefileDomain, out)
		conv.AddReply(out, MakefileDomain, false)
		return true
	}
	out := fmt.Sprintf("I don't know a make target called %q. Say \"list the commands\" to see them all.", target)
	fmt.Printf("gollemer [%s]> %s\n", MakefileDomain, out)
	conv.AddReply(out, MakefileDomain, false)
	return true
}

// tryExplainMake handles "explain make <target>": it prints what the
// target does (from the Makefile's own ## docs plus its recipe), then
// offers to run it.
func tryExplainMake(line string, r *LineReader, projectRoot string, conv *Conversation) bool {
	m := explainMakeRe.FindStringSubmatch(line)
	if m == nil {
		return false
	}
	target := strings.ToLower(m[1])
	var b strings.Builder
	if doc, ok := makeTargetDocs[target]; ok && doc != "" {
		fmt.Fprintf(&b, "make %s — %s\n", target, doc)
	} else if isMakeTarget(target) {
		fmt.Fprintf(&b, "make %s — no description in the Makefile.\n", target)
	} else {
		out := fmt.Sprintf("I don't know a make target called %q. Say \"list the commands\" to see them all.", target)
		fmt.Printf("gollemer [%s]> %s\n", MakefileDomain, out)
		conv.AddReply(out, MakefileDomain, false)
		return true
	}
	if recipe, ok := makeTargetRecipes[target]; ok {
		b.WriteString("It runs:\n")
		for _, rl := range recipe {
			fmt.Fprintf(&b, "  %s\n", rl)
		}
	}
	out := strings.TrimRight(b.String(), "\n")
	fmt.Printf("gollemer [%s]> %s\n", MakefileDomain, out)
	conv.AddReply(out, MakefileDomain, false)
	if t := runnableMakeTarget("run make " + target); t != "" {
		offerRunMakeCommand(r, projectRoot, t)
	}
	return true
}

// isMakeTarget reports whether name is a known Makefile target, including
// the chat entry points that are excluded from execution.
func isMakeTarget(name string) bool {
	if makeTargetsAllowlist[name] {
		return true
	}
	return name == "chat" || name == "debug-chat"
}
