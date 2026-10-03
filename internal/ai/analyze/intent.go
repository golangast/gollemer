package analyze

// Intent synthesis: what code is FOR, in one plain line. The model
// already knew what code does (effects, calls, guards); intent is the
// purpose behind it — the author's own doc comment when present,
// otherwise a synthesis of the name's verb, its effects, and the flags
// that gate it. General English, no per-feature logic.

import (
	"fmt"
	"strings"
)

// verbLexicon maps a function's leading name-verb to its infinitive.
// A starting set — extend as real usage shows gaps.
var verbLexicon = map[string]string{
	"Add": "add", "Apply": "apply", "Backup": "back up",
	"Build": "build", "Check": "check", "Clean": "clean",
	"Clone": "clone", "Close": "close", "Collect": "collect",
	"Compress": "compress", "Compute": "compute", "Connect": "connect",
	"Copy": "copy", "Create": "create", "Decode": "decode",
	"Delete": "delete", "Disable": "disable", "Display": "display",
	"Download": "download", "Enable": "enable", "Encode": "encode",
	"Encrypt": "encrypt", "Export": "export", "Fetch": "fetch",
	"Filter": "filter", "Find": "find", "Format": "format",
	"Gather": "gather", "Generate": "generate", "Get": "get",
	"Handle": "handle", "Hash": "hash", "Hide": "hide",
	"Import": "import", "Init": "initialize", "Join": "join",
	"List": "list", "Listen": "listen", "Load": "load",
	"Log": "log", "Make": "make", "Merge": "merge",
	"Move": "move", "Open": "open", "Parse": "parse",
	"Print": "print", "Process": "process", "Put": "put",
	"Read": "read", "Register": "register", "Remove": "remove",
	"Render": "render", "Restore": "restore", "Route": "route",
	"Run": "run", "Save": "save", "Scan": "scan",
	"Search": "search", "Send": "send", "Serve": "serve",
	"Set": "set", "Show": "show", "Sort": "sort",
	"Split": "split", "Start": "start", "Stop": "stop",
	"Sync": "sync", "Toggle": "toggle", "Unregister": "unregister",
	"Update": "update", "Upload": "upload", "Validate": "validate",
	"Walk": "walk", "Watch": "watch", "Write": "write",
}

// thirdPerson renders an infinitive verb in third person:
// "copy" -> "copies", "watch" -> "watches", "delete" -> "deletes".
func thirdPerson(v string) string {
	if v == "back up" {
		return "backs up"
	}
	if len(v) >= 2 && strings.HasSuffix(v, "y") &&
		!strings.ContainsRune("aeiou", rune(v[len(v)-2])) {
		return v[:len(v)-1] + "ies"
	}
	for _, suf := range []string{"s", "x", "z", "ch", "sh"} {
		if strings.HasSuffix(v, suf) {
			return v + "es"
		}
	}
	return v + "s"
}

// Intent is the one-line plain-English purpose of a function.
func (fn *Func) Intent() string {
	if s := firstSentence(fn.Doc); s != "" {
		return s
	}
	if fn.IsMain {
		return "Program entry point."
	}
	if fn.IsInit {
		return "Package initializer (runs automatically on import)."
	}
	if fn.IsTest {
		return testIntent(fn.Name)
	}
	if isConstructor(fn.Name) {
		return fmt.Sprintf("Creates a new %s.", strings.TrimPrefix(fn.Name, "New"))
	}
	var b strings.Builder
	b.WriteString(capitalize(verbPhrase(fn.Name, true)))
	b.WriteString(".")
	if eff := effectPhrases(fn); len(eff) > 0 {
		b.WriteString(" " + capitalize(joinWithAnd(eff)) + ".")
	}
	if g := checkGuard(fn); g != "" {
		b.WriteString(fmt.Sprintf(" Checks the `%s` flag.", g))
	}
	return b.String()
}

// splitName splits a CamelCase identifier for intent reading:
// "DeleteFile" -> ["Delete", "File"], "ServeHTTP" -> ["Serve", "HTTP"],
// "Copy" -> ["Copy"]. (The shared camelSplit over-splits single words.)
func splitName(s string) []string {
	var words []string
	start := 0
	runes := []rune(s)
	isUpper := func(r rune) bool { return r >= 'A' && r <= 'Z' }
	isLower := func(r rune) bool { return r >= 'a' && r <= 'z' }
	for i := 1; i < len(runes); i++ {
		r, prev := runes[i], runes[i-1]
		split := false
		switch {
		case isUpper(r) && (isLower(prev) || prev >= '0' && prev <= '9'):
			split = true // Delete|File
		case isUpper(r) && isUpper(prev) && i+1 < len(runes) && isLower(runes[i+1]):
			split = true // HTTP|Server
		}
		if split {
			words = append(words, string(runes[start:i]))
			start = i
		}
	}
	words = append(words, string(runes[start:]))
	return words
}

// verbPhrase turns a function name into a plain verb phrase:
// "DeleteFile" -> "deletes a file" (third=true) or "delete a file"
// (third=false, for "why does X call Y" — "to delete a file").
// "CopyOrCloneFile" -> "copies or clones a file".
func verbPhrase(name string, third bool) string {
	words := splitName(name)
	var parts []string
	i := 0
	for ; i < len(words); i++ {
		w := words[i]
		if w == "Or" || w == "And" {
			parts = append(parts, strings.ToLower(w))
			continue
		}
		v, ok := verbLexicon[w]
		if !ok {
			break
		}
		if third {
			v = thirdPerson(v)
		}
		parts = append(parts, v)
	}
	rest := words[i:]
	if len(parts) == 0 {
		return "handles " + articleJoin(rest)
	}
	out := strings.Join(parts, " ")
	if len(rest) > 0 {
		out += " " + articleJoin(rest)
	}
	return out
}

// articleJoin renders ["File"] as "a file", ["For", "Win"] as "for win"
// (prepositions take no article), ["HTTP"] as "an http".
func articleJoin(words []string) string {
	if len(words) == 0 {
		return ""
	}
	lower := make([]string, len(words))
	for i, w := range words {
		lower[i] = strings.ToLower(w)
	}
	s := strings.Join(lower, " ")
	if s == "" {
		return ""
	}
	// Prepositions and conjunctions take no article.
	switch lower[0] {
	case "for", "to", "with", "by", "from", "of", "in", "on", "at", "as", "via", "and", "or":
		return s
	}
	art := "a"
	if strings.ContainsRune("aeiou", rune(s[0])) {
		art = "an"
	} else if w := words[0]; w == strings.ToUpper(w) && strings.ContainsRune("HFMLNRSX", rune(w[0])) {
		// Acronyms pronounced with a leading vowel sound: HTTP, FAQ.
		art = "an"
	}
	return art + " " + s
}

// effectPhrases renders a function's direct effects in plain words:
// "writes files (os.WriteFile)".
func effectPhrases(fn *Func) []string {
	var writes, reads, net []string
	for _, e := range fn.ExtCalls {
		for _, fx := range callEffects[e] {
			switch fx {
			case fxWrite:
				writes = append(writes, e)
			case fxRead:
				reads = append(reads, e)
			case fxNet:
				net = append(net, e)
			}
		}
	}
	var out []string
	if len(writes) > 0 {
		out = append(out, "writes files ("+strings.Join(writes, ", ")+")")
	}
	if len(reads) > 0 {
		out = append(out, "reads files ("+strings.Join(reads, ", ")+")")
	}
	if len(net) > 0 {
		out = append(out, "uses the network ("+strings.Join(net, ", ")+")")
	}
	return out
}

// checkGuard returns the first flag-like guard condition: the option
// the function honors.
func checkGuard(fn *Func) string {
	for _, g := range fn.Guards {
		if g.FlagLike {
			return g.Cond
		}
	}
	return ""
}

// testIntent renders "TestCopyDir_nested" as "Tests CopyDir (nested).".
func testIntent(name string) string {
	rest := strings.TrimPrefix(name, "Test")
	subj, extra := rest, ""
	if i := strings.Index(rest, "_"); i >= 0 {
		subj = rest[:i]
		extra = strings.ToLower(strings.ReplaceAll(rest[i+1:], "_", " "))
	}
	if extra != "" {
		return fmt.Sprintf("Tests %s (%s).", subj, extra)
	}
	return fmt.Sprintf("Tests %s.", subj)
}

func capitalize(s string) string {
	if s == "" {
		return ""
	}
	r := []rune(s)
	if r[0] >= 'a' && r[0] <= 'z' {
		r[0] -= 'a' - 'A'
	}
	return string(r)
}

func joinWithAnd(parts []string) string {
	if len(parts) <= 2 {
		return strings.Join(parts, " and ")
	}
	return strings.Join(parts[:len(parts)-1], ", ") + ", and " + parts[len(parts)-1]
}
