package analyze

import (
	"fmt"
	"strings"
)

// This file detects common Go code patterns and turns them into intent
// phrases. The analyzer already extracts structure (calls, effects, guards);
// patterns recognize *idioms* — retry loops, HTTP handlers, getters — and
// describe what the code is doing in plain terms.

// patternIntent returns a plain-words description if fn matches a known
// idiom, or "" if it doesn't.
func patternIntent(fn *Func) string {
	// Closer: func (T) Close() error. Check before getter (Close looks like one).
	if fn.Name == "Close" {
		return "Releases resources held by the value."
	}
	// HTTP handler: func(w http.ResponseWriter, r *http.Request).
	if isHTTPHandler(fn) {
		name := strings.TrimSuffix(fn.Name, "Handler")
		if name == "" || name == fn.Name {
			return "Handles HTTP requests."
		}
		return fmt.Sprintf("Handles HTTP requests for %s.", camelToWords(name))
	}
	// Stringer: func (T) String() string.
	if fn.Name == "String" && fn.Params == 0 && strings.Contains(fn.Sig, ") string") {
		recv := fn.Receiver
		if recv == "" {
			recv = "the value"
		}
		return fmt.Sprintf("Returns a string representation of %s.", recv)
	}
	// Getter: func (r T) Field() Type — no params, single return, name matches a field.
	if isGetter(fn) {
		name := strings.TrimPrefix(fn.Name, "Get")
		return fmt.Sprintf("Returns the %s.", camelToWords(name))
	}
	// Setter: func (r *T) SetField(v Type).
	if isSetter(fn) {
		field := strings.TrimPrefix(fn.Name, "Set")
		return fmt.Sprintf("Sets the %s.", camelToWords(field))
	}
	// Predicate: IsX(), HasX(), CanX() returning bool.
	if isPredicate(fn) {
		return fmt.Sprintf("Reports whether %s.", predicatePhrase(fn.Name))
	}
	// Closer: func (T) Close() error. (Handled first in patternIntent.)
	// Retry loop: for + sleep + error check.
	if isRetryLoop(fn) {
		return "Retries an operation with backoff until it succeeds or gives up."
	}
	// Cache: map read + map write in the same function.
	if isCache(fn) {
		return "Caches results to avoid recomputing them."
	}
	return ""
}

func isHTTPHandler(fn *Func) bool {
	for _, t := range fn.ParamTypes {
		// Type may be "http.ResponseWriter" or "net/http.ResponseWriter".
		if strings.HasSuffix(t, "ResponseWriter") {
			return true
		}
	}
	return false
}

func isGetter(fn *Func) bool {
	// No params, has receiver. Either "Field" or "GetField".
	if fn.Params != 0 || fn.Receiver == "" || fn.Name == "Close" {
		return false
	}
	name := fn.Name
	if strings.HasPrefix(name, "Get") {
		rest := strings.TrimPrefix(name, "Get")
		if rest == "" || rest[0] < 'A' || rest[0] > 'Z' {
			return false
		}
		name = rest
	}
	words := splitName(name)
	if len(words) != 1 {
		return false
	}
	// Must return something (not just empty).
	return strings.Contains(fn.Sig, ") ") && !strings.HasSuffix(fn.Sig, "()")
}

func isSetter(fn *Func) bool {
	// "SetField(v ...)" — one or more params, name starts with Set.
	if !strings.HasPrefix(fn.Name, "Set") || fn.Params < 1 {
		return false
	}
	rest := strings.TrimPrefix(fn.Name, "Set")
	return rest != "" && rest[0] >= 'A' && rest[0] <= 'Z'
}

func isPredicate(fn *Func) bool {
	if !strings.Contains(fn.Sig, ") bool") {
		return false
	}
	for _, prefix := range []string{"Is", "Has", "Can", "Should", "Needs", "Wants"} {
		if strings.HasPrefix(fn.Name, prefix) {
			rest := strings.TrimPrefix(fn.Name, prefix)
			return rest != "" && rest[0] >= 'A' && rest[0] <= 'Z'
		}
	}
	return false
}

func predicatePhrase(name string) string {
	for _, prefix := range []string{"Is", "Has", "Can", "Should", "Needs", "Wants"} {
		if strings.HasPrefix(name, prefix) {
			rest := strings.TrimPrefix(name, prefix)
			words := camelToWords(rest)
			switch prefix {
			case "Is":
				return words
			case "Has":
				return "it has " + words
			case "Can":
				return "it can " + words
			default:
				return words
			}
		}
	}
	return camelToWords(name)
}

func isRetryLoop(fn *Func) bool {
	// Heuristic: has a loop (LoopCalls non-empty) + sleeps + checks errors.
	if len(fn.LoopCalls) == 0 {
		return false
	}
	sleeps := false
	for _, e := range fn.ExtCalls {
		if e == "time.Sleep" {
			sleeps = true
			break
		}
	}
	return sleeps
}

func isCache(fn *Func) bool {
	// Heuristic: reads and writes map-like state (field reads + writes).
	return len(fn.Reads) > 0 && len(fn.Writes) > 0
}

func camelToWords(s string) string {
	words := splitName(s)
	for i, w := range words {
		words[i] = strings.ToLower(w)
	}
	return strings.Join(words, " ")
}
