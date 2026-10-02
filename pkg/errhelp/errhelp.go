// Package errhelp translates Go compiler errors, vet findings, and
// runtime panics into plain words for beginners: what the message
// means and how to fix it. It is a deterministic pattern table — no
// model needed — covering the errors beginners hit over and over.
//
// The chat auto-detects pasted Go errors (LooksLikeGoError) and runs
// them through Translate; anything unrecognized falls back to the
// Go-concept brain.
package errhelp

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Help is the plain-words translation of one error line.
type Help struct {
	// Title names the problem in human words, e.g. "Undefined name".
	Title string
	// Meaning explains what the compiler is complaining about.
	Meaning string
	// Fix suggests the concrete repair.
	Fix string
	// Matched is the error line that produced this help.
	Matched string
}

type pattern struct {
	re      *regexp.Regexp
	title   string
	meaning string // $1-style template over the regexp groups
	fix     string
	// custom overrides the templates when the message needs computed
	// values (e.g. length-1). It returns meaning, fix.
	custom func(m []string) (string, string)
}

func mustCompile(s string) *regexp.Regexp { return regexp.MustCompile(s) }

// patterns is the whole beginner error vocabulary. Order matters:
// more specific shapes come before the general ones.
var patterns = []pattern{
	{
		re:      mustCompile(`undefined: (\S+)`),
		title:   "Undefined name",
		meaning: `Go doesn't know the name "$1" here. It might be a typo, it might not be declared yet, or it might live in a package you haven't imported.`,
		fix:     `Check the spelling first. If it's yours, declare it before this line; if it's from a package, add the import.`,
	},
	{
		re:      mustCompile(`cannot use (\S+) \(.*?type (\S+?)\) as (?:type )?(\S+)`),
		title:   "Wrong type",
		meaning: `"$1" is a $2, but this spot needs a $3. Go never converts types silently — a number and text are different things even when they look alike.`,
		fix:     `Convert it explicitly, e.g. $3($1), or pass a value that already is a $3.`,
	},
	{
		re:      mustCompile(`missing return`),
		title:   "Missing return",
		meaning: `The function's signature promises a value back, but at least one path through it ends without returning anything.`,
		fix:     `Add a return with the right type at the end of the function, and check every if/else branch returns too.`,
	},
	{
		re:      mustCompile(`declared and not used: (\S+)`),
		title:   "Unused variable",
		meaning: `You created "$1" but never used it. Go treats this as a mistake worth stopping for, not a warning.`,
		fix:     `Either use "$1", delete the line, or assign to the blank identifier _ if you must keep the call.`,
	},
	{
		re:      mustCompile(`imported and not used: (\S+)`),
		title:   "Unused import",
		meaning: `The package $1 is imported but nothing in the file uses it. Go refuses to compile until imports match reality.`,
		fix:     `Delete the import line. If you need the import only for its side effects, write _ $1 instead.`,
	},
	{
		re:      mustCompile(`too many arguments in call to (\S+)`),
		title:   "Too many arguments",
		meaning: `You're handing "$1" more values than it accepts. Its signature defines exactly how many it takes.`,
		fix:     `Look at $1's definition and remove the extra argument.`,
	},
	{
		re:      mustCompile(`not enough arguments in call to (\S+)`),
		title:   "Not enough arguments",
		meaning: `"$1" needs more values than you're giving it. Every parameter in its signature wants one.`,
		fix:     `Look at $1's definition and add the missing argument.`,
	},
	{
		re:      mustCompile(`cannot assign to (\S+)`),
		title:   "Can't assign here",
		meaning: `"$1" can't be assigned to — it may be a constant, a function name, or something else Go considers read-only.`,
		fix:     `Use a different variable name, or change the constant into a variable if the value is meant to change.`,
	},
	{
		re:      mustCompile(`no new variables on left side of :=`),
		title:   ":= with nothing new",
		meaning: `:= declares AND assigns, so it needs at least one brand-new variable on the left. Here every name already exists.`,
		fix:     `Use plain = to reassign existing variables, or give the new value a fresh name with :=.`,
	},
	{
		re:      mustCompile(`(\S+) redeclared in this block`),
		title:   "Declared twice",
		meaning: `"$1" is already declared in this block. Go won't let two different things share one name in the same scope.`,
		fix:     `Rename one of them, or drop the second := if you meant to reuse the first.`,
	},
	{
		re:      mustCompile(`syntax error: unexpected newline, expecting comma or \}`),
		title:   "Missing comma",
		meaning: `In a multi-line list — function call, struct, slice — every line except the last needs a comma, including the last item before the closing bracket.`,
		fix:     `Add a comma at the end of the line the error points to.`,
	},
	{
		re:      mustCompile(`multiple-value (\S+?)\(\) in single-value context`),
		title:   "Two values, one slot",
		meaning: `"$1" hands back two values (usually a result and an error), but you're catching only one.`,
		fix:     `Catch both: result, err := $1(). Always check the error before using the result.`,
	},
	{
		re:      mustCompile(`assignment mismatch: (\d+) variable`),
		title:   "Count mismatch",
		meaning: `The number of variables on the left doesn't match the number of values on the right.`,
		fix:     `Make the counts agree — add a variable, or use _ to ignore a value you don't need.`,
	},
	{
		re:      mustCompile(`unknown field '(\S+)' in struct literal`),
		title:   "Unknown field",
		meaning: `The struct has no field called "$1". It's usually a typo, or the field starts lowercase and can't be set from here.`,
		fix:     `Check the struct's definition for the exact field name and its capitalization.`,
	},
	{
		re:      mustCompile(`cannot convert (\S+) \(.*?type (\S+?)\) to type (\S+)`),
		title:   "Can't convert",
		meaning: `"$1" is a $2 and there's no direct conversion to $3 — not all type pairs convert.`,
		fix:     `Convert in steps (e.g. via string or []byte), or rethink whether you need a $3 at all.`,
	},
	{
		re:      mustCompile(`mismatched types (\S+) and (\S+)`),
		title:   "Mismatched types",
		meaning: `An operation mixes a $1 with a $2. Go needs both sides of +, ==, and friends to be the same type.`,
		fix:     `Convert one side so both match.`,
	},
	{
		re:    mustCompile(`index out of range \[(\d+)\] with length (\d+)`),
		title: "Index out of range",
		custom: func(m []string) (string, string) {
			length := 0
			fmt.Sscanf(m[2], "%d", &length)
			meaning := fmt.Sprintf("You asked for slot %s of a slice that only holds %s items (slots are numbered 0 to %d). The program panicked rather than guess.", m[1], m[2], length-1)
			return meaning, "Guard the access: if i < len(s) { ... }, and check where the index comes from."
		},
	},
	{
		re:      mustCompile(`invalid memory address or nil pointer dereference`),
		title:   "Nil pointer",
		meaning: `The code followed a pointer that points at nothing — a variable that was declared but never given a real value, or a function that returned nil.`,
		fix:     `Find which value is nil (print it just before the crash line) and check for nil before using it.`,
	},
	{
		re:      mustCompile(`assignment to entry in nil map`),
		title:   "Nil map",
		meaning: `You can read from a nil map, but writing to one panics. The map was declared but never created.`,
		fix:     `Create it first: m := make(map[string]int), or m := map[string]int{}.`,
	},
	{
		re:      mustCompile(`all goroutines are asleep - deadlock!`),
		title:   "Deadlock",
		meaning: `Every goroutine is stuck waiting on a channel that nobody will ever send to or receive from. The program is frozen, so Go kills it.`,
		fix:     `Trace each channel: someone must send for every receive. Look for a missing go statement or a receive with no matching send.`,
	},
	{
		re:      mustCompile(`send on closed channel`),
		title:   "Send on closed channel",
		meaning: `The code sent a value on a channel that was already closed. Closing means "no more values, ever."`,
		fix:     `Close the channel exactly once, after the last send — usually with defer close(ch) in the sender.`,
	},
	{
		re:      mustCompile(`close of closed channel`),
		title:   "Double close",
		meaning: `The channel was closed twice. One close is the whole signal; a second is a bug.`,
		fix:     `Make sure only one place closes it — typically the sender, once, when done.`,
	},
	{
		re:      mustCompile(`interface conversion: .*not (\S+)`),
		title:   "Failed type assertion",
		meaning: `A type assertion claimed the value was a $1, but it wasn't, so the program panicked.`,
		fix:     `Use the safe form: v, ok := x.($1); then check ok before using v.`,
	},
	{
		re:      mustCompile(`no required module provides package (\S+)`),
		title:   "Missing dependency",
		meaning: `Your code imports $1, but no module in go.mod provides it.`,
		fix:     `Run: go get $1 — then go mod tidy.`,
	},
	{
		re:      mustCompile(`go\.mod file not found`),
		title:   "No go.mod",
		meaning: `Go can't find the module file, so it doesn't know what your project is called or what it depends on.`,
		fix:     `Run go mod init <module-name> in the project root (or cd into the project first).`,
	},
}

// Translate matches one error line against the pattern table and
// returns the plain-words help. It reports false when nothing matches,
// so the caller can fall back to the language model.
func Translate(line string) (Help, bool) {
	trimmed := strings.TrimSpace(line)
	for _, p := range patterns {
		if m := p.re.FindStringSubmatch(trimmed); m != nil {
			var meaning, fix string
			if p.custom != nil {
				meaning, fix = p.custom(m)
			} else {
				expand := func(tmpl string) string {
					out := tmpl
					for i, g := range m[1:] {
						out = strings.ReplaceAll(out, "$"+strconv.Itoa(i+1), g)
					}
					return out
				}
				meaning, fix = expand(p.meaning), expand(p.fix)
			}
			return Help{
				Title:   p.title,
				Meaning: meaning,
				Fix:     fix,
				Matched: trimmed,
			}, true
		}
	}
	return Help{}, false
}

// errorMarkers are phrases that, outside a Go error, basically never
// appear in chat. Any one of them (or a file.go:line prefix) marks the
// line as a pasted Go error.
var errorMarkers = []string{
	"undefined:",
	"cannot use",
	"missing return",
	"declared and not used",
	"imported and not used",
	"too many arguments in call",
	"not enough arguments in call",
	"cannot assign to",
	"no new variables on left side",
	"redeclared in this block",
	"syntax error",
	"multiple-value",
	"assignment mismatch",
	"unknown field",
	"cannot convert",
	"mismatched types",
	"index out of range",
	"nil pointer dereference",
	"assignment to entry in nil map",
	"all goroutines are asleep",
	"send on closed channel",
	"close of closed channel",
	"interface conversion:",
	"no required module provides package",
	"go.mod file not found",
	"not in std",
}

var fileLineRe = regexp.MustCompile(`\.go:\d+`)

// LooksLikeGoError reports whether the line looks like a pasted Go
// compiler error, vet finding, or panic — as opposed to a chat
// question. It is deliberately conservative: a false negative just
// means the normal router handles the line.
func LooksLikeGoError(line string) bool {
	t := strings.ToLower(strings.TrimSpace(line))
	if t == "" {
		return false
	}
	if fileLineRe.MatchString(t) {
		return true
	}
	for _, m := range errorMarkers {
		if strings.Contains(t, m) {
			return true
		}
	}
	return false
}
