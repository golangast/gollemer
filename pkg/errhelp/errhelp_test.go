package errhelp

import (
	"strings"
	"testing"
)

// Every pattern translates a real compiler/panic line into plain words
// with the specifics filled in.
func TestTranslatePatterns(t *testing.T) {
	cases := []struct {
		line      string
		title     string
		wantWords []string
	}{
		{`./prog.go:12:5: undefined: foo`, "Undefined name", []string{`"foo"`, "typo"}},
		{`./prog.go:12:9: cannot use x (variable of type int) as string value in argument to f`, "Wrong type", []string{`"x"`, "int", "string"}},
		{`./prog.go:10:1: missing return`, "Missing return", []string{"promises"}},
		{`./prog.go:8:6: declared and not used: x`, "Unused variable", []string{`"x"`}},
		{`./prog.go:3:8: imported and not used: "fmt"`, "Unused import", []string{`"fmt"`}},
		{`./prog.go:12:9: too many arguments in call to save`, "Too many arguments", []string{`"save"`}},
		{`./prog.go:12:9: not enough arguments in call to save`, "Not enough arguments", []string{`"save"`}},
		{`./prog.go:12:5: cannot assign to count`, "Can't assign here", []string{`"count"`}},
		{`./prog.go:9:7: no new variables on left side of :=`, ":= with nothing new", []string{"="}},
		{`./prog.go:7:6: x redeclared in this block`, "Declared twice", []string{`"x"`}},
		{`./prog.go:14:2: syntax error: unexpected newline, expecting comma or }`, "Missing comma", []string{"comma"}},
		{`./prog.go:9:3: multiple-value get() in single-value context`, "Two values, one slot", []string{`"get"`}},
		{`assignment mismatch: 1 variable but get returns 2 values`, "Count mismatch", nil},
		{`unknown field 'Name' in struct literal of type User`, "Unknown field", []string{`"Name"`}},
		{`panic: runtime error: index out of range [3] with length 3`, "Index out of range", []string{"3"}},
		{`panic: runtime error: invalid memory address or nil pointer dereference`, "Nil pointer", []string{"nil"}},
		{`panic: runtime error: assignment to entry in nil map`, "Nil map", []string{"make"}},
		{`fatal error: all goroutines are asleep - deadlock!`, "Deadlock", []string{"channel"}},
		{`panic: send on closed channel`, "Send on closed channel", []string{"closed"}},
		{`panic: interface conversion: interface {} is string, not int`, "Failed type assertion", []string{"int"}},
		{`no required module provides package example.com/foo; to add it:`, "Missing dependency", []string{"go get"}},
		{`go.mod file not found in current directory or any parent directory`, "No go.mod", []string{"go mod init"}},
	}
	for _, c := range cases {
		h, ok := Translate(c.line)
		if !ok {
			t.Errorf("Translate(%q) = no match", c.line)
			continue
		}
		if h.Title != c.title {
			t.Errorf("Translate(%q).Title = %q, want %q", c.line, h.Title, c.title)
		}
		for _, w := range c.wantWords {
			if !strings.Contains(h.Meaning+h.Fix, w) {
				t.Errorf("Translate(%q) missing %q in:\n%s\n%s", c.line, w, h.Meaning, h.Fix)
			}
		}
	}
}

// Unrecognized lines report false so the caller can fall back.
func TestTranslateUnknown(t *testing.T) {
	if _, ok := Translate("hello there"); ok {
		t.Errorf("Translate(chat) = match, want false")
	}
}

// The detector catches pasted errors and leaves chat alone.
func TestLooksLikeGoError(t *testing.T) {
	positives := []string{
		`./prog.go:12:5: undefined: foo`,
		`panic: runtime error: index out of range [3] with length 3`,
		`./x.go:3:8: imported and not used: "fmt"`,
		`fatal error: all goroutines are asleep - deadlock!`,
	}
	for _, line := range positives {
		if !LooksLikeGoError(line) {
			t.Errorf("LooksLikeGoError(%q) = false, want true", line)
		}
	}
	negatives := []string{
		"what does routeDomain do",
		"how do i run the evals",
		"hello",
		"",
		"make an http handler",
	}
	for _, line := range negatives {
		if LooksLikeGoError(line) {
			t.Errorf("LooksLikeGoError(%q) = true, want false", line)
		}
	}
}
