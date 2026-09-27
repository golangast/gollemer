package chat

import "testing"

func TestValidatePairAcceptsGoodPair(t *testing.T) {
	seen := map[string]bool{}
	if err := ValidatePair(ChatPair{Input: "hello there", Output: "hi, how are you?"}, seen); err != nil {
		t.Fatalf("good pair rejected: %v", err)
	}
}

func TestValidatePairRejectsBadPairs(t *testing.T) {
	seen := map[string]bool{}
	cases := []ChatPair{
		{Input: "", Output: "hi"},               // empty input
		{Input: "hello", Output: ""},            // empty output
		{Input: "hello", Output: "hello"},       // identical
		{Input: "hi\x00there", Output: "hello"}, // control char
		{Input: "hello", Output: "hi\x07there"}, // control char
		{Input: "word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word word", Output: "hi"}, // too long
	}
	for i, p := range cases {
		if err := ValidatePair(p, seen); err == nil {
			t.Fatalf("case %d accepted but should be quarantined: %+v", i, p)
		}
	}
}

func TestValidatePairRejectsDuplicates(t *testing.T) {
	seen := map[string]bool{}
	p := ChatPair{Input: "Hello there", Output: "Hi!"}
	if err := ValidatePair(p, seen); err != nil {
		t.Fatalf("first occurrence rejected: %v", err)
	}
	seen[normalizePairKey(p)] = true
	dup := ChatPair{Input: "  hello   THERE ", Output: "hi!"} // same after normalization
	if err := ValidatePair(dup, seen); err == nil {
		t.Fatalf("normalized duplicate accepted")
	}
}

func TestClassifyDomain(t *testing.T) {
	if d := ClassifyDomain("What is a goroutine?", ""); d != "go" {
		t.Fatalf("go question classified as %q", d)
	}
	if d := ClassifyDomain("How are you doing today?", ""); d != SocialDomain {
		t.Fatalf("social question classified as %q", d)
	}
	// "How did your day go?" is social — the verb "go" must not trigger it.
	if d := ClassifyDomain("How did your day go?", ""); d != SocialDomain {
		t.Fatalf("social 'go' verb classified as %q", d)
	}
}

func TestClassifyDomainGoVariants(t *testing.T) {
	goQs := []string{
		"What is a WaitGroup?", "What is a pointer?", "Why should I use Go?",
		"Where are binary executables saved?", "How does error handling work?",
		"What does new() do?", "How do I handle dependencies?",
	}
	for _, q := range goQs {
		if d := ClassifyDomain(q, ""); d != "go" {
			t.Errorf("%q classified as %q, want go", q, d)
		}
	}
}

func TestClassifyDomainMakefile(t *testing.T) {
	mkQs := []struct{ in, out string }{
		{"how do i train the model", "run make train-real-seq2seq"},
		{"how do i chat with the model", "run make real-chat"},
		{"clean up the checkpoints", "run make clean"},
		{"what does the makefile do", "it defines targets like make train-real-seq2seq"},
		{"resume training", "run make train-resume"},
	}
	for _, q := range mkQs {
		if d := ClassifyDomain(q.in, q.out); d != MakefileDomain {
			t.Errorf("(%q, %q) classified as %q, want makefile", q.in, q.out, d)
		}
	}
	// "make the most of it" is social — the verb "make" must not trigger it.
	if d := ClassifyDomain("make the most of your day", ""); d != SocialDomain {
		t.Errorf("social 'make' verb classified as %q", d)
	}
	// makefile wins over go when both match.
	if d := ClassifyDomain("how do i train", "run make train-real-seq2seq"); d != MakefileDomain {
		t.Errorf("makefile+go overlap classified as %q", d)
	}
}

func TestClassifyDomainGocode(t *testing.T) {
	codeQs := []struct{ in, out string }{
		{"write a function that adds two ints", "func add(a int, b int) int { return a + b }"},
		{"give me go code for hello world", "package main"},
		{"how do i sum a slice", "total := 0"},
		{"write a method on a struct", "func (r rect) area() float64 { return r.w * r.h }"},
	}
	for _, q := range codeQs {
		if d := ClassifyDomain(q.in, q.out); d != GocodeDomain {
			t.Errorf("(%q, %q) classified as %q, want gocode", q.in, q.out, d)
		}
	}
	// gocode wins over go when both match (code output contains "func").
	if d := ClassifyDomain("write a function", "func add(a int, b int) int { return a + b }"); d != GocodeDomain {
		t.Errorf("gocode+go overlap classified as %q", d)
	}
	// Plain Go Q&A stays in the go domain.
	if d := ClassifyDomain("What is a goroutine?", ""); d != "go" {
		t.Errorf("go question classified as %q", d)
	}
	// Social chatter mentioning code casually stays social.
	if d := ClassifyDomain("how did your day go?", ""); d != SocialDomain {
		t.Errorf("social classified as %q", d)
	}
}

func TestValidatePairRejectsMixedChatterAndCode(t *testing.T) {
	seen := map[string]bool{}
	mixed := []ChatPair{
		// John's example: half sentence, half code.
		{Input: "write a function", Output: "I am func doing (x int) int { return x }", Domain: GocodeDomain},
		{Input: "add two numbers", Output: "sure! here is your function: func add(a int, b int) int { return a + b }", Domain: GocodeDomain},
		{Input: "sum a slice", Output: "func sum(nums []int) int { total := 0; return total } hope this helps!", Domain: GocodeDomain},
		// Pure chatter teaches the code model the wrong mode entirely.
		{Input: "write a function", Output: "sure thing, happy to help!", Domain: GocodeDomain},
	}
	for i, p := range mixed {
		if err := ValidatePair(p, seen); err == nil {
			t.Fatalf("mixed case %d accepted but should be quarantined: %+v", i, p)
		}
	}
}

func TestValidatePairAcceptsCleanCodeAndChat(t *testing.T) {
	seen := map[string]bool{}
	good := []ChatPair{
		// "hello" inside the string literal must not count as chatter.
		{Input: "write a hello world program in go", Output: `func main() { fmt.Println("hello world") }`, Domain: GocodeDomain},
		{Input: "add two ints", Output: "func add(a int, b int) int { return a + b }", Domain: GocodeDomain},
		// Social chatter stays admissible in the social domain.
		{Input: "hello there", Output: "hi, how are you?", Domain: SocialDomain},
	}
	for i, p := range good {
		if err := ValidatePair(p, seen); err != nil {
			t.Fatalf("clean case %d rejected: %v", i, err)
		}
		seen[normalizePairKey(p)] = true
	}
}

func TestTidyGoCode(t *testing.T) {
	caseMap := map[string]string{
		"fmt.println":     "fmt.Println",
		"strings.toupper": "strings.ToUpper",
		"iseven":          "isEven",
	}
	got := tidyGoCode(`func iseven(n int) bool { return n % 2 == 0 }`, caseMap)
	want := `func isEven(n int) bool { return n % 2 == 0 }`
	if got != want {
		t.Errorf("case restore: got %q, want %q", got, want)
	}
	got = tidyGoCode(`func sum(nums [ ] int) int { total: = 0 }`, caseMap)
	want = `func sum(nums [] int) int { total := 0 }`
	if got != want {
		t.Errorf("spacing fix: got %q, want %q", got, want)
	}
	got = tidyGoCode(`func shout(s string) string { return strings.toupper(s) }`, caseMap)
	want = `func shout(s string) string { return strings.ToUpper(s) }`
	if got != want {
		t.Errorf("dotted ident: got %q, want %q", got, want)
	}
}
