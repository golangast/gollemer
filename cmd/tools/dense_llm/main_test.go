package main

import "testing"

func TestFunctionSnippetFromPrompt(t *testing.T) {
	prompt := "add function jim to file jim/jim.go"
	got := functionSnippetFromPrompt(prompt)
	if got == "" {
		t.Fatal("functionSnippetFromPrompt returned empty for function creation prompt")
	}
	if want := "func Jim()"; got[:len(want)] != want {
		t.Fatalf("functionSnippetFromPrompt(%q) = %q, want prefix %q", prompt, got, want)
	}
}
