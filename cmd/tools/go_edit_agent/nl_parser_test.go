package main

import (
	"strings"
	"testing"
)

func TestNLPreprocessorExamples(t *testing.T) {
	// Using inline examples instead of dataset file for portability in tests.

	cases := []struct {
		q    string
		file string
	}{
		{"add the return string to f/j.go", "f/j.go"},
		{"add return type string to f/j.go", "f/j.go"},
		{"add string after the return type of F", "f/j.go"},
		{"add \"dfdfds\" to the return of F to f/j.go", "f/j.go"},
		{"return \"fddsf\" to F on f/j.go", "f/j.go"},
	}

	for _, c := range cases {
		ops := parseNaturalLanguageQuery(c.file, c.q)
		if ops == nil || len(ops) == 0 {
			t.Errorf("expected ops for query %q, got none", c.q)
		}
	}
}

func TestGatherContext(t *testing.T) {
	root := findProjectRoot(".")
	if root == "" {
		t.Skip("no project root")
	}
	snips := gatherContext(root, "F", "f/j.go", 3)
	if len(snips) == 0 {
		t.Errorf("expected snippets for symbol F, got none")
	}
}

func TestParamsAndImportPatterns(t *testing.T) {
	cases := []struct{ q, f string }{
		{"add parameters (int, int) to function foo in f/j.go", "f/j.go"},
		{"add import \"fmt\" to f/j.go", "f/j.go"},
	}
	for _, c := range cases {
		ops := nlPreprocessor(c.f, strings.ToLower(c.q))
		if ops == nil || len(ops) == 0 {
			t.Errorf("expected ops for query %q, got none", c.q)
		}
	}
}
