package chat

import (
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// Package lookup (2026-10-31): explicit "which package …"
// questions answered from the stdlib catalog snapshot. These
// tests pin the question shape, the answers (against a fixture
// catalog in the shipped file's shape, and once against the
// shipped snapshot itself), the strength gate, and the
// non-interference lines: project questions and code questions
// keep falling through.

// stdFixtureCatalog is a miniature snapshot in the shipped
// file's shape, with real synopses.
func stdFixtureCatalog() []stdPackageEntry {
	return []stdPackageEntry{
		{Path: "database/sql", Synopsis: "Package sql provides a generic interface around SQL (or SQL-like) databases."},
		{Path: "encoding/json", Synopsis: "Package json implements encoding and decoding of JSON."},
		{Path: "fmt", Synopsis: "Package fmt implements formatted I/O with functions analogous to C's printf and scanf."},
		{Path: "html/template", Synopsis: "Package template (html/template) implements data-driven templates for generating HTML output safe against code injection."},
		{Path: "net/http", Synopsis: "Package http provides HTTP client and server implementations."},
		{Path: "net/url", Synopsis: "Package url parses URLs and implements query escaping."},
		{Path: "strings", Synopsis: "Package strings implements simple functions to manipulate UTF-8 encoded strings."},
		{Path: "testing", Synopsis: "Package testing provides support for automated testing of Go packages."},
		{Path: "text/template", Synopsis: "Package template (text/template) implements data-driven templates for generating textual output."},
		{Path: "time", Synopsis: "Package time provides functionality for measuring and displaying time."},
		{Path: "crypto/rand", Synopsis: "Package rand implements a cryptographically secure random number generator."},
		{Path: "crypto/internal/sysrand", Synopsis: "Package rand provides cryptographically secure random bytes from the operating system."},
	}
}

// withStdPackages swaps the loaded catalog for a test's (the
// advocates/concept-book cache-swap pattern).
func withStdPackages(t *testing.T, pkgs []stdPackageEntry) {
	t.Helper()
	oldCache, oldOnce := stdPackagesCache, stdPackagesOnce
	t.Cleanup(func() { stdPackagesCache, stdPackagesOnce = oldCache, oldOnce })
	stdPackagesCache = pkgs
	stdPackagesOnce = &sync.Once{}
	stdPackagesOnce.Do(func() {})
}

func TestParsePackageLookupShapes(t *testing.T) {
	cases := []struct {
		line string
		rest string
		ok   bool
	}{
		{"which package parses URLs", "parses URLs", true},
		{"which package parses URLs?", "parses URLs", true},
		{"which package is fmt in?", "fmt", true},
		{"what package is net/url in", "net/url", true},
		{"what package should i use for testing", "testing", true},
		{"which package does JSON encoding", "JSON encoding", true},
		{"what's the package for json", "json", true},
		{"what is the package for testing", "testing", true},
		// Not library lookups: project questions and other
		// shapes keep their owners.
		{"which package is my server in", "", false},
		{"what package is this function in", "", false},
		{"how do i parse a url", "", false},
		{"which function parses URLs", "", false},
		{"what does the url package do", "", false},
		{"which package", "", false},
	}
	for _, c := range cases {
		rest, ok := parsePackageLookup(c.line)
		if ok != c.ok || rest != c.rest {
			t.Errorf("parsePackageLookup(%q) = (%q, %v), want (%q, %v)", c.line, rest, ok, c.rest, c.ok)
		}
	}
}

func TestTryPackageLookupAnswers(t *testing.T) {
	withStdPackages(t, stdFixtureCatalog())

	out := captureStdout(t, func() {
		if !tryPackageLookup("which package parses URLs") {
			t.Error("url lookup not claimed")
		}
	})
	for _, want := range []string{
		"That's package net/url — Package url parses URLs and implements query escaping.",
		"Docs: https://pkg.go.dev/net/url",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("url answer missing %q:\n%s", want, out)
		}
	}

	out = captureStdout(t, func() {
		if !tryPackageLookup("what package is url in") {
			t.Error("named-package lookup not claimed")
		}
	})
	if !strings.Contains(out, "That's package net/url") {
		t.Errorf("named-package answer wrong:\n%s", out)
	}

	out = captureStdout(t, func() {
		if !tryPackageLookup("which package does JSON encoding") {
			t.Error("json lookup not claimed")
		}
	})
	if !strings.Contains(out, "That's package encoding/json") {
		t.Errorf("json answer wrong:\n%s", out)
	}

	// An ambiguous bare name names both packages rather than
	// guessing between them.
	out = captureStdout(t, func() {
		if !tryPackageLookup("what package is template in") {
			t.Error("template lookup not claimed")
		}
	})
	for _, want := range []string{
		"More than one package matches that:",
		"html/template",
		"text/template",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("ambiguous answer missing %q:\n%s", want, out)
		}
	}

	// Internal packages are in the catalog but never offered:
	// the secure-random answer is crypto/rand, not its internal
	// helper — whose synopsis actually matches more words.
	out = captureStdout(t, func() {
		if !tryPackageLookup("which package provides cryptographically secure random bytes") {
			t.Error("rand lookup not claimed")
		}
	})
	if !strings.Contains(out, "That's package crypto/rand") {
		t.Errorf("rand answer wrong:\n%s", out)
	}
	if strings.Contains(out, "internal") {
		t.Errorf("an internal package was offered:\n%s", out)
	}
}

// The strength gate: weak topical brushes and non-catalog
// topics fall through, so the brains and the project Q&A keep
// their lines.
func TestTryPackageLookupFallsThrough(t *testing.T) {
	withStdPackages(t, stdFixtureCatalog())
	for _, line := range []string{
		"which package is my server in",    // a project question
		"what package is this function in", // a project question
		"which package does main.go use",   // a file, not a topic
		"which package is the server in",   // one synopsis word is not a match
		"what package handles websockets",  // not in the standard library
		"which package parses dates",       // the synopsis doesn't say so
		"what does the url package do",     // not the lookup shape at all
		"how do i parse a url in go",       // a how-to: the brains'
	} {
		if tryPackageLookup(line) {
			t.Errorf("lookup claimed a line it should fall through on: %q", line)
		}
	}
}

// Absent snapshot = feature off: the same line is not claimed.
func TestPackageLookupAbsentCatalogOff(t *testing.T) {
	withStdPackages(t, nil)
	if tryPackageLookup("which package parses URLs") {
		t.Error("lookup claimed with no catalog loaded")
	}
}

// Through the claim chain, in the loops' own order: the lookup
// answers, and the neighbouring claims never see the line.
func TestPackageLookupThroughChain(t *testing.T) {
	withStdPackages(t, stdFixtureCatalog())
	root := liveSessionRoot(t)
	withChatSession(t, root, true)
	profile := &UserProfile{}
	var handled bool
	out := captureStdout(t, func() { handled = claimChain("which package parses URLs", yes, profile) })
	if !handled {
		t.Fatalf("lookup line fell through the claim chain:\n%s", out)
	}
	if !strings.Contains(out, "That's package net/url") {
		t.Errorf("chain answer wrong:\n%s", out)
	}
}

// The shipped snapshot itself answers: the test points the
// loader at the repo root and asks the two canonical questions
// against the real catalog the importer wrote.
func TestPackageLookupRealCatalog(t *testing.T) {
	oldCache, oldOnce, oldRoot := stdPackagesCache, stdPackagesOnce, unifiedProjectRoot
	t.Cleanup(func() {
		stdPackagesCache, stdPackagesOnce, unifiedProjectRoot = oldCache, oldOnce, oldRoot
	})
	stdPackagesCache = nil
	stdPackagesOnce = &sync.Once{}
	unifiedProjectRoot = filepath.Join("..", "..", "..", "..")
	if got := stdPackages(); len(got) < 100 {
		t.Fatalf("shipped catalog loaded %d packages, want the full index", len(got))
	}

	out := captureStdout(t, func() {
		if !tryPackageLookup("which package parses URLs") {
			t.Error("real-catalog url lookup not claimed")
		}
	})
	if !strings.Contains(out, "That's package net/url — Package url parses URLs and implements query escaping.") {
		t.Errorf("real-catalog url answer wrong:\n%s", out)
	}
	out = captureStdout(t, func() {
		if !tryPackageLookup("which package does JSON encoding") {
			t.Error("real-catalog json lookup not claimed")
		}
	})
	if !strings.Contains(out, "That's package encoding/json") {
		t.Errorf("real-catalog json answer wrong:\n%s", out)
	}
}
