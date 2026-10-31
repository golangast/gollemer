package chat

import (
	"strings"
	"testing"
)

// The curated reference sources (2026-10-31): Effective Go
// sections and the Go Wiki's Switch page riding along in
// resource packs. These tests pin the matcher at unit level
// (which goals earn which sections), the ride-along in both
// pack modes, and the no-spray discipline: a near miss and a
// foreign goal earn no reference lines at all.

// referenceLabels flattens matched entries to "label|url" for
// compact assertions.
func referenceLabels(goal string) []string {
	var out []string
	for _, e := range matchReferences(goal) {
		out = append(out, e.label+"|"+e.url)
	}
	return out
}

func TestMatchReferences(t *testing.T) {
	// A topic the doc names exactly: one entry, the verified
	// section anchor.
	if got := referenceLabels("interfaces"); len(got) != 1 ||
		got[0] != "Interfaces and other types|"+effectiveGoURL {
		t.Errorf("matchReferences(interfaces) = %v", got)
	}
	// Naming interfaces is a different section from interfaces
	// themselves: it leads; the general Names and Interfaces
	// sections ride along (their primary terms are in the goal
	// too), best first.
	if got := referenceLabels("interface names"); len(got) != 3 ||
		got[0] != "Interface names|"+effectiveGoURL+"#interface-names" ||
		got[1] != "Names|"+effectiveGoURL+"#names" ||
		got[2] != "Interfaces and other types|"+effectiveGoURL {
		t.Errorf("matchReferences(interface names) = %v", got)
	}
	// Switch earns both sources: Effective Go's section first,
	// then the Wiki page (Part 4's page-level resource).
	got := referenceLabels("switch statements")
	if len(got) != 2 ||
		got[0] != "Switch|"+effectiveGoURL+"#switch" ||
		got[1] != "Switch|https://go.dev/wiki/Switch" {
		t.Errorf("matchReferences(switch statements) = %v", got)
	}
	// A type-switch ask leads with the type-switch section.
	got = referenceLabels("type switch")
	if len(got) == 0 || got[0] != "Type switch|"+effectiveGoURL+"#type_switch" {
		t.Errorf("matchReferences(type switch) = %v", got)
	}
	// Single-topic goals earn exactly their section.
	for goal, want := range map[string]string{
		"goroutines": "Goroutines|" + effectiveGoURL + "#goroutines",
		"channels":   "Channels|" + effectiveGoURL + "#channels",
		"defer":      "Defer|" + effectiveGoURL + "#defer",
		"formatting": "Formatting|" + effectiveGoURL + "#formatting",
	} {
		if got := referenceLabels(goal); len(got) != 1 || got[0] != want {
			t.Errorf("matchReferences(%q) = %v, want [%s]", goal, got, want)
		}
	}
	// The worker-pool goal reaches the concurrency section by
	// its expanded vocabulary (worker → goroutine/channel/
	// concurrency), the resources matcher's own synonyms.
	found := false
	for _, l := range referenceLabels("worker pool") {
		if l == "Concurrency|"+effectiveGoURL+"#concurrency" {
			found = true
		}
	}
	if !found {
		t.Errorf("matchReferences(worker pool) lacks the concurrency section: %v", referenceLabels("worker pool"))
	}
	// Nothing in the curated sources is about basket weaving.
	if got := referenceLabels("underwater basket weaving"); len(got) != 0 {
		t.Errorf("matchReferences(underwater basket weaving) = %v, want none", got)
	}
	// The cap holds even for a goal that brushes many topics.
	if got := referenceLabels("interfaces channels goroutines errors formatting"); len(got) > referenceMatchCap {
		t.Errorf("matchReferences returned %d entries, cap is %d", len(got), referenceMatchCap)
	}
}

// Doc mode: an explicit ask the Advocates doc has no topic for
// still earns its Effective Go section — labeled as a curated
// source, never dressed up as a doc match, and with no pack
// entries printed.
func TestBuildResourcesReferencesExplicitDocMiss(t *testing.T) {
	withAdvocatesDoc(t, resourcesFixtureDoc())
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("resources for interfaces", nil) {
			t.Error("explicit resources ask not claimed")
		}
	})
	if !strings.Contains(out, `I don't have a topic for "interfaces"`) {
		t.Errorf("doc miss not answered plainly:\n%s", out)
	}
	for _, want := range []string{
		"From Effective Go:",
		"Interfaces and other types — " + effectiveGoURL,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("reference ride-along missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "• ") {
		t.Errorf("a doc miss printed pack entries:\n%s", out)
	}
}

// Seed mode: a qualified pack (worker pools are in the seed)
// gains the concurrency section after its entries.
func TestBuildResourcesReferencesSeedPack(t *testing.T) {
	withAdvocatesDoc(t, nil)
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("i want to build a worker pool", nil) {
			t.Error("seed-mode worker pool goal not claimed")
		}
	})
	for _, want := range []string{
		"https://gobyexample.com/worker-pools",
		"From Effective Go:",
		"Concurrency — " + effectiveGoURL + "#concurrency",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("seed pack with references missing %q:\n%s", want, out)
		}
	}
}

// No spray: a near miss (claimed on a project example, not on
// qualification, and not explicit) earns no reference lines —
// even though its words brush a curated topic. The doc is the
// near-miss test's own: one general topic the goal only
// label-brushes.
func TestBuildResourcesReferencesNotSprayed(t *testing.T) {
	withAdvocatesDoc(t, &advocatesDoc{Topics: []advocatesTopic{
		{Name: "General", Entries: []advocatesEntry{
			{Label: "Server patterns", URL: "https://example.com/server-patterns"},
			{Label: "HTTP caching", URL: "https://example.com/http-caching"},
		}},
	}})
	useProject(t, serverProjectFixture(t))

	out := captureStdout(t, func() {
		if !tryBuildResources("i want to build a server", nil) {
			t.Error("near-miss goal not claimed")
		}
	})
	if strings.Contains(out, "From Effective Go:") || strings.Contains(out, "From the Go Wiki:") {
		t.Errorf("near miss sprayed reference lines:\n%s", out)
	}
}
